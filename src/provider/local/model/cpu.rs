//! The generic transformer on candle CPU tensors; see the [module
//! docs](super) for what it covers and what it costs in memory.
//!
//! [`Transformer::load`] builds the weights from [`Hparams`] and the
//! per-layer [`LayerLayout`]; [`Transformer::forward`] runs the layers over a
//! batch of tokens against the K/V cache. The burn GPU engine mirrors it
//! operation for operation and shares the host-side pieces (hyperparameters,
//! RoPE tables, masks, MoE routing).

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use candle_core::quantized::{GgmlDType, QMatMul as CandleMatMul, QTensor, gguf_file};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::ops::{rms_norm, sigmoid, softmax_last_dim};
use candle_nn::rotary_emb::{rope, rope_i};

use super::arch::Act;
use super::delta::{DeltaCell, DeltaKernel, DeltaState, Rows};
use super::layout::{Ffn, LayerLayout, Qkv, ffn_norm_part, name, tensor_info};
use super::moe::route;
#[cfg(test)]
use super::paged::Solo;
use super::paged::{BLOCK_SIZE, BlockTable, KvLayout, Sizing};
use super::params::{DeltaSpec, Hparams, LayerSpec, MoeSpec};
use super::qmm::QmmWeight;
use super::quant::{RowTable, TokenEmbedding};
use super::rope::RopeKind;
use super::tier::{BuildFn, ExpertSlice, ExpertStore, LayerExperts, TierShared, add};
use super::{
    Batch, Want, attention_mask, build_rope_tables, candle_err, check_context, err, read_hparams,
};
use crate::error::CucaError;
use crate::provider::local::{ExpertPlacement, ExpertTiers};

mod head;
pub(super) mod paged_attn;
pub(crate) use head::NextnHead;
use paged_attn::{PoolShape, attend_paged_rows, gather, store_rows};

thread_local! {
    /// Whether the current forward pass must reproduce, row by row, exactly
    /// what single-token decoding computes; see [`RowExact`].
    static ROW_EXACT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Scope guard of the shape-stable mode.
///
/// A batched forward pass computes each row with the same arithmetic as a
/// single-row pass except where an operation's rounding depends on the batch
/// shape (a float `gemm` over `t` rows versus one row, a softmax over a
/// longer masked vector). The difference is ~1e-7 and harmless, but the
/// quantized matmuls round their activations to Q8_0, which turns it into
/// occasional one-step jumps that grow to logit differences around one.
/// Speculative verification needs batch rows that equal decode steps
/// bit for bit, so while this guard lives every float matmul runs per row and
/// every attention runs per query row over exactly its own keys, as a decode
/// step would; the quantized matmuls and the element-wise operations are
/// row-independent already.
struct RowExact(bool);

impl RowExact {
    fn enter(on: bool) -> Self {
        let previous = ROW_EXACT.replace(on);
        Self(previous)
    }
}

impl Drop for RowExact {
    fn drop(&mut self) {
        ROW_EXACT.set(self.0);
    }
}

/// A matmul weight: candle's, or the row-batched kernel's.
enum Weight {
    Candle(CandleMatMul),
    /// A Q4_0 or Q8_0 weight read once for all activation rows (see [`qmm`]).
    RowBatched(QmmWeight),
}

/// A matmul weight, applying [`RowExact`] to float weights.
struct QMatMul(Weight);

impl QMatMul {
    fn tensor(t: Tensor) -> Self {
        Self(Weight::Candle(CandleMatMul::Tensor(t)))
    }

    fn from_arc(qt: Arc<QTensor>) -> candle_core::Result<Self> {
        CandleMatMul::from_arc(qt).map(|m| Self(Weight::Candle(m)))
    }

    /// The row-batched kernel over `qt` when it serves its dtype and shape,
    /// else candle's matmul.
    fn from_arc_row_batched(qt: Arc<QTensor>) -> candle_core::Result<Self> {
        match QmmWeight::new(&qt) {
            Some(w) => Ok(Self(Weight::RowBatched(w))),
            None => Self::from_arc(qt),
        }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let m = match &self.0 {
            Weight::Candle(m) => m,
            // Row-independent by construction: no per-row split needed.
            Weight::RowBatched(w) => return row_batched(w, x, w.n()),
        };
        let float_weight = !matches!(m, CandleMatMul::QTensor(_));
        if float_weight && ROW_EXACT.get() && x.rank() >= 2 && x.dim(D::Minus2)? > 1 {
            let rows = x.dim(D::Minus2)?;
            let parts = (0..rows)
                .map(|i| m.forward(&x.narrow(D::Minus2, i, 1)?))
                .collect::<candle_core::Result<Vec<_>>>()?;
            return Tensor::cat(&parts, x.rank() - 2);
        }
        m.forward(x)
    }

    /// [`QMatMul::forward`] keeping only the first `rows` outputs (the first
    /// `rows` weight rows). The row-batched kernel computes just those; any
    /// other weight computes the whole product and drops the rest.
    fn forward_rows(&self, x: &Tensor, rows: usize) -> candle_core::Result<Tensor> {
        if let Weight::RowBatched(w) = &self.0 {
            return row_batched(w, x, rows.min(w.n()));
        }
        let y = self.forward(x)?;
        if rows >= y.dim(D::Minus1)? {
            Ok(y)
        } else {
            y.narrow(D::Minus1, 0, rows)
        }
    }
}

/// `x * w^T` over the first `rows` rows of `w`.
fn row_batched(w: &QmmWeight, x: &Tensor, rows: usize) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    let Some((&width, lead)) = dims.split_last() else {
        candle_core::bail!("matmul: a scalar input");
    };
    if width != w.k() {
        candle_core::bail!("matmul: input shape {dims:?}, weight width {}", w.k());
    }
    let n_rows: usize = lead.iter().product();
    let data = x
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let out = w.forward_rows(&data, n_rows, rows)?;
    let mut shape = lead.to_vec();
    shape.push(rows);
    Tensor::from_vec(out, shape, x.device())
}

/// A matrix weight with its optional bias.
struct Linear {
    w: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.w.forward(x)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// Query, key and value projections of one layer.
enum QkvWeights {
    /// One matmul whose output is `[q | k | v]`.
    Fused(Linear),
    /// Separate projections; K/V exist only on layers that own K/V.
    Split {
        q: Linear,
        kv: Option<(Linear, Linear)>,
    },
}

/// Attention weights of one layer.
struct AttnWeights {
    qkv: QkvWeights,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    /// Weights of the weightless value-norm (ones), Gemma 4 only.
    v_norm: Option<Tensor>,
    wo: Linear,
}

/// Gate and up projections of a dense FFN.
enum GateUp {
    Split {
        gate: QMatMul,
        up: QMatMul,
    },
    /// `up` yields `[gate | up]`, `ff` wide each.
    Fused {
        up: QMatMul,
        ff: usize,
    },
}

/// One routed expert's three matmul weights.
pub(crate) struct CpuExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

/// The experts of one MoE layer.
enum ExpertBank {
    /// Every expert in RAM (`ExpertPlacement::Resident`).
    Resident(Vec<CpuExpert>),
    /// In the file, behind the RAM tier of the store
    /// (`ExpertPlacement::Tiered`).
    Tiered {
        il: usize,
        store: Arc<ExpertStore<CpuExpert>>,
    },
}

/// The shared expert of a Qwen 3.5 MoE layer: a dense gated FFN every token
/// runs, scaled per token by `sigmoid(gate_inp . x)`.
struct SharedExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    /// `ffn_gate_inp_shexp`, `n_embd` floats.
    gate_inp: Vec<f32>,
}

impl SharedExpert {
    /// The shared expert's output for the rows `flat` `(t, n_embd)`.
    ///
    /// The scalar gate is a sequential host dot product per row, so a row's
    /// value never depends on how many rows share the call.
    fn forward(&self, flat: &Tensor, act: Act) -> candle_core::Result<Tensor> {
        let (t, n_embd) = flat.dims2()?;
        let hidden = (activate(&self.gate.forward(flat)?, act)? * self.up.forward(flat)?)?;
        let y = self.down.forward(&hidden)?;
        let rows = flat.flatten_all()?.to_vec1::<f32>()?;
        let gates: Vec<f32> = rows
            .chunks_exact(n_embd)
            .map(|x| {
                let dot: f32 = x.iter().zip(&self.gate_inp).map(|(a, b)| a * b).sum();
                1.0 / (1.0 + (-dot).exp())
            })
            .collect();
        y.broadcast_mul(&Tensor::from_vec(gates, (t, 1), flat.device())?)
    }
}

enum FfnWeights {
    Dense {
        gate_up: GateUp,
        down: QMatMul,
    },
    Moe {
        spec: MoeSpec,
        router: QMatMul,
        experts: ExpertBank,
        shared: Option<SharedExpert>,
    },
}

/// Per-layer-embedding block of one layer (Gemma 4).
struct PleLayer {
    inp_gate: QMatMul,
    proj: QMatMul,
    post_norm: Tensor,
}

/// Gated delta-net weights of one layer: the five projections and, in
/// [`DeltaKernel`], the convolution, gate and norm parameters of its
/// recurrence; each cell holds the recurrent state of one sequence slot and
/// its rollback record.
struct DeltaWeights {
    kernel: DeltaKernel,
    /// `attn_qkv`: the convolution inputs `[q | k | v]`.
    qkv: QMatMul,
    /// `attn_gate`: the output gate `z`.
    gate: QMatMul,
    /// `ssm_beta`: the write strength logits.
    beta: QMatMul,
    /// `ssm_alpha`: the decay logits.
    alpha: QMatMul,
    /// `ssm_out`.
    out: QMatMul,
    /// One cell per state slot, each behind a mutex because a layer runs on a
    /// shared `&Transformer`; only the engine thread ever locks them.
    cells: Vec<Mutex<DeltaCell>>,
}

/// What mixes the tokens of one layer. The delta-net weights are boxed: they
/// are much larger than an attention layer's handles.
enum Mixer {
    Attention(AttnWeights),
    Delta(Box<DeltaWeights>),
}

/// Weights of one transformer block.
struct LayerWeights {
    /// The pre-attention norm; `None` when the architecture has no pre-norms.
    attn_norm: Option<Tensor>,
    mixer: Mixer,
    post_attn_norm: Option<Tensor>,
    /// The norm in front of the FFN; `None` when the architecture has no
    /// pre-norms.
    ffn_norm: Option<Tensor>,
    ffn: FfnWeights,
    post_ffw_norm: Option<Tensor>,
    ple: Option<PleLayer>,
    out_scale: Option<f64>,
}

/// Model-level per-layer-embedding weights (Gemma 4).
struct PleWeights {
    table: RowTable,
    proj: QMatMul,
    proj_norm: Tensor,
}

/// RoPE cos/sin tensors of shape `(positions, n_dims / 2)`.
struct RopeTable {
    cos: Tensor,
    sin: Tensor,
}

/// The key and value tensors of one contiguous K/V cache: the in-trunk
/// prediction head's own (see [`head`]); the trunk's is paged.
type KvPair = (Tensor, Tensor);

/// The K and V data of one K/V-owning layer: `[n_blocks][n_kv_head]
/// [BLOCK_SIZE][head_dim]` floats each, indexed by the block ids of the
/// layer's group.
struct KvData {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// The paged K/V cache of one K/V-owning layer. A layer runs on a shared
/// `&Transformer`, so the data sits behind a lock that only the engine thread
/// ever takes; a drafter holds a handle (`Arc`) to read the target's data.
pub(crate) struct KvLayer {
    data: Mutex<KvData>,
    shape: PoolShape,
}

impl KvLayer {
    /// An empty pool of `blocks` blocks for the K/V-owning layer `spec`.
    fn zeros(spec: &LayerSpec, blocks: usize) -> Arc<Self> {
        let floats = blocks * spec.n_kv_head * BLOCK_SIZE * spec.head_dim;
        Arc::new(Self {
            data: Mutex::new(KvData {
                k: vec![0.0; floats],
                v: vec![0.0; floats],
            }),
            shape: PoolShape {
                n_head: spec.n_head,
                n_kv_head: spec.n_kv_head,
                head_dim: spec.head_dim,
            },
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, KvData> {
        self.data.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Reads tensors from the GGUF file, checking each against its expected shape.
struct Loader<'a> {
    content: &'a gguf_file::Content,
    reader: BufReader<File>,
    /// Hold matmul weights as dequantized f32 instead of quantized (test
    /// reference only: candle's quantized matmul rounds activations to Q8).
    dequantize: bool,
    /// Store float matrices as Q8_0 (a drafter's: it only proposes tokens, so
    /// its rounding cannot change the output, and a quarter of the bytes are
    /// read per step).
    requantize: bool,
}

impl Loader<'_> {
    fn has(&self, tensor_name: &str) -> bool {
        self.content.tensor_infos.contains_key(tensor_name)
    }

    fn qtensor(&mut self, tensor_name: &str, shape: &[usize]) -> Result<QTensor, CucaError> {
        let qt = self
            .content
            .tensor(&mut self.reader, tensor_name, &Device::Cpu)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        if qt.shape().dims() != shape {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {shape:?}, found {:?}",
                qt.shape().dims()
            )));
        }
        Ok(qt)
    }

    fn matmul(&mut self, tensor_name: &str, shape: &[usize]) -> Result<QMatMul, CucaError> {
        let qt = Arc::new(self.qtensor(tensor_name, shape)?);
        self.to_matmul(qt, tensor_name)
    }

    fn to_matmul(&self, qt: Arc<QTensor>, tensor_name: &str) -> Result<QMatMul, CucaError> {
        let float = matches!(
            qt.dtype(),
            GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32
        );
        let matmul = if self.dequantize {
            qt.dequantize(&Device::Cpu).map(QMatMul::tensor)
        } else if self.requantize && float && qt.shape().dims().last().is_some_and(|c| c % 32 == 0)
        {
            qt.dequantize(&Device::Cpu)
                .and_then(|t| QTensor::quantize(&t, GgmlDType::Q8_0))
                .and_then(|q| QMatMul::from_arc_row_batched(Arc::new(q)))
        } else {
            // Q4_0 and Q8_0 go through the row-batched kernel (every row
            // independent of its batch, each weight read once per pass);
            // everything else through candle.
            QMatMul::from_arc_row_batched(qt)
        };
        matmul.map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))
    }

    /// A matrix weight plus its bias when `bias` is set.
    fn linear(
        &mut self,
        il: usize,
        part: &str,
        shape: [usize; 2],
        bias: bool,
    ) -> Result<Linear, CucaError> {
        Ok(Linear {
            w: self.matmul(&name::blk(il, part), &shape)?,
            bias: if bias {
                Some(self.vector(&name::blk_bias(il, part), shape[0])?)
            } else {
                None
            },
        })
    }

    fn vector(&mut self, tensor_name: &str, len: usize) -> Result<Tensor, CucaError> {
        self.qtensor(tensor_name, &[len])?
            .dequantize(&Device::Cpu)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))
    }

    /// The values of the (float or quantized) tensor `tensor_name`, whose
    /// shape must be `shape`, as row-major f32.
    fn floats(&mut self, tensor_name: &str, shape: &[usize]) -> Result<Vec<f32>, CucaError> {
        self.qtensor(tensor_name, shape)?
            .dequantize(&Device::Cpu)
            .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))
    }

    fn opt_vector(
        &mut self,
        present: bool,
        tensor_name: &str,
        len: usize,
    ) -> Result<Option<Tensor>, CucaError> {
        present.then(|| self.vector(tensor_name, len)).transpose()
    }

    fn scalar(&mut self, tensor_name: &str) -> Result<f64, CucaError> {
        let value = self
            .vector(tensor_name, 1)?
            .to_vec1::<f32>()
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?
            .first()
            .copied()
            .ok_or_else(|| err(format!("tensor `{tensor_name}`: empty")))?;
        Ok(f64::from(value))
    }

    /// The `[n_expert, rows, cols]` expert tensor `tensor_name`, split into
    /// one quantized `[rows, cols]` matmul weight per expert.
    fn experts(
        &mut self,
        tensor_name: &str,
        n_expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<QMatMul>, CucaError> {
        let slice = ExpertSlice::read(self.content, tensor_name, n_expert, rows, cols)?;
        let mut raw = vec![0u8; n_expert * slice.file_bytes];
        let io = |e: std::io::Error| err(format!("tensor `{tensor_name}`: {e}"));
        self.reader
            .seek(SeekFrom::Start(slice.offset(0)))
            .map_err(io)?;
        self.reader.read_exact(&mut raw).map_err(io)?;
        raw.chunks_exact(slice.file_bytes.max(1))
            .map(|chunk| {
                let qt = expert_qtensor(&slice, chunk)
                    .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
                self.to_matmul(Arc::new(qt), tensor_name)
            })
            .collect()
    }

    /// The experts of MoE layer `il` (FFN width `ff`): every expert read and
    /// split now, or, with `tiers`, left in the file for the store to load.
    fn expert_bank(
        &mut self,
        il: usize,
        n_expert: usize,
        ff: usize,
        n_embd: usize,
        tiers: Option<&Arc<ExpertStore<CpuExpert>>>,
    ) -> Result<ExpertBank, CucaError> {
        if let Some(store) = tiers {
            return Ok(ExpertBank::Tiered {
                il,
                store: Arc::clone(store),
            });
        }
        let blk = |part: &str| name::blk(il, part);
        let gate = self.experts(&blk(name::FFN_GATE_EXPS), n_expert, ff, n_embd)?;
        let up = self.experts(&blk(name::FFN_UP_EXPS), n_expert, ff, n_embd)?;
        let down = self.experts(&blk(name::FFN_DOWN_EXPS), n_expert, n_embd, ff)?;
        Ok(ExpertBank::Resident(
            gate.into_iter()
                .zip(up)
                .zip(down)
                .map(|((gate, up), down)| CpuExpert { gate, up, down })
                .collect(),
        ))
    }
}

/// One expert's `[rows, cols]` matrix of `slice` from its file bytes.
fn expert_qtensor(slice: &ExpertSlice, bytes: &[u8]) -> candle_core::Result<QTensor> {
    gguf_file::TensorInfo {
        ggml_dtype: slice.dtype,
        shape: (slice.rows, slice.cols).into(),
        offset: 0,
    }
    .read(&mut Cursor::new(bytes), 0, &Device::Cpu)
}

/// Build one expert of `layer` from its three matrices' file bytes, as the
/// resident loader builds it (`Loader::to_matmul` without the test modes).
fn build_cpu_expert(
    layer: &LayerExperts,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
) -> Result<CpuExpert, CucaError> {
    let one = |slice: &ExpertSlice, bytes: &[u8], part: &str| {
        expert_qtensor(slice, bytes)
            .and_then(|qt| QMatMul::from_arc_row_batched(Arc::new(qt)))
            .map_err(|e| err(format!("tensor `{}`: {e}", name::blk(layer.il, part))))
    };
    Ok(CpuExpert {
        gate: one(&layer.gate, gate, name::FFN_GATE_EXPS)?,
        up: one(&layer.up, up, name::FFN_UP_EXPS)?,
        down: one(&layer.down, down, name::FFN_DOWN_EXPS)?,
    })
}

/// The expert store of a tiered model (`ExpertPlacement::Tiered`): every
/// MoE layer's experts located in the file, the RAM tier sized by `tiers`,
/// warmed from `tiers.warm_from`.
///
/// # Errors
///
/// [`CucaError::Config`] for a model without experts, a RAM budget below
/// `n_used` experts per layer or heat of another shape; the loader's errors
/// for unreadable expert tensors.
fn open_expert_store(
    content: &gguf_file::Content,
    path: &Path,
    hp: &Hparams,
    tiers: &ExpertTiers,
) -> Result<Arc<ExpertStore<CpuExpert>>, CucaError> {
    let spec = hp.moe.ok_or_else(|| {
        CucaError::Config(format!(
            "expert tiers need a mixture-of-experts model; `{}` has no experts",
            hp.arch.name()
        ))
    })?;
    let mut layers = Vec::new();
    for (il, ls) in hp.layers.iter().enumerate() {
        if let Ffn::Moe { ff, .. } = LayerLayout::read(hp, il, ls, content)?.ffn {
            let slice = |part: &str, rows, cols| {
                ExpertSlice::read(content, &name::blk(il, part), spec.n_expert, rows, cols)
            };
            let gate = slice(name::FFN_GATE_EXPS, ff, hp.n_embd)?;
            let up = slice(name::FFN_UP_EXPS, ff, hp.n_embd)?;
            let down = slice(name::FFN_DOWN_EXPS, hp.n_embd, ff)?;
            layers.push(LayerExperts {
                il,
                gate,
                up,
                down,
                stored_bytes: gate.file_bytes + up.file_bytes + down.file_bytes,
            });
        }
    }
    let shared = Arc::new(TierShared::new(hp.n_layer, spec.n_expert));
    if let Some(heat) = &tiers.warm_from {
        shared.check(heat)?;
    }
    let build: Arc<BuildFn<CpuExpert>> = Arc::new(build_cpu_expert);
    let store = ExpertStore::open(
        path,
        hp.n_layer,
        layers,
        spec.n_expert,
        spec.n_used,
        tiers.ram_bytes,
        false,
        tiers.loader_threads,
        build,
        Arc::clone(&shared),
    )?;
    if let Some(heat) = &tiers.warm_from {
        shared.seed(heat);
        store.warm(heat, 0, store.ram_slots())?;
    }
    Ok(Arc::new(store))
}

/// Most rows of one `Want::All` call whose intermediate positions a hybrid
/// model can still roll back to (one verification round of a drafter: the
/// pending token and up to sixteen drafts). A longer call leaves only its end
/// position restorable.
pub(crate) const MAX_VERIFY_ROWS: usize = 17;

/// What a hybrid model keeps for one sequence slot besides its K/V: where
/// the recurrent state sits and which earlier positions it can return to. The
/// recurrent state is not positional, so a position needs a snapshot. See
/// [`Transformer::rewind`].
#[derive(Debug, Default)]
struct Slot {
    /// The position the slot's state sits at: where the next forward call for
    /// the slot must start. An attention-only model reads every position
    /// from the block tables and keeps no slots.
    len: usize,
    /// First position and row count of the slot's last forward call that
    /// recorded its rows (`Want::All` over `2..=MAX_VERIFY_ROWS` rows), until
    /// the slot's next forward call: positions `first..first + rows` can be
    /// rebuilt.
    verify: Option<(usize, usize)>,
}

/// A loaded model with its K/V cache and, for hybrid models, its recurrent
/// state.
pub(crate) struct Transformer {
    hp: Hparams,
    context_length: usize,
    vocab: usize,
    /// Token embedding table; `None` on a drafter, whose input is a hidden
    /// state.
    embed: Option<TokenEmbedding>,
    ple: Option<PleWeights>,
    layers: Vec<LayerWeights>,
    output_norm: Tensor,
    lm_head: QMatMul,
    /// Drafter projections between the target's hidden width and its own.
    nextn: Option<Nextn>,
    ropes: Vec<RopeTable>,
    /// The paged K/V of every K/V-owning layer (`None` for the others).
    kv: Vec<Option<Arc<KvLayer>>>,
    layout: KvLayout,
    /// One state slot per sequence the model can run at once (hybrid models
    /// only; empty otherwise).
    slots: Vec<Slot>,
    #[cfg(test)]
    solo: Solo,
    /// Whether any layer is a delta-net layer (has a recurrent state).
    hybrid: bool,
    /// Counters and heat of the expert tiers (`ExpertPlacement::Tiered`).
    tiers: Option<Arc<TierShared>>,
}

/// Which hidden-state rows a forward pass returns next to the logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Hidden {
    None,
    /// The final-norm output the LM head reads, one row of `n_embd` floats
    /// per row of logits [`Want`] selects.
    PostNorm,
    /// The final-norm output of every row of the call, whatever `want`
    /// selects (the multi-token-prediction head reads one per position,
    /// prompt positions included): `tokens * n_embd` floats. With
    /// `Want::None` the LM head is skipped, only the norm runs. The logits
    /// `want` selects are bit-equal to what `PostNorm` returns.
    PostNormAll,
}

/// What a forward pass returns.
#[derive(Default)]
pub(crate) struct Forward {
    pub(crate) logits: Option<Vec<f32>>,
    pub(crate) hidden: Option<Vec<f32>>,
}

/// The target's paged K/V a drafter attends to: the pools of the last
/// sliding-window layer's source and of the last full-attention layer's
/// source (the layers Gemma 4's shared-K/V layers read) with the block table
/// of the sequence being drafted for. The pools are shared handles, so the
/// target's later writes are visible without copying; the tables are copies:
/// take a fresh one per draft.
pub(crate) struct ExtKv {
    sliding: (Arc<KvLayer>, BlockTable),
    full: (Arc<KvLayer>, BlockTable),
}

/// Where one attention call finds its keys and values.
#[derive(Clone, Copy)]
enum KvAccess<'a> {
    /// One contiguous cache (the in-trunk prediction head's): the call's
    /// rows fill the slots `start..`, which are not RoPE positions.
    Slab(&'a KvPair),
    /// A paged layer cache: position `p` is row `p % BLOCK_SIZE` of block
    /// `table[p / BLOCK_SIZE]`.
    Paged {
        layer: &'a KvLayer,
        /// The layer owns the cache: the call stores its own keys and
        /// values first (a shared-K/V layer or a drafter layer only reads).
        write: bool,
        table: &'a [u32],
        /// A drafter layer's single query reads exactly this many positions
        /// of the target's cache; the trunk's row `i` reads up to itself.
        read_len: Option<usize>,
    },
}

/// One attention call: the rows one sequence contributes and the K/V storage it
/// uses. The trunk runs a layer at `rope_start == start` over the paged cache;
/// the multi-token-prediction head (see [`head`]) runs its block over its own
/// cache, whose slots are not RoPE positions.
#[derive(Clone, Copy)]
struct AttnCall<'a> {
    spec: &'a LayerSpec,
    /// Row of the call's first row among the rows of the whole forward step
    /// (the rows of several sequences are concatenated).
    offset: usize,
    /// Cache index of the first row: the call's `t` rows fill `start..`.
    start: usize,
    /// RoPE position of the first row (the rows are consecutive positions).
    rope_start: usize,
    t: usize,
    kv: KvAccess<'a>,
}

/// One sequence's share of a forward step: its rows are `offset..offset + t`
/// of the step's concatenated rows.
struct Seg<'a> {
    start: usize,
    offset: usize,
    t: usize,
    tables: &'a [BlockTable],
    /// The state slot (hybrid models).
    slot: usize,
    /// A delta-net layer keeps what cache rollback needs from this call (see
    /// [`Transformer::rewind`]).
    record: bool,
    want: Want,
    hidden: Hidden,
}

/// A drafter's input and output projections: `pre` maps the concatenation of
/// the target's hidden state and the next token's embedding (`2 * n_embd_out`
/// wide) to the drafter width; `post` maps the drafter's output back to the
/// target's hidden width.
struct Nextn {
    pre: QMatMul,
    post: QMatMul,
}

impl Transformer {
    /// [`Transformer::load_placed`] with every expert resident (tests).
    #[cfg(test)]
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::load_placed(
            content,
            path,
            context_length,
            &ExpertPlacement::Resident,
            &Sizing::single(context_length),
        )
    }

    /// Load the model from the GGUF file at `path`, whose index `content`
    /// was read already, with a K/V cache for `context_length` positions and
    /// the experts of a mixture-of-experts model placed as `experts` says;
    /// `sizing` says how many sequences the pools and the recurrent-state
    /// slots serve.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum, or for tiered placement on a model without
    /// experts or with budgets below what the model needs;
    /// [`CucaError::Provider`] for unsupported or malformed checkpoints;
    /// [`CucaError::Io`] when the file cannot be read.
    pub(crate) fn load_placed(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        experts: &ExpertPlacement,
        sizing: &Sizing,
    ) -> Result<Self, CucaError> {
        Self::load_with(content, path, context_length, false, false, experts, sizing)
    }

    /// Load a multi-token-prediction drafter (`gemma4-assistant`): its
    /// layers own no K/V and read the target's cache through
    /// [`Transformer::draft`].
    ///
    /// # Errors
    ///
    /// As [`Transformer::load_placed`], plus [`CucaError::Provider`] when the file
    /// is not a drafter architecture.
    pub(crate) fn load_drafter(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::load_with(
            content,
            path,
            context_length,
            false,
            true,
            &ExpertPlacement::Resident,
            &Sizing::single(context_length),
        )
    }

    /// [`Transformer::load`] with every matmul weight dequantized to f32: the
    /// exact-arithmetic reference the GPU engine is compared against.
    #[cfg(all(test, feature = "provider-local-gpu"))]
    pub(crate) fn load_dequantized(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::load_with(
            content,
            path,
            context_length,
            true,
            false,
            &ExpertPlacement::Resident,
            &Sizing::single(context_length),
        )
    }

    /// Vocabulary size.
    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Counters and heat of the expert tiers; `None` unless the model loaded
    /// with `ExpertPlacement::Tiered`.
    pub(crate) fn expert_tiers(&self) -> Option<Arc<TierShared>> {
        self.tiers.clone()
    }

    fn load_with(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        dequantize: bool,
        drafter: bool,
        experts: &ExpertPlacement,
        sizing: &Sizing,
    ) -> Result<Self, CucaError> {
        let hp = read_hparams(content)?;
        match (drafter, hp.arch.is_drafter()) {
            (false, true) => {
                return Err(err(format!(
                    "`{}` is a multi-token-prediction drafter, not a standalone model: pass it as the `drafter` of a Gemma 4 model",
                    hp.arch.name()
                )));
            }
            (true, false) => {
                return Err(err(format!(
                    "`{}` is not a drafter architecture",
                    hp.arch.name()
                )));
            }
            _ => {}
        }
        check_context(&hp, context_length)?;
        let store = match experts {
            ExpertPlacement::Resident => None,
            ExpertPlacement::Tiered(_) if dequantize || drafter => {
                return Err(CucaError::Config(
                    "expert tiers apply to a quantized main model only".into(),
                ));
            }
            ExpertPlacement::Tiered(tiers) => Some(open_expert_store(content, path, &hp, tiers)?),
        };
        let mut loader = Loader {
            content,
            reader: BufReader::new(File::open(path)?),
            dequantize,
            requantize: drafter,
        };
        let n_embd = hp.n_embd;

        // Token embedding: shared by the lookup and (when tied) the LM head.
        // A drafter never looks tokens up: its table is only the head.
        let vocab = *tensor_info(content, name::TOKEN_EMBD)?
            .shape
            .dims()
            .first()
            .ok_or_else(|| err(format!("tensor `{}`: scalar", name::TOKEN_EMBD)))?;
        let embd = Arc::new(loader.qtensor(name::TOKEN_EMBD, &[vocab, n_embd])?);
        let embed = if drafter {
            None
        } else {
            Some(TokenEmbedding::new(Arc::clone(&embd))?)
        };
        let lm_head = if loader.has(name::OUTPUT) {
            loader.matmul(name::OUTPUT, &[vocab, n_embd])?
        } else {
            loader.to_matmul(embd, name::TOKEN_EMBD)?
        };
        let output_norm = loader.vector(name::OUTPUT_NORM, n_embd)?;

        let ple = match hp.ple_dim {
            Some(dim) => {
                let row_len = hp.n_layer * dim;
                Some(PleWeights {
                    table: RowTable::open(
                        path,
                        content,
                        name::PER_LAYER_TOKEN_EMBD,
                        "per-layer embedding table",
                        row_len,
                    )?,
                    proj: loader.matmul(name::PER_LAYER_MODEL_PROJ, &[row_len, n_embd])?,
                    proj_norm: loader.vector(name::PER_LAYER_PROJ_NORM, dim)?,
                })
            }
            None => None,
        };

        let ropes = build_rope_tables(content, path, &hp, context_length)?
            .into_iter()
            .map(|t| {
                let shape = (context_length, t.half);
                Ok(RopeTable {
                    cos: Tensor::from_vec(t.cos, shape, &Device::Cpu).map_err(candle_err)?,
                    sin: Tensor::from_vec(t.sin, shape, &Device::Cpu).map_err(candle_err)?,
                })
            })
            .collect::<Result<Vec<_>, CucaError>>()?;

        let kv_layout = KvLayout::sized(&hp, context_length, sizing);
        let mut layers = Vec::with_capacity(hp.n_layer);
        let mut kv_layers = Vec::with_capacity(hp.n_layer);
        for (il, spec) in hp.layers.iter().enumerate() {
            let layout = LayerLayout::read(&hp, il, spec, content)?;
            match kv_layout.layer_group[il].filter(|_| spec.owns_kv(il)) {
                Some(g) => kv_layers.push(Some(KvLayer::zeros(spec, kv_layout.blocks[g]))),
                None => kv_layers.push(None),
            }
            layers.push(load_layer(
                &mut loader,
                &hp,
                il,
                spec,
                &layout,
                store.as_ref(),
            )?);
        }

        let nextn = if drafter {
            Some(Nextn {
                pre: loader.matmul("nextn.pre_projection.weight", &[n_embd, 2 * hp.n_embd_out])?,
                post: loader.matmul("nextn.post_projection.weight", &[hp.n_embd_out, n_embd])?,
            })
        } else {
            None
        };

        let hybrid = hp.is_hybrid();
        for lw in &mut layers {
            if let Mixer::Delta(dw) = &mut lw.mixer {
                dw.set_slots(sizing.max_sequences);
            }
        }
        #[cfg(test)]
        let solo = Solo::new(&kv_layout);
        Ok(Self {
            hp,
            context_length,
            vocab,
            embed,
            ple,
            layers,
            output_norm,
            lm_head,
            nextn,
            ropes,
            kv: kv_layers,
            layout: kv_layout,
            #[cfg(test)]
            solo,
            hybrid,
            slots: (0..if hybrid {
                sizing.max_sequences.max(1)
            } else {
                0
            })
                .map(|_| Slot::default())
                .collect(),
            tiers: store.map(|s| Arc::clone(s.shared())),
        })
    }

    /// Run the rows of every sequence of `batch` through the model in one
    /// pass, storing their K/V in the blocks of each sequence's tables (one
    /// block table per group of the model's [`KvLayout`], already holding
    /// every block the rows write); returns, per sequence, the logits `want`
    /// selects and the final-norm hidden rows `hidden` selects. The dense
    /// layers run over all rows at once (each weight is read once for the
    /// whole step), attention per sequence. With `batch.exact`, every row
    /// equals, bit for bit, what a single-token step at its position computes
    /// (see [`RowExact`]), so the output of a sequence does not depend on the
    /// other sequences in the step.
    ///
    /// A hybrid model's recurrent state sits at one position per state slot
    /// (the end of the slot's last call): `start_pos` must equal it, and a
    /// slot appears at most once per step.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for an empty batch or a sequence without
    /// tokens, a hybrid model's `start_pos` other than its slot's position,
    /// a state slot out of range or used twice, more positions than the
    /// context length or than a sequence's tables hold, a drafter model (it
    /// has no token embedding) or a failed pass; a hybrid model then drops
    /// the recurrent state of every slot in the step.
    pub(crate) fn forward_batch(&mut self, batch: &Batch<'_>) -> Result<Vec<Forward>, CucaError> {
        let _exact = RowExact::enter(batch.exact);
        if batch.rows.is_empty() {
            return Err(err("a forward step requires at least one sequence"));
        }
        let mut tokens: Vec<u32> = Vec::new();
        let mut segs = Vec::with_capacity(batch.rows.len());
        for (i, r) in batch.rows.iter().enumerate() {
            if r.tokens.is_empty() {
                return Err(err("forward requires at least one token"));
            }
            let hi = r.start_pos + r.tokens.len();
            if hi > self.context_length {
                return Err(err(format!(
                    "{hi} positions exceed the context length {}",
                    self.context_length
                )));
            }
            if self.hybrid {
                let Some(state) = self.slots.get(r.state.0) else {
                    return Err(err(format!(
                        "state slot {} is out of range (the model has {})",
                        r.state.0,
                        self.slots.len()
                    )));
                };
                if r.start_pos != state.len {
                    return Err(err(format!(
                        "forward start {} does not match the recurrent state's position {}",
                        r.start_pos, state.len
                    )));
                }
                if batch.rows[..i].iter().any(|o| o.state == r.state) {
                    return Err(err(format!(
                        "state slot {} appears twice in one step",
                        r.state.0
                    )));
                }
            }
            segs.push(Seg {
                start: r.start_pos,
                offset: tokens.len(),
                t: r.tokens.len(),
                tables: r.tables,
                slot: r.state.0,
                record: self.hybrid
                    && r.want == Want::All
                    && (2..=MAX_VERIFY_ROWS).contains(&r.tokens.len()),
                want: r.want,
                hidden: r.hidden,
            });
            tokens.extend_from_slice(r.tokens);
        }
        let embed = self
            .embed
            .as_ref()
            .ok_or_else(|| err("a drafter has no token embedding; use `draft`"))?;
        let embeddings = embed.rows(&tokens, self.hp.embd_scale)?;
        let per_layer = match (&mut self.ple, self.hp.ple_dim) {
            (Some(ple), Some(dim)) => Some(ple.table.rows(&tokens, (dim as f32).sqrt())?),
            _ => None,
        };
        // The call that recorded rows for rollback is over once another one
        // starts. A hybrid model whose pass fails half-way holds recurrent
        // states some layers advanced and others did not: they are discarded.
        if self.hybrid {
            for seg in &segs {
                self.slots[seg.slot].verify = None;
            }
        }
        let outs = match self.forward_layers(embeddings, per_layer, &segs) {
            Ok(outs) => outs,
            Err(e) => {
                for seg in &segs {
                    self.reset_slot(seg.slot);
                }
                return Err(candle_err(e));
            }
        };
        if self.hybrid {
            for seg in &segs {
                let slot = &mut self.slots[seg.slot];
                slot.len = seg.start + seg.t;
                if seg.record {
                    slot.verify = Some((seg.start, seg.t));
                }
            }
        }
        Ok(outs)
    }

    /// Copies of the recurrent state of every delta-net layer for `slot`
    /// (after its last forward call); `None` for an attention-only model or a
    /// slot out of range.
    pub(crate) fn snapshot(&self, slot: usize) -> Option<Vec<DeltaState>> {
        (self.hybrid && slot < self.slots.len()).then(|| {
            self.delta_layers()
                .map(|dw| dw.with_cell(slot, |cell| cell.snapshot()))
                .collect()
        })
    }

    /// The block pools' layout: groups and blocks per group.
    pub(crate) fn kv_layout(&self) -> &KvLayout {
        &self.layout
    }

    /// Bytes of K/V data the pools hold.
    pub(crate) fn kv_pool_bytes(&self) -> u64 {
        self.layout.pool_bytes(&self.hp)
    }

    /// The target's paged K/V a drafter reads, for the sequence whose block
    /// tables are `tables`: the pool of the last sliding-window layer's
    /// source and of the last full-attention layer's source (the layers
    /// Gemma 4's shared-K/V layers read), sharing storage with this model's
    /// cache.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the model has no sliding or no
    /// full-attention K/V-owning layer to share.
    pub(crate) fn ext_kv(&self, tables: &[BlockTable]) -> Result<ExtKv, CucaError> {
        let source = |sliding: bool| -> Result<(Arc<KvLayer>, BlockTable), CucaError> {
            let layer = self
                .hp
                .layers
                .iter()
                .rev()
                .find(|l| l.window.is_some() == sliding)
                .ok_or_else(|| {
                    err(format!(
                        "the drafter needs a target with {} attention layers",
                        if sliding { "sliding-window" } else { "full" }
                    ))
                })?;
            let pool = self.cache(layer.kv_source).map_err(candle_err)?;
            let table = self.layout.layer_group[layer.kv_source]
                .and_then(|g| tables.get(g))
                .ok_or_else(|| err("the sequence has no block table for the drafter's layer"))?;
            Ok((Arc::clone(pool), table.clone()))
        };
        Ok(ExtKv {
            sliding: source(true)?,
            full: source(false)?,
        })
    }

    /// Width of the hidden state the target exchanges with a drafter.
    pub(crate) fn n_embd(&self) -> usize {
        self.hp.n_embd
    }

    /// A drafter's hidden width on the target side (`embedding_length_out`).
    pub(crate) fn n_embd_out(&self) -> usize {
        self.hp.n_embd_out
    }

    /// The architecture.
    pub(crate) fn arch(&self) -> super::arch::Arch {
        self.hp.arch
    }

    /// The `n_embd`-wide embedding row of `token` (scaled as the first layer
    /// sees it): a drafter's input half next to the target's hidden state.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a drafter model, which has no token
    /// embedding.
    pub(crate) fn embedding_row(&self, token: u32) -> Result<Vec<f32>, CucaError> {
        self.embed
            .as_ref()
            .ok_or_else(|| err("a drafter has no token embedding"))?
            .rows(&[token], self.hp.embd_scale)
    }

    /// One drafting step. `input` is the drafter's `2 * n_embd_out` floats
    /// (the next token's embedding followed by the target's hidden state);
    /// the drafter's query sits at position `pos` and attends to the first
    /// `len` positions of the target's cache `ext`. Returns the drafter's logits
    /// over the vocabulary and the hidden state (`n_embd_out` floats) to feed
    /// the next step.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the model is not a drafter, `input` has
    /// the wrong width or a layer fails.
    pub(crate) fn draft(
        &self,
        ext: &ExtKv,
        len: usize,
        input: &[f32],
        pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CucaError> {
        let nextn = self
            .nextn
            .as_ref()
            .ok_or_else(|| err("`draft` needs a drafter model"))?;
        let hp = &self.hp;
        if input.len() != 2 * hp.n_embd_out {
            return Err(err(format!(
                "drafter input has {} floats, expected {}",
                input.len(),
                2 * hp.n_embd_out
            )));
        }
        let run = || -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
            let dev = &Device::Cpu;
            let eps = hp.eps as f32;
            let input = Tensor::from_vec(input.to_vec(), (1, 1, input.len()), dev)?;
            let mut x = nextn.pre.forward(&input)?;
            for il in 0..hp.n_layer {
                let seg = Seg {
                    start: pos,
                    offset: 0,
                    t: 1,
                    tables: &[],
                    slot: 0,
                    record: false,
                    want: Want::None,
                    hidden: Hidden::None,
                };
                x = self.layer(il, &x, None, &[seg], Some((ext, len)))?;
            }
            let normed = rms(&x, &self.output_norm, eps)?;
            let logits = self.head(&normed)?;
            let next_hidden = nextn.post.forward(&normed)?;
            Ok((
                logits.flatten_all()?.to_vec1::<f32>()?,
                next_hidden.flatten_all()?.to_vec1::<f32>()?,
            ))
        };
        run().map_err(candle_err)
    }

    /// Take the state of `slot` back to at most `len` positions and return
    /// the position it now sits at: the next forward call of a hybrid model
    /// for the slot must start there. K/V rows are positional and live in the
    /// sequence's blocks, so an attention-only model has no state to move and
    /// returns `len`; the caller truncates its block tables to what it gets
    /// back.
    ///
    /// A hybrid model keeps a recurrent state that is not positional (it
    /// summarises everything before the current end and cannot be undone), so
    /// it retains only positions it holds a state for:
    ///
    /// - the slot's current end (`len >=` the state's position retains
    ///   everything);
    /// - every position `first..first + rows` of the slot's last forward call
    ///   that recorded its rows (`Want::All` over `2..=`[`MAX_VERIFY_ROWS`]
    ///   rows), until the slot's next forward call: the state there is
    ///   rebuilt from the pre-call copy and the logged rows, bit for bit as
    ///   the original run computed it.
    ///
    /// For any other `len` it falls back to `0` (the zero state): the caller
    /// recomputes from there, or restores a snapshot ([`Transformer::restore`]).
    pub(crate) fn rewind(&mut self, slot: usize, len: usize) -> usize {
        if !self.hybrid {
            return len;
        }
        let Some(state) = self.slots.get(slot) else {
            return 0;
        };
        // An empty cache always means the zero state, even when a pass that
        // died half-way left some layers' state advanced past `len`.
        if len == 0 {
            self.reset_slot(slot);
            return 0;
        }
        if len >= state.len {
            return state.len;
        }
        match state.verify {
            Some((first, rows)) if len >= first && len < first + rows => {
                for dw in self.delta_layers() {
                    dw.restore_prefix(slot, len - first);
                }
                self.slots[slot].len = len;
                len
            }
            _ => {
                self.reset_slot(slot);
                0
            }
        }
    }

    /// Make `snapshot` (one [`DeltaState`] per delta-net layer, from a
    /// [`Forward`] that took it) the state of `slot`, which then sits at
    /// position `len`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a model without a recurrent state, a slot
    /// out of range or a snapshot of another model.
    pub(crate) fn restore(
        &mut self,
        slot: usize,
        snapshot: &[DeltaState],
        len: usize,
    ) -> Result<(), CucaError> {
        if slot >= self.slots.len() || snapshot.len() != self.delta_layers().count() {
            return Err(err(
                "the snapshot does not fit this model's recurrent state",
            ));
        }
        for (dw, state) in self.delta_layers().zip(snapshot) {
            dw.with_cell(slot, |cell| cell.load(state));
        }
        self.slots[slot] = Slot { len, verify: None };
        Ok(())
    }

    /// Whether the model keeps a recurrent state (see
    /// [`Transformer::rewind`]).
    pub(crate) fn is_hybrid(&self) -> bool {
        self.hybrid
    }

    /// Most rows of one forward call whose every intermediate position can be
    /// returned to afterwards: unbounded for an attention-only model,
    /// [`MAX_VERIFY_ROWS`] for a hybrid one.
    pub(crate) fn rollback_rows(&self) -> usize {
        if self.hybrid {
            MAX_VERIFY_ROWS
        } else {
            usize::MAX
        }
    }

    /// The delta-net layers' weights and state.
    fn delta_layers(&self) -> impl Iterator<Item = &DeltaWeights> {
        self.layers.iter().filter_map(|l| match &l.mixer {
            Mixer::Delta(dw) => Some(dw.as_ref()),
            Mixer::Attention(_) => None,
        })
    }

    /// The zero state at position 0 for `slot` (a sequence start, or after a
    /// failed pass).
    pub(crate) fn reset_slot(&mut self, slot: usize) {
        if !self.hybrid || slot >= self.slots.len() {
            return;
        }
        for dw in self.delta_layers() {
            dw.with_cell(slot, DeltaCell::reset);
        }
        self.slots[slot] = Slot::default();
    }

    /// The layers over the concatenated rows `embeddings` of the sequences
    /// `segs` describe, then the final norm and the LM head over the rows the
    /// sequences want.
    fn forward_layers(
        &self,
        embeddings: Vec<f32>,
        per_layer: Option<Vec<f32>>,
        segs: &[Seg<'_>],
    ) -> candle_core::Result<Vec<Forward>> {
        let hp = &self.hp;
        let dev = &Device::Cpu;
        let eps = hp.eps as f32;
        let t = embeddings.len() / hp.n_embd;
        let mut x = Tensor::from_vec(embeddings, (1, t, hp.n_embd), dev)?;

        // Gemma 4 per-layer inputs: projected embeddings merged with the
        // token-identity rows, `(ctx + tok) / sqrt(2)`.
        let ple = match (&self.ple, hp.ple_dim, per_layer) {
            (Some(ple), Some(dim), Some(rows)) => {
                let tokens = Tensor::from_vec(rows, (1, t, hp.n_layer, dim), dev)?;
                let ctx = ple
                    .proj
                    .forward(&x)?
                    .affine(1.0 / (hp.n_embd as f64).sqrt(), 0.0)?
                    .reshape((1, t, hp.n_layer, dim))?;
                let ctx = rms(&ctx, &ple.proj_norm, eps)?;
                Some((&ctx + &tokens)?.affine(std::f64::consts::FRAC_1_SQRT_2, 0.0)?)
            }
            _ => None,
        };

        for il in 0..hp.n_layer {
            x = self.layer(il, &x, ple.as_ref(), segs, None)?;
        }

        // The final norm runs over every row of a sequence that wants all
        // logits or every position's hidden row, else over its last row
        // only. It is row-independent, so a row is the same bits either way.
        // The selected rows of all sequences share one norm and one LM head
        // pass: the head is the largest matrix of a small model.
        let mut outs: Vec<Forward> = (0..segs.len()).map(|_| Forward::default()).collect();
        let mut parts = Vec::new();
        // Per sequence: first row among the normed rows, row count, whether
        // those are all of its rows.
        let mut plan = Vec::with_capacity(segs.len());
        let mut at = 0;
        for seg in segs {
            let all_hidden = seg.hidden == Hidden::PostNormAll;
            if seg.want == Want::None && !all_hidden {
                plan.push(None);
                continue;
            }
            let all_rows = all_hidden || seg.want == Want::All;
            let (from, n) = if all_rows {
                (seg.offset, seg.t)
            } else {
                (seg.offset + seg.t - 1, 1)
            };
            parts.push(x.narrow(1, from, n)?);
            plan.push(Some((at, n, all_rows)));
            at += n;
        }
        if parts.is_empty() {
            return Ok(outs);
        }
        let rows = Tensor::cat(&parts, 1)?.contiguous()?;
        let normed = rms(&rows, &self.output_norm, eps)?;
        let mut logit_parts = Vec::new();
        // Per sequence: first logits row, row count.
        let mut spans = Vec::with_capacity(segs.len());
        let mut lat = 0;
        for (seg, p) in segs.iter().zip(&plan) {
            let Some((at, n, all_rows)) = *p else {
                spans.push(None);
                continue;
            };
            if seg.hidden != Hidden::None {
                outs[spans.len()].hidden =
                    Some(normed.narrow(1, at, n)?.flatten_all()?.to_vec1::<f32>()?);
            }
            let (from, rows) = match seg.want {
                Want::None => {
                    spans.push(None);
                    continue;
                }
                Want::Last if all_rows => (at + n - 1, 1),
                Want::Last | Want::All => (at, n),
            };
            logit_parts.push(normed.narrow(1, from, rows)?);
            spans.push(Some((lat, rows)));
            lat += rows;
        }
        if logit_parts.is_empty() {
            return Ok(outs);
        }
        let head_in = Tensor::cat(&logit_parts, 1)?.contiguous()?;
        let logits = self.head(&head_in)?;
        let vocab = logits.dim(2)?;
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        if let [Some((_, _))] = spans.as_slice() {
            outs[0].logits = Some(flat);
        } else {
            for (out, span) in outs.iter_mut().zip(&spans) {
                if let Some((at, n)) = *span {
                    out.logits = Some(flat[at * vocab..(at + n) * vocab].to_vec());
                }
            }
        }
        Ok(outs)
    }

    /// LM head over final-norm output rows: scale, soft-cap.
    fn head(&self, normed: &Tensor) -> candle_core::Result<Tensor> {
        self.cap_logits(self.lm_head.forward(normed)?)
    }

    /// [`Transformer::head`] over the first `rows` vocabulary entries only
    /// (a proposal head that need not rank the whole vocabulary).
    fn head_rows(&self, normed: &Tensor, rows: usize) -> candle_core::Result<Tensor> {
        self.cap_logits(self.lm_head.forward_rows(normed, rows)?)
    }

    /// The logit scale and soft-cap.
    fn cap_logits(&self, mut logits: Tensor) -> candle_core::Result<Tensor> {
        let hp = &self.hp;
        if hp.logit_scale != 1.0 {
            logits = logits.affine(1.0 / f64::from(hp.logit_scale), 0.0)?;
        }
        if let Some(cap) = hp.logit_softcap {
            logits = logits.affine(1.0 / cap, 0.0)?.tanh()?.affine(cap, 0.0)?;
        }
        Ok(logits)
    }

    /// One transformer block over the concatenated rows `x` of the
    /// sequences `segs` describe.
    fn layer(
        &self,
        il: usize,
        x: &Tensor,
        ple: Option<&Tensor>,
        segs: &[Seg<'_>],
        ext: Option<(&ExtKv, usize)>,
    ) -> candle_core::Result<Tensor> {
        let lw = &self.layers[il];
        let t = x.dim(1)?;
        self.block(lw, x, ple.map(|p| (p, il)), t, |h| match &lw.mixer {
            Mixer::Attention(aw) => self.attention(il, aw, h, segs, ext),
            Mixer::Delta(dw) => dw.forward_segs(h, segs),
        })
    }

    /// The block `lw` over `x` `(1, t, n_embd)`: the normed input goes
    /// through `mix` (the layer's attention or delta-net mixer, already
    /// bound to its positions and cache), then the FFN, each with its
    /// residual; `ple` is Gemma 4's per-layer-embedding rows with the layer
    /// index. The trunk layers and the multi-token-prediction head share it.
    fn block(
        &self,
        lw: &LayerWeights,
        x: &Tensor,
        ple: Option<(&Tensor, usize)>,
        t: usize,
        mix: impl FnOnce(&Tensor) -> candle_core::Result<Tensor>,
    ) -> candle_core::Result<Tensor> {
        let hp = &self.hp;
        let eps = hp.eps as f32;
        let rs = f64::from(hp.residual_scale);
        let scaled = |branch: Tensor| -> candle_core::Result<Tensor> {
            if hp.residual_scale == 1.0 {
                Ok(branch)
            } else {
                branch.affine(rs, 0.0)
            }
        };

        let h = match &lw.attn_norm {
            Some(w) => rms(x, w, eps)?,
            None => x.clone(),
        };
        let mut attn = mix(&h)?;
        if let Some(w) = &lw.post_attn_norm {
            attn = rms(&attn, w, eps)?;
        }
        let x = (x + scaled(attn)?)?;

        let f = match &lw.ffn_norm {
            Some(w) => rms(&x, w, eps)?,
            None => x.clone(),
        };
        let mut f = lw.ffn.forward(&f, hp.act())?;
        if let Some(w) = &lw.post_ffw_norm {
            f = rms(&f, w, eps)?;
        }
        let mut x = (&x + scaled(f)?)?;

        if let (Some(layer), Some((ple, il)), Some(dim)) = (&lw.ple, ple, hp.ple_dim) {
            let slice = ple.narrow(2, il, 1)?.contiguous()?.reshape((1, t, dim))?;
            let g = (layer.inp_gate.forward(&x)?.gelu()? * slice)?;
            x = (&x + rms(&layer.proj.forward(&g)?, &layer.post_norm, eps)?)?;
        }
        match lw.out_scale {
            Some(s) => x.affine(s, 0.0),
            None => Ok(x),
        }
    }

    /// The attention of layer `il` over the normed input `h` `(1, T, n_embd)`
    /// of the concatenated rows of the sequences `segs` describe: the
    /// projections, QK-norm, RoPE and the output gate and projection run over
    /// all rows, the K/V cache write and the attention itself per sequence.
    fn attention(
        &self,
        il: usize,
        aw: &AttnWeights,
        h: &Tensor,
        segs: &[Seg<'_>],
        ext: Option<(&ExtKv, usize)>,
    ) -> candle_core::Result<Tensor> {
        let spec = &self.hp.layers[il];
        let calls = segs
            .iter()
            .map(|seg| {
                let kv = if spec.kv_source != usize::MAX {
                    let source = spec.kv_source;
                    let table = self.layout.layer_group[source]
                        .and_then(|g| seg.tables.get(g))
                        .ok_or_else(|| {
                            candle_core::Error::msg("the sequence has no block table")
                        })?;
                    KvAccess::Paged {
                        layer: self.cache(source)?,
                        write: spec.owns_kv(il),
                        table,
                        read_len: None,
                    }
                } else {
                    let (e, len) = ext.ok_or_else(|| {
                        candle_core::Error::msg("a drafter layer needs the target's K/V")
                    })?;
                    let (layer, table) = if spec.window.is_some() {
                        &e.sliding
                    } else {
                        &e.full
                    };
                    KvAccess::Paged {
                        layer,
                        write: false,
                        table,
                        read_len: Some(len),
                    }
                };
                Ok(AttnCall {
                    spec,
                    offset: seg.offset,
                    start: seg.start,
                    rope_start: seg.start,
                    t: seg.t,
                    kv,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        self.attention_in(&calls, aw, h)
    }

    /// [`Transformer::attention`] for the calls `calls` describe (all of one
    /// layer), which also lets the multi-token-prediction head run it over
    /// its own cache.
    fn attention_in(
        &self,
        calls: &[AttnCall<'_>],
        aw: &AttnWeights,
        h: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hp = &self.hp;
        let Some(first) = calls.first() else {
            return Err(candle_core::Error::msg("attention needs a sequence"));
        };
        let spec = first.spec;
        let t = h.dim(1)?;
        let eps = hp.eps as f32;
        let (nh, d) = (spec.n_head, spec.head_dim);

        let (q, projected) = aw.project(h, nh, spec.n_kv_head, d)?;
        // A full-width Q/K norm (OLMo 2) normalises the whole projection
        // before the heads are split.
        let full_qk = hp.behavior.qk_norm_full;
        let q = match &aw.q_norm {
            Some(w) if full_qk => rms(&q, w, eps)?,
            _ => q,
        };
        // A gated attention packs `[query | gate]` per head into the query
        // projection (each head's `2 * d` values contiguous).
        let (q, gate) = if hp.behavior.gated_attention {
            let packed = q.reshape((1, t, nh, 2 * d))?;
            (
                packed.narrow(3, 0, d)?.contiguous()?,
                Some(
                    packed
                        .narrow(3, d, d)?
                        .contiguous()?
                        .reshape((1, t, nh * d))?,
                ),
            )
        } else {
            (q.reshape((1, t, nh, d))?, None)
        };
        let q = match &aw.q_norm {
            Some(w) if !full_qk => rms(&q, w, eps)?,
            _ => q,
        };
        let q = q.transpose(1, 2)?.contiguous()?;
        let q = self.rotate_calls(spec, &q, calls)?;

        if let Some((k, v)) = &projected {
            self.store_kv(calls, aw, k, v)?;
        }

        let mut outs = Vec::with_capacity(calls.len());
        for call in calls {
            if let [_] = calls {
                outs.push(self.attend_call(call, &q)?);
            } else {
                outs.push(self.attend_call(call, &q.narrow(2, call.offset, call.t)?)?);
            }
        }
        let attn = match outs.as_mut_slice() {
            [only] => only.clone(),
            _ => Tensor::cat(&outs, 1)?,
        };
        let attn = match gate {
            Some(gate) => (attn * sigmoid(&gate)?)?,
            None => attn,
        };
        aw.wo.forward(&attn)
    }

    /// The attention output `(1, t, n_head * d)` of one sequence's queries
    /// `q` `(1, n_head, t, d)` (RoPE applied, K/V already stored).
    fn attend_call(&self, call: &AttnCall<'_>, q: &Tensor) -> candle_core::Result<Tensor> {
        let hp = &self.hp;
        let AttnCall {
            spec, start, t, kv, ..
        } = *call;
        let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
        Ok(match kv {
            KvAccess::Slab((cache_k, cache_v)) => {
                let hi = start + t;
                let lo = window_lo(spec, start, hi);
                let l = hi - lo;
                let k = cache_k.narrow(2, lo, l)?;
                let v = cache_v.narrow(2, lo, l)?;
                let mask = if t > 1 {
                    let mask = attention_mask(start, t, lo, hi, spec.window);
                    Some(Tensor::from_vec(mask, (t, l), &Device::Cpu)?)
                } else {
                    None
                };
                attend(q, &k, &v, mask.as_ref(), spec.scale, hp.attn_softcap)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((1, t, nh * d))?
            }
            KvAccess::Paged {
                layer,
                table,
                read_len,
                ..
            } => {
                let shape = PoolShape {
                    n_head: nh,
                    ..layer.shape
                };
                let data = layer.lock();
                if t == 1 || ROW_EXACT.get() {
                    // Every query row over exactly the keys a decode step at
                    // its position reads, straight out of the blocks.
                    let rows = q
                        .transpose(1, 2)?
                        .contiguous()?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let range = |i: usize| match read_len {
                        Some(len) => (window_lo(spec, start, len), len),
                        None => (window_lo(spec, start + i, start + i + 1), start + i + 1),
                    };
                    let out = attend_paged_rows(
                        &rows,
                        t,
                        range,
                        table,
                        &data.k,
                        &data.v,
                        shape,
                        spec.scale,
                        hp.attn_softcap,
                    )?;
                    Tensor::from_vec(out, (1, t, nh * d), &Device::Cpu)?
                } else {
                    // One batched matmul over the gathered keys.
                    let hi = start + t;
                    let lo = window_lo(spec, start, hi);
                    let l = hi - lo;
                    let k = Tensor::from_vec(
                        gather(table, lo, hi, &data.k, shape)?,
                        (1, nkv, l, d),
                        &Device::Cpu,
                    )?;
                    let v = Tensor::from_vec(
                        gather(table, lo, hi, &data.v, shape)?,
                        (1, nkv, l, d),
                        &Device::Cpu,
                    )?;
                    let mask = attention_mask(start, t, lo, hi, spec.window);
                    let mask = Tensor::from_vec(mask, (t, l), &Device::Cpu)?;
                    attend(q, &k, &v, Some(&mask), spec.scale, hp.attn_softcap)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((1, t, nh * d))?
                }
            }
        })
    }

    /// Rotate the rows `m` `(1, heads, t, d)` of a layer, the first at RoPE
    /// position `rope_start` (a NoPE layer passes them through).
    fn rotate_rows(
        &self,
        spec: &LayerSpec,
        m: &Tensor,
        rope_start: usize,
        t: usize,
    ) -> candle_core::Result<Tensor> {
        match spec.rope {
            Some(r) => {
                let (table, rope_spec) = (&self.ropes[r], &self.hp.ropes[r]);
                apply_rope(
                    m,
                    &table.cos.narrow(0, rope_start, t)?,
                    &table.sin.narrow(0, rope_start, t)?,
                    rope_spec.kind,
                    rope_spec.n_dims,
                )
            }
            None => Ok(m.clone()),
        }
    }

    /// [`Transformer::rotate_rows`] over the concatenated rows `m` `(1,
    /// heads, T, d)` of several sequences, each at its own positions.
    fn rotate_calls(
        &self,
        spec: &LayerSpec,
        m: &Tensor,
        calls: &[AttnCall<'_>],
    ) -> candle_core::Result<Tensor> {
        match (spec.rope, calls) {
            (None, _) => Ok(m.clone()),
            (Some(_), [only]) => self.rotate_rows(spec, m, only.rope_start, only.t),
            (Some(r), _) => {
                let (table, rope_spec) = (&self.ropes[r], &self.hp.ropes[r]);
                let rows = |t: &Tensor| {
                    calls
                        .iter()
                        .map(|c| t.narrow(0, c.rope_start, c.t))
                        .collect::<candle_core::Result<Vec<_>>>()
                };
                apply_rope(
                    m,
                    &Tensor::cat(&rows(&table.cos)?, 0)?,
                    &Tensor::cat(&rows(&table.sin)?, 0)?,
                    rope_spec.kind,
                    rope_spec.n_dims,
                )
            }
        }
    }

    /// Normalise, rotate and store the key and value projections `k` and `v`
    /// `(1, T, n_kv * d)` of a K/V-owning layer, the rows of each call into
    /// the cache indices `start..` of its cache.
    fn store_kv(
        &self,
        calls: &[AttnCall<'_>],
        aw: &AttnWeights,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<()> {
        let Some(first) = calls.first() else {
            return Ok(());
        };
        let spec = first.spec;
        let t = k.dim(1)?;
        let eps = self.hp.eps as f32;
        let (nkv, d) = (spec.n_kv_head, spec.head_dim);
        let full_qk = self.hp.behavior.qk_norm_full;
        let k = match &aw.k_norm {
            Some(w) if full_qk => rms(k, w, eps)?,
            _ => k.clone(),
        };
        let k = k.reshape((1, t, nkv, d))?;
        let k = match &aw.k_norm {
            Some(w) if !full_qk => rms(&k, w, eps)?,
            _ => k,
        };
        let k = self.rotate_calls(spec, &k.transpose(1, 2)?.contiguous()?, calls)?;
        let v = v.reshape((1, t, nkv, d))?;
        let v = match &aw.v_norm {
            Some(ones) => rms(&v, ones, eps)?,
            None => v,
        };
        let v = v.transpose(1, 2)?.contiguous()?;
        for call in calls {
            let (k, v) = match calls {
                [_] => (k.clone(), v.clone()),
                _ => (
                    k.narrow(2, call.offset, call.t)?,
                    v.narrow(2, call.offset, call.t)?,
                ),
            };
            match call.kv {
                KvAccess::Slab((cache_k, cache_v)) => {
                    cache_k.slice_set(&k.contiguous()?, 2, call.start)?;
                    cache_v.slice_set(&v.contiguous()?, 2, call.start)?;
                }
                KvAccess::Paged {
                    layer,
                    write: true,
                    table,
                    ..
                } => {
                    let ks = k.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                    let vs = v.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                    let mut data = layer.lock();
                    let KvData {
                        k: pool_k,
                        v: pool_v,
                    } = &mut *data;
                    store_rows(table, call.start, call.t, &ks, pool_k, layer.shape)?;
                    store_rows(table, call.start, call.t, &vs, pool_v, layer.shape)?;
                }
                KvAccess::Paged { write: false, .. } => {
                    return Err(candle_core::Error::msg(
                        "a K/V projection needs a cache to store in",
                    ));
                }
            }
        }
        Ok(())
    }

    /// The paged K/V of the K/V-owning layer `il`.
    fn cache(&self, il: usize) -> candle_core::Result<&Arc<KvLayer>> {
        self.kv
            .get(il)
            .and_then(Option::as_ref)
            .ok_or_else(|| candle_core::Error::msg(format!("layer {il} owns no K/V cache")))
    }
}

/// Read every weight of layer `il`; a MoE layer's experts stay in the file
/// when `tiers` is set.
fn load_layer(
    loader: &mut Loader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &LayerSpec,
    layout: &LayerLayout,
    tiers: Option<&Arc<ExpertStore<CpuExpert>>>,
) -> Result<LayerWeights, CucaError> {
    let n_embd = hp.n_embd;
    let blk = |part: &str| name::blk(il, part);
    let mixer = match spec.delta() {
        Some(delta) => Mixer::Delta(Box::new(load_delta(loader, hp, il, delta)?)),
        None => Mixer::Attention(load_attention(loader, hp, il, spec, layout)?),
    };

    let ffn = match (layout.ffn, hp.moe) {
        (Ffn::Dense { ff, fused }, _) => FfnWeights::Dense {
            gate_up: if fused {
                GateUp::Fused {
                    up: loader.matmul(&blk(name::FFN_UP), &[2 * ff, n_embd])?,
                    ff,
                }
            } else {
                GateUp::Split {
                    gate: loader.matmul(&blk(name::FFN_GATE), &[ff, n_embd])?,
                    up: loader.matmul(&blk(name::FFN_UP), &[ff, n_embd])?,
                }
            },
            down: loader.matmul(&blk(name::FFN_DOWN), &[n_embd, ff])?,
        },
        (Ffn::Moe { ff, shared_ff }, Some(spec)) => FfnWeights::Moe {
            spec,
            router: loader.matmul(&blk(name::FFN_GATE_INP), &[spec.n_expert, n_embd])?,
            experts: loader.expert_bank(il, spec.n_expert, ff, n_embd, tiers)?,
            shared: match shared_ff {
                Some(sff) => Some(SharedExpert {
                    gate: loader.matmul(&blk(name::FFN_GATE_SHEXP), &[sff, n_embd])?,
                    up: loader.matmul(&blk(name::FFN_UP_SHEXP), &[sff, n_embd])?,
                    down: loader.matmul(&blk(name::FFN_DOWN_SHEXP), &[n_embd, sff])?,
                    gate_inp: loader.floats(&blk(name::FFN_GATE_INP_SHEXP), &[n_embd])?,
                }),
                None => None,
            },
        },
        (Ffn::Moe { .. }, None) => return Err(err("expert tensors without an expert count")),
    };

    let (ple, out_scale) = if hp.behavior.gemma4_extras {
        let ple = match hp.ple_dim {
            Some(dim) => Some(PleLayer {
                inp_gate: loader.matmul(&blk(name::INP_GATE), &[dim, n_embd])?,
                proj: loader.matmul(&blk(name::PROJ), &[n_embd, dim])?,
                post_norm: loader.vector(&blk(name::POST_NORM), n_embd)?,
            }),
            None => None,
        };
        (ple, Some(loader.scalar(&blk(name::LAYER_OUTPUT_SCALE))?))
    } else {
        (None, None)
    };

    Ok(LayerWeights {
        attn_norm: loader.opt_vector(hp.behavior.pre_norms, &blk(name::ATTN_NORM), n_embd)?,
        mixer,
        post_attn_norm: loader.opt_vector(
            layout.post_attn_norm,
            &blk(name::POST_ATTENTION_NORM),
            n_embd,
        )?,
        ffn_norm: loader.opt_vector(
            hp.behavior.pre_norms,
            &blk(ffn_norm_part(hp.behavior.ffn_norm_is_post_attention)),
            n_embd,
        )?,
        ffn,
        post_ffw_norm: loader.opt_vector(
            layout.post_ffw_norm,
            &blk(name::POST_FFW_NORM),
            n_embd,
        )?,
        ple,
        out_scale,
    })
}

/// Read the attention weights of layer `il`. A gated attention (Qwen 3.5)
/// stores `[query | gate]` per head in `attn_q`, twice the query width.
fn load_attention(
    loader: &mut Loader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &LayerSpec,
    layout: &LayerLayout,
) -> Result<AttnWeights, CucaError> {
    let n_embd = hp.n_embd;
    let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
    let q_width = nh * d;
    let q_proj_width = if hp.behavior.gated_attention {
        2 * q_width
    } else {
        q_width
    };
    let kv_width = nkv * d;
    let blk = |part: &str| name::blk(il, part);
    let owns = spec.owns_kv(il);

    let qkv = match layout.qkv {
        Qkv::Fused => {
            if !owns {
                return Err(err(format!(
                    "layer {il}: a fused attn_qkv layer cannot share another layer's K/V"
                )));
            }
            QkvWeights::Fused(loader.linear(
                il,
                name::ATTN_QKV,
                [q_width + 2 * kv_width, n_embd],
                layout.biases.q,
            )?)
        }
        Qkv::Split => QkvWeights::Split {
            q: loader.linear(il, name::ATTN_Q, [q_proj_width, n_embd], layout.biases.q)?,
            kv: if owns {
                Some((
                    loader.linear(il, name::ATTN_K, [kv_width, n_embd], layout.biases.k)?,
                    loader.linear(il, name::ATTN_V, [kv_width, n_embd], layout.biases.v)?,
                ))
            } else {
                None
            },
        },
    };
    let v_norm = if hp.behavior.gemma4_extras && owns {
        Some(Tensor::ones(d, DType::F32, &Device::Cpu).map_err(candle_err)?)
    } else {
        None
    };
    Ok(AttnWeights {
        qkv,
        q_norm: loader.opt_vector(
            layout.q_norm,
            &blk(name::ATTN_Q_NORM),
            if hp.behavior.qk_norm_full { q_width } else { d },
        )?,
        k_norm: loader.opt_vector(
            layout.k_norm,
            &blk(name::ATTN_K_NORM),
            if hp.behavior.qk_norm_full {
                kv_width
            } else {
                d
            },
        )?,
        v_norm,
        wo: loader.linear(il, name::ATTN_OUTPUT, [n_embd, q_width], layout.biases.o)?,
    })
}

/// Read the gated delta-net weights of layer `il`.
///
/// The GGUF converter bakes everything into the stored values: `ssm_a` is
/// already `-exp(A_log)`, and the gated-norm weight multiplies directly. The
/// convolution tensor is `[conv_dim][conv_kernel]` (each channel's taps
/// contiguous; GGUF dims `[conv_kernel, conv_dim]`).
fn load_delta(
    loader: &mut Loader<'_>,
    hp: &Hparams,
    il: usize,
    delta: &DeltaSpec,
) -> Result<DeltaWeights, CucaError> {
    let n_embd = hp.n_embd;
    let (nv, dv) = (delta.n_v_heads, delta.head_v_dim);
    let conv = loader.floats(
        &name::blk(il, name::SSM_CONV1D),
        &[delta.conv_dim(), delta.conv_kernel],
    )?;
    let kernel = DeltaKernel::new(
        *delta,
        &conv,
        loader.floats(&name::blk(il, name::SSM_NORM), &[dv])?,
        hp.eps as f32,
        loader.floats(&name::blk_bare(il, name::SSM_A), &[nv])?,
        loader.floats(&name::blk_bias(il, name::SSM_DT), &[nv])?,
    );
    Ok(DeltaWeights {
        kernel,
        qkv: loader.matmul(&name::blk(il, name::ATTN_QKV), &[delta.conv_dim(), n_embd])?,
        gate: loader.matmul(
            &name::blk(il, name::ATTN_GATE),
            &[delta.value_dim(), n_embd],
        )?,
        beta: loader.matmul(&name::blk(il, name::SSM_BETA), &[nv, n_embd])?,
        alpha: loader.matmul(&name::blk(il, name::SSM_ALPHA), &[nv, n_embd])?,
        out: loader.matmul(&name::blk(il, name::SSM_OUT), &[n_embd, delta.value_dim()])?,
        cells: vec![Mutex::new(DeltaCell::new(delta))],
    })
}

impl DeltaWeights {
    /// Run `f` on the recurrent state and rollback record of `slot`.
    fn with_cell<R>(&self, slot: usize, f: impl FnOnce(&mut DeltaCell) -> R) -> R {
        f(&mut self.cells[slot]
            .lock()
            .unwrap_or_else(PoisonError::into_inner))
    }

    /// Rebuild the state of `slot` after the first `n` rows of its recorded
    /// call.
    fn restore_prefix(&self, slot: usize, n: usize) {
        self.with_cell(slot, |cell| cell.restore_prefix(&self.kernel, n));
    }

    /// Grow to `n` state slots.
    fn set_slots(&mut self, n: usize) {
        let spec = *self.kernel.spec();
        self.cells
            .resize_with(n.max(1), || Mutex::new(DeltaCell::new(&spec)));
    }

    /// The layer's output `(1, T, n_embd)` for the normed input `h` `(1, T,
    /// n_embd)` of the concatenated rows of the sequences `segs` describe,
    /// advancing each sequence's slot by its rows. A sequence with `record`
    /// keeps its rows and the pre-call state for rollback.
    ///
    /// Everything per row (the five projections, the gates, the convolution,
    /// the recurrence, the gated norm) is a function of that row and the
    /// state alone, so the output rows and the final state are the same
    /// whatever the batch split ([`DeltaKernel`]): the projections run over
    /// all rows at once, the recurrence per sequence.
    fn forward_segs(&self, h: &Tensor, segs: &[Seg<'_>]) -> candle_core::Result<Tensor> {
        let t = h.dim(1)?;
        let rows = |w: &QMatMul| -> candle_core::Result<Vec<f32>> {
            w.forward(h)?.flatten_all()?.to_vec1::<f32>()
        };
        let mixed = rows(&self.qkv)?;
        let z = rows(&self.gate)?;
        let (beta, decay) = self.kernel.gates(&rows(&self.beta)?, &rows(&self.alpha)?);
        let spec = self.kernel.spec();
        let (cd, nv, vd) = (spec.conv_dim(), spec.n_v_heads, spec.value_dim());
        let mut out = vec![0.0f32; t * vd];
        for seg in segs {
            let (lo, hi) = (seg.offset, seg.offset + seg.t);
            let inputs = Rows {
                t: seg.t,
                mixed: &mixed[lo * cd..hi * cd],
                beta: &beta[lo * nv..hi * nv],
                decay: &decay[lo * nv..hi * nv],
            };
            let rows_out = self.with_cell(seg.slot, |cell| {
                if seg.record {
                    cell.begin_record(&inputs);
                }
                self.kernel
                    .scan(cell.live_mut(), &inputs, Some(&z[lo * vd..hi * vd]))
            });
            let rows_out = rows_out
                .ok_or_else(|| candle_core::Error::msg("delta-net scan returned no output"))?;
            out[lo * vd..hi * vd].copy_from_slice(&rows_out);
        }
        let out = Tensor::from_vec(out, (1, t, vd), h.device())?;
        self.out.forward(&out)
    }
}

impl AttnWeights {
    /// Query `(1, t, nh * d)` and, on K/V-owning layers, key and value
    /// `(1, t, nkv * d)` projections of `h`.
    fn project(
        &self,
        h: &Tensor,
        nh: usize,
        nkv: usize,
        d: usize,
    ) -> candle_core::Result<(Tensor, Option<KvPair>)> {
        match &self.qkv {
            QkvWeights::Fused(w) => {
                let y = w.forward(h)?;
                let (q, k) = (nh * d, nkv * d);
                Ok((
                    y.narrow(2, 0, q)?,
                    Some((y.narrow(2, q, k)?, y.narrow(2, q + k, k)?)),
                ))
            }
            QkvWeights::Split { q, kv } => {
                let q = q.forward(h)?;
                match kv {
                    Some((k, v)) => Ok((q, Some((k.forward(h)?, v.forward(h)?)))),
                    None => Ok((q, None)),
                }
            }
        }
    }

    /// Key and value `(1, t, nkv * d)` projections of `h` for a layer that
    /// owns K/V, without the query projection (a fused `attn_qkv` computes
    /// all three anyway).
    fn project_kv(
        &self,
        h: &Tensor,
        nh: usize,
        nkv: usize,
        d: usize,
    ) -> candle_core::Result<KvPair> {
        match &self.qkv {
            QkvWeights::Split {
                kv: Some((k, v)), ..
            } => Ok((k.forward(h)?, v.forward(h)?)),
            _ => self
                .project(h, nh, nkv, d)?
                .1
                .ok_or_else(|| candle_core::Error::msg("this layer owns no K/V projections")),
        }
    }
}

impl FfnWeights {
    /// The FFN over `f` of shape `(1, t, n_embd)`.
    fn forward(&self, f: &Tensor, act: Act) -> candle_core::Result<Tensor> {
        match self {
            FfnWeights::Dense { gate_up, down } => {
                let (gate, up) = match gate_up {
                    GateUp::Split { gate, up } => (gate.forward(f)?, up.forward(f)?),
                    GateUp::Fused { up, ff } => {
                        let y = up.forward(f)?;
                        (y.narrow(2, 0, *ff)?, y.narrow(2, *ff, *ff)?)
                    }
                };
                down.forward(&(activate(&gate, act)? * up)?)
            }
            FfnWeights::Moe {
                spec,
                router,
                experts,
                shared,
            } => {
                let (_, t, n_embd) = f.dims3()?;
                let flat = f.reshape((t, n_embd))?;
                let logits = router.forward(&flat)?.flatten_all()?.to_vec1::<f32>()?;
                let routing = route(&logits, t, spec);
                let mut pass = match experts {
                    ExpertBank::Resident(_) => None,
                    ExpertBank::Tiered { il, store } => {
                        let shared = store.shared();
                        shared.record(*il, &routing, t);
                        let needed: Vec<u32> = (0..routing.per_expert.len() as u32)
                            .filter(|&e| !routing.per_expert[e as usize].is_empty())
                            .collect();
                        add(&shared.lookups, needed.len() as u64);
                        Some(store.pass(*il, &needed))
                    }
                };
                let mut out = Tensor::zeros((t, n_embd), DType::F32, f.device())?;
                for (e, picks) in routing.per_expert.iter().enumerate() {
                    if picks.is_empty() {
                        continue;
                    }
                    // Experts run in ascending order in both placements, so
                    // the accumulation order (and the output) is the same.
                    let fetched;
                    let x = match (experts, pass.as_mut()) {
                        (ExpertBank::Resident(all), _) => &all[e],
                        (ExpertBank::Tiered { .. }, Some(pass)) => {
                            fetched = pass
                                .next(e as u32)
                                .map_err(|e| candle_core::Error::msg(e.to_string()))?;
                            &*fetched
                        }
                        (ExpertBank::Tiered { .. }, None) => {
                            return Err(candle_core::Error::msg("tiered experts without a pass"));
                        }
                    };
                    let (ids, weights): (Vec<u32>, Vec<f32>) = picks.iter().copied().unzip();
                    let m = ids.len();
                    let ids = Tensor::from_vec(ids, m, f.device())?;
                    let weights = Tensor::from_vec(weights, (m, 1), f.device())?;
                    let rows = flat.index_select(&ids, 0)?;
                    let hidden =
                        (activate(&x.gate.forward(&rows)?, act)? * x.up.forward(&rows)?)?;
                    let y = x.down.forward(&hidden)?.broadcast_mul(&weights)?;
                    out = out.index_add(&ids, &y, 0)?;
                }
                if let Some(shared) = shared {
                    out = (out + shared.forward(&flat, act)?)?;
                }
                out.reshape((1, t, n_embd))
            }
        }
    }
}

fn activate(x: &Tensor, act: Act) -> candle_core::Result<Tensor> {
    match act {
        Act::Silu => x.silu(),
        Act::Gelu => x.gelu(),
    }
}

/// RMSNorm with a learned weight (no `1 + w`).
fn rms(x: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    rms_norm(&x.contiguous()?, weight, eps)
}

/// Rotate the first `n_dims` values of every head of `x` `(1, heads, t, d)`
/// under `kind`; the remaining `d - n_dims` values pass through. `cos` and
/// `sin` are `(t, n_dims / 2)`.
pub(crate) fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    kind: RopeKind,
    n_dims: usize,
) -> candle_core::Result<Tensor> {
    let rotate = match kind {
        RopeKind::Neox => rope,
        RopeKind::Norm => rope_i,
    };
    let d = x.dim(3)?;
    if n_dims == d {
        return rotate(&x.contiguous()?, cos, sin);
    }
    let rotated = rotate(&x.narrow(3, 0, n_dims)?.contiguous()?, cos, sin)?;
    Tensor::cat(&[&rotated, &x.narrow(3, n_dims, d - n_dims)?], 3)
}

/// Grouped-query attention: `q` is `(1, n_head, t, d)`, `k` and `v` are
/// `(1, n_kv, l, d)` and `mask` the additive `t x l` mask (`None` for a single
/// query, which sees every key). Query head `h` reads KV head
/// `h / (n_head / n_kv)`. The scores are scaled by `scale` and, with
/// `softcap`, squashed as `cap * tanh(score / cap)` before the mask. The
/// result is `(1, n_head, t, d)`.
///
/// The `n_head / n_kv` query heads of one KV head are stacked into the rows of
/// one matmul against that head's keys, so K and V are never repeated.
pub(crate) fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    softcap: Option<f64>,
) -> candle_core::Result<Tensor> {
    let (_, n_head, t, d) = q.dims4()?;
    let (_, n_kv, l, _) = k.dims4()?;
    let rows = n_head / n_kv * t;
    let mut scores = q.reshape((1, n_kv, rows, d))?.matmul(&k.transpose(2, 3)?)?;
    if scale != 1.0 {
        scores = scores.affine(f64::from(scale), 0.0)?;
    }
    if let Some(cap) = softcap {
        scores = scores.affine(1.0 / cap, 0.0)?.tanh()?.affine(cap, 0.0)?;
    }
    let scores = match mask {
        Some(mask) => scores
            .reshape((1, n_head, t, l))?
            .broadcast_add(mask)?
            .reshape((1, n_kv, rows, l))?,
        None => scores,
    };
    softmax_last_dim(&scores.contiguous()?)?
        .matmul(v)?
        .reshape((1, n_head, t, d))
}

/// The first key a query at cache slot `query` attends to when `hi` slots are
/// filled: a sliding window reaches back `window - 1` slots, a full-attention
/// layer starts at the first.
fn window_lo(spec: &LayerSpec, query: usize, hi: usize) -> usize {
    match spec.window {
        Some(w) => query.saturating_sub(w - 1).min(hi.saturating_sub(1)),
        None => 0,
    }
}

/// The single-sequence calls the unit tests use: one implicit sequence over
/// the model's own pools ([`Solo`]), so a test reads like the old contiguous
/// cache. The engine passes its sequences' block tables explicitly.
#[cfg(test)]
impl Transformer {
    /// [`Transformer::forward_ext`] logits only, not exact.
    pub(crate) fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want: Want,
    ) -> Result<Option<Vec<f32>>, CucaError> {
        Ok(self
            .forward_ext(tokens, start_pos, want, Hidden::None, false)?
            .logits)
    }

    /// [`Transformer::forward_batch`] over the implicit sequence (state slot
    /// 0): its blocks are allocated here; `start_pos` must equal its length.
    pub(crate) fn forward_ext(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want: Want,
        hidden: Hidden,
        exact: bool,
    ) -> Result<Forward, CucaError> {
        let seq = self.solo.begin(start_pos, tokens.len())?;
        let batch = Batch::single(tokens, start_pos, &seq.tables, want, hidden, exact);
        let out = self
            .forward_batch(&batch)
            .map(|mut outs| outs.swap_remove(0));
        self.solo
            .finish(seq, out.is_ok().then_some(start_pos + tokens.len()));
        out
    }

    /// Drop the positions from `len` on of the implicit sequence and return
    /// how many it retains (see [`Transformer::rewind`] and
    /// [`KvPools::truncate`](super::paged::KvPools::truncate)).
    pub(crate) fn truncate_cache(&mut self, len: usize) -> usize {
        let state = self.rewind(0, len).min(self.solo.len());
        let kept = self.solo.truncate(state);
        if kept != state {
            self.rewind(0, kept);
        }
        kept
    }

    /// Positions of the implicit sequence.
    pub(crate) fn cache_len(&self) -> usize {
        self.solo.len()
    }

    /// The implicit sequence's block tables, for a drafter that reads this
    /// target.
    #[cfg(feature = "provider-local-gpu")]
    pub(crate) fn solo_tables(&self) -> &[BlockTable] {
        self.solo.tables()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::GgmlDType;

    use super::super::rope::{RopeKind, rotate_reference};
    use super::super::tiny::{
        PROMPT, TinyMoe, TinyOptions, TinyQwen35, TinySpec, VOCAB, assert_close, load,
        write_qwen35, write_spec, write_tiny_model, write_tiny_model_with,
    };
    use super::*;

    fn logits(m: &mut Transformer, tokens: &[u32], start: usize) -> Vec<f32> {
        m.forward(tokens, start, Want::Last).unwrap().unwrap()
    }

    /// Cache and batching invariants every architecture must satisfy: a
    /// chunked prefill, a token-by-token decode and a replay after
    /// truncation all reproduce the single-shot logits.
    fn check_invariants(model: &super::super::tiny::TinyModel, what: &str) {
        let mut whole = load(model, 32);
        let expected = logits(&mut whole, &PROMPT, 0);
        assert_eq!(expected.len(), VOCAB, "{what}");
        assert!(expected.iter().all(|v| v.is_finite()), "{what}: finite");
        let spread = expected.iter().fold(0f32, |m, v| m.max(v.abs()))
            - expected.iter().fold(f32::INFINITY, |m, v| m.min(v.abs()));
        assert!(spread > 1e-3, "{what}: logits vary across the vocabulary");

        let mut chunked = load(model, 32);
        assert!(
            chunked
                .forward(&PROMPT[..4], 0, Want::None)
                .unwrap()
                .is_none()
        );
        assert!(
            chunked
                .forward(&PROMPT[4..9], 4, Want::None)
                .unwrap()
                .is_none()
        );
        assert_close(&expected, &logits(&mut chunked, &PROMPT[9..], 9));

        let mut stepped = load(model, 32);
        assert!(
            stepped
                .forward(&PROMPT[..10], 0, Want::None)
                .unwrap()
                .is_none()
        );
        assert_close(&expected, &logits(&mut stepped, &PROMPT[10..], 10));

        // Truncate to 6 and replay the rest.
        let mut replay = load(model, 32);
        logits(&mut replay, &PROMPT, 0);
        replay.truncate_cache(6);
        assert_eq!(replay.cache_len(), 6);
        assert_close(&expected, &logits(&mut replay, &PROMPT[6..], 6));

        // `Want::All` row i equals the Last logits after i + 1 tokens.
        let mut all = load(model, 32);
        let rows = all.forward(&PROMPT, 0, Want::All).unwrap().unwrap();
        assert_eq!(rows.len(), PROMPT.len() * VOCAB);
        assert_close(&expected, &rows[(PROMPT.len() - 1) * VOCAB..]);
        let mut prefix = load(model, 32);
        let third = logits(&mut prefix, &PROMPT[..3], 0);
        assert_close(&third, &rows[2 * VOCAB..3 * VOCAB]);
    }

    #[test]
    fn gemma4_tiny_model_satisfies_the_invariants() {
        check_invariants(&write_tiny_model(), "gemma4");
    }

    #[test]
    fn every_architecture_satisfies_the_invariants() {
        for spec in TinySpec::all() {
            check_invariants(&write_spec(&spec), spec.arch);
        }
    }

    /// The rows of several sequences in one forward step equal, bit for bit,
    /// the same rows run one sequence at a time: the dense layers see all
    /// rows, attention and the recurrent state stay per sequence. The steps
    /// mix a joint prefill, decode rows, a sequence sitting one step out, a
    /// prefill chunk next to decode rows and a verification-shaped call.
    fn check_batched(model: &super::super::tiny::TinyModel, what: &str) {
        use super::super::StateSlot;
        use super::super::paged::KvPools;
        use super::super::tiny::load_sized;
        type Call = (usize, &'static [u32], Want, Hidden);
        let steps: [&[Call]; 5] = [
            &[
                (0, &PROMPT[..5], Want::Last, Hidden::PostNorm),
                (1, &PROMPT[..9], Want::Last, Hidden::PostNorm),
                (2, &PROMPT[3..6], Want::None, Hidden::PostNormAll),
            ],
            &[
                (0, &[7], Want::Last, Hidden::PostNorm),
                (1, &[9], Want::Last, Hidden::None),
                (2, &[4, 5], Want::All, Hidden::PostNorm),
            ],
            &[
                (0, &[8], Want::Last, Hidden::PostNorm),
                (2, &[6], Want::Last, Hidden::PostNorm),
            ],
            &[
                (0, &[6, 3, 2, 5], Want::Last, Hidden::None),
                (1, &[2], Want::Last, Hidden::PostNorm),
                (2, &[3], Want::Last, Hidden::PostNormAll),
            ],
            &[
                (1, &[4, 9, 2], Want::All, Hidden::PostNormAll),
                (2, &[1], Want::Last, Hidden::PostNorm),
            ],
        ];
        let sizing = Sizing {
            max_sequences: 3,
            kv_tokens: 64,
            max_rows: 16,
        };
        let mut batched = load_sized(model, 32, &sizing);
        let mut pools = KvPools::new(batched.kv_layout());
        let mut seqs: Vec<_> = (0..3).map(|_| pools.seq()).collect();
        let mut solo: Vec<Transformer> = (0..3).map(|_| load(model, 32)).collect();
        let mut lens = [0usize; 3];
        for (n, step) in steps.iter().enumerate() {
            for &(i, tokens, ..) in *step {
                pools.reserve(&mut seqs[i], lens[i] + tokens.len()).unwrap();
            }
            let batch = Batch {
                rows: step
                    .iter()
                    .map(|&(i, tokens, want, hidden)| super::super::SeqRows {
                        tokens,
                        start_pos: lens[i],
                        tables: &seqs[i].tables,
                        state: StateSlot(i),
                        want,
                        hidden,
                    })
                    .collect(),
                exact: true,
            };
            let outs = batched.forward_batch(&batch).unwrap();
            for (&(i, tokens, want, hidden), out) in step.iter().zip(&outs) {
                let alone = solo[i]
                    .forward_ext(tokens, lens[i], want, hidden, true)
                    .unwrap();
                assert_eq!(out.logits, alone.logits, "{what}: step {n}, sequence {i}");
                assert_eq!(out.hidden, alone.hidden, "{what}: step {n}, sequence {i}");
                lens[i] += tokens.len();
            }
        }
    }

    #[test]
    fn batched_rows_equal_solo_rows_for_every_architecture() {
        for spec in TinySpec::all() {
            check_batched(&write_spec(&spec), spec.arch);
        }
        check_batched(&write_tiny_model(), "gemma4");
        check_batched(&write_qwen35(&TinyQwen35::default()), "qwen35");
    }

    #[test]
    fn a_batched_step_is_validated_before_anything_runs() {
        use super::super::StateSlot;
        use super::super::paged::KvPools;
        use super::super::tiny::load_sized;
        let model = write_qwen35(&TinyQwen35::default());
        let sizing = Sizing {
            max_sequences: 2,
            kv_tokens: 64,
            max_rows: 16,
        };
        let mut m = load_sized(&model, 32, &sizing);
        let mut pools = KvPools::new(m.kv_layout());
        let mut a = pools.seq();
        pools.reserve(&mut a, 4).unwrap();
        let rows = |slot: usize, start: usize| super::super::SeqRows {
            tokens: &PROMPT[..2],
            start_pos: start,
            tables: &a.tables,
            state: StateSlot(slot),
            want: Want::Last,
            hidden: Hidden::None,
        };
        let run = |m: &mut Transformer, rows: Vec<super::super::SeqRows<'_>>| {
            m.forward_batch(&Batch { rows, exact: true })
        };
        assert!(run(&mut m, vec![]).is_err(), "empty batch");
        assert!(run(&mut m, vec![rows(2, 0)]).is_err(), "slot out of range");
        assert!(run(&mut m, vec![rows(0, 0), rows(0, 0)]).is_err(), "twice");
        assert!(run(&mut m, vec![rows(1, 3)]).is_err(), "wrong start");
        assert!(run(&mut m, vec![rows(1, 0)]).is_ok());
    }

    #[test]
    fn gqa_gemma4_model_satisfies_the_invariants() {
        let model = write_tiny_model_with(TinyOptions {
            n_head: 4,
            n_kv_head: 2,
            ..TinyOptions::default()
        });
        check_invariants(&model, "gemma4 gqa");
    }

    #[test]
    fn gemma4_loads_with_other_embedding_dtypes() {
        for (embd_dtype, ple_dtype) in [
            (GgmlDType::Q4_1, GgmlDType::Q5_0),
            (GgmlDType::Q5_1, GgmlDType::Q8_0),
            (GgmlDType::F16, GgmlDType::Q4_1),
        ] {
            let model = write_tiny_model_with(TinyOptions {
                embd_dtype,
                ple_dtype,
                ..TinyOptions::default()
            });
            let mut m = load(&model, 32);
            let l = logits(&mut m, &PROMPT, 0);
            assert!(
                l.iter().all(|v| v.is_finite()),
                "{embd_dtype:?}/{ple_dtype:?}"
            );
        }
    }

    #[test]
    fn forward_rejects_overflow_and_bad_input() {
        let model = write_tiny_model();
        let mut m = load(&model, 8);
        assert!(m.forward(&[], 0, Want::Last).is_err());
        assert!(
            m.forward(&PROMPT, 0, Want::Last).is_err(),
            "11 > 8 positions"
        );
        assert!(
            m.forward(&[3], 1, Want::Last).is_err(),
            "start != cache length"
        );
        assert!(
            m.forward(&[999], 0, Want::Last).is_err(),
            "token outside the vocabulary"
        );
    }

    #[test]
    fn load_rejects_context_length_above_the_model_maximum() {
        let model = write_tiny_model();
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        for n in [0, 33] {
            let e = Transformer::load(&content, &model.path, n).err().unwrap();
            assert!(matches!(e, CucaError::Config(_)), "{n}: {e}");
        }
    }

    /// A model of one family, stored with NORM rows, and the same weights with
    /// Q/K rows in half-split order loaded under a NEOX architecture, compute
    /// the same function: the two RoPE kinds are the same rotation in two
    /// layouts.
    #[test]
    fn norm_and_neox_rope_are_the_same_rotation_in_two_layouts() {
        let norm = TinySpec::llama();
        let mut neox = norm.clone();
        neox.arch = "qwen2";
        neox.neox_rows = true;
        let (a, b) = (write_spec(&norm), write_spec(&neox));
        let mut m_norm = load(&a, 32);
        let mut m_neox = load(&b, 32);
        assert_eq!(m_norm.hp.ropes[0].kind, RopeKind::Norm);
        assert_eq!(m_neox.hp.ropes[0].kind, RopeKind::Neox);
        assert_close(
            &logits(&mut m_norm, &PROMPT, 0),
            &logits(&mut m_neox, &PROMPT, 0),
        );
        // And a wrong pairing really is different.
        let mut wrong = norm.clone();
        wrong.arch = "qwen2";
        let c = write_spec(&wrong);
        let mut m_wrong = load(&c, 32);
        let (x, y) = (
            logits(&mut m_norm, &[4], 11),
            logits(&mut m_wrong, &PROMPT, 0),
        );
        let _ = x;
        let reference = logits(&mut load(&a, 32), &PROMPT, 0);
        let diff: f32 = reference.iter().zip(&y).map(|(p, q)| (p - q).abs()).sum();
        assert!(diff > 1e-2, "NEOX pairing of NORM rows must differ");
    }

    /// Fused `attn_qkv` / gate-up tensors hold the same values as separate
    /// ones and produce identical logits.
    #[test]
    fn fused_projections_match_separate_ones() {
        let fused = TinySpec::phi3();
        let mut split = fused.clone();
        split.fused_qkv = false;
        split.fused_gate_up = false;
        let (a, b) = (write_spec(&fused), write_spec(&split));
        assert_close(
            &logits(&mut load(&a, 32), &PROMPT, 0),
            &logits(&mut load(&b, 32), &PROMPT, 0),
        );
    }

    /// LongRoPE: the long factor list applies when the configured context
    /// exceeds the original context (16), the short list otherwise.
    #[test]
    fn longrope_picks_factors_by_context_length() {
        let model = write_spec(&TinySpec::phi3());
        let tokens = &PROMPT[..8];
        let long = logits(&mut load(&model, 32), tokens, 0);
        let short = logits(&mut load(&model, 16), tokens, 0);
        let diff: f32 = long.iter().zip(&short).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "long and short factor lists must differ");
    }

    #[test]
    fn biases_change_the_output() {
        let with = write_spec(&TinySpec::qwen2());
        let mut without = TinySpec::qwen2();
        without.bias = false;
        let without = write_spec(&without);
        let a = logits(&mut load(&with, 32), &PROMPT, 0);
        let b = logits(&mut load(&without, 32), &PROMPT, 0);
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-2);
        let mut m = load(&with, 32);
        match &m.layers[0].mixer {
            Mixer::Attention(AttnWeights {
                qkv:
                    QkvWeights::Split {
                        q,
                        kv: Some((k, v)),
                    },
                ..
            }) => {
                assert!(q.bias.is_some() && k.bias.is_some() && v.bias.is_some());
            }
            _ => panic!("qwen2 projections are separate"),
        }
        let _ = logits(&mut m, &PROMPT, 0);
    }

    #[test]
    fn linear_adds_its_bias() {
        let w = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let bias = Tensor::from_vec(vec![10f32, 20.0], 2, &Device::Cpu).unwrap();
        let lin = Linear {
            w: QMatMul::tensor(w),
            bias: Some(bias),
        };
        let x = Tensor::from_vec(vec![1f32, 2.0], (1, 1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, [11.0, 22.0]);
    }

    #[test]
    fn apply_rope_matches_the_direct_pair_formula() {
        let (heads, t, d, n_dims) = (2usize, 3usize, 8usize, 6usize);
        let values: Vec<f32> = (0..heads * t * d)
            .map(|i| ((i * 7 % 13) as f32) - 6.0)
            .collect();
        let x = Tensor::from_vec(values.clone(), (1, heads, t, d), &Device::Cpu).unwrap();
        let half = n_dims / 2;
        let cos_v: Vec<f32> = (0..t * half).map(|i| (i as f32 * 0.37).cos()).collect();
        let sin_v: Vec<f32> = (0..t * half).map(|i| (i as f32 * 0.37).sin()).collect();
        let cos = Tensor::from_vec(cos_v.clone(), (t, half), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(sin_v.clone(), (t, half), &Device::Cpu).unwrap();
        for kind in [RopeKind::Norm, RopeKind::Neox] {
            let got = apply_rope(&x, &cos, &sin, kind, n_dims)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for h in 0..heads {
                for p in 0..t {
                    let at = (h * t + p) * d;
                    let want = rotate_reference(
                        kind,
                        &values[at..at + d],
                        &cos_v[p * half..(p + 1) * half],
                        &sin_v[p * half..(p + 1) * half],
                    );
                    for (a, b) in got[at..at + d].iter().zip(&want) {
                        assert!(
                            (a - b).abs() < 1e-5,
                            "{kind:?} head {h} pos {p}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    /// The MoE FFN equals a per-token reference that runs every selected
    /// expert densely and sums with the routing weights.
    #[test]
    fn moe_ffn_matches_a_per_token_reference() {
        let model = write_spec(&TinySpec::granitemoe());
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let hp = read_hparams(&content).unwrap();
        let mut loader = Loader {
            content: &content,
            reader: BufReader::new(File::open(&model.path).unwrap()),
            dequantize: true,
            requantize: false,
        };
        let layout = LayerLayout::read(&hp, 0, &hp.layers[0], &content).unwrap();
        let lw = load_layer(&mut loader, &hp, 0, &hp.layers[0], &layout, None).unwrap();
        let t = 5;
        let values: Vec<f32> = (0..t * 64)
            .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.1)
            .collect();
        let f = Tensor::from_vec(values, (1, t, 64), &Device::Cpu).unwrap();
        let got = lw.ffn.forward(&f, Act::Silu).unwrap();
        let FfnWeights::Moe {
            spec,
            router,
            experts: ExpertBank::Resident(experts),
            shared: None,
        } = &lw.ffn
        else {
            panic!("granitemoe FFN is a mixture of resident experts");
        };
        let flat = f.reshape((t, 64)).unwrap();
        let logits = router
            .forward(&flat)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let routing = route(&logits, t, spec);
        let mut want = vec![0f32; t * 64];
        for (e, picks) in routing.per_expert.iter().enumerate() {
            for &(token, weight) in picks {
                let row = flat.narrow(0, token as usize, 1).unwrap();
                let h = (experts[e].gate.forward(&row).unwrap().silu().unwrap()
                    * experts[e].up.forward(&row).unwrap())
                .unwrap();
                let y = experts[e]
                    .down
                    .forward(&h)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (o, v) in want[token as usize * 64..][..64].iter_mut().zip(&y) {
                    *o += weight * v;
                }
            }
        }
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Every expert received a share of the tokens at least once overall,
        // and each token went to exactly two experts.
        let routed: usize = routing.per_expert.iter().map(Vec::len).sum();
        assert_eq!(routed, t * spec.n_used);
    }

    fn q35moe() -> TinyQwen35 {
        TinyQwen35 {
            moe: Some(TinyMoe {
                n_expert: 8,
                n_used: 2,
                ff: 64,
                shared_ff: 32,
            }),
            ..TinyQwen35::default()
        }
    }

    /// Qwen 3.5 MoE: the FFN is the routed experts (softmax over every
    /// expert, top-k, renormalised) plus the shared expert scaled by
    /// `sigmoid(ffn_gate_inp_shexp . x)`, checked against a per-token f64
    /// reference built from the raw tensors.
    #[test]
    fn qwen35moe_ffn_adds_the_gated_shared_expert() {
        let model = write_qwen35(&q35moe());
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap()))
                .unwrap();
        let hp = read_hparams(&content).unwrap();
        assert_eq!(hp.arch.name(), "qwen35moe");
        let moe = hp.moe.unwrap();
        assert!(moe.norm_weights);
        let mut loader = Loader {
            content: &content,
            reader: BufReader::new(File::open(&model.path).unwrap()),
            dequantize: true,
            requantize: false,
        };
        let layout = LayerLayout::read(&hp, 0, &hp.layers[0], &content).unwrap();
        assert_eq!(
            layout.ffn,
            Ffn::Moe {
                ff: 64,
                shared_ff: Some(32)
            }
        );
        let lw = load_layer(&mut loader, &hp, 0, &hp.layers[0], &layout, None).unwrap();
        let t = 6;
        let values: Vec<f32> = (0..t * 64)
            .map(|i| ((i * 29 % 19) as f32 - 9.0) * 0.12)
            .collect();
        let f = Tensor::from_vec(values.clone(), (1, t, 64), &Device::Cpu).unwrap();
        let got = lw.ffn.forward(&f, Act::Silu).unwrap();
        let FfnWeights::Moe {
            spec,
            router,
            experts: ExpertBank::Resident(experts),
            shared: Some(shared),
        } = &lw.ffn
        else {
            panic!("qwen35moe FFN is resident experts plus a shared expert");
        };
        let flat = f.reshape((t, 64)).unwrap();
        let vec_of = |x: Tensor| x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let routing = route(&vec_of(router.forward(&flat).unwrap()), t, spec);
        let mut want = vec![0f64; t * 64];
        for (e, picks) in routing.per_expert.iter().enumerate() {
            for &(token, weight) in picks {
                let row = flat.narrow(0, token as usize, 1).unwrap();
                let h = (experts[e].gate.forward(&row).unwrap().silu().unwrap()
                    * experts[e].up.forward(&row).unwrap())
                .unwrap();
                let y = vec_of(experts[e].down.forward(&h).unwrap());
                for (o, v) in want[token as usize * 64..][..64].iter_mut().zip(&y) {
                    *o += f64::from(weight) * f64::from(*v);
                }
            }
        }
        // The routing weights of a token sum to one.
        for token in 0..t as u32 {
            let sum: f32 = routing
                .per_expert
                .iter()
                .flatten()
                .filter(|(tk, _)| *tk == token)
                .map(|(_, w)| *w)
                .sum();
            assert!((sum - 1.0).abs() < 1e-5, "token {token}: {sum}");
        }
        for token in 0..t {
            let row = flat.narrow(0, token, 1).unwrap();
            let h = (shared.gate.forward(&row).unwrap().silu().unwrap()
                * shared.up.forward(&row).unwrap())
            .unwrap();
            let y = vec_of(shared.down.forward(&h).unwrap());
            let dot: f64 = values[token * 64..][..64]
                .iter()
                .zip(&shared.gate_inp)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let gate = 1.0 / (1.0 + (-dot).exp());
            for (o, v) in want[token * 64..][..64].iter_mut().zip(&y) {
                *o += gate * f64::from(*v);
            }
        }
        for (a, b) in vec_of(got).iter().zip(&want) {
            assert!((f64::from(*a) - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// The trunk loads without reading the prediction block's experts, and
    /// the in-trunk head of an MoE model fails loudly instead of silently.
    #[test]
    fn qwen35moe_loads_and_refuses_the_in_trunk_head() {
        let model = write_qwen35(&q35moe());
        let m = load(&model, 32);
        assert!(m.is_hybrid());
        assert_eq!((m.hp.n_layer, m.hp.nextn_predict_layers), (5, 1));
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap()))
                .unwrap();
        let e = m.load_head(&content, &model.path).err().unwrap();
        assert!(
            matches!(&e, CucaError::Config(msg) if msg.contains("qwen35moe") && msg.contains("mixture-of-experts")),
            "{e:?}"
        );
    }

    /// A hybrid MoE model under tiered placement computes the resident
    /// model's logits bit for bit, at the minimum budget and at one that
    /// holds a few experts per layer.
    #[test]
    fn qwen35moe_tiered_experts_match_resident_bit_for_bit() {
        let model = write_qwen35(&q35moe());
        let per_index = expert_bytes(&model);
        let want = greedy_run(&mut load(&model, 32));
        for experts in [2u64, 5] {
            let budget = experts * per_index;
            let mut tiered = load_tiered(&model, budget, None).unwrap();
            assert_eq!(greedy_run(&mut tiered), want, "{experts} experts");
            let stats = tiered.expert_tiers().unwrap().stats();
            assert!(stats.disk_reads > 0, "{stats:?}");
            assert!(stats.ram_resident_bytes <= budget, "{stats:?}");
        }
        let e = load_tiered(&model, 2 * per_index - 1, None).err().unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains(&format!("at least 2 ({} bytes)", 2 * per_index))),
            "{e:?}"
        );
    }

    #[test]
    fn kv_pool_bytes_follow_the_layout() {
        // At 32 positions every group holds 4 blocks (twice the context in
        // blocks of 16, a sliding group at most as many): the bytes are 64
        // positions of every K/V-owning layer.
        let per_position = |m: &Transformer| -> usize {
            m.hp.kv_owners()
                .map(|(_, l)| 2 * 4 * l.n_kv_head * l.head_dim)
                .sum()
        };
        // Gemma 4 tiny: 3 owners (two sliding, one full), one KV head.
        let m = load(&write_tiny_model(), 32);
        assert_eq!(m.kv_layout().blocks, [4, 4]);
        assert_eq!(m.kv_pool_bytes(), (64 * per_position(&m)) as u64);
        // Llama tiny: 4 layers x 2 KV heads x 16 dims.
        let m = load(&write_spec(&TinySpec::llama()), 32);
        assert_eq!(m.kv_layout().blocks, [4]);
        assert_eq!(m.kv_pool_bytes(), 64 * 4 * 2 * 4 * 2 * 16);
        // Qwen 3: head dim 32 differs from the embedding split.
        let m = load(&write_spec(&TinySpec::qwen3()), 32);
        assert_eq!(m.kv_pool_bytes(), 64 * 4 * 2 * 4 * 2 * 32);
        // Gemma 3 tiny: four sliding owners (window 4) and two full ones, one
        // KV head of 32 dims.
        let m = load(&write_spec(&TinySpec::gemma3()), 32);
        assert_eq!(m.kv_pool_bytes(), (64 * per_position(&m)) as u64);
        assert_eq!(m.kv_pool_bytes(), 2 * 4 * 32 * 64 * 6);
    }

    /// The Gemma 3 tiny model loaded for 1400 positions twice: the second
    /// keeps every block of its sliding layers (the reference a releasing
    /// sequence must match).
    fn releasing_and_unbounded() -> (Transformer, Transformer) {
        let model = write_spec(&TinySpec {
            context: 1400,
            ..TinySpec::gemma3()
        });
        let releasing = load(&model, 1400);
        let mut unbounded = load(&model, 1400);
        unbounded.solo.release = false;
        (releasing, unbounded)
    }

    /// Blocks of the sliding group that still hold a block, and the group
    /// of a model's sliding layers.
    fn live_sliding_blocks(m: &Transformer) -> usize {
        let g = m
            .kv_layout()
            .groups
            .iter()
            .position(|g| g.window.is_some())
            .unwrap();
        m.solo.tables()[g]
            .iter()
            .filter(|&&b| b != super::super::paged::NO_BLOCK)
            .count()
    }

    /// A sequence that releases the sliding blocks left behind its window
    /// computes what keeping every position computes, bit for bit; it holds
    /// a bounded number of blocks, rolls back within its verification rows
    /// and, asked for a length whose window was released, empties itself.
    #[test]
    fn sliding_blocks_released_behind_the_window_change_nothing() {
        let (mut releasing, mut unbounded) = releasing_and_unbounded();
        let tokens: Vec<u32> = (0..1350u32)
            .map(|i| (i * 7 + i / 5) % VOCAB as u32)
            .collect();
        let step = |m: &mut Transformer, chunk: &[u32], at: usize| {
            m.forward_ext(chunk, at, Want::Last, Hidden::None, true)
                .unwrap()
                .logits
                .unwrap()
        };
        for (i, chunk) in tokens.chunks(256).enumerate() {
            let at = i * 256;
            assert_eq!(
                step(&mut releasing, chunk, at),
                step(&mut unbounded, chunk, at),
                "chunk at {at}"
            );
        }
        // Window 4 needs three positions behind the end, rows of the
        // verification lag and the block they start in: a handful of blocks
        // of the 85 the whole sequence spans.
        assert!(live_sliding_blocks(&releasing) <= 4);
        assert_eq!(live_sliding_blocks(&unbounded), 1350usize.div_ceil(16));
        for (j, &token) in [3u32, 9, 1, 14, 6, 2, 11, 8, 5, 12].iter().enumerate() {
            let at = 1350 + j;
            assert_eq!(
                step(&mut releasing, &[token], at),
                step(&mut unbounded, &[token], at),
                "step at {at}"
            );
        }
        // A verification round's rollback stays inside what is held.
        assert_eq!(releasing.truncate_cache(1355), 1355);
        assert_eq!(unbounded.truncate_cache(1355), 1355);
        assert_eq!(
            step(&mut releasing, &[4], 1355),
            step(&mut unbounded, &[4], 1355)
        );
        assert_eq!(
            releasing.truncate_cache(100),
            0,
            "the window before 100 was released: emptied"
        );
        assert_eq!(unbounded.truncate_cache(100), 100);
        assert_eq!(live_sliding_blocks(&releasing), 0);
        // Refilled from scratch, the releasing model agrees again.
        for (i, chunk) in tokens[..101].chunks(64).enumerate() {
            step(&mut releasing, chunk, i * 64);
        }
        step(&mut unbounded, &tokens[100..101], 100);
        assert_eq!(
            step(&mut releasing, &[7], 101),
            step(&mut unbounded, &[7], 101)
        );
    }

    #[test]
    fn unknown_architecture_fails_the_load_by_name() {
        let mut spec = TinySpec::llama();
        spec.arch = "mamba2";
        let model = write_spec(&spec);
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let e = Transformer::load(&content, &model.path, 32).err().unwrap();
        assert!(e.to_string().contains("`mamba2`"), "{e}");
    }

    /// A tiny MoE model of `spec`'s family with eight experts (two used) in
    /// four layers.
    fn moe_spec(spec: TinySpec) -> TinySpec {
        TinySpec {
            moe: Some((8, 2)),
            n_layer: 4,
            ..spec
        }
    }

    /// Bytes of one expert summed over the MoE layers of `model`.
    fn expert_bytes(model: &super::super::tiny::TinyModel) -> u64 {
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap()))
                .unwrap();
        let hp = read_hparams(&content).unwrap();
        let n_expert = hp.moe.unwrap().n_expert;
        (0..hp.n_layer)
            .flat_map(|il| {
                [name::FFN_GATE_EXPS, name::FFN_UP_EXPS, name::FFN_DOWN_EXPS].map(|part| {
                    let info = tensor_info(&content, &name::blk(il, part)).unwrap();
                    let [_, rows, cols] = info.shape.dims() else {
                        panic!("expert tensors are 3-D");
                    };
                    ExpertSlice::read(&content, &name::blk(il, part), n_expert, *rows, *cols)
                        .unwrap()
                        .file_bytes as u64
                })
            })
            .sum()
    }

    fn load_tiered(
        model: &super::super::tiny::TinyModel,
        ram_bytes: u64,
        warm_from: Option<crate::provider::local::ExpertHeat>,
    ) -> Result<Transformer, CucaError> {
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap()))
                .unwrap();
        let tiers = ExpertTiers {
            ram_bytes,
            loader_threads: 2,
            warm_from,
            ..ExpertTiers::default()
        };
        Transformer::load_placed(
            &content,
            &model.path,
            32,
            &ExpertPlacement::Tiered(tiers),
            &Sizing::single(32),
        )
    }

    /// A 24-token prompt in chunks of 16 then 8 greedy steps: the logits of
    /// every call.
    fn greedy_run(m: &mut Transformer) -> Vec<Vec<f32>> {
        let prompt: Vec<u32> = (0..24u32).map(|i| (i * 5 + 3) % VOCAB as u32).collect();
        let mut out: Vec<Vec<f32>> = Vec::new();
        let mut at = 0;
        for chunk in prompt.chunks(16) {
            let f = m.forward_ext(chunk, at, Want::Last, Hidden::None, true);
            out.push(f.unwrap().logits.unwrap());
            at += chunk.len();
        }
        for _ in 0..8 {
            let last = out.last().unwrap();
            let next = (0..last.len())
                .max_by(|&a, &b| last[a].total_cmp(&last[b]))
                .unwrap() as u32;
            let f = m.forward_ext(&[next], at, Want::Last, Hidden::None, true);
            out.push(f.unwrap().logits.unwrap());
            at += 1;
        }
        out
    }

    /// Placement only decides speed: at the smallest RAM budget (two of
    /// eight experts per layer, so nearly every routing evicts) the logits
    /// equal the resident load's bit for bit, for every MoE architecture.
    #[test]
    fn tiered_experts_match_resident_bit_for_bit() {
        for spec in [
            TinySpec::granitemoe(),
            TinySpec::qwen3moe(),
            TinySpec::olmoe(),
        ] {
            let arch = spec.arch;
            let model = write_spec(&moe_spec(spec));
            let min = 2 * expert_bytes(&model);
            let want = greedy_run(&mut load(&model, 32));
            let mut tiered = load_tiered(&model, min, None).unwrap();
            assert_eq!(greedy_run(&mut tiered), want, "{arch}");
            let stats = tiered.expert_tiers().unwrap().stats();
            assert!(stats.disk_reads > 0, "{arch}: {stats:?}");
            assert!(stats.ram_resident_bytes <= min, "{arch}: {stats:?}");
            assert_eq!(
                stats.lookups,
                stats.ram_hits + stats.disk_reads,
                "{arch}: {stats:?}"
            );
        }
    }

    #[test]
    fn tiered_load_rejects_a_budget_below_n_used() {
        let model = write_spec(&moe_spec(TinySpec::granitemoe()));
        let min = 2 * expert_bytes(&model);
        let e = load_tiered(&model, min - 1, None).err().unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains(&format!("at least 2 ({min} bytes)"))),
            "{e:?}"
        );
        let dense = write_spec(&TinySpec::llama());
        let e = load_tiered(&dense, 1 << 30, None).err().unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("`llama` has no experts")),
            "{e:?}"
        );
    }

    /// Heat exported from one run, fed to the next load, preloads the hot
    /// experts: the first forward pass reads fewer from the file.
    #[test]
    fn warm_from_loads_the_hottest_experts_first() {
        let model = write_spec(&moe_spec(TinySpec::granitemoe()));
        let budget = 4 * expert_bytes(&model);
        let first_pass_reads = |m: &mut Transformer| {
            let before = m.expert_tiers().unwrap().stats().disk_reads;
            let prompt: Vec<u32> = (0..24u32).map(|i| (i * 5 + 3) % VOCAB as u32).collect();
            m.forward_ext(&prompt[..16], 0, Want::Last, Hidden::None, true)
                .unwrap();
            m.expert_tiers().unwrap().stats().disk_reads - before
        };
        let mut cold = load_tiered(&model, budget, None).unwrap();
        let cold_reads = first_pass_reads(&mut cold);
        cold.truncate_cache(0);
        greedy_run(&mut cold);
        let heat = cold.expert_tiers().unwrap().heat();
        drop(cold);
        let mut warm = load_tiered(&model, budget, Some(heat)).unwrap();
        assert!(warm.expert_tiers().unwrap().stats().ram_resident_bytes > 0);
        let warm_reads = first_pass_reads(&mut warm);
        assert!(warm_reads < cold_reads, "{warm_reads} vs {cold_reads}");

        let other: crate::provider::local::ExpertHeat =
            serde_json::from_str(r#"{"n_layer":1,"n_expert":8,"counts":[1,0,0,0,0,0,0,0]}"#)
                .unwrap();
        let e = load_tiered(&model, budget, Some(other)).err().unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("1x8") && m.contains("4x8")),
            "{e:?}"
        );
    }
}

/// The hybrid (gated delta-net) architecture: arithmetic against direct f64
/// references, bit-exact batching, and the cache snapshot semantics.
#[cfg(test)]
mod hybrid_tests {
    use candle_core::quantized::gguf_file::Value;

    use super::super::delta::tests::{Inputs, reference};
    use super::super::paged::KvPools;
    use super::super::tiny::{
        CONTEXT, Lcg, N_EMBD, PROMPT, Q35_DK, Q35_DV, Q35_HEAD, Q35_N_HEAD, Q35_N_KV,
        Q35_ROPE_DIMS, TinyModel, TinyQwen35, TinySpec, VOCAB, load, write_qwen35, write_spec,
    };
    use super::*;

    pub(super) fn content_of(model: &TinyModel) -> gguf_file::Content {
        gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap())).unwrap()
    }

    /// The model with every matmul weight dequantized to f32.
    pub(super) fn load_dequantized(model: &TinyModel) -> Transformer {
        Transformer::load_with(
            &content_of(model),
            &model.path,
            CONTEXT,
            true,
            false,
            &ExpertPlacement::Resident,
            &Sizing::single(CONTEXT),
        )
        .unwrap()
    }

    /// Run `tokens` at `start` without asking for logits.
    fn feed(m: &mut Transformer, tokens: &[u32], start: usize) {
        m.forward_ext(tokens, start, Want::None, Hidden::None, true)
            .unwrap();
    }

    /// Run `tokens` at `start` and return the logits `want` selects.
    fn step(m: &mut Transformer, tokens: &[u32], start: usize, want: Want) -> Vec<f32> {
        m.forward_ext(tokens, start, want, Hidden::None, true)
            .unwrap()
            .logits
            .unwrap()
    }

    /// One sequence's rows `0..t` at slot 0, as a forward step builds it.
    fn solo_seg(t: usize, tables: &[BlockTable]) -> Seg<'_> {
        Seg {
            start: 0,
            offset: 0,
            t,
            tables,
            slot: 0,
            record: false,
            want: Want::None,
            hidden: Hidden::None,
        }
    }

    /// A model that ran `tokens` one at a time (the cold reference).
    fn cold(model: &TinyModel, tokens: &[u32]) -> Transformer {
        let mut m = load(model, CONTEXT);
        for (i, t) in tokens.iter().enumerate() {
            feed(&mut m, &[*t], i);
        }
        m
    }

    /// Every delta-net layer's live state (history, then matrices).
    fn live(m: &Transformer) -> Vec<Vec<f32>> {
        m.delta_layers()
            .map(|dw| {
                dw.with_cell(0, |c| {
                    let s = c.live();
                    [s.history(), s.matrices()].concat()
                })
            })
            .collect()
    }

    /// Floats held by every delta-net layer's state and snapshots.
    fn held(m: &Transformer) -> usize {
        m.delta_layers()
            .map(|dw| dw.with_cell(0, |c| c.held_floats()))
            .sum()
    }

    pub(super) fn tensor(model: &TinyModel, name: &str) -> Vec<f64> {
        let content = content_of(model);
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        content
            .tensor(&mut reader, name, &Device::Cpu)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// `w` is `rows x cols` row-major.
    pub(super) fn matvec(w: &[f64], cols: usize, x: &[f64]) -> Vec<f64> {
        w.chunks(cols)
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn the_trunk_loads_and_the_prediction_block_stays_in_the_file() {
        let model = write_qwen35(&TinyQwen35::default());
        let m = load(&model, CONTEXT);
        assert!(m.is_hybrid());
        assert_eq!((m.hp.n_layer, m.hp.nextn_predict_layers), (5, 1));
        let kinds: Vec<bool> = m
            .layers
            .iter()
            .map(|l| matches!(l.mixer, Mixer::Delta(_)))
            .collect();
        assert_eq!(kinds, [true, true, true, false, true]);
        // Block 5 is not a trunk layer, but its tensors stay loadable.
        let content = content_of(&model);
        let mut loader = Loader {
            content: &content,
            reader: BufReader::new(File::open(&model.path).unwrap()),
            dequantize: false,
            requantize: false,
        };
        loader
            .matmul(&name::blk(5, name::NEXTN_EH_PROJ), &[N_EMBD, 2 * N_EMBD])
            .unwrap();
        for part in [
            name::NEXTN_ENORM,
            name::NEXTN_HNORM,
            name::NEXTN_SHARED_HEAD_NORM,
        ] {
            loader.vector(&name::blk(5, part), N_EMBD).unwrap();
        }
        // The logits are finite and depend on the prompt.
        let mut m = load(&model, CONTEXT);
        let a = step(&mut m, &PROMPT, 0, Want::Last);
        assert_eq!(a.len(), VOCAB);
        assert!(a.iter().all(|v| v.is_finite()));
        let b = step(&mut load(&model, CONTEXT), &PROMPT[..8], 0, Want::Last);
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    /// The tiny model's delta-net layer, run on random rows, equals a direct
    /// f64 implementation of Qwen3_5GatedDeltaNet (projections, gates,
    /// convolution, l2 norms, the delta rule, the gated norm, `ssm_out`) on
    /// the same dequantized weights; so do the final convolution history and
    /// state matrices.
    #[test]
    fn delta_layer_matches_an_f64_reference() {
        let model = write_qwen35(&TinyQwen35::default());
        let m = load_dequantized(&model);
        let Mixer::Delta(dw) = &m.layers[1].mixer else {
            panic!("layer 1 is a delta-net layer");
        };
        let spec = *m.hp.layers[1].delta().unwrap();
        let (vd, nv) = (spec.value_dim(), spec.n_v_heads);
        let t = 7;
        let h = Lcg(77).values(t * N_EMBD, 1.0);

        let blk = |part: &str| name::blk(1, part);
        let wqkv = tensor(&model, &blk(name::ATTN_QKV));
        let wz = tensor(&model, &blk(name::ATTN_GATE));
        let wb = tensor(&model, &blk(name::SSM_BETA));
        let wa = tensor(&model, &blk(name::SSM_ALPHA));
        let wo = tensor(&model, &blk(name::SSM_OUT));
        let a_neg = tensor(&model, &name::blk_bare(1, name::SSM_A));
        let dt_bias = tensor(&model, &name::blk_bias(1, name::SSM_DT));
        let conv: Vec<f32> = tensor(&model, &blk(name::SSM_CONV1D))
            .iter()
            .map(|&v| v as f32)
            .collect();

        let (mut mixed, mut z, mut beta, mut decay) = (vec![], vec![], vec![], vec![]);
        for row in h.chunks(N_EMBD) {
            let x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            mixed.extend(matvec(&wqkv, N_EMBD, &x).iter().map(|&v| v as f32));
            z.extend(matvec(&wz, N_EMBD, &x).iter().map(|&v| v as f32));
            // beta = sigmoid(ssm_beta x); g = ssm_a * softplus(ssm_alpha x + dt_bias).
            for b in matvec(&wb, N_EMBD, &x) {
                beta.push((1.0 / (1.0 + (-b).exp())) as f32);
            }
            for (head, a) in matvec(&wa, N_EMBD, &x).iter().enumerate() {
                let pre = a + dt_bias[head];
                let softplus = if pre > 20.0 { pre } else { pre.exp().ln_1p() };
                decay.push((a_neg[head] * softplus).exp() as f32);
            }
        }
        assert_eq!((beta.len(), decay.len()), (t * nv, t * nv));
        let inputs = Inputs {
            mixed,
            z,
            beta,
            decay,
        };
        let (o, want_s, want_hist) = reference(&spec, &dw.kernel, &conv, &inputs, t);
        let want: Vec<f64> = o.chunks(vd).flat_map(|r| matvec(&wo, vd, r)).collect();

        let input = Tensor::from_vec(h, (1, t, N_EMBD), &Device::Cpu).unwrap();
        let got = dw
            .forward_segs(&input, &[solo_seg(t, &[])])
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-4 * scale,
                "output {i}: {g} vs {w}"
            );
        }
        let (matrices, history) = dw.with_cell(0, |c| {
            let s = c.live();
            (s.matrices().to_vec(), s.history().to_vec())
        });
        for (what, got, want) in [("state", matrices, want_s), ("history", history, want_hist)] {
            assert_eq!(got.len(), want.len(), "{what}");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(*g) - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "{what} {i}: {g} vs {w}"
                );
            }
        }
    }

    /// The tiny model's gated attention layer equals a direct f64 attention:
    /// the `[query | gate]` per-head packing, per-head QK-RMSNorm (weight
    /// multiplies directly), partial NEOX rotary over the first
    /// `rope.dimension_count` dimensions, grouped-query causal attention, and
    /// the output gated by `sigmoid(gate)`.
    #[test]
    fn gated_attention_layer_matches_an_f64_reference() {
        let model = write_qwen35(&TinyQwen35::default());
        let m = load_dequantized(&model);
        let Mixer::Attention(aw) = &m.layers[3].mixer else {
            panic!("layer 3 is the attention layer");
        };
        let (nh, nkv, d, rd) = (Q35_N_HEAD, Q35_N_KV, Q35_HEAD, Q35_ROPE_DIMS);
        let t = 6;
        let h = Lcg(78).values(t * N_EMBD, 1.0);
        let w = |part: &str| tensor(&model, &name::blk(3, part));
        let (wq, wk, wv, wo) = (
            w(name::ATTN_Q),
            w(name::ATTN_K),
            w(name::ATTN_V),
            w(name::ATTN_OUTPUT),
        );
        let (qn, kn) = (w(name::ATTN_Q_NORM), w(name::ATTN_K_NORM));
        let eps = 1e-6f64;
        let rms = |v: &[f64], weight: &[f64]| -> Vec<f64> {
            let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64 + eps).sqrt();
            v.iter().zip(weight).map(|(x, w)| x * inv * w).collect()
        };
        // NEOX over the first `rd` values: pair (i, i + rd / 2).
        let rotate = |v: &mut [f64], pos: usize| {
            for i in 0..rd / 2 {
                let angle = pos as f64 * 1e7f64.powf(-(2.0 * i as f64) / rd as f64);
                let (c, s) = (angle.cos(), angle.sin());
                let (a, b) = (v[i], v[i + rd / 2]);
                v[i] = a * c - b * s;
                v[i + rd / 2] = b * c + a * s;
            }
        };
        let mut qs = vec![vec![0f64; nh * d]; t];
        let mut gates = vec![vec![0f64; nh * d]; t];
        let mut ks = vec![vec![0f64; nkv * d]; t];
        let mut vs = vec![vec![0f64; nkv * d]; t];
        for (pos, row) in h.chunks(N_EMBD).enumerate() {
            let x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let packed = matvec(&wq, N_EMBD, &x);
            for head in 0..nh {
                let at = head * 2 * d;
                let mut q = rms(&packed[at..at + d], &qn);
                rotate(&mut q, pos);
                qs[pos][head * d..(head + 1) * d].copy_from_slice(&q);
                gates[pos][head * d..(head + 1) * d].copy_from_slice(&packed[at + d..at + 2 * d]);
            }
            let (k, v) = (matvec(&wk, N_EMBD, &x), matvec(&wv, N_EMBD, &x));
            for head in 0..nkv {
                let mut kh = rms(&k[head * d..(head + 1) * d], &kn);
                rotate(&mut kh, pos);
                ks[pos][head * d..(head + 1) * d].copy_from_slice(&kh);
            }
            vs[pos].copy_from_slice(&v);
        }
        let mut want = Vec::new();
        for pos in 0..t {
            let mut o = vec![0f64; nh * d];
            for head in 0..nh {
                let kv = head / (nh / nkv);
                let q = &qs[pos][head * d..(head + 1) * d];
                let scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        let k = &ks[j][kv * d..(kv + 1) * d];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() / (d as f64).sqrt()
                    })
                    .collect();
                let top = scores.iter().fold(f64::MIN, |a, &b| a.max(b));
                let exps: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (j, e) in exps.iter().enumerate() {
                    for c in 0..d {
                        o[head * d + c] += e / sum * vs[j][kv * d + c];
                    }
                }
            }
            for (x, g) in o.iter_mut().zip(&gates[pos]) {
                *x *= 1.0 / (1.0 + (-g).exp());
            }
            want.extend(matvec(&wo, nh * d, &o));
        }

        let input = Tensor::from_vec(h, (1, t, N_EMBD), &Device::Cpu).unwrap();
        let mut pools = KvPools::new(m.kv_layout());
        let mut seq = pools.seq();
        pools.reserve(&mut seq, t).unwrap();
        let got = m
            .attention(3, aw, &input, &[solo_seg(t, &seq.tables)], None)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-4 * scale,
                "output {i}: {g} vs {w}"
            );
        }
    }

    /// A prefill in any chunking, with any rows requested, equals token by
    /// token: the same logits and hidden rows and the same recurrent state,
    /// bit for bit.
    #[test]
    fn prefill_in_any_chunking_equals_token_by_token_bit_for_bit() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut stepped = load(&model, CONTEXT);
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut hidden: Vec<Vec<f32>> = Vec::new();
        for (i, t) in PROMPT.iter().enumerate() {
            let f = stepped
                .forward_ext(&[*t], i, Want::Last, Hidden::PostNorm, true)
                .unwrap();
            rows.push(f.logits.unwrap());
            hidden.push(f.hidden.unwrap());
        }
        assert!(rows[10].iter().any(|v| v.abs() > 1e-3));

        for chunks in [
            vec![11],
            vec![4, 5, 2],
            vec![1, 10],
            vec![10, 1],
            vec![3, 3, 3, 2],
        ] {
            let mut m = load(&model, CONTEXT);
            let (mut at, mut got, mut got_hidden) = (0, Vec::new(), Vec::new());
            for n in &chunks {
                let f = m
                    .forward_ext(&PROMPT[at..at + n], at, Want::All, Hidden::PostNorm, true)
                    .unwrap();
                got.extend(f.logits.unwrap());
                got_hidden.extend(f.hidden.unwrap());
                at += n;
            }
            assert_eq!(got, rows.concat(), "chunks {chunks:?}");
            assert_eq!(got_hidden, hidden.concat(), "chunks {chunks:?}");
            assert_eq!(live(&m), live(&stepped), "chunks {chunks:?}");
        }

        // Only the last row requested (what prefill does).
        let mut last = load(&model, CONTEXT);
        feed(&mut last, &PROMPT[..6], 0);
        assert_eq!(step(&mut last, &PROMPT[6..], 6, Want::Last), rows[10]);
    }

    /// A rejected call (a token outside the vocabulary, a wrong start) is
    /// refused before any state moves: the model carries on exactly as a
    /// cold one.
    #[test]
    fn a_rejected_call_leaves_the_state_untouched() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut m = load(&model, CONTEXT);
        feed(&mut m, &PROMPT[..4], 0);
        assert!(m.forward(&[999], 4, Want::Last).is_err());
        assert!(m.forward(&PROMPT[4..], 5, Want::Last).is_err());
        assert_eq!(m.cache_len(), 4);
        assert_eq!(live(&m), live(&cold(&model, &PROMPT[..4])));
    }

    #[test]
    fn truncate_returns_the_retained_length_and_attention_models_keep_min() {
        let model = write_spec(&TinySpec::llama());
        let mut m = load(&model, CONTEXT);
        assert!(!m.is_hybrid());
        assert_eq!(m.rollback_rows(), usize::MAX);
        step(&mut m, &PROMPT, 0, Want::Last);
        assert_eq!(m.truncate_cache(99), 11);
        assert_eq!(m.truncate_cache(11), 11);
        assert_eq!(m.truncate_cache(6), 6);
        assert_eq!(m.cache_len(), 6);
        assert_eq!(m.truncate_cache(0), 0);
        assert_eq!(m.truncate_cache(5), 0);
    }

    /// A hybrid model can only return to its end (or into the rows of a
    /// recorded call) and falls back to zero for anything else; a snapshot
    /// brings back a position bit for bit.
    #[test]
    fn rewind_falls_back_to_zero_and_a_snapshot_restores_a_position() {
        let model = write_qwen35(&TinyQwen35::default());
        let want_full = step(
            &mut cold(&model, &PROMPT[..10]),
            &PROMPT[10..],
            10,
            Want::Last,
        );

        // Anything below the end restarts from zero.
        let mut m = load(&model, CONTEXT);
        step(&mut m, &PROMPT, 0, Want::Last);
        assert_eq!(m.rollback_rows(), MAX_VERIFY_ROWS);
        assert_eq!(m.truncate_cache(11), 11);
        assert_eq!(m.truncate_cache(40), 11);
        assert_eq!(m.truncate_cache(10), 0);
        assert_eq!(m.cache_len(), 0);
        assert!(live(&m).iter().flatten().all(|&v| v == 0.0));

        // A snapshot after the 6-token prefix brings that position back,
        // keys and values of the first six positions untouched.
        let mut m = load(&model, CONTEXT);
        feed(&mut m, &PROMPT[..6], 0);
        let snapshot = m.snapshot(0).unwrap();
        step(&mut m, &PROMPT[6..], 6, Want::Last);
        assert_eq!(m.rewind(0, 6), 0, "the state itself cannot go back");
        m.restore(0, &snapshot, 6).unwrap();
        assert_eq!(m.solo.truncate(6), 6);
        assert_eq!(live(&m), live(&cold(&model, &PROMPT[..6])));
        // Replaying from there reproduces the cold run bit for bit.
        feed(&mut m, &PROMPT[6..10], 6);
        assert_eq!(step(&mut m, &PROMPT[10..], 10, Want::Last), want_full);
        // The snapshot is a copy: it restores again, and a foreign one or a
        // slot out of range is refused.
        m.rewind(0, 6);
        m.restore(0, &snapshot, 6).unwrap();
        assert_eq!(live(&m), live(&cold(&model, &PROMPT[..6])));
        assert!(m.restore(1, &snapshot, 6).is_err(), "one slot only");
        assert!(m.restore(0, &snapshot[1..], 6).is_err(), "wrong layers");
    }

    /// After a `Want::All` call over several rows every intermediate length
    /// can be returned to, the state is the one a cold run has there, and
    /// the logits that follow are bit-identical to the cold run's.
    #[test]
    fn verify_rows_roll_back_to_every_prefix_bit_for_bit() {
        let model = write_qwen35(&TinyQwen35::default());
        let first = 4;
        let mut stepped = load(&model, CONTEXT);
        let mut per_row = Vec::new();
        for (i, t) in PROMPT.iter().enumerate() {
            per_row.push(step(&mut stepped, &[*t], i, Want::Last));
        }
        for keep in first..PROMPT.len() {
            let mut m = load(&model, CONTEXT);
            feed(&mut m, &PROMPT[..first], 0);
            let all = step(&mut m, &PROMPT[first..], first, Want::All);
            // The rows themselves are the single-step logits.
            assert_eq!(all, per_row[first..].concat(), "rows, keep {keep}");

            assert_eq!(m.truncate_cache(keep), keep, "keep {keep}");
            assert_eq!(m.cache_len(), keep);
            let mut reference = cold(&model, &PROMPT[..keep]);
            assert_eq!(live(&m), live(&reference), "state, keep {keep}");
            assert_eq!(
                step(&mut m, &PROMPT[keep..], keep, Want::Last),
                step(&mut reference, &PROMPT[keep..], keep, Want::Last),
                "next logits, keep {keep}"
            );
        }

        // The record survives rollbacks: back to the call's start in any
        // order, until the next forward call replaces it.
        let mut m = load(&model, CONTEXT);
        feed(&mut m, &PROMPT[..first], 0);
        step(&mut m, &PROMPT[first..], first, Want::All);
        assert_eq!(m.truncate_cache(9), 9);
        assert_eq!(m.truncate_cache(6), 6);
        assert_eq!(m.truncate_cache(first), first, "the pre-call state");
        assert_eq!(live(&m), live(&cold(&model, &PROMPT[..first])));
        step(&mut m, &PROMPT[first..8], first, Want::All);
        assert_eq!(m.truncate_cache(6), 6);
        assert_eq!(live(&m), live(&cold(&model, &PROMPT[..6])));
    }

    #[test]
    fn the_verify_record_ends_with_the_next_call_and_is_bounded() {
        let model = write_qwen35(&TinyQwen35::default());
        let tokens: Vec<u32> = (0..24).map(|i| (i % 15 + 1) as u32).collect();

        // Another forward call ends the record.
        let mut m = load(&model, CONTEXT);
        feed(&mut m, &tokens[..3], 0);
        step(&mut m, &tokens[3..8], 3, Want::All);
        step(&mut m, &tokens[8..9], 8, Want::Last);
        assert_eq!(m.truncate_cache(5), 0, "the record ended");

        // A call over one row records nothing (there is no inside).
        let mut m = load(&model, CONTEXT);
        feed(&mut m, &tokens[..3], 0);
        step(&mut m, &tokens[3..4], 3, Want::All);
        assert_eq!(m.truncate_cache(3), 0);

        // The bound: 17 rows are restorable, 18 are not.
        for (rows, restorable) in [(MAX_VERIFY_ROWS, true), (MAX_VERIFY_ROWS + 1, false)] {
            let mut m = load(&model, CONTEXT);
            feed(&mut m, &tokens[..2], 0);
            step(&mut m, &tokens[2..2 + rows], 2, Want::All);
            let kept = m.truncate_cache(10);
            assert_eq!(kept, if restorable { 10 } else { 0 }, "{rows} rows");
            if restorable {
                assert_eq!(live(&m), live(&cold(&model, &tokens[..10])));
            }
        }

        // Below the record there is no state to return to, inside it the
        // record answers.
        let prepare = || {
            let mut m = load(&model, CONTEXT);
            feed(&mut m, &tokens[..3], 0);
            feed(&mut m, &tokens[3..5], 3);
            step(&mut m, &tokens[5..10], 5, Want::All);
            m
        };
        let mut m = prepare();
        assert_eq!(m.truncate_cache(4), 0, "below the record: zero");
        assert!(live(&m).iter().flatten().all(|&v| v == 0.0));
        let mut m = prepare();
        assert_eq!(m.truncate_cache(7), 7, "inside the record");
        assert_eq!(live(&m), live(&cold(&model, &tokens[..7])));
    }

    #[test]
    fn memory_follows_the_documented_formulas() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut m = load(&model, CONTEXT);
        let spec = *m.hp.layers[0].delta().unwrap();
        assert_eq!(
            (spec.conv_dim(), spec.key_dim(), spec.value_dim()),
            (96, 16, 64)
        );
        // (conv_kernel - 1) x conv_dim history plus n_v matrices dk x dv.
        assert_eq!(spec.state_floats(), 3 * 96 + 4 * Q35_DK * Q35_DV);
        assert_eq!(m.hp.recurrent_state_bytes(), 4 * 4 * spec.state_floats());
        // One attention layer (layer 3) owns K/V.
        // 4 blocks (twice the 32 positions, 16 each) of one layer's rows.
        assert_eq!(
            m.kv_pool_bytes(),
            (4 * 16 * 2 * 4 * Q35_N_KV * Q35_HEAD) as u64
        );
        assert_eq!(m.hp.kv_owners().count(), 1);

        let state = 4 * spec.state_floats();
        let per_row = spec.conv_dim() + 2 * spec.n_v_heads;
        assert_eq!(held(&m), state, "only the live state at first");
        feed(&mut m, &PROMPT[..5], 0);
        assert_eq!(held(&m), state);
        let rows = 4;
        step(&mut m, &PROMPT[5..5 + rows], 5, Want::All);
        // Plus the pre-call copy and `conv_dim + 2 n_v` floats per row.
        assert_eq!(held(&m), 2 * state + 4 * rows * per_row);
        // Nothing grows with the context: the next record reuses the buffers.
        step(&mut m, &PROMPT[9..11], 9, Want::All);
        assert_eq!(held(&m), 2 * state + 4 * 2 * per_row);
    }

    #[test]
    fn inconsistent_hybrid_files_fail_the_load_naming_the_cause() {
        let load_err = |s: TinyQwen35| {
            let model = write_qwen35(&s);
            let content = content_of(&model);
            Transformer::load(&content, &model.path, CONTEXT)
                .err()
                .unwrap()
                .to_string()
        };
        // The convolution tensor holds 4 taps, the metadata says 3.
        let msg = load_err(TinyQwen35 {
            meta: vec![("ssm.conv_kernel".into(), Value::U32(3))],
            ..TinyQwen35::default()
        });
        assert!(
            msg.contains("ssm_conv1d") && msg.contains("expected shape"),
            "{msg}"
        );
        // Interval 5 makes layer 3 a delta-net layer, which has attention
        // tensors only.
        let msg = load_err(TinyQwen35 {
            meta: vec![("full_attention_interval".into(), Value::U32(5))],
            ..TinyQwen35::default()
        });
        assert!(msg.contains("blk.3.ssm_conv1d.weight"), "{msg}");
        // A value-head count the tensors do not match.
        let msg = load_err(TinyQwen35 {
            meta: vec![("ssm.time_step_rank".into(), Value::U32(8))],
            ..TinyQwen35::default()
        });
        assert!(msg.contains("expected shape"), "{msg}");
    }

    #[test]
    fn equal_head_counts_and_a_two_tap_convolution_work_too() {
        // As many key heads as value heads (the 0.8B shape) and a 2-tap
        // convolution: the generic code path, not just the defaults.
        let spec = TinyQwen35 {
            n_k_heads: 4,
            n_v_heads: 4,
            conv_kernel: 2,
            ..TinyQwen35::default()
        };
        let model = write_qwen35(&spec);
        let mut whole = load(&model, CONTEXT);
        let rows = step(&mut whole, &PROMPT, 0, Want::All);
        let mut stepped = load(&model, CONTEXT);
        let mut want = Vec::new();
        for (i, t) in PROMPT.iter().enumerate() {
            want.extend(step(&mut stepped, &[*t], i, Want::Last));
        }
        assert_eq!(rows, want);
        assert!(rows.iter().all(|v| v.is_finite()));
    }
}
