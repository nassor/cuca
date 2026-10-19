//! Gemma 4 dense forward pass on a GPU through burn (CubeCL), loaded from the
//! same GGUF file as the CPU engine.
//!
//! [`GpuGemma4`] mirrors [`Gemma4`](super::Gemma4) operation for operation:
//! the same hyperparameters ([`Gemma4Params`]), the same RoPE tables, masks,
//! prefill chunking and K/V sharing, and the same architecture invariants
//! (RMSNorm multiplies by `w` with no `1 + w`, attention scale 1.0, a
//! weightless value norm, NEOX half-split RoPE, sliding layers reading
//! `start - (window - 1)..`, per-layer embeddings scaled by `sqrt(ple_dim)`
//! and merged as `(ctx + tok) / sqrt(2)`, shared-K/V layers reading the cache
//! of `kv_source[il]`, logits soft-capped as `30 * tanh(x / 30)`). GELU is
//! the tanh approximation, as in candle.
//!
//! # Weights
//!
//! Matmul weights must be Q4_0, Q8_0, F32, F16 or BF16. Q4_0 and Q8_0 are
//! repacked once at load into burn's block-quantized layout (`Q4S` / `Q8S`
//! values packed into `u32` words, one f16 scale per 32-element block, see
//! [`pack_blocks`]) and stay quantized on the device: burn's matmul reads
//! them directly and dequantizes inside the kernel. Float weights are
//! uploaded as f32. Any other dtype (K-quants, Q4_1, Q5_x) fails the load
//! with an error naming the tensor; the CPU device reads those.
//!
//! # Memory
//!
//! - Device memory holds the matmul weights at their file size, except
//!   `per_layer_model_proj` (BF16 in the published files), held as f32 (92 MB
//!   for E2B), and the tied token embedding, uploaded once as the LM head
//!   (Q8_0 in the E2B Q4_0 file, 427 MB). Norm vectors are f32.
//! - Embedding lookups never touch the device copy: the host reads each
//!   token's row of `token_embd` and of `per_layer_token_embd` from the GGUF
//!   file ([`RowTable`]) and uploads the dequantized rows. The per-layer
//!   table is never uploaded.
//! - The K/V cache lives on the device, preallocated for `context_length`
//!   positions with the CPU formula, `context_length * 2 * 4 *
//!   sum(n_kv_head * head_dim)` bytes over the K/V-owning layers (E2B:
//!   `context_length * 36,864`). Writes are in place: a layer takes its
//!   cache tensors out of the cache, assigns the new positions, and puts
//!   them back, and attention views are dropped before the next write, so no
//!   write copies the cache.
//! - RoPE cos/sin tables for `context_length` positions, one pair per
//!   attention type, live on the device.
//! - The host side is one reused read buffer of the largest weight during
//!   load, and per forward pass the embedding rows and the attention masks.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "provider-local-gpu copies GGUF little-endian f16 scales into burn's native-endian layout"
);

use std::f64::consts::FRAC_1_SQRT_2;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{DType, Device, Tensor, TensorData, activation};
use candle_core::quantized::{GgmlDType, gguf_file};

use super::{
    Gemma4Params, RowTable, attention_mask, dequant_row, err, name, rope_cos_sin, rope_inv_freq,
    tensor_info,
};
use crate::error::CucaError;
use crate::provider::local::{LocalDevice, device_unavailable};

/// burn's quantization scheme equivalent to a GGUF block dtype, `None` for
/// every dtype the GPU engine does not keep quantized.
fn quant_scheme(dtype: GgmlDType) -> Option<QuantScheme> {
    let value = match dtype {
        GgmlDType::Q4_0 => QuantValue::Q4S,
        GgmlDType::Q8_0 => QuantValue::Q8S,
        _ => return None,
    };
    Some(
        QuantScheme::default()
            .with_value(value)
            .per_block([32], ScaleDtype::F16)
            .with_store(QuantStore::PackedU32(0)),
    )
}

/// Repack the GGUF bytes of a `[rows, cols]` Q4_0 or Q8_0 tensor into the
/// byte layout of a burn quantized tensor with [`quant_scheme`]: every
/// block's values, packed into little-endian `u32` words, followed by every
/// block's f16 scale, both in row-major block order.
///
/// - Q4_0 element `e = nibble - 8` is stored as its 4-bit two's complement
///   (`nibble ^ 8`); element `i` of each run of 8 sits at bits `4 * (i % 8)`
///   of word `i / 8`. GGUF holds elements `0..16` in the low nibbles of a
///   block's 16 bytes and `16..32` in the high nibbles.
/// - Q8_0 values are signed bytes already in burn's order (element `i` at
///   bits `8 * (i % 4)` of word `i / 4`) and are copied verbatim.
/// - Scales are copied verbatim: GGUF stores them little-endian and burn
///   native-endian, equal on the little-endian hosts this module builds for.
fn pack_blocks(
    dtype: GgmlDType,
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, CucaError> {
    if quant_scheme(dtype).is_none() {
        return Err(err(format!("{dtype:?} has no burn block-quantized layout")));
    }
    let block_bytes = dtype.type_size();
    if !cols.is_multiple_of(32) || raw.len() != rows * cols / 32 * block_bytes {
        return Err(err(format!(
            "{} bytes do not hold a [{rows}, {cols}] {dtype:?} tensor",
            raw.len()
        )));
    }
    let blocks = rows * cols / 32;
    // Values: 16 bytes per Q4 block, 32 per Q8 block; scales: 2 per block.
    let value_bytes = block_bytes - 2;
    let mut out = Vec::with_capacity(blocks * block_bytes);
    for block in raw.chunks_exact(block_bytes) {
        let q = &block[2..];
        if dtype == GgmlDType::Q4_0 {
            for word in 0..4 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let element = 8 * word + i;
                    let byte = q[element % 16];
                    let nibble = if element < 16 { byte & 0x0F } else { byte >> 4 };
                    packed |= u32::from(nibble ^ 8) << (4 * i);
                }
                out.extend_from_slice(&packed.to_le_bytes());
            }
        } else {
            out.extend_from_slice(&q[..value_bytes]);
        }
    }
    for block in raw.chunks_exact(block_bytes) {
        out.extend_from_slice(&block[..2]);
    }
    Ok(out)
}

/// The burn device for `device`, opened now so that a missing adapter or
/// driver is an error here rather than a panic at the first tensor.
fn open_device(device: LocalDevice) -> Result<Device, CucaError> {
    match device {
        #[cfg(feature = "provider-local-vulkan")]
        LocalDevice::Vulkan(adapter) => open_wgpu(
            device,
            adapter,
            burn::tensor::wgpu::WgpuBackend::Vulkan,
            Device::vulkan,
        ),
        #[cfg(feature = "provider-local-metal")]
        LocalDevice::Metal(adapter) => open_wgpu(
            device,
            adapter,
            burn::tensor::wgpu::WgpuBackend::Metal,
            Device::metal,
        ),
        #[cfg(feature = "provider-local-cuda")]
        LocalDevice::Cuda(index) => {
            require_index(device, burn::tensor::DeviceType::Cuda, index)?;
            Ok(Device::cuda(index))
        }
        #[cfg(feature = "provider-local-rocm")]
        LocalDevice::Rocm(index) => {
            require_index(device, burn::tensor::DeviceType::Rocm, index)?;
            Ok(Device::rocm(index))
        }
        other => Err(device_unavailable(other).unwrap_or_else(|| {
            CucaError::Config(format!("device {other:?} does not run the GPU engine"))
        })),
    }
}

/// Initialize a wgpu-backed device. A second model on the same adapter finds
/// its runtime registered already and shares it through the lazy
/// constructor `shared`.
#[cfg(any(feature = "provider-local-vulkan", feature = "provider-local-metal"))]
fn open_wgpu(
    device: LocalDevice,
    adapter: crate::provider::local::GpuAdapter,
    api: burn::tensor::wgpu::WgpuBackend,
    shared: fn(burn::tensor::DeviceKind) -> Device,
) -> Result<Device, CucaError> {
    use crate::provider::local::GpuAdapter;
    use burn::tensor::DeviceKind;
    use burn::tensor::wgpu::WgpuInitError;

    let kind = || match adapter {
        GpuAdapter::Default => DeviceKind::DefaultDevice,
        GpuAdapter::Discrete(n) => DeviceKind::DiscreteGpu(n),
        GpuAdapter::Integrated(n) => DeviceKind::IntegratedGpu(n),
    };
    match Device::wgpu_options()
        .device_kind(kind())
        .graphics_api(api)
        .init()
    {
        Ok(opened) => Ok(opened),
        Err(WgpuInitError::Registration { .. }) => Ok(shared(kind())),
        Err(e) => Err(err(format!("device {device:?}: {e}"))),
    }
}

/// Fail unless the backend reports a device at `index`.
#[cfg(any(feature = "provider-local-cuda", feature = "provider-local-rocm"))]
fn require_index(
    device: LocalDevice,
    kind: burn::tensor::DeviceType,
    index: usize,
) -> Result<(), CucaError> {
    let found = Device::enumerate(kind).into_vec().len();
    if index >= found {
        return Err(err(format!(
            "device {device:?}: {found} such device(s) found (missing driver or GPU?)"
        )));
    }
    Ok(())
}

/// Reads tensors from the GGUF file into one reused buffer and uploads them.
struct WeightReader<'a> {
    content: &'a gguf_file::Content,
    file: File,
    buf: Vec<u8>,
    device: &'a Device,
}

impl WeightReader<'_> {
    /// Read `tensor_name` into `buf` after checking its shape; returns its
    /// dtype.
    fn read(&mut self, tensor_name: &str, shape: &[usize]) -> Result<GgmlDType, CucaError> {
        let info = tensor_info(self.content, tensor_name)?;
        let dims = info.shape.dims();
        if dims != shape {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {shape:?}, found {dims:?}"
            )));
        }
        let dtype = info.ggml_dtype;
        let elements: usize = shape.iter().product();
        if !elements.is_multiple_of(dtype.block_size()) {
            return Err(err(format!(
                "tensor `{tensor_name}`: {elements} elements are not whole {dtype:?} blocks"
            )));
        }
        self.buf
            .resize(elements / dtype.block_size() * dtype.type_size(), 0);
        self.file.seek(SeekFrom::Start(
            self.content.tensor_data_offset + info.offset,
        ))?;
        self.file.read_exact(&mut self.buf)?;
        Ok(dtype)
    }

    /// A `[out, in]` matmul weight on the device: quantized for Q4_0 and
    /// Q8_0, f32 for the float dtypes.
    fn weight(
        &mut self,
        tensor_name: &str,
        out: usize,
        inn: usize,
    ) -> Result<Tensor<2>, CucaError> {
        let dtype = self.read(tensor_name, &[out, inn])?;
        let data = match (quant_scheme(dtype), dtype) {
            (Some(scheme), _) => TensorData::from_bytes_vec(
                pack_blocks(dtype, &self.buf, out, inn)?,
                [out, inn],
                DType::QFloat(scheme),
            ),
            (None, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) => {
                let mut values = vec![0f32; out * inn];
                dequant_row(dtype, &self.buf, 1.0, &mut values)
                    .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
                TensorData::new(values, [out, inn])
            }
            (None, other) => {
                return Err(err(format!(
                    "tensor `{tensor_name}`: dtype {other:?} is not supported on a GPU device; use a Q4_0 or Q8_0 checkpoint"
                )));
            }
        };
        Ok(Tensor::from_data(data, self.device))
    }

    /// A 1-D tensor dequantized on the host.
    fn host_vector(&mut self, tensor_name: &str, len: usize) -> Result<Vec<f32>, CucaError> {
        let dtype = self.read(tensor_name, &[len])?;
        let mut out = vec![0f32; len];
        dequant_row(dtype, &self.buf, 1.0, &mut out)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        Ok(out)
    }

    /// A 1-D f32 tensor on the device.
    fn vector(&mut self, tensor_name: &str, len: usize) -> Result<Tensor<1>, CucaError> {
        let values = self.host_vector(tensor_name, len)?;
        Ok(Tensor::from_data(
            TensorData::new(values, [len]),
            self.device,
        ))
    }
}

/// RoPE cos/sin tables of shape `[positions, head_dim / 2]` on the device.
struct GpuRope {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl GpuRope {
    fn new(
        head_dim: usize,
        base: f64,
        factors: Option<&[f32]>,
        positions: usize,
        device: &Device,
    ) -> Self {
        let (cos, sin) = rope_cos_sin(&rope_inv_freq(head_dim, base, factors), positions);
        let shape = [positions, head_dim / 2];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
        }
    }
}

/// Weights of one transformer block.
struct GpuLayer {
    attn_norm: Tensor<1>,
    wq: Tensor<2>,
    q_norm: Tensor<1>,
    /// K/V projections exist only on layers below `n_layer_kv`.
    kv: Option<GpuKvWeights>,
    wo: Tensor<2>,
    post_attn_norm: Tensor<1>,
    ffn_norm: Tensor<1>,
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
    post_ffw_norm: Tensor<1>,
    inp_gate: Tensor<2>,
    proj: Tensor<2>,
    post_norm: Tensor<1>,
    out_scale: f64,
}

struct GpuKvWeights {
    wk: Tensor<2>,
    k_norm: Tensor<1>,
    wv: Tensor<2>,
}

/// K and V cache tensors, `[1, n_kv_head, context_length, head_dim]`, of
/// every layer (`None` for layers that share another layer's cache).
type KvLayers = Vec<Option<(Tensor<4>, Tensor<4>)>>;

/// The per-forward inputs of one attention type: RoPE rows of the new
/// positions, the first visible cached position and the additive mask.
struct AttnInputs {
    /// `[1, 1, t, head_dim / 2]`.
    cos: Tensor<4>,
    sin: Tensor<4>,
    lo: usize,
    /// `[1, 1, t, hi - lo]`; `None` for a single query, which sees every key.
    mask: Option<Tensor<4>>,
}

/// Everything read-only during a forward pass.
struct Weights {
    params: Gemma4Params,
    device: Device,
    ple_proj: Tensor<2>,
    ple_proj_norm: Tensor<1>,
    layers: Vec<GpuLayer>,
    output_norm: Tensor<1>,
    lm_head: Tensor<2>,
    rope_full: GpuRope,
    rope_swa: GpuRope,
}

/// A loaded Gemma 4 model on a GPU with its K/V cache; see the module docs.
pub(crate) struct GpuGemma4 {
    weights: Weights,
    context_length: usize,
    embed: RowTable,
    ple: RowTable,
    kv: KvLayers,
    kv_len: usize,
}

impl GpuGemma4 {
    /// Load the model from the GGUF file at `path`, whose index `content`
    /// was read already, onto `device` with a K/V cache for
    /// `context_length` positions.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum or `device` is not a GPU compiled into this build;
    /// [`CucaError::Provider`] for unsupported or malformed checkpoints
    /// (including a weight dtype this engine does not read) and a device
    /// that fails to open; [`CucaError::Io`] when the file cannot be read.
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
    ) -> Result<Self, CucaError> {
        let params = Gemma4Params::from_metadata(&content.metadata)?;
        if context_length == 0 || context_length > params.model_ctx {
            return Err(CucaError::Config(format!(
                "context_length {context_length} exceeds the model maximum {}",
                params.model_ctx
            )));
        }
        let device = open_device(device)?;
        let p = &params;

        let embed = RowTable::open(path, content, name::TOKEN_EMBD, "embedding table", p.n_embd)?;
        let ple_row_len = p.n_layer * p.ple_dim;
        let ple = RowTable::open(
            path,
            content,
            name::PER_LAYER_TOKEN_EMBD,
            "per-layer embedding table",
            ple_row_len,
        )?;

        let mut reader = WeightReader {
            content,
            file: File::open(path)?,
            buf: Vec::new(),
            device: &device,
        };
        let lm_head = reader.weight(name::TOKEN_EMBD, embed.rows, p.n_embd)?;
        let ple_proj = reader.weight(name::PER_LAYER_MODEL_PROJ, ple_row_len, p.n_embd)?;
        let ple_proj_norm = reader.vector(name::PER_LAYER_PROJ_NORM, p.ple_dim)?;
        let output_norm = reader.vector(name::OUTPUT_NORM, p.n_embd)?;

        let factors = if content.tensor_infos.contains_key(name::ROPE_FREQS) {
            let ff = reader.host_vector(name::ROPE_FREQS, p.head_dim_full / 2)?;
            if ff.iter().any(|f| !f.is_finite() || *f <= 0.0) {
                return Err(err(format!(
                    "tensor `{}`: factors must be finite and positive",
                    name::ROPE_FREQS
                )));
            }
            Some(ff)
        } else {
            None
        };
        let rope_full = GpuRope::new(
            p.head_dim_full,
            p.rope_base,
            factors.as_deref(),
            context_length,
            &device,
        );
        let rope_swa = GpuRope::new(
            p.head_dim_swa,
            p.rope_base_swa,
            None,
            context_length,
            &device,
        );

        let mut layers = Vec::with_capacity(p.n_layer);
        let mut kv: KvLayers = Vec::with_capacity(p.n_layer);
        for il in 0..p.n_layer {
            let d = if p.is_sliding[il] {
                p.head_dim_swa
            } else {
                p.head_dim_full
            };
            let q_width = p.n_head * d;
            let kv_width = p.n_kv_head * d;
            let gate_name = name::blk(il, name::FFN_GATE);
            let ff = *tensor_info(content, &gate_name)?
                .shape
                .dims()
                .first()
                .ok_or_else(|| err(format!("tensor `{gate_name}`: scalar")))?;
            let blk = |part: &str| name::blk(il, part);

            let kv_weights = if il < p.n_layer_kv {
                let shape = [1, p.n_kv_head, context_length, d];
                kv.push(Some((
                    Tensor::zeros(shape, &device),
                    Tensor::zeros(shape, &device),
                )));
                Some(GpuKvWeights {
                    wk: reader.weight(&blk(name::ATTN_K), kv_width, p.n_embd)?,
                    k_norm: reader.vector(&blk(name::ATTN_K_NORM), d)?,
                    wv: reader.weight(&blk(name::ATTN_V), kv_width, p.n_embd)?,
                })
            } else {
                kv.push(None);
                None
            };
            layers.push(GpuLayer {
                attn_norm: reader.vector(&blk(name::ATTN_NORM), p.n_embd)?,
                wq: reader.weight(&blk(name::ATTN_Q), q_width, p.n_embd)?,
                q_norm: reader.vector(&blk(name::ATTN_Q_NORM), d)?,
                kv: kv_weights,
                wo: reader.weight(&blk(name::ATTN_OUTPUT), p.n_embd, q_width)?,
                post_attn_norm: reader.vector(&blk(name::POST_ATTENTION_NORM), p.n_embd)?,
                ffn_norm: reader.vector(&blk(name::FFN_NORM), p.n_embd)?,
                gate: reader.weight(&gate_name, ff, p.n_embd)?,
                up: reader.weight(&blk(name::FFN_UP), ff, p.n_embd)?,
                down: reader.weight(&blk(name::FFN_DOWN), p.n_embd, ff)?,
                post_ffw_norm: reader.vector(&blk(name::POST_FFW_NORM), p.n_embd)?,
                inp_gate: reader.weight(&blk(name::INP_GATE), p.ple_dim, p.n_embd)?,
                proj: reader.weight(&blk(name::PROJ), p.n_embd, p.ple_dim)?,
                post_norm: reader.vector(&blk(name::POST_NORM), p.n_embd)?,
                out_scale: f64::from(reader.host_vector(&blk(name::LAYER_OUTPUT_SCALE), 1)?[0]),
            });
        }
        drop(reader);

        Ok(Self {
            weights: Weights {
                params,
                device,
                ple_proj,
                ple_proj_norm,
                layers,
                output_norm,
                lm_head,
                rope_full,
                rope_swa,
            },
            context_length,
            embed,
            ple,
            kv,
            kv_len: 0,
        })
    }

    /// See [`Gemma4::forward`](super::Gemma4::forward): same contract, same
    /// errors.
    pub(crate) fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, CucaError> {
        if tokens.is_empty() {
            return Err(err("forward requires at least one token"));
        }
        if start_pos != self.kv_len {
            return Err(err(format!(
                "forward start {start_pos} does not match the cache length {}",
                self.kv_len
            )));
        }
        let hi = start_pos + tokens.len();
        if hi > self.context_length {
            return Err(err(format!(
                "{hi} positions exceed the context length {}",
                self.context_length
            )));
        }
        let p = &self.weights.params;
        let embeddings = self.embed.rows(tokens, (p.n_embd as f32).sqrt())?;
        let per_layer = self.ple.rows(tokens, (p.ple_dim as f32).sqrt())?;
        let logits = self.weights.forward_layers(
            &mut self.kv,
            embeddings,
            per_layer,
            tokens.len(),
            start_pos,
            want_logits,
        )?;
        self.kv_len = hi;
        Ok(logits)
    }

    /// Drop every cached position from `len` on; see
    /// [`Gemma4::truncate_cache`](super::Gemma4::truncate_cache).
    pub(crate) fn truncate_cache(&mut self, len: usize) {
        self.kv_len = len.min(self.kv_len);
    }

    /// Number of cached positions (test-only).
    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        self.kv_len
    }
}

impl Weights {
    fn forward_layers(
        &self,
        kv: &mut KvLayers,
        embeddings: Vec<f32>,
        per_layer: Vec<f32>,
        t: usize,
        start: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, CucaError> {
        let p = &self.params;
        let dev = &self.device;
        let eps = p.eps;
        let mut x = Tensor::<2>::from_data(TensorData::new(embeddings, [t, p.n_embd]), dev);
        let per_layer_tokens =
            Tensor::<3>::from_data(TensorData::new(per_layer, [t, p.n_layer, p.ple_dim]), dev);
        let per_layer_ctx = linear(x.clone(), &self.ple_proj)
            .mul_scalar(1.0 / (p.n_embd as f64).sqrt())
            .reshape([t, p.n_layer, p.ple_dim]);
        let per_layer_ctx = rms(per_layer_ctx, Some(&self.ple_proj_norm), eps);
        let ple = (per_layer_ctx + per_layer_tokens).mul_scalar(FRAC_1_SQRT_2);

        let full = self.attn_inputs(false, start, t);
        let swa = self.attn_inputs(true, start, t);
        for il in 0..p.n_layer {
            let inputs = if p.is_sliding[il] { &swa } else { &full };
            x = self.layer(il, x, &ple, inputs, kv, start, t)?;
        }

        if !want_logits {
            return Ok(None);
        }
        let last = x.narrow(0, t - 1, 1);
        let logits = linear(rms(last, Some(&self.output_norm), eps), &self.lm_head)
            .div_scalar(p.softcap)
            .tanh()
            .mul_scalar(p.softcap);
        let data = logits
            .try_into_data()
            .map_err(|e| err(format!("reading logits from the device: {e}")))?;
        let values = data
            .try_into_vec::<f32>()
            .map_err(|e| err(format!("reading logits from the device: {e:?}")))?;
        Ok(Some(values))
    }

    /// RoPE rows, visible range and mask shared by every layer of one
    /// attention type for `t` queries at `start..`.
    fn attn_inputs(&self, sliding: bool, start: usize, t: usize) -> AttnInputs {
        let p = &self.params;
        let (rope, d) = if sliding {
            (&self.rope_swa, p.head_dim_swa)
        } else {
            (&self.rope_full, p.head_dim_full)
        };
        let half = d / 2;
        let rows = |table: &Tensor<2>| table.clone().narrow(0, start, t).reshape([1, 1, t, half]);
        let hi = start + t;
        let lo = if sliding {
            start.saturating_sub(p.window - 1)
        } else {
            0
        };
        let mask = (t > 1).then(|| {
            let values = attention_mask(start, t, lo, hi, sliding.then_some(p.window));
            Tensor::from_data(TensorData::new(values, [1, 1, t, hi - lo]), &self.device)
        });
        AttnInputs {
            cos: rows(&rope.cos),
            sin: rows(&rope.sin),
            lo,
            mask,
        }
    }

    /// One transformer block over `t` positions starting at `start`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        il: usize,
        x: Tensor<2>,
        ple: &Tensor<3>,
        a: &AttnInputs,
        kv: &mut KvLayers,
        start: usize,
        t: usize,
    ) -> Result<Tensor<2>, CucaError> {
        let p = &self.params;
        let lw = &self.layers[il];
        let eps = p.eps;
        let d = if p.is_sliding[il] {
            p.head_dim_swa
        } else {
            p.head_dim_full
        };
        let hi = start + t;

        let h = rms(x.clone(), Some(&lw.attn_norm), eps);
        let q = linear(h.clone(), &lw.wq).reshape([1, t, p.n_head, d]);
        let q = rope(rms(q, Some(&lw.q_norm), eps).swap_dims(1, 2), a);

        if let Some(kvw) = &lw.kv {
            let k = linear(h.clone(), &kvw.wk).reshape([1, t, p.n_kv_head, d]);
            let k = rope(rms(k, Some(&kvw.k_norm), eps).swap_dims(1, 2), a);
            let v = linear(h, &kvw.wv).reshape([1, t, p.n_kv_head, d]);
            let v = rms(v, None, eps).swap_dims(1, 2);
            // Take the cache out so the assignment owns the only handle and
            // writes in place.
            let (cache_k, cache_v) = kv[il]
                .take()
                .ok_or_else(|| err(format!("layer {il} owns no K/V cache")))?;
            let at = cache_ranges(p.n_kv_head, start..hi, d);
            kv[il] = Some((
                cache_k.slice_assign(at.clone(), k),
                cache_v.slice_assign(at, v),
            ));
        }

        let src = p.kv_source[il];
        let (cache_k, cache_v) = kv[src]
            .as_ref()
            .ok_or_else(|| err(format!("layer {src} owns no K/V cache")))?;
        let at = cache_ranges(p.n_kv_head, a.lo..hi, d);
        let k = cache_k.clone().slice(at.clone());
        let v = cache_v.clone().slice(at);
        let attn = attend(q, k, v, a.mask.clone())
            .swap_dims(1, 2)
            .reshape([t, p.n_head * d]);
        let attn = linear(attn, &lw.wo);
        let x = x + rms(attn, Some(&lw.post_attn_norm), eps);

        let f = rms(x.clone(), Some(&lw.ffn_norm), eps);
        let gated = gelu(linear(f.clone(), &lw.gate)) * linear(f, &lw.up);
        let f = linear(gated, &lw.down);
        let x = x + rms(f, Some(&lw.post_ffw_norm), eps);

        let slice = ple.clone().narrow(1, il, 1).reshape([t, p.ple_dim]);
        let g = gelu(linear(x.clone(), &lw.inp_gate)) * slice;
        let x = x + rms(linear(g, &lw.proj), Some(&lw.post_norm), eps);
        Ok(x.mul_scalar(lw.out_scale))
    }
}

/// The cache region of positions `positions` of every KV head.
fn cache_ranges(n_kv_head: usize, positions: Range<usize>, d: usize) -> [Range<usize>; 4] {
    [0..1, 0..n_kv_head, positions, 0..d]
}

/// `x [t, in]` times the `[out, in]` weight `w`, transposed: `[t, out]`. A
/// quantized `w` is read by the matmul kernel directly (dequantized in
/// registers); the transposes only swap strides.
///
/// The operand order depends on `t`, working around burn 0.22 kernel bugs
/// seen on Vulkan (RX 9070 XT), both pinned by `gpu_tiny_model_matches_cpu`
/// and the live suite:
///
/// - `t == 1` (decode): `x · wᵀ`. `(w · xᵀ)ᵀ` selects a matrix-vector
///   kernel that returns wrong values for a quantized left operand.
/// - `t > 1` (prefill): `(w · xᵀ)ᵀ`, the weight on the left in its stored
///   layout. With the stride-swapped quantized weight on the right, some
///   autotune candidates lose the device.
fn linear(x: Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
    if x.dims()[0] == 1 {
        x.matmul(w.clone().transpose())
    } else {
        w.clone().matmul(x.transpose()).transpose()
    }
}

/// RMSNorm over the last dimension, multiplied by `weight` (no `1 + w`);
/// `None` is the weightless norm.
fn rms<const D: usize>(x: Tensor<D>, weight: Option<&Tensor<1>>, eps: f64) -> Tensor<D> {
    let inv = x
        .clone()
        .square()
        .mean_dim(D - 1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    let y = x * inv;
    match weight {
        Some(w) => y * w.clone().unsqueeze::<D>(),
        None => y,
    }
}

/// GELU, tanh approximation (candle's `Tensor::gelu`). The cube is written
/// as `x * x * x`: a GPU `pow` with a negative base is undefined.
fn gelu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let cube = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cube.mul_scalar(0.044_715)).mul_scalar(SQRT_2_OVER_PI);
    (x * inner.tanh().add_scalar(1.0)).mul_scalar(0.5)
}

/// NEOX (half-split) RoPE of `x [1, heads, t, d]`.
fn rope(x: Tensor<4>, a: &AttnInputs) -> Tensor<4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    Tensor::cat(
        vec![
            x1.clone() * a.cos.clone() - x2.clone() * a.sin.clone(),
            x1 * a.sin.clone() + x2 * a.cos.clone(),
        ],
        3,
    )
}

/// Grouped-query attention without a score scale, as the CPU `attend`: `q`
/// is `[1, n_head, t, d]`, `k` and `v` are `[1, n_kv, l, d]`, `mask` the
/// additive `[1, 1, t, l]` mask. The `n_head / n_kv` query heads of one KV
/// head are stacked into the rows of one matmul. Returns `[1, n_head, t, d]`.
fn attend(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, mask: Option<Tensor<4>>) -> Tensor<4> {
    let [_, n_head, t, d] = q.dims();
    let [_, n_kv, l, _] = k.dims();
    let rows = n_head / n_kv * t;
    let scores = q.reshape([1, n_kv, rows, d]).matmul(k.transpose());
    let scores = match mask {
        Some(mask) => (scores.reshape([1, n_head, t, l]) + mask).reshape([1, n_kv, rows, l]),
        None => scores,
    };
    activation::softmax(scores, 3)
        .matmul(v)
        .reshape([1, n_head, t, d])
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use super::super::Gemma4;
    use super::super::tiny::{PROMPT, TinyModel, TinyOptions, write_tiny_model_with};
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    /// The `index`-th signed element of a packed values region, decoded by
    /// the documented bit layout: `bits`-wide fields, `32 / bits` per
    /// little-endian `u32`, lowest field first, two's complement.
    fn packed_element(values: &[u8], bits: u32, index: usize) -> i32 {
        let per_word = (32 / bits) as usize;
        let at = index / per_word * 4;
        let word = u32::from_le_bytes(values[at..at + 4].try_into().unwrap());
        let field = (word >> (bits * (index % per_word) as u32)) & ((1 << bits) - 1);
        // Sign-extend the field.
        ((field << (32 - bits)) as i32) >> (32 - bits)
    }

    /// Two Q4_0 blocks (one row of 64) whose nibbles cover 0..16 in both
    /// halves, so every value -8..=7 appears.
    fn q4_0_row() -> Vec<u8> {
        let mut raw = Vec::new();
        for scale in [0.5f32, -0.25] {
            raw.extend(f16_bytes(scale));
            raw.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        }
        raw
    }

    #[test]
    fn pack_q4_0_matches_dequant_row() {
        let raw = q4_0_row();
        let packed = pack_blocks(GgmlDType::Q4_0, &raw, 1, 64).unwrap();
        // 64 nibbles in 8 words, then 2 scales.
        assert_eq!(packed.len(), 32 + 4);
        let mut expected = [0f32; 64];
        dequant_row(GgmlDType::Q4_0, &raw, 1.0, &mut expected).unwrap();
        for (i, &e) in expected.iter().enumerate() {
            let scale = if i < 32 { 0.5 } else { -0.25 };
            assert_eq!(
                packed_element(&packed, 4, i) as f32,
                e / scale,
                "element {i}"
            );
        }
        assert_eq!(&packed[32..34], &raw[0..2]);
        assert_eq!(&packed[34..36], &raw[18..20]);
    }

    #[test]
    fn pack_q8_0_matches_dequant_row() {
        // Two rows of one block each.
        let mut raw = Vec::new();
        for (scale, salt) in [(0.25f32, 3i32), (2.0, 11)] {
            raw.extend(f16_bytes(scale));
            raw.extend((0..32).map(|j| ((j * 8 + salt) % 256 - 128) as i8 as u8));
        }
        let packed = pack_blocks(GgmlDType::Q8_0, &raw, 2, 32).unwrap();
        assert_eq!(packed.len(), 64 + 4);
        let mut expected = [0f32; 64];
        dequant_row(GgmlDType::Q8_0, &raw, 1.0, &mut expected).unwrap();
        for (i, &e) in expected.iter().enumerate() {
            let scale = if i < 32 { 0.25 } else { 2.0 };
            assert_eq!(
                packed_element(&packed, 8, i) as f32,
                e / scale,
                "element {i}"
            );
        }
        assert_eq!(&packed[64..66], &raw[0..2]);
        assert_eq!(&packed[66..68], &raw[34..36]);
    }

    #[test]
    fn quant_scheme_rejects_other_dtypes() {
        assert!(quant_scheme(GgmlDType::Q4_0).is_some());
        assert!(quant_scheme(GgmlDType::Q8_0).is_some());
        for dtype in [
            GgmlDType::F32,
            GgmlDType::BF16,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
        ] {
            assert!(quant_scheme(dtype).is_none(), "{dtype:?}");
            assert!(pack_blocks(dtype, &[], 0, 32).is_err(), "{dtype:?}");
        }
        // A byte count that is not the tensor's block count.
        assert!(pack_blocks(GgmlDType::Q4_0, &q4_0_row()[..35], 1, 64).is_err());
    }

    /// The GPU device under test, from `CUCA_LOCAL_DEVICE`; `None` (skip)
    /// when unset or `cpu`.
    fn test_device() -> Option<LocalDevice> {
        let device: LocalDevice = std::env::var("CUCA_LOCAL_DEVICE")
            .ok()?
            .parse()
            .expect("CUCA_LOCAL_DEVICE");
        (device != LocalDevice::Cpu).then_some(device)
    }

    fn load_both(model: &TinyModel, device: LocalDevice) -> (Gemma4, GpuGemma4) {
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        (
            Gemma4::load_dequantized(&content, &model.path, 32).unwrap(),
            GpuGemma4::load(&content, &model.path, 32, device).unwrap(),
        )
    }

    fn assert_matches_cpu(cpu: &[f32], gpu: &[f32], what: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{what}");
        for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (g - c).abs() <= 1e-3 + 1e-2 * c.abs(),
                "{what}: logit {i}: gpu {g} vs cpu {c}"
            );
        }
    }

    /// Run the same sequence of forward calls on both engines and compare
    /// every returned logit vector.
    fn compare(cpu: &mut Gemma4, gpu: &mut GpuGemma4, tokens: &[u32], start: usize, what: &str) {
        let c = cpu.forward(tokens, start, true).unwrap().unwrap();
        let g = gpu.forward(tokens, start, true).unwrap().unwrap();
        assert_matches_cpu(&c, &g, what);
    }

    /// The GPU engine must reproduce the CPU engine on the tiny model:
    /// single-shot and chunked prefill, decode, and truncate-and-replay, for
    /// one KV head and for grouped-query attention. The reference holds its
    /// weights dequantized: candle's quantized matmul rounds the activations
    /// to Q8_0, which the GPU kernels (f32 activations) do not, and over six
    /// random-weight layers that alone moves logits by several percent.
    #[test]
    fn gpu_tiny_model_matches_cpu() {
        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
            return;
        };
        for options in [
            TinyOptions::default(),
            TinyOptions {
                n_head: 4,
                n_kv_head: 2,
                ..TinyOptions::default()
            },
        ] {
            let model = write_tiny_model_with(options);
            let n = PROMPT.len();

            // (a) single-shot prefill, then (c) one decode step.
            let (mut cpu, mut gpu) = load_both(&model, device);
            compare(&mut cpu, &mut gpu, &PROMPT, 0, "prefill");
            compare(&mut cpu, &mut gpu, &[7], n, "decode");

            // (b) chunked prefill in chunks of 3, crossing the window.
            let (mut cpu, mut gpu) = load_both(&model, device);
            let mut start = 0;
            while start + 3 < n {
                let chunk = &PROMPT[start..start + 3];
                assert!(gpu.forward(chunk, start, false).unwrap().is_none());
                cpu.forward(chunk, start, false).unwrap();
                start += 3;
            }
            compare(&mut cpu, &mut gpu, &PROMPT[start..], start, "chunked");
            assert_eq!(gpu.cache_len(), n);

            // (d) truncate and replay a different suffix.
            cpu.truncate_cache(5);
            gpu.truncate_cache(5);
            assert!(
                gpu.forward(&PROMPT[6..], 6, true).is_err(),
                "gap after truncate"
            );
            compare(&mut cpu, &mut gpu, &[11, 13, 15, 14, 13, 12], 5, "replay");
        }
    }
}
