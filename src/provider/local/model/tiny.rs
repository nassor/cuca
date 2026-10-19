//! Tiny random-weight models written to a temporary GGUF file: small enough
//! to run in a unit test, structured like the real architectures they stand
//! for.
//!
//! [`write_tiny_model_with`] builds the Gemma 4 model (shared K/V layers,
//! sliding and full layers with a window shorter than the prompt, partial
//! RoPE factors, per-layer embeddings). [`write_spec`] builds the others from
//! a [`TinySpec`]: Llama (NORM RoPE, Llama 3 factors), Qwen 2 (biases), Qwen 3
//! (QK-norm, query width different from the embedding width), Phi 3 (fused
//! QKV and gate-up, LongRoPE, partial rotary), SmolLM 3 (NoPE layers),
//! Granite MoE (experts, scales) and Gemma 3 (post-norms, sliding pattern).
//! [`write_qwen35`] builds the hybrid Qwen 3.5 model from a [`TinyQwen35`]:
//! delta-net, delta-net, delta-net, gated attention, delta-net, then a
//! trailing multi-token-prediction block with its `nextn.*` tensors, four
//! value heads over two key heads of different widths so every index of the
//! recurrence is exercised.
//! Weights are drawn in a fixed order, so two specs that differ only in
//! storage (fused or split, NORM or NEOX row order) hold the same values.

use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::{AtomicU32, Ordering};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

use super::cpu::Transformer;

pub(crate) const VOCAB: usize = 16;
pub(crate) const CONTEXT: usize = 32;
pub(crate) const N_EMBD: usize = 64;

/// Deterministic pseudo-random values in `[-scale, scale]`.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

pub(crate) struct TinyModel {
    pub(crate) path: std::path::PathBuf,
}

impl Drop for TinyModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    std::env::temp_dir().join(format!(
        "cuca-tiny-{tag}-{}-{}.gguf",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

fn quantize(values: Vec<f32>, dims: &[usize], dtype: GgmlDType) -> QTensor {
    let t = Tensor::from_vec(values, dims, &Device::Cpu).unwrap();
    QTensor::quantize(&t, dtype).unwrap()
}

fn quantized(rng: &mut Lcg, dims: &[usize], dtype: GgmlDType, scale: f32) -> QTensor {
    quantize(rng.values(dims.iter().product(), scale), dims, dtype)
}

fn norm(rng: &mut Lcg, len: usize) -> QTensor {
    let values: Vec<f32> = rng.values(len, 0.25).iter().map(|v| 1.0 + v).collect();
    quantize(values, &[len], GgmlDType::F32)
}

fn f32_tensor(values: Vec<f32>) -> QTensor {
    let n = values.len();
    quantize(values, &[n], GgmlDType::F32)
}

fn write(path: &std::path::Path, meta: &[(String, Value)], tensors: &[(String, QTensor)]) {
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut file = File::create(path).unwrap();
    gguf_file::write(&mut file, &meta_refs, &tensor_refs).unwrap();
}

fn kv(key: &str, value: Value) -> (String, Value) {
    (key.to_string(), value)
}

fn u(n: usize) -> Value {
    Value::U32(n as u32)
}

// ---------------------------------------------------------------------------
// Gemma 4
// ---------------------------------------------------------------------------

const G4_N_LAYER: usize = 6;
const G4_N_LAYER_KV: usize = 3;
const G4_N_HEAD: usize = 2;
const G4_HEAD_FULL: usize = 64;
const G4_HEAD_SWA: usize = 32;
const G4_WINDOW: usize = 4;
const G4_PLE: usize = 32;
const G4_SLIDING: [bool; G4_N_LAYER] = [true, true, false, true, true, false];

/// Head counts and embedding dtypes of the tiny Gemma 4 model.
#[derive(Clone, Copy)]
pub(crate) struct TinyOptions {
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    /// Dtype of `token_embd.weight` (rows of `N_EMBD` = 64).
    pub(crate) embd_dtype: GgmlDType,
    /// Dtype of `per_layer_token_embd.weight` (rows of 192).
    pub(crate) ple_dtype: GgmlDType,
}

impl Default for TinyOptions {
    fn default() -> Self {
        Self {
            n_head: G4_N_HEAD,
            n_kv_head: 1,
            embd_dtype: GgmlDType::Q8_0,
            ple_dtype: GgmlDType::Q4_0,
        }
    }
}

pub(crate) fn write_tiny_model() -> TinyModel {
    write_tiny_model_with(TinyOptions::default())
}

pub(crate) fn write_tiny_model_with(o: TinyOptions) -> TinyModel {
    let path = temp_path("gemma4");
    let mut rng = Lcg(7);
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let mut add = |n: &str, t: QTensor| tensors.push((n.to_string(), t));

    add(
        "token_embd.weight",
        quantized(&mut rng, &[VOCAB, N_EMBD], o.embd_dtype, 0.5),
    );
    add(
        "per_layer_token_embd.weight",
        quantized(&mut rng, &[VOCAB, G4_N_LAYER * G4_PLE], o.ple_dtype, 0.5),
    );
    add(
        "per_layer_model_proj.weight",
        quantized(
            &mut rng,
            &[G4_N_LAYER * G4_PLE, N_EMBD],
            GgmlDType::BF16,
            0.2,
        ),
    );
    add("per_layer_proj_norm.weight", norm(&mut rng, G4_PLE));
    add("output_norm.weight", norm(&mut rng, N_EMBD));
    let mut factors = vec![1.0f32; G4_HEAD_FULL / 4];
    factors.resize(G4_HEAD_FULL / 2, 1e30);
    add("rope_freqs.weight", f32_tensor(factors));
    for (il, &sliding) in G4_SLIDING.iter().enumerate() {
        let d = if sliding { G4_HEAD_SWA } else { G4_HEAD_FULL };
        let ff = if il < G4_N_LAYER_KV { 64 } else { 128 };
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        let q4 = GgmlDType::Q4_0;
        add(&b("attn_norm"), norm(&mut rng, N_EMBD));
        add(
            &b("attn_q"),
            quantized(&mut rng, &[o.n_head * d, N_EMBD], q4, 0.3),
        );
        add(&b("attn_q_norm"), norm(&mut rng, d));
        if il < G4_N_LAYER_KV {
            add(
                &b("attn_k"),
                quantized(&mut rng, &[o.n_kv_head * d, N_EMBD], q4, 0.3),
            );
            add(&b("attn_k_norm"), norm(&mut rng, d));
            add(
                &b("attn_v"),
                quantized(&mut rng, &[o.n_kv_head * d, N_EMBD], q4, 0.3),
            );
        }
        add(
            &b("attn_output"),
            quantized(&mut rng, &[N_EMBD, o.n_head * d], q4, 0.3),
        );
        add(&b("post_attention_norm"), norm(&mut rng, N_EMBD));
        add(&b("ffn_norm"), norm(&mut rng, N_EMBD));
        add(&b("ffn_gate"), quantized(&mut rng, &[ff, N_EMBD], q4, 0.3));
        add(&b("ffn_up"), quantized(&mut rng, &[ff, N_EMBD], q4, 0.3));
        add(&b("ffn_down"), quantized(&mut rng, &[N_EMBD, ff], q4, 0.3));
        add(&b("post_ffw_norm"), norm(&mut rng, N_EMBD));
        add(
            &b("inp_gate"),
            quantized(&mut rng, &[G4_PLE, N_EMBD], q4, 0.3),
        );
        add(&b("proj"), quantized(&mut rng, &[N_EMBD, G4_PLE], q4, 0.3));
        add(&b("post_norm"), norm(&mut rng, N_EMBD));
        add(&b("layer_output_scale"), f32_tensor(vec![0.9]));
    }

    let pattern = Value::Array(G4_SLIDING.iter().copied().map(Value::Bool).collect());
    let meta = vec![
        kv("general.architecture", Value::String("gemma4".into())),
        kv("gemma4.block_count", u(G4_N_LAYER)),
        kv("gemma4.context_length", u(CONTEXT)),
        kv("gemma4.embedding_length", u(N_EMBD)),
        kv("gemma4.attention.head_count", u(o.n_head)),
        kv("gemma4.attention.head_count_kv", u(o.n_kv_head)),
        kv("gemma4.attention.key_length", u(G4_HEAD_FULL)),
        kv("gemma4.attention.key_length_swa", u(G4_HEAD_SWA)),
        kv("gemma4.attention.sliding_window", u(G4_WINDOW)),
        kv("gemma4.attention.sliding_window_pattern", pattern),
        kv(
            "gemma4.attention.shared_kv_layers",
            u(G4_N_LAYER - G4_N_LAYER_KV),
        ),
        kv("gemma4.embedding_length_per_layer_input", u(G4_PLE)),
        kv("gemma4.rope.freq_base", Value::F32(1e6)),
        kv("gemma4.rope.freq_base_swa", Value::F32(1e4)),
        kv("gemma4.rope.dimension_count", u(G4_HEAD_FULL)),
        kv("gemma4.rope.dimension_count_swa", u(G4_HEAD_SWA)),
        kv("gemma4.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        kv("gemma4.final_logit_softcapping", Value::F32(30.0)),
    ];
    write(&path, &meta, &tensors);
    TinyModel { path }
}

// ---------------------------------------------------------------------------
// Every other architecture
// ---------------------------------------------------------------------------

/// A tiny model of the `arch` family. Defaults describe a plain Llama; the
/// constructors below set the features of the other architectures.
#[derive(Clone)]
pub(crate) struct TinySpec {
    /// `general.architecture` and the metadata key prefix.
    pub(crate) arch: &'static str,
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) ff: usize,
    pub(crate) qk_norm: bool,
    pub(crate) bias: bool,
    pub(crate) fused_qkv: bool,
    pub(crate) fused_gate_up: bool,
    pub(crate) post_norms: bool,
    /// `attn_norm` and `ffn_norm` exist (false for OLMo 2).
    pub(crate) pre_norms: bool,
    /// The Q/K norm weights span the whole projection (OLMo 2).
    pub(crate) qk_norm_full: bool,
    /// No `output.weight`: the LM head is the token embedding.
    pub(crate) tied: bool,
    /// `(experts, used)`; the FFN is then a mixture of experts of width `ff`.
    pub(crate) moe: Option<(usize, usize)>,
    pub(crate) embd_dtype: GgmlDType,
    pub(crate) weight_dtype: GgmlDType,
    /// Permute the rows of Q and K per head from adjacent-pair order to
    /// half-split order: a NORM model written this way, loaded as a NEOX
    /// architecture, computes the same function.
    pub(crate) neox_rows: bool,
    /// Extra metadata keys, relative to the architecture prefix.
    pub(crate) meta: Vec<(String, Value)>,
    pub(crate) rope_freqs: Option<Vec<f32>>,
    pub(crate) rope_long_short: Option<(Vec<f32>, Vec<f32>)>,
}

impl TinySpec {
    fn base(arch: &'static str) -> Self {
        Self {
            arch,
            n_layer: 4,
            n_head: 4,
            n_kv_head: 2,
            head_dim: N_EMBD / 4,
            ff: 96,
            qk_norm: false,
            bias: false,
            fused_qkv: false,
            fused_gate_up: false,
            post_norms: false,
            pre_norms: true,
            qk_norm_full: false,
            tied: false,
            moe: None,
            embd_dtype: GgmlDType::Q8_0,
            weight_dtype: GgmlDType::Q4_0,
            neox_rows: false,
            meta: Vec::new(),
            rope_freqs: None,
            rope_long_short: None,
        }
    }

    /// Llama: NORM RoPE with Llama 3 per-pair factors, separate output head.
    pub(crate) fn llama() -> Self {
        let mut s = Self::base("llama");
        s.rope_freqs = Some(
            (0..s.head_dim / 2)
                .map(|i| if i < 4 { 1.0 } else { 2.0 + i as f32 * 0.25 })
                .collect(),
        );
        s.meta.push(kv("rope.freq_base", Value::F32(500_000.0)));
        s
    }

    /// Qwen 2: NEOX RoPE, Q/K/V biases, tied head.
    pub(crate) fn qwen2() -> Self {
        let mut s = Self::base("qwen2");
        s.bias = true;
        s.tied = true;
        s.meta.push(kv("rope.freq_base", Value::F32(1_000_000.0)));
        s
    }

    /// Qwen 3: QK-norm, head dimension 32 so the query is wider than the
    /// embedding (128 vs 64).
    pub(crate) fn qwen3() -> Self {
        let mut s = Self::base("qwen3");
        s.head_dim = 32;
        s.qk_norm = true;
        s.tied = true;
        s.meta.push(kv("rope.freq_base", Value::F32(1_000_000.0)));
        s
    }

    /// Phi 3: fused QKV and gate-up, partial rotary (24 of 32 dimensions),
    /// LongRoPE factors (the model's original context is 16 < `CONTEXT`, so
    /// the long list applies) and an attention factor.
    pub(crate) fn phi3() -> Self {
        let mut s = Self::base("phi3");
        s.head_dim = 32;
        s.n_kv_head = 4;
        s.fused_qkv = true;
        s.fused_gate_up = true;
        s.meta.push(kv("rope.dimension_count", u(24)));
        s.meta.push(kv("rope.freq_base", Value::F32(10_000.0)));
        s.meta
            .push(kv("rope.scaling.original_context_length", u(16)));
        s.meta
            .push(kv("rope.scaling.attn_factor", Value::F32(1.19)));
        s.meta.push(kv("attention.sliding_window", u(262_144)));
        s.rope_long_short = Some((
            (0..12).map(|i| 1.5 + i as f32 * 0.5).collect(),
            (0..12).map(|i| 1.0 + i as f32 * 0.1).collect(),
        ));
        s
    }

    /// SmolLM 3: NORM RoPE and every fourth layer without any.
    pub(crate) fn smollm3() -> Self {
        let mut s = Self::base("smollm3");
        s.n_layer = 8;
        s.meta.push(kv("rope.freq_base", Value::F32(5_000_000.0)));
        s
    }

    /// Granite MoE: four experts, two used, Granite scales.
    pub(crate) fn granitemoe() -> Self {
        let mut s = Self::base("granitemoe");
        s.moe = Some((4, 2));
        s.ff = 64;
        s.meta.push(kv("attention.scale", Value::F32(0.1)));
        s.meta.push(kv("embedding_scale", Value::F32(2.0)));
        s.meta.push(kv("residual_scale", Value::F32(0.5)));
        s.meta.push(kv("logit_scale", Value::F32(2.0)));
        s.meta.push(kv("rope.freq_base", Value::F32(1_500_000.0)));
        s
    }

    /// Granite dense: the Granite scales without experts.
    pub(crate) fn granite() -> Self {
        let mut s = Self::base("granite");
        s.meta.push(kv("attention.scale", Value::F32(0.1)));
        s.meta.push(kv("embedding_scale", Value::F32(2.0)));
        s.meta.push(kv("residual_scale", Value::F32(0.5)));
        s.meta.push(kv("logit_scale", Value::F32(2.0)));
        s.meta.push(kv("rope.freq_base", Value::F32(10_000_000.0)));
        s.tied = true;
        s
    }

    /// OLMo 2: no pre-norms, post-norms, full-width Q/K norms, NEOX RoPE.
    pub(crate) fn olmo2() -> Self {
        let mut s = Self::base("olmo2");
        s.qk_norm = true;
        s.qk_norm_full = true;
        s.post_norms = true;
        s.pre_norms = false;
        s.meta.push(kv("rope.freq_base", Value::F32(500_000.0)));
        s
    }

    /// ERNIE 4.5 dense: a head dimension of 32 (wider than the embedding
    /// split), tied head.
    pub(crate) fn ernie4_5() -> Self {
        let mut s = Self::base("ernie4_5");
        s.head_dim = 32;
        s.tied = true;
        s.meta.push(kv("rope.freq_base", Value::F32(500_000.0)));
        s
    }

    /// Gemma 2: post-norms, attention and final logit soft-caps, alternating
    /// sliding and full layers with a window shorter than the prompt.
    pub(crate) fn gemma2() -> Self {
        let mut s = Self::base("gemma2");
        s.n_layer = 6;
        s.n_kv_head = 2;
        s.head_dim = 32;
        s.post_norms = true;
        s.tied = true;
        s.meta.push(kv("attention.sliding_window", u(4)));
        s.meta.push(kv("attn_logit_softcapping", Value::F32(50.0)));
        s.meta.push(kv("final_logit_softcapping", Value::F32(30.0)));
        s.meta.push(kv("rope.freq_base", Value::F32(10_000.0)));
        s
    }

    /// Gemma 3: QK-norm, post-norms, three-layer sliding pattern (two
    /// sliding, one full) with a window shorter than the prompt.
    pub(crate) fn gemma3() -> Self {
        let mut s = Self::base("gemma3");
        s.n_layer = 6;
        s.n_kv_head = 1;
        s.head_dim = 32;
        s.qk_norm = true;
        s.post_norms = true;
        s.tied = true;
        s.meta.push(kv("attention.sliding_window", u(4)));
        s.meta.push(kv("attention.sliding_window_pattern", u(3)));
        s.meta.push(kv("rope.freq_base", Value::F32(1_000_000.0)));
        s
    }

    /// Every spec the cross-architecture tests run.
    pub(crate) fn all() -> Vec<Self> {
        vec![
            Self::llama(),
            Self::qwen2(),
            Self::qwen3(),
            Self::phi3(),
            Self::smollm3(),
            Self::granitemoe(),
            Self::gemma3(),
            Self::gemma2(),
            Self::granite(),
            Self::olmo2(),
            Self::ernie4_5(),
        ]
    }
}

/// Permute the `rows = heads * head_dim` rows of `[rows, cols]` row-major
/// values so each head's adjacent pairs `(2i, 2i + 1)` land at `(i, i + d/2)`.
fn neox_row_order(values: &[f32], heads: usize, d: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; values.len()];
    for h in 0..heads {
        for i in 0..d / 2 {
            for (dst, src) in [(i, 2 * i), (i + d / 2, 2 * i + 1)] {
                let to = (h * d + dst) * cols;
                let from = (h * d + src) * cols;
                out[to..to + cols].copy_from_slice(&values[from..from + cols]);
            }
        }
    }
    out
}

/// Write the model `s` describes.
pub(crate) fn write_spec(s: &TinySpec) -> TinyModel {
    let path = temp_path(s.arch);
    let mut rng = Lcg(11);
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let dt = s.weight_dtype;
    let (nh, nkv, d, ff) = (s.n_head, s.n_kv_head, s.head_dim, s.ff);
    let qw = nh * d;
    let kw = nkv * d;

    tensors.push((
        "token_embd.weight".into(),
        quantized(&mut rng, &[VOCAB, N_EMBD], s.embd_dtype, 0.5),
    ));
    tensors.push(("output_norm.weight".into(), norm(&mut rng, N_EMBD)));
    if !s.tied {
        tensors.push((
            "output.weight".into(),
            quantized(&mut rng, &[VOCAB, N_EMBD], GgmlDType::Q8_0, 0.5),
        ));
    }
    if let Some(f) = &s.rope_freqs {
        tensors.push(("rope_freqs.weight".into(), f32_tensor(f.clone())));
    }
    if let Some((long, short)) = &s.rope_long_short {
        tensors.push(("rope_factors_long.weight".into(), f32_tensor(long.clone())));
        tensors.push((
            "rope_factors_short.weight".into(),
            f32_tensor(short.clone()),
        ));
    }

    for il in 0..s.n_layer {
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        let bb = |part: &str| format!("blk.{il}.{part}.bias");
        if s.pre_norms {
            tensors.push((b("attn_norm"), norm(&mut rng, N_EMBD)));
        }

        let mut q = rng.values(qw * N_EMBD, 0.3);
        let mut k = rng.values(kw * N_EMBD, 0.3);
        let v = rng.values(kw * N_EMBD, 0.3);
        if s.neox_rows {
            q = neox_row_order(&q, nh, d, N_EMBD);
            k = neox_row_order(&k, nkv, d, N_EMBD);
        }
        let mut bq = s.bias.then(|| rng.values(qw, 0.2));
        let mut bk = s.bias.then(|| rng.values(kw, 0.2));
        let bv = s.bias.then(|| rng.values(kw, 0.2));
        if s.neox_rows {
            bq = bq.map(|b| neox_row_order(&b, nh, d, 1));
            bk = bk.map(|b| neox_row_order(&b, nkv, d, 1));
        }
        if s.fused_qkv {
            let fused: Vec<f32> = q.iter().chain(&k).chain(&v).copied().collect();
            tensors.push((b("attn_qkv"), quantize(fused, &[qw + 2 * kw, N_EMBD], dt)));
            if let (Some(bq), Some(bk), Some(bv)) = (&bq, &bk, &bv) {
                let fb: Vec<f32> = bq.iter().chain(bk).chain(bv).copied().collect();
                tensors.push((bb("attn_qkv"), f32_tensor(fb)));
            }
        } else {
            tensors.push((b("attn_q"), quantize(q, &[qw, N_EMBD], dt)));
            tensors.push((b("attn_k"), quantize(k, &[kw, N_EMBD], dt)));
            tensors.push((b("attn_v"), quantize(v, &[kw, N_EMBD], dt)));
            for (part, bias) in [("attn_q", bq), ("attn_k", bk), ("attn_v", bv)] {
                if let Some(bias) = bias {
                    tensors.push((bb(part), f32_tensor(bias)));
                }
            }
        }
        if s.qk_norm {
            let (qn, kn) = if s.qk_norm_full { (qw, kw) } else { (d, d) };
            tensors.push((b("attn_q_norm"), norm(&mut rng, qn)));
            tensors.push((b("attn_k_norm"), norm(&mut rng, kn)));
        }
        tensors.push((
            b("attn_output"),
            quantized(&mut rng, &[N_EMBD, qw], dt, 0.3),
        ));
        if s.post_norms {
            tensors.push((b("post_attention_norm"), norm(&mut rng, N_EMBD)));
        }
        if s.pre_norms {
            tensors.push((b("ffn_norm"), norm(&mut rng, N_EMBD)));
        }
        if let Some((n_expert, _)) = s.moe {
            tensors.push((
                b("ffn_gate_inp"),
                quantized(&mut rng, &[n_expert, N_EMBD], GgmlDType::F32, 0.8),
            ));
            for part in ["ffn_gate_exps", "ffn_up_exps"] {
                tensors.push((
                    b(part),
                    quantized(&mut rng, &[n_expert, ff, N_EMBD], dt, 0.3),
                ));
            }
            tensors.push((
                b("ffn_down_exps"),
                quantized(&mut rng, &[n_expert, N_EMBD, ff], dt, 0.3),
            ));
        } else {
            let gate = rng.values(ff * N_EMBD, 0.3);
            let up = rng.values(ff * N_EMBD, 0.3);
            if s.fused_gate_up {
                let fused: Vec<f32> = gate.iter().chain(&up).copied().collect();
                tensors.push((b("ffn_up"), quantize(fused, &[2 * ff, N_EMBD], dt)));
            } else {
                tensors.push((b("ffn_gate"), quantize(gate, &[ff, N_EMBD], dt)));
                tensors.push((b("ffn_up"), quantize(up, &[ff, N_EMBD], dt)));
            }
            tensors.push((b("ffn_down"), quantized(&mut rng, &[N_EMBD, ff], dt, 0.3)));
        }
        if s.post_norms {
            tensors.push((b("post_ffw_norm"), norm(&mut rng, N_EMBD)));
        }
    }

    let p = s.arch;
    let mut meta = vec![
        kv("general.architecture", Value::String(p.into())),
        kv(&format!("{p}.block_count"), u(s.n_layer)),
        kv(&format!("{p}.context_length"), u(CONTEXT)),
        kv(&format!("{p}.embedding_length"), u(N_EMBD)),
        kv(&format!("{p}.attention.head_count"), u(nh)),
        kv(&format!("{p}.attention.head_count_kv"), u(nkv)),
        kv(&format!("{p}.attention.key_length"), u(d)),
        kv(
            &format!("{p}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-5),
        ),
    ];
    if let Some((n_expert, n_used)) = s.moe {
        meta.push(kv(&format!("{p}.expert_count"), u(n_expert)));
        meta.push(kv(&format!("{p}.expert_used_count"), u(n_used)));
    }
    for (key, value) in &s.meta {
        meta.push(kv(&format!("{p}.{key}"), value.clone()));
    }
    write(&path, &meta, &tensors);
    TinyModel { path }
}

pub(crate) fn load(model: &TinyModel, context_length: usize) -> Transformer {
    let mut reader = BufReader::new(File::open(&model.path).unwrap());
    let content = gguf_file::Content::read(&mut reader).unwrap();
    Transformer::load(&content, &model.path, context_length).unwrap()
}

pub(crate) const PROMPT: [u32; 11] = [3, 5, 7, 1, 9, 2, 4, 6, 8, 10, 12];

pub(crate) fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite(), "logit {i} not finite");
        assert!(
            (x - y).abs() <= 2e-3 * scale,
            "logit {i}: {x} vs {y} (scale {scale})"
        );
    }
}

// ---------------------------------------------------------------------------
// Qwen 3.5 (hybrid: gated delta-net + gated full attention, trailing MTP block)
// ---------------------------------------------------------------------------

/// Query heads, KV heads and head width of the attention layers.
pub(crate) const Q35_N_HEAD: usize = 4;
pub(crate) const Q35_N_KV: usize = 2;
pub(crate) const Q35_HEAD: usize = 16;
/// Rotated dimensions of an attention head (partial rotary).
pub(crate) const Q35_ROPE_DIMS: usize = 8;
/// Delta-net query/key and value head widths (`ssm.state_size` and
/// `ssm.inner_size / ssm.time_step_rank`): different on purpose.
pub(crate) const Q35_DK: usize = 8;
pub(crate) const Q35_DV: usize = 16;
pub(crate) const Q35_FF: usize = 96;

/// A tiny `qwen35` model. The defaults give five trunk layers (delta-net,
/// delta-net, delta-net, attention, delta-net) and one trailing
/// multi-token-prediction block (block 5, with its `nextn.*` tensors); four
/// value heads share two query/key heads, so the repeat-interleave is
/// exercised.
#[derive(Clone)]
pub(crate) struct TinyQwen35 {
    pub(crate) trunk_layers: usize,
    pub(crate) nextn_layers: usize,
    pub(crate) interval: usize,
    pub(crate) n_k_heads: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) dtype: GgmlDType,
    /// Metadata keys (relative to `qwen35.`) written over the defaults.
    pub(crate) meta: Vec<(String, Value)>,
}

impl Default for TinyQwen35 {
    fn default() -> Self {
        Self {
            trunk_layers: 5,
            nextn_layers: 1,
            interval: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 4,
            dtype: GgmlDType::Q8_0,
            meta: Vec::new(),
        }
    }
}

impl TinyQwen35 {
    pub(crate) fn key_dim(&self) -> usize {
        self.n_k_heads * Q35_DK
    }

    pub(crate) fn value_dim(&self) -> usize {
        self.n_v_heads * Q35_DV
    }

    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Whether block `il` is an attention block (full-attention interval or
    /// the trailing prediction block).
    pub(crate) fn is_attention(&self, il: usize) -> bool {
        il >= self.trunk_layers || (il + 1).is_multiple_of(self.interval)
    }
}

/// Write the model `s` describes.
pub(crate) fn write_qwen35(s: &TinyQwen35) -> TinyModel {
    let path = temp_path("qwen35");
    let mut rng = Lcg(35);
    let dt = s.dtype;
    let (nv, conv_dim, value_dim) = (s.n_v_heads, s.conv_dim(), s.value_dim());
    let q_width = Q35_N_HEAD * Q35_HEAD;
    let kv_width = Q35_N_KV * Q35_HEAD;
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let mut add = |n: String, t: QTensor| tensors.push((n, t));

    add(
        "token_embd.weight".into(),
        quantized(&mut rng, &[VOCAB, N_EMBD], GgmlDType::Q8_0, 0.5),
    );
    add("output_norm.weight".into(), norm(&mut rng, N_EMBD));
    for il in 0..s.trunk_layers + s.nextn_layers {
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        add(b("attn_norm"), norm(&mut rng, N_EMBD));
        // `post_attention_norm` is the norm in front of the FFN.
        add(b("post_attention_norm"), norm(&mut rng, N_EMBD));
        if s.is_attention(il) {
            // `[query | gate]` per head: twice the query width.
            add(
                b("attn_q"),
                quantized(&mut rng, &[2 * q_width, N_EMBD], dt, 0.3),
            );
            add(
                b("attn_k"),
                quantized(&mut rng, &[kv_width, N_EMBD], dt, 0.3),
            );
            add(
                b("attn_v"),
                quantized(&mut rng, &[kv_width, N_EMBD], dt, 0.3),
            );
            add(b("attn_q_norm"), norm(&mut rng, Q35_HEAD));
            add(b("attn_k_norm"), norm(&mut rng, Q35_HEAD));
            add(
                b("attn_output"),
                quantized(&mut rng, &[N_EMBD, q_width], dt, 0.3),
            );
        } else {
            add(
                b("attn_qkv"),
                quantized(&mut rng, &[conv_dim, N_EMBD], dt, 0.3),
            );
            add(
                b("attn_gate"),
                quantized(&mut rng, &[value_dim, N_EMBD], dt, 0.3),
            );
            add(b("ssm_beta"), quantized(&mut rng, &[nv, N_EMBD], dt, 0.5));
            add(b("ssm_alpha"), quantized(&mut rng, &[nv, N_EMBD], dt, 0.5));
            add(
                b("ssm_out"),
                quantized(&mut rng, &[N_EMBD, value_dim], dt, 0.3),
            );
            add(
                b("ssm_conv1d"),
                quantized(&mut rng, &[conv_dim, s.conv_kernel], GgmlDType::F32, 0.6),
            );
            add(b("ssm_norm"), norm(&mut rng, Q35_DV));
            // `ssm_a` is stored as `-exp(A_log)` and has no `.weight` suffix.
            let a: Vec<f32> = rng
                .values(nv, 1.0)
                .iter()
                .map(|x| -(0.2 + 1.5 * x.abs()))
                .collect();
            add(format!("blk.{il}.ssm_a"), f32_tensor(a));
            add(
                format!("blk.{il}.ssm_dt.bias"),
                f32_tensor(rng.values(nv, 1.0)),
            );
        }
        add(
            b("ffn_gate"),
            quantized(&mut rng, &[Q35_FF, N_EMBD], dt, 0.3),
        );
        add(b("ffn_up"), quantized(&mut rng, &[Q35_FF, N_EMBD], dt, 0.3));
        add(
            b("ffn_down"),
            quantized(&mut rng, &[N_EMBD, Q35_FF], dt, 0.3),
        );
        if il >= s.trunk_layers {
            add(
                b("nextn.eh_proj"),
                quantized(&mut rng, &[N_EMBD, 2 * N_EMBD], dt, 0.3),
            );
            add(b("nextn.enorm"), norm(&mut rng, N_EMBD));
            add(b("nextn.hnorm"), norm(&mut rng, N_EMBD));
            add(b("nextn.shared_head_norm"), norm(&mut rng, N_EMBD));
        }
    }

    let mut meta = vec![
        kv("general.architecture", Value::String("qwen35".into())),
        kv("qwen35.block_count", u(s.trunk_layers + s.nextn_layers)),
        kv("qwen35.context_length", u(CONTEXT)),
        kv("qwen35.embedding_length", u(N_EMBD)),
        kv("qwen35.feed_forward_length", u(Q35_FF)),
        kv("qwen35.attention.head_count", u(Q35_N_HEAD)),
        kv("qwen35.attention.head_count_kv", u(Q35_N_KV)),
        kv("qwen35.attention.key_length", u(Q35_HEAD)),
        kv("qwen35.attention.value_length", u(Q35_HEAD)),
        kv("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        kv(
            "qwen35.rope.dimension_sections",
            Value::Array(vec![u(2), u(1), u(1), u(0)]),
        ),
        kv("qwen35.rope.freq_base", Value::F32(1e7)),
        kv("qwen35.rope.dimension_count", u(Q35_ROPE_DIMS)),
        kv("qwen35.ssm.conv_kernel", u(s.conv_kernel)),
        kv("qwen35.ssm.state_size", u(Q35_DK)),
        kv("qwen35.ssm.group_count", u(s.n_k_heads)),
        kv("qwen35.ssm.time_step_rank", u(s.n_v_heads)),
        kv("qwen35.ssm.inner_size", u(value_dim)),
        kv("qwen35.full_attention_interval", u(s.interval)),
        kv("qwen35.nextn_predict_layers", u(s.nextn_layers)),
    ];
    for (key, value) in &s.meta {
        let full = format!("qwen35.{key}");
        match meta.iter_mut().find(|(k, _)| *k == full) {
            Some(entry) => entry.1 = value.clone(),
            None => meta.push((full, value.clone())),
        }
    }
    write(&path, &meta, &tensors);
    TinyModel { path }
}

// ---------------------------------------------------------------------------
// Gemma 4 drafter
// ---------------------------------------------------------------------------

/// A tiny `gemma4-assistant` drafter for the tiny Gemma 4 model: width 32,
/// two layers (sliding, full) whose head dimensions (32 / 64) and single KV
/// head match the target's caches, projections to and from the target's
/// width of 64.
pub(crate) fn write_tiny_drafter() -> TinyModel {
    const N: usize = 32;
    const HEADS: usize = 2;
    let path = temp_path("gemma4-assistant");
    let mut rng = Lcg(23);
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let q4 = GgmlDType::Q4_0;
    tensors.push((
        "token_embd.weight".into(),
        quantized(&mut rng, &[VOCAB, N], GgmlDType::F16, 0.5),
    ));
    tensors.push(("output_norm.weight".into(), norm(&mut rng, N)));
    tensors.push((
        "nextn.pre_projection.weight".into(),
        quantized(&mut rng, &[N, 2 * N_EMBD], q4, 0.3),
    ));
    tensors.push((
        "nextn.post_projection.weight".into(),
        quantized(&mut rng, &[N_EMBD, N], q4, 0.3),
    ));
    for (il, d) in [G4_HEAD_SWA, G4_HEAD_FULL].into_iter().enumerate() {
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        tensors.push((b("attn_norm"), norm(&mut rng, N)));
        tensors.push((b("attn_q"), quantized(&mut rng, &[HEADS * d, N], q4, 0.3)));
        tensors.push((b("attn_q_norm"), norm(&mut rng, d)));
        tensors.push((
            b("attn_output"),
            quantized(&mut rng, &[N, HEADS * d], q4, 0.3),
        ));
        tensors.push((b("post_attention_norm"), norm(&mut rng, N)));
        tensors.push((b("ffn_norm"), norm(&mut rng, N)));
        tensors.push((b("ffn_gate"), quantized(&mut rng, &[64, N], q4, 0.3)));
        tensors.push((b("ffn_up"), quantized(&mut rng, &[64, N], q4, 0.3)));
        tensors.push((b("ffn_down"), quantized(&mut rng, &[N, 64], q4, 0.3)));
        tensors.push((b("post_ffw_norm"), norm(&mut rng, N)));
        tensors.push((b("layer_output_scale"), f32_tensor(vec![0.9])));
    }
    let p = "gemma4-assistant";
    let meta = vec![
        kv("general.architecture", Value::String(p.into())),
        kv(&format!("{p}.block_count"), u(2)),
        kv(&format!("{p}.context_length"), u(CONTEXT)),
        kv(&format!("{p}.embedding_length"), u(N)),
        kv(&format!("{p}.embedding_length_out"), u(N_EMBD)),
        kv(&format!("{p}.attention.head_count"), u(HEADS)),
        kv(&format!("{p}.attention.head_count_kv"), u(1)),
        kv(&format!("{p}.attention.key_length"), u(G4_HEAD_FULL)),
        kv(&format!("{p}.attention.key_length_swa"), u(G4_HEAD_SWA)),
        kv(&format!("{p}.attention.sliding_window"), u(G4_WINDOW)),
        kv(
            &format!("{p}.attention.sliding_window_pattern"),
            Value::Array(vec![Value::Bool(true), Value::Bool(false)]),
        ),
        kv(&format!("{p}.attention.shared_kv_layers"), u(2)),
        kv(&format!("{p}.embedding_length_per_layer_input"), u(0)),
        kv(&format!("{p}.rope.freq_base"), Value::F32(1e6)),
        kv(&format!("{p}.rope.freq_base_swa"), Value::F32(1e4)),
        kv(&format!("{p}.rope.dimension_count"), u(G4_HEAD_FULL)),
        kv(&format!("{p}.rope.dimension_count_swa"), u(G4_HEAD_SWA)),
        kv(
            &format!("{p}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-6),
        ),
    ];
    write(&path, &meta, &tensors);
    TinyModel { path }
}
