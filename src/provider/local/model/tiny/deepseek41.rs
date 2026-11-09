//! A tiny `deepseek41` model: hyper-connection streams, latent attention with
//! a sliding window, compressed sparse attention with an indexer and
//! candidate blocks, engram layers, and a `sqrtsoftplus`-routed MoE with a
//! SwiGLU clamp.
//!
//! Six layers over the usual tiny width (64): compression ratios
//! `[0, 2, 2, 1, 1, 1]`, so layer 1 owns ratio-2 rows (and scores them),
//! layer 2 reads layer 1's rows and picks, layer 3 owns ratio-1 rows and
//! publishes the candidate blocks, layer 4 scores inside them and layer 5
//! reuses its picks; engram on layers 1 and 4. The window is 8 and the
//! indexer keeps 4 rows, so a prompt of a few dozen tokens uses every path.

use std::fs::File;

use super::{
    GgmlDType, Lcg, N_EMBD, TinyModel, VOCAB, Value, gguf, kv, norm, quantized, temp_path, u,
};
use crate::provider::local::gguf::GgufType;

pub(crate) const DS_LAYERS: usize = 6;
pub(crate) const DS_STREAMS: usize = 4;
pub(crate) const DS_HEADS: usize = 4;
pub(crate) const DS_HEAD_DIM: usize = 32;
pub(crate) const DS_ROT: usize = 8;
pub(crate) const DS_WINDOW: usize = 8;
pub(crate) const DS_Q_LORA: usize = 32;
pub(crate) const DS_OUT_GROUPS: usize = 2;
pub(crate) const DS_OUT_RANK: usize = 16;
pub(crate) const DS_INDEX_HEADS: usize = 4;
pub(crate) const DS_INDEX_DIM: usize = 32;
pub(crate) const DS_TOP_K: usize = 4;
pub(crate) const DS_EXPERTS: usize = 8;
pub(crate) const DS_USED: usize = 3;
pub(crate) const DS_FF: usize = 32;
pub(crate) const DS_NGRAM: usize = 3;
pub(crate) const DS_ENGRAM_HEADS: usize = 2;
pub(crate) const DS_ENGRAM_DIM: usize = 32;
/// Context of the tiny model: room for a prompt past two 32-token blocks.
pub(crate) const DS_CONTEXT: usize = 96;
pub(crate) const DS_RATIOS: [usize; DS_LAYERS] = [0, 2, 2, 1, 1, 1];
pub(crate) const DS_KV_SOURCES: [usize; 2] = [1, 3];
pub(crate) const DS_INDEX_SOURCES: [usize; 3] = [1, 3, 4];
pub(crate) const DS_ENGRAM_LAYERS: [usize; 2] = [1, 4];
/// Bucket counts of the engram heads of the two engram layers, order 2 then
/// 3, two heads each.
const BUCKETS: [u64; 8] = [11, 13, 17, 19, 23, 29, 31, 37];

/// What the tiny model is made of.
#[derive(Clone)]
pub(crate) struct TinyDeepseek41 {
    /// Dtype of the dense matrices.
    pub(crate) dtype: GgmlDType,
    /// The routed experts in MXFP4 (the checkpoint's native type) instead of
    /// `dtype`.
    pub(crate) mxfp4: bool,
    /// The candidate source layer (`None` omits the key: every indexer layer
    /// scores every row).
    pub(crate) candidates: Option<usize>,
    /// Metadata keys (relative to `deepseek41.`) written over the defaults.
    pub(crate) meta: Vec<(String, Value)>,
}

impl Default for TinyDeepseek41 {
    fn default() -> Self {
        Self {
            dtype: GgmlDType::Q8_0,
            mxfp4: false,
            candidates: Some(3),
            meta: Vec::new(),
        }
    }
}

/// MXFP4 blocks (17 bytes per 32 values: the E8M0 scale, then the nibbles of
/// the first and second half) of `values`, whose length is a multiple of 32.
pub(crate) fn mxfp4_bytes(values: &[f32]) -> Vec<u8> {
    const GRID: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];
    let nearest = |v: f32| {
        let m = GRID
            .iter()
            .enumerate()
            .min_by(|a, b| (v.abs() - a.1).abs().total_cmp(&(v.abs() - b.1).abs()))
            .map_or(0, |(i, _)| i as u8);
        if v < 0.0 && m != 0 { m | 8 } else { m }
    };
    let mut out = Vec::with_capacity(values.len() / 32 * 17);
    for block in values.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, x| m.max(x.abs()));
        let e = if amax > 0.0 {
            ((amax.log2().floor() as i32) - 2 + 127).clamp(0, 254) as u8
        } else {
            0
        };
        let inv = 1.0 / crate::provider::local::model::quant::e8m0_half(e);
        out.push(e);
        let (lo, hi) = block.split_at(16);
        for (a, b) in lo.iter().zip(hi) {
            out.push(nearest(a * inv) | (nearest(b * inv) << 4));
        }
    }
    out
}

/// One tensor to write: name, type, dimensions (slowest first), bytes.
pub(super) type Raw = (String, GgufType, Vec<usize>, Vec<u8>);

pub(super) fn raw_q(name: String, qt: &candle_core::quantized::QTensor) -> Raw {
    (
        name,
        GgufType::Ggml(qt.dtype()),
        qt.shape().dims().to_vec(),
        qt.data().unwrap().into_owned(),
    )
}

pub(super) fn raw_f32(name: String, dims: &[usize], values: &[f32]) -> Raw {
    (
        name,
        GgufType::Ggml(GgmlDType::F32),
        dims.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// Write the model `s` describes.
pub(crate) fn write_deepseek41(s: &TinyDeepseek41) -> TinyModel {
    let p = "deepseek41";
    let path = temp_path(p);
    let mut rng = Lcg(41);
    let dt = s.dtype;
    let mut tensors: Vec<Raw> = Vec::new();

    let mat = |rng: &mut Lcg, name: String, dims: &[usize], scale: f32| {
        raw_q(name, &quantized(rng, dims, dt, scale))
    };
    tensors.push(raw_q(
        "token_embd.weight".into(),
        &quantized(&mut rng, &[VOCAB, N_EMBD], GgmlDType::Q8_0, 0.5),
    ));
    tensors.push(raw_q("output_norm.weight".into(), &norm(&mut rng, N_EMBD)));
    tensors.push(raw_q(
        "output.weight".into(),
        &quantized(&mut rng, &[VOCAB, N_EMBD], GgmlDType::Q8_0, 0.5),
    ));
    // The table of every engram layer's rows, one after the other.
    let rows: u64 = BUCKETS.iter().sum();
    tensors.push(raw_q(
        "per_layer_token_embd.weight".into(),
        &quantized(
            &mut rng,
            &[rows as usize, DS_ENGRAM_DIM],
            GgmlDType::Q8_0,
            0.5,
        ),
    ));

    let mix = (2 + DS_STREAMS) * DS_STREAMS;
    for (il, &ratio) in DS_RATIOS.iter().enumerate() {
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        tensors.push(raw_q(b("attn_norm"), &norm(&mut rng, N_EMBD)));
        tensors.push(raw_q(b("ffn_norm"), &norm(&mut rng, N_EMBD)));
        for (site, scale) in [("attn", 0.5f32), ("ffn", 0.5)] {
            tensors.push(raw_f32(
                b(&format!("hc_{site}_fn")),
                &[mix, DS_STREAMS * N_EMBD],
                &rng.values(mix * DS_STREAMS * N_EMBD, 0.15),
            ));
            tensors.push(raw_f32(
                b(&format!("hc_{site}_base")),
                &[mix],
                &rng.values(mix, 0.6),
            ));
            tensors.push(raw_f32(
                b(&format!("hc_{site}_scale")),
                &[3],
                &[scale, scale, scale],
            ));
        }
        tensors.push(mat(&mut rng, b("attn_q_a"), &[DS_Q_LORA, N_EMBD], 0.3));
        tensors.push(raw_q(b("attn_q_a_norm"), &norm(&mut rng, DS_Q_LORA)));
        tensors.push(mat(
            &mut rng,
            b("attn_q_b"),
            &[DS_HEADS * DS_HEAD_DIM, DS_Q_LORA],
            0.3,
        ));
        tensors.push(mat(&mut rng, b("attn_kv"), &[DS_HEAD_DIM, N_EMBD], 0.3));
        tensors.push(raw_q(b("attn_kv_a_norm"), &norm(&mut rng, DS_HEAD_DIM)));
        tensors.push(mat(
            &mut rng,
            b("attn_output_a"),
            &[
                DS_OUT_GROUPS * DS_OUT_RANK,
                DS_HEADS * DS_HEAD_DIM / DS_OUT_GROUPS,
            ],
            0.3,
        ));
        tensors.push(mat(
            &mut rng,
            b("attn_output_b"),
            &[N_EMBD, DS_OUT_GROUPS * DS_OUT_RANK],
            0.3,
        ));
        tensors.push(raw_f32(
            b("attn_sinks"),
            &[DS_HEADS],
            &rng.values(DS_HEADS, 0.5),
        ));
        if DS_KV_SOURCES.contains(&il) {
            tensors.push(mat(
                &mut rng,
                b("attn_compressor_kv"),
                &[DS_HEAD_DIM, N_EMBD],
                0.3,
            ));
            if ratio == 2 {
                tensors.push(mat(
                    &mut rng,
                    b("attn_compressor_gate"),
                    &[DS_HEAD_DIM, N_EMBD],
                    0.5,
                ));
            }
            tensors.push(raw_q(
                b("attn_compressor_norm"),
                &norm(&mut rng, DS_HEAD_DIM),
            ));
            tensors.push(mat(
                &mut rng,
                b("indexer.attn_k"),
                &[DS_INDEX_DIM, DS_HEAD_DIM],
                0.5,
            ));
            tensors.push(raw_q(b("indexer.k_norm"), &norm(&mut rng, DS_INDEX_DIM)));
        }
        if DS_INDEX_SOURCES.contains(&il) {
            tensors.push(mat(
                &mut rng,
                b("indexer.attn_q_b"),
                &[DS_INDEX_HEADS * DS_INDEX_DIM, DS_Q_LORA],
                0.5,
            ));
            tensors.push(mat(
                &mut rng,
                b("indexer.proj"),
                &[DS_INDEX_HEADS, N_EMBD],
                0.5,
            ));
        }
        if let Some(ordinal) = DS_ENGRAM_LAYERS.iter().position(|&l| l == il) {
            let _ = ordinal;
            tensors.push(mat(
                &mut rng,
                b("engram_kv"),
                &[
                    (DS_STREAMS + 1) * N_EMBD,
                    (DS_NGRAM - 1) * DS_ENGRAM_HEADS * DS_ENGRAM_DIM,
                ],
                0.3,
            ));
            tensors.push(raw_f32(
                b("engram_gate"),
                &[DS_STREAMS, N_EMBD],
                &rng.values(DS_STREAMS * N_EMBD, 0.8),
            ));
        }
        tensors.push(raw_f32(
            format!("blk.{il}.exp_probs_b.bias"),
            &[DS_EXPERTS],
            &rng.values(DS_EXPERTS, 0.3),
        ));
        tensors.push(raw_f32(
            b("ffn_gate_inp"),
            &[DS_EXPERTS, N_EMBD],
            &rng.values(DS_EXPERTS * N_EMBD, 0.8),
        ));
        for (part, dims) in [
            ("ffn_gate_exps", [DS_EXPERTS, DS_FF, N_EMBD]),
            ("ffn_up_exps", [DS_EXPERTS, DS_FF, N_EMBD]),
            ("ffn_down_exps", [DS_EXPERTS, N_EMBD, DS_FF]),
        ] {
            let values = rng.values(dims.iter().product(), 0.3);
            if s.mxfp4 {
                tensors.push((
                    b(part),
                    GgufType::Mxfp4,
                    dims.to_vec(),
                    mxfp4_bytes(&values),
                ));
            } else {
                let t = candle_core::Tensor::from_vec(values, &dims[..], &candle_core::Device::Cpu)
                    .unwrap();
                tensors.push(raw_q(
                    b(part),
                    &candle_core::quantized::QTensor::quantize(&t, dt).unwrap(),
                ));
            }
        }
        for (part, dims) in [
            ("ffn_gate_shexp", [DS_FF, N_EMBD]),
            ("ffn_up_shexp", [DS_FF, N_EMBD]),
            ("ffn_down_shexp", [N_EMBD, DS_FF]),
        ] {
            tensors.push(mat(&mut rng, b(part), &dims, 0.3));
        }
    }

    let kf = |key: &str, value: Value| kv(&format!("{p}.{key}"), value);
    let ints = |v: &[usize]| Value::Array(v.iter().map(|&n| Value::U32(n as u32)).collect());
    let mut multipliers = Vec::new();
    for layer in 0..DS_ENGRAM_LAYERS.len() {
        for order in 0..DS_NGRAM {
            multipliers.push(Value::U64(
                2 * (3 + 5 * layer as u64 + 7 * order as u64) + 1,
            ));
        }
    }
    let mut offsets = Vec::new();
    let mut at = 0u64;
    for &b in &BUCKETS {
        offsets.push(Value::U64(at));
        at += b;
    }
    let mut meta = vec![
        kv("general.architecture", Value::String(p.into())),
        kf("block_count", u(DS_LAYERS)),
        kf("context_length", u(DS_CONTEXT)),
        kf("embedding_length", u(N_EMBD)),
        kf("attention.head_count", u(DS_HEADS)),
        kf("attention.head_count_kv", u(1)),
        kf("attention.key_length", u(DS_HEAD_DIM)),
        kf("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        kf("attention.sliding_window", u(DS_WINDOW)),
        kf("attention.q_lora_rank", u(DS_Q_LORA)),
        kf("attention.output_group_count", u(DS_OUT_GROUPS)),
        kf("attention.output_lora_rank", u(DS_OUT_RANK)),
        kf("attention.compress_ratios", ints(&DS_RATIOS)),
        kf("attention.kv_source_layers", ints(&DS_KV_SOURCES)),
        kf("attention.compress_rope_freq_base", Value::F32(160_000.0)),
        kf("attention.indexer.source_layers", ints(&DS_INDEX_SOURCES)),
        kf("attention.indexer.head_count", u(DS_INDEX_HEADS)),
        kf("attention.indexer.key_length", u(DS_INDEX_DIM)),
        kf("attention.indexer.top_k", u(DS_TOP_K)),
        kf("rope.dimension_count", u(DS_ROT)),
        kf("rope.freq_base", Value::F32(10_000.0)),
        kf("rope.scaling.type", Value::String("yarn".into())),
        kf("rope.scaling.factor", Value::F32(4.0)),
        kf("rope.scaling.original_context_length", u(16)),
        kf("rope.scaling.yarn_beta_fast", Value::F32(32.0)),
        kf("rope.scaling.yarn_beta_slow", Value::F32(1.0)),
        kf("expert_count", u(DS_EXPERTS)),
        kf("expert_used_count", u(DS_USED)),
        kf("expert_shared_count", u(1)),
        kf("expert_feed_forward_length", u(DS_FF)),
        kf("expert_weights_scale", Value::F32(1.5)),
        kf("expert_weights_norm", Value::Bool(true)),
        kf("expert_gating_func", u(4)),
        kf(
            "swiglu_clamp_exp",
            Value::Array(vec![Value::F32(2.0); DS_LAYERS]),
        ),
        kf(
            "swiglu_clamp_shexp",
            Value::Array(vec![Value::F32(2.0); DS_LAYERS]),
        ),
        kf("hyper_connection.count", u(DS_STREAMS)),
        kf("hyper_connection.sinkhorn_iterations", u(20)),
        kf("hyper_connection.epsilon", Value::F32(1e-6)),
        kf("ple.layers", ints(&DS_ENGRAM_LAYERS)),
        kf("ple.ngram_size", u(DS_NGRAM)),
        kf("ple.heads_per_ngram", u(DS_ENGRAM_HEADS)),
        kf("embedding_length_per_layer_input", u(DS_ENGRAM_DIM)),
        kf("ple.layer_multipliers", Value::Array(multipliers)),
        kf("ple.head_offsets", Value::Array(offsets)),
        kf(
            "ple.head_vocab_sizes",
            Value::Array(BUCKETS.iter().map(|&b| Value::U64(b)).collect()),
        ),
        kf("engram.pad_token_id", u(0)),
        kf(
            "engram.token_map",
            Value::Array((0..VOCAB).map(|t| Value::I32((t / 2) as i32)).collect()),
        ),
    ];
    if let Some(src) = s.candidates {
        meta.push(kf("attention.indexer.candidate_source_layer", u(src)));
        meta.push(kf("attention.indexer.candidate_top_k_blocks", u(2)));
        meta.push(kf("attention.indexer.candidate_block_size", u(4)));
    }
    for (key, value) in &s.meta {
        let full = format!("{p}.{key}");
        match meta.iter_mut().find(|(k, _)| *k == full) {
            Some(entry) => entry.1 = value.clone(),
            None => meta.push((full, value.clone())),
        }
    }

    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let raw: Vec<gguf::RawTensor> = tensors
        .iter()
        .map(|(name, dtype, dims, data)| gguf::RawTensor {
            name,
            dims,
            dtype: *dtype,
            data,
        })
        .collect();
    let mut file = File::create(&path).unwrap();
    gguf::write_raw(&mut file, &meta_refs, &raw).unwrap();
    TinyModel { path }
}
