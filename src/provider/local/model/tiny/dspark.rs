//! A tiny DSpark drafter (`dflash` with hyper-connection streams) for the tiny
//! [`TinyDeepseek41`](super::TinyDeepseek41) target: three window-only stages
//! over the target's geometry, six experts (two used), a Markov head of rank
//! 32 and a confidence head. It reads the inputs of the target's layers 3, 4
//! and 5, with block 5 and noise token 14.

use std::fs::File;

use super::deepseek41::{
    DS_FF, DS_HEAD_DIM, DS_HEADS, DS_OUT_GROUPS, DS_OUT_RANK, DS_Q_LORA, DS_ROT, DS_STREAMS,
    DS_WINDOW, Raw, mxfp4_bytes, raw_f32, raw_q,
};
use super::{
    DFLASH_MASK, GgmlDType, Lcg, N_EMBD, TinyModel, VOCAB, Value, gguf, kv, norm, quantized,
    temp_path, u,
};
use crate::provider::local::gguf::GgufType;

const DSPARK_STAGES: usize = 3;
const DSPARK_BLOCK: usize = 5;
const MARKOV_RANK: usize = 32;
const EXPERTS: usize = 6;
const USED: usize = 2;
const TARGET_LAYERS: [usize; 3] = [3, 4, 5];

/// What the tiny drafter is made of.
#[derive(Clone)]
pub(crate) struct TinyDspark {
    pub(crate) seed: u64,
    /// Dtype of the dense matrices.
    pub(crate) dtype: GgmlDType,
    /// The routed experts in MXFP4 instead of `dtype`.
    pub(crate) mxfp4: bool,
    /// Metadata keys (relative to `dflash.`) written over the defaults.
    pub(crate) meta: Vec<(String, Value)>,
}

impl Default for TinyDspark {
    fn default() -> Self {
        Self {
            seed: 17,
            dtype: GgmlDType::Q8_0,
            mxfp4: true,
            meta: Vec::new(),
        }
    }
}

/// Write the drafter `s` describes.
pub(crate) fn write_tiny_dspark(s: &TinyDspark) -> TinyModel {
    let p = "dflash";
    let path = temp_path("dspark");
    let mut rng = Lcg(s.seed);
    let dt = s.dtype;
    let mut tensors: Vec<Raw> = Vec::new();
    let mat = |rng: &mut Lcg, name: String, dims: &[usize], scale: f32| {
        raw_q(name, &quantized(rng, dims, dt, scale))
    };
    tensors.push(mat(
        &mut rng,
        "fc.weight".into(),
        &[N_EMBD, TARGET_LAYERS.len() * N_EMBD],
        0.2,
    ));
    tensors.push(raw_q(
        "enc.output_norm.weight".into(),
        &norm(&mut rng, N_EMBD),
    ));
    tensors.push(raw_q("output_norm.weight".into(), &norm(&mut rng, N_EMBD)));
    for name in ["markov_w1.weight", "markov_w2.weight"] {
        tensors.push(mat(&mut rng, name.into(), &[VOCAB, MARKOV_RANK], 0.6));
    }
    tensors.push(raw_f32(
        "conf_proj.weight".into(),
        &[N_EMBD + MARKOV_RANK],
        &rng.values(N_EMBD + MARKOV_RANK, 0.3),
    ));
    let mix = (2 + DS_STREAMS) * DS_STREAMS;
    for il in 0..DSPARK_STAGES {
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        tensors.push(raw_q(b("attn_norm"), &norm(&mut rng, N_EMBD)));
        tensors.push(raw_q(b("ffn_norm"), &norm(&mut rng, N_EMBD)));
        for site in ["attn", "ffn"] {
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
                &[0.5, 0.5, 0.5],
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
        tensors.push(raw_f32(
            format!("blk.{il}.exp_probs_b.bias"),
            &[EXPERTS],
            &rng.values(EXPERTS, 0.3),
        ));
        tensors.push(raw_f32(
            b("ffn_gate_inp"),
            &[EXPERTS, N_EMBD],
            &rng.values(EXPERTS * N_EMBD, 0.8),
        ));
        for (part, dims) in [
            ("ffn_gate_exps", [EXPERTS, DS_FF, N_EMBD]),
            ("ffn_up_exps", [EXPERTS, DS_FF, N_EMBD]),
            ("ffn_down_exps", [EXPERTS, N_EMBD, DS_FF]),
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
    let mut meta = vec![
        kv("general.architecture", Value::String(p.into())),
        kf("block_count", u(DSPARK_STAGES)),
        kf("context_length", u(96)),
        kf("embedding_length", u(N_EMBD)),
        kf("feed_forward_length", u(DS_FF)),
        kf("attention.head_count", u(DS_HEADS)),
        kf("attention.head_count_kv", u(1)),
        kf("attention.key_length", u(DS_HEAD_DIM)),
        kf("attention.value_length", u(DS_HEAD_DIM)),
        kf("rope.dimension_count", u(DS_ROT)),
        kf("rope.freq_base", Value::F32(10_000.0)),
        kf("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        kf("attention.q_lora_rank", u(DS_Q_LORA)),
        kf("attention.sliding_window", u(DS_WINDOW)),
        kf("expert_count", u(EXPERTS)),
        kf("expert_used_count", u(USED)),
        kf("expert_feed_forward_length", u(DS_FF)),
        kf("expert_shared_count", u(1)),
        kf("expert_weights_scale", Value::F32(1.5)),
        kf("expert_weights_norm", Value::Bool(true)),
        kf("expert_gating_func", u(4)),
        kf(
            "swiglu_clamp_exp",
            Value::Array(vec![Value::F32(2.0); DSPARK_STAGES]),
        ),
        kf(
            "swiglu_clamp_shexp",
            Value::Array(vec![Value::F32(2.0); DSPARK_STAGES]),
        ),
        kf("attention.output_group_count", u(DS_OUT_GROUPS)),
        kf("attention.output_lora_rank", u(DS_OUT_RANK)),
        kf("attention.compress_ratios", ints(&[0; DSPARK_STAGES])),
        kf("hyper_connection.count", u(DS_STREAMS)),
        kf("hyper_connection.sinkhorn_iterations", u(20)),
        kf("hyper_connection.epsilon", Value::F32(1e-6)),
        kf("hyper_connection.single_pass", Value::Bool(true)),
        kf("block_size", u(DSPARK_BLOCK)),
        kf("target_layers", ints(&TARGET_LAYERS)),
        kf("sample_from_anchor", Value::Bool(true)),
        kf("has_confidence_head", Value::Bool(true)),
        kv("tokenizer.ggml.mask_token_id", Value::U32(DFLASH_MASK)),
    ];
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
