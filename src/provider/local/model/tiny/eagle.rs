//! A tiny `mistral4-eagle` drafter for the tiny `mistral4` model of
//! [`write_mistral4`](super::write_mistral4): two latent-attention layers of
//! the target's geometry over a dense FFN, the fusion projection and the
//! final norm, with the target's RoPE, YaRN and query scaling.

use super::{
    CONTEXT, GgmlDType, Lcg, M4_DENSE_FF, M4_KV_LORA, M4_N_HEAD, M4_NOPE, M4_ORIG_CTX, M4_Q_LORA,
    M4_ROT, M4_TEMP_LEN, M4_V_HEAD, N_EMBD, TinyModel, VOCAB, Value, kv, norm, quantized,
    temp_path, u, write,
};

/// What the tiny drafter is made of.
#[derive(Clone)]
pub(crate) struct TinyEagle {
    pub(crate) layers: usize,
    /// Dtype of the dense matrices.
    pub(crate) dtype: GgmlDType,
    /// Metadata keys (relative to `mistral4-eagle.`) written over the
    /// defaults.
    pub(crate) meta: Vec<(String, Value)>,
}

impl Default for TinyEagle {
    fn default() -> Self {
        Self {
            layers: 2,
            dtype: GgmlDType::Q8_0,
            meta: Vec::new(),
        }
    }
}

/// Write the drafter `s` describes.
pub(crate) fn write_tiny_eagle(s: &TinyEagle) -> TinyModel {
    let p = "mistral4-eagle";
    let path = temp_path(p);
    let mut rng = Lcg(31);
    let dt = s.dtype;
    let q_head = M4_NOPE + M4_ROT;
    let mut tensors = Vec::new();
    let mut add = |n: String, t| tensors.push((n, t));

    add(
        "nextn.eh_proj.weight".into(),
        quantized(&mut rng, &[N_EMBD, 2 * N_EMBD], dt, 0.2),
    );
    add("output_norm.weight".into(), norm(&mut rng, N_EMBD));
    for il in 0..s.layers {
        let b = |part: &str| format!("blk.{il}.{part}.weight");
        add(b("attn_norm"), norm(&mut rng, N_EMBD));
        add(
            b("attn_q_a"),
            quantized(&mut rng, &[M4_Q_LORA, N_EMBD], dt, 0.3),
        );
        add(b("attn_q_a_norm"), norm(&mut rng, M4_Q_LORA));
        add(
            b("attn_q_b"),
            quantized(&mut rng, &[M4_N_HEAD * q_head, M4_Q_LORA], dt, 0.3),
        );
        add(
            b("attn_kv_a_mqa"),
            quantized(&mut rng, &[M4_KV_LORA + M4_ROT, N_EMBD], dt, 0.3),
        );
        add(b("attn_kv_a_norm"), norm(&mut rng, M4_KV_LORA));
        add(
            b("attn_k_b"),
            quantized(
                &mut rng,
                &[M4_N_HEAD, M4_KV_LORA, M4_NOPE],
                GgmlDType::F32,
                0.5,
            ),
        );
        add(
            b("attn_v_b"),
            quantized(
                &mut rng,
                &[M4_N_HEAD, M4_V_HEAD, M4_KV_LORA],
                GgmlDType::F32,
                0.5,
            ),
        );
        add(
            b("attn_output"),
            quantized(&mut rng, &[N_EMBD, M4_N_HEAD * M4_V_HEAD], dt, 0.3),
        );
        add(b("ffn_norm"), norm(&mut rng, N_EMBD));
        for part in ["ffn_gate", "ffn_up"] {
            add(
                b(part),
                quantized(&mut rng, &[M4_DENSE_FF, N_EMBD], dt, 0.3),
            );
        }
        add(
            b("ffn_down"),
            quantized(&mut rng, &[N_EMBD, M4_DENSE_FF], dt, 0.3),
        );
    }

    let mut meta = vec![
        kv("general.architecture", Value::String(p.into())),
        kv(&format!("{p}.block_count"), u(s.layers)),
        kv(&format!("{p}.context_length"), u(CONTEXT)),
        kv(&format!("{p}.embedding_length"), u(N_EMBD)),
        kv(&format!("{p}.embedding_length_out"), u(N_EMBD)),
        kv(&format!("{p}.vocab_size"), u(VOCAB)),
        kv(&format!("{p}.feed_forward_length"), u(M4_DENSE_FF)),
        kv(&format!("{p}.attention.head_count"), u(M4_N_HEAD)),
        kv(&format!("{p}.attention.head_count_kv"), u(1)),
        kv(&format!("{p}.attention.key_length"), u(M4_KV_LORA + M4_ROT)),
        kv(&format!("{p}.attention.value_length"), u(M4_KV_LORA)),
        kv(&format!("{p}.attention.q_lora_rank"), u(M4_Q_LORA)),
        kv(&format!("{p}.attention.kv_lora_rank"), u(M4_KV_LORA)),
        kv(&format!("{p}.attention.key_length_mla"), u(q_head)),
        kv(&format!("{p}.attention.value_length_mla"), u(M4_V_HEAD)),
        kv(
            &format!("{p}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-6),
        ),
        kv(&format!("{p}.rope.dimension_count"), u(M4_ROT)),
        kv(&format!("{p}.rope.freq_base"), Value::F32(10_000.0)),
        kv(
            &format!("{p}.rope.scaling.type"),
            Value::String("yarn".into()),
        ),
        kv(&format!("{p}.rope.scaling.factor"), Value::F32(2.0)),
        kv(
            &format!("{p}.rope.scaling.original_context_length"),
            u(M4_ORIG_CTX),
        ),
        kv(
            &format!("{p}.rope.scaling.yarn_beta_fast"),
            Value::F32(32.0),
        ),
        kv(&format!("{p}.rope.scaling.yarn_beta_slow"), Value::F32(1.0)),
        kv(
            &format!("{p}.rope.scaling.yarn_log_multiplier"),
            Value::F32(0.1),
        ),
        kv(&format!("{p}.attention.temperature_scale"), Value::F32(0.3)),
        kv(&format!("{p}.attention.temperature_length"), u(M4_TEMP_LEN)),
        kv(&format!("{p}.leading_dense_block_count"), u(s.layers)),
    ];
    for (key, value) in &s.meta {
        let full = format!("{p}.{key}");
        match meta.iter_mut().find(|(k, _)| *k == full) {
            Some(entry) => entry.1 = value.clone(),
            None => meta.push((full, value.clone())),
        }
    }
    write(&path, &meta, &tensors);
    TinyModel { path }
}
