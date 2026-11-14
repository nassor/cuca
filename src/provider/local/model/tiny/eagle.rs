//! A tiny `mistral4-eagle` drafter for the tiny `mistral4` model of
//! [`write_mistral4`](super::write_mistral4): two latent-attention layers of
//! the target's geometry over a dense FFN, the fusion projection and the
//! final norm, with the target's RoPE, YaRN and query scaling.

use super::{
    GgmlDType, Lcg, N_EMBD, TinyModel, VOCAB, Value, kv, m4_attention, m4_dense_ffn, m4_meta, norm,
    overlay, quantized, temp_path, u, write,
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
    let mut tensors = Vec::new();
    let mut add = |n: String, t| tensors.push((n, t));

    add(
        "nextn.eh_proj.weight".into(),
        quantized(&mut rng, &[N_EMBD, 2 * N_EMBD], dt, 0.2),
    );
    add("output_norm.weight".into(), norm(&mut rng, N_EMBD));
    for il in 0..s.layers {
        m4_attention(&mut add, &mut rng, il, dt);
        m4_dense_ffn(&mut add, &mut rng, il, dt);
    }

    let mut meta = m4_meta(p, s.layers);
    meta.extend([
        kv(&format!("{p}.embedding_length_out"), u(N_EMBD)),
        kv(&format!("{p}.vocab_size"), u(VOCAB)),
        kv(&format!("{p}.leading_dense_block_count"), u(s.layers)),
    ]);
    overlay(&mut meta, p, &s.meta);
    write(&path, &meta, &tensors);
    TinyModel { path }
}
