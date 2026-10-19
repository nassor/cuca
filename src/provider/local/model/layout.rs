//! GGUF tensor names and the per-layer tensor layout.
//!
//! [`LayerLayout::read`] inspects the tensor index (names and shapes, no
//! data) and records what the layer has: fused or separate QKV and gate/up
//! projections, biases, QK-norms, the FFN width or the experts. Both loaders
//! build their weights from it, so optional tensors are handled the same on
//! the CPU and on a GPU.

use candle_core::quantized::gguf_file;

use super::err;
use super::params::{Hparams, LayerSpec};
use crate::error::CucaError;

/// GGUF tensor names: the one table of the naming scheme. Per-layer names are
/// `blk.{layer}.{part}.weight` (or `.bias`).
pub(crate) mod name {
    pub(crate) const TOKEN_EMBD: &str = "token_embd.weight";
    pub(crate) const OUTPUT: &str = "output.weight";
    pub(crate) const OUTPUT_NORM: &str = "output_norm.weight";
    pub(crate) const ROPE_FREQS: &str = "rope_freqs.weight";
    pub(crate) const ROPE_FACTORS_LONG: &str = "rope_factors_long.weight";
    pub(crate) const ROPE_FACTORS_SHORT: &str = "rope_factors_short.weight";
    pub(crate) const PER_LAYER_TOKEN_EMBD: &str = "per_layer_token_embd.weight";
    pub(crate) const PER_LAYER_MODEL_PROJ: &str = "per_layer_model_proj.weight";
    pub(crate) const PER_LAYER_PROJ_NORM: &str = "per_layer_proj_norm.weight";

    pub(crate) const ATTN_NORM: &str = "attn_norm";
    pub(crate) const ATTN_Q: &str = "attn_q";
    pub(crate) const ATTN_Q_NORM: &str = "attn_q_norm";
    pub(crate) const ATTN_K: &str = "attn_k";
    pub(crate) const ATTN_K_NORM: &str = "attn_k_norm";
    pub(crate) const ATTN_V: &str = "attn_v";
    pub(crate) const ATTN_QKV: &str = "attn_qkv";
    pub(crate) const ATTN_OUTPUT: &str = "attn_output";
    pub(crate) const POST_ATTENTION_NORM: &str = "post_attention_norm";
    pub(crate) const FFN_NORM: &str = "ffn_norm";
    pub(crate) const FFN_GATE: &str = "ffn_gate";
    pub(crate) const FFN_UP: &str = "ffn_up";
    pub(crate) const FFN_DOWN: &str = "ffn_down";
    pub(crate) const FFN_GATE_INP: &str = "ffn_gate_inp";
    pub(crate) const FFN_GATE_EXPS: &str = "ffn_gate_exps";
    pub(crate) const FFN_UP_EXPS: &str = "ffn_up_exps";
    pub(crate) const FFN_DOWN_EXPS: &str = "ffn_down_exps";
    pub(crate) const POST_FFW_NORM: &str = "post_ffw_norm";
    pub(crate) const INP_GATE: &str = "inp_gate";
    pub(crate) const PROJ: &str = "proj";
    pub(crate) const POST_NORM: &str = "post_norm";
    pub(crate) const LAYER_OUTPUT_SCALE: &str = "layer_output_scale";

    // Gated delta-net layers (Qwen 3.5). `ssm_a` carries no `.weight` suffix
    // and the time-step bias is `ssm_dt.bias`.
    pub(crate) const ATTN_GATE: &str = "attn_gate";
    pub(crate) const SSM_A: &str = "ssm_a";
    pub(crate) const SSM_ALPHA: &str = "ssm_alpha";
    pub(crate) const SSM_BETA: &str = "ssm_beta";
    pub(crate) const SSM_CONV1D: &str = "ssm_conv1d";
    pub(crate) const SSM_DT: &str = "ssm_dt";
    pub(crate) const SSM_NORM: &str = "ssm_norm";
    pub(crate) const SSM_OUT: &str = "ssm_out";

    // The multi-token-prediction block that trails the trunk
    // (`blk.{n_layer}.nextn.*`, next to that block's own attention and FFN
    // tensors). The trunk loaders never read them; the in-trunk head does.
    pub(crate) const NEXTN_EH_PROJ: &str = "nextn.eh_proj";
    pub(crate) const NEXTN_ENORM: &str = "nextn.enorm";
    pub(crate) const NEXTN_HNORM: &str = "nextn.hnorm";
    pub(crate) const NEXTN_SHARED_HEAD_NORM: &str = "nextn.shared_head_norm";

    /// Full tensor name of the weight `part` in block `layer`.
    pub(crate) fn blk(layer: usize, part: &str) -> String {
        format!("blk.{layer}.{part}.weight")
    }

    /// Full tensor name of the bias of `part` in block `layer`.
    pub(crate) fn blk_bias(layer: usize, part: &str) -> String {
        format!("blk.{layer}.{part}.bias")
    }

    /// Full tensor name of `part` in block `layer` when it has no suffix
    /// (`ssm_a`).
    pub(crate) fn blk_bare(layer: usize, part: &str) -> String {
        format!("blk.{layer}.{part}")
    }
}

/// The part name of the tensor that norms the input of a layer's FFN:
/// `ffn_norm`, or `post_attention_norm` for architectures that name it so
/// ([`Behavior::ffn_norm_is_post_attention`](super::arch::Behavior)).
pub(crate) fn ffn_norm_part(post_attention_is_ffn_norm: bool) -> &'static str {
    if post_attention_is_ffn_norm {
        name::POST_ATTENTION_NORM
    } else {
        name::FFN_NORM
    }
}

/// The index entry of `tensor_name`, without reading its data.
pub(crate) fn tensor_info<'a>(
    content: &'a gguf_file::Content,
    tensor_name: &str,
) -> Result<&'a gguf_file::TensorInfo, CucaError> {
    content.tensor_infos.get(tensor_name).ok_or_else(|| {
        err(format!(
            "tensor `{tensor_name}`: missing from the GGUF file"
        ))
    })
}

/// How a layer's query, key and value projections are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qkv {
    /// One `attn_qkv` tensor: query, key, value rows concatenated.
    Fused,
    /// Separate `attn_q`, and `attn_k` / `attn_v` on K/V-owning layers.
    Split,
}

/// The FFN of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ffn {
    /// Gated dense FFN of width `ff`; `fused` when `ffn_up` holds the gate
    /// rows followed by the up rows.
    Dense { ff: usize, fused: bool },
    /// Mixture of experts, each of width `ff`.
    Moe { ff: usize },
}

/// Optional tensors of the attention block that this layer has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AttnBiases {
    pub(crate) q: bool,
    pub(crate) k: bool,
    pub(crate) v: bool,
    pub(crate) o: bool,
}

/// What one layer's tensors look like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerLayout {
    pub(crate) qkv: Qkv,
    pub(crate) biases: AttnBiases,
    pub(crate) q_norm: bool,
    /// K-norm; only layers that own K/V have one.
    pub(crate) k_norm: bool,
    pub(crate) post_attn_norm: bool,
    pub(crate) post_ffw_norm: bool,
    pub(crate) ffn: Ffn,
}

fn dims<'a>(content: &'a gguf_file::Content, tensor: &str) -> Result<&'a [usize], CucaError> {
    tensor_info(content, tensor).map(|i| i.shape.dims())
}

impl LayerLayout {
    /// Read the layout of layer `il`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when a tensor the architecture requires is
    /// missing or has an unexpected rank.
    pub(crate) fn read(
        hp: &Hparams,
        il: usize,
        spec: &LayerSpec,
        content: &gguf_file::Content,
    ) -> Result<Self, CucaError> {
        let has = |tensor: &str| content.tensor_infos.contains_key(tensor);
        let has_part = |part: &str| has(&name::blk(il, part));
        let owns = spec.owns_kv(il);

        // A delta-net layer's `attn_qkv` is its own convolution input, not a
        // fused attention projection.
        let qkv = if spec.delta().is_none() && has_part(name::ATTN_QKV) {
            Qkv::Fused
        } else {
            Qkv::Split
        };
        let biases = AttnBiases {
            q: has(&name::blk_bias(il, name::ATTN_Q)) || has(&name::blk_bias(il, name::ATTN_QKV)),
            k: owns && has(&name::blk_bias(il, name::ATTN_K)),
            v: owns && has(&name::blk_bias(il, name::ATTN_V)),
            o: has(&name::blk_bias(il, name::ATTN_OUTPUT)),
        };

        let ffn = if hp.moe.is_some() {
            let gate = name::blk(il, name::FFN_GATE_EXPS);
            let d = dims(content, &gate)?;
            let [_, ff, _] = d else {
                return Err(err(format!(
                    "tensor `{gate}`: expected a rank-3 expert tensor, found {d:?}"
                )));
            };
            Ffn::Moe { ff: *ff }
        } else {
            let down = name::blk(il, name::FFN_DOWN);
            let d = dims(content, &down)?;
            let [_, ff] = d else {
                return Err(err(format!(
                    "tensor `{down}`: expected a matrix, found {d:?}"
                )));
            };
            Ffn::Dense {
                ff: *ff,
                fused: !has_part(name::FFN_GATE),
            }
        };

        let post_attn_norm = hp.behavior.post_norms;
        let post_ffw_norm = hp.behavior.post_norms;
        if post_attn_norm {
            for part in [name::POST_ATTENTION_NORM, name::POST_FFW_NORM] {
                tensor_info(content, &name::blk(il, part))?;
            }
        }
        Ok(Self {
            qkv,
            biases,
            q_norm: has_part(name::ATTN_Q_NORM),
            k_norm: owns && has_part(name::ATTN_K_NORM),
            post_attn_norm,
            post_ffw_norm,
            ffn,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_gguf_scheme() {
        assert_eq!(name::blk(3, name::ATTN_Q), "blk.3.attn_q.weight");
        assert_eq!(name::blk_bias(0, name::ATTN_K), "blk.0.attn_k.bias");
        assert_eq!(name::blk_bare(2, name::SSM_A), "blk.2.ssm_a");
        assert_eq!(name::blk_bias(2, name::SSM_DT), "blk.2.ssm_dt.bias");
        assert_eq!(
            name::blk(24, name::NEXTN_EH_PROJ),
            "blk.24.nextn.eh_proj.weight"
        );
    }

    #[test]
    fn ffn_norm_part_follows_the_architecture_flag() {
        assert_eq!(ffn_norm_part(false), "ffn_norm");
        assert_eq!(ffn_norm_part(true), "post_attention_norm");
    }
}
