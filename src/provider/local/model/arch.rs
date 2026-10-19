//! The architecture table: the one place that maps `general.architecture` to
//! model behaviour, following llama.cpp's `llm_build_*` graphs.
//!
//! Everything an architecture decides without looking at the weights lives in
//! [`Arch::behavior`]. Hyperparameters (sizes, head counts, bases, scales) are
//! read from the `{arch}.*` metadata by [`super::params::Hparams`], and which
//! optional tensors a layer has (biases, QK-norm, fused projections) is read
//! from the tensor index by [`super::layout`]; neither duplicates this table.

use super::rope::RopeKind;
use crate::error::CucaError;

/// An architecture this engine can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Arch {
    /// Gemma 4 dense: sliding/full layers with per-type head sizes and RoPE,
    /// shared K/V layers, per-layer embeddings, weightless value norm.
    Gemma4,
    /// Gemma 2: sliding/full layers alternating (every other full), attention
    /// and final logit soft-caps, post-norms, no QK-norm.
    Gemma2,
    /// Gemma 3: sliding/full layers (every sixth full), QK-norm, post-norms.
    Gemma3,
    /// Llama 1/2/3 and every model that publishes as `llama` (Mistral, ...).
    Llama,
    /// Qwen 2 / 2.5 (and DeepSeek-R1 distills of it): QKV biases.
    Qwen2,
    /// Qwen 3: QK-norm.
    Qwen3,
    /// Phi 3 / 3.5 / 4-mini: fused QKV and gate-up, LongRoPE, partial rotary.
    Phi3,
    /// SmolLM 3: Llama-style with NoPE layers.
    Smollm3,
    /// Granite 3 dense: Llama-style layers with Granite's embedding, attention,
    /// residual and logit scales.
    Granite,
    /// Granite 3 MoE: Llama-style attention with Granite scales and a top-k
    /// mixture of experts.
    GraniteMoe,
    /// Qwen 3.5 (`qwen35`): a hybrid stack of gated delta-net layers
    /// (linear attention with a recurrent state) and, every
    /// `full_attention_interval`-th layer, gated full attention with partial
    /// NEOX RoPE; an optional trailing multi-token-prediction block.
    ///
    /// The checkpoint's `rope.dimension_sections` (multimodal RoPE) are
    /// ignored: for text, the temporal, height and width positions of every
    /// token are equal, which reduces the interleaved multimodal rotation to
    /// plain NEOX over the first `rope.dimension_count` dimensions.
    Qwen35,
    /// OLMo 2: no pre-norms, post-attention and post-FFN norms, full-width
    /// Q/K RMSNorm.
    Olmo2,
    /// ERNIE 4.5 dense: Llama-style layers with a large head dimension.
    Ernie4_5,
    /// The Gemma 4 multi-token-prediction drafter (`gemma4-assistant`): a
    /// four-layer query-only decoder that reads the K/V cache of a Gemma 4
    /// target model. It is loaded through `LocalModelConfig::drafter`, never
    /// as a main model.
    Gemma4Assistant,
}

/// FFN gate activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Act {
    Silu,
    /// GELU, tanh approximation (`ggml_gelu`).
    Gelu,
}

/// How layers are split between sliding-window and full attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Swa {
    /// Every layer attends to the whole context.
    None,
    /// `{arch}.attention.sliding_window_pattern` is a per-layer boolean array.
    PatternArray,
    /// Layer `il` is sliding unless `il % n == n - 1` (llama.cpp
    /// `set_swa_pattern(n)`); `{arch}.attention.sliding_window_pattern` may
    /// override `n`.
    Every(usize),
}

/// Where the attention score scale comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttnScale {
    /// `1 / sqrt(head_dim)`.
    HeadDim,
    /// A constant `1.0` (Gemma 4: the QK norms do the scaling).
    One,
    /// Gemma 3: `1 / sqrt(head_dim)`, except the 62-layer 27B model, which
    /// uses `1 / sqrt(n_embd / n_head)`.
    Gemma3,
    /// Gemma 2: `1 / sqrt(head_dim)`, except the 46-layer 27B model, which
    /// uses `1 / sqrt(n_embd / n_head)`.
    Gemma2,
    /// `{arch}.attention.scale` (Granite).
    Key,
}

/// How token embeddings are scaled after lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbdScale {
    None,
    /// `sqrt(n_embd)` (Gemma).
    SqrtEmbd,
    /// `{arch}.embedding_scale` (Granite).
    Key,
}

/// Everything an architecture fixes without looking at the weights.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Behavior {
    /// Pairing of head values under RoPE.
    pub(crate) rope: RopeKind,
    pub(crate) act: Act,
    /// `post_attention_norm` and `post_ffw_norm` wrap the two sub-blocks
    /// (Gemma 2/3/4); the tensors must then exist.
    pub(crate) post_norms: bool,
    pub(crate) embd_scale: EmbdScale,
    pub(crate) swa: Swa,
    pub(crate) attn_scale: AttnScale,
    /// Every n-th layer (`(il + 1) % n == 0`) applies no rotary embedding.
    pub(crate) no_rope_interval: Option<usize>,
    /// Granite: `residual_scale` multiplies each sub-block output and
    /// `logit_scale` divides the logits.
    pub(crate) granite_scales: bool,
    /// Gemma 4: per-type head sizes and bases (`key_length_swa`,
    /// `freq_base_swa`, `dimension_count_swa`), `shared_kv_layers`,
    /// per-layer embeddings, weightless value norm, per-layer output scale
    /// and the final logit soft-cap.
    pub(crate) gemma4_extras: bool,
    /// The FFN is a top-k mixture of experts (`expert_count` > 0 required).
    pub(crate) moe: bool,
    /// Hybrid layer stack (Qwen 3.5): layer `il` is a gated delta-net layer
    /// unless `(il + 1) % {arch}.full_attention_interval == 0`, in which
    /// case it is full attention. The `{arch}.ssm.*` and
    /// `{arch}.full_attention_interval` keys are required.
    pub(crate) hybrid_delta_net: bool,
    /// `attn_q` holds `[query | gate]` per head (twice the query width) and
    /// the attention output is multiplied by `sigmoid(gate)` before
    /// `attn_output`.
    pub(crate) gated_attention: bool,
    /// The norm in front of the FFN is the tensor `post_attention_norm`
    /// (llama.cpp's `attn_post_norm`), not `ffn_norm`; no post-norm wraps the
    /// attention output.
    pub(crate) ffn_norm_is_post_attention: bool,
    /// The trailing `{arch}.nextn_predict_layers` blocks of `block_count`
    /// hold a multi-token-prediction head, not trunk layers: the trunk has
    /// `block_count - nextn_predict_layers` layers.
    pub(crate) nextn_blocks: bool,
    /// `attn_norm` and `ffn_norm` precede the two sub-blocks. OLMo 2 has none:
    /// its sub-blocks read the residual stream directly and only the post
    /// norms exist.
    pub(crate) pre_norms: bool,
    /// The Q and K RMSNorm weights span the whole projection (`n_head *
    /// head_dim`, `n_kv_head * head_dim`) and normalise before the heads are
    /// split, instead of one `head_dim` weight per head (OLMo 2).
    pub(crate) qk_norm_full: bool,
}

const DEFAULT: Behavior = Behavior {
    rope: RopeKind::Norm,
    act: Act::Silu,
    post_norms: false,
    embd_scale: EmbdScale::None,
    swa: Swa::None,
    attn_scale: AttnScale::HeadDim,
    no_rope_interval: None,
    granite_scales: false,
    gemma4_extras: false,
    moe: false,
    hybrid_delta_net: false,
    gated_attention: false,
    ffn_norm_is_post_attention: false,
    nextn_blocks: false,
    pre_norms: true,
    qk_norm_full: false,
};

impl Arch {
    /// Every supported architecture, for error messages.
    pub(crate) const ALL: [Arch; 14] = [
        Arch::Gemma4,
        Arch::Gemma3,
        Arch::Gemma2,
        Arch::Llama,
        Arch::Qwen2,
        Arch::Qwen3,
        Arch::Phi3,
        Arch::Smollm3,
        Arch::Granite,
        Arch::GraniteMoe,
        Arch::Qwen35,
        Arch::Olmo2,
        Arch::Ernie4_5,
        Arch::Gemma4Assistant,
    ];

    /// The `general.architecture` string, which is also the metadata key
    /// prefix.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Arch::Gemma4 => "gemma4",
            Arch::Gemma3 => "gemma3",
            Arch::Gemma2 => "gemma2",
            Arch::Llama => "llama",
            Arch::Qwen2 => "qwen2",
            Arch::Qwen3 => "qwen3",
            Arch::Phi3 => "phi3",
            Arch::Smollm3 => "smollm3",
            Arch::Granite => "granite",
            Arch::GraniteMoe => "granitemoe",
            Arch::Qwen35 => "qwen35",
            Arch::Olmo2 => "olmo2",
            Arch::Ernie4_5 => "ernie4_5",
            Arch::Gemma4Assistant => "gemma4-assistant",
        }
    }

    /// Resolve `general.architecture`; an unknown name is an error naming it
    /// and the supported set.
    pub(crate) fn from_name(name: &str) -> Result<Self, CucaError> {
        Self::ALL
            .into_iter()
            .find(|a| a.name() == name)
            .ok_or_else(|| {
                super::err(format!(
                    "unsupported GGUF architecture `{name}`; supported: {}",
                    Self::ALL.map(Arch::name).join(", ")
                ))
            })
    }

    /// Whether this architecture is a drafter rather than a standalone model.
    pub(crate) fn is_drafter(self) -> bool {
        self == Arch::Gemma4Assistant
    }

    /// The behaviours of this architecture.
    pub(crate) fn behavior(self) -> Behavior {
        match self {
            Arch::Gemma4 | Arch::Gemma4Assistant => Behavior {
                rope: RopeKind::Neox,
                act: Act::Gelu,
                post_norms: true,
                embd_scale: EmbdScale::SqrtEmbd,
                swa: Swa::PatternArray,
                attn_scale: AttnScale::One,
                gemma4_extras: true,
                ..DEFAULT
            },
            Arch::Gemma3 => Behavior {
                rope: RopeKind::Neox,
                act: Act::Gelu,
                post_norms: true,
                embd_scale: EmbdScale::SqrtEmbd,
                swa: Swa::Every(6),
                attn_scale: AttnScale::Gemma3,
                ..DEFAULT
            },
            Arch::Gemma2 => Behavior {
                rope: RopeKind::Neox,
                act: Act::Gelu,
                post_norms: true,
                embd_scale: EmbdScale::SqrtEmbd,
                swa: Swa::Every(2),
                attn_scale: AttnScale::Gemma2,
                ..DEFAULT
            },
            Arch::Olmo2 => Behavior {
                rope: RopeKind::Neox,
                post_norms: true,
                pre_norms: false,
                qk_norm_full: true,
                ..DEFAULT
            },
            Arch::Llama | Arch::Ernie4_5 => DEFAULT,
            Arch::Qwen2 | Arch::Qwen3 | Arch::Phi3 => Behavior {
                rope: RopeKind::Neox,
                ..DEFAULT
            },
            Arch::Smollm3 => Behavior {
                no_rope_interval: Some(4),
                ..DEFAULT
            },
            Arch::Granite => Behavior {
                embd_scale: EmbdScale::Key,
                attn_scale: AttnScale::Key,
                granite_scales: true,
                ..DEFAULT
            },
            Arch::GraniteMoe => Behavior {
                embd_scale: EmbdScale::Key,
                attn_scale: AttnScale::Key,
                granite_scales: true,
                moe: true,
                ..DEFAULT
            },
            Arch::Qwen35 => Behavior {
                rope: RopeKind::Neox,
                hybrid_delta_net: true,
                gated_attention: true,
                ffn_norm_is_post_attention: true,
                nextn_blocks: true,
                ..DEFAULT
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_unknown_names_are_listed() {
        for arch in Arch::ALL {
            assert_eq!(Arch::from_name(arch.name()).unwrap(), arch);
        }
        let msg = Arch::from_name("mamba").unwrap_err().to_string();
        assert!(msg.contains("`mamba`"), "{msg}");
        assert!(msg.contains("qwen3"), "{msg}");
    }

    #[test]
    fn rope_kinds_follow_llama_cpp() {
        assert_eq!(Arch::Llama.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::Smollm3.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::GraniteMoe.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::Granite.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::Ernie4_5.behavior().rope, RopeKind::Norm);
        for arch in [
            Arch::Gemma4,
            Arch::Gemma3,
            Arch::Gemma2,
            Arch::Olmo2,
            Arch::Qwen2,
            Arch::Qwen3,
            Arch::Qwen35,
            Arch::Phi3,
        ] {
            assert_eq!(arch.behavior().rope, RopeKind::Neox, "{arch:?}");
        }
    }

    #[test]
    fn only_qwen35_is_hybrid_gated_and_embeds_prediction_blocks() {
        for arch in Arch::ALL {
            let b = arch.behavior();
            let qwen35 = arch == Arch::Qwen35;
            assert_eq!(b.hybrid_delta_net, qwen35, "{arch:?}");
            assert_eq!(b.gated_attention, qwen35, "{arch:?}");
            assert_eq!(b.ffn_norm_is_post_attention, qwen35, "{arch:?}");
            assert_eq!(b.nextn_blocks, qwen35, "{arch:?}");
        }
        assert!(!Arch::Qwen35.is_drafter());
    }
}
