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
    /// Gemma 3: sliding/full layers (every sixth full), QK-norm, post-norms.
    Gemma3,
    /// Llama 3.x-style models and every model that publishes as `llama`
    /// (DeepSeek-R1-Distill-Llama, Mistral, ...).
    Llama,
    /// Qwen 2 / 2.5 (and DeepSeek-R1 distills of it): QKV biases.
    Qwen2,
    /// Qwen 3: QK-norm.
    Qwen3,
    /// Phi 4-mini (and Phi 3.x): fused QKV and gate-up, LongRoPE, partial rotary.
    Phi3,
    /// SmolLM 3: Llama-style with NoPE layers.
    Smollm3,
    /// Granite 3 dense: Llama-style layers with Granite's embedding, attention,
    /// residual and logit scales.
    Granite,
    /// Qwen 3 MoE: Qwen 3 attention (QK-norm) with a top-k mixture of
    /// experts in every layer.
    Qwen3Moe,
    /// OLMoE: Llama-style pre-norm layers with full-width Q/K RMSNorm and a
    /// top-k mixture of experts whose selected weights are not renormalised.
    Olmoe,
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
    /// Qwen 3.5 MoE (`qwen35moe`): the [`Arch::Qwen35`] hybrid stack with a
    /// mixture-of-experts FFN in every layer (softmax over all experts,
    /// top-k, renormalised) plus a shared expert scaled by a sigmoid gate
    /// (transformers `Qwen3_5MoeSparseMoeBlock`).
    Qwen35Moe,
    /// Muse Glimmer (`muse-glimmer`): a dense decoder with a sliding-window
    /// pattern whose full-attention layers apply no rotary embedding, a
    /// separate sigmoid attention-output gate projection, Gemma-style
    /// sandwich norms (the post norms with their own epsilon), a scale-less
    /// norm on the embeddings and a logit multiplier ahead of the soft-cap.
    MuseGlimmer,
    /// Ministral 3 and Devstral Small 2 (`mistral3`): the `llama` graph with
    /// YaRN whose magnitude follows `rope.scaling.yarn_log_multiplier` and the
    /// Llama 4 per-position query scaling, its length being the YaRN original
    /// context.
    Mistral3,
    /// Mistral Small 4 (`mistral4`): DeepSeek-style multi-head latent
    /// attention (low-rank Q, a compressed K/V latent shared by every head
    /// plus one shared rotary key, absorbed into the query so that only the
    /// latent is cached), YaRN with a softmax-scale `mscale`, Llama 4 style
    /// per-position query scaling, and a softmax-routed mixture of experts
    /// (fused gate/up expert tensor, renormalised weights) plus an ungated
    /// shared expert in every layer.
    Mistral4,
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
    /// The Mistral Small 4 EAGLE drafter (`mistral4-eagle`): two latent
    /// attention layers with a dense FFN that read the target's final-norm
    /// hidden state and the embedding of the token after it, and predict the
    /// token after that. Unlike the Gemma 4 drafter it keeps a K/V cache of
    /// its own and shares the target's embedding table and LM head. It is
    /// loaded through `LocalModelConfig::drafter`, never as a main model.
    Mistral4Eagle,
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
    /// (Gemma 3/4); the tensors must then exist.
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
    /// Renormalise the selected experts' softmax weights to sum to one
    /// (llama.cpp `norm_w`). Only read when `moe` is set.
    pub(crate) moe_norm_weights: bool,
    /// Every MoE layer also runs a shared expert (`ffn_{gate,up,down}_shexp`)
    /// whose output is scaled by `sigmoid(ffn_gate_inp_shexp . x)` and added
    /// to the routed experts' output (Qwen 3.5 MoE). Only read when `moe` is
    /// set.
    pub(crate) shared_expert: bool,
    /// Every attention layer has an `attn_gate` projection (`n_embd` to
    /// `n_head * head_dim`) of its normed input whose sigmoid multiplies the
    /// attention output before `attn_output` (Muse Glimmer). Unlike
    /// [`Behavior::gated_attention`], the gate is a tensor of its own.
    pub(crate) attn_gate_tensor: bool,
    /// Full-attention layers apply no rotary embedding (llama.cpp's
    /// `rope_pattern` follows the sliding pattern); only meaningful with a
    /// sliding pattern.
    pub(crate) nope_full_layers: bool,
    /// Without `{arch}.rope.freq_base_swa`, sliding layers use
    /// `{arch}.rope.freq_base` like the full layers (the default is 10000).
    pub(crate) swa_base_from_freq_base: bool,
    /// A scale-less RMSNorm (the layer-norm epsilon) over every embedding row
    /// after the embedding scale.
    pub(crate) embd_norm: bool,
    /// Epsilon of the post-attention and post-FFN norms when it differs from
    /// `attention.layer_norm_rms_epsilon`; the file does not carry it.
    pub(crate) post_norm_eps: Option<f64>,
    /// `{arch}.logit_scale` multiplies the final logits (Granite's divides),
    /// before the soft-cap.
    pub(crate) logit_multiplier: bool,
    /// Multi-head latent attention: `attn_q_a`/`attn_q_b` (low-rank Q),
    /// `attn_kv_a_mqa` (latent plus shared rotary key), `attn_k_b` and
    /// `attn_v_b` (the per-head up-projections, absorbed at run time) and
    /// `attn_output`; the K/V cache holds the latent and the rotary key only.
    pub(crate) mla: bool,
    /// The shared expert has no `ffn_gate_inp_shexp` gate: its output is added
    /// as it is (llama.cpp's deepseek2 family). Only read with
    /// [`Behavior::shared_expert`].
    pub(crate) shared_expert_ungated: bool,
    /// Llama 4 per-position query scaling `1 + beta * ln(1 + floor(pos /
    /// length))` (`beta`, `length`), used when the file carries no
    /// `attention.temperature_scale` / `attention.temperature_length`. The
    /// values are the checkpoint's own configuration; the GGUF does not hold
    /// them for Mistral Small 4.
    pub(crate) attn_temp_default: Option<(f64, usize)>,
    /// Llama 4 query scaling from `attention.temperature_scale` (absent or 0:
    /// none) whose length is `rope.scaling.original_context_length` (llama.cpp
    /// `mistral3`).
    pub(crate) attn_temp_orig_ctx: bool,
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
    moe_norm_weights: true,
    shared_expert: false,
    attn_gate_tensor: false,
    nope_full_layers: false,
    swa_base_from_freq_base: false,
    embd_norm: false,
    post_norm_eps: None,
    logit_multiplier: false,
    mla: false,
    shared_expert_ungated: false,
    attn_temp_default: None,
    attn_temp_orig_ctx: false,
};

impl Arch {
    /// Every supported architecture, for error messages.
    pub(crate) const ALL: [Arch; 19] = [
        Arch::Gemma4,
        Arch::Gemma3,
        Arch::Llama,
        Arch::Qwen2,
        Arch::Qwen3,
        Arch::Phi3,
        Arch::Smollm3,
        Arch::Granite,
        Arch::Qwen3Moe,
        Arch::Olmoe,
        Arch::Qwen35,
        Arch::Qwen35Moe,
        Arch::Olmo2,
        Arch::Ernie4_5,
        Arch::MuseGlimmer,
        Arch::Mistral3,
        Arch::Mistral4,
        Arch::Gemma4Assistant,
        Arch::Mistral4Eagle,
    ];

    /// The `general.architecture` string, which is also the metadata key
    /// prefix.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Arch::Gemma4 => "gemma4",
            Arch::Gemma3 => "gemma3",
            Arch::Llama => "llama",
            Arch::Qwen2 => "qwen2",
            Arch::Qwen3 => "qwen3",
            Arch::Phi3 => "phi3",
            Arch::Smollm3 => "smollm3",
            Arch::Granite => "granite",
            Arch::Qwen3Moe => "qwen3moe",
            Arch::Olmoe => "olmoe",
            Arch::Qwen35 => "qwen35",
            Arch::Qwen35Moe => "qwen35moe",
            Arch::Olmo2 => "olmo2",
            Arch::Ernie4_5 => "ernie4_5",
            Arch::Gemma4Assistant => "gemma4-assistant",
            Arch::MuseGlimmer => "muse-glimmer",
            Arch::Mistral3 => "mistral3",
            Arch::Mistral4 => "mistral4",
            Arch::Mistral4Eagle => "mistral4-eagle",
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
        matches!(self, Arch::Gemma4Assistant | Arch::Mistral4Eagle)
    }

    /// Whether this drafter has no K/V of its own and attends to the
    /// target's cache (the Gemma 4 assistant), as opposed to keeping its own
    /// (the EAGLE drafter).
    pub(crate) fn reads_target_kv(self) -> bool {
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
            Arch::MuseGlimmer => Behavior {
                post_norms: true,
                swa: Swa::PatternArray,
                attn_gate_tensor: true,
                nope_full_layers: true,
                swa_base_from_freq_base: true,
                embd_norm: true,
                post_norm_eps: Some(1e-8),
                logit_multiplier: true,
                ..DEFAULT
            },
            Arch::Mistral3 => Behavior {
                attn_temp_orig_ctx: true,
                ..DEFAULT
            },
            Arch::Mistral4 => Behavior {
                moe: true,
                shared_expert: true,
                shared_expert_ungated: true,
                mla: true,
                attn_temp_default: Some((0.1, 8192)),
                ..DEFAULT
            },
            // The drafter's layers are the target's attention with a dense
            // FFN: no experts, no shared expert.
            Arch::Mistral4Eagle => Behavior {
                mla: true,
                attn_temp_default: Some((0.1, 8192)),
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
            Arch::Qwen3Moe => Behavior {
                rope: RopeKind::Neox,
                moe: true,
                ..DEFAULT
            },
            Arch::Olmoe => Behavior {
                rope: RopeKind::Neox,
                qk_norm_full: true,
                moe: true,
                moe_norm_weights: false,
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
            Arch::Qwen35Moe => Behavior {
                rope: RopeKind::Neox,
                hybrid_delta_net: true,
                gated_attention: true,
                ffn_norm_is_post_attention: true,
                nextn_blocks: true,
                moe: true,
                shared_expert: true,
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
        for removed in ["gemma2", "granitemoe"] {
            let msg = Arch::from_name(removed).unwrap_err().to_string();
            assert!(msg.contains(&format!("`{removed}`")), "{msg}");
        }
    }

    #[test]
    fn rope_kinds_follow_llama_cpp() {
        assert_eq!(Arch::Llama.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::Smollm3.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::Granite.behavior().rope, RopeKind::Norm);
        assert_eq!(Arch::Ernie4_5.behavior().rope, RopeKind::Norm);
        for arch in [
            Arch::Gemma4,
            Arch::Gemma3,
            Arch::Olmo2,
            Arch::Qwen2,
            Arch::Qwen3,
            Arch::Qwen35,
            Arch::Qwen35Moe,
            Arch::Phi3,
            Arch::Qwen3Moe,
            Arch::Olmoe,
        ] {
            assert_eq!(arch.behavior().rope, RopeKind::Neox, "{arch:?}");
        }
    }

    /// Only the Gemma 4 assistant and the EAGLE head are drafters, and only
    /// the assistant reads the target's K/V.
    #[test]
    fn only_the_assistant_and_the_eagle_head_are_drafters() {
        for arch in Arch::ALL {
            let drafter = matches!(arch, Arch::Gemma4Assistant | Arch::Mistral4Eagle);
            assert_eq!(arch.is_drafter(), drafter, "{arch:?}");
            assert_eq!(arch.reads_target_kv(), arch == Arch::Gemma4Assistant);
        }
    }
}
