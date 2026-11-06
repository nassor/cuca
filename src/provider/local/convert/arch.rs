//! The checkpoint architectures the converter knows, keyed by the
//! `architectures[0]` of `config.json`.
//!
//! One entry per architecture of `model/arch.rs`, each ported from the
//! llama.cpp converter's model class (`conversion/*.py`, commit
//! `67107ac34f62882d8b53ea41ad68010ea635b9be`) that handles the same HF
//! class.

use crate::error::CucaError;

/// A checkpoint family the converter handles; one per engine architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ConvArch {
    /// `LlamaForCausalLM` (`LlamaModel`, Q/K rows permuted).
    Llama,
    /// `SmolLM3ForCausalLM`.
    Smollm3,
    /// `GraniteForCausalLM`.
    Granite,
    /// `Qwen2ForCausalLM`.
    Qwen2,
    /// `Qwen3ForCausalLM`.
    Qwen3,
    /// `Qwen3MoeForCausalLM`.
    Qwen3Moe,
    /// `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`.
    Qwen35,
    /// `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`.
    Qwen35Moe,
    /// `Phi3ForCausalLM` (`Phi3MiniModel`).
    Phi3,
    /// `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`.
    Gemma3,
    /// `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`.
    Gemma4,
    /// `Gemma4AssistantForCausalLM`, the multi-token-prediction drafter.
    Gemma4Assistant,
    /// `Olmo2ForCausalLM`, `Olmo3ForCausalLM`.
    Olmo2,
    /// `OlmoeForCausalLM`.
    Olmoe,
    /// `Ernie4_5ForCausalLM`.
    Ernie4_5,
}

impl ConvArch {
    /// The converter for an HF `architectures[0]` string.
    pub(crate) fn from_architecture(name: &str) -> Option<Self> {
        Some(match name {
            "LlamaForCausalLM" => Self::Llama,
            "SmolLM3ForCausalLM" => Self::Smollm3,
            "GraniteForCausalLM" => Self::Granite,
            "Qwen2ForCausalLM" => Self::Qwen2,
            "Qwen3ForCausalLM" => Self::Qwen3,
            "Qwen3MoeForCausalLM" => Self::Qwen3Moe,
            "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration" => Self::Qwen35,
            "Qwen3_5MoeForCausalLM" | "Qwen3_5MoeForConditionalGeneration" => Self::Qwen35Moe,
            "Phi3ForCausalLM" => Self::Phi3,
            "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => Self::Gemma3,
            "Gemma4ForCausalLM" | "Gemma4ForConditionalGeneration" => Self::Gemma4,
            "Gemma4AssistantForCausalLM" => Self::Gemma4Assistant,
            "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => Self::Olmo2,
            "OlmoeForCausalLM" => Self::Olmoe,
            "Ernie4_5ForCausalLM" | "Ernie4_5_ForCausalLM" => Self::Ernie4_5,
            _ => return None,
        })
    }

    /// The converter for `config.json`'s first architecture, or the error
    /// [`CucaError::Config`] naming the unsupported one.
    pub(crate) fn from_config(config: &serde_json::Value) -> Result<Self, CucaError> {
        let name = config
            .get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|a| a.as_str())
            .ok_or_else(|| CucaError::Config("config.json has no `architectures` entry".into()))?;
        Self::from_architecture(name).ok_or_else(|| {
            CucaError::Config(format!(
                "architecture `{name}` has no checkpoint converter; convert it to GGUF with llama.cpp"
            ))
        })
    }

    /// The GGUF `general.architecture` (and metadata key prefix).
    pub(crate) fn gguf_name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Smollm3 => "smollm3",
            Self::Granite => "granite",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Phi3 => "phi3",
            Self::Gemma3 => "gemma3",
            Self::Gemma4 => "gemma4",
            Self::Gemma4Assistant => "gemma4-assistant",
            Self::Olmo2 => "olmo2",
            Self::Olmoe => "olmoe",
            Self::Ernie4_5 => "ernie4_5",
        }
    }

    /// Llama-family models whose Q/K rows the converter permutes into
    /// adjacent pairs (`LlamaModel.undo_permute`).
    pub(crate) fn permutes_qk(self) -> bool {
        matches!(self, Self::Llama | Self::Smollm3 | Self::Granite)
    }

    /// The Gemma family, whose HF checkpoints keep `norm.weight` as an offset
    /// from 1 for Gemma 3 (see [`ConvArch::norm_shift`]).
    pub(crate) fn is_gemma(self) -> bool {
        matches!(self, Self::Gemma3 | Self::Gemma4 | Self::Gemma4Assistant)
    }

    /// Added to every tensor whose name ends in `norm.weight`: Gemma 3
    /// store `w` and compute `x * (1 + w)`, so the GGUF holds `1 + w`.
    /// Gemma 4 and every other family store the multiplier itself.
    /// (Qwen 3.5's shift applies to all norms but one and is its own rule.)
    pub(crate) fn norm_shift(self) -> f32 {
        match self {
            Self::Gemma3 => 1.0,
            _ => 0.0,
        }
    }

    /// Whether the checkpoint's FFN is a mixture of experts.
    pub(crate) fn is_moe(self) -> bool {
        matches!(self, Self::Qwen3Moe | Self::Qwen35Moe | Self::Olmoe)
    }

    /// The hybrid gated delta-net family.
    pub(crate) fn is_qwen35(self) -> bool {
        matches!(self, Self::Qwen35 | Self::Qwen35Moe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::local::model::arch::Arch;

    #[test]
    fn every_converter_names_an_engine_architecture() {
        let all = [
            ConvArch::Llama,
            ConvArch::Smollm3,
            ConvArch::Granite,
            ConvArch::Qwen2,
            ConvArch::Qwen3,
            ConvArch::Qwen3Moe,
            ConvArch::Qwen35,
            ConvArch::Qwen35Moe,
            ConvArch::Phi3,
            ConvArch::Gemma3,
            ConvArch::Gemma4,
            ConvArch::Gemma4Assistant,
            ConvArch::Olmo2,
            ConvArch::Olmoe,
            ConvArch::Ernie4_5,
        ];
        for conv in all {
            assert!(Arch::from_name(conv.gguf_name()).is_ok(), "{conv:?}");
        }
        for arch in Arch::ALL {
            assert!(all.iter().any(|c| c.gguf_name() == arch.name()), "{arch:?}");
        }
    }

    #[test]
    fn unknown_architectures_are_named() {
        for class in [
            "MixtralForCausalLM",
            "Gemma2ForCausalLM",
            "GraniteMoeForCausalLM",
        ] {
            let cfg = serde_json::json!({"architectures": [class]});
            let err = ConvArch::from_config(&cfg).unwrap_err().to_string();
            assert!(
                err.contains(&format!("`{class}`")) && err.contains("llama.cpp"),
                "{err}"
            );
        }
        assert!(ConvArch::from_config(&serde_json::json!({})).is_err());
    }
}
