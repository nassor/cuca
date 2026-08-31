//! tiktoken-rs encoder resolution shared by the token-counting plugins.
//!
//! [`load_encoder`] is the single place base-encoder names and model names are
//! resolved to a `CoreBPE`. It lives in core rather than in one plugin because
//! `plugin-memory` and `plugin-cost` both need it and a foundational plugin
//! must not reach into a peer (see AGENTS.md, *Plugin layering*).

use crate::error::PluginError;

/// Load the tiktoken encoder for `encoder_name`.
///
/// tiktoken-rs 0.12 exposes the base encoders through typed helpers
/// (`cl100k_base()`, `o200k_base()`, …) and model names through
/// `bpe_for_model` (e.g. `"gpt-4o"`); `bpe_for_model` alone does
/// NOT accept base encoder names like `"cl100k_base"`. Resolve base names
/// first, then fall back to the model-name lookup (`bpe_for_model` returns
/// a `&'static CoreBPE` singleton, so it is cloned into the owned value).
///
/// # Errors
///
/// Returns [`PluginError::Internal`] when `encoder_name` is neither a known
/// base encoder nor a model name tiktoken-rs maps to a tokenizer.
pub(crate) fn load_encoder(encoder_name: &str) -> Result<tiktoken_rs::CoreBPE, PluginError> {
    let encoder = match encoder_name {
        "r50k_base" => tiktoken_rs::r50k_base(),
        "p50k_base" => tiktoken_rs::p50k_base(),
        "p50k_edit" => tiktoken_rs::p50k_edit(),
        "cl100k_base" => tiktoken_rs::cl100k_base(),
        "o200k_base" => tiktoken_rs::o200k_base(),
        other => tiktoken_rs::bpe_for_model(other).cloned(),
    };
    encoder.map_err(|e| {
        PluginError::Internal(format!(
            "failed to load tiktoken encoder '{encoder_name}': {e}"
        ))
    })
}
