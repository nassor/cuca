//! Provider adapter layer.
//!
//! Each concrete provider adapter lives in its own submodule, declared here
//! behind its `#[cfg(feature = "...")]` gate. The adapter modules own the
//! `dispatch_*` methods on [`crate::client::CucaClient`] (`impl CucaClient`
//! blocks), which `generate_stream` calls under the matching feature.

/// Shared Anthropic Messages-API protocol module: auth,
/// request-body translation, and SSE parsing. Serves the `provider-anthropic`
/// dispatch and is reused by the DeepSeek bridge.
#[cfg(any(feature = "provider-anthropic", feature = "provider-deepseek",))]
pub(crate) mod anthropic;

/// Shared OpenAI-compatible `/chat/completions` adapter:
/// one request-body builder + SSE translator serving OpenAI, DeepSeek-native,
/// vLLM, LM Studio, and llama.cpp.
#[cfg(any(
    feature = "provider-openai",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
    feature = "provider-deepseek",
    feature = "provider-llamacpp",
))]
pub(crate) mod openai_compat;

/// OpenAI dispatch: default base URL and `dispatch_openai`.
#[cfg(feature = "provider-openai")]
pub(crate) mod openai;

/// DeepSeek dispatch: native OpenAI-compatible route over the shared
/// `openai_compat` adapter plus the Anthropic bridge (model-id translation and
/// the shared `anthropic` protocol module) selected by base URL.
#[cfg(feature = "provider-deepseek")]
pub(crate) mod deepseek;

/// vLLM dispatch: OpenAI-compatible local server over the shared
/// `openai_compat` adapter, defaulting to `http://127.0.0.1:8000/v1`.
#[cfg(feature = "provider-vllm")]
pub(crate) mod vllm;

/// LM Studio dispatch: OpenAI-compatible local server over the
/// shared `openai_compat` adapter, defaulting to `http://127.0.0.1:1234/v1`.
#[cfg(feature = "provider-lmstudio")]
pub(crate) mod lmstudio;

/// Gemini dispatch: Google `streamGenerateContent` over the
/// `x-goog-api-key` header, translating the unified model into Gemini's
/// `contents`/`parts` hierarchy and parsing the SSE response stream.
#[cfg(feature = "provider-gemini")]
pub(crate) mod gemini;

/// llama.cpp dispatch: OpenAI-compatible chat route over the shared
/// `openai_compat` adapter plus the native `/completion` route with raw-token
/// frames.
#[cfg(feature = "provider-llamacpp")]
pub(crate) mod llamacpp;

/// In-process GGUF language-model engine: model download, GGUF load, tokenizer, chat template, and generation on a dedicated thread.
#[cfg(feature = "provider-local")]
pub(crate) mod local;

/// Wire key names for the per-request sampling fields of one adapter.
#[cfg(any(
    feature = "provider-openai",
    feature = "provider-anthropic",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
    feature = "provider-deepseek",
    feature = "provider-gemini",
    feature = "provider-llamacpp",
))]
pub(crate) struct SamplingKeys {
    pub top_p: &'static str,
    pub top_k: &'static str,
    /// `None` when the API has no seed (the caller rejects a set seed first).
    pub seed: Option<&'static str>,
    pub stop: &'static str,
}

/// Insert `top_p`, `top_k`, `seed` and `stop` into a request body under the
/// adapter's key names, each only when set (`stop` only when non-empty).
#[cfg(any(
    feature = "provider-openai",
    feature = "provider-anthropic",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
    feature = "provider-deepseek",
    feature = "provider-gemini",
    feature = "provider-llamacpp",
))]
pub(crate) fn insert_sampling(
    body: &mut serde_json::Map<String, serde_json::Value>,
    req: &crate::request::UnifiedRequest,
    keys: &SamplingKeys,
) {
    use serde_json::Value;
    if let Some(top_p) = req.top_p {
        body.insert(keys.top_p.to_string(), Value::from(top_p));
    }
    if let Some(top_k) = req.top_k {
        body.insert(keys.top_k.to_string(), Value::from(top_k));
    }
    if let (Some(key), Some(seed)) = (keys.seed, req.seed) {
        body.insert(key.to_string(), Value::from(seed));
    }
    if !req.stop.is_empty() {
        let stop = req.stop.iter().cloned().map(Value::String).collect();
        body.insert(keys.stop.to_string(), Value::Array(stop));
    }
}
