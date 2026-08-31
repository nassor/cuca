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
