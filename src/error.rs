//! Error types for the CUCA core client and its plugins.
//!
//! [`CucaError`] is the client-facing error surfaced to callers of the crate;
//! [`PluginError`] is the error type a plugin hook returns. Both implement
//! [`std::fmt::Display`], [`std::error::Error`], and are `Clone` so they can be
//! threaded through plugins and response streams.

use crate::types::ProviderEndpoint;

/// Client-facing error produced by the CUCA core and its providers.
///
/// Every variant is a readable sentence via [`Display`](std::fmt::Display) and
/// carries enough detail for diagnostics without exposing transport internals.
/// `CucaError` is `Clone` and `Send + Sync + 'static`, so it can be moved across
/// await points and into `AgentResponseStream` error channels.
#[derive(Debug, Clone)]
pub enum CucaError {
    /// Transport-level failure (connection, TLS, body read).
    ///
    // `reqwest::Error` is not `Clone`, so this variant carries the wrapped
    // error's `Display` text as a `String` and does not chain `source()`.
    Transport {
        /// The underlying transport error's message.
        message: String,
    },
    /// Non-2xx response; body captured for diagnostics.
    Http {
        /// The HTTP status code of the failed response.
        status: u16,
        /// The response body, captured for diagnostics.
        body: String,
    },
    /// SSE framing/parsing failure.
    SseParse(String),
    /// JSON decode/encode failure.
    ///
    // `serde_json::Error` is not `Clone`, so this variant carries the wrapped
    // error's `Display` text as a `String` and does not chain `source()`.
    Json {
        /// The underlying JSON error's message.
        message: String,
    },
    /// Provider adapter rejected or failed on a request.
    Provider {
        /// The provider endpoint that failed.
        provider: ProviderEndpoint,
        /// Human-readable failure detail.
        message: String,
    },
    /// The request targets a provider whose feature flag is not compiled.
    ProviderNotEnabled(&'static str),
    /// A plugin hook returned an error.
    Plugin(PluginError),
    /// Invalid builder/configuration state.
    Config(String),
    /// IO failure (file, subprocess).
    ///
    // `std::io::Error` is not `Clone`, so this variant carries the wrapped
    // error's `Display` text as a `String` and does not chain `source()`.
    Io {
        /// The underlying I/O error's message.
        message: String,
    },
}

/// Error returned by a plugin hook.
///
/// Plugins return this type from their hooks; the core converts it into
/// [`CucaError::Plugin`] via [`From`]. It is `Clone` and `Send + Sync +
/// 'static` so plugin failures can be captured and re-injected into response
/// streams.
#[derive(Debug, Clone)]
pub enum PluginError {
    /// A hook failed; `plugin` is the [`CucaPlugin::name()`](crate) identifier.
    HookFailure {
        /// The plugin's name, as reported by its `name()`.
        plugin: &'static str,
        /// The hook stage that failed (e.g. `"stream"`, `"tool"`).
        stage: &'static str,
        /// Human-readable failure detail.
        message: String,
    },
    /// A guardrail/schema validation failure; `message` is re-injected into the
    /// stream.
    Validation {
        /// The schema (or schema key) the payload failed to satisfy.
        schema: String,
        /// Human-readable validation detail.
        message: String,
    },
    /// The plugin was asked for something it does not support (e.g. transport).
    NotSupported(String),
    /// IO failure inside a plugin.
    ///
    // `std::io::Error` is not `Clone`, so this variant carries the wrapped
    // error's `Display` text as a `String` and does not chain `source()`.
    Io(String),
    /// An internal plugin failure with no more specific classification.
    Internal(String),
}

impl CucaError {
    /// Build a [`CucaError::Provider`] from an endpoint and a message.
    pub fn provider(endpoint: ProviderEndpoint, msg: impl Into<String>) -> Self {
        CucaError::Provider {
            provider: endpoint,
            message: msg.into(),
        }
    }
}

impl PluginError {
    /// Build a [`PluginError::HookFailure`] from a plugin name, hook stage, and
    /// message.
    pub fn hook(plugin: &'static str, stage: &'static str, msg: impl Into<String>) -> Self {
        PluginError::HookFailure {
            plugin,
            stage,
            message: msg.into(),
        }
    }
}

impl std::fmt::Display for CucaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CucaError::Transport { message } => write!(f, "transport failure: {message}"),
            CucaError::Http { status, body } => write!(f, "HTTP error {status}: {body}"),
            CucaError::SseParse(msg) => write!(f, "SSE parse failure: {msg}"),
            CucaError::Json { message } => write!(f, "JSON error: {message}"),
            CucaError::Provider { provider, message } => {
                write!(f, "provider {provider} failed: {message}")
            }
            CucaError::ProviderNotEnabled(flag) => {
                write!(f, "provider feature not enabled: {flag}")
            }
            CucaError::Plugin(e) => write!(f, "plugin error: {e}"),
            CucaError::Config(msg) => write!(f, "configuration error: {msg}"),
            CucaError::Io { message } => write!(f, "I/O error: {message}"),
        }
    }
}

impl std::error::Error for CucaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // The one variant holding a typed, `std::error::Error`-implementing
            // payload chains its source; the string-based variants (by design)
            // return `None`.
            CucaError::Plugin(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::HookFailure {
                plugin,
                stage,
                message,
            } => write!(f, "plugin {plugin} failed at stage {stage}: {message}"),
            PluginError::Validation { schema, message } => {
                write!(f, "validation failed for schema {schema}: {message}")
            }
            PluginError::NotSupported(msg) => write!(f, "not supported: {msg}"),
            PluginError::Io(msg) => write!(f, "plugin I/O error: {msg}"),
            PluginError::Internal(msg) => write!(f, "internal plugin error: {msg}"),
        }
    }
}

impl std::error::Error for PluginError {}

impl From<PluginError> for CucaError {
    fn from(e: PluginError) -> Self {
        CucaError::Plugin(e)
    }
}

impl From<serde_json::Error> for CucaError {
    fn from(e: serde_json::Error) -> Self {
        CucaError::Json {
            message: e.to_string(),
        }
    }
}

impl From<std::io::Error> for CucaError {
    fn from(e: std::io::Error) -> Self {
        CucaError::Io {
            message: e.to_string(),
        }
    }
}

impl From<std::io::Error> for PluginError {
    fn from(e: std::io::Error) -> Self {
        PluginError::Io(e.to_string())
    }
}

#[cfg(any(
    feature = "provider-openai",
    feature = "provider-anthropic",
    feature = "provider-deepseek",
    feature = "provider-gemini",
    feature = "provider-llamacpp",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
))]
impl From<reqwest::Error> for CucaError {
    fn from(e: reqwest::Error) -> Self {
        CucaError::Transport {
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static assertions that both error types are `Send + Sync + 'static`.
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    #[test]
    fn errors_are_send_sync_static() {
        assert_send_sync::<CucaError>();
        assert_send_sync::<PluginError>();
    }

    /// `From<PluginError> for CucaError` wraps correctly and `Display` chains the
    /// plugin name.
    #[test]
    fn plugin_into_cuca_chains_plugin_name() {
        let pe = PluginError::hook("router", "stream", "timeout");
        let ce: CucaError = pe.into();
        let s = ce.to_string();
        assert!(s.contains("plugin error"), "unexpected display: {s}");
        assert!(s.contains("router"), "missing plugin name: {s}");
        assert!(s.contains("stream"), "missing stage: {s}");
    }

    /// `CucaError::Plugin` chains `source()` to the inner `PluginError`.
    #[test]
    fn plugin_source_chains() {
        let pe = PluginError::Internal("boom".into());
        let ce = CucaError::Plugin(pe);
        let src = std::error::Error::source(&ce).expect("Plugin must expose a source");
        // The source must be the same `PluginError` payload.
        let down = src
            .downcast_ref::<PluginError>()
            .expect("source must be a PluginError");
        assert_eq!(down.to_string(), "internal plugin error: boom");
    }

    /// `Display` for every `CucaError` variant is non-empty and mentions the key
    /// detail.
    #[test]
    fn display_mentions_key_detail() {
        let cases: Vec<(CucaError, &str)> = vec![
            (
                CucaError::Transport {
                    message: "conn reset".into(),
                },
                "transport failure",
            ),
            (
                CucaError::Http {
                    status: 503,
                    body: "unavailable".into(),
                },
                "503",
            ),
            (CucaError::SseParse("bad frame".into()), "bad frame"),
            (
                CucaError::Json {
                    message: "invalid digit".into(),
                },
                "invalid digit",
            ),
            (
                CucaError::provider(ProviderEndpoint::OpenAi, "rate limited"),
                "rate limited",
            ),
            (
                CucaError::ProviderNotEnabled("provider-openai"),
                "provider-openai",
            ),
            (CucaError::Config("bad token".into()), "bad token"),
            (
                CucaError::Io {
                    message: "denied".into(),
                },
                "denied",
            ),
        ];
        for (err, detail) in cases {
            let s = err.to_string();
            assert!(!s.is_empty(), "display must be non-empty");
            assert!(
                s.contains(detail),
                "expected {detail:?} in display, got: {s:?}"
            );
        }
    }

    /// `Display` for `PluginError` variants mentions the key detail.
    #[test]
    fn plugin_display_mentions_key_detail() {
        let cases: Vec<(PluginError, &str)> = vec![
            (
                PluginError::hook("mcp", "tool", "boom"),
                // The schema is validated separately below; here check the
                // stage and plugin are surfaced.
                "tool",
            ),
            (
                PluginError::Validation {
                    schema: "schema-x".into(),
                    message: "nope".into(),
                },
                "schema-x",
            ),
            (PluginError::NotSupported("http2".into()), "http2"),
            (PluginError::Io("disk full".into()), "disk full"),
            (PluginError::Internal("oops".into()), "oops"),
        ];
        for (err, detail) in cases {
            let s = err.to_string();
            assert!(!s.is_empty(), "display must be non-empty");
            assert!(
                s.contains(detail),
                "expected {detail:?} in display, got: {s:?}"
            );
        }
    }

    /// `HookFailure` display mentions the plugin name and stage.
    #[test]
    fn hook_failure_display_mentions_plugin_and_stage() {
        let err = PluginError::hook("agent-sampler", "on_stream", "failed");
        let s = err.to_string();
        assert!(s.contains("agent-sampler"), "missing plugin name: {s}");
        assert!(s.contains("on_stream"), "missing stage: {s}");
    }

    /// `From<serde_json::Error>` captures the message.
    #[test]
    fn from_serde_json() {
        let je = serde_json::from_str::<u32>("x").unwrap_err();
        let ce: CucaError = je.into();
        match ce {
            CucaError::Json { message } => assert!(message.contains("x") || !message.is_empty()),
            other => panic!("expected Json variant, got {other:?}"),
        }
    }

    /// `From<std::io::Error>` captures the message into `CucaError::Io`.
    #[test]
    fn from_io_error_into_cuca() {
        let ie = std::io::Error::other("boom");
        let ce: CucaError = ie.into();
        match ce {
            CucaError::Io { message } => assert_eq!(message, "boom"),
            other => panic!("expected Io variant, got {other:?}"),
        }
    }

    /// `From<std::io::Error>` maps into `PluginError::Io`.
    #[test]
    fn from_io_error_into_plugin() {
        let ie = std::io::Error::other("boom");
        let pe: PluginError = ie.into();
        match pe {
            PluginError::Io(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Io variant, got {other:?}"),
        }
    }

    #[cfg(any(
        feature = "provider-openai",
        feature = "provider-anthropic",
        feature = "provider-deepseek",
        feature = "provider-gemini",
        feature = "provider-llamacpp",
        feature = "provider-vllm",
        feature = "provider-lmstudio",
    ))]
    /// `From<reqwest::Error>` maps into `CucaError::Transport` without network.
    #[test]
    fn from_reqwest_error() {
        let re = reqwest::Client::builder()
            .user_agent("bad\nvalue")
            .build()
            .expect_err("an invalid User-Agent header must fail to build");
        let ce: CucaError = re.into();
        match ce {
            CucaError::Transport { message } => assert!(!message.is_empty()),
            other => panic!("expected Transport variant, got {other:?}"),
        }
    }
}
