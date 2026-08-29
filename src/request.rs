//! Normalized request/response contracts for the unified abstraction.
//!
//! [`UnifiedRequest`] is the single outbound shape every provider consumes; its
//! builder-style methods keep callsites readable while routing and telemetry
//! stay uniform. [`UnifiedResponse`] is the terminal, aggregated result handed
//! to `on_response_complete` hooks and telemetry.
//!
//! # Field-usage contract
//!
//! Downstream plans rely on specific fields here:
//! - `req.model` and `req.provider` are read by `plugin-telemetry`.
//! - `request.provider` is overwritten by `CucaClient::generate_stream`
//!   with `self.selected_provider` before `on_request` hooks run, so
//!   callers must not treat the value passed to `new` as authoritative.

use std::pin::Pin;

use futures_core::Stream;

use crate::error::CucaError;
use crate::types::{MessageContentBlock, ProviderEndpoint, ToolDefinition, UnifiedMessage};

/// Unified thinking/reasoning effort levels, translated per provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEffort {
    /// Minimal reasoning; the provider's cheapest mode.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort (often the provider default).
    Medium,
    /// High reasoning effort.
    High,
    /// Maximum reasoning effort; the provider's deepest, most expensive mode.
    #[serde(rename = "xhigh")]
    XHigh,
}

/// Provider-specific thinking parameters: knobs with no unified equivalent.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "provider", content = "params", rename_all = "snake_case")]
pub enum ThinkingParams {
    /// No provider-specific parameters; the unified effort alone applies.
    None,
    /// Anthropic: legacy budget mode (`type: "enabled"`) or the newer
    /// adaptive mode (`type: "adaptive"`).
    Anthropic {
        /// Explicit token budget for budget mode; `None` defers to the
        /// unified-effort budget map.
        budget_tokens: Option<u32>,
        /// Switch to adaptive mode (`type: "adaptive"`) instead of a fixed
        /// budget.
        adaptive: bool,
    },
    /// Google Gemini: `thinkingBudget` (2.5) and `thinkingLevel` (3+).
    Gemini {
        /// `thinkingBudget` token cap; wins over the unified effort when set.
        thinking_budget: Option<u32>,
        /// Raw `thinkingLevel` string; wins over the unified effort when set.
        thinking_level: Option<String>,
    },
    /// OpenAI-compatible: raw `reasoning_effort` override.
    OpenAi {
        /// Overrides the unified-effort map verbatim when `Some`.
        reasoning_effort: Option<String>,
    },
}

impl Default for ThinkingParams {
    /// No provider-specific parameters by default.
    fn default() -> Self {
        Self::None
    }
}

/// Optional thinking capability on a request.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThinkingConfig {
    /// false = explicitly disabled (silently ignored by always-on reasoning
    /// models).
    pub enabled: bool,
    /// Unified effort level; None = the provider's default effort.
    pub effort: Option<ThinkingEffort>,
    /// Provider-specific parameters.
    pub params: ThinkingParams,
}

impl Default for ThinkingConfig {
    /// Thinking enabled with the provider's default effort and no
    /// provider-specific parameters.
    fn default() -> Self {
        Self {
            enabled: true,
            effort: None,
            params: ThinkingParams::None,
        }
    }
}

impl ThinkingConfig {
    /// Thinking explicitly disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            effort: None,
            params: ThinkingParams::None,
        }
    }

    /// Thinking enabled at a unified effort level.
    pub fn with_effort(effort: ThinkingEffort) -> Self {
        Self {
            enabled: true,
            effort: Some(effort),
            params: ThinkingParams::None,
        }
    }
}

/// A location where a provider may establish a prompt-cache breakpoint.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph properties.
/// Treat the JSON as sensitive data; do not log or publish it. CUCA does not
/// encrypt, redact, or write it. The caller owns access control, encryption,
/// storage, and deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PromptCacheBreakpoint {
    /// Index of the message containing the breakpoint.
    pub message_index: usize,
    /// Index of the content block containing the breakpoint.
    pub block_index: usize,
}

/// Provider-neutral prompt-cache behavior requested for a completion.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph properties.
/// Treat the JSON as sensitive data; do not log or publish it. CUCA does not
/// encrypt, redact, or write it. The caller owns access control, encryption,
/// storage, and deletion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PromptCacheDirective {
    /// Do not request provider prompt caching.
    Disabled,
    /// Request ephemeral caching at the ordered content locations.
    Ephemeral {
        /// Breakpoints at which the provider may cache prompt content.
        breakpoints: Vec<PromptCacheBreakpoint>,
    },
}

impl Default for PromptCacheDirective {
    /// Prompt caching is disabled unless explicitly requested.
    fn default() -> Self {
        Self::Disabled
    }
}

/// Normalized provider prompt-cache token usage.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph properties.
/// Treat the JSON as sensitive data; do not log or publish it. CUCA does not
/// encrypt, redact, or write it. The caller owns access control, encryption,
/// storage, and deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptCacheUsage {
    /// Number of prompt tokens read from the provider cache.
    pub read_tokens: u32,
    /// Number of prompt tokens written to the provider cache.
    pub write_tokens: u32,
}

/// Normalized outbound request: one shape for every provider.
///
/// Built with the chainable builder methods; the model names the upstream model,
/// `provider` the endpoint it is routed to (see the [field-usage contract](crate::request)),
/// and `messages` the ordered conversation. Optional knobs are `None` unless set.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may contain
/// confidential system prompts, user messages, tool arguments and results,
/// base64 image data, model output, signatures, and graph properties. Treat the
/// JSON as sensitive data; do not log or publish it. CUCA does not encrypt,
/// redact, or write it. The caller owns access control, encryption, storage, and
/// deletion.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnifiedRequest {
    /// The upstream model identifier.
    pub model: String,
    /// The provider endpoint the request is routed to; set by
    /// `CucaClient::generate_stream` before plugin hooks run.
    pub provider: ProviderEndpoint,
    /// The ordered conversation messages.
    pub messages: Vec<UnifiedMessage>,
    /// Optional sampling temperature; `None` defers to the provider default.
    pub temperature: Option<f32>,
    /// Optional cap on completion tokens; `None` defers to the provider default.
    pub max_tokens: Option<u32>,
    /// Whether the request streams its response.
    pub stream: bool,
    /// Optional thinking capability; `None` defers to the provider's default
    /// behavior (no thinking keys emitted on the wire).
    pub thinking: Option<ThinkingConfig>,
    /// Provider-neutral tool definitions available to the model.
    pub tools: Vec<ToolDefinition>,
    /// Provider-neutral prompt-cache behavior; disabled by default.
    #[serde(default)]
    pub prompt_cache: PromptCacheDirective,
}

impl UnifiedRequest {
    /// Start a request for `model` with no messages and defaults: `Custom("")`
    /// provider, no temperature/max-token caps, streaming enabled.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: ProviderEndpoint::Custom(String::new()),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            stream: true,
            thinking: None,
            tools: Vec::new(),
            prompt_cache: PromptCacheDirective::Disabled,
        }
    }

    /// Append a provider-neutral tool definition.
    pub fn add_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set the provider-neutral prompt-cache behavior.
    pub fn with_prompt_cache(mut self, prompt_cache: PromptCacheDirective) -> Self {
        self.prompt_cache = prompt_cache;
        self
    }

    /// Append a system message from a single text block.
    pub fn add_system_message(mut self, text: impl Into<String>) -> Self {
        self.messages.push(UnifiedMessage::system(text));
        self
    }

    /// Append a user message from a single text block.
    pub fn add_user_message(mut self, text: impl Into<String>) -> Self {
        self.messages.push(UnifiedMessage::user(text));
        self
    }

    /// Append an arbitrary already-built message.
    pub fn add_message(mut self, msg: UnifiedMessage) -> Self {
        self.messages.push(msg);
        self
    }

    /// Set the sampling temperature.
    pub fn set_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the cap on completion tokens.
    pub fn set_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Toggle streaming on or off.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set the optional thinking capability; `None` keeps the provider default
    /// (no thinking keys emitted on the wire).
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Enable thinking at the given unified effort level, or the provider's
    /// default effort when `None`.
    pub fn enable_thinking(mut self, effort: Option<ThinkingEffort>) -> Self {
        self.thinking = Some(ThinkingConfig {
            enabled: true,
            effort,
            params: ThinkingParams::None,
        });
        self
    }

    /// Explicitly disable thinking on this request.
    pub fn disable_thinking(mut self) -> Self {
        self.thinking = Some(ThinkingConfig::disabled());
        self
    }

    /// Number of messages (informational; used by complexity evaluation).
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

/// Normalized terminal response; assembled by the client stream wrapper.
///
/// Carries the aggregated completion: the model and provider that served it,
/// wall-clock duration and token usage for telemetry, the stop reason, and the
/// ordered content blocks that make up the answer.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes the
/// complete memory graph and local-cache request/response values. It may contain
/// confidential system prompts, user messages, tool arguments and results,
/// base64 image data, model output, signatures, and graph properties. Treat the
/// JSON as sensitive data; do not log or publish it. CUCA does not encrypt,
/// redact, or write it. The caller owns access control, encryption, storage, and
/// deletion.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnifiedResponse {
    /// The upstream model that produced the completion.
    pub model: String,
    /// The provider endpoint that served the request.
    pub provider: ProviderEndpoint,
    /// Wall-clock duration of the request, in seconds.
    pub duration_secs: f64,
    /// Prompt tokens consumed by the request.
    pub prompt_tokens: u32,
    /// Completion tokens generated by the model.
    pub completion_tokens: u32,
    /// Provider stop reason, e.g. `"stop"` or `"length"`, when reported.
    pub finish_reason: Option<String>,
    /// The ordered content blocks of the aggregated response.
    pub content: Vec<MessageContentBlock>,
    /// Provider-reported prompt-cache token usage, when available.
    #[serde(default)]
    pub prompt_cache_usage: Option<PromptCacheUsage>,
}

/// The stream contract: normalized content blocks, or errors.
pub type AgentResponseStream =
    Pin<Box<dyn Stream<Item = Result<MessageContentBlock, CucaError>> + Send>>;

#[cfg(test)]
mod tests {
    use tokio_stream::StreamExt;

    use super::{
        AgentResponseStream, PromptCacheBreakpoint, PromptCacheDirective, PromptCacheUsage,
        ThinkingConfig, ThinkingEffort, ThinkingParams, UnifiedRequest, UnifiedResponse,
    };
    use crate::error::CucaError;
    use crate::types::{
        MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
    };
    #[test]
    fn prompt_cache_fields_default_when_deserializing_legacy_json() {
        let mut request_json = serde_json::to_value(UnifiedRequest::new("legacy-model"))
            .expect("legacy request serialization should succeed");
        request_json
            .as_object_mut()
            .expect("serialized request should be a JSON object")
            .remove("prompt_cache");
        let request: UnifiedRequest = serde_json::from_value(request_json)
            .expect("legacy request deserialization should succeed");
        assert_eq!(request.prompt_cache, PromptCacheDirective::Disabled);

        let mut response_json = serde_json::to_value(UnifiedResponse {
            model: "legacy-model".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 0.5,
            prompt_tokens: 1,
            completion_tokens: 2,
            finish_reason: Some("stop".into()),
            content: vec![MessageContentBlock::Text("done".into())],
            prompt_cache_usage: None,
        })
        .expect("legacy response serialization should succeed");
        response_json
            .as_object_mut()
            .expect("serialized response should be a JSON object")
            .remove("prompt_cache_usage");
        let response: UnifiedResponse = serde_json::from_value(response_json)
            .expect("legacy response deserialization should succeed");
        assert_eq!(response.prompt_cache_usage, None);
    }

    #[test]
    fn prompt_cache_directive_round_trips_ordered_breakpoints() {
        let directive = PromptCacheDirective::Ephemeral {
            breakpoints: vec![
                PromptCacheBreakpoint {
                    message_index: 2,
                    block_index: 1,
                },
                PromptCacheBreakpoint {
                    message_index: 4,
                    block_index: 0,
                },
            ],
        };
        let request = UnifiedRequest::new("model").with_prompt_cache(directive.clone());
        let json =
            serde_json::to_value(&request).expect("directive request serialization should succeed");
        let restored: UnifiedRequest =
            serde_json::from_value(json).expect("directive request deserialization should succeed");
        assert_eq!(restored.prompt_cache, directive);
    }

    #[test]
    fn prompt_cache_usage_round_trips() {
        let usage = PromptCacheUsage {
            read_tokens: 123,
            write_tokens: 45,
        };
        let json = serde_json::to_value(usage).expect("usage serialization should succeed");
        let restored: PromptCacheUsage =
            serde_json::from_value(json).expect("usage deserialization should succeed");
        assert_eq!(restored, usage);
    }

    #[test]
    fn new_request_starts_with_prompt_cache_disabled() {
        assert_eq!(
            UnifiedRequest::new("model").prompt_cache,
            PromptCacheDirective::Disabled
        );
    }
    #[test]
    fn add_tool_preserves_definition() {
        let request = UnifiedRequest::new("model").add_tool(ToolDefinition {
            name: "read_tool_result".into(),
            description: "Read stored output".into(),
            input_schema: serde_json::json!({ "type": "object" }),
        });

        assert_eq!(request.tools[0].name, "read_tool_result");
    }

    #[test]
    fn builder_chain_produces_ordered_system_user_messages() {
        let req = UnifiedRequest::new("gpt-4o")
            .add_system_message("be concise")
            .add_user_message("hello");

        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, MessageRole::System);
        assert_eq!(
            req.messages[0].content,
            vec![MessageContentBlock::Text("be concise".into())]
        );
        assert_eq!(req.messages[1].role, MessageRole::User);
        assert_eq!(
            req.messages[1].content,
            vec![MessageContentBlock::Text("hello".into())]
        );
    }

    #[test]
    fn builders_mutate_expected_fields_and_new_uses_defaults() {
        let req = UnifiedRequest::new("deepseek-v3")
            .add_message(UnifiedMessage::assistant("ok"))
            .set_temperature(0.7)
            .set_max_tokens(512)
            .with_stream(false);

        assert_eq!(req.model, "deepseek-v3");
        assert_eq!(req.provider, ProviderEndpoint::Custom(String::new()));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(512));
        assert!(!req.stream);
        assert_eq!(req.message_count(), 1);

        let default = UnifiedRequest::new("x");
        assert_eq!(default.provider, ProviderEndpoint::Custom(String::new()));
        assert!(default.stream);
        assert_eq!(default.temperature, None);
        assert_eq!(default.max_tokens, None);
        assert_eq!(default.message_count(), 0);
    }

    #[test]
    fn request_serde_round_trips() {
        let req = UnifiedRequest::new("gpt-4o")
            .add_system_message("you are helpful")
            .add_user_message("hi")
            .set_temperature(1.2)
            .set_max_tokens(2048)
            .with_stream(true);

        let json = serde_json::to_value(&req).expect("serialize should succeed");
        let back: UnifiedRequest =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(back, req);
    }

    #[test]
    fn response_serde_round_trips() {
        let res = UnifiedResponse {
            model: "gpt-4o".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.5,
            prompt_tokens: 12,
            completion_tokens: 34,
            finish_reason: Some("stop".into()),
            content: vec![MessageContentBlock::Text("hi there".into())],
            prompt_cache_usage: None,
        };

        let json = serde_json::to_value(&res).expect("serialize should succeed");
        let back: UnifiedResponse =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(back, res);
    }

    #[tokio::test]
    async fn agent_response_stream_yields_blocks_then_ends() {
        let items: Vec<Result<MessageContentBlock, CucaError>> = vec![
            Ok(MessageContentBlock::Text("first".into())),
            Ok(MessageContentBlock::Text("second".into())),
        ];
        let mut stream: AgentResponseStream = Box::pin(tokio_stream::iter(items.into_iter()));

        let first = stream.next().await;
        assert!(matches!(&first, Some(Ok(MessageContentBlock::Text(t))) if t == "first"));
        let second = stream.next().await;
        assert!(matches!(&second, Some(Ok(MessageContentBlock::Text(t))) if t == "second"));
        assert!(stream.next().await.is_none());
    }

    // --- thinking ---

    #[test]
    fn thinking_builders_set_and_clear_the_field() {
        let req = UnifiedRequest::new("gpt-5").enable_thinking(Some(ThinkingEffort::High));
        let thinking = req.thinking.as_ref().unwrap();
        assert!(thinking.enabled);
        assert_eq!(thinking.effort, Some(ThinkingEffort::High));
        assert_eq!(thinking.params, ThinkingParams::None);

        let req = UnifiedRequest::new("gpt-5").enable_thinking(None);
        assert_eq!(req.thinking.as_ref().unwrap().effort, None);
        assert!(req.thinking.as_ref().unwrap().enabled);

        let req = UnifiedRequest::new("gpt-5")
            .enable_thinking(Some(ThinkingEffort::Medium))
            .disable_thinking();
        let thinking = req.thinking.as_ref().unwrap();
        assert!(!thinking.enabled);
        assert_eq!(thinking.effort, None);

        let req = UnifiedRequest::new("gpt-5")
            .with_thinking(ThinkingConfig::with_effort(ThinkingEffort::Low))
            .disable_thinking();
        assert!(!req.thinking.as_ref().unwrap().enabled);

        let req = UnifiedRequest::new("gpt-5");
        assert!(req.thinking.is_none());
    }

    #[test]
    fn thinking_config_constructors() {
        let default = ThinkingConfig::default();
        assert!(default.enabled);
        assert_eq!(default.effort, None);
        assert_eq!(default.params, ThinkingParams::None);

        let disabled = ThinkingConfig::disabled();
        assert!(!disabled.enabled);
        assert_eq!(disabled.effort, None);
        assert_eq!(disabled.params, ThinkingParams::None);

        let effort = ThinkingConfig::with_effort(ThinkingEffort::XHigh);
        assert!(effort.enabled);
        assert_eq!(effort.effort, Some(ThinkingEffort::XHigh));
        assert_eq!(effort.params, ThinkingParams::None);
    }

    #[test]
    fn thinking_types_serde_round_trip() {
        for effort in [
            ThinkingEffort::Minimal,
            ThinkingEffort::Low,
            ThinkingEffort::Medium,
            ThinkingEffort::High,
            ThinkingEffort::XHigh,
        ] {
            let json = serde_json::to_value(effort).unwrap();
            assert!(json.is_string());
            let back: ThinkingEffort = serde_json::from_value(json).unwrap();
            assert_eq!(back, effort);
        }
        // snake_case wire names.
        assert_eq!(
            serde_json::to_value(ThinkingEffort::XHigh).unwrap(),
            serde_json::json!("xhigh")
        );

        for params in [
            ThinkingParams::None,
            ThinkingParams::Anthropic {
                budget_tokens: Some(8192),
                adaptive: false,
            },
            ThinkingParams::Anthropic {
                budget_tokens: None,
                adaptive: true,
            },
            ThinkingParams::Gemini {
                thinking_budget: Some(20_000),
                thinking_level: None,
            },
            ThinkingParams::Gemini {
                thinking_budget: None,
                thinking_level: Some("LOW".into()),
            },
            ThinkingParams::OpenAi {
                reasoning_effort: Some("high".into()),
            },
            ThinkingParams::OpenAi {
                reasoning_effort: None,
            },
        ] {
            let json = serde_json::to_value(&params).unwrap();
            let back: ThinkingParams = serde_json::from_value(json).unwrap();
            assert_eq!(back, params);
        }

        for config in [
            ThinkingConfig::default(),
            ThinkingConfig::disabled(),
            ThinkingConfig::with_effort(ThinkingEffort::Medium),
            ThinkingConfig {
                enabled: true,
                effort: Some(ThinkingEffort::Low),
                params: ThinkingParams::OpenAi {
                    reasoning_effort: Some("high".into()),
                },
            },
        ] {
            let json = serde_json::to_value(&config).unwrap();
            let back: ThinkingConfig = serde_json::from_value(json).unwrap();
            assert_eq!(back, config);
        }
    }

    #[test]
    fn request_serde_round_trips_with_thinking() {
        let req = UnifiedRequest::new("gpt-5")
            .add_user_message("hi")
            .with_thinking(ThinkingConfig {
                enabled: true,
                effort: Some(ThinkingEffort::High),
                params: ThinkingParams::OpenAi {
                    reasoning_effort: Some("high".into()),
                },
            });

        let json = serde_json::to_value(&req).expect("serialize should succeed");
        let back: UnifiedRequest =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(back, req);
        assert_eq!(
            back.thinking.as_ref().unwrap().effort,
            Some(ThinkingEffort::High)
        );
    }
}
