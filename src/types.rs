//! Core unified wire types shared by every provider adapter.

use serde::{Deserialize, Serialize};

/// The provider endpoint a request is routed to.
///
/// This is the canonical, provider-agnostic identifier used across the crate.
/// Each variant maps to a concrete upstream service; [`Self::Custom`] lets
/// callers address a bespoke gateway without adding a first-class variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderEndpoint {
    /// llama.cpp server.
    LlamaCpp,
    /// vLLM server.
    Vllm,
    /// LM Studio local server.
    LmStudio,
    /// OpenAI-compatible API.
    OpenAi,
    /// Anthropic API.
    Anthropic,
    /// Google Gemini API.
    GoogleGemini,
    /// DeepSeek API.
    DeepSeek,
    /// A bespoke gateway, identified by its opaque key.
    Custom(String),
}

impl Default for ProviderEndpoint {
    /// The default endpoint is [`ProviderEndpoint::OpenAi`], the most commonly
    /// used provider; callers that need a different default opt in explicitly.
    fn default() -> Self {
        ProviderEndpoint::OpenAi
    }
}

impl std::fmt::Display for ProviderEndpoint {
    /// Human-readable label used in telemetry attributes and logs. Note this is
    /// distinct from the serde name: `openai` here, `"open_ai"` when serialized.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ProviderEndpoint::LlamaCpp => "llamacpp",
            ProviderEndpoint::Vllm => "vllm",
            ProviderEndpoint::LmStudio => "lmstudio",
            ProviderEndpoint::OpenAi => "openai",
            ProviderEndpoint::Anthropic => "anthropic",
            ProviderEndpoint::GoogleGemini => "gemini",
            ProviderEndpoint::DeepSeek => "deepseek",
            ProviderEndpoint::Custom(inner) => inner.as_str(),
        };
        f.write_str(s)
    }
}

/// The role of a participant in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// System prompt / instructions.
    System,
    /// The end user.
    User,
    /// The model / assistant.
    Assistant,
    /// A tool result.
    Tool,
}

/// A single, provider-agnostic message content item.
///
/// Messages carry a heterogeneous sequence of these blocks. The variants cover
/// plain text, images, chain-of-thought, tool invocations, and tool results;
/// provider adapters translate them to each vendor's own wire shape.
//
// spec deviation: the spec derives only Debug/Clone here; the serde repr is an
// addition, adjacently tagged (`type` + `value`) rather than internally tagged
// because internal tagging cannot represent the newtype String variant `Text`:
// session logging serializes Text blocks, so every variant must round-trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum MessageContentBlock {
    /// A plain text block.
    Text(String),
    /// A base64-encoded image, with its MIME media type.
    ImageBase64 {
        /// MIME type of the encoded image, e.g. `"image/png"`.
        media_type: String,
        /// Base64-encoded image bytes.
        data: String,
    },
    /// Reasoning / chain-of-thought content emitted by the model.
    Thinking {
        /// The reasoning text.
        reasoning: String,
        /// An optional provider signature authenticating the reasoning.
        signature: Option<String>,
    },
    /// An invocation of a tool by the model.
    ToolCall {
        /// Provider-assigned id of the call.
        id: String,
        /// Name of the tool being invoked.
        name: String,
        /// JSON-encoded arguments passed to the tool.
        arguments: serde_json::Value,
    },
    /// The result of a previously issued tool call.
    ToolResult {
        /// Id of the tool call this result answers.
        tool_call_id: String,
        /// String output produced by the tool.
        output: String,
    },
}

/// A provider-neutral tool definition available to a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Stable name used by model tool calls.
    pub name: String,
    /// Human-readable guidance for when to call the tool.
    pub description: String,
    /// JSON Schema describing the tool's input object.
    pub input_schema: serde_json::Value,
}

/// A single unified message in a conversation.
///
/// Combines a [`MessageRole`] with an ordered list of content blocks and
/// optional per-message annotations used by tool flows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedMessage {
    /// The role of the speaker.
    pub role: MessageRole,
    /// The ordered content blocks that make up the message.
    pub content: Vec<MessageContentBlock>,
    /// Optional speaker name annotation (e.g. for multi-agent routing).
    pub name: Option<String>,
    /// Id of the tool call this message answers (present on tool-result messages).
    pub tool_call_id: Option<String>,
}

impl UnifiedMessage {
    /// Build a system message from a single text block.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![MessageContentBlock::Text(text.into())],
            name: None,
            tool_call_id: None,
        }
    }

    /// Build a user message from a single text block.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![MessageContentBlock::Text(text.into())],
            name: None,
            tool_call_id: None,
        }
    }

    /// Build an assistant message from a single text block.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::Text(text.into())],
            name: None,
            tool_call_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip<T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug>(
        value: &T,
    ) {
        let json = serde_json::to_value(value).expect("serialize should succeed");
        let back: T = serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(*value, back);
    }

    #[test]
    fn message_content_block_round_trips() {
        let text = MessageContentBlock::Text("hello".into());
        assert_eq!(
            serde_json::to_value(&text).expect("serialize should succeed"),
            json!({ "type": "text", "value": "hello" })
        );
        round_trip(&text);
        round_trip(&MessageContentBlock::ImageBase64 {
            media_type: "image/png".into(),
            data: "aGVsbG8=".into(),
        });
        round_trip(&MessageContentBlock::Thinking {
            reasoning: "think step by step".into(),
            signature: None,
        });
        round_trip(&MessageContentBlock::Thinking {
            reasoning: "think step by step".into(),
            signature: Some("sig-123".into()),
        });
        round_trip(&MessageContentBlock::ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: json!({ "query": "cuca", "limit": 5 }),
        });
        round_trip(&MessageContentBlock::ToolResult {
            tool_call_id: "call_1".into(),
            output: "42".into(),
        });
    }

    #[test]
    fn provider_endpoint_round_trips_and_serde_names() {
        let cases = [
            (ProviderEndpoint::LlamaCpp, json!("llama_cpp")),
            (ProviderEndpoint::Vllm, json!("vllm")),
            (ProviderEndpoint::LmStudio, json!("lm_studio")),
            (ProviderEndpoint::OpenAi, json!("open_ai")),
            (ProviderEndpoint::Anthropic, json!("anthropic")),
            (ProviderEndpoint::GoogleGemini, json!("google_gemini")),
            (ProviderEndpoint::DeepSeek, json!("deep_seek")),
            // A newtype variant serializes under its variant name as the map key, so
            // `Custom("my-gateway")` becomes `{"custom": "my-gateway"}`.
            (
                ProviderEndpoint::Custom("my-gateway".into()),
                json!({ "custom": "my-gateway" }),
            ),
        ];
        for (endpoint, expected) in cases {
            let serialized = serde_json::to_value(&endpoint).expect("serialize should succeed");
            assert_eq!(serialized, expected, "unexpected name for {endpoint:?}");
            round_trip(&endpoint);
        }
    }

    #[test]
    fn message_role_round_trips_and_serde_names() {
        let cases = [
            (MessageRole::System, "system"),
            (MessageRole::User, "user"),
            (MessageRole::Assistant, "assistant"),
            (MessageRole::Tool, "tool"),
        ];
        for (role, expected) in cases {
            let serialized = serde_json::to_value(role).expect("serialize should succeed");
            assert_eq!(serialized, json!(expected), "unexpected name for {role:?}");
            round_trip(&role);
        }
    }

    #[test]
    fn constructors_build_expected_messages() {
        let system = UnifiedMessage::system("sys");
        assert_eq!(system.role, MessageRole::System);
        assert_eq!(
            system.content,
            vec![MessageContentBlock::Text("sys".into())]
        );
        assert_eq!(system.name, None);
        assert_eq!(system.tool_call_id, None);

        let user = UnifiedMessage::user("usr");
        assert_eq!(user.role, MessageRole::User);
        assert_eq!(user.content, vec![MessageContentBlock::Text("usr".into())]);
        assert_eq!(user.name, None);
        assert_eq!(user.tool_call_id, None);

        let assistant = UnifiedMessage::assistant("asst");
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert_eq!(
            assistant.content,
            vec![MessageContentBlock::Text("asst".into())]
        );
        assert_eq!(assistant.name, None);
        assert_eq!(assistant.tool_call_id, None);
    }

    #[test]
    fn provider_endpoint_default_is_openai() {
        assert_eq!(ProviderEndpoint::default(), ProviderEndpoint::OpenAi);
    }

    #[test]
    fn provider_endpoint_display() {
        let cases = [
            (ProviderEndpoint::LlamaCpp, "llamacpp"),
            (ProviderEndpoint::Vllm, "vllm"),
            (ProviderEndpoint::LmStudio, "lmstudio"),
            (ProviderEndpoint::OpenAi, "openai"),
            (ProviderEndpoint::Anthropic, "anthropic"),
            (ProviderEndpoint::GoogleGemini, "gemini"),
            (ProviderEndpoint::DeepSeek, "deepseek"),
            (ProviderEndpoint::Custom("my-gateway".into()), "my-gateway"),
        ];
        for (endpoint, expected) in cases {
            assert_eq!(
                endpoint.to_string(),
                expected,
                "unexpected display for {endpoint:?}"
            );
        }
    }
    #[test]
    fn tool_definition_round_trips() {
        round_trip(&ToolDefinition {
            name: "read_tool_result".into(),
            description: "Read stored output".into(),
            input_schema: json!({ "type": "object" }),
        });
    }
}
