//! Chat prompt construction for the local engine.
//!
//! [`chat_prompt`] maps a [`UnifiedRequest`] to the OpenAI-shaped `messages` and
//! `tools` JSON the model's own Jinja chat template expects,
//! [`ChatTemplate`] renders that through minijinja, and [`ChatFormat`] derives
//! from the GGUF what the output side needs: end-of-generation ids, reasoning
//! markers and the tool-call syntax. The template ships inside the GGUF
//! (`tokenizer.chat_template`); its text is never modified here (beyond
//! neutralising HuggingFace's no-output `{% generation %}` tags). Two
//! deliberate differences from the OpenAI wire mapping: tool-call `arguments`
//! stay JSON objects (templates reject strings), and `ImageBase64` blocks are
//! refused because the engine is text-only.

mod format;
#[cfg(test)]
mod golden;
mod template;

pub(crate) use self::format::{ChatFormat, ToolSyntax};
pub(crate) use self::template::ChatTemplate;

use serde_json::json;

use crate::error::CucaError;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};

/// Build a local-provider error.
fn local_err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

/// The template inputs derived from one request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatPrompt {
    /// OpenAI-shaped message array.
    pub(crate) messages: serde_json::Value,
    /// OpenAI-shaped tool array (`[]` when the request has no tools).
    pub(crate) tools: serde_json::Value,
    /// Whether the template should open a thinking channel.
    pub(crate) enable_thinking: bool,
}

/// Join the text blocks of `msg` with newlines.
fn joined_text(msg: &UnifiedMessage) -> String {
    msg.content
        .iter()
        .filter_map(|b| match b {
            MessageContentBlock::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Normalise tool-call arguments to a JSON object (or `null`).
fn call_arguments(
    name: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, CucaError> {
    let not_object = || {
        local_err(format!(
            "tool call `{name}` arguments must be a JSON object"
        ))
    };
    match arguments {
        serde_json::Value::Object(_) | serde_json::Value::Null => Ok(arguments.clone()),
        serde_json::Value::String(text) => match serde_json::from_str(text) {
            Ok(parsed @ serde_json::Value::Object(_)) => Ok(parsed),
            _ => Err(not_object()),
        },
        _ => Err(not_object()),
    }
}

/// Map one message to its OpenAI-shaped JSON.
fn message_to_json(msg: &UnifiedMessage) -> Result<serde_json::Value, CucaError> {
    if msg
        .content
        .iter()
        .any(|b| matches!(b, MessageContentBlock::ImageBase64 { .. }))
    {
        return Err(local_err(
            "the local engine is text-only; ImageBase64 blocks are not supported",
        ));
    }
    let mut wire = serde_json::Map::new();
    match msg.role {
        MessageRole::System | MessageRole::User => {
            let role = if msg.role == MessageRole::System {
                "system"
            } else {
                "user"
            };
            wire.insert("role".into(), json!(role));
            wire.insert("content".into(), joined_text(msg).into());
        }
        MessageRole::Assistant => {
            wire.insert("role".into(), json!("assistant"));
            wire.insert("content".into(), joined_text(msg).into());
            let reasoning: Vec<&str> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    MessageContentBlock::Thinking { reasoning, .. } => Some(reasoning.as_str()),
                    _ => None,
                })
                .collect();
            if !reasoning.is_empty() {
                wire.insert("reasoning_content".into(), reasoning.concat().into());
            }
            let mut calls = Vec::new();
            for block in &msg.content {
                if let MessageContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } = block
                {
                    calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": call_arguments(name, arguments)? },
                    }));
                }
            }
            if !calls.is_empty() {
                wire.insert("tool_calls".into(), calls.into());
            }
        }
        MessageRole::Tool => {
            wire.insert("role".into(), json!("tool"));
            let tool_call_id = msg.tool_call_id.clone().or_else(|| {
                msg.content.iter().find_map(|b| match b {
                    MessageContentBlock::ToolResult { tool_call_id, .. } => {
                        Some(tool_call_id.clone())
                    }
                    _ => None,
                })
            });
            if let Some(id) = tool_call_id {
                wire.insert("tool_call_id".into(), id.into());
            }
            if let Some(name) = &msg.name {
                wire.insert("name".into(), json!(name));
            }
            let output = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    MessageContentBlock::ToolResult { output, .. } => Some(output.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            wire.insert("content".into(), output.into());
        }
    }
    Ok(serde_json::Value::Object(wire))
}

/// Map a request to the template inputs.
///
/// Fails when any message carries an `ImageBase64` block or a tool call whose
/// arguments are not a JSON object (or a string holding one).
pub(crate) fn chat_prompt(req: &UnifiedRequest) -> Result<ChatPrompt, CucaError> {
    let messages = req
        .messages
        .iter()
        .map(message_to_json)
        .collect::<Result<Vec<_>, _>>()?;
    let tools: Vec<serde_json::Value> = req
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })
        })
        .collect();
    Ok(ChatPrompt {
        messages: messages.into(),
        tools: tools.into(),
        enable_thinking: req.thinking.as_ref().is_some_and(|t| t.enabled),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::ThinkingConfig;
    use crate::types::ToolDefinition;

    fn assistant_call(arguments: serde_json::Value) -> UnifiedMessage {
        UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![
                MessageContentBlock::Thinking {
                    reasoning: "need ".into(),
                    signature: None,
                },
                MessageContentBlock::Thinking {
                    reasoning: "weather".into(),
                    signature: Some("sig".into()),
                },
                MessageContentBlock::Text("Checking.".into()),
                MessageContentBlock::Text("One moment.".into()),
                MessageContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments,
                },
            ],
            name: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn maps_roles_text_and_tools() {
        let mut req = UnifiedRequest::new("m")
            .add_system_message("Be brief.")
            .add_user_message("Hi")
            .add_tool(ToolDefinition {
                name: "get_weather".into(),
                description: "Look up weather".into(),
                input_schema: json!({ "type": "object" }),
            })
            .with_thinking(ThinkingConfig::default());
        req.messages.push(UnifiedMessage::assistant("Hello"));
        let prompt = chat_prompt(&req).expect("maps");
        assert_eq!(
            prompt.messages,
            json!([
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "Hi" },
                { "role": "assistant", "content": "Hello" },
            ])
        );
        assert_eq!(
            prompt.tools,
            json!([{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": { "type": "object" },
                },
            }])
        );
        assert!(prompt.enable_thinking);
        let plain = chat_prompt(&UnifiedRequest::new("m").add_user_message("x")).expect("maps");
        assert_eq!(plain.tools, json!([]));
        assert!(!plain.enable_thinking);
        let disabled = UnifiedRequest::new("m").with_thinking(ThinkingConfig::disabled());
        assert!(!chat_prompt(&disabled).expect("maps").enable_thinking);
    }

    #[test]
    fn assistant_tool_call_keeps_object_arguments() {
        let mut req = UnifiedRequest::new("m");
        req.messages
            .push(assistant_call(json!({ "city": "Paris", "days": 3 })));
        let prompt = chat_prompt(&req).expect("maps");
        assert_eq!(
            prompt.messages,
            json!([{
                "role": "assistant",
                "content": "Checking.\nOne moment.",
                "reasoning_content": "need weather",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": { "city": "Paris", "days": 3 },
                    },
                }],
            }])
        );
    }

    #[test]
    fn string_arguments_are_parsed_and_null_is_kept() {
        let mut req = UnifiedRequest::new("m");
        req.messages
            .push(assistant_call(json!("{\"city\":\"Paris\"}")));
        req.messages.push(assistant_call(serde_json::Value::Null));
        let prompt = chat_prompt(&req).expect("maps");
        assert_eq!(
            prompt.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({ "city": "Paris" })
        );
        assert_eq!(
            prompt.messages[1]["tool_calls"][0]["function"]["arguments"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        for bad in [json!("[1,2]"), json!("not json"), json!(7), json!([1])] {
            let mut req = UnifiedRequest::new("m");
            req.messages.push(assistant_call(bad));
            let err = chat_prompt(&req).expect_err("must reject");
            assert!(
                err.to_string()
                    .contains("tool call `get_weather` arguments must be a JSON object"),
                "{err}"
            );
        }
    }

    #[test]
    fn tool_message_carries_id_and_name() {
        let mut req = UnifiedRequest::new("m");
        req.messages.push(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![
                MessageContentBlock::ToolResult {
                    tool_call_id: "block_id".into(),
                    output: "a".into(),
                },
                MessageContentBlock::ToolResult {
                    tool_call_id: "block_id".into(),
                    output: "b".into(),
                },
            ],
            name: Some("get_weather".into()),
            tool_call_id: Some("call_1".into()),
        });
        req.messages.push(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "block_id".into(),
                output: "c".into(),
            }],
            name: None,
            tool_call_id: None,
        });
        let prompt = chat_prompt(&req).expect("maps");
        assert_eq!(
            prompt.messages,
            json!([
                { "role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "a\nb" },
                { "role": "tool", "tool_call_id": "block_id", "content": "c" },
            ])
        );
    }

    #[test]
    fn image_blocks_are_rejected() {
        let mut req = UnifiedRequest::new("m");
        req.messages.push(UnifiedMessage {
            role: MessageRole::User,
            content: vec![
                MessageContentBlock::Text("look".into()),
                MessageContentBlock::ImageBase64 {
                    media_type: "image/png".into(),
                    data: "AAAA".into(),
                },
            ],
            name: None,
            tool_call_id: None,
        });
        let err = chat_prompt(&req).expect_err("images unsupported");
        assert!(
            err.to_string()
                .contains("the local engine is text-only; ImageBase64 blocks are not supported")
        );
    }
}
