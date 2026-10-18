//! Chat prompt construction for the local Gemma 4 engine.
//!
//! [`chat_prompt`] maps a [`UnifiedRequest`] to the OpenAI-shaped `messages` and
//! `tools` JSON the model's own Jinja chat template expects, and
//! [`ChatTemplate`] renders that through minijinja. The template ships inside
//! the GGUF (`tokenizer.chat_template`); its text is never modified here. Two
//! deliberate differences from the OpenAI wire mapping: tool-call `arguments`
//! stay JSON objects (the template rejects strings), and `ImageBase64` blocks are
//! refused because the engine is text-only.

use minijinja::{Environment, Error, ErrorKind, Value, context};
use serde_json::json;

use crate::error::CucaError;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};

/// Name under which the template is registered in the environment.
const TEMPLATE_NAME: &str = "gemma4";
/// Rendered text that closes a turn; a prompt ending here needs a model turn.
const TURN_END: &str = "<turn|>\n";
/// Opener appended when a rendered prompt ends with a closed turn.
const MODEL_TURN: &str = "<|turn>model\n";

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
            wire.insert("content".into(), json!(joined_text(msg)));
        }
        MessageRole::Assistant => {
            wire.insert("role".into(), json!("assistant"));
            wire.insert("content".into(), json!(joined_text(msg)));
            let reasoning: Vec<&str> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    MessageContentBlock::Thinking { reasoning, .. } => Some(reasoning.as_str()),
                    _ => None,
                })
                .collect();
            if !reasoning.is_empty() {
                wire.insert("reasoning_content".into(), json!(reasoning.concat()));
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
                wire.insert("tool_calls".into(), json!(calls));
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
                wire.insert("tool_call_id".into(), json!(id));
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
            wire.insert("content".into(), json!(output));
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
        messages: json!(messages),
        tools: json!(tools),
        enable_thinking: req.thinking.as_ref().is_some_and(|t| t.enabled),
    })
}

/// A compiled Gemma 4 chat template.
pub(crate) struct ChatTemplate {
    /// Environment owning the compiled template.
    env: Environment<'static>,
    /// BOS text the template prepends (`<bos>`).
    bos_token: String,
}

impl ChatTemplate {
    /// Compile `source`, the Jinja chat template, for rendering with `bos_token`.
    pub(crate) fn new(source: String, bos_token: String) -> Result<Self, CucaError> {
        let template_err = |e: Error| local_err(format!("chat template: {e}"));
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(template_err)?;
        Ok(Self { env, bos_token })
    }

    /// Render `prompt` to the text fed to the tokenizer.
    ///
    /// The generation prompt is always requested; when the result still ends in
    /// a closed turn (`<turn|>` plus newline), a model turn opener is appended.
    pub(crate) fn render(&self, prompt: &ChatPrompt) -> Result<String, CucaError> {
        let template_err = |e: Error| local_err(format!("chat template: {e}"));
        let template = self.env.get_template(TEMPLATE_NAME).map_err(template_err)?;
        let mut text = template
            .render(context! {
                messages => Value::from_serialize(&prompt.messages),
                tools => Value::from_serialize(&prompt.tools),
                add_generation_prompt => true,
                enable_thinking => prompt.enable_thinking,
                preserve_thinking => false,
                bos_token => self.bos_token.as_str(),
                eos_token => "<eos>",
            })
            .map_err(template_err)?;
        if text.ends_with(TURN_END) {
            text.push_str(MODEL_TURN);
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::ThinkingConfig;
    use crate::types::ToolDefinition;

    /// The chat template embedded in the pinned GGUF (`tokenizer.chat_template` of
    /// revision b4243c156154b6dca9324415f8c7ccc098b4aed1), byte-identical to
    /// google/gemma-4-E2B-it `chat_template.jinja`; the engine renders this text.
    const TEMPLATE: &str = include_str!("testdata/gemma4_chat_template.jinja");

    fn template() -> ChatTemplate {
        ChatTemplate::new(TEMPLATE.to_string(), "<bos>".to_string()).expect("template compiles")
    }

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

    #[test]
    fn invalid_template_source_is_reported() {
        let err = ChatTemplate::new("{% if %}".to_string(), "<bos>".to_string())
            .err()
            .expect("syntax error");
        assert!(err.to_string().contains("chat template:"), "{err}");
    }

    #[test]
    fn raise_exception_surfaces_as_a_template_error() {
        let t = ChatTemplate::new(
            "{{ raise_exception('boom') }}".to_string(),
            "<bos>".to_string(),
        )
        .expect("compiles");
        let prompt = chat_prompt(&UnifiedRequest::new("m")).expect("maps");
        let err = t.render(&prompt).expect_err("raises");
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn closed_final_turn_gets_a_model_turn_appended() {
        let source = "{{ 'x<turn|>\\n' }}".to_string();
        let t = ChatTemplate::new(source, "<bos>".to_string()).expect("compiles");
        let prompt = chat_prompt(&UnifiedRequest::new("m")).expect("maps");
        assert_eq!(
            t.render(&prompt).expect("renders"),
            "x<turn|>\n<|turn>model\n"
        );
    }

    /// The four golden requests, by case name.
    fn golden_requests() -> Vec<(&'static str, UnifiedRequest)> {
        let weather_tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Look up the current weather".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" },
                    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                },
                "required": ["city"],
            }),
        };
        let mut tool_turn = UnifiedRequest::new("m")
            .add_user_message("What is the weather in Paris?")
            .add_tool(weather_tool);
        tool_turn.messages.push(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: json!({ "city": "Paris", "days": 3, "metric": true, "tags": ["a", "b"] }),
            }],
            name: None,
            tool_call_id: None,
        });
        tool_turn.messages.push(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "call_1".into(),
                output: "{\"temp\":18}".into(),
            }],
            name: Some("get_weather".into()),
            tool_call_id: Some("call_1".into()),
        });
        let mut history = UnifiedRequest::new("m")
            .add_system_message("You are a calculator.")
            .add_user_message("What is 2+2?");
        history.messages.push(UnifiedMessage::assistant("It is 4."));
        history.messages.push(UnifiedMessage::user("And 3+3?"));
        history.messages.push(UnifiedMessage::assistant("6."));
        history.messages.push(UnifiedMessage::user("Thanks!"));
        vec![
            (
                "system_user",
                UnifiedRequest::new("m")
                    .add_system_message("You are concise.")
                    .add_user_message("Hello"),
            ),
            ("multi_turn", history),
            ("tool_round_trip", tool_turn),
            (
                "thinking",
                UnifiedRequest::new("m")
                    .add_system_message("Reason carefully.")
                    .add_user_message("Is 17 prime?")
                    .with_thinking(ThinkingConfig::default()),
            ),
        ]
    }

    /// Expected renders, generated once with Python
    /// `jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True,
    /// lstrip_blocks=True)` over the exact `messages`/`tools` JSON `chat_prompt`
    /// produces for [`golden_requests`], with `bos_token="<bos>"`.
    #[test]
    fn golden_renders_match_reference_jinja() {
        let expected: [(&str, &str); 4] = [
            (
                "system_user",
                "<bos><|turn>system\nYou are concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n",
            ),
            (
                "multi_turn",
                "<bos><|turn>system\nYou are a calculator.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\nIt is 4.<turn|>\n<|turn>user\nAnd 3+3?<turn|>\n<|turn>model\n6.<turn|>\n<|turn>user\nThanks!<turn|>\n<|turn>model\n",
            ),
            (
                "tool_round_trip",
                "<bos><|turn>system\n<|tool>declaration:get_weather{description:<|\"|>Look up the current weather<|\"|>,parameters:{properties:{city:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>},unit:{enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\nWhat is the weather in Paris?<turn|>\n<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3,metric:true,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|><|tool_response>response:get_weather{value:<|\"|>{\"temp\":18}<|\"|>}<tool_response|>",
            ),
            (
                "thinking",
                "<bos><|turn>system\n<|think|>\nReason carefully.<turn|>\n<|turn>user\nIs 17 prime?<turn|>\n<|turn>model\n",
            ),
        ];
        let template = template();
        for (name, req) in golden_requests() {
            let want = expected
                .iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("no expectation for {name}"))
                .1;
            let prompt = chat_prompt(&req).expect("maps");
            assert_eq!(
                template.render(&prompt).expect("renders"),
                want,
                "case {name}"
            );
        }
    }
}
