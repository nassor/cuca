//! The OpenAI-compatible wire format of `POST /v1/chat/completions` and
//! `GET /v1/models`, as vLLM and llama.cpp's server speak it, and its mapping
//! to and from cuca's [`UnifiedRequest`] / [`MessageContentBlock`].
//!
//! This module is data and pure functions; the routes live in `http`.
//!
//! # Supported request fields
//!
//! `model`, `messages` (roles `system`, `developer`, `user`, `assistant`,
//! `tool`; `content` a string or an array of `text` parts; assistant
//! `tool_calls`, `reasoning_content`), `temperature`, `top_p`, `top_k`,
//! `seed`, `max_tokens` / `max_completion_tokens`, `stop`, `stream`,
//! `stream_options.include_usage`, `tools`, `tool_choice` (`none` drops the
//! tools, a named function keeps only that one; `required` is treated as
//! `auto`, since the engine cannot force a call), `reasoning_effort` and
//! `chat_template_kwargs.enable_thinking`.
//!
//! Fields whose meaning the engine cannot honour are rejected with a 400
//! rather than ignored: image or audio parts, `n` other than 1, `logprobs`,
//! non-zero `frequency_penalty` / `presence_penalty`, `logit_bias`,
//! `response_format` other than `text`, and `function_call` style messages.
//! Fields that do not change the output (`user`, `store`, `metadata`, ...)
//! are ignored.
//!
//! The engine has no raw-prompt entry (every request goes through the chat
//! template), so `POST /v1/completions` is not served.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::http::ApiError;
use crate::request::{
    PromptCacheUsage, ThinkingConfig, ThinkingEffort, TokenUsage, UnifiedRequest,
};
use crate::types::{MessageContentBlock, MessageRole, ToolDefinition, UnifiedMessage};

/// A `POST /v1/chat/completions` body.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// The model to run; optional when one model is loaded.
    pub model: Option<String>,
    /// The conversation.
    pub messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    stop: Option<Stop>,
    /// Stream the response as server-sent events.
    #[serde(default)]
    pub stream: bool,
    stream_options: Option<StreamOptions>,
    tools: Option<Vec<WireTool>>,
    tool_choice: Option<Value>,
    reasoning_effort: Option<String>,
    chat_template_kwargs: Option<Map<String, Value>>,
    n: Option<u32>,
    logprobs: Option<bool>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    logit_bias: Option<Map<String, Value>>,
    response_format: Option<Value>,
}

/// `stop`: one string or a list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

/// `stream_options`.
#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// One message of the request.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    role: String,
    content: Option<Content>,
    name: Option<String>,
    tool_calls: Option<Vec<WireToolCall>>,
    tool_call_id: Option<String>,
    reasoning_content: Option<String>,
    /// vLLM's newer spelling of `reasoning_content`.
    reasoning: Option<String>,
}

/// A message's `content`: a string or an array of typed parts.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

/// A tool call, in a request's assistant message or a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireToolCall {
    /// Call id.
    #[serde(default)]
    pub id: String,
    /// Always `function`.
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    /// Name and JSON-text arguments.
    pub function: WireFunctionCall,
    /// Position in the message; streaming chunks only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

fn function_kind() -> String {
    "function".to_string()
}

/// The function of a [`WireToolCall`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments as JSON text.
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: String,
    function: WireToolFunction,
}

#[derive(Debug, Deserialize)]
struct WireToolFunction {
    name: String,
    #[serde(default)]
    description: String,
    parameters: Option<Value>,
}

/// What the handler needs besides the [`UnifiedRequest`].
#[derive(Debug, Clone, Copy)]
pub struct RequestOptions {
    /// Send a final usage chunk when streaming.
    pub include_usage: bool,
}

impl ChatCompletionRequest {
    /// Map to a request for the local engine, rejecting what it cannot honour.
    ///
    /// # Errors
    ///
    /// A 400 [`ApiError`] naming the unsupported or invalid field.
    pub fn into_unified(self, model: &str) -> Result<(UnifiedRequest, RequestOptions), ApiError> {
        self.reject_unsupported()?;
        let thinking = self.thinking()?;
        let tools = self.tools()?;
        let mut request = UnifiedRequest::new(model);
        request.stream = true;
        request.temperature = self.temperature;
        request.top_p = self.top_p;
        request.top_k = self.top_k;
        request.seed = self.seed;
        request.max_tokens = self.max_completion_tokens.or(self.max_tokens);
        request.stop = match self.stop {
            None => Vec::new(),
            Some(Stop::One(s)) => vec![s],
            Some(Stop::Many(list)) => list,
        };
        request.thinking = thinking;
        request.tools = tools;
        if self.messages.is_empty() {
            return Err(ApiError::invalid_request("`messages` must not be empty").param("messages"));
        }
        request.messages = self
            .messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| m.into_unified(i))
            .collect::<Result<_, _>>()?;
        let include_usage = self.stream_options.is_some_and(|o| o.include_usage);
        Ok((request, RequestOptions { include_usage }))
    }

    fn reject_unsupported(&self) -> Result<(), ApiError> {
        let unsupported = |param: &str, why: &str| {
            Err(
                ApiError::invalid_request(format!("`{param}` is not supported: {why}"))
                    .param(param),
            )
        };
        if self.n.is_some_and(|n| n != 1) {
            return unsupported("n", "only one choice per request");
        }
        if self.logprobs == Some(true) {
            return unsupported("logprobs", "the engine does not return log probabilities");
        }
        if self.frequency_penalty.is_some_and(|p| p != 0.0) {
            return unsupported(
                "frequency_penalty",
                "the engine has no repetition penalties",
            );
        }
        if self.presence_penalty.is_some_and(|p| p != 0.0) {
            return unsupported("presence_penalty", "the engine has no repetition penalties");
        }
        if self.logit_bias.as_ref().is_some_and(|b| !b.is_empty()) {
            return unsupported("logit_bias", "the engine has no logit bias");
        }
        if let Some(format) = &self.response_format
            && format
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| t != "text")
        {
            return unsupported("response_format", "only free text is generated");
        }
        Ok(())
    }

    /// `reasoning_effort` and `chat_template_kwargs.enable_thinking` as the
    /// request's thinking configuration.
    fn thinking(&self) -> Result<Option<ThinkingConfig>, ApiError> {
        let effort = match self.reasoning_effort.as_deref() {
            None => None,
            Some("none") => return Ok(Some(ThinkingConfig::disabled())),
            Some("minimal") => Some(ThinkingEffort::Minimal),
            Some("low") => Some(ThinkingEffort::Low),
            Some("medium") => Some(ThinkingEffort::Medium),
            Some("high") => Some(ThinkingEffort::High),
            Some("xhigh") => Some(ThinkingEffort::XHigh),
            Some(other) => {
                return Err(ApiError::invalid_request(format!(
                    "unknown reasoning_effort `{other}`: expected none, minimal, low, medium, high or xhigh"
                ))
                .param("reasoning_effort"));
            }
        };
        let enable = self
            .chat_template_kwargs
            .as_ref()
            .and_then(|k| k.get("enable_thinking"))
            .map(|v| {
                v.as_bool().ok_or_else(|| {
                    ApiError::invalid_request(
                        "`chat_template_kwargs.enable_thinking` must be a boolean",
                    )
                    .param("chat_template_kwargs")
                })
            })
            .transpose()?;
        Ok(match (enable, effort) {
            (Some(false), _) => Some(ThinkingConfig::disabled()),
            (_, Some(effort)) => Some(ThinkingConfig::with_effort(effort)),
            (Some(true), None) => Some(ThinkingConfig::default()),
            (None, None) => None,
        })
    }

    /// `tools` filtered by `tool_choice`.
    fn tools(&self) -> Result<Vec<ToolDefinition>, ApiError> {
        let Some(tools) = &self.tools else {
            return Ok(Vec::new());
        };
        let mut defs = Vec::with_capacity(tools.len());
        for tool in tools {
            if tool.kind != "function" {
                return Err(ApiError::invalid_request(format!(
                    "tool type `{}` is not supported: only `function`",
                    tool.kind
                ))
                .param("tools"));
            }
            defs.push(ToolDefinition {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                input_schema: tool
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
            });
        }
        match &self.tool_choice {
            None | Some(Value::Null) => Ok(defs),
            Some(Value::String(choice)) => match choice.as_str() {
                "none" => Ok(Vec::new()),
                "auto" | "required" => Ok(defs),
                other => Err(ApiError::invalid_request(format!(
                    "unknown tool_choice `{other}`: expected none, auto, required or a function"
                ))
                .param("tool_choice")),
            },
            Some(choice) => {
                let name = choice
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ApiError::invalid_request(
                            "`tool_choice` must be a string or {\"type\":\"function\",\"function\":{\"name\":...}}",
                        )
                        .param("tool_choice")
                    })?;
                defs.retain(|d| d.name == name);
                if defs.is_empty() {
                    return Err(ApiError::invalid_request(format!(
                        "tool_choice names `{name}`, which is not in `tools`"
                    ))
                    .param("tool_choice"));
                }
                Ok(defs)
            }
        }
    }
}

impl ChatMessage {
    fn into_unified(self, index: usize) -> Result<UnifiedMessage, ApiError> {
        let param = format!("messages[{index}]");
        let text = self.text(&param)?;
        let role = match self.role.as_str() {
            "system" | "developer" => MessageRole::System,
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            other => {
                return Err(ApiError::invalid_request(format!(
                    "unsupported message role `{other}`: expected system, developer, user, assistant or tool"
                ))
                .param(param));
            }
        };
        let mut content = Vec::new();
        let mut tool_call_id = None;
        match role {
            MessageRole::Assistant => {
                if let Some(reasoning) = self.reasoning_content.or(self.reasoning)
                    && !reasoning.is_empty()
                {
                    content.push(MessageContentBlock::Thinking {
                        reasoning,
                        signature: None,
                    });
                }
                if !text.is_empty() {
                    content.push(MessageContentBlock::Text(text));
                }
                for call in self.tool_calls.unwrap_or_default() {
                    // OpenAI sends arguments as JSON text; the engine's chat
                    // template wants the parsed object.
                    let arguments = serde_json::from_str(&call.function.arguments)
                        .unwrap_or(Value::String(call.function.arguments));
                    content.push(MessageContentBlock::ToolCall {
                        id: call.id,
                        name: call.function.name,
                        arguments,
                    });
                }
            }
            MessageRole::Tool => {
                let id = self.tool_call_id.ok_or_else(|| {
                    ApiError::invalid_request("a `tool` message needs `tool_call_id`").param(&param)
                })?;
                content.push(MessageContentBlock::ToolResult {
                    tool_call_id: id.clone(),
                    output: text,
                });
                tool_call_id = Some(id);
            }
            MessageRole::System | MessageRole::User => {
                content.push(MessageContentBlock::Text(text))
            }
        }
        Ok(UnifiedMessage {
            role,
            content,
            name: self.name,
            tool_call_id,
        })
    }

    /// The text of `content`: a string, or the `text` parts joined by a
    /// newline; any other part type is rejected.
    fn text(&self, param: &str) -> Result<String, ApiError> {
        match &self.content {
            None => Ok(String::new()),
            Some(Content::Text(text)) => Ok(text.clone()),
            Some(Content::Parts(parts)) => {
                let mut texts = Vec::with_capacity(parts.len());
                for part in parts {
                    if part.kind != "text" {
                        return Err(ApiError::invalid_request(format!(
                            "content part type `{}` is not supported: the engine is text-only",
                            part.kind
                        ))
                        .param(param));
                    }
                    texts.push(part.text.as_deref().unwrap_or(""));
                }
                Ok(texts.join("\n"))
            }
        }
    }
}

/// Token usage of a response.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens, reasoning included.
    pub completion_tokens: u32,
    /// Their sum.
    pub total_tokens: u32,
    /// Where the prompt tokens came from.
    pub prompt_tokens_details: PromptTokensDetails,
    /// Reasoning share of the completion, when the model reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// `usage.prompt_tokens_details`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PromptTokensDetails {
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: u32,
}

/// `usage.completion_tokens_details`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CompletionTokensDetails {
    /// Tokens spent on reasoning.
    pub reasoning_tokens: u32,
}

impl Usage {
    /// The usage of one finished request from the engine's exact counts.
    pub fn new(usage: TokenUsage, cache: Option<PromptCacheUsage>) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.prompt_tokens.saturating_add(usage.completion_tokens),
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: cache.map_or(0, |c| c.read_tokens),
            },
            completion_tokens_details: usage
                .reasoning_tokens
                .map(|reasoning_tokens| CompletionTokensDetails { reasoning_tokens }),
        }
    }
}

/// The reply of a non-streaming request.
#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: [Choice; 1],
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
}

/// The blocks of a finished generation gathered into one message.
#[derive(Debug, Default)]
pub struct Collected {
    text: String,
    reasoning: String,
    tool_calls: Vec<WireToolCall>,
}

impl Collected {
    /// Take one block from the stream.
    pub fn push(&mut self, block: MessageContentBlock) {
        match block {
            MessageContentBlock::Text(t) => self.text.push_str(&t),
            MessageContentBlock::Thinking { reasoning, .. } => self.reasoning.push_str(&reasoning),
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                let index = self.tool_calls.len();
                self.tool_calls
                    .push(wire_tool_call(id, name, &arguments, Some(index)));
            }
            // The engine emits neither; a tool result or image is not output.
            MessageContentBlock::ToolResult { .. } | MessageContentBlock::ImageBase64 { .. } => {}
        }
    }

    /// Whether the model called a tool.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The `chat.completion` object.
    pub fn into_completion(
        self,
        id: String,
        model: String,
        finish_reason: &'static str,
        usage: Usage,
    ) -> ChatCompletion {
        let tool_calls: Vec<WireToolCall> = self
            .tool_calls
            .into_iter()
            .map(|mut c| {
                c.index = None;
                c
            })
            .collect();
        ChatCompletion {
            id,
            object: "chat.completion",
            created: unix_secs(),
            model,
            choices: [Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: (!self.text.is_empty() || tool_calls.is_empty()).then_some(self.text),
                    reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
                    tool_calls,
                },
                finish_reason,
            }],
            usage,
        }
    }
}

/// The finish reason of a response: a tool call wins, then the engine's
/// (`stop` or `length`); a stream that ended without a report is a `stop`.
pub fn finish_reason(called_tool: bool, engine: Option<&'static str>) -> &'static str {
    if called_tool {
        "tool_calls"
    } else {
        engine.unwrap_or("stop")
    }
}

fn wire_tool_call(
    id: String,
    name: String,
    arguments: &Value,
    index: Option<usize>,
) -> WireToolCall {
    let arguments = match arguments {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    WireToolCall {
        id,
        kind: function_kind(),
        function: WireFunctionCall { name, arguments },
        index,
    }
}

/// Builds the `chat.completion.chunk` events of one streamed response.
#[derive(Debug)]
pub struct ChunkWriter {
    id: String,
    model: String,
    created: u64,
    /// `usage: null` on every chunk but the last (`include_usage`).
    null_usage: bool,
    tool_calls: usize,
}

impl ChunkWriter {
    /// A writer for response `id` of `model`.
    pub fn new(id: String, model: String, options: RequestOptions) -> Self {
        Self {
            id,
            model,
            created: unix_secs(),
            null_usage: options.include_usage,
            tool_calls: 0,
        }
    }

    fn chunk(&self, choices: Value, usage: Option<Value>) -> String {
        let mut chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        });
        if let Some(usage) = usage.or(self.null_usage.then_some(Value::Null))
            && let Some(map) = chunk.as_object_mut()
        {
            map.insert("usage".into(), usage);
        }
        chunk.to_string()
    }

    fn delta(&self, delta: Value, finish_reason: Option<&str>) -> String {
        self.chunk(
            json!([{ "index": 0, "delta": delta, "finish_reason": finish_reason }]),
            None,
        )
    }

    /// The first chunk, announcing the assistant role.
    pub fn role(&self) -> String {
        self.delta(json!({ "role": "assistant", "content": "" }), None)
    }

    /// The chunk of one block, `None` for a block that is not output.
    pub fn block(&mut self, block: MessageContentBlock) -> Option<String> {
        match block {
            MessageContentBlock::Text(text) => Some(self.delta(json!({ "content": text }), None)),
            MessageContentBlock::Thinking { reasoning, .. } => {
                Some(self.delta(json!({ "reasoning_content": reasoning }), None))
            }
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                let call = wire_tool_call(id, name, &arguments, Some(self.tool_calls));
                self.tool_calls += 1;
                Some(self.delta(json!({ "tool_calls": [call] }), None))
            }
            MessageContentBlock::ToolResult { .. } | MessageContentBlock::ImageBase64 { .. } => {
                None
            }
        }
    }

    /// Whether the request asked for a final usage chunk.
    pub fn wants_usage(&self) -> bool {
        self.null_usage
    }

    /// Whether a tool call went out.
    pub fn called_tool(&self) -> bool {
        self.tool_calls > 0
    }

    /// The last content chunk, carrying the finish reason.
    pub fn finish(&self, reason: &str) -> String {
        self.delta(json!({}), Some(reason))
    }

    /// The usage chunk (`choices: []`) of `include_usage`.
    pub fn usage(&self, usage: &Usage) -> String {
        self.chunk(json!([]), serde_json::to_value(usage).ok())
    }
}

/// One entry of `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelCard {
    /// The name clients pass as `model`.
    pub id: String,
    /// Always `model`.
    pub object: &'static str,
    /// When the model finished loading, seconds since the Unix epoch.
    pub created: u64,
    /// Always `cuca`.
    pub owned_by: &'static str,
    /// The context length, as vLLM names it.
    pub max_model_len: usize,
}

impl ModelCard {
    /// The card of a ready model.
    pub fn new(id: String, created: u64, max_model_len: usize) -> Self {
        Self {
            id,
            object: "model",
            created,
            owned_by: "cuca",
            max_model_len,
        }
    }
}

/// The `GET /v1/models` body.
pub fn model_list(cards: Vec<ModelCard>) -> Value {
    json!({ "object": "list", "data": cards })
}

/// Seconds since the Unix epoch.
pub fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `chatcmpl-` plus 24 random alphanumerics.
pub fn completion_id() -> String {
    format!("chatcmpl-{}", random_token(24))
}

/// `n` random lowercase alphanumerics; falls back to a time-derived string if
/// the system's randomness is unavailable (an id only needs to be distinct).
pub fn random_token(n: usize) -> String {
    const ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut bytes = vec![0u8; n];
    if getrandom::fill(&mut bytes).is_err() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (nanos >> ((i % 16) * 8)) as u8 ^ (i as u8).wrapping_mul(31);
        }
    }
    bytes
        .iter()
        .map(|b| char::from(ALPHABET[usize::from(*b) % ALPHABET.len()]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: Value) -> ChatCompletionRequest {
        serde_json::from_value(body).expect("valid body")
    }

    fn unified(body: Value) -> UnifiedRequest {
        parse(body).into_unified("m").expect("maps").0
    }

    fn rejected(body: Value) -> String {
        parse(body)
            .into_unified("m")
            .expect_err("rejected")
            .message()
            .to_string()
    }

    #[test]
    fn roles_content_forms_and_sampling_map_to_the_unified_request() {
        let req = unified(json!({
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": "Dev note."},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            ],
            "temperature": 0.2, "top_p": 0.9, "top_k": 40, "seed": 7,
            "max_tokens": 10, "max_completion_tokens": 20,
            "stop": "END",
        }));
        assert_eq!(req.messages[0].role, MessageRole::System);
        assert_eq!(
            req.messages[1].role,
            MessageRole::System,
            "developer is a system message"
        );
        assert_eq!(
            req.messages[2].content,
            [MessageContentBlock::Text("a\nb".into())]
        );
        assert_eq!(
            (req.temperature, req.top_p, req.top_k, req.seed),
            (Some(0.2), Some(0.9), Some(40), Some(7))
        );
        assert_eq!(
            req.max_tokens,
            Some(20),
            "max_completion_tokens wins over max_tokens"
        );
        assert_eq!(req.stop, ["END"]);
        let req =
            unified(json!({"messages": [{"role": "user", "content": "x"}], "stop": ["a", "b"]}));
        assert_eq!(req.stop, ["a", "b"]);
    }

    #[test]
    fn an_assistant_turn_keeps_reasoning_text_and_parsed_tool_calls() {
        let req = unified(json!({"messages": [
            {"role": "assistant", "content": "Checking.", "reasoning_content": "need weather",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]}));
        assert_eq!(
            req.messages[0].content,
            [
                MessageContentBlock::Thinking {
                    reasoning: "need weather".into(),
                    signature: None
                },
                MessageContentBlock::Text("Checking.".into()),
                MessageContentBlock::ToolCall {
                    id: "c1".into(),
                    name: "get_weather".into(),
                    arguments: json!({"city": "Rome"}),
                },
            ]
        );
        assert_eq!(req.messages[1].role, MessageRole::Tool);
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(
            req.messages[1].content,
            [MessageContentBlock::ToolResult {
                tool_call_id: "c1".into(),
                output: "sunny".into()
            }]
        );
    }

    #[test]
    fn tool_choice_filters_the_tools() {
        let tool = |name: &str| json!({"type": "function", "function": {"name": name, "parameters": {"type": "object"}}});
        let body = |choice: Value| json!({"messages": [{"role": "user", "content": "x"}], "tools": [tool("a"), tool("b")], "tool_choice": choice});
        assert_eq!(unified(body(json!("auto"))).tools.len(), 2);
        assert_eq!(unified(body(json!("required"))).tools.len(), 2);
        assert_eq!(unified(body(json!("none"))).tools.len(), 0);
        let named = unified(body(json!({"type": "function", "function": {"name": "b"}})));
        assert_eq!(named.tools.len(), 1);
        assert_eq!(named.tools[0].name, "b");
        assert!(
            rejected(body(
                json!({"type": "function", "function": {"name": "zzz"}})
            ))
            .contains("zzz")
        );
    }

    #[test]
    fn reasoning_effort_and_enable_thinking_map_to_the_thinking_config() {
        let body = |extra: Value| {
            let mut b = json!({"messages": [{"role": "user", "content": "x"}]});
            b.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            unified(b).thinking
        };
        assert_eq!(body(json!({})), None);
        assert_eq!(
            body(json!({"reasoning_effort": "none"})),
            Some(ThinkingConfig::disabled())
        );
        assert_eq!(
            body(json!({"reasoning_effort": "high"})),
            Some(ThinkingConfig::with_effort(ThinkingEffort::High))
        );
        assert_eq!(
            body(json!({"chat_template_kwargs": {"enable_thinking": false}})),
            Some(ThinkingConfig::disabled())
        );
        assert_eq!(
            body(json!({"chat_template_kwargs": {"enable_thinking": true}})),
            Some(ThinkingConfig::default())
        );
        // An explicit off beats an effort.
        assert_eq!(
            body(
                json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": false}})
            ),
            Some(ThinkingConfig::disabled())
        );
    }

    #[test]
    fn what_the_engine_cannot_honour_is_rejected_naming_the_field() {
        let user = json!([{"role": "user", "content": "x"}]);
        let with = |k: &str, v: Value| rejected(json!({"messages": user, k: v}));
        assert!(with("n", json!(2)).contains("`n`"));
        assert!(with("logprobs", json!(true)).contains("logprobs"));
        assert!(with("presence_penalty", json!(0.5)).contains("presence_penalty"));
        assert!(with("logit_bias", json!({"5": 1})).contains("logit_bias"));
        assert!(
            with("response_format", json!({"type": "json_object"})).contains("response_format")
        );
        assert!(with("reasoning_effort", json!("max")).contains("max"));
        let image = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]);
        let msg = rejected(json!({"messages": image}));
        assert!(
            msg.contains("image_url") && msg.contains("text-only"),
            "{msg}"
        );
        assert!(rejected(json!({"messages": []})).contains("messages"));
        assert!(
            rejected(json!({"messages": [{"role": "tool", "content": "x"}]}))
                .contains("tool_call_id")
        );
        assert!(
            rejected(json!({"messages": [{"role": "function", "content": "x"}]}))
                .contains("function")
        );
    }

    #[test]
    fn harmless_defaults_are_accepted() {
        let req = parse(json!({"messages": [{"role": "user", "content": "x"}],
            "n": 1, "logprobs": false, "frequency_penalty": 0, "logit_bias": {},
            "response_format": {"type": "text"}, "user": "u", "store": false}));
        assert!(req.into_unified("m").is_ok());
    }

    #[test]
    fn usage_reports_exact_counts_cached_and_reasoning_tokens() {
        let usage = Usage::new(
            TokenUsage {
                prompt_tokens: 30,
                completion_tokens: 12,
                reasoning_tokens: Some(5),
            },
            Some(PromptCacheUsage {
                read_tokens: 16,
                write_tokens: 14,
            }),
        );
        assert_eq!(
            serde_json::to_value(usage).unwrap(),
            json!({"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42,
                   "prompt_tokens_details": {"cached_tokens": 16},
                   "completion_tokens_details": {"reasoning_tokens": 5}})
        );
        let plain = Usage::new(
            TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                reasoning_tokens: None,
            },
            None,
        );
        let v = serde_json::to_value(plain).unwrap();
        assert!(v.get("completion_tokens_details").is_none());
        assert_eq!(v["prompt_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn a_collected_reply_becomes_one_message_with_reasoning_and_tool_calls() {
        let mut c = Collected::default();
        c.push(MessageContentBlock::Thinking {
            reasoning: "hm".into(),
            signature: None,
        });
        c.push(MessageContentBlock::Text("He".into()));
        c.push(MessageContentBlock::Text("llo".into()));
        c.push(MessageContentBlock::ToolCall {
            id: "c1".into(),
            name: "f".into(),
            arguments: json!({"x": 1}),
        });
        assert!(c.has_tool_calls());
        let usage = Usage::new(
            TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 4,
                reasoning_tokens: None,
            },
            None,
        );
        let v = serde_json::to_value(c.into_completion(
            "chatcmpl-1".into(),
            "m".into(),
            "tool_calls",
            usage,
        ))
        .unwrap();
        assert_eq!(v["object"], "chat.completion");
        let message = &v["choices"][0]["message"];
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "Hello");
        assert_eq!(message["reasoning_content"], "hm");
        assert_eq!(
            message["tool_calls"],
            json!([{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1}"}}])
        );
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn a_reply_that_is_only_a_tool_call_has_null_content() {
        let mut c = Collected::default();
        c.push(MessageContentBlock::ToolCall {
            id: "c".into(),
            name: "f".into(),
            arguments: json!({}),
        });
        let usage = Usage::new(
            TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                reasoning_tokens: None,
            },
            None,
        );
        let v =
            serde_json::to_value(c.into_completion("i".into(), "m".into(), "tool_calls", usage))
                .unwrap();
        assert!(v["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn the_finish_reason_prefers_a_tool_call_then_the_engine() {
        assert_eq!(finish_reason(true, Some("length")), "tool_calls");
        assert_eq!(finish_reason(false, Some("length")), "length");
        assert_eq!(finish_reason(false, None), "stop");
    }

    #[test]
    fn stream_chunks_follow_the_openai_shape() {
        let mut w = ChunkWriter::new(
            "chatcmpl-x".into(),
            "m".into(),
            RequestOptions {
                include_usage: false,
            },
        );
        let first: Value = serde_json::from_str(&w.role()).unwrap();
        assert_eq!(first["object"], "chat.completion.chunk");
        assert_eq!(
            first["choices"][0]["delta"],
            json!({"role": "assistant", "content": ""})
        );
        assert!(first["choices"][0]["finish_reason"].is_null());
        assert!(
            first.get("usage").is_none(),
            "no usage key without include_usage"
        );

        let text: Value =
            serde_json::from_str(&w.block(MessageContentBlock::Text("hi".into())).unwrap())
                .unwrap();
        assert_eq!(text["choices"][0]["delta"], json!({"content": "hi"}));
        let think: Value = serde_json::from_str(
            &w.block(MessageContentBlock::Thinking {
                reasoning: "r".into(),
                signature: None,
            })
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            think["choices"][0]["delta"],
            json!({"reasoning_content": "r"})
        );

        let call: Value = serde_json::from_str(
            &w.block(MessageContentBlock::ToolCall {
                id: "c".into(),
                name: "f".into(),
                arguments: json!({"a": 1}),
            })
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            call["choices"][0]["delta"]["tool_calls"],
            json!([{"index": 0, "id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}])
        );
        assert!(w.called_tool());
        let last: Value = serde_json::from_str(&w.finish("tool_calls")).unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(last["choices"][0]["delta"], json!({}));
    }

    #[test]
    fn include_usage_adds_null_usage_to_chunks_and_a_final_usage_chunk() {
        let w = ChunkWriter::new(
            "i".into(),
            "m".into(),
            RequestOptions {
                include_usage: true,
            },
        );
        let first: Value = serde_json::from_str(&w.role()).unwrap();
        assert!(first["usage"].is_null() && first.get("usage").is_some());
        let usage = Usage::new(
            TokenUsage {
                prompt_tokens: 2,
                completion_tokens: 3,
                reasoning_tokens: None,
            },
            None,
        );
        let last: Value = serde_json::from_str(&w.usage(&usage)).unwrap();
        assert_eq!(last["choices"], json!([]));
        assert_eq!(last["usage"]["total_tokens"], 5);
    }

    #[test]
    fn completion_ids_are_prefixed_and_distinct() {
        let (a, b) = (completion_id(), completion_id());
        assert!(a.starts_with("chatcmpl-") && a.len() == "chatcmpl-".len() + 24);
        assert_ne!(a, b);
    }

    #[test]
    fn the_model_list_is_an_openai_list_with_max_model_len() {
        let v = model_list(vec![ModelCard::new("m".into(), 5, 4096)]);
        assert_eq!(
            v,
            json!({"object": "list", "data": [
                {"id": "m", "object": "model", "created": 5, "owned_by": "cuca", "max_model_len": 4096}
            ]})
        );
    }
}
