//! Google Gemini provider adapter.
//!
//! Implements the Gemini `streamGenerateContent` contract behind the
//! `provider-gemini` feature: request bodies are built from the unified
//! message/block model into Gemini's `contents`/`parts` hierarchy, streaming
//! responses are parsed through [`SseStreamParser`] + [`GeminiTranslator`], and
//! `CucaClient::dispatch_gemini` wires the two into `generate_stream`.
//!
//! # Tool-call round-trip
//!
//! Gemini models call tools by emitting `functionCall` parts and receive
//! results as `functionResponse` parts inside a `user`-role turn. The unified
//! model expresses the same flow as `ToolCall` (assistant) and `ToolResult`
//! (tool) blocks, so `Tool`-role messages are translated to `"user"` turns
//! carrying `functionResponse` parts, a documented role-name deviation, not a
//! semantic one. Because Gemini's `functionResponse` names the *function* being
//! answered rather than the call id, a `ToolResult` block (which carries only
//! `tool_call_id`) is matched by `UnifiedMessage::name` when the message sets
//! it, falling back to the call id as a lossy stand-in.
//!
//! # Streaming end
//!
//! Gemini sends no `[DONE]` marker: the final frame carries only
//! `usageMetadata` (which translates to no blocks) and the SSE byte stream
//! simply ends, terminating the stream.
//!
//! # Out of scope
//!
//! Non-streaming `generateContent` batching, and Google auth beyond the
//! `x-goog-api-key` header (OAuth for GCP service accounts is not implemented).

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use tokio_stream::Stream;

use crate::client::{CucaClient, ProviderDispatch, ResponseMetadataHandle};
use crate::error::CucaError;
use crate::request::{ThinkingEffort, ThinkingParams, UnifiedRequest};
use crate::sse::SseStreamParser;
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint};

/// Build the Gemini `streamGenerateContent` request body for a
/// [`UnifiedRequest`].
///
/// Top-level keys:
/// - `system_instruction: { parts: [{ text }] }`: every `Text` block of every
///   `System` message, joined with `\n`; omitted entirely when there is none.
///   Non-text blocks in system messages are dropped (Gemini has no system
///   image/thinking parts).
/// - `generationConfig`: `temperature` and `maxOutputTokens` only when set;
///   the whole key is omitted when neither is set.
/// - `contents`: one entry per non-system message: role `"user"` for
///   `User`/`Tool` messages and `"model"` for `Assistant`, with the parts built
///   by [`block_to_part`].
/// - `tools`: one entry holding a `functionDeclarations` array with every
///   [`UnifiedRequest::tools`] definition as `{name, description, parameters}`,
///   `parameters` holding the definition's `input_schema`; the key is omitted
///   when the request declares no tool, and `toolConfig` is never emitted, so
///   the API's own selection default applies.
/// - `thinkingConfig`: only when `req.thinking` is set: `includeThoughts:
///   false` when disabled; otherwise `thinkingBudget` (params override), else
///   `thinkingLevel` (params override, then the unified-effort map), else
///   `includeThoughts: true`.
pub fn build_generate_content_body(req: &UnifiedRequest) -> Result<serde_json::Value, CucaError> {
    let system_text: String = req
        .messages
        .iter()
        .filter(|m| m.role == MessageRole::System)
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            MessageContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut body = serde_json::Map::new();
    if !system_text.is_empty() {
        body.insert(
            "system_instruction".to_string(),
            serde_json::json!({ "parts": [{ "text": system_text }] }),
        );
    }

    let mut generation = serde_json::Map::new();
    if let Some(temperature) = req.temperature {
        generation.insert("temperature".to_string(), serde_json::json!(temperature));
    }
    if let Some(max_tokens) = req.max_tokens {
        generation.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
    }
    if !generation.is_empty() {
        body.insert(
            "generationConfig".to_string(),
            serde_json::Value::Object(generation),
        );
    }

    if let Some(thinking) = &req.thinking {
        let config = if !thinking.enabled {
            serde_json::json!({ "includeThoughts": false })
        } else {
            match &thinking.params {
                // Params override the unified effort when set; the budget wins
                // over the raw level.
                ThinkingParams::Gemini {
                    thinking_budget: Some(budget),
                    ..
                } => serde_json::json!({ "thinkingBudget": budget }),
                ThinkingParams::Gemini {
                    thinking_level: Some(level),
                    ..
                } => serde_json::json!({ "thinkingLevel": level }),
                // Non-Gemini params variants fall through to the unified path.
                _ => match thinking.effort {
                    Some(effort) => {
                        serde_json::json!({ "thinkingLevel": gemini_level_for(effort) })
                    }
                    None => serde_json::json!({ "includeThoughts": true }),
                },
            }
        };
        body.insert("thinkingConfig".to_string(), config);
    }

    let contents: Vec<serde_json::Value> = req
        .messages
        .iter()
        .filter(|m| m.role != MessageRole::System)
        .map(|m| {
            let role = match m.role {
                MessageRole::User | MessageRole::Tool => "user",
                MessageRole::Assistant => "model",
                // System messages are filtered out above: Gemini has no system
                // role, and their text lives in `system_instruction` instead.
                MessageRole::System => "user",
            };
            let parts: Vec<serde_json::Value> = m
                .content
                .iter()
                .map(|b| block_to_part(b, m.name.as_deref()))
                .collect();
            serde_json::json!({ "role": role, "parts": parts })
        })
        .collect();
    body.insert("contents".to_string(), serde_json::json!(contents));
    if !req.tools.is_empty() {
        let declarations: Vec<serde_json::Value> = req
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                })
            })
            .collect();
        body.insert(
            "tools".to_string(),
            serde_json::json!([{ "functionDeclarations": declarations }]),
        );
    }

    Ok(serde_json::Value::Object(body))
}

/// Unified thinking effort -> Gemini `thinkingLevel` string.
///
/// `Minimal`/`Low` share `LOW` and `High`/`XHigh` share `HIGH`: Gemini exposes
/// no finer-grained levels.
fn gemini_level_for(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Minimal | ThinkingEffort::Low => "LOW",
        ThinkingEffort::Medium => "MEDIUM",
        ThinkingEffort::High | ThinkingEffort::XHigh => "HIGH",
    }
}

/// Translate one unified content block to a Gemini `parts` entry.
///
/// - `Text` -> `{ text }`
/// - `ImageBase64` -> `{ inline_data: { mime_type, data } }`
/// - `Thinking` -> `{ thought: true, text }`, lossy: Gemini has no signature
///   field, so the block's `signature` is dropped.
/// - `ToolCall` -> `{ functionCall: { name, args } }`, Gemini `functionCall`
///   parts carry no call id, so `id` is dropped on the wire.
/// - `ToolResult` -> `{ functionResponse: { name, response: { output } } }`
///   with `name` from `tool_name` (the message's `name` annotation, tool-result
///   messages should set it), falling back to the block's `tool_call_id`.
fn block_to_part(block: &MessageContentBlock, tool_name: Option<&str>) -> serde_json::Value {
    match block {
        MessageContentBlock::Text(text) => serde_json::json!({ "text": text }),
        MessageContentBlock::ImageBase64 { media_type, data } => serde_json::json!({
            "inline_data": { "mime_type": media_type, "data": data },
        }),
        MessageContentBlock::Thinking { reasoning, .. } => serde_json::json!({
            "thought": true,
            "text": reasoning,
        }),
        MessageContentBlock::ToolCall {
            name, arguments, ..
        } => serde_json::json!({
            "functionCall": { "name": name, "args": arguments },
        }),
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => {
            let name = tool_name.unwrap_or(tool_call_id.as_str());
            serde_json::json!({
                "functionResponse": { "name": name, "response": { "output": output } },
            })
        }
    }
}

/// Stateless translator from Gemini-shaped `data:` frames to unified blocks.
///
/// Each frame's `candidates[0].content.parts[]` maps to blocks: `text` ->
/// [`MessageContentBlock::Text`], `thought` -> [`MessageContentBlock::Thinking`]
/// (signature `None`: Gemini does not sign thoughts), and `functionCall` ->
/// [`MessageContentBlock::ToolCall`] with the `args` object used directly as
/// the arguments `Value`. A frame without a usable candidate (e.g. the final
/// frame carrying only `usageMetadata`) yields an empty vec; usage accounting
/// is the client's aggregation, so it is ignored here.
pub struct GeminiTranslator;

impl GeminiTranslator {
    /// Translate one `data:` payload into the blocks its first candidate
    /// carries.
    ///
    /// Malformed JSON yields [`CucaError::Json`]; a payload carrying an `error`
    /// field yields [`CucaError::Provider`] naming
    /// [`ProviderEndpoint::GoogleGemini`].
    pub fn translate(&self, payload: &str) -> Result<Vec<MessageContentBlock>, CucaError> {
        let mut value: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| CucaError::Json {
                message: format!("invalid gemini frame: {e}"),
            })?;

        // Gemini surfaces API errors as JSON frames shaped {"error": {...}}.
        if let Some(message) = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(CucaError::provider(ProviderEndpoint::GoogleGemini, message));
        }

        let Some(candidate) = value
            .get_mut("candidates")
            .and_then(|c| c.as_array_mut())
            .and_then(|c| c.first_mut())
        else {
            return Ok(Vec::new());
        };

        let Some(parts) = candidate
            .get_mut("content")
            .and_then(|content| content.get_mut("parts"))
            .and_then(|p| p.as_array_mut())
        else {
            return Ok(Vec::new());
        };

        let mut blocks = Vec::with_capacity(parts.len());
        // Payload strings and the tool-call argument tree are moved out of the
        // frame instead of copied: `value` is a local that dies at the end of
        // this call, and the text arms run on every streamed part.
        for part in parts.iter_mut() {
            if part.get("thought").and_then(|t| t.as_bool()) == Some(true) {
                // Gemini thought parts carry no signature; the unified block's
                // signature slot stays None (lossy by design).
                blocks.push(MessageContentBlock::Thinking {
                    reasoning: take_string(part, "text").unwrap_or_default(),
                    signature: None,
                });
            } else if let Some(text) = take_string(part, "text") {
                blocks.push(MessageContentBlock::Text(text));
            } else if let Some(call) = part.get_mut("functionCall") {
                let name = take_string(call, "name").unwrap_or_default();
                let id = take_string(call, "id").unwrap_or_default();
                // `args` is already a JSON object; a missing args field
                // translates to null rather than an invented empty object.
                let arguments = call
                    .get_mut("args")
                    .map(serde_json::Value::take)
                    .unwrap_or(serde_json::Value::Null);
                blocks.push(MessageContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }
        Ok(blocks)
    }
}

/// Move the string at `field` out of `value`, leaving `Null` behind.
///
/// `None` when the field is absent or is not a JSON string. Taking the
/// `String` avoids the copy `as_str().to_string()` makes on every streamed
/// part of every frame.
fn take_string(value: &mut serde_json::Value, field: &str) -> Option<String> {
    match value.get_mut(field).map(serde_json::Value::take) {
        Some(serde_json::Value::String(text)) => Some(text),
        _ => None,
    }
}

/// Feed one transport chunk through the SSE parser and translator.
///
/// Pure helper so translation is testable without the network: parses every
/// complete frame in `chunk` and maps each non-empty `data:` payload through
/// [`GeminiTranslator::translate`], appending every block it yields. Frames
/// with empty data are skipped.
pub fn translate_sse(
    parser: &mut SseStreamParser,
    translator: &GeminiTranslator,
    chunk: &[u8],
) -> Result<Vec<MessageContentBlock>, CucaError> {
    let events = parser.feed_chunk(chunk)?;
    let mut blocks = Vec::new();
    for event in events {
        if event.data.is_empty() {
            continue;
        }
        blocks.extend(translator.translate(&event.data)?);
    }
    Ok(blocks)
}

/// Stream a [`UnifiedRequest`] through the Gemini `streamGenerateContent` endpoint.
///
/// POSTs `{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse` (the
/// base is trimmed of a trailing `/`) with the `x-goog-api-key` header, then
/// pipes the SSE response through [`SseStreamParser`] and
/// [`GeminiTranslator`].
///
/// # Errors
///
/// [`CucaError::Http`] on a non-2xx response (body captured); the body
/// builder's errors; transport errors via the usual conversions.
pub async fn gemini_stream(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    req: UnifiedRequest,
) -> Result<ProviderDispatch, CucaError> {
    let url = format!(
        "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
        base_url.trim_end_matches('/'),
        req.model
    );
    let response = http
        .post(&url)
        .header("x-goog-api-key", api_key)
        .json(&build_generate_content_body(&req)?)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await?;
        return Err(CucaError::Http {
            status: status.as_u16(),
            body,
        });
    }
    Ok(ProviderDispatch {
        stream: Box::pin(GeminiStream {
            inner: Box::pin(response.bytes_stream()),
            parser: SseStreamParser::new(),
            translator: GeminiTranslator,
            buffer: VecDeque::new(),
            ended: false,
        }),
        metadata: ResponseMetadataHandle::empty(),
    })
}

/// Stream adapter: reqwest byte stream -> SSE parser -> block translator.
///
/// Yields at most one block per poll; the byte stream ending terminates the
/// stream once the current chunk's blocks are drained.
struct GeminiStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseStreamParser,
    translator: GeminiTranslator,
    /// Blocks awaiting emission within the current chunk.
    buffer: VecDeque<MessageContentBlock>,
    /// True once the byte stream ended; the stream then emits nothing more.
    ended: bool,
}

impl Stream for GeminiStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(block) = this.buffer.pop_front() {
                return Poll::Ready(Some(Ok(block)));
            }
            if this.ended {
                return Poll::Ready(None);
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    match translate_sse(&mut this.parser, &this.translator, &bytes) {
                        Ok(blocks) => {
                            this.buffer.extend(blocks);
                        }
                        Err(e) => {
                            // A malformed frame poisons the stream: report it
                            // once and stop reading further chunks.
                            this.ended = true;
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    this.ended = true;
                    return Poll::Ready(Some(Err(CucaError::Transport {
                        message: e.to_string(),
                    })));
                }
                Poll::Ready(None) => {
                    this.ended = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(feature = "provider-gemini")]
impl CucaClient {
    /// Dispatch a unified request to the Gemini `streamGenerateContent` endpoint.
    ///
    /// Requires the builder's API key (Gemini authenticates every request with
    /// `x-goog-api-key`); uses the client's base URL when set, otherwise the
    /// Google API default (`https://generativelanguage.googleapis.com`). Called
    /// by `generate_stream` under the `provider-gemini` feature.
    pub(crate) async fn dispatch_gemini(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let api_key = self
            .api_key()
            .ok_or_else(|| CucaError::Config("gemini requires an api key".into()))?;
        let base_url = if self.base_url().is_empty() {
            "https://generativelanguage.googleapis.com"
        } else {
            self.base_url()
        };
        gemini_stream(self.http_client(), base_url, api_key, req).await
    }
}

#[cfg(all(test, feature = "provider-gemini"))]
mod tests {
    use std::sync::{Arc, mpsc};

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_stream::StreamExt;

    use super::*;
    use crate::error::PluginError;
    use crate::plugin::CucaPlugin;
    use crate::request::{ThinkingConfig, UnifiedResponse};
    use crate::types::{ToolDefinition, UnifiedMessage};

    #[test]
    fn build_body_maps_system_instruction_and_generation_config() {
        let req = UnifiedRequest::new("gemini-2.0-flash")
            .add_system_message("be concise")
            .add_user_message("hello")
            .set_temperature(0.7)
            .set_max_tokens(256);
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(
            body["system_instruction"],
            json!({ "parts": [{ "text": "be concise" }] })
        );
        assert_eq!(
            body["generationConfig"],
            json!({ "temperature": 0.7_f32, "maxOutputTokens": 256 })
        );
        assert_eq!(
            body["contents"],
            json!([{ "role": "user", "parts": [{ "text": "hello" }] }])
        );
    }

    #[test]
    fn build_body_omits_system_instruction_and_generation_config_when_unset() {
        let req = UnifiedRequest::new("gemini-2.0-flash").add_user_message("hi");
        let body = build_generate_content_body(&req).unwrap();

        assert!(body.get("system_instruction").is_none());
        assert!(body.get("generationConfig").is_none());
        assert_eq!(
            body["contents"],
            json!([{ "role": "user", "parts": [{ "text": "hi" }] }])
        );
    }

    #[test]
    fn build_body_maps_user_text_and_image_parts() {
        let req = UnifiedRequest::new("gemini-2.0-flash").add_message(UnifiedMessage {
            role: MessageRole::User,
            content: vec![
                MessageContentBlock::Text("what is this".into()),
                MessageContentBlock::ImageBase64 {
                    media_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                },
            ],
            name: None,
            tool_call_id: None,
        });
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(
            body["contents"][0],
            json!({
                "role": "user",
                "parts": [
                    { "text": "what is this" },
                    { "inline_data": { "mime_type": "image/png", "data": "aGVsbG8=" } },
                ],
            })
        );
    }

    #[test]
    fn build_body_maps_assistant_thinking_and_tool_call_as_model_role() {
        let req = UnifiedRequest::new("gemini-2.0-flash").add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![
                MessageContentBlock::Thinking {
                    reasoning: "let me think".into(),
                    signature: Some("sig-1".into()),
                },
                MessageContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: json!({ "location": "NYC" }),
                },
            ],
            name: None,
            tool_call_id: None,
        });
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(body["contents"][0]["role"], json!("model"));
        // The Thinking signature is dropped: Gemini has no signature field.
        assert_eq!(
            body["contents"][0]["parts"],
            json!([
                { "thought": true, "text": "let me think" },
                { "functionCall": { "name": "get_weather", "args": { "location": "NYC" } } },
            ])
        );
    }

    #[test]
    fn build_body_maps_tool_result_to_function_response_in_user_turn() {
        let req = UnifiedRequest::new("gemini-2.0-flash").add_message(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "call_1".into(),
                output: "42".into(),
            }],
            name: Some("get_weather".into()),
            tool_call_id: Some("call_1".into()),
        });
        let body = build_generate_content_body(&req).unwrap();

        // Tool responses ride in a "user" turn, named after the function they
        // answer (from the message annotation), not the call id.
        assert_eq!(body["contents"][0]["role"], json!("user"));
        assert_eq!(
            body["contents"][0]["parts"],
            json!([
                { "functionResponse": { "name": "get_weather", "response": { "output": "42" } } }
            ])
        );
    }

    #[test]
    fn build_body_falls_back_to_tool_call_id_for_function_response_name() {
        let req = UnifiedRequest::new("gemini-2.0-flash").add_message(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "call_7".into(),
                output: "ok".into(),
            }],
            name: None,
            tool_call_id: Some("call_7".into()),
        });
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(
            body["contents"][0]["parts"][0],
            json!({ "functionResponse": { "name": "call_7", "response": { "output": "ok" } } })
        );
    }

    #[test]
    fn build_body_maps_tool_definitions_to_function_declarations() {
        let req = UnifiedRequest::new("gemini-2.0-flash")
            .add_user_message("what is the weather in NYC")
            .add_tool(ToolDefinition {
                name: "get_weather".into(),
                description: "Look up the current weather for a city".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                }),
            });
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(
            body["tools"],
            json!([{
                "functionDeclarations": [{
                    "name": "get_weather",
                    "description": "Look up the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"],
                    },
                }],
            }])
        );
        // No toolConfig: the API's own selection default applies.
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    fn build_body_omits_tools_when_none_declared() {
        let req = UnifiedRequest::new("gemini-2.0-flash").add_user_message("hi");
        let body = build_generate_content_body(&req).unwrap();

        assert!(body.get("tools").is_none());
    }

    // --- thinking ---

    #[test]
    fn build_body_thinking_default_is_include_thoughts_true() {
        let req = UnifiedRequest::new("gemini-2.5-flash")
            .add_user_message("hi")
            .enable_thinking(None);
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(body["thinkingConfig"], json!({ "includeThoughts": true }));
    }

    #[test]
    fn build_body_thinking_maps_effort_to_thinking_level() {
        for (effort, expected) in [
            (ThinkingEffort::Minimal, "LOW"),
            (ThinkingEffort::Low, "LOW"),
            (ThinkingEffort::Medium, "MEDIUM"),
            (ThinkingEffort::High, "HIGH"),
            // XHigh shares HIGH: Gemini has no finer-grained level.
            (ThinkingEffort::XHigh, "HIGH"),
        ] {
            let req = UnifiedRequest::new("gemini-2.5-flash")
                .add_user_message("hi")
                .enable_thinking(Some(effort));
            let body = build_generate_content_body(&req).unwrap();

            assert_eq!(body["thinkingConfig"], json!({ "thinkingLevel": expected }));
        }
    }

    #[test]
    fn build_body_thinking_budget_param_wins_over_effort() {
        let req = UnifiedRequest::new("gemini-2.5-flash")
            .add_user_message("hi")
            .with_thinking(ThinkingConfig {
                enabled: true,
                effort: Some(ThinkingEffort::High),
                params: ThinkingParams::Gemini {
                    thinking_budget: Some(20_000),
                    thinking_level: None,
                },
            });
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(body["thinkingConfig"], json!({ "thinkingBudget": 20_000 }));
    }

    #[test]
    fn build_body_thinking_level_param_wins_over_effort() {
        let req = UnifiedRequest::new("gemini-3")
            .add_user_message("hi")
            .with_thinking(ThinkingConfig {
                enabled: true,
                effort: Some(ThinkingEffort::Medium),
                params: ThinkingParams::Gemini {
                    thinking_budget: None,
                    thinking_level: Some("LOW".into()),
                },
            });
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(body["thinkingConfig"], json!({ "thinkingLevel": "LOW" }));
    }

    #[test]
    fn build_body_disabled_thinking_is_include_thoughts_false() {
        let req = UnifiedRequest::new("gemini-2.5-flash")
            .add_user_message("hi")
            .enable_thinking(Some(ThinkingEffort::High))
            .disable_thinking();
        let body = build_generate_content_body(&req).unwrap();

        assert_eq!(body["thinkingConfig"], json!({ "includeThoughts": false }));
    }

    #[test]
    fn build_body_unset_thinking_emits_no_thinking_config() {
        let req = UnifiedRequest::new("gemini-2.5-flash").add_user_message("hi");
        let body = build_generate_content_body(&req).unwrap();

        assert!(body.get("thinkingConfig").is_none());
    }

    #[test]
    fn translate_text_frame_yields_text_block() {
        let translator = GeminiTranslator;
        let blocks = translator
            .translate(r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#)
            .unwrap();
        assert_eq!(blocks, vec![MessageContentBlock::Text("Hello".into())]);
    }

    #[test]
    fn translate_thought_frame_yields_thinking_without_signature() {
        let translator = GeminiTranslator;
        let blocks = translator
            .translate(r#"{"candidates":[{"content":{"parts":[{"thought":true,"text":"let me think"}]}}]}"#)
            .unwrap();
        assert_eq!(
            blocks,
            vec![MessageContentBlock::Thinking {
                reasoning: "let me think".into(),
                signature: None,
            }]
        );
    }

    #[test]
    fn translate_function_call_frame_yields_tool_call_with_parsed_args() {
        let translator = GeminiTranslator;
        let blocks = translator
            .translate(r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC"}}}]}}]}"#)
            .unwrap();
        assert_eq!(
            blocks,
            vec![MessageContentBlock::ToolCall {
                id: "".into(),
                name: "get_weather".into(),
                arguments: json!({ "location": "NYC" }),
            }]
        );
    }

    #[test]
    fn translate_multi_part_frame_preserves_block_order() {
        let translator = GeminiTranslator;
        let blocks = translator
            .translate(r#"{"candidates":[{"content":{"parts":[{"text":"hello"},{"thought":true,"text":"hm"},{"functionCall":{"name":"f","args":{}}},{"text":"bye"}]}}]}"#)
            .unwrap();
        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("hello".into()),
                MessageContentBlock::Thinking {
                    reasoning: "hm".into(),
                    signature: None,
                },
                MessageContentBlock::ToolCall {
                    id: "".into(),
                    name: "f".into(),
                    arguments: json!({}),
                },
                MessageContentBlock::Text("bye".into()),
            ]
        );
    }

    #[test]
    fn translate_usage_only_and_empty_candidate_frames_yield_no_blocks() {
        let translator = GeminiTranslator;
        // The final frame carries only usageMetadata.
        assert!(
            translator
                .translate(r#"{"usageMetadata":{"promptTokenCount":4}}"#)
                .unwrap()
                .is_empty()
        );
        // An empty candidates array is equally content-free.
        assert!(
            translator
                .translate(r#"{"candidates":[]}"#)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn translate_error_payload_is_provider_error() {
        let translator = GeminiTranslator;
        let err = translator
            .translate(r#"{"error":{"code":400,"message":"invalid request"}}"#)
            .unwrap_err();
        match err {
            CucaError::Provider { provider, message } => {
                assert_eq!(provider, ProviderEndpoint::GoogleGemini);
                assert_eq!(message, "invalid request");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[test]
    fn translate_malformed_json_is_json_error() {
        let translator = GeminiTranslator;
        let err = translator.translate("not json").unwrap_err();
        assert!(matches!(err, CucaError::Json { .. }));
    }

    /// Feed every chunk through a fresh parser/translator pair, returning the
    /// emitted blocks in order.
    fn collect_blocks(chunks: &[&[u8]]) -> Vec<MessageContentBlock> {
        let mut parser = SseStreamParser::new();
        let translator = GeminiTranslator;
        let mut blocks = Vec::new();
        for chunk in chunks {
            blocks.extend(translate_sse(&mut parser, &translator, chunk).unwrap());
        }
        blocks
    }

    #[test]
    fn translate_sse_chunk_split_matches_whole_and_bytewise_feeds() {
        let frames = [
            r#"data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#,
            r#"data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"think"}]}}]}"#,
            r#"data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"a":1}}}]}}]}"#,
            r#"data: {"usageMetadata":{"promptTokenCount":3}}"#,
        ];
        let wire: Vec<u8> = frames
            .iter()
            .flat_map(|f| format!("{f}\n\n").into_bytes())
            .collect();

        // Split five bytes into the third frame so both chunk boundaries and
        // intra-frame accumulation are exercised.
        let boundary = frames.iter().take(2).map(|f| f.len() + 2).sum::<usize>() + 5;

        let whole_blocks = collect_blocks(&[&wire]);
        let split_blocks = collect_blocks(&[&wire[..boundary], &wire[boundary..]]);

        // Byte-at-a-time feeding through the pure helper must agree too.
        let mut parser = SseStreamParser::new();
        let translator = GeminiTranslator;
        let mut bytewise_blocks = Vec::new();
        for &b in &wire {
            bytewise_blocks.extend(translate_sse(&mut parser, &translator, &[b]).unwrap());
        }

        assert_eq!(whole_blocks, split_blocks);
        assert_eq!(whole_blocks, bytewise_blocks);
        assert_eq!(
            whole_blocks,
            vec![
                MessageContentBlock::Text("Hi".into()),
                MessageContentBlock::Thinking {
                    reasoning: "think".into(),
                    signature: None,
                },
                MessageContentBlock::ToolCall {
                    id: "".into(),
                    name: "f".into(),
                    arguments: json!({ "a": 1 }),
                },
            ]
        );
    }

    /// Records every `on_response_complete` payload for assertion.
    struct RecordingPlugin {
        /// The completion hook runs synchronously inside `poll_next`; an
        /// unbounded mpsc sender records each response without any locking
        /// (a tokio mutex's blocking_lock would panic inside the runtime).
        tx: mpsc::Sender<UnifiedResponse>,
    }

    impl RecordingPlugin {
        fn new() -> (Self, mpsc::Receiver<UnifiedResponse>) {
            let (tx, rx) = mpsc::channel();
            (Self { tx }, rx)
        }
    }

    impl CucaPlugin for RecordingPlugin {
        fn name(&self) -> &'static str {
            "recording-gemini"
        }

        fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
            self.tx.send(res.clone()).map_err(|_| {
                PluginError::Internal("recording channel closed before completion".into())
            })
        }
    }

    /// Canned Gemini-shaped SSE frames: text part, thought part, then a
    /// usage-only final frame.
    fn canned_frames() -> Vec<&'static str> {
        vec![
            r#"data: {"candidates":[{"content":{"parts":[{"text":"Hi there"}]}}]}"#,
            r#"data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"thinking hard"}]}}]}"#,
            r#"data: {"usageMetadata":{"promptTokenCount":2}}"#,
        ]
    }

    #[tokio::test]
    async fn end_to_end_stream_translates_gemini_sse_and_completes_plugin() {
        // In-process stub: a Gemini-shaped SSE server on an ephemeral port.
        // The request head is captured so the path, auth header, and body can
        // be asserted after the stream completes.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let frames = canned_frames();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let n = socket.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).into_owned();
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in &frames {
                // Each SSE frame needs its terminating blank line inside the
                // chunk body for the parser to complete it.
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
            request
        });

        let (plugin, rx) = RecordingPlugin::new();
        let plugin = Arc::new(plugin);
        // Register through the trait-object type the builder expects; the
        // concrete handle is kept for asserting on the recorded responses.
        let plugin_dyn = Arc::clone(&plugin) as Arc<dyn CucaPlugin>;
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::GoogleGemini)
            .with_base_url(format!("http://{addr}"))
            .with_api_key("test-key")
            .register_plugin(plugin_dyn)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));

        let stream = client
            .generate_stream(UnifiedRequest::new("gemini-2.0-flash").add_user_message("hi"))
            .await
            .unwrap_or_else(|e| panic!("generate_stream must succeed: {e}"));
        let mut blocks = Vec::new();
        let mut stream = stream;
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap_or_else(|e| panic!("stream block must be Ok: {e}")));
        }
        let request = server.await.unwrap();

        // The stub saw the streamGenerateContent path with alt=sse, the
        // x-goog-api-key header, and a contents-shaped JSON body.
        let request_line = request.lines().next().unwrap_or("");
        let path = request_line.split_whitespace().nth(1).unwrap_or("");
        let path_without_query = path.split('?').next().unwrap_or(path);
        assert!(
            path_without_query.ends_with(":streamGenerateContent"),
            "unexpected path: {path}"
        );
        assert!(path.contains("alt=sse"), "missing alt=sse: {path}");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("x-goog-api-key: test-key"),
            "missing x-goog-api-key header: {request}"
        );
        let body_str = request.split("\r\n\r\n").nth(1).unwrap_or("");
        let body: serde_json::Value = serde_json::from_str(body_str)
            .unwrap_or_else(|e| panic!("stub request body is not JSON: {e}"));
        assert_eq!(body["contents"][0]["parts"][0]["text"], json!("hi"));

        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("Hi there".into()),
                MessageContentBlock::Thinking {
                    reasoning: "thinking hard".into(),
                    signature: None,
                },
            ]
        );

        // The completion hook fired exactly once, with the aggregated response.
        let completed: Vec<UnifiedResponse> = rx.try_iter().collect();
        assert_eq!(
            completed.len(),
            1,
            "on_response_complete must fire exactly once"
        );
        assert_eq!(completed[0].content, blocks);
        assert_eq!(completed[0].model, "gemini-2.0-flash");
        assert_eq!(completed[0].provider, ProviderEndpoint::GoogleGemini);
    }
}
