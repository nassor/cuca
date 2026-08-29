//! Shared OpenAI-compatible `/chat/completions` adapter.
//!
//! OpenAI, DeepSeek-native, vLLM, LM Studio, and llama.cpp all expose the same
//! `/v1/chat/completions` contract: identical request JSON, SSE frames shaped
//! as `choices[0].delta.{content,reasoning_content,tool_calls}` and terminated
//! by `data: [DONE]`. They differ only in default base URL and auth header, so
//! one adapter serves all of them; each provider adds a thin `dispatch_*`
//! method that builds an [`OpenAiCompatConfig`].
//!
//! # Tool-call accumulation design
//!
//! [`ChatCompletionTranslator::translate`] emits at most one
//! [`MessageContentBlock`] per call, but one response can carry several
//! interleaved tool calls (accumulated by `index` across frames) plus text and
//! reasoning deltas. The translator therefore keeps:
//!
//! - `tool_acc`: `index -> (id, name, arguments-fragment)` for calls whose
//!   argument fragments are still arriving frame by frame;
//! - `pending`: a FIFO of completed blocks (e.g. tool calls flushed by
//!   `finish_reason` or `[DONE]`) that `translate` drains one block per call.
//!
//! When `finish_reason` or `[DONE]` arrives, every accumulated call is flushed
//! into `pending` in `index` order; subsequent `translate` calls pop them one
//! at a time. [`openai_compat_stream`] additionally drains `pending` directly
//! when the frame stream ends, so flushed tool calls are never lost.

#![cfg(any(
    feature = "provider-openai",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
    feature = "provider-deepseek",
    feature = "provider-llamacpp",
))]

use std::collections::{HashMap, VecDeque, hash_map::Entry};
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use tokio_stream::Stream;

use crate::client::{ProviderDispatch, ResponseMetadataHandle};
use crate::error::CucaError;
use crate::request::{ThinkingEffort, ThinkingParams, UnifiedRequest};
use crate::sse::SseStreamParser;
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};

/// Configuration for one OpenAI-compatible endpoint.
///
/// The union of what every compatible provider needs; the per-provider
/// defaults (base URL, key presence) are chosen by each `dispatch_*` method.
pub struct OpenAiCompatConfig {
    /// Base URL including the API-version suffix, e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    /// Optional bearer token; the `Authorization: Bearer` header is only sent
    /// when this is `Some` (local servers like LM Studio need none).
    pub api_key: Option<String>,
    /// The upstream model identifier.
    pub model: String,
}

/// Build the `/chat/completions` request body for a [`UnifiedRequest`].
///
/// `stream` is always `true`: this adapter only produces streaming responses,
/// and a non-streaming response could not be parsed by the SSE pipeline.
/// `temperature`/`max_tokens` are included only when set.
/// When `req.thinking` is set, DeepSeek receives a top-level `thinking` mode
/// object (`{"type": "enabled"}` / `{"type": "disabled"}`, it has no effort
/// knob), while the other OpenAI-compatible endpoints receive
/// `reasoning_effort`: a `ThinkingParams::OpenAi` override, else the unified
/// effort map, else `"medium"`; a disabled config omits the key there.
/// `CucaClient::generate_stream` overwrites `request.provider` with the
/// client's selected provider before dispatch, so at body-build time
/// `req.provider` is the effective provider.
pub fn build_chat_completion_body(req: &UnifiedRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = req.messages.iter().map(message_to_wire).collect();
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), serde_json::json!(req.model));
    body.insert("messages".to_string(), serde_json::json!(messages));
    if let Some(temperature) = req.temperature {
        body.insert("temperature".to_string(), serde_json::json!(temperature));
    }
    if let Some(max_tokens) = req.max_tokens {
        body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
    }
    body.insert("stream".to_string(), serde_json::Value::Bool(true));
    if let Some(thinking) = &req.thinking {
        if thinking.enabled {
            match req.provider {
                // DeepSeek has no effort knob: the mode is on/off only.
                ProviderEndpoint::DeepSeek => {
                    body.insert(
                        "thinking".to_string(),
                        serde_json::json!({ "type": "enabled" }),
                    );
                }
                _ => {
                    // OpenAI-compatible `reasoning_effort`: a raw params
                    // override wins, else the unified effort map, else the
                    // conventional default.
                    let effort = match &thinking.params {
                        ThinkingParams::OpenAi {
                            reasoning_effort: Some(e),
                        } => e.clone(),
                        _ => thinking
                            .effort
                            .map(reasoning_effort_for)
                            .unwrap_or("medium")
                            .to_string(),
                    };
                    body.insert("reasoning_effort".to_string(), serde_json::json!(effort));
                }
            }
        } else {
            // Explicitly disabled: DeepSeek wants an explicit `disabled` mode;
            // OpenAI-compatible servers get no reasoning_effort key at all.
            if req.provider == ProviderEndpoint::DeepSeek {
                body.insert(
                    "thinking".to_string(),
                    serde_json::json!({ "type": "disabled" }),
                );
            }
        }
    }
    serde_json::Value::Object(body)
}

/// Unified thinking effort -> OpenAI `reasoning_effort` string.
///
/// `XHigh` has no native OpenAI value, so the closest available level
/// (`high`) is used.
fn reasoning_effort_for(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Minimal => "minimal",
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::XHigh => "high",
    }
}

/// Translate one unified message to the wire `{role, content, ...}` shape.
///
/// - `Text` blocks become plain strings (joined with `\n` when several).
/// - A message containing any `ImageBase64` block becomes a content *array* of
///   `{type: "text"|"image_url"}` parts instead of a plain string.
/// - `Thinking` becomes `reasoning_content` on assistant messages carrying
///   exactly one thinking block (OpenAI/DeepSeek); anywhere else it is dropped
///   for strict-compat servers.
/// - `ToolCall` blocks become the assistant `tool_calls` array with the
///   arguments `Value` stringified per the wire contract; an assistant message
///   whose only blocks are tool calls carries `content: null`.
/// - `ToolResult` becomes a `role: "tool"` message with `tool_call_id` and the
///   output as `content`.
fn message_to_wire(msg: &UnifiedMessage) -> serde_json::Value {
    let role = match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };

    let mut wire = serde_json::Map::new();
    wire.insert(
        "role".to_string(),
        serde_json::Value::String(role.to_string()),
    );

    if msg.role == MessageRole::Tool {
        // A tool message answers exactly one call; the wire requires the id.
        // The id lives on the message annotation, with the block's own id as
        // fallback for messages built block-first.
        let tool_call_id = msg
            .tool_call_id
            .clone()
            .or_else(|| match msg.content.first() {
                Some(MessageContentBlock::ToolResult { tool_call_id, .. }) => {
                    Some(tool_call_id.clone())
                }
                _ => None,
            });
        if let Some(id) = tool_call_id {
            wire.insert("tool_call_id".to_string(), serde_json::Value::String(id));
        }
        let output: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        wire.insert("content".to_string(), serde_json::Value::String(output));
        return serde_json::Value::Object(wire);
    }

    let has_tool_calls = msg.role == MessageRole::Assistant
        && msg
            .content
            .iter()
            .any(|b| matches!(b, MessageContentBlock::ToolCall { .. }));

    // Assistant-only: `reasoning_content` for exactly one Thinking block.
    if msg.role == MessageRole::Assistant {
        let thinkings: Vec<&str> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::Thinking { reasoning, .. } => Some(reasoning.as_str()),
                _ => None,
            })
            .collect();
        if thinkings.len() == 1 {
            wire.insert(
                "reasoning_content".to_string(),
                serde_json::Value::String(thinkings[0].to_string()),
            );
        }
    }

    // Assistant-only: `tool_calls` array from ToolCall blocks.
    if has_tool_calls {
        let calls: Vec<serde_json::Value> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    // Stringify: the wire contract requires arguments as a JSON
                    // string, not an object.
                    let args = serde_json::to_string(arguments).unwrap_or_default();
                    Some(serde_json::json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": args },
                    }))
                }
                _ => None,
            })
            .collect();
        wire.insert("tool_calls".to_string(), serde_json::json!(calls));
    }

    // Content: an array of parts when an image is present, else the joined
    // text. An assistant message with only tool calls carries `null`.
    let has_image = msg
        .content
        .iter()
        .any(|b| matches!(b, MessageContentBlock::ImageBase64 { .. }));
    let content = if has_image {
        let parts: Vec<serde_json::Value> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::Text(text) => {
                    Some(serde_json::json!({ "type": "text", "text": text }))
                }
                MessageContentBlock::ImageBase64 { media_type, data } => Some(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{media_type};base64,{data}") },
                })),
                // Thinking/ToolCall/ToolResult have no content-array
                // representation and are dropped for strict compatibility.
                _ => None,
            })
            .collect();
        serde_json::json!(parts)
    } else {
        let text: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if has_tool_calls && text.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::json!(text)
        }
    };
    wire.insert("content".to_string(), content);

    serde_json::Value::Object(wire)
}

/// Stateful translator from OpenAI-shaped `data:` payloads to unified blocks.
///
/// Holds tool-call accumulation across frames (see the [module docs](crate::provider::openai_compat))
/// so argument fragments split across multiple SSE frames are reassembled
/// before the call is emitted.
pub struct ChatCompletionTranslator {
    /// `delta.tool_calls[index] -> (id, name, arguments-fragment)`.
    tool_acc: HashMap<usize, (String, String, String)>,
    /// Completed blocks awaiting emission (tool calls flushed at
    /// `finish_reason`/`[DONE]`), drained one per `translate` call.
    pending: VecDeque<MessageContentBlock>,
    /// Set once `data: [DONE]` was seen; the frame stream is over.
    done: bool,
}

impl ChatCompletionTranslator {
    /// Start a translator with no accumulated state.
    pub fn new() -> Self {
        Self {
            tool_acc: HashMap::new(),
            pending: VecDeque::new(),
            done: false,
        }
    }

    /// Translate one `data:` payload into at most one block.
    ///
    /// `[DONE]` flushes every accumulated tool call into the pending queue (in
    /// `index` order), marks the stream finished, and returns `None`; the
    /// flushed calls are drained one per subsequent call. Text deltas become
    /// [`MessageContentBlock::Text`], `reasoning_content` becomes
    /// [`MessageContentBlock::Thinking`], and a tool call is emitted once its
    /// argument fragment accumulates to valid JSON, or at `finish_reason`/`[DONE]`, whichever comes first. OpenAI error
    /// bodies
    /// (`{"error":{"message":...}}`) yield [`CucaError::Provider`].
    pub fn translate(&mut self, payload: &str) -> Result<Option<MessageContentBlock>, CucaError> {
        // [DONE] terminates the stream; check it before draining pending so the
        // flushed tool calls surface on subsequent calls and the stream wrapper
        // can observe completion via `done`.
        if payload == "[DONE]" {
            self.flush_accumulated();
            self.done = true;
            return Ok(None);
        }
        // Drain pending first: at most one block per call, in completion order.
        if let Some(block) = self.pending.pop_front() {
            return Ok(Some(block));
        }

        let mut value: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| CucaError::Json {
                message: format!("invalid chat completion frame: {e}"),
            })?;

        // OpenAI error bodies are JSON, not frames: {"error": {"message": ...}}.
        if let Some(message) = value
            .get("error")
            .and_then(|err| err.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(CucaError::provider(ProviderEndpoint::OpenAi, message));
        }

        let Some(choice) = value
            .get_mut("choices")
            .and_then(|c| c.as_array_mut())
            .and_then(|c| c.first_mut())
        else {
            return Ok(None);
        };

        if let Some(delta) = choice.get_mut("delta") {
            // `take` moves each payload string out of the frame rather than
            // copying it: `value` is a local that dies at the end of this
            // call, and these two arms run on every streamed token.
            if let Some(serde_json::Value::String(text)) =
                delta.get_mut("content").map(serde_json::Value::take)
                && !text.is_empty()
            {
                return Ok(Some(MessageContentBlock::Text(text)));
            }
            if let Some(serde_json::Value::String(reasoning)) = delta
                .get_mut("reasoning_content")
                .map(serde_json::Value::take)
                && !reasoning.is_empty()
            {
                return Ok(Some(MessageContentBlock::Thinking {
                    reasoning,
                    signature: None,
                }));
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for entry in tool_calls {
                    if let Some(block) = self.accumulate_tool_call(entry) {
                        return Ok(Some(block));
                    }
                }
            }
        }

        if choice.get("finish_reason").is_some() {
            self.flush_accumulated();
        }
        Ok(None)
    }

    /// Fold one `delta.tool_calls` entry into the accumulator; returns the
    /// completed call when its arguments just finished accumulating.
    fn accumulate_tool_call(&mut self, entry: &serde_json::Value) -> Option<MessageContentBlock> {
        let index = entry
            .get("index")
            .and_then(|i| i.as_u64())
            .map(|i| i as usize)
            .unwrap_or(0);
        let id = entry.get("id").and_then(|i| i.as_str()).map(str::to_string);
        let function = entry.get("function");
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(str::to_string);
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
            .map(str::to_string);

        match self.tool_acc.entry(index) {
            Entry::Occupied(mut slot) => match arguments {
                // A non-empty fragment continues the accumulation.
                Some(fragment) if !fragment.is_empty() => {
                    slot.get_mut().2.push_str(&fragment);
                    None
                }
                // Empty or missing arguments after accumulation started: the
                // server signalled the call is complete. Complete it only when
                // the accumulated fragment is valid JSON; otherwise keep
                // accumulating (the fragment is still mid-flight).
                _ => self.try_complete(index),
            },
            Entry::Vacant(slot) => {
                // First frame for this index: seed the accumulator, which
                // requires id + name. The initial arguments (usually "") start
                // the fragment.
                if let (Some(id), Some(name)) = (id, name) {
                    slot.insert((id, name, arguments.unwrap_or_default()));
                }
                None
            }
        }
    }

    /// Emit the accumulated call at `index` once its fragment parses as JSON.
    fn try_complete(&mut self, index: usize) -> Option<MessageContentBlock> {
        let fragment = self.tool_acc.get(&index)?.2.as_str();
        if fragment.is_empty() {
            return None;
        }
        let arguments = serde_json::from_str::<serde_json::Value>(fragment).ok()?;
        let (id, name, _) = self.tool_acc.remove(&index)?;
        Some(MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        })
    }

    /// Move every accumulated tool call into `pending`, in `index` order.
    ///
    /// Called at `finish_reason` and `[DONE]`. Fragments that do not parse as
    /// JSON (malformed server output) are emitted with `null` arguments rather
    /// than dropped, so the call id/name are never lost.
    fn flush_accumulated(&mut self) {
        if self.tool_acc.is_empty() {
            return;
        }
        let mut indices: Vec<usize> = self.tool_acc.keys().copied().collect();
        indices.sort_unstable();
        for index in indices {
            if let Some((id, name, fragment)) = self.tool_acc.remove(&index) {
                let arguments = match serde_json::from_str::<serde_json::Value>(&fragment) {
                    Ok(value) => value,
                    Err(_) => serde_json::Value::Null,
                };
                self.pending.push_back(MessageContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }
    }
}

impl Default for ChatCompletionTranslator {
    fn default() -> Self {
        Self::new()
    }
}

/// Feed one transport chunk through the SSE parser and translator.
///
/// Pure helper so translation is testable without the network: parses every
/// complete frame in `chunk` and maps each non-empty `data:` payload through
/// [`ChatCompletionTranslator::translate`]. Frames with empty data are skipped.
pub fn translate_sse(
    parser: &mut SseStreamParser,
    translator: &mut ChatCompletionTranslator,
    chunk: &[u8],
) -> Result<Vec<Option<MessageContentBlock>>, CucaError> {
    let events = parser.feed_chunk(chunk)?;
    let mut blocks = Vec::with_capacity(events.len());
    for event in events {
        if event.data.is_empty() {
            continue;
        }
        blocks.push(translator.translate(&event.data)?);
    }
    Ok(blocks)
}

/// Stream a [`UnifiedRequest`] through an OpenAI-compatible endpoint.
///
/// POSTs `{base_url}/chat/completions` (the base URL already carries the `/v1`
/// suffix) with a bearer token when configured, then pipes the SSE response
/// through [`SseStreamParser`] and [`ChatCompletionTranslator`]. Non-2xx
/// responses surface as [`CucaError::Http`] with the captured body.
pub async fn openai_compat_stream(
    http: &reqwest::Client,
    config: &OpenAiCompatConfig,
    req: UnifiedRequest,
) -> Result<ProviderDispatch, CucaError> {
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    // The config carries the effective model: each dispatch fills it from the
    // request (or a provider default), and the body is built from that single
    // source of truth.
    let mut req = req;
    req.model = config.model.clone();
    let mut request = http.post(&url).json(&build_chat_completion_body(&req));
    if let Some(api_key) = &config.api_key {
        request = request.bearer_auth(api_key);
    }
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await?;
        return Err(CucaError::Http {
            status: status.as_u16(),
            body,
        });
    }
    Ok(ProviderDispatch {
        stream: Box::pin(OpenAiCompatStream {
            inner: Box::pin(response.bytes_stream()),
            parser: SseStreamParser::new(),
            translator: ChatCompletionTranslator::new(),
            buffer: VecDeque::new(),
            ended: false,
        }),
        metadata: ResponseMetadataHandle::empty(),
    })
}

/// Stream adapter: reqwest byte stream -> SSE parser -> block translator.
///
/// Yields at most one block per poll; `data: [DONE]` (or the byte stream
/// ending) terminates the stream after any tool calls flushed into the
/// translator's pending queue are emitted.
struct OpenAiCompatStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseStreamParser,
    translator: ChatCompletionTranslator,
    /// Blocks awaiting emission within the current chunk.
    buffer: VecDeque<MessageContentBlock>,
    /// True once `[DONE]` was seen or the byte stream ended; the stream then
    /// emits only what is left in `buffer`/`pending`.
    ended: bool,
}

impl Stream for OpenAiCompatStream {
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
                    match translate_sse(&mut this.parser, &mut this.translator, &bytes) {
                        Ok(blocks) => {
                            for block in blocks.into_iter().flatten() {
                                this.buffer.push_back(block);
                            }
                            if this.translator.done {
                                // [DONE] ended the frame stream; emit any tool
                                // calls it flushed into pending, then stop.
                                this.ended = true;
                                this.buffer.extend(this.translator.pending.drain(..));
                            }
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
                    // Byte stream ended without [DONE]: emit whatever was
                    // flushed (e.g. by finish_reason), then end.
                    this.ended = true;
                    this.buffer.extend(this.translator.pending.drain(..));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(all(test, feature = "provider-openai"))]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::request::ThinkingConfig;

    #[test]
    fn build_body_maps_text_system_and_user_messages() {
        let req = UnifiedRequest::new("gpt-4o")
            .add_system_message("be concise")
            .add_user_message("hello")
            .set_temperature(0.7)
            .set_max_tokens(128);
        let body = build_chat_completion_body(&req);

        assert_eq!(body["model"], json!("gpt-4o"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["temperature"], json!(0.7_f32));
        assert_eq!(body["max_tokens"], json!(128));
        assert_eq!(
            body["messages"],
            json!([
                { "role": "system", "content": "be concise" },
                { "role": "user", "content": "hello" },
            ])
        );
    }

    #[test]
    fn build_body_omits_unset_knobs_and_always_streams() {
        let req = UnifiedRequest::new("gpt-4o").add_user_message("hi");
        let body = build_chat_completion_body(&req);

        assert!(body.get("temperature").is_none());
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn build_body_maps_image_block_to_data_uri() {
        let req = UnifiedRequest::new("gpt-4o").add_message(UnifiedMessage {
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
        let body = build_chat_completion_body(&req);

        assert_eq!(
            body["messages"][0]["content"],
            json!([
                { "type": "text", "text": "what is this" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,aGVsbG8=" } },
            ])
        );
    }

    #[test]
    fn build_body_maps_assistant_tool_calls() {
        let req = UnifiedRequest::new("gpt-4o").add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: json!({ "q": "cuca", "limit": 5 }),
            }],
            name: None,
            tool_call_id: None,
        });
        let body = build_chat_completion_body(&req);

        // No text accompanies the call, so content is null.
        assert_eq!(body["messages"][0]["content"], json!(null));
        // Arguments are stringified per the wire contract; the expected string
        // is produced through the same serde path so key ordering matches.
        let expected_args = serde_json::to_string(&json!({ "q": "cuca", "limit": 5 })).unwrap();
        assert_eq!(
            body["messages"][0]["tool_calls"],
            json!([{
                "id": "call_1",
                "type": "function",
                "function": { "name": "search", "arguments": expected_args },
            }])
        );
    }

    #[test]
    fn build_body_maps_tool_result_message() {
        let req = UnifiedRequest::new("gpt-4o").add_message(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "call_1".into(),
                output: "42".into(),
            }],
            name: None,
            tool_call_id: Some("call_1".into()),
        });
        let body = build_chat_completion_body(&req);

        assert_eq!(
            body["messages"][0],
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "42" })
        );
    }

    #[test]
    fn build_body_maps_single_assistant_thinking_to_reasoning_content() {
        let req = UnifiedRequest::new("deepseek-reasoner").add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::Thinking {
                reasoning: "step by step".into(),
                signature: None,
            }],
            name: None,
            tool_call_id: None,
        });
        let body = build_chat_completion_body(&req);

        assert_eq!(
            body["messages"][0]["reasoning_content"],
            json!("step by step")
        );
        // The thinking block has no content-array representation; the joined
        // text is empty.
        assert_eq!(body["messages"][0]["content"], json!(""));
    }

    #[test]
    fn build_body_drops_thinking_without_single_assistant_block() {
        // Two thinking blocks: not the single-block shape, so both are dropped.
        let req = UnifiedRequest::new("gpt-4o").add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![
                MessageContentBlock::Thinking {
                    reasoning: "first".into(),
                    signature: None,
                },
                MessageContentBlock::Thinking {
                    reasoning: "second".into(),
                    signature: None,
                },
            ],
            name: None,
            tool_call_id: None,
        });
        let body = build_chat_completion_body(&req);

        assert!(body["messages"][0].get("reasoning_content").is_none());
    }

    // --- thinking ---

    #[test]
    fn build_body_maps_each_effort_level_to_reasoning_effort() {
        for (effort, expected) in [
            (ThinkingEffort::Minimal, "minimal"),
            (ThinkingEffort::Low, "low"),
            (ThinkingEffort::Medium, "medium"),
            (ThinkingEffort::High, "high"),
            // XHigh has no native value; the closest available level is used.
            (ThinkingEffort::XHigh, "high"),
        ] {
            let req = UnifiedRequest::new("gpt-5")
                .add_user_message("hi")
                .enable_thinking(Some(effort));
            let body = build_chat_completion_body(&req);

            assert_eq!(body["reasoning_effort"], json!(expected));
            assert!(body.get("thinking").is_none());
        }
    }

    #[test]
    fn build_body_defaults_reasoning_effort_to_medium_when_effort_unset() {
        let req = UnifiedRequest::new("gpt-5")
            .add_user_message("hi")
            .enable_thinking(None);
        let body = build_chat_completion_body(&req);

        assert_eq!(body["reasoning_effort"], json!("medium"));
    }

    #[test]
    fn build_body_params_reasoning_effort_override_wins_over_effort_map() {
        let req = UnifiedRequest::new("gpt-5")
            .add_user_message("hi")
            .with_thinking(ThinkingConfig {
                enabled: true,
                effort: Some(ThinkingEffort::Low),
                params: ThinkingParams::OpenAi {
                    reasoning_effort: Some("high".into()),
                },
            });
        let body = build_chat_completion_body(&req);

        assert_eq!(body["reasoning_effort"], json!("high"));
    }

    #[test]
    fn build_body_disabled_thinking_omits_reasoning_effort() {
        let req = UnifiedRequest::new("gpt-5")
            .add_user_message("hi")
            .enable_thinking(Some(ThinkingEffort::High))
            .disable_thinking();
        let body = build_chat_completion_body(&req);

        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn build_body_omits_thinking_keys_when_unset() {
        let req = UnifiedRequest::new("gpt-5").add_user_message("hi");
        let body = build_chat_completion_body(&req);

        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn build_body_deepseek_uses_thinking_mode_object() {
        let mut req = UnifiedRequest::new("deepseek-reasoner")
            .add_user_message("hi")
            .enable_thinking(Some(ThinkingEffort::High));
        req.provider = ProviderEndpoint::DeepSeek;
        let body = build_chat_completion_body(&req);

        assert_eq!(body["thinking"], json!({ "type": "enabled" }));
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn build_body_deepseek_disabled_thinking_emits_disabled_mode() {
        let mut req = UnifiedRequest::new("deepseek-reasoner")
            .add_user_message("hi")
            .disable_thinking();
        req.provider = ProviderEndpoint::DeepSeek;
        let body = build_chat_completion_body(&req);

        assert_eq!(body["thinking"], json!({ "type": "disabled" }));
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn translate_text_delta() {
        let mut translator = ChatCompletionTranslator::new();
        let block = translator
            .translate(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#)
            .unwrap();
        assert_eq!(block, Some(MessageContentBlock::Text("Hello".into())));
    }

    #[test]
    fn translate_reasoning_content_to_thinking_without_signature() {
        let mut translator = ChatCompletionTranslator::new();
        let block = translator
            .translate(r#"{"choices":[{"delta":{"reasoning_content":"let me think"}}]}"#)
            .unwrap();
        assert_eq!(
            block,
            Some(MessageContentBlock::Thinking {
                reasoning: "let me think".into(),
                signature: None,
            })
        );
    }

    #[test]
    fn translate_accumulates_tool_call_arguments_across_frames() {
        let mut translator = ChatCompletionTranslator::new();

        // First frame: id + name seed the accumulator.
        assert_eq!(
            translator
                .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#)
                .unwrap(),
            None
        );
        // Argument fragments concatenate across frames.
        assert_eq!(
            translator
                .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\""}}]}}]}"#)
                .unwrap(),
            None
        );
        assert_eq!(
            translator
                .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"NYC\"}"}}]}}]}"#)
                .unwrap(),
            None
        );
        // finish_reason flushes the accumulated call into pending.
        assert_eq!(
            translator
                .translate(r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#)
                .unwrap(),
            None
        );
        // The next call drains it, one block at a time.
        assert_eq!(
            translator.translate("{}").unwrap(),
            Some(MessageContentBlock::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: json!({ "location": "NYC" }),
            })
        );
        assert_eq!(translator.translate("{}").unwrap(), None);
    }

    #[test]
    fn translate_done_flushes_accumulated_and_returns_none() {
        let mut translator = ChatCompletionTranslator::new();
        translator
            .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]}}]}"#)
            .unwrap();
        translator
            .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#)
            .unwrap();

        // [DONE] flushes and returns None...
        assert_eq!(translator.translate("[DONE]").unwrap(), None);
        assert!(translator.done);
        // ...and the flushed call is drained one translate call at a time.
        assert_eq!(
            translator.translate("{}").unwrap(),
            Some(MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "f".into(),
                arguments: json!({}),
            })
        );
        assert_eq!(translator.translate("{}").unwrap(), None);
    }

    #[test]
    fn translate_completes_when_frame_omits_arguments() {
        let mut translator = ChatCompletionTranslator::new();
        translator
            .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]}}]}"#)
            .unwrap();
        translator
            .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]}}]}"#)
            .unwrap();

        // A frame whose entry has no arguments field completes the call once
        // the accumulated fragment is valid JSON.
        let block = translator
            .translate(r#"{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}"#)
            .unwrap();
        assert_eq!(
            block,
            Some(MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "f".into(),
                arguments: json!({ "x": 1 }),
            })
        );
        assert_eq!(translator.translate("{}").unwrap(), None);
    }

    #[test]
    fn translate_two_parallel_tool_calls_flush_in_index_order() {
        let mut translator = ChatCompletionTranslator::new();

        // Both calls seed first, then fragments interleave by index.
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"a","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"b","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"y\":2}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ] {
            translator.translate(payload).unwrap();
        }

        // Flushed in index order, one per call.
        assert_eq!(
            translator.translate("{}").unwrap(),
            Some(MessageContentBlock::ToolCall {
                id: "call_a".into(),
                name: "a".into(),
                arguments: json!({ "x": 1 }),
            })
        );
        assert_eq!(
            translator.translate("{}").unwrap(),
            Some(MessageContentBlock::ToolCall {
                id: "call_b".into(),
                name: "b".into(),
                arguments: json!({ "y": 2 }),
            })
        );
        assert_eq!(translator.translate("{}").unwrap(), None);
    }

    #[test]
    fn translate_error_payload_is_provider_error() {
        let mut translator = ChatCompletionTranslator::new();
        let err = translator
            .translate(r#"{"error":{"message":"rate limited"}}"#)
            .unwrap_err();
        match err {
            CucaError::Provider { provider, message } => {
                assert_eq!(provider, ProviderEndpoint::OpenAi);
                assert_eq!(message, "rate limited");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    /// Feed every frame in `chunks` through a fresh parser/translator pair and
    /// return the emitted blocks, draining the pending queue afterwards.
    fn collect_blocks(chunks: &[&[u8]]) -> Vec<MessageContentBlock> {
        let mut parser = SseStreamParser::new();
        let mut translator = ChatCompletionTranslator::new();
        let mut blocks = Vec::new();
        for chunk in chunks {
            for block in translate_sse(&mut parser, &mut translator, chunk)
                .unwrap()
                .into_iter()
                .flatten()
            {
                blocks.push(block);
            }
        }
        // Drain whatever finish_reason/[DONE] flushed into the pending queue.
        while let Some(block) = translator.translate("{}").unwrap() {
            blocks.push(block);
        }
        blocks
    }

    #[test]
    fn translate_sse_chunk_split_matches_whole_chunk() {
        let frames = [
            r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: [DONE]"#,
        ];
        let wire: Vec<u8> = frames
            .iter()
            .flat_map(|f| format!("{f}\n\n").into_bytes())
            .collect();

        // Split mid-frame (five bytes into the third frame) so both chunk
        // boundaries and frame accumulation are exercised.
        let boundary = frames.iter().take(2).map(|f| f.len() + 2).sum::<usize>() + 5;

        let whole_blocks = collect_blocks(&[&wire]);
        let split_blocks = collect_blocks(&[&wire[..boundary], &wire[boundary..]]);

        assert_eq!(whole_blocks, split_blocks);
        assert_eq!(
            whole_blocks,
            vec![
                MessageContentBlock::Text("Hi".into()),
                MessageContentBlock::Thinking {
                    reasoning: "think".into(),
                    signature: None,
                },
                MessageContentBlock::ToolCall {
                    id: "c1".into(),
                    name: "f".into(),
                    arguments: json!({ "a": 1 }),
                },
            ]
        );
    }
}
