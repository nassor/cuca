//! Shared Anthropic Messages-API protocol module.
//!
//! Implements the Anthropic Claude adapter's protocol layer behind the
//! `any(provider-anthropic, provider-deepseek)` gate so the DeepSeek
//! bridge can reuse the same translation pieces: auth + request headers
//! ([`headers`]), OAuth 2.0 PKCE helpers (gated to `provider-anthropic`, which
//! pulls in the optional `sha2`/`getrandom`/`base64` deps), Messages-API body
//! translation ([`build_anthropic_request`]), and SSE frame parsing
//! ([`AnthropicTranslator`]). The `provider-anthropic` dispatch
//! (`CucaClient::dispatch_anthropic`) lives here too.
//!
//! # Block-sequential translation design
//!
//! Anthropic emits content blocks strictly sequentially: a block's
//! `content_block_delta` events always follow its `content_block_start` and
//! precede its `content_block_stop`, so the `index` carried on delta events is
//! never needed. [`AnthropicTranslator`] tracks at most one block in flight
//! with plain `Option` accumulators (one each for the tool-call input
//! fragment, the thinking text, and the thinking signature).

#![cfg(any(feature = "provider-anthropic", feature = "provider-deepseek",))]

use std::collections::{HashSet, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(feature = "provider-anthropic")]
use base64::Engine;
#[cfg(feature = "provider-anthropic")]
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use bytes::Bytes;
#[cfg(feature = "provider-anthropic")]
use getrandom::fill;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
#[cfg(feature = "provider-anthropic")]
use sha2::{Digest, Sha256};
use tokio_stream::Stream;

#[cfg(feature = "provider-anthropic")]
use crate::client::CucaClient;
use crate::client::{ProviderDispatch, ResponseMetadataHandle};
use crate::error::CucaError;
use crate::request::{
    PromptCacheBreakpoint, PromptCacheDirective, PromptCacheUsage, ThinkingEffort, ThinkingParams,
    UnifiedRequest,
};
use crate::sse::{SseEvent, SseStreamParser};
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint};

/// How a request authenticates to Anthropic.
///
/// Exactly one of the two modes is sent: [`Self::ApiKey`] as the `x-api-key`
/// header, [`Self::Bearer`] (an OAuth PKCE-issued access token) as
/// `Authorization: Bearer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicAuth {
    /// Static API key; sent as `x-api-key`.
    ApiKey(String),
    /// OAuth PKCE-issued access token; sent as `Authorization: Bearer`.
    // Constructed only by `provider-anthropic`-gated code (dispatch_anthropic
    // and its tests), so dead_code fires under `provider-deepseek` alone,
    // which compiles this module for the Anthropic bridge. The expectation is
    // cfg-scoped because it would be unfulfilled (and warn) under
    // `provider-anthropic`.
    #[cfg_attr(
        not(feature = "provider-anthropic"),
        expect(
            dead_code,
            reason = "only provider-anthropic constructs Bearer; the deepseek bridge compiles the enum without it"
        )
    )]
    Bearer(String),
}

/// Build the Anthropic request headers for one auth mode.
///
/// Exactly one of `x-api-key` (static key) or `Authorization: Bearer` (OAuth
/// PKCE-issued token) is sent, plus `anthropic-version: 2023-06-01` and one
/// `anthropic-beta` header per entry in `betas` (e.g. `claude-code-20250219`,
/// `prompt-caching-2024-07-31`).
///
/// Deviation: returns `Result<HeaderMap, CucaError>` instead of a bare
/// `HeaderMap`: dynamic values (key, token, betas) may be invalid HTTP header
/// values, and this crate's non-test code never panics, so they surface as
/// [`CucaError::Config`] instead of being dropped or panicking.
pub fn headers(auth: &AnthropicAuth, betas: &[&str]) -> Result<HeaderMap, CucaError> {
    let mut map = HeaderMap::new();
    map.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    match auth {
        AnthropicAuth::ApiKey(key) => {
            let value = HeaderValue::from_bytes(key.as_bytes()).map_err(|_| {
                CucaError::Config("anthropic api key is not a valid header value".into())
            })?;
            map.insert(HeaderName::from_static("x-api-key"), value);
        }
        AnthropicAuth::Bearer(token) => {
            let value =
                HeaderValue::from_bytes(format!("Bearer {token}").as_bytes()).map_err(|_| {
                    CucaError::Config("anthropic bearer token is not a valid header value".into())
                })?;
            map.insert(AUTHORIZATION, value);
        }
    }
    for beta in betas {
        let value = HeaderValue::from_bytes(beta.as_bytes()).map_err(|_| {
            CucaError::Config(format!("invalid anthropic-beta header value: {beta}"))
        })?;
        map.append(HeaderName::from_static("anthropic-beta"), value);
    }
    Ok(map)
}

// --- OAuth 2.0 PKCE (needs the provider-anthropic-only sha2/getrandom/base64 deps) ---

/// Anthropic OAuth 2.0 PKCE application configuration.
///
/// The client id, authorization/token endpoints, and requested scopes for the
/// authorization-code flow with PKCE (RFC 7636).
#[cfg(feature = "provider-anthropic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthPkceConfig {
    /// The registered OAuth client id.
    pub client_id: String,
    /// The authorization endpoint URL.
    pub authorize_url: String,
    /// The token endpoint URL.
    pub token_url: String,
    /// Scopes requested in the authorization request.
    pub scopes: Vec<String>,
}

/// A PKCE code-verifier/challenge pair.
///
/// The verifier is sent in the token exchange; the challenge is sent in the
/// authorization URL.
// OAuth helpers are exercised by the test suite; non-test builds have no
// caller yet, so the expectation is `not(test)`-scoped: under `cfg(test)` the
// tests construct the pair and an unscoped `#[expect]` would be unfulfilled.
#[cfg(feature = "provider-anthropic")]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "PKCE helper type: only the test suite constructs it so far"
    )
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PkcePair {
    /// The code verifier (base64url, no padding).
    pub code_verifier: String,
    /// The S256 challenge: base64url(SHA-256(verifier)).
    pub code_challenge: String,
}

/// base64url(SHA-256(verifier)): the RFC 7636 S256 challenge derivation.
#[cfg(feature = "provider-anthropic")]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "PKCE helper: only the test suite calls it so far")
)]
fn pkce_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let digest = hasher.finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

/// Generate a fresh PKCE pair.
///
/// The verifier is 64 random bytes (from `getrandom::fill`) base64url-encoded
/// without padding (86 chars, within the RFC 7636 43 to 128 range); the challenge
/// is base64url(SHA-256(verifier)).
///
/// Deviation: the result is `Result<PkcePair, CucaError>`; `getrandom` can
/// fail (no OS entropy source), and this crate's non-test code never panics,
/// so the failure surfaces as [`CucaError::Io`].
#[cfg(feature = "provider-anthropic")]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "PKCE helper: only the test suite calls it so far")
)]
pub fn generate_pkce_pair() -> Result<PkcePair, CucaError> {
    let mut bytes = [0u8; 64];
    fill(&mut bytes).map_err(|e| CucaError::Io {
        message: format!("system randomness unavailable for PKCE verifier: {e}"),
    })?;
    let code_verifier = URL_SAFE_NO_PAD.encode(bytes);
    let code_challenge = pkce_challenge(&code_verifier);
    Ok(PkcePair {
        code_verifier,
        code_challenge,
    })
}

/// Exchange an authorization code for an access token (PKCE token request).
///
/// POSTs `grant_type=authorization_code`, `code`, `code_verifier`,
/// `client_id`, and `redirect_uri` to the token endpoint and returns the
/// `access_token` from the JSON response.
///
/// # Errors
///
/// [`CucaError::Http`] on a non-2xx token response (body captured);
/// [`CucaError::Provider`] when the response carries no `access_token`;
/// transport errors via the usual conversions.
#[cfg(feature = "provider-anthropic")]
// `exchange_code` has no caller in either build profile: dispatch uses the
// static api key, and no test drives a live token endpoint. dead_code
// therefore fires everywhere and the expectation needs no cfg scope.
#[expect(
    dead_code,
    reason = "PKCE token exchange: no dispatch or test caller yet"
)]
pub async fn exchange_code(
    cfg: &OAuthPkceConfig,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<String, CucaError> {
    let params = [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("code_verifier", verifier),
        ("client_id", cfg.client_id.as_str()),
        ("redirect_uri", redirect_uri),
    ];
    let response = reqwest::Client::new()
        .post(&cfg.token_url)
        .form(&params)
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
    let value: serde_json::Value = response.json().await?;
    value
        .get("access_token")
        .and_then(|t| t.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            CucaError::provider(
                ProviderEndpoint::Anthropic,
                "token response missing access_token",
            )
        })
}

/// Build the authorization URL for the PKCE code flow.
///
/// Query parameters: `response_type=code`, `client_id`, `code_challenge`,
/// `code_challenge_method=S256`, `state`, and (when non-empty) `scope` with the
/// scopes joined by `+`. Values are expected to be URL-safe (client ids are
/// plain tokens, the challenge is base64url, state is caller-chosen).
#[cfg(feature = "provider-anthropic")]
#[cfg_attr(not(test), allow(dead_code))]
pub fn authorization_url(cfg: &OAuthPkceConfig, challenge: &str, state: &str) -> String {
    let mut url = format!(
        "{}?response_type=code&client_id={}&code_challenge={}&code_challenge_method=S256&state={}",
        cfg.authorize_url, cfg.client_id, challenge, state
    );
    if !cfg.scopes.is_empty() {
        url.push_str(&format!("&scope={}", cfg.scopes.join("+")));
    }
    url
}

/// Build the Anthropic Messages-API request body for a [`UnifiedRequest`].
///
/// - `max_tokens` is REQUIRED by the API and defaults to 1024 when unset.
/// - `stream` is always `true`: this adapter only produces streaming
///   responses, and a non-streaming response could not be parsed by the SSE
///   pipeline.
/// - System-message text blocks are joined (with `\n`) into the top-level
///   `system` string; non-text blocks in system messages are dropped, and no
///   `system` key is emitted when there are no system messages.
/// - `messages` carries user/assistant messages only. Tool results ride inside
///   a user message as `tool_result` blocks per the Anthropic wire contract,
///   so standalone `Tool`-role messages are skipped.
///
/// When `req.thinking` is set, a top-level `thinking` key is emitted: adaptive
/// mode (`{"type": "adaptive"}` plus `effort` when a unified effort is set)
/// for `ThinkingParams::Anthropic { adaptive: true, .. }`, otherwise budget
/// mode (`{"type": "enabled", "budget_tokens": N}` with the params override or
/// the unified-effort budget map). A disabled config emits no `thinking` key.
///
/// Deviation: auth headers are the [`headers`] function's concern, not the
/// body builder's, so `build_anthropic_request` takes only the request.
pub fn build_anthropic_request(req: &UnifiedRequest) -> Result<serde_json::Value, CucaError> {
    let marks = cache_marks(req)?;
    // `(text, cache-marked)` per system text block, in unified order.
    let mut system: Vec<(String, bool)> = Vec::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for (message_index, msg) in req.messages.iter().enumerate() {
        match msg.role {
            MessageRole::System => {
                for (block_index, block) in msg.content.iter().enumerate() {
                    if let MessageContentBlock::Text(text) = block {
                        system.push((text.clone(), marks.contains(&(message_index, block_index))));
                    }
                }
            }
            MessageRole::User | MessageRole::Assistant => {
                let role = if matches!(msg.role, MessageRole::User) {
                    "user"
                } else {
                    "assistant"
                };
                let content: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .enumerate()
                    .map(|(block_index, block)| {
                        let mut value = block_to_anthropic(block);
                        if marks.contains(&(message_index, block_index)) {
                            mark_ephemeral(&mut value);
                        }
                        value
                    })
                    .collect();
                messages.push(serde_json::json!({ "role": role, "content": content }));
            }
            // Tool results are represented as tool_result blocks inside a user
            // message; standalone tool-role messages are skipped.
            MessageRole::Tool => {}
        }
    }

    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), serde_json::json!(req.model));
    body.insert(
        "max_tokens".to_string(),
        serde_json::json!(req.max_tokens.unwrap_or(1024)),
    );
    if let Some(temperature) = req.temperature {
        body.insert("temperature".to_string(), serde_json::json!(temperature));
    }
    body.insert("stream".to_string(), serde_json::Value::Bool(true));
    if !system.is_empty() {
        body.insert("system".to_string(), system_value(&system));
    }
    // Explicitly disabled (or unset): emit no thinking key.
    if let Some(thinking) = &req.thinking
        && thinking.enabled
    {
        if let ThinkingParams::Anthropic { adaptive: true, .. } = thinking.params {
            // Newer adaptive mode: no fixed budget; the unified effort
            // becomes the optional `effort` field.
            let mut config = serde_json::Map::new();
            config.insert("type".to_string(), serde_json::json!("adaptive"));
            if let Some(effort) = thinking.effort {
                config.insert(
                    "effort".to_string(),
                    serde_json::json!(adaptive_effort_for(effort)),
                );
            }
            body.insert("thinking".to_string(), serde_json::Value::Object(config));
        } else {
            // Legacy budget mode: a fixed token budget from the params
            // override or the unified-effort budget map.
            let budget = match &thinking.params {
                ThinkingParams::Anthropic {
                    budget_tokens: Some(n),
                    ..
                } => *n,
                _ => anthropic_budget_for(thinking.effort),
            };
            body.insert(
                "thinking".to_string(),
                serde_json::json!({ "type": "enabled", "budget_tokens": budget }),
            );
        }
    }
    body.insert("messages".to_string(), serde_json::json!(messages));
    Ok(serde_json::Value::Object(body))
}

/// The `anthropic-beta` values this request requires.
///
/// Exactly one prompt-caching beta when the *selected* endpoint is Anthropic
/// and the directive carries at least one breakpoint; empty otherwise. The
/// decision is keyed on [`UnifiedRequest::provider`], never on this module
/// being compiled: the DeepSeek bridge reuses this protocol module and must
/// stay on the unsupported path.
pub(crate) fn prompt_cache_betas(req: &UnifiedRequest) -> &'static [&'static str] {
    if cache_breakpoints(req).is_empty() {
        &[]
    } else {
        &["prompt-caching-2024-07-31"]
    }
}

/// The breakpoints this adapter will honor for `req`.
///
/// Empty for every non-Anthropic endpoint (whose wire request must stay
/// unchanged, breakpoints and all) and for a [`PromptCacheDirective::Disabled`]
/// request.
fn cache_breakpoints(req: &UnifiedRequest) -> &[PromptCacheBreakpoint] {
    if req.provider != ProviderEndpoint::Anthropic {
        return &[];
    }
    match &req.prompt_cache {
        PromptCacheDirective::Disabled => &[],
        PromptCacheDirective::Ephemeral { breakpoints } => breakpoints,
    }
}

/// Translate unified `(message_index, block_index)` breakpoints into the set of
/// coordinates to mark, rejecting anything this adapter cannot represent.
///
/// Translation happens here, against the unified message list, *before* the
/// body builder drops standalone tool messages or flattens system text, so a
/// breakpoint is never silently moved onto a different block.
///
/// # Errors
///
/// [`CucaError::Config`] for a duplicate breakpoint, an out-of-range message or
/// block index, a breakpoint on a non-text system block (which this adapter
/// does not send), or a breakpoint on a standalone tool-role message (which
/// this adapter drops).
fn cache_marks(req: &UnifiedRequest) -> Result<HashSet<(usize, usize)>, CucaError> {
    let breakpoints = cache_breakpoints(req);
    let mut marks = HashSet::with_capacity(breakpoints.len());
    for bp in breakpoints {
        let Some(msg) = req.messages.get(bp.message_index) else {
            return Err(CucaError::Config(format!(
                "prompt-cache breakpoint message_index {} is out of range: the request has {} messages",
                bp.message_index,
                req.messages.len()
            )));
        };
        let Some(block) = msg.content.get(bp.block_index) else {
            return Err(CucaError::Config(format!(
                "prompt-cache breakpoint block_index {} is out of range: message {} has {} blocks",
                bp.block_index,
                bp.message_index,
                msg.content.len()
            )));
        };
        match msg.role {
            MessageRole::Tool => {
                return Err(CucaError::Config(format!(
                    "prompt-cache breakpoint (message {}, block {}) targets a standalone tool message, which the anthropic adapter does not send",
                    bp.message_index, bp.block_index
                )));
            }
            MessageRole::System if !matches!(block, MessageContentBlock::Text(_)) => {
                return Err(CucaError::Config(format!(
                    "prompt-cache breakpoint (message {}, block {}) targets a non-text system block, which the anthropic adapter does not send",
                    bp.message_index, bp.block_index
                )));
            }
            _ => {}
        }
        if !marks.insert((bp.message_index, bp.block_index)) {
            return Err(CucaError::Config(format!(
                "duplicate prompt-cache breakpoint at message {} block {}",
                bp.message_index, bp.block_index
            )));
        }
    }
    Ok(marks)
}

/// Render the `system` body value: the existing joined scalar unless a system
/// breakpoint requires array-form text blocks.
fn system_value(system: &[(String, bool)]) -> serde_json::Value {
    if !system.iter().any(|(_, marked)| *marked) {
        let joined: Vec<&str> = system.iter().map(|(text, _)| text.as_str()).collect();
        return serde_json::json!(joined.join("\n"));
    }
    serde_json::Value::Array(
        system
            .iter()
            .map(|(text, marked)| {
                let mut value = serde_json::json!({ "type": "text", "text": text });
                if *marked {
                    mark_ephemeral(&mut value);
                }
                value
            })
            .collect(),
    )
}

/// Add `"cache_control": {"type": "ephemeral"}` to one content block.
fn mark_ephemeral(block: &mut serde_json::Value) {
    if let serde_json::Value::Object(map) = block {
        map.insert(
            "cache_control".to_string(),
            serde_json::json!({ "type": "ephemeral" }),
        );
    }
}

/// Unified thinking effort -> Anthropic extended-thinking token budget.
///
/// `XHigh` shares `High`'s budget; an unset effort gets the conventional
/// default of 10000.
fn anthropic_budget_for(effort: Option<ThinkingEffort>) -> u32 {
    match effort {
        None => 10_000,
        Some(ThinkingEffort::Minimal) => 1024,
        Some(ThinkingEffort::Low) => 2048,
        Some(ThinkingEffort::Medium) => 8192,
        Some(ThinkingEffort::High) => 16_384,
        Some(ThinkingEffort::XHigh) => 16_384,
    }
}

/// Unified thinking effort -> Anthropic adaptive-mode `effort` string.
fn adaptive_effort_for(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Minimal | ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::XHigh => "xhigh",
    }
}

/// Translate one unified content block to the Anthropic wire shape.
fn block_to_anthropic(block: &MessageContentBlock) -> serde_json::Value {
    match block {
        MessageContentBlock::Text(text) => serde_json::json!({ "type": "text", "text": text }),
        MessageContentBlock::ImageBase64 { media_type, data } => serde_json::json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": data },
        }),
        MessageContentBlock::Thinking {
            reasoning,
            signature,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".to_string(), serde_json::json!("thinking"));
            obj.insert("thinking".to_string(), serde_json::json!(reasoning));
            // The signature authenticates extended thinking; omit it when
            // absent rather than emitting null.
            if let Some(sig) = signature {
                obj.insert("signature".to_string(), serde_json::json!(sig));
            }
            serde_json::Value::Object(obj)
        }
        MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": arguments,
        }),
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": output,
        }),
    }
}

/// Which Anthropic content block is currently accumulating, if any.
///
/// Anthropic streams blocks sequentially, so one slot is sufficient (see the
/// [module docs](crate::provider::anthropic)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    /// text block: deltas emit [`MessageContentBlock::Text`] directly.
    Text,
    /// thinking block: thinking/signature deltas accumulate until stop.
    Thinking,
    /// tool_use block: input_json_delta fragments accumulate until stop.
    ToolUse,
}

/// Stateful translator from Anthropic-shaped SSE frames to unified blocks.
///
/// Consumes the frames produced by [`SseStreamParser`] (one [`SseEvent`] per
/// call to [`Self::translate`]) and reassembles block deltas: `text_delta`
/// frames emit [`MessageContentBlock::Text`] immediately, a thinking block's
/// `thinking_delta`/`signature_delta` frames accumulate until its
/// `content_block_stop` emits [`MessageContentBlock::Thinking`], and a
/// tool_use block's `input_json_delta` fragments accumulate until its stop
/// emits a parsed [`MessageContentBlock::ToolCall`].
pub struct AnthropicTranslator {
    /// The block kind currently accumulating, if any.
    current: Option<BlockKind>,
    /// `(id, name, input-json fragment)` for the tool_use block in flight.
    tool_acc: Option<(String, String, String)>,
    /// Reasoning text accumulated for the thinking block in flight.
    reasoning: Option<String>,
    /// Signature accumulated for the thinking block in flight.
    signature: Option<String>,
    /// True once `message_stop` was seen; the frame stream is over.
    done: bool,
    /// Normalized prompt-cache usage read from `message_start.message.usage`,
    /// until [`Self::take_prompt_cache_usage`] claims it.
    prompt_cache_usage: Option<PromptCacheUsage>,
}

impl AnthropicTranslator {
    /// Start a translator with no accumulated state.
    pub fn new() -> Self {
        Self {
            current: None,
            tool_acc: None,
            reasoning: None,
            signature: None,
            done: false,
            prompt_cache_usage: None,
        }
    }

    /// Translate one Anthropic event frame into at most one block.
    ///
    /// `message_start`/`message_delta`/`message_stop`/`ping` produce no
    /// output; `message_stop` additionally marks the frame stream done.
    /// `content_block_start`/`content_block_delta`/`content_block_stop` drive
    /// the accumulators; an `error` event yields [`CucaError::Provider`] with
    /// the error message. Unknown event types are ignored for forward
    /// compatibility.
    pub fn translate(&mut self, ev: &SseEvent) -> Result<Option<MessageContentBlock>, CucaError> {
        match ev.event.as_str() {
            // Terminal event: mark the frame stream over; no block output.
            "message_stop" => {
                self.done = true;
                return Ok(None);
            }
            // Usage rides on `message_start`; it never produces a block, and a
            // frame this adapter cannot parse leaves usage unreported instead
            // of poisoning the stream (control frames were never parsed
            // before, so a malformed one must stay harmless).
            "message_start" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                    self.prompt_cache_usage = prompt_cache_usage_of(&value);
                }
                return Ok(None);
            }
            // Remaining control events carry no content and need no parsing.
            "message_delta" | "ping" => return Ok(None),
            _ => {}
        }

        let mut value: serde_json::Value =
            serde_json::from_str(&ev.data).map_err(|e| CucaError::Json {
                message: format!("invalid anthropic event frame: {e}"),
            })?;

        match ev.event.as_str() {
            "error" => {
                let message = value
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| "anthropic stream error".to_string());
                Err(CucaError::provider(ProviderEndpoint::Anthropic, message))
            }
            "content_block_start" => {
                self.on_block_start(&value);
                Ok(None)
            }
            "content_block_delta" => Ok(self.on_block_delta(&mut value)),
            "content_block_stop" => Ok(self.on_block_stop()),
            _ => Ok(None),
        }
    }

    /// Seed the accumulator for a new `content_block_start`.
    fn on_block_start(&mut self, value: &serde_json::Value) {
        let Some(block) = value.get("content_block") else {
            return;
        };
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => self.current = Some(BlockKind::Text),
            Some("thinking") => {
                self.current = Some(BlockKind::Thinking);
                self.reasoning = Some(String::new());
                self.signature = None;
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or_default()
                    .to_string();
                self.current = Some(BlockKind::ToolUse);
                self.tool_acc = Some((id, name, String::new()));
            }
            _ => {}
        }
    }

    /// Fold one `content_block_delta` into the accumulators; a `text_delta`
    /// emits immediately, everything else buffers until the block stops.
    fn on_block_delta(&mut self, value: &mut serde_json::Value) -> Option<MessageContentBlock> {
        let delta = value.get_mut("delta")?;
        match delta.get("type").and_then(|t| t.as_str()) {
            Some("text_delta") => {
                // `take` moves the delta text out of the frame instead of
                // copying it: `value` is a per-frame local, and this is
                // Anthropic's per-token event.
                match delta.get_mut("text").map(serde_json::Value::take) {
                    Some(serde_json::Value::String(text)) if !text.is_empty() => {
                        Some(MessageContentBlock::Text(text))
                    }
                    _ => None,
                }
            }
            Some("thinking_delta") => {
                let thinking = delta
                    .get("thinking")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default();
                if let Some(acc) = &mut self.reasoning {
                    acc.push_str(thinking);
                }
                None
            }
            Some("signature_delta") => {
                let signature = delta
                    .get("signature")
                    .and_then(|s| s.as_str())
                    .unwrap_or_default();
                if !signature.is_empty() {
                    self.signature = Some(signature.to_string());
                }
                None
            }
            Some("input_json_delta") => {
                let fragment = delta
                    .get("partial_json")
                    .and_then(|p| p.as_str())
                    .unwrap_or_default();
                if let Some((_, _, acc)) = &mut self.tool_acc {
                    acc.push_str(fragment);
                }
                None
            }
            _ => None,
        }
    }

    /// Claim the normalized prompt-cache usage parsed from `message_start`,
    /// leaving the translator with none.
    ///
    /// `None` means the provider reported no prompt-cache counters. Normalized
    /// `prompt_tokens`/`completion_tokens` accounting is unaffected.
    pub fn take_prompt_cache_usage(&mut self) -> Option<PromptCacheUsage> {
        self.prompt_cache_usage.take()
    }

    /// Emit the finished block at `content_block_stop`.
    fn on_block_stop(&mut self) -> Option<MessageContentBlock> {
        let kind = self.current.take()?;
        match kind {
            // Text blocks already emitted their deltas; nothing to flush.
            BlockKind::Text => None,
            BlockKind::Thinking => {
                let reasoning = self.reasoning.take().unwrap_or_default();
                let signature = self.signature.take();
                Some(MessageContentBlock::Thinking {
                    reasoning,
                    signature,
                })
            }
            BlockKind::ToolUse => {
                let (id, name, fragment) = self.tool_acc.take().unwrap_or_default();
                // Fragments that do not parse as JSON (malformed server output)
                // are emitted with `null` arguments rather than dropped.
                let arguments = serde_json::from_str::<serde_json::Value>(&fragment)
                    .unwrap_or(serde_json::Value::Null);
                Some(MessageContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                })
            }
        }
    }
}

impl Default for AnthropicTranslator {
    fn default() -> Self {
        Self::new()
    }
}

/// Read `message.usage.cache_read_input_tokens` /
/// `message.usage.cache_creation_input_tokens` from a `message_start` frame.
///
/// Returns `None` when the frame carries no usage object or neither cache
/// counter, so "provider reported nothing" stays distinguishable from
/// "provider reported zero". A counter present alone normalizes with a zero
/// partner. Anthropic's field names never leave this function.
fn prompt_cache_usage_of(frame: &serde_json::Value) -> Option<PromptCacheUsage> {
    let usage = frame.get("message")?.get("usage")?;
    let counter = |name: &str| usage.get(name).and_then(serde_json::Value::as_u64);
    let read = counter("cache_read_input_tokens");
    let write = counter("cache_creation_input_tokens");
    if read.is_none() && write.is_none() {
        return None;
    }
    Some(PromptCacheUsage {
        read_tokens: read.unwrap_or(0) as u32,
        write_tokens: write.unwrap_or(0) as u32,
    })
}

/// Feed one transport chunk through the SSE parser and translator.
///
/// Pure helper so translation is testable without the network: parses every
/// complete frame in `chunk` and maps each non-empty `data:` payload through
/// [`AnthropicTranslator::translate`]. Frames with empty data are skipped.
pub fn translate_sse(
    parser: &mut SseStreamParser,
    translator: &mut AnthropicTranslator,
    chunk: &[u8],
) -> Result<Vec<Option<MessageContentBlock>>, CucaError> {
    let events = parser.feed_chunk(chunk)?;
    let mut blocks = Vec::with_capacity(events.len());
    for event in events {
        if event.data.is_empty() {
            continue;
        }
        blocks.push(translator.translate(&event)?);
    }
    Ok(blocks)
}

/// Stream a [`UnifiedRequest`] through the Anthropic Messages API.
///
/// POSTs `{base_url}/messages` (the base URL carries the `/v1` suffix) with
/// [`headers`] for `auth` plus whatever [`prompt_cache_betas`] requires (one
/// prompt-caching beta for an Anthropic-endpoint request with breakpoints,
/// nothing otherwise), then pipes the SSE response through [`SseStreamParser`]
/// and [`AnthropicTranslator`]. Normalized prompt-cache usage parsed from
/// `message_start` is published through the returned
/// [`ResponseMetadataHandle`]. Non-2xx responses surface as
/// [`CucaError::Http`] with the captured body; `error` events surface as
/// [`CucaError::Provider`].
pub async fn anthropic_stream(
    http: &reqwest::Client,
    base_url: &str,
    auth: &AnthropicAuth,
    req: UnifiedRequest,
) -> Result<ProviderDispatch, CucaError> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let body = build_anthropic_request(&req)?;
    let response = http
        .post(&url)
        .headers(headers(auth, prompt_cache_betas(&req))?)
        .json(&body)
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
    let metadata = ResponseMetadataHandle::empty();
    Ok(ProviderDispatch {
        stream: Box::pin(AnthropicStream {
            inner: Box::pin(response.bytes_stream()),
            parser: SseStreamParser::new(),
            translator: AnthropicTranslator::new(),
            buffer: VecDeque::new(),
            ended: false,
            metadata: metadata.clone(),
        }),
        metadata,
    })
}

/// Stream adapter: reqwest byte stream -> SSE parser -> block translator.
///
/// Yields at most one block per poll; `message_stop` (or the byte stream
/// ending) terminates the stream once the current chunk's blocks are drained.
struct AnthropicStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseStreamParser,
    translator: AnthropicTranslator,
    /// Blocks awaiting emission within the current chunk.
    buffer: VecDeque<MessageContentBlock>,
    /// True once `message_stop` was seen or the byte stream ended.
    ended: bool,
    /// Publishes normalized prompt-cache usage to the dispatch caller.
    metadata: ResponseMetadataHandle,
}

impl Stream for AnthropicStream {
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
                            // `message_start` usage becomes response
                            // metadata, never a content block.
                            if let Some(usage) = this.translator.take_prompt_cache_usage() {
                                this.metadata.set(usage);
                            }
                            if this.translator.done {
                                // message_stop ended the frame stream; emit
                                // what this chunk produced, then stop.
                                this.ended = true;
                            }
                        }
                        Err(e) => {
                            // A malformed frame or provider error poisons the
                            // stream: report it once and stop reading.
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
                    // Byte stream ended without message_stop: emit whatever
                    // the last chunk produced, then end.
                    this.ended = true;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(feature = "provider-anthropic")]
impl CucaClient {
    /// Dispatch a unified request to the Anthropic Messages API.
    ///
    /// Defaults the base URL to the Anthropic API (`https://api.anthropic.com/v1`)
    /// when the builder did not set one; auth comes from the configured bearer
    /// token (OAuth PKCE path) or API key. Called by `generate_stream` under
    /// the `provider-anthropic` feature.
    pub(crate) async fn dispatch_anthropic(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let base = if self.base_url().is_empty() {
            "https://api.anthropic.com/v1".to_string()
        } else {
            self.base_url().to_string()
        };
        let auth = match (self.bearer_token(), self.api_key()) {
            (Some(token), _) => AnthropicAuth::Bearer(token.to_string()),
            (None, Some(key)) => AnthropicAuth::ApiKey(key.to_string()),
            (None, None) => {
                return Err(CucaError::Config(
                    "anthropic requires an api key or bearer token".into(),
                ));
            }
        };
        anthropic_stream(self.http_client(), &base, &auth, req).await
    }
}

#[cfg(all(test, feature = "provider-anthropic"))]
mod tests {
    use std::sync::{Arc, mpsc};

    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_stream::StreamExt;

    use crate::client::CucaClient;
    use crate::error::PluginError;
    use crate::plugin::CucaPlugin;
    use crate::request::{
        PromptCacheBreakpoint, PromptCacheDirective, PromptCacheUsage, ThinkingConfig,
        UnifiedRequest, UnifiedResponse,
    };
    use crate::sse::SseEvent;
    use crate::types::UnifiedMessage;

    use super::*;

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
            "recording-anthropic"
        }

        fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
            self.tx.send(res.clone()).map_err(|_| {
                PluginError::Internal("recording channel closed before completion".into())
            })
        }
    }

    // --- headers ---

    #[test]
    fn headers_api_key_mode_sends_x_api_key_only() {
        let map = headers(&AnthropicAuth::ApiKey("sk-test".into()), &[]).unwrap();

        assert_eq!(map.get("x-api-key").unwrap().to_str().unwrap(), "sk-test");
        assert!(map.get("authorization").is_none());
        assert_eq!(
            map.get("anthropic-version").unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }

    #[test]
    fn headers_bearer_mode_sends_authorization_only() {
        let map = headers(&AnthropicAuth::Bearer("tok-123".into()), &[]).unwrap();

        assert_eq!(
            map.get("authorization").unwrap().to_str().unwrap(),
            "Bearer tok-123"
        );
        assert!(map.get("x-api-key").is_none());
        assert_eq!(
            map.get("anthropic-version").unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }

    #[test]
    fn headers_include_beta_flags() {
        let map = headers(
            &AnthropicAuth::ApiKey("sk-test".into()),
            &["claude-code-20250219", "prompt-caching-2024-07-31"],
        )
        .unwrap();

        let betas: Vec<&str> = map
            .get_all("anthropic-beta")
            .iter()
            .map(|v| v.to_str().unwrap())
            .collect();
        assert_eq!(
            betas,
            vec!["claude-code-20250219", "prompt-caching-2024-07-31"]
        );
    }

    // --- OAuth PKCE ---

    #[test]
    fn generate_pkce_pair_produces_valid_verifier_and_matching_challenge() {
        let pair = generate_pkce_pair().unwrap();

        // RFC 7636: verifiers are 43..=128 base64url characters; 64 random
        // bytes encode to exactly 86.
        assert!(
            (43..=128).contains(&pair.code_verifier.len()),
            "verifier length {} outside 43..=128",
            pair.code_verifier.len()
        );
        assert_eq!(pair.code_verifier.len(), 86);
        assert!(
            pair.code_verifier
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "verifier must use the base64url charset"
        );

        // The challenge must be base64url(SHA-256(verifier)), recomputed
        // independently here.
        let mut hasher = Sha256::new();
        hasher.update(pair.code_verifier.as_bytes());
        let expected = URL_SAFE_NO_PAD.encode(hasher.finalize());
        assert_eq!(pair.code_challenge, expected);
    }

    #[test]
    fn pkce_challenge_matches_rfc7636_vector_for_fixed_verifier() {
        // RFC 7636 Appendix B fixed vector: verifier -> known S256 challenge.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let expected = URL_SAFE_NO_PAD.encode(hasher.finalize());
        assert_eq!(expected, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
        assert_eq!(pkce_challenge(verifier), expected);
    }

    #[test]
    fn authorization_url_contains_pkce_params() {
        let cfg = OAuthPkceConfig {
            client_id: "client-123".into(),
            authorize_url: "https://auth.anthropic.com/authorize".into(),
            token_url: "https://auth.anthropic.com/token".into(),
            scopes: vec!["user:read".into(), "agent:write".into()],
        };

        let url = authorization_url(
            &cfg,
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            "state-abc",
        );

        assert!(url.starts_with(&cfg.authorize_url));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=client-123"));
        assert!(url.contains("code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=state-abc"));
        assert!(url.contains("scope=user:read+agent:write"));
    }

    // --- request body ---

    #[test]
    fn build_body_extracts_system_and_defaults_max_tokens() {
        let req = UnifiedRequest::new("claude-3-5-sonnet-20241022")
            .add_system_message("be concise")
            .add_system_message("use tools")
            .add_user_message("hello");
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(body["model"], json!("claude-3-5-sonnet-20241022"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["system"], json!("be concise\nuse tools"));
        assert_eq!(body["max_tokens"], json!(1024));
        assert_eq!(body["messages"][0]["role"], json!("user"));
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({ "type": "text", "text": "hello" })
        );
        assert!(
            body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["role"] != json!("system")),
            "system messages must be extracted out of the messages array"
        );
    }

    #[test]
    fn build_body_omits_system_when_absent_and_uses_explicit_knobs() {
        let req = UnifiedRequest::new("claude-3-5-sonnet-20241022")
            .add_user_message("hi")
            .set_max_tokens(2048)
            .set_temperature(0.5);
        let body = build_anthropic_request(&req).unwrap();

        assert!(body.get("system").is_none());
        assert_eq!(body["max_tokens"], json!(2048));
        assert_eq!(body["temperature"], json!(0.5_f32));
    }

    #[test]
    fn build_body_maps_image_source_object() {
        let req = UnifiedRequest::new("claude-3-5-sonnet-20241022").add_message(UnifiedMessage {
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
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(
            body["messages"][0]["content"],
            json!([
                { "type": "text", "text": "what is this" },
                {
                    "type": "image",
                    "source": { "type": "base64", "media_type": "image/png", "data": "aGVsbG8=" },
                },
            ])
        );
    }

    #[test]
    fn build_body_maps_tool_use_and_tool_result_blocks() {
        let req = UnifiedRequest::new("claude-3-5-sonnet-20241022")
            .add_message(UnifiedMessage {
                role: MessageRole::Assistant,
                content: vec![MessageContentBlock::ToolCall {
                    id: "toolu_1".into(),
                    name: "get_weather".into(),
                    arguments: json!({ "location": "NYC" }),
                }],
                name: None,
                tool_call_id: None,
            })
            .add_message(UnifiedMessage {
                role: MessageRole::User,
                content: vec![MessageContentBlock::ToolResult {
                    tool_call_id: "toolu_1".into(),
                    output: "72F".into(),
                }],
                name: None,
                tool_call_id: None,
            });
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(body["messages"][0]["role"], json!("assistant"));
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": { "location": "NYC" },
            })
        );
        assert_eq!(body["messages"][1]["role"], json!("user"));
        assert_eq!(
            body["messages"][1]["content"][0],
            json!({ "type": "tool_result", "tool_use_id": "toolu_1", "content": "72F" })
        );
    }

    #[test]
    fn build_body_maps_thinking_block_with_optional_signature() {
        let req = UnifiedRequest::new("claude-3-5-sonnet-20241022")
            .add_message(UnifiedMessage {
                role: MessageRole::Assistant,
                content: vec![MessageContentBlock::Thinking {
                    reasoning: "let me think".into(),
                    signature: Some("sig-1".into()),
                }],
                name: None,
                tool_call_id: None,
            })
            .add_message(UnifiedMessage {
                role: MessageRole::Assistant,
                content: vec![MessageContentBlock::Thinking {
                    reasoning: "plain".into(),
                    signature: None,
                }],
                name: None,
                tool_call_id: None,
            });
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(
            body["messages"][0]["content"][0],
            json!({ "type": "thinking", "thinking": "let me think", "signature": "sig-1" })
        );
        let plain = body["messages"][1]["content"][0].as_object().unwrap();
        assert_eq!(plain["type"], json!("thinking"));
        assert_eq!(plain["thinking"], json!("plain"));
        assert!(plain.get("signature").is_none());
    }

    // --- thinking ---

    #[test]
    fn build_body_thinking_budget_mode_defaults_and_effort_budgets() {
        for (effort, expected) in [
            (None, 10_000),
            (Some(ThinkingEffort::Minimal), 1024),
            (Some(ThinkingEffort::Low), 2048),
            (Some(ThinkingEffort::Medium), 8192),
            (Some(ThinkingEffort::High), 16_384),
            // XHigh shares High's budget in budget mode.
            (Some(ThinkingEffort::XHigh), 16_384),
        ] {
            let req = UnifiedRequest::new("claude-sonnet-4-5")
                .add_user_message("hi")
                .enable_thinking(effort);
            let body = build_anthropic_request(&req).unwrap();

            assert_eq!(
                body["thinking"],
                json!({ "type": "enabled", "budget_tokens": expected })
            );
        }
    }

    #[test]
    fn build_body_thinking_budget_tokens_override_wins() {
        let req = UnifiedRequest::new("claude-sonnet-4-5")
            .add_user_message("hi")
            .with_thinking(ThinkingConfig {
                enabled: true,
                effort: Some(ThinkingEffort::High),
                params: ThinkingParams::Anthropic {
                    budget_tokens: Some(42_000),
                    adaptive: false,
                },
            });
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(
            body["thinking"],
            json!({ "type": "enabled", "budget_tokens": 42_000 })
        );
    }

    #[test]
    fn build_body_thinking_adaptive_mode_with_effort() {
        for (effort, expected) in [
            (ThinkingEffort::Minimal, "low"),
            (ThinkingEffort::Low, "low"),
            (ThinkingEffort::Medium, "medium"),
            (ThinkingEffort::High, "high"),
            (ThinkingEffort::XHigh, "xhigh"),
        ] {
            let req = UnifiedRequest::new("claude-sonnet-4-5")
                .add_user_message("hi")
                .with_thinking(ThinkingConfig {
                    enabled: true,
                    effort: Some(effort),
                    params: ThinkingParams::Anthropic {
                        budget_tokens: None,
                        adaptive: true,
                    },
                });
            let body = build_anthropic_request(&req).unwrap();

            assert_eq!(body["thinking"]["type"], json!("adaptive"));
            assert_eq!(body["thinking"]["effort"], json!(expected));
            assert!(body["thinking"].get("budget_tokens").is_none());
        }
    }

    #[test]
    fn build_body_thinking_adaptive_mode_without_effort_omits_effort_key() {
        let req = UnifiedRequest::new("claude-sonnet-4-5")
            .add_user_message("hi")
            .with_thinking(ThinkingConfig {
                enabled: true,
                effort: None,
                params: ThinkingParams::Anthropic {
                    budget_tokens: None,
                    adaptive: true,
                },
            });
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(body["thinking"], json!({ "type": "adaptive" }));
        assert!(body["thinking"].get("effort").is_none());
    }

    #[test]
    fn build_body_disabled_thinking_emits_no_thinking_key() {
        let req = UnifiedRequest::new("claude-sonnet-4-5")
            .add_user_message("hi")
            .enable_thinking(Some(ThinkingEffort::High))
            .disable_thinking();
        let body = build_anthropic_request(&req).unwrap();

        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn build_body_unset_thinking_emits_no_thinking_key() {
        let req = UnifiedRequest::new("claude-sonnet-4-5").add_user_message("hi");
        let body = build_anthropic_request(&req).unwrap();

        assert!(body.get("thinking").is_none());
    }

    // --- translator ---

    /// Drive a translator through a canned (event, data) sequence, returning
    /// the translator (for `done` assertions) and the emitted blocks.
    fn translate_frames(
        frames: &[(&str, &str)],
    ) -> (AnthropicTranslator, Vec<MessageContentBlock>) {
        let mut translator = AnthropicTranslator::new();
        let mut out = Vec::new();
        for (event, data) in frames {
            let ev = SseEvent {
                event: (*event).into(),
                data: (*data).into(),
                id: None,
                retry: None,
            };
            if let Some(block) = translator.translate(&ev).unwrap() {
                out.push(block);
            }
        }
        (translator, out)
    }

    #[test]
    fn translate_text_sequence_yields_text_blocks_and_marks_done() {
        let (translator, out) = translate_frames(&[
            (
                "message_start",
                r#"{"type":"message_start","message":{"role":"assistant","content":[]}}"#,
            ),
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            ("message_stop", r#"{"type":"message_stop"}"#),
        ]);

        assert_eq!(
            out,
            vec![
                MessageContentBlock::Text("Hello".into()),
                MessageContentBlock::Text(" world".into()),
            ]
        );
        assert!(
            translator.done,
            "message_stop must mark the translator done"
        );
    }

    #[test]
    fn translate_thinking_sequence_yields_thinking_with_signature() {
        let (_, out) = translate_frames(&[
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me think"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" step by step"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-123"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
        ]);

        assert_eq!(
            out,
            vec![MessageContentBlock::Thinking {
                reasoning: "let me think step by step".into(),
                signature: Some("sig-123".into()),
            }]
        );
    }

    #[test]
    fn translate_tool_use_frames_accumulate_to_one_tool_call() {
        let (_, out) = translate_frames(&[
            (
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"NYC\"}"}}"#,
            ),
            (
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
        ]);

        assert_eq!(
            out,
            vec![MessageContentBlock::ToolCall {
                id: "toolu_1".into(),
                name: "get_weather".into(),
                arguments: json!({ "location": "NYC" }),
            }]
        );
    }

    #[test]
    fn translate_error_event_yields_provider_error() {
        let mut translator = AnthropicTranslator::new();
        let ev = SseEvent {
            event: "error".into(),
            data: r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#
                .into(),
            id: None,
            retry: None,
        };

        match translator.translate(&ev).unwrap_err() {
            CucaError::Provider { provider, message } => {
                assert_eq!(provider, ProviderEndpoint::Anthropic);
                assert_eq!(message, "overloaded");
            }
            other => panic!("expected Provider error, got {other}"),
        }
    }

    // --- end-to-end ---

    /// Canned Anthropic-shaped SSE frames: message -> text block -> deltas ->
    /// stop -> message_stop.
    fn canned_anthropic_frames() -> Vec<&'static str> {
        vec![
            r#"event: message_start
data: {"type":"message_start","message":{"role":"assistant","content":[]}}"#,
            r#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            r#"event: content_block_stop
data: {"type":"content_block_stop","index":0}"#,
            r#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"event: message_stop
data: {"type":"message_stop"}"#,
        ]
    }

    #[tokio::test]
    async fn end_to_end_stream_translates_anthropic_sse_and_completes_plugin() {
        // In-process stub: an Anthropic-shaped SSE server on an ephemeral port.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let frames = canned_anthropic_frames();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            // Capture and verify the request head before responding.
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf);
            assert!(
                head.contains("x-api-key: sk-test"),
                "missing x-api-key in request head:\n{head}"
            );
            assert!(
                head.contains("anthropic-version: 2023-06-01"),
                "missing anthropic-version in request head:\n{head}"
            );
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
        });

        let (plugin, rx) = RecordingPlugin::new();
        let plugin = Arc::new(plugin);
        // Register through the trait-object type the builder expects; the
        // concrete handle is kept for asserting on the recorded responses.
        let plugin_dyn = Arc::clone(&plugin) as Arc<dyn CucaPlugin>;
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::Anthropic)
            .with_base_url(format!("http://{addr}/v1"))
            .with_api_key("sk-test")
            .register_plugin(plugin_dyn)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));

        let stream = client
            .generate_stream(
                UnifiedRequest::new("claude-3-5-sonnet-20241022").add_user_message("hi"),
            )
            .await
            .unwrap_or_else(|e| panic!("generate_stream must succeed: {e}"));
        let mut blocks = Vec::new();
        let mut stream = stream;
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap_or_else(|e| panic!("stream block must be Ok: {e}")));
        }
        server.await.unwrap();

        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("Hello".into()),
                MessageContentBlock::Text(" world".into()),
            ]
        );

        // The completion hook fired exactly once, with the aggregated response.
        let completed: Vec<UnifiedResponse> = rx.try_iter().collect();
        assert_eq!(
            completed.len(),
            1,
            "on_response_complete must fire exactly once"
        );
        assert!(completed[0].completion_tokens >= 1);
        assert_eq!(completed[0].content, blocks);
        assert_eq!(completed[0].model, "claude-3-5-sonnet-20241022");
        assert_eq!(completed[0].provider, ProviderEndpoint::Anthropic);
        assert_eq!(
            completed[0].prompt_cache_usage, None,
            "a response without usage fields reports no prompt-cache usage"
        );
    }

    // --- prompt-cache mapping (request side) ---

    fn msg(role: MessageRole, content: Vec<MessageContentBlock>) -> UnifiedMessage {
        UnifiedMessage {
            role,
            content,
            name: None,
            tool_call_id: None,
        }
    }

    fn text(s: &str) -> MessageContentBlock {
        MessageContentBlock::Text(s.to_string())
    }

    /// System(2 text blocks) / User(2 text blocks) / Assistant(thinking +
    /// tool_use) / standalone Tool message, all on the Anthropic endpoint.
    fn cache_request() -> UnifiedRequest {
        let mut req = UnifiedRequest::new("claude-3-5-sonnet-20241022");
        req.provider = ProviderEndpoint::Anthropic;
        req.messages = vec![
            msg(
                MessageRole::System,
                vec![text("policy a"), text("policy b")],
            ),
            msg(MessageRole::User, vec![text("question"), text("context")]),
            msg(
                MessageRole::Assistant,
                vec![
                    MessageContentBlock::Thinking {
                        reasoning: "hmm".to_string(),
                        signature: None,
                    },
                    MessageContentBlock::ToolCall {
                        id: "call-1".to_string(),
                        name: "lookup".to_string(),
                        arguments: json!({"q": 1}),
                    },
                ],
            ),
            msg(
                MessageRole::Tool,
                vec![MessageContentBlock::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    output: "42".to_string(),
                }],
            ),
        ];
        req
    }

    fn ephemeral(points: &[(usize, usize)]) -> PromptCacheDirective {
        PromptCacheDirective::Ephemeral {
            breakpoints: points
                .iter()
                .map(|(message_index, block_index)| PromptCacheBreakpoint {
                    message_index: *message_index,
                    block_index: *block_index,
                })
                .collect(),
        }
    }

    fn cache_control_of(block: &serde_json::Value) -> Option<&serde_json::Value> {
        block.get("cache_control")
    }

    #[test]
    fn disabled_directive_keeps_the_existing_body_and_no_betas() {
        let req = cache_request();
        assert_eq!(req.prompt_cache, PromptCacheDirective::Disabled);
        let body = build_anthropic_request(&req).unwrap();

        // Joined scalar system form.
        assert_eq!(body["system"], json!("policy a\npolicy b"));
        assert!(
            !serde_json::to_string(&body)
                .unwrap()
                .contains("cache_control"),
            "the disabled path must not emit cache_control: {body}"
        );
        assert!(prompt_cache_betas(&req).is_empty());
    }

    #[test]
    fn ephemeral_without_breakpoints_is_identical_to_disabled() {
        let mut req = cache_request();
        let disabled = build_anthropic_request(&req).unwrap();
        req.prompt_cache = ephemeral(&[]);
        assert_eq!(build_anthropic_request(&req).unwrap(), disabled);
        assert!(
            prompt_cache_betas(&req).is_empty(),
            "no breakpoints means no beta header"
        );
    }

    #[test]
    fn a_user_breakpoint_marks_only_that_block_and_keeps_system_scalar() {
        let mut req = cache_request();
        req.prompt_cache = ephemeral(&[(1, 1)]);
        let body = build_anthropic_request(&req).unwrap();

        // System is untouched: still the joined scalar form.
        assert_eq!(body["system"], json!("policy a\npolicy b"));
        let user = &body["messages"][0];
        assert_eq!(user["role"], json!("user"));
        assert_eq!(
            user["content"][0],
            json!({"type": "text", "text": "question"})
        );
        assert_eq!(
            user["content"][1],
            json!({
                "type": "text",
                "text": "context",
                "cache_control": {"type": "ephemeral"},
            })
        );
        // The assistant message is untouched.
        assert!(cache_control_of(&body["messages"][1]["content"][0]).is_none());
        assert!(cache_control_of(&body["messages"][1]["content"][1]).is_none());
    }

    #[test]
    fn a_system_breakpoint_switches_system_to_array_form() {
        let mut req = cache_request();
        req.prompt_cache = ephemeral(&[(0, 1)]);
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(
            body["system"],
            json!([
                {"type": "text", "text": "policy a"},
                {"type": "text", "text": "policy b", "cache_control": {"type": "ephemeral"}},
            ]),
            "array-form system preserves order and marks only the target"
        );
        // Messages keep their existing shape.
        assert!(
            !serde_json::to_string(&body["messages"])
                .unwrap()
                .contains("cache_control")
        );
    }

    #[test]
    fn an_assistant_breakpoint_marks_the_translated_block_only() {
        let mut req = cache_request();
        req.prompt_cache = ephemeral(&[(2, 1)]);
        let body = build_anthropic_request(&req).unwrap();

        let assistant = &body["messages"][1];
        assert_eq!(assistant["role"], json!("assistant"));
        assert_eq!(
            assistant["content"][0],
            json!({"type": "thinking", "thinking": "hmm"}),
            "unmarked blocks keep their exact previous value"
        );
        assert_eq!(
            assistant["content"][1],
            json!({
                "type": "tool_use",
                "id": "call-1",
                "name": "lookup",
                "input": {"q": 1},
                "cache_control": {"type": "ephemeral"},
            })
        );
    }

    #[test]
    fn multiple_breakpoints_mark_every_target() {
        let mut req = cache_request();
        req.prompt_cache = ephemeral(&[(0, 0), (1, 0), (2, 0)]);
        let body = build_anthropic_request(&req).unwrap();

        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(cache_control_of(&body["system"][1]).is_none());
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(cache_control_of(&body["messages"][0]["content"][1]).is_none());
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn anthropic_endpoint_with_breakpoints_adds_exactly_one_beta_header() {
        let mut req = cache_request();
        req.prompt_cache = ephemeral(&[(1, 0)]);
        assert_eq!(prompt_cache_betas(&req), &["prompt-caching-2024-07-31"]);

        let map = headers(
            &AnthropicAuth::ApiKey("sk-test".into()),
            prompt_cache_betas(&req),
        )
        .unwrap();
        let betas: Vec<&str> = map
            .get_all("anthropic-beta")
            .iter()
            .map(|v| v.to_str().unwrap())
            .collect();
        assert_eq!(betas, vec!["prompt-caching-2024-07-31"]);
        // Existing auth/version headers are still present and unchanged.
        assert_eq!(map.get("x-api-key").unwrap().to_str().unwrap(), "sk-test");
        assert_eq!(
            map.get("anthropic-version").unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }

    /// Support is keyed on the selected endpoint, not on this module being
    /// compiled: a DeepSeek request never gets caching fields even though it
    /// goes through this very builder.
    #[test]
    fn a_non_anthropic_endpoint_ignores_the_directive_entirely() {
        let mut req = cache_request();
        req.provider = ProviderEndpoint::DeepSeek;
        let baseline = build_anthropic_request(&req).unwrap();

        // Even coordinates that would be errors on Anthropic are ignored.
        req.prompt_cache = ephemeral(&[(0, 1), (99, 99), (3, 0)]);
        let body = build_anthropic_request(&req)
            .expect("an unsupported endpoint must not fail on breakpoints");
        assert_eq!(body, baseline, "the wire request is unchanged");
        assert!(prompt_cache_betas(&req).is_empty());
    }

    // --- coordinate validation (Anthropic endpoint only) ---

    fn config_error_for(points: &[(usize, usize)]) -> String {
        let mut req = cache_request();
        req.prompt_cache = ephemeral(points);
        match build_anthropic_request(&req) {
            Err(CucaError::Config(message)) => message,
            Ok(body) => panic!("expected a configuration error, got body {body}"),
            Err(other) => panic!("expected CucaError::Config, got {other}"),
        }
    }

    #[test]
    fn duplicate_breakpoints_are_a_configuration_error() {
        let message = config_error_for(&[(1, 0), (1, 0)]);
        assert!(
            message.contains("duplicate"),
            "error must name the duplicate: {message}"
        );
    }

    #[test]
    fn out_of_range_breakpoints_are_a_configuration_error() {
        let message = config_error_for(&[(9, 0)]);
        assert!(
            message.contains('9'),
            "error must name the bad message index: {message}"
        );
        let message = config_error_for(&[(1, 7)]);
        assert!(
            message.contains('7'),
            "error must name the bad block index: {message}"
        );
    }

    #[test]
    fn a_system_breakpoint_on_a_non_text_block_is_a_configuration_error() {
        let mut req = cache_request();
        req.messages[0]
            .content
            .push(MessageContentBlock::ImageBase64 {
                media_type: "image/png".to_string(),
                data: "AAAA".to_string(),
            });
        req.prompt_cache = ephemeral(&[(0, 2)]);
        match build_anthropic_request(&req) {
            Err(CucaError::Config(message)) => assert!(
                message.contains("system"),
                "error must explain the unsupported system block: {message}"
            ),
            other => panic!("expected CucaError::Config, got {other:?}"),
        }
    }

    /// Standalone tool-role messages are dropped from the Anthropic body, so a
    /// breakpoint aimed at one is rejected instead of silently moved to another
    /// block.
    #[test]
    fn a_breakpoint_on_a_dropped_tool_message_is_a_configuration_error() {
        let message = config_error_for(&[(3, 0)]);
        assert!(
            message.contains("tool"),
            "error must explain the dropped tool message: {message}"
        );
    }

    // --- normalized prompt-cache usage (response side) ---

    fn message_start(data: &str) -> SseEvent {
        SseEvent {
            event: "message_start".to_string(),
            data: data.to_string(),
            id: None,
            retry: None,
        }
    }

    #[test]
    fn message_start_usage_normalizes_both_counters_and_emits_no_block() {
        let mut translator = AnthropicTranslator::new();
        let block = translator
            .translate(&message_start(
                r#"{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":100,"cache_read_input_tokens":37,"cache_creation_input_tokens":11}}}"#,
            ))
            .expect("a control event never fails");
        assert!(
            block.is_none(),
            "usage control events emit no content block"
        );

        assert_eq!(
            translator.take_prompt_cache_usage(),
            Some(PromptCacheUsage {
                read_tokens: 37,
                write_tokens: 11,
            })
        );
        assert_eq!(
            translator.take_prompt_cache_usage(),
            None,
            "taking the usage leaves the translator empty"
        );
    }

    #[test]
    fn a_single_reported_counter_still_normalizes_with_a_zero_partner() {
        let mut translator = AnthropicTranslator::new();
        translator
            .translate(&message_start(
                r#"{"type":"message_start","message":{"usage":{"cache_read_input_tokens":5}}}"#,
            ))
            .unwrap();
        assert_eq!(
            translator.take_prompt_cache_usage(),
            Some(PromptCacheUsage {
                read_tokens: 5,
                write_tokens: 0,
            })
        );
    }

    #[test]
    fn missing_or_unusable_usage_stays_none() {
        for data in [
            r#"{"type":"message_start","message":{"role":"assistant","content":[]}}"#,
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10}}}"#,
            r#"{"type":"message_start"}"#,
            // A malformed control frame must not become a stream error either.
            r#"not json"#,
        ] {
            let mut translator = AnthropicTranslator::new();
            let block = translator
                .translate(&message_start(data))
                .unwrap_or_else(|e| panic!("control frames never fail, got {e} for {data}"));
            assert!(block.is_none());
            assert_eq!(
                translator.take_prompt_cache_usage(),
                None,
                "no usage reported for {data}"
            );
        }
    }

    /// Usage arrives only from `message_start`: a `message_delta` carrying
    /// cache counters is not a source of normalized usage.
    #[test]
    fn usage_is_read_only_from_message_start() {
        let mut translator = AnthropicTranslator::new();
        translator
            .translate(&SseEvent {
                event: "message_delta".to_string(),
                data: r#"{"type":"message_delta","message":{"usage":{"cache_read_input_tokens":9,"cache_creation_input_tokens":9}}}"#
                    .to_string(),
                id: None,
                retry: None,
            })
            .unwrap();
        assert_eq!(translator.take_prompt_cache_usage(), None);
    }

    /// Canned frames whose `message_start` reports both cache counters.
    fn canned_frames_with_cache_usage() -> Vec<&'static str> {
        vec![
            r#"event: message_start
data: {"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":100,"cache_read_input_tokens":64,"cache_creation_input_tokens":8}}}"#,
            r#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cached"}}"#,
            r#"event: content_block_stop
data: {"type":"content_block_stop","index":0}"#,
            r#"event: message_stop
data: {"type":"message_stop"}"#,
        ]
    }

    #[tokio::test]
    async fn end_to_end_usage_reaches_the_normalized_response() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let frames = canned_frames_with_cache_usage();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 2048];
            let _ = socket.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf);
            assert!(
                head.contains("anthropic-beta: prompt-caching-2024-07-31"),
                "a request with breakpoints must carry exactly one prompt-caching beta:\n{head}"
            );
            assert_eq!(
                head.matches("anthropic-beta").count(),
                1,
                "exactly one beta header:\n{head}"
            );
            assert!(
                head.contains("x-api-key: sk-test")
                    && head.contains("anthropic-version: 2023-06-01"),
                "existing auth/version headers must survive:\n{head}"
            );
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in &frames {
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        });

        let (plugin, rx) = RecordingPlugin::new();
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::Anthropic)
            .with_base_url(format!("http://{addr}/v1"))
            .with_api_key("sk-test")
            .register_plugin(Arc::new(plugin) as Arc<dyn CucaPlugin>)
            .build()
            .unwrap();

        let request = UnifiedRequest::new("claude-3-5-sonnet-20241022")
            .add_system_message("policy")
            .add_user_message("hi")
            .with_prompt_cache(PromptCacheDirective::Ephemeral {
                breakpoints: vec![PromptCacheBreakpoint {
                    message_index: 0,
                    block_index: 0,
                }],
            });
        let mut stream = client.generate_stream(request).await.unwrap();
        let mut blocks = Vec::new();
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap());
        }
        server.await.unwrap();

        assert_eq!(
            blocks,
            vec![MessageContentBlock::Text("cached".into())],
            "usage never becomes a content block"
        );
        let completed: Vec<UnifiedResponse> = rx.try_iter().collect();
        assert_eq!(completed.len(), 1);
        assert_eq!(
            completed[0].prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 64,
                write_tokens: 8,
            })
        );
        // Existing normalized token semantics are unchanged.
        assert!(completed[0].completion_tokens >= 1);
        assert_eq!(completed[0].provider, ProviderEndpoint::Anthropic);
    }
}
