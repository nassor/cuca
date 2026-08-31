//! llama.cpp provider adapter.
//!
//! llama-server exposes two API styles: an OpenAI-compatible
//! `/v1/chat/completions` route and the native `/completion` route with
//! raw-token SSE frames (`{"content": "tok", "stop": false}`). This module
//! implements both: the chat route reuses the shared
//! [`openai_compat`](crate::provider::openai_compat) building blocks
//! ([`ChatCompletionTranslator`], [`translate_sse`]), and the native route
//! translates frames with [`CompletionTranslator`].
//!
//! One configured [`LlamaCppConfig`] serves both routes: the dispatch appends
//! `/v1` to the base URL for the Chat route when it is not already present, so
//! a single bare default (`http://127.0.0.1:8080`) reaches
//! `/v1/chat/completions` on the chat route and `/completion` on the native
//! route.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use serde_json::json;
use tokio_stream::Stream;

use crate::client::{CucaClient, ProviderDispatch, ResponseMetadataHandle};
use crate::error::CucaError;
use crate::provider::openai_compat::{OpenAiCompatConfig, openai_compat_stream};
use crate::request::UnifiedRequest;
use crate::sse::SseStreamParser;
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint};

/// The route a llama.cpp request is served by.
///
/// llama-server's OpenAI-compatible endpoint (`/v1/chat/completions`) and its
/// native endpoint (`/completion`) use different request bodies and frame
/// shapes, so the dispatch picks the adapter from the route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRoute {
    /// OpenAI-compatible `/v1/chat/completions` route (shared `openai_compat`
    /// translation).
    Chat,
    /// Native `/completion` route with raw-token frames.
    Completion,
}

/// llama.cpp adapter configuration.
///
/// Fields mirror the llama-server body parameters this adapter sends (thread
/// affinity, flash attention, GPU offload). An empty `base_url`/`model`
/// defers to the client's base URL and the request's model at dispatch time.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaCppConfig {
    /// Base URL of the llama-server; defaults to `http://127.0.0.1:8080` when
    /// both this and the client's base URL are empty.
    pub base_url: String,
    /// Optional auth header value; sent as `Authorization: Bearer` when set.
    pub api_key: Option<String>,
    /// The effective model; overrides the request's model in the body when
    /// non-empty (empty defers to the request model).
    pub model: String,
    /// CPU thread count (`n_threads` body parameter); `None` defers to the
    /// server default.
    pub n_threads: Option<u32>,
    /// GPU offload depth (`n_gpu_layers` body parameter); `None` defers to the
    /// server default.
    pub n_gpu_layers: Option<u32>,
    /// Whether to enable flash attention (`flash_attn` body parameter).
    pub flash_attn: bool,
    /// Which server route serves requests (chat vs native completion).
    pub route: LlamaRoute,
}

impl Default for LlamaCppConfig {
    /// A chat-route config with no auth and no runtime knobs;
    /// `base_url`/`model` empty means the dispatch falls back to the
    /// client's base URL and the request's model.
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: None,
            model: String::new(),
            n_threads: None,
            n_gpu_layers: None,
            flash_attn: false,
            route: LlamaRoute::Chat,
        }
    }
}

/// The `n_predict` cap used when the request sets no `max_tokens`.
const DEFAULT_N_PREDICT: u32 = 128;

/// Build the native `/completion` request body for a [`UnifiedRequest`].
///
/// `stream` is always `true` (the client only consumes SSE); `temperature` is
/// included only when set; `n_predict` carries `max_tokens` (or
/// [`DEFAULT_N_PREDICT`]); `n_threads`/`n_gpu_layers`/`flash_attn` appear only
/// when configured. The prompt is the role-marked conversation assembled by
/// [`assemble_prompt`]. Following the effective-model pattern of
/// [`openai_compat_stream`](crate::provider::openai_compat::openai_compat_stream),
/// `cfg.model` overrides the request model in the body (llama.cpp ignores the
/// field, but the body stays self-describing for proxies and logging).
///
/// The native `/completion` route has no thinking knob: `req.thinking` is
/// silently ignored here. (The chat route inherits the OpenAI-compatible
/// `reasoning_effort` translation from
/// [`build_chat_completion_body`](crate::provider::openai_compat::build_chat_completion_body).)
pub fn build_completion_body(req: &UnifiedRequest, cfg: &LlamaCppConfig) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    body.insert("prompt".to_string(), json!(assemble_prompt(req)));
    body.insert("stream".to_string(), json!(true));
    if let Some(temperature) = req.temperature {
        body.insert("temperature".to_string(), json!(temperature));
    }
    body.insert(
        "n_predict".to_string(),
        json!(req.max_tokens.unwrap_or(DEFAULT_N_PREDICT)),
    );
    if let Some(threads) = cfg.n_threads {
        body.insert("n_threads".to_string(), json!(threads));
    }
    if let Some(layers) = cfg.n_gpu_layers {
        body.insert("n_gpu_layers".to_string(), json!(layers));
    }
    if cfg.flash_attn {
        body.insert("flash_attn".to_string(), json!(true));
    }
    body.insert("model".to_string(), json!(cfg.model));
    serde_json::Value::Object(body)
}

/// Assemble the native `/completion` prompt from the unified conversation.
///
/// System messages contribute their text as bare context at the top (the
/// unified model's convention of leading system instructions); User and
/// Assistant messages are marked with the classic `### User:` / `### Assistant:`
/// markers used by llama.cpp chat-style prompting. Only `Text` blocks
/// contribute: images, reasoning, and tool blocks have no native
/// representation and are dropped, and Tool messages are skipped entirely (the
/// native endpoint has no tool protocol).
fn assemble_prompt(req: &UnifiedRequest) -> String {
    let mut prompt = String::new();
    for msg in &req.messages {
        let text: Vec<&str> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        let text = text.join("\n");
        if text.is_empty() {
            continue;
        }
        match msg.role {
            MessageRole::System => {
                prompt.push_str(&text);
                prompt.push('\n');
            }
            MessageRole::User => {
                prompt.push_str("### User:\n");
                prompt.push_str(&text);
                prompt.push('\n');
            }
            MessageRole::Assistant => {
                prompt.push_str("### Assistant:\n");
                prompt.push_str(&text);
                prompt.push('\n');
            }
            MessageRole::Tool => {}
        }
    }
    prompt
}

/// Stateless translator for native `/completion` SSE frames.
///
/// llama.cpp emits one frame per token (`{"content": "tok", "stop": false}`)
/// and terminates with `{"content": "", "stop": true}`. Each non-empty
/// `content` delta becomes a [`MessageContentBlock::Text`]; empty or missing
/// content yields `None`, so the final stop frames flow through as no-ops. An
/// `error` field surfaces as [`CucaError::Provider`].
pub struct CompletionTranslator;

impl CompletionTranslator {
    /// Translate one `data:` payload into at most one block.
    ///
    /// - non-empty `content` -> [`MessageContentBlock::Text`];
    /// - empty/absent `content` (e.g. the `{"content": "", "stop": true}`
    ///   end-of-stream frame) -> `None`;
    /// - an `error` field -> [`CucaError::Provider`] for llama.cpp;
    /// - malformed JSON -> [`CucaError::Json`].
    pub fn translate(&self, payload: &str) -> Result<Option<MessageContentBlock>, CucaError> {
        let mut value: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| CucaError::Json {
                message: format!("invalid llama.cpp completion frame: {e}"),
            })?;

        // llama.cpp error bodies are JSON objects with an `error` field
        // (usually a string message; some builds nest {"message": ...}).
        if let Some(error) = value.get("error") {
            let message = error
                .as_str()
                .or_else(|| error.get("message").and_then(|m| m.as_str()))
                .unwrap_or("llama.cpp completion error");
            return Err(CucaError::provider(ProviderEndpoint::LlamaCpp, message));
        }

        // `take` moves the payload string out of the frame instead of copying
        // it: `value` is a local that dies at the end of this call, and this
        // route emits one frame per token.
        match value.get_mut("content").map(serde_json::Value::take) {
            Some(serde_json::Value::String(content)) if !content.is_empty() => {
                Ok(Some(MessageContentBlock::Text(content)))
            }
            _ => Ok(None),
        }
    }
}

/// Feed one transport chunk through the SSE parser and the completion
/// translator (pure helper, mirroring the other providers).
///
/// Parses every complete frame in `chunk` and maps each non-empty `data:`
/// payload through [`CompletionTranslator::translate`]; frames with empty data
/// are skipped.
pub fn translate_sse(
    parser: &mut SseStreamParser,
    translator: &CompletionTranslator,
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

/// Stream a [`UnifiedRequest`] through llama.cpp's native `/completion` route.
///
/// POSTs `{base_url}/completion` (the base URL's trailing `/` is trimmed) with
/// a bearer token when `api_key` is set; the body is [`build_completion_body`].
/// Non-2xx responses surface as [`CucaError::Http`] with the captured body;
/// frames flow through [`SseStreamParser`] and [`CompletionTranslator`].
pub async fn llamacpp_completion_stream(
    http: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
    req: UnifiedRequest,
    cfg: &LlamaCppConfig,
) -> Result<ProviderDispatch, CucaError> {
    let url = format!("{}/completion", base_url.trim_end_matches('/'));
    let body = build_completion_body(&req, cfg);
    let mut request = http.post(&url).json(&body);
    if let Some(api_key) = api_key {
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
        stream: Box::pin(CompletionStream {
            inner: Box::pin(response.bytes_stream()),
            parser: SseStreamParser::new(),
            translator: CompletionTranslator,
            buffer: VecDeque::new(),
            ended: false,
        }),
        metadata: ResponseMetadataHandle::empty(),
    })
}

/// Stream adapter for the native route: reqwest byte stream -> SSE parser ->
/// [`CompletionTranslator`].
///
/// Yields at most one block per poll; the stream ends when the byte stream
/// ends. llama.cpp's final `{"content": "", "stop": true}` frames translate to
/// `None` and are skipped, so the stream ends naturally right after them.
struct CompletionStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseStreamParser,
    translator: CompletionTranslator,
    /// Blocks awaiting emission within the current chunk.
    buffer: VecDeque<MessageContentBlock>,
    /// True once the byte stream ended; only `buffer` leftovers are emitted.
    ended: bool,
}

impl Stream for CompletionStream {
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
                            for block in blocks.into_iter().flatten() {
                                this.buffer.push_back(block);
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
                    this.ended = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(feature = "provider-llamacpp")]
impl CucaClient {
    /// Dispatch a unified request to a llama.cpp server.
    ///
    /// Resolves the effective base URL (`cfg.base_url` > client `base_url` >
    /// `http://127.0.0.1:8080`), the API key (`cfg.api_key` > client
    /// `api_key`), and the effective model (`cfg.model` overrides the
    /// request's). The [`LlamaRoute`] then selects the adapter: `Chat` posts
    /// `/v1/chat/completions` (the `/v1` suffix is appended when missing),
    /// `Completion` posts the native `/completion`. Called by `generate_stream`
    /// under the `provider-llamacpp` feature.
    pub(crate) async fn dispatch_llamacpp(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let mut cfg = self.llamacpp_config().cloned().unwrap_or_default();
        let base = if !cfg.base_url.is_empty() {
            cfg.base_url.clone()
        } else if !self.base_url().is_empty() {
            self.base_url().to_string()
        } else {
            "http://127.0.0.1:8080".to_string()
        };
        let api_key = cfg
            .api_key
            .clone()
            .or_else(|| self.api_key().map(str::to_string));
        if cfg.model.is_empty() {
            cfg.model = req.model.clone();
        }

        match cfg.route {
            LlamaRoute::Chat => {
                // The shared chat adapter appends /chat/completions to the
                // base URL, which must therefore end in /v1; append it here so
                // one configured base serves both routes.
                let base = base.trim_end_matches('/');
                let base = if base.ends_with("/v1") {
                    base.to_string()
                } else {
                    format!("{base}/v1")
                };
                let config = OpenAiCompatConfig {
                    base_url: base,
                    api_key,
                    model: cfg.model,
                };
                openai_compat_stream(self.http_client(), &config, req).await
            }
            LlamaRoute::Completion => {
                llamacpp_completion_stream(self.http_client(), &base, api_key.as_deref(), req, &cfg)
                    .await
            }
        }
    }
}

#[cfg(all(test, feature = "provider-llamacpp"))]
mod tests {
    use std::sync::{Arc, mpsc};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_stream::StreamExt;

    use crate::error::PluginError;
    use crate::plugin::CucaPlugin;
    use crate::request::UnifiedResponse;
    use crate::types::UnifiedMessage;

    use super::*;

    /// A llama.cpp config with the given route, a non-empty model, and no
    /// runtime knobs.
    fn config(route: LlamaRoute) -> LlamaCppConfig {
        LlamaCppConfig {
            base_url: String::new(),
            api_key: None,
            model: "qwen2.5-coder".into(),
            n_threads: None,
            n_gpu_layers: None,
            flash_attn: false,
            route,
        }
    }

    #[test]
    fn completion_body_assembles_role_marked_prompt() {
        let req = UnifiedRequest::new("qwen2.5-coder")
            .add_system_message("you are a coding assistant")
            .add_user_message("write a sort")
            .add_message(UnifiedMessage::assistant("sure"));
        let body = build_completion_body(&req, &config(LlamaRoute::Completion));

        assert_eq!(
            body["prompt"],
            json!("you are a coding assistant\n### User:\nwrite a sort\n### Assistant:\nsure\n")
        );
    }

    #[test]
    fn completion_body_n_predict_defaults_to_128_and_streams() {
        let req = UnifiedRequest::new("qwen2.5-coder").add_user_message("hi");
        let body = build_completion_body(&req, &config(LlamaRoute::Completion));
        assert_eq!(body["n_predict"], json!(128));
        assert_eq!(body["stream"], json!(true));

        let req = UnifiedRequest::new("qwen2.5-coder")
            .add_user_message("hi")
            .set_max_tokens(512);
        let body = build_completion_body(&req, &config(LlamaRoute::Completion));
        assert_eq!(body["n_predict"], json!(512));
    }

    #[test]
    fn completion_body_includes_configured_knobs_only() {
        let mut cfg = config(LlamaRoute::Completion);
        cfg.n_threads = Some(8);
        cfg.n_gpu_layers = Some(24);
        cfg.flash_attn = true;
        let body = build_completion_body(&UnifiedRequest::new("m"), &cfg);
        assert_eq!(body["n_threads"], json!(8));
        assert_eq!(body["n_gpu_layers"], json!(24));
        assert_eq!(body["flash_attn"], json!(true));

        let cfg = config(LlamaRoute::Completion);
        let body = build_completion_body(&UnifiedRequest::new("m"), &cfg);
        assert!(body.get("n_threads").is_none());
        assert!(body.get("n_gpu_layers").is_none());
        assert!(body.get("flash_attn").is_none());
    }

    #[test]
    fn completion_body_model_overrides_request_model() {
        let req = UnifiedRequest::new("req-model").add_user_message("hi");
        let body = build_completion_body(&req, &config(LlamaRoute::Completion));
        assert_eq!(body["model"], json!("qwen2.5-coder"));
    }

    #[test]
    fn completion_translator_content_frame_to_text() {
        let translator = CompletionTranslator;
        assert_eq!(
            translator
                .translate(r#"{"content":"Hel","stop":false}"#)
                .unwrap(),
            Some(MessageContentBlock::Text("Hel".into()))
        );
    }

    #[test]
    fn completion_translator_empty_content_yields_none() {
        let translator = CompletionTranslator;
        assert_eq!(
            translator
                .translate(r#"{"content":"","stop":true}"#)
                .unwrap(),
            None
        );
        assert_eq!(translator.translate(r#"{"stop":true}"#).unwrap(), None);
    }

    #[test]
    fn completion_translator_error_frame_is_provider_error() {
        let translator = CompletionTranslator;
        let err = translator
            .translate(r#"{"error":"slot exhausted"}"#)
            .unwrap_err();
        match err {
            CucaError::Provider { provider, message } => {
                assert_eq!(provider, ProviderEndpoint::LlamaCpp);
                assert_eq!(message, "slot exhausted");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[test]
    fn completion_translator_malformed_json_is_json_error() {
        let translator = CompletionTranslator;
        let err = translator.translate("not json").unwrap_err();
        assert!(matches!(err, CucaError::Json { .. }));
    }

    #[test]
    fn completion_translate_sse_chunk_split_matches_whole_chunk() {
        let frames = [
            r#"data: {"content":"Hel","stop":false}"#,
            r#"data: {"content":"lo","stop":false}"#,
            r#"data: {"content":"","stop":true}"#,
        ];
        let wire: Vec<u8> = frames
            .iter()
            .flat_map(|f| format!("{f}\n\n").into_bytes())
            .collect();

        // Split three bytes into the second frame so both chunk boundaries and
        // frame accumulation are exercised.
        let boundary = frames[0].len() + 2 + 3;

        let collect = |chunks: &[&[u8]]| -> Vec<MessageContentBlock> {
            let mut parser = SseStreamParser::new();
            let translator = CompletionTranslator;
            let mut blocks = Vec::new();
            for chunk in chunks {
                for block in translate_sse(&mut parser, &translator, chunk)
                    .unwrap()
                    .into_iter()
                    .flatten()
                {
                    blocks.push(block);
                }
            }
            blocks
        };

        let whole = collect(&[&wire]);
        let split = collect(&[&wire[..boundary], &wire[boundary..]]);

        assert_eq!(whole, split);
        assert_eq!(
            whole,
            vec![
                MessageContentBlock::Text("Hel".into()),
                MessageContentBlock::Text("lo".into()),
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
            "recording-llamacpp"
        }

        fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
            self.tx.send(res.clone()).map_err(|_| {
                PluginError::Internal("recording channel closed before completion".into())
            })
        }
    }

    /// End-to-end chat route: a bare base URL (no `/v1`) must be suffixed so
    /// the stub sees `/v1/chat/completions`, and the plugin pipeline must run
    /// (blocks emitted, `on_response_complete` once).
    #[tokio::test]
    async fn chat_route_hits_v1_chat_completions_and_completes_plugin() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            // Read the request head and capture the request-line path.
            let mut buf = [0u8; 1024];
            let n = socket.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or_default()
                .to_string();
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in [
                r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                r#"data: [DONE]"#,
            ] {
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
            path
        });

        let (plugin, rx) = RecordingPlugin::new();
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::LlamaCpp)
            // No /v1 on purpose: the dispatch must append it for the chat route.
            .with_base_url(format!("http://127.0.0.1:{}", addr.port()))
            .with_llamacpp_config(LlamaCppConfig::default())
            .register_plugin(Arc::new(plugin) as Arc<dyn CucaPlugin>)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));

        let stream = client
            .generate_stream(UnifiedRequest::new("qwen2.5-coder").add_user_message("hi"))
            .await
            .unwrap_or_else(|e| panic!("generate_stream must succeed: {e}"));
        let mut blocks = Vec::new();
        let mut stream = stream;
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap_or_else(|e| panic!("stream block must be Ok: {e}")));
        }
        let path = server.await.unwrap();

        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(blocks, vec![MessageContentBlock::Text("Hi".into())]);

        // The completion hook fired exactly once, with the aggregated response.
        let completed: Vec<UnifiedResponse> = rx.try_iter().collect();
        assert_eq!(
            completed.len(),
            1,
            "on_response_complete must fire exactly once"
        );
        assert_eq!(completed[0].content, blocks);
        assert_eq!(completed[0].model, "qwen2.5-coder");
        assert_eq!(completed[0].provider, ProviderEndpoint::LlamaCpp);
    }

    /// End-to-end native route: the resolved base must post to `/completion`
    /// and raw-token frames must stream through as `Text` blocks.
    #[tokio::test]
    async fn completion_route_hits_native_endpoint_and_streams_tokens() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let n = socket.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or_default()
                .to_string();
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in [
                r#"data: {"content":"Hel","stop":false}"#,
                r#"data: {"content":"lo","stop":false}"#,
                r#"data: {"content":"","stop":true}"#,
            ] {
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
            path
        });

        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::LlamaCpp)
            .with_base_url(format!("http://127.0.0.1:{}", addr.port()))
            .with_llamacpp_config(LlamaCppConfig {
                route: LlamaRoute::Completion,
                ..Default::default()
            })
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));

        let stream = client
            .generate_stream(UnifiedRequest::new("qwen2.5-coder").add_user_message("hi"))
            .await
            .unwrap_or_else(|e| panic!("generate_stream must succeed: {e}"));
        let mut blocks = Vec::new();
        let mut stream = stream;
        while let Some(block) = stream.next().await {
            blocks.push(block.unwrap_or_else(|e| panic!("stream block must be Ok: {e}")));
        }
        let path = server.await.unwrap();

        assert_eq!(path, "/completion");
        assert_eq!(
            blocks,
            vec![
                MessageContentBlock::Text("Hel".into()),
                MessageContentBlock::Text("lo".into()),
            ]
        );
    }
}
