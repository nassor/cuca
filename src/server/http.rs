//! The HTTP routes: the OpenAI-compatible API, health, metrics and the stats
//! API, plus the request-id/tracing and bearer-auth middleware.
//!
//! | Route | Notes |
//! | --- | --- |
//! | `GET /health` | `200 {"status":"ok"}`, or `503` until every model is loaded; never needs the API key |
//! | `GET /version` | the cuca version |
//! | `GET /v1/models` | the ready models, with vLLM's `max_model_len` |
//! | `POST /v1/chat/completions` | non-streaming JSON or SSE (`stream: true`) |
//! | `GET /metrics` | Prometheus text |
//! | `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `--stats-api` |
//!
//! Bounds: request bodies are limited to [`MAX_BODY_BYTES`]; at most
//! [`MAX_STATS_STREAMS`] stats streams are open at once; a chat stream
//! buffers at most [`SSE_BUFFER`] events between the engine and the socket
//! (a slow client then stalls only its own request); chat requests in flight
//! are bounded by the engine's queue, which makes further callers wait.
//!
//! A client that disconnects stops its generation: dropping the response body
//! drops the stream task's channel, which drops the engine request.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Extension, Query, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::{IntervalStream, ReceiverStream};
use tracing::{Instrument, Span};

use super::collector;
use super::latency::Outcome;
use super::metrics::{self, ModelView};
use super::openai::{
    ChatCompletionRequest, ChunkWriter, Collected, ModelCard, RequestOptions, Usage, completion_id,
    finish_reason, model_list, random_token,
};
use super::registry::{LoadedModel, ModelSlot, ResolveError, SlotStatus};
use super::state::{RequestTracker, ServerState};
use super::stats::ModelState;
use crate::{CucaError, LocalGeneration};

/// Largest accepted request body.
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Most stats streams open at once.
pub const MAX_STATS_STREAMS: usize = 64;
/// Events buffered between the engine and the socket of one chat stream.
pub const SSE_BUFFER: usize = 32;

/// An error answered as OpenAI's error body,
/// `{"error":{"message","type","param","code"}}`.
#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: &'static str,
    message: String,
    param: Option<String>,
}

impl ApiError {
    pub(crate) fn new(
        status: StatusCode,
        kind: &'static str,
        code: &'static str,
        message: String,
    ) -> Self {
        Self {
            status,
            kind,
            code,
            message,
            param: None,
        }
    }

    /// 400: the request is malformed or asks for something unsupported.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_request",
            message.into(),
        )
    }

    /// Name the request field the error is about.
    #[must_use]
    pub fn param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    pub(crate) fn model_not_found(name: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "model_not_found",
            format!("the model `{name}` does not exist"),
        )
        .param("model")
    }

    pub(crate) fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "missing or invalid API key: send `Authorization: Bearer <key>`".into(),
        )
    }

    fn model_unavailable(slot: &ModelSlot) -> Self {
        let message = match slot.status() {
            SlotStatus::Failed(error) => format!("model `{}` failed to load: {error}", slot.name),
            _ => format!("model `{}` is still loading", slot.name),
        };
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_error",
            "model_loading",
            message,
        )
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
            message.into(),
        )
    }

    /// The error of a request the engine refused at dispatch.
    fn from_dispatch(error: &CucaError) -> Self {
        match error {
            CucaError::Config(message) => Self::invalid_request(message.clone()),
            CucaError::Provider { message, .. } if message.contains("context_length is") => {
                Self::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "context_length_exceeded",
                    message.clone(),
                )
                .param("messages")
            }
            CucaError::Provider { message, .. } if message.contains("shut down") => Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable_error",
                "engine_unavailable",
                message.clone(),
            ),
            // Anything else at dispatch is the request failing to render
            // through the chat template.
            other => Self::invalid_request(other.to_string()),
        }
    }

    /// The error of a generation that failed after it started.
    fn from_generation(error: &CucaError) -> Self {
        Self::internal(error.to_string())
    }

    /// The status code.
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// The human-readable message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The JSON body.
    pub fn body(&self) -> Value {
        json!({ "error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code,
        }})
    }

    fn outcome(&self) -> Outcome {
        if self.status.is_server_error() {
            Outcome::ServerError
        } else {
            Outcome::ClientError
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(self.body())).into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

/// What the middleware knows about the request, for the handlers.
#[derive(Clone)]
pub(crate) struct RequestCtx {
    pub(crate) id: Arc<str>,
    pub(crate) span: Span,
    pub(crate) arrived: Instant,
}

/// The router of a server over `state`.
pub fn router(state: Arc<ServerState>) -> Router {
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/version", get(version))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/metrics", get(prometheus));
    if state.stats_api {
        app = app
            .route("/cuca/v1/stats", get(stats))
            .route("/cuca/v1/stats/stream", get(stats_stream));
    }
    app.fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), auth))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), observe))
        .with_state(state)
}

/// Trace the request and count its response: honour an incoming
/// `x-request-id` when it is short printable ASCII (otherwise make one), open
/// the `http` span, echo the id, and call `count` with the status.
pub(crate) async fn trace_request(
    mut req: Request,
    next: Next,
    count: impl FnOnce(u16),
) -> Response {
    let arrived = Instant::now();
    let id: Arc<str> = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 128 && v.bytes().all(|b| b.is_ascii_graphic()))
        .map_or_else(|| Arc::from(format!("req-{}", random_token(20))), Arc::from);
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let span = tracing::info_span!(
        "http",
        request_id = %id,
        method = %method,
        path = %path,
        model = tracing::field::Empty,
        status = tracing::field::Empty,
        prompt_tokens = tracing::field::Empty,
        completion_tokens = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
        e2e_ms = tracing::field::Empty,
    );
    req.extensions_mut().insert(RequestCtx {
        id: Arc::clone(&id),
        span: span.clone(),
        arrived,
    });
    let mut response = next.run(req).instrument(span.clone()).await;
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", value);
    }
    let status = response.status().as_u16();
    count(status);
    let latency_ms = arrived.elapsed().as_secs_f64() * 1000.0;
    // Probes and scrapes would drown the log at info.
    let quiet = matches!(path.as_str(), "/health" | "/metrics" | "/cuca/v1/stats");
    span.in_scope(|| {
        if quiet {
            tracing::debug!(target: "cuca::access", status, latency_ms, "request");
        } else {
            tracing::info!(target: "cuca::access", status, latency_ms, "response headers sent");
        }
    });
    response
}

async fn observe(State(state): State<Arc<ServerState>>, req: Request, next: Next) -> Response {
    trace_request(req, next, |status| state.metrics.http.inc(status)).await
}

/// Constant-time byte comparison (the length is not secret).
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Require `Authorization: Bearer <key>` on every route but `/health` when a
/// key is configured.
async fn auth(State(state): State<Arc<ServerState>>, req: Request, next: Next) -> Response {
    match &state.api_key {
        Some(key) if req.uri().path() != "/health" && !bearer_matches(&req, key) => {
            ApiError::unauthorized().into_response()
        }
        _ => next.run(req).await,
    }
}

/// Whether the request carries `Authorization: Bearer <key>`.
pub(crate) fn bearer_matches(req: &Request, key: &str) -> bool {
    req.headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .is_some_and(|(_, token)| constant_time_eq(token.trim().as_bytes(), key.as_bytes()))
}

async fn not_found() -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "not_found",
        "no such route".into(),
    )
}

async fn health(State(state): State<Arc<ServerState>>) -> Response {
    if state.registry.all_ready() {
        return Json(json!({ "status": "ok" })).into_response();
    }
    let failed = state
        .registry
        .slots()
        .iter()
        .find_map(|s| match s.status() {
            SlotStatus::Failed(e) => Some(format!("model `{}` failed to load: {e}", s.name)),
            _ => None,
        });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": {
            "code": 503,
            "message": failed.unwrap_or_else(|| "Loading model".to_string()),
            "type": "unavailable_error",
        }})),
    )
        .into_response()
}

async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

async fn models(State(state): State<Arc<ServerState>>) -> Json<Value> {
    let cards = state
        .registry
        .slots()
        .iter()
        .filter_map(|slot| {
            slot.ready()
                .map(|m| ModelCard::new(slot.name.to_string(), m.loaded_at, m.context_length))
        })
        .collect();
    Json(model_list(cards))
}

async fn prometheus(State(state): State<Arc<ServerState>>) -> Response {
    let statuses: Vec<(Arc<ModelSlot>, SlotStatus)> = state
        .registry
        .slots()
        .iter()
        .map(|s| (Arc::clone(s), s.status()))
        .collect();
    let views: Vec<ModelView<'_>> = statuses
        .iter()
        .map(|(slot, status)| {
            let ready = match status {
                SlotStatus::Ready(m) => Some(m),
                _ => None,
            };
            ModelView {
                name: &slot.name,
                state: match status {
                    SlotStatus::Loading => ModelState::Loading,
                    SlotStatus::Ready(_) => ModelState::Ready,
                    SlotStatus::Failed(_) => ModelState::Failed,
                },
                metrics: &slot.metrics,
                serving: ready.map(|m| m.model.serving_stats()),
                kv_bytes: ready.map_or(0, |m| m.kv_bytes),
                weights_bytes: ready.map_or(0, |m| m.weights_bytes),
            }
        })
        .collect();
    let text = metrics::render(
        &views,
        &state.metrics,
        &state.system.snapshot(),
        state.started.elapsed().as_secs_f64(),
    );
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        text,
    )
        .into_response()
}

async fn stats(State(state): State<Arc<ServerState>>) -> Json<Value> {
    Json(serde_json::to_value(collector::snapshot(&state)).unwrap_or(Value::Null))
}

#[derive(Deserialize)]
struct StreamQuery {
    interval_ms: Option<u64>,
}

/// Decrements the open-streams count when the stream ends.
struct StreamSlot(Arc<ServerState>);

impl Drop for StreamSlot {
    fn drop(&mut self) {
        self.0.stats_streams.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn stats_stream(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<StreamQuery>,
) -> Result<Response, ApiError> {
    if state.stats_streams.fetch_add(1, Ordering::Relaxed) >= MAX_STATS_STREAMS {
        state.stats_streams.fetch_sub(1, Ordering::Relaxed);
        return Err(ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "too_many_streams",
            format!("at most {MAX_STATS_STREAMS} stats streams at once"),
        ));
    }
    let slot = StreamSlot(Arc::clone(&state));
    let interval_ms = query.interval_ms.unwrap_or(1000).clamp(250, 10_000);
    let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let events = IntervalStream::new(interval).map(move |_| {
        let _open = &slot;
        let event = Event::default()
            .json_data(collector::snapshot(&state))
            .unwrap_or_else(|_| Event::default().comment("snapshot could not be serialized"));
        Ok::<_, Infallible>(event)
    });
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// A chat request after parsing and model selection.
struct Prepared {
    slot: Arc<ModelSlot>,
    loaded: Arc<LoadedModel>,
    request: crate::UnifiedRequest,
    options: RequestOptions,
    stream: bool,
}

/// Parse the body, select the model, and map the request.
fn prepare(state: &ServerState, body: Result<Bytes, BytesRejection>) -> Result<Prepared, ApiError> {
    let body = body.map_err(|rejection| {
        let status = rejection.status();
        if status == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(
                status,
                "invalid_request_error",
                "request_too_large",
                format!("the request body exceeds {MAX_BODY_BYTES} bytes"),
            )
        } else {
            ApiError::invalid_request(rejection.body_text())
        }
    })?;
    let parsed: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))?;
    let slot = state
        .registry
        .resolve(parsed.model.as_deref())
        .map_err(|e| match e {
            ResolveError::Missing => {
                ApiError::invalid_request("`model` is required: several models are loaded")
                    .param("model")
            }
            ResolveError::Unknown(name) => ApiError::model_not_found(&name),
        })?
        .clone();
    let loaded = slot
        .ready()
        .ok_or_else(|| ApiError::model_unavailable(&slot))?;
    let stream = parsed.stream;
    let (request, options) = parsed.into_unified(&slot.name)?;
    Ok(Prepared {
        slot,
        loaded,
        request,
        options,
        stream,
    })
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Extension(ctx): Extension<RequestCtx>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let prepared = match prepare(&state, body) {
        Ok(prepared) => prepared,
        Err(error) => {
            state.metrics.rejected.inc(error.status.as_u16());
            return Err(error);
        }
    };
    run(state, ctx, prepared).await
}

async fn run(
    state: Arc<ServerState>,
    ctx: RequestCtx,
    prepared: Prepared,
) -> Result<Response, ApiError> {
    let Prepared {
        slot,
        loaded,
        request,
        options,
        stream,
    } = prepared;
    let model = slot.name.to_string();
    let tracker = RequestTracker::start(state, slot, ctx.id, ctx.span.clone(), ctx.arrived);
    // Dispatch waits when the engine's queue is full; a client that goes
    // away meanwhile cancels this future and the tracker records it.
    let generation = match loaded.model.generate(request).await {
        Ok(generation) => generation,
        Err(error) => {
            let error = ApiError::from_dispatch(&error);
            tracker.finish(error.status.as_u16(), error.outcome(), None);
            return Err(error);
        }
    };
    let id = completion_id();
    if stream {
        let writer = ChunkWriter::new(id, model, options);
        let (tx, rx) = mpsc::channel(SSE_BUFFER);
        tokio::spawn(pump(generation, writer, tracker, tx).instrument(ctx.span));
        let events = ReceiverStream::new(rx);
        Ok(Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        collect(generation, tracker, id, model).await
    }
}

/// A non-streaming reply: drain the generation, then answer once.
async fn collect(
    mut generation: LocalGeneration,
    mut tracker: RequestTracker,
    id: String,
    model: String,
) -> Result<Response, ApiError> {
    let mut collected = Collected::default();
    while let Some(item) = generation.next().await {
        match item {
            Ok(block) => {
                tracker.block();
                collected.push(block);
            }
            Err(error) => {
                let error = ApiError::from_generation(&error);
                tracker.finish(
                    error.status.as_u16(),
                    Outcome::ServerError,
                    Some(&generation.report()),
                );
                return Err(error);
            }
        }
    }
    let report = generation.report();
    let Some(usage) = report.usage else {
        tracker.finish(500, Outcome::ServerError, Some(&report));
        return Err(ApiError::internal(
            "the engine ended the reply without its token usage",
        ));
    };
    let reason = finish_reason(collected.has_tool_calls(), report.finish_reason);
    let completion =
        collected.into_completion(id, model, reason, Usage::new(usage, report.prompt_cache));
    tracker.finish(200, Outcome::Ok, Some(&report));
    Ok(Json(completion).into_response())
}

/// Forward a generation to the SSE channel until it ends or the client goes.
async fn pump(
    mut generation: LocalGeneration,
    mut writer: ChunkWriter,
    mut tracker: RequestTracker,
    tx: mpsc::Sender<Result<Event, Infallible>>,
) {
    if !send(&tx, &mut tracker, writer.role()).await {
        return;
    }
    loop {
        let item = tokio::select! {
            biased;
            // The client hung up: drop the generation (the engine stops) and
            // the tracker (recorded as cancelled).
            () = tx.closed() => return,
            item = generation.next() => item,
        };
        match item {
            Some(Ok(block)) => {
                tracker.block();
                if let Some(chunk) = writer.block(block)
                    && !send(&tx, &mut tracker, chunk).await
                {
                    return;
                }
            }
            Some(Err(error)) => {
                let body = ApiError::from_generation(&error).body().to_string();
                let report = generation.report();
                let _ = send(&tx, &mut tracker, body).await;
                let _ = send(&tx, &mut tracker, "[DONE]".to_string()).await;
                tracker.finish(500, Outcome::ServerError, Some(&report));
                return;
            }
            None => break,
        }
    }
    let report = generation.report();
    let reason = finish_reason(writer.called_tool(), report.finish_reason);
    let mut open = send(&tx, &mut tracker, writer.finish(reason)).await;
    if open
        && writer.wants_usage()
        && let Some(usage) = report.usage
    {
        open = send(
            &tx,
            &mut tracker,
            writer.usage(&Usage::new(usage, report.prompt_cache)),
        )
        .await;
    }
    if open {
        send(&tx, &mut tracker, "[DONE]".to_string()).await;
        tracker.finish(200, Outcome::Ok, Some(&report));
    }
}

/// Queue one SSE `data:` event, timing how long the client made us wait.
/// `false` when the client is gone.
async fn send(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    tracker: &mut RequestTracker,
    data: String,
) -> bool {
    let started = Instant::now();
    let sent = tx.send(Ok(Event::default().data(data))).await.is_ok();
    tracker.stalled(started.elapsed());
    sent
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::*;
    use crate::server::config::{ModelOptions, ModelSpec, ServerConfig};
    use crate::server::registry::ModelRegistry;
    use crate::server::sysmon::SystemSampler;

    /// A server over models that never finish loading, listening on a free
    /// port; returns its base URL.
    async fn serve(names: &[&str], api_key: Option<&str>) -> String {
        let specs: Vec<ModelSpec> = names
            .iter()
            .map(|n| {
                ModelSpec::from_arg(&format!("{n}=qwen3-0.6b-q8_0"), &ModelOptions::default())
                    .unwrap()
            })
            .collect();
        let mut config = ServerConfig::new("127.0.0.1:0".parse().unwrap(), specs.clone());
        config.api_key = api_key.map(str::to_string);
        config.stats_api = true;
        let registry = ModelRegistry::pending(&specs);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let state = Arc::new(ServerState::new(
            &config,
            addr,
            registry,
            SystemSampler::start(Duration::from_secs(1)),
        ));
        tokio::spawn(async move {
            let _ = axum::serve(listener, router(state)).await;
        });
        format!("http://{addr}")
    }

    async fn get(url: &str, key: Option<&str>) -> reqwest::Response {
        let mut req = reqwest::Client::new().get(url);
        if let Some(key) = key {
            req = req.bearer_auth(key);
        }
        req.send().await.unwrap()
    }

    async fn post_chat(base: &str, body: &str) -> reqwest::Response {
        reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn health_is_503_while_a_model_loads_and_never_needs_the_key() {
        let base = serve(&["m"], Some("secret")).await;
        let r = get(&format!("{base}/health"), None).await;
        assert_eq!(r.status(), 503);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["message"], "Loading model");
    }

    #[tokio::test]
    async fn every_other_route_demands_the_bearer_key() {
        let base = serve(&["m"], Some("secret")).await;
        let denied = get(&format!("{base}/v1/models"), None).await;
        assert_eq!(denied.status(), 401);
        assert_eq!(denied.headers()["www-authenticate"], "Bearer");
        let body: Value = denied.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_api_key");
        assert_eq!(
            get(&format!("{base}/v1/models"), Some("wrong"))
                .await
                .status(),
            401
        );
        assert_eq!(
            get(&format!("{base}/v1/models"), Some("secret"))
                .await
                .status(),
            200
        );
        assert_eq!(get(&format!("{base}/metrics"), None).await.status(), 401);
    }

    #[tokio::test]
    async fn a_request_id_is_honoured_or_made() {
        let base = serve(&["m"], None).await;
        let r = reqwest::Client::new()
            .get(format!("{base}/version"))
            .header("x-request-id", "abc-123")
            .send()
            .await
            .unwrap();
        assert_eq!(r.headers()["x-request-id"], "abc-123");
        let made = get(&format!("{base}/version"), None).await;
        assert!(
            made.headers()["x-request-id"]
                .to_str()
                .unwrap()
                .starts_with("req-")
        );
    }

    #[tokio::test]
    async fn a_chat_request_to_a_loading_model_is_a_503_error_body() {
        let base = serve(&["m"], None).await;
        let r = post_chat(&base, r#"{"messages":[{"role":"user","content":"hi"}]}"#).await;
        assert_eq!(r.status(), 503);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "model_loading");
        assert_eq!(body["error"]["type"], "unavailable_error");
    }

    #[tokio::test]
    async fn with_several_models_an_unknown_name_is_404_and_a_missing_one_400() {
        let base = serve(&["a", "b"], None).await;
        let unknown = post_chat(&base, r#"{"model":"zzz","messages":[]}"#).await;
        assert_eq!(unknown.status(), 404);
        let body: Value = unknown.json().await.unwrap();
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
        let missing = post_chat(&base, r#"{"messages":[]}"#).await;
        assert_eq!(missing.status(), 400);
    }

    #[tokio::test]
    async fn a_malformed_body_is_a_400_openai_error() {
        let base = serve(&["m"], None).await;
        let r = post_chat(&base, "{not json").await;
        assert_eq!(r.status(), 400);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid request body")
        );
    }

    #[tokio::test]
    async fn the_stats_and_models_routes_answer_while_loading() {
        let base = serve(&["m"], None).await;
        let models: Value = get(&format!("{base}/v1/models"), None)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(
            models["data"],
            json!([]),
            "a model that is not ready is not listed"
        );
        let stats: Value = get(&format!("{base}/cuca/v1/stats"), None)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(stats["models"][0]["state"], "loading");
        assert_eq!(stats["node"]["role"], "server");
        assert_eq!(stats["diagnosis"][0]["cause"], "model_loading");
        let metrics = get(&format!("{base}/metrics"), None)
            .await
            .text()
            .await
            .unwrap();
        assert!(
            metrics.contains("cuca_model_ready{model=\"m\"} 0"),
            "{metrics}"
        );
    }

    #[test]
    fn the_key_comparison_rejects_a_different_length_and_a_different_byte() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secre"));
    }
}
