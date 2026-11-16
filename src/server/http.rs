//! The HTTP routes: the OpenAI-compatible API, health, metrics and the stats
//! API, plus the request-id/tracing and bearer-auth middleware.
//!
//! | Route | Notes |
//! | --- | --- |
//! | `GET /health` | `200 {"status":"ok"}`, or `503` until every model is loaded; never needs the API key |
//! | `GET /version` | the cuca version |
//! | `GET /v1/models` | the ready models, with vLLM's `max_model_len` |
//! | `POST /v1/chat/completions` | non-streaming JSON or SSE (`stream: true`) |
//! | `POST /cuca/v1/kv/export` | streams the K/V rows of a prompt to a decode node (see [`kv`](super::kv)) |
//! | `GET /metrics` | Prometheus text |
//! | `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `stats_api #true` |
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Extension, Path, Query, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
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
use crate::{CucaError, LocalGeneration, STAGE_COMPUTE_HEADER};

/// Largest accepted request body.
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Most stats streams open at once.
pub const MAX_STATS_STREAMS: usize = 64;
/// Events buffered between the engine and the socket of one chat stream.
pub const SSE_BUFFER: usize = 32;
/// Set by the balancer on a request routed to a decode node: the base URL of
/// the prefill node that should compute its prompt's K/V rows.
pub(crate) const PREFILL_HEADER: &str = "x-cuca-prefill";
/// Largest accepted body of a layer-stage call: the hidden states and K/V
/// rows of one forward pass, which `max_step_tokens` bounds.
pub const MAX_STAGE_BODY_BYTES: usize = 1 << 30;

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

    pub(crate) fn model_unavailable(slot: &ModelSlot) -> Self {
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
        .route("/cuca/v1/kv/export", post(super::kv::export))
        .route(
            "/cuca/v1/stage",
            post(stage_only).layer(DefaultBodyLimit::max(MAX_STAGE_BODY_BYTES)),
        )
        .route(
            "/cuca/v1/stage/{model}",
            post(stage_named).layer(DefaultBodyLimit::max(MAX_STAGE_BODY_BYTES)),
        )
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
            // A layer-stage worker serves stage calls only: keeping it out
            // of the list keeps a balancer from routing chat to it.
            slot.ready()
                .filter(|m| !m.model.is_stage())
                .map(|m| ModelCard::new(slot.name.to_string(), m.loaded_at, m.context_length))
        })
        .collect();
    Json(model_list(cards))
}

/// `POST /cuca/v1/stage`: one frame for the only model, a layer-stage worker.
async fn stage_only(
    State(state): State<Arc<ServerState>>,
    body: Bytes,
) -> Result<Response, ApiError> {
    stage_frame(&state, None, body).await
}

/// `POST /cuca/v1/stage/{model}`: one frame for the named model.
async fn stage_named(
    State(state): State<Arc<ServerState>>,
    Path(model): Path<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    stage_frame(&state, Some(&model), body).await
}

/// Run a coordinator's frame on the worker model and answer its reply frame.
/// A frame the worker cannot take because its configuration differs from the
/// coordinator's is a 409, the one status the coordinator reads as a
/// mismatch; any other failure is a 500.
async fn stage_frame(
    state: &ServerState,
    model: Option<&str>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let slot = state.registry.resolve(model).map_err(|e| match e {
        ResolveError::Missing => {
            ApiError::invalid_request("the model is part of the path: several models are loaded")
                .param("model")
        }
        ResolveError::Unknown(name) => ApiError::model_not_found(&name),
    })?;
    let loaded = slot
        .ready()
        .ok_or_else(|| ApiError::model_unavailable(slot))?;
    match loaded.model.stage_call(body.to_vec()).await {
        Ok(reply) => {
            let compute = reply.compute.as_nanos().to_string();
            let mut response = Response::new(axum::body::Body::from(reply.frame));
            let headers = response.headers_mut();
            headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            if let (Ok(name), Ok(value)) = (
                HeaderName::from_bytes(STAGE_COMPUTE_HEADER.as_bytes()),
                HeaderValue::from_str(&compute),
            ) {
                headers.insert(name, value);
            }
            Ok(response)
        }
        Err(CucaError::Config(message)) => Err(ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "stage_mismatch",
            message,
        )),
        Err(error) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "stage_failed",
            error.to_string(),
        )),
    }
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
                stages: ready.map_or_else(Vec::new, |m| m.model.stage_stats()),
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
pub(crate) struct StreamQuery {
    interval_ms: Option<u64>,
}

/// Decrements the open-streams count when the stream ends.
struct StreamSlot<S>(Arc<S>, fn(&S) -> &AtomicUsize);

impl<S> Drop for StreamSlot<S> {
    fn drop(&mut self) {
        (self.1)(&self.0).fetch_sub(1, Ordering::Relaxed);
    }
}

/// The server-sent stream of `snapshot(&state)` every `query.interval_ms`,
/// counted in `open(&state)` and refused beyond [`MAX_STATS_STREAMS`] streams.
pub(crate) fn stats_events<S: Send + Sync + 'static, T: Serialize + 'static>(
    state: Arc<S>,
    open: fn(&S) -> &AtomicUsize,
    query: &StreamQuery,
    snapshot: fn(&S) -> T,
) -> Result<Response, ApiError> {
    if open(&state).fetch_add(1, Ordering::Relaxed) >= MAX_STATS_STREAMS {
        open(&state).fetch_sub(1, Ordering::Relaxed);
        return Err(ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "too_many_streams",
            format!("at most {MAX_STATS_STREAMS} stats streams at once"),
        ));
    }
    let slot = StreamSlot(Arc::clone(&state), open);
    let interval_ms = query.interval_ms.unwrap_or(1000).clamp(250, 10_000);
    let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let events = IntervalStream::new(interval).map(move |_| {
        let _open = &slot;
        let event = Event::default()
            .json_data(snapshot(&state))
            .unwrap_or_else(|_| Event::default().comment("snapshot could not be serialized"));
        Ok::<_, Infallible>(event)
    });
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn stats_stream(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<StreamQuery>,
) -> Result<Response, ApiError> {
    stats_events(state, |s| &s.stats_streams, &query, collector::snapshot)
}

/// The error for a request body that could not be read.
pub(crate) fn body_rejection(rejection: BytesRejection) -> ApiError {
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::new(
            rejection.status(),
            "invalid_request_error",
            "request_too_large",
            format!("the request body exceeds {MAX_BODY_BYTES} bytes"),
        )
    } else {
        ApiError::invalid_request(rejection.body_text())
    }
}

/// A chat request after parsing and model selection.
struct Prepared {
    slot: Arc<ModelSlot>,
    loaded: Arc<LoadedModel>,
    request: crate::UnifiedRequest,
    options: RequestOptions,
    stream: bool,
    /// The prefill node the balancer named (`x-cuca-prefill`).
    prefill: Option<String>,
    /// The request's `Authorization`, sent on to that peer.
    authorization: Option<HeaderValue>,
}

/// Parse the body, select the model, and map the request.
fn prepare(state: &ServerState, body: Result<Bytes, BytesRejection>) -> Result<Prepared, ApiError> {
    let body = body.map_err(body_rejection)?;
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
        prefill: None,
        authorization: None,
    })
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Extension(ctx): Extension<RequestCtx>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let mut prepared = match prepare(&state, body) {
        Ok(prepared) => prepared,
        Err(error) => {
            state.metrics.rejected.inc(error.status.as_u16());
            return Err(error);
        }
    };
    prepared.prefill = headers
        .get(PREFILL_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    prepared.authorization = headers.get(AUTHORIZATION).cloned();
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
        prefill,
        authorization,
    } = prepared;
    let model = slot.name.to_string();
    let (kv_state, kv_slot) = (Arc::clone(&state), Arc::clone(&slot));
    let tracker = RequestTracker::start(state, slot, ctx.id, ctx.span.clone(), ctx.arrived);
    // The prompt is tokenized here so that a decode node can fetch its K/V
    // rows from the prefill node the balancer named before the engine sees
    // the request. Without a `kv_import` section the header is ignored.
    let prompt = match loaded.model.tokenize(&request) {
        Ok(prompt) => prompt,
        Err(error) => {
            let error = ApiError::from_dispatch(&error);
            tracker.finish(error.status.as_u16(), error.outcome(), None);
            return Err(error);
        }
    };
    if let (Some(peer), Some(cfg)) = (&prefill, &kv_slot.kv_import) {
        super::kv::remote_prefill(
            &kv_state,
            &kv_slot,
            &loaded,
            cfg,
            peer,
            &prompt,
            authorization.as_ref(),
        )
        .await;
    }
    // Dispatch waits when the engine's queue is full; a client that goes
    // away meanwhile cancels this future and the tracker records it.
    let generation = match loaded.model.generate_tokens(request, prompt).await {
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
    use crate::LocalModelConfig;
    use crate::server::config::{ModelSpec, ServerConfig};
    use crate::server::registry::ModelRegistry;
    use crate::sysmon::SystemSampler;

    /// A server over models that never finish loading, listening on a free
    /// port; returns its base URL.
    async fn serve(names: &[&str], api_key: Option<&str>) -> String {
        let specs: Vec<ModelSpec> = names
            .iter()
            .map(|n| {
                ModelSpec::new(
                    Some(n.to_string()),
                    LocalModelConfig {
                        source: "qwen3-0.6b-q8_0".parse().unwrap(),
                        ..Default::default()
                    },
                )
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
            crate::server::state::peer_client().unwrap(),
        ));
        tokio::spawn(async move {
            let _ = axum::serve(listener, router(state)).await;
        });
        format!("http://{addr}")
    }

    async fn get(url: &str, key: Option<&str>) -> reqwest::Response {
        let mut req = crate::tls::http_client().get(url);
        if let Some(key) = key {
            req = req.bearer_auth(key);
        }
        req.send().await.unwrap()
    }

    async fn post_chat(base: &str, body: &str) -> reqwest::Response {
        crate::tls::http_client()
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
        let r = crate::tls::http_client()
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

    /// A server over the tiny model `build` makes, named `m`, listening on a
    /// free port; returns its base URL and the model's fingerprint.
    async fn tiny_node(
        path: &std::path::Path,
        api_key: Option<&str>,
        build: impl FnOnce(&std::path::Path) -> crate::provider::local::engine::Engine + Send + 'static,
    ) -> (String, u64) {
        use crate::provider::local::engine::testing::tiny_local_model;
        let model = tiny_local_model(path.to_path_buf(), 96, build).await;
        let fingerprint = model.fingerprint();
        let spec = ModelSpec::new(Some("m".into()), LocalModelConfig::default());
        let mut config = ServerConfig::new("127.0.0.1:0".parse().unwrap(), vec![spec.clone()]);
        config.api_key = api_key.map(str::to_string);
        let registry = ModelRegistry::ready_spec(&spec, model);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let state = Arc::new(ServerState::new(
            &config,
            addr,
            registry,
            SystemSampler::start(Duration::from_secs(1)),
            crate::server::state::peer_client().unwrap(),
        ));
        tokio::spawn(async move {
            let _ = axum::serve(listener, router(state)).await;
        });
        (format!("http://{addr}"), fingerprint)
    }

    async fn chat_text(base: &str) -> Value {
        let response = post_chat(
            base,
            &json!({
                "model": "m",
                "messages": [{"role": "user", "content": "abcdefgabcdefgabcdefg"}],
                "temperature": 0,
                "max_tokens": 4,
            })
            .to_string(),
        )
        .await;
        assert_eq!(response.status(), 200);
        response.json().await.unwrap()
    }

    /// A coordinator whose first layer runs on a worker server answers what
    /// one process does; the worker lists no model, refuses chat, and turns
    /// a frame for a different configuration away with a 409.
    #[tokio::test]
    async fn a_coordinator_over_a_worker_server_answers_like_one_process() {
        use crate::RemoteStage;
        use crate::provider::local::engine::testing::{
            engine_at, http_chain_at, stage_worker_layers_at, tiny_llama_file, tiny_sizing,
        };
        use crate::provider::local::stageframe::{Open, open_frame};

        let file = tiny_llama_file(96);
        let (worker, fingerprint) = tiny_node(&file.path, Some("stage-key"), |path| {
            stage_worker_layers_at(path, 96, 0, 1)
        })
        .await;
        let stage = RemoteStage {
            url: worker.clone(),
            layers: 1,
            api_key: Some("stage-key".into()),
            ..RemoteStage::default()
        };
        let (coordinator, _) = tiny_node(&file.path, None, move |path| {
            http_chain_at(path, 96, &[stage])
        })
        .await;
        let (single, _) = tiny_node(&file.path, None, |path| engine_at(path, 96)).await;

        let want = chat_text(&single).await;
        let got = chat_text(&coordinator).await;
        assert_eq!(
            got["choices"][0]["message"]["content"],
            want["choices"][0]["message"]["content"]
        );
        assert_eq!(got["usage"], want["usage"]);
        // A second turn of the same prompt reuses the cache on both sides.
        assert_eq!(
            chat_text(&coordinator).await["usage"],
            chat_text(&single).await["usage"]
        );

        // The coordinator counted its calls, and the compute time came back
        // in the worker's header.
        let metrics = get(&format!("{coordinator}/metrics"), None)
            .await
            .text()
            .await
            .unwrap();
        let value = |series: &str| -> f64 {
            metrics
                .lines()
                .find_map(|l| l.strip_prefix(series))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("{series} is missing:\n{metrics}"))
        };
        assert!(value("cuca_stage_calls_total{model=\"m\",stage=\"0\"} ") > 1.0);
        assert_eq!(
            value("cuca_stage_errors_total{model=\"m\",stage=\"0\"} "),
            0.0
        );
        assert!(value("cuca_stage_seconds_total{model=\"m\",stage=\"0\",part=\"compute\"} ") > 0.0);
        assert!(value("cuca_stage_seconds_total{model=\"m\",stage=\"0\",part=\"network\"} ") > 0.0);
        assert!(value("cuca_stage_bytes_total{model=\"m\",stage=\"0\",direction=\"sent\"} ") > 0.0);
        assert!(
            value("cuca_stage_bytes_total{model=\"m\",stage=\"0\",direction=\"received\"} ") > 0.0
        );

        let models: Value = get(&format!("{worker}/v1/models"), Some("stage-key"))
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(models["data"], json!([]));
        let models: Value = get(&format!("{coordinator}/v1/models"), None)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(models["data"][0]["id"], "m");

        // Chat to a worker is a client error, not a 500.
        let refused = crate::tls::http_client()
            .post(format!("{worker}/v1/chat/completions"))
            .bearer_auth("stage-key")
            .json(&json!({"model": "m", "messages": [{"role": "user", "content": "abcd"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(refused.status(), 400);
        let body: Value = refused.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("layer stage"),
            "{body}"
        );

        // The stage route needs the key like the others.
        let post_frame = |key: Option<&str>, frame: Vec<u8>| {
            let mut request = crate::tls::http_client()
                .post(format!("{worker}/cuca/v1/stage"))
                .header("content-type", "application/octet-stream")
                .body(frame);
            if let Some(key) = key {
                request = request.bearer_auth(key);
            }
            request.send()
        };
        let (sizing, _) = tiny_sizing(96, 1, None, None);
        let open = Open {
            fingerprint,
            first: 0,
            end: 1,
            context_length: 95,
            max_sequences: 1,
            kv_tokens: sizing.kv_tokens as u64,
            max_rows: sizing.max_rows as u64,
            session: 9,
        };
        assert_eq!(
            post_frame(None, open_frame(&open)).await.unwrap().status(),
            401
        );
        let mismatch = post_frame(Some("stage-key"), open_frame(&open))
            .await
            .unwrap();
        assert_eq!(mismatch.status(), 409);
        let body: Value = mismatch.json().await.unwrap();
        assert_eq!(body["error"]["code"], "stage_mismatch");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("context_length"),
            "{body}"
        );

        // The stage route has its own, larger body limit.
        let big = post_frame(Some("stage-key"), vec![0u8; MAX_BODY_BYTES + 1])
            .await
            .unwrap();
        assert_ne!(big.status(), 413);
    }

    /// A TCP proxy to `target` that, once `armed`, cuts both connections
    /// when the worker's next answer bytes arrive, instead of relaying them:
    /// a worker that ran the frame and went away before its answer reached
    /// the caller. Returns the proxy's base URL.
    async fn lossy_proxy(target: SocketAddr, armed: Arc<std::sync::atomic::AtomicBool>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut client, _)) = listener.accept().await else {
                    return;
                };
                let armed = Arc::clone(&armed);
                tokio::spawn(async move {
                    let Ok(mut upstream) = tokio::net::TcpStream::connect(target).await else {
                        return;
                    };
                    let (mut from_client, mut to_client) = client.split();
                    let (mut from_worker, mut to_worker) = upstream.split();
                    let requests = tokio::io::copy(&mut from_client, &mut to_worker);
                    let answers = async {
                        let mut buf = [0u8; 16 * 1024];
                        loop {
                            let n = from_worker.read(&mut buf).await.unwrap_or(0);
                            if n == 0 || armed.swap(false, Ordering::SeqCst) {
                                break;
                            }
                            if to_client.write_all(&buf[..n]).await.is_err() {
                                break;
                            }
                        }
                    };
                    tokio::select! {
                        _ = requests => {}
                        () = answers => {}
                    }
                });
            }
        });
        format!("http://{addr}")
    }

    /// An answer lost between the worker and the coordinator after the
    /// worker ran the frame: the coordinator sends the frame again, the
    /// worker answers from memory, and the request goes on as if nothing
    /// happened. The retry shows in the metrics.
    #[tokio::test]
    async fn a_stage_answer_lost_on_the_wire_is_retried_and_the_reply_is_unchanged() {
        use crate::RemoteStage;
        use crate::provider::local::engine::testing::{
            engine_at, http_chain_at, stage_worker_layers_at, tiny_llama_file,
        };

        let file = tiny_llama_file(96);
        let (worker, _) = tiny_node(&file.path, Some("stage-key"), |path| {
            stage_worker_layers_at(path, 96, 0, 1)
        })
        .await;
        let worker_addr: SocketAddr = worker.trim_start_matches("http://").parse().unwrap();
        let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let proxy = lossy_proxy(worker_addr, Arc::clone(&armed)).await;
        let stage = RemoteStage {
            url: proxy,
            layers: 1,
            api_key: Some("stage-key".into()),
            ..RemoteStage::default()
        };
        let (coordinator, _) = tiny_node(&file.path, None, move |path| {
            http_chain_at(path, 96, &[stage])
        })
        .await;
        let (single, _) = tiny_node(&file.path, None, |path| engine_at(path, 96)).await;

        // A first turn over the intact wire, then a second with an answer
        // lost; the single process answers both as the reference.
        assert_eq!(
            chat_text(&coordinator).await["choices"],
            chat_text(&single).await["choices"]
        );
        armed.store(true, Ordering::SeqCst);
        let got = chat_text(&coordinator).await;
        let want = chat_text(&single).await;
        assert!(!armed.load(Ordering::SeqCst), "an answer was cut");
        assert_eq!(got["choices"], want["choices"]);
        assert_eq!(got["usage"], want["usage"]);

        let metrics = get(&format!("{coordinator}/metrics"), None)
            .await
            .text()
            .await
            .unwrap();
        let value = |series: &str| -> f64 {
            metrics
                .lines()
                .find_map(|l| l.strip_prefix(series))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("{series} is missing:\n{metrics}"))
        };
        assert_eq!(
            value("cuca_stage_retries_total{model=\"m\",stage=\"0\"} "),
            1.0
        );
        assert_eq!(
            value("cuca_stage_errors_total{model=\"m\",stage=\"0\"} "),
            0.0
        );
    }
}
