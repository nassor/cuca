//! The balancer's HTTP routes and the proxy.
//!
//! | Route | Notes |
//! | --- | --- |
//! | `POST /v1/chat/completions` | routed to an upstream (see `routing`); the upstream's response streams through unchanged |
//! | `GET /health` | `200` when at least one upstream is healthy, else `503`; lists every upstream |
//! | `GET /v1/models` | the union of the healthy upstreams' models, each with the upstreams serving it |
//! | `GET /version`, `GET /metrics` | as on a model server; the metrics are the balancer's |
//! | `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `--stats-api` (`role: balancer`) |
//!
//! # Proxying
//!
//! The balancer parses the request body only to read `model` and the prompt
//! prefix, and forwards the original bytes. The upstream's status, headers
//! (minus hop-by-hop ones) and body stream back as they arrive, with no
//! buffering. A request is retried on the next candidate (at most
//! [`MAX_RETRIES`] times) only when it never reached an upstream (connect
//! error) or the upstream answered 503 before any body byte; either ejects
//! that upstream until its next successful poll. Any other 5xx ejects the
//! upstream too but is passed through. A client that disconnects drops the
//! upstream response, which closes the upstream connection, so the upstream's
//! engine stops generating.
//!
//! Bounds: bodies are limited to [`MAX_BODY_BYTES`]; at most
//! [`MAX_STATS_STREAMS`] stats streams are open; nothing else grows with
//! traffic (counters, histograms and the bounded latency rings).

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Extension, Query, Request, State};
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_core::Stream;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::IntervalStream;
use tracing::Span;

use super::metrics as balancer_metrics;
use super::routing::{Decision, NoCandidate, prefix_key, resolve_model};
use super::snapshot;
use super::state::{BalancerState, as_views, decision_slot};
use super::upstream::{Upstream, short_error};
use crate::server::http::{ApiError, RequestCtx, bearer_matches, trace_request};
use crate::server::latency::{Outcome, RequestRecord};

/// Largest accepted request body.
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Most stats streams open at once.
pub const MAX_STATS_STREAMS: usize = 64;
/// Most extra attempts after the first.
pub const MAX_RETRIES: u32 = 2;

/// Headers that describe one connection, not the message.
const HOP_BY_HOP: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// The router of a balancer over `state`.
pub fn router(state: Arc<BalancerState>) -> Router {
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

async fn observe(State(state): State<Arc<BalancerState>>, req: Request, next: Next) -> Response {
    trace_request(req, next, |status| state.metrics.http.inc(status)).await
}

async fn auth(State(state): State<Arc<BalancerState>>, req: Request, next: Next) -> Response {
    match &state.api_key {
        Some(key) if req.uri().path() != "/health" && !bearer_matches(&req, key) => {
            ApiError::unauthorized().into_response()
        }
        _ => next.run(req).await,
    }
}

async fn not_found() -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "not_found",
        "no such route".into(),
    )
}

fn no_healthy(message: String) -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "unavailable_error",
        "no_healthy_upstream",
        message,
    )
}

fn upstream_error(upstream: &Upstream, error: &reqwest::Error) -> ApiError {
    ApiError::new(
        StatusCode::BAD_GATEWAY,
        "server_error",
        "upstream_error",
        format!("upstream {} failed: {}", upstream.id, short_error(error)),
    )
}

async fn health(State(state): State<Arc<BalancerState>>) -> Response {
    let now = Instant::now();
    let upstreams: Vec<Value> = state
        .upstreams
        .iter()
        .map(|u| {
            json!({
                "id": &*u.id,
                "url": u.url,
                "healthy": state.is_healthy(u, now),
                "error": u.error(),
            })
        })
        .collect();
    let healthy = upstreams.iter().filter(|u| u["healthy"] == true).count();
    if healthy > 0 {
        Json(json!({
            "status": "ok",
            "healthy": healthy,
            "total": upstreams.len(),
            "upstreams": upstreams,
        }))
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "code": 503,
                    "message": "no healthy upstream",
                    "type": "unavailable_error",
                },
                "healthy": 0,
                "total": upstreams.len(),
                "upstreams": upstreams,
            })),
        )
            .into_response()
    }
}

async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION"), "role": "balancer" }))
}

async fn models(State(state): State<Arc<BalancerState>>) -> Json<Value> {
    let now = Instant::now();
    // id -> (created, max_model_len, upstreams serving it)
    let mut union: BTreeMap<String, (u64, u64, Vec<String>)> = BTreeMap::new();
    for u in state.upstreams.iter().filter(|u| state.is_healthy(u, now)) {
        for m in u.model_info() {
            let entry = union
                .entry(m.id)
                .or_insert((m.created, m.max_model_len, Vec::new()));
            entry.0 = entry.0.min(m.created);
            entry.1 = entry.1.max(m.max_model_len);
            entry.2.push(u.id.to_string());
        }
    }
    let data: Vec<Value> = union
        .into_iter()
        .map(|(id, (created, max_model_len, upstreams))| {
            json!({
                "id": id,
                "object": "model",
                "created": created,
                "owned_by": "cuca",
                "max_model_len": max_model_len,
                "upstreams": upstreams,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn prometheus(State(state): State<Arc<BalancerState>>) -> Response {
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        balancer_metrics::render(&state),
    )
        .into_response()
}

async fn stats(State(state): State<Arc<BalancerState>>) -> Json<Value> {
    Json(serde_json::to_value(snapshot::build(&state)).unwrap_or(Value::Null))
}

#[derive(Deserialize)]
struct StreamQuery {
    interval_ms: Option<u64>,
}

/// Decrements the open-streams count when the stream ends.
struct StreamSlot(Arc<BalancerState>);

impl Drop for StreamSlot {
    fn drop(&mut self) {
        self.0.stats_streams.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn stats_stream(
    State(state): State<Arc<BalancerState>>,
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
            .json_data(snapshot::build(&state))
            .unwrap_or_else(|_| Event::default().comment("snapshot could not be serialized"));
        Ok::<_, Infallible>(event)
    });
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// What the balancer reads of a chat body.
#[derive(Deserialize)]
struct Peek {
    model: Option<String>,
    #[serde(default)]
    messages: Vec<Value>,
}

async fn chat_completions(
    State(state): State<Arc<BalancerState>>,
    Extension(ctx): Extension<RequestCtx>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let routed = match route(&state, body) {
        Ok(routed) => routed,
        Err(error) => {
            state.metrics.rejected.inc(error.status().as_u16());
            return Err(error);
        }
    };
    forward(state, ctx, headers, routed).await
}

/// A chat request after parsing and model resolution.
struct Routed {
    body: Bytes,
    model: Arc<str>,
    key: Option<String>,
}

fn route(state: &BalancerState, body: Result<Bytes, BytesRejection>) -> Result<Routed, ApiError> {
    let body = body.map_err(|rejection| {
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
    })?;
    let peek: Peek = serde_json::from_slice(&body)
        .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))?;
    let owned = state.views();
    let views = as_views(&owned);
    let (model, _) =
        resolve_model(peek.model.as_deref(), &views, state.warmed_up()).map_err(|e| match e {
            NoCandidate::MissingModel => {
                ApiError::invalid_request("`model` is required: the upstreams serve several models")
                    .param("model")
            }
            NoCandidate::UnknownModel(name) => ApiError::model_not_found(&name),
            NoCandidate::NoneHealthy => no_healthy(match peek.model.as_deref() {
                Some(name) => format!("no healthy upstream serves the model `{name}`"),
                None => "no healthy upstream".to_string(),
            }),
        })?;
    Ok(Routed {
        key: prefix_key(&peek.messages),
        model: Arc::from(model),
        body,
    })
}

/// Records one proxied request, however it ends: counters, histograms, the
/// latency rings, the `proxy` span and one `info` line. Dropped before
/// [`finish`](Self::finish) (the client went away) it records a 499.
struct ProxyTracker {
    state: Arc<BalancerState>,
    upstream: Arc<Upstream>,
    model: Arc<str>,
    request_id: Arc<str>,
    span: Span,
    arrived: Instant,
    first_byte: Option<Instant>,
    decision: Decision,
    retries: u32,
    done: bool,
    /// The response is `text/event-stream`: events are counted.
    sse: bool,
    newlines: u8,
    counted: bool,
    events: u32,
    first_event: Option<Instant>,
    last_event: Option<Instant>,
}

impl ProxyTracker {
    fn start(
        state: Arc<BalancerState>,
        upstream: Arc<Upstream>,
        model: Arc<str>,
        ctx: &RequestCtx,
        span: Span,
        decision: Decision,
        retries: u32,
    ) -> Self {
        upstream.in_flight.fetch_add(1, Ordering::Relaxed);
        upstream.routed.fetch_add(1, Ordering::Relaxed);
        state.metrics.decisions[decision_slot(decision)].fetch_add(1, Ordering::Relaxed);
        if decision == Decision::Fallback {
            state.metrics.fallbacks.fetch_add(1, Ordering::Relaxed);
        }
        span.record("upstream", &*upstream.id);
        span.record("decision", decision.as_str());
        span.record("retries", retries);
        Self {
            state,
            upstream,
            model,
            request_id: Arc::clone(&ctx.id),
            span,
            arrived: ctx.arrived,
            first_byte: None,
            decision,
            retries,
            done: false,
            sse: false,
            newlines: 0,
            counted: false,
            events: 0,
            first_event: None,
            last_event: None,
        }
    }

    fn first_byte(&mut self) {
        self.first_byte.get_or_insert_with(Instant::now);
    }

    /// Count the SSE events in a forwarded chunk (a blank line ends one) and
    /// note when they arrived. Chunks are inspected as they pass, never
    /// buffered. `data: [DONE]` is not an event of the reply.
    fn observe(&mut self, chunk: &[u8]) {
        if !self.sse {
            return;
        }
        let mut events = 0u32;
        for &b in chunk {
            match b {
                // Two newlines in a row end an event; more are padding.
                b'\n' => {
                    self.newlines = (self.newlines + 1).min(2);
                    if self.newlines == 2 && !self.counted {
                        events += 1;
                        self.counted = true;
                    }
                }
                b'\r' => {}
                _ => {
                    self.newlines = 0;
                    self.counted = false;
                }
            }
        }
        if memchr::memmem::find(chunk, b"[DONE]").is_some() {
            events = events.saturating_sub(1);
        }
        if events > 0 {
            let now = Instant::now();
            self.events += events;
            if self.events >= 2 {
                self.first_event.get_or_insert(now);
            }
            self.last_event = Some(now);
        }
    }

    fn finish(&mut self, status: u16, outcome: Outcome) {
        if std::mem::replace(&mut self.done, true) {
            return;
        }
        let now = Instant::now();
        let u = &self.upstream;
        let e2e = now.duration_since(self.arrived).as_secs_f64();
        let ttft = self
            .first_byte
            .map(|t| t.duration_since(self.arrived).as_secs_f64());
        // Balancer-side inter-token latency of a streamed reply: the mean gap
        // between SSE events as they pass through, from the first content
        // event (the first event is the role prelude, sent before prefill) to
        // the last. An approximation of the engine's per-token gap: a chunk
        // may carry several events, and the usage chunk counts as one.
        let itl = match (self.first_event, self.last_event) {
            (Some(first), Some(last)) if self.events > 2 => {
                Some(last.duration_since(first).as_secs_f64() / f64::from(self.events - 2))
            }
            _ => None,
        };
        u.in_flight.fetch_sub(1, Ordering::Relaxed);
        u.requests.inc(status);
        if outcome == Outcome::ServerError {
            u.errors.fetch_add(1, Ordering::Relaxed);
        }
        if outcome == Outcome::Ok {
            u.e2e.observe(e2e);
            self.state.metrics.e2e.observe(e2e);
            if let Some(t) = ttft {
                u.ttft.observe(t);
                self.state.metrics.ttft.observe(t);
            }
        }
        let record = RequestRecord {
            finished: now,
            model: Arc::clone(&self.model),
            outcome,
            queue_s: None,
            ttft_s: ttft,
            itl_s: itl,
            e2e_s: e2e,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            stall_s: 0.0,
        };
        u.latency.record(record.clone());
        self.state.latency.record(record);
        let ms = |s: f64| (s * 1000.0 * 10.0).round() / 10.0;
        self.span.record("status", status);
        if let Some(t) = ttft {
            self.span.record("ttft_ms", ms(t));
        }
        let _entered = self.span.enter();
        tracing::info!(
            target: "cuca::access",
            request_id = %self.request_id,
            model = %self.model,
            upstream = %u.id,
            decision = self.decision.as_str(),
            retries = self.retries,
            status,
            outcome = ?outcome,
            ttft_ms = ttft.map(ms),
            e2e_ms = ms(e2e),
            "proxied request finished"
        );
    }
}

impl Drop for ProxyTracker {
    fn drop(&mut self) {
        self.finish(499, Outcome::Cancelled);
    }
}

fn outcome_of(status: StatusCode) -> Outcome {
    if status.is_server_error() {
        Outcome::ServerError
    } else if status.is_client_error() {
        Outcome::ClientError
    } else {
        Outcome::Ok
    }
}

/// The upstream response body on its way to the client; the tracker records
/// the first byte and the end.
struct ProxyBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    tracker: Option<ProxyTracker>,
    status: StatusCode,
    /// Bytes still due when the upstream declared a `content-length`.
    remaining: Option<u64>,
}

impl Stream for ProxyBody {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(t) = &mut this.tracker {
                    t.first_byte();
                    t.observe(&chunk);
                }
                // hyper stops polling once a declared content-length is
                // delivered, so the end is detected by counting.
                if let Some(remaining) = &mut this.remaining {
                    *remaining = remaining.saturating_sub(chunk.len() as u64);
                    if *remaining == 0
                        && let Some(mut t) = this.tracker.take()
                    {
                        t.finish(this.status.as_u16(), outcome_of(this.status));
                    }
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(mut t) = this.tracker.take() {
                    t.finish(502, Outcome::ServerError);
                }
                Poll::Ready(Some(Err(std::io::Error::other(error))))
            }
            Poll::Ready(None) => {
                if let Some(mut t) = this.tracker.take() {
                    t.finish(this.status.as_u16(), outcome_of(this.status));
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// The response headers worth passing on.
fn passthrough_headers(from: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::with_capacity(from.len());
    for (name, value) in from {
        if !HOP_BY_HOP.contains(&name.as_str()) && name.as_str() != "x-request-id" {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

/// Send the request to `upstream`.
async fn send(
    state: &BalancerState,
    upstream: &Upstream,
    headers: &HeaderMap,
    body: Bytes,
    request_id: &str,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = state
        .client
        .post(format!("{}/v1/chat/completions", upstream.url))
        .header(
            CONTENT_TYPE,
            headers
                .get(CONTENT_TYPE)
                .cloned()
                .unwrap_or_else(|| HeaderValue::from_static("application/json")),
        )
        .header(HeaderName::from_static("x-request-id"), request_id)
        .body(body);
    if let Some(accept) = headers.get(ACCEPT) {
        req = req.header(ACCEPT, accept);
    }
    if let Some(key) = &state.upstream_key {
        req = req.bearer_auth(key);
    }
    req.send().await
}

/// Try the candidates in policy order until one takes the request.
async fn forward(
    state: Arc<BalancerState>,
    ctx: RequestCtx,
    headers: HeaderMap,
    routed: Routed,
) -> Result<Response, ApiError> {
    let span = tracing::info_span!(
        parent: &ctx.span,
        "proxy",
        upstream = tracing::field::Empty,
        decision = tracing::field::Empty,
        retries = tracing::field::Empty,
        status = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
    );
    let mut tried: Vec<usize> = Vec::new();
    let mut retries = 0u32;
    // The last 503 an upstream answered, returned when no candidate is left.
    let mut last_refusal: Option<Response> = None;
    let mut last_failure: Option<ApiError> = None;
    loop {
        let (index, decision) = {
            let candidates = state.candidates(&routed.model, &tried);
            match state
                .router
                .choose(&routed.model, routed.key.as_deref(), &candidates)
            {
                Some(choice) => choice,
                None => break,
            }
        };
        tried.push(index);
        let upstream = Arc::clone(&state.upstreams[index]);
        let mut tracker = ProxyTracker::start(
            Arc::clone(&state),
            Arc::clone(&upstream),
            Arc::clone(&routed.model),
            &ctx,
            span.clone(),
            decision,
            retries,
        );
        match send(&state, &upstream, &headers, routed.body.clone(), &ctx.id).await {
            Err(error) if error.is_connect() => {
                upstream.eject(short_error(&error));
                tracker.finish(502, Outcome::ServerError);
                last_failure = Some(upstream_error(&upstream, &error));
                if retries < MAX_RETRIES {
                    retries += 1;
                    state.metrics.retries.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                break;
            }
            Err(error) => {
                // The request may have reached the upstream: no retry.
                upstream.eject(short_error(&error));
                tracker.finish(502, Outcome::ServerError);
                return Err(upstream_error(&upstream, &error));
            }
            Ok(response) => {
                let status = response.status();
                if status.is_server_error() {
                    upstream.eject(format!("HTTP {}", status.as_u16()));
                }
                if status == StatusCode::SERVICE_UNAVAILABLE && retries < MAX_RETRIES {
                    let headers = passthrough_headers(response.headers());
                    let body = response.bytes().await.unwrap_or_default();
                    tracker.finish(503, Outcome::ServerError);
                    let mut refusal = Response::new(Body::from(body));
                    *refusal.status_mut() = status;
                    *refusal.headers_mut() = headers;
                    last_refusal = Some(refusal);
                    retries += 1;
                    state.metrics.retries.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let headers = passthrough_headers(response.headers());
                let remaining = response.content_length();
                tracker.sse = headers
                    .get(CONTENT_TYPE)
                    .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"));
                // An empty declared body is never polled: it is over now.
                let tracker = if remaining == Some(0) {
                    tracker.finish(status.as_u16(), outcome_of(status));
                    None
                } else {
                    Some(tracker)
                };
                let body = ProxyBody {
                    remaining,
                    inner: Box::pin(response.bytes_stream()),
                    tracker,
                    status,
                };
                let mut out = Response::new(Body::from_stream(body));
                *out.status_mut() = status;
                *out.headers_mut() = headers;
                return Ok(out);
            }
        }
    }
    if let Some(refusal) = last_refusal {
        return Ok(refusal);
    }
    Err(last_failure.unwrap_or_else(|| {
        no_healthy(format!(
            "no healthy upstream serves the model `{}`",
            routed.model
        ))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_by_hop_and_request_id_headers_are_not_passed_on() {
        let mut from = HeaderMap::new();
        from.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        from.insert("connection", HeaderValue::from_static("keep-alive"));
        from.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        from.insert("x-request-id", HeaderValue::from_static("up"));
        from.insert("cache-control", HeaderValue::from_static("no-cache"));
        let out = passthrough_headers(&from);
        let mut names: Vec<&str> = out.keys().map(HeaderName::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["cache-control", "content-type"]);
    }

    #[test]
    fn status_classes_map_to_outcomes() {
        assert_eq!(outcome_of(StatusCode::OK), Outcome::Ok);
        assert_eq!(outcome_of(StatusCode::NOT_FOUND), Outcome::ClientError);
        assert_eq!(outcome_of(StatusCode::BAD_GATEWAY), Outcome::ServerError);
    }
}
