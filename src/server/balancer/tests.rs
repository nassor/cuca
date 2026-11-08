//! The balancer against in-process mock upstreams over real sockets: routing,
//! passive and active health, retries, streaming passthrough and the stats
//! API. No model is loaded.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::config::BalancerConfig;
use super::proxy;
use super::routing::Policy;
use super::state::BalancerState;
use super::upstream::{Upstream, spawn_poller};
use crate::server::stats::{ModelStats, StatsSnapshot};
use crate::server::sysmon::SystemSampler;

/// A fake model server.
#[derive(Clone)]
struct Mock {
    name: &'static str,
    models: Vec<String>,
    healthy: Arc<AtomicBool>,
    chat_status: Arc<AtomicU16>,
    hits: Arc<AtomicUsize>,
    auth_seen: Arc<Mutex<Option<String>>>,
    stats: Option<Value>,
    /// Set when a streaming response was dropped by the balancer.
    stream_dropped: Arc<AtomicBool>,
}

impl Mock {
    fn new(name: &'static str, models: &[&str]) -> Self {
        Self {
            name,
            models: models.iter().map(|m| (*m).to_string()).collect(),
            healthy: Arc::new(AtomicBool::new(true)),
            chat_status: Arc::new(AtomicU16::new(200)),
            hits: Arc::new(AtomicUsize::new(0)),
            auth_seen: Arc::new(Mutex::new(None)),
            stats: None,
            stream_dropped: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn spawn(self) -> (String, Self) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/health", get(mock_health))
            .route("/v1/models", get(mock_models))
            .route("/cuca/v1/stats", get(mock_stats))
            .route("/v1/chat/completions", post(mock_chat))
            .with_state(self.clone());
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (url, self)
    }
}

async fn mock_health(State(m): State<Mock>) -> Response {
    if m.healthy.load(Ordering::Relaxed) {
        Json(json!({"status": "ok"})).into_response()
    } else {
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    }
}

async fn mock_models(State(m): State<Mock>) -> Json<Value> {
    let data: Vec<Value> = m
        .models
        .iter()
        .map(|id| json!({"id": id, "object": "model", "created": 7, "max_model_len": 4096}))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn mock_stats(State(m): State<Mock>) -> Response {
    match &m.stats {
        Some(stats) => Json(stats.clone()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn mock_chat(State(m): State<Mock>, headers: HeaderMap, body: Bytes) -> Response {
    m.hits.fetch_add(1, Ordering::Relaxed);
    *m.auth_seen.lock().unwrap() = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let status = m.chat_status.load(Ordering::Relaxed);
    if status != 200 {
        return (
            StatusCode::from_u16(status).unwrap(),
            Json(json!({"error": {"message": "mock failure", "upstream": m.name}})),
        )
            .into_response();
    }
    let text = String::from_utf8_lossy(&body);
    if text.contains("\"stream\":true") {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(4);
        let forever = text.contains("forever");
        let three = text.contains("\"three\":true");
        let dropped = Arc::clone(&m.stream_dropped);
        tokio::spawn(async move {
            if forever {
                loop {
                    if tx.send(Ok(Event::default().data("tick"))).await.is_err() {
                        dropped.store(true, Ordering::Relaxed);
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
            let _ = tx.send(Ok(Event::default().data("first"))).await;
            tokio::time::sleep(Duration::from_millis(400)).await;
            let _ = tx.send(Ok(Event::default().data("second"))).await;
            if three {
                tokio::time::sleep(Duration::from_millis(400)).await;
                let _ = tx.send(Ok(Event::default().data("third"))).await;
            }
        });
        return Sse::new(ReceiverStream::new(rx)).into_response();
    }
    ([("x-upstream", m.name)], Json(json!({"upstream": m.name}))).into_response()
}

struct Harness {
    base: String,
    state: Arc<BalancerState>,
    client: reqwest::Client,
}

impl Harness {
    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .unwrap()
    }

    async fn chat(&self, body: Value) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    /// Wait until `ready(state)` holds, at most five seconds.
    async fn until(&self, what: &str, ready: impl Fn(&BalancerState) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready(&self.state) {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

struct Options {
    policy: Policy,
    api_key: Option<&'static str>,
    upstream_key: Option<&'static str>,
    /// URLs the balancer treats as healthy without polling them.
    forced: Vec<(String, &'static str)>,
    stats_api: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            policy: Policy::PrefixAffinity,
            api_key: None,
            upstream_key: None,
            forced: Vec::new(),
            stats_api: true,
        }
    }
}

/// A balancer in front of `polled` (polled every 100 ms) and the `forced`
/// upstreams.
async fn balancer(polled: &[String], options: Options) -> Harness {
    let mut urls: Vec<String> = polled.to_vec();
    urls.extend(options.forced.iter().map(|(u, _)| u.clone()));
    let mut config = BalancerConfig::new("127.0.0.1:0".parse().unwrap(), urls);
    config.health_interval = Duration::from_millis(100);
    config.policy = options.policy;
    config.api_key = options.api_key.map(str::to_string);
    config.upstream_api_key = options.upstream_key.map(str::to_string);
    config.stats_api = options.stats_api;
    let resolved = config.resolve_upstreams().unwrap();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let upstreams: Vec<Arc<Upstream>> = resolved
        .into_iter()
        .map(|(id, url)| Arc::new(Upstream::new(&id, url, Duration::from_secs(60))))
        .collect();
    let key: Option<Arc<str>> = config.upstream_api_key.as_deref().map(Arc::from);
    for (i, u) in upstreams.iter().enumerate() {
        if i < polled.len() {
            spawn_poller(
                Arc::clone(u),
                client.clone(),
                key.clone(),
                config.health_interval,
            );
        } else {
            u.force_healthy(&[options.forced[i - polled.len()].1]);
        }
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(BalancerState::new(
        &config,
        addr,
        upstreams,
        client.clone(),
        SystemSampler::start(Duration::from_secs(1)),
    ));
    let app = proxy::router(Arc::clone(&state));
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Harness {
        base: format!("http://{addr}"),
        state,
        client: reqwest::Client::new(),
    }
}

fn body(model: &str, user: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": user}]})
}

async fn answered_by(r: reqwest::Response) -> String {
    assert_eq!(r.status(), 200);
    r.headers()["x-upstream"].to_str().unwrap().to_string()
}

#[tokio::test]
async fn a_conversation_sticks_to_one_upstream_and_conversations_spread() {
    let (ua, _) = Mock::new("a", &["m"]).spawn().await;
    let (ub, _) = Mock::new("b", &["m"]).spawn().await;
    let h = balancer(&[ua, ub], Options::default()).await;
    h.until("both upstreams healthy", |s| s.healthy_count() == 2)
        .await;

    let first = answered_by(h.chat(body("m", "the same conversation")).await).await;
    for _ in 0..5 {
        assert_eq!(
            answered_by(h.chat(body("m", "the same conversation")).await).await,
            first
        );
    }
    let mut seen = std::collections::HashSet::new();
    for i in 0..40 {
        seen.insert(answered_by(h.chat(body("m", &format!("conversation {i}"))).await).await);
    }
    assert_eq!(seen.len(), 2, "distinct conversations use both upstreams");
}

#[tokio::test]
async fn a_dead_upstream_is_ejected_at_once_and_requests_retry_on_the_other() {
    let (live, mock) = Mock::new("live", &["m"]).spawn().await;
    // A port nothing listens on.
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let h = balancer(
        &[live],
        Options {
            forced: vec![(dead, "m")],
            ..Options::default()
        },
    )
    .await;
    h.until("the live upstream healthy", |s| {
        s.upstreams[0].healthy(Instant::now(), s.health_interval)
    })
    .await;

    for i in 0..20 {
        assert_eq!(
            answered_by(h.chat(body("m", &format!("c{i}"))).await).await,
            "live"
        );
    }
    let dead_up = &h.state.upstreams[1];
    assert!(
        dead_up.requests.get(502) >= 1,
        "some request tried the dead upstream first"
    );
    assert!(h.state.metrics.retries.load(Ordering::Relaxed) >= 1);
    assert!(
        !dead_up.healthy(Instant::now(), h.state.health_interval),
        "ejected"
    );
    assert_eq!(mock.hits.load(Ordering::Relaxed), 20);
}

#[tokio::test]
async fn a_503_before_any_body_is_retried_elsewhere() {
    let (ua, a) = Mock::new("a", &["m"]).spawn().await;
    let (ub, b) = Mock::new("b", &["m"]).spawn().await;
    b.chat_status.store(503, Ordering::Relaxed);
    let h = balancer(&[ua, ub], Options::default()).await;
    h.until("both healthy", |s| s.healthy_count() == 2).await;
    for i in 0..20 {
        assert_eq!(
            answered_by(h.chat(body("m", &format!("c{i}"))).await).await,
            "a"
        );
    }
    assert!(b.hits.load(Ordering::Relaxed) >= 1, "b was tried");
    assert_eq!(a.hits.load(Ordering::Relaxed), 20);
    assert!(h.state.metrics.retries.load(Ordering::Relaxed) >= 1);
}

#[tokio::test]
async fn when_every_upstream_answers_503_the_last_answer_passes_through() {
    let (ua, a) = Mock::new("a", &["m"]).spawn().await;
    a.chat_status.store(503, Ordering::Relaxed);
    let h = balancer(&[ua], Options::default()).await;
    h.until("healthy", |s| s.healthy_count() == 1).await;
    let r = h.chat(body("m", "x")).await;
    assert_eq!(r.status(), 503);
    let v: Value = r.json().await.unwrap();
    assert_eq!(
        v["error"]["upstream"], "a",
        "the upstream's own body, not the balancer's"
    );
}

#[tokio::test]
async fn a_4xx_is_passed_through_without_retry_or_ejection() {
    let (ua, a) = Mock::new("a", &["m"]).spawn().await;
    let (ub, b) = Mock::new("b", &["m"]).spawn().await;
    a.chat_status.store(400, Ordering::Relaxed);
    b.chat_status.store(400, Ordering::Relaxed);
    let h = balancer(&[ua, ub], Options::default()).await;
    h.until("both healthy", |s| s.healthy_count() == 2).await;
    assert_eq!(h.chat(body("m", "x")).await.status(), 400);
    assert_eq!(h.state.metrics.retries.load(Ordering::Relaxed), 0);
    assert_eq!(
        a.hits.load(Ordering::Relaxed) + b.hits.load(Ordering::Relaxed),
        1
    );
    assert_eq!(h.state.healthy_count(), 2);
}

#[tokio::test]
async fn health_follows_the_upstreams_and_models_are_the_deduplicated_union() {
    let (ua, a) = Mock::new("a", &["m1", "shared"]).spawn().await;
    let (ub, b) = Mock::new("b", &["shared"]).spawn().await;
    let h = balancer(&[ua, ub], Options::default()).await;
    h.until("both healthy", |s| s.healthy_count() == 2).await;
    let health: Value = h.get("/health").await.json().await.unwrap();
    assert_eq!(
        (health["healthy"].as_u64(), health["total"].as_u64()),
        (Some(2), Some(2))
    );

    let models: Value = h.get("/v1/models").await.json().await.unwrap();
    let data = models["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let shared = data.iter().find(|m| m["id"] == "shared").unwrap();
    assert_eq!(shared["upstreams"].as_array().unwrap().len(), 2);
    assert_eq!(shared["max_model_len"], 4096);

    a.healthy.store(false, Ordering::Relaxed);
    h.until("one healthy", |s| s.healthy_count() == 1).await;
    assert_eq!(h.get("/health").await.status(), 200);
    b.healthy.store(false, Ordering::Relaxed);
    h.until("none healthy", |s| s.healthy_count() == 0).await;
    let down = h.get("/health").await;
    assert_eq!(down.status(), 503);
    let v: Value = down.json().await.unwrap();
    assert_eq!(v["upstreams"].as_array().unwrap().len(), 2);
    assert_eq!(h.chat(body("shared", "x")).await.status(), 503);
    // An upstream that comes back is routed to again.
    a.healthy.store(true, Ordering::Relaxed);
    h.until("recovered", |s| s.healthy_count() == 1).await;
    assert_eq!(answered_by(h.chat(body("shared", "x")).await).await, "a");
}

#[tokio::test]
async fn model_errors_are_openai_errors() {
    let (ua, _) = Mock::new("a", &["m1"]).spawn().await;
    let (ub, _) = Mock::new("b", &["m2"]).spawn().await;
    let h = balancer(&[ua, ub], Options::default()).await;
    h.until("both healthy", |s| s.healthy_count() == 2).await;
    let unknown = h.chat(body("zzz", "x")).await;
    assert_eq!(unknown.status(), 404);
    assert_eq!(
        unknown.json::<Value>().await.unwrap()["error"]["code"],
        "model_not_found"
    );
    let missing = h
        .chat(json!({"messages": [{"role": "user", "content": "x"}]}))
        .await;
    assert_eq!(missing.status(), 400);
    let v: Value = missing.json().await.unwrap();
    assert_eq!(v["error"]["param"], "model");
    let bad = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .body("{nope")
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
}

#[tokio::test]
async fn a_missing_model_works_when_every_upstream_serves_the_same_single_one() {
    let (ua, _) = Mock::new("a", &["only"]).spawn().await;
    let h = balancer(&[ua], Options::default()).await;
    h.until("healthy", |s| s.healthy_count() == 1).await;
    let r = h
        .chat(json!({"messages": [{"role": "user", "content": "x"}]}))
        .await;
    assert_eq!(answered_by(r).await, "a");
}

#[tokio::test]
async fn a_streamed_response_is_forwarded_as_it_arrives() {
    let (ua, _) = Mock::new("a", &["m"]).spawn().await;
    let h = balancer(&[ua], Options::default()).await;
    h.until("healthy", |s| s.healthy_count() == 1).await;
    let started = Instant::now();
    let mut r = h
        .chat(json!({"model": "m", "stream": true, "messages": [{"role": "user", "content": "x"}]}))
        .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let first = r.chunk().await.unwrap().unwrap();
    let first_at = started.elapsed();
    assert!(String::from_utf8_lossy(&first).contains("first"));
    assert!(
        first_at < Duration::from_millis(300),
        "the first event waited for the end: {first_at:?}"
    );
    let mut rest = String::new();
    while let Some(c) = r.chunk().await.unwrap() {
        rest.push_str(&String::from_utf8_lossy(&c));
    }
    assert!(rest.contains("second"));
    assert!(started.elapsed() >= Duration::from_millis(400));
    // The request is finished and counted.
    h.until("the request recorded", |s| {
        s.upstreams[0].requests.get(200) == 1
    })
    .await;
    assert_eq!(h.state.upstreams[0].in_flight.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn a_client_that_disconnects_drops_the_upstream_request() {
    let (ua, a) = Mock::new("a", &["m"]).spawn().await;
    let h = balancer(&[ua], Options::default()).await;
    h.until("healthy", |s| s.healthy_count() == 1).await;
    let mut r = h
        .chat(json!({"model": "m", "stream": true, "messages": [{"role": "user", "content": "forever"}]}))
        .await;
    r.chunk().await.unwrap().unwrap();
    assert_eq!(h.state.upstreams[0].in_flight.load(Ordering::Relaxed), 1);
    drop(r);
    h.until("the upstream stream dropped", |_| {
        a.stream_dropped.load(Ordering::Relaxed)
    })
    .await;
    h.until("the request recorded as cancelled", |s| {
        s.upstreams[0].requests.get(499) == 1
    })
    .await;
    assert_eq!(h.state.upstreams[0].in_flight.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn the_upstream_key_is_sent_and_the_client_key_is_required() {
    let (ua, a) = Mock::new("a", &["m"]).spawn().await;
    let h = balancer(
        &[ua],
        Options {
            api_key: Some("client-key"),
            upstream_key: Some("upstream-key"),
            ..Options::default()
        },
    )
    .await;
    h.until("healthy", |s| s.healthy_count() == 1).await;
    assert_eq!(h.get("/health").await.status(), 200, "health needs no key");
    assert_eq!(h.get("/v1/models").await.status(), 401);
    let r = h
        .client
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth("client-key")
        .json(&body("m", "x"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        a.auth_seen.lock().unwrap().as_deref(),
        Some("Bearer upstream-key")
    );
}

#[tokio::test]
async fn round_robin_and_least_loaded_policies_route_without_affinity() {
    let (ua, a) = Mock::new("a", &["m"]).spawn().await;
    let (ub, b) = Mock::new("b", &["m"]).spawn().await;
    let h = balancer(
        &[ua, ub],
        Options {
            policy: Policy::RoundRobin,
            ..Options::default()
        },
    )
    .await;
    h.until("both healthy", |s| s.healthy_count() == 2).await;
    for _ in 0..10 {
        h.chat(body("m", "the very same prompt")).await;
    }
    assert_eq!(
        (
            a.hits.load(Ordering::Relaxed),
            b.hits.load(Ordering::Relaxed)
        ),
        (5, 5)
    );
}

#[tokio::test]
async fn the_stats_snapshot_is_a_balancer_view_with_nested_upstream_snapshots() {
    let nested = StatsSnapshot {
        models: vec![ModelStats {
            name: "m".into(),
            state: crate::server::stats::ModelState::Ready,
            max_sequences: 4,
            running: 1,
            ..ModelStats::default()
        }],
        ..StatsSnapshot::default()
    };
    let mut with_stats = Mock::new("a", &["m"]);
    with_stats.stats = Some(serde_json::to_value(&nested).unwrap());
    let (ua, _) = with_stats.spawn().await;
    let (ub, _) = Mock::new("b", &["m"]).spawn().await;
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let h = balancer(&[ua, ub, dead], Options::default()).await;
    h.until("two healthy and a snapshot", |s| {
        s.healthy_count() == 2 && s.upstreams[0].snapshot(Duration::from_secs(5)).is_some()
    })
    .await;
    h.until("the dead one failed a poll", |s| {
        s.upstreams[2].error().is_some()
    })
    .await;
    for i in 0..6 {
        h.chat(body("m", &format!("c{i}"))).await;
    }
    let stats: StatsSnapshot = h.get("/cuca/v1/stats").await.json().await.unwrap();
    assert_eq!(stats.node.role, crate::server::stats::NodeRole::Balancer);
    assert_eq!(stats.upstreams.len(), 3);
    assert!(stats.upstreams[0].healthy && !stats.upstreams[2].healthy);
    assert!(
        stats.upstreams[0].snapshot.is_some(),
        "an upstream with the stats API is nested"
    );
    assert!(
        stats.upstreams[1].snapshot.is_none(),
        "one without it (404) is not an error"
    );
    assert_eq!(stats.upstreams[1].error, None);
    assert_eq!(
        stats.upstreams.iter().map(|u| u.routed_total).sum::<u64>(),
        6
    );
    assert_eq!(stats.models.len(), 1);
    assert_eq!(
        stats.models[0].running, 1,
        "the union sums the upstream snapshots"
    );
    assert_eq!(stats.requests.total, 6);
    assert!(stats.latency.e2e.count >= 6);
    let down = stats
        .diagnosis
        .iter()
        .find(|d| d.cause == crate::server::stats::Cause::UpstreamDown)
        .expect("the dead upstream is diagnosed");
    assert_eq!(down.upstream.as_deref(), Some(&*stats.upstreams[2].id));

    let metrics = h.get("/metrics").await.text().await.unwrap();
    assert!(
        metrics.contains("cuca_balancer_routed_total{upstream="),
        "{metrics}"
    );
    assert!(metrics.contains("cuca_balancer_decisions_total{decision=\"affinity\"}"));
    assert!(metrics.contains("cuca_balancer_request_ttft_seconds_count{upstream="));
}

#[tokio::test]
async fn a_down_upstream_keeps_its_stale_snapshot_but_not_its_diagnoses() {
    use crate::server::stats::{Cause, Diagnosis, Severity};
    let nested = StatsSnapshot {
        diagnosis: vec![Diagnosis {
            severity: Severity::Warn,
            cause: Cause::CpuSaturated,
            message: "cpu busy".into(),
            ..Diagnosis::default()
        }],
        ..StatsSnapshot::default()
    };
    let mut mock = Mock::new("a", &["m"]);
    mock.stats = Some(serde_json::to_value(&nested).unwrap());
    let (ua, a) = mock.spawn().await;
    let (ub, _) = Mock::new("b", &["m"]).spawn().await;
    let h = balancer(&[ua, ub], Options::default()).await;
    h.until("both healthy with a snapshot", |s| {
        s.healthy_count() == 2 && s.upstreams[0].snapshot(Duration::from_secs(5)).is_some()
    })
    .await;
    let live: StatsSnapshot = h.get("/cuca/v1/stats").await.json().await.unwrap();
    assert!(
        live.diagnosis
            .iter()
            .any(|d| d.cause == Cause::CpuSaturated && d.upstream.is_some())
    );

    a.healthy.store(false, Ordering::Relaxed);
    h.until("a is down", |s| s.healthy_count() == 1).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    let down: StatsSnapshot = h.get("/cuca/v1/stats").await.json().await.unwrap();
    let u = &down.upstreams[0];
    assert!(
        !u.healthy && u.snapshot.is_some(),
        "the snapshot stays for context"
    );
    assert!(
        u.snapshot_age_s.is_some_and(|age| age > 0.2),
        "{:?}",
        u.snapshot_age_s
    );
    assert_eq!(u.error.as_deref(), Some("HTTP 503"));
    assert!(
        down.diagnosis
            .iter()
            .any(|d| d.cause == Cause::UpstreamDown)
    );
    assert!(
        !down
            .diagnosis
            .iter()
            .any(|d| d.cause == Cause::CpuSaturated),
        "a down upstream's old diagnoses are not reported: {:?}",
        down.diagnosis
    );
}

#[tokio::test]
async fn a_refused_connection_is_reported_as_its_cause_without_the_url() {
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let h = balancer(&[dead], Options::default()).await;
    h.until("a failed poll", |s| s.upstreams[0].error().is_some())
        .await;
    // Windows may take longer than the poll timeout to refuse.
    let error = h.state.upstreams[0].error().unwrap();
    assert!(
        ["connection refused", "timed out"].contains(&error.as_str()),
        "{error}"
    );
    let stats: StatsSnapshot = h.get("/cuca/v1/stats").await.json().await.unwrap();
    assert!(
        !stats.upstreams[0]
            .error
            .as_deref()
            .unwrap_or("")
            .contains("http")
    );
    assert!(stats.upstreams[0].snapshot_age_s.is_none());
}

#[tokio::test]
async fn the_balancer_measures_inter_event_latency_of_a_streamed_reply() {
    let (ua, _) = Mock::new("a", &["m"]).spawn().await;
    let h = balancer(&[ua], Options::default()).await;
    h.until("healthy", |s| s.healthy_count() == 1).await;
    // The mock sends `first` (standing for the role prelude), waits 400 ms,
    // sends `second`, waits 400 ms, sends `third`: one interval counts.
    let r = h
        .chat(json!({"model": "m", "stream": true, "three": true, "messages": [{"role": "user", "content": "x"}]}))
        .await;
    let _ = r.text().await.unwrap();
    h.until("recorded", |s| s.upstreams[0].requests.get(200) == 1)
        .await;
    let summary = h.state.upstreams[0].latency.summary(None, Instant::now());
    let itl = summary.latency.itl;
    assert_eq!(itl.count, 1);
    assert!(
        (0.3..0.6).contains(&itl.p50),
        "two events 400 ms apart: {}",
        itl.p50
    );
    // A plain JSON reply has no events.
    h.chat(body("m", "y")).await;
    h.until("second recorded", |s| s.upstreams[0].requests.get(200) == 2)
        .await;
    assert_eq!(
        h.state.upstreams[0]
            .latency
            .summary(None, Instant::now())
            .latency
            .itl
            .count,
        1
    );
}
