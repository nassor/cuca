//! One model server behind the balancer: its health, its models, its latest
//! stats snapshot, and the balancer's own counters for it.
//!
//! A task per upstream ([`spawn_poller`]) polls `GET /health`, `GET /v1/models`
//! and (when the upstream has it) `GET /cuca/v1/stats` every interval, each
//! with a timeout, and keeps only the latest result. An upstream is healthy
//! when its last `/health` answered 200 within three intervals and the
//! proxy has not ejected it since: a connect error, or a 5xx before the first
//! byte, ejects it at once until the next successful poll. A dead upstream is
//! polled with a growing delay, up to [`MAX_BACKOFF`].
//!
//! Bounds: one snapshot, one model list and one 1,024-record latency ring per
//! upstream; the number of upstreams is capped at
//! [`MAX_UPSTREAMS`](super::config::MAX_UPSTREAMS).

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use reqwest::StatusCode;
use serde::Deserialize;
use tokio::task::JoinHandle;

use crate::server::latency::LatencyTracker;
use crate::server::metrics::{E2E_BUCKETS, Histogram, StatusCounters, TTFT_BUCKETS};
use crate::server::stats::StatsSnapshot;

/// Longest delay between polls of a dead upstream.
pub const MAX_BACKOFF: Duration = Duration::from_secs(10);
/// Records of the per-upstream latency ring.
pub const LATENCY_RECORDS: usize = 1024;

/// The cause of a failed request in a few words, from the error's source
/// chain, with no URL (the upstream's URL is its own field): `connection
/// refused`, `timed out`, `connection reset`, or the innermost message.
pub fn short_error(error: &reqwest::Error) -> String {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(e) = source {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            use std::io::ErrorKind::{
                ConnectionAborted, ConnectionRefused, ConnectionReset, TimedOut,
            };
            return match io.kind() {
                ConnectionRefused => "connection refused".into(),
                ConnectionReset | ConnectionAborted => "connection reset".into(),
                TimedOut => "timed out".into(),
                _ => io.to_string(),
            };
        }
        source = e.source();
    }
    if error.is_timeout() {
        "timed out".into()
    } else if error.is_connect() {
        "connection failed".into()
    } else if error.is_body() || error.is_decode() {
        "bad response body".into()
    } else {
        "request failed".into()
    }
}

/// A model an upstream serves, from its `/v1/models`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct UpstreamModel {
    /// The name clients pass as `model`.
    pub id: String,
    /// The context length, as vLLM names it.
    pub max_model_len: u64,
    /// When the model was loaded, seconds since the Unix epoch.
    pub created: u64,
}

#[derive(Debug, Default, Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<UpstreamModel>,
}

#[derive(Debug, Default)]
struct Health {
    /// When `/health` last answered 200.
    last_ok: Option<Instant>,
    /// Set by the proxy on a connect error or a 5xx before the first byte;
    /// cleared by the next successful poll.
    ejected: bool,
    /// Since when the upstream has been failing.
    down_since: Option<Instant>,
    /// Polls failed in a row.
    failures: u32,
    /// The last failure.
    error: Option<String>,
    /// A poll round (either result) has completed.
    polled: bool,
    models: Arc<[String]>,
    model_info: Vec<UpstreamModel>,
    /// `Some(false)` once the upstream answered 404 to the stats route.
    has_stats_api: Option<bool>,
    snapshot: Option<(Instant, Arc<StatsSnapshot>)>,
}

/// One upstream and everything the balancer tracks about it.
#[derive(Debug)]
pub struct Upstream {
    /// `host:port`, the node's identity in hashes, logs and metrics.
    pub id: Arc<str>,
    /// Base URL without a trailing slash.
    pub url: String,
    /// Requests the balancer has outstanding to it.
    pub in_flight: AtomicU64,
    /// Requests routed to it since start.
    pub routed: AtomicU64,
    /// Those that failed (connection error, 5xx, or a stream that broke).
    pub errors: AtomicU64,
    /// Responses by status code (a failed connection counts as 502).
    pub requests: StatusCounters,
    /// Time to the first body byte at the balancer.
    pub ttft: Histogram,
    /// Time to the end of the body.
    pub e2e: Histogram,
    /// The latest requests, for percentiles.
    pub latency: LatencyTracker,
    health: Mutex<Health>,
}

impl Upstream {
    /// An upstream at `url` (already normalized) known as `id`.
    pub fn new(id: &str, url: String, latency_window: Duration) -> Self {
        Self {
            id: Arc::from(id),
            url,
            in_flight: AtomicU64::new(0),
            routed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            requests: StatusCounters::default(),
            ttft: Histogram::new(TTFT_BUCKETS),
            e2e: Histogram::new(E2E_BUCKETS),
            latency: LatencyTracker::new(latency_window, LATENCY_RECORDS),
            health: Mutex::new(Health::default()),
        }
    }

    fn health(&self) -> std::sync::MutexGuard<'_, Health> {
        self.health.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether requests may be routed to it: `/health` answered 200 within
    /// three `interval`s and the proxy has not ejected it since.
    pub fn healthy(&self, now: Instant, interval: Duration) -> bool {
        let h = self.health();
        !h.ejected
            && h.last_ok
                .is_some_and(|t| now.saturating_duration_since(t) <= interval * 3)
    }

    /// Passive health: stop routing here until the next successful poll.
    pub fn eject(&self, reason: String) {
        let mut h = self.health();
        h.ejected = true;
        h.down_since.get_or_insert_with(Instant::now);
        h.error = Some(reason);
    }

    /// The last failure, if the upstream is failing.
    pub fn error(&self) -> Option<String> {
        self.health().error.clone()
    }

    /// How long the upstream has been failing.
    pub fn down_for(&self, now: Instant) -> Option<Duration> {
        self.health()
            .down_since
            .map(|t| now.saturating_duration_since(t))
    }

    /// The names of the models it serves, as last reported.
    pub fn models(&self) -> Arc<[String]> {
        Arc::clone(&self.health().models)
    }

    /// The models with their details.
    pub fn model_info(&self) -> Vec<UpstreamModel> {
        self.health().model_info.clone()
    }

    /// Whether a poll round has completed, so that its models are known or
    /// known to be unreachable.
    pub fn polled(&self) -> bool {
        self.health().polled
    }

    /// Whether the upstream is known to have no stats API.
    pub fn lacks_stats_api(&self) -> bool {
        self.health().has_stats_api == Some(false)
    }

    /// The latest stats snapshot if it is no older than `max_age`.
    pub fn snapshot(&self, max_age: Duration) -> Option<Arc<StatsSnapshot>> {
        let h = self.health();
        h.snapshot
            .as_ref()
            .filter(|(at, _)| at.elapsed() <= max_age)
            .map(|(_, s)| Arc::clone(s))
    }

    /// The last snapshot ever stored and its age, however old: a down
    /// upstream keeps it for context.
    pub fn latest_snapshot(&self) -> Option<(Duration, Arc<StatsSnapshot>)> {
        let h = self.health();
        h.snapshot
            .as_ref()
            .map(|(at, s)| (at.elapsed(), Arc::clone(s)))
    }

    fn poll_succeeded(&self, models: Vec<UpstreamModel>, stats: Option<StatsOutcome>) {
        let mut h = self.health();
        // An upstream restarted with `--stats-api` must be probed again.
        if h.down_since.is_some() || h.ejected {
            h.has_stats_api = None;
        }
        h.last_ok = Some(Instant::now());
        h.ejected = false;
        h.down_since = None;
        h.failures = 0;
        h.error = None;
        h.polled = true;
        // An empty answer (a failed `/v1/models`) keeps the last list.
        if !models.is_empty() || h.model_info.is_empty() {
            h.models = models.iter().map(|m| m.id.clone()).collect();
            h.model_info = models;
        }
        match stats {
            Some(StatsOutcome::Snapshot(s)) => {
                h.has_stats_api = Some(true);
                h.snapshot = Some((Instant::now(), Arc::new(*s)));
            }
            Some(StatsOutcome::Absent) => {
                h.has_stats_api = Some(false);
                h.snapshot = None;
            }
            None => {}
        }
    }

    /// Make the upstream look healthy serving `models`, for an hour, without
    /// a poller (tests of the proxy's passive health).
    #[cfg(test)]
    pub(crate) fn force_healthy(&self, models: &[&str]) {
        let mut h = self.health();
        h.last_ok = Some(Instant::now() + Duration::from_secs(3600));
        h.polled = true;
        h.models = models.iter().map(|m| (*m).to_string()).collect();
        h.model_info = models
            .iter()
            .map(|m| UpstreamModel {
                id: (*m).to_string(),
                ..UpstreamModel::default()
            })
            .collect();
    }

    fn poll_failed(&self, error: String) -> u32 {
        let mut h = self.health();
        h.down_since.get_or_insert_with(Instant::now);
        h.failures = h.failures.saturating_add(1);
        h.error = Some(error);
        h.polled = true;
        h.failures
    }
}

enum StatsOutcome {
    Snapshot(Box<StatsSnapshot>),
    Absent,
}

/// The delay before polling again after `failures` failures in a row.
pub fn poll_delay(interval: Duration, failures: u32) -> Duration {
    if failures == 0 {
        return interval;
    }
    interval
        .saturating_mul(1u32 << failures.min(4))
        .min(MAX_BACKOFF)
        .max(interval)
}

/// One poll round: health, then models, then stats. `Err` means the upstream
/// is down (its `/health` did not answer 200); a failure of the optional
/// parts is not.
async fn poll_once(
    upstream: &Upstream,
    client: &reqwest::Client,
    key: Option<&str>,
    timeout: Duration,
) -> Result<(), String> {
    let get = |path: &str| {
        let mut req = client
            .get(format!("{}{path}", upstream.url))
            .timeout(timeout);
        if let Some(key) = key {
            req = req.bearer_auth(key);
        }
        req.send()
    };
    let health = get("/health").await.map_err(|e| short_error(&e))?;
    if health.status() != StatusCode::OK {
        return Err(format!("HTTP {}", health.status().as_u16()));
    }
    let models = match get("/v1/models").await {
        Ok(r) if r.status().is_success() => r
            .json::<ModelList>()
            .await
            .map(|l| l.data)
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    // Do not ask again after a 404: the upstream has no stats API.
    let stats = if upstream.lacks_stats_api() {
        None
    } else {
        match get("/cuca/v1/stats").await {
            Ok(r) if r.status() == StatusCode::NOT_FOUND => Some(StatsOutcome::Absent),
            Ok(r) if r.status().is_success() => r
                .json::<StatsSnapshot>()
                .await
                .ok()
                .map(|s| StatsOutcome::Snapshot(Box::new(s))),
            _ => None,
        }
    };
    upstream.poll_succeeded(models, stats);
    Ok(())
}

/// Poll `upstream` every `interval` until the task is aborted.
pub fn spawn_poller(
    upstream: Arc<Upstream>,
    client: reqwest::Client,
    key: Option<Arc<str>>,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let timeout = (interval * 2).clamp(Duration::from_secs(1), Duration::from_secs(5));
        loop {
            let delay = match poll_once(&upstream, &client, key.as_deref(), timeout).await {
                Ok(()) => poll_delay(interval, 0),
                Err(error) => {
                    let failures = upstream.poll_failed(error.clone());
                    if failures == 1 {
                        tracing::warn!(upstream = %upstream.id, %error, "upstream is down");
                    }
                    poll_delay(interval, failures)
                }
            };
            tokio::time::sleep(delay).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upstream() -> Upstream {
        Upstream::new("a:1", "http://a:1".into(), Duration::from_secs(60))
    }

    #[test]
    fn an_upstream_is_unhealthy_until_it_answers_and_after_three_missed_intervals() {
        let u = upstream();
        let interval = Duration::from_millis(100);
        assert!(!u.healthy(Instant::now(), interval), "unpolled");
        u.poll_succeeded(vec![], None);
        assert!(u.healthy(Instant::now(), interval));
        assert!(!u.healthy(Instant::now() + interval * 4, interval), "stale");
    }

    #[test]
    fn ejection_lasts_until_the_next_successful_poll() {
        let u = upstream();
        u.poll_succeeded(vec![], None);
        u.eject("connection refused".into());
        assert!(!u.healthy(Instant::now(), Duration::from_secs(1)));
        assert_eq!(u.error().as_deref(), Some("connection refused"));
        assert!(u.down_for(Instant::now()).is_some());
        u.poll_succeeded(vec![], None);
        assert!(u.healthy(Instant::now(), Duration::from_secs(1)));
        assert_eq!(u.error(), None);
        assert!(u.down_for(Instant::now()).is_none());
    }

    #[test]
    fn a_failed_poll_marks_the_round_complete_and_counts_failures() {
        let u = upstream();
        assert!(!u.polled());
        assert_eq!(u.poll_failed("refused".into()), 1);
        assert_eq!(u.poll_failed("refused".into()), 2);
        assert!(u.polled());
    }

    #[test]
    fn polling_a_dead_upstream_backs_off_to_ten_seconds() {
        let i = Duration::from_millis(1000);
        assert_eq!(poll_delay(i, 0), i);
        assert_eq!(poll_delay(i, 1), Duration::from_secs(2));
        assert_eq!(poll_delay(i, 2), Duration::from_secs(4));
        assert_eq!(poll_delay(i, 10), MAX_BACKOFF);
        // A long interval is never shortened.
        assert_eq!(
            poll_delay(Duration::from_secs(30), 3),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn the_models_it_reports_replace_the_old_ones_but_an_empty_answer_keeps_them() {
        let u = upstream();
        let model = |id: &str| UpstreamModel {
            id: id.into(),
            ..UpstreamModel::default()
        };
        u.poll_succeeded(vec![model("m1"), model("m2")], None);
        assert_eq!(&*u.models(), ["m1", "m2"]);
        u.poll_succeeded(vec![], None);
        assert_eq!(
            &*u.models(),
            ["m1", "m2"],
            "a failed /v1/models keeps the last list"
        );
        u.poll_succeeded(vec![model("m3")], None);
        assert_eq!(&*u.models(), ["m3"]);
    }

    #[test]
    fn a_snapshot_goes_stale_and_a_404_marks_the_stats_api_absent() {
        let u = upstream();
        u.poll_succeeded(vec![], Some(StatsOutcome::Snapshot(Box::default())));
        assert!(u.snapshot(Duration::from_secs(5)).is_some());
        std::thread::sleep(Duration::from_millis(5));
        assert!(
            u.snapshot(Duration::from_millis(1)).is_none(),
            "older than max_age"
        );
        u.poll_succeeded(vec![], Some(StatsOutcome::Absent));
        assert!(u.lacks_stats_api());
        assert!(u.snapshot(Duration::from_secs(5)).is_none());
    }
}

#[cfg(test)]
mod short_error_tests {
    use super::short_error;

    #[tokio::test]
    async fn a_refused_connection_is_named_by_its_cause_alone() {
        let dead = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}/health", l.local_addr().unwrap())
        };
        let error = reqwest::Client::new().get(&dead).send().await.unwrap_err();
        assert_eq!(short_error(&error), "connection refused");
        assert!(!short_error(&error).contains("http"));
    }
}
