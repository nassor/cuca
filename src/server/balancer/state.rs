//! State shared by every balancer route.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::config::BalancerConfig;
use super::routing::{Candidate, Decision, Router, UpstreamView, load_score};
use super::upstream::Upstream;
use crate::server::latency::{LatencyTracker, MAX_RECORDS};
use crate::server::metrics::{E2E_BUCKETS, Histogram, StatusCounters, TTFT_BUCKETS};
use crate::server::sysmon::SystemSampler;

/// The balancer's own counters.
#[derive(Debug)]
pub struct BalancerMetrics {
    /// Responses by status code, counted when the headers go out.
    pub http: StatusCounters,
    /// Chat requests refused before any upstream was tried.
    pub rejected: StatusCounters,
    /// Attempts repeated on another upstream.
    pub retries: AtomicU64,
    /// Affinity targets passed over for being too loaded.
    pub fallbacks: AtomicU64,
    /// Routing decisions by kind (see [`Decision`]).
    pub decisions: [AtomicU64; 4],
    /// Time to the first body byte, over all upstreams.
    pub ttft: Histogram,
    /// Time to the end of the body, over all upstreams.
    pub e2e: Histogram,
}

impl Default for BalancerMetrics {
    fn default() -> Self {
        Self {
            http: StatusCounters::default(),
            rejected: StatusCounters::default(),
            retries: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
            decisions: Default::default(),
            ttft: Histogram::new(TTFT_BUCKETS),
            e2e: Histogram::new(E2E_BUCKETS),
        }
    }
}

/// The metric slot of a decision.
pub fn decision_slot(decision: Decision) -> usize {
    match decision {
        Decision::Affinity => 0,
        Decision::Fallback => 1,
        Decision::LeastLoaded => 2,
        Decision::RoundRobin => 3,
    }
}

/// The decision labels in slot order.
pub const DECISIONS: [Decision; 4] = [
    Decision::Affinity,
    Decision::Fallback,
    Decision::LeastLoaded,
    Decision::RoundRobin,
];

/// Everything the routes share.
pub struct BalancerState {
    /// The address the HTTP listener is bound to.
    pub listen: SocketAddr,
    /// Bearer key clients must send (every route but `/health`).
    pub api_key: Option<Arc<str>>,
    /// Sent to the upstreams.
    pub upstream_key: Option<Arc<str>>,
    /// Whether the stats routes are mounted.
    pub stats_api: bool,
    /// The upstreams, in configuration order.
    pub upstreams: Vec<Arc<Upstream>>,
    /// The routing policy and its rotation counter.
    pub router: Router,
    /// The HTTP client of the proxy and the pollers.
    pub client: reqwest::Client,
    /// The poll interval; health and snapshot staleness derive from it.
    pub health_interval: Duration,
    /// Finished requests over all upstreams, for percentiles.
    pub latency: LatencyTracker,
    /// Counters.
    pub metrics: BalancerMetrics,
    /// The balancer host's sampler.
    pub system: SystemSampler,
    /// When the balancer started.
    pub started: Instant,
    /// The same moment, milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// `<host>:<port>`.
    pub node_id: String,
    /// Stats streams open now.
    pub stats_streams: AtomicUsize,
}

impl BalancerState {
    /// The state for `config` over `upstreams`.
    pub fn new(
        config: &BalancerConfig,
        listen: SocketAddr,
        upstreams: Vec<Arc<Upstream>>,
        client: reqwest::Client,
        system: SystemSampler,
    ) -> Self {
        let host = sysinfo::System::host_name().unwrap_or_else(|_| "localhost".to_string());
        Self {
            listen,
            api_key: config.api_key.as_deref().map(Arc::from),
            upstream_key: config.upstream_api_key.as_deref().map(Arc::from),
            stats_api: config.stats_api,
            upstreams,
            router: Router::new(config.policy),
            client,
            health_interval: config.health_interval,
            latency: LatencyTracker::new(config.latency_window, MAX_RECORDS),
            metrics: BalancerMetrics::default(),
            system,
            started: Instant::now(),
            started_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
            node_id: format!("{host}:{}", listen.port()),
            stats_streams: AtomicUsize::new(0),
        }
    }

    /// The oldest stats snapshot a score may use.
    pub fn snapshot_max_age(&self) -> Duration {
        self.health_interval * 3
    }

    /// Whether `upstream` takes requests now.
    pub fn is_healthy(&self, upstream: &Upstream, now: Instant) -> bool {
        upstream.healthy(now, self.health_interval)
    }

    /// How many upstreams take requests now.
    pub fn healthy_count(&self) -> usize {
        let now = Instant::now();
        self.upstreams
            .iter()
            .filter(|u| self.is_healthy(u, now))
            .count()
    }

    /// Every upstream has completed a poll round.
    pub fn warmed_up(&self) -> bool {
        self.upstreams.iter().all(|u| u.polled())
    }

    /// The healthy upstreams of `model` that are not in `exclude`, with their
    /// load scores, in configuration order.
    pub fn candidates<'a>(&'a self, model: &str, exclude: &[usize]) -> Vec<Candidate<'a>> {
        let now = Instant::now();
        self.upstreams
            .iter()
            .enumerate()
            .filter(|(i, u)| !exclude.contains(i) && self.is_healthy(u, now))
            .filter(|(_, u)| u.models().iter().any(|m| m == model))
            .map(|(index, u)| Candidate {
                index,
                id: &u.id,
                score: load_score(
                    u.in_flight.load(std::sync::atomic::Ordering::Relaxed),
                    u.snapshot(self.snapshot_max_age()).as_deref(),
                    model,
                ),
            })
            .collect()
    }

    /// The routing view of every upstream: owned pieces to borrow from.
    pub fn views(&self) -> Vec<OwnedView> {
        let now = Instant::now();
        self.upstreams
            .iter()
            .map(|u| OwnedView {
                id: Arc::clone(&u.id),
                healthy: self.is_healthy(u, now),
                models: u.models(),
            })
            .collect()
    }
}

/// An upstream's id, health and models, owned so that a request can borrow
/// routing views from them (see [`as_views`]).
#[derive(Debug, Clone)]
pub struct OwnedView {
    id: Arc<str>,
    healthy: bool,
    models: Arc<[String]>,
}

/// Borrow [`BalancerState::views`] as routing views.
pub fn as_views(owned: &[OwnedView]) -> Vec<UpstreamView<'_>> {
    owned
        .iter()
        .map(|v| UpstreamView {
            id: &v.id,
            healthy: v.healthy,
            models: &v.models,
        })
        .collect()
}
