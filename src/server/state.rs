//! State shared by every request: the registry, the counters, the latency
//! window, the system sampler, and the per-request tracker that feeds them.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tracing::Span;

use super::collector::StatsCollector;
use super::config::ServerConfig;
use super::latency::{LatencyTracker, MAX_RECORDS, Outcome, RequestRecord};
use super::metrics::ServerMetrics;
use super::registry::{ModelRegistry, ModelSlot};
use super::sysmon::SystemSampler;
use crate::GenerationReport;

/// A producer wait shorter than this is a scheduling blip, not a slow client.
const MIN_STALL: Duration = Duration::from_millis(2);

/// Everything the routes share.
pub struct ServerState {
    /// The address the HTTP listener is bound to.
    pub listen: SocketAddr,
    /// Bearer key every route but `/health` requires, when set.
    pub api_key: Option<Arc<str>>,
    /// Whether the stats API routes are mounted.
    pub stats_api: bool,
    /// The models.
    pub registry: Arc<ModelRegistry>,
    /// Finished requests of the sliding window.
    pub latency: LatencyTracker,
    /// Counters not tied to a model.
    pub metrics: ServerMetrics,
    /// The machine sampler.
    pub system: SystemSampler,
    /// The sample rings the stats API computes rates from.
    pub collector: StatsCollector,
    /// When the server started.
    pub started: Instant,
    /// The same moment, milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// `<host>:<port>`, the node's stable id.
    pub node_id: String,
    /// Stats streams open now (bounded by `http::MAX_STATS_STREAMS`).
    pub stats_streams: AtomicUsize,
}

impl ServerState {
    /// The state of a server for `config` over `registry`, reading the
    /// machine through `system`; `listen` is the bound address.
    pub fn new(
        config: &ServerConfig,
        listen: SocketAddr,
        registry: Arc<ModelRegistry>,
        system: SystemSampler,
    ) -> Self {
        let host = sysinfo::System::host_name().unwrap_or_else(|_| "localhost".to_string());
        Self {
            listen,
            api_key: config.api_key.as_deref().map(Arc::from),
            stats_api: config.stats_api,
            collector: StatsCollector::new(registry.slots().len()),
            registry,
            latency: LatencyTracker::new(config.latency_window, MAX_RECORDS),
            metrics: ServerMetrics::default(),
            system,
            started: Instant::now(),
            started_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
            node_id: format!("{host}:{}", listen.port()),
            stats_streams: AtomicUsize::new(0),
        }
    }
}

/// Follows one chat request from arrival to its last byte and, however it
/// ends, records it once: per-model counters and histograms, the latency
/// window, the request's span and one `info` line.
///
/// A tracker dropped before [`finish`](Self::finish) (the client went away
/// and the handler was cancelled) records the request as cancelled (499).
pub struct RequestTracker {
    state: Arc<ServerState>,
    slot: Arc<ModelSlot>,
    request_id: Arc<str>,
    span: Span,
    arrived: Instant,
    first_block: Option<Instant>,
    last_block: Option<Instant>,
    blocks: u32,
    stall: Duration,
    done: bool,
}

impl RequestTracker {
    /// Start tracking a request that arrived at `arrived` for `slot`.
    pub fn start(
        state: Arc<ServerState>,
        slot: Arc<ModelSlot>,
        request_id: Arc<str>,
        span: Span,
        arrived: Instant,
    ) -> Self {
        slot.metrics.in_flight.fetch_add(1, Ordering::Relaxed);
        span.record("model", &*slot.name);
        Self {
            state,
            slot,
            request_id,
            span,
            arrived,
            first_block: None,
            last_block: None,
            blocks: 0,
            stall: Duration::ZERO,
            done: false,
        }
    }

    /// The request's id, as sent in `x-request-id`.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// A content block went out (or was collected).
    pub fn block(&mut self) {
        let now = Instant::now();
        self.first_block.get_or_insert(now);
        self.last_block = Some(now);
        self.blocks = self.blocks.saturating_add(1);
    }

    /// Generation waited `waited` for the client to take output.
    pub fn stalled(&mut self, waited: Duration) {
        if waited >= MIN_STALL {
            self.stall += waited;
        }
    }

    /// The request ended with HTTP `status`; `report` is the engine's account
    /// when it produced one.
    pub fn finish(mut self, status: u16, outcome: Outcome, report: Option<&GenerationReport>) {
        self.record(status, outcome, report);
    }

    fn record(&mut self, status: u16, outcome: Outcome, report: Option<&GenerationReport>) {
        if std::mem::replace(&mut self.done, true) {
            return;
        }
        let now = Instant::now();
        let m = &self.slot.metrics;
        let e2e = now.duration_since(self.arrived).as_secs_f64();
        let (prompt, completion) = report
            .and_then(|r| r.usage)
            .map_or((0, 0), |u| (u.prompt_tokens, u.completion_tokens));
        let cached = report
            .and_then(|r| r.prompt_cache)
            .map_or(0, |c| c.read_tokens);
        let ttft = self
            .first_block
            .map(|t| t.duration_since(self.arrived).as_secs_f64());
        // Gaps between tokens after the first, over the exact token count.
        let tokens = if completion > 0 {
            completion
        } else {
            self.blocks
        };
        let itl = match (self.first_block, self.last_block) {
            (Some(first), Some(last)) if tokens > 1 => {
                Some(last.duration_since(first).as_secs_f64() / f64::from(tokens - 1))
            }
            _ => None,
        };
        let queue = report.and_then(|r| r.queue_secs);

        m.in_flight.fetch_sub(1, Ordering::Relaxed);
        m.requests.inc(status);
        m.stall_ns.fetch_add(
            u64::try_from(self.stall.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        m.prompt_tokens
            .fetch_add(u64::from(prompt), Ordering::Relaxed);
        m.cached_prompt_tokens
            .fetch_add(u64::from(cached), Ordering::Relaxed);
        m.generation_tokens
            .fetch_add(u64::from(completion), Ordering::Relaxed);
        if outcome == Outcome::Ok {
            m.e2e.observe(e2e);
            ttft.into_iter().for_each(|v| m.ttft.observe(v));
            itl.into_iter().for_each(|v| m.itl.observe(v));
            queue.into_iter().for_each(|v| m.queue.observe(v));
        }
        self.state.latency.record(RequestRecord {
            finished: now,
            model: Arc::clone(&self.slot.name),
            outcome,
            queue_s: queue,
            ttft_s: ttft,
            itl_s: itl,
            e2e_s: e2e,
            prompt_tokens: prompt,
            completion_tokens: completion,
            cached_tokens: cached,
            stall_s: self.stall.as_secs_f64(),
        });

        let ms = |s: f64| (s * 1000.0 * 10.0).round() / 10.0;
        self.span.record("status", status);
        self.span.record("prompt_tokens", prompt);
        self.span.record("completion_tokens", completion);
        self.span.record("e2e_ms", ms(e2e));
        if let Some(t) = ttft {
            self.span.record("ttft_ms", ms(t));
        }
        let _entered = self.span.enter();
        tracing::info!(
            target: "cuca::access",
            request_id = %self.request_id,
            model = %self.slot.name,
            status,
            outcome = ?outcome,
            prompt_tokens = prompt,
            completion_tokens = completion,
            cached_tokens = cached,
            queue_ms = queue.map(ms),
            ttft_ms = ttft.map(ms),
            itl_ms = itl.map(ms),
            e2e_ms = ms(e2e),
            finish_reason = report.and_then(|r| r.finish_reason),
            "chat completion finished"
        );
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        self.record(499, Outcome::Cancelled, None);
    }
}
