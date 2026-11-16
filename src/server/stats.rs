//! Wire types of the stats API (`GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream`).
//!
//! One [`StatsSnapshot`] describes one node: a model server today, a load
//! balancer later (`NodeRole::Balancer`, with [`StatsSnapshot::upstreams`]
//! filled in). The types are plain serde data with no behaviour, so a client
//! (the balancer polling its servers, the terminal UI) depends on them alone.
//!
//! Compatibility: every struct is `#[serde(default)]`, so a snapshot from an
//! older or newer cuca deserializes (missing fields take their defaults,
//! unknown ones are ignored); fields are added, never renamed. Durations are
//! seconds (`*_s`) or milliseconds (`*_ms`) as the name says, sizes are bytes,
//! ratios are `0.0..=1.0` and percentages `0.0..=100.0`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::sysmon::SystemSnapshot;

/// What kind of node produced a snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    /// A model server (a `serve` section).
    #[default]
    Server,
    /// A load balancer in front of model servers.
    Balancer,
}

/// Identity of the node and how long it has been up.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeInfo {
    /// `<host>:<port>` of the listener; stable across restarts.
    pub id: String,
    /// Server or balancer.
    pub role: NodeRole,
    /// The cuca version.
    pub version: String,
    /// The address the HTTP API listens on.
    pub listen: String,
    /// Process start time, RFC 3339 in UTC (`2026-11-08T12:00:00Z`).
    pub started_at: String,
    /// Process start time, milliseconds since the Unix epoch.
    pub started_at_ms: u64,
    /// Seconds since the process started.
    pub uptime_s: f64,
}

/// Where a model is in its life.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    /// Downloading, converting or reading the weights; requests get 503.
    #[default]
    Loading,
    /// Serving.
    Ready,
    /// The load failed; see [`ModelStats::error`].
    Failed,
}

/// The K/V block pool of a model (full-attention group).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KvStats {
    /// Blocks of 16 positions in the pool, fixed at load.
    pub total_blocks: u64,
    /// Blocks nobody references and no prefix names.
    pub free_blocks: u64,
    /// Unreferenced blocks kept for prefix reuse (evictable).
    pub cached_blocks: u64,
    /// Blocks a running request references.
    pub used_blocks: u64,
    /// Bytes the K/V pools occupy (RAM, or device memory on a GPU).
    pub bytes: u64,
    /// `used_blocks / total_blocks`.
    pub usage_ratio: f64,
}

/// Token throughput over the sampling window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TokensPerSecond {
    /// Prompt tokens the engine computed per second (prefix-cache hits are
    /// not computed, so they do not count).
    pub prompt: f64,
    /// Tokens generated per second.
    pub generation: f64,
}

/// What a model occupies.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelMemory {
    /// Size of the GGUF the model runs from: the weights' footprint, up to
    /// what the engine leaves on disk or re-encodes (`0` while loading).
    pub weights_bytes: u64,
    /// Bytes of the K/V pools.
    pub kv_bytes: u64,
    /// Where the forward pass runs (`cpu`, `vulkan:0`, `cuda:0+cuda:1`, ...).
    pub device: String,
}

/// Tiered-expert counters (mixture-of-experts model loaded with
/// `experts = tiered:...`), the source of disk reads that slow a model down.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExpertTierRates {
    /// Experts read from the GGUF per second over the window.
    pub disk_reads_per_s: f64,
    /// Bytes per second those reads took.
    pub read_bytes_per_s: f64,
    /// Fraction of expert lookups served from a cache, over the window.
    pub hit_ratio: Option<f64>,
    /// Bytes of experts the RAM tier holds.
    pub ram_resident_bytes: u64,
    /// Bytes of experts the device tier holds.
    pub device_resident_bytes: u64,
}

/// The speculative-decoding drafter of a model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DrafterStats {
    /// `in-trunk`, or the drafter file's source (a preset's
    /// `repo:revision:file`, a path).
    pub source: String,
    /// Most tokens drafted per verification round.
    pub draft_tokens: u64,
    /// Fraction of drafted tokens the target accepted over the sampling
    /// window (`None` without drafts in it: the model is idle, or several
    /// requests run and speculation is off).
    pub acceptance_ratio: Option<f64>,
}

/// Percentiles of one latency over the sliding window, in seconds. `count`
/// `0` means no request in the window: the other fields are `0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Quantiles {
    /// Median.
    pub p50: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Requests the figures come from.
    pub count: u64,
}

/// The latencies of the requests that finished inside the window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LatencyStats {
    /// The window length in seconds.
    pub window_s: f64,
    /// Time to first token: from the request's arrival until its first
    /// content block (queue wait plus prefill).
    pub ttft: Quantiles,
    /// Inter-token latency: the mean gap between a request's tokens after the
    /// first (one figure per request).
    pub itl: Quantiles,
    /// End to end: from arrival until the last token.
    pub e2e: Quantiles,
    /// Wait for admission to the engine (a free slot and K/V blocks).
    pub queue: Quantiles,
}

/// Request counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RequestStats {
    /// Requests being handled now (queued, prefilling or generating).
    pub in_flight: u64,
    /// Requests finished since start, whatever the outcome.
    pub total: u64,
    /// Those that ended in an error status or failed mid-stream.
    pub errors: u64,
    /// Requests finished per second over the window.
    pub rate_per_s: f64,
}

/// What a coordinator measured of one remote layer stage, cumulative since
/// load (the means are over every answered call).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageCallStats {
    /// The stage in messages: `stage {url}`.
    pub name: String,
    /// Calls answered.
    pub calls: u64,
    /// Calls that failed.
    pub errors: u64,
    /// Calls sent a second time after the connection failed before any
    /// answer arrived.
    pub retries: u64,
    /// Mean time of an answered call, milliseconds (`compute + network`).
    pub mean_call_ms: f64,
    /// Mean time the worker reported computing per call, milliseconds.
    pub mean_compute_ms: f64,
    /// Mean time of a call outside the worker's compute (transfer, HTTP
    /// handling and the worker's queue), milliseconds.
    pub mean_network_ms: f64,
    /// Bytes of frames sent.
    pub sent_bytes: u64,
    /// Bytes of frames received.
    pub received_bytes: u64,
}

/// One loaded (or loading) model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelStats {
    /// The id clients select it by.
    pub name: String,
    /// Where it came from (a preset, `repo:revision:file`, a path).
    pub source: String,
    /// Device the forward pass runs on.
    pub device: String,
    /// Maximum prompt plus completion tokens per request (`0` while loading).
    pub context_length: u64,
    /// Loading, ready or failed.
    pub state: ModelState,
    /// Why the load failed.
    pub error: Option<String>,
    /// Fraction of the download or conversion done, when it reports one.
    pub load_progress: Option<f64>,
    /// Most requests the engine runs at once.
    pub max_sequences: u64,
    /// Requests the engine is running now.
    pub running: u64,
    /// Requests admitted to the engine but waiting for a slot or for blocks.
    pub waiting: u64,
    /// The K/V block pool.
    pub kv: KvStats,
    /// Share of prompt tokens served from the prefix cache over the window
    /// (`None` without prompts in it).
    pub prefix_hit_ratio: Option<f64>,
    /// Sequences preempted per minute over the window.
    pub preemptions_per_min: f64,
    /// Token throughput over the window.
    pub tokens_per_s: TokensPerSecond,
    /// Forward steps per second over the window.
    pub steps_per_s: f64,
    /// Rows (tokens) those steps carried per second.
    pub rows_per_s: f64,
    /// Mean rows per step over the window; above the running count means
    /// prefill chunks share steps with decoding.
    pub mean_rows_per_step: f64,
    /// Share of the window's rows that were prompt rows.
    pub prefill_row_share: f64,
    /// Mean forward-step duration over the window, milliseconds.
    pub mean_step_ms: f64,
    /// Duration of the last forward step, milliseconds.
    pub last_step_ms: f64,
    /// Mean wait for admission of the requests admitted in the window, from
    /// the engine's own counters (milliseconds).
    pub mean_queue_ms: f64,
    /// What the model occupies.
    pub memory: ModelMemory,
    /// Tiered-expert counters (`None` unless the model streams experts).
    pub expert_tiers: Option<ExpertTierRates>,
    /// The drafter (`None` without one).
    pub drafter: Option<DrafterStats>,
    /// Request counters of this model.
    pub requests: RequestStats,
    /// Latencies of this model's requests over the window.
    pub latency: LatencyStats,
    /// The remote layer stages of a coordinator, in layer order (empty
    /// otherwise).
    pub stages: Vec<StageCallStats>,
}

/// How bad a [`Diagnosis`] is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Context, nothing to fix.
    #[default]
    Info,
    /// Costs latency now.
    Warn,
    /// Costs a lot, or requests fail.
    Critical,
}

/// The reason a [`Diagnosis`] names. New causes are added over time; a
/// reader treats an unknown one as `Other`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    /// A model is still downloading or loading.
    ModelLoading,
    /// A model failed to load.
    ModelFailed,
    /// Every sequence slot is busy and requests queue (`max_sequences`).
    QueueSaturated,
    /// The K/V block pool is nearly full, so requests wait for blocks.
    KvPoolFull,
    /// The engine preempts running sequences to free K/V blocks.
    Preemptions,
    /// Long prompts with a cold prefix cache dominate time to first token.
    PrefillBound,
    /// Prompt chunks of other requests share steps with decoding rows.
    PrefillInterference,
    /// Many rows per step make each step, and so each token, slower.
    LargeBatch,
    /// Forward steps take longer than they did a moment ago.
    StepTimeGrowth,
    /// The CPU is saturated.
    CpuSaturated,
    /// RAM is nearly exhausted or swap is in use.
    MemoryPressure,
    /// A GPU's memory is nearly full.
    VramNearFull,
    /// A GPU is at full utilization.
    GpuSaturated,
    /// Weights or experts are read from disk while serving.
    DiskReads,
    /// Clients read their responses slowly and stall generation.
    SlowConsumer,
    /// A balancer has no healthy upstream to route to.
    NoHealthyUpstream,
    /// An upstream failed its health checks or refused a request.
    UpstreamDown,
    /// Every healthy upstream of a balancer is queueing requests.
    UpstreamSaturated,
    /// One upstream carries far more load than the others while prefix
    /// affinity pins traffic to it.
    LoadImbalance,
    /// One upstream's time to first byte is far above the others'.
    UpstreamSlow,
    /// A cause this reader does not know.
    #[default]
    #[serde(other)]
    Other,
}

/// One ranked reason for the current latency, with the numbers behind it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Diagnosis {
    /// How bad it is; the list is sorted most severe first.
    pub severity: Severity,
    /// The reason as a code.
    pub cause: Cause,
    /// The model it concerns (`None` for a machine-wide cause).
    pub model: Option<String>,
    /// The upstream it concerns, on a balancer's snapshot (`None` for the
    /// balancer itself and for every diagnosis of a model server).
    pub upstream: Option<String>,
    /// A sentence for a human.
    pub message: String,
    /// The measurements that triggered it, by name (`waiting`,
    /// `kv_usage_ratio`, `step_ms`, ...), in the units the name says.
    pub evidence: BTreeMap<String, f64>,
}

/// One server behind a balancer; reserved, always empty on a model server.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpstreamStats {
    /// Name of the upstream.
    pub id: String,
    /// Base URL of the upstream.
    pub url: String,
    /// Whether the balancer routes to it.
    pub healthy: bool,
    /// Requests the balancer has in flight to it.
    pub in_flight: u64,
    /// Latencies of the requests it served.
    pub latency: LatencyStats,
    /// The last failure, if any.
    pub error: Option<String>,
    /// Requests the balancer has routed to it since start.
    pub routed_total: u64,
    /// Those that failed (connection error or error status).
    pub errors_total: u64,
    /// The upstream's own latest snapshot (`None` before the first poll).
    pub snapshot: Option<Box<StatsSnapshot>>,
    /// Seconds since `snapshot` was taken (`None` without one). An age far
    /// above the poll interval means the upstream stopped answering and the
    /// snapshot is stale context.
    pub snapshot_age_s: Option<f64>,
}

/// Everything the stats API reports about one node at one moment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StatsSnapshot {
    /// Identity and uptime.
    pub node: NodeInfo,
    /// Models.
    pub models: Vec<ModelStats>,
    /// Latencies over all models.
    pub latency: LatencyStats,
    /// Request counters over all models.
    pub requests: RequestStats,
    /// The machine.
    pub system: SystemSnapshot,
    /// Ranked reasons for the current latency, most severe first.
    pub diagnosis: Vec<Diagnosis>,
    /// Balancer upstreams (empty on a model server).
    pub upstreams: Vec<UpstreamStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_with_missing_and_unknown_fields_still_deserializes() {
        let json = r#"{
            "node": {"id": "h:8000", "future_field": 1},
            "models": [{"name": "m", "state": "ready", "kv": {"usage_ratio": 0.5}}],
            "diagnosis": [{"severity": "warn", "cause": "queue_saturated", "evidence": {"waiting": 3}}]
        }"#;
        let snap: StatsSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snap.node.role, NodeRole::Server);
        assert_eq!(snap.models[0].state, ModelState::Ready);
        assert_eq!(snap.models[0].kv.usage_ratio, 0.5);
        assert_eq!(snap.models[0].kv.total_blocks, 0);
        assert_eq!(snap.diagnosis[0].cause, Cause::QueueSaturated);
        assert_eq!(snap.diagnosis[0].evidence["waiting"], 3.0);
        assert!(snap.upstreams.is_empty());
    }

    #[test]
    fn severity_orders_info_below_warn_below_critical() {
        assert!(Severity::Info < Severity::Warn && Severity::Warn < Severity::Critical);
    }
}
