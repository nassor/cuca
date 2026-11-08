//! Sampling and the stats snapshot.
//!
//! One background task ([`run_sampler`]) wakes every second, reads each ready
//! model's [`ServingStats`] and expert-tier counters and
//! the system sampler's latest snapshot, and appends them to bounded rings:
//! [`RING`] samples per model (5 minutes) and per machine. Everything the
//! stats API reports as a rate (tokens per second, steps per second,
//! preemptions per minute, expert reads) is a difference between two ring
//! entries; gauges (running, waiting, K/V blocks) are read live when a
//! snapshot is built. [`snapshot`] builds a [`StatsSnapshot`] from the rings,
//! the latency window and the machine; it allocates one snapshot and holds no
//! lock across the work.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

use super::diagnosis::{ModelSignals, SystemSignals, Thresholds, diagnose};
use super::latency::WindowSummary;
use super::registry::{ModelSlot, SlotStatus};
use super::state::ServerState;
use super::stats::{
    ExpertTierRates, KvStats, ModelMemory, ModelState, ModelStats, NodeInfo, NodeRole,
    RequestStats, StatsSnapshot, SystemSnapshot, TokensPerSecond,
};
use crate::{ExpertTierStats, ServingStats};

/// Samples kept per ring: five minutes at one per second.
pub const RING: usize = 300;
/// The interval between samples.
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
/// The span rates are measured over.
const RATE_WINDOW: Duration = Duration::from_secs(10);
/// The span preemptions are counted over (and scaled to a minute).
const PREEMPT_WINDOW: Duration = Duration::from_secs(60);
/// The span the step-time trend compares, and the recent part of it.
const TREND_WINDOW: Duration = Duration::from_secs(60);
const TREND_RECENT: Duration = Duration::from_secs(15);
/// The span swap growth is measured over.
const SWAP_WINDOW: Duration = Duration::from_secs(30);
/// Steps each half of the trend needs before it is trusted.
const TREND_MIN_STEPS: u64 = 5;

struct ModelSample {
    at: Instant,
    serving: ServingStats,
    tier: Option<ExpertTierStats>,
}

struct SystemSample {
    at: Instant,
    swap_used_bytes: u64,
}

/// The sample rings; see the module docs for the bounds.
pub struct StatsCollector {
    models: Mutex<Vec<VecDeque<ModelSample>>>,
    system: Mutex<VecDeque<SystemSample>>,
}

fn push_bounded<T>(ring: &mut VecDeque<T>, value: T) {
    if ring.len() >= RING {
        ring.pop_front();
    }
    ring.push_back(value);
}

impl StatsCollector {
    /// Rings for `models` models, all empty.
    pub fn new(models: usize) -> Self {
        Self {
            models: Mutex::new((0..models).map(|_| VecDeque::new()).collect()),
            system: Mutex::new(VecDeque::new()),
        }
    }

    /// Take one sample of every ready model and of the machine at `now`.
    pub fn sample(&self, slots: &[Arc<ModelSlot>], system: &SystemSnapshot, now: Instant) {
        // Read the engines before locking: the reads are atomics, the lock is
        // only held to push.
        let samples: Vec<Option<ModelSample>> = slots
            .iter()
            .map(|slot| {
                slot.ready().map(|loaded| ModelSample {
                    at: now,
                    serving: loaded.model.serving_stats(),
                    tier: loaded.model.expert_tier_stats(),
                })
            })
            .collect();
        {
            let mut rings = self.models.lock().unwrap_or_else(PoisonError::into_inner);
            for (ring, sample) in rings.iter_mut().zip(samples) {
                if let Some(sample) = sample {
                    push_bounded(ring, sample);
                }
            }
        }
        let mut ring = self.system.lock().unwrap_or_else(PoisonError::into_inner);
        push_bounded(
            &mut ring,
            SystemSample {
                at: now,
                swap_used_bytes: system.memory.swap_used_bytes,
            },
        );
    }

    /// The rates of model `index` as of its newest sample.
    fn rates(&self, index: usize) -> Rates {
        let rings = self.models.lock().unwrap_or_else(PoisonError::into_inner);
        rings.get(index).map_or_else(Rates::default, rates_of)
    }

    /// How much swap use grew over the last [`SWAP_WINDOW`].
    fn swap_growth_bytes(&self) -> f64 {
        let ring = self.system.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(newest) = ring.back() else {
            return 0.0;
        };
        let base = first_within(&ring, newest.at, SWAP_WINDOW, |s| s.at).unwrap_or(newest);
        newest.swap_used_bytes as f64 - base.swap_used_bytes as f64
    }
}

/// The oldest entry no more than `span` older than `newest`.
fn first_within<T>(
    ring: &VecDeque<T>,
    newest: Instant,
    span: Duration,
    at: impl Fn(&T) -> Instant,
) -> Option<&T> {
    ring.iter()
        .find(|s| newest.saturating_duration_since(at(s)) <= span)
}

/// Rates of one model over its sample ring.
#[derive(Debug, Default)]
struct Rates {
    prompt_per_s: f64,
    generation_per_s: f64,
    steps_per_s: f64,
    rows_per_s: f64,
    mean_rows_per_step: f64,
    prefill_row_share: f64,
    mean_step_ms: f64,
    mean_queue_ms: f64,
    prefix_hit_ratio: Option<f64>,
    preemptions_per_min: f64,
    step_trend: Option<(f64, f64)>,
    tier: Option<(f64, f64, Option<f64>)>,
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn rates_of(ring: &VecDeque<ModelSample>) -> Rates {
    let Some(newest) = ring.back() else {
        return Rates::default();
    };
    let Some(base) = first_within(ring, newest.at, RATE_WINDOW, |s| s.at) else {
        return Rates::default();
    };
    let dt = newest.at.duration_since(base.at).as_secs_f64();
    if dt <= 0.0 {
        return Rates::default();
    }
    let (a, b) = (&base.serving, &newest.serving);
    let d = |new: u64, old: u64| new.saturating_sub(old) as f64;
    let steps = d(b.steps, a.steps);
    let rows = d(b.batched_rows, a.batched_rows);
    let prefill = d(b.prefill_rows, a.prefill_rows);
    let hits = d(b.prefix_hit_tokens, a.prefix_hit_tokens);

    let preempt_base = first_within(ring, newest.at, PREEMPT_WINDOW, |s| s.at).unwrap_or(base);
    let preempt_dt = newest
        .at
        .duration_since(preempt_base.at)
        .as_secs_f64()
        .max(RATE_WINDOW.as_secs_f64());

    Rates {
        prompt_per_s: prefill / dt,
        generation_per_s: d(b.generated_tokens, a.generated_tokens) / dt,
        steps_per_s: steps / dt,
        rows_per_s: rows / dt,
        mean_rows_per_step: ratio(rows, steps),
        prefill_row_share: ratio(prefill, rows),
        mean_step_ms: ratio(d(b.step_ns, a.step_ns), steps) / 1e6,
        mean_queue_ms: ratio(
            d(b.queue_wait_ns, a.queue_wait_ns),
            d(b.admissions, a.admissions),
        ) / 1e6,
        prefix_hit_ratio: {
            let total = hits + prefill;
            (total > 0.0).then(|| hits / total)
        },
        preemptions_per_min: d(b.preemptions, preempt_base.serving.preemptions) / preempt_dt * 60.0,
        step_trend: step_trend(ring, newest),
        tier: match (&base.tier, &newest.tier) {
            (Some(x), Some(y)) => {
                let lookups = d(y.lookups, x.lookups);
                let hits = d(y.device_hits + y.ram_hits, x.device_hits + x.ram_hits);
                Some((
                    d(y.disk_reads, x.disk_reads) / dt,
                    d(y.bytes_read, x.bytes_read) / dt,
                    (lookups > 0.0).then(|| hits / lookups),
                ))
            }
            _ => None,
        },
    }
}

/// Mean step time (ms) before and during the last [`TREND_RECENT`] of the
/// last [`TREND_WINDOW`], when each part ran enough steps.
fn step_trend(ring: &VecDeque<ModelSample>, newest: &ModelSample) -> Option<(f64, f64)> {
    let oldest = first_within(ring, newest.at, TREND_WINDOW, |s| s.at)?;
    let split = first_within(ring, newest.at, TREND_RECENT, |s| s.at)?;
    let mean_ms = |from: &ModelSample, to: &ModelSample| {
        let steps = to.serving.steps.saturating_sub(from.serving.steps);
        (steps >= TREND_MIN_STEPS).then(|| {
            to.serving.step_ns.saturating_sub(from.serving.step_ns) as f64 / steps as f64 / 1e6
        })
    };
    Some((mean_ms(oldest, split)?, mean_ms(split, newest)?))
}

/// Run the sampler until the task is aborted: one sample per
/// [`SAMPLE_INTERVAL`].
pub fn run_sampler(state: Arc<ServerState>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(SAMPLE_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let system = state.system.snapshot();
            state
                .collector
                .sample(state.registry.slots(), &system, Instant::now());
        }
    })
}

/// RFC 3339 UTC of `unix_ms` (`2026-11-08T12:00:00Z`).
pub(crate) fn rfc3339(unix_ms: u64) -> String {
    let secs = unix_ms / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil-from-days (Howard Hinnant), days since 1970-01-01.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn model_stats(
    state: &ServerState,
    index: usize,
    slot: &ModelSlot,
    window: &WindowSummary,
    span_s: f64,
) -> (ModelStats, Option<(f64, f64)>) {
    let mut stats = ModelStats {
        name: slot.name.to_string(),
        source: slot.source.clone(),
        device: slot.device.clone(),
        memory: ModelMemory {
            device: slot.device.clone(),
            ..ModelMemory::default()
        },
        requests: RequestStats {
            in_flight: slot
                .metrics
                .in_flight
                .load(std::sync::atomic::Ordering::Relaxed),
            total: slot.metrics.requests.total(),
            errors: slot.metrics.requests.errors(),
            rate_per_s: ratio(window.finished as f64, span_s),
        },
        latency: window.latency,
        ..ModelStats::default()
    };
    let loaded = match slot.status() {
        SlotStatus::Loading => {
            stats.state = ModelState::Loading;
            stats.load_progress = slot.load_progress();
            return (stats, None);
        }
        SlotStatus::Failed(message) => {
            stats.state = ModelState::Failed;
            stats.error = Some(message.to_string());
            return (stats, None);
        }
        SlotStatus::Ready(loaded) => loaded,
    };
    let serving = loaded.model.serving_stats();
    let rates = state.collector.rates(index);
    stats.state = ModelState::Ready;
    stats.context_length = loaded.context_length as u64;
    stats.max_sequences = serving.max_sequences as u64;
    stats.running = serving.running as u64;
    stats.waiting = serving.waiting as u64;
    stats.kv = KvStats {
        total_blocks: serving.total_blocks as u64,
        free_blocks: serving.free_blocks as u64,
        cached_blocks: serving.cached_blocks as u64,
        used_blocks: serving.used_blocks as u64,
        bytes: loaded.kv_bytes,
        usage_ratio: ratio(serving.used_blocks as f64, serving.total_blocks as f64),
    };
    stats.prefix_hit_ratio = rates.prefix_hit_ratio;
    stats.preemptions_per_min = rates.preemptions_per_min;
    stats.tokens_per_s = TokensPerSecond {
        prompt: rates.prompt_per_s,
        generation: rates.generation_per_s,
    };
    stats.steps_per_s = rates.steps_per_s;
    stats.rows_per_s = rates.rows_per_s;
    stats.mean_rows_per_step = rates.mean_rows_per_step;
    stats.prefill_row_share = rates.prefill_row_share;
    stats.mean_step_ms = rates.mean_step_ms;
    stats.last_step_ms = serving.last_step_ns as f64 / 1e6;
    stats.mean_queue_ms = rates.mean_queue_ms;
    stats.memory.weights_bytes = loaded.weights_bytes;
    stats.memory.kv_bytes = loaded.kv_bytes;
    stats.expert_tiers = loaded.model.expert_tier_stats().map(|tier| {
        let (disk_reads_per_s, read_bytes_per_s, hit_ratio) =
            rates.tier.unwrap_or((0.0, 0.0, None));
        ExpertTierRates {
            disk_reads_per_s,
            read_bytes_per_s,
            hit_ratio,
            ram_resident_bytes: tier.ram_resident_bytes,
            device_resident_bytes: tier.device_resident_bytes,
        }
    });
    (stats, rates.step_trend)
}

/// Build the snapshot the stats API serves, from the rings, the latency
/// window and the machine's latest sample.
pub fn snapshot(state: &ServerState) -> StatsSnapshot {
    let now = Instant::now();
    let uptime = now.duration_since(state.started);
    let window = state.latency.window();
    // Rates over a window longer than the process has lived would dilute.
    let span_s = uptime.min(window).as_secs_f64().max(1.0);
    let system = state.system.snapshot();

    let mut models = Vec::new();
    let mut windows = Vec::new();
    let mut trends = Vec::new();
    for (index, slot) in state.registry.slots().iter().enumerate() {
        let w = state.latency.summary(Some(&slot.name), now);
        let (stats, trend) = model_stats(state, index, slot, &w, span_s);
        models.push(stats);
        windows.push(w);
        trends.push(trend);
    }
    let overall = state.latency.summary(None, now);

    let diagnosis = {
        let signals: Vec<ModelSignals<'_>> = models
            .iter()
            .zip(&windows)
            .zip(&trends)
            .map(|((stats, window), step_trend)| ModelSignals {
                stats,
                window,
                step_trend: *step_trend,
            })
            .collect();
        diagnose(
            &signals,
            &SystemSignals {
                snapshot: &system,
                swap_growth_bytes: state.collector.swap_growth_bytes(),
            },
            &Thresholds::default(),
        )
    };

    let requests = RequestStats {
        in_flight: models.iter().map(|m| m.requests.in_flight).sum(),
        total: models.iter().map(|m| m.requests.total).sum::<u64>()
            + state.metrics.rejected.total(),
        errors: models.iter().map(|m| m.requests.errors).sum::<u64>()
            + state.metrics.rejected.errors(),
        rate_per_s: ratio(overall.finished as f64, span_s),
    };
    StatsSnapshot {
        node: NodeInfo {
            id: state.node_id.clone(),
            role: NodeRole::Server,
            version: env!("CARGO_PKG_VERSION").to_string(),
            listen: state.listen.to_string(),
            started_at: rfc3339(state.started_unix_ms),
            started_at_ms: state.started_unix_ms,
            uptime_s: uptime.as_secs_f64(),
        },
        models,
        latency: overall.latency,
        requests,
        system,
        diagnosis,
        upstreams: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(at: Instant, f: impl FnOnce(&mut ServingStats)) -> ModelSample {
        let mut serving = ServingStats::default();
        f(&mut serving);
        ModelSample {
            at,
            serving,
            tier: None,
        }
    }

    /// A ring of one sample per second for `n` seconds, built by `f(second)`.
    fn ring(n: u64, f: impl Fn(u64, &mut ServingStats)) -> VecDeque<ModelSample> {
        let t0 = Instant::now();
        (0..=n)
            .map(|i| sample(t0 + Duration::from_secs(i), |s| f(i, s)))
            .collect()
    }

    #[test]
    fn rates_are_deltas_over_the_last_ten_seconds() {
        // 100 generated tokens, 40 prefill rows, 20 steps of 5 ms per second.
        let r = ring(30, |i, s| {
            s.generated_tokens = 100 * i;
            s.prefill_rows = 40 * i;
            s.batched_rows = 140 * i;
            s.steps = 20 * i;
            s.step_ns = 20 * i * 5_000_000;
            s.admissions = i;
            s.queue_wait_ns = i * 30_000_000;
        });
        let rates = rates_of(&r);
        assert_eq!(rates.generation_per_s, 100.0);
        assert_eq!(rates.prompt_per_s, 40.0);
        assert_eq!(rates.steps_per_s, 20.0);
        assert_eq!(rates.mean_rows_per_step, 7.0);
        assert_eq!(rates.prefill_row_share, 40.0 / 140.0);
        assert!((rates.mean_step_ms - 5.0).abs() < 1e-9);
        assert!((rates.mean_queue_ms - 30.0).abs() < 1e-9);
        assert_eq!(rates.prefix_hit_ratio, Some(0.0), "prompts, no hits");
    }

    #[test]
    fn the_prefix_hit_ratio_is_hits_over_hits_plus_computed_prompt_rows() {
        let r = ring(12, |i, s| {
            s.prefix_hit_tokens = 300 * i;
            s.prefill_rows = 100 * i;
        });
        assert_eq!(rates_of(&r).prefix_hit_ratio, Some(0.75));
        let idle = ring(12, |_, _| {});
        assert_eq!(rates_of(&idle).prefix_hit_ratio, None);
    }

    #[test]
    fn preemptions_are_scaled_to_a_minute_but_never_over_a_span_under_ten_seconds() {
        // 1 preemption in the first seconds of a young ring is not 60/min.
        let young = ring(2, |i, s| s.preemptions = i.min(1));
        assert!(rates_of(&young).preemptions_per_min <= 6.0);
        // 30 over a minute of ring.
        let r = ring(60, |i, s| s.preemptions = i / 2);
        assert!((rates_of(&r).preemptions_per_min - 30.0).abs() < 1.5);
    }

    #[test]
    fn the_step_trend_compares_the_recent_steps_with_the_earlier_ones() {
        // 10 steps per second; 20 ms steps for 45 s, then 60 ms.
        let r = ring(60, |i, s| {
            s.steps = 10 * i;
            s.step_ns = (10 * i.min(45) * 20 + 10 * i.saturating_sub(45) * 60) * 1_000_000;
        });
        let newest = r.back().unwrap();
        let (baseline, recent) = step_trend(&r, newest).expect("both halves have steps");
        assert!((baseline - 20.0).abs() < 0.5, "{baseline}");
        assert!((recent - 60.0).abs() < 0.5, "{recent}");
        // Too few steps in the recent part: no trend.
        let sparse = ring(60, |i, s| s.steps = i / 20);
        assert_eq!(step_trend(&sparse, sparse.back().unwrap()), None);
    }

    #[test]
    fn the_ring_is_bounded() {
        let mut ring = VecDeque::new();
        for i in 0..(RING + 50) {
            push_bounded(&mut ring, i);
        }
        assert_eq!((ring.len(), ring.front().copied()), (RING, Some(50)));
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_000_000_000_000), "2001-09-09T01:46:40Z");
        assert_eq!(rfc3339(1_709_164_800_000), "2024-02-29T00:00:00Z");
    }
}
