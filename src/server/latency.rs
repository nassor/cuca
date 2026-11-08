//! Sliding-window latency tracking.
//!
//! Every finished request leaves one [`RequestRecord`] (phase timings, token
//! counts, outcome). The tracker keeps the records of the last
//! [`ServerConfig::latency_window`](super::ServerConfig::latency_window)
//! (60 s by default) in a ring capped at [`MAX_RECORDS`] entries, and
//! [`LatencyTracker::summary`] turns them into percentiles for the stats API
//! and the diagnosis rules.
//!
//! Bounds: at the cap the oldest record is dropped, so under more than
//! `MAX_RECORDS` requests per window the figures describe the most recent
//! `MAX_RECORDS` ones; a summary sorts at most that many numbers per metric.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::stats::{LatencyStats, Quantiles};

/// Most records the tracker keeps.
pub const MAX_RECORDS: usize = 4096;

/// How a request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Completed (2xx, stream finished).
    Ok,
    /// Rejected for the client's fault (4xx).
    ClientError,
    /// Failed on the server (5xx, or an error while streaming).
    ServerError,
    /// The client went away before the response ended.
    Cancelled,
}

impl Outcome {
    /// Whether this counts as an error in the request counters.
    pub fn is_error(self) -> bool {
        matches!(self, Self::ClientError | Self::ServerError)
    }
}

/// What one finished request took.
#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// When it finished.
    pub finished: Instant,
    /// The model it ran on.
    pub model: Arc<str>,
    /// How it ended.
    pub outcome: Outcome,
    /// Seconds waiting for admission to the engine.
    pub queue_s: Option<f64>,
    /// Seconds from arrival to the first content block.
    pub ttft_s: Option<f64>,
    /// Mean seconds between tokens after the first.
    pub itl_s: Option<f64>,
    /// Seconds from arrival to the end.
    pub e2e_s: f64,
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: u32,
    /// Seconds the producer waited for the client to take output.
    pub stall_s: f64,
}

/// The figures of one summary: percentiles plus the aggregates the diagnosis
/// rules read.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindowSummary {
    /// Percentiles over the successful requests of the window.
    pub latency: LatencyStats,
    /// Requests that finished in the window, whatever the outcome.
    pub finished: u64,
    /// Those that ended in an error.
    pub errors: u64,
    /// Mean prompt tokens of the successful requests.
    pub mean_prompt_tokens: f64,
    /// Share of their prompt tokens served from the prefix cache.
    pub cached_share: f64,
    /// Producer wait for slow clients as a share of the end-to-end time.
    pub stall_share: f64,
}

/// The sliding window of finished requests; see the module docs for the
/// bounds.
#[derive(Debug)]
pub struct LatencyTracker {
    window: Duration,
    cap: usize,
    records: Mutex<VecDeque<RequestRecord>>,
}

impl LatencyTracker {
    /// A tracker over `window`, keeping at most `cap` records.
    pub fn new(window: Duration, cap: usize) -> Self {
        Self {
            window,
            cap: cap.max(1),
            records: Mutex::new(VecDeque::with_capacity(cap.clamp(1, 256))),
        }
    }

    /// The window length.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Add a finished request; drops the records that left the window and,
    /// at the cap, the oldest.
    pub fn record(&self, record: RequestRecord) {
        let mut records = self.records.lock().unwrap_or_else(PoisonError::into_inner);
        let now = record.finished;
        evict(&mut records, now, self.window);
        if records.len() >= self.cap {
            records.pop_front();
        }
        records.push_back(record);
    }

    /// The figures of the records inside the window ending at `now`, for one
    /// model or (`None`) all of them.
    pub fn summary(&self, model: Option<&str>, now: Instant) -> WindowSummary {
        let records = self.records.lock().unwrap_or_else(PoisonError::into_inner);
        let in_window = records.iter().filter(|r| {
            now.saturating_duration_since(r.finished) <= self.window
                && model.is_none_or(|m| &*r.model == m)
        });

        let mut queue = Vec::new();
        let mut ttft = Vec::new();
        let mut itl = Vec::new();
        let mut e2e = Vec::new();
        let (mut finished, mut errors, mut ok) = (0u64, 0u64, 0u64);
        let (mut prompt, mut cached, mut stall, mut e2e_sum) = (0f64, 0f64, 0f64, 0f64);
        for r in in_window {
            finished += 1;
            errors += u64::from(r.outcome.is_error());
            if r.outcome != Outcome::Ok {
                continue;
            }
            ok += 1;
            queue.extend(r.queue_s);
            ttft.extend(r.ttft_s);
            itl.extend(r.itl_s);
            e2e.push(r.e2e_s);
            prompt += f64::from(r.prompt_tokens);
            cached += f64::from(r.cached_tokens);
            stall += r.stall_s;
            e2e_sum += r.e2e_s;
        }
        WindowSummary {
            latency: LatencyStats {
                window_s: self.window.as_secs_f64(),
                ttft: quantiles(&mut ttft),
                itl: quantiles(&mut itl),
                e2e: quantiles(&mut e2e),
                queue: quantiles(&mut queue),
            },
            finished,
            errors,
            mean_prompt_tokens: if ok > 0 { prompt / ok as f64 } else { 0.0 },
            cached_share: if prompt > 0.0 { cached / prompt } else { 0.0 },
            stall_share: if e2e_sum > 0.0 { stall / e2e_sum } else { 0.0 },
        }
    }
}

/// Drop the records older than `window` before `now` (the ring is in finish
/// order, so they are at the front).
fn evict(records: &mut VecDeque<RequestRecord>, now: Instant, window: Duration) {
    while records
        .front()
        .is_some_and(|r| now.saturating_duration_since(r.finished) > window)
    {
        records.pop_front();
    }
}

/// Nearest-rank percentiles of `values` (sorted in place).
fn quantiles(values: &mut [f64]) -> Quantiles {
    if values.is_empty() {
        return Quantiles::default();
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let at = |q: f64| {
        let rank = (q * n as f64).ceil() as usize;
        values[rank.clamp(1, n) - 1]
    };
    Quantiles {
        p50: at(0.50),
        p95: at(0.95),
        p99: at(0.99),
        mean: values.iter().sum::<f64>() / n as f64,
        count: n as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(model: &str, finished: Instant, e2e_s: f64) -> RequestRecord {
        RequestRecord {
            finished,
            model: model.into(),
            outcome: Outcome::Ok,
            queue_s: Some(0.1),
            ttft_s: Some(e2e_s / 2.0),
            itl_s: Some(0.02),
            e2e_s,
            prompt_tokens: 100,
            completion_tokens: 10,
            cached_tokens: 16,
            stall_s: 0.0,
        }
    }

    #[test]
    fn percentiles_use_the_nearest_rank() {
        let mut values: Vec<f64> = (1..=100).map(f64::from).collect();
        let q = quantiles(&mut values);
        assert_eq!((q.p50, q.p95, q.p99, q.count), (50.0, 95.0, 99.0, 100));
        assert_eq!(q.mean, 50.5);
        assert_eq!(quantiles(&mut []), Quantiles::default());
    }

    #[test]
    fn records_older_than_the_window_leave_the_summary_and_the_ring() {
        let tracker = LatencyTracker::new(Duration::from_secs(10), 100);
        let start = Instant::now();
        tracker.record(record("m", start, 1.0));
        let later = start + Duration::from_secs(30);
        tracker.record(record("m", later, 2.0));
        let summary = tracker.summary(None, later);
        assert_eq!(summary.finished, 1);
        assert_eq!(summary.latency.e2e.p50, 2.0);
        assert_eq!(tracker.records.lock().unwrap().len(), 1);
    }

    #[test]
    fn the_ring_keeps_the_newest_records_at_the_cap() {
        let tracker = LatencyTracker::new(Duration::from_secs(60), 3);
        let now = Instant::now();
        for i in 0..5 {
            tracker.record(record("m", now, f64::from(i)));
        }
        let summary = tracker.summary(None, now);
        assert_eq!(summary.finished, 3);
        assert_eq!(summary.latency.e2e.p50, 3.0, "records 2, 3, 4 remain");
    }

    #[test]
    fn a_summary_filters_by_model_and_counts_errors_apart_from_latency() {
        let tracker = LatencyTracker::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        tracker.record(record("a", now, 1.0));
        tracker.record(record("b", now, 9.0));
        let mut failed = record("a", now, 0.0);
        failed.outcome = Outcome::ServerError;
        tracker.record(failed);
        let a = tracker.summary(Some("a"), now);
        assert_eq!((a.finished, a.errors, a.latency.e2e.count), (2, 1, 1));
        assert_eq!(tracker.summary(None, now).finished, 3);
    }

    #[test]
    fn the_cached_share_is_weighted_by_prompt_tokens() {
        let tracker = LatencyTracker::new(Duration::from_secs(60), 100);
        let now = Instant::now();
        let mut long = record("m", now, 1.0);
        long.prompt_tokens = 900;
        long.cached_tokens = 0;
        tracker.record(long);
        tracker.record(record("m", now, 1.0));
        let s = tracker.summary(None, now);
        assert_eq!(s.mean_prompt_tokens, 500.0);
        assert_eq!(s.cached_share, 16.0 / 1000.0);
    }
}
