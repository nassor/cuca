//! What a coordinator measures of each remote stage: calls, time and bytes.
//!
//! One [`StageCounters`] per remote, a fixed set of atomics the transport
//! writes and the [`LocalModel`](crate::LocalModel) handle reads, so the
//! figures never grow with traffic. The wall time of a call is what the
//! coordinator's thread waited for the answer; the compute time is what the
//! worker's engine thread spent running the frame (the worker reports it with
//! its answer). Their difference is the network: the transfer both ways, the
//! HTTP handling, and the time the frame waited in the worker's queue.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// What the coordinator measured of one remote stage, cumulative since load
/// (see [`LocalModel::stage_stats`](crate::LocalModel::stage_stats)).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct StageStats {
    /// The stage in messages: `stage {url}` for an HTTP stage.
    pub name: String,
    /// Calls that were answered.
    pub calls: u64,
    /// Calls that failed (unreachable, refused, a non-2xx answer); their time
    /// is in none of the totals below.
    pub errors: u64,
    /// Total nanoseconds the answered calls took, from sending the frame to
    /// the whole answer.
    pub wall_ns: u64,
    /// Total nanoseconds the worker reported it computed for those calls. A
    /// worker that reports nothing (an older cuca) counts zero, so its calls
    /// show as all network.
    pub compute_ns: u64,
    /// Bytes of frames sent.
    pub sent_bytes: u64,
    /// Bytes of frames received.
    pub received_bytes: u64,
}

impl StageStats {
    /// `wall_ns - compute_ns`: what the calls spent outside the worker's
    /// compute.
    pub fn network_ns(&self) -> u64 {
        self.wall_ns.saturating_sub(self.compute_ns)
    }
}

/// The atomics behind one [`StageStats`].
#[derive(Debug, Default)]
pub(crate) struct StageCounters {
    calls: AtomicU64,
    errors: AtomicU64,
    wall_ns: AtomicU64,
    compute_ns: AtomicU64,
    sent_bytes: AtomicU64,
    received_bytes: AtomicU64,
}

/// A duration in nanoseconds, saturating.
fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

impl StageCounters {
    /// Count one answered call.
    pub(crate) fn record(&self, wall: Duration, compute: Duration, sent: usize, received: usize) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.wall_ns.fetch_add(nanos(wall), Ordering::Relaxed);
        self.compute_ns.fetch_add(nanos(compute), Ordering::Relaxed);
        self.sent_bytes.fetch_add(sent as u64, Ordering::Relaxed);
        self.received_bytes
            .fetch_add(received as u64, Ordering::Relaxed);
    }

    /// Count one failed call; `sent` bytes still left the coordinator.
    pub(crate) fn record_error(&self, sent: usize) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        self.sent_bytes.fetch_add(sent as u64, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self, name: String) -> StageStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        StageStats {
            name,
            calls: get(&self.calls),
            errors: get(&self.errors),
            wall_ns: get(&self.wall_ns),
            compute_ns: get(&self.compute_ns),
            sent_bytes: get(&self.sent_bytes),
            received_bytes: get(&self.received_bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_add_up_and_network_is_wall_minus_compute() {
        let c = StageCounters::default();
        c.record(Duration::from_millis(5), Duration::from_millis(3), 100, 40);
        c.record(Duration::from_millis(7), Duration::from_millis(4), 10, 4);
        c.record_error(8);
        let s = c.snapshot("stage x".to_owned());
        assert_eq!((s.calls, s.errors), (2, 1));
        assert_eq!((s.wall_ns, s.compute_ns), (12_000_000, 7_000_000));
        assert_eq!(s.network_ns(), 5_000_000);
        assert_eq!((s.sent_bytes, s.received_bytes), (118, 44));
        assert_eq!(s.name, "stage x");
    }

    #[test]
    fn a_compute_time_above_the_wall_time_gives_no_negative_network() {
        let s = StageStats {
            wall_ns: 1,
            compute_ns: 2,
            ..StageStats::default()
        };
        assert_eq!(s.network_ns(), 0);
    }
}
