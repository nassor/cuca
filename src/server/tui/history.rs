//! Bounded history of the numbers the sparklines and charts draw.
//!
//! One [`Sample`] is derived from each snapshot; the ring keeps the last
//! [`HISTORY_CAP`] and drops the oldest at the bound, so memory stays constant
//! however long the terminal UI runs.

use std::collections::VecDeque;

use crate::server::stats::StatsSnapshot;

/// Samples kept per node (five minutes at one snapshot per second).
pub const HISTORY_CAP: usize = 300;

/// One GPU at one moment.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuSample {
    /// Utilization `0..=100`, `0` when the driver does not report it.
    pub util_pct: f64,
    /// Device memory in use, bytes.
    pub vram_used: f64,
}

/// The numbers derived from one snapshot.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sample {
    /// Whole-machine CPU usage `0..=100`.
    pub cpu_pct: f64,
    /// RAM in use, bytes.
    pub ram_used: f64,
    /// Swap in use, bytes.
    pub swap_used: f64,
    /// Bytes per second paged in from swap (`0` when the OS does not report it).
    pub swap_in: f64,
    /// Bytes per second paged out to swap (`0` when the OS does not report it).
    pub swap_out: f64,
    /// Disk read throughput, bytes per second.
    pub disk_read: f64,
    /// Disk write throughput, bytes per second.
    pub disk_write: f64,
    /// 95th percentile time to first token, seconds.
    pub ttft_p95: f64,
    /// 95th percentile inter-token latency, seconds.
    pub itl_p95: f64,
    /// 95th percentile end-to-end latency, seconds.
    pub e2e_p95: f64,
    /// 95th percentile queue wait, seconds.
    pub queue_p95: f64,
    /// Generated tokens per second over all models.
    pub tokens_per_s: f64,
    /// Requests in flight.
    pub in_flight: f64,
    /// Per GPU, in snapshot order.
    pub gpus: Vec<GpuSample>,
}

/// A ring of the last [`HISTORY_CAP`] samples of one node.
#[derive(Debug, Clone)]
pub struct History {
    samples: VecDeque<Sample>,
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    /// An empty history.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(HISTORY_CAP),
        }
    }

    /// Derives a sample from `snapshot` and appends it, dropping the oldest at
    /// the bound.
    pub fn push(&mut self, snapshot: &StatsSnapshot) {
        let system = &snapshot.system;
        let sample = Sample {
            cpu_pct: f64::from(system.cpu.usage_pct),
            ram_used: system.memory.used_bytes as f64,
            swap_used: system.memory.swap_used_bytes as f64,
            swap_in: system.memory.swap_in_bytes_per_s.unwrap_or(0.0),
            swap_out: system.memory.swap_out_bytes_per_s.unwrap_or(0.0),
            disk_read: system.disk_io.read_bytes_per_s,
            disk_write: system.disk_io.write_bytes_per_s,
            ttft_p95: snapshot.latency.ttft.p95,
            itl_p95: snapshot.latency.itl.p95,
            e2e_p95: snapshot.latency.e2e.p95,
            queue_p95: snapshot.latency.queue.p95,
            tokens_per_s: snapshot
                .models
                .iter()
                .map(|model| model.tokens_per_s.generation)
                .sum(),
            in_flight: snapshot.requests.in_flight as f64,
            gpus: system
                .gpus
                .iter()
                .map(|gpu| GpuSample {
                    util_pct: gpu.util_pct.map_or(0.0, f64::from),
                    vram_used: gpu.vram_used_bytes.map_or(0.0, |bytes| bytes as f64),
                })
                .collect(),
        };
        if self.samples.len() >= HISTORY_CAP {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Samples kept.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no snapshot arrived yet.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The newest `limit` values of one field, oldest first.
    pub fn series(&self, limit: usize, field: impl Fn(&Sample) -> f64) -> Vec<f64> {
        let skip = self.samples.len().saturating_sub(limit);
        self.samples.iter().skip(skip).map(field).collect()
    }

    /// The newest `limit` values of one GPU's field, oldest first (`0` for a
    /// sample taken before the adapter appeared).
    pub fn gpu_series(
        &self,
        gpu: usize,
        limit: usize,
        field: impl Fn(&GpuSample) -> f64,
    ) -> Vec<f64> {
        self.series(limit, |sample| sample.gpus.get(gpu).map_or(0.0, &field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with(cpu: f32) -> StatsSnapshot {
        let mut snapshot = StatsSnapshot::default();
        snapshot.system.cpu.usage_pct = cpu;
        snapshot
    }

    #[test]
    fn the_ring_never_grows_past_its_cap_and_keeps_the_newest() {
        let mut history = History::new();
        for i in 0..(HISTORY_CAP * 3 + 7) {
            history.push(&snapshot_with(i as f32 % 100.0));
        }
        assert_eq!(history.len(), HISTORY_CAP);
        let last = (HISTORY_CAP * 3 + 6) as f64 % 100.0;
        assert_eq!(history.series(1, |sample| sample.cpu_pct), [last]);
        assert_eq!(history.series(5, |sample| sample.cpu_pct).len(), 5);
        assert_eq!(
            history.series(10_000, |sample| sample.cpu_pct).len(),
            HISTORY_CAP
        );
    }

    #[test]
    fn a_gpu_that_appears_late_reads_as_zero_before_it() {
        let mut history = History::new();
        history.push(&StatsSnapshot::default());
        let mut with_gpu = StatsSnapshot::default();
        with_gpu.system.gpus = vec![crate::server::stats::GpuStats {
            util_pct: Some(50.0),
            ..Default::default()
        }];
        history.push(&with_gpu);
        assert_eq!(history.gpu_series(0, 10, |gpu| gpu.util_pct), [0.0, 50.0]);
    }
}
