//! GPU probe for platforms without a GPU source: reports no GPUs.

use crate::server::stats::GpuStats;

pub(super) struct GpuProbe;

impl GpuProbe {
    pub(super) fn new() -> Self {
        Self
    }

    #[expect(clippy::unused_self, reason = "same shape as the platform probes")]
    pub(super) fn sample(&mut self) -> Vec<GpuStats> {
        Vec::new()
    }
}

/// No paging source: both rates are unknown.
pub(super) struct SwapProbe;

impl SwapProbe {
    pub(super) fn new() -> Self {
        Self
    }

    #[expect(clippy::unused_self, reason = "same shape as the platform probes")]
    pub(super) fn sample(&mut self, _dt_s: f64) -> (Option<f64>, Option<f64>) {
        (None, None)
    }
}
