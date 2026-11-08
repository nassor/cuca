//! macOS GPU probe: `ioreg` accelerator statistics.

use std::process::Command;
use std::time::{Duration, Instant};

use super::MAX_GPUS;
use super::parse::parse_ioreg;
use crate::server::stats::GpuStats;

/// `ioreg` is a subprocess: run it at most this often, reuse the last result between.
const IOREG_INTERVAL: Duration = Duration::from_secs(2);

/// Reads GPU utilisation and in-use memory from
/// `ioreg -r -d 1 -c IOAccelerator` (`"Device Utilization %"` and
/// `"In use system memory"`).
///
/// Apple GPUs share system memory, so no VRAM size is reported: only the memory
/// the GPU holds. Temperature, power and per-process memory are unavailable.
pub(super) struct GpuProbe {
    last_run: Option<Instant>,
    cached: Vec<GpuStats>,
}

impl GpuProbe {
    pub(super) fn new() -> Self {
        Self {
            last_run: None,
            cached: Vec::new(),
        }
    }

    pub(super) fn sample(&mut self) -> Vec<GpuStats> {
        if self.last_run.is_none_or(|t| t.elapsed() >= IOREG_INTERVAL) {
            self.last_run = Some(Instant::now());
            self.cached = run_ioreg();
        }
        self.cached.clone()
    }
}

/// macOS exposes no paging rate here: both are unknown.
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

fn run_ioreg() -> Vec<GpuStats> {
    let out = match Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            tracing::debug!("sysmon: ioreg exited with {}", o.status);
            return Vec::new();
        }
        Err(e) => {
            tracing::debug!("sysmon: ioreg failed: {e}");
            return Vec::new();
        }
    };
    parse_ioreg(&String::from_utf8_lossy(&out.stdout))
        .into_iter()
        .take(MAX_GPUS)
        .enumerate()
        .map(|(index, g)| GpuStats {
            index,
            vendor: vendor_of(&g.name).to_owned(),
            name: g.name,
            util_pct: g.util_pct,
            vram_total_bytes: None,
            vram_used_bytes: g.in_use_bytes,
            process_vram_bytes: None,
            shared_vram_used_bytes: None,
            temperature_c: None,
            power_w: None,
            source: "ioreg".to_owned(),
        })
        .collect()
}

/// The wire vendor name from a model name (`Apple M2 Pro`, `AMD Radeon Pro 5500M`).
fn vendor_of(name: &str) -> &'static str {
    [
        ("Apple", "apple"),
        ("AMD", "amd"),
        ("Intel", "intel"),
        ("NVIDIA", "nvidia"),
    ]
    .into_iter()
    .find(|(prefix, _)| name.starts_with(prefix))
    .map_or("unknown", |(_, vendor)| vendor)
}
