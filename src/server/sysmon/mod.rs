//! System sampler: CPU, memory, disk and GPU figures for the stats API and the TUI.
//!
//! [`SystemSampler::start`] spawns one background thread (`cuca-sysmon`) that
//! probes the machine every `interval` and publishes a [`SystemSnapshot`];
//! [`SystemSampler::snapshot`] clones the latest one under a short lock and never
//! runs a probe, so a request handler can call it freely. The thread holds only
//! a weak reference and exits within one interval of the last clone's drop.
//!
//! # Sources
//!
//! | Figure | Source |
//! | --- | --- |
//! | CPU, RAM, swap, disk space, own process | `sysinfo`, refreshing only what the snapshot reads |
//! | Paging rates | Windows: WMI raw `PerfOS_Memory` page counters (page-ins include memory-mapped file reads); Linux: `/proc/vmstat` `pswpin`/`pswpout`; macOS: unknown (`None`) |
//! | Disk I/O | system-wide per-disk counters from `sysinfo`; this process's counters when the platform reports none |
//! | GPU, Windows | WMI performance counters (`GPUEngine`, `GPUAdapterMemory`, `GPUProcessMemory`) and the registry's DirectX adapter table (name, LUID, 64-bit VRAM size), all vendors |
//! | GPU, Linux | `sysfs` (`/sys/class/drm/card*/device`, hwmon) for amdgpu, i915 and xe, `/proc/self/fdinfo` for this process's VRAM; NVIDIA through NVML, loaded at run time |
//! | GPU, macOS | `ioreg -r -d 1 -c IOAccelerator`, at most every two seconds |
//!
//! Each [`GpuStats::source`](crate::server::stats::GpuStats::source) names what produced that entry. A probe that fails
//! leaves its fields `None` and never stops the sampler.
//!
//! # Rates
//!
//! Byte rates and GPU utilisation come from deltas between ticks over the real
//! elapsed time; the first tick reports 0 (or `None` for GPU utilisation).
//!
//! # Not available
//!
//! * Windows: GPU temperature and power (no counter exposes them), per-process
//!   thread count, load average.
//! * Linux: GPU utilisation on i915 and xe (no sysfs counter); VRAM size on
//!   Intel integrated GPUs (shared memory).
//! * macOS: GPU memory size (unified memory; only the amount in use is shown),
//!   temperature and power.
//! * Every platform: the GPU list is capped at 16 entries and the disk list at 64;
//!   the sampler keeps one snapshot and no history.

use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::server::stats::{MemoryStats, SystemSnapshot};

mod host;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "each platform uses only its own parsers")
)]
mod parse;
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod unsupported;
#[cfg(windows)]
mod windows;

use host::HostProbe;
#[cfg(target_os = "linux")]
use linux::{GpuProbe, SwapProbe};
#[cfg(target_os = "macos")]
use macos::{GpuProbe, SwapProbe};
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
use unsupported::{GpuProbe, SwapProbe};
#[cfg(windows)]
use windows::{GpuProbe, SwapProbe};

/// At most this many GPUs are reported.
const MAX_GPUS: usize = 16;
/// At most this many disks are listed.
const MAX_DISKS: usize = 64;
/// Shortest sampling interval honoured; a smaller one is raised to it.
const MIN_INTERVAL: Duration = Duration::from_millis(200);

/// Handle to the background system sampler.
///
/// Clones share one sampler thread; it stops when the last clone is dropped.
#[derive(Clone)]
pub struct SystemSampler {
    shared: Arc<Shared>,
}

struct Shared {
    latest: Mutex<SystemSnapshot>,
}

impl Shared {
    fn update(&self, f: impl FnOnce(&mut SystemSnapshot)) {
        f(&mut self.latest.lock().unwrap_or_else(PoisonError::into_inner));
    }
}

impl SystemSampler {
    /// Starts the sampler thread `cuca-sysmon`, probing every `interval`
    /// (raised to 200 ms if shorter).
    ///
    /// The snapshot is empty (all zeros, `sampled_at_ms == 0`) until the first
    /// probe lands. If the thread cannot be spawned the snapshot stays empty and
    /// a warning is logged.
    #[must_use]
    pub fn start(interval: Duration) -> SystemSampler {
        let shared = Arc::new(Shared {
            latest: Mutex::new(SystemSnapshot::default()),
        });
        let weak = Arc::downgrade(&shared);
        let interval = interval.max(MIN_INTERVAL);
        if let Err(e) = thread::Builder::new()
            .name("cuca-sysmon".to_owned())
            .spawn(move || run(&weak, interval))
        {
            tracing::warn!("sysmon: cannot spawn the sampler thread: {e}");
        }
        SystemSampler { shared }
    }

    /// The latest sample; a clone under a short lock, never a probe.
    #[must_use]
    pub fn snapshot(&self) -> SystemSnapshot {
        self.shared
            .latest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// The sampler thread body: probe, publish, sleep until the next tick.
fn run(shared: &Weak<Shared>, interval: Duration) {
    // Probes are built here, not in `start`: a WMI connection is bound to its thread.
    let mut host = HostProbe::new();
    let mut gpu = GpuProbe::new();
    let mut swap = SwapProbe::new();
    let mut last = Instant::now();
    loop {
        let tick = Instant::now();
        let dt = tick.duration_since(last).as_secs_f64();
        last = tick;

        // Host figures publish first: a slow first GPU probe must not delay them.
        let mut host_part = SystemSnapshot::default();
        host.sample(dt, &mut host_part);
        let Some(strong) = shared.upgrade() else {
            return;
        };
        strong.update(|s| {
            s.cpu = host_part.cpu;
            // The paging rates come from the slower probe below; keep the last ones.
            let MemoryStats {
                swap_in_bytes_per_s: swap_in,
                swap_out_bytes_per_s: swap_out,
                ..
            } = s.memory;
            s.memory = MemoryStats {
                swap_in_bytes_per_s: swap_in,
                swap_out_bytes_per_s: swap_out,
                ..host_part.memory
            };
            s.disks = host_part.disks;
            s.disk_io = host_part.disk_io;
            s.process = host_part.process;
            s.sampled_at_ms = now_ms();
        });
        drop(strong);

        let mut gpus = gpu.sample();
        gpus.truncate(MAX_GPUS);
        let (swap_in, swap_out) = swap.sample(dt);
        let Some(strong) = shared.upgrade() else {
            return;
        };
        strong.update(|s| {
            s.gpus = gpus;
            s.memory.swap_in_bytes_per_s = swap_in;
            s.memory.swap_out_bytes_per_s = swap_out;
            s.sampled_at_ms = now_ms();
        });
        drop(strong);

        thread::sleep(interval.saturating_sub(tick.elapsed()));
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
