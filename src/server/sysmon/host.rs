//! CPU, memory, disk and own-process probes (`sysinfo`).

use std::collections::HashMap;

use sysinfo::{
    CpuRefreshKind, DiskKind, DiskRefreshKind, Disks, MemoryRefreshKind, Pid, ProcessRefreshKind,
    ProcessesToUpdate, System,
};

use super::MAX_DISKS;
use crate::server::stats::{
    CpuStats, DiskIo, DiskStats, MemoryStats, ProcessStats, SystemSnapshot,
};

/// Disks are re-listed with their kind (SSD/HDD) every this many ticks.
const KIND_REFRESH_TICKS: u32 = 30;

/// The `sysinfo` state behind the CPU, RAM, swap, disk and process figures.
///
/// Each tick refreshes only what the snapshot reads: CPU usage, RAM and swap,
/// the disk list with space and I/O counters, and this process alone.
pub(super) struct HostProbe {
    sys: Option<System>,
    disks: Option<Disks>,
    pid: Pid,
    brand: String,
    /// Cumulative `(read, written)` bytes per disk at the previous tick.
    prev_disks: HashMap<String, (u64, u64)>,
    /// Cumulative `(read, written)` bytes of this process at the previous tick.
    prev_process: Option<(u64, u64)>,
    ticks: u32,
}

impl HostProbe {
    pub(super) fn new() -> Self {
        let sys = System::new()
            .inspect_err(|e| tracing::warn!("sysmon: sysinfo System unavailable: {e}"))
            .ok();
        let disks = Disks::new()
            .inspect_err(|e| tracing::warn!("sysmon: sysinfo Disks unavailable: {e}"))
            .ok();
        Self {
            sys,
            disks,
            pid: Pid::from_u32(std::process::id()),
            brand: String::new(),
            prev_disks: HashMap::new(),
            prev_process: None,
            ticks: 0,
        }
    }

    /// Refreshes and writes the CPU, memory, disk, disk I/O and process sections
    /// of `out`. `dt_s` is the time since the previous call, in seconds.
    pub(super) fn sample(&mut self, dt_s: f64, out: &mut SystemSnapshot) {
        let dt_s = dt_s.max(1e-3);
        let kind_tick = self.ticks.is_multiple_of(KIND_REFRESH_TICKS);
        self.ticks = self.ticks.wrapping_add(1);

        let mut proc_io = (0.0, 0.0);
        if let Some(sys) = &mut self.sys {
            sys.refresh_cpu_specifics(CpuRefreshKind::nothing().with_usage());
            sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram().with_swap());
            sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[self.pid]),
                true,
                ProcessRefreshKind::nothing()
                    .with_cpu()
                    .with_memory()
                    .with_disk_usage()
                    .with_tasks(),
            );

            if self.brand.is_empty() {
                self.brand = sys
                    .cpus()
                    .first()
                    .map(|c| c.brand().trim().to_owned())
                    .unwrap_or_default();
            }
            out.cpu = CpuStats {
                usage_pct: sys.global_cpu_usage(),
                per_core_pct: sys.cpus().iter().map(sysinfo::Cpu::usage).collect(),
                cores: sys.cpus().len(),
                load_avg: load_average(),
                brand: self.brand.clone(),
            };
            out.memory = MemoryStats {
                total_bytes: sys.total_memory(),
                used_bytes: sys.used_memory(),
                available_bytes: sys.available_memory(),
                swap_total_bytes: sys.total_swap(),
                swap_used_bytes: sys.used_swap(),
                ..MemoryStats::default()
            };
            if let Some(p) = sys.process(self.pid) {
                let usage = p.disk_usage();
                let now = (usage.total_read_bytes, usage.total_written_bytes);
                if let Some(prev) = self.prev_process.replace(now) {
                    proc_io = (rate(now.0, prev.0, dt_s), rate(now.1, prev.1, dt_s));
                }
                out.process = ProcessStats {
                    pid: self.pid.as_u32(),
                    cpu_pct: p.cpu_usage(),
                    rss_bytes: p.memory(),
                    virtual_bytes: p.virtual_memory(),
                    threads: p.tasks().map(std::collections::HashSet::len),
                    read_bytes_per_s: proc_io.0,
                    write_bytes_per_s: proc_io.1,
                };
            }
        }

        let (disks, io) = self.sample_disks(dt_s, kind_tick);
        out.disks = disks;
        // System-wide counters when the platform reports any, else this process.
        out.disk_io = io.unwrap_or(DiskIo {
            read_bytes_per_s: proc_io.0,
            write_bytes_per_s: proc_io.1,
        });
    }

    /// The disk list and the system-wide I/O rate (`None` when the platform
    /// exposes no per-disk counters).
    fn sample_disks(&mut self, dt_s: f64, kind_tick: bool) -> (Vec<DiskStats>, Option<DiskIo>) {
        let Some(disks) = &mut self.disks else {
            return (Vec::new(), None);
        };
        let mut refresh = DiskRefreshKind::nothing().with_storage().with_io_usage();
        if kind_tick {
            refresh = refresh.with_kind();
        }
        disks.refresh_specifics(true, refresh);

        let mut list = Vec::new();
        let mut counters: HashMap<String, (u64, u64)> = HashMap::new();
        let (mut read, mut written) = (0.0, 0.0);
        let mut any_counter = false;
        for d in disks.iter().take(MAX_DISKS) {
            let mount = d.mount_point().to_string_lossy().into_owned();
            let name = d.name().to_string_lossy().into_owned();
            if d.total_space() > 0 {
                list.push(DiskStats {
                    mount: mount.clone(),
                    name: name.clone(),
                    total_bytes: d.total_space(),
                    available_bytes: d.available_space(),
                    kind: disk_kind(d.kind()).to_owned(),
                });
            }
            // Several mounts of one device share the device's counters: count the
            // device once. A Windows volume name is a label, not an identity.
            let key = if cfg!(windows) || name.is_empty() {
                mount
            } else {
                name
            };
            let usage = d.usage();
            let now = (usage.total_read_bytes, usage.total_written_bytes);
            any_counter |= now != (0, 0);
            if counters.contains_key(&key) {
                continue;
            }
            if let Some(prev) = self.prev_disks.get(&key) {
                read += rate(now.0, prev.0, dt_s);
                written += rate(now.1, prev.1, dt_s);
            }
            counters.insert(key, now);
        }
        self.prev_disks = counters;

        let io = any_counter.then_some(DiskIo {
            read_bytes_per_s: read,
            write_bytes_per_s: written,
        });
        (list, io)
    }
}

/// The wire name of a disk kind: `SSD`, `HDD` or `unknown`.
fn disk_kind(kind: DiskKind) -> &'static str {
    match kind {
        DiskKind::SSD => "SSD",
        DiskKind::HDD => "HDD",
        DiskKind::Unknown(_) => "unknown",
    }
}

/// Bytes per second between two cumulative counters; a counter that went
/// backwards (device re-listed, wrap) counts as no traffic.
#[expect(clippy::cast_precision_loss, reason = "a byte delta far below 2^53")]
fn rate(now: u64, prev: u64, dt_s: f64) -> f64 {
    now.saturating_sub(prev) as f64 / dt_s
}

/// The 1, 5 and 15 minute load average, where the platform has one.
fn load_average() -> Option<[f64; 3]> {
    if !cfg!(unix) {
        return None;
    }
    let l = System::load_average().ok()?;
    Some([l.one, l.five, l.fifteen])
}
