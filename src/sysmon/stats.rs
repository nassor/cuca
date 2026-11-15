//! Wire types of one system sample; embedded in the server's `StatsSnapshot`.

use serde::{Deserialize, Serialize};

/// A CPU's load.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuStats {
    /// Whole-machine usage, `0..=100`.
    pub usage_pct: f32,
    /// Usage of each logical core, `0..=100`.
    pub per_core_pct: Vec<f32>,
    /// Logical cores.
    pub cores: usize,
    /// 1, 5 and 15 minute load averages (`None` where the OS has none).
    pub load_avg: Option<[f64; 3]>,
    /// CPU model name.
    pub brand: String,
}

/// RAM and swap.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryStats {
    /// Installed RAM.
    pub total_bytes: u64,
    /// RAM in use.
    pub used_bytes: u64,
    /// RAM a program can still take without swapping.
    pub available_bytes: u64,
    /// Swap or page-file size.
    pub swap_total_bytes: u64,
    /// Swap or page file in use.
    pub swap_used_bytes: u64,
    /// Bytes per second paged in from swap or the page file (Windows: all
    /// hard-fault page-ins, including memory-mapped files). `None` where the OS
    /// does not report it, so unknown is distinct from 0.
    pub swap_in_bytes_per_s: Option<f64>,
    /// Bytes per second paged out to swap or the page file. `None` where the
    /// OS does not report it.
    pub swap_out_bytes_per_s: Option<f64>,
}

/// One mounted disk.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DiskStats {
    /// Mount point (`C:\`, `/`).
    pub mount: String,
    /// Device name.
    pub name: String,
    /// Capacity.
    pub total_bytes: u64,
    /// Free space.
    pub available_bytes: u64,
    /// `SSD`, `HDD` or `unknown`.
    pub kind: String,
}

/// Disk throughput: system-wide where the OS reports per-disk counters, else
/// this process's own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DiskIo {
    /// Bytes read per second.
    pub read_bytes_per_s: f64,
    /// Bytes written per second.
    pub write_bytes_per_s: f64,
}

/// One GPU. A field is `None` when the driver or OS does not report it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuStats {
    /// Index in the order the sampler found the adapters.
    pub index: usize,
    /// Marketing name.
    pub name: String,
    /// `nvidia`, `amd`, `intel`, `apple` or `unknown`.
    pub vendor: String,
    /// Utilization, `0..=100`.
    pub util_pct: Option<f32>,
    /// Device memory capacity.
    pub vram_total_bytes: Option<u64>,
    /// Device memory in use (all processes).
    pub vram_used_bytes: Option<u64>,
    /// Device memory this process uses.
    pub process_vram_bytes: Option<u64>,
    /// System memory the GPU uses (Windows shared usage, amdgpu GTT), all
    /// processes. `None` where not reported.
    pub shared_vram_used_bytes: Option<u64>,
    /// Temperature in degrees Celsius.
    pub temperature_c: Option<f32>,
    /// Power draw in watts.
    pub power_w: Option<f32>,
    /// Where the figures come from (`nvml`, `wmi`, ...).
    pub source: String,
}

/// This process.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessStats {
    /// Process id.
    pub pid: u32,
    /// CPU time as a percentage of one core (above 100 on several cores).
    pub cpu_pct: f32,
    /// Resident memory.
    pub rss_bytes: u64,
    /// Virtual memory.
    pub virtual_bytes: u64,
    /// Threads, where the OS reports them.
    pub threads: Option<usize>,
    /// Bytes read from disk per second.
    pub read_bytes_per_s: f64,
    /// Bytes written to disk per second.
    pub write_bytes_per_s: f64,
}

/// The machine the node runs on, sampled by [`SystemSampler`](super::SystemSampler).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemSnapshot {
    /// CPU load.
    pub cpu: CpuStats,
    /// RAM and swap.
    pub memory: MemoryStats,
    /// Mounted disks.
    pub disks: Vec<DiskStats>,
    /// Disk throughput, see [`DiskIo`].
    pub disk_io: DiskIo,
    /// GPUs.
    pub gpus: Vec<GpuStats>,
    /// This process.
    pub process: ProcessStats,
    /// When it was sampled, milliseconds since the Unix epoch (`0` before the
    /// first sample).
    pub sampled_at_ms: u64,
}
