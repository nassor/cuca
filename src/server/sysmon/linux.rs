//! Linux GPU probe: `sysfs` for amdgpu, i915 and xe, NVML for NVIDIA.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::struct_wrappers::device::ProcessInfo;

use super::MAX_GPUS;
use super::parse::{
    parse_drm_fdinfo, parse_kernel_page_size, parse_sysfs_hex, parse_sysfs_u64,
    parse_uevent_pci_id, parse_vmstat_swap, vendor_name,
};
use crate::server::stats::GpuStats;

const DRM_DIR: &str = "/sys/class/drm";
const NVIDIA_VENDOR_ID: u32 = 0x10de;

/// Reads GPU figures from the kernel's DRM `sysfs` tree and from NVML.
///
/// * NVIDIA: NVML (`libnvidia-ml`, loaded at run time; its absence is not an
///   error): utilisation, memory, temperature, power and this process's memory.
///   A card NVML lists is skipped in the `sysfs` pass.
/// * Others: `/sys/class/drm/card*/device`: `gpu_busy_percent`,
///   `mem_info_vram_used`/`_total`, hwmon `temp1_input` and `power1_average`
///   (amdgpu; i915 and xe expose fewer files, the rest stay `None`). This
///   process's VRAM is read from `/proc/self/fdinfo` DRM clients, matched to the
///   card by PCI address.
pub(super) struct GpuProbe {
    pid: u32,
    nvml: Option<Nvml>,
}

impl GpuProbe {
    pub(super) fn new() -> Self {
        let nvml = Nvml::init()
            .inspect_err(|e| tracing::debug!("sysmon: NVML unavailable: {e}"))
            .ok();
        Self {
            pid: std::process::id(),
            nvml,
        }
    }

    pub(super) fn sample(&mut self) -> Vec<GpuStats> {
        let mut gpus = Vec::new();
        if let Some(nvml) = &self.nvml {
            sample_nvml(nvml, self.pid, &mut gpus);
        }
        let nvml_listed = !gpus.is_empty();
        let cards = drm_cards();
        // fdinfo is only worth scanning when a non-NVML card will use it.
        let own_vram = cards
            .iter()
            .any(|c| !(nvml_listed && vendor_of(c) == Some(NVIDIA_VENDOR_ID)))
            .then(own_drm_vram)
            .flatten();
        for card in &cards {
            if gpus.len() == MAX_GPUS {
                break;
            }
            let vendor_id = vendor_of(card);
            if nvml_listed && vendor_id == Some(NVIDIA_VENDOR_ID) {
                continue;
            }
            if let Some(g) = sysfs_gpu(card, gpus.len(), own_vram.as_ref()) {
                gpus.push(g);
            }
        }
        gpus
    }
}

/// NVIDIA devices through NVML, appended to `out`.
fn sample_nvml(nvml: &Nvml, pid: u32, out: &mut Vec<GpuStats>) {
    let Ok(count) = nvml.device_count() else {
        return;
    };
    for i in 0..count {
        if out.len() == MAX_GPUS {
            return;
        }
        let Ok(dev) = nvml.device_by_index(i) else {
            continue;
        };
        let mem = dev.memory_info().ok();
        out.push(GpuStats {
            index: out.len(),
            name: dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}")),
            vendor: "nvidia".to_owned(),
            util_pct: dev.utilization_rates().ok().map(|u| u.gpu as f32),
            vram_total_bytes: mem.as_ref().map(|m| m.total),
            vram_used_bytes: mem.as_ref().map(|m| m.used),
            shared_vram_used_bytes: None,
            process_vram_bytes: nvml_process_vram(
                [
                    dev.running_compute_processes(),
                    dev.running_graphics_processes(),
                ],
                pid,
            ),
            temperature_c: dev
                .temperature(TemperatureSensor::Gpu)
                .ok()
                .map(|t| t as f32),
            power_w: dev.power_usage().ok().map(|mw| mw as f32 / 1000.0),
            source: "nvml".to_owned(),
        });
    }
}

/// Memory `pid` holds across NVML's compute and graphics process lists; `None`
/// if neither list could be read.
fn nvml_process_vram<E>(lists: [Result<Vec<ProcessInfo>, E>; 2], pid: u32) -> Option<u64> {
    let mut total = None;
    for list in lists.into_iter().flatten() {
        let held = list
            .iter()
            .filter(|p| p.pid == pid)
            .map(|p| match p.used_gpu_memory {
                UsedGpuMemory::Used(b) => b,
                UsedGpuMemory::Unavailable => 0,
            })
            .sum::<u64>();
        *total.get_or_insert(0) += held;
    }
    total
}

/// The `/sys/class/drm/card<N>` directories (not their connectors), by number.
fn drm_cards() -> Vec<PathBuf> {
    let Ok(dir) = fs::read_dir(DRM_DIR) else {
        return Vec::new();
    };
    let mut cards: Vec<(u32, PathBuf)> = dir
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_str()?.strip_prefix("card")?.parse().ok()?;
            Some((n, e.path()))
        })
        .collect();
    cards.sort_unstable();
    cards.into_iter().map(|(_, p)| p).collect()
}

fn read(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn vendor_of(card: &Path) -> Option<u32> {
    parse_sysfs_hex(&read(&card.join("device/vendor"))?)
}

/// One card's `sysfs` figures; `None` for a node with no PCI vendor (virtual).
fn sysfs_gpu(
    card: &Path,
    index: usize,
    own_vram: Option<&HashMap<String, u64>>,
) -> Option<GpuStats> {
    let dev = card.join("device");
    let vendor_id = vendor_of(card)?;
    let num = |name: &str| read(&dev.join(name)).and_then(|t| parse_sysfs_u64(&t));
    let pci_id = read(&dev.join("uevent")).and_then(|t| parse_uevent_pci_id(&t));
    let vendor = vendor_name(vendor_id);
    let name = read(&dev.join("product_name"))
        .map(|n| n.trim().to_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| match &pci_id {
            Some(id) => format!("GPU {id}"),
            None => "GPU".to_owned(),
        });
    let hwmon = first_hwmon(&dev);
    let hw = |file: &str| {
        hwmon
            .as_ref()
            .and_then(|d| read(&d.join(file)))
            .and_then(|t| parse_sysfs_u64(&t))
    };
    #[expect(clippy::cast_precision_loss, reason = "sensor values far below 2^24")]
    let (temperature_c, power_w) = (
        hw("temp1_input").map(|m| m as f32 / 1000.0),
        hw("power1_average")
            .or_else(|| hw("power1_input"))
            .map(|uw| uw as f32 / 1_000_000.0),
    );
    // The card's PCI address, the key of its fdinfo clients.
    let process_vram_bytes = own_vram.and_then(|m| {
        let pdev = fs::canonicalize(&dev)
            .ok()?
            .file_name()?
            .to_str()?
            .to_owned();
        Some(m.get(&pdev).copied().unwrap_or(0))
    });
    Some(GpuStats {
        index,
        name,
        vendor: vendor.to_owned(),
        #[expect(clippy::cast_precision_loss, reason = "a percentage")]
        util_pct: num("gpu_busy_percent").map(|p| p as f32),
        vram_total_bytes: num("mem_info_vram_total"),
        vram_used_bytes: num("mem_info_vram_used"),
        process_vram_bytes,
        shared_vram_used_bytes: num("mem_info_gtt_used"),
        temperature_c,
        power_w,
        source: "sysfs".to_owned(),
    })
}

/// The first `device/hwmon/hwmon*` directory.
fn first_hwmon(dev: &Path) -> Option<PathBuf> {
    fs::read_dir(dev.join("hwmon"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .min()
}

/// Device memory this process holds per PCI address, from the DRM clients among
/// its open files; `None` if `/proc/self/fdinfo` cannot be listed.
fn own_drm_vram() -> Option<HashMap<String, u64>> {
    let mut seen: HashSet<(String, u64)> = HashSet::new();
    let mut per_device: HashMap<String, u64> = HashMap::new();
    for entry in fs::read_dir("/proc/self/fdinfo").ok()?.flatten() {
        let Some(client) = read(&entry.path()).and_then(|t| parse_drm_fdinfo(&t)) else {
            continue;
        };
        // One client may be open on several descriptors; count it once.
        if seen.insert((client.pdev.clone(), client.client_id)) {
            *per_device.entry(client.pdev).or_default() += client.vram_bytes;
        }
    }
    Some(per_device)
}

/// Page-in and page-out rates from `/proc/vmstat` (`pswpin`, `pswpout`).
pub(super) struct SwapProbe {
    page_bytes: u64,
    prev: Option<(u64, u64)>,
}

impl SwapProbe {
    pub(super) fn new() -> Self {
        // The kernel page size from this process's first mapping; 4 KiB if unreadable.
        let page_bytes = read(Path::new("/proc/self/smaps"))
            .and_then(|t| parse_kernel_page_size(&t))
            .unwrap_or(4096);
        Self {
            page_bytes,
            prev: None,
        }
    }

    /// `(in, out)` in bytes per second; `(None, None)` on the first tick or if
    /// `/proc/vmstat` is unreadable.
    pub(super) fn sample(&mut self, dt_s: f64) -> (Option<f64>, Option<f64>) {
        let now = read(Path::new("/proc/vmstat")).and_then(|t| parse_vmstat_swap(&t));
        let prev = std::mem::replace(&mut self.prev, now);
        let (Some(now), Some(prev)) = (now, prev) else {
            return (None, None);
        };
        #[expect(clippy::cast_precision_loss, reason = "counters far below 2^53")]
        let rate = |now: u64, prev: u64| {
            Some(now.saturating_sub(prev).saturating_mul(self.page_bytes) as f64 / dt_s.max(1e-3))
        };
        (rate(now.0, prev.0), rate(now.1, prev.1))
    }
}
