//! Windows GPU probe: WMI performance counters plus the DirectX adapter registry.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use winreg::RegKey;
use winreg::enums::HKEY_LOCAL_MACHINE;
use wmi::WMIConnection;

use super::MAX_GPUS;
use super::parse::{parse_gpu_instance, vendor_name};
use crate::server::stats::GpuStats;

/// Ticks between WMI reconnection attempts after a failed connect.
const WMI_RETRY_TICKS: u32 = 15;
/// Minimum ticks between registry re-reads triggered by an unknown LUID.
const REGISTRY_RETRY_TICKS: u32 = 10;
/// Registry subkeys scanned under the DirectX key (stale adapters accumulate).
const MAX_REGISTRY_ADAPTERS: usize = 64;
/// Vendor id of the Microsoft Basic Render Driver (WARP), not a GPU to show.
const MICROSOFT_VENDOR_ID: u32 = 0x1414;

const ADAPTER_MEMORY_WQL: &str = "SELECT Name, DedicatedUsage, SharedUsage FROM \
    Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory";
const PROCESS_MEMORY_WQL: &str = "SELECT Name, DedicatedUsage, SharedUsage FROM \
    Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory";
// The raw class, not the formatted one: WMI keeps no refresher between one-shot
// queries, so the formatted `UtilizationPercentage` is always 0. `RunningTime`
// is the engine's cumulative busy time in 100 ns units, stamped with
// `Timestamp_Sys100NS`; the probe takes the delta itself. Idle instances are
// filtered out server-side.
const ENGINE_WQL: &str = "SELECT Name, RunningTime, Timestamp_Sys100NS FROM \
    Win32_PerfRawData_GPUPerformanceCounters_GPUEngine WHERE RunningTime > 0";

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct MemoryRow {
    name: String,
    dedicated_usage: Option<u64>,
    /// Selected by both memory queries (a missing column fails the row).
    shared_usage: Option<u64>,
}

/// Cumulative paging counters of `Win32_PerfRawData_PerfOS_Memory`: pages read
/// in and written out since boot, stamped in 100 ns units.
const PAGING_WQL: &str = "SELECT PagesInputPersec, PagesOutputPersec, Timestamp_Sys100NS FROM \
    Win32_PerfRawData_PerfOS_Memory";
/// Windows page size on x64 and ARM64.
const PAGE_BYTES: u64 = 4096;

#[derive(Deserialize)]
struct PagingRow {
    #[serde(rename = "PagesInputPersec")]
    pages_in: Option<u64>,
    #[serde(rename = "PagesOutputPersec")]
    pages_out: Option<u64>,
    #[serde(rename = "Timestamp_Sys100NS")]
    timestamp: Option<u64>,
}

/// Page-in and page-out rates from the raw paging counters.
///
/// Pages input counts every hard fault resolved from disk, memory-mapped file
/// reads included, not only pagefile traffic. Owns its own WMI connection (it
/// lives on the sampler thread like [`GpuProbe`]).
pub(super) struct SwapProbe {
    wmi: Option<WMIConnection>,
    connect_in: u32,
    /// `(pages in, pages out, stamp)` at the previous tick.
    prev: Option<(u64, u64, u64)>,
}

impl SwapProbe {
    pub(super) fn new() -> Self {
        Self {
            wmi: None,
            connect_in: 0,
            prev: None,
        }
    }

    /// `(in, out)` in bytes per second; `(None, None)` on the first tick or if
    /// the counters cannot be read.
    pub(super) fn sample(&mut self, _dt_s: f64) -> (Option<f64>, Option<f64>) {
        if self.wmi.is_none() {
            self.connect_in = self.connect_in.saturating_sub(1);
            if self.connect_in == 0 {
                match WMIConnection::new() {
                    Ok(con) => self.wmi = Some(con),
                    Err(e) => {
                        tracing::debug!("sysmon: WMI connect failed: {e}");
                        self.connect_in = WMI_RETRY_TICKS;
                    }
                }
            }
        }
        let row = self
            .wmi
            .as_ref()
            .and_then(|c| query::<PagingRow>(c, PAGING_WQL))
            .and_then(|r| r.into_iter().next());
        let now = row.and_then(|r| Some((r.pages_in?, r.pages_out?, r.timestamp?)));
        let prev = std::mem::replace(&mut self.prev, now);
        let (Some(now), Some(prev)) = (now, prev) else {
            return (None, None);
        };
        let Some(dt) = now.2.checked_sub(prev.2).filter(|d| *d > 0) else {
            return (None, None);
        };
        #[expect(clippy::cast_precision_loss, reason = "counters far below 2^53")]
        let rate = |now: u64, prev: u64| {
            Some(now.saturating_sub(prev).saturating_mul(PAGE_BYTES) as f64 * 1e7 / dt as f64)
        };
        (rate(now.0, prev.0), rate(now.1, prev.1))
    }
}

#[derive(Deserialize)]
struct EngineRow {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "RunningTime")]
    running_time: Option<u64>,
    #[serde(rename = "Timestamp_Sys100NS")]
    timestamp: Option<u64>,
}

/// An adapter from the registry's DirectX adapter table.
struct RegistryAdapter {
    luid: u64,
    name: String,
    vendor_id: u32,
    /// True dedicated VRAM size: `DedicatedVideoMemory` is a 64-bit value,
    /// unlike `Win32_VideoController.AdapterRAM`, which caps at 4 GiB.
    dedicated_bytes: u64,
    last_seen: u64,
}

/// Reads GPU load and memory from WMI performance counters.
///
/// * Adapters come from `HKLM\SOFTWARE\Microsoft\DirectX\{guid}`, which carries
///   the LUID, name, vendor and the 64-bit VRAM size. Mapping counters to
///   adapters is therefore exact by LUID, never by order. An adapter the
///   registry does not know is listed as `GPU 0x<luid>` without a size.
/// * Utilisation is the busiest engine: busy time summed over processes per
///   engine, then the maximum over engines, as Task Manager's GPU figure.
/// * VRAM in use is `GPUAdapterMemory.DedicatedUsage`; this process's share is
///   `GPUProcessMemory.DedicatedUsage` for its pid.
///
/// One instance lives on the sampler thread (a WMI connection is `!Send`).
pub(super) struct GpuProbe {
    pid: u32,
    wmi: Option<WMIConnection>,
    connect_in: u32,
    registry: Vec<RegistryAdapter>,
    registry_age: u32,
    /// Cumulative busy time per engine instance at the previous tick, and its stamp.
    engines: HashMap<String, u64>,
    engines_stamp: Option<u64>,
    /// Whether an engine sample has been taken, so a later empty one means idle.
    engines_seeded: bool,
}

impl GpuProbe {
    pub(super) fn new() -> Self {
        Self {
            pid: std::process::id(),
            wmi: None,
            connect_in: 0,
            registry: read_registry_adapters(),
            registry_age: 0,
            engines: HashMap::new(),
            engines_stamp: None,
            engines_seeded: false,
        }
    }

    pub(super) fn sample(&mut self) -> Vec<GpuStats> {
        self.registry_age = self.registry_age.saturating_add(1);
        self.connect();

        let (mem, process, util) = match &self.wmi {
            Some(con) => {
                let mem = query::<MemoryRow>(con, ADAPTER_MEMORY_WQL).map(|r| adapter_memory(&r));
                let process = query::<MemoryRow>(con, PROCESS_MEMORY_WQL)
                    .map(|r| process_memory(&r, self.pid));
                let engines = query::<EngineRow>(con, ENGINE_WQL);
                (mem, process, engines)
            }
            None => (None, None, None),
        };
        let engines_ok = util.is_some();
        let util = util.and_then(|rows| self.engine_utilization(&rows));
        // The first engine sample only seeds the delta; it is not a failure.
        let counters_ok =
            usize::from(mem.is_some()) + usize::from(process.is_some()) + usize::from(engines_ok);
        let source = match counters_ok {
            3 => "wmi+registry",
            0 => "registry",
            _ => "wmi-partial+registry",
        };

        // Without adapter counters, fall back to the adapters the registry saw last.
        let luids: Vec<u64> = match &mem {
            Some(m) => m.keys().copied().collect(),
            None => self.live_registry_luids(),
        };
        if luids.iter().any(|l| self.registry_adapter(*l).is_none())
            && self.registry_age >= REGISTRY_RETRY_TICKS
        {
            self.registry = read_registry_adapters();
            self.registry_age = 0;
        }

        let mut gpus = Vec::new();
        for luid in luids {
            let reg = self.registry_adapter(luid);
            if reg.is_some_and(|r| r.vendor_id == MICROSOFT_VENDOR_ID) {
                continue;
            }
            gpus.push(GpuStats {
                index: gpus.len(),
                name: reg.map_or_else(|| format!("GPU 0x{luid:x}"), |r| r.name.clone()),
                vendor: reg
                    .map_or("unknown", |r| vendor_name(r.vendor_id))
                    .to_owned(),
                util_pct: util.as_ref().map(|u| u.get(&luid).copied().unwrap_or(0.0)),
                vram_total_bytes: reg.map(|r| r.dedicated_bytes).filter(|b| *b > 0),
                vram_used_bytes: mem.as_ref().and_then(|m| m.get(&luid)).map(|m| m.0),
                shared_vram_used_bytes: mem.as_ref().and_then(|m| m.get(&luid)).map(|m| m.1),
                process_vram_bytes: process.as_ref().map(|p| p.get(&luid).copied().unwrap_or(0)),
                temperature_c: None,
                power_w: None,
                source: source.to_owned(),
            });
            if gpus.len() == MAX_GPUS {
                break;
            }
        }
        gpus
    }

    /// Connects to WMI on this thread if not connected (COM is initialised by
    /// the `wmi` crate), retrying every [`WMI_RETRY_TICKS`] after a failure.
    fn connect(&mut self) {
        if self.wmi.is_some() {
            return;
        }
        self.connect_in = self.connect_in.saturating_sub(1);
        if self.connect_in > 0 {
            return;
        }
        match WMIConnection::new() {
            Ok(con) => self.wmi = Some(con),
            Err(e) => {
                tracing::debug!("sysmon: WMI connect failed: {e}");
                self.connect_in = WMI_RETRY_TICKS;
            }
        }
    }

    /// The registry entry for `luid`; of stale duplicates, the most recently seen.
    fn registry_adapter(&self, luid: u64) -> Option<&RegistryAdapter> {
        self.registry
            .iter()
            .filter(|a| a.luid == luid)
            .max_by_key(|a| a.last_seen)
    }

    /// LUIDs of the adapters seen in the latest registry generation.
    fn live_registry_luids(&self) -> Vec<u64> {
        let newest = self.registry.iter().map(|a| a.last_seen).max().unwrap_or(0);
        let mut luids: Vec<u64> = self
            .registry
            .iter()
            .filter(|a| a.last_seen == newest)
            .map(|a| a.luid)
            .collect();
        luids.sort_unstable();
        luids.dedup();
        luids
    }

    /// Busiest-engine utilisation per adapter LUID from the raw busy-time delta
    /// since the previous call; `None` on the first call (nothing to diff yet).
    /// An engine query that returns no busy engine is an idle machine: `Some` of
    /// an empty map, which the caller reads as 0 % for every adapter.
    fn engine_utilization(&mut self, rows: &[EngineRow]) -> Option<BTreeMap<u64, f32>> {
        let stamp = rows.iter().find_map(|r| r.timestamp);
        let prev_stamp = self.engines_stamp;
        let prev = std::mem::take(&mut self.engines);
        let mut busy: HashMap<(u64, u32, u32), u64> = HashMap::new();
        for row in rows {
            let (Some(running), Some(inst)) = (row.running_time, parse_gpu_instance(&row.name))
            else {
                continue;
            };
            let Some(engine) = inst.engine else { continue };
            // A new instance counts from zero; one that went backwards was recreated.
            let before = prev.get(&row.name).copied().unwrap_or(0);
            let delta = if running >= before {
                running - before
            } else {
                running
            };
            *busy.entry((inst.luid, inst.phys, engine)).or_default() += delta;
            self.engines.insert(row.name.clone(), running);
        }
        // An empty result carries no stamp: an idle machine keeps the previous one.
        self.engines_stamp = stamp.or(prev_stamp);

        // No stamped row: nothing ran on any engine, which after the first call is 0 %.
        let seeded = std::mem::replace(&mut self.engines_seeded, true);
        let Some(stamp) = stamp else {
            return seeded.then(BTreeMap::new);
        };
        let dt = stamp.checked_sub(prev_stamp?).filter(|d| *d > 0)?;
        let mut util: BTreeMap<u64, f32> = BTreeMap::new();
        for ((luid, _, _), delta) in busy {
            #[expect(
                clippy::cast_precision_loss,
                reason = "busy time and interval far below 2^53"
            )]
            let pct = (delta as f64 / dt as f64 * 100.0).min(100.0) as f32;
            let slot = util.entry(luid).or_insert(0.0);
            *slot = slot.max(pct);
        }
        Some(util)
    }
}

/// Runs a WQL query, logging (at debug) and swallowing a failure.
fn query<T: serde::de::DeserializeOwned>(con: &WMIConnection, wql: &str) -> Option<Vec<T>> {
    con.raw_query(wql)
        .inspect_err(|e| tracing::debug!("sysmon: WMI query failed ({wql}): {e}"))
        .ok()
}

/// `(dedicated, shared)` memory in use per adapter LUID (linked physical
/// adapters summed).
fn adapter_memory(rows: &[MemoryRow]) -> BTreeMap<u64, (u64, u64)> {
    let mut out: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    for row in rows {
        if let Some(inst) = parse_gpu_instance(&row.name) {
            let slot = out.entry(inst.luid).or_default();
            slot.0 += row.dedicated_usage.unwrap_or(0);
            slot.1 += row.shared_usage.unwrap_or(0);
        }
    }
    out
}

/// Dedicated VRAM held by `pid`, per adapter LUID.
fn process_memory(rows: &[MemoryRow], pid: u32) -> BTreeMap<u64, u64> {
    let mut out = BTreeMap::new();
    for row in rows {
        if let Some(inst) = parse_gpu_instance(&row.name).filter(|i| i.pid == Some(pid)) {
            *out.entry(inst.luid).or_default() += row.dedicated_usage.unwrap_or(0);
        }
    }
    out
}

/// The DirectX adapter table; empty if the key is absent (very old Windows).
fn read_registry_adapters() -> Vec<RegistryAdapter> {
    let Ok(root) = RegKey::predef(HKEY_LOCAL_MACHINE).open_subkey(r"SOFTWARE\Microsoft\DirectX")
    else {
        return Vec::new();
    };
    root.enum_keys()
        .flatten()
        .filter(|k| k.starts_with('{'))
        .take(MAX_REGISTRY_ADAPTERS)
        .filter_map(|k| {
            let key = root.open_subkey(k).ok()?;
            Some(RegistryAdapter {
                luid: key.get_value("AdapterLuid").ok()?,
                name: key.get_value("Description").ok()?,
                vendor_id: key.get_value("VendorId").unwrap_or(0),
                dedicated_bytes: key.get_value("DedicatedVideoMemory").unwrap_or(0),
                last_seen: key.get_value("LastSeen").unwrap_or(0),
            })
        })
        .collect()
}
