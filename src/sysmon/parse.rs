//! Pure text parsers for the per-platform GPU probes.
//!
//! They live apart from the probes so their tests run on every platform; each
//! platform uses only its own subset, hence the non-test `dead_code` expectation
//! on the module declaration.

/// A parsed Windows GPU performance-counter instance name.
///
/// The three GPU counter classes name their instances alike:
/// `luid_0x<high>_0x<low>_phys_<n>` (adapter memory),
/// `pid_<pid>_luid_..._phys_<n>` (process memory) and
/// `pid_<pid>_luid_..._phys_<n>_eng_<n>_engtype_<type>` (engine time).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GpuInstance {
    /// Owning process, absent for adapter-wide instances.
    pub(super) pid: Option<u32>,
    /// The adapter's LUID as `high << 32 | low`, the form of the `AdapterLuid`
    /// registry value.
    pub(super) luid: u64,
    /// Physical adapter index inside the LUID (linked adapters).
    pub(super) phys: u32,
    /// Engine index, present on engine instances only.
    pub(super) engine: Option<u32>,
}

/// Parses a WMI GPU counter instance name, `None` if it is not one.
pub(super) fn parse_gpu_instance(name: &str) -> Option<GpuInstance> {
    let (pid, rest) = match name.strip_prefix("pid_") {
        Some(r) => {
            let (pid, rest) = r.split_once('_')?;
            (Some(pid.parse().ok()?), rest)
        }
        None => (None, name),
    };
    let rest = rest.strip_prefix("luid_0x")?;
    let (high, rest) = rest.split_once("_0x")?;
    let (low, rest) = rest.split_once("_phys_")?;
    let luid = u64::from(u32::from_str_radix(high, 16).ok()?) << 32
        | u64::from(u32::from_str_radix(low, 16).ok()?);
    let (phys, engine) = match rest.split_once("_eng_") {
        None => (rest.parse().ok()?, None),
        Some((phys, tail)) => {
            let (engine, _engine_type) = tail.split_once("_engtype_")?;
            (phys.parse().ok()?, Some(engine.parse().ok()?))
        }
    };
    Some(GpuInstance {
        pid,
        luid,
        phys,
        engine,
    })
}

/// The wire vendor name (`amd`, `nvidia`, `intel`, `apple`, else `unknown`) for a
/// PCI/DXGI vendor id.
pub(super) fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 | 0x1022 => "amd",
        0x10de => "nvidia",
        0x8086 => "intel",
        0x106b => "apple",
        _ => "unknown",
    }
}

/// Parses a sysfs integer file such as `gpu_busy_percent` or `mem_info_vram_used`.
pub(super) fn parse_sysfs_u64(text: &str) -> Option<u64> {
    text.trim().parse().ok()
}

/// Parses a sysfs hexadecimal id file such as `vendor` (`0x1002`).
pub(super) fn parse_sysfs_hex(text: &str) -> Option<u32> {
    let t = text.trim();
    u32::from_str_radix(t.strip_prefix("0x").unwrap_or(t), 16).ok()
}

/// `(pswpin, pswpout)` page counts from `/proc/vmstat`.
pub(super) fn parse_vmstat_swap(text: &str) -> Option<(u64, u64)> {
    let (mut pin, mut pout) = (None, None);
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("pswpin", v)) => pin = v.trim().parse().ok(),
            Some(("pswpout", v)) => pout = v.trim().parse().ok(),
            _ => {}
        }
    }
    Some((pin?, pout?))
}

/// The first `KernelPageSize:` of a `smaps` file, in bytes.
pub(super) fn parse_kernel_page_size(text: &str) -> Option<u64> {
    let rest = text
        .lines()
        .find_map(|l| l.strip_prefix("KernelPageSize:"))?;
    let kib: u64 = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kib.checked_mul(1024)
}

/// The `PCI_ID=` value of a sysfs `uevent` file, as `vendor:device` lowercase hex.
pub(super) fn parse_uevent_pci_id(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("PCI_ID="))
        .map(|id| id.trim().to_ascii_lowercase())
}

/// One DRM client read from `/proc/<pid>/fdinfo/<fd>`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct DrmClient {
    /// PCI address of the device, `drm-pdev` (`0000:03:00.0`).
    pub(super) pdev: String,
    /// Per-driver client id; several fds of one client repeat it.
    pub(super) client_id: u64,
    /// Device-local memory the client holds, in bytes.
    pub(super) vram_bytes: u64,
}

/// Parses the DRM fdinfo text of one file descriptor, `None` if it is not a DRM
/// client with a PCI address and client id.
///
/// Device memory is `drm-memory-vram` (amdgpu), else the sum of the
/// `drm-total-vram*` and `drm-total-local*` regions (xe, i915).
pub(super) fn parse_drm_fdinfo(text: &str) -> Option<DrmClient> {
    let (mut pdev, mut client_id) = (None, None);
    let (mut memory_vram, mut total) = (None, 0u64);
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key {
            "drm-pdev" => pdev = Some(value.to_owned()),
            "drm-client-id" => client_id = value.parse().ok(),
            "drm-memory-vram" => memory_vram = parse_drm_size(value),
            k if k.starts_with("drm-total-vram") || k.starts_with("drm-total-local") => {
                total = total.saturating_add(parse_drm_size(value).unwrap_or(0));
            }
            _ => {}
        }
    }
    Some(DrmClient {
        pdev: pdev?,
        client_id: client_id?,
        vram_bytes: memory_vram.unwrap_or(total),
    })
}

/// `"123 KiB"` / `"4 MiB"` / `"512"` (bytes) to bytes.
fn parse_drm_size(value: &str) -> Option<u64> {
    let mut it = value.split_whitespace();
    let n: u64 = it.next()?.parse().ok()?;
    let shift = match it.next() {
        None => 0,
        Some("KiB") => 10,
        Some("MiB") => 20,
        Some("GiB") => 30,
        Some(_) => return None,
    };
    n.checked_mul(1 << shift)
}

/// One accelerator parsed from `ioreg -r -d 1 -c IOAccelerator`.
#[derive(Debug, PartialEq)]
pub(super) struct IoregGpu {
    pub(super) name: String,
    /// `"Device Utilization %"`.
    pub(super) util_pct: Option<f32>,
    /// `"In use system memory"`, bytes of unified memory the GPU holds.
    pub(super) in_use_bytes: Option<u64>,
}

/// Parses `ioreg -r -d 1 -c IOAccelerator` output, one entry per `+-o` node.
pub(super) fn parse_ioreg(text: &str) -> Vec<IoregGpu> {
    let mut blocks: Vec<&str> = Vec::new();
    let mut start = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.trim_start().starts_with("+-o ")
            && let Some(s) = start.replace(offset)
        {
            blocks.push(&text[s..offset]);
        }
        offset += line.len();
    }
    if let Some(s) = start {
        blocks.push(&text[s..]);
    }
    blocks
        .into_iter()
        .map(|b| IoregGpu {
            name: ioreg_model(b).unwrap_or_else(|| ioreg_class(b)),
            util_pct: ioreg_number(b, "Device Utilization %").map(|n| n as f32),
            in_use_bytes: ioreg_number(b, "In use system memory"),
        })
        .collect()
}

/// `"model" = "Apple M2 Pro"`.
fn ioreg_model(block: &str) -> Option<String> {
    let rest = block.split_once("\"model\" = \"")?.1;
    Some(rest.split_once('"')?.0.to_owned())
}

/// The class name on the node's `+-o <class>  <...>` line.
fn ioreg_class(block: &str) -> String {
    block
        .trim_start()
        .strip_prefix("+-o ")
        .and_then(|r| r.split_whitespace().next())
        .unwrap_or("GPU")
        .to_owned()
}

/// The integer after `"<key>"=` (spaces around `=` tolerated).
fn ioreg_number(block: &str, key: &str) -> Option<u64> {
    let rest = block.split_once(&format!("\"{key}\""))?.1;
    let rest = rest.trim_start().strip_prefix('=')?.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_instance_names_parse_with_pid_luid_and_engine() {
        let i = parse_gpu_instance(
            "pid_55564_luid_0x00000000_0x00017C37_phys_0_eng_7_engtype_Video Codec Engine",
        );
        assert_eq!(
            i,
            Some(GpuInstance {
                pid: Some(55564),
                luid: 0x17c37,
                phys: 0,
                engine: Some(7)
            })
        );
    }

    #[test]
    fn adapter_and_process_memory_instance_names_parse() {
        let a = parse_gpu_instance("luid_0x00000001_0x0001A4B0_phys_2").unwrap();
        assert_eq!(
            (a.pid, a.luid, a.phys, a.engine),
            (None, 0x1_0001_a4b0, 2, None)
        );
        let p = parse_gpu_instance("pid_14840_luid_0x00000000_0x0001A4B0_phys_0").unwrap();
        assert_eq!((p.pid, p.luid), (Some(14840), 0x1a4b0));
    }

    #[test]
    fn malformed_instance_names_are_rejected() {
        for bad in [
            "",
            "_Total",
            "pid_x_luid_0x0_0x1_phys_0",
            "luid_0x0_0x1",
            "luid_0x0_0x1_phys_0_eng_1",
        ] {
            assert_eq!(parse_gpu_instance(bad), None, "{bad}");
        }
    }

    #[test]
    fn sysfs_values_parse_with_trailing_newline() {
        assert_eq!(parse_sysfs_u64("17163091968\n"), Some(17_163_091_968));
        assert_eq!(parse_sysfs_u64("n/a\n"), None);
        assert_eq!(parse_sysfs_hex("0x1002\n"), Some(0x1002));
        assert_eq!(parse_sysfs_hex("10de"), Some(0x10de));
        assert_eq!(
            parse_uevent_pci_id("DRIVER=amdgpu\nPCI_ID=1002:7550\n"),
            Some("1002:7550".into())
        );
    }

    #[test]
    fn vmstat_swap_counters_and_kernel_page_size_parse() {
        let vm = "nr_free_pages 10\npswpin 120\npswpout 7\npgpgin 5\n";
        assert_eq!(parse_vmstat_swap(vm), Some((120, 7)));
        assert_eq!(parse_vmstat_swap("pswpin 1\n"), None);
        let smaps = "Size: 4 kB\nKernelPageSize:        64 kB\nMMUPageSize: 4 kB\n";
        assert_eq!(parse_kernel_page_size(smaps), Some(65536));
    }

    #[test]
    fn amdgpu_fdinfo_reports_vram_in_bytes() {
        let text = "pos:\t0\ndrm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t42\n\
                    drm-memory-vram:\t2048 KiB\ndrm-memory-gtt:\t64 KiB\n";
        assert_eq!(
            parse_drm_fdinfo(text),
            Some(DrmClient {
                pdev: "0000:03:00.0".into(),
                client_id: 42,
                vram_bytes: 2 << 20
            })
        );
    }

    #[test]
    fn xe_fdinfo_sums_device_regions_and_ignores_system_memory() {
        let text = "drm-pdev:\t0000:00:02.0\ndrm-client-id:\t7\ndrm-total-vram0:\t3 MiB\n\
                    drm-total-system0:\t9 MiB\n";
        assert_eq!(parse_drm_fdinfo(text).map(|c| c.vram_bytes), Some(3 << 20));
    }

    #[test]
    fn non_drm_fdinfo_is_ignored() {
        assert_eq!(parse_drm_fdinfo("pos:\t0\nflags:\t02\n"), None);
    }

    #[test]
    fn ioreg_output_yields_name_utilization_and_memory_per_accelerator() {
        let text = "+-o AGXAcceleratorG14X  <class AGXAcceleratorG14X, id 0x100000a3c, registered>\n\
            {\n  \"model\" = \"Apple M2 Pro\"\n  \"PerformanceStatistics\" = {\"Alloc system memory\"=1883242496,\
            \"Device Utilization %\"=17,\"In use system memory\"=1018314752}\n}\n\
            +-o AGXAcceleratorG14X  <class AGXAcceleratorG14X, id 0x100000a3d, registered>\n\
            {\n  \"PerformanceStatistics\" = {\"Device Utilization %\"=0}\n}\n";
        assert_eq!(
            parse_ioreg(text),
            vec![
                IoregGpu {
                    name: "Apple M2 Pro".into(),
                    util_pct: Some(17.0),
                    in_use_bytes: Some(1_018_314_752)
                },
                IoregGpu {
                    name: "AGXAcceleratorG14X".into(),
                    util_pct: Some(0.0),
                    in_use_bytes: None
                },
            ]
        );
    }

    #[test]
    fn empty_ioreg_output_has_no_accelerators() {
        assert!(parse_ioreg("").is_empty());
    }

    #[test]
    fn vendor_ids_map_to_names() {
        assert_eq!(vendor_name(0x1002), "amd");
        assert_eq!(vendor_name(0x10de), "nvidia");
        assert_eq!(vendor_name(0xabcd), "unknown");
    }
}
