//! The balancer's fleet: every upstream's machine folded into one snapshot.
//!
//! The terminal UI draws a balancer's resource panel with the same code as a
//! server's, so the fleet is a [`StatsSnapshot`] whose `system` and `models`
//! add up the upstreams' own snapshots; latency, requests and diagnosis stay
//! the balancer's.

use crate::server::stats::{
    CpuStats, DiskIo, DiskStats, GpuStats, MemoryStats, StatsSnapshot, SystemSnapshot,
};

/// A sum where an unreported side counts as nothing, and two unreported
/// sides stay unreported.
fn add(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    }
}

/// Folds the nested snapshots of `balancer` into one fleet snapshot (its
/// `upstreams` stay empty; the caller keeps the balancer's own).
pub fn fleet(balancer: &StatsSnapshot) -> StatsSnapshot {
    let nested: Vec<(&str, &StatsSnapshot)> = balancer
        .upstreams
        .iter()
        // A down upstream's last snapshot is stale: it would still count its machine.
        .filter(|upstream| upstream.healthy)
        .filter_map(|upstream| {
            upstream
                .snapshot
                .as_deref()
                .map(|nested| (upstream.id.as_str(), nested))
        })
        .collect();

    let mut cpu = CpuStats::default();
    let mut memory = MemoryStats::default();
    let mut disk_io = DiskIo::default();
    let mut disks = Vec::new();
    let mut gpus: Vec<GpuStats> = Vec::new();
    let mut models = Vec::new();
    let mut sampled_at_ms = 0;
    for (id, snapshot) in &nested {
        let system = &snapshot.system;
        cpu.usage_pct += system.cpu.usage_pct;
        cpu.cores += system.cpu.cores;
        memory.total_bytes += system.memory.total_bytes;
        memory.used_bytes += system.memory.used_bytes;
        memory.available_bytes += system.memory.available_bytes;
        memory.swap_total_bytes += system.memory.swap_total_bytes;
        memory.swap_used_bytes += system.memory.swap_used_bytes;
        memory.swap_in_bytes_per_s = add(
            memory.swap_in_bytes_per_s,
            system.memory.swap_in_bytes_per_s,
        );
        memory.swap_out_bytes_per_s = add(
            memory.swap_out_bytes_per_s,
            system.memory.swap_out_bytes_per_s,
        );
        disk_io.read_bytes_per_s += system.disk_io.read_bytes_per_s;
        disk_io.write_bytes_per_s += system.disk_io.write_bytes_per_s;
        sampled_at_ms = sampled_at_ms.max(system.sampled_at_ms);
        disks.extend(system.disks.iter().map(|disk| DiskStats {
            mount: format!("{id} {}", disk.mount),
            ..disk.clone()
        }));
        let base = gpus.len();
        gpus.extend(system.gpus.iter().enumerate().map(|(i, gpu)| GpuStats {
            index: base + i,
            name: format!("{id}: {}", gpu.name),
            ..gpu.clone()
        }));
        models.extend(snapshot.models.iter().cloned());
    }
    if !nested.is_empty() {
        // The mean across machines: a fleet of four at 50% is a fleet at 50%.
        cpu.usage_pct /= nested.len() as f32;
    }

    StatsSnapshot {
        node: balancer.node.clone(),
        models,
        latency: balancer.latency,
        requests: balancer.requests,
        system: SystemSnapshot {
            cpu,
            memory,
            disks,
            disk_io,
            gpus,
            sampled_at_ms,
            ..SystemSnapshot::default()
        },
        diagnosis: balancer.diagnosis.clone(),
        upstreams: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tui::tests::balancer_snapshot;

    #[test]
    fn the_fleet_adds_up_memory_and_averages_the_cpu() {
        let balancer = balancer_snapshot();
        let rolled = fleet(&balancer);
        let nested: Vec<&StatsSnapshot> = balancer
            .upstreams
            .iter()
            .filter(|upstream| upstream.healthy)
            .filter_map(|upstream| upstream.snapshot.as_deref())
            .collect();
        assert_eq!(
            rolled.system.memory.used_bytes,
            nested
                .iter()
                .map(|s| s.system.memory.used_bytes)
                .sum::<u64>()
        );
        let mean = nested.iter().map(|s| s.system.cpu.usage_pct).sum::<f32>() / nested.len() as f32;
        assert!((rolled.system.cpu.usage_pct - mean).abs() < 1e-3);
        assert_eq!(
            rolled.system.gpus.len(),
            nested.iter().map(|s| s.system.gpus.len()).sum::<usize>()
        );
        assert!(rolled.upstreams.is_empty());
    }
}
