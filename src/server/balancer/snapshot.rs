//! The balancer's [`StatsSnapshot`] (`role: balancer`).
//!
//! * `upstreams`: per upstream, health, in-flight and totals, the latencies
//!   the balancer measured (time to first body byte, end to end) and the
//!   upstream's own latest snapshot when it has the stats API.
//! * `models`: the union of the models the upstreams serve, summed over the
//!   healthy upstreams' snapshots ([`union_models`]).
//! * `latency` / `requests`: the balancer's own, over all upstreams.
//! * `system`: the balancer host's.
//! * `diagnosis`: the balancer's causes ([`super::diagnosis`]), then each
//!   upstream's diagnoses tagged with the upstream.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::diagnosis::{Thresholds, UpstreamSignal, diagnose, relabel};
use super::routing::{Policy, load_score};
use super::state::BalancerState;
use super::upstream::UpstreamModel;
use crate::server::stats::{
    KvStats, ModelMemory, ModelState, ModelStats, NodeInfo, NodeRole, RequestStats, StatsSnapshot,
    TokensPerSecond, UpstreamStats,
};

/// One upstream as [`union_models`] sees it.
pub struct UpstreamInput {
    /// The upstream's id.
    pub id: String,
    /// Whether it takes requests.
    pub healthy: bool,
    /// The models it last reported.
    pub models: Vec<UpstreamModel>,
    /// Its latest stats snapshot, when fresh.
    pub snapshot: Option<Arc<StatsSnapshot>>,
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// The models of all upstreams as one list, sorted by name. A model served
/// by a healthy upstream is `Ready`, one only known from unhealthy upstreams
/// `Failed`. Counts and rates are summed over the healthy upstreams whose
/// snapshot lists the model; means are weighted by the rates they describe;
/// latency and request rates are left for the caller to fill in.
pub fn union_models(inputs: &[UpstreamInput]) -> Vec<ModelStats> {
    let mut names: BTreeMap<&str, (bool, u64)> = BTreeMap::new();
    for input in inputs {
        for m in &input.models {
            let entry = names.entry(&m.id).or_insert((false, 0));
            entry.0 |= input.healthy;
            entry.1 = entry.1.max(m.max_model_len);
        }
        if input.healthy
            && let Some(snapshot) = &input.snapshot
        {
            for m in &snapshot.models {
                let entry = names.entry(&m.name).or_insert((true, 0));
                entry.0 = true;
                entry.1 = entry.1.max(m.context_length);
            }
        }
    }
    names
        .into_iter()
        .map(|(name, (ready, context))| {
            let mut stats = ModelStats {
                name: name.to_string(),
                context_length: context,
                state: if ready {
                    ModelState::Ready
                } else {
                    ModelState::Failed
                },
                error: (!ready).then(|| "no healthy upstream serves it".to_string()),
                ..ModelStats::default()
            };
            let parts: Vec<&ModelStats> = inputs
                .iter()
                .filter(|i| i.healthy)
                .filter_map(|i| i.snapshot.as_deref())
                .flat_map(|s| s.models.iter().filter(|m| m.name == name))
                .filter(|m| m.state == ModelState::Ready)
                .collect();
            aggregate(&mut stats, &parts);
            stats
        })
        .collect()
}

fn aggregate(out: &mut ModelStats, parts: &[&ModelStats]) {
    if parts.is_empty() {
        return;
    }
    let sum_u = |f: fn(&ModelStats) -> u64| parts.iter().map(|m| f(m)).sum::<u64>();
    let sum_f = |f: fn(&ModelStats) -> f64| parts.iter().map(|m| f(m)).sum::<f64>();
    let weighted = |value: fn(&ModelStats) -> f64, weight: fn(&ModelStats) -> f64| {
        ratio(
            parts.iter().map(|m| value(m) * weight(m)).sum(),
            parts.iter().map(|m| weight(m)).sum(),
        )
    };
    out.source = parts
        .iter()
        .map(|m| m.source.as_str())
        .find(|s| !s.is_empty())
        .unwrap_or("")
        .to_string();
    let mut devices: Vec<&str> = parts.iter().map(|m| m.device.as_str()).collect();
    devices.sort_unstable();
    devices.dedup();
    out.device = devices.join(",");
    out.memory = ModelMemory {
        weights_bytes: sum_u(|m| m.memory.weights_bytes),
        kv_bytes: sum_u(|m| m.memory.kv_bytes),
        device: out.device.clone(),
    };
    out.max_sequences = sum_u(|m| m.max_sequences);
    out.running = sum_u(|m| m.running);
    out.waiting = sum_u(|m| m.waiting);
    out.kv = KvStats {
        total_blocks: sum_u(|m| m.kv.total_blocks),
        free_blocks: sum_u(|m| m.kv.free_blocks),
        cached_blocks: sum_u(|m| m.kv.cached_blocks),
        used_blocks: sum_u(|m| m.kv.used_blocks),
        bytes: sum_u(|m| m.kv.bytes),
        usage_ratio: 0.0,
    };
    out.kv.usage_ratio = ratio(out.kv.used_blocks as f64, out.kv.total_blocks as f64);
    let hits: Vec<f64> = parts.iter().filter_map(|m| m.prefix_hit_ratio).collect();
    out.prefix_hit_ratio = (!hits.is_empty()).then(|| hits.iter().sum::<f64>() / hits.len() as f64);
    out.preemptions_per_min = sum_f(|m| m.preemptions_per_min);
    out.tokens_per_s = TokensPerSecond {
        prompt: sum_f(|m| m.tokens_per_s.prompt),
        generation: sum_f(|m| m.tokens_per_s.generation),
    };
    out.steps_per_s = sum_f(|m| m.steps_per_s);
    out.rows_per_s = sum_f(|m| m.rows_per_s);
    out.mean_rows_per_step = ratio(out.rows_per_s, out.steps_per_s);
    out.prefill_row_share = weighted(|m| m.prefill_row_share, |m| m.rows_per_s);
    out.mean_step_ms = weighted(|m| m.mean_step_ms, |m| m.steps_per_s);
    out.last_step_ms = parts.iter().map(|m| m.last_step_ms).fold(0.0, f64::max);
    out.mean_queue_ms = parts.iter().map(|m| m.mean_queue_ms).sum::<f64>() / parts.len() as f64;
    out.requests = RequestStats {
        in_flight: sum_u(|m| m.requests.in_flight),
        total: sum_u(|m| m.requests.total),
        errors: sum_u(|m| m.requests.errors),
        rate_per_s: sum_f(|m| m.requests.rate_per_s),
    };
}

/// Build the snapshot the stats API serves.
pub fn build(state: &BalancerState) -> StatsSnapshot {
    let now = Instant::now();
    let uptime = now.duration_since(state.started);
    let window = state.latency.window();
    let span_s = uptime.min(window).as_secs_f64().max(1.0);
    let max_age = state.snapshot_max_age();

    let mut inputs = Vec::new();
    let mut upstreams = Vec::new();
    let mut signals = Vec::new();
    let mut own_diagnoses = Vec::new();
    for u in &state.upstreams {
        let healthy = state.is_healthy(u, now);
        let snapshot = u.snapshot(max_age);
        let latest = u.latest_snapshot();
        let summary = u.latency.summary(None, now);
        let in_flight = u.in_flight.load(Ordering::Relaxed);
        let (mut waiting, mut running, mut max_sequences) = (0, 0, 0);
        if let Some(s) = &snapshot {
            for m in &s.models {
                waiting += m.waiting;
                running += m.running;
                max_sequences += m.max_sequences;
            }
            // A down upstream's old diagnoses describe a moment that is over:
            // only `upstream_down` speaks for it.
            if healthy {
                own_diagnoses.extend(relabel(&u.id, &s.diagnosis));
            }
        }
        signals.push(UpstreamSignal {
            id: u.id.to_string(),
            healthy,
            error: u.error(),
            down_for_s: u.down_for(now).map(|d| d.as_secs_f64()),
            score: load_score(in_flight, snapshot.as_deref(), ""),
            has_stats: snapshot.is_some(),
            waiting,
            running,
            max_sequences,
            ttft_p95_s: summary.latency.ttft.p95,
            ttft_count: summary.latency.ttft.count,
        });
        upstreams.push(UpstreamStats {
            id: u.id.to_string(),
            url: u.url.clone(),
            healthy,
            in_flight,
            latency: summary.latency,
            error: u.error(),
            routed_total: u.routed.load(Ordering::Relaxed),
            errors_total: u.errors.load(Ordering::Relaxed),
            snapshot: latest.as_ref().map(|(_, s)| Box::new((**s).clone())),
            snapshot_age_s: latest.as_ref().map(|(age, _)| age.as_secs_f64()),
        });
        inputs.push(UpstreamInput {
            id: u.id.to_string(),
            healthy,
            models: u.model_info(),
            snapshot,
        });
    }

    let mut models = union_models(&inputs);
    for m in &mut models {
        let w = state.latency.summary(Some(&m.name), now);
        m.latency = w.latency;
        m.requests.rate_per_s = ratio(w.finished as f64, span_s);
    }

    let mut diagnosis = diagnose(
        &signals,
        state.router.policy() == Policy::PrefixAffinity,
        &Thresholds::default(),
    );
    diagnosis.extend(own_diagnoses);
    diagnosis.sort_by_key(|d| std::cmp::Reverse(d.severity));

    let overall = state.latency.summary(None, now);
    let requests = RequestStats {
        in_flight: upstreams.iter().map(|u| u.in_flight).sum(),
        total: state
            .upstreams
            .iter()
            .map(|u| u.requests.total())
            .sum::<u64>()
            + state.metrics.rejected.total(),
        errors: state
            .upstreams
            .iter()
            .map(|u| u.requests.errors())
            .sum::<u64>()
            + state.metrics.rejected.errors(),
        rate_per_s: ratio(overall.finished as f64, span_s),
    };
    StatsSnapshot {
        node: NodeInfo {
            id: state.node_id.clone(),
            role: NodeRole::Balancer,
            version: env!("CARGO_PKG_VERSION").to_string(),
            listen: state.listen.to_string(),
            started_at: crate::server::collector::rfc3339(state.started_unix_ms),
            started_at_ms: state.started_unix_ms,
            uptime_s: uptime.as_secs_f64(),
        },
        models,
        latency: overall.latency,
        requests,
        system: state.system.snapshot(),
        diagnosis,
        upstreams,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, running: u64, waiting: u64, steps: f64, step_ms: f64) -> ModelStats {
        ModelStats {
            name: name.into(),
            state: ModelState::Ready,
            context_length: 4096,
            max_sequences: 4,
            running,
            waiting,
            steps_per_s: steps,
            rows_per_s: steps * 2.0,
            mean_step_ms: step_ms,
            kv: KvStats {
                total_blocks: 100,
                used_blocks: running * 10,
                ..KvStats::default()
            },
            tokens_per_s: TokensPerSecond {
                prompt: 1.0,
                generation: 10.0,
            },
            device: "cpu".into(),
            ..ModelStats::default()
        }
    }

    fn input(id: &str, healthy: bool, models: Vec<ModelStats>) -> UpstreamInput {
        UpstreamInput {
            id: id.into(),
            healthy,
            models: models
                .iter()
                .map(|m| UpstreamModel {
                    id: m.name.clone(),
                    max_model_len: m.context_length,
                    created: 0,
                })
                .collect(),
            snapshot: Some(Arc::new(StatsSnapshot {
                models,
                ..StatsSnapshot::default()
            })),
        }
    }

    #[test]
    fn the_union_sums_counts_and_rates_and_weights_the_means() {
        let inputs = [
            input("a", true, vec![model("m", 2, 1, 10.0, 20.0)]),
            input("b", true, vec![model("m", 4, 3, 30.0, 40.0)]),
        ];
        let union = union_models(&inputs);
        assert_eq!(union.len(), 1);
        let m = &union[0];
        assert_eq!((m.running, m.waiting, m.max_sequences), (6, 4, 8));
        assert_eq!(m.tokens_per_s.generation, 20.0);
        assert_eq!(m.steps_per_s, 40.0);
        assert_eq!(m.mean_rows_per_step, 2.0);
        // (20*10 + 40*30) / 40 = 35
        assert_eq!(m.mean_step_ms, 35.0);
        assert_eq!(m.kv.total_blocks, 200);
        assert_eq!(m.kv.usage_ratio, 60.0 / 200.0);
        assert_eq!((m.state, m.device.as_str()), (ModelState::Ready, "cpu"));
    }

    #[test]
    fn an_unhealthy_upstreams_snapshot_is_left_out_and_its_models_stay_listed() {
        let inputs = [
            input("a", true, vec![model("m", 2, 0, 10.0, 20.0)]),
            input(
                "b",
                false,
                vec![
                    model("m", 4, 0, 10.0, 20.0),
                    model("only-b", 1, 0, 1.0, 1.0),
                ],
            ),
        ];
        let union = union_models(&inputs);
        assert_eq!(
            union.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
            ["m", "only-b"]
        );
        assert_eq!(
            union[0].running, 2,
            "b is down: its numbers are not counted"
        );
        assert_eq!(union[0].state, ModelState::Ready);
        assert_eq!(union[1].state, ModelState::Failed);
        assert!(union[1].error.is_some());
    }

    #[test]
    fn a_model_without_a_snapshot_still_lists_with_its_context_length() {
        let mut i = input("a", true, vec![model("m", 0, 0, 0.0, 0.0)]);
        i.snapshot = None;
        let union = union_models(&[i]);
        assert_eq!(
            (union[0].context_length, union[0].state),
            (4096, ModelState::Ready)
        );
        assert_eq!(union[0].running, 0);
    }
}
