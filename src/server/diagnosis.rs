//! Latency diagnosis: why requests are slow right now.
//!
//! [`diagnose`] reads one moment's figures (per-model engine rates and window
//! percentiles, the machine's snapshot) and returns the [`Diagnosis`] entries
//! that apply, most severe first. Each rule is one function of the table
//! `RULES`: it reads the figures, compares them with a named threshold of
//! [`Thresholds`] and, when it fires, says so with the measurements as
//! evidence. Rules are independent and run in table order, which is also the
//! tie-break between entries of equal severity (queueing causes first, the
//! machine's resources last). A quiet server (nothing running, waiting or
//! finished in the window) produces no resource diagnosis.

use std::collections::BTreeMap;

use super::latency::WindowSummary;
use super::stats::{Cause, Diagnosis, ModelState, ModelStats, Severity, SystemSnapshot};

const MIB: f64 = 1024.0 * 1024.0;

/// The limits the rules compare against.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    /// K/V pool usage that warns / is critical.
    pub kv_usage: (f64, f64),
    /// Preemptions per minute that warn / are critical.
    pub preemptions_per_min: (f64, f64),
    /// Waiting requests per sequence slot that make queueing critical.
    pub queue_critical_per_slot: f64,
    /// Mean prompt tokens from which a cold prefix cache is a prefill cost.
    pub prefill_min_prompt_tokens: f64,
    /// Prefix hit ratio below which the cache is cold.
    pub prefill_max_hit_ratio: f64,
    /// Median time to first token (seconds) from which prefill is a problem.
    pub prefill_min_ttft_p50_s: f64,
    /// Mean rows per step and step time (ms) from which batching slows tokens.
    pub large_batch: (f64, f64),
    /// Share of prompt rows in the steps, with decoding rows beside them and
    /// at least this step time (ms), that makes prefill interfere.
    pub interference: (f64, f64),
    /// Recent over baseline step time, and the minimum growth in ms.
    pub step_growth: (f64, f64),
    /// CPU usage percent that warns / is critical.
    pub cpu_pct: (f64, f64),
    /// Available RAM as a share of the total below which memory is short /
    /// critical.
    pub mem_available_ratio: (f64, f64),
    /// Swap growth in bytes over the sampling window that warns; used only
    /// where the OS reports no paging rate.
    pub swap_growth_bytes: f64,
    /// Paging-in rate (bytes/s) that warns, when the OS reports it.
    pub swap_in_bytes_per_s: f64,
    /// VRAM usage that warns / is critical.
    pub vram_usage: (f64, f64),
    /// GPU utilization percent from which the GPU is the bottleneck.
    pub gpu_util_pct: f64,
    /// Process disk read rate (bytes/s) that warns.
    pub disk_read_bytes_per_s: f64,
    /// Expert read rate (bytes/s) of a tiered model that warns.
    pub expert_read_bytes_per_s: f64,
    /// Share of end-to-end time spent waiting for clients that warns, and
    /// the requests needed to say so.
    pub stall: (f64, u64),
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            kv_usage: (0.90, 0.97),
            preemptions_per_min: (1.0, 10.0),
            queue_critical_per_slot: 2.0,
            prefill_min_prompt_tokens: 512.0,
            prefill_max_hit_ratio: 0.25,
            prefill_min_ttft_p50_s: 1.0,
            large_batch: (8.0, 100.0),
            interference: (0.5, 50.0),
            step_growth: (1.5, 20.0),
            cpu_pct: (90.0, 97.0),
            mem_available_ratio: (0.10, 0.03),
            swap_growth_bytes: 64.0 * MIB,
            swap_in_bytes_per_s: 32.0 * MIB,
            vram_usage: (0.92, 0.98),
            gpu_util_pct: 95.0,
            disk_read_bytes_per_s: 50.0 * MIB,
            expert_read_bytes_per_s: 20.0 * MIB,
            stall: (0.2, 3),
        }
    }
}

/// What the rules know about one model.
#[derive(Debug, Clone, Copy)]
pub struct ModelSignals<'a> {
    /// The model's current stats (engine rates, latency percentiles).
    pub stats: &'a ModelStats,
    /// Aggregates of its requests in the latency window.
    pub window: &'a WindowSummary,
    /// Mean step time in ms of the older part of the sample ring and of its
    /// newest part, when both have steps.
    pub step_trend: Option<(f64, f64)>,
}

/// The machine, with the history one snapshot lacks.
#[derive(Debug, Clone, Copy)]
pub struct SystemSignals<'a> {
    /// The latest sample.
    pub snapshot: &'a SystemSnapshot,
    /// How much swap use grew over the sampling window (negative: shrank).
    pub swap_growth_bytes: f64,
}

struct Ctx<'a> {
    models: &'a [ModelSignals<'a>],
    system: &'a SystemSignals<'a>,
    th: &'a Thresholds,
    /// Something is, or just was, being served.
    busy: bool,
}

type Rule = fn(&Ctx<'_>, &mut Vec<Diagnosis>);

/// The rules in precedence order.
const RULES: &[Rule] = &[
    model_state,
    queue_saturated,
    kv_pool_full,
    preemptions,
    slow_consumer,
    prefill_bound,
    prefill_interference,
    large_batch,
    step_time_growth,
    gpu_saturated,
    vram_near_full,
    cpu_saturated,
    memory_pressure,
    disk_reads,
];

/// The diagnoses that apply to these figures, most severe first.
pub fn diagnose(
    models: &[ModelSignals<'_>],
    system: &SystemSignals<'_>,
    th: &Thresholds,
) -> Vec<Diagnosis> {
    let busy = models.iter().any(|m| {
        m.stats.running + m.stats.waiting > 0
            || m.stats.requests.in_flight > 0
            || m.window.finished > 0
    });
    let ctx = Ctx {
        models,
        system,
        th,
        busy,
    };
    let mut out = Vec::new();
    for rule in RULES {
        rule(&ctx, &mut out);
    }
    // Stable: equal severities keep the table order.
    out.sort_by_key(|d| std::cmp::Reverse(d.severity));
    out
}

fn entry(
    severity: Severity,
    cause: Cause,
    model: Option<&str>,
    message: String,
    evidence: &[(&str, f64)],
) -> Diagnosis {
    Diagnosis {
        severity,
        cause,
        model: model.map(str::to_string),
        upstream: None,
        message,
        evidence: evidence
            .iter()
            .map(|&(k, v)| (k.to_string(), round(v)))
            .collect::<BTreeMap<_, _>>(),
    }
}

/// Three significant decimals are plenty for a human and keep the JSON short.
fn round(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn severity_of(value: f64, (warn, critical): (f64, f64)) -> Option<Severity> {
    if value >= critical {
        Some(Severity::Critical)
    } else if value >= warn {
        Some(Severity::Warn)
    } else {
        None
    }
}

fn ready(m: &ModelSignals<'_>) -> bool {
    m.stats.state == ModelState::Ready
}

fn model_state(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    for m in ctx.models {
        let s = m.stats;
        match s.state {
            ModelState::Loading => out.push(entry(
                Severity::Info,
                Cause::ModelLoading,
                Some(&s.name),
                match s.load_progress {
                    Some(p) => format!(
                        "model `{}` is still loading ({:.0}%); requests get 503 until it is ready",
                        s.name,
                        p * 100.0
                    ),
                    None => format!(
                        "model `{}` is still loading; requests get 503 until it is ready",
                        s.name
                    ),
                },
                &[("load_progress", s.load_progress.unwrap_or(0.0))],
            )),
            ModelState::Failed => out.push(entry(
                Severity::Critical,
                Cause::ModelFailed,
                Some(&s.name),
                format!(
                    "model `{}` failed to load: {}",
                    s.name,
                    s.error.as_deref().unwrap_or("unknown error")
                ),
                &[],
            )),
            ModelState::Ready => {}
        }
    }
}

fn queue_saturated(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let s = m.stats;
        if s.waiting == 0 || s.running < s.max_sequences || s.max_sequences == 0 {
            continue;
        }
        let critical = s.waiting as f64 >= ctx.th.queue_critical_per_slot * s.max_sequences as f64;
        out.push(entry(
            if critical { Severity::Critical } else { Severity::Warn },
            Cause::QueueSaturated,
            Some(&s.name),
            format!(
                "{} request(s) wait for a free slot: all {} sequence slots of `{}` are busy (queue wait p95 {:.0} ms); more `seqs` raises throughput at the cost of each request's token rate",
                s.waiting,
                s.max_sequences,
                s.name,
                s.latency.queue.p95 * 1000.0
            ),
            &[
                ("waiting", s.waiting as f64),
                ("running", s.running as f64),
                ("max_sequences", s.max_sequences as f64),
                ("queue_p95_ms", s.latency.queue.p95 * 1000.0),
                ("mean_queue_ms", s.mean_queue_ms),
            ],
        ));
    }
}

fn kv_pool_full(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let s = m.stats;
        let Some(severity) = severity_of(s.kv.usage_ratio, ctx.th.kv_usage) else {
            continue;
        };
        out.push(entry(
            severity,
            Cause::KvPoolFull,
            Some(&s.name),
            format!(
                "the K/V pool of `{}` is {:.0}% in use ({} of {} blocks, {} cached): requests wait for blocks and the youngest sequences are preempted; raise `kv=` or lower `context_length`",
                s.name,
                s.kv.usage_ratio * 100.0,
                s.kv.used_blocks,
                s.kv.total_blocks,
                s.kv.cached_blocks
            ),
            &[
                ("kv_usage_ratio", s.kv.usage_ratio),
                ("used_blocks", s.kv.used_blocks as f64),
                ("total_blocks", s.kv.total_blocks as f64),
                ("cached_blocks", s.kv.cached_blocks as f64),
                ("waiting", s.waiting as f64),
            ],
        ));
    }
}

fn preemptions(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let s = m.stats;
        let Some(severity) = severity_of(s.preemptions_per_min, ctx.th.preemptions_per_min) else {
            continue;
        };
        out.push(entry(
            severity,
            Cause::Preemptions,
            Some(&s.name),
            format!(
                "`{}` preempts {:.1} sequences per minute to free K/V blocks; each one re-prefills what the pool lost",
                s.name, s.preemptions_per_min
            ),
            &[
                ("preemptions_per_min", s.preemptions_per_min),
                ("kv_usage_ratio", s.kv.usage_ratio),
                ("running", s.running as f64),
            ],
        ));
    }
}

fn slow_consumer(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    let (share, min_requests) = ctx.th.stall;
    for m in ctx.models.iter().filter(|m| ready(m)) {
        if m.window.latency.e2e.count < min_requests || m.window.stall_share < share {
            continue;
        }
        out.push(entry(
            Severity::Warn,
            Cause::SlowConsumer,
            Some(&m.stats.name),
            format!(
                "clients of `{}` spend {:.0}% of the response time not reading: generation stalls on their backpressure (only their own requests slow down)",
                m.stats.name,
                m.window.stall_share * 100.0
            ),
            &[
                ("stall_share", m.window.stall_share),
                ("requests", m.window.latency.e2e.count as f64),
            ],
        ));
    }
}

fn prefill_bound(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    let th = ctx.th;
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let w = m.window;
        let hit = m.stats.prefix_hit_ratio.unwrap_or(0.0);
        if w.latency.ttft.count == 0
            || w.mean_prompt_tokens < th.prefill_min_prompt_tokens
            || hit >= th.prefill_max_hit_ratio
            || w.latency.ttft.p50 < th.prefill_min_ttft_p50_s
        {
            continue;
        }
        out.push(entry(
            Severity::Warn,
            Cause::PrefillBound,
            Some(&m.stats.name),
            format!(
                "`{}` is prefill bound: prompts average {:.0} tokens, only {:.0}% come from the prefix cache, and the median time to first token is {:.2} s ({:.0} prompt tokens/s computed)",
                m.stats.name,
                w.mean_prompt_tokens,
                hit * 100.0,
                w.latency.ttft.p50,
                m.stats.tokens_per_s.prompt
            ),
            &[
                ("mean_prompt_tokens", w.mean_prompt_tokens),
                ("prefix_hit_ratio", hit),
                ("ttft_p50_s", w.latency.ttft.p50),
                ("prompt_tokens_per_s", m.stats.tokens_per_s.prompt),
            ],
        ));
    }
}

fn prefill_interference(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    let (share, min_step_ms) = ctx.th.interference;
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let s = m.stats;
        if s.running < 2 || s.prefill_row_share < share || s.mean_step_ms < min_step_ms {
            continue;
        }
        out.push(entry(
            Severity::Info,
            Cause::PrefillInterference,
            Some(&s.name),
            format!(
                "{:.0}% of the rows `{}` computes are prompt chunks that share steps with decoding: steps take {:.0} ms, so decoding requests see that gap between tokens while prompts prefill",
                s.prefill_row_share * 100.0,
                s.name,
                s.mean_step_ms
            ),
            &[
                ("prefill_row_share", s.prefill_row_share),
                ("mean_step_ms", s.mean_step_ms),
                ("running", s.running as f64),
                ("itl_mean_ms", s.latency.itl.mean * 1000.0),
            ],
        ));
    }
}

fn large_batch(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    let (min_rows, min_step_ms) = ctx.th.large_batch;
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let s = m.stats;
        if s.mean_rows_per_step < min_rows || s.mean_step_ms < min_step_ms {
            continue;
        }
        out.push(entry(
            Severity::Info,
            Cause::LargeBatch,
            Some(&s.name),
            format!(
                "steps of `{}` carry {:.1} rows and take {:.0} ms: every running sequence waits a whole step per token, so inter-token latency tracks the batch (mean {:.0} ms)",
                s.name,
                s.mean_rows_per_step,
                s.mean_step_ms,
                s.latency.itl.mean * 1000.0
            ),
            &[
                ("mean_rows_per_step", s.mean_rows_per_step),
                ("mean_step_ms", s.mean_step_ms),
                ("running", s.running as f64),
                ("itl_mean_ms", s.latency.itl.mean * 1000.0),
            ],
        ));
    }
}

fn step_time_growth(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    let (ratio, min_delta_ms) = ctx.th.step_growth;
    for m in ctx.models.iter().filter(|m| ready(m)) {
        let Some((baseline, recent)) = m.step_trend else {
            continue;
        };
        if baseline <= 0.0 || recent / baseline < ratio || recent - baseline < min_delta_ms {
            continue;
        }
        out.push(entry(
            Severity::Warn,
            Cause::StepTimeGrowth,
            Some(&m.stats.name),
            format!(
                "forward steps of `{}` slowed from {:.0} ms to {:.0} ms ({:.1}x); check the CPU, memory and disk lines for the cause",
                m.stats.name,
                baseline,
                recent,
                recent / baseline
            ),
            &[
                ("baseline_step_ms", baseline),
                ("recent_step_ms", recent),
                ("ratio", recent / baseline),
            ],
        ));
    }
}

fn gpu_saturated(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    if !ctx.busy {
        return;
    }
    for g in &ctx.system.snapshot.gpus {
        let Some(util) = g.util_pct.map(f64::from) else {
            continue;
        };
        if util < ctx.th.gpu_util_pct {
            continue;
        }
        out.push(entry(
            Severity::Warn,
            Cause::GpuSaturated,
            None,
            format!(
                "GPU {} ({}) is {:.0}% utilized: compute bound, so more concurrent sequences add latency instead of throughput",
                g.index, g.name, util
            ),
            &[("gpu", g.index as f64), ("util_pct", util)],
        ));
    }
}

fn vram_near_full(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    for g in &ctx.system.snapshot.gpus {
        let (Some(total), Some(used)) = (g.vram_total_bytes, g.vram_used_bytes) else {
            continue;
        };
        if total == 0 {
            continue;
        }
        let ratio = used as f64 / total as f64;
        let Some(severity) = severity_of(ratio, ctx.th.vram_usage) else {
            continue;
        };
        out.push(entry(
            severity,
            Cause::VramNearFull,
            None,
            format!(
                "GPU {} ({}) memory is {:.0}% full ({:.0} of {:.0} MiB): a larger K/V pool, more layers on the GPU or another process may not fit",
                g.index,
                g.name,
                ratio * 100.0,
                used as f64 / MIB,
                total as f64 / MIB
            ),
            &[
                ("gpu", g.index as f64),
                ("vram_usage_ratio", ratio),
                ("vram_used_mib", used as f64 / MIB),
                ("vram_total_mib", total as f64 / MIB),
                ("process_vram_mib", g.process_vram_bytes.unwrap_or(0) as f64 / MIB),
            ],
        ));
    }
}

fn cpu_saturated(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    if !ctx.busy {
        return;
    }
    let sys = ctx.system.snapshot;
    let usage = f64::from(sys.cpu.usage_pct);
    let Some(severity) = severity_of(usage, ctx.th.cpu_pct) else {
        return;
    };
    let cores = sys.cpu.cores.max(1) as f64;
    let process = f64::from(sys.process.cpu_pct);
    let others = (usage * cores - process).max(0.0);
    let who = if others > process {
        format!(
            "other processes use more of it than cuca does ({:.0}% of a core against {:.0}%)",
            others, process
        )
    } else {
        format!("cuca itself uses {process:.0}% of a core across {cores:.0} cores")
    };
    out.push(entry(
        severity,
        Cause::CpuSaturated,
        None,
        format!("the CPU is {usage:.0}% busy: {who}"),
        &[
            ("cpu_usage_pct", usage),
            ("process_cpu_pct", process),
            ("other_cpu_pct", others),
            ("cores", cores),
        ],
    ));
}

fn memory_pressure(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    if !ctx.busy {
        return;
    }
    let mem = &ctx.system.snapshot.memory;
    if mem.total_bytes == 0 {
        return;
    }
    let available = mem.available_bytes as f64 / mem.total_bytes as f64;
    let growth = ctx.system.swap_growth_bytes;
    let short = severity_of(
        1.0 - available,
        (
            1.0 - ctx.th.mem_available_ratio.0,
            1.0 - ctx.th.mem_available_ratio.1,
        ),
    );
    // The paging-in rate is the direct measure; where the OS reports none,
    // swap growth stands in for it.
    // Windows counts page-ins of memory-mapped files too, so a cold model
    // read shows a high rate with no swap in use: the rate counts only
    // while swap holds pages.
    let swapping = match mem.swap_in_bytes_per_s {
        Some(rate) => rate >= ctx.th.swap_in_bytes_per_s && mem.swap_used_bytes > 0,
        None => growth >= ctx.th.swap_growth_bytes,
    };
    let severity = match (short, swapping) {
        (Some(Severity::Critical), _) => Severity::Critical,
        (Some(_), _) | (None, true) => Severity::Warn,
        (None, false) => return,
    };
    let paging = match mem.swap_in_bytes_per_s {
        Some(rate) => format!("pages are read back from swap at {:.0} MiB/s", rate / MIB),
        None => format!("swap grew by {:.0} MiB over the window", growth / MIB),
    };
    let mut evidence = vec![
        ("available_ratio", available),
        ("available_mib", mem.available_bytes as f64 / MIB),
        ("swap_used_mib", mem.swap_used_bytes as f64 / MIB),
        ("swap_growth_mib", growth / MIB),
    ];
    if let Some(rate) = mem.swap_in_bytes_per_s {
        evidence.push(("swap_in_mib_per_s", rate / MIB));
    }
    out.push(entry(
        severity,
        Cause::MemoryPressure,
        None,
        format!(
            "RAM is {:.0}% available ({:.0} MiB) and {paging} ({:.0} MiB of swap in use): weights or K/V pages may be evicted to disk",
            available * 100.0,
            mem.available_bytes as f64 / MIB,
            mem.swap_used_bytes as f64 / MIB
        ),
        &evidence,
    ));
}

fn disk_reads(ctx: &Ctx<'_>, out: &mut Vec<Diagnosis>) {
    if !ctx.busy {
        return;
    }
    let process_read = ctx.system.snapshot.process.read_bytes_per_s;
    let tiered: Vec<(&ModelSignals<'_>, f64)> = ctx
        .models
        .iter()
        .filter_map(|m| {
            m.stats
                .expert_tiers
                .as_ref()
                .map(|t| (m, t.read_bytes_per_s))
        })
        .filter(|&(_, bps)| bps >= ctx.th.expert_read_bytes_per_s)
        .collect();
    for (m, bps) in &tiered {
        let rates = m.stats.expert_tiers.clone().unwrap_or_default();
        out.push(entry(
            Severity::Warn,
            Cause::DiskReads,
            Some(&m.stats.name),
            format!(
                "`{}` reads experts from disk at {:.0} MiB/s ({:.0} reads/s, cache hit {:.0}%): the tiered expert caches are too small for the traffic",
                m.stats.name,
                bps / MIB,
                rates.disk_reads_per_s,
                rates.hit_ratio.unwrap_or(0.0) * 100.0
            ),
            &[
                ("expert_read_mib_per_s", bps / MIB),
                ("expert_reads_per_s", rates.disk_reads_per_s),
                ("expert_hit_ratio", rates.hit_ratio.unwrap_or(0.0)),
                ("process_read_mib_per_s", process_read / MIB),
            ],
        ));
    }
    if tiered.is_empty() && process_read >= ctx.th.disk_read_bytes_per_s {
        out.push(entry(
            Severity::Warn,
            Cause::DiskReads,
            None,
            format!(
                "the process reads {:.0} MiB/s from disk while serving: per-layer embeddings read per token, or weights paged back in after swapping",
                process_read / MIB
            ),
            &[("process_read_mib_per_s", process_read / MIB)],
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::stats::{
        ExpertTierRates, GpuStats, KvStats, LatencyStats, Quantiles, RequestStats,
    };

    /// A busy, healthy model: nothing fires for it.
    fn model() -> ModelStats {
        ModelStats {
            name: "m".into(),
            state: ModelState::Ready,
            max_sequences: 4,
            running: 2,
            kv: KvStats {
                total_blocks: 1000,
                used_blocks: 100,
                usage_ratio: 0.1,
                ..KvStats::default()
            },
            mean_rows_per_step: 2.0,
            mean_step_ms: 30.0,
            requests: RequestStats {
                in_flight: 2,
                ..RequestStats::default()
            },
            ..ModelStats::default()
        }
    }

    fn run(model: &ModelStats, window: &WindowSummary, system: &SystemSnapshot) -> Vec<Diagnosis> {
        run_with(model, window, system, None, 0.0)
    }

    fn run_with(
        model: &ModelStats,
        window: &WindowSummary,
        system: &SystemSnapshot,
        step_trend: Option<(f64, f64)>,
        swap_growth_bytes: f64,
    ) -> Vec<Diagnosis> {
        diagnose(
            &[ModelSignals {
                stats: model,
                window,
                step_trend,
            }],
            &SystemSignals {
                snapshot: system,
                swap_growth_bytes,
            },
            &Thresholds::default(),
        )
    }

    fn causes(d: &[Diagnosis]) -> Vec<Cause> {
        d.iter().map(|d| d.cause).collect()
    }

    #[test]
    fn a_healthy_busy_server_has_no_diagnosis() {
        let d = run(
            &model(),
            &WindowSummary::default(),
            &SystemSnapshot::default(),
        );
        assert!(d.is_empty(), "{d:?}");
    }

    #[test]
    fn a_full_slot_set_with_a_queue_is_queue_saturation_with_the_numbers() {
        let mut m = model();
        m.running = 4;
        m.waiting = 3;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::QueueSaturated]);
        assert_eq!(d[0].severity, Severity::Warn);
        assert_eq!(d[0].evidence["waiting"], 3.0);
        assert_eq!(d[0].evidence["max_sequences"], 4.0);
        // Twice the slots waiting is critical.
        m.waiting = 8;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(d[0].severity, Severity::Critical);
    }

    #[test]
    fn waiting_with_a_free_slot_is_not_slot_saturation() {
        let mut m = model();
        m.running = 2;
        m.waiting = 1;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert!(!causes(&d).contains(&Cause::QueueSaturated));
    }

    #[test]
    fn kv_usage_warns_at_ninety_percent_and_is_critical_at_ninety_seven() {
        let mut m = model();
        m.kv.usage_ratio = 0.91;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(
            (d[0].cause, d[0].severity),
            (Cause::KvPoolFull, Severity::Warn)
        );
        m.kv.usage_ratio = 0.98;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(d[0].severity, Severity::Critical);
    }

    #[test]
    fn critical_causes_sort_before_warnings_whatever_the_rule_order() {
        let mut m = model();
        m.running = 4;
        m.waiting = 3; // queue: Warn, first in the table
        m.kv.usage_ratio = 0.99; // kv: Critical, second in the table
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::KvPoolFull, Cause::QueueSaturated]);
    }

    #[test]
    fn preemptions_scale_in_severity() {
        let mut m = model();
        m.preemptions_per_min = 3.0;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(
            (d[0].cause, d[0].severity),
            (Cause::Preemptions, Severity::Warn)
        );
        m.preemptions_per_min = 12.0;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(d[0].severity, Severity::Critical);
    }

    fn window(prompt: f64, ttft_p50: f64) -> WindowSummary {
        WindowSummary {
            latency: LatencyStats {
                ttft: Quantiles {
                    p50: ttft_p50,
                    count: 5,
                    ..Quantiles::default()
                },
                e2e: Quantiles {
                    count: 5,
                    ..Quantiles::default()
                },
                ..LatencyStats::default()
            },
            finished: 5,
            mean_prompt_tokens: prompt,
            ..WindowSummary::default()
        }
    }

    #[test]
    fn long_cold_prompts_with_a_slow_first_token_are_prefill_bound() {
        let mut m = model();
        m.prefix_hit_ratio = Some(0.05);
        let d = run(&m, &window(2000.0, 3.0), &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::PrefillBound]);
        assert_eq!(d[0].evidence["mean_prompt_tokens"], 2000.0);
    }

    #[test]
    fn a_warm_prefix_cache_or_short_prompts_are_not_prefill_bound() {
        let mut m = model();
        m.prefix_hit_ratio = Some(0.8);
        assert!(run(&m, &window(2000.0, 3.0), &SystemSnapshot::default()).is_empty());
        m.prefix_hit_ratio = Some(0.0);
        assert!(run(&m, &window(100.0, 3.0), &SystemSnapshot::default()).is_empty());
        assert!(run(&m, &window(2000.0, 0.2), &SystemSnapshot::default()).is_empty());
    }

    #[test]
    fn prompt_chunks_beside_decoding_rows_are_interference() {
        let mut m = model();
        m.prefill_row_share = 0.7;
        m.mean_step_ms = 120.0;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::PrefillInterference]);
        // A lone request has nobody to interfere with.
        m.running = 1;
        assert!(run(&m, &WindowSummary::default(), &SystemSnapshot::default()).is_empty());
    }

    #[test]
    fn many_rows_in_slow_steps_are_a_large_batch() {
        let mut m = model();
        m.mean_rows_per_step = 16.0;
        m.mean_step_ms = 250.0;
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::LargeBatch]);
        assert_eq!(d[0].evidence["mean_rows_per_step"], 16.0);
    }

    #[test]
    fn step_time_growth_needs_both_the_ratio_and_the_absolute_gap() {
        let m = model();
        let sys = SystemSnapshot::default();
        let w = WindowSummary::default();
        let d = run_with(&m, &w, &sys, Some((40.0, 90.0)), 0.0);
        assert_eq!(causes(&d), [Cause::StepTimeGrowth]);
        assert_eq!(d[0].evidence["ratio"], 2.25);
        // 2x but only 10 ms: noise.
        assert!(run_with(&m, &w, &sys, Some((10.0, 20.0)), 0.0).is_empty());
        // 40 ms more but only 1.2x.
        assert!(run_with(&m, &w, &sys, Some((200.0, 240.0)), 0.0).is_empty());
    }

    #[test]
    fn a_saturated_cpu_names_who_uses_it() {
        let mut sys = SystemSnapshot::default();
        sys.cpu.usage_pct = 99.0;
        sys.cpu.cores = 8;
        sys.process.cpu_pct = 150.0;
        let d = run(&model(), &WindowSummary::default(), &sys);
        assert_eq!(
            (d[0].cause, d[0].severity),
            (Cause::CpuSaturated, Severity::Critical)
        );
        assert!(d[0].message.contains("other processes"), "{}", d[0].message);
        sys.process.cpu_pct = 790.0;
        let d = run(&model(), &WindowSummary::default(), &sys);
        assert!(d[0].message.contains("cuca itself"), "{}", d[0].message);
    }

    #[test]
    fn an_idle_server_reports_no_resource_diagnosis() {
        let mut sys = SystemSnapshot::default();
        sys.cpu.usage_pct = 99.0;
        let mut m = model();
        m.running = 0;
        m.requests.in_flight = 0;
        assert!(run(&m, &WindowSummary::default(), &sys).is_empty());
    }

    #[test]
    fn low_available_ram_or_growing_swap_is_memory_pressure() {
        let mut sys = SystemSnapshot::default();
        sys.memory.total_bytes = 16 << 30;
        sys.memory.available_bytes = 4 << 30;
        let w = WindowSummary::default();
        assert!(run(&model(), &w, &sys).is_empty());
        // 5% available: warn.
        sys.memory.available_bytes = (16u64 << 30) / 20;
        let d = run(&model(), &w, &sys);
        assert_eq!(
            (d[0].cause, d[0].severity),
            (Cause::MemoryPressure, Severity::Warn)
        );
        // 1% available: critical.
        sys.memory.available_bytes = (16u64 << 30) / 100;
        assert_eq!(run(&model(), &w, &sys)[0].severity, Severity::Critical);
        // Plenty of RAM but swap growing 100 MiB, on an OS that reports no
        // paging rate: growth decides.
        sys.memory.available_bytes = 8 << 30;
        let d = run_with(&model(), &w, &sys, None, 100.0 * MIB);
        assert_eq!(causes(&d), [Cause::MemoryPressure]);
        assert!(!d[0].evidence.contains_key("swap_in_mib_per_s"));
        // A reported paging rate wins over growth: low rate with growth is quiet...
        sys.memory.swap_in_bytes_per_s = Some(1.0 * MIB);
        assert!(run_with(&model(), &w, &sys, None, 100.0 * MIB).is_empty());
        // ...a high rate with no swap in use is a mapped-file read: quiet...
        sys.memory.swap_in_bytes_per_s = Some(200.0 * MIB);
        assert!(run_with(&model(), &w, &sys, None, 0.0).is_empty());
        // ...and with swap in use it fires without any growth, with the rate as evidence.
        sys.memory.swap_used_bytes = 1 << 30;
        let d = run_with(&model(), &w, &sys, None, 0.0);
        assert_eq!(causes(&d), [Cause::MemoryPressure]);
        assert_eq!(d[0].evidence["swap_in_mib_per_s"], 200.0);
    }

    fn gpu(used: u64, total: u64, util: f32) -> GpuStats {
        GpuStats {
            index: 0,
            name: "G".into(),
            vram_total_bytes: Some(total),
            vram_used_bytes: Some(used),
            util_pct: Some(util),
            ..GpuStats::default()
        }
    }

    #[test]
    fn a_full_busy_gpu_reports_vram_and_utilization_separately() {
        let mut sys = SystemSnapshot::default();
        sys.gpus.push(gpu(95 << 20, 100 << 20, 100.0));
        let d = run(&model(), &WindowSummary::default(), &sys);
        assert_eq!(causes(&d), [Cause::GpuSaturated, Cause::VramNearFull]);
        assert_eq!(d[1].evidence["vram_usage_ratio"], 0.95);
        // Half full and half busy: nothing.
        sys.gpus[0] = gpu(50 << 20, 100 << 20, 50.0);
        assert!(run(&model(), &WindowSummary::default(), &sys).is_empty());
    }

    #[test]
    fn tiered_expert_reads_name_the_model_and_the_process_rate_covers_the_rest() {
        let mut m = model();
        m.expert_tiers = Some(ExpertTierRates {
            disk_reads_per_s: 40.0,
            read_bytes_per_s: 200.0 * MIB,
            hit_ratio: Some(0.4),
            ..ExpertTierRates::default()
        });
        let d = run(&m, &WindowSummary::default(), &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::DiskReads]);
        assert_eq!(d[0].model.as_deref(), Some("m"));
        assert_eq!(d[0].evidence["expert_read_mib_per_s"], 200.0);
        // No tiers: the process-wide rate speaks.
        let mut sys = SystemSnapshot::default();
        sys.process.read_bytes_per_s = 80.0 * MIB;
        let d = run(&model(), &WindowSummary::default(), &sys);
        assert_eq!((d[0].cause, d[0].model.clone()), (Cause::DiskReads, None));
    }

    #[test]
    fn slow_clients_need_a_few_requests_before_they_are_blamed() {
        let mut w = window(10.0, 0.1);
        w.stall_share = 0.5;
        let d = run(&model(), &w, &SystemSnapshot::default());
        assert_eq!(causes(&d), [Cause::SlowConsumer]);
        w.latency.e2e.count = 2;
        assert!(run(&model(), &w, &SystemSnapshot::default()).is_empty());
    }

    #[test]
    fn a_loading_or_failed_model_is_reported_first() {
        let mut loading = model();
        loading.state = ModelState::Loading;
        loading.load_progress = Some(0.4);
        let d = run(
            &loading,
            &WindowSummary::default(),
            &SystemSnapshot::default(),
        );
        assert_eq!(
            (d[0].cause, d[0].severity),
            (Cause::ModelLoading, Severity::Info)
        );
        assert!(d[0].message.contains("40%"), "{}", d[0].message);
        let mut failed = model();
        failed.state = ModelState::Failed;
        failed.error = Some("no such file".into());
        let d = run(
            &failed,
            &WindowSummary::default(),
            &SystemSnapshot::default(),
        );
        assert_eq!(
            (d[0].cause, d[0].severity),
            (Cause::ModelFailed, Severity::Critical)
        );
    }
}
