//! Prometheus metrics: lock-free counters and fixed-bucket histograms, and the
//! text exposition (`GET /metrics`), written by hand (no metrics crate).
//!
//! Bounds: every metric is a fixed set of atomics created at startup. The
//! only labels are the model name (at most [`MAX_MODELS`](super::MAX_MODELS)
//! values), the status code (a fixed list), a GPU index and the bucket bound,
//! so the series count does not grow with traffic.

use std::fmt::{Display, Write as _};
use std::sync::atomic::{AtomicU64, Ordering};

use super::stats::{ModelState, SystemSnapshot};
use crate::ServingStats;

/// Buckets of time-to-first-token histograms, seconds (vLLM's).
pub const TTFT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
/// Buckets of inter-token-latency histograms, seconds (vLLM's).
pub const ITL_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
/// Buckets of end-to-end histograms, seconds (vLLM's).
pub const E2E_BUCKETS: &[f64] = &[
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];
/// Buckets of queue-wait histograms, seconds.
pub const QUEUE_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// A cumulative histogram over fixed bucket bounds (seconds).
#[derive(Debug)]
pub struct Histogram {
    bounds: &'static [f64],
    /// One counter per bound plus one for `+Inf`, each counting the
    /// observations that fell in that bucket only (the exposition sums them).
    buckets: Box<[AtomicU64]>,
    sum_ns: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// An empty histogram over ascending `bounds`.
    pub fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            buckets: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record one observation, in seconds; a negative or NaN value counts as
    /// zero.
    pub fn observe(&self, seconds: f64) {
        let seconds = if seconds.is_nan() {
            0.0
        } else {
            seconds.max(0.0)
        };
        // A bucket's bound is inclusive (`le`).
        let slot = self.bounds.partition_point(|&b| b < seconds);
        self.buckets[slot].fetch_add(1, Ordering::Relaxed);
        self.sum_ns
            .fetch_add((seconds * 1e9) as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Observations so far.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Append the `_bucket`, `_sum` and `_count` lines of series `name` with
    /// `labels` (already formatted, without braces, possibly empty).
    pub(crate) fn render_into(&self, out: &mut String, name: &str, labels: &str) {
        let sep = if labels.is_empty() { "" } else { "," };
        let mut cumulative = 0u64;
        for (bound, bucket) in self.bounds.iter().zip(self.buckets.iter()) {
            cumulative += bucket.load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "{name}_bucket{{{labels}{sep}le=\"{bound}\"}} {cumulative}"
            );
        }
        cumulative += self.buckets[self.bounds.len()].load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "{name}_bucket{{{labels}{sep}le=\"+Inf\"}} {cumulative}"
        );
        let sum = self.sum_ns.load(Ordering::Relaxed) as f64 / 1e9;
        let braces = if labels.is_empty() {
            String::new()
        } else {
            format!("{{{labels}}}")
        };
        let _ = writeln!(out, "{name}_sum{braces} {sum}");
        let _ = writeln!(out, "{name}_count{braces} {cumulative}");
    }
}

/// The status codes the counters tell apart; any other code counts in its
/// class's first entry (`4xx` as 400, `5xx` as 500). 499 is a client that
/// went away (nginx's code), not an error of the server.
pub const STATUS_CODES: [u16; 9] = [200, 400, 401, 404, 413, 499, 500, 503, 502];

/// Request counters by status code.
#[derive(Debug, Default)]
pub struct StatusCounters([AtomicU64; STATUS_CODES.len()]);

impl StatusCounters {
    fn slot(code: u16) -> usize {
        let position = |wanted: u16| STATUS_CODES.iter().position(|&c| c == wanted).unwrap_or(0);
        STATUS_CODES
            .iter()
            .position(|&c| c == code)
            .unwrap_or_else(|| match code {
                400..=498 => position(400),
                500..=599 => position(500),
                _ => position(200),
            })
    }

    /// Count one request that ended with `code`.
    pub fn inc(&self, code: u16) {
        self.0[Self::slot(code)].fetch_add(1, Ordering::Relaxed);
    }

    /// Requests counted with `code`'s slot.
    pub fn get(&self, code: u16) -> u64 {
        self.0[Self::slot(code)].load(Ordering::Relaxed)
    }

    /// Every request counted.
    pub fn total(&self) -> u64 {
        self.0.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Requests that ended in a 4xx or 5xx other than 499.
    pub fn errors(&self) -> u64 {
        STATUS_CODES
            .iter()
            .zip(&self.0)
            .filter(|(code, _)| **code >= 400 && **code != 499)
            .map(|(_, c)| c.load(Ordering::Relaxed))
            .sum()
    }

    pub(crate) fn render_into(&self, out: &mut String, name: &str, labels: &str) {
        let sep = if labels.is_empty() { "" } else { "," };
        for (code, counter) in STATUS_CODES.iter().zip(&self.0) {
            let _ = writeln!(
                out,
                "{name}{{{labels}{sep}code=\"{code}\"}} {}",
                counter.load(Ordering::Relaxed)
            );
        }
    }
}

/// The metrics of one model.
#[derive(Debug)]
pub struct ModelMetrics {
    /// Requests by status code.
    pub requests: StatusCounters,
    /// Requests being handled now.
    pub in_flight: AtomicU64,
    /// Prompt tokens of finished requests.
    pub prompt_tokens: AtomicU64,
    /// Prompt tokens of finished requests served from the prefix cache.
    pub cached_prompt_tokens: AtomicU64,
    /// Completion tokens of finished requests.
    pub generation_tokens: AtomicU64,
    /// Nanoseconds producers waited for a client to take output.
    pub stall_ns: AtomicU64,
    /// Time to first token.
    pub ttft: Histogram,
    /// Mean inter-token latency of a request.
    pub itl: Histogram,
    /// End-to-end time.
    pub e2e: Histogram,
    /// Wait for admission to the engine.
    pub queue: Histogram,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            requests: StatusCounters::default(),
            in_flight: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            cached_prompt_tokens: AtomicU64::new(0),
            generation_tokens: AtomicU64::new(0),
            stall_ns: AtomicU64::new(0),
            ttft: Histogram::new(TTFT_BUCKETS),
            itl: Histogram::new(ITL_BUCKETS),
            e2e: Histogram::new(E2E_BUCKETS),
            queue: Histogram::new(QUEUE_BUCKETS),
        }
    }
}

/// The metrics not tied to a model.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Every HTTP response by status code, counted when its headers go out
    /// (a streamed response that fails later still counts as 200 here; the
    /// per-model counters hold the final outcome).
    pub http: StatusCounters,
    /// Chat requests refused before they reached a model.
    pub rejected: StatusCounters,
}

/// What [`render`] reads about one model.
pub struct ModelView<'a> {
    /// Model name, the `model` label.
    pub name: &'a str,
    /// Loading, ready or failed.
    pub state: ModelState,
    /// The model's counters and histograms.
    pub metrics: &'a ModelMetrics,
    /// The engine's counters (`None` until the model is ready).
    pub serving: Option<ServingStats>,
    /// Bytes of K/V pools and of weights (`0` until ready).
    pub kv_bytes: u64,
    /// Size of the GGUF the model runs from (`0` until ready).
    pub weights_bytes: u64,
}

/// A per-model sample read from the engine's counters; `None` until the
/// model is ready.
fn serving(f: fn(&ServingStats) -> u64) -> impl Fn(&ModelView<'_>) -> Option<u64> {
    move |m| m.serving.as_ref().map(f)
}

/// Escape a label value per the exposition format.
pub(crate) fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

/// Builds the exposition text one metric family at a time.
pub(crate) struct Exposition {
    pub(crate) out: String,
}

impl Exposition {
    pub(crate) fn new() -> Self {
        Self {
            out: String::with_capacity(8 * 1024),
        }
    }

    pub(crate) fn family(&mut self, name: &str, kind: &str, help: &str) {
        let _ = writeln!(self.out, "# HELP {name} {help}");
        let _ = writeln!(self.out, "# TYPE {name} {kind}");
    }

    pub(crate) fn sample(&mut self, name: &str, labels: &str, value: impl Display) {
        if labels.is_empty() {
            let _ = writeln!(self.out, "{name} {value}");
        } else {
            let _ = writeln!(self.out, "{name}{{{labels}}} {value}");
        }
    }

    /// One family with a sample per model.
    fn per_model<V: Display>(
        &mut self,
        models: &[ModelView<'_>],
        name: &str,
        kind: &str,
        help: &str,
        value: impl Fn(&ModelView<'_>) -> Option<V>,
    ) {
        self.family(name, kind, help);
        for m in models {
            if let Some(v) = value(m) {
                self.sample(name, &format!("model=\"{}\"", escape(m.name)), v);
            }
        }
    }

    fn per_model_histogram(
        &mut self,
        models: &[ModelView<'_>],
        name: &str,
        help: &str,
        pick: impl Fn(&ModelMetrics) -> &Histogram,
    ) {
        self.family(name, "histogram", help);
        for m in models {
            pick(m.metrics).render_into(
                &mut self.out,
                name,
                &format!("model=\"{}\"", escape(m.name)),
            );
        }
    }
}

/// Render every metric in the Prometheus text exposition format (version
/// 0.0.4): per-model request, latency and engine series, then the machine's.
pub fn render(
    models: &[ModelView<'_>],
    server: &ServerMetrics,
    system: &SystemSnapshot,
    uptime_s: f64,
) -> String {
    let mut x = Exposition {
        out: String::with_capacity(8 * 1024),
    };
    x.family(
        "cuca_uptime_seconds",
        "gauge",
        "Seconds since the server started.",
    );
    x.sample("cuca_uptime_seconds", "", uptime_s);

    x.family(
        "cuca_http_requests_total",
        "counter",
        "HTTP responses by status code, counted when the headers are sent.",
    );
    server
        .http
        .render_into(&mut x.out, "cuca_http_requests_total", "");
    x.family(
        "cuca_rejected_requests_total",
        "counter",
        "Chat requests refused before reaching a model (bad body, unknown model, model loading), by status code.",
    );
    server
        .rejected
        .render_into(&mut x.out, "cuca_rejected_requests_total", "");

    x.per_model(
        models,
        "cuca_model_ready",
        "gauge",
        "1 when the model serves requests.",
        |m| Some(u8::from(m.state == ModelState::Ready)),
    );
    x.family(
        "cuca_requests_total",
        "counter",
        "Requests that reached a model, by status code.",
    );
    for m in models {
        m.metrics.requests.render_into(
            &mut x.out,
            "cuca_requests_total",
            &format!("model=\"{}\"", escape(m.name)),
        );
    }
    x.per_model(
        models,
        "cuca_requests_in_flight",
        "gauge",
        "Requests being handled (queued, prefilling or generating).",
        |m| Some(m.metrics.in_flight.load(Ordering::Relaxed)),
    );
    x.per_model(
        models,
        "cuca_prompt_tokens_total",
        "counter",
        "Prompt tokens of finished requests.",
        |m| Some(m.metrics.prompt_tokens.load(Ordering::Relaxed)),
    );
    x.per_model(
        models,
        "cuca_prompt_tokens_cached_total",
        "counter",
        "Prompt tokens of finished requests served from the prefix cache.",
        |m| Some(m.metrics.cached_prompt_tokens.load(Ordering::Relaxed)),
    );
    x.per_model(
        models,
        "cuca_generation_tokens_total",
        "counter",
        "Completion tokens of finished requests.",
        |m| Some(m.metrics.generation_tokens.load(Ordering::Relaxed)),
    );
    x.per_model(
        models,
        "cuca_response_stall_seconds_total",
        "counter",
        "Seconds generation waited for clients to read their responses.",
        |m| Some(m.metrics.stall_ns.load(Ordering::Relaxed) as f64 / 1e9),
    );
    x.per_model_histogram(
        models,
        "cuca_request_ttft_seconds",
        "Time from arrival to the first token.",
        |m| &m.ttft,
    );
    x.per_model_histogram(
        models,
        "cuca_request_inter_token_seconds",
        "Mean time between the tokens of a request.",
        |m| &m.itl,
    );
    x.per_model_histogram(
        models,
        "cuca_request_e2e_seconds",
        "Time from arrival to the last token.",
        |m| &m.e2e,
    );
    x.per_model_histogram(
        models,
        "cuca_request_queue_seconds",
        "Time from arrival to admission to the engine.",
        |m| &m.queue,
    );

    x.per_model(
        models,
        "cuca_requests_running",
        "gauge",
        "Requests the engine is running.",
        serving(|s| s.running as u64),
    );
    x.per_model(
        models,
        "cuca_requests_waiting",
        "gauge",
        "Requests waiting for a slot or K/V blocks.",
        serving(|s| s.waiting as u64),
    );
    x.per_model(
        models,
        "cuca_max_sequences",
        "gauge",
        "Most requests the engine runs at once.",
        serving(|s| s.max_sequences as u64),
    );
    x.per_model(
        models,
        "cuca_kv_cache_usage_ratio",
        "gauge",
        "Share of the K/V block pool that running requests reference.",
        |m| {
            m.serving
                .as_ref()
                .filter(|s| s.total_blocks > 0)
                .map(|s| s.used_blocks as f64 / s.total_blocks as f64)
        },
    );
    x.per_model(
        models,
        "cuca_kv_cache_blocks_total",
        "gauge",
        "Blocks in the K/V pool.",
        serving(|s| s.total_blocks as u64),
    );
    x.per_model(
        models,
        "cuca_kv_cache_blocks_cached",
        "gauge",
        "Unreferenced blocks kept for prefix reuse.",
        serving(|s| s.cached_blocks as u64),
    );
    x.per_model(
        models,
        "cuca_kv_cache_bytes",
        "gauge",
        "Bytes the K/V pools occupy.",
        |m| Some(m.kv_bytes),
    );
    x.per_model(
        models,
        "cuca_model_weights_bytes",
        "gauge",
        "Size of the model file the engine runs from.",
        |m| Some(m.weights_bytes),
    );
    x.per_model(
        models,
        "cuca_prefix_cache_hit_tokens_total",
        "counter",
        "Prompt tokens served from the prefix cache.",
        serving(|s| s.prefix_hit_tokens),
    );
    x.per_model(
        models,
        "cuca_preemptions_total",
        "counter",
        "Sequences preempted to free K/V blocks.",
        serving(|s| s.preemptions),
    );
    x.per_model(
        models,
        "cuca_engine_steps_total",
        "counter",
        "Forward steps run.",
        serving(|s| s.steps),
    );
    x.family(
        "cuca_engine_rows_total",
        "counter",
        "Rows (tokens) the forward steps processed.",
    );
    for m in models {
        if let Some(s) = &m.serving {
            let name = escape(m.name);
            x.sample(
                "cuca_engine_rows_total",
                &format!("model=\"{name}\",kind=\"prefill\""),
                s.prefill_rows,
            );
            x.sample(
                "cuca_engine_rows_total",
                &format!("model=\"{name}\",kind=\"decode\""),
                s.batched_rows.saturating_sub(s.prefill_rows),
            );
        }
    }
    x.per_model(
        models,
        "cuca_engine_step_seconds_total",
        "counter",
        "Seconds spent in forward passes.",
        |m| m.serving.as_ref().map(|s| s.step_ns as f64 / 1e9),
    );
    x.per_model(
        models,
        "cuca_engine_last_step_seconds",
        "gauge",
        "Duration of the last forward pass.",
        |m| m.serving.as_ref().map(|s| s.last_step_ns as f64 / 1e9),
    );
    x.per_model(
        models,
        "cuca_engine_last_step_rows",
        "gauge",
        "Rows of the last forward pass.",
        serving(|s| s.last_step_rows as u64),
    );
    x.per_model(
        models,
        "cuca_engine_admissions_total",
        "counter",
        "Requests admitted to the engine.",
        serving(|s| s.admissions),
    );
    x.per_model(
        models,
        "cuca_engine_queue_wait_seconds_total",
        "counter",
        "Seconds admitted requests waited for admission.",
        |m| m.serving.as_ref().map(|s| s.queue_wait_ns as f64 / 1e9),
    );
    x.per_model(
        models,
        "cuca_engine_generated_tokens_total",
        "counter",
        "Tokens the engine generated.",
        serving(|s| s.generated_tokens),
    );

    render_system(&mut x, system);
    x.out
}

/// The machine's series from a sysmon snapshot.
pub(crate) fn render_system(x: &mut Exposition, s: &SystemSnapshot) {
    let gauge = |x: &mut Exposition, name: &str, help: &str, value: f64| {
        x.family(name, "gauge", help);
        x.sample(name, "", value);
    };
    gauge(
        x,
        "cuca_system_cpu_usage_percent",
        "Whole-machine CPU usage.",
        f64::from(s.cpu.usage_pct),
    );
    gauge(
        x,
        "cuca_system_cpu_cores",
        "Logical CPU cores.",
        s.cpu.cores as f64,
    );
    gauge(
        x,
        "cuca_system_memory_total_bytes",
        "Installed RAM.",
        s.memory.total_bytes as f64,
    );
    gauge(
        x,
        "cuca_system_memory_used_bytes",
        "RAM in use.",
        s.memory.used_bytes as f64,
    );
    gauge(
        x,
        "cuca_system_memory_available_bytes",
        "RAM available without swapping.",
        s.memory.available_bytes as f64,
    );
    gauge(
        x,
        "cuca_system_swap_total_bytes",
        "Swap size.",
        s.memory.swap_total_bytes as f64,
    );
    gauge(
        x,
        "cuca_system_swap_used_bytes",
        "Swap in use.",
        s.memory.swap_used_bytes as f64,
    );
    for (name, help, rate) in [
        (
            "cuca_system_swap_in_bytes_per_second",
            "Bytes per second paged in from swap (absent where the OS does not report it).",
            s.memory.swap_in_bytes_per_s,
        ),
        (
            "cuca_system_swap_out_bytes_per_second",
            "Bytes per second paged out to swap (absent where the OS does not report it).",
            s.memory.swap_out_bytes_per_s,
        ),
    ] {
        if let Some(rate) = rate {
            gauge(x, name, help, rate);
        }
    }
    gauge(
        x,
        "cuca_process_cpu_percent",
        "CPU time of this process as a percentage of one core.",
        f64::from(s.process.cpu_pct),
    );
    gauge(
        x,
        "cuca_process_resident_memory_bytes",
        "Resident memory of this process.",
        s.process.rss_bytes as f64,
    );
    gauge(
        x,
        "cuca_system_disk_read_bytes_per_second",
        "Disk read throughput, see the stats API's disk_io.",
        s.disk_io.read_bytes_per_s,
    );
    gauge(
        x,
        "cuca_system_disk_write_bytes_per_second",
        "Disk write throughput, see the stats API's disk_io.",
        s.disk_io.write_bytes_per_s,
    );
    gauge(
        x,
        "cuca_process_disk_read_bytes_per_second",
        "Disk read throughput of this process.",
        s.process.read_bytes_per_s,
    );
    gauge(
        x,
        "cuca_process_disk_write_bytes_per_second",
        "Disk write throughput of this process.",
        s.process.write_bytes_per_s,
    );

    let per_gpu = |x: &mut Exposition,
                   name: &str,
                   help: &str,
                   value: &dyn Fn(&super::stats::GpuStats) -> Option<f64>| {
        x.family(name, "gauge", help);
        for g in &s.gpus {
            if let Some(v) = value(g) {
                let labels = format!("gpu=\"{}\",name=\"{}\"", g.index, escape(&g.name));
                x.sample(name, &labels, v);
            }
        }
    };
    per_gpu(
        x,
        "cuca_gpu_utilization_percent",
        "GPU utilization.",
        &|g| g.util_pct.map(f64::from),
    );
    per_gpu(
        x,
        "cuca_gpu_memory_total_bytes",
        "Device memory capacity.",
        &|g| g.vram_total_bytes.map(|v| v as f64),
    );
    per_gpu(
        x,
        "cuca_gpu_memory_used_bytes",
        "Device memory in use (all processes).",
        &|g| g.vram_used_bytes.map(|v| v as f64),
    );
    per_gpu(
        x,
        "cuca_gpu_process_memory_bytes",
        "Device memory this process uses.",
        &|g| g.process_vram_bytes.map(|v| v as f64),
    );
    per_gpu(
        x,
        "cuca_gpu_shared_memory_used_bytes",
        "System memory the GPU uses (all processes).",
        &|g| g.shared_vram_used_bytes.map(|v| v as f64),
    );
    per_gpu(
        x,
        "cuca_gpu_temperature_celsius",
        "GPU temperature.",
        &|g| g.temperature_c.map(f64::from),
    );
    per_gpu(x, "cuca_gpu_power_watts", "GPU power draw.", &|g| {
        g.power_w.map(f64::from)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_histogram_renders_cumulative_buckets_with_inclusive_bounds() {
        static BOUNDS: &[f64] = &[0.1, 1.0];
        let h = Histogram::new(BOUNDS);
        h.observe(0.05);
        h.observe(0.1); // on the bound: counts in le="0.1"
        h.observe(0.5);
        h.observe(30.0); // beyond every bound
        let mut out = String::new();
        h.render_into(&mut out, "t_seconds", "model=\"m\"");
        assert_eq!(
            out,
            "t_seconds_bucket{model=\"m\",le=\"0.1\"} 2\n\
             t_seconds_bucket{model=\"m\",le=\"1\"} 3\n\
             t_seconds_bucket{model=\"m\",le=\"+Inf\"} 4\n\
             t_seconds_sum{model=\"m\"} 30.65\n\
             t_seconds_count{model=\"m\"} 4\n"
        );
    }

    #[test]
    fn a_histogram_without_labels_renders_bare_sum_and_count() {
        static BOUNDS: &[f64] = &[1.0];
        let h = Histogram::new(BOUNDS);
        h.observe(f64::NAN);
        let mut out = String::new();
        h.render_into(&mut out, "t", "");
        assert!(out.contains("t_bucket{le=\"1\"} 1\n"), "{out}");
        assert!(out.contains("t_count 1\n"), "{out}");
    }

    #[test]
    fn unknown_status_codes_count_in_their_class() {
        let c = StatusCounters::default();
        c.inc(200);
        c.inc(422);
        c.inc(504);
        c.inc(499);
        assert_eq!((c.get(400), c.get(500), c.get(499)), (1, 1, 1));
        assert_eq!(c.get(502), 0, "502 has its own slot");
        assert_eq!(
            (c.total(), c.errors()),
            (4, 2),
            "a cancelled request is no error"
        );
    }

    #[test]
    fn the_exposition_labels_each_model_and_escapes_names() {
        let metrics = ModelMetrics::default();
        metrics.requests.inc(200);
        metrics.ttft.observe(0.2);
        let models = [ModelView {
            name: "we\"ird\\name",
            state: ModelState::Ready,
            metrics: &metrics,
            serving: Some(ServingStats::default()),
            kv_bytes: 7,
            weights_bytes: 9,
        }];
        let text = render(
            &models,
            &ServerMetrics::default(),
            &SystemSnapshot::default(),
            1.5,
        );
        assert!(text.contains("# TYPE cuca_request_ttft_seconds histogram"));
        assert!(
            text.contains(r#"cuca_requests_total{model="we\"ird\\name",code="200"} 1"#),
            "{text}"
        );
        assert!(
            text.contains(r#"cuca_request_ttft_seconds_bucket{model="we\"ird\\name",le="0.25"} 1"#)
        );
        assert!(text.contains(r#"cuca_model_ready{model="we\"ird\\name"} 1"#));
        assert!(text.contains(r#"cuca_kv_cache_bytes{model="we\"ird\\name"} 7"#));
        assert!(text.contains("cuca_uptime_seconds 1.5"));
    }
}
