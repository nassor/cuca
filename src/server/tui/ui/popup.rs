//! Overlays: the key help and the details of one model.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use super::widgets::{ACCENT, CRIT, DIM, OK, WARN, panel};
use crate::server::stats::{LatencyStats, ModelState, ModelStats, Quantiles};
use crate::server::tui::format;

/// A `width` x `height` rectangle centered in `area` (clamped to it).
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [at] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(row);
    at
}

fn key(name: &str, what: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{name:>12}  "),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw(what.to_string()),
    ])
}

/// The key help.
pub fn draw_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        key("q / Esc", "back from a drilled upstream, else quit"),
        key("Ctrl-C", "quit"),
        key(
            "Tab / S-Tab",
            "next / previous pane (the only pane shown on a small terminal)",
        ),
        key(
            "Up / Down",
            "select a row; in the Latency pane, scroll the reasons",
        ),
        key("Enter", "details of the model / drill into the upstream"),
        key("p", "pause or resume updates"),
        key("? / h", "this help"),
        Line::default(),
        Line::from(Span::styled(
            "Colors: green is fine, yellow costs latency, red is a bottleneck.",
            Style::new().fg(DIM),
        )),
    ];
    let at = centered(area, 78, lines.len() as u16 + 2);
    frame.render_widget(Clear, at);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(panel(" Help ", true)),
        at,
    );
}

fn field(label: &str, value: String) -> Span<'static> {
    Span::raw(format!("{label} {value}   "))
}

fn quantile_line(name: &str, quantiles: &Quantiles) -> Line<'static> {
    if quantiles.count == 0 {
        return Line::from(Span::styled(
            format!("  {name:<6} no request in the window"),
            Style::new().fg(DIM),
        ));
    }
    Line::from(format!(
        "  {name:<6} p50 {:>8}  p95 {:>8}  p99 {:>8}  mean {:>8}  n {}",
        format::seconds(quantiles.p50),
        format::seconds(quantiles.p95),
        format::seconds(quantiles.p99),
        format::seconds(quantiles.mean),
        quantiles.count,
    ))
}

fn latency_lines(latency: &LatencyStats) -> Vec<Line<'static>> {
    vec![
        quantile_line("TTFT", &latency.ttft),
        quantile_line("ITL", &latency.itl),
        quantile_line("E2E", &latency.e2e),
        quantile_line("Queue", &latency.queue),
    ]
}

fn heading(text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_string(),
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
    ))
}

/// Every figure of one model.
fn model_lines(model: &ModelStats) -> Vec<Line<'static>> {
    let (state_color, state) = match model.state {
        ModelState::Ready => (OK, "ready".to_string()),
        ModelState::Loading => (
            WARN,
            model.load_progress.map_or_else(
                || "loading".to_string(),
                |p| format!("loading {:.0}%", p * 100.0),
            ),
        ),
        ModelState::Failed => (CRIT, "failed".to_string()),
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{} ", model.name),
            Style::new().add_modifier(Modifier::BOLD),
        ),
        Span::styled(state, Style::new().fg(state_color)),
    ])];
    if let Some(error) = &model.error {
        lines.push(Line::from(Span::styled(
            format!("error: {error}"),
            Style::new().fg(CRIT),
        )));
    }
    if !model.source.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("source {}", model.source),
            Style::new().fg(DIM),
        )));
    }
    lines.push(Line::from(vec![
        field("device", model.device.clone()),
        field("context", format::count(model.context_length)),
        field("max seqs", model.max_sequences.to_string()),
    ]));

    lines.push(heading("Engine"));
    lines.push(Line::from(vec![
        field("running", model.running.to_string()),
        field("waiting", model.waiting.to_string()),
        field("steps/s", format::per_second(model.steps_per_s)),
        field("rows/s", format::per_second(model.rows_per_s)),
    ]));
    lines.push(Line::from(vec![
        field("mean step", format::millis(model.mean_step_ms)),
        field("last step", format::millis(model.last_step_ms)),
        field("rows/step", format!("{:.1}", model.mean_rows_per_step)),
        field(
            "prefill share",
            format::ratio_percent(Some(model.prefill_row_share)),
        ),
    ]));
    lines.push(Line::from(vec![
        field(
            "prompt tok/s",
            format::per_second(model.tokens_per_s.prompt),
        ),
        field(
            "gen tok/s",
            format::per_second(model.tokens_per_s.generation),
        ),
        field("mean queue", format::millis(model.mean_queue_ms)),
        field(
            "preemptions/min",
            format::per_second(model.preemptions_per_min),
        ),
    ]));

    lines.push(heading("K/V cache and memory"));
    lines.push(Line::from(vec![
        field(
            "blocks used",
            format!("{}/{}", model.kv.used_blocks, model.kv.total_blocks),
        ),
        field("free", model.kv.free_blocks.to_string()),
        field("cached", model.kv.cached_blocks.to_string()),
        field("usage", format::ratio_percent(Some(model.kv.usage_ratio))),
        field("prefix hit", format::ratio_percent(model.prefix_hit_ratio)),
    ]));
    lines.push(Line::from(vec![
        field("weights", format::bytes(model.memory.weights_bytes)),
        field("K/V", format::bytes(model.memory.kv_bytes)),
        field("on", model.memory.device.clone()),
    ]));
    if let Some(tiers) = &model.expert_tiers {
        lines.push(heading("Expert tiers"));
        lines.push(Line::from(vec![
            field("disk reads/s", format::per_second(tiers.disk_reads_per_s)),
            field("read", format::rate(tiers.read_bytes_per_s)),
            field("hit", format::ratio_percent(tiers.hit_ratio)),
            field("RAM tier", format::bytes(tiers.ram_resident_bytes)),
            field("device tier", format::bytes(tiers.device_resident_bytes)),
        ]));
    }

    lines.push(heading("Requests"));
    lines.push(Line::from(vec![
        field("in flight", model.requests.in_flight.to_string()),
        field("total", format::count(model.requests.total)),
        field("errors", format::count(model.requests.errors)),
        field("req/s", format::per_second(model.requests.rate_per_s)),
    ]));
    lines.push(heading("Latency of this model"));
    lines.extend(latency_lines(&model.latency));
    lines
}

/// The details of the model named `name`.
pub fn draw_model(frame: &mut Frame, area: Rect, models: &[ModelStats], name: &str) {
    let lines = match models.iter().find(|model| model.name == name) {
        Some(model) => model_lines(model),
        None => vec![Line::from(Span::styled(
            format!("{name} is no longer loaded."),
            Style::new().fg(Color::Yellow),
        ))],
    };
    let at = centered(area, 96, lines.len() as u16 + 2);
    frame.render_widget(Clear, at);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(panel(" Model details · Esc closes ", true)),
        at,
    );
}
