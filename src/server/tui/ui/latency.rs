//! The latency panel: a quantile table colored against thresholds, a p95
//! trend beside each row and, when there is room, a chart of the p95 history.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::Line;
use ratatui::widgets::{Axis, Chart, Dataset, GraphType, Paragraph, Row, Table};

use super::widgets::{ACCENT, DIM, INFO, OK, WARN, cell, level, panel, spark};
use crate::server::stats::{LatencyStats, Quantiles};
use crate::server::tui::format;
use crate::server::tui::history::{HISTORY_CAP, History, Sample};

/// Width of the quantile table: metric, four quantiles and the count.
const TABLE_WIDTH: u16 = 5 + 4 * 7 + 5 + 5;
/// Rows of the table: header plus one per metric.
const TABLE_HEIGHT: u16 = 5;
/// Rows the chart needs to be worth drawing.
const CHART_HEIGHT: u16 = 6;
/// Rows of the tokens per second and in-flight sparklines.
const THROUGHPUT_ROWS: u16 = 2;

/// One of the four latencies.
#[derive(Debug, Clone, Copy)]
enum Metric {
    Ttft,
    Itl,
    E2e,
    Queue,
}

impl Metric {
    /// Yellow and red thresholds in seconds, for the p95 and the other cells
    /// alike: a slow median is as bad as a slow tail.
    fn color(self, seconds: f64) -> Color {
        match self {
            Self::Ttft => level(seconds, 1.0, 2.0),
            Self::Itl => level(seconds, 0.1, 0.25),
            Self::E2e => level(seconds, 10.0, 30.0),
            Self::Queue => level(seconds, 0.25, 1.0),
        }
    }

    fn p95(self, sample: &Sample) -> f64 {
        match self {
            Self::Ttft => sample.ttft_p95,
            Self::Itl => sample.itl_p95,
            Self::E2e => sample.e2e_p95,
            Self::Queue => sample.queue_p95,
        }
    }
}

/// Draws the panel.
pub fn draw(
    frame: &mut Frame,
    area: Rect,
    latency: &LatencyStats,
    history: &History,
    focused: bool,
) {
    let title = if latency.window_s > 0.0 {
        format!(" Latency · last {} ", format::seconds(latency.window_s))
    } else {
        " Latency ".to_string()
    };
    let block = panel(title, focused);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < TABLE_HEIGHT || inner.width < 20 {
        return;
    }

    let rows: [(&str, Metric, &Quantiles); 4] = [
        ("TTFT", Metric::Ttft, &latency.ttft),
        ("ITL", Metric::Itl, &latency.itl),
        ("E2E", Metric::E2e, &latency.e2e),
        ("Queue", Metric::Queue, &latency.queue),
    ];
    let trend_width = inner.width.saturating_sub(TABLE_WIDTH + 1);
    let with_trend = trend_width >= 10;
    let table_area = Rect {
        height: TABLE_HEIGHT,
        width: if with_trend { TABLE_WIDTH } else { inner.width },
        ..inner
    };

    let header = Row::new(["", "p50", "p95", "p99", "mean", "n"].map(|title| {
        let line = Line::from(title);
        if title.is_empty() {
            line
        } else {
            line.right_aligned()
        }
    }))
    .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD));
    let body = rows.iter().map(|(name, metric, quantiles)| {
        let quiet = quantiles.count == 0;
        let value = |seconds: f64, bold: bool| {
            if quiet {
                cell("-", Some(DIM), true)
            } else {
                let cell = cell(format::seconds(seconds), Some(metric.color(seconds)), true);
                if bold {
                    cell.style(
                        Style::new()
                            .fg(metric.color(seconds))
                            .add_modifier(Modifier::BOLD),
                    )
                } else {
                    cell
                }
            }
        };
        Row::new([
            cell(*name, None, false),
            value(quantiles.p50, false),
            value(quantiles.p95, true),
            value(quantiles.p99, false),
            value(quantiles.mean, false),
            cell(format::count(quantiles.count), Some(DIM), true),
        ])
    });
    let widths = [
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(5),
    ];
    frame.render_widget(Table::new(body, widths).header(header), table_area);

    if with_trend {
        let x = inner.x + TABLE_WIDTH + 1;
        frame.render_widget(
            Paragraph::new(
                Line::from("p95 trend").style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            ),
            Rect {
                x,
                y: inner.y,
                width: trend_width,
                height: 1,
            },
        );
        for (row, (_, metric, quantiles)) in rows.iter().enumerate() {
            let values = history.series(usize::from(trend_width), |sample| metric.p95(sample));
            let color = if quantiles.count == 0 {
                DIM
            } else {
                metric.color(quantiles.p95)
            };
            let at = Rect {
                x,
                y: inner.y + 1 + row as u16,
                width: trend_width,
                height: 1,
            };
            spark(frame, at, &values, None, color);
        }
    }

    let mut spare = inner.height - TABLE_HEIGHT;
    let mut y = inner.y + TABLE_HEIGHT;
    if spare >= THROUGHPUT_ROWS {
        draw_throughput(
            frame,
            Rect {
                y,
                height: 1,
                ..inner
            },
            "gen tok/s",
            history,
            |s| s.tokens_per_s,
            OK,
        );
        draw_throughput(
            frame,
            Rect {
                y: y + 1,
                height: 1,
                ..inner
            },
            "in flight",
            history,
            |s| s.in_flight,
            INFO,
        );
        spare -= THROUGHPUT_ROWS;
        y += THROUGHPUT_ROWS;
    }
    if spare >= CHART_HEIGHT && !history.is_empty() {
        draw_chart(
            frame,
            Rect {
                y,
                height: spare,
                ..inner
            },
            history,
        );
    }
}

/// `gen tok/s   94.3 ▁▂▅█`: the latest value, then its history.
fn draw_throughput(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    history: &History,
    field: fn(&Sample) -> f64,
    color: Color,
) {
    let values = history.series(usize::from(area.width), field);
    let latest = values.last().copied().unwrap_or(0.0);
    let label = Line::from(format!("{label} {:>7}", format::per_second(latest)));
    let width = 18.min(area.width / 2);
    frame.render_widget(Paragraph::new(label), Rect { width, ..area });
    let at = Rect {
        x: area.x + width + 1,
        width: area.width.saturating_sub(width + 1),
        ..area
    };
    spark(frame, at, &values, None, color);
}

/// TTFT, queue and E2E p95 over the kept history, newest at the right edge.
fn draw_chart(frame: &mut Frame, area: Rect, history: &History) {
    let series = |metric: Metric| -> Vec<(f64, f64)> {
        let values = history.series(HISTORY_CAP, |sample| metric.p95(sample));
        let offset = HISTORY_CAP - values.len();
        values
            .into_iter()
            .enumerate()
            .map(|(i, value)| ((offset + i) as f64, value))
            .collect()
    };
    let ttft = series(Metric::Ttft);
    let queue = series(Metric::Queue);
    let e2e = series(Metric::E2e);
    let top = [&ttft, &queue, &e2e]
        .iter()
        .flat_map(|points| points.iter().map(|(_, value)| *value))
        .fold(0.0, f64::max);
    let top = if top > 0.0 { top * 1.1 } else { 1.0 };

    let chart = Chart::new(vec![
        dataset("TTFT p95", &ttft, ACCENT),
        dataset("Queue p95", &queue, WARN),
        dataset("E2E p95", &e2e, Color::Magenta),
    ])
    .x_axis(
        Axis::default()
            .bounds([0.0, (HISTORY_CAP - 1) as f64])
            .style(Style::new().fg(DIM)),
    )
    .y_axis(
        Axis::default()
            .bounds([0.0, top])
            .labels(["0".to_string(), format::seconds(top)])
            .style(Style::new().fg(DIM)),
    );
    frame.render_widget(chart, area);
}

fn dataset<'a>(name: &'static str, points: &'a [(f64, f64)], color: Color) -> Dataset<'a> {
    Dataset::default()
        .name(name)
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(color))
        .data(points)
}
