//! Small drawing helpers shared by the panels: colors, panel frames, gauge
//! rows with a sparkline, the table column fitter.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Cell, LineGauge, Row, Sparkline, Table};

pub const OK: Color = Color::Green;
pub const WARN: Color = Color::Yellow;
pub const CRIT: Color = Color::Red;
pub const INFO: Color = Color::LightBlue;
pub const ACCENT: Color = Color::Cyan;
pub const DIM: Color = Color::DarkGray;

/// Green below `warn`, yellow below `crit`, red above.
pub fn level(value: f64, warn: f64, crit: f64) -> Color {
    if value >= crit {
        CRIT
    } else if value >= warn {
        WARN
    } else {
        OK
    }
}

/// The color of a usage ratio: yellow from 70%, red from 90%.
pub fn usage_color(ratio: f64) -> Color {
    level(ratio, 0.7, 0.9)
}

/// `used / total` clamped to `0..=1`, zero for an empty or unknown total.
pub fn ratio(used: f64, total: f64) -> f64 {
    if total > 0.0 && used.is_finite() {
        (used / total).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// A bordered panel; the focused one has an accent border.
pub fn panel<'a>(title: impl Into<Line<'a>>, focused: bool) -> Block<'a> {
    let border = if focused { ACCENT } else { DIM };
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border))
        .title(
            title
                .into()
                .style(Style::new().add_modifier(Modifier::BOLD)),
        )
}

/// `text` cut or padded with spaces to exactly `width` characters.
pub fn fit(text: &str, width: usize) -> String {
    let mut out: String = text.chars().take(width).collect();
    let missing = width.saturating_sub(out.chars().count());
    out.extend(std::iter::repeat_n(' ', missing));
    out
}

/// A text bar of `width` cells: `█████░░░`.
pub fn text_bar(ratio: f64, width: usize) -> String {
    let filled = ((ratio.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    let mut bar = "█".repeat(filled);
    bar.push_str(&"░".repeat(width - filled));
    bar
}

/// A one-row line gauge: the label, then a bar.
pub fn gauge(label: String, ratio: f64, color: Color) -> LineGauge<'static> {
    let ratio = if ratio.is_nan() {
        0.0
    } else {
        ratio.clamp(0.0, 1.0)
    };
    LineGauge::default()
        .label(label)
        .ratio(ratio)
        .filled_symbol("━")
        .unfilled_symbol("─")
        .filled_style(Style::new().fg(color))
        .unfilled_style(Style::new().fg(DIM))
}

/// Draws `values` (oldest first) as a one-row sparkline filling `area`; the
/// newest value sits at the right edge. `max` fixes the scale, `None` scales
/// to the largest value.
pub fn spark(frame: &mut Frame, area: Rect, values: &[f64], max: Option<f64>, color: Color) {
    if area.is_empty() {
        return;
    }
    let width = usize::from(area.width);
    let tail = &values[values.len().saturating_sub(width)..];
    let top = max
        .unwrap_or_else(|| tail.iter().copied().fold(0.0, f64::max))
        .max(1e-9);
    let mut data: Vec<Option<u64>> = vec![None; width - tail.len()];
    data.extend(tail.iter().map(|value| {
        let scaled = (value.max(0.0) / top * 1000.0).round().min(1000.0);
        Some(scaled as u64)
    }));
    frame.render_widget(
        Sparkline::default()
            .data(data)
            .max(1000)
            .style(Style::new().fg(color)),
        area,
    );
}

/// Where a gauge row puts its sparkline, if it has one.
pub struct Trend<'a> {
    /// Oldest first.
    pub values: &'a [f64],
    /// Fixed scale, or the largest value when `None`.
    pub max: Option<f64>,
    /// Line color.
    pub color: Color,
}

/// One row: a labelled gauge and, to its right, a sparkline when the row is
/// wide enough.
pub fn gauge_row(
    frame: &mut Frame,
    area: Rect,
    label: String,
    ratio: f64,
    color: Color,
    trend: Option<Trend<'_>>,
) {
    let spark_width = match trend {
        Some(_) if area.width >= 44 => (area.width / 3).clamp(10, 40),
        _ => 0,
    };
    let gauge_width = area
        .width
        .saturating_sub(if spark_width > 0 { spark_width + 1 } else { 0 });
    frame.render_widget(
        gauge(label, ratio, color),
        Rect {
            width: gauge_width,
            ..area
        },
    );
    if let (Some(trend), true) = (trend, spark_width > 0) {
        let at = Rect {
            x: area.x + gauge_width + 1,
            width: spark_width,
            ..area
        };
        spark(frame, at, trend.values, trend.max, trend.color);
    }
}

/// One table column.
pub struct Column<T> {
    /// Header text.
    pub title: &'static str,
    /// Width in cells (the minimum when `fill`).
    pub width: u16,
    /// Lower is kept longer when the table is too narrow.
    pub priority: u8,
    /// Right-aligned numbers.
    pub right: bool,
    /// Takes the width the others leave.
    pub fill: bool,
    /// The cell of one row.
    pub cell: fn(&T) -> Cell<'static>,
}

/// A numeric or text cell; `right` aligns it to the column's right edge.
pub fn cell(text: impl Into<String>, color: Option<Color>, right: bool) -> Cell<'static> {
    let line = Line::from(text.into());
    let line = if right { line.right_aligned() } else { line };
    let cell = Cell::from(line);
    match color {
        Some(color) => cell.style(Style::new().fg(color)),
        None => cell,
    }
}

/// Builds a table of `rows`, dropping the columns of highest priority number
/// (last first) until the rest fits `available` cells.
pub fn table<'a, T>(
    rows: &[T],
    columns: &[Column<T>],
    available: u16,
    block: Block<'a>,
) -> Table<'a> {
    let mut keep: Vec<usize> = (0..columns.len()).collect();
    let total = |keep: &[usize]| -> u16 {
        let widths: u16 = keep.iter().map(|&i| columns[i].width).sum();
        widths + keep.len().saturating_sub(1) as u16
    };
    while keep.len() > 1 && total(&keep) > available {
        let worst = keep.iter().map(|&i| columns[i].priority).max().unwrap_or(0);
        let Some(at) = keep.iter().rposition(|&i| columns[i].priority == worst) else {
            break;
        };
        keep.remove(at);
    }

    let widths = keep.iter().map(|&i| {
        if columns[i].fill {
            Constraint::Fill(1)
        } else {
            Constraint::Length(columns[i].width)
        }
    });
    let header = Row::new(keep.iter().map(|&i| {
        let line = Line::from(columns[i].title);
        Cell::from(if columns[i].right {
            line.right_aligned()
        } else {
            line
        })
    }))
    .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD));
    let body = rows
        .iter()
        .map(|row| Row::new(keep.iter().map(|&i| (columns[i].cell)(row))));
    Table::new(body, widths)
        .header(header)
        .block(block)
        .row_highlight_style(
            Style::new()
                .bg(Color::Rgb(40, 44, 60))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_bars_have_a_fixed_width() {
        assert_eq!(text_bar(0.5, 8), "████░░░░");
        assert_eq!(text_bar(2.0, 4), "████");
        assert_eq!(text_bar(f64::NAN.max(0.0), 3), "░░░");
    }

    #[test]
    fn fit_pads_and_cuts() {
        assert_eq!(fit("ab", 4), "ab  ");
        assert_eq!(fit("abcdef", 3), "abc");
    }

    #[test]
    fn ratio_guards_an_empty_total() {
        assert_eq!(ratio(5.0, 0.0), 0.0);
        assert_eq!(ratio(5.0, 10.0), 0.5);
        assert_eq!(ratio(50.0, 10.0), 1.0);
    }
}
