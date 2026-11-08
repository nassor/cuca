//! The models table of a server view.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::{Cell, Paragraph, TableState};

use super::widgets::{
    CRIT, Column, DIM, OK, WARN, cell, level, panel, table, text_bar, usage_color,
};
use crate::server::stats::{ModelState, ModelStats};
use crate::server::tui::format;

/// Longest state text kept in the table; the details popup has the rest.
const MAX_STATE_WIDTH: usize = 28;

fn state_text(model: &ModelStats) -> String {
    match model.state {
        ModelState::Ready => "ready".to_string(),
        ModelState::Loading => match model.load_progress {
            Some(progress) => format!("loading {:.0}%", progress.clamp(0.0, 1.0) * 100.0),
            None => "loading".to_string(),
        },
        ModelState::Failed => match model.error.as_deref() {
            Some(error) => format!("failed: {}", error.lines().next().unwrap_or("")),
            None => "failed".to_string(),
        },
    }
}

fn columns(models: &[ModelStats]) -> Vec<Column<ModelStats>> {
    let name_width = models
        .iter()
        .map(|model| model.name.chars().count())
        .max()
        .unwrap_or(0)
        .clamp(5, 24);
    let state_width = models
        .iter()
        .map(|model| state_text(model).chars().count())
        .max()
        .unwrap_or(0)
        .clamp(5, MAX_STATE_WIDTH);
    let column =
        |title, width: usize, priority, right, render: fn(&ModelStats) -> Cell<'static>| Column {
            title,
            width: width as u16,
            priority,
            right,
            fill: false,
            cell: render,
        };
    vec![
        column("Model", name_width, 0, false, |m| {
            cell(format::truncate(&m.name, 24), None, false)
        }),
        column("State", state_width, 0, false, |m| {
            let color = match m.state {
                ModelState::Ready => OK,
                ModelState::Loading => WARN,
                ModelState::Failed => CRIT,
            };
            cell(
                format::truncate(&state_text(m), MAX_STATE_WIDTH),
                Some(color),
                false,
            )
        }),
        column("Device", 10, 1, false, |m| {
            cell(format::truncate(device_of(m), 10), None, false)
        }),
        column("Run/Wait/Max", 12, 0, true, |m| {
            let color = if m.waiting > 0 { Some(WARN) } else { None };
            cell(
                format!("{}/{}/{}", m.running, m.waiting, m.max_sequences),
                color,
                true,
            )
        }),
        column("KV usage", 13, 0, false, |m| {
            let ratio = m.kv.usage_ratio;
            cell(
                format!("{} {:>3.0}%", text_bar(ratio, 8), ratio * 100.0),
                Some(usage_color(ratio)),
                false,
            )
        }),
        column("Cached", 6, 3, true, |m| {
            cell(format::count(m.kv.cached_blocks), Some(DIM), true)
        }),
        column("Prefix", 6, 2, true, |m| {
            cell(format::ratio_percent(m.prefix_hit_ratio), None, true)
        }),
        column("Gen t/s", 7, 1, true, |m| {
            cell(format::per_second(m.tokens_per_s.generation), None, true)
        }),
        column("Pp t/s", 7, 2, true, |m| {
            cell(format::per_second(m.tokens_per_s.prompt), None, true)
        }),
        column("Step ms", 7, 2, true, |m| {
            cell(
                format!("{:.1}", m.mean_step_ms),
                Some(level(m.mean_step_ms, 100.0, 400.0)),
                true,
            )
        }),
        column("Rows/st", 7, 3, true, |m| {
            cell(format!("{:.1}", m.mean_rows_per_step), None, true)
        }),
        column("Pre/min", 7, 1, true, |m| {
            let color = if m.preemptions_per_min > 0.0 {
                Some(WARN)
            } else {
                None
            };
            cell(format::per_second(m.preemptions_per_min), color, true)
        }),
        column("Weights", 8, 3, true, |m| {
            cell(format::bytes_compact(m.memory.weights_bytes), None, true)
        }),
        column("KV size", 8, 3, true, |m| {
            cell(format::bytes_compact(m.memory.kv_bytes), None, true)
        }),
    ]
}

/// The device the memory report names, falling back to the model's own.
fn device_of(model: &ModelStats) -> &str {
    if model.memory.device.is_empty() {
        &model.device
    } else {
        &model.memory.device
    }
}

/// Draws the table; the selected row follows `state`.
pub fn draw(
    frame: &mut Frame,
    area: Rect,
    models: &[ModelStats],
    state: &mut TableState,
    focused: bool,
) {
    let title = if focused {
        format!(" Models ({}) · ↑↓ select · Enter details ", models.len())
    } else {
        format!(" Models ({}) ", models.len())
    };
    let block = panel(title, focused);
    let available = block.inner(area).width.saturating_sub(2);
    if models.is_empty() {
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(Paragraph::new("No model loaded."), inner);
        return;
    }
    if state.selected().is_none_or(|at| at >= models.len()) {
        state.select(Some(models.len() - 1));
    }
    let widget = table(models, &columns(models), available, block);
    frame.render_stateful_widget(widget, area, state);
}
