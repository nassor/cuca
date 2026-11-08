//! Rendering: the frame layout and the panels.
//!
//! A terminal of at least [`FULL_WIDTH`] x [`FULL_HEIGHT`] shows every panel at
//! once. A smaller one shows the focused pane alone (Tab switches), under a
//! one-line summary of the worst diagnosis, so 80x24 stays usable.

mod header;
mod latency;
mod models;
mod popup;
mod resources;
mod upstreams;
mod why;
mod widgets;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::app::{App, Data, Pane, Popup, UiState, View};
use super::history::History;
use crate::server::stats::{NodeRole, StatsSnapshot, UpstreamStats};
use widgets::{ACCENT, CRIT, DIM, panel};

/// Smallest terminal width that shows every panel at once.
pub const FULL_WIDTH: u16 = 110;
/// Smallest terminal height that shows every panel at once.
pub const FULL_HEIGHT: u16 = 34;
/// Below this the UI only asks for a bigger terminal.
const MIN_WIDTH: u16 = 40;
const MIN_HEIGHT: u16 = 10;
/// Rows of the header block.
const HEADER_HEIGHT: u16 = 4;

/// What the bottom list shows.
#[derive(Clone, Copy)]
enum List<'a> {
    Models,
    Upstreams(&'a [UpstreamStats]),
}

/// Draws one frame.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        frame.render_widget(
            Paragraph::new(format!(
                "Terminal too small: {}x{}, need {MIN_WIDTH}x{MIN_HEIGHT}",
                area.width, area.height
            )),
            area,
        );
        return;
    }
    let [header_area, body, footer] = Layout::vertical([
        Constraint::Length(HEADER_HEIGHT),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(area);

    let App { data, ui } = app;
    ui.compact = area.width < FULL_WIDTH || area.height < FULL_HEIGHT;

    let shown = match &ui.view {
        View::Root => data.snapshot.as_ref(),
        View::Upstream(id) => data.upstream(id).and_then(|up| up.snapshot.as_deref()),
    };
    header::draw(frame, header_area, data, ui, shown);
    footer_line(frame, footer, ui, data);

    let view = ui.view.clone();
    match (&view, shown) {
        (_, None) => draw_waiting(frame, body, data, ui),
        (View::Root, Some(root)) if root.node.role == NodeRole::Balancer => {
            if let Some(rolled) = data.fleet.as_ref() {
                let title = format!(
                    "Fleet resources ({}/{} upstreams up)",
                    root.upstreams
                        .iter()
                        .filter(|up| up.healthy && up.snapshot.is_some())
                        .count(),
                    root.upstreams.len()
                );
                dashboard(
                    frame,
                    body,
                    rolled,
                    &data.history,
                    &title,
                    List::Upstreams(&root.upstreams),
                    ui,
                );
            }
        }
        (View::Root, Some(root)) => {
            dashboard(
                frame,
                body,
                root,
                &data.history,
                "Resources",
                List::Models,
                ui,
            );
        }
        (View::Upstream(id), Some(nested)) => {
            let empty = History::new();
            let history = data.upstream_history.get(id).unwrap_or(&empty);
            dashboard(frame, body, nested, history, "Resources", List::Models, ui);
        }
    }

    match &ui.popup {
        Popup::None => {}
        Popup::Help => popup::draw_help(frame, area),
        Popup::Model(name) => {
            let models = shown
                .map(|snapshot| snapshot.models.as_slice())
                .unwrap_or_default();
            popup::draw_model(frame, area, models, name);
        }
    }
}

/// The key hints and, on a small terminal, which pane is showing.
fn footer_line(frame: &mut Frame, area: Rect, ui: &UiState, data: &Data) {
    let mut spans = Vec::new();
    if ui.compact {
        for pane in App::PANES {
            let name = pane_name(pane, data.is_balancer() && ui.view == View::Root);
            if pane == ui.focus {
                spans.push(Span::styled(format!("[{name}]"), Style::new().fg(ACCENT)));
            } else {
                spans.push(Span::styled(format!(" {name} "), Style::new().fg(DIM)));
            }
        }
        spans.push(Span::styled(
            " Tab · ↑↓ · Enter · p pause · ? help · q quit",
            Style::new().fg(DIM),
        ));
    } else {
        spans.push(Span::styled(
            " q quit · Tab pane · ↑↓ select · Enter details · p pause · ? help ",
            Style::new().fg(DIM),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn pane_name(pane: Pane, balancer: bool) -> &'static str {
    match pane {
        Pane::Resources => "Resources",
        Pane::Latency => "Latency",
        Pane::List if balancer => "Upstreams",
        Pane::List => "Models",
    }
}

/// What the body says before the first snapshot, or when a drilled upstream
/// has none.
fn draw_waiting(frame: &mut Frame, area: Rect, data: &Data, ui: &UiState) {
    let mut lines = vec![Line::default()];
    if matches!(ui.view, View::Upstream(_)) {
        lines.push(Line::from(
            "This upstream has not reported a snapshot yet. Esc goes back.",
        ));
    } else {
        lines.push(Line::from(Span::styled(
            format!("Waiting for {} ...", data.url),
            Style::new().fg(ACCENT),
        )));
        if let super::client::Connection::Reconnecting { error, .. } = &data.connection {
            lines.push(Line::from(Span::styled(
                error.clone(),
                Style::new().fg(CRIT),
            )));
        }
        lines.push(Line::from(Span::styled(
            "The stats API is opt-in: start the server with --stats-api.",
            Style::new().fg(DIM),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .centered()
            .block(panel(" cuca tui ", false)),
        area,
    );
}

/// Lays out the three panes (or the focused one) for `snapshot`.
fn dashboard(
    frame: &mut Frame,
    area: Rect,
    snapshot: &StatsSnapshot,
    history: &History,
    title: &str,
    list: List<'_>,
    ui: &mut UiState,
) {
    // A fleet's adapters are numbered across machines, so the models' own
    // device indices do not apply to them.
    let models = match list {
        List::Models => snapshot.models.as_slice(),
        List::Upstreams(_) => &[],
    };
    let resources = resources::Resources {
        title,
        system: &snapshot.system,
        models,
        history,
    };
    if ui.compact {
        draw_compact(frame, area, snapshot, &resources, history, list, ui);
        return;
    }

    let list_rows = match list {
        List::Models => snapshot.models.len(),
        List::Upstreams(upstreams) => upstreams.len(),
    };
    let list_height = (list_rows as u16 + 3).clamp(6, 10);
    let top_height = area.height.saturating_sub(list_height).max(12);
    let [top, bottom] =
        Layout::vertical([Constraint::Length(top_height), Constraint::Fill(1)]).areas(area);
    let left_width = (u32::from(area.width) * 52 / 100).clamp(56, 70) as u16;
    let [left, right] =
        Layout::horizontal([Constraint::Length(left_width), Constraint::Fill(1)]).areas(top);
    resources::draw(frame, left, &resources, ui.focus == Pane::Resources);
    draw_right(frame, right, snapshot, history, ui);
    draw_list(frame, bottom, snapshot, list, ui);
}

/// The latency panel over the reasons.
fn draw_right(
    frame: &mut Frame,
    area: Rect,
    snapshot: &StatsSnapshot,
    history: &History,
    ui: &UiState,
) {
    // Table only; plus the two throughput rows; plus the p95 chart.
    let latency_height = match area.height {
        24.. => 15,
        16.. => 9,
        _ => 7,
    };
    let latency_height = latency_height.min(area.height.saturating_sub(5)).max(1);
    let [latency_area, why_area] =
        Layout::vertical([Constraint::Length(latency_height), Constraint::Fill(1)]).areas(area);
    latency::draw(
        frame,
        latency_area,
        &snapshot.latency,
        history,
        ui.focus == Pane::Latency,
    );
    why::draw(
        frame,
        why_area,
        &snapshot.diagnosis,
        ui.diag_scroll,
        ui.focus == Pane::Latency,
    );
}

fn draw_list(
    frame: &mut Frame,
    area: Rect,
    snapshot: &StatsSnapshot,
    list: List<'_>,
    ui: &mut UiState,
) {
    let focused = ui.focus == Pane::List;
    match list {
        List::Models => models::draw(frame, area, &snapshot.models, &mut ui.model_table, focused),
        List::Upstreams(upstreams) => {
            upstreams::draw(frame, area, upstreams, &mut ui.upstream_table, focused)
        }
    }
}

/// The focused pane alone, under the one-line diagnosis summary.
fn draw_compact(
    frame: &mut Frame,
    area: Rect,
    snapshot: &StatsSnapshot,
    resources: &resources::Resources<'_>,
    history: &History,
    list: List<'_>,
    ui: &mut UiState,
) {
    let show_summary = ui.focus != Pane::Latency;
    let [summary, pane] = Layout::vertical([
        Constraint::Length(u16::from(show_summary)),
        Constraint::Fill(1),
    ])
    .areas(area);
    if show_summary {
        frame.render_widget(Paragraph::new(why::summary(&snapshot.diagnosis)), summary);
    }
    match ui.focus {
        Pane::Resources => resources::draw(frame, pane, resources, true),
        Pane::Latency => draw_right(frame, pane, snapshot, history, ui),
        Pane::List => draw_list(frame, pane, snapshot, list, ui),
    }
}
