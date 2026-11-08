//! The header: who is being watched, the connection, the request counters.

use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::widgets::{ACCENT, CRIT, DIM, OK, WARN, panel};
use crate::server::stats::{NodeRole, StatsSnapshot};
use crate::server::tui::app::{Data, UiState, View};
use crate::server::tui::client::Connection;
use crate::server::tui::format;

fn separator() -> Span<'static> {
    Span::styled(" │ ", Style::new().fg(DIM))
}

fn stat(label: &str, value: String, color: Option<Color>) -> [Span<'static>; 2] {
    [
        Span::styled(format!("{label} "), Style::new().fg(DIM)),
        Span::styled(
            value,
            color.map_or_else(Style::new, |color| Style::new().fg(color)),
        ),
    ]
}

/// The connection badge and, while it is down, the reason.
fn connection(data: &Data) -> (Vec<Span<'static>>, Option<String>) {
    match &data.connection {
        Connection::Connecting => (
            vec![Span::styled("◌ connecting", Style::new().fg(WARN))],
            None,
        ),
        Connection::Connected => (vec![Span::styled("● connected", Style::new().fg(OK))], None),
        Connection::Reconnecting {
            attempt,
            retry_at,
            error,
        } => {
            let wait = retry_at
                .saturating_duration_since(Instant::now())
                .as_secs_f64()
                .ceil() as u64;
            let age = data
                .last_update
                .map(|at| format!(", data {} old", format::uptime(at.elapsed().as_secs_f64())))
                .unwrap_or_default();
            let text = if wait == 0 {
                format!("✖ reconnecting now (attempt {attempt}{age})")
            } else {
                format!("✖ reconnecting in {wait}s (attempt {attempt}{age})")
            };
            (
                vec![Span::styled(
                    text,
                    Style::new().fg(CRIT).add_modifier(Modifier::BOLD),
                )],
                Some(error.clone()),
            )
        }
    }
}

/// Draws the header over `area` (four rows: border, two lines, border).
pub fn draw(
    frame: &mut Frame,
    area: Rect,
    data: &Data,
    ui: &UiState,
    shown: Option<&StatsSnapshot>,
) {
    let block = panel(" cuca tui ", false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut first = Vec::new();
    if let Some(snapshot) = shown {
        let node = &snapshot.node;
        let (badge, color) = match node.role {
            NodeRole::Server => (" SERVER ", OK),
            NodeRole::Balancer => (" BALANCER ", Color::Magenta),
        };
        first.push(Span::styled(
            badge,
            Style::new()
                .fg(Color::Black)
                .bg(color)
                .add_modifier(Modifier::BOLD),
        ));
        first.push(Span::styled(
            format!(
                " {} ",
                if node.id.is_empty() {
                    &node.listen
                } else {
                    &node.id
                }
            ),
            Style::new().add_modifier(Modifier::BOLD),
        ));
        if !node.version.is_empty() {
            first.push(Span::styled(
                format!("v{} ", node.version),
                Style::new().fg(DIM),
            ));
        }
        first.push(Span::raw(format!("up {} ", format::uptime(node.uptime_s))));
    }
    first.push(Span::styled(data.url.clone(), Style::new().fg(ACCENT)));
    if let View::Upstream(id) = &ui.view {
        first.push(Span::styled(
            format!("  ▸ upstream {id}"),
            Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ));
        let stale = data
            .upstream(id)
            .and_then(|up| up.snapshot_age_s)
            .filter(|age| *age > 5.0);
        if let Some(age) = stale {
            first.push(Span::styled(
                format!("  STALE {}", format::uptime(age)),
                Style::new().fg(WARN).add_modifier(Modifier::BOLD),
            ));
        }
    }
    let (mut second, error) = connection(data);
    if let Some(error) = &error {
        first.push(Span::styled(
            format!("  ✖ {}", format::truncate(error, 48)),
            Style::new().fg(CRIT),
        ));
    }

    if data.paused {
        second.push(Span::raw(" "));
        second.push(Span::styled(
            " PAUSED ",
            Style::new()
                .fg(Color::Black)
                .bg(WARN)
                .add_modifier(Modifier::BOLD),
        ));
    }
    second.push(separator());
    if let Some(snapshot) = shown {
        let requests = &snapshot.requests;
        let share = if requests.total > 0 {
            format!(
                " ({:.1}%)",
                requests.errors as f64 / requests.total as f64 * 100.0
            )
        } else {
            String::new()
        };
        let groups = [
            stat("req/s", format::per_second(requests.rate_per_s), None),
            stat("in-flight", requests.in_flight.to_string(), None),
            stat("total", format::count(requests.total), None),
            stat(
                "errors",
                format!("{}{share}", format::count(requests.errors)),
                if requests.errors > 0 {
                    Some(WARN)
                } else {
                    None
                },
            ),
        ];
        for (i, group) in groups.into_iter().enumerate() {
            if i > 0 {
                second.push(separator());
            }
            second.extend(group);
        }
    } else {
        second.push(Span::styled(
            format!("every {} ms", data.interval_ms),
            Style::new().fg(DIM),
        ));
    }
    frame.render_widget(
        Paragraph::new(vec![Line::from(first), Line::from(second)]),
        inner,
    );
}
