//! The upstreams table of a balancer view.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::{Cell, Paragraph, TableState};

use super::widgets::{CRIT, Column, DIM, OK, WARN, cell, level, panel, table};
use crate::server::stats::{Quantiles, UpstreamStats};
use crate::server::tui::format;

fn quantile_cell(quantiles: &Quantiles, warn: f64, crit: f64) -> Cell<'static> {
    if quantiles.count == 0 {
        cell("-", Some(DIM), true)
    } else {
        cell(
            format::seconds(quantiles.p95),
            Some(level(quantiles.p95, warn, crit)),
            true,
        )
    }
}

fn columns(upstreams: &[UpstreamStats]) -> Vec<Column<UpstreamStats>> {
    let id_width = upstreams
        .iter()
        .map(|u| u.id.chars().count())
        .max()
        .unwrap_or(0)
        .clamp(8, 24) as u16;
    let column =
        |title, width: u16, priority, right, fill, render: fn(&UpstreamStats) -> Cell<'static>| {
            Column {
                title,
                width,
                priority,
                right,
                fill,
                cell: render,
            }
        };
    vec![
        column("Upstream", id_width, 0, false, false, |u| {
            cell(
                format::truncate(&u.id, 24),
                if u.healthy { None } else { Some(CRIT) },
                false,
            )
        }),
        column("URL", 26, 2, false, false, |u| {
            cell(format::truncate(&u.url, 26), Some(DIM), false)
        }),
        column("Health", 6, 0, false, false, |u| {
            if u.healthy {
                cell("✓ up", Some(OK), false)
            } else {
                cell("✗ down", Some(CRIT), false)
            }
        }),
        column("In-flight", 9, 0, true, false, |u| {
            cell(u.in_flight.to_string(), None, true)
        }),
        column("Routed", 8, 2, true, false, |u| {
            cell(format::count(u.routed_total), None, true)
        }),
        column("Errors", 7, 1, true, false, |u| {
            let color = if u.errors_total > 0 {
                Some(WARN)
            } else {
                Some(DIM)
            };
            cell(format::count(u.errors_total), color, true)
        }),
        column("TTFT p95", 9, 0, true, false, |u| {
            quantile_cell(&u.latency.ttft, 1.0, 2.0)
        }),
        column("E2E p95", 9, 1, true, false, |u| {
            quantile_cell(&u.latency.e2e, 10.0, 30.0)
        }),
        column("Models", 6, 3, true, false, |u| {
            let text = u.snapshot.as_deref().map_or_else(
                || "-".to_string(),
                |snapshot| snapshot.models.len().to_string(),
            );
            cell(text, None, true)
        }),
        column("Last error", 14, 1, false, true, |u| {
            cell(u.error.as_deref().unwrap_or(""), Some(CRIT), false)
        }),
    ]
}

/// Draws the table; the selected row follows `state`.
pub fn draw(
    frame: &mut Frame,
    area: Rect,
    upstreams: &[UpstreamStats],
    state: &mut TableState,
    focused: bool,
) {
    let healthy = upstreams.iter().filter(|upstream| upstream.healthy).count();
    let title = if focused {
        format!(
            " Upstreams ({healthy}/{} up) · ↑↓ select · Enter drill down ",
            upstreams.len()
        )
    } else {
        format!(" Upstreams ({healthy}/{} up) ", upstreams.len())
    };
    let block = panel(title, focused);
    let inner = block.inner(area);
    if upstreams.is_empty() {
        frame.render_widget(block, area);
        frame.render_widget(Paragraph::new("The balancer has no upstream."), inner);
        return;
    }
    if state.selected().is_none_or(|at| at >= upstreams.len()) {
        state.select(Some(upstreams.len() - 1));
    }
    let widget = table(
        upstreams,
        &columns(upstreams),
        inner.width.saturating_sub(2),
        block,
    );
    frame.render_stateful_widget(widget, area, state);
}
