//! The "Why is it slow?" panel: the node's diagnoses, most severe first.

use std::cmp::Reverse;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::widgets::{CRIT, DIM, INFO, OK, WARN, panel};
use crate::server::stats::{Cause, Diagnosis, Severity};
use crate::server::tui::format;

/// The color and badge of a severity.
pub fn severity_style(severity: Severity) -> (Color, &'static str) {
    match severity {
        Severity::Critical => (CRIT, " CRIT "),
        Severity::Warn => (WARN, " WARN "),
        Severity::Info => (INFO, " INFO "),
    }
}

/// The diagnoses sorted most severe first (stable within a severity).
pub fn ranked(diagnosis: &[Diagnosis]) -> Vec<&Diagnosis> {
    let mut ranked: Vec<&Diagnosis> = diagnosis.iter().collect();
    ranked.sort_by_key(|entry| Reverse(entry.severity));
    ranked
}

/// A cause as words: `QueueSaturated` is `queue saturated`.
pub fn cause_words(cause: Cause) -> String {
    let name = format!("{cause:?}");
    let mut words = String::with_capacity(name.len() + 4);
    for (i, letter) in name.chars().enumerate() {
        if letter.is_uppercase() {
            if i > 0 {
                words.push(' ');
            }
            words.extend(letter.to_lowercase());
        } else {
            words.push(letter);
        }
    }
    words
}

/// The lines of one diagnosis: the sentence, then its evidence.
fn lines(entry: &Diagnosis) -> [Line<'static>; 2] {
    let (color, badge) = severity_style(entry.severity);
    let mut head = vec![
        Span::styled(
            badge,
            Style::new()
                .fg(Color::Black)
                .bg(color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ];
    // A balancer already prefixes the message of an upstream's diagnosis with `[id]`.
    let upstream = entry
        .upstream
        .as_deref()
        .filter(|id| !entry.message.starts_with(&format!("[{id}]")));
    let scope: Vec<&str> = [upstream, entry.model.as_deref()]
        .into_iter()
        .flatten()
        .collect();
    if !scope.is_empty() {
        head.push(Span::styled(
            format!("[{}] ", scope.join("/")),
            Style::new().fg(color),
        ));
    }
    let message = if entry.message.is_empty() {
        cause_words(entry.cause)
    } else {
        entry.message.clone()
    };
    head.push(Span::styled(
        message,
        Style::new().add_modifier(Modifier::BOLD),
    ));

    let mut evidence = vec![Span::styled(
        format!("       {} ", cause_words(entry.cause)),
        Style::new().fg(color),
    )];
    for (key, value) in &entry.evidence {
        evidence.push(Span::styled(format!("{key}="), Style::new().fg(DIM)));
        evidence.push(Span::raw(format!("{} ", format::number(*value))));
    }
    [Line::from(head), Line::from(evidence)]
}

/// Draws the panel; `scroll` is the first visible line.
pub fn draw(frame: &mut Frame, area: Rect, diagnosis: &[Diagnosis], scroll: usize, focused: bool) {
    let ranked = ranked(diagnosis);
    let (title_color, count) = match ranked.first() {
        Some(worst) => (
            severity_style(worst.severity).0,
            format!(" ({})", ranked.len()),
        ),
        None => (OK, String::new()),
    };
    let title = Line::from(vec![
        Span::raw(" Why is it slow?"),
        Span::styled(count, Style::new().fg(title_color)),
        Span::raw(" "),
    ]);
    let block = panel(title, focused);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }

    if ranked.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "✓ No latency issues detected",
                Style::new().fg(OK).add_modifier(Modifier::BOLD),
            )),
            inner,
        );
        return;
    }
    let text: Vec<Line<'static>> = ranked.iter().flat_map(|entry| lines(entry)).collect();
    let scroll = scroll.min(text.len().saturating_sub(1));
    frame.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .scroll((scroll as u16, 0)),
        inner,
    );
}

/// The one-line summary for a compact screen: the worst diagnosis or the
/// all-clear.
pub fn summary(diagnosis: &[Diagnosis]) -> Line<'static> {
    match ranked(diagnosis).first() {
        Some(worst) => {
            let (color, badge) = severity_style(worst.severity);
            let mut spans = vec![Span::styled(
                badge,
                Style::new()
                    .fg(Color::Black)
                    .bg(color)
                    .add_modifier(Modifier::BOLD),
            )];
            let message = if worst.message.is_empty() {
                cause_words(worst.cause)
            } else {
                worst.message.clone()
            };
            spans.push(Span::raw(format!(" {message}")));
            if diagnosis.len() > 1 {
                spans.push(Span::styled(
                    format!("  (+{} more)", diagnosis.len() - 1),
                    Style::new().fg(DIM),
                ));
            }
            Line::from(spans)
        }
        None => Line::from(Span::styled(
            "✓ No latency issues detected",
            Style::new().fg(OK),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causes_read_as_words() {
        assert_eq!(cause_words(Cause::QueueSaturated), "queue saturated");
        assert_eq!(cause_words(Cause::Other), "other");
    }

    #[test]
    fn the_most_severe_diagnosis_ranks_first() {
        let make = |severity| Diagnosis {
            severity,
            ..Diagnosis::default()
        };
        let all = [
            make(Severity::Info),
            make(Severity::Critical),
            make(Severity::Warn),
        ];
        let order: Vec<Severity> = ranked(&all).iter().map(|entry| entry.severity).collect();
        assert_eq!(order, [Severity::Critical, Severity::Warn, Severity::Info]);
    }
}
