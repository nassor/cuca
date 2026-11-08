//! `cuca tui`: a real-time terminal view of a model server or a load balancer.
//!
//! The UI follows the node's stats stream (`GET /cuca/v1/stats/stream`, which
//! the server serves only with `--stats-api`) and shows resources (CPU, RAM,
//! disks, GPUs, VRAM), the latency over the sliding window and the ranked
//! reasons for it, then the models of a server or the upstreams of a
//! balancer (Enter drills into an upstream's own view).
//!
//! Layout of the module: [`client`] reads the stream on a tokio task and
//! reconnects with a capped backoff; [`app`] holds the state and the keys;
//! [`history`] keeps the last [`history::HISTORY_CAP`] samples for sparklines;
//! [`ui`] draws; [`mod@format`] holds the human number formatting. The draw loop
//! runs on a blocking thread and never waits for the network: it drains the
//! channel, redraws on new data, on input and about every 250 ms.
//!
//! The terminal is restored on exit, on error and on panic (ratatui's
//! `init` installs the panic hook).

pub mod app;
pub mod client;
pub mod format;
pub mod history;
pub mod rollup;
pub mod ui;

#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

use crossterm::event::{self, Event};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;

use self::app::App;
use self::client::Update;
use crate::server::error::ServerError;

/// How long one wait for input lasts; also the latency of noticing new data.
const TICK: Duration = Duration::from_millis(50);
/// Longest time between two redraws, so countdowns and ages keep moving.
const REDRAW_EVERY: Duration = Duration::from_millis(250);

/// What `cuca tui` was asked to watch.
#[derive(Debug, Clone)]
pub struct TuiOptions {
    /// Base URL of the server or balancer (`http://127.0.0.1:8000`).
    pub url: String,
    /// Bearer token the stats API wants, if the node has an API key.
    pub api_key: Option<String>,
    /// Interval between snapshots asked of the server, milliseconds (at least 100).
    pub interval_ms: u64,
}

/// Runs the terminal UI until the user quits.
///
/// # Errors
///
/// [`ServerError::Config`] for a bad URL or interval, [`ServerError::Io`]
/// when the terminal cannot be set up (standard output is not a terminal) or
/// breaks while drawing.
pub async fn run(options: TuiOptions) -> Result<(), ServerError> {
    let (updates, network) = client::spawn(&options)?;
    let app = App::new(&options.url, options.interval_ms);
    let result = tokio::task::spawn_blocking(move || event_loop(app, updates)).await;
    network.abort();
    match result {
        Ok(result) => result,
        Err(error) => Err(ServerError::Io(format!("terminal UI stopped: {error}"))),
    }
}

/// Owns the terminal for the life of the UI.
fn event_loop(mut app: App, mut updates: mpsc::Receiver<Update>) -> Result<(), ServerError> {
    let mut terminal =
        ratatui::try_init().map_err(|error| ServerError::Io(format!("terminal: {error}")))?;
    let result = drive(&mut terminal, &mut app, &mut updates);
    let restored =
        ratatui::try_restore().map_err(|error| ServerError::Io(format!("terminal: {error}")));
    result.and(restored)
}

fn drive(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    updates: &mut mpsc::Receiver<Update>,
) -> Result<(), ServerError> {
    let io = |error: std::io::Error| ServerError::Io(format!("terminal: {error}"));
    let mut dirty = true;
    let mut drawn = Instant::now();
    while !app.ui.quit {
        while let Ok(update) = updates.try_recv() {
            app.apply(update);
            dirty = true;
        }
        if dirty || drawn.elapsed() >= REDRAW_EVERY {
            terminal.draw(|frame| ui::draw(frame, app)).map_err(io)?;
            dirty = false;
            drawn = Instant::now();
        }
        if event::poll(TICK).map_err(io)? {
            match event::read().map_err(io)? {
                Event::Key(key) => app.on_key(key),
                Event::Resize(..) => {}
                _ => continue,
            }
            dirty = true;
        }
    }
    Ok(())
}
