//! Application state and key handling.
//!
//! [`Data`] is what the stream delivered (snapshots, histories, connection);
//! [`UiState`] is what the user chose (view, pane, selection, popup). The
//! renderer reads `Data` and mutates `UiState` (table scroll offsets), so the
//! two live in separate fields and borrow independently.

use std::collections::HashMap;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::widgets::TableState;

use super::client::{Connection, Update};
use super::history::History;
use super::rollup::fleet;
use crate::server::stats::{NodeRole, StatsSnapshot};

/// A region of the dashboard that can hold the focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    /// CPU, RAM, disks, GPUs, process.
    Resources,
    /// Latency table and the ranked reasons.
    Latency,
    /// The models table (server) or the upstreams table (balancer).
    List,
}

/// Which node is on screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum View {
    /// The node the URL points at.
    Root,
    /// One upstream of a balancer, drilled into by its id.
    Upstream(String),
}

/// An overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Popup {
    /// Nothing on top.
    None,
    /// The key help.
    Help,
    /// Everything about one model, by name.
    Model(String),
}

/// What the stream delivered.
#[derive(Debug)]
pub struct Data {
    /// The base URL being watched.
    pub url: String,
    /// The refresh interval asked of the server.
    pub interval_ms: u64,
    /// State of the stream.
    pub connection: Connection,
    /// The latest snapshot of the node at the URL.
    pub snapshot: Option<StatsSnapshot>,
    /// For a balancer: its snapshot with the upstreams' machines rolled up.
    pub fleet: Option<StatsSnapshot>,
    /// History of the node at the URL (the rolled-up fleet for a balancer).
    pub history: History,
    /// History of each upstream that reported a snapshot, by id.
    pub upstream_history: HashMap<String, History>,
    /// Whether new snapshots are being ignored.
    pub paused: bool,
    /// When the latest snapshot arrived.
    pub last_update: Option<Instant>,
}

/// What the user chose.
#[derive(Debug)]
pub struct UiState {
    /// The node on screen.
    pub view: View,
    /// The focused pane.
    pub focus: Pane,
    /// The overlay.
    pub popup: Popup,
    /// Scroll and selection of the models table.
    pub model_table: TableState,
    /// Scroll and selection of the upstreams table.
    pub upstream_table: TableState,
    /// First visible line of the reasons list.
    pub diag_scroll: usize,
    /// The terminal was too small for every pane at once on the last draw, so
    /// only the focused pane shows.
    pub compact: bool,
    /// Set when the user asked to leave.
    pub quit: bool,
}

/// The whole state.
#[derive(Debug)]
pub struct App {
    /// Stream data.
    pub data: Data,
    /// User choices.
    pub ui: UiState,
}

impl App {
    /// A state showing "connecting" for `url`.
    pub fn new(url: &str, interval_ms: u64) -> Self {
        Self {
            data: Data {
                url: url.trim_end_matches('/').to_string(),
                interval_ms,
                connection: Connection::Connecting,
                snapshot: None,
                fleet: None,
                history: History::new(),
                upstream_history: HashMap::new(),
                paused: false,
                last_update: None,
            },
            ui: UiState {
                view: View::Root,
                focus: Pane::Resources,
                popup: Popup::None,
                model_table: TableState::default().with_selected(0),
                upstream_table: TableState::default().with_selected(0),
                diag_scroll: 0,
                compact: false,
                quit: false,
            },
        }
    }

    /// Applies one update from the network task.
    pub fn apply(&mut self, update: Update) {
        match update {
            Update::Connection(connection) => self.data.connection = connection,
            Update::Snapshot(snapshot) => {
                if !self.data.paused {
                    self.data.accept(*snapshot);
                }
            }
        }
    }

    /// Handles one key press.
    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.ui.quit = true;
            return;
        }
        if self.ui.popup != Popup::None {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Char('q' | '?') | KeyCode::Enter
            ) {
                self.ui.popup = Popup::None;
            }
            return;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.back_or_quit(),
            KeyCode::Tab => self.cycle_focus(1),
            KeyCode::BackTab => self.cycle_focus(2),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Enter => self.enter(),
            KeyCode::Char('p') => self.data.paused = !self.data.paused,
            KeyCode::Char('?' | 'h') | KeyCode::F(1) => self.ui.popup = Popup::Help,
            _ => {}
        }
    }

    /// The snapshot on screen: the root node, or the drilled upstream's.
    pub fn shown(&self) -> Option<&StatsSnapshot> {
        match &self.ui.view {
            View::Root => self.data.snapshot.as_ref(),
            View::Upstream(id) => self.data.upstream(id).and_then(|up| up.snapshot.as_deref()),
        }
    }

    /// Whether the root node is a balancer.
    pub fn is_balancer(&self) -> bool {
        self.data.is_balancer()
    }

    /// The panes of the view on screen, in Tab order.
    pub const PANES: [Pane; 3] = [Pane::Resources, Pane::Latency, Pane::List];

    fn back_or_quit(&mut self) {
        if self.ui.view == View::Root {
            self.ui.quit = true;
        } else {
            self.ui.view = View::Root;
            self.ui.focus = Pane::Resources;
            self.ui.diag_scroll = 0;
            self.ui.model_table.select(Some(0));
        }
    }

    fn cycle_focus(&mut self, step: usize) {
        let at = Self::PANES
            .iter()
            .position(|pane| *pane == self.ui.focus)
            .unwrap_or(0);
        self.ui.focus = Self::PANES[(at + step) % Self::PANES.len()];
    }

    /// Whether Up, Down and Enter act on the list (models or upstreams): the
    /// list pane has the focus, or every pane is on screen and the focus is
    /// on one that has nothing to select.
    fn list_active(&self) -> bool {
        match self.ui.focus {
            Pane::List => true,
            Pane::Resources => !self.ui.compact,
            Pane::Latency => false,
        }
    }

    fn list_len(&self) -> usize {
        match (&self.ui.view, &self.data.snapshot) {
            (View::Root, Some(root)) if root.node.role == NodeRole::Balancer => {
                root.upstreams.len()
            }
            _ => self.shown().map_or(0, |snapshot| snapshot.models.len()),
        }
    }

    fn list_state(&mut self) -> &mut TableState {
        if self.ui.view == View::Root && self.data.is_balancer() {
            &mut self.ui.upstream_table
        } else {
            &mut self.ui.model_table
        }
    }

    fn move_selection(&mut self, delta: isize) {
        if self.ui.focus == Pane::Latency {
            self.ui.diag_scroll = self.ui.diag_scroll.saturating_add_signed(delta);
            return;
        }
        if !self.list_active() {
            return;
        }
        let len = self.list_len();
        if len == 0 {
            return;
        }
        let state = self.list_state();
        let at = state.selected().unwrap_or(0).min(len - 1);
        state.select(Some(at.saturating_add_signed(delta).min(len - 1)));
    }

    fn enter(&mut self) {
        if !self.list_active() {
            return;
        }
        let len = self.list_len();
        let selected = self.list_state().selected().unwrap_or(0);
        if selected >= len {
            return;
        }
        if self.ui.view == View::Root && self.data.is_balancer() {
            let Some(upstream) = self
                .data
                .snapshot
                .as_ref()
                .and_then(|root| root.upstreams.get(selected))
            else {
                return;
            };
            if upstream.snapshot.is_some() {
                self.ui.view = View::Upstream(upstream.id.clone());
                self.ui.focus = Pane::Resources;
                self.ui.diag_scroll = 0;
                self.ui.model_table.select(Some(0));
            }
        } else if let Some(model) = self
            .shown()
            .and_then(|snapshot| snapshot.models.get(selected))
        {
            self.ui.popup = Popup::Model(model.name.clone());
        }
    }
}

impl Data {
    /// Whether the root node is a balancer.
    pub fn is_balancer(&self) -> bool {
        self.snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.node.role == NodeRole::Balancer)
    }

    /// An upstream of the root balancer by id.
    pub fn upstream(&self, id: &str) -> Option<&crate::server::stats::UpstreamStats> {
        self.snapshot
            .as_ref()?
            .upstreams
            .iter()
            .find(|upstream| upstream.id == id)
    }

    fn accept(&mut self, snapshot: StatsSnapshot) {
        self.last_update = Some(Instant::now());
        if snapshot.node.role == NodeRole::Balancer {
            let rolled = fleet(&snapshot);
            self.history.push(&rolled);
            for upstream in &snapshot.upstreams {
                if let Some(nested) = upstream.snapshot.as_deref() {
                    self.upstream_history
                        .entry(upstream.id.clone())
                        .or_default()
                        .push(nested);
                }
            }
            // Upstreams that left the balancer's list free their history.
            self.upstream_history
                .retain(|id, _| snapshot.upstreams.iter().any(|upstream| &upstream.id == id));
            self.fleet = Some(rolled);
        } else {
            self.history.push(&snapshot);
            self.fleet = None;
            self.upstream_history.clear();
        }
        self.snapshot = Some(snapshot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tui::tests::{balancer_snapshot, press, server_snapshot};

    fn app_with(snapshot: StatsSnapshot) -> App {
        let mut app = App::new("http://h:1/", 1000);
        app.apply(Update::Snapshot(Box::new(snapshot)));
        app
    }

    #[test]
    fn q_quits_from_the_root_and_esc_leaves_a_drilled_upstream_first() {
        let mut app = app_with(balancer_snapshot());
        app.on_key(press(KeyCode::Tab));
        app.on_key(press(KeyCode::Tab));
        assert_eq!(app.ui.focus, Pane::List);
        app.on_key(press(KeyCode::Enter));
        assert!(matches!(app.ui.view, View::Upstream(_)), "drilled in");
        assert!(app.shown().is_some());
        app.on_key(press(KeyCode::Esc));
        assert_eq!(app.ui.view, View::Root);
        assert!(!app.ui.quit);
        app.on_key(press(KeyCode::Char('q')));
        assert!(app.ui.quit);
    }

    #[test]
    fn an_upstream_without_a_snapshot_cannot_be_drilled_into() {
        let mut snapshot = balancer_snapshot();
        snapshot.upstreams[1].snapshot = None;
        let mut app = app_with(snapshot);
        app.ui.focus = Pane::List;
        app.on_key(press(KeyCode::Down));
        app.on_key(press(KeyCode::Enter));
        assert_eq!(app.ui.view, View::Root);
    }

    #[test]
    fn enter_opens_the_selected_model_and_selection_stops_at_the_ends() {
        let mut app = app_with(server_snapshot());
        app.ui.focus = Pane::List;
        for _ in 0..5 {
            app.on_key(press(KeyCode::Down));
        }
        assert_eq!(app.ui.model_table.selected(), Some(1));
        app.on_key(press(KeyCode::Enter));
        assert_eq!(app.ui.popup, Popup::Model("qwen3-0.6b-q8_0".into()));
        app.on_key(press(KeyCode::Esc));
        assert_eq!(app.ui.popup, Popup::None);
    }

    #[test]
    fn pause_ignores_snapshots_but_not_connection_changes() {
        let mut app = app_with(server_snapshot());
        app.on_key(press(KeyCode::Char('p')));
        let before = app.data.history.len();
        app.apply(Update::Snapshot(Box::new(server_snapshot())));
        assert_eq!(app.data.history.len(), before);
        app.apply(Update::Connection(Connection::Connected));
        assert_eq!(app.data.connection, Connection::Connected);
    }
}
