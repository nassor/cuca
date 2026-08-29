//! Child subagent delegation and Git worktree isolation.
//!
//! [`SubagentPlugin`] turns two tool calls emitted by the parent model, `spawn_subagent` and `collect_subagent`, into asynchronous child
//! subagent runs with restricted tool scopes and, optionally, isolated Git
//! worktrees, then aggregates each child's summary back to the parent stream.
//!
//! # Delegation plumbing vs. the agent brain
//!
//! CUCA provides the *plumbing*, not a model agent: the actual child
//! execution is delegated to the caller-supplied [`SubagentRunner`] seam
//! (a real deployment wraps an external agent process/CLI; tests inject canned
//! runners). Each spawned child is executed by `runner.spawn(spec)` on a
//! background tokio task, so the plugin-level spawn is **non-blocking**: only [`SubagentPlugin::collect`] blocks, when the parent
//! asks for a child's
//! finished summary.
//!
//! # Async fan-out design
//!
//! [`SubagentPlugin::spawn_subagent`] generates a unique id, registers a
//! receiver under that id, and `tokio::spawn`s a task that awaits
//! `runner.spawn(spec)` and delivers the [`SubagentResult`] through the
//! channel. The delivery channel is **`std::sync::mpsc`** rather than a tokio
//! oneshot: [`SubagentPlugin::collect`] is synchronous
//! and `std::sync::mpsc::Receiver::recv()` blocks safely on any thread with no
//! runtime guard: tokio's `blocking_recv` instead panics when called from
//! inside a runtime, which is exactly the context `on_stream_chunk` runs in.
//! The std primitive parks only the caller's thread, leaving the spawned task
//! free to make progress on its own runtime worker.
//!
//! Pause semantics: `collect` pauses the stream pipeline until the child
//! finishes. This is the same background-oneshot pause the MCP connector
//! documents.
//! The structural caveat is identical: the blocking call must never
//! run on a thread that itself has to make progress on the blocked child's
//! runtime, which is safe here because the child runs on tokio's own worker
//! threads, independent of whatever thread calls `collect`.
//!
//! # Worktree isolation
//!
//! When [`SubagentSpec::worktree`] is set, [`SubagentPlugin::spawn_subagent`]
//! first runs `git worktree add <path> [-b <branch>]` in the current working
//! directory ([`SubagentPlugin::prepare_worktree`]); a non-git cwd or a failed
//! add surfaces as [`PluginError::NotSupported`]. The child then runs with
//! `cwd` = the worktree path (the runner reads it from `spec.worktree.path`).
//! Worktree cleanup (removal) is deliberately out of scope: the child's
//! [`SubagentResult::worktree_path`] lets the caller remove the worktree after
//! collection, and the parent (or a separate lifecycle owner) owns that policy.
//!
//! # Diagnostic metric
//!
//! The spec's diagnostic (`child subagent spawns`: `parent_session_id`,
//! `worktree_path`) is exposed through [`SubagentPlugin::spawns`], an append-only log of every spawn, alongside the scalar
//! [`SubagentPlugin::spawn_count`]. `tracing` is not a dependency of this
//! feature, so this is an accessor rather than emitted structured telemetry:
//! emitting it belongs to `plugin-telemetry` when both features are enabled.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::types::MessageContentBlock;

/// A description of one child subagent run.
///
/// This is the input contract handed to the [`SubagentRunner`] seam; it carries
/// everything the child needs to execute in isolation and be attributed to the
/// parent session.
pub struct SubagentSpec {
    /// Caller-facing name of the child (a label, not the unique spawn id).
    pub name: String,
    /// Prompt/instructions for the child agent.
    pub task: String,
    /// Restricted tool names the child may use. An empty vec means the child
    /// has an unrestricted tool scope.
    pub tool_scope: Vec<String>,
    /// Optional Git worktree isolation: when set, a worktree is prepared and
    /// the child runs with `cwd` = that worktree path.
    pub worktree: Option<WorktreeConfig>,
    /// Parent session id, recorded for the diagnostic metric.
    pub session_id: Option<String>,
}

/// Configuration for Git worktree isolation.
pub struct WorktreeConfig {
    /// Filesystem path at which the worktree is created.
    pub path: String,
    /// Branch checked out in the worktree; `None` creates a detached worktree.
    pub branch: Option<String>,
}

/// The outcome of a finished child subagent run.
#[derive(Debug, Clone, PartialEq)]
pub struct SubagentResult {
    /// The child's unique spawn id (matches the id [`SubagentPlugin::spawn_subagent`] returned).
    pub subagent_id: String,
    /// The child's aggregated summary, surfaced to the parent.
    pub summary: String,
    /// The worktree path the child ran in, when one was requested.
    pub worktree_path: Option<String>,
    /// Whether the child exited successfully.
    pub exit_ok: bool,
}

/// Seam that executes one child subagent.
///
/// Real deployments wrap an external agent process/CLI; tests inject canned
/// runners. The returned future resolves to the child's
/// [`SubagentResult`] once the run completes.
pub trait SubagentRunner: Send + Sync {
    /// Execute the child described by `spec`, resolving when the run finishes.
    fn spawn(&self, spec: SubagentSpec) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>>;
}

/// Child subagent delegation plugin.
///
/// Holds the injected [`SubagentRunner`] seam plus the async fan-out registry
/// (`pending` receivers), a spawn counter, and the spawn metric log. All shared
/// state is behind `Mutex`/atomic so the stream hook can run from any thread
/// (the `CucaPlugin` supertrait requires `Send + Sync`).
///
/// # Growth
///
/// Two structures grow with traffic, both bounded:
///
/// - The pending-result registry holds one receiver per spawned-but-uncollected
///   child, capped at [`Self::max_pending`] ([`Self::new`] uses
///   [`Self::DEFAULT_MAX_PENDING`]; [`Self::with_max_pending`] validates a
///   custom bound). At the cap [`Self::spawn_subagent`] refuses rather than
///   evicting: dropping a receiver would discard a running child's result.
///   [`Self::pending_len`] is the O(1) usage gauge; [`Self::collect`] is what
///   drains it.
/// - The diagnostic spawn log keeps the most recent
///   [`Self::MAX_SPAWN_LOG`] entries, dropping the oldest at the cap. It is a
///   rolling sample for the spawn metric, not an audit trail — the durable
///   record of a child run is the child's own [`SubagentResult`].
pub struct SubagentPlugin {
    runner: Arc<dyn SubagentRunner>,
    // `std::sync::mpsc` is used instead of tokio oneshot so the sync `collect`
    // can block on any thread without the runtime guard tokio's blocking_recv
    // imposes (see module docs).
    pending: Mutex<HashMap<String, mpsc::Receiver<SubagentResult>>>,
    /// Upper bound on outstanding uncollected children.
    max_pending: usize,
    spawn_counter: AtomicU64,
    // Rolling spawn log for the diagnostic metric, newest at the back:
    // (parent_session_id, worktree_path).
    spawn_log: Mutex<VecDeque<(Option<String>, Option<String>)>>,
}

impl SubagentPlugin {
    /// Outstanding-children cap used by [`Self::new`].
    pub const DEFAULT_MAX_PENDING: usize = 1024;

    /// Number of spawns retained in the diagnostic log.
    pub const MAX_SPAWN_LOG: usize = 4096;

    /// Build the plugin around an injected runner seam, allowing
    /// [`Self::DEFAULT_MAX_PENDING`] uncollected children.
    pub fn new(runner: Arc<dyn SubagentRunner>) -> Self {
        Self {
            runner,
            pending: Mutex::new(HashMap::new()),
            max_pending: Self::DEFAULT_MAX_PENDING,
            spawn_counter: AtomicU64::new(0),
            spawn_log: Mutex::new(VecDeque::new()),
        }
    }

    /// Build the plugin with an explicit cap on uncollected children.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when `max_pending` is zero.
    pub fn with_max_pending(
        runner: Arc<dyn SubagentRunner>,
        max_pending: usize,
    ) -> Result<Self, PluginError> {
        if max_pending == 0 {
            return Err(PluginError::Validation {
                schema: "max_pending".into(),
                message: "max_pending must be non-zero".into(),
            });
        }
        let mut plugin = Self::new(runner);
        plugin.max_pending = max_pending;
        Ok(plugin)
    }

    /// Number of spawned children not yet collected: the O(1) usage gauge
    /// against [`Self::max_pending`].
    pub fn pending_len(&self) -> usize {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// The configured cap on uncollected children.
    pub fn max_pending(&self) -> usize {
        self.max_pending
    }

    /// Spawn a child subagent and return its unique id.
    ///
    /// Validates that `spec.task` is non-empty, prepares the worktree when
    /// requested, records the spawn metric, and fires the child on a background
    /// tokio task. The plugin-level spawn is non-blocking: only
    /// [`Self::collect`] blocks, when the result is wanted. The id is
    /// `sub-<n>` where `n` is a monotonically increasing per-plugin counter
    /// (atomic only, no timestamp, so ids are deterministic and cheap).
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] for a blank `spec.task`;
    /// [`PluginError::NotSupported`] when a requested worktree cannot be
    /// prepared (see [`Self::prepare_worktree`]); [`PluginError::Internal`]
    /// when [`Self::max_pending`] children are already awaiting collection.
    pub fn spawn_subagent(&self, spec: SubagentSpec) -> Result<String, PluginError> {
        if spec.task.trim().is_empty() {
            return Err(PluginError::Validation {
                schema: "SubagentSpec.task".into(),
                message: "task must be non-empty".into(),
            });
        }
        // Prepare the worktree synchronously so the child runs in a ready cwd;
        // a non-git cwd or failed add surfaces here as NotSupported.
        let worktree_path = match &spec.worktree {
            Some(config) => Some(Self::prepare_worktree(config)?),
            None => None,
        };
        // Register the reply receiver before anything is spawned, so a full
        // registry refuses the spawn instead of leaking a running child.
        let (tx, rx) = mpsc::channel();
        let id = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            if pending.len() >= self.max_pending {
                return Err(PluginError::Internal(format!(
                    "subagent registry full: {} of {} children awaiting collection; \
                     collect() finished children or raise max_pending",
                    pending.len(),
                    self.max_pending
                )));
            }
            let id = format!("sub-{}", self.spawn_counter.fetch_add(1, Ordering::Relaxed));
            pending.insert(id.clone(), rx);
            id
        };
        {
            // Rolling window: the oldest sample leaves at the cap.
            let mut log = self.spawn_log.lock().unwrap_or_else(|p| p.into_inner());
            while log.len() >= Self::MAX_SPAWN_LOG {
                log.pop_front();
            }
            log.push_back((
                spec.session_id.clone(),
                worktree_path.map(|p| p.to_string_lossy().into_owned()),
            ));
        }
        let runner = self.runner.clone();
        tokio::spawn(async move {
            let result = runner.spawn(spec).await;
            // The receiver may have been dropped if collect was never called.
            let _ = tx.send(result);
        });
        Ok(id)
    }

    /// Collect a finished child's result, blocking until it completes.
    ///
    /// Removes the child's receiver from `pending` and waits on it. Returns
    /// [`PluginError::NotSupported`] for an unknown id, or
    /// [`PluginError::Internal`] if the background task died without delivering
    /// a result. The wait uses a std blocking `recv()`, which parks only the
    /// calling thread and carries no tokio runtime guard, so it may be called
    /// from a runtime worker such as the pipeline's stream hook.
    pub fn collect(&self, subagent_id: &str) -> Result<SubagentResult, PluginError> {
        let rx = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            pending.remove(subagent_id).ok_or_else(|| {
                PluginError::NotSupported(format!("unknown subagent id {subagent_id:?}"))
            })?
        };
        rx.recv().map_err(|_| {
            PluginError::Internal("subagent result channel closed before a result arrived".into())
        })
    }

    /// Total number of children spawned by this plugin.
    pub fn spawn_count(&self) -> u64 {
        self.spawn_counter.load(Ordering::Relaxed)
    }

    /// The spawn log for the diagnostic metric.
    ///
    /// Each entry is `(parent_session_id, worktree_path)`, one per spawn, in
    /// spawn order, for at most the most recent [`Self::MAX_SPAWN_LOG`] spawns.
    pub fn spawns(&self) -> Vec<(Option<String>, Option<String>)> {
        self.spawn_log
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Dry-run: the exact `git worktree add` argv for a config.
    ///
    /// Prepend `"git"` to get a complete command line:
    /// `["git", "worktree", "add", <path>]` for a detached worktree, or
    /// `["git", "worktree", "add", "-b", <branch>, <path>]` with a branch.
    pub fn worktree_args(config: &WorktreeConfig) -> Vec<String> {
        let mut argv = vec!["worktree".to_owned(), "add".to_owned()];
        if let Some(branch) = &config.branch {
            argv.push("-b".to_owned());
            argv.push(branch.clone());
        }
        argv.push(config.path.clone());
        argv
    }

    /// Prepare a Git worktree for a child run.
    ///
    /// Runs `git worktree add <path> [-b <branch>]` in the current working
    /// directory and returns the worktree path. A non-git cwd or a failed add
    /// surfaces as [`PluginError::NotSupported`]. The child is documented to run
    /// with `cwd` = the returned path.
    fn prepare_worktree(config: &WorktreeConfig) -> Result<PathBuf, PluginError> {
        let status = std::process::Command::new("git")
            .args(Self::worktree_args(config))
            .status()
            .map_err(|e| {
                PluginError::NotSupported(format!(
                    "cannot run `git worktree add` (is cwd inside a git repo?): {e}"
                ))
            })?;
        if !status.success() {
            return Err(PluginError::NotSupported(format!(
                "`git worktree add` failed with {status}"
            )));
        }
        Ok(PathBuf::from(&config.path))
    }
}

impl CucaPlugin for SubagentPlugin {
    /// Stable plugin name.
    fn name(&self) -> &'static str {
        "subagent-delegation"
    }

    /// Route the subagent tool calls in the stream.
    ///
    /// `spawn_subagent` (arguments: `{ name?, task, tool_scope?, worktree?,
    /// session_id? }`) builds a [`SubagentSpec`], spawns it, and replaces the
    /// block with a `ToolResult` carrying the child's id. `collect_subagent`
    /// (arguments: `{ subagent_id }`) collects the child and replaces the block
    /// with a `ToolResult` carrying the summary (or the error text on failure).
    /// Unknown tool names pass through untouched, so other plugins' and
    /// provider-native tools are unaffected.
    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        else {
            return Ok(());
        };
        match name.as_str() {
            "spawn_subagent" => {
                let spec = spec_from_args(arguments)?;
                let subagent_id = self.spawn_subagent(spec)?;
                // `id` is moved, not cloned: this assignment replaces the
                // block it was borrowed from.
                *chunk = MessageContentBlock::ToolResult {
                    tool_call_id: std::mem::take(id),
                    output: subagent_id,
                };
            }
            "collect_subagent" => {
                let subagent_id = arguments
                    .get("subagent_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                    .ok_or_else(|| PluginError::Validation {
                        schema: "collect_subagent.arguments".into(),
                        message: "missing string field `subagent_id`".into(),
                    })?;
                let output = match self.collect(&subagent_id) {
                    Ok(result) => result.summary,
                    Err(err) => err.to_string(),
                };
                *chunk = MessageContentBlock::ToolResult {
                    tool_call_id: std::mem::take(id),
                    output,
                };
            }
            _ => {}
        }
        Ok(())
    }

    fn on_request(&self, _req: &mut crate::request::UnifiedRequest) -> Result<(), PluginError> {
        Ok(())
    }

    fn on_response_complete(
        &self,
        _res: &crate::request::UnifiedResponse,
    ) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Build a [`SubagentSpec`] from `spawn_subagent` tool arguments.
///
/// Missing or non-object arguments, and a missing/empty `task`, yield
/// [`PluginError::Validation`]. `tool_scope` defaults to empty (unrestricted);
/// `worktree` defaults to `None` and, when present, must be an object with a
/// string `path` and optional string `branch`.
fn spec_from_args(arguments: &serde_json::Value) -> Result<SubagentSpec, PluginError> {
    let obj = arguments
        .as_object()
        .ok_or_else(|| PluginError::Validation {
            schema: "spawn_subagent.arguments".into(),
            message: "arguments must be a JSON object".into(),
        })?;
    let task = obj
        .get("task")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| PluginError::Validation {
            schema: "spawn_subagent.arguments.task".into(),
            message: "missing string field `task`".into(),
        })?;
    if task.trim().is_empty() {
        return Err(PluginError::Validation {
            schema: "spawn_subagent.arguments.task".into(),
            message: "`task` must be non-empty".into(),
        });
    }
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .unwrap_or_default();
    // Default to an empty scope (unrestricted); ignore non-string entries.
    let tool_scope = obj
        .get("tool_scope")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let worktree = match obj.get("worktree") {
        None | Some(serde_json::Value::Null) => None,
        Some(w) => {
            let wobj = w.as_object().ok_or_else(|| PluginError::Validation {
                schema: "spawn_subagent.arguments.worktree".into(),
                message: "`worktree` must be an object".into(),
            })?;
            let path = wobj
                .get("path")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| PluginError::Validation {
                    schema: "spawn_subagent.arguments.worktree.path".into(),
                    message: "missing string field `path`".into(),
                })?;
            let branch = wobj
                .get("branch")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            Some(WorktreeConfig { path, branch })
        }
    };
    let session_id = obj
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    Ok(SubagentSpec {
        name,
        task,
        tool_scope,
        worktree,
        session_id,
    })
}

#[cfg(all(test, feature = "plugin-subagent"))]
mod tests {
    use super::*;
    use serde_json::json;

    /// A recorded fake runner invocation: (task, tool_scope).
    type FakeCall = (String, Vec<String>);

    /// Canned runner: records every (task, tool_scope) it is given and resolves
    /// each spawn to a fixed, configurable result.
    #[derive(Clone)]
    struct FakeRunner {
        calls: Arc<Mutex<Vec<FakeCall>>>,
        summary: String,
        exit_ok: bool,
        worktree_path: Option<String>,
    }

    impl FakeRunner {
        fn new(summary: &str, exit_ok: bool) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                summary: summary.into(),
                exit_ok,
                worktree_path: None,
            }
        }
    }

    impl SubagentRunner for FakeRunner {
        fn spawn(
            &self,
            spec: SubagentSpec,
        ) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>> {
            self.calls
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((spec.task.clone(), spec.tool_scope.clone()));
            let summary = self.summary.clone();
            let exit_ok = self.exit_ok;
            let worktree_path = self.worktree_path.clone();
            Box::pin(async move {
                SubagentResult {
                    subagent_id: "fake-id".into(),
                    summary,
                    worktree_path,
                    exit_ok,
                }
            })
        }
    }

    fn spec(task: &str, scope: Vec<String>) -> SubagentSpec {
        SubagentSpec {
            name: "child".into(),
            task: task.into(),
            tool_scope: scope,
            worktree: None,
            session_id: Some("sess-1".into()),
        }
    }

    #[test]
    fn worktree_args_with_branch() {
        let cfg = WorktreeConfig {
            path: "/tmp/wt".into(),
            branch: Some("feat".into()),
        };
        assert_eq!(
            SubagentPlugin::worktree_args(&cfg),
            vec!["worktree", "add", "-b", "feat", "/tmp/wt"]
        );
    }

    #[test]
    fn worktree_args_detached() {
        let cfg = WorktreeConfig {
            path: "/tmp/wt".into(),
            branch: None,
        };
        assert_eq!(
            SubagentPlugin::worktree_args(&cfg),
            vec!["worktree", "add", "/tmp/wt"]
        );
    }

    #[test]
    fn empty_task_rejected() {
        let runner = FakeRunner::new("done", true);
        let plugin = SubagentPlugin::new(Arc::new(runner));
        match plugin.spawn_subagent(spec("   ", vec![])) {
            Err(PluginError::Validation { .. }) => {}
            other => panic!("expected Validation, got {other:?}"),
        }
        assert_eq!(plugin.spawn_count(), 0);
    }

    #[tokio::test]
    async fn empty_tool_scope_allowed() {
        let runner = FakeRunner::new("done", true);
        let plugin = Arc::new(SubagentPlugin::new(Arc::new(runner)));
        // Empty scope means unrestricted; spawn must be accepted. `collect`
        // blocks on a std mpsc receiver while the result task runs on the
        // tokio runtime, so the interaction runs on the blocking pool, on a current-thread runtime (this crate never enables
        // `rt-multi-thread`) the runtime worker must stay free to poll the
        // spawned task.
        let res = tokio::task::spawn_blocking(move || {
            let id = plugin
                .spawn_subagent(spec("do work", vec![]))
                .expect("empty tool scope must be allowed");
            plugin.collect(&id).expect("collect must succeed")
        })
        .await
        .expect("spawn_blocking must complete");
        assert!(res.exit_ok);
    }

    #[tokio::test]
    async fn spawn_ids_unique_and_count() {
        let runner = FakeRunner::new("done", true);
        let plugin = SubagentPlugin::new(Arc::new(runner));
        let a = plugin.spawn_subagent(spec("a", vec![])).unwrap();
        let b = plugin.spawn_subagent(spec("b", vec![])).unwrap();
        assert_ne!(a, b);
        assert_eq!(plugin.spawn_count(), 2);
    }

    #[test]
    fn pending_cap_of_zero_is_rejected() {
        let runner = FakeRunner::new("done", true);
        // `SubagentPlugin` is not `Debug`, so the Result is matched in place
        // rather than unwrapped.
        assert!(
            matches!(
                SubagentPlugin::with_max_pending(Arc::new(runner), 0),
                Err(PluginError::Validation { schema, .. }) if schema == "max_pending"
            ),
            "a zero pending cap must be refused"
        );
    }

    #[tokio::test]
    async fn spawn_past_the_pending_cap_is_refused() {
        let runner = FakeRunner::new("done", true);
        let plugin =
            SubagentPlugin::with_max_pending(Arc::new(runner), 1).expect("a non-zero cap is valid");
        plugin
            .spawn_subagent(spec("first", vec![]))
            .expect("the first child fits the cap");
        let err = plugin
            .spawn_subagent(spec("second", vec![]))
            .expect_err("a spawn past the cap must be refused");
        assert!(
            matches!(&err, PluginError::Internal(message) if message.contains("registry full")),
            "err: {err}"
        );
        assert_eq!(
            plugin.pending_len(),
            1,
            "the refused spawn leaves the registry untouched"
        );
        assert_eq!(
            plugin.spawn_count(),
            1,
            "a refused spawn never consumes an id"
        );
    }

    #[tokio::test]
    async fn spawn_collect_roundtrip() {
        let runner = FakeRunner::new("summarized!", true);
        let plugin = Arc::new(SubagentPlugin::new(Arc::new(runner.clone())));
        // `collect` blocks on a std mpsc receiver while the spawned task runs
        // on the tokio runtime; see `empty_tool_scope_allowed`.
        let (res, calls, spawns) = tokio::task::spawn_blocking(move || {
            let id = plugin
                .spawn_subagent(spec("summarize docs", vec!["read".into()]))
                .unwrap();
            let res = plugin.collect(&id).unwrap();

            // The fake recorded the task + scope it was handed.
            let calls = runner
                .calls
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();

            // The spawn metric logged (session_id, worktree_path).
            let spawns = plugin.spawns();
            (res, calls, spawns)
        })
        .await
        .expect("spawn_blocking must complete");
        assert_eq!(res.summary, "summarized!");
        assert!(res.exit_ok);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "summarize docs");
        assert_eq!(calls[0].1, vec!["read".to_owned()]);
        assert_eq!(spawns.len(), 1);
        assert_eq!(spawns[0].0.as_deref(), Some("sess-1"));
        assert_eq!(spawns[0].1, None);
    }

    #[tokio::test]
    async fn on_stream_chunk_routing() {
        let runner = FakeRunner::new("the summary", true);
        let plugin = Arc::new(SubagentPlugin::new(Arc::new(runner)));
        // The collect_subagent hook blocks on `collect`'s std mpsc receiver
        // while the spawned task runs on the tokio runtime; see
        // `empty_tool_scope_allowed`.
        let (spawned_id, collect_output, foreign_name) = tokio::task::spawn_blocking(move || {
            // spawn_subagent ToolCall -> ToolResult carrying the child id.
            let mut spawn = MessageContentBlock::ToolCall {
                id: "call_1".into(),
                name: "spawn_subagent".into(),
                arguments: json!({ "name": "child-a", "task": "list files", "tool_scope": ["read"] }),
            };
            plugin.on_stream_chunk(&mut spawn).unwrap();
            let spawned_id = match spawn {
                MessageContentBlock::ToolResult {
                    tool_call_id,
                    output,
                } => {
                    assert_eq!(tool_call_id, "call_1");
                    assert!(output.starts_with("sub-"), "id = {output}");
                    output
                }
                other => panic!("expected ToolResult, got {other:?}"),
            };

            // collect_subagent ToolCall -> ToolResult carrying the summary.
            let mut collect = MessageContentBlock::ToolCall {
                id: "call_2".into(),
                name: "collect_subagent".into(),
                arguments: json!({ "subagent_id": spawned_id }),
            };
            plugin.on_stream_chunk(&mut collect).unwrap();
            let collect_output = match collect {
                MessageContentBlock::ToolResult {
                    tool_call_id,
                    output,
                } => {
                    assert_eq!(tool_call_id, "call_2");
                    output
                }
                other => panic!("expected ToolResult, got {other:?}"),
            };

            // Unknown tool passes through untouched.
            let mut foreign = MessageContentBlock::ToolCall {
                id: "call_3".into(),
                name: "unrelated".into(),
                arguments: json!({}),
            };
            plugin.on_stream_chunk(&mut foreign).unwrap();
            let foreign_name = match foreign {
                MessageContentBlock::ToolCall { name, .. } => name,
                other => panic!("expected ToolCall passthrough, got {other:?}"),
            };
            (spawned_id, collect_output, foreign_name)
        })
        .await
        .expect("spawn_blocking must complete");
        assert!(spawned_id.starts_with("sub-"), "id = {spawned_id}");
        assert_eq!(collect_output, "the summary");
        assert_eq!(foreign_name, "unrelated");
    }
    #[tokio::test]
    async fn collect_error_rendered_as_tool_result() {
        let runner = FakeRunner::new("x", true);
        let plugin = SubagentPlugin::new(Arc::new(runner));
        let mut call = MessageContentBlock::ToolCall {
            id: "call_1".into(),
            name: "collect_subagent".into(),
            arguments: json!({ "subagent_id": "sub-999" }),
        };
        // Unknown id must not blow up the hook; the error text becomes the output.
        plugin.on_stream_chunk(&mut call).unwrap();
        match call {
            MessageContentBlock::ToolResult { output, .. } => {
                assert!(output.contains("unknown subagent id"), "output = {output}")
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn name_is_subagent_delegation() {
        let runner = FakeRunner::new("x", true);
        let plugin = SubagentPlugin::new(Arc::new(runner));
        assert_eq!(plugin.name(), "subagent-delegation");
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn plugin_is_send_sync() {
        assert_send_sync::<SubagentPlugin>();
    }
}
