//! Human-in-the-loop approval interceptors.
//!
//! [`HitlPlugin`] classifies streamed [`ToolCall`](crate::types::MessageContentBlock::ToolCall)
//! blocks by risk: shell/exec/bash, file-write/edit/delete, and external
//! HTTP-POST/PUT/DELETE-style tools are [`Risk::High`] and must pass through an
//! interactive approval gate before the model's call executes; read-only and
//! unrecognized tools are [`Risk::Low`] and never prompt.
//!
//! # Pause semantics and the sync channel seam
//!
//! The pipeline pauses at a gated call and waits for interactive approval, so
//! the hook blocks on the injected [`ApprovalChannel`]. The trait is
//! deliberately synchronous: [`ApprovalChannel::request_approval`] parks the
//! calling thread until an approver decides. A real deployment bridges this to
//! an async UI (e.g. a oneshot populated by a UI task); the crate ships the
//! [`OneshotApprovalChannel`] seam for exactly that programmatic/UI-driven
//! case.
//!
//! # Failure-closed default
//!
//! A channel that closes without a decision (e.g. the UI vanished) is treated
//! as a denial, so a tool never executes on a lost approval round-trip.
//!
//! # Audit trail
//!
//! Every gated decision is recorded with `action_requested`, `approver_id`,
//! and `status`, plus a wall-clock timestamp in [`HitlAuditEntry`], readable
//! via [`HitlPlugin::audit_log`].
//! Durable/persisted storage of this log belongs to the session-log plugin when
//! co-registered (out of scope here).

use std::sync::Arc;
use std::sync::Mutex;

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::types::MessageContentBlock;

/// Risk class of a tool call.
///
/// [`Risk::High`] calls pause the pipeline for interactive approval;
/// [`Risk::Low`] calls stream through untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    /// No interactive gate; the call streams through untouched.
    Low,
    /// Requires approval; the pipeline pauses until an approver decides.
    High,
}

impl Risk {
    /// A human-readable reason for the risk class, if one applies.
    ///
    /// Returns `Some` for [`Risk::High`] (a category reason: this operation is
    /// potentially destructive and needs an approver) and `None` for
    /// [`Risk::Low`] (nothing to explain: the call is safe to stream).
    pub fn reason(self) -> Option<&'static str> {
        match self {
            Risk::Low => None,
            Risk::High => {
                Some("high-risk operation: requires interactive approval before execution")
            }
        }
    }
}

/// Classify a tool call by its name.
///
/// Shell/exec/bash-style names (`shell`, `exec`, `bash`, `run_command`,
/// `terminal`), file-write/edit/delete-style names (`write`, `edit`,
/// `delete`, `remove`, `rm_`, `mv_`, `move`, `create_file`), and
/// external-API-write-style names (`http_post`, `http_put`, `http_delete`,
/// `api_write`, `post_`, `put_`, `delete_`) are [`Risk::High`]. Read-only
/// names (`read`, `search`, `get`, `list`, `web_search`, `query`) and every
/// unrecognized tool are [`Risk::Low`].
///
/// Unknown tools default to [`Risk::Low`] so the plugin never blocks an
/// unrecognized read path; this is a conservative choice tuned to not break
/// model-native read tools. The keyword table below is a simple, tunable
/// `match`-free list: extend it to tighten or relax the policy.
pub fn classify_tool_call(name: &str) -> Risk {
    let n = name.to_ascii_lowercase();
    const HIGH_KEYWORDS: [&str; 20] = [
        "shell",
        "exec",
        "bash",
        "run_command",
        "terminal",
        "write",
        "edit",
        "delete",
        "remove",
        "rm_",
        "mv_",
        "move",
        "create_file",
        "http_post",
        "http_put",
        "http_delete",
        "api_write",
        "post_",
        "put_",
        "delete_",
    ];
    if HIGH_KEYWORDS.iter().any(|k| n.contains(k)) {
        return Risk::High;
    }
    Risk::Low
}

/// A pending approval request describing one high-risk tool call.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// The id of the gated tool call (matches `ToolCall::id`).
    pub tool_call_id: String,
    /// The action category, e.g. `"shell_exec"`, `"file_write"`, or
    /// `"external_api_write"`.
    pub action: String,
    /// Human-readable detail: the tool name plus its JSON arguments truncated
    /// to ~200 characters.
    pub detail: String,
}

/// An approver's ruling on an [`ApprovalRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// The call may proceed; the block streams through unchanged.
    Approved,
    /// The call is blocked; the block is replaced by a denial `ToolResult`.
    Denied,
}

/// Seam that resolves an [`ApprovalRequest`] to an [`ApprovalDecision`].
///
/// Synchronous by design: the pipeline pauses at a gated call and waits for
/// interactive approval, so [`Self::request_approval`] blocks the calling
/// thread until an approver decides. Implementations bridge to async UI, e.g. a
/// [`OneshotApprovalChannel`] populated by a UI task. `Send + Sync` lets the
/// channel be shared across `await` points in the async client pipeline.
pub trait ApprovalChannel: Send + Sync {
    /// Block until an approver rules on `req`, returning the decision.
    ///
    /// Implementations must be failure-closed: a channel that cannot reach an
    /// approver should return [`ApprovalDecision::Denied`] so the tool never
    /// executes on a lost round-trip.
    fn request_approval(&self, req: &ApprovalRequest) -> ApprovalDecision;

    /// Identity of the approver who issued the last decision, if any.
    ///
    /// Defaults to `None`; implementations that can attribute a decision to a
    /// human/agent override this. Recorded in the audit log.
    fn approver_id(&self) -> Option<String> {
        None
    }
}

/// Programmatic [`ApprovalChannel`] backed by a oneshot.
///
/// [`OneshotApprovalChannel::new`] splits a tokio oneshot: the caller (the UI
/// or test harness) holds the returned `Sender` and answers one request by
/// sending an [`ApprovalDecision`]; the channel itself holds the `Receiver`
/// and resolves the request against it.
///
/// # Runtime caveat
///
/// [`ApprovalChannel::request_approval`] blocks on the receiver. Tokio's
/// blocking receive/`blocking_lock` panic when invoked from inside a runtime
/// worker, which is exactly the context the stream hook runs in, so this seam
/// must be driven from a dedicated OS thread (see the test). It is intended as
/// a test/UI seam, not a call-on-the-pipeline-thread channel. A lost sender
/// (dropped without sending) fails closed to [`ApprovalDecision::Denied`].
#[derive(Debug)]
pub struct OneshotApprovalChannel {
    receiver: tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<ApprovalDecision>>>,
}

impl OneshotApprovalChannel {
    /// Create a fresh (channel, sender) pair.
    ///
    /// The returned `Sender` must be held by the UI side; the channel holds
    /// the `Receiver` and resolves one request per pair.
    pub fn new() -> (Self, tokio::sync::oneshot::Sender<ApprovalDecision>) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (
            Self {
                receiver: tokio::sync::Mutex::new(Some(rx)),
            },
            tx,
        )
    }
}

impl Default for OneshotApprovalChannel {
    fn default() -> Self {
        // Construct and drop the sender: a default channel fails closed on any
        // request, which is a safe no-approver baseline.
        let (channel, _tx) = Self::new();
        channel
    }
}

impl ApprovalChannel for OneshotApprovalChannel {
    fn request_approval(&self, _req: &ApprovalRequest) -> ApprovalDecision {
        let rx = self.receiver.blocking_lock().take();
        match rx.and_then(|r| r.blocking_recv().ok()) {
            Some(decision) => decision,
            // Receiver already consumed, or sender dropped without a ruling:
            // fail closed.
            None => ApprovalDecision::Denied,
        }
    }
}

/// One audit record for a gated tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct HitlAuditEntry {
    /// The action category that was requested (`"shell_exec"`,
    /// `"file_write"`, `"external_api_write"`, ...).
    pub action_requested: String,
    /// Identity of the approver who ruled, if the channel can attribute it.
    pub approver_id: Option<String>,
    /// The ruling outcome: `"approved"` or `"denied"`.
    pub status: String,
    /// Wall-clock timestamp of the decision, in milliseconds since the Unix
    /// epoch.
    pub timestamp_ms: u64,
}

/// HITL permission plugin: gates high-risk tool calls on approval.
///
/// Holds the injected [`ApprovalChannel`] seam plus an in-memory append-only
/// audit log. All shared state is behind a `Mutex` so the stream hook can run
/// from any thread (the `CucaPlugin` supertrait requires `Send + Sync`).
///
/// # Growth
///
/// The audit log grows by one entry per gated (high-risk) decision and is
/// capped at [`Self::max_audit_entries`] ([`Self::new`] uses
/// [`Self::DEFAULT_MAX_AUDIT_ENTRIES`]; [`Self::with_max_audit_entries`]
/// validates a custom bound). At the cap the hook fails instead of evicting,
/// exactly as [`crate::plugins::session_log::InMemoryBackend`] does: this is
/// an approval audit trail, and dropping rulings would quietly erase the
/// record of who allowed what. A gated call whose ruling cannot be recorded is
/// therefore refused, matching the module's failure-closed default. Drain the
/// log through [`Self::audit_log`], or offload it to the session-log plugin,
/// before the cap. [`Self::audit_len`] is the O(1) usage gauge.
pub struct HitlPlugin {
    channel: Arc<dyn ApprovalChannel>,
    /// Append-only audit log of every gated decision, bounded by
    /// `max_audit_entries`.
    audit: Mutex<Vec<HitlAuditEntry>>,
    /// Upper bound on retained audit entries.
    max_audit_entries: usize,
}

impl HitlPlugin {
    /// Audit-log cap used by [`Self::new`].
    pub const DEFAULT_MAX_AUDIT_ENTRIES: usize = 65_536;

    /// Build the plugin around an approval channel seam, capped at
    /// [`Self::DEFAULT_MAX_AUDIT_ENTRIES`] audit entries.
    pub fn new(channel: Arc<dyn ApprovalChannel>) -> Self {
        Self {
            channel,
            audit: Mutex::new(Vec::new()),
            max_audit_entries: Self::DEFAULT_MAX_AUDIT_ENTRIES,
        }
    }

    /// Build the plugin with an explicit audit-log cap.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when `max_audit_entries` is zero.
    pub fn with_max_audit_entries(
        channel: Arc<dyn ApprovalChannel>,
        max_audit_entries: usize,
    ) -> Result<Self, PluginError> {
        if max_audit_entries == 0 {
            return Err(PluginError::Validation {
                schema: "max_audit_entries".to_string(),
                message: "max_audit_entries must be non-zero".to_string(),
            });
        }
        Ok(Self {
            channel,
            audit: Mutex::new(Vec::new()),
            max_audit_entries,
        })
    }

    /// Snapshot of the audit log, in decision order.
    pub fn audit_log(&self) -> Vec<HitlAuditEntry> {
        self.audit.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Number of retained audit entries: the O(1) usage gauge against
    /// [`Self::max_audit_entries`].
    pub fn audit_len(&self) -> usize {
        self.audit.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// The configured audit-log cap.
    pub fn max_audit_entries(&self) -> usize {
        self.max_audit_entries
    }

    /// Append an audit entry for a gated request.
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when the log is already at
    /// [`Self::max_audit_entries`] entries; see the type's *Growth* section.
    fn record_audit(&self, req: &ApprovalRequest, status: &str) -> Result<(), PluginError> {
        let mut audit = self.audit.lock().unwrap_or_else(|p| p.into_inner());
        if audit.len() >= self.max_audit_entries {
            return Err(PluginError::Internal(format!(
                "hitl audit log full: {} of {} entries stored; drain it with \
                 audit_log() or raise max_audit_entries",
                audit.len(),
                self.max_audit_entries
            )));
        }
        audit.push(HitlAuditEntry {
            action_requested: req.action.clone(),
            approver_id: self.channel.approver_id(),
            status: status.to_owned(),
            timestamp_ms: now_ms(),
        });
        Ok(())
    }
}

impl CucaPlugin for HitlPlugin {
    /// Stable plugin name.
    fn name(&self) -> &'static str {
        "hitl-approvals"
    }

    /// Gate high-risk tool calls on approval.
    ///
    /// A [`Risk::High`] [`ToolCall`](crate::types::MessageContentBlock::ToolCall)
    /// builds an [`ApprovalRequest`], blocks on the channel, and:
    /// - [`ApprovalDecision::Approved`]: the block passes through unchanged,
    ///   audited as `"approved"`;
    /// - [`ApprovalDecision::Denied`]: the block is replaced with a
    ///   [`ToolResult`](crate::types::MessageContentBlock::ToolResult) carrying
    ///   `"denied by approver"`, audited as `"denied"`.
    ///
    /// [`Risk::Low`] calls stream through without touching the channel.
    fn on_stream_chunk(
        &self,
        chunk: &mut MessageContentBlock,
    ) -> Result<(), crate::error::PluginError> {
        let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        else {
            return Ok(());
        };
        if classify_tool_call(name) == Risk::Low {
            return Ok(());
        }
        let action = classify_action(name).unwrap_or("tool_execution");
        let detail = format!("{name} {}", truncated_json(arguments));
        let req = ApprovalRequest {
            tool_call_id: id.clone(),
            action: action.to_owned(),
            detail,
        };
        match self.channel.request_approval(&req) {
            ApprovalDecision::Approved => self.record_audit(&req, "approved"),
            ApprovalDecision::Denied => {
                self.record_audit(&req, "denied")?;
                // `req` is dead once its ruling is recorded, so its owned id
                // replaces a second clone of the block's id.
                *chunk = MessageContentBlock::ToolResult {
                    tool_call_id: req.tool_call_id,
                    output: "denied by approver".into(),
                };
                Ok(())
            }
        }
    }
}

/// Map a tool name to its high-risk action category.
///
/// The category is chosen by which keyword group matched (shell first), so
/// shell/exec/bash → `"shell_exec"`, write/edit/delete/rm/mv → `"file_write"`,
/// and HTTP/API write → `"external_api_write"`. Returns `None` for a name that
/// [`classify_tool_call`] deems low-risk.
fn classify_action(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    let has = |needle: &str| n.contains(needle);
    if ["shell", "exec", "bash", "run_command", "terminal"]
        .iter()
        .any(|k| has(k))
    {
        return Some("shell_exec");
    }
    if [
        "write",
        "edit",
        "delete",
        "remove",
        "rm_",
        "mv_",
        "move",
        "create_file",
    ]
    .iter()
    .any(|k| has(k))
    {
        return Some("file_write");
    }
    if [
        "http_post",
        "http_put",
        "http_delete",
        "api_write",
        "post_",
        "put_",
        "delete_",
    ]
    .iter()
    .any(|k| has(k))
    {
        return Some("external_api_write");
    }
    None
}

/// Render tool arguments as a JSON string truncated to ~200 chars.
///
/// Truncation is character-based so it never splits a UTF-8 sequence; a
/// truncated payload is suffixed with `…`.
fn truncated_json(value: &serde_json::Value) -> String {
    const LIMIT: usize = 200;
    let s = value.to_string();
    if s.chars().count() <= LIMIT {
        return s;
    }
    let cut: String = s.chars().take(LIMIT).collect();
    format!("{cut}…")
}

/// Current wall-clock time in milliseconds since the Unix epoch.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(all(test, feature = "plugin-hitl"))]
mod tests {
    use super::*;

    /// Channel that approves every request.
    #[derive(Clone, Copy)]
    struct AutoApprove;

    impl ApprovalChannel for AutoApprove {
        fn request_approval(&self, _req: &ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Approved
        }
    }

    /// Channel that denies every request.
    #[derive(Clone, Copy)]
    struct AutoDeny;

    impl ApprovalChannel for AutoDeny {
        fn request_approval(&self, _req: &ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Denied
        }
    }

    /// Channel that panics if ever consulted (proves low-risk calls never prompt).
    struct PanicChannel;

    impl ApprovalChannel for PanicChannel {
        fn request_approval(&self, _req: &ApprovalRequest) -> ApprovalDecision {
            panic!("low-risk tool must not reach the approval channel")
        }
    }

    #[test]
    fn classify_table() {
        for high in [
            "shell",
            "exec",
            "bash",
            "run_command",
            "terminal",
            "write_file",
            "edit_file",
            "delete_file",
            "create_file",
            "http_post",
            "api_write",
        ] {
            assert_eq!(classify_tool_call(high), Risk::High, "{high}");
        }
        for low in ["read_file", "search", "web_search", "get_weather"] {
            assert_eq!(classify_tool_call(low), Risk::Low, "{low}");
        }
        assert!(Risk::High.reason().is_some());
        assert!(Risk::Low.reason().is_none());
    }

    #[test]
    fn high_risk_approved_passes_through() {
        let plugin = HitlPlugin::new(Arc::new(AutoApprove));
        let mut chunk = MessageContentBlock::ToolCall {
            id: "call_1".into(),
            name: "run_command".into(),
            arguments: serde_json::json!({ "cmd": "ls -la" }),
        };
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert!(
            matches!(&chunk, MessageContentBlock::ToolCall { name, .. } if name == "run_command"),
            "approved call must pass through unchanged: {chunk:?}"
        );
        let audit = plugin.audit_log();
        assert_eq!(audit.len(), 1);
        assert_eq!(audit[0].status, "approved");
        assert_eq!(audit[0].action_requested, "shell_exec");
    }

    #[test]
    fn high_risk_denied_becomes_tool_result() {
        let plugin = HitlPlugin::new(Arc::new(AutoDeny));
        let mut chunk = MessageContentBlock::ToolCall {
            id: "call_2".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({ "path": "/tmp/x" }),
        };
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match &chunk {
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => {
                assert_eq!(tool_call_id, "call_2");
                assert!(output.contains("denied"), "output: {output}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        let audit = plugin.audit_log();
        assert_eq!(audit.len(), 1);
        assert_eq!(audit[0].status, "denied");
        assert_eq!(audit[0].action_requested, "file_write");
    }

    #[test]
    fn low_risk_never_hits_channel() {
        let plugin = HitlPlugin::new(Arc::new(PanicChannel));
        let mut chunk = MessageContentBlock::ToolCall {
            id: "call_3".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({ "city": "Berlin" }),
        };
        // A panicking channel would abort the test if a low-risk call prompted.
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert!(
            matches!(&chunk, MessageContentBlock::ToolCall { name, .. } if name == "get_weather"),
            "low-risk call must pass through untouched: {chunk:?}"
        );
        assert!(
            plugin.audit_log().is_empty(),
            "low-risk calls are not audited"
        );
    }

    #[test]
    fn name_and_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HitlPlugin>();
        assert_send_sync::<OneshotApprovalChannel>();
        let plugin = HitlPlugin::new(Arc::new(AutoApprove));
        assert_eq!(plugin.name(), "hitl-approvals");
    }

    #[test]
    fn audit_log_cap_of_zero_is_rejected() {
        // `HitlPlugin` is not `Debug`, so the Result is matched in place rather
        // than unwrapped.
        assert!(
            matches!(
                HitlPlugin::with_max_audit_entries(Arc::new(AutoApprove), 0),
                Err(PluginError::Validation { schema, .. }) if schema == "max_audit_entries"
            ),
            "a zero audit cap must be refused"
        );
    }

    #[test]
    fn gated_call_past_the_audit_cap_fails_instead_of_dropping_the_ruling() {
        let plugin = HitlPlugin::with_max_audit_entries(Arc::new(AutoApprove), 1)
            .expect("a non-zero cap is valid");
        let call = || MessageContentBlock::ToolCall {
            id: "call_1".into(),
            name: "shell_exec".into(),
            arguments: serde_json::json!({}),
        };
        let mut first = call();
        plugin
            .on_stream_chunk(&mut first)
            .expect("the first ruling fits the cap");
        assert_eq!(plugin.audit_len(), 1);

        let mut second = call();
        let err = plugin
            .on_stream_chunk(&mut second)
            .expect_err("a ruling that cannot be audited must fail the block");
        assert!(
            matches!(&err, PluginError::Internal(message) if message.contains("audit log full")),
            "err: {err}"
        );
        assert_eq!(
            plugin.audit_len(),
            1,
            "the cap holds: no ruling is silently dropped to make room"
        );
    }

    #[tokio::test]
    async fn oneshot_channel_driven_from_thread() {
        let (channel, sender) = OneshotApprovalChannel::new();
        let plugin = HitlPlugin::new(Arc::new(channel));
        // Drive the blocking request off the tokio worker so blocking_lock/
        // blocking_recv are legal (documented caveat), then answer from here.
        let handle = std::thread::spawn(move || {
            let mut chunk = MessageContentBlock::ToolCall {
                id: "call_4".into(),
                name: "exec_command".into(),
                arguments: serde_json::json!({}),
            };
            plugin.on_stream_chunk(&mut chunk).unwrap();
            chunk
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = sender.send(ApprovalDecision::Approved);
        let chunk = handle.join().unwrap();
        assert!(
            matches!(&chunk, MessageContentBlock::ToolCall { name, .. } if name == "exec_command"),
            "approved oneshot call must pass through: {chunk:?}"
        );
    }
}
