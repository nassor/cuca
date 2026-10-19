+++
title = "Human approval"
description = "The human-in-the-loop approval plugin: risk classification, the failure-closed default, and the audit log cap."
template = "page.html"
weight = 7
+++

# Human approval

<dl class="page-facts">
<dt>In one line</dt>
<dd>Classifies streamed tool calls by risk and pauses on high-risk calls for an interactive approval decision.</dd>
<dt>You need</dt>
<dd>The <code>plugin-hitl</code> feature and a caller-supplied <code>ApprovalChannel</code>.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>HitlPlugin</code> or implementing an <code>ApprovalChannel</code>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-hitl` (enables `dep:tokio`) |
| Entry type | `HitlPlugin` |
| Supporting types | `ApprovalChannel`, `ApprovalDecision`, `ApprovalRequest`, `OneshotApprovalChannel`, `Risk` |
| Also in `cuca::plugins::hitl` | `HitlAuditEntry`, `classify_tool_call` |
| Plugin name | `"hitl-approvals"` |
| Attachment | `register_plugin` |
| Peer features | None |
| Needs | An `ApprovalChannel` implementation |

## Hooks

`HitlPlugin` overrides `on_stream_chunk` only. Blocks other than `ToolCall` pass through untouched.

`classify_tool_call(name)` lowercases the tool name and checks it for each keyword as a substring. Three groups are `Risk::High`, and each group sets the `ApprovalRequest::action` string:

- Shell and exec, action `shell_exec`: `shell`, `exec`, `bash`, `run_command`, `terminal`.
- File write, action `file_write`: `write`, `edit`, `delete`, `remove`, `rm_`, `mv_`, `move`, `create_file`.
- External API write, action `external_api_write`: `http_post`, `http_put`, `http_delete`, `api_write`, `post_`, `put_`, `delete_`.

Every other name, including an unrecognized one, is `Risk::Low`. The shell group is checked first.

- A `Risk::Low` call streams through without touching the channel and without an audit entry.
- A `Risk::High` call builds an `ApprovalRequest` and blocks on `ApprovalChannel::request_approval`. The request carries `tool_call_id`, `action`, and `detail`, which is the tool name plus its JSON arguments cut to 200 characters with a trailing ellipsis.
- `ApprovalDecision::Approved` streams the call through unchanged and records `"approved"`.
- `ApprovalDecision::Denied` replaces the block with a `ToolResult` whose output is `"denied by approver"` and records `"denied"`. The application executes nothing.

An approved call is not executed by the plugin. The application runs it.

`ApprovalChannel` is synchronous: `request_approval` parks the calling thread until an approver rules. Its optional `approver_id()` defaults to `None` and fills the audit entry.

A channel that cannot reach an approver must return `ApprovalDecision::Denied`. A tool then never runs on a lost approval round trip.

`OneshotApprovalChannel::new()` returns the channel plus a Tokio oneshot `Sender<ApprovalDecision>` for the UI side, and resolves one request per pair. A dropped sender, or a second request, resolves to `Denied`. `OneshotApprovalChannel::default()` has no sender and denies every request. The channel blocks with Tokio's blocking receive, which panics inside a runtime worker, so drive it from a dedicated OS thread. It is a test and UI seam.

`HitlAuditEntry` has `action_requested`, `approver_id`, `status` (`"approved"` or `"denied"`) and `timestamp_ms`, milliseconds since the Unix epoch.

## Configuration

`HitlPlugin` has no config struct. Constructors take the settings.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `channel` | `Arc<dyn ApprovalChannel>` | required | Rules on each high-risk call. |
| `max_audit_entries` | `usize` | `65_536` | Cap on retained audit entries. |

- `HitlPlugin::new(channel)` uses `DEFAULT_MAX_AUDIT_ENTRIES`, 65_536.
- `HitlPlugin::with_max_audit_entries(channel, max_audit_entries)` sets the cap and rejects zero.

## Caps and at-cap policy

| Bound | At-cap policy | Usage gauge |
|---|---|---|
| `DEFAULT_MAX_AUDIT_ENTRIES`, 65_536, or `max_audit_entries()` | The hook fails; no entry is evicted | `audit_len()` against `max_audit_entries()` |

- The audit log gains one entry per gated decision, approved or denied.
- At the cap, a gated call whose ruling cannot be recorded is refused. Nothing evicts an old ruling, so the record of who allowed what stays complete.
- `audit_log()` returns a snapshot in decision order. No method removes entries, so a full log refuses every gated call for the life of the plugin instance.

## Errors

- `PluginError::Internal` when a gated call arrives with the log at its cap. The message reads `hitl audit log full: <n> of <cap> entries stored`. The stream yields it as `CucaError::Plugin`, and the call is neither streamed through nor replaced by a denial.
- `PluginError::Validation` from `HitlPlugin::with_max_audit_entries` for a cap of zero.

A denial is not an error. It is a `ToolResult` in the stream.

## Example

`examples/hitl.rs` runs one policy that approves a note bound for the `scratch` notebook and denies the same tool aimed at `audit`. The application executes only the approved call, and the audit log prints both rulings. It needs a `llama-server` on port 1234 with the demo model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```rust,name=An approval policy and its registration, from examples/hitl.rs
struct NotebookPolicy;

impl ApprovalChannel for NotebookPolicy {
    fn request_approval(&self, req: &ApprovalRequest) -> ApprovalDecision {
        let decision = if req.detail.contains(OPEN_NOTEBOOK) {
            ApprovalDecision::Approved
        } else {
            ApprovalDecision::Denied
        };
        println!("  gate: {} {} -> {decision:?}", req.action, req.detail);
        decision
    }

    fn approver_id(&self) -> Option<String> {
        Some("ops-oncall".to_string())
    }
}

let hitl = Arc::new(HitlPlugin::new(Arc::new(NotebookPolicy)));
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::clone(&hitl) as Arc<dyn CucaPlugin>)
    .build()?;
```

```bash,name=Runs the same on all three platforms
cargo run --example hitl --features "provider-llamacpp plugin-hitl"
```

```text,name=Expected output
Stage 1: a call the policy approves
  gate: file_write write_note {"notebook":"scratch","text":"ship it"} -> Approved
  delivered: [ToolCall { id: "AIG5T6wty1u1kN5zRNPtMxvOjy8hTYpo", name: "write_note", arguments: Object {"notebook": String("scratch"), "text": String("ship it")} }]
  app executed: appended note 1 to notebook "scratch"
  reply: OK. I've saved the note "ship it" in your scratch notebook.
  thinking blocks: 73 then 0

Stage 2: the same tool, a notebook the policy refuses
  gate: file_write write_note {"notebook":"audit","text":"ship it"} -> Denied
  delivered: [ToolResult { tool_call_id: "POLkDMwHOUZcg35grlKbmhCSxvOQ3SoT", output: "denied by approver" }]
  reply: I'm sorry, but I was unable to save the note to the audit notebook because it was denied by the approver.
  thinking blocks: 55 then 250

Audit log, 2 of 65536 entries
  approved file_write   approver ops-oncall
  denied   file_write   approver ops-oncall
  notebook "scratch" holds ["ship it"]
```

Whether the model calls the tool, its wording and the thinking-block counts depend on the model; this output came from `google/gemma-4-12b-qat`. The full program also registers a recorder plugin before `HitlPlugin`, because a denied call otherwise erases the call the model made.

## Related pages

- [Plugins](@/plugins/_index.md) lists every plugin, its hooks and its caps.
- [Memory discipline](@/concepts/memory-discipline.md) explains why a full audit log refuses instead of evicting.
- [Write a custom plugin](@/guides/custom-plugin.md) walks through the hook contract.
- [Error types](@/reference/errors.md) lists `PluginError`.
