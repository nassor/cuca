+++
title = "Human approval"
description = "Ask a person before a risky tool call goes ahead. Shell, file write and external write calls are held for a decision."
template = "page.html"
weight = 7
+++

# Human approval

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>HitlPlugin</code> sorts streamed tool calls by risk and pauses on high-risk ones until a person or a policy approves or denies them.</dd>
<dt>You need</dt>
<dd>The <code>plugin-hitl</code> feature and an <code>ApprovalChannel</code> that you write.</dd>
<dt>Read this if</dt>
<dd>The model can call tools that run commands or change data, and you want a check before they run.</dd>
</dl>

HITL stands for human in the loop. The plugin does not run tools. It decides whether a tool call may reach your code. If the call is denied, your code gets a `ToolResult` that says so instead.

## Register it

You write an `ApprovalChannel`: one method that gets a request and returns `Approved` or `Denied`. This one approves notes for the `scratch` notebook and denies the rest.

```rust,name=An approval policy and its registration
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

Expected effect: a call to a high-risk tool, such as `write_note`, reaches `request_approval`. `Approved` lets the `ToolCall` through to your code. `Denied` replaces it with a `ToolResult` that reads `denied by approver`.

## How a call is decided

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 200" role="img" aria-labelledby="hitl-title hitl-desc">
  <title id="hitl-title">How a tool call is sorted, approved or denied, and logged.</title>
  <desc id="hitl-desc">A tool call is classified by its name. A low-risk call streams through with no audit entry. A high-risk call goes to the approval channel. Approved calls stream through unchanged and are logged. Denied calls are replaced by a ToolResult that reads denied by approver, and are logged.</desc>
  <defs>
    <marker id="hitl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">on_stream_chunk for a ToolCall</text>
  <rect class="blk blk-data" x="5" y="40" width="120" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="65" y="62">ToolCall</text>
  <text class="t-sm t-mid" x="65" y="78">classify by name</text>
  <rect class="blk blk-data" x="215" y="40" width="190" height="34" rx="6"/>
  <text class="t-sm t-mid" x="310" y="61">low risk: stream through</text>
  <rect class="blk blk-ctl" x="215" y="108" width="140" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="285" y="130">high risk</text>
  <text class="t-sm t-mid" x="285" y="146">ask the channel</text>
  <rect class="blk blk-data" x="450" y="92" width="205" height="34" rx="6"/>
  <text class="t-sm t-mid" x="552" y="113">Approved: stream through</text>
  <rect class="blk blk-data" x="450" y="140" width="205" height="50" rx="6"/>
  <text class="t-sm t-mid" x="552" y="161">Denied: ToolResult</text>
  <text class="t-sm t-mid" x="552" y="177">"denied by approver"</text>
  <path class="arw arw-data" d="M126 57 H213" marker-end="url(#hitl-arrow)"/>
  <path class="arw arw-ctl" d="M95 92 V133 H213" marker-end="url(#hitl-arrow)"/>
  <path class="arw arw-ctl" d="M356 125 H380 V109 H448" marker-end="url(#hitl-arrow)"/>
  <path class="arw arw-ctl" d="M356 141 H380 V165 H448" marker-end="url(#hitl-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Only high-risk calls wait for a decision.</b> Each decision, approved or denied, adds one audit entry. A low-risk call adds none. The plugin never runs the tool. Your code runs approved calls.</p>
</div>

## Run the example

`examples/hitl.rs` approves a note for the `scratch` notebook and denies the same tool aimed at `audit`. Your code runs only the approved call, and the audit log shows both decisions. It needs a `llama-server` on port 1234 with a tool-calling model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

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

Whether the model calls the tool, its wording and the thinking block counts depend on the model. The full program also registers a small recorder plugin before `HitlPlugin`, because a denied call otherwise erases the call the model made.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-hitl` |
| Entry type | `HitlPlugin` |
| Supporting types | `ApprovalChannel`, `ApprovalDecision`, `ApprovalRequest`, `OneshotApprovalChannel`, `Risk` |
| Also in `cuca::plugins::hitl` | `HitlAuditEntry`, `classify_tool_call` |
| Plugin name | `"hitl-approvals"` |
| Attachment | `register_plugin` |
| Needs | An `ApprovalChannel` implementation |

## Hooks

`HitlPlugin` acts on `on_stream_chunk` only. Blocks other than `ToolCall` pass through.

`classify_tool_call(name)` lowercases the tool name and checks whether it contains a keyword. Three groups are `Risk::High`, and each sets the `ApprovalRequest::action` string:

| Action | Keywords in the tool name |
|---|---|
| `shell_exec` | `shell`, `exec`, `bash`, `run_command`, `terminal` |
| `file_write` | `write`, `edit`, `delete`, `remove`, `rm_`, `mv_`, `move`, `create_file` |
| `external_api_write` | `http_post`, `http_put`, `http_delete`, `api_write`, `post_`, `put_`, `delete_` |

The shell group is checked first. Every other name is `Risk::Low`, including names the plugin does not recognize.

- A `Risk::Low` call streams through. It does not touch the channel and adds no audit entry.
- A `Risk::High` call builds an `ApprovalRequest` with `tool_call_id`, `action` and `detail`. `detail` is the tool name plus its JSON arguments, cut to 200 characters with a trailing ellipsis. The plugin then waits on `ApprovalChannel::request_approval`.
- `ApprovalDecision::Approved` streams the call through unchanged and logs `"approved"`.
- `ApprovalDecision::Denied` replaces the block with a `ToolResult` whose output is `"denied by approver"` and logs `"denied"`.

### Writing a channel

- `request_approval` is synchronous. It blocks the calling thread until an approver decides.
- `approver_id()` is optional and defaults to `None`. It fills the audit entry.
- A channel that cannot reach an approver must return `ApprovalDecision::Denied`. Then a lost approval never lets a tool run.
- `OneshotApprovalChannel::new()` returns the channel plus a Tokio oneshot `Sender<ApprovalDecision>` for the UI side. It resolves one request per pair. A dropped sender, or a second request, gives `Denied`. `OneshotApprovalChannel::default()` has no sender and denies every request.
- `OneshotApprovalChannel` blocks with Tokio's blocking receive, which panics inside a runtime worker. Drive it from a dedicated OS thread. It suits tests and simple UIs.

`HitlAuditEntry` has `action_requested`, `approver_id`, `status` (`"approved"` or `"denied"`) and `timestamp_ms` (milliseconds since the Unix epoch).

## Configuration

`HitlPlugin` has no config struct. Its constructors take the settings.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `channel` | `Arc<dyn ApprovalChannel>` | required | Decides each high-risk call. |
| `max_audit_entries` | `usize` | `65_536` | Limit on kept audit entries. |

- `HitlPlugin::new(channel)` uses `DEFAULT_MAX_AUDIT_ENTRIES`, 65_536.
- `HitlPlugin::with_max_audit_entries(channel, max_audit_entries)` sets the limit. Zero is rejected.

## Caps and at-cap policy

| Limit | At the limit | Usage gauge |
|---|---|---|
| `DEFAULT_MAX_AUDIT_ENTRIES`, 65_536, or `max_audit_entries()` | The hook fails. No entry is evicted. | `audit_len()` against `max_audit_entries()` |

- The audit log gains one entry per decision, approved or denied.
- At the limit, a gated call whose decision cannot be recorded is refused. Old decisions are never dropped, so the record of who allowed what stays complete.
- `audit_log()` returns a copy in decision order. Nothing removes entries, so a full log refuses every gated call for the life of the plugin.

## Errors

- `PluginError::Internal` when a gated call arrives and the log is full. The message reads `hitl audit log full: <n> of <cap> entries stored`. The stream yields it as `CucaError::Plugin`. The call is neither streamed through nor replaced by a denial.
- `PluginError::Validation` from `HitlPlugin::with_max_audit_entries` for a limit of zero.

A denial is not an error. It is a `ToolResult` in the stream.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Memory discipline](@/concepts/memory-discipline.md): why a full audit log refuses instead of evicting.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract.
- [Error types](@/reference/errors.md): `PluginError`.
