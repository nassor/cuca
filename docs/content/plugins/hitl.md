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

## Entry types

`HitlPlugin`, `ApprovalChannel`, `ApprovalDecision`, `ApprovalRequest`, `OneshotApprovalChannel`, `Risk`, `HitlAuditEntry`.

## `CucaPlugin`

`HitlPlugin` implements `CucaPlugin` with the plugin name `"hitl-approvals"`. It overrides `on_stream_chunk` only.

## Risk classification

`classify_tool_call(name)` matches the tool name, case-insensitively, against keyword groups:

| Group | Keywords | `Risk` |
|---|---|---|
| Shell and exec | `shell`, `exec`, `bash`, `run_command`, `terminal` | `High` |
| File write | `write`, `edit`, `delete`, `remove`, `rm_`, `mv_`, `move`, `create_file` | `High` |
| External API write | `http_post`, `http_put`, `http_delete`, `api_write`, `post_`, `put_`, `delete_` | `High` |
| Everything else, including unrecognized names | | `Low` |

`Risk::Low` calls stream through without touching the channel. A `Risk::High` call blocks on `ApprovalChannel::request_approval`: `ApprovalDecision::Approved` streams the call through unchanged; `ApprovalDecision::Denied` replaces the block with a `ToolResult` carrying `"denied by approver"`.

## Failure-closed default

An `ApprovalChannel` implementation that cannot reach an approver must return `ApprovalDecision::Denied`, so a tool never executes on a lost approval round trip. `OneshotApprovalChannel` follows this rule: a dropped sender resolves to `Denied`.

## Capacity

| | |
|---|---|
| Bound | `HitlPlugin::DEFAULT_MAX_AUDIT_ENTRIES`, 65536 audit entries |
| At-cap policy | The hook fails rather than evicting; a gated call whose ruling cannot be recorded is refused |
| Usage gauge | `HitlPlugin::audit_len()` against `max_audit_entries()` |
