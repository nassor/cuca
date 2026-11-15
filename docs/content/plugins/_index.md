+++
title = "Plugins"
description = "Twelve optional plugins that watch or change a request and its streamed reply: cost caps, redaction, memory, tools and more."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>A plugin is a small add-on you register on the client. It can edit a request, change or replace streamed blocks, and see the finished response.</dd>
<dt>You need</dt>
<dd>The <code>plugin-*</code> features you want, named in your <code>cargo</code> command. None is on by default.</dd>
<dt>Read this if</dt>
<dd>You want a plugin's feature flag, entry type, hooks, config defaults or caps.</dd>
</dl>

## Register a plugin

Every plugin attaches with `register_plugin`. This example adds a cost tracker and an in-memory session log:

```rust,name=Register two plugins
use std::sync::Arc;

let cost = Arc::new(CostPlugin::new(CostConfig::default())?);
let log = Arc::new(SessionLogPlugin::new_in_memory());

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(cost as Arc<dyn CucaPlugin>)
    .register_plugin(log as Arc<dyn CucaPlugin>)
    .build()?;
```

Build it with the features named explicitly:

```bash,name=Runs the same on all three platforms
cargo build --features "provider-llamacpp plugin-cost plugin-session-log"
```

Expected effect: every request through `client` is priced and recorded. Keep your own `Arc` handle (as `cost` and `log` above, before the cast) to read the totals later.

## How plugins fit into a turn

A hook is a method the client calls at a fixed moment of a turn. A content block is one piece of a streamed reply: text, thinking, a tool call or a tool result. A plugin implements only the hooks it needs.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 196" role="img" aria-labelledby="hub-title hub-desc">
  <title id="hub-title">The four steps of one turn, and the order plugins run in at each hook.</title>
  <desc id="hub-desc">A request passes through on_request, goes to the provider, and each streamed block passes through on_stream_chunk. When the stream ends, on_response_complete runs once. At every hook the registered plugins run in registration order: first, second, third.</desc>
  <defs>
    <marker id="hub-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One turn through the client</text>
  <rect class="blk blk-ctl" x="5" y="38" width="140" height="56" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="75" y="62">on_request</text>
  <text class="t-sm t-mid" x="75" y="80">edit the request</text>
  <rect class="blk blk-bnd" x="175" y="38" width="140" height="56" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="245" y="62">provider</text>
  <text class="t-sm t-mid" x="245" y="80">model call</text>
  <rect class="blk blk-ctl" x="345" y="38" width="140" height="56" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="415" y="62">on_stream_chunk</text>
  <text class="t-sm t-mid" x="415" y="80">once per block</text>
  <rect class="blk blk-ctl" x="515" y="38" width="140" height="56" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="585" y="62">on_response_complete</text>
  <text class="t-sm t-mid" x="585" y="80">once, at the end</text>
  <path class="arw arw-ctl" d="M146 66 H173" marker-end="url(#hub-arrow)"/>
  <path class="arw arw-data" d="M316 66 H343" marker-end="url(#hub-arrow)"/>
  <path class="arw arw-ctl" d="M486 66 H513" marker-end="url(#hub-arrow)"/>
  <text class="t-sm" x="6" y="130">At every hook, plugins run in registration order:</text>
  <rect class="blk blk-ctl" x="5" y="146" width="120" height="34" rx="6"/>
  <text class="t-sm t-mid" x="65" y="167">first plugin</text>
  <rect class="blk blk-ctl" x="155" y="146" width="120" height="34" rx="6"/>
  <text class="t-sm t-mid" x="215" y="167">second plugin</text>
  <rect class="blk blk-ctl" x="305" y="146" width="120" height="34" rx="6"/>
  <text class="t-sm t-mid" x="365" y="167">third plugin</text>
  <path class="arw arw-ctl" d="M126 163 H153" marker-end="url(#hub-arrow)"/>
  <path class="arw arw-ctl" d="M276 163 H303" marker-end="url(#hub-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>A turn has four steps and three hook sites.</b> A plugin registered earlier sees the request first and the block first. The next plugin sees what the earlier one left.</p>
</div>

| Hook | Runs | If a plugin returns an error |
|---|---|---|
| `on_request` | Once per turn, before the request is sent. May change the request. | The turn fails before it is sent. Later plugins do not run. |
| `execute_local_tool` | For each tool call block. The first plugin that returns `Some` wins. | The stream yields the error in place of the block. |
| `on_stream_chunk` | Once per block. May change or replace the block. | The stream yields the error in place of the block. |
| `on_response_complete` | Once, when the stream ends. | The error is not passed on. The next plugin's hook still runs. |

- A local tool result must carry the same `tool_call_id` as the call it answers. A mismatch is `PluginError::Validation` with schema `local tool result`. No bundled plugin overrides `execute_local_tool`.
- Plugins turn expected problems into blocks. For example, a failed tool call becomes a `ToolResult` that holds the error text, so the model can react.
- With a prompt cache hit (`service-prompt-cache`), the stored blocks replay. `on_stream_chunk` and `execute_local_tool` are skipped. Every `on_response_complete` still runs.

## The twelve plugins

| Plugin | Feature | What it is for |
|---|---|---|
| [MCP](@/plugins/mcp.md) | `plugin-mcp` | Calls the tools of a Model Context Protocol server. |
| [Sandbox](@/plugins/sandbox.md) | `plugin-sandbox` | Runs model-written WebAssembly with strict limits. |
| [Memory](@/plugins/memory.md) | `plugin-memory` | Shrinks long conversations and adds a working knowledge graph. |
| [Guardrails](@/plugins/guardrails.md) | `plugin-guardrails` | Checks tool arguments and JSON replies against a JSON Schema. |
| [Subagent](@/plugins/subagent.md) | `plugin-subagent` | Lets the model start and collect child agent runs. |
| [Human approval](@/plugins/hitl.md) | `plugin-hitl` | Asks a person before a risky tool call goes ahead. |
| [Web search](@/plugins/web-search.md) | `plugin-web-search` | Gives the model live web search and page extraction. |
| [Skills](@/plugins/skills.md) | `plugin-skills` | Loads reusable `SKILL.md` instructions for the model. |
| [Telemetry](@/plugins/telemetry.md) | `plugin-telemetry` | Reports request, token and latency metrics through OpenTelemetry. |
| [Session log](@/plugins/session-log.md) | `plugin-session-log` | Records every turn as an append-only log you can replay and fork. |
| [Redaction](@/plugins/redaction.md) | `plugin-redaction` | Hides secrets and personal data before a request leaves your process. |
| [Cost accounting](@/plugins/cost.md) | `plugin-cost` | Estimates tokens and spend, and refuses turns over a budget. |

The entry type and the hooks each plugin uses are in its own page.

## Order and independence

- Plugins run in registration order. No bundled plugin needs a particular position, but order changes what later plugins see. For example, register [Redaction](@/plugins/redaction.md) before [Session log](@/plugins/session-log.md) so the log stores scrubbed text.
- No plugin depends on another plugin. Each `plugin-*` feature builds alone.
- A [service](@/services/_index.md) can depend on a plugin: `service-entity-extraction` and `service-vector-store` need `plugin-memory`, and `service-replay` needs `plugin-session-log`. A plugin never depends on a service.
- A service is called explicitly. It does not implement `CucaPlugin`, so `register_plugin` rejects it at compile time.

[Plugins and services](@/concepts/plugin-layering.md) explains why the tiers are split.

## Caps and what happens at a cap

Every structure that could grow has a limit, a rule for what happens at the limit, and a way to read how full it is. The rule is to refuse or fold, not to silently drop old data. [Memory discipline](@/concepts/memory-discipline.md) explains why.

| Plugin | Limit | At the limit | How to read usage |
|---|---|---|---|
| Guardrails | 4096 tracked tool call ids | Evicts the oldest. | `tracked_calls()` |
| Subagent | 1024 uncollected children | Refuses the spawn. | `pending_len()` |
| Subagent | 4096 spawn log entries | Evicts the oldest. | `spawns()` |
| Human approval | 65536 audit entries | Refuses the gated call. | `audit_len()` |
| Session log | 65536 records (in memory) | Refuses the append or fork. | `len()` |
| Cost accounting | 64 tracked models | Folds extra models into one overflow bucket. Totals stay exact. | `usage()`, `breakdown()` |
| Memory | The message budget you set | Runs the compaction strategies. | `ContextUsage`, `count_tokens()` |
| Memory | Working graph: no limit | You own the bound. | `MemoryGraph::len()` |
| Redaction | 256 rules, 4096 matches per string, JSON depth 64 | Rejects the policy, or refuses the request. | `rule_count()`, `match_cap()` |
| Sandbox | Memory, instructions, time and output per call | Stops that call. | `last_diagnostic()` |
| MCP, Skills, Telemetry, Web search | Nothing grows with traffic | Not applicable. | None |

## Related pages

- [The feature matrix](@/reference/features.md): every feature and what it enables.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Write a custom plugin](@/guides/custom-plugin.md): implement `CucaPlugin` yourself.

## Per-plugin pages
