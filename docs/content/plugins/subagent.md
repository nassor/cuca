+++
title = "Subagent"
description = "Let the model start child agent runs in the background, then collect their results."
template = "page.html"
weight = 6
+++

# Subagent

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SubagentPlugin</code> turns <code>spawn_subagent</code> and <code>collect_subagent</code> tool calls into background child agent runs. A child can work in its own Git worktree.</dd>
<dt>You need</dt>
<dd>The <code>plugin-subagent</code> feature and a <code>SubagentRunner</code> that you write. It runs one child. <code>git</code> must be on the path if you use worktrees.</dd>
<dt>Read this if</dt>
<dd>You want a parent agent to hand work to child agents and read their summaries later.</dd>
</dl>

A subagent is a second agent that does one task for the parent. The plugin handles the hand-off: it starts the child, remembers it, and returns the result when the parent asks. It does not decide how a child runs. Your `SubagentRunner` does that, for example by calling a second `CucaClient` or an outside process.

## Register it

```rust,name=Register SubagentPlugin
// LiveRunner implements SubagentRunner; examples/subagent.rs defines it.
let subagent = Arc::new(SubagentPlugin::new(Arc::new(LiveRunner {
    client: child_client,
    model: model.clone(),
    gate: Arc::clone(&gate),
    done: Arc::clone(&done),
})));
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::clone(&subagent) as Arc<dyn CucaPlugin>)
    .build()?;
```

Expected effect: a `spawn_subagent` call in the stream is replaced by a `ToolResult` holding a child id such as `sub-0`, and the child starts at once. A later `collect_subagent` call with that id is replaced by the child's summary. Declare both tools in your request yourself. The plugin does not add them.

## How spawn and collect work

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="sub-title sub-desc">
  <title id="sub-title">A parent spawns a child, gets an id, and later collects the summary.</title>
  <desc id="sub-desc">In turn one the parent sends spawn_subagent. The plugin starts the child in the background through the runner and returns the child id as a ToolResult. In a later turn the parent sends collect_subagent with that id. The plugin waits for the child to finish and returns its summary as a ToolResult.</desc>
  <defs>
    <marker id="sub-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">SubagentPlugin on_stream_chunk</text>
  <rect class="blk blk-data" x="5" y="40" width="150" height="40" rx="6"/>
  <text class="t-sm t-mid" x="80" y="64">spawn_subagent</text>
  <rect class="blk blk-ctl" x="255" y="40" width="150" height="40" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="330" y="64">SubagentRunner</text>
  <rect class="blk blk-data" x="505" y="40" width="150" height="40" rx="6"/>
  <text class="t-sm t-mid" x="580" y="64">ToolResult: sub-0</text>
  <rect class="blk blk-data" x="5" y="124" width="150" height="40" rx="6"/>
  <text class="t-sm t-mid" x="80" y="148">collect_subagent</text>
  <rect class="blk blk-ctl" x="255" y="124" width="150" height="40" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="330" y="148">wait for the child</text>
  <rect class="blk blk-data" x="505" y="124" width="150" height="40" rx="6"/>
  <text class="t-sm t-mid" x="580" y="148">ToolResult: summary</text>
  <path class="arw arw-ctl" d="M156 60 H253" marker-end="url(#sub-arrow)"/>
  <path class="arw arw-data" d="M406 60 H503" marker-end="url(#sub-arrow)"/>
  <path class="arw arw-ctl" d="M156 144 H253" marker-end="url(#sub-arrow)"/>
  <path class="arw arw-data" d="M406 144 H503" marker-end="url(#sub-arrow)"/>
  <text class="t-sm" x="6" y="108">Turn 1: spawning does not block.</text>
  <text class="t-sm" x="6" y="184">A later turn: collecting blocks until the child finishes.</text>
</svg>
</div>
<p class="dgm-cap"><b>Spawn returns at once. Collect waits.</b> The child runs in the background between the two calls. Each uncollected child counts toward <code>max_pending</code>.</p>
</div>

## Run the example

`examples/subagent.rs` runs a parent turn that spawns a child, waits for it to finish, then runs a second parent turn that collects the summary. It needs a `llama-server` on port 1234 with a tool-calling model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example subagent --features "provider-llamacpp plugin-subagent"
```

```text,name=Expected output
Pending gauge: 0 of 1024 children awaiting collection

Turn 1: the parent delegates, and spawning does not block
  child id from the tool result: sub-0
  pending 1 of 1024, spawns so far 1
  thinking blocks: 121

The child runs its own turn while the parent waits
  the child delivered its result

Turn 2: the parent collects the summary through the same pipeline
  collected summary: "The largest moon of Saturn is Titan."
  pending 0 of 1024, spawns so far 1
  thinking blocks: 83

Spawn log, 1 entr(ies)
  parent session "unset", worktree "none"
```

The reply wording and the thinking block counts depend on the model.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-subagent` |
| Entry type | `SubagentPlugin` |
| Supporting types | `SubagentSpec`, `SubagentResult`, `SubagentRunner`, `WorktreeConfig` |
| Plugin name | `"subagent-delegation"` |
| Attachment | `register_plugin` |
| Needs | A `SubagentRunner`, a running Tokio runtime when a child starts, and `git` on the path when a worktree is requested |

## Hooks

`SubagentPlugin` acts on `on_stream_chunk` only. Tool calls with other names pass through.

- `spawn_subagent` builds a `SubagentSpec` from the call arguments, starts the child on a background Tokio task, and replaces the block with a `ToolResult` holding the child id. Ids look like `sub-<n>`. Spawning does not block.
- `collect_subagent` takes `subagent_id`, waits until that child finishes, and replaces the block with a `ToolResult` holding `SubagentResult::summary`. On failure the output is the error text.

`spawn_subagent` arguments:

| Argument | Required | Meaning |
|---|---|---|
| `task` | yes | What the child should do. Must not be blank. |
| `name` | no | A label for the child. Default is the empty string. |
| `tool_scope` | no | Array of tool names for the child. Non-string entries are ignored. The default, an empty array, means unrestricted. The plugin passes it to the runner and does not enforce it. |
| `worktree` | no | An object with a string `path` and an optional string `branch`. |
| `session_id` | no | The parent session id, kept in the spawn log. |

When `worktree` is set, the plugin first runs `git worktree add <path>` (or `git worktree add -b <branch> <path>`) in the process working directory. The runner reads the path from `spec.worktree`. The plugin never removes the worktree. After collecting, you can remove it using `SubagentResult::worktree_path`.

`SubagentResult` has `subagent_id`, `summary`, `worktree_path` and `exit_ok`. The `collect_subagent` tool result carries only `summary`.

On a current-thread Tokio runtime, collecting blocks the only thread the child can run on. The child must already have delivered its result before the collect call. `examples/subagent.rs` waits for delivery before the parent collects.

Direct methods: `spawn_subagent(spec)`, `collect(id)`, `pending_len()`, `max_pending()`, `spawn_count()`, `spawns()` and `worktree_args(config)`, which returns the `git worktree add` arguments without running them.

## Configuration

`SubagentPlugin` has no config struct. Its constructors take the settings.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `runner` | `Arc<dyn SubagentRunner>` | required | Runs one child. |
| `max_pending` | `usize` | `1024` | Limit on spawned children that are not yet collected. |

- `SubagentPlugin::new(runner)` uses `DEFAULT_MAX_PENDING`, 1024.
- `SubagentPlugin::with_max_pending(runner, max_pending)` sets the limit. Zero is rejected.
- `MAX_SPAWN_LOG`, 4096, is fixed.

## Caps and at-cap policy

| Structure | Limit | At the limit | Usage gauge |
|---|---|---|---|
| Pending children | `DEFAULT_MAX_PENDING`, 1024, or `max_pending()` | `spawn_subagent` refuses the spawn. No pending child is dropped. | `pending_len()` against `max_pending()` |
| Spawn log | `MAX_SPAWN_LOG`, 4096 entries | The oldest entry is dropped. | `spawns()` |

- Each pending child keeps its result until `collect` takes it. Dropping one would lose a running child's result, so a full registry refuses instead. `collect` is what lowers `pending_len()`.
- The spawn log holds `(parent_session_id, worktree_path)` for each spawn. It is a rolling sample, not an audit trail.
- `spawn_count()` is a `u64` total of all spawns since the plugin was built. It is one integer and has no limit.

## Errors

A `spawn_subagent` call fails the stream with `CucaError::Plugin` carrying one of these:

- `PluginError::Validation` when the arguments are not a JSON object, `task` is missing or blank, `worktree` is not an object, or `worktree.path` is missing.
- `PluginError::NotSupported` when a requested worktree cannot be made: `git` cannot run, the directory is not a repository, or `git worktree add` fails.
- `PluginError::Internal` when `max_pending` children already wait for collection. The message names the count and the limit.

A `collect_subagent` call with no string `subagent_id` fails the stream with `PluginError::Validation`. Any other `collect` failure becomes the `ToolResult` text and the stream goes on:

- `PluginError::NotSupported` for an unknown id.
- `PluginError::Internal` when the background task ends without a result.

`SubagentPlugin::with_max_pending` returns `PluginError::Validation` for a limit of zero.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Memory discipline](@/concepts/memory-discipline.md): why a full registry refuses instead of dropping.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract.
- [Error types](@/reference/errors.md): `PluginError`.
