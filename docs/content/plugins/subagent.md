+++
title = "Subagent"
description = "The child subagent delegation plugin: spawn and collect tools, Git worktree isolation, and the pending-child cap."
template = "page.html"
weight = 6
+++

# Subagent

<dl class="page-facts">
<dt>In one line</dt>
<dd>Turns <code>spawn_subagent</code> and <code>collect_subagent</code> tool calls into asynchronous child agent runs, optionally isolated in a Git worktree.</dd>
<dt>You need</dt>
<dd>The <code>plugin-subagent</code> feature and a caller-supplied <code>SubagentRunner</code> that executes one child run.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SubagentPlugin</code> or implementing <code>SubagentRunner</code>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-subagent` (enables `dep:tokio`) |
| Entry type | `SubagentPlugin` |
| Supporting types | `SubagentSpec`, `SubagentResult`, `SubagentRunner`, `WorktreeConfig` |
| Plugin name | `"subagent-delegation"` |
| Attachment | `register_plugin` |
| Peer features | None |
| Needs | A `SubagentRunner`; a running Tokio runtime when a child spawns; `git` on the path when a worktree is requested |

`SubagentPlugin` provides the fan-out plumbing only. The `SubagentRunner` seam executes each child, for example by wrapping an external agent process or a second `CucaClient`.

## Hooks

`SubagentPlugin` overrides `on_stream_chunk`. Its `on_request` and `on_response_complete` are explicit no-ops. Tool calls with any other name pass through untouched.

- `spawn_subagent` builds a `SubagentSpec` from the call arguments, starts the child on a background Tokio task, and replaces the block with a `ToolResult` whose output is the child id. Ids have the form `sub-<n>` from a per-plugin counter. Spawning does not block.
- `collect_subagent` takes `subagent_id`, blocks the calling thread until that child finishes, and replaces the block with a `ToolResult` carrying `SubagentResult::summary`. On failure the output is the error text instead.

`spawn_subagent` arguments:

- `task` is required and must not be blank.
- `name` is a label for the child; default is the empty string.
- `tool_scope` is an array of tool names for the child; non-string entries are ignored, and the default empty array means unrestricted. The plugin passes the scope to the runner in the spec and does not enforce it.
- `worktree` is an object with a string `path` and an optional string `branch`; default is none.
- `session_id` is the parent session id, recorded in the spawn log.

When `worktree` is set, `spawn_subagent` first runs `git worktree add <path>` (or `git worktree add -b <branch> <path>` with a branch) in the process working directory. The runner reads the path from `spec.worktree`. The plugin never removes the worktree. `SubagentResult::worktree_path` lets the caller do so after collection.

`SubagentResult` fields are `subagent_id`, `summary`, `worktree_path` and `exit_ok`. The `collect_subagent` tool result carries only `summary`.

`collect` parks the calling thread on a std channel. On a current-thread Tokio runtime that thread is the only executor the child has, so the child must have delivered its result before the collect call. `examples/subagent.rs` waits for the delivery before it asks the parent to collect.

The plugin also exposes direct methods: `spawn_subagent(spec)`, `collect(id)`, `pending_len()`, `max_pending()`, `spawn_count()`, `spawns()` and `worktree_args(config)`, the dry-run `git worktree add` arguments.

## Configuration

`SubagentPlugin` has no config struct. Constructors take the settings.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `runner` | `Arc<dyn SubagentRunner>` | required | Executes one child. |
| `max_pending` | `usize` | `1024` | Cap on spawned, uncollected children. |

- `SubagentPlugin::new(runner)` uses `DEFAULT_MAX_PENDING`, 1024.
- `SubagentPlugin::with_max_pending(runner, max_pending)` sets the cap and rejects zero.
- `MAX_SPAWN_LOG`, 4096, is a constant and cannot be set.

## Caps and at-cap policy

| Structure | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| Pending children | `DEFAULT_MAX_PENDING`, 1024, or `max_pending()` | `spawn_subagent` refuses the spawn; no pending child is evicted | `pending_len()` against `max_pending()` |
| Spawn log | `MAX_SPAWN_LOG`, 4096 entries | The oldest entry is dropped | `spawns()` |

- A full registry never evicts: each pending child keeps its result receiver until `collect` removes it, and dropping one would discard a running child's result. `collect` is what drains `pending_len()`.
- The spawn log holds `(parent_session_id, worktree_path)` per spawn. It is a rolling sample, not an audit trail.
- `spawn_count()` is a `u64` total of every spawn since construction. It is not capped because it holds one integer.

## Errors

A `spawn_subagent` tool call fails the stream with `CucaError::Plugin` carrying one of these:

- `PluginError::Validation` when the arguments are not a JSON object, `task` is missing or blank, `worktree` is not an object, or `worktree.path` is missing.
- `PluginError::NotSupported` when a requested worktree cannot be prepared: `git` cannot run, the directory is not a repository, or `git worktree add` exits with failure.
- `PluginError::Internal` when `max_pending` children already await collection. The message names the count and the cap.

A `collect_subagent` call with no string `subagent_id` fails the stream with `PluginError::Validation`. Any other `collect` failure becomes the `ToolResult` output text and the stream continues:

- `PluginError::NotSupported` for an unknown id.
- `PluginError::Internal` when the background task ends without delivering a result.

`SubagentPlugin::with_max_pending` returns `PluginError::Validation` for a cap of zero.

## Example

`examples/subagent.rs` runs a parent turn that spawns a child, waits for its result, then runs a second parent turn that collects the summary. It needs a `llama-server` on port 1234 with the demo model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```rust,name=Register SubagentPlugin, from examples/subagent.rs
// LiveRunner implements SubagentRunner; it is defined in examples/subagent.rs.
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

The reply wording and the thinking-block counts depend on the model; this output came from `google/gemma-4-12b-qat`.

## Related pages

- [Plugins](@/plugins/_index.md) lists every plugin, its hooks and its caps.
- [Memory discipline](@/concepts/memory-discipline.md) explains why a full registry refuses instead of evicting.
- [Write a custom plugin](@/guides/custom-plugin.md) walks through the hook contract.
- [Error types](@/reference/errors.md) lists `PluginError`.
