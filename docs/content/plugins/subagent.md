+++
title = "Subagent"
description = "The child subagent delegation plugin: spawn and collect tools, Git worktree isolation, and the pending-child cap."
template = "page.html"
weight = 6
+++

# Subagent

<dl class="page-facts">
<dt>In one line</dt>
<dd>Turns spawn_subagent and collect_subagent tool calls into asynchronous child agent runs, optionally isolated in a Git worktree.</dd>
<dt>You need</dt>
<dd>The <code>plugin-subagent</code> feature and a caller-supplied <code>SubagentRunner</code> that executes one child run.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SubagentPlugin</code> or implementing <code>SubagentRunner</code>.</dd>
</dl>

`SubagentPlugin` turns `spawn_subagent`/`collect_subagent` tool calls, or direct `spawn_subagent`/`collect` calls, into asynchronous child agent runs executed by a caller-supplied `SubagentRunner`, optionally isolated in a Git worktree. Spawning is non-blocking: it starts the child on a background task and returns its id immediately, while `collect` blocks until that child's `SubagentResult` is ready. Reach for it to fan a task out to an isolated child agent and collect its summary back into the parent conversation.

```rust,name=Spawn a child and collect its summary
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::ProviderEndpoint;
use cuca::{CucaClient, SubagentPlugin, SubagentResult, SubagentRunner, SubagentSpec};

struct FixedRunner;

impl SubagentRunner for FixedRunner {
    fn spawn(&self, spec: SubagentSpec) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>> {
        Box::pin(async move {
            SubagentResult {
                subagent_id: String::new(),
                summary: format!("did: {}", spec.task),
                worktree_path: None,
                exit_ok: true,
            }
        })
    }
}

let subagent = Arc::new(SubagentPlugin::new(Arc::new(FixedRunner)));

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&subagent) as Arc<dyn CucaPlugin>)
    .build()?;

let id = subagent.spawn_subagent(SubagentSpec {
    name: "docs-child".into(),
    task: "summarize the repository".into(),
    tool_scope: vec![],
    worktree: None,
    session_id: None,
})?;
let result = subagent.collect(&id)?;
```

```text,name=The runner's summary comes back through collect
result.summary    "did: summarize the repository"
result.exit_ok    true
```

## Entry types

`SubagentPlugin`, `SubagentSpec`, `SubagentResult`, `SubagentRunner`, `WorktreeConfig`.

## `CucaPlugin`

`SubagentPlugin` implements `CucaPlugin` with the plugin name `"subagent-delegation"`. It overrides `on_stream_chunk` only.

## Tools

| Tool | Arguments | Behavior |
|---|---|---|
| `spawn_subagent` | `name?`, `task` (required, non-blank), `tool_scope?`, `worktree?`, `session_id?` | Registers the child, starts it on a background task, and replaces the block with a `ToolResult` carrying the child's id |
| `collect_subagent` | `subagent_id` | Blocks until the named child finishes, then replaces the block with a `ToolResult` carrying its summary, or the error text on failure |

Spawning is non-blocking; only `collect` blocks. When `SubagentSpec::worktree` is set, `spawn_subagent` first runs `git worktree add <path> [-b <branch>]` in the current working directory; a non-git working directory or a failed add surfaces as `PluginError::NotSupported`.

## Capacity, pending children

| | |
|---|---|
| Bound | `SubagentPlugin::DEFAULT_MAX_PENDING`, 1024 spawned-but-uncollected children |
| At-cap policy | `spawn_subagent` refuses the spawn rather than evicting a pending child |
| Usage gauge | `SubagentPlugin::pending_len()` against `max_pending()` |

## Capacity, spawn log

| | |
|---|---|
| Bound | `SubagentPlugin::MAX_SPAWN_LOG`, 4096 entries |
| At-cap policy | The oldest logged spawn is dropped |
| Usage gauge | `SubagentPlugin::spawns()` |

`SubagentPlugin::spawn_count()` is a separate, uncapped total counter of every spawn since construction.
