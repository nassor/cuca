+++
title = "Skills"
description = "The reusable agent skills plugin: SKILL.md loading, the catalog injection, and the skill, skill_read, and skill_search tools."
template = "page.html"
weight = 9
+++

# Skills

<dl class="page-facts">
<dt>In one line</dt>
<dd>Loads reusable SKILL.md instructions and resolves skill, skill_read, and skill_search tool calls against them.</dd>
<dt>You need</dt>
<dd>The <code>plugin-skills</code> feature and either a skills directory or inline <code>Skill</code> values.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SkillsPlugin</code> or authoring a <code>SKILL.md</code> file.</dd>
</dl>

`SkillsPlugin` loads reusable `SKILL.md` instructions, either from a directory of `SKILL.md` subdirectories or provided inline, and resolves the `skill`, `skill_read`, and `skill_search` tool calls a model issues against them. When `inject_catalog` is set (the default), `on_request` appends a system message naming the three tools and listing every loaded skill's name and description. Reach for it to give a model a discoverable library of task-specific instructions without hand-writing them into every system prompt.

```rust,name=Load one inline skill and answer a skill tool call
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, Skill, SkillsPlugin};
use serde_json::json;

let skills = Arc::new(SkillsPlugin::inline(vec![Skill::inline(
    "math",
    "Solve arithmetic problems",
    "Add, subtract, multiply and divide exactly as asked.",
)]));

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&skills) as Arc<dyn CucaPlugin>)
    .build()?;

let mut call = MessageContentBlock::ToolCall {
    id: "c1".into(),
    name: "skill".into(),
    arguments: json!({ "name": "math" }),
};
skills.on_stream_chunk(&mut call)?;
```

```text,name=The tool call resolves to the skill's instructions
ToolResult { tool_call_id: "c1", output: "{\"description\":\"Solve arithmetic problems\",\"instructions\":\"Add, subtract, multiply and divide exactly as asked.\",\"name\":\"math\",\"references\":[]}" }
```

## Entry types

`SkillsPlugin`, `SkillsConfig`, `Skill`.

## `CucaPlugin`

`SkillsPlugin` implements `CucaPlugin` with the plugin name `"skills"`.

| Hook | Behavior |
|---|---|
| `on_request` | Injects a system message naming the three tools and listing every loaded skill's name and description, when `SkillsConfig::inject_catalog` is true and at least one skill is loaded |
| `on_stream_chunk` | Resolves `skill`, `skill_read`, and `skill_search` tool calls into `ToolResult` blocks |

## Config

`SkillsConfig` defaults: `skills_dir: None`, `inline_skills: []`, `inject_catalog: true`, `max_search_results: 5`.

`SkillsConfig::skills_dir` points at a directory whose direct subdirectories each hold a `SKILL.md`: YAML-ish frontmatter with required `name:` and `description:` keys, an instructions body, and an optional `references/` directory of bundled files. `inline_skills` are provided programmatically and win over directory-discovered skills on a name conflict.

## Tools

| Tool | Arguments | Behavior |
|---|---|---|
| `skill` | `name` | Returns the named skill's full instructions |
| `skill_read` | `skill`, `file` | Returns a reference file bundled with the named skill |
| `skill_search` | `query` | Returns skills ranked by a substring match score, capped at `max_search_results` |

`skill_search` scores each whitespace-split query term at three points per match in the skill name, two in the description, and one in the instructions; a blank query returns every loaded skill.

## Capacity

No growth cap. The skill list is fixed once at construction, deduplicated by name and sorted.
