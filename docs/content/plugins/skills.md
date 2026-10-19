+++
title = "Skills"
description = "The reusable agent skills plugin: SKILL.md loading, the catalog injection, and the skill, skill_read, and skill_search tools."
template = "page.html"
weight = 9
+++

# Skills

<dl class="page-facts">
<dt>In one line</dt>
<dd>Loads reusable <code>SKILL.md</code> instructions and resolves <code>skill</code>, <code>skill_read</code> and <code>skill_search</code> tool calls against them.</dd>
<dt>You need</dt>
<dd>The <code>plugin-skills</code> feature and either a skills directory or inline <code>Skill</code> values.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SkillsPlugin</code> or authoring a <code>SKILL.md</code> file.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-skills` (no dependencies) |
| Entry type | `SkillsPlugin` |
| Supporting types | `SkillsConfig`, `Skill` |
| Plugin name | `"skills"` |
| Attachment | `register_plugin` |
| Peer features | None |
| Needs | A skills directory, inline `Skill` values, or both |

Build the plugin with `SkillsPlugin::new(config)`, `SkillsPlugin::from_dir(path)` (default config, scanning `path`) or `SkillsPlugin::inline(skills)` (default config, no directory). `Skill::inline(name, description, instructions)` builds a skill without references. The accessors are `skill(name)`, `list_skills()`, `search(query, limit)` and `read_reference(skill, file)`.

## Hooks

`SkillsPlugin` overrides `on_request` and `on_stream_chunk`.

- `on_request` appends one system message to the request when `inject_catalog` is true and at least one skill is loaded. The message names the three tools, tells the model to use `skill_search` first when unsure which skill applies, and lists each skill as `- <name>: <description>`.
- `on_stream_chunk` replaces a `skill`, `skill_read` or `skill_search` `ToolCall` with a `ToolResult`. Any other block passes through untouched.

Tool arguments and results:

- `skill` takes a non-blank `name`. The output is a JSON object `{name, description, instructions, references}`, where `references` lists the file names.
- `skill_read` takes non-blank `skill` and `file`. The output is the content of that reference file.
- `skill_search` takes a non-blank `query`. The output is a JSON array of `{name, description}` objects, at most `max_search_results` long.

Search is case-insensitive. Each whitespace-separated query term scores 3 points per occurrence in the skill name, 2 in the description and 1 in the instructions. Skills with a positive score are ranked by score descending, then name ascending. `SkillsPlugin::search` called directly with a blank query returns every skill.

Failures never fail the hook. The error text becomes the `ToolResult` output, so the model can react to it in the conversation.

## Configuration

`SkillsConfig` defaults:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `skills_dir` | `Option<PathBuf>` | `None` | Directory whose direct subdirectories each hold a `SKILL.md`. |
| `inline_skills` | `Vec<Skill>` | empty | Skills supplied in code. |
| `inject_catalog` | `bool` | `true` | Whether `on_request` injects the catalog message. |
| `max_search_results` | `usize` | `5` | Cap on `skill_search` results. |

- `skills_dir` subdirectories without a `SKILL.md` are skipped silently.
- `inline_skills` win over directory skills on a name conflict. Among duplicates, the later entry wins, and the final list is sorted by name.

### SKILL.md format

A `SKILL.md` file has this shape:

- The first line is `---`, followed by `key: value` lines, then a closing `---` line.
- `name:` and `description:` are required and must not be blank. Values may be unquoted or double-quoted.
- Other keys, such as `license` or `allowed-tools`, are read and ignored.
- Everything after the closing line, trimmed, is the instructions.
- Every regular file in an optional `references/` directory next to `SKILL.md` becomes a reference keyed by file name. Content is read as UTF-8, lossy.

The skill name is the `name:` value. The directory name does not matter.

## Caps and at-cap policy

No cap: the skill list is fixed at construction, deduplicated by name and sorted, so nothing grows with traffic. The usage gauge is none.

- `max_search_results` bounds one `skill_search` result, not the plugin.
- The plugin loads every skill and every reference file into memory at construction and sets no size limit on them.
- The catalog message grows with the number of loaded skills and is added to every request while `inject_catalog` is true.

## Errors

Construction returns `PluginError::Io`, from `SkillsPlugin::new` and `from_dir`, when:

- `skills_dir` cannot be read.
- A `SKILL.md` or a reference file is unreadable.
- A `SKILL.md` has no leading `---` line or no closing `---` line.
- A `SKILL.md` lacks a non-blank `name:` or `description:`.

The message names the offending file. Inline skills are never rejected.

At call time the hook returns no error. Each variant below becomes `ToolResult` output text:

- `PluginError::Validation` when a required tool argument is missing, not a string, or blank.
- `PluginError::Internal` for an unknown skill name (the text points to `skill_search`), an unknown skill in `skill_read`, or a skill with no such reference file.

## Example

`examples/skills.rs` writes a `SKILL.md` and one `references/factors.md` into a temporary directory and loads them with `SkillsPlugin::from_dir`. It takes three turns: the model calls `skill` and gets the instructions back, then calls `skill_read` for the bundled file, then answers. The directory is removed when the program ends. It needs a `llama-server` on port 1234 with the demo model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```text,name=The SKILL.md the example writes
---
name: unit-convert
description: Convert between metric and imperial units using exact factors
license: Apache-2.0
---

# Unit conversion

Read `factors.md` for the exact factor, multiply, and answer with the number
followed by the unit and nothing else.
```

```rust,name=Load a skills directory and register the plugin, from examples/skills.rs
let skills = Arc::new(SkillsPlugin::from_dir(dir.path())?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::clone(&skills) as Arc<dyn CucaPlugin>)
    .build()?;
```

```bash,name=Runs the same on all three platforms
cargo run --example skills --features "provider-llamacpp plugin-skills"
```

```text,name=Expected output
Loaded from /tmp/cuca-skills-3284142
  unit-convert: Convert between metric and imperial units using exact factors (1 reference file(s))

The model reaches for the skill out of the injected catalog
  catalog in the outbound request, last line: - unit-convert: Convert between metric and imperial units using exact factors
  skill returned: {"description":"Convert between metric and imperial units using exact factors","instructions":"# Unit conversion\n\nRead `factors.md` for the exact factor, multiply, and answer with the number\nfollowed by the unit and nothing else.","name":"unit-convert","references":["factors.md"]}
  thinking blocks: 138

The model reaches for the reference file
  catalog in the outbound request, last line: - unit-convert: Convert between metric and imperial units using exact factors
  skill_read returned: 1 mile = 1.609344 kilometers
1 pound = 0.45359237 kilograms

  thinking blocks: 122

The answer, from the instructions and the factor
  reply: 19.312128 kilometers
  thinking blocks: 369
```

The temporary directory path depends on the platform. Which tool the model calls, its wording and the thinking-block counts depend on the model; this output came from `google/gemma-4-12b-qat`. The full program also registers a recorder plugin around `SkillsPlugin` to print the catalog and the model's tool call.

## Related pages

- [Plugins](@/plugins/_index.md) lists every plugin, its hooks and its caps.
- [Write a custom plugin](@/guides/custom-plugin.md) walks through the hook contract.
- [Error types](@/reference/errors.md) lists `PluginError`.
