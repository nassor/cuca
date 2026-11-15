+++
title = "Skills"
description = "Load reusable SKILL.md instructions and let the model look them up and read their files on demand."
template = "page.html"
weight = 9
+++

# Skills

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SkillsPlugin</code> loads reusable instructions from <code>SKILL.md</code> files, lists them to the model, and answers its <code>skill</code>, <code>skill_read</code> and <code>skill_search</code> tool calls.</dd>
<dt>You need</dt>
<dd>The <code>plugin-skills</code> feature, and a skills directory, inline <code>Skill</code> values, or both.</dd>
<dt>Read this if</dt>
<dd>You want to give the model a library of how-to instructions it loads only when it needs them.</dd>
</dl>

A skill is a named block of instructions, plus optional reference files. The plugin adds a short catalog of skill names and descriptions to every request. The model asks for the full instructions of a skill only when it needs them, so unused skills cost few tokens.

## Register it

Put each skill in its own folder with a `SKILL.md` file:

```text,name=skills/unit-convert/SKILL.md
---
name: unit-convert
description: Convert between metric and imperial units using exact factors
---

# Unit conversion

Read `factors.md` for the exact factor, multiply, and answer with the number
followed by the unit and nothing else.
```

Add reference files in a `references/` folder next to it (here `skills/unit-convert/references/factors.md`). Then load the directory:

```rust,name=Load a skills directory and register the plugin
let skills = Arc::new(SkillsPlugin::from_dir("skills")?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::clone(&skills) as Arc<dyn CucaPlugin>)
    .build()?;
```

Expected effect: every request gets a system message listing `- unit-convert: Convert between metric and imperial units using exact factors`. When the model calls `skill`, it gets the instructions back. When it calls `skill_read`, it gets a reference file.

## Run the example

`examples/skills.rs` writes a `SKILL.md` and a `references/factors.md` file into a temporary folder and loads them. It needs a `llama-server` on port 1234 with a tool-calling model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

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

The temporary folder path depends on the platform. Which tool the model calls, its wording and the thinking block counts depend on the model.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-skills` (no dependencies) |
| Entry type | `SkillsPlugin` |
| Supporting types | `SkillsConfig`, `Skill` |
| Plugin name | `"skills"` |
| Attachment | `register_plugin` |
| Needs | A skills directory, inline `Skill` values, or both |

Build the plugin with one of:

- `SkillsPlugin::new(config)`
- `SkillsPlugin::from_dir(path)`: default config, scanning `path`.
- `SkillsPlugin::inline(skills)`: default config, no directory.

`Skill::inline(name, description, instructions)` makes a skill with no reference files. Read accessors: `skill(name)`, `list_skills()`, `search(query, limit)` and `read_reference(skill, file)`.

## Hooks

`SkillsPlugin` acts on `on_request` and `on_stream_chunk`.

- `on_request` adds one system message when `inject_catalog` is true and at least one skill is loaded. The message names the three tools, says to use `skill_search` first when unsure which skill fits, and lists each skill as `- <name>: <description>`.
- `on_stream_chunk` replaces a `skill`, `skill_read` or `skill_search` `ToolCall` with a `ToolResult`. Other blocks pass through.

| Tool | Arguments | Output |
|---|---|---|
| `skill` | `name` (non-blank) | A JSON object `{name, description, instructions, references}`. `references` lists file names. |
| `skill_read` | `skill` and `file` (non-blank) | The content of that reference file. |
| `skill_search` | `query` (non-blank) | A JSON array of `{name, description}`, at most `max_search_results` long. |

Search ignores case. Each space-separated query word scores 3 points per hit in the skill name, 2 in the description and 1 in the instructions. Skills with a score above zero are ranked by score, highest first, then by name. `SkillsPlugin::search` called directly with a blank query returns every skill.

A failure never fails the hook. The error text becomes the `ToolResult` output, so the model can see it. The plugin does not declare the three tools in your request. Add their `ToolDefinition`s yourself.

## Configuration

`SkillsConfig` keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `skills_dir` | `Option<PathBuf>` | `None` | A directory whose direct subdirectories each hold a `SKILL.md`. |
| `inline_skills` | `Vec<Skill>` | empty | Skills you build in code. |
| `inject_catalog` | `bool` | `true` | Whether `on_request` adds the catalog message. |
| `max_search_results` | `usize` | `5` | Limit on `skill_search` results. |

- Subdirectories of `skills_dir` with no `SKILL.md` are skipped silently.
- On a name clash, an inline skill wins over a directory skill. Among duplicates the later one wins. The final list is sorted by name.

### SKILL.md format

- The first line is `---`, then `key: value` lines, then a closing `---` line.
- `name:` and `description:` are required and must not be blank. Values may be plain or in double quotes.
- Other keys, such as `license` or `allowed-tools`, are read and ignored.
- Everything after the closing line, trimmed, is the instructions.
- Every regular file in an optional `references/` folder next to `SKILL.md` is a reference, keyed by file name. Content is read as UTF-8, with bad bytes replaced.

The skill's name is the `name:` value. The folder name does not matter.

## Caps and at-cap policy

No cap. The skill list is fixed when the plugin is built, so nothing grows with traffic. There is no usage gauge.

- `max_search_results` limits one `skill_search` result, not the plugin.
- The plugin loads every skill and reference file into memory at construction and sets no size limit on them.
- The catalog message grows with the number of skills. It is added to every request while `inject_catalog` is true.

## Errors

`SkillsPlugin::new` and `from_dir` return `PluginError::Io` when:

- `skills_dir` cannot be read.
- A `SKILL.md` or reference file cannot be read.
- A `SKILL.md` has no opening `---` line or no closing `---` line.
- A `SKILL.md` has no non-blank `name:` or `description:`.

The message names the file. Inline skills are never rejected.

At call time the hook returns no error. These become `ToolResult` text:

- `PluginError::Validation` when a required tool argument is missing, not a string or blank.
- `PluginError::Internal` for an unknown skill name (the text points to `skill_search`), an unknown skill in `skill_read`, or a skill with no such reference file.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract.
- [Error types](@/reference/errors.md): `PluginError`.
