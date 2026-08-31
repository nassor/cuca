//! Reusable agent skills (agentskills.io `SKILL.md` format).
//!
//! [`SkillsPlugin`] implements the "agent skills" concept (reusable
//! instructions that teach an agent how to perform specific tasks, discovered
//! during a conversation via tools) in CUCA's everything-is-a-plugin
//! architecture. Skills are plain `SKILL.md` documents: YAML-ish frontmatter
//! with required `name:`/`description:` keys, an instructions body, and an
//! optional `references/` directory of bundled files. They are loaded either
//! by scanning a directory for `SKILL.md` subdirectories or by providing them
//! inline in the config.
//!
//! The plugin injects a catalog system message into every request
//! ([`CucaPlugin::on_request`]) and intercepts three model-issued tool calls on
//! [`CucaPlugin::on_stream_chunk`], replacing them with
//! [`ToolResult`](crate::types::MessageContentBlock::ToolResult) blocks:
//!
//! - `skill {name}`: load a skill's full instructions.
//! - `skill_read {skill, file}`: read a reference file bundled with a skill.
//! - `skill_search {query}`: find skills by description.
//!
//! Tool-call validation failures, unknown skills, and missing references
//! surface as descriptive error text inside the `ToolResult` output rather
//! than failing the hook, so the model can react to them mid-conversation (the
//! same contract as [`crate::plugins::web_search`]).

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, UnifiedMessage};

/// A single reusable agent skill.
///
/// Mirrors the agentskills.io `SKILL.md` shape: a human-readable `name` and
/// `description` used for discovery, the `instructions` body the agent should
/// follow, and an optional set of `references` (file name → content) the agent
/// can pull in on demand via the `skill_read` tool.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Skill {
    /// The skill's unique name (the `name:` frontmatter key / tool argument).
    pub name: String,
    /// One-line description used by `skill_search` and the injected catalog.
    pub description: String,
    /// The full instructions the agent follows when the skill is loaded.
    pub instructions: String,
    /// Bundled reference files (file name → content), loaded from a skill's
    /// `references/` directory; empty for skills without references.
    #[serde(default)]
    pub references: BTreeMap<String, String>,
}

impl Skill {
    /// Build a skill inline with no bundled references.
    pub fn inline(
        name: impl Into<String>,
        description: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
            references: BTreeMap::new(),
        }
    }

    /// Parse a skill from `<dir>/SKILL.md` (agentskills.io frontmatter format).
    ///
    /// The file must start with a `---` line, followed by `key: value`
    /// frontmatter lines (required `name:` and `description:`, other keys such
    /// as `license` or `allowed-tools` parsed and ignored; values may be
    /// unquoted or double-quoted), a closing `---` line, and the instructions
    /// body. If `<dir>/references/` exists, every regular file in it becomes a
    /// reference keyed by file name (UTF-8, lossy). Read and parse failures
    /// are reported as [`PluginError::Io`] errors that name the offending
    /// file.
    pub(crate) fn from_skill_md(dir: &std::path::Path) -> Result<Skill, PluginError> {
        let skill_md = dir.join("SKILL.md");
        let text = std::fs::read_to_string(&skill_md)
            .map_err(|e| PluginError::Io(format!("failed to read {}: {e}", skill_md.display())))?;
        let mut lines = text.lines();
        // Frontmatter must start at the very first line of the file.
        let Some(first) = lines.next() else {
            return Err(PluginError::Io(format!(
                "{} has no frontmatter: expected a leading `---` line",
                skill_md.display()
            )));
        };
        if first.trim_end() != "---" {
            return Err(PluginError::Io(format!(
                "{} has no frontmatter: expected a leading `---` line",
                skill_md.display()
            )));
        }
        let mut name: Option<String> = None;
        let mut description: Option<String> = None;
        let mut body: Vec<&str> = Vec::new();
        let mut in_frontmatter = true;
        for line in lines {
            if in_frontmatter {
                if line.trim_end() == "---" {
                    in_frontmatter = false;
                } else if let Some((key, value)) = line.split_once(':') {
                    let value = value.trim();
                    let value = value
                        .strip_prefix('"')
                        .and_then(|v| v.strip_suffix('"'))
                        .unwrap_or(value);
                    match key.trim() {
                        "name" => name = Some(value.to_string()),
                        "description" => description = Some(value.to_string()),
                        _ => {}
                    }
                }
            } else {
                body.push(line);
            }
        }
        if in_frontmatter {
            return Err(PluginError::Io(format!(
                "{} has unterminated frontmatter: missing a closing `---` line",
                skill_md.display()
            )));
        }
        let name = match name {
            Some(n) if !n.trim().is_empty() => n,
            _ => {
                return Err(PluginError::Io(format!(
                    "{} is missing a required non-blank `name:` frontmatter key",
                    skill_md.display()
                )));
            }
        };
        let description = match description {
            Some(d) if !d.trim().is_empty() => d,
            _ => {
                return Err(PluginError::Io(format!(
                    "{} is missing a required non-blank `description:` frontmatter key",
                    skill_md.display()
                )));
            }
        };
        let mut references = BTreeMap::new();
        let references_dir = dir.join("references");
        if references_dir.is_dir() {
            let entries = std::fs::read_dir(&references_dir).map_err(|e| {
                PluginError::Io(format!("failed to list {}: {e}", references_dir.display()))
            })?;
            for entry in entries {
                let entry = entry.map_err(|e| {
                    PluginError::Io(format!(
                        "failed to read an entry of {}: {e}",
                        references_dir.display()
                    ))
                })?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let bytes = std::fs::read(&path).map_err(|e| {
                    PluginError::Io(format!("failed to read {}: {e}", path.display()))
                })?;
                let content = String::from_utf8_lossy(&bytes).into_owned();
                references.insert(entry.file_name().to_string_lossy().into_owned(), content);
            }
        }
        Ok(Skill {
            name,
            description,
            instructions: body.join("\n").trim().to_string(),
            references,
        })
    }
}

/// Configuration for a [`SkillsPlugin`].
///
/// [`skills_dir`](Self::skills_dir) points at a directory whose direct
/// subdirectories each hold a `SKILL.md`; [`inline_skills`](Self::inline_skills)
/// are provided programmatically and win on name conflicts.
#[derive(Debug, Clone)]
pub struct SkillsConfig {
    /// Directory scanned for direct `SKILL.md` subdirectories.
    pub skills_dir: Option<std::path::PathBuf>,
    /// Skills provided inline; take precedence over directory-discovered ones.
    pub inline_skills: Vec<Skill>,
    /// Whether `on_request` injects the tool-instructions + catalog system
    /// message into every request (default `true`).
    pub inject_catalog: bool,
    /// Cap on `skill_search` results (default 5).
    pub max_search_results: usize,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            skills_dir: None,
            inline_skills: Vec::new(),
            inject_catalog: true,
            max_search_results: 5,
        }
    }
}

/// Reusable agent skills plugin: resolves `skill`, `skill_read`, and
/// `skill_search` tool calls in the stream pipeline against a loaded skill
/// list, and injects a discoverable catalog into each request.
#[derive(Clone)]
pub struct SkillsPlugin {
    config: SkillsConfig,
    /// Deduplicated skills, sorted by name (inline skills win on conflict).
    skills: Vec<Skill>,
}

impl SkillsPlugin {
    /// Build a plugin from a config, loading directory skills (if
    /// [`SkillsConfig::skills_dir`] is set) and appending inline skills.
    ///
    /// Directory loading reads `<dir>/<subdir>/SKILL.md` for every direct
    /// subdirectory that contains one; subdirectories without a `SKILL.md` are
    /// skipped silently. Duplicate names are resolved with inline skills
    /// winning, and the final list is sorted by name.
    ///
    /// # Errors
    ///
    /// [`PluginError::Io`] when [`SkillsConfig::skills_dir`] cannot be read, or
    /// when a subdirectory's `SKILL.md` is unreadable, has no `---`
    /// frontmatter, or is missing a non-blank `name:`/`description:` (see
    /// [`Skill::from_skill_md`]). Inline skills are never rejected.
    pub fn new(config: SkillsConfig) -> Result<Self, PluginError> {
        let mut skills = Vec::new();
        if let Some(dir) = &config.skills_dir {
            skills.extend(Self::load_dir(dir)?);
        }
        // Inline skills are appended last so the dedup pass lets them overwrite
        // directory-discovered skills of the same name.
        skills.extend(config.inline_skills.iter().cloned());
        Ok(Self {
            config,
            skills: Self::dedup_sorted(skills),
        })
    }

    /// Build a plugin from inline skills only, with default config.
    pub fn inline(skills: Vec<Skill>) -> Self {
        Self {
            config: SkillsConfig::default(),
            skills: Self::dedup_sorted(skills),
        }
    }

    /// Build a plugin scanning `path` for `SKILL.md` skill subdirectories, with
    /// default config.
    ///
    /// # Errors
    ///
    /// The [`Self::new`] errors, for `skills_dir = Some(path)`.
    pub fn from_dir(path: impl AsRef<std::path::Path>) -> Result<Self, PluginError> {
        Self::new(SkillsConfig {
            skills_dir: Some(path.as_ref().to_path_buf()),
            ..SkillsConfig::default()
        })
    }

    /// Look up a loaded skill by exact name.
    pub fn skill(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    /// All loaded skills in name order.
    pub fn list_skills(&self) -> Vec<&Skill> {
        self.skills.iter().collect()
    }

    /// Search skills case-insensitively by substring scoring.
    ///
    /// Each whitespace-split query term scores `occurrences × 3` in the name,
    /// `× 2` in the description, and `× 1` in the instructions. Results are
    /// ranked by score descending, then name ascending; only skills with a
    /// positive score are returned, capped at `limit`. A blank query returns
    /// every skill.
    pub fn search(&self, query: &str, limit: usize) -> Vec<&Skill> {
        if query.trim().is_empty() {
            return self.list_skills();
        }
        let terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        let mut scored: Vec<(usize, &Skill)> = self
            .skills
            .iter()
            .map(|skill| {
                let name = skill.name.to_lowercase();
                let description = skill.description.to_lowercase();
                let instructions = skill.instructions.to_lowercase();
                let score = terms.iter().fold(0usize, |acc, term| {
                    acc + name.matches(term.as_str()).count() * 3
                        + description.matches(term.as_str()).count() * 2
                        + instructions.matches(term.as_str()).count()
                });
                (score, skill)
            })
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, skill)| skill)
            .collect()
    }

    /// Read a reference file bundled with a skill; `None` if the skill or the
    /// reference does not exist.
    pub fn read_reference(&self, skill: &str, reference: &str) -> Option<&str> {
        self.skill(skill)?
            .references
            .get(reference)
            .map(String::as_str)
    }

    fn load_dir(dir: &std::path::Path) -> Result<Vec<Skill>, PluginError> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            PluginError::Io(format!("failed to read skills dir {}: {e}", dir.display()))
        })?;
        let mut skills = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                PluginError::Io(format!("failed to read an entry of {}: {e}", dir.display()))
            })?;
            let path = entry.path();
            if !path.is_dir() || !path.join("SKILL.md").is_file() {
                continue;
            }
            skills.push(Skill::from_skill_md(&path)?);
        }
        Ok(skills)
    }

    /// Deduplicate by name (later entries win) and sort by name.
    fn dedup_sorted(skills: Vec<Skill>) -> Vec<Skill> {
        let mut by_name: BTreeMap<String, Skill> = BTreeMap::new();
        for skill in skills {
            by_name.insert(skill.name.clone(), skill);
        }
        by_name.into_values().collect()
    }
}

/// Validate the `skill` tool arguments and extract the required `name`.
///
/// A missing, non-string, or blank `name` yields [`PluginError::Validation`].
/// Testable seam for the stream hook's argument validation.
pub(crate) fn parse_skill_args(arguments: &Value) -> Result<String, PluginError> {
    let name = arguments.get("name").and_then(Value::as_str).unwrap_or("");
    if name.trim().is_empty() {
        return Err(PluginError::Validation {
            schema: "skill".to_string(),
            message: "skill requires a non-empty string `name` argument".to_string(),
        });
    }
    Ok(name.to_string())
}

/// Validate the `skill_read` tool arguments and extract `skill` and `file`.
///
/// A missing, non-string, or blank `skill` or `file` yields
/// [`PluginError::Validation`]. Testable seam for the stream hook's argument
/// validation.
pub(crate) fn parse_skill_read_args(arguments: &Value) -> Result<(String, String), PluginError> {
    let skill = arguments.get("skill").and_then(Value::as_str).unwrap_or("");
    let file = arguments.get("file").and_then(Value::as_str).unwrap_or("");
    if skill.trim().is_empty() || file.trim().is_empty() {
        return Err(PluginError::Validation {
            schema: "skill_read".to_string(),
            message: "skill_read requires non-empty string `skill` and `file` arguments"
                .to_string(),
        });
    }
    Ok((skill.to_string(), file.to_string()))
}

/// Validate the `skill_search` tool arguments and extract the required `query`.
///
/// A missing, non-string, or blank (whitespace-trimmed) `query` yields
/// [`PluginError::Validation`]. Testable seam for the stream hook's argument
/// validation.
pub(crate) fn parse_skill_search_args(arguments: &Value) -> Result<String, PluginError> {
    let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
    if query.trim().is_empty() {
        return Err(PluginError::Validation {
            schema: "skill_search".to_string(),
            message: "skill_search requires a non-empty string `query` argument".to_string(),
        });
    }
    Ok(query.to_string())
}

/// Serialize a skill into the JSON object string used as a
/// [`ToolResult`](crate::types::MessageContentBlock::ToolResult) `output`:
/// `{ name, description, instructions, references: [names...] }`.
///
/// [`Skill`] fields are serializable, so the output round-trips through
/// [`serde_json`]; the fallback preserves the message rather than panicking.
pub(crate) fn format_skill(skill: &Skill) -> String {
    let references: Vec<&str> = skill.references.keys().map(String::as_str).collect();
    serde_json::to_string(&json!({
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
        "references": references,
    }))
    .unwrap_or_else(|e| format!("failed to serialize skill: {e}"))
}

/// Serialize a ranked skill list into the JSON array string used as a
/// [`ToolResult`](crate::types::MessageContentBlock::ToolResult) `output`:
/// an array of `{ name, description }` objects.
///
/// The projected shape is serializable, so the output round-trips through
/// [`serde_json`]; the fallback preserves the message rather than panicking.
pub(crate) fn format_skill_list(skills: &[&Skill]) -> String {
    let list: Vec<Value> = skills
        .iter()
        .map(|skill| {
            json!({
                "name": skill.name,
                "description": skill.description,
            })
        })
        .collect();
    serde_json::to_string(&list).unwrap_or_else(|e| format!("failed to serialize skill list: {e}"))
}

impl CucaPlugin for SkillsPlugin {
    fn name(&self) -> &'static str {
        "skills"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        if !self.config.inject_catalog || self.skills.is_empty() {
            return Ok(());
        }
        let mut catalog = String::from(
            "You have access to agent skills via three tools:\n\
             - `skill {name}`: loads a skill's full instructions by name.\n\
             - `skill_read {skill, file}`: reads a reference file bundled with a skill.\n\
             - `skill_search {query}`: finds skills by description.\n\
             Use `skill_search` first when unsure which skill applies.\n\
             Available skills:\n",
        );
        for skill in &self.skills {
            catalog.push_str(&format!("- {}: {}\n", skill.name, skill.description));
        }
        req.messages.push(UnifiedMessage::system(catalog));
        Ok(())
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        if let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        {
            let output = match name.as_str() {
                "skill" => match parse_skill_args(arguments) {
                    Ok(skill_name) => match self.skill(&skill_name) {
                        Some(skill) => Ok(format_skill(skill)),
                        None => Err(PluginError::Internal(format!(
                            "unknown skill `{skill_name}`; use `skill_search` to find an applicable skill"
                        ))),
                    },
                    Err(e) => Err(e),
                },
                "skill_read" => match parse_skill_read_args(arguments) {
                    Ok((skill_name, file)) => match self.read_reference(&skill_name, &file) {
                        Some(content) => Ok(content.to_string()),
                        None => {
                            let detail = if self.skill(&skill_name).is_none() {
                                format!("unknown skill `{skill_name}`")
                            } else {
                                format!("skill `{skill_name}` has no reference file `{file}`")
                            };
                            Err(PluginError::Internal(detail))
                        }
                    },
                    Err(e) => Err(e),
                },
                "skill_search" => match parse_skill_search_args(arguments) {
                    Ok(query) => {
                        let results = self.search(&query, self.config.max_search_results);
                        Ok(format_skill_list(&results))
                    }
                    Err(e) => Err(e),
                },
                // Unknown tools are not this plugin's responsibility; leave
                // the block untouched.
                _ => return Ok(()),
            };
            // Validation and lookup failures surface inside the ToolResult
            // output rather than failing the hook, so the model can react to
            // them in the conversation.
            let output = output.unwrap_or_else(|e| e.to_string());
            // `id` is moved, not cloned: this assignment replaces the block it
            // was borrowed from.
            *chunk = MessageContentBlock::ToolResult {
                tool_call_id: std::mem::take(id),
                output,
            };
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-skills"))]
mod tests {
    use super::*;
    use crate::request::UnifiedRequest;

    /// Removes the temp directory on drop so tests leave no files behind.
    struct TestDir(std::path::PathBuf);
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fresh_temp_dir() -> TestDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        TestDir(
            std::env::temp_dir().join(format!("cuca-skills-test-{}-{nanos}", std::process::id())),
        )
    }

    fn write_skill_md(dir: &std::path::Path, name: &str, description: &str, instructions: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!(
                "---\nname: {name}\ndescription: \"{description}\"\nlicense: MIT\nallowed-tools: [bash]\n---\n{instructions}\n"
            ),
        )
        .unwrap();
    }

    fn sample_plugin() -> SkillsPlugin {
        SkillsPlugin::inline(vec![
            Skill::inline(
                "math",
                "Arithmetic operations",
                "Add, subtract, multiply, divide.",
            ),
            Skill::inline(
                "web",
                "Fetch web pages",
                "Use HTTP requests to retrieve pages.",
            ),
        ])
    }

    fn tool_call(name: &str, arguments: Value) -> MessageContentBlock {
        MessageContentBlock::ToolCall {
            id: "call_1".to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn skill_md_parses_frontmatter_and_ignores_other_keys() {
        let guard = fresh_temp_dir();
        write_skill_md(
            &guard.0,
            "math",
            "Arithmetic operations",
            "Add and subtract.",
        );
        let skill = Skill::from_skill_md(&guard.0).unwrap();
        assert_eq!(skill.name, "math");
        assert_eq!(skill.description, "Arithmetic operations");
        assert_eq!(skill.instructions, "Add and subtract.");
        assert!(skill.references.is_empty());
    }

    #[test]
    fn skill_md_missing_name_is_error() {
        let guard = fresh_temp_dir();
        std::fs::create_dir_all(&guard.0).unwrap();
        std::fs::write(
            guard.0.join("SKILL.md"),
            "---\ndescription: \"d\"\n---\nbody\n",
        )
        .unwrap();
        let err = Skill::from_skill_md(&guard.0).unwrap_err();
        assert!(matches!(err, PluginError::Io(_)));
        assert!(err.to_string().contains("SKILL.md"), "err: {err}");
    }

    #[test]
    fn skill_md_blank_name_is_error() {
        let guard = fresh_temp_dir();
        std::fs::create_dir_all(&guard.0).unwrap();
        std::fs::write(
            guard.0.join("SKILL.md"),
            "---\nname: \"   \"\ndescription: \"d\"\n---\nbody\n",
        )
        .unwrap();
        assert!(matches!(
            Skill::from_skill_md(&guard.0),
            Err(PluginError::Io(_))
        ));
    }

    #[test]
    fn skill_md_missing_description_is_error() {
        let guard = fresh_temp_dir();
        std::fs::create_dir_all(&guard.0).unwrap();
        std::fs::write(guard.0.join("SKILL.md"), "---\nname: math\n---\nbody\n").unwrap();
        let err = Skill::from_skill_md(&guard.0).unwrap_err();
        assert!(matches!(err, PluginError::Io(_)));
        assert!(err.to_string().contains("SKILL.md"), "err: {err}");
    }

    #[test]
    fn skill_md_without_frontmatter_is_error() {
        let guard = fresh_temp_dir();
        std::fs::create_dir_all(&guard.0).unwrap();
        std::fs::write(guard.0.join("SKILL.md"), "just instructions\n").unwrap();
        let err = Skill::from_skill_md(&guard.0).unwrap_err();
        assert!(matches!(err, PluginError::Io(_)));
        assert!(err.to_string().contains("frontmatter"), "err: {err}");
    }

    #[test]
    fn references_dir_populates_reference_map() {
        let guard = fresh_temp_dir();
        write_skill_md(&guard.0, "math", "Arithmetic", "Add.");
        std::fs::create_dir_all(guard.0.join("references")).unwrap();
        std::fs::write(guard.0.join("references").join("formulas.txt"), "a+b=c").unwrap();
        std::fs::write(
            guard.0.join("references").join("constants.md"),
            "# pi = 3.14",
        )
        .unwrap();
        // A nested directory is not a regular file and must be skipped.
        std::fs::create_dir_all(guard.0.join("references").join("nested")).unwrap();
        let skill = Skill::from_skill_md(&guard.0).unwrap();
        assert_eq!(skill.references.len(), 2);
        assert_eq!(skill.references["formulas.txt"], "a+b=c");
        assert_eq!(skill.references["constants.md"], "# pi = 3.14");
    }

    #[test]
    fn from_dir_discovers_sorted_skills_and_skips_dirs_without_skill_md() {
        let guard = fresh_temp_dir();
        write_skill_md(&guard.0.join("zeta"), "zeta", "Last skill", "Z.");
        write_skill_md(&guard.0.join("alpha"), "alpha", "First skill", "A.");
        // A subdirectory without SKILL.md is skipped silently.
        std::fs::create_dir_all(guard.0.join("scratch")).unwrap();
        std::fs::write(guard.0.join("scratch").join("notes.txt"), "nope").unwrap();
        // A plain file at the top level is skipped too.
        std::fs::write(guard.0.join("README.md"), "nope").unwrap();

        let plugin = SkillsPlugin::from_dir(&guard.0).unwrap();
        let names: Vec<&str> = plugin
            .list_skills()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
    }

    #[test]
    fn inline_skills_win_on_name_conflict() {
        let guard = fresh_temp_dir();
        write_skill_md(
            &guard.0.join("math"),
            "math",
            "From directory",
            "Dir instructions.",
        );
        let plugin = SkillsPlugin::new(SkillsConfig {
            skills_dir: Some(guard.0.clone()),
            inline_skills: vec![Skill::inline("math", "From inline", "Inline instructions.")],
            ..SkillsConfig::default()
        })
        .unwrap();
        assert_eq!(plugin.list_skills().len(), 1);
        let math = plugin.skill("math").unwrap();
        assert_eq!(math.description, "From inline");
        assert_eq!(math.instructions, "Inline instructions.");
    }

    #[test]
    fn on_request_injects_tool_instructions_and_catalog() {
        let plugin = sample_plugin();
        let mut req = UnifiedRequest::new("model");
        plugin.on_request(&mut req).unwrap();
        assert_eq!(req.messages.len(), 1);
        let text = match &req.messages[0].content[0] {
            MessageContentBlock::Text(t) => t.clone(),
            other => panic!("expected text block, got {other:?}"),
        };
        assert!(text.contains("skill_search"), "text: {text}");
        assert!(text.contains("skill_read"), "text: {text}");
        assert!(text.contains("Use `skill_search` first"), "text: {text}");
        assert!(
            text.contains("- math: Arithmetic operations"),
            "text: {text}"
        );
        assert!(text.contains("- web: Fetch web pages"), "text: {text}");
    }

    #[test]
    fn on_request_skips_injection_when_disabled() {
        let plugin = SkillsPlugin::new(SkillsConfig {
            inline_skills: vec![Skill::inline("math", "Arithmetic", "Add.")],
            inject_catalog: false,
            ..SkillsConfig::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("model");
        plugin.on_request(&mut req).unwrap();
        assert!(req.messages.is_empty());
    }

    #[test]
    fn on_request_skips_injection_without_skills() {
        let plugin = SkillsPlugin::inline(Vec::new());
        let mut req = UnifiedRequest::new("model");
        plugin.on_request(&mut req).unwrap();
        assert!(req.messages.is_empty());
    }

    #[test]
    fn stream_chunk_skill_returns_serialized_skill() {
        let plugin = sample_plugin();
        let mut chunk = tool_call("skill", json!({ "name": "math" }));
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => {
                assert_eq!(tool_call_id, "call_1");
                let value: Value = serde_json::from_str(&output).unwrap();
                assert_eq!(value["name"], "math");
                assert_eq!(value["description"], "Arithmetic operations");
                assert_eq!(value["instructions"], "Add, subtract, multiply, divide.");
                assert_eq!(value["references"], json!([]));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_unknown_skill_returns_error_text() {
        let plugin = sample_plugin();
        let mut chunk = tool_call("skill", json!({ "name": "nope" }));
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult { output, .. } => {
                assert!(output.contains("unknown skill"), "output: {output}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_validation_error_goes_into_output() {
        let plugin = sample_plugin();
        let mut chunk = tool_call("skill", json!({}));
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult { output, .. } => {
                assert!(output.contains("requires"), "output: {output}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_skill_read_returns_reference_content() {
        let guard = fresh_temp_dir();
        write_skill_md(&guard.0.join("math"), "math", "Arithmetic", "Add.");
        std::fs::create_dir_all(guard.0.join("math").join("references")).unwrap();
        std::fs::write(
            guard.0.join("math").join("references").join("formulas.txt"),
            "a+b=c",
        )
        .unwrap();
        let plugin = SkillsPlugin::from_dir(&guard.0).unwrap();
        let mut chunk = tool_call(
            "skill_read",
            json!({ "skill": "math", "file": "formulas.txt" }),
        );
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult { output, .. } => assert_eq!(output, "a+b=c"),
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_skill_read_missing_reference_returns_error_text() {
        let plugin = sample_plugin();
        let mut chunk = tool_call("skill_read", json!({ "skill": "math", "file": "nope.txt" }));
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult { output, .. } => {
                assert!(output.contains("no reference"), "output: {output}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_skill_read_unknown_skill_returns_error_text() {
        let plugin = sample_plugin();
        let mut chunk = tool_call("skill_read", json!({ "skill": "nope", "file": "f.txt" }));
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult { output, .. } => {
                assert!(output.contains("unknown skill"), "output: {output}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_skill_search_returns_json_array() {
        let plugin = sample_plugin();
        let mut chunk = tool_call("skill_search", json!({ "query": "fetch" }));
        plugin.on_stream_chunk(&mut chunk).unwrap();
        match chunk {
            MessageContentBlock::ToolResult { output, .. } => {
                let value: Value = serde_json::from_str(&output).unwrap();
                let arr = value.as_array().expect("expected an array");
                assert!(!arr.is_empty());
                // `fetch` matches only the `web` skill's description.
                assert_eq!(arr[0]["name"], "web");
                assert!(arr[0].get("description").is_some());
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tool_passes_through() {
        let plugin = sample_plugin();
        let original = tool_call("some_other_tool", json!({ "x": 1 }));
        let mut chunk = original.clone();
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert_eq!(chunk, original);
    }

    #[test]
    fn parse_skill_args_requires_non_empty_name() {
        assert_eq!(
            parse_skill_args(&json!({ "name": "math" })).unwrap(),
            "math"
        );
        for bad in [json!({}), json!({ "name": "  " }), json!({ "name": 42 })] {
            assert!(matches!(
                parse_skill_args(&bad),
                Err(PluginError::Validation { .. })
            ));
        }
    }

    #[test]
    fn parse_skill_read_args_requires_skill_and_file() {
        assert_eq!(
            parse_skill_read_args(&json!({ "skill": "math", "file": "f.txt" })).unwrap(),
            ("math".to_string(), "f.txt".to_string())
        );
        for bad in [
            json!({}),
            json!({ "skill": "math" }),
            json!({ "file": "f.txt" }),
            json!({ "skill": "  ", "file": "f.txt" }),
            json!({ "skill": "math", "file": 42 }),
        ] {
            assert!(matches!(
                parse_skill_read_args(&bad),
                Err(PluginError::Validation { .. })
            ));
        }
    }

    #[test]
    fn parse_skill_search_args_requires_non_empty_query() {
        assert_eq!(
            parse_skill_search_args(&json!({ "query": "fetch" })).unwrap(),
            "fetch"
        );
        for bad in [
            json!({}),
            json!({ "query": "   " }),
            json!({ "query": true }),
        ] {
            assert!(matches!(
                parse_skill_search_args(&bad),
                Err(PluginError::Validation { .. })
            ));
        }
    }

    #[test]
    fn search_ranks_name_matches_above_description_matches() {
        let plugin = SkillsPlugin::inline(vec![
            Skill::inline("web-scraper", "Extract data", "Write scrapers."),
            Skill::inline(
                "data-extractor",
                "Web scraping with HTML",
                "Parse HTML tables.",
            ),
        ]);
        let results = plugin.search("web", 5);
        assert_eq!(results.len(), 2);
        // "web" occurs in web-scraper's name (×3) and in data-extractor's
        // description (×2), so the name match ranks first.
        assert_eq!(results[0].name, "web-scraper");
        assert_eq!(results[1].name, "data-extractor");
    }

    #[test]
    fn search_blank_query_returns_everything() {
        let plugin = sample_plugin();
        assert_eq!(plugin.search("", 1).len(), 2);
        assert_eq!(plugin.search("   ", 1).len(), 2);
    }

    #[test]
    fn search_caps_results_at_limit() {
        let plugin = sample_plugin();
        let results = plugin.search("web", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "web");
    }

    #[test]
    fn search_no_match_returns_empty() {
        let plugin = sample_plugin();
        assert!(plugin.search("xyzzy", 5).is_empty());
    }

    #[test]
    fn accessors_skill_list_and_read_reference() {
        let guard = fresh_temp_dir();
        write_skill_md(&guard.0.join("math"), "math", "Arithmetic", "Add.");
        std::fs::create_dir_all(guard.0.join("math").join("references")).unwrap();
        std::fs::write(
            guard.0.join("math").join("references").join("formulas.txt"),
            "a+b=c",
        )
        .unwrap();
        let plugin = SkillsPlugin::from_dir(&guard.0).unwrap();

        assert!(plugin.skill("math").is_some());
        assert!(plugin.skill("nope").is_none());
        let names: Vec<&str> = plugin
            .list_skills()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, vec!["math"]);
        assert_eq!(plugin.read_reference("math", "formulas.txt"), Some("a+b=c"));
        assert_eq!(plugin.read_reference("math", "missing.txt"), None);
        assert_eq!(plugin.read_reference("nope", "formulas.txt"), None);
    }

    #[test]
    fn name_is_skills() {
        assert_eq!(sample_plugin().name(), "skills");
    }

    #[test]
    fn plugin_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SkillsPlugin>();
    }
}
