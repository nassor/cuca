//! The KDL reader: a KDL document becomes a [`ConfigValue`] tree, which any
//! `Deserialize` configuration struct reads with the serde attributes it
//! already has.
//!
//! # Mapping
//!
//! The mapping is schema blind: it knows no key names, so the same seven rules
//! apply to every section.
//!
//! 1. A document, and every children block, is a table. Each node in it
//!    contributes one entry keyed by the node's name.
//! 2. Properties (`key=value`) are scalar entries of the node's own table.
//!    Naming one property twice on a node is an error, not KDL's rightmost
//!    wins override: a configuration that says one thing twice is a mistake.
//! 3. A node with arguments only holds a scalar when it has one argument and
//!    an array of scalars when it has two or more.
//! 4. A node with properties or children is a table built from both. One
//!    leading argument, if present, is stored under the key `id`.
//! 5. Sibling nodes sharing a name collapse into an array of their values, in
//!    document order. Any other key collision is an error naming the key.
//! 6. A node with nothing on it is an empty table.
//! 7. Numbers map to `i64`, `u64` or `f64`. Type annotations, `#null`, `#inf`
//!    and `#nan` are errors: the mapping is schema blind, so silently dropping
//!    syntax would hide a typo.
//!
//! Rule 5 needs two occurrences to build an array, so a list-valued field
//! written once is a single value. Such fields carry
//! [`one_or_many`](super::one_or_many), which accepts both shapes.
//!
//! # Variables
//!
//! [`load_kdl_str`] and [`load_kdl_file`] substitute placeholders in the raw
//! text before the parser sees it: `${VAR}` is replaced with the value of
//! `VAR`, and `${VAR:-default}` falls back to `default` when `VAR` is unset.
//! A name is looked up in the file's own top-level `variables { NAME "value" }`
//! block first, then in the process environment; this is the only place cuca
//! reads the environment for configuration, and only for names the file
//! writes. A declared value may reference other names; a cycle is an error.
//! A value is inserted verbatim, so one holding a backslash or a double quote
//! lands as syntax inside a KDL quoted string and the line stops parsing; that
//! error names the variables whose values were inserted and which of them
//! carried such a character, never the value itself.

use std::collections::HashMap;
use std::path::Path;

use kdl::{KdlDocument, KdlEntry, KdlNode, KdlValue};
use serde::de::DeserializeOwned;

use super::ConfigValue;
use crate::error::CucaError;

/// The table form of a [`ConfigValue`].
type ConfigMap = serde_json::Map<String, ConfigValue>;

/// A variable lookup outside the file: the process environment, or a fixed
/// list in tests.
type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

/// The process environment as an [`Env`].
fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Parse KDL `text` into a [`ConfigValue`], always a table at the top level.
/// No placeholder is substituted.
///
/// # Errors
///
/// [`CucaError::Config`] when the document is not valid KDL, or when it
/// violates one of the mapping rules. Every message opens with `parsing KDL:
/// line:column:`, then a parser diagnostic or the offending node and key.
pub fn from_kdl_str(text: &str) -> Result<ConfigValue, CucaError> {
    let document = KdlDocument::parse(text).map_err(|error| {
        CucaError::Config(format!("parsing KDL: {}", render_parse_error(&error)))
    })?;
    let mut table = ConfigMap::new();
    extend_with_document(&mut table, &document, "document", text)?;
    Ok(ConfigValue::Object(table))
}

/// Substitute `raw`'s placeholders against `declared` (then the process
/// environment), then parse it with [`from_kdl_str`].
///
/// When the substituted text does not parse, the error also names the
/// variables whose values were inserted, and says which of them holds a
/// backslash or a double quote. A value is never printed: a variable can hold
/// a credential.
///
/// # Errors
///
/// [`CucaError::Config`] for an unclosed `${`, a placeholder that is neither
/// declared nor set and has no default, a cyclic declaration, or anything
/// [`from_kdl_str`] rejects.
pub fn from_kdl_str_with_vars(
    raw: &str,
    declared: &HashMap<String, String>,
) -> Result<ConfigValue, CucaError> {
    parse_with_vars(raw, declared, &process_env)
}

/// Read a whole configuration from KDL `text` into `T`.
///
/// The top-level `variables` block is read from the unsubstituted text,
/// every placeholder is substituted (declared names first, then the process
/// environment), the result is parsed, the `variables` key is removed, and the
/// rest deserializes into `T`.
///
/// # Errors
///
/// [`CucaError::Config`] for anything [`from_kdl_str_with_vars`] rejects, a
/// `variables` block that is not name/string pairs or declares a name outside
/// `[A-Za-z0-9_]`, and a document that does not match `T` (an unknown key
/// under `deny_unknown_fields` is named).
pub fn load_kdl_str<T: DeserializeOwned>(text: &str) -> Result<T, CucaError> {
    load_with_env(text, &process_env)
}

/// [`load_kdl_str`] over the file at `path`. A relative path inside the file
/// is not rewritten: it resolves against the working directory.
///
/// # Errors
///
/// [`CucaError::Config`] naming `path` when the file cannot be read or
/// [`load_kdl_str`] rejects it.
pub fn load_kdl_file<T: DeserializeOwned>(path: &Path) -> Result<T, CucaError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| CucaError::Config(format!("reading `{}`: {e}", path.display())))?;
    load_kdl_str(&text).map_err(|error| match error {
        CucaError::Config(m) => CucaError::Config(format!("`{}`: {m}", path.display())),
        other => other,
    })
}

fn load_with_env<T: DeserializeOwned>(text: &str, env: Env<'_>) -> Result<T, CucaError> {
    let unsubstituted = from_kdl_str(text)?;
    let declared = declared_variables(&unsubstituted)?;
    let mut value = parse_with_vars(text, &declared, env)?;
    if let ConfigValue::Object(table) = &mut value {
        table.remove("variables");
    }
    T::deserialize(value).map_err(|e| CucaError::Config(format!("configuration: {e}")))
}

/// The top-level `variables` table of an unsubstituted parse, empty when the
/// document declares none. Names are restricted to `[A-Za-z0-9_]` so a typo in
/// a `${...}` reference cannot alias a name the scanner would not recognise.
fn declared_variables(value: &ConfigValue) -> Result<HashMap<String, String>, CucaError> {
    let Some(vars) = value.get("variables") else {
        return Ok(HashMap::new());
    };
    let declared =
        serde_json::from_value::<HashMap<String, String>>(vars.clone()).map_err(|e| {
            CucaError::Config(format!("variables block must be name/string pairs: {e}"))
        })?;
    for name in declared.keys() {
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(CucaError::Config(format!(
                "invalid variable name {name:?}: use [A-Za-z0-9_]"
            )));
        }
    }
    Ok(declared)
}

fn parse_with_vars(
    raw: &str,
    declared: &HashMap<String, String>,
    env: Env<'_>,
) -> Result<ConfigValue, CucaError> {
    let substituted = substitute_vars(raw, declared, env)?;
    from_kdl_str(&substituted).map_err(|error| with_substitution_context(error, raw, declared, env))
}

/// Substitute `${VAR}` and `${VAR:-default}` placeholders in `raw` by asking
/// `resolve` for each variable name.
///
/// `Ok(Some(value))` pushes `value` into the output verbatim, never
/// re-scanned. `Ok(None)` means the name is unset: `${VAR:-default}`
/// substitutes `default` and a bare `${VAR}` is an error.
fn substitute_with(
    raw: &str,
    resolve: &mut dyn FnMut(&str) -> Result<Option<String>, CucaError>,
) -> Result<String, CucaError> {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next();

            let mut placeholder = String::new();
            let mut closed = false;
            for inner in chars.by_ref() {
                if inner == '}' {
                    closed = true;
                    break;
                }
                placeholder.push(inner);
            }

            if !closed {
                return Err(CucaError::Config(
                    "unclosed '${' in configuration: missing '}'".into(),
                ));
            }

            let (var_name, fallback) = match placeholder.find(":-") {
                Some(pos) => (&placeholder[..pos], Some(&placeholder[pos + 2..])),
                None => (placeholder.as_str(), None),
            };

            match resolve(var_name)? {
                Some(val) => result.push_str(&val),
                None => match fallback {
                    Some(default) => result.push_str(default),
                    None => {
                        return Err(CucaError::Config(format!(
                            "variable '${{{var_name}}}' is neither declared in `variables` nor set in the environment, and has no default"
                        )));
                    }
                },
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Substitute `${name}` and `${name:-default}` placeholders in `raw`, with
/// `declared` names taking precedence over `env`.
///
/// A declared value may itself contain placeholders, resolved recursively
/// under the same rule; a value that reaches itself again through the chain
/// is a cycle error. Values from `env` are inserted literally and never
/// re-expanded.
fn substitute_vars(
    raw: &str,
    declared: &HashMap<String, String>,
    env: Env<'_>,
) -> Result<String, CucaError> {
    substitute_with(raw, &mut |name| {
        resolve_declared_or_env(name, declared, env, &mut Vec::new())
    })
}

/// Resolve one placeholder name: `declared` first, then `env`.
///
/// A declared value is expanded under the same rule, with `stack` holding the
/// names currently being expanded so a chain that reaches one of them again
/// is a cycle error instead of an infinite descent.
fn resolve_declared_or_env(
    name: &str,
    declared: &HashMap<String, String>,
    env: Env<'_>,
    stack: &mut Vec<String>,
) -> Result<Option<String>, CucaError> {
    if let Some(value) = declared.get(name) {
        if stack.iter().any(|on_stack| on_stack == name) {
            return Err(CucaError::Config(format!(
                "cyclic variable reference: ${{{name}}}"
            )));
        }
        stack.push(name.to_string());
        let expanded = substitute_with(value, &mut |inner| {
            resolve_declared_or_env(inner, declared, env, stack)
        })?;
        stack.pop();
        return Ok(Some(expanded));
    }
    Ok(env(name))
}

/// One placeholder whose resolved value reached the parser.
struct Insertion {
    name: String,
    /// The value held `\`, which opens an escape sequence in a quoted string.
    backslash: bool,
    /// The value held `"`, which closes a quoted string.
    quote: bool,
}

/// Tell the reader of a failed parse that the text was substituted first, and
/// which variables carried syntax into it.
///
/// The names come from scanning `raw` a second time, which happens only here:
/// the substitution that succeeded keeps no bookkeeping, so a file that parses
/// pays nothing for a note no one reads.
fn with_substitution_context(
    error: CucaError,
    raw: &str,
    declared: &HashMap<String, String>,
    env: Env<'_>,
) -> CucaError {
    let CucaError::Config(original) = &error else {
        return error;
    };
    let mut inserted: Vec<Insertion> = Vec::new();
    let rescan = substitute_with(raw, &mut |name| {
        let resolved = resolve_declared_or_env(name, declared, env, &mut Vec::new())?;
        if let Some(value) = &resolved {
            let backslash = value.contains('\\');
            let quote = value.contains('"');
            match inserted.iter_mut().find(|entry| entry.name == name) {
                Some(entry) => {
                    entry.backslash |= backslash;
                    entry.quote |= quote;
                }
                None => inserted.push(Insertion {
                    name: name.to_string(),
                    backslash,
                    quote,
                }),
            }
        }
        Ok(resolved)
    });

    if rescan.is_err() || inserted.is_empty() {
        return error;
    }

    let names = inserted
        .iter()
        .map(|entry| format!("${{{}}}", entry.name))
        .collect::<Vec<_>>()
        .join(", ");
    let mut message =
        format!("{original}\n  note: substitution ran before the parser and inserted {names}");
    for entry in &inserted {
        let held = match (entry.backslash, entry.quote) {
            (true, true) => "a backslash and a double quote",
            (true, false) => "a backslash",
            (false, true) => "a double quote",
            (false, false) => continue,
        };
        message.push_str(&format!(
            "\n  note: ${{{}}} holds {held}, which KDL reads as syntax inside a quoted string",
            entry.name
        ));
    }
    CucaError::Config(message)
}

/// Add every node of `document` to `table`, collapsing same-named siblings.
///
/// `owner` names the enclosing scope in an error message and `input` is the
/// whole document text, which every error resolves its `line:column` against.
fn extend_with_document(
    table: &mut ConfigMap,
    document: &KdlDocument,
    owner: &str,
    input: &str,
) -> Result<(), CucaError> {
    // Grouped in document order in a Vec rather than a map: a node has a
    // handful of children, and the linear scan keeps declaration order.
    let mut groups: Vec<(&str, Vec<ConfigValue>, usize)> = Vec::new();

    for node in document.nodes() {
        let name = node.name().value();
        let value = node_to_value(node, input)?;
        match groups.iter_mut().find(|(key, _, _)| *key == name) {
            Some((_, values, _)) => values.push(value),
            None => groups.push((name, vec![value], node.span().offset())),
        }
    }

    for (name, values, offset) in groups {
        if table.contains_key(name) {
            return Err(duplicate_key(owner, name, input, offset));
        }
        let value = match <[ConfigValue; 1]>::try_from(values) {
            Ok([single]) => single,
            Err(many) => ConfigValue::Array(many),
        };
        table.insert(name.to_string(), value);
    }

    Ok(())
}

/// Map one node to its value under rules 2 through 6.
fn node_to_value(node: &KdlNode, input: &str) -> Result<ConfigValue, CucaError> {
    let name = node.name().value();
    let owner = format!("node \"{name}\"");
    let node_at = node.span().offset();

    if let Some(ty) = node.ty() {
        return Err(config_error(
            input,
            node_at,
            format!(
                "{owner}: type annotation '({})' is not a configuration value",
                ty.value()
            ),
        ));
    }

    let mut arguments: Vec<&KdlEntry> = Vec::new();
    let mut properties: Vec<(&str, &KdlEntry)> = Vec::new();
    // Iterating entries rather than calling `KdlNode::get`: the parser retains
    // every duplicate property and `get` returns only the last, which would
    // hide the repetition rule 2 rejects.
    for entry in node.entries() {
        if let Some(ty) = entry.ty() {
            return Err(config_error(
                input,
                entry.span().offset(),
                format!(
                    "{owner}: type annotation '({})' is not a configuration value",
                    ty.value()
                ),
            ));
        }
        match entry.name() {
            None => arguments.push(entry),
            Some(key) => properties.push((key.value(), entry)),
        }
    }

    let children = node.children();
    if properties.is_empty() && children.is_none() {
        return match arguments.len() {
            0 => Ok(ConfigValue::Object(ConfigMap::new())),
            1 => scalar_to_value(arguments[0], &owner, input),
            _ => arguments
                .into_iter()
                .map(|entry| scalar_to_value(entry, &owner, input))
                .collect::<Result<Vec<_>, _>>()
                .map(ConfigValue::Array),
        };
    }

    let mut table = ConfigMap::new();

    if arguments.len() > 1 {
        return Err(config_error(
            input,
            node_at,
            format!(
                "{owner}: a node with properties or children takes at most one leading argument, \
                 found {}",
                arguments.len()
            ),
        ));
    }
    if let Some(argument) = arguments.first() {
        table.insert("id".to_string(), scalar_to_value(argument, &owner, input)?);
    }

    for (key, entry) in properties {
        if table.contains_key(key) {
            return Err(duplicate_key(&owner, key, input, entry.span().offset()));
        }
        table.insert(key.to_string(), scalar_to_value(entry, &owner, input)?);
    }

    if let Some(document) = children {
        extend_with_document(&mut table, document, &owner, input)?;
    }

    Ok(ConfigValue::Object(table))
}

/// Map one KDL scalar under rule 7.
fn scalar_to_value(entry: &KdlEntry, owner: &str, input: &str) -> Result<ConfigValue, CucaError> {
    let at = entry.span().offset();
    match entry.value() {
        KdlValue::String(text) => Ok(ConfigValue::String(text.clone())),
        KdlValue::Bool(flag) => Ok(ConfigValue::Bool(*flag)),
        KdlValue::Integer(number) => integer_to_value(*number, owner, input, at),
        // `Number::from_f64` rejects exactly the non-finite floats KDL spells
        // `#inf`, `#-inf` and `#nan`.
        KdlValue::Float(number) => serde_json::Number::from_f64(*number)
            .map(ConfigValue::Number)
            .ok_or_else(|| {
                config_error(
                    input,
                    at,
                    format!("{owner}: float '{number}' is not representable in configuration"),
                )
            }),
        KdlValue::Null => Err(config_error(
            input,
            at,
            format!("{owner}: #null is not a configuration value; omit the key instead"),
        )),
    }
}

fn integer_to_value(
    number: i128,
    owner: &str,
    input: &str,
    at: usize,
) -> Result<ConfigValue, CucaError> {
    if let Ok(signed) = i64::try_from(number) {
        return Ok(ConfigValue::Number(signed.into()));
    }
    if let Ok(unsigned) = u64::try_from(number) {
        return Ok(ConfigValue::Number(unsigned.into()));
    }
    Err(config_error(
        input,
        at,
        format!("{owner}: integer '{number}' does not fit in 64 bits"),
    ))
}

fn duplicate_key(owner: &str, key: &str, input: &str, at: usize) -> CucaError {
    config_error(input, at, format!("{owner}: duplicate key \"{key}\""))
}

/// A mapping-rule violation, positioned the same way a parser diagnostic is.
fn config_error(input: &str, at: usize, message: String) -> CucaError {
    let (line, column) = line_column(input, at);
    CucaError::Config(format!("parsing KDL: {line}:{column}: {message}"))
}

/// Render every parser diagnostic as `line:column: message (help)`.
///
/// `KdlError` alone prints "Failed to parse KDL document", and its detail
/// lives in the diagnostics, whose spans are byte offsets into the input.
fn render_parse_error(error: &kdl::KdlError) -> String {
    let input = error.input.as_str();
    let mut rendered = String::new();

    for diagnostic in &error.diagnostics {
        if !rendered.is_empty() {
            rendered.push_str("; ");
        }
        let (line, column) = line_column(input, diagnostic.span.offset());
        let message = diagnostic.message.as_deref().unwrap_or("invalid syntax");
        rendered.push_str(&format!("{line}:{column}: {message}"));
        if let Some(help) = &diagnostic.help {
            rendered.push_str(&format!(" ({help})"));
        }
    }

    if rendered.is_empty() {
        rendered.push_str("invalid KDL document");
    }
    rendered
}

fn line_column(input: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut column = 1;
    for (index, ch) in input.char_indices() {
        if index >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> ConfigValue {
        from_kdl_str(text).expect("valid config")
    }

    fn message(error: CucaError) -> String {
        match error {
            CucaError::Config(m) => m,
            other => panic!("expected a configuration error, got {other:?}"),
        }
    }

    fn error(text: &str) -> String {
        message(from_kdl_str(text).expect_err("invalid config"))
    }

    /// An environment holding exactly `pairs`.
    fn fake_env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn a_single_argument_node_is_a_scalar() {
        let value = parse("mode \"standalone\"\nid 7\nenabled #true\nratio 0.5\n");
        assert_eq!(value["mode"], "standalone");
        assert_eq!(value["id"], 7);
        assert_eq!(value["enabled"], true);
        assert_eq!(value["ratio"], 0.5);
    }

    #[test]
    fn several_arguments_on_one_node_are_an_array() {
        let value = parse("servers \"http://a:8000\" \"http://b:8000\"\nbackoff_ms 100 500\n");
        assert_eq!(
            value["servers"],
            serde_json::json!(["http://a:8000", "http://b:8000"])
        );
        assert_eq!(value["backoff_ms"], serde_json::json!([100, 500]));
    }

    #[test]
    fn properties_build_the_nodes_own_table() {
        let value = parse("node id=1 name=\"cuca\" data_dir=\"/tmp/cuca\"\n");
        assert_eq!(
            value["node"],
            serde_json::json!({ "id": 1, "name": "cuca", "data_dir": "/tmp/cuca" })
        );
    }

    #[test]
    fn a_leading_argument_on_a_table_node_becomes_the_id_key() {
        let value = parse("model \"chat\" device=\"cpu\"\n");
        assert_eq!(
            value["model"],
            serde_json::json!({ "id": "chat", "device": "cpu" })
        );
    }

    #[test]
    fn children_nest_as_tables_under_their_node_name() {
        let value = parse(
            r#"
model "chat" device="cpu" {
    sampling temperature=0.5 {
        nested flag=#true
    }
}
"#,
        );
        assert_eq!(
            value["model"],
            serde_json::json!({
                "id": "chat",
                "device": "cpu",
                "sampling": {
                    "temperature": 0.5,
                    "nested": { "flag": true },
                },
            })
        );
    }

    #[test]
    fn same_named_siblings_collapse_into_an_array_in_document_order() {
        let value = parse(
            r#"
part "a.gguf" sha256="x"
part "b.gguf"
"#,
        );
        assert_eq!(
            value["part"],
            serde_json::json!([
                { "id": "a.gguf", "sha256": "x" },
                "b.gguf",
            ])
        );
    }

    #[test]
    fn one_sibling_stays_a_single_value_that_one_or_many_accepts() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Spec {
            #[serde(deserialize_with = "crate::config::one_or_many")]
            peer: Vec<Peer>,
        }
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Peer {
            id: u64,
        }

        let one: Spec = serde::Deserialize::deserialize(parse("peer id=1\n")).expect("one peer");
        assert_eq!(one.peer, vec![Peer { id: 1 }]);

        let two: Spec =
            serde::Deserialize::deserialize(parse("peer id=1\npeer id=2\n")).expect("two peers");
        assert_eq!(two.peer, vec![Peer { id: 1 }, Peer { id: 2 }]);
    }

    #[test]
    fn a_bare_node_is_an_empty_table() {
        let value = parse("pipeline\nhttp {}\n");
        assert_eq!(value["pipeline"], serde_json::json!({}));
        assert_eq!(value["http"], serde_json::json!({}));
    }

    #[test]
    fn an_integer_above_i64_maps_to_u64() {
        let value = parse("big 18446744073709551615\n");
        assert_eq!(value["big"], serde_json::json!(18446744073709551615u64));
    }

    #[test]
    fn a_repeated_property_is_an_error_naming_the_key() {
        assert_eq!(
            error("config path=\"a.csv\" path=\"b.csv\"\n"),
            "parsing KDL: 1:21: node \"config\": duplicate key \"path\""
        );
    }

    #[test]
    fn a_leading_argument_beside_an_id_property_is_a_duplicate_key() {
        assert_eq!(
            error("source \"a\" id=\"b\"\n"),
            "parsing KDL: 1:12: node \"source\": duplicate key \"id\""
        );
    }

    #[test]
    fn two_leading_arguments_on_a_table_node_are_an_error() {
        assert_eq!(
            error("source \"a\" \"b\" type=\"FileSource\"\n"),
            "parsing KDL: 1:1: node \"source\": a node with properties or children takes at most \
             one leading argument, found 2"
        );
    }

    #[test]
    fn a_property_colliding_with_a_child_node_is_an_error() {
        assert_eq!(
            error("config format=\"csv\" {\n    format \"ndjson\"\n}\n"),
            "parsing KDL: 2:5: node \"config\": duplicate key \"format\""
        );
    }

    #[test]
    fn null_is_rejected_with_the_advice_to_omit_the_key() {
        assert_eq!(
            error("sha256 #null\n"),
            "parsing KDL: 1:8: node \"sha256\": #null is not a configuration value; omit the \
             key instead"
        );
    }

    #[test]
    fn a_non_finite_float_is_rejected() {
        assert_eq!(
            error("top_p #inf\n"),
            "parsing KDL: 1:7: node \"top_p\": float 'inf' is not representable in configuration"
        );
    }

    #[test]
    fn an_integer_beyond_u64_is_rejected() {
        assert_eq!(
            error("big 18446744073709551616\n"),
            "parsing KDL: 1:5: node \"big\": integer '18446744073709551616' does not fit in 64 bits"
        );
    }

    #[test]
    fn a_type_annotation_is_rejected() {
        assert_eq!(
            error("node id=(u8)1\n"),
            "parsing KDL: 1:6: node \"node\": type annotation '(u8)' is not a configuration value"
        );
        assert_eq!(
            error("(person)node id=1\n"),
            "parsing KDL: 1:1: node \"node\": type annotation '(person)' is not a configuration value"
        );
    }

    #[test]
    fn a_malformed_document_reports_line_and_column() {
        let message = error("mode \"standalone\nnode id=1\n");
        assert!(message.starts_with("parsing KDL: 1:6:"), "got: {message}");
    }

    #[test]
    fn a_slashdash_elides_the_node_it_precedes() {
        let value = parse("/-source \"gone\" type=\"FileSource\"\nsink \"kept\"\n");
        assert!(value.get("source").is_none());
        assert_eq!(value["sink"], "kept");
    }

    #[test]
    fn env_vars_substitute_before_the_parser_sees_the_text() {
        let env = fake_env(&[("CUCA_TEST_KDL_DIR", "/var/lib/cuca")]);
        let raw = "node data_dir=\"${CUCA_TEST_KDL_DIR}\" name=\"${CUCA_TEST_KDL_NAME:-cuca}\"\n";
        let value = parse(&substitute_vars(raw, &HashMap::new(), &env).expect("substituted"));
        assert_eq!(value["node"]["data_dir"], "/var/lib/cuca");
        assert_eq!(value["node"]["name"], "cuca");
    }

    #[test]
    fn an_unset_variable_without_a_default_is_an_error() {
        let err = substitute_vars(
            "node data_dir=\"${CUCA_TEST_KDL_UNSET}\"\n",
            &HashMap::new(),
            &no_env,
        )
        .expect_err("unset var");
        assert_eq!(
            message(err),
            "variable '${CUCA_TEST_KDL_UNSET}' is neither declared in `variables` nor set in the environment, and has no default"
        );
    }

    #[test]
    fn declared_variables_substitute_in_place_of_placeholders() {
        let declared = HashMap::from([("mod_dir".to_string(), "wasm".to_string())]);
        let out = substitute_vars(r#"node p="/tmp/${mod_dir}/x""#, &declared, &no_env)
            .expect("substituted");
        assert_eq!(out, r#"node p="/tmp/wasm/x""#);
    }

    #[test]
    fn a_declared_variable_shadows_a_same_named_env_var() {
        let env = fake_env(&[("CUCA_TEST_SHADOW", "from-env")]);
        let declared = HashMap::from([("CUCA_TEST_SHADOW".to_string(), "from-file".to_string())]);
        let out = substitute_vars("node p=\"${CUCA_TEST_SHADOW}\"", &declared, &env)
            .expect("substituted");
        assert_eq!(out, r#"node p="from-file""#);
    }

    #[test]
    fn an_undeclared_variable_falls_back_to_the_env() {
        let env = fake_env(&[("CUCA_TEST_UNDECLARED", "/env/path")]);
        let out = substitute_vars("node p=\"${CUCA_TEST_UNDECLARED}\"", &HashMap::new(), &env)
            .expect("substituted");
        assert_eq!(out, r#"node p="/env/path""#);
    }

    #[test]
    fn a_default_is_used_when_a_variable_is_nowhere_and_loses_to_a_declared_value() {
        let out = substitute_vars(
            "node p=\"${CUCA_TEST_NOWHERE:-/fallback}\"",
            &HashMap::new(),
            &no_env,
        )
        .expect("substituted");
        assert_eq!(out, r#"node p="/fallback""#);

        let declared = HashMap::from([("CUCA_TEST_NOWHERE".to_string(), "/declared".to_string())]);
        let out = substitute_vars(
            "node p=\"${CUCA_TEST_NOWHERE:-/fallback}\"",
            &declared,
            &no_env,
        )
        .expect("substituted");
        assert_eq!(out, r#"node p="/declared""#);
    }

    #[test]
    fn declared_values_expand_recursively_but_env_values_stay_literal() {
        let declared = HashMap::from([
            ("a".to_string(), "${b}/x".to_string()),
            ("b".to_string(), "2".to_string()),
        ]);
        let out = substitute_vars("node p=\"${a}\"", &declared, &no_env).expect("substituted");
        assert_eq!(out, r#"node p="2/x""#);

        let env = fake_env(&[
            ("CUCA_TEST_INNER", "expanded"),
            ("CUCA_TEST_LITERAL", "${CUCA_TEST_INNER}"),
        ]);
        let out = substitute_vars("node p=\"${CUCA_TEST_LITERAL}\"", &HashMap::new(), &env)
            .expect("substituted");
        assert_eq!(out, r#"node p="${CUCA_TEST_INNER}""#);
    }

    #[test]
    fn a_cycle_in_declared_values_is_an_error() {
        let declared = HashMap::from([("a".to_string(), "${a}".to_string())]);
        let err = message(
            substitute_vars("node p=\"${a}\"", &declared, &no_env).expect_err("self cycle"),
        );
        assert!(err.contains("cyclic variable reference"), "{err}");

        let declared = HashMap::from([
            ("a".to_string(), "${b}".to_string()),
            ("b".to_string(), "${a}".to_string()),
        ]);
        let err = message(
            substitute_vars("node p=\"${a}\"", &declared, &no_env).expect_err("mutual cycle"),
        );
        assert!(err.contains("cyclic variable reference"), "{err}");
    }

    #[test]
    fn a_substituted_backslash_names_its_variable_and_never_its_value() {
        let declared = HashMap::from([("out_dir".to_string(), r"C:\Users\me\out".to_string())]);
        let err = parse_with_vars(
            "sink \"orders\" {\n    config path=\"${out_dir}/orders.csv\"\n}\n",
            &declared,
            &no_env,
        )
        .expect_err("a backslash breaks the quoted string");

        let message = message(err);
        assert!(message.contains("${out_dir}"), "got: {message}");
        assert!(message.contains("backslash"), "got: {message}");
        assert!(!message.contains(r"C:\"), "leaked the value: {message}");
        assert!(!message.contains("Users"), "leaked the value: {message}");
    }

    #[derive(Debug, serde::Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Service {
        listen: String,
        api_key: Option<String>,
    }

    #[test]
    fn load_kdl_str_reads_variables_and_drops_the_block() {
        let env = fake_env(&[("CUCA_TEST_HOST", "0.0.0.0"), ("KEY", "from-env")]);
        let text = r#"
variables {
    KEY "from-file"
}
listen "${CUCA_TEST_HOST}:8000"
api_key "${KEY}"
"#;
        let service: Service = load_with_env(text, &env).expect("loads");
        assert_eq!(
            service,
            Service {
                listen: "0.0.0.0:8000".into(),
                api_key: Some("from-file".into()),
            }
        );

        let err = message(
            load_with_env::<Service>("variables {\n    \"a-b\" \"x\"\n}\nlisten \"x\"\n", &no_env)
                .unwrap_err(),
        );
        assert!(err.contains("invalid variable name"), "{err}");
    }

    #[test]
    fn load_kdl_str_names_an_unknown_key() {
        let err = message(
            load_with_env::<Service>("listen \"x\"\napi_kee \"k\"\n", &no_env).unwrap_err(),
        );
        assert!(err.starts_with("configuration: "), "{err}");
        assert!(err.contains("api_kee"), "{err}");
    }

    #[test]
    fn load_kdl_file_names_the_file() {
        let err = message(
            load_kdl_file::<Service>(Path::new("target/no-such-cuca-config.kdl")).unwrap_err(),
        );
        assert!(
            err.starts_with("reading `target/no-such-cuca-config.kdl`: "),
            "{err}"
        );
    }
}
