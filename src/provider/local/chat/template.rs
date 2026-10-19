//! Rendering of the GGUF's own Jinja chat template.
//!
//! [`ChatTemplate`] compiles `tokenizer.chat_template` with minijinja in the
//! environment HuggingFace's `apply_chat_template` uses: `trim_blocks` and
//! `lstrip_blocks` on, Python string methods through the `pycompat` callback,
//! and the helpers templates in the wild rely on:
//!
//! - `raise_exception(message)`, surfaced as a render error;
//! - `strftime_now(format)`, the current UTC time (`%Y %m %d %e %H %M %S %y %b
//!   %B %a %A %j %p %I %Z %z %%`);
//! - a `tojson` filter with Python's `json.dumps(ensure_ascii=False)` layout
//!   (`", "` / `": "` separators, `indent`, `sort_keys`, `ensure_ascii`
//!   keywords) instead of minijinja's compact, HTML-escaped one, so tool
//!   declarations are byte-identical to what the model saw in training. It
//!   keeps the order of an object's keys (minijinja's own maps are sorted,
//!   Python's `dict` keeps insertion order): the `tools` variable is built from
//!   insertion-ordered objects, in the OpenAI order a Python caller writes
//!   (`type`, then `function` with `name`, `description`, `parameters`). The
//!   key order *inside* a tool's JSON schema is the order of the request's
//!   `serde_json` map: sorted by key, unless the application enables
//!   `serde_json`'s `preserve_order`, in which case it is the order written;
//! - a `string` filter that renders maps, lists, `None` and booleans the way
//!   Python's `str()` does (`{'a': 1}`), as templates that interpolate a tool
//!   dict with `| string` expect;
//! - the HuggingFace-only `{% generation %}` / `{% endgeneration %}` tags
//!   (assistant-mask markers with no output) are rewritten to a no-op
//!   `{% if true %}` block before compilation.
//!
//! The template text is otherwise never modified, and nothing in this module is
//! specific to one model family.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::value::{Enumerator, Kwargs, Object, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value};

use super::ChatPrompt;
use crate::error::CucaError;
use crate::types::ProviderEndpoint;

/// Name under which the template is registered in the environment.
const TEMPLATE_NAME: &str = "chat";

/// Build a local-provider error.
fn local_err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

/// Wrap a minijinja error as a provider error.
fn template_err(e: Error) -> CucaError {
    local_err(format!("chat template: {e}"))
}

/// A compiled chat template.
pub(crate) struct ChatTemplate {
    /// Environment owning the compiled template.
    env: Environment<'static>,
    /// Text of the BOS token (`bos_token` in the template), or empty.
    bos_token: String,
    /// Text of the EOS token (`eos_token` in the template), or empty.
    eos_token: String,
    /// Whether the template reads `enable_thinking` (or the `thinking`
    /// variable some templates use instead, Granite's), i.e. whether a request
    /// can switch its reasoning on or off.
    reads_enable_thinking: bool,
}

impl ChatTemplate {
    /// Compile `source`, the Jinja chat template, for rendering with the given
    /// BOS and EOS token texts.
    ///
    /// Fails when the source does not parse.
    pub(crate) fn new(
        source: String,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, CucaError> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_filter("tojson", tojson);
        env.add_filter("string", py_string);
        let reads_enable_thinking =
            source.contains("enable_thinking") || reads_variable(&source, "thinking");
        env.add_template_owned(TEMPLATE_NAME, strip_generation_tags(&source))
            .map_err(template_err)?;
        Ok(Self {
            env,
            bos_token,
            eos_token,
            reads_enable_thinking,
        })
    }

    /// Whether the template reads `enable_thinking`.
    pub(crate) fn reads_enable_thinking(&self) -> bool {
        self.reads_enable_thinking
    }

    /// Render `prompt` to the text fed to the tokenizer, with the generation
    /// prompt requested (`add_generation_prompt = true`).
    ///
    /// `tools` is defined only when the request declares tools, as templates
    /// distinguish `tools is none` from an empty list.
    pub(crate) fn render(&self, prompt: &ChatPrompt) -> Result<String, CucaError> {
        self.render_at(prompt, unix_now())
    }

    /// [`ChatTemplate::render`] with `strftime_now` pinned to `unix_secs`.
    pub(crate) fn render_at(
        &self,
        prompt: &ChatPrompt,
        unix_secs: i64,
    ) -> Result<String, CucaError> {
        let template = self.env.get_template(TEMPLATE_NAME).map_err(template_err)?;
        let mut vars: BTreeMap<&str, Value> = BTreeMap::new();
        vars.insert("messages", Value::from_serialize(&prompt.messages));
        if let Some(tools) = prompt.tools.as_array().filter(|tools| !tools.is_empty()) {
            vars.insert(
                "tools",
                Value::from(tools.iter().map(tool_value).collect::<Vec<_>>()),
            );
        }
        vars.insert("add_generation_prompt", Value::from(true));
        vars.insert("enable_thinking", Value::from(prompt.enable_thinking));
        vars.insert("thinking", Value::from(prompt.enable_thinking));
        vars.insert("preserve_thinking", Value::from(false));
        vars.insert("bos_token", Value::from(self.bos_token.as_str()));
        vars.insert("eos_token", Value::from(self.eos_token.as_str()));
        vars.insert(
            "strftime_now",
            Value::from_function(move |format: String| strftime(&format, unix_secs)),
        );
        template.render(Value::from(vars)).map_err(template_err)
    }
}

/// A JSON object whose keys keep their order, which minijinja's own maps (sorted
/// by key) do not.
#[derive(Debug)]
struct OrderedMap(Vec<(String, Value)>);

impl Object for OrderedMap {
    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Values(
            self.0
                .iter()
                .map(|(k, _)| Value::from(k.as_str()))
                .collect(),
        )
    }
}

/// `json` as a template value whose objects keep the order of `serde_json`'s
/// map.
fn json_to_value(json: &serde_json::Value) -> Value {
    ordered_object(json, &[], |_, child| json_to_value(child))
}

/// A JSON object as an [`OrderedMap`], the keys of `head` first (those it has,
/// in that order) and the others after them in the map's own order; each value
/// goes through `child`. Anything else converts as a plain value.
fn ordered_object(
    json: &serde_json::Value,
    head: &[&str],
    child: impl Fn(&str, &serde_json::Value) -> Value,
) -> Value {
    match json {
        serde_json::Value::Object(map) => {
            let first = head.iter().filter_map(|key| map.get_key_value(*key));
            let rest = map.iter().filter(|(key, _)| !head.contains(&key.as_str()));
            Value::from_object(OrderedMap(
                first
                    .chain(rest)
                    .map(|(key, value)| (key.clone(), child(key, value)))
                    .collect(),
            ))
        }
        serde_json::Value::Array(items) => {
            Value::from(items.iter().map(json_to_value).collect::<Vec<_>>())
        }
        serde_json::Value::String(text) => Value::from(text.as_str()),
        serde_json::Value::Bool(flag) => Value::from(*flag),
        serde_json::Value::Number(number) => Value::from_serialize(number),
        serde_json::Value::Null => Value::from(()),
    }
}

/// One tool declaration in the order OpenAI documents and Python callers write:
/// `{"type", "function": {"name", "description", "parameters"}}`.
fn tool_value(tool: &serde_json::Value) -> Value {
    ordered_object(tool, &["type", "function"], |key, child| {
        if key == "function" {
            ordered_object(child, &["name", "description", "parameters"], |_, v| {
                json_to_value(v)
            })
        } else {
            json_to_value(child)
        }
    })
}

/// Whether a `{% %}` or `{{ }}` block of `source` reads the variable `name`
/// (outside string literals, as a whole identifier that is not an attribute).
fn reads_variable(source: &str, name: &str) -> bool {
    let mut rest = source;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let Some(kind) = after.chars().next().filter(|c| matches!(c, '%' | '{')) else {
            rest = after;
            continue;
        };
        let close = if kind == '%' { "%}" } else { "}}" };
        let Some(end) = after.find(close) else {
            return false;
        };
        let mut code = String::new();
        let mut quote: Option<char> = None;
        for c in after[1..end].chars() {
            match quote {
                Some(q) if c == q => quote = None,
                Some(_) => {}
                None if c == '\'' || c == '"' => quote = Some(c),
                None => code.push(c),
            }
        }
        let bytes = code.as_bytes();
        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        if code.match_indices(name).any(|(at, _)| {
            let before = at.checked_sub(1).map(|i| bytes[i]);
            let next = bytes.get(at + name.len()).copied();
            !before.is_some_and(|b| is_ident(b) || b == b'.') && !next.is_some_and(is_ident)
        }) {
            return true;
        }
        rest = &after[end + close.len()..];
    }
    false
}

/// Seconds since the Unix epoch (0 when the clock is before it).
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// Rewrite HuggingFace's `{% generation %}` / `{% endgeneration %}` assistant
/// mask tags to an equivalent no-output `if` block, keeping the whitespace
/// control markers (`-`) of the original tag.
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];
        let Some(close) = after.find("%}") else {
            out.push_str(&rest[open..]);
            return out;
        };
        let inner = &after[..close];
        let core = inner.trim_matches(['-', ' ', '\t', '\r', '\n']);
        let replacement = match core {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        match replacement {
            Some(keyword) => {
                let left = if inner.starts_with('-') { "-" } else { "" };
                let right = if inner.ends_with('-') { "-" } else { "" };
                let _ = write!(out, "{{%{left} {keyword} {right}%}}");
            }
            None => out.push_str(&rest[open..open + 2 + close + 2]),
        }
        rest = &after[close + 2..];
    }
    out.push_str(rest);
    out
}

/// The `tojson` filter: `json.dumps` with Python's default separators.
fn tojson(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let mut out = String::new();
    JsonDumper {
        indent,
        sort_keys: sort_keys.unwrap_or(false),
        ensure_ascii: ensure_ascii.unwrap_or(false),
    }
    .dump(&value, 0, &mut out)?;
    Ok(out)
}

/// The `string` filter: Python `str()` of the value.
fn py_string(value: Value) -> String {
    match value.kind() {
        ValueKind::String => value.as_str().unwrap_or_default().to_string(),
        _ => {
            let mut out = String::new();
            py_repr(&value, &mut out);
            out
        }
    }
}

/// Append Python's `repr()` of `value`.
fn py_repr(value: &Value, out: &mut String) {
    match value.kind() {
        ValueKind::None | ValueKind::Undefined => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::String => py_repr_str(value.as_str().unwrap_or_default(), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            if let Ok(items) = value.try_iter() {
                for (i, item) in items.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_repr(&item, out);
                }
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            if let Ok(keys) = value.try_iter() {
                for (i, key) in keys.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_repr(&key, out);
                    out.push_str(": ");
                    py_repr(&value.get_item(&key).unwrap_or(Value::UNDEFINED), out);
                }
            }
            out.push('}');
        }
        _ => out.push_str(&value.to_string()),
    }
}

/// Append Python's `repr()` of a string: single quotes unless the text holds a
/// single quote and no double quote.
fn py_repr_str(text: &str, out: &mut String) {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c < ' ' || c == '\u{7f}' => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// The error of a value `tojson` cannot write.
fn unserializable(cause: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("cannot serialize value to JSON: {cause}"),
    )
}

/// Python-`json.dumps`-style serializer.
struct JsonDumper {
    /// Spaces per nesting level; `None` keeps everything on one line.
    indent: Option<usize>,
    /// Emit object keys in sorted order.
    sort_keys: bool,
    /// Escape every non-ASCII character as `\uXXXX`.
    ensure_ascii: bool,
}

impl JsonDumper {
    /// Append `value` at nesting `level`; objects keep their enumeration order
    /// unless `sort_keys` is set.
    fn dump(&self, value: &Value, level: usize, out: &mut String) -> Result<(), Error> {
        match value.kind() {
            ValueKind::None | ValueKind::Undefined => out.push_str("null"),
            ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
            ValueKind::Number => {
                let number = serde_json::to_value(value).map_err(unserializable)?;
                out.push_str(&number.to_string());
            }
            ValueKind::String => self.string(value.as_str().unwrap_or_default(), out),
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = value.try_iter()?.collect();
                if items.is_empty() {
                    out.push_str("[]");
                    return Ok(());
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    self.separator(i == 0, level + 1, out);
                    self.dump(item, level + 1, out)?;
                }
                self.close(level, out);
                out.push(']');
            }
            ValueKind::Map => {
                let mut entries = Vec::new();
                for key in value.try_iter()? {
                    let item = value.get_item(&key)?;
                    let key = key.as_str().map_or_else(|| key.to_string(), str::to_string);
                    entries.push((key, item));
                }
                if entries.is_empty() {
                    out.push_str("{}");
                    return Ok(());
                }
                if self.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (i, (key, item)) in entries.iter().enumerate() {
                    self.separator(i == 0, level + 1, out);
                    self.string(key, out);
                    out.push_str(": ");
                    self.dump(item, level + 1, out)?;
                }
                self.close(level, out);
                out.push('}');
            }
            _ => {
                return Err(unserializable("not a JSON value"));
            }
        }
        Ok(())
    }

    /// Emit the separator before an element: `", "` inline, `","` plus a
    /// newline and indent when indenting.
    fn separator(&self, first: bool, level: usize, out: &mut String) {
        match self.indent {
            Some(width) => {
                if !first {
                    out.push(',');
                }
                out.push('\n');
                out.extend(std::iter::repeat_n(' ', width * level));
            }
            None if !first => out.push_str(", "),
            None => {}
        }
    }

    /// Emit the line break and indent before a closing bracket.
    fn close(&self, level: usize, out: &mut String) {
        if let Some(width) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * level));
        }
    }

    /// Append `text` as a JSON string literal with Python's escapes.
    fn string(&self, text: &str, out: &mut String) {
        out.push('"');
        for c in text.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if c < ' ' || (self.ensure_ascii && !c.is_ascii()) => {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        let _ = write!(out, "\\u{unit:04x}");
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// Days-since-epoch to proleptic Gregorian `(year, month, day)`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Month names, January first.
const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
/// Weekday names, Sunday first.
const WEEKDAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// Format `unix_secs` (UTC) with a C `strftime` subset; unknown directives are
/// copied verbatim.
fn strftime(format: &str, unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs = unix_secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (secs / 3600, secs % 3600 / 60, secs % 60);
    let weekday = (days + 4).rem_euclid(7) as usize;
    let day_of_year = days - days_from_civil(year, 1, 1) + 1;
    let mut out = String::with_capacity(format.len() + 8);
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(directive) = chars.next() else {
            out.push('%');
            break;
        };
        let month_name = MONTHS[(month - 1) as usize];
        let weekday_name = WEEKDAYS[weekday];
        match directive {
            'Y' => {
                let _ = write!(out, "{year}");
            }
            'y' => {
                let _ = write!(out, "{:02}", year.rem_euclid(100));
            }
            'm' => {
                let _ = write!(out, "{month:02}");
            }
            'd' => {
                let _ = write!(out, "{day:02}");
            }
            'e' => {
                let _ = write!(out, "{day:2}");
            }
            'H' => {
                let _ = write!(out, "{hour:02}");
            }
            'I' => {
                let _ = write!(out, "{:02}", (hour + 11) % 12 + 1);
            }
            'M' => {
                let _ = write!(out, "{minute:02}");
            }
            'S' => {
                let _ = write!(out, "{second:02}");
            }
            'p' => out.push_str(if hour < 12 { "AM" } else { "PM" }),
            'b' | 'h' => out.push_str(&month_name[..3]),
            'B' => out.push_str(month_name),
            'a' => out.push_str(&weekday_name[..3]),
            'A' => out.push_str(weekday_name),
            'j' => {
                let _ = write!(out, "{day_of_year:03}");
            }
            'Z' => out.push_str("UTC"),
            'z' => out.push_str("+0000"),
            '%' => out.push('%'),
            other => {
                out.push('%');
                out.push(other);
            }
        }
    }
    out
}

/// Proleptic Gregorian `(year, month, day)` to days since the epoch.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from((month + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A prompt from raw messages.
    fn prompt(messages: serde_json::Value) -> ChatPrompt {
        ChatPrompt {
            messages,
            tools: json!([]),
            enable_thinking: false,
        }
    }

    fn render(source: &str, prompt: &ChatPrompt) -> Result<String, CucaError> {
        ChatTemplate::new(source.to_string(), "<s>".into(), "</s>".into())?
            .render_at(prompt, 1_792_281_600)
    }

    #[test]
    fn invalid_template_source_is_reported() {
        let err = ChatTemplate::new("{% if %}".into(), String::new(), String::new())
            .err()
            .expect("syntax error");
        assert!(matches!(err, CucaError::Provider { .. }), "{err}");
    }

    #[test]
    fn raise_exception_surfaces_as_a_template_error() {
        let err = render("{{ raise_exception('boom') }}", &prompt(json!([]))).expect_err("raises");
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn tokens_and_generation_flags_are_defined() {
        let text = render(
            "{{ bos_token }}|{{ eos_token }}|{{ add_generation_prompt }}|{{ enable_thinking }}",
            &prompt(json!([])),
        )
        .expect("renders");
        assert_eq!(text, "<s>|</s>|True|False");
    }

    #[test]
    fn thinking_variables_are_read_from_code_not_prose() {
        let reads = |source: &str| {
            ChatTemplate::new(source.to_string(), String::new(), String::new())
                .expect("compiles")
                .reads_enable_thinking()
        };
        assert!(reads("{% if enable_thinking %}x{% endif %}"));
        assert!(reads("{% if a %}a{%- elif thinking %} b {% endif %}"));
        assert!(reads("{{ 'x' if thinking else 'y' }}"));
        assert!(!reads("{{ 'a thinking process' }} and thinking here"));
        assert!(!reads("{% if message.thinking %}x{% endif %}"));
        assert!(!reads("{% set rethinking = 1 %}"));
    }

    #[test]
    fn tools_are_undefined_unless_declared() {
        let source = "{% if tools is defined %}defined{% else %}undefined{% endif %}";
        let mut p = prompt(json!([]));
        assert_eq!(render(source, &p).expect("renders"), "undefined");
        p.tools = json!([{ "type": "function" }]);
        assert_eq!(render(source, &p).expect("renders"), "defined");
    }

    #[test]
    fn generation_tags_are_no_ops() {
        let source =
            "a{% generation %}b{% endgeneration %}c{%- generation -%} d {%- endgeneration -%}e";
        assert_eq!(
            render(source, &prompt(json!([]))).expect("renders"),
            "abcde"
        );
        assert_eq!(
            strip_generation_tags("x {%- generation -%} {% endgeneration %} {% set a = 1 %}"),
            "x {%- if true -%} {% endif %} {% set a = 1 %}"
        );
    }

    #[test]
    fn pycompat_string_methods_work() {
        let source = "{{ 'abc'.startswith('a') }}{{ '  x\\n'.strip() }}{{ 'a/b'.split('/')[-1] }}{{ '\\nq'.lstrip('\\n') }}{{ 'aXb'.replace('X', '-') }}";
        assert_eq!(
            render(source, &prompt(json!([]))).expect("renders"),
            "Truexbqa-b"
        );
    }

    #[test]
    fn qwen35_template_writes_tool_calls_as_xml() {
        let source = include_str!("../testdata/templates/qwen35.jinja");
        let prompt = ChatPrompt {
            messages: json!([
                { "role": "user", "content": "Weather?" },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": { "city": "Paris", "days": 3, "opts": { "a": [1] } },
                        },
                    }],
                },
                { "role": "tool", "content": "sunny" },
            ]),
            tools: json!([{ "type": "function", "function": { "name": "get_weather" } }]),
            enable_thinking: false,
        };
        let rendered = render(source, &prompt).expect("renders");
        assert!(rendered.contains("<tool_call>\n<function=example_function_name>"));
        assert!(
            rendered.contains(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n\
                 <parameter=city>\nParis\n</parameter>\n\
                 <parameter=days>\n3\n</parameter>\n\
                 <parameter=opts>\n{\"a\": [1]}\n</parameter>\n\
                 </function>\n</tool_call><|im_end|>\n"
            ),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n"),
            "{rendered}"
        );
    }

    #[test]
    fn tools_keep_the_openai_key_order_and_stay_ordinary_mappings() {
        let prompt = ChatPrompt {
            messages: json!([]),
            // serde_json hands the keys over sorted: wrapper and function swap.
            tools: json!([{
                "function": {
                    "parameters": { "type": "object", "properties": {} },
                    "name": "f",
                    "description": "d",
                    "strict": true,
                },
                "type": "function",
            }]),
            enable_thinking: false,
        };
        let source = "{{ tools[0] | tojson }}|{{ tools[0] is mapping }}|{{ tools[0].function.name }}|\
                      {% for k, v in tools[0].function.items() %}{{ k }},{% endfor %}|\
                      {{ tools[0].function | length }}|{{ 'type' in tools[0] }}|{{ tools | length }}";
        let rendered = render(source, &prompt).expect("renders");
        assert_eq!(
            rendered,
            "{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"d\", \
             \"parameters\": {\"properties\": {}, \"type\": \"object\"}, \"strict\": true}}|True|f|\
             name,description,parameters,strict,|4|True|1"
        );
        let sorted = render("{{ tools[0] | tojson(sort_keys=true) }}", &prompt).expect("renders");
        assert!(
            sorted.starts_with("{\"function\": {\"description\""),
            "{sorted}"
        );
    }

    #[test]
    fn tojson_matches_python_dumps() {
        let p =
            prompt(json!([{ "a": [1, 2.5, "é<>&'\n"], "b": {}, "c": [], "d": null, "e": true }]));
        let text = render("{{ messages[0] | tojson }}", &p).expect("renders");
        assert_eq!(
            text,
            "{\"a\": [1, 2.5, \"é<>&'\\n\"], \"b\": {}, \"c\": [], \"d\": null, \"e\": true}"
        );
        let indented = render("{{ messages[0].a | tojson(indent=2) }}", &p).expect("renders");
        assert_eq!(indented, "[\n  1,\n  2.5,\n  \"é<>&'\\n\"\n]");
        let ascii = render("{{ 'é😀' | tojson(ensure_ascii=true) }}", &p).expect("renders");
        assert_eq!(ascii, "\"\\u00e9\\ud83d\\ude00\"");
        let sorted =
            render("{{ {'b': 1, 'a': 2} | tojson(sort_keys=true) }}", &p).expect("renders");
        assert_eq!(sorted, "{\"a\": 2, \"b\": 1}");
    }

    #[test]
    fn string_filter_matches_python_str() {
        let p = prompt(json!([{ "a": [1, "it's", null, true], "b": { "k": "v\n" } }]));
        let text = render("{{ messages[0] | string }}", &p).expect("renders");
        assert_eq!(text, "{'a': [1, \"it's\", None, True], 'b': {'k': 'v\\n'}}");
        assert_eq!(
            render("{{ 'plain' | string }}", &p).expect("renders"),
            "plain"
        );
    }

    #[test]
    fn strftime_formats_utc_dates() {
        // 2026-10-18 05:07:09 UTC, a Sunday.
        let secs = 1_792_300_029;
        assert_eq!(strftime("%d %b %Y", secs), "18 Oct 2026");
        assert_eq!(strftime("%d %B %Y", secs), "18 October 2026");
        assert_eq!(
            strftime("%Y-%m-%d %H:%M:%S %a %A %j %p %I %e %y %%", secs),
            "2026-10-18 05:07:09 Sun Sunday 291 AM 05 18 26 %"
        );
        assert_eq!(strftime("%Q%", 0), "%Q%");
        assert_eq!(strftime("%d %b %Y", 0), "01 Jan 1970");
        assert_eq!(strftime("%Y-%m-%d", 951_782_400), "2000-02-29");
        let text = render("{{ strftime_now('%d %b %Y') }}", &prompt(json!([]))).expect("renders");
        assert_eq!(text, "18 Oct 2026");
    }
}
