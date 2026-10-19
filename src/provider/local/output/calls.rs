//! Parsers for the tool-call bodies a model writes between its markers.
//!
//! [`parse_tool_call`] reads the custom `call:name{key:<|"|>value<|"|>}` grammar;
//! [`parse_json_calls`] reads `{"name": ..., "arguments"|"parameters": ...}`
//! objects, alone or in a JSON list; [`parse_xml_call`] reads the
//! `<function=name><parameter=key>value</parameter></function>` blocks of
//! Qwen 3.5. All return a description of the first
//! problem on malformed input; the caller wraps it into a provider error that
//! quotes the raw body.

use serde_json::{Map, Number, Value};

/// Opening and closing delimiter of a string value inside a custom-grammar call.
const STRING_QUOTE: &str = "<|\"|>";
/// Maximum nesting depth of dicts and arrays inside one custom-grammar call.
const MAX_DEPTH: usize = 64;

/// Parse a Gemma 4 tool-call body such as
/// `call:get_time{city:<|"|>London<|"|>}` into the tool name and a JSON object
/// of arguments.
///
/// Values are `<|"|>`-delimited raw strings (no escapes), nested dicts, arrays,
/// JSON numbers, `true`/`false`/`null`, or bare identifiers kept as strings.
/// Returns a description of the first problem on malformed input.
pub(crate) fn parse_tool_call(text: &str) -> Result<(String, Value), String> {
    let rest = text
        .trim_start()
        .strip_prefix("call:")
        .ok_or("missing `call:` prefix")?;
    let brace = rest.find('{').ok_or("missing `{` after the tool name")?;
    let name = rest[..brace].trim();
    if name.is_empty() {
        return Err("empty tool name".to_string());
    }
    let mut parser = ValueParser {
        src: rest,
        pos: brace,
    };
    let arguments = parser.dict(0)?;
    parser.skip_ws();
    if parser.pos != rest.len() {
        return Err(format!(
            "unexpected text after the call at byte {}",
            parser.pos
        ));
    }
    Ok((name.to_string(), arguments))
}

/// Opening of a function block: `<function=` then the name up to `>`.
const FUNCTION_OPEN: &str = "<function=";
/// Closing of a function block.
const FUNCTION_CLOSE: &str = "</function>";
/// Opening of a parameter element: `<parameter=` then the key up to `>`.
const PARAMETER_OPEN: &str = "<parameter=";
/// Closing of a parameter element.
const PARAMETER_CLOSE: &str = "</parameter>";

/// Parse a Qwen 3.5 tool-call body such as
/// `\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n`
/// into the tool name and a JSON object of arguments.
///
/// The template writes a string argument as its raw text and any other value
/// (number, boolean, object, array) as JSON, each between a newline after the
/// opening tag and a newline before the closing one. A value is therefore read
/// as JSON when it is valid JSON and as a string otherwise; the one newline on
/// each side is removed and nothing else is trimmed. A string that happens to
/// be valid JSON (`42`, `true`) reads back as that JSON value, the ambiguity
/// the format has.
pub(crate) fn parse_xml_call(text: &str) -> Result<(String, Value), String> {
    let rest = text
        .trim()
        .strip_prefix(FUNCTION_OPEN)
        .ok_or("missing `<function=` prefix")?;
    let (name, mut rest) = rest
        .split_once('>')
        .ok_or("missing `>` after the function name")?;
    let name = name.trim();
    if name.is_empty() {
        return Err("empty tool name".to_string());
    }
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix(FUNCTION_CLOSE) {
            if !after.trim().is_empty() {
                return Err("unexpected text after `</function>`".to_string());
            }
            return Ok((name.to_string(), Value::Object(arguments)));
        }
        let after = rest
            .strip_prefix(PARAMETER_OPEN)
            .ok_or("expected `<parameter=` or `</function>`")?;
        let (key, after) = after
            .split_once('>')
            .ok_or("missing `>` after a parameter name")?;
        let key = key.trim();
        if key.is_empty() {
            return Err("empty parameter name".to_string());
        }
        let (value, after) = after
            .split_once(PARAMETER_CLOSE)
            .ok_or_else(|| format!("parameter `{key}` is not closed"))?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let value =
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
        arguments.insert(key.to_string(), value);
        rest = after;
    }
}

/// Recursive-descent parser over the tool-call argument syntax.
struct ValueParser<'a> {
    /// Source text.
    src: &'a str,
    /// Current byte offset.
    pos: usize,
}

impl<'a> ValueParser<'a> {
    /// The unconsumed input.
    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    /// Next character without consuming it.
    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    /// Skip whitespace.
    fn skip_ws(&mut self) {
        let trimmed = self.rest().trim_start();
        self.pos = self.src.len() - trimmed.len();
    }

    /// Consume `expected` or fail.
    fn expect(&mut self, expected: char) -> Result<(), String> {
        if self.peek() == Some(expected) {
            self.pos += expected.len_utf8();
            Ok(())
        } else {
            Err(format!("expected `{expected}` at byte {}", self.pos))
        }
    }

    /// Parse `{ member (, member)* }` into a JSON object.
    fn dict(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("nesting too deep".to_string());
        }
        self.expect('{')?;
        let mut map = Map::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            let colon = self
                .rest()
                .find(':')
                .ok_or_else(|| format!("missing `:` after a key at byte {}", self.pos))?;
            let raw_key = self.rest()[..colon].trim();
            let key = raw_key
                .strip_prefix(STRING_QUOTE)
                .and_then(|k| k.strip_suffix(STRING_QUOTE))
                .unwrap_or(raw_key);
            if key.is_empty() || key.contains(['{', '}']) {
                return Err(format!("invalid key `{raw_key}`"));
            }
            self.pos += colon + 1;
            self.skip_ws();
            let value = self.value(depth)?;
            map.insert(key.to_string(), value);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                    self.skip_ws();
                }
                Some('}') => {
                    self.pos += 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(format!("expected `,` or `}}` at byte {}", self.pos)),
            }
        }
    }

    /// Parse `[ value (, value)* ]` into a JSON array.
    fn array(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("nesting too deep".to_string());
        }
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                    self.skip_ws();
                }
                Some(']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(format!("expected `,` or `]` at byte {}", self.pos)),
            }
        }
    }

    /// Parse one value: string, dict, array, number, literal or bare word.
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        let rest = self.rest();
        if let Some(body) = rest.strip_prefix(STRING_QUOTE) {
            let end = body
                .find(STRING_QUOTE)
                .ok_or_else(|| format!("unterminated string at byte {}", self.pos))?;
            self.pos += STRING_QUOTE.len() * 2 + end;
            return Ok(Value::String(body[..end].to_string()));
        }
        match self.peek() {
            Some('{') => return self.dict(depth + 1),
            Some('[') => return self.array(depth + 1),
            _ => {}
        }
        let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
        let word = rest[..end].trim_end();
        if word.is_empty() {
            return Err(format!("empty value at byte {}", self.pos));
        }
        self.pos += end;
        if is_json_number(word)
            && let Ok(number) = serde_json::from_str::<Number>(word)
        {
            return Ok(Value::Number(number));
        }
        Ok(match word {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            other => Value::String(other.to_string()),
        })
    }
}

/// Whether `word` matches `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?` in full.
fn is_json_number(word: &str) -> bool {
    let b = word.as_bytes();
    let digits = |mut i: usize| {
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        (i, i > start)
    };
    let mut i = usize::from(b.first() == Some(&b'-'));
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits(i).0,
        _ => return false,
    }
    if b.get(i) == Some(&b'.') {
        let (next, any) = digits(i + 1);
        if !any {
            return false;
        }
        i = next;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let (next, any) = digits(i);
        if !any {
            return false;
        }
        i = next;
    }
    i == b.len()
}

/// Parse a JSON tool-call body: one object or a list of objects, each
/// `{"name": ..., "arguments": ...}` (or `"parameters"`, or the OpenAI nesting
/// `{"type": "function", "function": {...}}`).
///
/// `arguments` may be an object, `null`, or a string holding a JSON object; an
/// absent one means no arguments unless `require_arguments` is set, which a
/// bare-JSON reply needs to tell a call from an ordinary JSON answer.
pub(crate) fn parse_json_calls(
    body: &str,
    require_arguments: bool,
) -> Result<Vec<(String, Value)>, String> {
    let value: Value = serde_json::from_str(body.trim()).map_err(|e| e.to_string())?;
    match value {
        Value::Array(items) if items.is_empty() => Err("empty tool-call list".to_string()),
        Value::Array(items) => items
            .iter()
            .map(|item| json_call(item, require_arguments))
            .collect(),
        other => json_call(&other, require_arguments).map(|call| vec![call]),
    }
}

/// Read one tool-call object.
///
/// The name comes from `name`, from a string `function` (`{"function": "f",
/// "arguments": {...}}`, which small models write for `name`), or from the
/// nested OpenAI shape `{"type": "function", "function": {"name": "f",
/// "arguments": {...}}}`. The arguments come from `arguments` or `parameters`,
/// in the nested object first and then beside it; they may be an object, `null`,
/// or a string holding a JSON object.
fn json_call(value: &Value, require_arguments: bool) -> Result<(String, Value), String> {
    let outer = value.as_object().ok_or("tool call is not a JSON object")?;
    let nested = outer.get("function").and_then(Value::as_object);
    let inner = nested.unwrap_or(outer);
    let name = inner
        .get("name")
        .or_else(|| outer.get("name"))
        .or_else(|| outer.get("function").filter(|f| f.is_string()))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or("missing tool name")?;
    let arguments = [inner, outer]
        .iter()
        .find_map(|o| o.get("arguments").or_else(|| o.get("parameters")));
    let arguments = match arguments {
        None if require_arguments => return Err("missing arguments".to_string()),
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(object @ Value::Object(_)) => object.clone(),
        Some(Value::String(text)) => match serde_json::from_str(text) {
            Ok(parsed @ Value::Object(_)) => parsed,
            _ => return Err("arguments string is not a JSON object".to_string()),
        },
        Some(_) => return Err("arguments are not a JSON object".to_string()),
    };
    Ok((name.to_string(), arguments))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn xml_calls_read_strings_json_and_nothing_else() {
        let body = "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n\
                    <parameter=flags>\n{\"a\": [1, true]}\n</parameter>\n\
                    <parameter=note>\nline one\n  line two <b>x</b>\n</parameter>\n\
                    </function>\n";
        let (name, args) = parse_xml_call(body).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(
            args,
            json!({
                "city": "Paris",
                "days": 3,
                "flags": { "a": [1, true] },
                "note": "line one\n  line two <b>x</b>",
            })
        );
        // No parameters, and values keep inner whitespace and one-sided newlines.
        assert_eq!(
            parse_xml_call("<function=ping>\n</function>").unwrap(),
            ("ping".to_string(), json!({}))
        );
        assert_eq!(
            parse_xml_call("<function=f><parameter=a>  x  </parameter></function>")
                .unwrap()
                .1,
            json!({ "a": "  x  " })
        );
    }

    #[test]
    fn malformed_xml_calls_are_rejected() {
        for bad in [
            "",
            "get_weather",
            "<function=>\n</function>",
            "<function=f>\n<parameter=a>\nx\n</function>",
            "<function=f>\n<parameter=>\nx\n</parameter>\n</function>",
            "<function=f>\n<parameter=a>\nx\n</parameter>\n",
            "<function=f>\n</function>\ntrailing",
            "<function=f>\nstray\n</function>",
        ] {
            assert!(parse_xml_call(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parses_tool_call_shapes() {
        let cases = [
            (
                "call:get_time{city:<|\"|>London<|\"|>}",
                "get_time",
                json!({ "city": "London" }),
            ),
            (
                "call:special_function{arg1:42}",
                "special_function",
                json!({ "arg1": 42 }),
            ),
            (
                "call:set_config{config:{theme:<|\"|>dark<|\"|>,count:3}}",
                "set_config",
                json!({ "config": { "theme": "dark", "count": 3 } }),
            ),
            (
                "call:todo_list{todos:[<|\"|>buy milk<|\"|>,<|\"|>walk dog<|\"|>]}",
                "todo_list",
                json!({ "todos": ["buy milk", "walk dog"] }),
            ),
            ("call:empty_args{}", "empty_args", json!({})),
            (
                "  call:ws { a : 1 , b : true } \n",
                "ws",
                json!({ "a": 1, "b": true }),
            ),
            (
                "call:lit{a:null,b:false,c:-1.5e3,d:bare word,e:[],f:{},g:0}",
                "lit",
                json!({ "a": null, "b": false, "c": -1500.0, "d": "bare word", "e": [], "f": {}, "g": 0 }),
            ),
            ("call:quoted{<|\"|>k<|\"|>:1}", "quoted", json!({ "k": 1 })),
            (
                "call:ident{a:1.2.3,b:01,c:42abc}",
                "ident",
                json!({ "a": "1.2.3", "b": "01", "c": "42abc" }),
            ),
        ];
        for (input, name, args) in cases {
            let (n, a) = parse_tool_call(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!((n.as_str(), a), (name, args), "input {input}");
        }
    }

    #[test]
    fn string_values_keep_structural_characters() {
        let (_, args) = parse_tool_call("call:f{s:<|\"|>a,}:\"b<|\"|>}").expect("parses");
        assert_eq!(args, json!({ "s": "a,}:\"b" }));
    }

    #[test]
    fn malformed_tool_calls_are_rejected() {
        let bad = [
            "get_time{}",
            "call:{}",
            "call:f",
            "call:f{a:1",
            "call:f{a}",
            "call:f{a:<|\"|>x}",
            "call:f{a:1}}",
            "call:f{a:1,}",
            "call:f{a:}",
            "call:f{a:[1,}",
            "call:f{:1}",
        ];
        for input in bad {
            assert!(parse_tool_call(input).is_err(), "should reject {input:?}");
        }
        let deep = format!("call:f{{a:{}{}}}", "[".repeat(200), "]".repeat(200));
        assert!(parse_tool_call(&deep).is_err());
    }

    #[test]
    fn json_calls_accept_the_common_shapes() {
        let one = parse_json_calls(
            "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n",
            false,
        )
        .expect("parses");
        assert_eq!(
            one,
            [("get_weather".to_string(), json!({ "city": "Paris" }))]
        );
        let llama = parse_json_calls("{\"name\": \"f\", \"parameters\": {\"x\": 1}}", true)
            .expect("parses");
        assert_eq!(llama, [("f".to_string(), json!({ "x": 1 }))]);
        let list = parse_json_calls(
            "[{\"name\": \"a\", \"arguments\": {}}, {\"type\": \"function\", \"function\": {\"name\": \"b\", \"arguments\": \"{\\\"k\\\": 2}\"}}]",
            false,
        )
        .expect("parses");
        assert_eq!(
            list,
            [
                ("a".to_string(), json!({})),
                ("b".to_string(), json!({ "k": 2 }))
            ]
        );
        let bare = parse_json_calls("{\"name\": \"f\"}", false).expect("parses");
        assert_eq!(bare, [("f".to_string(), json!({}))]);
    }

    #[test]
    fn json_call_name_and_argument_shapes() {
        let expect = ("get_weather".to_string(), json!({ "city": "Paris" }));
        let shapes = [
            // `name` with `arguments` or `parameters`.
            r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#,
            r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#,
            // A string `function` for the name.
            r#"{"function": "get_weather", "arguments": {"city": "Paris"}}"#,
            r#"{"function": "get_weather", "parameters": {"city": "Paris"}}"#,
            // The OpenAI shape, arguments an object or a JSON string.
            r#"{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}"#,
            r#"{"type": "function", "function": {"name": "get_weather", "parameters": {"city": "Paris"}}}"#,
            r#"{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}"#,
            // Arguments beside the nested function object.
            r#"{"function": {"name": "get_weather"}, "arguments": {"city": "Paris"}}"#,
            // Arguments as a JSON string at the top level.
            r#"{"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}"#,
        ];
        for shape in shapes {
            for require in [false, true] {
                let got = parse_json_calls(shape, require)
                    .unwrap_or_else(|e| panic!("{shape} (require {require}): {e}"));
                assert_eq!(got, std::slice::from_ref(&expect), "{shape}");
            }
        }
        // The same shapes in a list, as Granite and Qwen write them.
        let list = format!("[{}, {}]", shapes[2], shapes[4]);
        assert_eq!(
            parse_json_calls(&list, true).expect("list parses"),
            [expect.clone(), expect]
        );
        // `name` wins over a `function` string; null arguments mean none.
        assert_eq!(
            parse_json_calls(
                r#"{"name": "a", "function": "b", "arguments": null}"#,
                false
            )
            .expect("parses"),
            [("a".to_string(), json!({}))]
        );
    }

    #[test]
    fn json_calls_reject_other_shapes() {
        for bad in [
            "",
            "not json",
            "[]",
            "[1]",
            "{\"arguments\": {}}",
            "{\"name\": \"\", \"arguments\": {}}",
            "{\"name\": \"f\", \"arguments\": [1]}",
            "{\"name\": \"f\", \"arguments\": \"nope\"}",
            "{\"name\": \"f\"} trailing",
            "{\"function\": 3, \"arguments\": {}}",
            "{\"function\": {\"arguments\": {}}}",
            "{\"function\": \"\", \"arguments\": {}}",
        ] {
            assert!(
                parse_json_calls(bad, false).is_err(),
                "should reject {bad:?}"
            );
        }
        assert!(parse_json_calls("{\"name\": \"f\"}", true).is_err());
    }
}
