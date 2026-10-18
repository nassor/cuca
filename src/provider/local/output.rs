//! Streaming parser from generated token ids to [`MessageContentBlock`]s.
//!
//! Gemma 4 interleaves plain text with two marked regions: a reasoning channel
//! (`<|channel>thought\n…<channel|>`) and tool calls
//! (`<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`). [`OutputParser`]
//! is a small state machine fed one id at a time:
//!
//! - text outside any region becomes [`MessageContentBlock::Text`];
//! - the channel header word `thought` and the whitespace after it are dropped
//!   and the channel body becomes [`MessageContentBlock::Thinking`];
//! - a tool-call body is buffered whole and parsed by [`parse_tool_call`] into
//!   [`MessageContentBlock::ToolCall`], or reported as a provider error when it
//!   is malformed.
//!
//! Token bytes are held back until they form complete UTF-8, so a character
//! split across byte tokens is emitted whole; an invalid sequence becomes
//! U+FFFD. End-of-generation ids are never pushed (the engine stops first);
//! [`OutputParser::finish`] flushes whatever is pending. Per token the parser
//! allocates only the emitted block; tool-call and channel-header buffers are
//! bounded by the generation length the engine already caps.

use serde_json::{Map, Number, Value};

use super::tokenizer::Tokenizer;
use crate::error::CucaError;
use crate::types::{MessageContentBlock, ProviderEndpoint};

/// Opening and closing delimiter of a string value inside a tool call.
const STRING_QUOTE: &str = "<|\"|>";
/// Channel header word that precedes reasoning text.
const THOUGHT: &str = "thought";
/// Maximum nesting depth of dicts and arrays inside one tool call.
const MAX_DEPTH: usize = 64;

/// Ids of the vocabulary markers the parser and engine react to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Markers {
    /// End of sequence, `<eos>`.
    pub(crate) eos: u32,
    /// End of turn, `<turn|>`.
    pub(crate) turn_end: u32,
    /// Start of a tool response, `<|tool_response>`.
    pub(crate) tool_response: u32,
    /// Start of a reasoning channel, `<|channel>`.
    pub(crate) channel_open: u32,
    /// End of a reasoning channel, `<channel|>`.
    pub(crate) channel_close: u32,
    /// Start of a tool call, `<|tool_call>`.
    pub(crate) tool_call_open: u32,
    /// End of a tool call, `<tool_call|>`.
    pub(crate) tool_call_close: u32,
}

impl Markers {
    /// Resolve the marker ids from the vocabulary; `eos_id` is the GGUF EOS id.
    pub(crate) fn from_tokenizer(t: &Tokenizer, eos_id: u32) -> Result<Self, CucaError> {
        let lookup = |text: &str| {
            t.token_id(text).ok_or_else(|| {
                CucaError::provider(
                    ProviderEndpoint::Local,
                    format!("vocabulary lacks `{text}`"),
                )
            })
        };
        Ok(Self {
            eos: eos_id,
            turn_end: lookup("<turn|>")?,
            tool_response: lookup("<|tool_response>")?,
            channel_open: lookup("<|channel>")?,
            channel_close: lookup("<channel|>")?,
            tool_call_open: lookup("<|tool_call>")?,
            tool_call_close: lookup("<tool_call|>")?,
        })
    }

    /// Whether `id` ends generation (`<eos>`, `<turn|>` or `<|tool_response>`).
    pub(crate) fn is_eog(&self, id: u32) -> bool {
        id == self.eos || id == self.turn_end || id == self.tool_response
    }
}

/// Where the parser currently is in the model's output grammar.
#[derive(Debug)]
enum State {
    /// Plain answer text.
    Content,
    /// Just after `<|channel>`: collecting the header word before the body.
    ChannelHeader(String),
    /// Inside a reasoning channel body.
    Thinking,
    /// Inside a tool call: the raw body collected so far.
    ToolCall(String),
}

/// Incremental parser turning generated ids into content blocks.
#[derive(Debug)]
pub(crate) struct OutputParser {
    /// Marker ids.
    markers: Markers,
    /// Current grammar state.
    state: State,
    /// Decoded bytes not yet forming complete UTF-8.
    pending: Vec<u8>,
    /// Ids pushed while inside (or delimiting) a reasoning channel.
    reasoning_tokens: u32,
}

impl OutputParser {
    /// Create a parser; with `start_in_thinking` the output is already inside a
    /// reasoning channel (the prompt ended after the channel header).
    pub(crate) fn new(markers: Markers, start_in_thinking: bool) -> Self {
        Self {
            markers,
            state: if start_in_thinking {
                State::Thinking
            } else {
                State::Content
            },
            pending: Vec::new(),
            reasoning_tokens: 0,
        }
    }

    /// Number of pushed ids from `<|channel>` through `<channel|>` inclusive (and
    /// every id before the first `<channel|>` when started in thinking).
    pub(crate) fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    /// Feed one generated id, appending any completed blocks to `out`.
    pub(crate) fn push(
        &mut self,
        id: u32,
        tok: &Tokenizer,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        let m = self.markers;
        if id == m.channel_open || matches!(self.state, State::ChannelHeader(_) | State::Thinking) {
            self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
        }

        if matches!(self.state, State::ToolCall(_)) {
            if id == m.tool_call_close {
                self.flush_pending(out);
                self.finish_tool_call(next_call_id, out);
            } else {
                let before = self.pending.len();
                tok.decode_into(id, &mut self.pending);
                if self.pending.len() == before {
                    // Control tokens such as the string quote decode to nothing
                    // but are part of the call text.
                    self.pending
                        .extend_from_slice(tok.token_text(id).as_bytes());
                }
                self.drain_pending(out);
            }
            return;
        }

        if id == m.channel_open {
            self.flush_pending(out);
            self.resolve_header(true, out);
            self.state = State::ChannelHeader(String::new());
        } else if id == m.channel_close
            && matches!(self.state, State::ChannelHeader(_) | State::Thinking)
        {
            self.flush_pending(out);
            self.resolve_header(true, out);
            self.state = State::Content;
        } else if id == m.tool_call_open {
            self.flush_pending(out);
            self.resolve_header(true, out);
            self.state = State::ToolCall(String::new());
        } else {
            tok.decode_into(id, &mut self.pending);
            self.drain_pending(out);
        }
    }

    /// Flush held-back bytes and any open channel header or tool call at the end
    /// of generation (EOG, `max_tokens`, or the context limit).
    pub(crate) fn finish(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        self.flush_pending(out);
        if matches!(self.state, State::ToolCall(_)) {
            self.finish_tool_call(next_call_id, out);
        } else {
            self.resolve_header(true, out);
        }
    }

    /// Route all complete UTF-8 in `pending`, keeping an incomplete tail; an
    /// invalid sequence becomes U+FFFD.
    fn drain_pending(&mut self, out: &mut Vec<Result<MessageContentBlock, CucaError>>) {
        loop {
            if self.pending.is_empty() {
                return;
            }
            let (valid, invalid_len) = match std::str::from_utf8(&self.pending) {
                Ok(_) => (self.pending.len(), None),
                Err(e) => (e.valid_up_to(), Some(e.error_len())),
            };
            if valid > 0 {
                let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                self.route(text, out);
            }
            match invalid_len {
                None => return,
                Some(None) => return,
                Some(Some(n)) => {
                    self.pending.drain(..n);
                    self.route("\u{FFFD}".to_string(), out);
                }
            }
        }
    }

    /// Route every pending byte, an incomplete tail included, as lossy text.
    fn flush_pending(&mut self, out: &mut Vec<Result<MessageContentBlock, CucaError>>) {
        self.drain_pending(out);
        if !self.pending.is_empty() {
            let text = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.route(text, out);
        }
    }

    /// Hand decoded text to the current state.
    fn route(&mut self, text: String, out: &mut Vec<Result<MessageContentBlock, CucaError>>) {
        if text.is_empty() {
            return;
        }
        match &mut self.state {
            State::Content => out.push(Ok(MessageContentBlock::Text(text))),
            State::Thinking => out.push(Ok(MessageContentBlock::Thinking {
                reasoning: text,
                signature: None,
            })),
            State::ChannelHeader(header) => {
                header.push_str(&text);
                self.resolve_header(false, out);
            }
            State::ToolCall(body) => body.push_str(&text),
        }
    }

    /// Decide whether the channel header is complete and, if so, move to
    /// `Thinking`, emitting the body text already seen.
    ///
    /// A header starting with `thought` drops that word and the whitespace
    /// after it; any other header is kept as body text. Unless `force` is set
    /// the decision waits until it is known: a header still being a prefix of
    /// `thought`, or `thought` followed only by whitespace, stays open.
    fn resolve_header(
        &mut self,
        force: bool,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        let State::ChannelHeader(header) = &mut self.state else {
            return;
        };
        let body = if let Some(rest) = header.strip_prefix(THOUGHT) {
            let trimmed = rest.trim_start_matches([' ', '\t', '\n']);
            if trimmed.is_empty() && !force {
                return;
            }
            trimmed.to_string()
        } else {
            if !force && THOUGHT.starts_with(header.as_str()) {
                return;
            }
            std::mem::take(header)
        };
        self.state = State::Thinking;
        if !body.is_empty() {
            out.push(Ok(MessageContentBlock::Thinking {
                reasoning: body,
                signature: None,
            }));
        }
    }

    /// Parse the buffered tool-call body, emit the result, and return to
    /// `Content`.
    fn finish_tool_call(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        let State::ToolCall(body) = std::mem::replace(&mut self.state, State::Content) else {
            return;
        };
        match parse_tool_call(&body) {
            Ok((name, arguments)) => out.push(Ok(MessageContentBlock::ToolCall {
                id: next_call_id(),
                name,
                arguments,
            })),
            Err(_) => out.push(Err(CucaError::provider(
                ProviderEndpoint::Local,
                format!("malformed tool call: {body}"),
            ))),
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Synthetic vocabulary: `(text, GGUF type)`; ids follow the order.
    const VOCAB: &[(&str, i32)] = &[
        ("<eos>", 3),
        ("<turn|>", 3),
        ("<|tool_response>", 3),
        ("<|channel>", 3),
        ("<channel|>", 3),
        ("<|tool_call>", 3),
        ("<tool_call|>", 3),
        ("<|\"|>", 3),
        ("thought", 1),
        ("\n", 1),
        ("abc", 1),
        ("ok", 1),
        ("hello", 1),
        ("call:get_time{city:", 1),
        ("London", 1),
        ("}", 1),
        ("call:a{x:1}", 1),
        ("call:b{y:2}", 1),
        ("call:cut{k:", 1),
        ("<0xE2>", 6),
        ("<0x82>", 6),
        ("<0xAC>", 6),
        ("<0xFF>", 6),
        ("\u{2581}world", 1),
    ];

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(
            VOCAB.iter().map(|(t, _)| t.to_string()).collect(),
            VOCAB.iter().map(|(_, ty)| *ty).collect(),
            &[],
        )
        .expect("synthetic vocabulary is valid")
    }

    fn id(t: &Tokenizer, text: &str) -> u32 {
        t.token_id(text).expect("token in synthetic vocabulary")
    }

    fn run(
        texts: &[&str],
        start_in_thinking: bool,
    ) -> (Vec<Result<MessageContentBlock, CucaError>>, u32) {
        let t = tokenizer();
        let markers = Markers::from_tokenizer(&t, 0).expect("markers present");
        let mut parser = OutputParser::new(markers, start_in_thinking);
        let mut counter = 0;
        let mut next = || {
            counter += 1;
            format!("call_{counter}")
        };
        let mut out = Vec::new();
        for text in texts {
            parser.push(id(&t, text), &t, &mut next, &mut out);
        }
        parser.finish(&mut next, &mut out);
        (out, parser.reasoning_tokens())
    }

    fn ok(out: Vec<Result<MessageContentBlock, CucaError>>) -> Vec<MessageContentBlock> {
        out.into_iter()
            .map(|r| r.expect("no parser error expected"))
            .collect()
    }

    fn text(s: &str) -> MessageContentBlock {
        MessageContentBlock::Text(s.to_string())
    }

    fn thinking(s: &str) -> MessageContentBlock {
        MessageContentBlock::Thinking {
            reasoning: s.to_string(),
            signature: None,
        }
    }

    fn call(id: &str, name: &str, arguments: Value) -> MessageContentBlock {
        MessageContentBlock::ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn markers_resolve_and_classify_eog() {
        let t = tokenizer();
        let m = Markers::from_tokenizer(&t, 0).expect("markers");
        assert!(m.is_eog(0) && m.is_eog(1) && m.is_eog(2));
        assert!(!m.is_eog(3) && !m.is_eog(10));
        let sparse = Tokenizer::new(vec!["<eos>".into()], vec![3], &[]).expect("valid");
        let err = Markers::from_tokenizer(&sparse, 0).expect_err("markers missing");
        assert!(err.to_string().contains("vocabulary lacks `<turn|>`"));
    }

    #[test]
    fn plain_text_passes_through() {
        let (out, reasoning) = run(&["hello", "\u{2581}world"], false);
        assert_eq!(ok(out), [text("hello"), text(" world")]);
        assert_eq!(reasoning, 0);
    }

    #[test]
    fn thought_channel_then_answer() {
        let (out, reasoning) = run(
            &["<|channel>", "thought", "\n", "abc", "<channel|>", "ok"],
            false,
        );
        assert_eq!(ok(out), [thinking("abc"), text("ok")]);
        assert_eq!(reasoning, 5);
    }

    #[test]
    fn channel_without_thought_header_keeps_body() {
        let (out, _) = run(&["<|channel>", "abc", "<channel|>", "ok"], false);
        assert_eq!(ok(out), [thinking("abc"), text("ok")]);
    }

    #[test]
    fn empty_thought_channel_emits_nothing() {
        let (out, reasoning) = run(&["<|channel>", "thought", "\n", "<channel|>", "ok"], false);
        assert_eq!(ok(out), [text("ok")]);
        assert_eq!(reasoning, 4);
    }

    #[test]
    fn start_in_thinking_counts_until_close() {
        let (out, reasoning) = run(&["abc", "hello", "<channel|>", "ok"], true);
        assert_eq!(ok(out), [thinking("abc"), thinking("hello"), text("ok")]);
        assert_eq!(reasoning, 3);
    }

    #[test]
    fn stray_channel_close_in_content_is_ignored() {
        let (out, reasoning) = run(&["<channel|>", "ok"], false);
        assert_eq!(ok(out), [text("ok")]);
        assert_eq!(reasoning, 0);
    }

    #[test]
    fn tool_call_with_quoted_string() {
        let (out, _) = run(
            &[
                "<|tool_call>",
                "call:get_time{city:",
                "<|\"|>",
                "London",
                "<|\"|>",
                "}",
                "<tool_call|>",
            ],
            false,
        );
        assert_eq!(
            ok(out),
            [call("call_1", "get_time", json!({ "city": "London" }))]
        );
    }

    #[test]
    fn two_back_to_back_tool_calls() {
        let (out, _) = run(
            &[
                "<|tool_call>",
                "call:a{x:1}",
                "<tool_call|>",
                "<|tool_call>",
                "call:b{y:2}",
                "<tool_call|>",
                "ok",
            ],
            false,
        );
        assert_eq!(
            ok(out),
            [
                call("call_1", "a", json!({ "x": 1 })),
                call("call_2", "b", json!({ "y": 2 })),
                text("ok"),
            ]
        );
    }

    #[test]
    fn tool_call_cut_off_by_finish_is_still_parsed() {
        let (out, _) = run(&["<|tool_call>", "call:a{x:1}"], false);
        assert_eq!(ok(out), [call("call_1", "a", json!({ "x": 1 }))]);
    }

    #[test]
    fn truncated_tool_call_body_is_an_error() {
        let (out, _) = run(&["<|tool_call>", "call:cut{k:"], false);
        assert_eq!(out.len(), 1);
        let err = out[0].as_ref().expect_err("truncated call is malformed");
        assert!(err.to_string().contains("malformed tool call: call:cut{k:"));
    }

    #[test]
    fn malformed_tool_call_reports_error_and_recovers() {
        let (out, _) = run(&["<|tool_call>", "abc", "<tool_call|>", "ok"], false);
        assert_eq!(out.len(), 2);
        assert!(
            out[0]
                .as_ref()
                .expect_err("malformed")
                .to_string()
                .contains("malformed tool call: abc")
        );
        assert_eq!(out[1].as_ref().expect("text"), &text("ok"));
    }

    #[test]
    fn multibyte_char_split_across_byte_tokens() {
        let (out, _) = run(&["hello", "<0xE2>", "<0x82>", "<0xAC>", "ok"], false);
        assert_eq!(ok(out), [text("hello"), text("\u{20AC}"), text("ok")]);
    }

    #[test]
    fn invalid_utf8_becomes_replacement_character() {
        let (out, _) = run(&["<0xFF>", "ok"], false);
        assert_eq!(ok(out), [text("\u{FFFD}"), text("ok")]);
    }

    #[test]
    fn incomplete_sequence_is_flushed_at_a_marker() {
        let (out, _) = run(&["<0xE2>", "<0x82>", "<|channel>"], false);
        assert_eq!(ok(out), [text("\u{FFFD}")]);
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
}
