//! The channel protocol of Muse Glimmer's chat template ("ATEM"), parsed from
//! generated ids (see [`Atem`] for the grammar).
//!
//! [`AtemParser`] reads a turn as a sequence of messages. A header (the text
//! between `<|start|>` and `<|message|>`, or from the start of generation)
//! names the recipient after `to=`: `self` makes the body a
//! [`MessageContentBlock::Thinking`], `user` (or no recipient) answer
//! [`MessageContentBlock::Text`], and anything else a tool call whose body is
//! buffered whole and parsed by [`parse_atem_calls`] when its message ends.
//! `<|eom|>` ends a message and the next header follows; `<|eot|>` (an
//! end-of-generation id) and the end of generation end the last one. Markup
//! inside a reasoning or answer message is ordinary text, so an answer that
//! quotes `<atem:function_calls>` stays an answer.
//!
//! The four markers are single control tokens, so they are recognised by id
//! ([`ChatFormat::is_marker`]); everything else is decoded and routed to the
//! current message. A character split across byte tokens is emitted whole.

use serde_json::{Map, Value};

use super::super::chat::{Atem, ChatFormat};
use super::super::tokenizer::Tokenizer;
use crate::error::CucaError;
use crate::types::{MessageContentBlock, ProviderEndpoint};

const FUNCTION_CALLS_OPEN: &str = "<atem:function_calls>";
const FUNCTION_CALLS_CLOSE: &str = "</atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke name=\"";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAMETER_OPEN: &str = "<atem:parameter name=\"";
const PARAMETER_CLOSE: &str = "</atem:parameter>";
/// Recipient of the reasoning channel.
const SELF_RECIPIENT: &str = "self";

/// The message being read.
#[derive(Debug)]
enum State {
    /// A header: the text collected since the start of generation or since
    /// `<|start|>`.
    Header(String),
    /// The body of a `to=self` message.
    Thinking,
    /// The body of an answer message.
    Content,
    /// The body of a tool-call message, collected whole.
    Call(String),
}

/// Incremental parser of the channel protocol.
#[derive(Debug)]
pub(super) struct AtemParser {
    markers: Atem,
    state: State,
    /// Decoded bytes not yet forming complete UTF-8.
    pending: Vec<u8>,
    /// Ids pushed since the current header began (counted as reasoning when
    /// the message turns out to be addressed to `self`).
    header_ids: u32,
    reasoning_tokens: u32,
}

type Out = Vec<Result<MessageContentBlock, CucaError>>;

impl AtemParser {
    pub(super) fn new(markers: &Atem) -> Self {
        Self {
            markers: markers.clone(),
            state: State::Header(String::new()),
            pending: Vec::new(),
            header_ids: 0,
            reasoning_tokens: 0,
        }
    }

    /// Ids pushed inside `to=self` messages: header, opening and closing
    /// markers included.
    pub(super) fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    /// Feed one generated id, appending completed blocks to `out`.
    pub(super) fn push(
        &mut self,
        id: u32,
        format: &ChatFormat,
        tok: &Tokenizer,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Out,
    ) {
        if matches!(self.state, State::Header(_)) {
            self.header_ids = self.header_ids.saturating_add(1);
        }
        let in_thinking = matches!(self.state, State::Thinking);
        if format.is_marker(id) {
            self.drain(true, out);
            let text = tok.token_text(id);
            let m = &self.markers;
            if text == m.start {
                self.end_message(next_call_id, out);
                self.begin_header();
            } else if text == m.message {
                self.open_body();
            } else if text == m.end_message || text == m.end_turn {
                self.end_message(next_call_id, out);
                self.begin_header();
            } else {
                Self::route(&mut self.state, text, out);
            }
        } else {
            tok.decode_into(id, &mut self.pending);
            self.drain(false, out);
        }
        if in_thinking || matches!(self.state, State::Thinking) {
            self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
        }
    }

    /// End of generation: flush undecoded bytes and finish the open message.
    pub(super) fn finish(&mut self, next_call_id: &mut dyn FnMut() -> String, out: &mut Out) {
        self.drain(true, out);
        self.end_message(next_call_id, out);
        self.begin_header();
    }

    fn begin_header(&mut self) {
        self.state = State::Header(String::new());
        self.header_ids = 0;
    }

    /// `<|message|>`: the header is complete; its recipient picks the channel.
    fn open_body(&mut self) {
        let State::Header(header) = &self.state else {
            // A stray marker inside a body is dropped.
            return;
        };
        self.state = match recipient(header) {
            Some(SELF_RECIPIENT) => {
                // The header's ids and this marker belong to the reasoning.
                self.reasoning_tokens = self
                    .reasoning_tokens
                    .saturating_add(self.header_ids.saturating_sub(1));
                State::Thinking
            }
            None | Some("user") => State::Content,
            Some(_) => State::Call(String::new()),
        };
    }

    /// The message ends: a collected tool call is parsed and emitted.
    fn end_message(&mut self, next_call_id: &mut dyn FnMut() -> String, out: &mut Out) {
        let State::Call(body) = std::mem::replace(&mut self.state, State::Content) else {
            return;
        };
        if body.trim().is_empty() {
            return;
        }
        match parse_atem_calls(&body) {
            Ok(calls) => {
                for (name, arguments) in calls {
                    out.push(Ok(MessageContentBlock::ToolCall {
                        id: next_call_id(),
                        name,
                        arguments,
                    }));
                }
            }
            Err(_) => out.push(Err(CucaError::provider(
                ProviderEndpoint::Local,
                format!("malformed tool call: {body}"),
            ))),
        }
    }

    /// Route all complete UTF-8 in `pending` (all of it, lossily, when
    /// `flush`), keeping an incomplete tail otherwise. An invalid sequence
    /// becomes U+FFFD.
    fn drain(&mut self, flush: bool, out: &mut Out) {
        let mut done = 0;
        while done < self.pending.len() {
            let rest = &self.pending[done..];
            let error = match std::str::from_utf8(rest) {
                Ok(text) => {
                    Self::route(&mut self.state, text, out);
                    done = self.pending.len();
                    break;
                }
                Err(e) => e,
            };
            let valid = error.valid_up_to();
            Self::route(
                &mut self.state,
                &String::from_utf8_lossy(&rest[..valid]),
                out,
            );
            done += valid;
            match error.error_len() {
                Some(n) => {
                    done += n;
                    Self::route(&mut self.state, "\u{FFFD}", out);
                }
                None if flush => {
                    let text = String::from_utf8_lossy(&self.pending[done..]);
                    Self::route(&mut self.state, &text, out);
                    done = self.pending.len();
                }
                None => break,
            }
        }
        self.pending.drain(..done);
    }

    /// Hand text to the current message.
    fn route(state: &mut State, text: &str, out: &mut Out) {
        if text.is_empty() {
            return;
        }
        match state {
            State::Header(header) => header.push_str(text),
            State::Call(body) => body.push_str(text),
            State::Thinking => out.push(Ok(MessageContentBlock::Thinking {
                reasoning: text.to_owned(),
                signature: None,
            })),
            State::Content => out.push(Ok(MessageContentBlock::Text(text.to_owned()))),
        }
    }
}

/// The recipient a header names: the word after `to=`.
fn recipient(header: &str) -> Option<&str> {
    let (_, rest) = header.split_once("to=")?;
    rest.split_whitespace().next()
}

/// Parse the body of a tool-call message,
///
/// ```text
/// <atem:function_calls>
/// <atem:invoke name="NAME">
/// <atem:parameter name="KEY">VALUE</atem:parameter>
/// </atem:invoke>
/// </atem:function_calls>
/// ```
///
/// into one `(name, arguments)` per `invoke`. A value is the raw text between
/// the tags, nothing trimmed: the template writes strings and numbers as they
/// are, booleans as `true`/`false`, none as `null`, and lists and objects as
/// JSON. A value is therefore read as JSON when it is a JSON number, boolean,
/// null, array or object spelled without surrounding whitespace, and as a
/// string otherwise; a string that looks like one of those (`42`, `true`)
/// reads back as that JSON value, the ambiguity the format has.
///
/// # Errors
///
/// A description of the first problem: no complete `invoke`, a missing tag.
pub(super) fn parse_atem_calls(body: &str) -> Result<Vec<(String, Value)>, String> {
    let rest = body.trim();
    let rest = rest
        .strip_prefix(FUNCTION_CALLS_OPEN)
        .ok_or_else(|| format!("expected {FUNCTION_CALLS_OPEN}"))?;
    let mut calls = Vec::new();
    let mut rest = rest.trim_start();
    while let Some(after) = rest.strip_prefix(INVOKE_OPEN) {
        let (name, after) = after.split_once("\">").ok_or("unterminated invoke name")?;
        let mut args = Map::new();
        let mut after = after.trim_start();
        loop {
            if let Some(next) = after.strip_prefix(INVOKE_CLOSE) {
                rest = next.trim_start();
                break;
            }
            let param = after
                .strip_prefix(PARAMETER_OPEN)
                .ok_or_else(|| format!("expected {PARAMETER_OPEN} or {INVOKE_CLOSE}"))?;
            let (key, param) = param
                .split_once("\">")
                .ok_or("unterminated parameter name")?;
            let (value, param) = param
                .split_once(PARAMETER_CLOSE)
                .ok_or_else(|| format!("missing {PARAMETER_CLOSE}"))?;
            args.insert(key.to_string(), parameter_value(value));
            after = param.trim_start();
        }
        calls.push((name.to_string(), Value::Object(args)));
    }
    if calls.is_empty() {
        return Err(format!("no {INVOKE_OPEN}...{INVOKE_CLOSE}"));
    }
    // The closing tag may be cut off by the end of generation.
    let rest = rest.trim();
    if !rest.is_empty() && rest != FUNCTION_CALLS_CLOSE {
        return Err(format!("unexpected text after the invokes: {rest}"));
    }
    Ok(calls)
}

/// One parameter value, see [`parse_atem_calls`].
fn parameter_value(raw: &str) -> Value {
    if raw == raw.trim()
        && let Ok(v) = serde_json::from_str::<Value>(raw)
        && !v.is_string()
    {
        return v;
    }
    Value::String(raw.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::quantized::gguf_file;
    use serde_json::json;

    use super::super::super::chat::ToolSyntax;
    use super::*;

    /// A vocabulary with the four protocol markers as control tokens and a
    /// few text pieces; ids follow the order.
    const VOCAB: &[(&str, i32)] = &[
        ("<|end_of_text|>", 3),
        ("<|start|>", 3),
        ("<|message|>", 3),
        ("<|eom|>", 3),
        ("<|eot|>", 3),
        ("assistant", 1),
        (" to", 1),
        ("=", 1),
        ("self", 1),
        ("user", 1),
        ("special_function", 1),
        ("Hello, world!\nWhat's up?", 1),
        ("I need to output the details", 1),
        (
            "<atem:function_calls>\n<atem:invoke name=\"special_function\">\n",
            1,
        ),
        ("<atem:parameter name=\"arg1\">1</atem:parameter>\n", 1),
        ("</atem:invoke>\n</atem:function_calls>", 1),
        ("I could use ", 1),
        (" here", 1),
        ("You invoke it like this:\n", 1),
        ("<0xE2>", 6),
        ("<0x82>", 6),
        ("<0xAC>", 6),
    ];
    const TEMPLATE: &str = "<atem:function_calls> <|eom|> <|eot|>";

    const CALL_MARKUP: &[&str] = &[
        "<atem:function_calls>\n<atem:invoke name=\"special_function\">\n",
        "<atem:parameter name=\"arg1\">1</atem:parameter>\n",
        "</atem:invoke>\n</atem:function_calls>",
    ];

    fn run(texts: &[&str]) -> (Out, u32) {
        let tokenizer = Tokenizer::new(
            VOCAB.iter().map(|(t, _)| t.to_string()).collect(),
            VOCAB.iter().map(|(_, ty)| *ty).collect(),
            &[],
        )
        .expect("vocabulary");
        let md = HashMap::from([(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(0),
        )]);
        let format = ChatFormat::detect(&md, &tokenizer, TEMPLATE).expect("format");
        let mut parser = AtemParser::new(atem_of(&format));
        let mut out = Vec::new();
        let mut counter = 0;
        let mut next = || {
            counter += 1;
            format!("call_{counter}")
        };
        for text in texts {
            let id = tokenizer
                .token_id(text)
                .unwrap_or_else(|| panic!("`{text}` is not in the vocabulary"));
            parser.push(id, &format, &tokenizer, &mut next, &mut out);
        }
        parser.finish(&mut next, &mut out);
        (out, parser.reasoning_tokens())
    }

    /// The `Atem` markers of a detected format.
    fn atem_of(format: &ChatFormat) -> &Atem {
        match format.tool_syntax() {
            ToolSyntax::Atem(a) => a,
            other => panic!("not the ATEM syntax: {other:?}"),
        }
    }

    fn ok(out: Out) -> Vec<MessageContentBlock> {
        out.into_iter().map(|r| r.expect("no error")).collect()
    }

    fn text(s: &str) -> MessageContentBlock {
        MessageContentBlock::Text(s.into())
    }

    fn thinking(s: &str) -> MessageContentBlock {
        MessageContentBlock::Thinking {
            reasoning: s.into(),
            signature: None,
        }
    }

    fn call(id: &str, name: &str, arguments: Value) -> MessageContentBlock {
        MessageContentBlock::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    // Fixtures ported from llama.cpp `tests/test-chat.cpp` ("Muse Glimmer
    // format tests"): the generated text starts at " to=" because
    // `<|start|>assistant` is the generation prompt. `<|eot|>` is an EOG id
    // the engine never pushes, so a final answer simply ends.

    #[test]
    fn a_plain_answer() {
        let (out, reasoning) = run(&[
            " to",
            "=",
            "user",
            "<|message|>",
            "Hello, world!\nWhat's up?",
        ]);
        assert_eq!(ok(out), [text("Hello, world!\nWhat's up?")]);
        assert_eq!(reasoning, 0);
    }

    #[test]
    fn a_tool_call_as_the_first_message() {
        let mut ids = vec![" to", "=", "special_function", "<|message|>"];
        ids.extend(CALL_MARKUP);
        let (out, _) = run(&ids);
        assert_eq!(
            ok(out),
            [call("call_1", "special_function", json!({"arg1": 1}))]
        );
    }

    #[test]
    fn inform_then_act_stops_the_answer_at_eom() {
        let mut ids = vec![
            " to",
            "=",
            "user",
            "<|message|>",
            "Hello, world!\nWhat's up?",
            "<|eom|>",
            "<|start|>",
            "assistant",
            " to",
            "=",
            "special_function",
            "<|message|>",
        ];
        ids.extend(CALL_MARKUP);
        let (out, _) = run(&ids);
        assert_eq!(
            ok(out),
            [
                text("Hello, world!\nWhat's up?"),
                call("call_1", "special_function", json!({"arg1": 1}))
            ]
        );
    }

    #[test]
    fn markup_quoted_in_an_answer_stays_text() {
        let mut ids = vec![
            " to",
            "=",
            "user",
            "<|message|>",
            "You invoke it like this:\n",
        ];
        ids.extend(CALL_MARKUP);
        let (out, _) = run(&ids);
        let blocks = ok(out);
        let joined: String = blocks
            .iter()
            .map(|b| match b {
                MessageContentBlock::Text(t) => t.as_str(),
                other => panic!("not text: {other:?}"),
            })
            .collect();
        assert_eq!(
            joined,
            "You invoke it like this:\n<atem:function_calls>\n<atem:invoke name=\"special_function\">\n\
             <atem:parameter name=\"arg1\">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
        );
    }

    #[test]
    fn reasoning_then_answer() {
        let (out, reasoning) = run(&[
            " to",
            "=",
            "self",
            "<|message|>",
            "I need to output the details",
            "<|eom|>",
            "<|start|>",
            "assistant",
            " to",
            "=",
            "user",
            "<|message|>",
            "Hello, world!\nWhat's up?",
        ]);
        assert_eq!(
            ok(out),
            [
                thinking("I need to output the details"),
                text("Hello, world!\nWhat's up?")
            ]
        );
        // " to", "=", "self", the message marker, the body, `<|eom|>`.
        assert_eq!(reasoning, 6);
    }

    #[test]
    fn tool_markup_inside_reasoning_is_reasoning() {
        let mut ids = vec![" to", "=", "self", "<|message|>", "I could use "];
        ids.extend(CALL_MARKUP);
        ids.extend([
            " here",
            "<|eom|>",
            "<|start|>",
            "assistant",
            " to",
            "=",
            "user",
            "<|message|>",
            "Hello, world!\nWhat's up?",
        ]);
        let (out, _) = run(&ids);
        let blocks = ok(out);
        let reasoning: String = blocks
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::Thinking { reasoning, .. } => Some(reasoning.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            reasoning.starts_with("I could use <atem:function_calls>"),
            "{reasoning}"
        );
        assert!(
            reasoning.ends_with("</atem:function_calls> here"),
            "{reasoning}"
        );
        assert_eq!(blocks.last(), Some(&text("Hello, world!\nWhat's up?")));
        assert!(
            blocks
                .iter()
                .all(|b| !matches!(b, MessageContentBlock::ToolCall { .. }))
        );
    }

    #[test]
    fn two_calls_in_two_messages_get_their_own_ids() {
        let mut ids = vec![" to", "=", "special_function", "<|message|>"];
        ids.extend(CALL_MARKUP);
        ids.extend([
            "<|eom|>",
            "<|start|>",
            "assistant",
            " to",
            "=",
            "special_function",
            "<|message|>",
        ]);
        ids.extend(CALL_MARKUP);
        let (out, _) = run(&ids);
        assert_eq!(
            ok(out),
            [
                call("call_1", "special_function", json!({"arg1": 1})),
                call("call_2", "special_function", json!({"arg1": 1}))
            ]
        );
    }

    #[test]
    fn a_truncated_call_is_a_provider_error() {
        let (out, _) = run(&[
            " to",
            "=",
            "special_function",
            "<|message|>",
            "<atem:function_calls>\n<atem:invoke name=\"special_function\">\n",
        ]);
        let err = out.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(err.contains("malformed tool call"), "{err}");
    }

    #[test]
    fn a_header_without_a_recipient_is_an_answer() {
        let (out, _) = run(&["<|message|>", "Hello, world!\nWhat's up?"]);
        assert_eq!(ok(out), [text("Hello, world!\nWhat's up?")]);
    }

    #[test]
    fn multibyte_characters_split_across_tokens_are_emitted_whole() {
        let (out, _) = run(&[
            " to",
            "=",
            "user",
            "<|message|>",
            "<0xE2>",
            "<0x82>",
            "<0xAC>",
        ]);
        assert_eq!(ok(out), [text("\u{20ac}")]);
    }

    #[test]
    fn parameters_read_as_json_unless_they_are_strings() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"f\">\n\
            <atem:parameter name=\"n\">42</atem:parameter>\n\
            <atem:parameter name=\"b\">true</atem:parameter>\n\
            <atem:parameter name=\"z\">null</atem:parameter>\n\
            <atem:parameter name=\"l\">[1, \"a\"]</atem:parameter>\n\
            <atem:parameter name=\"o\">{\"k\": 1}</atem:parameter>\n\
            <atem:parameter name=\"s\"> padded 42 </atem:parameter>\n\
            <atem:parameter name=\"q\">\"quoted\"</atem:parameter>\n\
            <atem:parameter name=\"m\">two\nlines\n</atem:parameter>\n\
            <atem:parameter name=\"e\"></atem:parameter>\n\
            </atem:invoke>\n</atem:function_calls>";
        let calls = parse_atem_calls(body).unwrap();
        assert_eq!(
            calls,
            [(
                "f".to_string(),
                json!({
                    "n": 42, "b": true, "z": null, "l": [1, "a"], "o": {"k": 1},
                    "s": " padded 42 ", "q": "\"quoted\"", "m": "two\nlines\n", "e": ""
                })
            )]
        );
    }

    #[test]
    fn several_invokes_and_no_parameters_parse() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"a\">\n</atem:invoke>\n\
            <atem:invoke name=\"ns.b\">\n<atem:parameter name=\"x\">1</atem:parameter>\n</atem:invoke>\n";
        let calls = parse_atem_calls(body).unwrap();
        assert_eq!(
            calls,
            [
                ("a".to_string(), json!({})),
                ("ns.b".to_string(), json!({"x": 1}))
            ]
        );
        assert!(parse_atem_calls("no markup").is_err());
        assert!(parse_atem_calls("<atem:function_calls>\n</atem:function_calls>").is_err());
        assert!(
            parse_atem_calls(
                "<atem:function_calls>\n<atem:invoke name=\"a\">\n</atem:invoke>\ngarbage"
            )
            .is_err()
        );
    }
}
