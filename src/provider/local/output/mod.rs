//! Streaming parser from generated token ids to [`MessageContentBlock`]s.
//!
//! What a chat model writes besides plain answer text depends on its family,
//! and [`ChatFormat`] (derived from the GGUF, see `chat::format`) says which of
//! these [`OutputParser`] looks for:
//!
//! - a **reasoning block** (`<|channel>thought\n…<channel|>` or
//!   `<think>…</think>`), which becomes [`MessageContentBlock::Thinking`]; a
//!   header word such as `thought` and the whitespace after it are dropped;
//! - **tool calls**, whose body is buffered whole and parsed into
//!   [`MessageContentBlock::ToolCall`]s, or reported as a provider error
//!   (`malformed tool call: <body>`) when malformed:
//!   the custom `call:name{key:<|"|>value<|"|>}` grammar between control tokens
//!   ([`calls::parse_tool_call`]), tagged JSON (`<tool_call>{"name": …,
//!   "arguments": …}</tool_call>`, also a JSON list, with or without a closing
//!   tag), the XML style of Qwen 3.5 (`<tool_call><function=name>
//!   <parameter=key>value</parameter></function></tool_call>`,
//!   [`calls::parse_xml_call`]), or a bare `{"name": …, "parameters": …}` object as the whole reply
//!   and a `<|python_tag|>` body that runs to the end of generation (code that
//!   is not JSON becomes a `python` call with a `code` argument);
//! - everything else is [`MessageContentBlock::Text`].
//!
//! Markers are matched on **text**, not ids, so a marker that the vocabulary
//! splits into several ordinary tokens works like one that is a single control
//! token: single-token markers contribute their raw spelling even though the
//! tokenizer renders control tokens as nothing, and text that could still grow
//! into a marker is held back until it cannot (at most one marker length).
//! A `</think>`-style close without an open is dropped.
//!
//! Token bytes are held back until they form complete UTF-8, so a character
//! split across byte tokens is emitted whole; an invalid sequence becomes
//! U+FFFD. Leading whitespace after a reasoning block opens or closes (and
//! after a tool call) is dropped, as is one leading space of the first text
//! for SentencePiece models. End-of-generation ids are never pushed (the engine
//! stops first); [`OutputParser::finish`] flushes whatever is pending. Per token
//! the parser allocates only the emitted block; the buffers for a reasoning
//! header, a tool-call body or a bare-JSON candidate are bounded by the
//! generation length the engine already caps.

mod calls;

use self::calls::{parse_json_calls, parse_tool_call, parse_xml_call};
use super::chat::{ChatFormat, ToolSyntax};
use super::tokenizer::Tokenizer;
use crate::error::CucaError;
use crate::types::{MessageContentBlock, ProviderEndpoint};

/// Tool name given to `<|python_tag|>` bodies that are code, not JSON.
const PYTHON_TOOL: &str = "python";

/// What recognising a marker does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Start a reasoning block.
    OpenReasoning,
    /// End the current reasoning block.
    CloseReasoning,
    /// A reasoning close outside a block: dropped.
    StrayClose,
    /// Start a tool-call body.
    OpenTool,
    /// End the current tool-call body.
    CloseTool,
    /// Start a `<|python_tag|>` body.
    OpenPython,
}

/// A marker text and its action.
#[derive(Debug, Clone)]
struct Marker {
    /// The spelling to find in the text stream.
    text: String,
    /// What to do when found.
    action: Action,
}

/// The markers to look for in each state.
#[derive(Debug, Default)]
struct MarkerTable {
    /// In answer text.
    content: Vec<Marker>,
    /// In a reasoning header or body.
    thinking: Vec<Marker>,
    /// In a tool-call body.
    tool: Vec<Marker>,
}

/// Running brace tracker for a bare-JSON tool-call candidate.
#[derive(Debug, Default)]
struct JsonCapture {
    /// The candidate text so far.
    text: String,
    /// Open-brace depth.
    depth: usize,
    /// Inside a string literal.
    in_string: bool,
    /// The previous character was a backslash inside a string.
    escaped: bool,
}

impl JsonCapture {
    /// Append characters of `text` up to and including the brace that closes
    /// the object; returns the bytes consumed and whether the object closed.
    fn advance(&mut self, text: &str) -> (usize, bool) {
        let mut consumed = 0;
        for c in text.chars() {
            consumed += c.len_utf8();
            self.text.push(c);
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if c == '\\' {
                    self.escaped = true;
                } else if c == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match c {
                '"' => self.in_string = true,
                '{' => self.depth += 1,
                '}' => {
                    self.depth = self.depth.saturating_sub(1);
                    if self.depth == 0 {
                        return (consumed, true);
                    }
                }
                _ => {}
            }
        }
        (consumed, false)
    }
}

/// Where the parser currently is in the model's output grammar.
#[derive(Debug)]
enum State {
    /// Plain answer text.
    Content,
    /// Just after a reasoning open marker with a header word: collecting the
    /// header before the body.
    Header(String),
    /// Inside a reasoning block body.
    Thinking,
    /// Inside a marked tool call: the raw body collected so far.
    ToolCall(String),
    /// After `<|python_tag|>`: the raw body collected so far.
    PythonTag(String),
    /// Inside a bare-JSON tool-call candidate.
    JsonCall(JsonCapture),
}

/// Incremental parser turning generated ids into content blocks.
#[derive(Debug)]
pub(crate) struct OutputParser {
    /// Output conventions of the model.
    format: ChatFormat,
    /// Marker spellings per state.
    markers: MarkerTable,
    /// Reasoning header word, empty when the format has none.
    header_word: String,
    /// Current grammar state.
    state: State,
    /// Decoded bytes not yet forming complete UTF-8.
    pending: Vec<u8>,
    /// Text not yet routed because it may be the start of a marker.
    hold: String,
    /// Ids pushed while inside (or delimiting) a reasoning block.
    reasoning_tokens: u32,
    /// Drop leading whitespace of the next text (after a reasoning edge or a
    /// tool call).
    trim_leading_ws: bool,
    /// Drop one leading space of the next text (SentencePiece prefix).
    strip_one_space: bool,
    /// Nothing but whitespace has been answered since the start or since the
    /// reasoning block closed: a `{` now may start a bare-JSON tool call.
    json_call_window: bool,
}

impl OutputParser {
    /// Create a parser for `format`; with `start_in_thinking` the output is
    /// already inside a reasoning block (the prompt ended after its opener).
    pub(crate) fn new(format: &ChatFormat, start_in_thinking: bool) -> Self {
        let mut markers = MarkerTable::default();
        let mut header_word = String::new();
        if let Some(r) = format.reasoning() {
            markers.content.push(Marker {
                text: r.open.clone(),
                action: Action::OpenReasoning,
            });
            markers.content.push(Marker {
                text: r.close.clone(),
                action: Action::StrayClose,
            });
            markers.thinking.push(Marker {
                text: r.close.clone(),
                action: Action::CloseReasoning,
            });
            header_word = r.header.clone().unwrap_or_default();
        }
        match format.tool_syntax() {
            ToolSyntax::Plain => {}
            ToolSyntax::Gemma4 { open, close } | ToolSyntax::TaggedXml { open, close } => {
                markers.content.push(open_tool(open));
                markers.thinking.push(open_tool(open));
                markers.tool.push(close_tool(close));
            }
            ToolSyntax::TaggedJson {
                open,
                alt_open,
                close,
            } => {
                for text in std::iter::once(open).chain(alt_open) {
                    markers.content.push(open_tool(text));
                    markers.thinking.push(open_tool(text));
                }
                if let Some(close) = close {
                    markers.tool.push(close_tool(close));
                }
            }
            ToolSyntax::LlamaJson { python_tag } => {
                if let Some(tag) = python_tag {
                    markers.content.push(Marker {
                        text: tag.clone(),
                        action: Action::OpenPython,
                    });
                }
            }
        }
        let thinking = start_in_thinking && format.reasoning().is_some();
        let state = if thinking {
            State::Thinking
        } else {
            State::Content
        };
        Self {
            markers,
            header_word,
            state,
            pending: Vec::new(),
            hold: String::new(),
            reasoning_tokens: 0,
            trim_leading_ws: thinking,
            strip_one_space: format.strips_leading_space(),
            json_call_window: matches!(format.tool_syntax(), ToolSyntax::LlamaJson { .. })
                && !thinking,
            format: format.clone(),
        }
    }

    /// Number of pushed ids inside a reasoning block, counting the opening and
    /// closing marker ids (and every id before the first close when started in
    /// thinking).
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
        let was_reasoning = self.in_reasoning();
        if self.format.is_marker(id) {
            self.flush_pending(next_call_id, out);
            self.feed_text(tok.token_text(id), next_call_id, out);
        } else {
            let before = self.pending.len();
            tok.decode_into(id, &mut self.pending);
            if self.pending.len() == before
                && tok.is_control(id)
                && matches!(self.state, State::ToolCall(_))
            {
                // Control tokens such as the string quote decode to nothing but
                // are part of the call text.
                self.pending
                    .extend_from_slice(tok.token_text(id).as_bytes());
            }
            self.drain_pending(next_call_id, out);
        }
        if was_reasoning || self.in_reasoning() {
            self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
        }
    }

    /// Flush held-back bytes and text and any open header, tool call or JSON
    /// candidate at the end of generation (EOG, `max_tokens`, or the context
    /// limit).
    pub(crate) fn finish(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        self.flush_pending(next_call_id, out);
        let held = std::mem::take(&mut self.hold);
        match std::mem::replace(&mut self.state, State::Content) {
            State::Content => self.emit_content(held, next_call_id, out),
            State::Thinking => {
                self.state = State::Thinking;
                self.emit_thinking(held, out);
            }
            State::Header(mut header) => {
                header.push_str(&held);
                self.state = State::Header(header);
                self.resolve_header(true, out);
            }
            State::ToolCall(mut body) => {
                body.push_str(&held);
                self.finish_tool_call(body, next_call_id, out);
            }
            State::PythonTag(mut body) => {
                body.push_str(&held);
                self.finish_python(&body, next_call_id, out);
            }
            State::JsonCall(mut capture) => {
                capture.text.push_str(&held);
                self.emit_text_block(capture.text, out);
            }
        }
    }

    /// Whether the parser is inside a reasoning header or body.
    fn in_reasoning(&self) -> bool {
        matches!(self.state, State::Header(_) | State::Thinking)
    }

    /// Route all complete UTF-8 in `pending`, keeping an incomplete tail; an
    /// invalid sequence becomes U+FFFD.
    fn drain_pending(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        loop {
            if self.pending.is_empty() {
                return;
            }
            let (valid, invalid_len) = match std::str::from_utf8(&self.pending) {
                Ok(_) => (self.pending.len(), None),
                Err(e) => (e.valid_up_to(), Some(e.error_len())),
            };
            if valid > 0 {
                // The prefix is valid UTF-8 by construction; it moves straight
                // into the hold buffer without an intermediate string.
                if let Ok(text) = std::str::from_utf8(&self.pending[..valid]) {
                    self.hold.push_str(text);
                }
                self.pending.drain(..valid);
                self.process_hold(next_call_id, out);
            }
            match invalid_len {
                None | Some(None) => return,
                Some(Some(n)) => {
                    self.pending.drain(..n);
                    self.feed_text("\u{FFFD}", next_call_id, out);
                }
            }
        }
    }

    /// Route every pending byte, an incomplete tail included, as lossy text.
    fn flush_pending(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        self.drain_pending(next_call_id, out);
        if !self.pending.is_empty() {
            self.hold.push_str(&String::from_utf8_lossy(&self.pending));
            self.pending.clear();
            self.process_hold(next_call_id, out);
        }
    }

    /// The markers to look for in the current state.
    fn active_markers(&self) -> &[Marker] {
        match self.state {
            State::Content => &self.markers.content,
            State::Header(_) | State::Thinking => &self.markers.thinking,
            State::ToolCall(_) => &self.markers.tool,
            State::PythonTag(_) | State::JsonCall(_) => &[],
        }
    }

    /// Append decoded text to the hold buffer and route everything that cannot
    /// be part of a marker.
    fn feed_text(
        &mut self,
        text: &str,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        self.hold.push_str(text);
        self.process_hold(next_call_id, out);
    }

    /// Route everything in the hold buffer that cannot be part of a marker.
    fn process_hold(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        loop {
            if matches!(self.state, State::JsonCall(_)) {
                if !self.scan_json(next_call_id, out) {
                    return;
                }
                continue;
            }
            let (found, keep) = {
                let markers = self.active_markers();
                match earliest_marker(&self.hold, markers) {
                    Some(hit) => (Some(hit), 0),
                    None => (None, partial_marker_suffix(&self.hold, markers)),
                }
            };
            match found {
                Some((pos, len, action)) => {
                    let before: String = self.hold.drain(..pos).collect();
                    self.hold.drain(..len);
                    self.route(before, next_call_id, out);
                    self.apply(action, next_call_id, out);
                }
                None => {
                    let ready_len = self.hold.len() - keep;
                    if ready_len > 0 {
                        self.route_hold(ready_len, next_call_id, out);
                    }
                    return;
                }
            }
        }
    }

    /// Hand the first `len` bytes of the hold buffer to the current state.
    /// States that only accumulate text take it by reference; states that emit
    /// a block take it as that block's string.
    fn route_hold(
        &mut self,
        len: usize,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        match &mut self.state {
            State::ToolCall(body) | State::PythonTag(body) => {
                body.push_str(&self.hold[..len]);
                self.hold.drain(..len);
            }
            State::JsonCall(capture) => {
                capture.text.push_str(&self.hold[..len]);
                self.hold.drain(..len);
            }
            _ => {
                let text: String = self.hold.drain(..len).collect();
                self.route(text, next_call_id, out);
            }
        }
    }

    /// Hand marker-free text to the current state.
    fn route(
        &mut self,
        text: String,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        if text.is_empty() {
            return;
        }
        match &mut self.state {
            State::Content => self.emit_content(text, next_call_id, out),
            State::Thinking => self.emit_thinking(text, out),
            State::Header(header) => {
                header.push_str(&text);
                self.resolve_header(false, out);
            }
            State::ToolCall(body) | State::PythonTag(body) => body.push_str(&text),
            State::JsonCall(capture) => capture.text.push_str(&text),
        }
    }

    /// Carry out a recognised marker.
    fn apply(
        &mut self,
        action: Action,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        if matches!(self.state, State::JsonCall(_))
            && let State::JsonCall(capture) = std::mem::replace(&mut self.state, State::Content)
        {
            self.emit_text_block(capture.text, out);
        }
        match action {
            Action::OpenReasoning => {
                self.trim_leading_ws = true;
                self.state = if self.header_word.is_empty() {
                    State::Thinking
                } else {
                    State::Header(String::new())
                };
            }
            Action::CloseReasoning => {
                self.resolve_header(true, out);
                self.state = State::Content;
                self.trim_leading_ws = true;
                self.json_call_window =
                    matches!(self.format.tool_syntax(), ToolSyntax::LlamaJson { .. });
            }
            Action::StrayClose => {}
            Action::OpenTool => {
                self.resolve_header(true, out);
                self.state = State::ToolCall(String::new());
            }
            Action::CloseTool => {
                let State::ToolCall(body) = std::mem::replace(&mut self.state, State::Content)
                else {
                    return;
                };
                self.finish_tool_call(body, next_call_id, out);
            }
            Action::OpenPython => self.state = State::PythonTag(String::new()),
        }
    }

    /// Emit answer text, dropping prefix whitespace as configured and starting
    /// a bare-JSON tool-call candidate when allowed.
    fn emit_content(
        &mut self,
        text: String,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        let Some(text) = self.trim_prefix(text) else {
            return;
        };
        if self.json_call_window {
            let trimmed = text.trim_start();
            if trimmed.is_empty() {
                // Whitespace before a possible call is dropped, like the
                // whitespace after a reasoning block.
                return;
            }
            self.json_call_window = false;
            if trimmed.starts_with('{') {
                let mut capture = JsonCapture::default();
                let (consumed, complete) = capture.advance(trimmed);
                if complete {
                    let rest = trimmed[consumed..].to_string();
                    self.complete_json(capture.text, next_call_id, out);
                    self.emit_content(rest, next_call_id, out);
                } else {
                    self.state = State::JsonCall(capture);
                }
                return;
            }
        }
        self.emit_text_block(text, out);
    }

    /// Emit reasoning text, dropping prefix whitespace as configured.
    fn emit_thinking(
        &mut self,
        text: String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        if let Some(text) = self.trim_prefix(text) {
            out.push(Ok(MessageContentBlock::Thinking {
                reasoning: text,
                signature: None,
            }));
        }
    }

    /// Push a text block unless it is empty.
    fn emit_text_block(&self, text: String, out: &mut Vec<Result<MessageContentBlock, CucaError>>) {
        if !text.is_empty() {
            out.push(Ok(MessageContentBlock::Text(text)));
        }
    }

    /// Apply the one-shot prefix trims; `None` when nothing is left.
    fn trim_prefix(&mut self, text: String) -> Option<String> {
        if text.is_empty() {
            return None;
        }
        let mut text = text;
        if self.strip_one_space {
            self.strip_one_space = false;
            if text.starts_with(' ') {
                text.remove(0);
            }
        }
        if self.trim_leading_ws {
            let trimmed = text.trim_start();
            if trimmed.is_empty() {
                return None;
            }
            if trimmed.len() != text.len() {
                text = trimmed.to_string();
            }
            self.trim_leading_ws = false;
        }
        (!text.is_empty()).then_some(text)
    }

    /// Consume the hold buffer into the bare-JSON candidate; returns whether the
    /// feed loop should continue (the candidate completed and text remains).
    fn scan_json(
        &mut self,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) -> bool {
        let State::JsonCall(capture) = &mut self.state else {
            return false;
        };
        let (consumed, complete) = capture.advance(&self.hold);
        self.hold.drain(..consumed);
        if !complete {
            return false;
        }
        let State::JsonCall(capture) = std::mem::replace(&mut self.state, State::Content) else {
            return false;
        };
        self.complete_json(capture.text, next_call_id, out);
        !self.hold.is_empty()
    }

    /// Resolve a closed bare-JSON candidate: a tool call when it has the call
    /// shape, answer text otherwise.
    fn complete_json(
        &mut self,
        text: String,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        match parse_json_calls(&text, true) {
            Ok(calls) => {
                for (name, arguments) in calls {
                    out.push(Ok(MessageContentBlock::ToolCall {
                        id: next_call_id(),
                        name,
                        arguments,
                    }));
                }
                self.trim_leading_ws = true;
            }
            Err(_) => self.emit_text_block(text, out),
        }
    }

    /// Decide whether the reasoning header is complete and, if so, move to
    /// `Thinking`, emitting the body text already seen.
    ///
    /// A header starting with the header word drops that word and the
    /// whitespace after it; any other header is kept as body text. Unless
    /// `force` is set the decision waits until it is known: a header still
    /// being a prefix of the word, or the word followed only by whitespace,
    /// stays open.
    fn resolve_header(
        &mut self,
        force: bool,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        let State::Header(header) = &mut self.state else {
            return;
        };
        let word = self.header_word.as_str();
        let body = if let Some(rest) = header.strip_prefix(word) {
            let trimmed = rest.trim_start_matches([' ', '\t', '\n']);
            if trimmed.is_empty() && !force {
                return;
            }
            trimmed.to_string()
        } else {
            if !force && word.starts_with(header.as_str()) {
                return;
            }
            std::mem::take(header)
        };
        self.state = State::Thinking;
        self.emit_thinking(body, out);
    }

    /// Parse a collected tool-call body and emit the result.
    fn finish_tool_call(
        &mut self,
        body: String,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        self.trim_leading_ws = true;
        let parsed = match self.format.tool_syntax() {
            ToolSyntax::Gemma4 { .. } => parse_tool_call(&body).map(|call| vec![call]),
            ToolSyntax::TaggedXml { .. } => parse_xml_call(&body).map(|call| vec![call]),
            _ => parse_json_calls(&body, false),
        };
        match parsed {
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

    /// Emit a `<|python_tag|>` body: JSON calls as such, anything else as a
    /// `python` call carrying the code.
    fn finish_python(
        &mut self,
        body: &str,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        if body.trim().is_empty() {
            return;
        }
        let calls = parse_json_calls(body, true).unwrap_or_else(|_| {
            vec![(
                PYTHON_TOOL.to_string(),
                serde_json::json!({ "code": body.trim() }),
            )]
        });
        for (name, arguments) in calls {
            out.push(Ok(MessageContentBlock::ToolCall {
                id: next_call_id(),
                name,
                arguments,
            }));
        }
    }
}

/// An `OpenTool` marker.
fn open_tool(text: &str) -> Marker {
    Marker {
        text: text.to_string(),
        action: Action::OpenTool,
    }
}

/// A `CloseTool` marker.
fn close_tool(text: &str) -> Marker {
    Marker {
        text: text.to_string(),
        action: Action::CloseTool,
    }
}

/// The leftmost (then longest) marker occurrence in `hold`, as
/// `(start, length, action)`.
fn earliest_marker(hold: &str, markers: &[Marker]) -> Option<(usize, usize, Action)> {
    markers
        .iter()
        .filter_map(|m| hold.find(m.text.as_str()).map(|pos| (pos, m)))
        .min_by(|a, b| a.0.cmp(&b.0).then(b.1.text.len().cmp(&a.1.text.len())))
        .map(|(pos, m)| (pos, m.text.len(), m.action))
}

/// Length in bytes of the longest suffix of `hold` that is a proper prefix of
/// some marker (text that may still grow into one).
fn partial_marker_suffix(hold: &str, markers: &[Marker]) -> usize {
    let mut best = 0;
    for marker in markers {
        let text = marker.text.as_str();
        let longest = text.len().saturating_sub(1).min(hold.len());
        for k in (best + 1..=longest).rev() {
            if text.is_char_boundary(k) && hold.ends_with(&text[..k]) {
                best = k;
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::quantized::gguf_file;
    use serde_json::{Value, json};

    use super::*;

    /// Gemma-style vocabulary: `(text, GGUF type)`; ids follow the order.
    const GEMMA_VOCAB: &[(&str, i32)] = &[
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

    /// Template text that makes detection pick the Gemma-style format.
    const GEMMA_TEMPLATE: &str = "<|channel>thought\n <|tool_call>call:f{}<tool_call|>";

    /// Hermes/Qwen-style vocabulary with `<think>` and `<tool_call>` tags, some
    /// as single tokens and some spelled over several.
    const HERMES_VOCAB: &[(&str, i32)] = &[
        ("<|im_end|>", 3),
        ("<think>", 4),
        ("</think>", 4),
        ("<tool_call>", 4),
        ("</tool_call>", 4),
        ("hello", 1),
        ("\n", 1),
        ("\n\n", 1),
        ("abc", 1),
        ("{\"name\": \"f\", \"arguments\": {\"x\": 1}}", 1),
        (
            "[{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\", \"arguments\": {}}]",
            1,
        ),
        ("{not json}", 1),
        ("<", 1),
        ("tool_call", 1),
        (">", 1),
        ("</", 1),
        ("think", 1),
        (" x", 1),
        ("\u{2581}hi", 1),
    ];
    /// Template text for the Hermes-style format.
    const HERMES_TEMPLATE: &str = "<think> </think> <tool_call> </tool_call>";

    /// Llama-3-style vocabulary: bare JSON calls and a python tag.
    const LLAMA_VOCAB: &[(&str, i32)] = &[
        ("<|eot_id|>", 3),
        ("<|eom_id|>", 3),
        ("<|python_tag|>", 3),
        ("hello", 1),
        ("{\"name\": \"f\", ", 1),
        ("\"parameters\": {\"x\": \"}\"}", 1),
        ("}", 1),
        (" tail", 1),
        ("{\"name\": \"only\"}", 1),
        ("{\"a\": 1}", 1),
        ("{", 1),
        ("{\"name\": \"f\", \"parameters\": {\"x\": 1}}", 1),
        ("print(1)", 1),
        ("\n", 1),
        ("  ", 1),
    ];
    /// Template text for the Llama-3-style format.
    const LLAMA_TEMPLATE: &str = "{% if message.tool_calls %}{\"name\": x, \"parameters\": y}";

    /// A parser over a synthetic vocabulary, fed token texts.
    struct Rig {
        tokenizer: Tokenizer,
        parser: OutputParser,
        out: Vec<Result<MessageContentBlock, CucaError>>,
        counter: u32,
    }

    impl Rig {
        fn new(
            vocab: &[(&str, i32)],
            template: &str,
            start_in_thinking: bool,
            strip_space: bool,
        ) -> Self {
            let tokenizer = Tokenizer::new(
                vocab.iter().map(|(t, _)| t.to_string()).collect(),
                vocab.iter().map(|(_, ty)| *ty).collect(),
                &[],
            )
            .expect("synthetic vocabulary is valid");
            let md = HashMap::from([(
                "tokenizer.ggml.eos_token_id".to_string(),
                gguf_file::Value::U32(0),
            )]);
            let format = ChatFormat::detect(&md, &tokenizer, template)
                .expect("format detects")
                .with_leading_space_strip(strip_space);
            Self {
                parser: OutputParser::new(&format, start_in_thinking),
                tokenizer,
                out: Vec::new(),
                counter: 0,
            }
        }

        fn push(&mut self, texts: &[&str]) {
            for text in texts {
                let id = self
                    .tokenizer
                    .token_id(text)
                    .unwrap_or_else(|| panic!("`{text}` is not in the synthetic vocabulary"));
                let counter = &mut self.counter;
                let mut next = || {
                    *counter += 1;
                    format!("call_{counter}")
                };
                self.parser
                    .push(id, &self.tokenizer, &mut next, &mut self.out);
            }
        }

        fn finish(mut self) -> (Vec<Result<MessageContentBlock, CucaError>>, u32) {
            let counter = &mut self.counter;
            let mut next = || {
                *counter += 1;
                format!("call_{counter}")
            };
            self.parser.finish(&mut next, &mut self.out);
            (self.out, self.parser.reasoning_tokens())
        }
    }

    fn run(
        texts: &[&str],
        start_in_thinking: bool,
    ) -> (Vec<Result<MessageContentBlock, CucaError>>, u32) {
        let mut rig = Rig::new(GEMMA_VOCAB, GEMMA_TEMPLATE, start_in_thinking, false);
        rig.push(texts);
        rig.finish()
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
    fn think_tags_reasoning_and_tool_call() {
        let mut rig = Rig::new(HERMES_VOCAB, HERMES_TEMPLATE, false, false);
        rig.push(&[
            "<think>",
            "\n",
            "abc",
            "\n",
            "</think>",
            "\n\n",
            "hello",
            "<tool_call>",
            "\n",
            "{\"name\": \"f\", \"arguments\": {\"x\": 1}}",
            "\n",
            "</tool_call>",
            "\n",
            "<tool_call>",
            "{\"name\": \"f\", \"arguments\": {\"x\": 1}}",
            "</tool_call>",
        ]);
        let (out, reasoning) = rig.finish();
        assert_eq!(
            ok(out),
            [
                thinking("abc"),
                thinking("\n"),
                text("hello"),
                call("call_1", "f", json!({ "x": 1 })),
                call("call_2", "f", json!({ "x": 1 })),
            ]
        );
        assert_eq!(reasoning, 5);
    }

    #[test]
    fn markers_spelled_over_several_tokens_are_matched_on_text() {
        let mut rig = Rig::new(HERMES_VOCAB, HERMES_TEMPLATE, false, false);
        rig.push(&["hello", "<"]);
        assert_eq!(rig.out.len(), 1, "a possible marker prefix is held back");
        rig.push(&["tool_call", ">"]);
        rig.push(&["{\"name\": \"f\", \"arguments\": {\"x\": 1}}"]);
        rig.push(&["</", "tool_call", ">", "<", "abc"]);
        let (out, _) = rig.finish();
        assert_eq!(
            ok(out),
            [
                text("hello"),
                call("call_1", "f", json!({ "x": 1 })),
                text("<abc"),
            ]
        );
        let mut rig = Rig::new(HERMES_VOCAB, HERMES_TEMPLATE, true, false);
        rig.push(&["abc", "</", "think", ">", "hello"]);
        let (out, reasoning) = rig.finish();
        assert_eq!(ok(out), [thinking("abc"), text("hello")]);
        assert_eq!(reasoning, 4);
    }

    #[test]
    fn dangling_marker_prefix_is_flushed_at_finish() {
        let mut rig = Rig::new(HERMES_VOCAB, HERMES_TEMPLATE, false, false);
        rig.push(&["hello", "<", "tool_call"]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [text("hello"), text("<tool_call")]);
    }

    #[test]
    fn tagged_json_lists_and_errors() {
        let mut rig = Rig::new(HERMES_VOCAB, HERMES_TEMPLATE, false, false);
        rig.push(&[
            "<tool_call>",
            "[{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\", \"arguments\": {}}]",
            "</tool_call>",
            "<tool_call>",
            "{not json}",
            "</tool_call>",
            "hello",
        ]);
        let (out, _) = rig.finish();
        assert_eq!(out.len(), 4);
        assert_eq!(
            out[0].as_ref().expect("call"),
            &call("call_1", "a", json!({}))
        );
        assert_eq!(
            out[1].as_ref().expect("call"),
            &call("call_2", "b", json!({}))
        );
        assert!(
            out[2]
                .as_ref()
                .expect_err("malformed")
                .to_string()
                .contains("malformed tool call: {not json}")
        );
        assert_eq!(out[3].as_ref().expect("text"), &text("hello"));
    }

    #[test]
    fn xml_tool_calls_follow_the_qwen35_template() {
        let template = include_str!("../testdata/templates/qwen35.jinja");
        let vocab: &[(&str, i32)] = &[
            ("<|im_end|>", 3),
            ("<think>", 4),
            ("</think>", 4),
            ("<tool_call>", 4),
            ("</tool_call>", 4),
            ("\n\n", 1),
            ("Checking.", 1),
            (
                "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
                1,
            ),
            ("<parameter=days>\n3\n</parameter>\n</function>\n", 1),
            ("<function=oops>\n<parameter=a>\n", 1),
        ];
        let mut rig = Rig::new(vocab, template, false, false);
        assert!(matches!(
            rig.parser.format.tool_syntax(),
            ToolSyntax::TaggedXml { .. }
        ));
        rig.push(&[
            "<think>",
            "\n\n",
            "</think>",
            "\n\n",
            "Checking.",
            "\n\n",
            "<tool_call>",
            "\n\n",
            "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
            "<parameter=days>\n3\n</parameter>\n</function>\n",
            "</tool_call>",
            "<tool_call>",
            "<function=oops>\n<parameter=a>\n",
            "</tool_call>",
        ]);
        let (out, _) = rig.finish();
        let calls: Vec<_> = out
            .iter()
            .filter_map(|b| match b {
                Ok(MessageContentBlock::ToolCall {
                    name, arguments, ..
                }) => Some((name.as_str(), arguments.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            [("get_weather", json!({ "city": "Paris", "days": 3 }))]
        );
        assert!(out.iter().any(|b| b.as_ref().err().is_some_and(|e| {
            e.to_string()
                .contains("malformed tool call: <function=oops>")
        })));
        assert!(
            out.iter()
                .any(|b| b.as_ref().is_ok_and(|b| b == &text("Checking.")))
        );
    }

    #[test]
    fn alternate_opener_spelling_is_accepted() {
        let mut rig = Rig::new(
            &[
                ("<eos>", 3),
                ("<|tool_call|>", 3),
                ("<", 1),
                ("tool_call", 1),
                (">", 1),
                ("[{\"name\": \"a\", \"arguments\": {\"k\": 1}}]", 1),
            ],
            "respond with <|tool_call|> then JSON",
            false,
            false,
        );
        rig.push(&[
            "<",
            "tool_call",
            ">",
            "[{\"name\": \"a\", \"arguments\": {\"k\": 1}}]",
        ]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [call("call_1", "a", json!({ "k": 1 }))]);
    }

    #[test]
    fn tag_without_close_runs_to_the_end_of_generation() {
        let mut rig = Rig::new(
            &[
                ("<eos>", 3),
                ("<|tool_call|>", 3),
                ("hello", 1),
                ("[{\"name\": \"a\", \"arguments\": {\"k\": 1}}]", 1),
            ],
            "respond with <|tool_call|> then JSON",
            false,
            false,
        );
        rig.push(&[
            "hello",
            "<|tool_call|>",
            "[{\"name\": \"a\", \"arguments\": {\"k\": 1}}]",
        ]);
        let (out, _) = rig.finish();
        assert_eq!(
            ok(out),
            [text("hello"), call("call_1", "a", json!({ "k": 1 }))]
        );
    }

    #[test]
    fn bare_json_reply_is_a_tool_call() {
        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&[
            "  ",
            "{\"name\": \"f\", ",
            "\"parameters\": {\"x\": \"}\"}",
            "}",
            " tail",
        ]);
        let (out, _) = rig.finish();
        assert_eq!(
            ok(out),
            [call("call_1", "f", json!({ "x": "}" })), text("tail")]
        );
    }

    #[test]
    fn bare_json_that_is_not_a_call_stays_text() {
        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&["{\"a\": 1}", " tail"]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [text("{\"a\": 1}"), text(" tail")]);

        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&["{\"name\": \"only\"}"]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [text("{\"name\": \"only\"}")]);

        // Truncated candidates come out as text at the end of generation.
        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&["{", "hello"]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [text("{hello")]);
    }

    #[test]
    fn brace_after_answer_text_is_not_a_call() {
        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&["hello", "{\"name\": \"f\", \"parameters\": {\"x\": 1}}"]);
        let (out, _) = rig.finish();
        assert_eq!(
            ok(out),
            [
                text("hello"),
                text("{\"name\": \"f\", \"parameters\": {\"x\": 1}}")
            ]
        );
    }

    #[test]
    fn python_tag_bodies_become_calls() {
        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&["hello", "<|python_tag|>", "print(1)"]);
        let (out, _) = rig.finish();
        assert_eq!(
            ok(out),
            [
                text("hello"),
                call("call_1", "python", json!({ "code": "print(1)" }))
            ]
        );
        let mut rig = Rig::new(LLAMA_VOCAB, LLAMA_TEMPLATE, false, false);
        rig.push(&[
            "<|python_tag|>",
            "{\"name\": \"f\", \"parameters\": {\"x\": 1}}",
        ]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [call("call_1", "f", json!({ "x": 1 }))]);
    }

    #[test]
    fn leading_space_of_the_first_text_is_stripped_when_configured() {
        let mut rig = Rig::new(HERMES_VOCAB, HERMES_TEMPLATE, false, true);
        rig.push(&["\u{2581}hi", " x"]);
        let (out, _) = rig.finish();
        assert_eq!(ok(out), [text("hi"), text(" x")]);
    }

    #[test]
    fn plain_format_never_looks_for_markers() {
        let mut rig = Rig::new(HERMES_VOCAB, "{{ messages }}", false, false);
        rig.push(&["<think>", "abc", "</think>", "<tool_call>"]);
        let (out, reasoning) = rig.finish();
        assert_eq!(
            ok(out),
            [
                text("<think>"),
                text("abc"),
                text("</think>"),
                text("<tool_call>")
            ]
        );
        assert_eq!(reasoning, 0);
    }
}
