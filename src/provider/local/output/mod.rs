//! Streaming parser from generated token ids to [`MessageContentBlock`]s.
//!
//! What a chat model writes besides plain answer text depends on its family,
//! and [`ChatFormat`] (derived from the GGUF, see `chat::format`) says which of
//! these [`OutputParser`] looks for:
//!
//! - a **reasoning block** (`<|channel>thought\n…<channel|>`,
//!   `<think>…</think>` or `[THINK]…[/THINK]`), which becomes
//!   [`MessageContentBlock::Thinking`]; a header word such as `thought` and
//!   the whitespace after it are dropped;
//! - **tool calls**, whose body is buffered whole and parsed into
//!   [`MessageContentBlock::ToolCall`]s, or reported as a provider error
//!   (`malformed tool call: <body>`) when malformed:
//!   the custom `call:name{key:<|"|>value<|"|>}` grammar between control tokens
//!   ([`calls::parse_tool_call`]), tagged JSON (`<tool_call>{"name": …,
//!   "arguments": …}</tool_call>`, also a JSON list, with or without a closing
//!   tag), the XML style of Qwen 3.5 (`<tool_call><function=name>
//!   <parameter=key>value</parameter></function></tool_call>`,
//!   [`calls::parse_xml_call`]), Mistral's `[TOOL_CALLS]name[ARGS]{json}`
//!   (repeated per call, no closing marker: a call ends at the next opener or
//!   at the end of generation, [`calls::parse_args_call`]; an opener inside
//!   reasoning is reasoning text), and DeepSeek V4.1's DSML block
//!   (`<｜DSML｜ calls>` with one `<｜DSML｜ invoke name="f">` per call and a
//!   `<｜DSML｜ parameter name="k" string="true|false">` per argument,
//!   [`calls::parse_dsml_calls`]; the blank line before the block is part of
//!   its opener, and the opener inside reasoning ends it and starts the call,
//!   as in DeepSeek's own stream parser);
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
//! header or a tool-call body are bounded by the
//! generation length the engine already caps.

mod atem;
mod calls;
#[cfg(test)]
mod golden_deepseek;

use self::atem::AtemParser;
use self::calls::{
    parse_args_call, parse_dsml_calls, parse_json_calls, parse_tool_call, parse_xml_call,
};
use super::chat::{ChatFormat, ToolSyntax};
use super::tokenizer::Tokenizer;
use crate::error::CucaError;
use crate::types::{MessageContentBlock, ProviderEndpoint};

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
    /// End the current tool-call body and start the next one (a format whose
    /// calls have no closing marker).
    NextTool,
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
    /// The channel-protocol parser that replaces the marker state machine for
    /// a format that writes messages addressed to `self`, `user` and tools.
    atem: Option<AtemParser>,
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
            ToolSyntax::Plain | ToolSyntax::Atem(_) => {}
            ToolSyntax::Gemma4 { open, close } | ToolSyntax::TaggedXml { open, close } => {
                markers.content.push(open_tool(open));
                markers.thinking.push(open_tool(open));
                markers.tool.push(close_tool(close));
            }
            // `\n\n<｜DSML｜ calls>` is what the model was trained to write after
            // its answer, so the blank line belongs to the opener (the longer
            // spelling wins at the same place, and a bare opener is accepted
            // too). DeepSeek's own parser treats the opener inside reasoning as
            // the start of a call, so the markers apply there as well.
            ToolSyntax::Dsml { dsml, calls } => {
                let open = format!("<{dsml}{calls}>");
                for text in [format!("\n\n{open}"), open] {
                    markers.content.push(open_tool(&text));
                    markers.thinking.push(open_tool(&text));
                }
                markers.tool.push(close_tool(&format!("</{dsml}{calls}>")));
            }
            // The opener is not a marker inside reasoning: a model may write
            // it there as text.
            ToolSyntax::MistralArgs { open, .. } => {
                markers.content.push(open_tool(open));
                markers.tool.push(Marker {
                    text: open.clone(),
                    action: Action::NextTool,
                });
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
            atem: match format.tool_syntax() {
                ToolSyntax::Atem(markers) => Some(AtemParser::new(markers)),
                _ => None,
            },
            format: format.clone(),
        }
    }

    /// Number of pushed ids inside a reasoning block, counting the opening and
    /// closing marker ids (and every id before the first close when started in
    /// thinking).
    pub(crate) fn reasoning_tokens(&self) -> u32 {
        match &self.atem {
            Some(atem) => atem.reasoning_tokens(),
            None => self.reasoning_tokens,
        }
    }

    /// Feed one generated id, appending any completed blocks to `out`.
    pub(crate) fn push(
        &mut self,
        id: u32,
        tok: &Tokenizer,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        if let Some(atem) = &mut self.atem {
            atem.push(id, &self.format, tok, next_call_id, out);
            return;
        }
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
        if let Some(atem) = &mut self.atem {
            atem.finish(next_call_id, out);
            return;
        }
        self.flush_pending(next_call_id, out);
        let held = std::mem::take(&mut self.hold);
        match std::mem::replace(&mut self.state, State::Content) {
            State::Content => self.emit_content(held, out),
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
        while !self.pending.is_empty() {
            let error = match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    self.hold.push_str(text);
                    self.pending.clear();
                    self.process_hold(next_call_id, out);
                    return;
                }
                Err(e) => e,
            };
            let valid = error.valid_up_to();
            if valid > 0 {
                self.hold
                    .push_str(&String::from_utf8_lossy(&self.pending[..valid]));
                self.pending.drain(..valid);
                self.process_hold(next_call_id, out);
            }
            let Some(invalid) = error.error_len() else {
                // An incomplete character: wait for the rest of it.
                return;
            };
            self.pending.drain(..invalid);
            self.feed_text("\u{FFFD}", next_call_id, out);
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
            let (found, keep) = {
                let markers = self.active_markers();
                match earliest_marker(&self.hold, markers) {
                    Some(hit) => (Some(hit), 0),
                    None => (None, partial_marker_suffix(&self.hold, markers)),
                }
            };
            match found {
                Some((pos, len, action)) => {
                    self.route_hold(pos, out);
                    self.hold.drain(..len);
                    self.apply(action, next_call_id, out);
                }
                None => {
                    let ready_len = self.hold.len() - keep;
                    self.route_hold(ready_len, out);
                    return;
                }
            }
        }
    }

    /// Hand the first `len` bytes of the hold buffer (marker-free text) to the
    /// current state: a tool-call body or a reasoning header appends it, the
    /// other states emit it as a block.
    fn route_hold(&mut self, len: usize, out: &mut Vec<Result<MessageContentBlock, CucaError>>) {
        if len == 0 {
            return;
        }
        match &mut self.state {
            State::ToolCall(body) => body.push_str(&self.hold[..len]),
            State::Header(header) => {
                header.push_str(&self.hold[..len]);
                self.resolve_header(false, out);
            }
            State::Content => {
                let text = self.hold[..len].to_owned();
                self.emit_content(text, out);
            }
            State::Thinking => {
                let text = self.hold[..len].to_owned();
                self.emit_thinking(text, out);
            }
        }
        self.hold.drain(..len);
    }

    /// Carry out a recognised marker.
    fn apply(
        &mut self,
        action: Action,
        next_call_id: &mut dyn FnMut() -> String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
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
            Action::NextTool => {
                let State::ToolCall(body) =
                    std::mem::replace(&mut self.state, State::ToolCall(String::new()))
                else {
                    return;
                };
                self.finish_tool_call(body, next_call_id, out);
            }
        }
    }

    /// Emit answer text, dropping prefix whitespace as configured.
    fn emit_content(
        &mut self,
        text: String,
        out: &mut Vec<Result<MessageContentBlock, CucaError>>,
    ) {
        if let Some(text) = self.trim_prefix(text) {
            out.push(Ok(MessageContentBlock::Text(text)));
        }
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
            ToolSyntax::Gemma4 { .. } => parse_tool_call(&body).map(|call| vec![(None, call)]),
            ToolSyntax::TaggedXml { .. } => parse_xml_call(&body).map(|call| vec![(None, call)]),
            ToolSyntax::Dsml { dsml, .. } => parse_dsml_calls(&body, dsml)
                .map(|calls| calls.into_iter().map(|c| (None, c)).collect()),
            ToolSyntax::MistralArgs { args, call_id, .. } => {
                parse_args_call(&body, args, call_id.as_deref())
                    .map(|(id, name, arguments)| vec![(id, (name, arguments))])
            }
            _ => parse_json_calls(&body)
                .map(|calls| calls.into_iter().map(|call| (None, call)).collect()),
        };
        match parsed {
            Ok(calls) => {
                for (id, (name, arguments)) in calls {
                    out.push(Ok(MessageContentBlock::ToolCall {
                        id: id.unwrap_or_else(&mut *next_call_id),
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

    use crate::provider::local::gguf;
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

    /// Mistral-style vocabulary: the `[THINK]`, `[TOOL_CALLS]`, `[ARGS]` and
    /// `[CALL_ID]` control tokens, and the pieces of llama.cpp's Ministral 3
    /// and Mistral Small 3.2 parser fixtures.
    const MISTRAL_VOCAB: &[(&str, i32)] = &[
        ("</s>", 3),
        ("[THINK]", 3),
        ("[/THINK]", 3),
        ("[TOOL_CALLS]", 3),
        ("[ARGS]", 3),
        ("[CALL_ID]", 3),
        ("Hello, world!", 1),
        ("\n", 1),
        ("What's up?", 1),
        ("I'm", 1),
        ("thinking", 1),
        ("special_function", 1),
        ("special_function_with_opt", 1),
        ("{\"arg1\":1}", 1),
        ("{\"arg1\": 1}", 1),
        ("{\"arg1\": 1, \"arg2\": 2}", 1),
        ("Let me think about ", 1),
        (" and more", 1),
        ("123456789", 1),
        ("000000001", 1),
        ("000000002", 1),
        ("{\"arg1\"", 1),
        (": 1}", 1),
        ("{not json}", 1),
        ("spec", 1),
        ("ial_function", 1),
    ];
    /// Template text that makes detection pick `[TOOL_CALLS]name[ARGS]json`.
    const MISTRAL_TEMPLATE: &str = "[THINK] [/THINK] [TOOL_CALLS] [ARGS]";
    /// The same with Mistral Small 3.2's `[CALL_ID]`.
    const MISTRAL_ID_TEMPLATE: &str = "[TOOL_CALLS] [CALL_ID] [ARGS]";

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
                gguf::Value::U32(0),
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

    /// Adjacent text blocks (and adjacent thinking blocks) joined: a model's
    /// words arrive token by token.
    fn merged(blocks: Vec<MessageContentBlock>) -> Vec<MessageContentBlock> {
        let mut out: Vec<MessageContentBlock> = Vec::new();
        for block in blocks {
            match (out.last_mut(), block) {
                (Some(MessageContentBlock::Text(a)), MessageContentBlock::Text(b)) => {
                    a.push_str(&b)
                }
                (
                    Some(MessageContentBlock::Thinking { reasoning: a, .. }),
                    MessageContentBlock::Thinking { reasoning: b, .. },
                ) => a.push_str(&b),
                (_, block) => out.push(block),
            }
        }
        out
    }

    fn mistral(template: &str, texts: &[&str]) -> (Vec<MessageContentBlock>, u32) {
        let mut rig = Rig::new(MISTRAL_VOCAB, template, false, false);
        rig.push(texts);
        let (out, reasoning) = rig.finish();
        (merged(ok(out)), reasoning)
    }

    // llama.cpp's Ministral 3 parser fixtures (tests/test-chat.cpp, "Ministral-3-14B-Reasoning-2512"),
    // Devstral's and Mistral Small 3.2's, over the same token spellings.

    #[test]
    fn mistral_plain_answer() {
        let (out, reasoning) = mistral(MISTRAL_TEMPLATE, &["Hello, world!", "\n", "What's up?"]);
        assert_eq!(out, [text("Hello, world!\nWhat's up?")]);
        assert_eq!(reasoning, 0);
    }

    #[test]
    fn mistral_think_block_then_answer() {
        let (out, reasoning) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "[THINK]",
                "I'm",
                "\n",
                "thinking",
                "[/THINK]",
                "Hello, world!",
                "\n",
                "What's up?",
            ],
        );
        assert_eq!(
            out,
            [thinking("I'm\nthinking"), text("Hello, world!\nWhat's up?")]
        );
        assert_eq!(reasoning, 5);
    }

    #[test]
    fn mistral_single_call() {
        let (out, _) = mistral(
            MISTRAL_TEMPLATE,
            &["[TOOL_CALLS]", "special_function", "[ARGS]", "{\"arg1\":1}"],
        );
        assert_eq!(
            out,
            [call("call_1", "special_function", json!({ "arg1": 1 }))]
        );
    }

    #[test]
    fn mistral_think_then_call() {
        let (out, _) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "[THINK]",
                "I'm",
                "\n",
                "thinking",
                "[/THINK]",
                "[TOOL_CALLS]",
                "special_function",
                "[ARGS]",
                "{\"arg1\":1}",
            ],
        );
        assert_eq!(
            out,
            [
                thinking("I'm\nthinking"),
                call("call_1", "special_function", json!({ "arg1": 1 }))
            ]
        );
    }

    #[test]
    fn mistral_parallel_calls_end_at_the_next_opener_and_at_the_end() {
        let (out, _) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "[TOOL_CALLS]",
                "special_function",
                "[ARGS]",
                "{\"arg1\": 1}",
                "[TOOL_CALLS]",
                "special_function_with_opt",
                "[ARGS]",
                "{\"arg1\": 1, \"arg2\": 2}",
            ],
        );
        assert_eq!(
            out,
            [
                call("call_1", "special_function", json!({ "arg1": 1 })),
                call(
                    "call_2",
                    "special_function_with_opt",
                    json!({ "arg1": 1, "arg2": 2 })
                ),
            ]
        );
    }

    #[test]
    fn mistral_opener_inside_reasoning_is_reasoning_text() {
        let (out, reasoning) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "[THINK]",
                "Let me think about ",
                "[TOOL_CALLS]",
                "special_function",
                "[ARGS]",
                "{\"arg1\":1}",
                " and more",
                "[/THINK]",
                "[TOOL_CALLS]",
                "special_function",
                "[ARGS]",
                "{\"arg1\": 1}",
            ],
        );
        assert_eq!(
            out,
            [
                thinking(
                    "Let me think about [TOOL_CALLS]special_function[ARGS]{\"arg1\":1} and more"
                ),
                call("call_1", "special_function", json!({ "arg1": 1 }))
            ]
        );
        assert_eq!(reasoning, 8);
    }

    #[test]
    fn mistral_answer_text_before_a_call() {
        // Devstral: content, then the call.
        let (out, _) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "Hello, world!",
                "\n",
                "What's up?",
                "[TOOL_CALLS]",
                "special_function",
                "[ARGS]",
                "{\"arg1\": 1}",
            ],
        );
        assert_eq!(
            out,
            [
                text("Hello, world!\nWhat's up?"),
                call("call_1", "special_function", json!({ "arg1": 1 }))
            ]
        );
    }

    #[test]
    fn mistral_small_3_2_calls_carry_the_models_ids() {
        let (out, _) = mistral(
            MISTRAL_ID_TEMPLATE,
            &[
                "[TOOL_CALLS]",
                "special_function",
                "[CALL_ID]",
                "000000001",
                "[ARGS]",
                "{\"arg1\": 1}",
                "[TOOL_CALLS]",
                "special_function_with_opt",
                "[CALL_ID]",
                "000000002",
                "[ARGS]",
                "{\"arg1\": 1, \"arg2\": 2}",
            ],
        );
        assert_eq!(
            out,
            [
                call("000000001", "special_function", json!({ "arg1": 1 })),
                call(
                    "000000002",
                    "special_function_with_opt",
                    json!({ "arg1": 1, "arg2": 2 })
                ),
            ]
        );
        // A template without `[CALL_ID]` leaves the parser without that
        // marker: the id text stays in the name.
        let (out, _) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "[TOOL_CALLS]",
                "special_function",
                "[ARGS]",
                "{\"arg1\": 1}",
            ],
        );
        assert_eq!(
            out,
            [call("call_1", "special_function", json!({ "arg1": 1 }))]
        );
    }

    #[test]
    fn mistral_names_and_arguments_may_span_several_tokens() {
        let (out, _) = mistral(
            MISTRAL_TEMPLATE,
            &[
                "[TOOL_CALLS]",
                "spec",
                "ial_function",
                "[ARGS]",
                "{\"arg1\"",
                ": 1}",
            ],
        );
        assert_eq!(
            out,
            [call("call_1", "special_function", json!({ "arg1": 1 }))]
        );
    }

    #[test]
    fn mistral_malformed_calls_are_errors_and_do_not_stop_the_next_call() {
        let mut rig = Rig::new(MISTRAL_VOCAB, MISTRAL_TEMPLATE, false, false);
        rig.push(&[
            "[TOOL_CALLS]",
            "special_function",
            "[ARGS]",
            "{not json}",
            "[TOOL_CALLS]",
            "special_function",
            "[ARGS]",
            "{\"arg1\"",
        ]);
        let (out, _) = rig.finish();
        assert_eq!(out.len(), 2);
        for r in out {
            let err = r.expect_err("both bodies are malformed").to_string();
            assert!(err.contains("malformed tool call"), "{err}");
        }
        // A call with no `[ARGS]` at all.
        let mut rig = Rig::new(MISTRAL_VOCAB, MISTRAL_TEMPLATE, false, false);
        rig.push(&["[TOOL_CALLS]", "special_function"]);
        let (out, _) = rig.finish();
        assert!(out[0].is_err());
    }

    #[test]
    fn mistral_syntax_is_not_claimed_without_the_args_token() {
        // The template mentions the markers but the vocabulary has no
        // `[ARGS]`: no tool calls, and the control token renders as nothing.
        let vocab: &[(&str, i32)] = &[("</s>", 3), ("[TOOL_CALLS]", 3), ("a", 1)];
        let mut rig = Rig::new(vocab, MISTRAL_TEMPLATE, false, false);
        rig.push(&["a", "[TOOL_CALLS]"]);
        let (out, _) = rig.finish();
        assert_eq!(merged(ok(out)), [text("a")]);
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
