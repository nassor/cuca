//! What the output side of a chat model looks like, derived from its GGUF.
//!
//! [`ChatFormat::detect`] inspects the vocabulary, the metadata and the Jinja
//! chat template source of a model and records three things, nothing of which
//! is hard-coded per model name:
//!
//! - the **end-of-generation** ids: the EOS/EOT/EOM ids of the metadata plus
//!   every vocabulary token spelled like a known turn or sequence terminator
//!   (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`, `<|end|>`, `<turn|>`, ...),
//!   the same set llama.cpp's loader treats as EOG;
//! - the **reasoning** markers: a channel (`<|channel>thought\n` ...
//!   `<channel|>`), `<think>` ... `</think>` tags or Mistral's `[THINK]` ...
//!   `[/THINK]` control tokens, whichever the template writes, with
//!   [`ChatFormat::starts_in_thinking`] deciding from a rendered prompt
//!   whether generation begins inside an open block;
//! - the **tool-call syntax** the template teaches the model: a custom
//!   `call:name{key:<|"|>value<|"|>}` body between control tokens, tagged JSON
//!   (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`), the XML
//!   style of Qwen 3.5, Mistral's `[TOOL_CALLS]name[ARGS]{json}` or
//!   DeepSeek V4.1's DSML block; a template without tool syntax leaves the
//!   output plain text.

use std::collections::HashMap;

use crate::provider::local::gguf;

use super::local_err;
use crate::error::CucaError;
use crate::provider::local::tokenizer::Tokenizer;

/// Vocabulary texts that end a turn or a sequence (llama.cpp's EOT/EOM list
/// plus the Gemma 4 turn end and tool-response stop).
const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
    "<|eot|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<turn|>",
    "<|tool_response>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<end_of_utterance>",
];

/// A reasoning block delimited by an open and a close marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reasoning {
    /// Text that opens the block (`<|channel>` or `<think>`).
    pub(crate) open: String,
    /// Text that closes it (`<channel|>` or `</think>`).
    pub(crate) close: String,
    /// Header word right after the open marker, dropped from the reasoning
    /// text together with the whitespace behind it (`thought`).
    pub(crate) header: Option<String>,
}

/// The marker texts of the channel protocol Muse Glimmer's chat template
/// teaches ("ATEM"). A turn is a sequence of messages, each
/// `<|start|>assistant to=RECIPIENT<|message|>BODY END`:
///
/// - `to=self`: the model's reasoning;
/// - `to=user`: answer text (a header without a recipient is the same);
/// - any other recipient: a tool call whose body is
///   `<atem:function_calls>` ... `</atem:function_calls>`.
///
/// `END` is `<|eom|>` when more messages follow and `<|eot|>` (an
/// end-of-generation token) at the end of the turn. The prompt already ends
/// with `<|start|>assistant`, so the generated text begins with the
/// ` to=...` of the first header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Atem {
    /// Starts a message header (`<|start|>`).
    pub(crate) start: String,
    /// Ends the header and starts the body (`<|message|>`).
    pub(crate) message: String,
    /// Ends a message that other messages follow (`<|eom|>`).
    pub(crate) end_message: String,
    /// Ends the turn (`<|eot|>`).
    pub(crate) end_turn: String,
}

/// How the model writes tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolSyntax {
    /// No tool-call syntax: every token is answer text.
    Plain,
    /// The channel protocol of [`Atem`]: reasoning, answers and tool calls are
    /// messages addressed to `self`, `user` and a tool.
    Atem(Atem),
    /// `open call:name{key:<|"|>value<|"|>} close` with a custom value grammar.
    Gemma4 {
        /// Text that opens a call (`<|tool_call>`).
        open: String,
        /// Text that closes it (`<tool_call|>`).
        close: String,
    },
    /// `open JSON close`, the JSON being `{"name": ..., "arguments": ...}` or a
    /// list of them. Without a `close` the call runs to the end of generation.
    TaggedJson {
        /// Text that opens a call (`<tool_call>`).
        open: String,
        /// A second spelling of the opener the model is known to write
        /// instead (Granite 3.x is taught `<|tool_call|>` but writes the plain
        /// text `<tool_call>`, in HuggingFace f32 as well).
        alt_open: Option<String>,
        /// Text that closes it (`</tool_call>`), if the format has one.
        close: Option<String>,
    },
    /// `[TOOL_CALLS]name[ARGS]{json}` (Mistral Small 4, Devstral, Ministral 3;
    /// `[TOOL_CALLS]name[CALL_ID]id[ARGS]{json}` for Mistral Small 3.2): one
    /// block per call, no closing marker. A call ends at the next opener or at
    /// the end of generation, and the arguments are one JSON object.
    MistralArgs {
        /// Text that opens each call (`[TOOL_CALLS]`).
        open: String,
        /// Text between the name (and call id) and the arguments (`[ARGS]`).
        args: String,
        /// Text between the name and the call id (`[CALL_ID]`), for templates
        /// that carry one; the id the model writes becomes the call's id.
        call_id: Option<String>,
    },
    /// `open <function=name><parameter=key>value</parameter>... </function> close`
    /// (the Qwen3-Coder XML style Qwen 3.5 is taught): one call per block,
    /// every parameter a `<parameter=key>` element whose value is raw text
    /// (JSON where it is valid JSON).
    TaggedXml {
        /// Text that opens a call (`<tool_call>`).
        open: String,
        /// Text that closes it (`</tool_call>`).
        close: String,
    },
    /// DeepSeek V4.1's DSML: a `<｜DSML｜ calls>` block (the tag name is
    /// `dsml` followed by `calls`) of `<｜DSML｜ invoke name="f">` elements,
    /// each holding `<｜DSML｜ parameter name="k" string="true|false">value
    /// </｜DSML｜ parameter>` elements, closed by `</｜DSML｜ calls>`. A
    /// string parameter is raw text; any other value is JSON. The model writes
    /// the block after the answer and a blank line.
    Dsml {
        /// The marker the tags carry after `<` and `</` (`｜DSML｜`, one
        /// vocabulary token).
        dsml: String,
        /// What follows it in the block's own tag (` calls` for V4.1,
        /// `tool_calls` for V4).
        calls: String,
    },
}

/// The output-side conventions of one chat model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatFormat {
    /// Sorted, deduplicated end-of-generation ids.
    eog: Vec<u32>,
    /// The reasoning block, when the template writes one.
    reasoning: Option<Reasoning>,
    /// The tool-call syntax.
    tools: ToolSyntax,
    /// Ids of single-token markers; the parser reads their raw text even when
    /// the tokenizer renders them as nothing (control tokens).
    marker_ids: Vec<u32>,
    /// Whether the first generated text loses one leading space (SentencePiece
    /// `add_space_prefix`).
    strip_leading_space: bool,
}

/// Read an id from a GGUF metadata value.
fn md_id(md: &HashMap<String, gguf::Value>, key: &str) -> Option<u32> {
    match md.get(key)? {
        gguf::Value::U32(n) => Some(*n),
        gguf::Value::I32(n) => u32::try_from(*n).ok(),
        gguf::Value::U64(n) => u32::try_from(*n).ok(),
        gguf::Value::I64(n) => u32::try_from(*n).ok(),
        gguf::Value::U16(n) => Some(u32::from(*n)),
        gguf::Value::U8(n) => Some(u32::from(*n)),
        _ => None,
    }
}

impl ChatFormat {
    /// Detect the format from the model's metadata, tokenizer and chat template
    /// source.
    ///
    /// Fails when no end-of-generation token can be determined.
    pub(crate) fn detect(
        md: &HashMap<String, gguf::Value>,
        tokenizer: &Tokenizer,
        template_source: &str,
    ) -> Result<Self, CucaError> {
        let vocab_len = tokenizer.vocab_size();
        let mut eog: Vec<u32> = [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ]
        .iter()
        .filter_map(|key| md_id(md, key))
        .filter(|&id| (id as usize) < vocab_len)
        .collect();
        eog.extend(EOG_TEXTS.iter().filter_map(|text| tokenizer.token_id(text)));
        eog.sort_unstable();
        eog.dedup();
        if eog.is_empty() {
            return Err(local_err(
                "cannot determine the end-of-generation token: the GGUF names no EOS/EOT/EOM id \
                 and the vocabulary has no known turn terminator",
            ));
        }

        let reasoning = detect_reasoning(tokenizer, template_source);
        let tools = detect_tools(tokenizer, template_source);
        let mut marker_ids = Vec::new();
        let mut note = |text: &str| {
            if let Some(id) = tokenizer.token_id(text) {
                marker_ids.push(id);
            }
        };
        if let Some(r) = &reasoning {
            note(&r.open);
            note(&r.close);
        }
        match &tools {
            ToolSyntax::Plain => {}
            ToolSyntax::Atem(a) => {
                for text in [&a.start, &a.message, &a.end_message, &a.end_turn] {
                    note(text);
                }
            }
            ToolSyntax::Gemma4 { open, close } | ToolSyntax::TaggedXml { open, close } => {
                note(open);
                note(close);
            }
            ToolSyntax::MistralArgs {
                open,
                args,
                call_id,
            } => {
                note(open);
                note(args);
                if let Some(call_id) = call_id {
                    note(call_id);
                }
            }
            ToolSyntax::TaggedJson {
                open,
                alt_open,
                close,
            } => {
                note(open);
                if let Some(alt) = alt_open {
                    note(alt);
                }
                if let Some(close) = close {
                    note(close);
                }
            }
            ToolSyntax::Dsml { dsml, .. } => note(dsml),
        }
        marker_ids.sort_unstable();
        marker_ids.dedup();

        Ok(Self {
            eog,
            reasoning,
            tools,
            marker_ids,
            strip_leading_space: tokenizer.strips_leading_space(),
        })
    }

    /// Override [`ChatFormat::strips_leading_space`] (SentencePiece behaviour
    /// on a synthetic vocabulary).
    #[cfg(test)]
    pub(crate) fn with_leading_space_strip(mut self, on: bool) -> Self {
        self.strip_leading_space = on;
        self
    }

    /// Whether `id` ends generation.
    pub(crate) fn is_eog(&self, id: u32) -> bool {
        self.eog.binary_search(&id).is_ok()
    }

    /// The reasoning block markers, if the model has any.
    pub(crate) fn reasoning(&self) -> Option<&Reasoning> {
        self.reasoning.as_ref()
    }

    /// Whether the model writes reasoning the parser reports as `Thinking`
    /// blocks: a marked block, or messages addressed to `self`.
    pub(crate) fn writes_reasoning(&self) -> bool {
        self.reasoning.is_some() || matches!(self.tools, ToolSyntax::Atem(_))
    }

    /// The tool-call syntax.
    pub(crate) fn tool_syntax(&self) -> &ToolSyntax {
        &self.tools
    }

    /// Whether `id` is a single-token marker whose raw text the parser needs.
    pub(crate) fn is_marker(&self, id: u32) -> bool {
        self.marker_ids.binary_search(&id).is_ok()
    }

    /// Whether the first generated text loses one leading space.
    pub(crate) fn strips_leading_space(&self) -> bool {
        self.strip_leading_space
    }

    /// Whether a rendered prompt ends inside an open reasoning block, so the
    /// generated text starts as reasoning: the last open marker has no close
    /// after it and only whitespace (or the header word) follows it.
    pub(crate) fn starts_in_thinking(&self, rendered: &str) -> bool {
        let Some(reasoning) = &self.reasoning else {
            return false;
        };
        let Some(open_at) = rendered.rfind(&reasoning.open) else {
            return false;
        };
        let after = &rendered[open_at + reasoning.open.len()..];
        if after.contains(&reasoning.close) {
            return false;
        }
        let tail = after.trim();
        tail.is_empty() || reasoning.header.as_deref() == Some(tail)
    }
}

/// Find the reasoning block the template writes.
fn detect_reasoning(tokenizer: &Tokenizer, template: &str) -> Option<Reasoning> {
    let both = |open: &str, close: &str| {
        tokenizer.token_id(open).is_some() && tokenizer.token_id(close).is_some()
    };
    if template.contains("<|channel>") && both("<|channel>", "<channel|>") {
        return Some(Reasoning {
            open: "<|channel>".into(),
            close: "<channel|>".into(),
            header: Some("thought".into()),
        });
    }
    if template.contains("[THINK]") && both("[THINK]", "[/THINK]") {
        return Some(Reasoning {
            open: "[THINK]".into(),
            close: "[/THINK]".into(),
            header: None,
        });
    }
    if template.contains("<think>") || template.contains("</think>") {
        return Some(Reasoning {
            open: "<think>".into(),
            close: "</think>".into(),
            header: None,
        });
    }
    None
}

/// Find the tool-call syntax the template teaches.
fn detect_tools(tokenizer: &Tokenizer, template: &str) -> ToolSyntax {
    if template.contains("<atem:function_calls>")
        && template.contains("<|eom|>")
        && ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
            .iter()
            .all(|t| tokenizer.token_id(t).is_some())
    {
        return ToolSyntax::Atem(Atem {
            start: "<|start|>".into(),
            message: "<|message|>".into(),
            end_message: "<|eom|>".into(),
            end_turn: "<|eot|>".into(),
        });
    }
    if template.contains("｜DSML｜")
        && tokenizer.token_id("｜DSML｜").is_some()
        && let Some(calls) = [" calls", "tool_calls"]
            .into_iter()
            .find(|calls| template.contains(&format!("｜DSML｜{calls}")))
    {
        return ToolSyntax::Dsml {
            dsml: "｜DSML｜".into(),
            calls: calls.into(),
        };
    }
    if template.contains("[TOOL_CALLS]")
        && template.contains("[ARGS]")
        && ["[TOOL_CALLS]", "[ARGS]"]
            .iter()
            .all(|t| tokenizer.token_id(t).is_some())
    {
        return ToolSyntax::MistralArgs {
            open: "[TOOL_CALLS]".into(),
            args: "[ARGS]".into(),
            call_id: (template.contains("[CALL_ID]") && tokenizer.token_id("[CALL_ID]").is_some())
                .then(|| "[CALL_ID]".to_string()),
        };
    }
    if template.contains("<|tool_call>")
        && tokenizer.token_id("<|tool_call>").is_some()
        && tokenizer.token_id("<tool_call|>").is_some()
    {
        return ToolSyntax::Gemma4 {
            open: "<|tool_call>".into(),
            close: "<tool_call|>".into(),
        };
    }
    if template.contains("<tool_call>")
        && template.contains("</tool_call>")
        && template.contains("<function=")
        && template.contains("<parameter=")
    {
        return ToolSyntax::TaggedXml {
            open: "<tool_call>".into(),
            close: "</tool_call>".into(),
        };
    }
    if template.contains("<tool_call>") {
        return ToolSyntax::TaggedJson {
            open: "<tool_call>".into(),
            alt_open: None,
            close: template
                .contains("</tool_call>")
                .then(|| "</tool_call>".to_string()),
        };
    }
    if template.contains("<|tool_call|>") {
        return ToolSyntax::TaggedJson {
            open: "<|tool_call|>".into(),
            alt_open: Some("<tool_call>".into()),
            close: template
                .contains("<|/tool_call|>")
                .then(|| "<|/tool_call|>".to_string()),
        };
    }
    ToolSyntax::Plain
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A synthetic tokenizer holding `texts` as control tokens, in order.
    pub(crate) fn tokenizer(texts: &[&str]) -> Tokenizer {
        Tokenizer::new(
            texts.iter().map(|t| (*t).to_string()).collect(),
            vec![3; texts.len()],
            &[],
        )
        .expect("synthetic vocabulary is valid")
    }

    fn metadata(eos: u32) -> HashMap<String, gguf::Value> {
        HashMap::from([(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf::Value::U32(eos),
        )])
    }

    #[test]
    fn mistral_templates_are_detected_from_their_markers() {
        let t = tokenizer(&[
            "</s>",
            "[THINK]",
            "[/THINK]",
            "[TOOL_CALLS]",
            "[ARGS]",
            "[CALL_ID]",
            "x",
        ]);
        let small_4 = include_str!("../testdata/templates/mistral-small-4.jinja");
        let f = ChatFormat::detect(&metadata(0), &t, small_4).expect("detects");
        assert_eq!(
            f.reasoning(),
            Some(&Reasoning {
                open: "[THINK]".into(),
                close: "[/THINK]".into(),
                header: None
            })
        );
        // Small 4's template has no call ids.
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::MistralArgs {
                open: "[TOOL_CALLS]".into(),
                args: "[ARGS]".into(),
                call_id: None
            }
        );
        assert!(f.writes_reasoning());
        for id in 1..=4 {
            assert!(f.is_marker(id), "{id}");
        }
        assert!(!f.is_marker(5) && f.is_eog(0));
        // The prompt ends after `[/INST]` (or the tool results): generation
        // starts in the answer, the model opens `[THINK]` itself.
        assert!(!f.starts_in_thinking("[INST]hi[/INST]"));
        assert!(f.starts_in_thinking("[INST]hi[/INST][THINK]"));

        // Mistral Small 3.2 names the call id, and has no reasoning.
        let small_3_2 = include_str!("../testdata/templates/mistral-small-3.2.jinja");
        let f = ChatFormat::detect(&metadata(0), &t, small_3_2).expect("detects");
        assert!(f.reasoning().is_none() && !f.writes_reasoning());
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::MistralArgs {
                open: "[TOOL_CALLS]".into(),
                args: "[ARGS]".into(),
                call_id: Some("[CALL_ID]".into())
            }
        );
        assert!(f.is_marker(5));

        // The current Ministral 3 Reasoning template (the one the preset
        // pins in place of the GGUF's): `[THINK]` and no call ids.
        let ministral = include_str!("../testdata/templates/ministral3-reasoning.jinja");
        let f = ChatFormat::detect(&metadata(0), &t, ministral).expect("detects");
        assert_eq!(f.reasoning().map(|r| r.open.as_str()), Some("[THINK]"));
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::MistralArgs {
                open: "[TOOL_CALLS]".into(),
                args: "[ARGS]".into(),
                call_id: None
            }
        );

        // A vocabulary without the control tokens leaves the output plain.
        let bare = tokenizer(&["</s>", "x"]);
        let f = ChatFormat::detect(&metadata(0), &bare, small_4).expect("detects");
        assert!(f.reasoning().is_none());
        assert_eq!(f.tool_syntax(), &ToolSyntax::Plain);
    }

    #[test]
    fn atem_channel_protocol_is_detected_and_eom_does_not_end_generation() {
        let t = tokenizer(&[
            "<|end_of_text|>",
            "<|start|>",
            "<|message|>",
            "<|eom|>",
            "<|eot|>",
            "x",
        ]);
        let template = "<atem:function_calls>...<|eom|>...<|eot|>";
        // The GGUF names `<|end_of_text|>` as EOS only; `<|eot|>` is an
        // end-of-generation id by its spelling.
        let f = ChatFormat::detect(&metadata(0), &t, template).expect("detects");
        assert!(f.is_eog(0) && f.is_eog(4));
        assert!(!f.is_eog(3), "<|eom|> ends a message, not the turn");
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::Atem(Atem {
                start: "<|start|>".into(),
                message: "<|message|>".into(),
                end_message: "<|eom|>".into(),
                end_turn: "<|eot|>".into(),
            })
        );
        for id in 1..=4 {
            assert!(f.is_marker(id), "{id}");
        }
        assert!(f.writes_reasoning() && f.reasoning().is_none());
        assert!(!f.starts_in_thinking("<|start|>assistant"));
        // Without the markers in the vocabulary the syntax is not claimed.
        let bare = tokenizer(&["<|end_of_text|>", "x"]);
        let f = ChatFormat::detect(&metadata(0), &bare, template).expect("detects");
        assert!(!f.writes_reasoning());
    }

    #[test]
    fn gemma4_style_format_is_detected() {
        let t = tokenizer(&[
            "<eos>",
            "<turn|>",
            "<|tool_response>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "x",
        ]);
        let f = ChatFormat::detect(
            &metadata(0),
            &t,
            "<|channel>thought\n <|tool_call>call:f{}<tool_call|>",
        )
        .expect("detects");
        assert!(f.is_eog(0) && f.is_eog(1) && f.is_eog(2));
        assert!(!f.is_eog(3) && !f.is_eog(7));
        assert_eq!(
            f.reasoning(),
            Some(&Reasoning {
                open: "<|channel>".into(),
                close: "<channel|>".into(),
                header: Some("thought".into()),
            })
        );
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::Gemma4 {
                open: "<|tool_call>".into(),
                close: "<tool_call|>".into(),
            }
        );
        assert!([3, 4, 5, 6].iter().all(|&id| f.is_marker(id)));
        assert!(!f.is_marker(1));
    }

    #[test]
    fn starts_in_thinking_requires_an_unclosed_open_marker() {
        let t = tokenizer(&["<|channel>", "<channel|>", "<eos>"]);
        let f = ChatFormat::detect(&metadata(2), &t, "<|channel>").expect("detects");
        assert!(f.starts_in_thinking("<bos>user\n<|channel>thought\n"));
        assert!(f.starts_in_thinking("a<|channel>"));
        assert!(!f.starts_in_thinking("a<|channel>thought\nhm<channel|>\n"));
        assert!(!f.starts_in_thinking("a<|channel>thought\nstill typing"));
        assert!(!f.starts_in_thinking("no markers"));
        assert!(f.starts_in_thinking("<|channel>thought\nx<channel|>y<|channel>thought\n"));

        let think = ChatFormat::detect(&metadata(0), &tokenizer(&["<eos>"]), "... <think> ...")
            .expect("detects");
        assert_eq!(
            think.reasoning(),
            Some(&Reasoning {
                open: "<think>".into(),
                close: "</think>".into(),
                header: None,
            })
        );
        assert!(think.starts_in_thinking("<|Assistant|><think>\n"));
        assert!(!think.starts_in_thinking("<think>\n\n</think>\n\n"));
        assert!(!think.starts_in_thinking("<think>\nreasoned so far"));
    }

    #[test]
    fn tool_syntax_follows_the_template() {
        let t = tokenizer(&["<eos>", "<|tool_call|>"]);
        let detect = |source: &str| {
            ChatFormat::detect(&metadata(0), &t, source)
                .expect("detects")
                .tool_syntax()
                .clone()
        };
        assert_eq!(
            detect("a <tool_call>\n{...}\n</tool_call>"),
            ToolSyntax::TaggedJson {
                open: "<tool_call>".into(),
                alt_open: None,
                close: Some("</tool_call>".into()),
            }
        );
        assert_eq!(
            detect(include_str!("../testdata/templates/qwen35.jinja")),
            ToolSyntax::TaggedXml {
                open: "<tool_call>".into(),
                close: "</tool_call>".into(),
            }
        );
        assert_eq!(
            detect("respond with <|tool_call|> then a JSON list"),
            ToolSyntax::TaggedJson {
                open: "<|tool_call|>".into(),
                alt_open: Some("<tool_call>".into()),
                close: None,
            }
        );
        assert_eq!(detect("{{ messages }}"), ToolSyntax::Plain);
    }

    #[test]
    fn eog_set_combines_metadata_and_known_terminators() {
        let t = tokenizer(&["<s>", "<|im_end|>", "<|eot_id|>", "other", "<eos>"]);
        let mut md = metadata(4);
        md.insert(
            "tokenizer.ggml.eom_token_id".to_string(),
            gguf::Value::I32(3),
        );
        md.insert(
            "tokenizer.ggml.eot_token_id".to_string(),
            gguf::Value::U32(99),
        );
        let f = ChatFormat::detect(&md, &t, "").expect("detects");
        assert!(f.is_eog(1) && f.is_eog(2) && f.is_eog(3) && f.is_eog(4));
        assert!(!f.is_eog(0) && !f.is_eog(99));
    }

    #[test]
    fn missing_end_of_generation_is_an_error() {
        let t = tokenizer(&["a", "b"]);
        let err = ChatFormat::detect(&HashMap::new(), &t, "").expect_err("no EOG");
        assert!(err.to_string().contains("end-of-generation"), "{err}");
    }

    #[test]
    fn deepseek_v41_is_detected_from_its_template_and_vocabulary() {
        let template = include_str!("templates/deepseek-v41.jinja");
        let t = tokenizer(&[
            "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
            "<think>",
            "</think>",
            "\u{ff5c}DSML\u{ff5c}",
            "x",
        ]);
        let f = ChatFormat::detect(&metadata(0), &t, template).expect("detects");
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::Dsml {
                dsml: "\u{ff5c}DSML\u{ff5c}".into(),
                calls: " calls".into(),
            }
        );
        assert_eq!(
            f.reasoning(),
            Some(&Reasoning {
                open: "<think>".into(),
                close: "</think>".into(),
                header: None
            })
        );
        assert!(f.is_eog(0) && !f.is_eog(1) && !f.is_eog(4));
        // The three single-token markers the parser reads raw, even when the
        // tokenizer renders them as control tokens.
        assert!([1, 2, 3].iter().all(|&id| f.is_marker(id)) && !f.is_marker(4));
        assert!(f.writes_reasoning());
        // Thinking mode ends the prompt in `<think>`, chat mode in `</think>`,
        // also when the tool prompt has quoted both tags.
        let tools = "## Tools ... inside <think>...</think> BEFORE any tool calls";
        assert!(f.starts_in_thinking("<\u{ff5c}Assistant\u{ff5c}><think>"));
        assert!(f.starts_in_thinking(&format!("{tools}<\u{ff5c}Assistant\u{ff5c}><think>")));
        assert!(!f.starts_in_thinking("<\u{ff5c}Assistant\u{ff5c}></think>"));
        assert!(!f.starts_in_thinking(&format!("{tools}<\u{ff5c}Assistant\u{ff5c}></think>")));

        // V4's spelling of the block follows the template that teaches it.
        let v4 = "<\u{ff5c}DSML\u{ff5c}tool_calls>";
        let f = ChatFormat::detect(&metadata(0), &t, v4).expect("detects");
        assert_eq!(
            f.tool_syntax(),
            &ToolSyntax::Dsml {
                dsml: "\u{ff5c}DSML\u{ff5c}".into(),
                calls: "tool_calls".into(),
            }
        );
        // Without the bar token in the vocabulary, or without a block spelling
        // in the template, the output stays plain.
        let bare = tokenizer(&["<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", "x"]);
        let f = ChatFormat::detect(&metadata(0), &bare, template).expect("detects");
        assert_eq!(f.tool_syntax(), &ToolSyntax::Plain);
        let f = ChatFormat::detect(&metadata(0), &t, "\u{ff5c}DSML\u{ff5c} only").expect("detects");
        assert_eq!(f.tool_syntax(), &ToolSyntax::Plain);
    }
}
