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
//!   `<channel|>`) or `<think>` ... `</think>` tags, whichever the template
//!   writes, with [`ChatFormat::starts_in_thinking`] deciding from a rendered
//!   prompt whether generation begins inside an open block;
//! - the **tool-call syntax** the template teaches the model: a custom
//!   `call:name{key:<|"|>value<|"|>}` body between control tokens, tagged JSON
//!   (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`), or bare JSON
//!   `{"name": ..., "parameters": ...}` optionally behind `<|python_tag|>`; a
//!   template without tool syntax leaves the output plain text.

use std::collections::HashMap;

use candle_core::quantized::gguf_file;

use super::local_err;
use crate::error::CucaError;
use crate::provider::local::tokenizer::Tokenizer;

/// Vocabulary texts that end a turn or a sequence (llama.cpp's EOT/EOM list
/// plus the Gemma 4 turn end and tool-response stop).
const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
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

/// How the model writes tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolSyntax {
    /// No tool-call syntax: every token is answer text.
    Plain,
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
    /// Bare JSON `{"name": ..., "parameters": ...}` as the whole reply, plus an
    /// optional `<|python_tag|>` prefix for code or JSON calls running to the
    /// end of generation.
    LlamaJson {
        /// Text of the python tag token, when the vocabulary has one.
        python_tag: Option<String>,
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
fn md_id(md: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    match md.get(key)? {
        gguf_file::Value::U32(n) => Some(*n),
        gguf_file::Value::I32(n) => u32::try_from(*n).ok(),
        gguf_file::Value::U64(n) => u32::try_from(*n).ok(),
        gguf_file::Value::I64(n) => u32::try_from(*n).ok(),
        gguf_file::Value::U16(n) => Some(u32::from(*n)),
        gguf_file::Value::U8(n) => Some(u32::from(*n)),
        _ => None,
    }
}

impl ChatFormat {
    /// Detect the format from the model's metadata, tokenizer and chat template
    /// source.
    ///
    /// Fails when no end-of-generation token can be determined.
    pub(crate) fn detect(
        md: &HashMap<String, gguf_file::Value>,
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
            ToolSyntax::Gemma4 { open, close } | ToolSyntax::TaggedXml { open, close } => {
                note(open);
                note(close);
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
            ToolSyntax::LlamaJson { python_tag } => {
                if let Some(tag) = python_tag {
                    note(tag);
                }
            }
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
    if template.contains("tool_calls") && template.contains("\"parameters\"") {
        return ToolSyntax::LlamaJson {
            python_tag: tokenizer
                .token_id("<|python_tag|>")
                .map(|_| "<|python_tag|>".to_string()),
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

    fn metadata(eos: u32) -> HashMap<String, gguf_file::Value> {
        HashMap::from([(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(eos),
        )])
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
        let t = tokenizer(&["<eos>", "<|python_tag|>", "<|tool_call|>"]);
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
        assert_eq!(
            detect("{% if 'tool_calls' in message %}{\"name\": x, \"parameters\": y}"),
            ToolSyntax::LlamaJson {
                python_tag: Some("<|python_tag|>".into()),
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
            gguf_file::Value::I32(3),
        );
        md.insert(
            "tokenizer.ggml.eot_token_id".to_string(),
            gguf_file::Value::U32(99),
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
}
