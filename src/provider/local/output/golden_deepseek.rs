//! Golden parses of DeepSeek V4.1 completions.
//!
//! `testdata/deepseek_v41_parse_cases.json` holds 20 completions, each
//! `{name, thinking, text, expect | error}`: the text the model writes after
//! the prompt (a thinking completion starts inside the reasoning, because the
//! prompt ends in `<think>`; the final `<｜end▁of▁sentence｜>` is in it) and
//! what the reference `parse_message_from_completion_text` of
//! `encoding/encoding.py` returned for it (`content`, `reasoning_content`,
//! `tool_calls` with their arguments decoded), or the error it raised.
//!
//! The streaming parser must produce the same blocks however the text is cut
//! into tokens: one character per token, groups of characters, with the
//! special tokens `<think>`, `</think>` and `｜DSML｜` whole (as the real
//! vocabulary has them) or spelled out character by character. The reference
//! raises on five completions that a streaming parser has to take as they come
//! (a reply cut before `</think>` or before the end token, the V4 spelling of
//! the block, text after the block) or that are malformed (a repeated
//! parameter); the tests below say what each becomes.

use std::collections::HashSet;

use serde_json::Value;

use super::*;
use crate::provider::local::chat::expand_template_name;
use crate::provider::local::gguf;

/// The end token, which the engine never feeds to the parser.
const EOS: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
/// The vocabulary entries that decode as text and that the model writes whole.
const SPECIALS: [&str; 3] = ["<think>", "</think>", "\u{ff5c}DSML\u{ff5c}"];

/// The reference's parse of one completion.
struct Case {
    name: String,
    thinking: bool,
    text: String,
    expect: Option<Value>,
}

fn cases() -> Vec<Case> {
    let cases: Value =
        serde_json::from_str(include_str!("../testdata/deepseek_v41_parse_cases.json"))
            .expect("parses");
    cases
        .as_array()
        .expect("array")
        .iter()
        .map(|case| Case {
            name: case["name"].as_str().expect("name").to_string(),
            thinking: case["thinking"].as_bool().expect("flag"),
            text: case["text"]
                .as_str()
                .expect("text")
                .strip_suffix(EOS)
                .unwrap_or_else(|| case["text"].as_str().expect("text"))
                .to_string(),
            expect: case.get("expect").cloned(),
        })
        .collect()
}

/// Cut `text` into tokens of `chunk` characters; with `whole_specials` the three
/// special spellings are single tokens and never part of another one.
fn pieces(text: &str, chunk: usize, whole_specials: bool) -> Vec<String> {
    let starts_special =
        |rest: &str| whole_specials && SPECIALS.iter().any(|special| rest.starts_with(special));
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(special) = SPECIALS
            .iter()
            .find(|special| whole_specials && rest.starts_with(**special))
        {
            out.push((*special).to_string());
            rest = &rest[special.len()..];
            continue;
        }
        let mut end = 0;
        for (n, (at, c)) in rest.char_indices().enumerate() {
            if n == chunk || (n > 0 && starts_special(&rest[at..])) {
                break;
            }
            end = at + c.len_utf8();
        }
        out.push(rest[..end].to_string());
        rest = &rest[end..];
    }
    out
}

/// A parser over a vocabulary of the end token (id 0), the three specials
/// (decoding as text, GGUF type 4) and `tokens`.
struct Rig {
    tokenizer: Tokenizer,
    format: ChatFormat,
}

impl Rig {
    fn new(template: &str, tokens: &[String]) -> Self {
        let mut vocab: Vec<(String, i32)> = vec![(EOS.to_string(), 3)];
        vocab.extend(SPECIALS.iter().map(|s| ((*s).to_string(), 4)));
        let mut seen: HashSet<String> = vocab.iter().map(|(text, _)| text.clone()).collect();
        for token in tokens {
            if seen.insert(token.clone()) {
                vocab.push((token.clone(), 1));
            }
        }
        let tokenizer = Tokenizer::new(
            vocab.iter().map(|(text, _)| text.clone()).collect(),
            vocab.iter().map(|(_, ty)| *ty).collect(),
            &[],
        )
        .expect("valid vocabulary");
        let md = std::collections::HashMap::from([(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf::Value::U32(0),
        )]);
        let format = ChatFormat::detect(&md, &tokenizer, template).expect("format detects");
        Self { tokenizer, format }
    }

    /// The blocks the parser emits for `tokens`, adjacent text (and adjacent
    /// reasoning) joined as a client sees it.
    fn parse(
        &self,
        tokens: &[String],
        start_in_thinking: bool,
    ) -> Vec<Result<MessageContentBlock, CucaError>> {
        let mut parser = OutputParser::new(&self.format, start_in_thinking);
        let mut out = Vec::new();
        let mut counter = 0;
        let mut next = || {
            counter += 1;
            format!("call_{counter}")
        };
        for token in tokens {
            let id = self
                .tokenizer
                .token_id(token)
                .unwrap_or_else(|| panic!("`{token}` is not in the vocabulary"));
            parser.push(id, &self.tokenizer, &mut next, &mut out);
        }
        parser.finish(&mut next, &mut out);
        merge(out)
    }
}

fn merge(
    blocks: Vec<Result<MessageContentBlock, CucaError>>,
) -> Vec<Result<MessageContentBlock, CucaError>> {
    let mut out: Vec<Result<MessageContentBlock, CucaError>> = Vec::new();
    for block in blocks {
        match (out.last_mut(), block) {
            (Some(Ok(MessageContentBlock::Text(a))), Ok(MessageContentBlock::Text(b))) => {
                a.push_str(&b);
            }
            (
                Some(Ok(MessageContentBlock::Thinking { reasoning: a, .. })),
                Ok(MessageContentBlock::Thinking { reasoning: b, .. }),
            ) => a.push_str(&b),
            (_, block) => out.push(block),
        }
    }
    out
}

fn template() -> String {
    expand_template_name("deepseek-v41")
        .expect("builtin name")
        .into_owned()
}

/// Every way of cutting a completion into tokens that the tests compare.
fn cuts() -> Vec<(usize, bool)> {
    let mut cuts = Vec::new();
    for whole_specials in [true, false] {
        for chunk in [1, 2, 3, 7, usize::MAX] {
            cuts.push((chunk, whole_specials));
        }
    }
    cuts
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

fn call(n: usize, name: &str, arguments: Value) -> MessageContentBlock {
    MessageContentBlock::ToolCall {
        id: format!("call_{n}"),
        name: name.to_string(),
        arguments,
    }
}

/// The blocks that the reference's message stands for: reasoning, answer, then
/// the calls, a namespaced name written `namespace::name` as the model wrote it.
fn expected_blocks(message: &Value) -> Vec<MessageContentBlock> {
    let mut blocks = Vec::new();
    let reasoning = message["reasoning_content"].as_str().expect("reasoning");
    if !reasoning.is_empty() {
        blocks.push(thinking(reasoning));
    }
    let content = message["content"].as_str().expect("content");
    if !content.is_empty() {
        blocks.push(text(content));
    }
    for (n, tool_call) in message["tool_calls"]
        .as_array()
        .expect("calls")
        .iter()
        .enumerate()
    {
        let name = tool_call["function"]["name"].as_str().expect("name");
        let name = match tool_call["namespace"].as_str() {
            Some(namespace) => format!("{namespace}::{name}"),
            None => name.to_string(),
        };
        blocks.push(call(
            n + 1,
            &name,
            tool_call["function"]["arguments"].clone(),
        ));
    }
    blocks
}

fn ok(out: Vec<Result<MessageContentBlock, CucaError>>) -> Vec<MessageContentBlock> {
    out.into_iter()
        .map(|block| block.expect("no parser error expected"))
        .collect()
}

#[test]
fn completions_parse_like_the_reference_parser() {
    let template = template();
    let cases = cases();
    assert_eq!(cases.len(), 20);
    let mut checked = 0;
    let mut failures = Vec::new();
    for case in cases.iter().filter(|case| case.expect.is_some()) {
        let want = expected_blocks(case.expect.as_ref().expect("filtered"));
        for (chunk, whole_specials) in cuts() {
            let tokens = pieces(&case.text, chunk, whole_specials);
            let got = Rig::new(&template, &tokens).parse(&tokens, case.thinking);
            let got: Result<Vec<_>, _> = got.into_iter().collect();
            if got.as_ref().ok() != Some(&want) {
                failures.push(format!(
                    "{} (chunk {chunk}, whole specials {whole_specials}): got {got:?}, want {want:?}",
                    case.name
                ));
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 15);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// What the five completions the reference rejects become in a streaming
/// parser, by case name.
#[test]
fn completions_the_reference_rejects_become_what_was_written() {
    let template = template();
    let cases = cases();
    let rejected: Vec<&Case> = cases.iter().filter(|case| case.expect.is_none()).collect();
    assert_eq!(rejected.len(), 5);
    for case in rejected {
        for (chunk, whole_specials) in cuts() {
            let tokens = pieces(&case.text, chunk, whole_specials);
            let got = Rig::new(&template, &tokens).parse(&tokens, case.thinking);
            let name = case.name.as_str();
            match name {
                // A reply cut before `</think>` is all reasoning; one cut before
                // the end token is all answer.
                "think missing close" => assert_eq!(ok(got), [thinking("no end of thinking")]),
                "chat missing eos" => assert_eq!(ok(got), [text("truncated")]),
                // V4's `tool_calls` spelling is not this template's block: plain
                // text, so nothing is executed.
                "old v4 tool_calls tag" => assert_eq!(ok(got), [text(&case.text)]),
                // A repeated parameter name is one malformed call.
                "duplicate parameter" => {
                    assert_eq!(got.len(), 1, "{got:?}");
                    let error = got[0].as_ref().expect_err("malformed").to_string();
                    assert!(error.contains("malformed tool call"), "{error}");
                }
                // The text behind the block is still the model's output.
                "text after calls" => {
                    assert_eq!(ok(got), [call(1, "f", serde_json::json!({})), text("tail")]);
                }
                other => panic!("unexpected rejected case `{other}`"),
            }
        }
    }
}

#[test]
fn the_opener_inside_reasoning_starts_the_call_as_in_deepseeks_recipe() {
    // The reference parser raises on a call before `</think>`; DeepSeek's own
    // stream parser takes it as the end of the reasoning.
    let completion = "plan\n\n<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"f\">\n\
                      <\u{ff5c}DSML\u{ff5c} parameter name=\"x\" string=\"false\">1</\u{ff5c}DSML\u{ff5c} parameter>\n\
                      </\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>";
    let template = template();
    for (chunk, whole_specials) in cuts() {
        let tokens = pieces(completion, chunk, whole_specials);
        let got = Rig::new(&template, &tokens).parse(&tokens, true);
        assert_eq!(
            ok(got),
            [thinking("plan"), call(1, "f", serde_json::json!({"x": 1}))]
        );
    }
}

#[test]
fn a_bare_opener_and_a_block_cut_short_still_parse() {
    let template = template();
    // No blank line before the opener, and generation ends inside the block
    // (a length stop): the body parses as far as it goes.
    let cut_short = "ok<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"f\">\n\
                     <\u{ff5c}DSML\u{ff5c} parameter name=\"a\" string=\"true\">v</\u{ff5c}DSML\u{ff5c} parameter>\n\
                     </\u{ff5c}DSML\u{ff5c} invoke>";
    let tokens = pieces(cut_short, 1, true);
    let got = Rig::new(&template, &tokens).parse(&tokens, false);
    assert_eq!(
        ok(got),
        [text("ok"), call(1, "f", serde_json::json!({"a": "v"}))]
    );
}

#[test]
fn an_answer_that_mentions_the_markup_is_text_until_the_exact_opener() {
    let template = template();
    let answer = "Write <\u{ff5c}DSML\u{ff5c} invoke> inside a block, or <b>bold</b> and </think>.";
    for (chunk, whole_specials) in cuts() {
        let tokens = pieces(answer, chunk, whole_specials);
        let got = Rig::new(&template, &tokens).parse(&tokens, false);
        // The stray `</think>` is dropped; everything else is the answer.
        assert_eq!(
            ok(got),
            [text(
                "Write <\u{ff5c}DSML\u{ff5c} invoke> inside a block, or <b>bold</b> and ."
            )]
        );
    }
}
