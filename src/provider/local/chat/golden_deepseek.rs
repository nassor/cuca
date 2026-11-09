//! Golden renders of the `deepseek-v41` template against DeepSeek's reference
//! `encoding/encoding.py` of DeepSeek-V4.1-Flash-1017.
//!
//! `testdata/deepseek_v41_chat_cases.json` holds 400 cases, each
//! `{name, messages, thinking, effort, tools, expect}`: OpenAI-shaped
//! `messages` (maps in the order the reference saw them), the global `tools`
//! list (or `null`), the thinking mode, the reasoning effort the reference was
//! given (`null`, `"low"`, `"high"`, `"max"` or an integer 1 to 100) and the
//! prompt `encode_messages` returned. They are the five official test pairs of
//! the `encoding` directory (the image case with its image blocks replaced by
//! text), about 80 hand-written conversations (every message kind, effort,
//! tool loop, merge, mid-conversation system message, per-message tools,
//! `response_format`, `latest_reminder` placement, task and `wo_eos`) and
//! seeded random conversations; `check_template.py` renders the same cases
//! under jinja2 (3.1.6) and finds the same text.
//!
//! The template must render each of them byte for byte, which also needs the
//! host to keep the key order of the message maps (the order of a DSML
//! call's parameters) and of the tool schemas.

use minijinja::Value;
use serde_json::{Value as Json, json};

use super::builtin::expand;
use super::{ChatPrompt, ChatTemplate, ThinkingRequest};
use crate::request::ThinkingEffort;

const BOS: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
const EOS: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";

fn template() -> ChatTemplate {
    let source = expand("deepseek-v41").expect("builtin name");
    ChatTemplate::new(source.into_owned(), BOS.into(), EOS.into()).expect("compiles")
}

fn prompt(messages: Json, tools: Json, thinking: ThinkingRequest) -> ChatPrompt {
    ChatPrompt {
        messages,
        tools,
        enable_thinking: matches!(thinking, ThinkingRequest::On(_)),
        thinking,
    }
}

#[test]
fn renders_match_the_reference_encoder() {
    let cases: Json =
        serde_json::from_str(include_str!("../testdata/deepseek_v41_chat_cases.json"))
            .expect("parses");
    let cases = cases.as_array().expect("array");
    assert_eq!(cases.len(), 400);
    let template = template();
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let thinking = if case["thinking"].as_bool().expect("flag") {
            ThinkingRequest::On(None)
        } else {
            ThinkingRequest::Off
        };
        let tools = case["tools"].clone();
        let prompt = prompt(
            case["messages"].clone(),
            if tools.is_null() { json!([]) } else { tools },
            thinking,
        );
        let rendered = match &case["effort"] {
            Json::String(effort) => {
                template.render_with_budget(&prompt, 0, Value::from(effort.as_str()))
            }
            Json::Number(effort) => {
                template.render_with_budget(&prompt, 0, Value::from(effort.as_i64().expect("int")))
            }
            _ => template.render_at(&prompt, 0),
        };
        let want = case["expect"].as_str().expect("expected prompt");
        match rendered {
            Ok(got) if got == want => {}
            Ok(got) => {
                let at = got
                    .chars()
                    .zip(want.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                let around = |s: &str| {
                    s.chars()
                        .skip(at.saturating_sub(40))
                        .take(100)
                        .collect::<String>()
                };
                failures.push(format!(
                    "{name}: differs at char {at}\n  want ..{:?}\n  got  ..{:?}",
                    around(want),
                    around(&got)
                ));
            }
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of 400 differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_request_selects_thinking_and_effort_as_deepseeks_stack_does() {
    let template = template();
    let messages = json!([{"role": "user", "content": "hi"}]);
    let render = |thinking| {
        template
            .render_at(&prompt(messages.clone(), json!([]), thinking), 0)
            .expect("renders")
    };
    let effort = |n: u32| {
        format!(
            "{BOS}<\u{ff5c}System\u{ff5c}>Reasoning Effort: {n} (range 1-100, the higher the value, the more thorough the reasoning)\n\n<\u{ff5c}User\u{ff5c}>hi<\u{ff5c}Assistant\u{ff5c}><think>"
        )
    };
    // No thinking configuration: thinking, at the default effort `high`.
    assert_eq!(render(ThinkingRequest::Unset), effort(75));
    assert_eq!(render(ThinkingRequest::On(None)), effort(75));
    // Minimal and low are `low`; medium, high and xhigh are `high`, as in
    // DeepSeek's recipe (`max`, 100, has no unified effort).
    for (request, n) in [
        (ThinkingEffort::Minimal, 50),
        (ThinkingEffort::Low, 50),
        (ThinkingEffort::Medium, 75),
        (ThinkingEffort::High, 75),
        (ThinkingEffort::XHigh, 75),
    ] {
        assert_eq!(
            render(ThinkingRequest::On(Some(request))),
            effort(n),
            "{request:?}"
        );
    }
    // Disabled: chat mode, no effort line and no system token.
    assert_eq!(
        render(ThinkingRequest::Off),
        format!("{BOS}<\u{ff5c}User\u{ff5c}>hi<\u{ff5c}Assistant\u{ff5c}></think>")
    );
}

#[test]
fn only_a_template_that_reads_the_budget_thinks_by_default() {
    let deepseek = template();
    let qwen = ChatTemplate::new("{{ enable_thinking }}".into(), String::new(), String::new())
        .expect("compiles");
    let unset = prompt(json!([]), json!([]), ThinkingRequest::Unset);
    let off = prompt(json!([]), json!([]), ThinkingRequest::Off);
    let on = prompt(json!([]), json!([]), ThinkingRequest::On(None));
    assert!(deepseek.thinking_enabled(&unset) && deepseek.thinking_enabled(&on));
    assert!(!deepseek.thinking_enabled(&off));
    assert!(!qwen.thinking_enabled(&unset) && qwen.thinking_enabled(&on));
    assert!(deepseek.reads_enable_thinking());
}

#[test]
fn a_tool_call_keeps_the_order_of_its_arguments() {
    let template = template();
    let messages: Json = serde_json::from_str(
        r#"[{"role":"user","content":"q"},{"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":{"zeta":"z","alpha":1,"mid":[2,1]}}}]}]"#,
    )
    .expect("parses");
    let rendered = template
        .render_at(&prompt(messages, json!([]), ThinkingRequest::Off), 0)
        .expect("renders");
    let zeta = rendered.find("name=\"zeta\"").expect("zeta");
    let alpha = rendered.find("name=\"alpha\"").expect("alpha");
    let mid = rendered.find("name=\"mid\"").expect("mid");
    assert!(zeta < alpha && alpha < mid, "{rendered}");
    assert!(rendered.contains("string=\"false\">[2, 1]</"), "{rendered}");
}

#[test]
fn invalid_roles_and_efforts_are_errors() {
    let template = template();
    let developer = prompt(
        json!([{"role": "developer", "content": "x"}]),
        json!([]),
        ThinkingRequest::On(None),
    );
    let e = template
        .render_at(&developer, 0)
        .expect_err("developer role");
    assert!(e.to_string().contains("unknown role developer"), "{e}");
    let user = prompt(
        json!([{"role": "user", "content": "x"}]),
        json!([]),
        ThinkingRequest::On(None),
    );
    for bad in [Value::from(0), Value::from(101), Value::from("medium")] {
        let e = template
            .render_with_budget(&user, 0, bad)
            .expect_err("effort out of range");
        assert!(e.to_string().contains("Invalid reasoning effort"), "{e}");
    }
    let image = prompt(
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
        json!([]),
        ThinkingRequest::Off,
    );
    let e = template.render_at(&image, 0).expect_err("image part");
    assert!(e.to_string().contains("only text content"), "{e}");
}

#[test]
fn the_image_placeholder_cannot_be_written_in_text() {
    let template = template();
    let placeholder = "see <\u{ff5c}deepseek_image\u{ff5c}> here";
    for messages in [
        json!([{"role": "user", "content": placeholder}]),
        json!([{"role": "user", "content": [{"type": "text", "text": placeholder}]}]),
        json!([{"role": "system", "content": placeholder}, {"role": "user", "content": "x"}]),
        json!([{"role": "user", "content": "x"}, {"role": "assistant", "content": placeholder}]),
        json!([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y", "reasoning_content": placeholder}]),
    ] {
        let e = template
            .render_at(&prompt(messages, json!([]), ThinkingRequest::Off), 0)
            .expect_err("placeholder");
        assert!(e.to_string().contains("image placeholder"), "{e}");
    }
}

/// The format derived from the template tells, for every reference prompt, whether
/// generation starts inside the reasoning: exactly when the prompt ends in an
/// open `<think>`.
#[test]
fn the_format_reads_every_reference_prompt() {
    use std::collections::HashMap;

    use super::ChatFormat;
    use super::format::tests::tokenizer;
    use crate::provider::local::gguf;

    let tokenizer = tokenizer(&[EOS, "<think>", "</think>", "\u{ff5c}DSML\u{ff5c}", "x"]);
    let md = HashMap::from([(
        "tokenizer.ggml.eos_token_id".to_string(),
        gguf::Value::U32(0),
    )]);
    let format =
        ChatFormat::detect(&md, &tokenizer, &expand("deepseek-v41").unwrap()).expect("detects");
    let cases: Json =
        serde_json::from_str(include_str!("../testdata/deepseek_v41_chat_cases.json"))
            .expect("parses");
    let mut thinking_prompts = 0;
    for case in cases.as_array().expect("array") {
        let expect = case["expect"].as_str().expect("prompt");
        let open = expect.ends_with("<think>");
        thinking_prompts += usize::from(open);
        assert_eq!(
            format.starts_in_thinking(expect),
            open,
            "{}: {expect:?}",
            case["name"]
        );
    }
    assert!(thinking_prompts > 100, "{thinking_prompts}");
}
