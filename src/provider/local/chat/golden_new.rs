//! Golden chat-template renders for the families `golden.rs` does not cover:
//! Muse Glimmer (the repository template, `meta-models/Muse-Glimmer-30B` at
//! cec94fcf), Mistral Small 4, Mistral Small 3.2 and Ministral 3 Reasoning
//! (the `chat_template.jinja` of `mistralai/Ministral-3-3B-Reasoning-2512` at
//! 4a36357c, the template the preset pins) and Ministral 3 Instruct
//! (`mistralai/Ministral-3-3B-Instruct-2512` at b35d4dfe, likewise).
//!
//! `testdata/chat_golden_new.json` maps an entry name to its `template`
//! (`testdata/templates/<name>.jinja`, the model repository's own
//! `chat_template.jinja`), the `bos`/`eos` texts, the `cases` (OpenAI-shaped
//! `messages`, `tools`, and `thinking`: `unset`, `off`, `on` or `on:<effort>`)
//! and the `renders` the HuggingFace environment produces for them
//! (`tokenizer.apply_chat_template(..., add_generation_prompt=True,
//! tokenize=False)` of transformers 5.20, `strftime_now` pinned to
//! 2026-10-18 UTC, template variables derived from `thinking` exactly like
//! [`ThinkingRequest`] derives them). A case whose template raises is
//! recorded as `{"error": ...}` and must raise here too.

use serde_json::Value;

use super::{ChatPrompt, ChatTemplate, ThinkingRequest};
use crate::request::ThinkingEffort;

/// 2026-10-18 00:00:00 UTC, the date the references pin `strftime_now` to.
const PINNED_NOW: i64 = 1_792_281_600;

fn template_source(name: &str) -> &'static str {
    match name {
        "muse-glimmer" => include_str!("../testdata/templates/muse-glimmer.jinja"),
        "mistral-small-4" => include_str!("../testdata/templates/mistral-small-4.jinja"),
        "mistral-small-3.2" => include_str!("../testdata/templates/mistral-small-3.2.jinja"),
        "ministral3-reasoning" => include_str!("../testdata/templates/ministral3-reasoning.jinja"),
        "ministral3-instruct" => include_str!("../testdata/templates/ministral3-instruct.jinja"),
        other => panic!("no template fixture for {other}"),
    }
}

/// `thinking` of a case as the request would carry it.
fn thinking(spec: &str) -> ThinkingRequest {
    match spec {
        "unset" => ThinkingRequest::Unset,
        "off" => ThinkingRequest::Off,
        "on" => ThinkingRequest::On(None),
        other => {
            let effort = match other.strip_prefix("on:").expect("on:<effort>") {
                "minimal" => ThinkingEffort::Minimal,
                "low" => ThinkingEffort::Low,
                "medium" => ThinkingEffort::Medium,
                "high" => ThinkingEffort::High,
                "xhigh" => ThinkingEffort::XHigh,
                e => panic!("unknown effort {e}"),
            };
            ThinkingRequest::On(Some(effort))
        }
    }
}

#[test]
fn renders_match_the_reference_jinja_environment() {
    let golden: Value =
        serde_json::from_str(include_str!("../testdata/chat_golden_new.json")).expect("parses");
    let mut checked = 0;
    let mut failures = Vec::new();
    for (model, entry) in golden.as_object().expect("object") {
        let template = ChatTemplate::new(
            template_source(entry["template"].as_str().expect("template")).to_string(),
            entry["bos"].as_str().expect("bos").to_string(),
            entry["eos"].as_str().expect("eos").to_string(),
        )
        .unwrap_or_else(|e| panic!("{model}: {e}"));
        for (case, expected) in entry["renders"].as_object().expect("renders") {
            let input = &entry["cases"][case];
            let mode = thinking(input["thinking"].as_str().unwrap_or("unset"));
            let prompt = ChatPrompt {
                messages: input["messages"].clone(),
                tools: input
                    .get("tools")
                    .cloned()
                    .unwrap_or(Value::Array(Vec::new())),
                enable_thinking: matches!(mode, ThinkingRequest::On(_)),
                thinking: mode,
            };
            match (expected, template.render_at(&prompt, PINNED_NOW)) {
                (Value::String(want), Ok(got)) if *want == got => {}
                (Value::String(want), Ok(got)) => {
                    let at = want
                        .bytes()
                        .zip(got.bytes())
                        .position(|(a, b)| a != b)
                        .unwrap_or(want.len().min(got.len()));
                    let around = |s: &str| {
                        let lo = (0..=at.saturating_sub(60))
                            .rev()
                            .find(|&i| s.is_char_boundary(i))
                            .unwrap_or(0);
                        let hi = (at + 120).min(s.len());
                        let hi = (hi..=s.len())
                            .find(|&i| s.is_char_boundary(i))
                            .unwrap_or(s.len());
                        s[lo..hi].to_string()
                    };
                    failures.push(format!(
                        "{model}/{case}: differ at byte {at}\n  want {:?}\n  got  {:?}",
                        around(want),
                        around(&got)
                    ));
                }
                (Value::String(want), Err(e)) => {
                    failures.push(format!("{model}/{case}: want {want:?}, error {e}"));
                }
                (Value::Object(_), Err(_)) => {}
                (want, Ok(got)) => {
                    failures.push(format!("{model}/{case}: want an error {want}, got {got:?}"));
                }
                (_, Err(e)) => failures.push(format!("{model}/{case}: unexpected error {e}")),
            }
            checked += 1;
        }
    }
    assert!(checked > 0);
    assert!(
        failures.is_empty(),
        "{} of {checked} renders differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
