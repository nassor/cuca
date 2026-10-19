//! Golden chat-template renders for the catalog models.
//!
//! `testdata/templates/<model>.jinja` are the chat templates embedded in the
//! catalog GGUFs (gemma4's is `testdata/gemma4_chat_template.jinja`);
//! `testdata/chat_cases.json` holds the `messages`/`tools`/`enable_thinking`
//! inputs (keys sorted, as `chat_prompt` produces them); `testdata/
//! chat_golden.json` holds, per model, the text HuggingFace's template
//! environment renders for each case, generated once with Python `jinja2`
//! (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`, a
//! `json.dumps(ensure_ascii=False)` `tojson`, `strftime_now` pinned to
//! 2026-10-18 UTC, `add_generation_prompt=True`). A case whose template raises
//! is recorded as `{"error": ...}` and must raise here too.

use serde_json::Value;

use super::{ChatPrompt, ChatTemplate};

/// 2026-10-18 00:00:00 UTC, the date the references pin `strftime_now` to.
const PINNED_NOW: i64 = 1_792_281_600;

/// Template source by golden entry name.
fn template_source(name: &str) -> &'static str {
    match name {
        "gemma3-1b" => include_str!("../testdata/templates/gemma3-1b.jinja"),
        "gemma4" => include_str!("../testdata/gemma4_chat_template.jinja"),
        "granitemoe" => include_str!("../testdata/templates/granitemoe.jinja"),
        "gemma2" => include_str!("../testdata/templates/gemma2.jinja"),
        "olmo2" => include_str!("../testdata/templates/olmo2.jinja"),
        "granite33" => include_str!("../testdata/templates/granite33.jinja"),
        "ernie45" => include_str!("../testdata/templates/ernie45.jinja"),
        "smollm2" => include_str!("../testdata/templates/smollm2.jinja"),
        "tinyllama" => include_str!("../testdata/templates/tinyllama.jinja"),
        "llama32" => include_str!("../testdata/templates/llama32.jinja"),
        "phi35" => include_str!("../testdata/templates/phi35.jinja"),
        "phi4mini" => include_str!("../testdata/templates/phi4mini.jinja"),
        "qwen25" => include_str!("../testdata/templates/qwen25.jinja"),
        "qwen3" => include_str!("../testdata/templates/qwen3.jinja"),
        "qwen35" => include_str!("../testdata/templates/qwen35.jinja"),
        "r1qwen" => include_str!("../testdata/templates/r1qwen.jinja"),
        "smollm3" => include_str!("../testdata/templates/smollm3.jinja"),
        other => panic!("no template fixture for {other}"),
    }
}

/// The rendered prompt for each model and case matches the references.
#[test]
fn renders_match_the_reference_jinja_environment() {
    let cases: Value =
        serde_json::from_str(include_str!("../testdata/chat_cases.json")).expect("cases parse");
    let golden: Value =
        serde_json::from_str(include_str!("../testdata/chat_golden.json")).expect("golden parses");
    let mut checked = 0;
    let mut failures = Vec::new();
    for (model, entry) in golden.as_object().expect("golden is an object") {
        let source = template_source(entry["template"].as_str().expect("template name"));
        let template = ChatTemplate::new(
            source.to_string(),
            entry["bos"].as_str().expect("bos").to_string(),
            entry["eos"].as_str().expect("eos").to_string(),
        )
        .unwrap_or_else(|e| panic!("{model}: {e}"));
        for (case, expected) in entry["renders"].as_object().expect("renders") {
            let input = &cases[case];
            let prompt = ChatPrompt {
                messages: input["messages"].clone(),
                tools: input
                    .get("tools")
                    .cloned()
                    .unwrap_or(Value::Array(Vec::new())),
                enable_thinking: input["enable_thinking"].as_bool().expect("flag"),
            };
            let rendered = template.render_at(&prompt, PINNED_NOW);
            match (expected, rendered) {
                (Value::String(want), Ok(got)) if *want == got => {}
                (Value::String(want), Ok(got)) => {
                    failures.push(format!("{model}/{case}:\n  want {want:?}\n  got  {got:?}"));
                }
                (Value::String(_), Err(e)) => failures.push(format!("{model}/{case}: {e}")),
                (_, Err(_)) => {}
                (_, Ok(_)) => failures.push(format!(
                    "{model}/{case}: the reference raised {expected}, this rendered"
                )),
            }
            checked += 1;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(checked, 13 * 7);
}

/// Catalog models: format detection and an end-to-end render + tokenize check
/// against the cached GGUFs (`CUCA_LOCAL_CATALOG_DIR`, see the tokenizer golden
/// docs; gemma4 through `CUCA_LOCAL_MODEL_PATH`). Absent files print `SKIP:`.
mod real_models {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::{PINNED_NOW, template_source};
    use crate::provider::local::chat::{ChatFormat, ChatPrompt, ChatTemplate, ToolSyntax};
    use crate::provider::local::tokenizer::Tokenizer;
    use crate::provider::local::tokenizer::golden::{catalog_dir, read_metadata, skip};

    /// What detection must find for one model.
    struct Expect {
        /// Catalog id (fixture key).
        id: &'static str,
        /// Template fixture name.
        template: &'static str,
        /// Reasoning open marker, if any.
        reasoning_open: Option<&'static str>,
        /// Tool syntax variant name: plain, gemma4, tagged, llama.
        tools: &'static str,
        /// Vocabulary texts that must end generation.
        eog: &'static [&'static str],
    }

    const EXPECTED: &[Expect] = &[
        Expect {
            id: "qwen3",
            template: "qwen3",
            reasoning_open: Some("<think>"),
            tools: "tagged",
            eog: &["<|im_end|>", "<|endoftext|>"],
        },
        Expect {
            id: "qwen25",
            template: "qwen25",
            reasoning_open: None,
            tools: "tagged",
            eog: &["<|im_end|>", "<|endoftext|>"],
        },
        Expect {
            id: "llama32",
            template: "llama32",
            reasoning_open: None,
            tools: "llama",
            eog: &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"],
        },
        Expect {
            id: "gemma3-270m",
            template: "gemma3-1b",
            reasoning_open: None,
            tools: "plain",
            eog: &["<end_of_turn>", "<eos>"],
        },
        Expect {
            id: "gemma3-1b",
            template: "gemma3-1b",
            reasoning_open: None,
            tools: "plain",
            eog: &["<end_of_turn>"],
        },
        Expect {
            id: "smollm3",
            template: "smollm3",
            reasoning_open: Some("<think>"),
            tools: "tagged",
            eog: &["<|im_end|>", "<|eot_id|>"],
        },
        Expect {
            id: "phi35",
            template: "phi35",
            reasoning_open: None,
            tools: "plain",
            eog: &["<|end|>", "<|endoftext|>"],
        },
        Expect {
            id: "phi4mini",
            template: "phi4mini",
            reasoning_open: None,
            tools: "plain",
            eog: &["<|end|>", "<|endoftext|>"],
        },
        Expect {
            id: "r1qwen",
            template: "r1qwen",
            reasoning_open: Some("<think>"),
            tools: "plain",
            eog: &["<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"],
        },
        Expect {
            id: "granitemoe",
            template: "granitemoe",
            reasoning_open: None,
            tools: "tagged",
            eog: &["<|end_of_text|>"],
        },
        Expect {
            id: "qwen35-08b-mtp",
            template: "qwen35",
            reasoning_open: Some("<think>"),
            tools: "xml",
            eog: &["<|im_end|>", "<|endoftext|>"],
        },
        Expect {
            id: "gemma4",
            template: "gemma4",
            reasoning_open: Some("<|channel>"),
            tools: "gemma4",
            eog: &["<eos>", "<turn|>", "<|tool_response>"],
        },
    ];

    /// Cached GGUF path of a catalog id, or `None` (after a `SKIP:`).
    fn model_path(id: &str) -> Option<PathBuf> {
        if id == "gemma4" {
            let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") else {
                skip("CUCA_LOCAL_MODEL_PATH not set");
                return None;
            };
            return Some(PathBuf::from(path));
        }
        let fixture: Value =
            serde_json::from_str(include_str!("../testdata/tokenizer_golden.json"))
                .expect("tokenizer fixture parses");
        let suffix = fixture["models"]
            .as_array()
            .and_then(|models| models.iter().find(|m| m["id"] == id))
            .and_then(|m| m["gguf_path_suffix"].as_str())
            .unwrap_or_else(|| panic!("no tokenizer fixture entry for {id}"));
        let Some(dir) = catalog_dir() else {
            skip("no model cache directory");
            return None;
        };
        let path = dir.join(suffix);
        if path.is_file() {
            Some(path)
        } else {
            skip(&format!("{id}: {} is not cached", path.display()));
            None
        }
    }

    /// The prompt of a chat case.
    fn prompt(cases: &Value, case: &str) -> ChatPrompt {
        let input = &cases[case];
        ChatPrompt {
            messages: input["messages"].clone(),
            tools: input
                .get("tools")
                .cloned()
                .unwrap_or(Value::Array(Vec::new())),
            enable_thinking: input["enable_thinking"].as_bool().expect("flag"),
        }
    }

    #[test]
    fn detection_and_prompt_tokenization_on_the_catalog() {
        let cases: Value =
            serde_json::from_str(include_str!("../testdata/chat_cases.json")).expect("cases parse");
        for expect in EXPECTED {
            let Some(path) = model_path(expect.id) else {
                continue;
            };
            let md = read_metadata(&path);
            let tokenizer = Tokenizer::from_gguf(&md).expect("tokenizer builds");
            let source = template_source(expect.template);
            let format = ChatFormat::detect(&md, &tokenizer, source)
                .unwrap_or_else(|e| panic!("{}: {e}", expect.id));
            let id = expect.id;
            assert_eq!(
                format.reasoning().map(|r| r.open.as_str()),
                expect.reasoning_open,
                "{id}: reasoning"
            );
            let kind = match format.tool_syntax() {
                ToolSyntax::Plain => "plain",
                ToolSyntax::Gemma4 { .. } => "gemma4",
                ToolSyntax::TaggedJson { .. } => "tagged",
                ToolSyntax::TaggedXml { .. } => "xml",
                ToolSyntax::LlamaJson { .. } => "llama",
            };
            assert_eq!(kind, expect.tools, "{id}: tool syntax");
            for text in expect.eog {
                let token = tokenizer
                    .token_id(text)
                    .unwrap_or_else(|| panic!("{id}: vocabulary lacks {text}"));
                assert!(format.is_eog(token), "{id}: {text} must end generation");
            }
            if let Some(eos) = tokenizer.eos_id() {
                assert!(format.is_eog(eos), "{id}: EOS must end generation");
            }

            let template = ChatTemplate::new(
                source.to_string(),
                tokenizer.bos_token().to_string(),
                tokenizer.eos_token().to_string(),
            )
            .expect("template compiles");
            for case in [
                "plain",
                "system_user",
                "thinking_on",
                "thinking_off",
                "tool_round_trip",
            ] {
                let Ok(rendered) = template.render_at(&prompt(&cases, case), PINNED_NOW) else {
                    continue; // e.g. gemma3 rejects a tool turn
                };
                // A template may leave the reasoning block open for the model
                // to continue (Qwen 3.5 with thinking on ends in `<think>\n`);
                // the format must say so exactly then.
                let open_block = format
                    .reasoning()
                    .is_some_and(|r| rendered.trim_end().ends_with(r.open.as_str()));
                assert_eq!(
                    format.starts_in_thinking(&rendered),
                    open_block,
                    "{id}/{case}: starts_in_thinking must follow the open reasoning block"
                );
                let ids = tokenizer.encode_prompt(&rendered);
                assert!(!ids.is_empty(), "{id}/{case}: empty prompt");
                if let Some(bos) = tokenizer.bos_id()
                    && tokenizer.add_bos()
                {
                    assert_eq!(ids[0], bos, "{id}/{case}: BOS first");
                    assert_ne!(ids.get(1), Some(&bos), "{id}/{case}: double BOS");
                }
                // Control tokens render as nothing, ordinary text must survive.
                let mut bytes = Vec::new();
                for &token in &ids {
                    tokenizer.decode_into(token, &mut bytes);
                }
                let decoded = String::from_utf8(bytes).expect("prompt decodes to UTF-8");
                let user = prompt(&cases, case).messages.as_array().and_then(|m| {
                    m.iter()
                        .find(|m| m["role"] == "user")
                        .map(|m| m["content"].clone())
                });
                let user = user.expect("every case has a user turn");
                assert!(
                    decoded.contains(user.as_str().expect("text")),
                    "{id}/{case}: {decoded:?}"
                );
            }
        }
    }

    #[test]
    fn reasoning_prompts_are_recognised() {
        let Some(path) = model_path("qwen3") else {
            return;
        };
        let md = read_metadata(&path);
        let tokenizer = Tokenizer::from_gguf(&md).expect("tokenizer builds");
        let format =
            ChatFormat::detect(&md, &tokenizer, template_source("qwen3")).expect("detects");
        assert!(format.starts_in_thinking("<|im_start|>assistant\n<think>\n"));
        assert!(!format.starts_in_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    /// Run generated `text` (tokenized like model output) through the parser.
    fn parse(
        id: &str,
        template: &str,
        text: &str,
    ) -> Option<Vec<Result<crate::types::MessageContentBlock, crate::error::CucaError>>> {
        let path = model_path(id)?;
        let md = read_metadata(&path);
        let tokenizer = Tokenizer::from_gguf(&md).expect("tokenizer builds");
        let format =
            ChatFormat::detect(&md, &tokenizer, template_source(template)).expect("detects");
        let mut parser = crate::provider::local::output::OutputParser::new(&format, false);
        let mut out = Vec::new();
        let mut n = 0;
        let mut next = || {
            n += 1;
            format!("call_{n}")
        };
        for token in tokenizer.encode(text) {
            assert!(!format.is_eog(token), "{id}: text contains an EOG token");
            parser.push(token, &tokenizer, &mut next, &mut out);
        }
        parser.finish(&mut next, &mut out);
        Some(out)
    }

    #[test]
    fn model_output_is_parsed_per_family() {
        use crate::types::MessageContentBlock as B;
        use serde_json::json;
        let text = |s: &str| B::Text(s.to_string());
        let thinking = |s: &str| B::Thinking {
            reasoning: s.to_string(),
            signature: None,
        };
        let call = |name: &str, args: Value| B::ToolCall {
            id: "call_1".into(),
            name: name.into(),
            arguments: args,
        };
        let weather = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let cases: Vec<(&str, &str, String, Vec<B>)> = vec![
            (
                "qwen3",
                "qwen3",
                format!(
                    "<think>\nIs it?\n</think>\n\nChecking.\n<tool_call>\n{weather}\n</tool_call>"
                ),
                vec![
                    thinking("Is it?"),
                    thinking("\n"),
                    text("Checking."),
                    text("\n"),
                    call("get_weather", json!({ "city": "Paris" })),
                ],
            ),
            (
                "smollm3",
                "smollm3",
                format!("<think>\nhm\n</think>\nOk<tool_call>\n{weather}\n</tool_call>"),
                vec![
                    thinking("hm"),
                    thinking("\n"),
                    text("Ok"),
                    call("get_weather", json!({ "city": "Paris" })),
                ],
            ),
            (
                "llama32",
                "llama32",
                r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#.to_string(),
                vec![call("get_weather", json!({ "city": "Paris" }))],
            ),
            (
                "llama32",
                "llama32",
                "Hello there, {not a call}.".to_string(),
                vec![text("Hello there, {not a call}.")],
            ),
            (
                "granitemoe",
                "granitemoe",
                r#"<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
                    .to_string(),
                vec![call("get_weather", json!({ "city": "Paris" }))],
            ),
            (
                "phi35",
                "phi35",
                "Hello there".to_string(),
                vec![text("Hello"), text(" there")],
            ),
            (
                "r1qwen",
                "r1qwen",
                "<think>\nhmm\n</think>\n\nThe answer.".to_string(),
                vec![thinking("hmm"), thinking("\n"), text("The answer.")],
            ),
        ];
        for (id, template, generated, expected) in cases {
            let Some(out) = parse(id, template, &generated) else {
                continue;
            };
            let blocks: Vec<B> = out
                .into_iter()
                .map(|r| r.unwrap_or_else(|e| panic!("{id}: {e}")))
                .collect();
            // Adjacent blocks of one kind may be split differently by token
            // boundaries; compare the merged form.
            assert_eq!(merge(blocks), merge(expected), "{id}: {generated:?}");
        }
    }

    /// Concatenate adjacent text blocks and adjacent thinking blocks.
    fn merge(
        blocks: Vec<crate::types::MessageContentBlock>,
    ) -> Vec<crate::types::MessageContentBlock> {
        use crate::types::MessageContentBlock as B;
        let mut out: Vec<B> = Vec::new();
        for block in blocks {
            match (out.last_mut(), &block) {
                (Some(B::Text(a)), B::Text(b)) => a.push_str(b),
                (Some(B::Thinking { reasoning: a, .. }), B::Thinking { reasoning: b, .. }) => {
                    a.push_str(b)
                }
                _ => out.push(block),
            }
        }
        out
    }

    /// Cross-check against the HuggingFace transformers oracle dumps
    /// (`.omp/local-models/ref/<id>.json`, directory in `CUCA_LOCAL_REF_DIR`):
    /// the GGUF template rendered for one user turn (`enable_thinking=False`,
    /// `strftime_now` pinned to 2025-01-15) must equal `chat_template.rendered`
    /// and `encode_prompt` of it the oracle's `chat_a` ids.
    #[test]
    fn render_and_ids_match_the_transformers_oracle() {
        let Some(dir) = std::env::var_os("CUCA_LOCAL_REF_DIR") else {
            skip("CUCA_LOCAL_REF_DIR not set");
            return;
        };
        let user = "What is the capital of France? Answer in one word.";
        let prompt = ChatPrompt {
            messages: serde_json::json!([{ "role": "user", "content": user }]),
            tools: Value::Array(Vec::new()),
            enable_thinking: false,
        };
        for expect in EXPECTED.iter().filter(|e| e.id != "gemma4") {
            let path = PathBuf::from(&dir).join(format!("{}.json", expect.id));
            let Ok(text) = std::fs::read_to_string(&path) else {
                skip(&format!("{} not found", path.display()));
                continue;
            };
            let reference: Value = serde_json::from_str(&text).expect("oracle dump parses");
            let Some(model) = model_path(expect.id) else {
                continue;
            };
            let md = read_metadata(&model);
            let tokenizer = Tokenizer::from_gguf(&md).expect("tokenizer builds");
            let template = ChatTemplate::new(
                template_source(expect.template).to_string(),
                tokenizer.bos_token().to_string(),
                tokenizer.eos_token().to_string(),
            )
            .expect("template compiles");
            let rendered = template
                .render_at(&prompt, 1_736_942_400)
                .unwrap_or_else(|e| panic!("{}: {e}", expect.id));
            assert_eq!(
                rendered,
                reference["chat_template"]["rendered"]
                    .as_str()
                    .expect("rendered"),
                "{}: rendered prompt",
                expect.id
            );
            let want: Vec<u64> = reference["prompts"][0]["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|n| n.as_u64().expect("id"))
                .collect();
            let got: Vec<u64> = tokenizer
                .encode_prompt(&rendered)
                .into_iter()
                .map(u64::from)
                .collect();
            assert_eq!(got, want, "{}: prompt ids", expect.id);
        }
    }
}

/// Later catalog models: the rendered chat prompt and the token ids of every
/// oracle prompt equal the transformers oracle's (`CUCA_LOCAL_REF_DIR`,
/// models from `CUCA_LOCAL_CATALOG_DIR`). Absent files print `SKIP:`.
mod later_models {
    use serde_json::Value;

    use super::template_source;
    use crate::provider::local::chat::{ChatFormat, ChatPrompt, ChatTemplate};
    use crate::provider::local::tokenizer::Tokenizer;
    use crate::provider::local::tokenizer::golden::{catalog_dir, read_metadata, skip};

    /// `(oracle id, template fixture, GGUF path under the catalog dir)`.
    const MODELS: &[(&str, &str, &str)] = &[
        (
            "gemma2",
            "gemma2",
            "second-state--gemma-2-2b-it-GGUF/b7d6a1a55228d89cc129757c687384b96781f97e/gemma-2-2b-it-Q4_0.gguf",
        ),
        (
            "olmo2",
            "olmo2",
            "allenai--OLMo-2-0425-1B-Instruct-GGUF/62f8c199538474c3e33ed5d7e0580abd66686a27/OLMo-2-0425-1B-Instruct-Q4_0.gguf",
        ),
        (
            "granite33",
            "granite33",
            "ibm-granite--granite-3.3-2b-instruct-GGUF/7cdf86ccd1f1bb3491c9b7017b033f2e51367397/granite-3.3-2b-instruct-Q4_0.gguf",
        ),
        (
            "ernie45",
            "ernie45",
            "unsloth--ERNIE-4.5-0.3B-PT-GGUF/32b9743ef4cb56af53d71a6ab4b822d191ccb4fc/ERNIE-4.5-0.3B-PT-Q8_0.gguf",
        ),
        (
            "smollm2",
            "smollm2",
            "bartowski--SmolLM2-360M-Instruct-GGUF/7be6f65f1db715fe5dc5a4634c0d459b4eed42ec/SmolLM2-360M-Instruct-Q8_0.gguf",
        ),
        (
            "tinyllama",
            "tinyllama",
            "TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        ),
    ];

    /// SentencePiece models whose ids differ from HF's by llama.cpp's space
    /// prefix rule (ERNIE 4.5 omits `tokenizer.ggml.add_space_prefix`;
    /// TinyLlama's text after `</s>` gets a space), see the comparison below.
    const SPACE_PREFIX: &[&str] = &["ernie45", "tinyllama"];

    #[test]
    fn prompts_and_ids_match_the_transformers_oracle() {
        let Some(refs) = std::env::var_os("CUCA_LOCAL_REF_DIR") else {
            skip("CUCA_LOCAL_REF_DIR not set");
            return;
        };
        let Some(dir) = catalog_dir() else {
            skip("no model cache directory");
            return;
        };
        for (id, template_name, suffix) in MODELS {
            let path = dir.join(suffix);
            let oracle_path = std::path::Path::new(&refs).join(format!("{id}.json"));
            let (Ok(text), true) = (std::fs::read_to_string(&oracle_path), path.is_file()) else {
                skip(&format!("{id}: model or oracle absent"));
                continue;
            };
            let oracle: Value = serde_json::from_str(&text).expect("oracle parses");
            let md = read_metadata(&path);
            let tokenizer = Tokenizer::from_gguf(&md).expect("tokenizer builds");
            let source = template_source(template_name);
            ChatFormat::detect(&md, &tokenizer, source).unwrap_or_else(|e| panic!("{id}: {e}"));
            let template = ChatTemplate::new(
                source.to_string(),
                tokenizer.bos_token().to_string(),
                tokenizer.eos_token().to_string(),
            )
            .expect("template compiles");
            let user = "What is the capital of France? Answer in one word.";
            let prompt = ChatPrompt {
                messages: serde_json::json!([{ "role": "user", "content": user }]),
                tools: Value::Array(Vec::new()),
                enable_thinking: false,
            };
            let rendered = template
                .render_at(&prompt, PINNED_NOW_2025)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(
                rendered,
                oracle["chat_template"]["rendered"]
                    .as_str()
                    .expect("rendered"),
                "{id}: rendered chat prompt"
            );
            for prompt in oracle["prompts"].as_array().expect("prompts") {
                let name = prompt["name"].as_str().expect("name");
                let want: Vec<u32> = prompt["ids"]
                    .as_array()
                    .expect("ids")
                    .iter()
                    .map(|n| n.as_u64().expect("id") as u32)
                    .collect();
                // The long prompts keep no text in the dump.
                let Some(text) = prompt["text"].as_str() else {
                    continue;
                };
                let got = if name == "chat_a" {
                    // The oracle tokenizes the rendered template as is.
                    tokenizer.encode(text)
                } else {
                    tokenizer.encode_prompt(text)
                };
                if SPACE_PREFIX.contains(id) {
                    // llama.cpp's SentencePiece default puts a space in front
                    // of the text after a special token (and of the whole
                    // text when the GGUF omits `tokenizer.ggml.add_space_prefix`),
                    // while the HF tokenizers add none: the ids differ in a
                    // bare `▁` token and in the words that start a text.
                    let bare = tokenizer.token_id("\u{2581}");
                    let strip = |ids: &[u32]| -> Vec<u32> {
                        ids.iter().copied().filter(|&i| Some(i) != bare).collect()
                    };
                    let (got, want) = (strip(&got), strip(&want));
                    let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                    assert!(
                        got.len() == want.len() && differing <= 2,
                        "{id}/{name}: {differing} differing ids\n  want {want:?}\n  got  {got:?}"
                    );
                } else {
                    assert_eq!(got, want, "{id}/{name}: token ids");
                }
            }
        }
    }

    /// 2025-01-15 12:00 UTC, the date the oracle pinned.
    const PINNED_NOW_2025: i64 = 1_736_942_400;
}
