//! The engine's named chat templates: llama.cpp's and this engine's own.
//!
//! A GGUF's `tokenizer.chat_template` (or a `--chat-template` argument) can be a
//! bare *name* instead of Jinja: llama.cpp's converter writes `mistral-v7-tekken`
//! for a Mistral checkpoint with a v7 `tekken.json`, and llama.cpp renders the
//! name in C++ (`llm_chat_apply_template`, `src/llama-chat.cpp`, the
//! `LLM_CHAT_TEMPLATE_MISTRAL_V7` and `..._V7_TEKKEN` branches). Other servers
//! render the same GGUFs, so [`expand`] turns a known name into the equivalent
//! Jinja text, which the engine then handles like any template.
//!
//! The llama.cpp table is limited to the names a 2025 or later model uses: the
//! v7 formats of Mistral Small 3.0/3.1 (2501, 2503) and Devstral Small 2505.
//! llama.cpp's other Mistral names (`mistral-v1`, `mistral-v3`,
//! `mistral-v3-tekken`, the Llama 2 and Mistral Nemo era) serve models from
//! before 2025 and are not ported. `deepseek-v41` is not a llama.cpp name (it
//! has no built-in for DeepSeek V4.x, whose checkpoint ships no Jinja either):
//! it is the template that reproduces DeepSeek's reference `encoding.py` for
//! V4.1, for the GGUFs the converter will write and for template overrides.
//!
//! The Mistral port follows the C++ branch exactly:
//!
//! - `system` writes `[SYSTEM_PROMPT]{sp}{content}[/SYSTEM_PROMPT]` wherever the
//!   message sits (no merging into the first user turn), `user` writes
//!   `[INST]{sp}{content}[/INST]`, and every other role (assistant, but also
//!   tool) writes `{sp}{content}</s>`, where `{sp}` is one space for
//!   `mistral-v7` and nothing for `mistral-v7-tekken`;
//! - no BOS is written (the tokenizer adds it, `add_bos_token`), nothing is
//!   written for the generation prompt, and `tools` are ignored: the model
//!   never sees them, as in llama.cpp;
//! - a message without text content (an assistant turn that only carries tool
//!   calls) contributes an empty content, as the C++ `std::string` does.

use std::borrow::Cow;

/// `deepseek-v41`: DeepSeek's `encoding/encoding.py` of
/// DeepSeek-V4.1-Flash-1017 as Jinja (the file `templates/deepseek-v41.jinja`).
pub(crate) const DEEPSEEK_V41: &str = include_str!("templates/deepseek-v41.jinja");

/// `mistral-v7`: the SentencePiece variant, a space after every opener and
/// before every assistant turn.
const MISTRAL_V7: &str = "{% for message in messages %}\
{% if message['role'] == 'system' %}{{ '[SYSTEM_PROMPT] ' ~ (message['content'] or '') ~ '[/SYSTEM_PROMPT]' }}\
{% elif message['role'] == 'user' %}{{ '[INST] ' ~ (message['content'] or '') ~ '[/INST]' }}\
{% else %}{{ ' ' ~ (message['content'] or '') ~ '</s>' }}\
{% endif %}{% endfor %}";

/// `mistral-v7-tekken`: the same without the spaces.
const MISTRAL_V7_TEKKEN: &str = "{% for message in messages %}\
{% if message['role'] == 'system' %}{{ '[SYSTEM_PROMPT]' ~ (message['content'] or '') ~ '[/SYSTEM_PROMPT]' }}\
{% elif message['role'] == 'user' %}{{ '[INST]' ~ (message['content'] or '') ~ '[/INST]' }}\
{% else %}{{ (message['content'] or '') ~ '</s>' }}\
{% endif %}{% endfor %}";

/// The Jinja text of the known name `name` (llama.cpp compares the whole
/// string, so a name with any other character is not one).
fn lookup(name: &str) -> Option<&'static str> {
    match name {
        "mistral-v7" => Some(MISTRAL_V7),
        "mistral-v7-tekken" => Some(MISTRAL_V7_TEKKEN),
        "deepseek-v41" => Some(DEEPSEEK_V41),
        _ => None,
    }
}

/// Whether `source` looks like a bare template name rather than Jinja: one
/// token of letters, digits, `.`, `-` and `_`, no whitespace and no Jinja
/// delimiter. Real templates always have a `{`.
fn looks_like_a_name(source: &str) -> bool {
    !source.is_empty()
        && source
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

/// The template text to compile for `source`: the Jinja equivalent of a known
/// llama.cpp template name, `source` itself when it is Jinja.
///
/// # Errors
///
/// A bare name this engine does not know, which would otherwise compile as a
/// literal template and make every prompt the name's text. The message names
/// the supported ones.
pub(crate) fn expand(source: &str) -> Result<Cow<'_, str>, String> {
    if let Some(jinja) = lookup(source) {
        return Ok(Cow::Borrowed(jinja));
    }
    if looks_like_a_name(source) {
        return Err(format!(
            "`{source}` is a bare chat template name, not a Jinja template; the names this engine renders are \
             `mistral-v7`, `mistral-v7-tekken` and `deepseek-v41` (llama.cpp's other built-in names belong to models from before 2025)"
        ));
    }
    Ok(Cow::Borrowed(source))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::provider::local::chat::{ChatPrompt, ChatTemplate, ThinkingRequest};

    fn render(name: &str, messages: Value) -> String {
        let source = expand(name).expect("known name");
        ChatTemplate::new(source.into_owned(), "<s>".into(), "</s>".into())
            .expect("compiles")
            .render(&ChatPrompt {
                messages,
                tools: json!([]),
                enable_thinking: false,
                thinking: ThinkingRequest::Unset,
            })
            .expect("renders")
    }

    /// The conversation of llama.cpp's `tests/test-chat-template.cpp`.
    fn conversation() -> Value {
        json!([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Who are you"},
            {"role": "assistant", "content": "   I am an assistant   "},
            {"role": "user", "content": "Another question"},
        ])
    }

    /// llama.cpp's own expected output for the `mistral-v7` case of
    /// `test-chat-template.cpp` ("Mistral-Large-Instruct-2411 (mistralai 'v7'
    /// template)", `expected_output`, the C++ path).
    #[test]
    fn mistral_v7_matches_llama_cpps_expected_output() {
        let want = "[SYSTEM_PROMPT] You are a helpful assistant[/SYSTEM_PROMPT][INST] Hello[/INST] Hi there</s>[INST] Who are you[/INST]    I am an assistant   </s>[INST] Another question[/INST]";
        assert_eq!(render("mistral-v7", conversation()), want);
        // That test's own Jinja template (the official v7 one) renders the
        // same bytes in this engine, with BOS empty as the test sets it.
        let official = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'system' %}{{ '[SYSTEM_PROMPT] ' + message['content'] + '[/SYSTEM_PROMPT]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% else %}{{ raise_exception('Only user, system and assistant roles are supported!') }}{% endif %}{% endfor %}";
        let got = ChatTemplate::new(official.into(), String::new(), "</s>".into())
            .unwrap()
            .render(&ChatPrompt {
                messages: conversation(),
                tools: json!([]),
                enable_thinking: false,
                thinking: ThinkingRequest::Unset,
            })
            .unwrap();
        assert_eq!(got, want);
    }

    /// `mistral-v7-tekken` is the C++ branch with no trailing space.
    #[test]
    fn mistral_v7_tekken_follows_the_cpp_branch() {
        assert_eq!(
            render("mistral-v7-tekken", conversation()),
            "[SYSTEM_PROMPT]You are a helpful assistant[/SYSTEM_PROMPT][INST]Hello[/INST]Hi there</s>[INST]Who are you[/INST]   I am an assistant   </s>[INST]Another question[/INST]"
        );
        // A system message anywhere, a tool message written like an assistant
        // one, an assistant turn of tool calls only (no content), and no
        // generation prompt, no BOS.
        let messages = json!([
            {"role": "user", "content": "a"},
            {"role": "system", "content": "s"},
            {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": {}}}]},
            {"role": "tool", "content": "result"},
        ]);
        assert_eq!(
            render("mistral-v7-tekken", messages),
            "[INST]a[/INST][SYSTEM_PROMPT]s[/SYSTEM_PROMPT]</s>result</s>"
        );
        assert_eq!(render("mistral-v7-tekken", json!([])), "");
    }

    /// `mistral-v7-tekken` equals what `mistral_common` (the reference Tekken v7
    /// instruct tokenizer, with the real Mistral Small 3.1 `tekken.json`)
    /// writes for the same conversations, apart from the leading `<s>` the
    /// engine's tokenizer adds.
    #[test]
    fn mistral_v7_tekken_matches_mistral_common() {
        let golden: Value =
            serde_json::from_str(include_str!("../testdata/chat_golden_v7.json")).unwrap();
        let cases = golden.as_object().unwrap();
        assert!(cases.len() >= 6);
        for (name, case) in cases {
            let want = case["text"].as_str().unwrap();
            let want = want
                .strip_prefix("<s>")
                .expect("mistral_common starts with BOS");
            assert_eq!(
                render("mistral-v7-tekken", case["messages"].clone()),
                want,
                "{name}"
            );
        }
    }

    #[test]
    fn jinja_passes_through_and_an_unknown_name_is_an_error() {
        for jinja in ["{{ messages }}", "[INST] {{ x }}", "plain text with spaces"] {
            assert_eq!(expand(jinja).unwrap(), jinja);
        }
        for name in ["mistral-v3-tekken", "mistral-v1", "chatml", "Mistral-V7"] {
            let e = expand(name).unwrap_err();
            assert!(e.contains(name) && e.contains("mistral-v7-tekken"), "{e}");
        }
    }
}
