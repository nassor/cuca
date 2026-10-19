//! Golden tests against the real cached GGUF models.
//!
//! The reference ids come from the HuggingFace `tokenizer.json` of each model
//! (`tokenizers` 0.23, `encode(text, add_special_tokens=False)`), stored in
//! `testdata/tokenizer_golden.json`:
//!
//! ```json
//! { "texts": [..shared inputs..], "prompts": [..],
//!   "models": [ { "id", "hf_repo", "gguf_path_suffix", "add_bos", "add_eos",
//!                 "ids": [[..per shared text..]], "chat_text", "chat_ids",
//!                 "prompt_ids": [[..per prompt, BOS/EOS applied..]] } ] }
//! ```
//!
//! `gguf_path_suffix` is `<repo with / as -->/<revision>/<file>` under the
//! `huggingface` directory of the model cache: `CUCA_LOCAL_CATALOG_DIR` names
//! that directory, else it is `<cache dir>/huggingface` with the cache dir
//! resolved exactly like the downloader does (`CUCA_CACHE_DIR`, then
//! `%LOCALAPPDATA%\cuca` and friends). A model whose file is absent prints
//! `SKIP:` and passes, unless `CUCA_REQUIRE_LIVE=1`.
//!
//! Where HuggingFace and llama.cpp semantics legitimately differ, the expected
//! ids follow llama.cpp, and the fixture is generated so the difference does not
//! show (`hf_repo` is the repository whose `tokenizer.json` produced a model's
//! ids):
//!
//! - `lstrip`/`rstrip` on added tokens: HuggingFace's Phi-3.5 `tokenizer.json`
//!   strips the whitespace after `<|user|>`, `<|end|>`, ...; this tokenizer
//!   reproduces it (llama.cpp's `RSTRIP` rule, widened to the Phi-3.5 GGUF whose
//!   `general.name` has no hyphen: see the module docs of `tokenizer`), so the
//!   Phi-3.5 vectors are HuggingFace's with its flags unchanged. Phi-4-mini also
//!   has the flags in `tokenizer.json` but is not a Phi-3 SPM model, gets no
//!   stripping here, and its vectors (no whitespace after a special) agree.
//! - Unicode normalization: Qwen-family `tokenizer.json` files NFC-normalize
//!   the text first; llama.cpp does not, so the shared texts are NFC-stable.
//! - GPT-4o-style case classes: llama.cpp approximates `\p{Lu}`/`\p{Ll}` with
//!   ASCII case only; this crate uses the real categories like HuggingFace, so
//!   the fixture's mixed-case non-ASCII text (`\u{c9}COLE \u{c9}cole`) is the
//!   HuggingFace result.
//! - The `refact` pre-tokenizer (IBM Granite 3.1): llama.cpp splits single digits
//!   out first and runs the GPT-2 stage on the fragments in between, so a
//!   whitespace run directly before a digit stays whole (`"  1"` is `"  "`, `1`),
//!   whereas HuggingFace's single GPT-2 regex over the whole text splits off the
//!   last space (`" "`, `" "`, `1`). This tokenizer follows llama.cpp. A 1500-text
//!   random comparison against HuggingFace found every one of its granite
//!   mismatches (120) to be exactly this case, and none for the other ten models.
//!
//! The Gemma 4 test (`CUCA_LOCAL_MODEL_PATH`) keeps its own vectors from the
//! reference tokenizer of google/gemma-4-E2B-it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use serde::Deserialize;

use super::Tokenizer;
use crate::provider::local::download::resolve_cache_dir;

/// The committed reference ids.
const FIXTURE: &str = include_str!("../testdata/tokenizer_golden.json");

/// The fixture file.
#[derive(Deserialize)]
struct Fixture {
    /// Inputs shared by every model.
    texts: Vec<String>,
    /// Inputs for the BOS/EOS (`encode_prompt`) check.
    prompts: Vec<String>,
    /// One entry per catalog model.
    models: Vec<ModelCase>,
}

/// Reference ids of one model.
#[derive(Deserialize)]
struct ModelCase {
    /// Catalog id (`qwen3`, ...).
    id: String,
    /// GGUF location below the `huggingface` cache directory.
    gguf_path_suffix: String,
    /// Expected [`Tokenizer::add_bos`].
    add_bos: bool,
    /// Expected [`Tokenizer::add_eos`].
    add_eos: bool,
    /// `ids[i]` encodes `texts[i]`.
    ids: Vec<Vec<u32>>,
    /// A chat-template-shaped text with the model's own special tokens.
    chat_text: String,
    /// Reference ids of `chat_text`.
    chat_ids: Vec<u32>,
    /// `prompt_ids[i]` is the prompt encoding of `prompts[i]`.
    prompt_ids: Vec<Vec<u32>>,
}

/// The directory that holds `<repo>/<revision>/<file>` model caches.
pub(crate) fn catalog_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("CUCA_LOCAL_CATALOG_DIR") {
        return Some(PathBuf::from(dir));
    }
    let cache = resolve_cache_dir(None, &|key| std::env::var_os(key)).ok()?;
    Some(cache.join("huggingface"))
}

/// Report a missing model: print `SKIP:`, or panic under `CUCA_REQUIRE_LIVE=1`.
pub(crate) fn skip(what: &str) {
    assert!(
        std::env::var("CUCA_REQUIRE_LIVE").as_deref() != Ok("1"),
        "required model unavailable: {what}"
    );
    eprintln!("SKIP: {what}");
}

/// Read the GGUF header metadata of `path` (tensor data is not touched).
pub(crate) fn read_metadata(path: &Path) -> HashMap<String, gguf_file::Value> {
    let mut file = std::fs::File::open(path).expect("model file opens");
    gguf_file::Content::read(&mut file)
        .expect("GGUF header parses")
        .metadata
}

/// Compare one id list, appending a readable failure to `failures`.
fn compare(
    failures: &mut Vec<String>,
    model: &str,
    what: &str,
    text: &str,
    expected: &[u32],
    got: &[u32],
) {
    if expected != got {
        let first = expected
            .iter()
            .zip(got)
            .position(|(a, b)| a != b)
            .unwrap_or(expected.len().min(got.len()));
        failures.push(format!(
            "{model} {what} {text:?}: first difference at {first}\n  expected {expected:?}\n  got      {got:?}"
        ));
    }
}

#[test]
fn golden_ids_match_reference_tokenizers() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    assert_eq!(fixture.models.len(), 11, "every catalog model has a vector");
    let Some(dir) = catalog_dir() else {
        skip("no model cache directory");
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0;
    for model in &fixture.models {
        let path = dir.join(&model.gguf_path_suffix);
        if !path.is_file() {
            skip(&format!("{} not cached at {}", model.id, path.display()));
            continue;
        }
        let tokenizer = Tokenizer::from_gguf(&read_metadata(&path))
            .unwrap_or_else(|e| panic!("{}: tokenizer builds: {e}", model.id));
        assert_eq!(tokenizer.add_bos(), model.add_bos, "{}: add_bos", model.id);
        assert_eq!(tokenizer.add_eos(), model.add_eos, "{}: add_eos", model.id);
        assert_eq!(model.ids.len(), fixture.texts.len());
        for (text, expected) in fixture.texts.iter().zip(&model.ids) {
            compare(
                &mut failures,
                &model.id,
                "encode",
                text,
                expected,
                &tokenizer.encode(text),
            );
        }
        compare(
            &mut failures,
            &model.id,
            "chat",
            &model.chat_text,
            &model.chat_ids,
            &tokenizer.encode(&model.chat_text),
        );
        assert_eq!(model.prompt_ids.len(), fixture.prompts.len());
        for (text, expected) in fixture.prompts.iter().zip(&model.prompt_ids) {
            compare(
                &mut failures,
                &model.id,
                "prompt",
                text,
                expected,
                &tokenizer.encode_prompt(text),
            );
        }
        checked += 1;
        eprintln!("golden ok: {}", model.id);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!(
        "golden: {checked} of {} models checked",
        fixture.models.len()
    );
}

/// A chat prompt that already starts with the model's BOS text must not get a
/// second BOS from `encode_prompt` (llama.cpp only warns about it).
#[test]
fn golden_prompt_never_doubles_bos() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    let Some(dir) = catalog_dir() else {
        skip("no model cache directory");
        return;
    };
    for model in fixture.models.iter().filter(|m| m.add_bos) {
        let path = dir.join(&model.gguf_path_suffix);
        if !path.is_file() {
            skip(&format!("{} not cached", model.id));
            continue;
        }
        let tokenizer = Tokenizer::from_gguf(&read_metadata(&path)).expect("tokenizer builds");
        let bos = tokenizer.bos_id().expect("add_bos models have a BOS id");
        let with_bos = format!("{}Hello", tokenizer.bos_token());
        let ids = tokenizer.encode_prompt(&with_bos);
        assert_eq!(ids.first(), Some(&bos), "{}", model.id);
        assert_ne!(ids.get(1), Some(&bos), "{}: double BOS", model.id);
        assert_eq!(ids, tokenizer.encode(&with_bos), "{}", model.id);
    }
}

/// Model-gated golden test against ids produced by the reference
/// HuggingFace tokenizer (`tokenizer.json` of google/gemma-4-E2B-it).
///
/// The reference merges across `\n`/non-`\n` boundaries while llama.cpp (and
/// this tokenizer) cut newline runs first. If a vector ever differs from the
/// reference only at a merge spanning such a boundary, the llama.cpp-style
/// (newline-run split) result is the expected one; all current vectors agree.
#[test]
fn gemma4_ids_match_reference_tokenizer() {
    let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") else {
        skip("CUCA_LOCAL_MODEL_PATH not set");
        return;
    };
    let t = Tokenizer::from_gguf(&read_metadata(Path::new(&path))).expect("tokenizer builds");
    assert_eq!(t.vocab_size(), 262_144);
    assert_eq!(t.token_id("<bos>"), Some(2));
    assert_eq!(t.token_id("<eos>"), Some(1));
    assert_eq!(t.bos_id(), Some(2));
    assert!(t.add_bos(), "gemma4 forces BOS");
    for (text, expected) in GEMMA4 {
        assert_eq!(t.encode(text), *expected, "encoding of {text:?}");
    }
    assert_eq!(t.encode_prompt("Hello world"), [2, 9259, 1902]);
}

/// Expected Gemma 4 ids per input.
const GEMMA4: &[(&str, &[u32])] = &[
    ("Hello world", &[9259, 1902]),
    (
        "naïve café 東京 🦀",
        &[1789, 238527, 560, 33443, 86459, 236743, 248714],
    ),
    ("  two leading spaces", &[138, 13498, 5830, 9952]),
    (
        "<bos><|turn>user\nHello<turn|>\n<|turn>model\n",
        &[2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107],
    ),
    (
        "line one\nline two\n\nthird  line",
        &[1257, 886, 107, 1257, 1156, 108, 23362, 138, 1257],
    ),
];
