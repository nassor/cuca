//! Checkpoints that ship their vocabulary only as `tekken.json`.
//!
//! Mistral Small 3.1/3.2, Magistral 2506-2509 and Devstral Small 2505/2507
//! publish Hugging Face shards with `tekken.json` (Mistral's own tokenizer
//! file) and no `tokenizer.json`. llama.cpp's `LlamaModel.set_vocab` sends such
//! a directory through `_set_vocab_mistral` and gguf-py's `MistralVocab`; this
//! module is that path, with `mistral_common`'s `Tekkenizer` loading rules:
//!
//! - the vocabulary has `default_vocab_size` entries: `default_num_special_tokens`
//!   special tokens first (those the file lists, completed with
//!   `<SPECIAL_{i}>`), then the first `default_vocab_size - default_num_special_tokens`
//!   ranked byte strings (`token_bytes`, base64), shown in GPT-2's byte-to-unicode
//!   alphabet;
//! - the BPE merges are rebuilt from the ranks: for every rank from 256 up, each
//!   split of the token into two ranked halves is a merge, ordered by the
//!   halves' ranks, and all merges are ordered by the merged token's rank
//!   (`MistralVocab.extract_vocab_merges_from_model`);
//! - written as `gpt2` with the `tekken` pre-tokenizer, scores 0, the first
//!   special tokens as control tokens, `<s>`, `</s>`, `<unk>` and `<pad>` ids,
//!   `add_bos_token` true, `add_eos_token` false;
//! - the chat template is the community template llama.cpp bundles for the
//!   file's `version` (`MistralModel.get_community_chat_template`): v11 the Mistral
//!   Small 3.2 one, v13 the Devstral Small 2507 one, and for v7 (Mistral Small
//!   3.0/3.1, Devstral Small 2505) the *name* `mistral-v7-tekken`, which
//!   llama.cpp renders in C++ and the engine expands to Jinja at load
//!   (`chat/builtin.rs`). For v1 and v3 (pre-2025 models) llama.cpp writes names
//!   this engine does not port, and any other version it refuses; both are an
//!   error here.
//!
//! The newer `tekken-num3` pre-tokenizer (right-aligned digit groups of three,
//! which no model this crate supports uses) is refused by name.

use std::collections::HashMap;
use std::path::Path;

use crate::provider::local::gguf::Value;
use serde_json::Value as Json;

use super::super::tokenizer::byte_to_char;
use super::cfg_err;
use super::vocab::{CONTROL, NORMAL, VocabMeta, string, strings, token_types};
use crate::error::CucaError;

/// llama.cpp's community template for tokenizer version v11 (Mistral Small
/// 3.2), `models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja`.
const TEMPLATE_V11: &str = include_str!("templates/Mistral-Small-3.2-24B-Instruct-2506.jinja");
/// ... and for v13 (Devstral Small 2507),
/// `models/templates/unsloth-mistral-Devstral-Small-2507.jinja`.
const TEMPLATE_V13: &str = include_str!("templates/unsloth-mistral-Devstral-Small-2507.jinja");

/// `Tekkenizer.DEPRECATED_SPECIAL_TOKENS` of mistral_common, the special
/// tokens of a tekken.json (v7 or older) that lists none, by rank.
const DEPRECATED_SPECIAL_TOKENS: [&str; 20] = [
    "<unk>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CALLS]",
    "[IMG]",
    "<pad>",
    "[IMG_BREAK]",
    "[IMG_END]",
    "[PREFIX]",
    "[MIDDLE]",
    "[SUFFIX]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[TOOL_CONTENT]",
];

/// Decode standard base64 (with padding), the encoding of `token_bytes`.
fn base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// `MistralVocab.token_bytes_to_string`: the bytes in GPT-2's byte-to-unicode
/// alphabet.
fn show(bytes: &[u8]) -> String {
    bytes.iter().map(|b| byte_to_char(*b)).collect()
}

/// llama.cpp's built-in template name for tokenizer version v7 (tekken).
const TEMPLATE_V7: &str = "mistral-v7-tekken";

/// The chat template for `version`: the bundled community template's text, or
/// the built-in name llama.cpp writes.
fn community_template(version: &str) -> Result<&'static str, CucaError> {
    match version {
        "v11" => Ok(TEMPLATE_V11),
        "v13" => Ok(TEMPLATE_V13),
        "v7" => Ok(TEMPLATE_V7),
        "v1" | "v3" => Err(cfg_err(format!(
            "tekken.json version {version}: llama.cpp names a template built into its C++ code for it (a pre-2025 format), which this engine does not render; \
             convert a checkpoint that ships a tokenizer.json and a chat template instead"
        ))),
        other => Err(cfg_err(format!(
            "tekken.json version {other}: no chat template is known for it (llama.cpp refuses it too)"
        ))),
    }
}

/// The merges of `MistralVocab.extract_vocab_merges_from_model`, as the
/// `left right` strings gguf stores.
fn merges(ranks: &HashMap<&[u8], usize>, by_rank: &[Vec<u8>]) -> Result<Vec<String>, CucaError> {
    let mut out = Vec::with_capacity(by_rank.len());
    for (rank, token) in by_rank.iter().enumerate().skip(256) {
        let mut local: Vec<(&[u8], &[u8])> = (1..token.len())
            .filter_map(|j| {
                let (left, right) = token.split_at(j);
                (ranks.contains_key(left) && ranks.contains_key(right)).then_some((left, right))
            })
            .collect();
        if local.is_empty() {
            return Err(cfg_err(format!(
                "tekken.json: no valid merge for the token of rank {rank}"
            )));
        }
        // `(left + right)` is the token itself, ranked by construction.
        local.sort_by_key(|(l, r)| (ranks[*l], ranks[*r]));
        out.extend(
            local
                .into_iter()
                .map(|(l, r)| format!("{} {}", show(l), show(r))),
        );
    }
    Ok(out)
}

/// The vocabulary metadata of `<dir>/tekken.json` for the architecture whose
/// metadata prefix is `arch_name`.
pub(crate) fn build(dir: &Path, arch_name: &str) -> Result<VocabMeta, CucaError> {
    let path = dir.join("tekken.json");
    let text = std::fs::read_to_string(&path).map_err(|e| CucaError::Io {
        message: format!("cannot read `{}`: {e}", path.display()),
    })?;
    let json: Json =
        serde_json::from_str(&text).map_err(|e| cfg_err(format!("`{}`: {e}", path.display())))?;
    let config = json
        .get("config")
        .ok_or_else(|| cfg_err("tekken.json has no `config`"))?;
    let int = |key: &str| -> Result<usize, CucaError> {
        config
            .get(key)
            .and_then(Json::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| cfg_err(format!("tekken.json config.{key} is missing")))
    };
    let vocab_size = int("default_vocab_size")?;
    let n_special = int("default_num_special_tokens")?;
    let version = config
        .get("version")
        .and_then(Json::as_str)
        .ok_or_else(|| cfg_err("tekken.json config.version is missing"))?;
    let pattern = config
        .get("pattern")
        .and_then(Json::as_str)
        .ok_or_else(|| cfg_err("tekken.json config.pattern is missing"))?;
    let template = community_template(version)?;
    if pattern.contains("\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))") {
        return Err(cfg_err(
            "tekken.json splits numbers into right-aligned groups of three (llama.cpp's `tekken-num3` pre-tokenizer), which the engine does not implement",
        ));
    }

    // Special tokens: the file's list (later than v7 must have one; v7 files
    // may omit it and get mistral_common's `DEPRECATED_SPECIAL_TOKENS`), then
    // `<SPECIAL_{i}>` fillers up to `default_num_special_tokens`.
    let deprecated: Vec<Json>;
    let listed = match json.get("special_tokens").and_then(Json::as_array) {
        Some(list) => list,
        None if version == "v7" => {
            deprecated = DEPRECATED_SPECIAL_TOKENS
                .iter()
                .enumerate()
                .map(|(rank, token_str)| serde_json::json!({"rank": rank, "token_str": token_str}))
                .collect();
            &deprecated
        }
        None => return Err(cfg_err("tekken.json has no `special_tokens`")),
    };
    let mut special: Vec<String> = Vec::with_capacity(n_special);
    for (i, t) in listed.iter().enumerate() {
        let rank = t.get("rank").and_then(Json::as_u64);
        let name = t.get("token_str").and_then(Json::as_str);
        match (rank, name) {
            (Some(r), Some(name)) if r == i as u64 => special.push(name.to_string()),
            _ => {
                return Err(cfg_err(format!(
                    "tekken.json special token {i} is not in rank order"
                )));
            }
        }
    }
    if special.len() > n_special {
        return Err(cfg_err(
            "tekken.json lists more special tokens than default_num_special_tokens",
        ));
    }
    for i in special.len()..n_special {
        special.push(format!("<SPECIAL_{i}>"));
    }
    let mut seen = std::collections::HashSet::new();
    if !special.iter().all(|s| seen.insert(s.as_str())) {
        return Err(cfg_err("tekken.json special tokens are not unique"));
    }

    // The ranked byte strings, cut to `default_vocab_size - n_special`.
    let inner = vocab_size
        .checked_sub(n_special)
        .ok_or_else(|| cfg_err("tekken.json default_vocab_size < default_num_special_tokens"))?;
    let entries = json
        .get("vocab")
        .and_then(Json::as_array)
        .ok_or_else(|| cfg_err("tekken.json has no `vocab`"))?;
    if entries.len() < inner {
        return Err(cfg_err(format!(
            "tekken.json has {} ranked tokens, fewer than default_vocab_size - default_num_special_tokens = {inner}",
            entries.len()
        )));
    }
    let mut bytes: Vec<Vec<u8>> = Vec::with_capacity(inner);
    for (i, entry) in entries.iter().take(inner).enumerate() {
        let decoded = entry
            .get("token_bytes")
            .and_then(Json::as_str)
            .and_then(base64)
            .ok_or_else(|| cfg_err(format!("tekken.json vocab entry {i} has no token_bytes")))?;
        if entry.get("rank").and_then(Json::as_u64) != Some(i as u64) {
            return Err(cfg_err(format!(
                "tekken.json vocab entry {i} is out of rank order"
            )));
        }
        if i < 256 && decoded != [i as u8] {
            return Err(cfg_err(format!("tekken.json rank {i} is not the byte {i}")));
        }
        bytes.push(decoded);
    }
    let mut ranks: HashMap<&[u8], usize> = HashMap::with_capacity(inner);
    for (i, b) in bytes.iter().enumerate() {
        ranks.insert(b, i);
    }
    if ranks.len() != inner {
        return Err(cfg_err("tekken.json ranked tokens are not unique"));
    }

    let merges = merges(&ranks, &bytes)?;

    let id = |name: &str| -> Result<u32, CucaError> {
        special
            .iter()
            .position(|s| s == name)
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| cfg_err(format!("tekken.json has no `{name}` special token")))
    };
    let (bos, eos, unk, pad) = (id("<s>")?, id("</s>")?, id("<unk>")?, id("<pad>")?);
    let tokens: Vec<String> = special
        .into_iter()
        .chain(bytes.iter().map(|b| show(b)))
        .collect();
    let mut types = vec![CONTROL; n_special];
    types.resize(vocab_size, NORMAL);
    let kv = vec![
        ("tokenizer.ggml.model".to_string(), string("gpt2")),
        ("tokenizer.ggml.pre".to_string(), string("tekken")),
        ("tokenizer.ggml.merges".to_string(), strings(merges)),
        ("tokenizer.ggml.bos_token_id".to_string(), Value::U32(bos)),
        ("tokenizer.ggml.eos_token_id".to_string(), Value::U32(eos)),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            Value::U32(unk),
        ),
        (
            "tokenizer.ggml.padding_token_id".to_string(),
            Value::U32(pad),
        ),
        ("tokenizer.ggml.tokens".to_string(), strings(tokens)),
        (
            "tokenizer.ggml.scores".to_string(),
            Value::Array(vec![Value::F32(0.0); vocab_size]),
        ),
        ("tokenizer.ggml.token_type".to_string(), token_types(&types)),
        (
            format!("{arch_name}.vocab_size"),
            Value::U32(u32::try_from(vocab_size).map_err(|_| cfg_err("vocabulary too large"))?),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Value::Bool(true),
        ),
        (
            "tokenizer.ggml.add_eos_token".to_string(),
            Value::Bool(false),
        ),
        ("tokenizer.chat_template".to_string(), string(template)),
    ];
    Ok(VocabMeta {
        kv,
        n_vocab: vocab_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_with_and_without_padding() {
        assert_eq!(base64("AA==").unwrap(), [0]);
        assert_eq!(base64("YQ==").unwrap(), b"a");
        assert_eq!(base64("YWI=").unwrap(), b"ab");
        assert_eq!(base64("YWJj").unwrap(), b"abc");
        assert_eq!(base64("w6k=").unwrap(), "é".as_bytes());
        assert!(base64("a$b=").is_none());
    }

    #[test]
    fn the_byte_alphabet_is_gpt2s() {
        assert_eq!(show(b"a"), "a");
        assert_eq!(show(b" "), "\u{120}");
        assert_eq!(show(b"\n"), "\u{10a}");
        assert_eq!(show(&[0]), "\u{100}");
        assert_eq!(show(&[0xA1]), "\u{a1}");
        assert_eq!(show(&[0xAD]), "\u{143}");
        assert_eq!(show("é".as_bytes()), "Ã©");
    }

    /// Every split into two ranked halves is a merge, ordered by the halves'
    /// ranks and then by the merged token's rank.
    #[test]
    fn merges_follow_the_ranks() {
        let ranked = |bytes: &[Vec<u8>]| -> Result<Vec<String>, CucaError> {
            let ranks: HashMap<&[u8], usize> = bytes
                .iter()
                .enumerate()
                .map(|(i, b)| (b.as_slice(), i))
                .collect();
            merges(&ranks, bytes)
        };
        let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        bytes.push(b"ab".to_vec()); // 256
        bytes.push(b"bc".to_vec()); // 257
        bytes.push(b"abc".to_vec()); // 258: a+bc and ab+c
        // a+bc has ranks (97, 257), ab+c has (256, 99): the byte half sorts first.
        assert_eq!(ranked(&bytes).unwrap(), ["a b", "b c", "a bc", "ab c"]);
        // A token with no ranked split is an error: x|yz!, xy|z!, xyz|! each
        // have an unranked half.
        bytes.push(b"xyz!".to_vec());
        assert!(ranked(&bytes).is_err());
    }

    /// A synthetic `tekken.json` (256 bytes, three merges, 12 special tokens)
    /// converts to a vocabulary the engine's tokenizer loads and encodes with.
    #[test]
    fn a_tekken_json_converts_to_a_loadable_gpt2_vocabulary() {
        use crate::provider::local::convert::safetensors::tests::temp_dir;
        use crate::provider::local::tokenizer::Tokenizer;

        let b64 = |b: &[u8]| {
            const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut out = String::new();
            for chunk in b.chunks(3) {
                let n = chunk.iter().fold(0u32, |a, x| (a << 8) | u32::from(*x))
                    << (8 * (3 - chunk.len()));
                for i in 0..=chunk.len() {
                    out.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
                }
                out.extend(std::iter::repeat_n('=', 3 - chunk.len()));
            }
            out
        };
        let mut vocab: Vec<Json> = (0..=255u8)
            .map(|i| serde_json::json!({"rank": i, "token_bytes": b64(&[i]), "token_str": ""}))
            .collect();
        for (rank, word) in [(256, "he"), (257, "ll"), (258, "hell")] {
            vocab.push(serde_json::json!({"rank": rank, "token_bytes": b64(word.as_bytes()), "token_str": word}));
        }
        let names = [
            "<unk>",
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "[TOOL_CALLS]",
            "[IMG]",
            "<pad>",
        ];
        let special: Vec<Json> = names
            .iter()
            .enumerate()
            .map(|(rank, s)| serde_json::json!({"rank": rank, "token_str": s, "is_control": true}))
            .collect();
        let json = serde_json::json!({
            "config": {"pattern": "\\s+|\\S+", "num_vocab_tokens": 259,
                "default_vocab_size": 14 + 259, "default_num_special_tokens": 14, "version": "v11"},
            "vocab": vocab, "special_tokens": special,
        });
        let dir = temp_dir("tekken-synth");
        std::fs::write(dir.join("tekken.json"), json.to_string()).unwrap();
        let meta = build(&dir, "llama").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let get = |k: &str| meta.kv.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(meta.n_vocab, 273);
        let Some(Value::Array(tokens)) = get("tokenizer.ggml.tokens") else {
            panic!("tokens");
        };
        assert_eq!(tokens.len(), 273);
        // 12 listed specials, 2 fillers, then the byte alphabet and the merged words.
        assert!(matches!(&tokens[12], Value::String(s) if s == "<SPECIAL_12>"));
        assert!(matches!(&tokens[14 + 104], Value::String(s) if s == "h"));
        assert!(matches!(&tokens[14 + 258], Value::String(s) if s == "hell"));
        let Some(Value::Array(merges)) = get("tokenizer.ggml.merges") else {
            panic!("merges");
        };
        let merges: Vec<&str> = merges
            .iter()
            .map(|m| match m {
                Value::String(s) => s.as_str(),
                _ => "",
            })
            .collect();
        // "hell" splits only as he|ll: "h|ell" and "hel|l" have an unranked half.
        assert_eq!(merges, ["h e", "l l", "he ll"]);
        assert!(matches!(
            get("tokenizer.ggml.padding_token_id"),
            Some(Value::U32(11))
        ));
        assert!(matches!(
            get("tokenizer.ggml.add_eos_token"),
            Some(Value::Bool(false))
        ));
        assert!(matches!(get("llama.vocab_size"), Some(Value::U32(273))));
        let md: HashMap<String, Value> = meta.kv.iter().cloned().collect();
        let tokenizer = Tokenizer::from_gguf(&md).expect("the converted vocabulary loads");
        assert_eq!(tokenizer.encode("hell"), [14 + 258]);
    }

    #[test]
    fn templates_follow_the_version() {
        assert!(community_template("v11").unwrap().contains("[TOOL_CALLS]"));
        assert!(community_template("v13").unwrap().contains("[ARGS]"));
        assert_eq!(community_template("v7").unwrap(), "mistral-v7-tekken");
        for v in ["v1", "v3"] {
            let e = community_template(v).unwrap_err().to_string();
            assert!(e.contains("built into"), "{e}");
        }
        assert!(community_template("v15").is_err());
    }
}
