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
//!   Small 3.2 one, v13 the Devstral Small 2507 one. For v1, v3 and v7 llama.cpp
//!   writes the *name* of a template built into its C++ code
//!   (`mistral-v7-tekken`), which this engine cannot render, and any other
//!   version it refuses; both are an error here.
//!
//! The newer `tekken-num3` pre-tokenizer (right-aligned digit groups of three,
//! which no model this crate supports uses) is refused by name.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file::Value;
use serde_json::Value as Json;

use super::vocab::VocabMeta;
use crate::error::CucaError;

fn cfg_err(msg: impl Into<String>) -> CucaError {
    CucaError::Config(msg.into())
}

/// `gguf.TokenType.NORMAL` and `CONTROL`.
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;

/// llama.cpp's community template for tokenizer version v11 (Mistral Small
/// 3.2), `models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja`.
const TEMPLATE_V11: &str = include_str!("templates/Mistral-Small-3.2-24B-Instruct-2506.jinja");
/// ... and for v13 (Devstral Small 2507),
/// `models/templates/unsloth-mistral-Devstral-Small-2507.jinja`.
const TEMPLATE_V13: &str = include_str!("templates/unsloth-mistral-Devstral-Small-2507.jinja");

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

/// GPT-2's `bytes_to_unicode`: printable bytes map to themselves, the rest to
/// `U+0100` upward in byte order.
fn byte_encoder() -> [char; 256] {
    let printable = |b: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b)
    };
    let mut table = ['\0'; 256];
    let mut n = 0;
    for b in 0..256u32 {
        let c = if printable(b) {
            b
        } else {
            n += 1;
            255 + n
        };
        table[b as usize] = char::from_u32(c).unwrap_or('\u{fffd}');
    }
    table
}

/// `MistralVocab.token_bytes_to_string`.
fn show(bytes: &[u8], table: &[char; 256]) -> String {
    bytes.iter().map(|b| table[usize::from(*b)]).collect()
}

/// The text of the bundled community chat template for `version`.
fn community_template(version: &str) -> Result<&'static str, CucaError> {
    match version {
        "v11" => Ok(TEMPLATE_V11),
        "v13" => Ok(TEMPLATE_V13),
        "v1" | "v3" | "v7" => Err(cfg_err(format!(
            "tekken.json version {version}: llama.cpp names a template built into its C++ code for it, which this engine cannot render; \
             convert a checkpoint that ships a tokenizer.json and a chat template instead"
        ))),
        other => Err(cfg_err(format!(
            "tekken.json version {other}: no chat template is known for it (llama.cpp refuses it too)"
        ))),
    }
}

/// The merges of `MistralVocab.extract_vocab_merges_from_model`, as the
/// `left right` strings gguf stores.
fn merges(
    ranks: &HashMap<Vec<u8>, usize>,
    by_rank: &[&[u8]],
    table: &[char; 256],
) -> Result<Vec<String>, CucaError> {
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
                .map(|(l, r)| format!("{} {}", show(l, table), show(r, table))),
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

    // Special tokens: the file's list (v11 and later must have one), then
    // `<SPECIAL_{i}>` fillers up to `default_num_special_tokens`.
    let listed = json
        .get("special_tokens")
        .and_then(Json::as_array)
        .ok_or_else(|| cfg_err("tekken.json has no `special_tokens`"))?;
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
    let mut ranks: HashMap<Vec<u8>, usize> = HashMap::with_capacity(inner);
    for (i, b) in bytes.iter().enumerate() {
        ranks.insert(b.clone(), i);
    }
    if ranks.len() != inner {
        return Err(cfg_err("tekken.json ranked tokens are not unique"));
    }

    let table = byte_encoder();
    let by_rank: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let merges = merges(&ranks, &by_rank, &table)?;

    let id = |name: &str| -> Result<u32, CucaError> {
        special
            .iter()
            .position(|s| s == name)
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| cfg_err(format!("tekken.json has no `{name}` special token")))
    };
    let tokens: Vec<String> = special
        .iter()
        .cloned()
        .chain(bytes.iter().map(|b| show(b, &table)))
        .collect();
    let mut types = vec![CONTROL; n_special];
    types.resize(vocab_size, NORMAL);
    let strings = |v: Vec<String>| Value::Array(v.into_iter().map(Value::String).collect());
    let kv = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::String("tekken".into()),
        ),
        ("tokenizer.ggml.merges".to_string(), strings(merges)),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            Value::U32(id("<s>")?),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            Value::U32(id("</s>")?),
        ),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            Value::U32(id("<unk>")?),
        ),
        (
            "tokenizer.ggml.padding_token_id".to_string(),
            Value::U32(id("<pad>")?),
        ),
        ("tokenizer.ggml.tokens".to_string(), strings(tokens)),
        (
            "tokenizer.ggml.scores".to_string(),
            Value::Array(vec![Value::F32(0.0); vocab_size]),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::Array(types.into_iter().map(Value::I32).collect()),
        ),
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
        (
            "tokenizer.chat_template".to_string(),
            Value::String(template.to_string()),
        ),
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
        let t = byte_encoder();
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], '\u{120}');
        assert_eq!(t[b'\n' as usize], '\u{10a}');
        assert_eq!(t[0], '\u{100}');
        assert_eq!(t[0xA1], '\u{a1}');
        assert_eq!(t[0xAD], '\u{143}');
        assert_eq!(show("é".as_bytes(), &t), "Ã©");
    }

    /// Every split into two ranked halves is a merge, ordered by the halves'
    /// ranks and then by the merged token's rank.
    #[test]
    fn merges_follow_the_ranks() {
        let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        bytes.push(b"ab".to_vec()); // 256
        bytes.push(b"bc".to_vec()); // 257
        bytes.push(b"abc".to_vec()); // 258: a+bc and ab+c
        let ranks: HashMap<Vec<u8>, usize> = bytes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let by_rank: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
        let m = merges(&ranks, &by_rank, &byte_encoder()).unwrap();
        // a+bc has ranks (97, 257), ab+c has (256, 99): the byte half sorts first.
        assert_eq!(m, ["a b", "b c", "a bc", "ab c"]);
        // A token with no ranked split is an error.
        let mut broken = bytes.clone();
        broken.push(b"xyz!".to_vec());
        let ranks: HashMap<Vec<u8>, usize> = broken
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let by_rank: Vec<&[u8]> = broken.iter().map(Vec::as_slice).collect();
        // "x" + "yz!" has no ranked right half but "xy"? none: x|yz!, xy|z!, xyz|!
        // only x, y, z, ! exist, so no split has both halves ranked.
        assert!(merges(&ranks, &by_rank, &byte_encoder()).is_err());
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
        for v in ["v1", "v3", "v7"] {
            let e = community_template(v).unwrap_err().to_string();
            assert!(e.contains("built into"), "{e}");
        }
        assert!(community_template("v15").is_err());
    }
}
