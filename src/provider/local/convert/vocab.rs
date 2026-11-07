//! Tokenizer metadata from the checkpoint's tokenizer files.
//!
//! Ports the pinned llama.cpp converter's vocabulary paths: the byte-level
//! BPE one (`_set_vocab_gpt2`, `tokenizer.ggml.model = gpt2`), the
//! SentencePiece ones (`_set_vocab_sentencepiece`),
//! Gemma 4's (`tokenizer.ggml.model = gemma4`) and gguf-py's `SpecialVocab`
//! (special token ids, BOS/EOS flags, merges, chat template). The BPE
//! pre-tokenizer is identified from `tokenizer.json`'s structure and then
//! fingerprinted with llama.cpp's probe-text hash table, using this engine's
//! own tokenizer, so a `pre` the engine cannot run is refused up front.

use std::collections::{BTreeMap, HashMap};

use candle_core::quantized::gguf_file::Value;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};

use super::arch::ConvArch;
use super::chktxt::{CHKTXT, PRE_BY_HASH};
use super::config::Hp;
use super::hf::CheckpointDir;
use super::spm::{self, PieceType};
use crate::error::CucaError;
use crate::provider::local::tokenizer::Tokenizer;

fn cfg_err(msg: impl Into<String>) -> CucaError {
    CucaError::Config(msg.into())
}

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;
const BYTE: i32 = 6;

/// `U+2581`, SentencePiece's space.
const SPACE_MARK: char = '\u{2581}';

/// The vocabulary metadata and the size the token embedding is trimmed to.
#[derive(Debug)]
pub(crate) struct VocabMeta {
    pub(crate) kv: Vec<(String, Value)>,
    /// Real vocabulary size (Gemma 3 and 4 drop embedding rows past it).
    pub(crate) n_vocab: usize,
}

fn string(s: impl Into<String>) -> Value {
    Value::String(s.into())
}

fn strings<I: IntoIterator<Item = String>>(it: I) -> Value {
    Value::Array(it.into_iter().map(Value::String).collect())
}

fn token_types(types: &[i32]) -> Value {
    Value::Array(types.iter().map(|t| Value::I32(*t)).collect())
}

/// `does_token_look_special`.
fn looks_special(t: &str) -> bool {
    matches!(t, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<\u{ff5c}") && t.ends_with("\u{ff5c}>"))
        || (t.starts_with("<unused") && t.ends_with('>'))
}

/// A token given as a string or as `{"content": ...}`.
fn token_str(v: &Json) -> Option<String> {
    match v {
        Json::String(s) => Some(s.clone()),
        Json::Object(o) => o.get("content").and_then(Json::as_str).map(str::to_string),
        _ => None,
    }
}

fn truthy(s: &Option<String>) -> bool {
    s.as_deref().is_some_and(|s| !s.is_empty())
}

/// gguf-py's `SpecialVocab`.
#[derive(Default)]
struct Special {
    merges: Vec<String>,
    add: Vec<(String, bool)>,
    ids: Vec<(String, u32)>,
    chat_template: Option<Json>,
    nfc: Option<bool>,
    lowercase: Option<bool>,
    strip_accents: Option<bool>,
    clean_spaces: Option<bool>,
    metaspace: Option<String>,
    types: Vec<String>,
    n_vocab: Option<usize>,
}

impl Special {
    fn set_add(&mut self, typ: &str, v: bool, overwrite: bool) {
        match self.add.iter_mut().find(|(t, _)| t == typ) {
            Some(slot) if overwrite => slot.1 = v,
            Some(_) => {}
            None => self.add.push((typ.to_string(), v)),
        }
    }

    fn set_id(&mut self, typ: &str, id: i64) -> Result<(), CucaError> {
        if id < 0 {
            return Err(cfg_err(format!(
                "invalid value for special token type {typ}: {id}"
            )));
        }
        if self.n_vocab.is_none_or(|n| (id as usize) < n) && !self.ids.iter().any(|(t, _)| t == typ)
        {
            self.ids.push((typ.to_string(), id as u32));
        }
        Ok(())
    }

    fn normalizer(&mut self, n: &Json) {
        match n.get("type").and_then(Json::as_str) {
            Some("Lowercase") => self.lowercase = Some(true),
            Some("NFC") => self.nfc = Some(true),
            Some("StripAccents") => self.strip_accents = Some(true),
            Some("BertNormalizer") => {
                if let Some(b) = n.get("lowercase").and_then(Json::as_bool) {
                    self.lowercase = Some(b);
                }
                if let Some(b) = n.get("strip_accents").and_then(Json::as_bool) {
                    self.strip_accents = Some(b);
                }
            }
            Some("Sequence") => {
                for sub in n
                    .get("normalizers")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                {
                    self.normalizer(sub);
                }
            }
            _ => {}
        }
    }

    fn pre_tokenizer(&mut self, p: &Json) {
        match p.get("type").and_then(Json::as_str) {
            Some("Metaspace") => {
                if let Some(s) = p.get("prepend_scheme").and_then(Json::as_str) {
                    self.metaspace = Some(s.to_string());
                }
            }
            Some("Sequence") => {
                for sub in p
                    .get("pretokenizers")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                {
                    self.pre_tokenizer(sub);
                }
            }
            _ => {}
        }
    }

    /// `SpecialVocab(path, load_merges, n_vocab=...)`.
    fn load(
        ck: &CheckpointDir,
        load_merges: bool,
        types: Option<&[&str]>,
        n_vocab: Option<usize>,
    ) -> Result<Self, CucaError> {
        let mut sv = Self {
            types: types
                .unwrap_or(&["bos", "eos", "unk", "sep", "pad", "cls", "mask"])
                .iter()
                .map(|s| s.to_string())
                .collect(),
            n_vocab,
            ..Self::default()
        };
        sv.read_tokenizer_json(ck, load_merges)?;
        if let Some(cfg) = ck.config.as_object() {
            for typ in sv.types.clone() {
                let key = format!("{typ}_token_id");
                let id = cfg
                    .get(&key)
                    .filter(|v| !v.is_null())
                    .or_else(|| cfg.get("text_config")?.get(&key))
                    .and_then(Json::as_i64);
                if let Some(id) = id {
                    sv.set_id(&typ, id)?;
                }
            }
        }
        // generation_config.json names the BOS and EOS ids of checkpoints
        // whose config.json does not.
        if let Some(Json::Object(gen_cfg)) = &ck.generation_config {
            for typ in ["bos", "eos"] {
                if let Some(id) = gen_cfg
                    .get(&format!("{typ}_token_id"))
                    .and_then(Json::as_i64)
                {
                    sv.set_id(typ, id)?;
                }
            }
        }
        Ok(sv)
    }

    fn read_tokenizer_json(
        &mut self,
        ck: &CheckpointDir,
        load_merges: bool,
    ) -> Result<(), CucaError> {
        let tok = ck.tokenizer_json.as_ref();
        let mut cfg: Option<Map<String, Json>> = ck
            .tokenizer_config
            .as_ref()
            .and_then(Json::as_object)
            .cloned()
            .filter(|m| !m.is_empty());
        let added: Vec<Json> = tok
            .and_then(|t| t.get("added_tokens"))
            .and_then(Json::as_array)
            .cloned()
            .unwrap_or_default();
        if let Some(t) = tok {
            if load_merges
                && let Some(Json::Array(merges)) = t.get("model").and_then(|m| m.get("merges"))
                && let Some(first) = merges.first()
            {
                if first.is_string() {
                    self.merges = merges
                        .iter()
                        .filter_map(Json::as_str)
                        .map(str::to_string)
                        .collect();
                } else if first.is_array() {
                    let fix = |s: &str| -> String {
                        s.chars()
                            .map(|c| {
                                if c == ' ' {
                                    char::from_u32(' ' as u32 + 256).unwrap_or(' ')
                                } else {
                                    c
                                }
                            })
                            .collect()
                    };
                    for pair in merges {
                        let parts: Option<Vec<&str>> = pair
                            .as_array()
                            .filter(|p| p.len() == 2)
                            .map(|p| p.iter().filter_map(Json::as_str).collect())
                            .filter(|p: &Vec<&str>| p.len() == 2);
                        let Some(p) = parts else {
                            return Err(cfg_err("unknown tokenizer merges format"));
                        };
                        self.merges.push(format!("{} {}", fix(p[0]), fix(p[1])));
                    }
                } else {
                    return Err(cfg_err("unknown tokenizer merges format"));
                }
            }
            if let Some(n) = t.get("normalizer").filter(|n| !n.is_null()) {
                self.normalizer(n);
            }
            if let Some(p) = t.get("pre_tokenizer").filter(|n| !n.is_null()) {
                self.pre_tokenizer(p);
            }
        }
        if tok.is_some() {
            let get = |cfg: &Option<Map<String, Json>>, k: &str| {
                cfg.as_ref().and_then(|c| c.get(k)).and_then(token_str)
            };
            let mut bos = get(&cfg, "bos_token");
            let mut cls = get(&cfg, "cls_token");
            let mut eos = get(&cfg, "eos_token");
            let sep = get(&cfg, "sep_token");
            if !truthy(&bos) && truthy(&cls) && cfg.is_some() {
                if let Some(c) = cfg.as_mut() {
                    c.insert(
                        "bos_token".into(),
                        Json::String(cls.clone().unwrap_or_default()),
                    );
                }
                bos = cls.clone();
            }
            if !truthy(&eos) && truthy(&sep) && cfg.is_some() {
                if let Some(c) = cfg.as_mut() {
                    c.insert(
                        "eos_token".into(),
                        Json::String(sep.clone().unwrap_or_default()),
                    );
                }
                eos = sep.clone();
            }
            if let Some(post) = tok
                .and_then(|t| t.get("post_processor"))
                .filter(|p| !p.is_null())
            {
                let processors: Vec<&Json> = match post.get("processors").and_then(Json::as_array) {
                    Some(list) => list.iter().collect(),
                    None => vec![post],
                };
                for processor in processors {
                    match processor.get("type").and_then(Json::as_str) {
                        Some("ByteLevel") => {
                            self.set_add("bos", false, false);
                            self.set_add("eos", false, false);
                        }
                        Some("RobertaProcessing") => {
                            for t in ["bos", "eos", "sep"] {
                                self.set_add(t, true, true);
                            }
                            if !truthy(&cls)
                                && let Some(c) = cfg.as_mut()
                            {
                                let v = processor
                                    .get("cls")
                                    .and_then(|c| c.get(0))
                                    .and_then(Json::as_str)
                                    .map(str::to_string)
                                    .or_else(|| bos.clone())
                                    .unwrap_or_default();
                                c.insert("cls_token".into(), Json::String(v.clone()));
                                cls = Some(v);
                            }
                        }
                        Some("TemplateProcessing") => {
                            self.template(processor, &mut cfg, &mut bos, &mut cls, &mut eos, &sep);
                        }
                        _ => {}
                    }
                }
            }
        }
        let Some(cfg) = cfg else {
            return Ok(());
        };
        let mut chat_template = cfg.get("chat_template").cloned();
        if let Some(text) = &ck.chat_template_file {
            chat_template = Some(Json::String(text.clone()));
        }
        if chat_template
            .as_ref()
            .is_none_or(|c| c.is_null() || c.is_string() || c.is_array())
        {
            self.chat_template = chat_template.filter(|c| !c.is_null());
        }
        if let Some(b) = cfg
            .get("clean_up_tokenization_spaces")
            .and_then(Json::as_bool)
        {
            self.clean_spaces = Some(b);
        }
        for typ in self.types.clone() {
            if let Some(b) = cfg.get(&format!("add_{typ}_token")).and_then(Json::as_bool) {
                self.set_add(&typ, b, false);
            }
            let Some(content) = cfg.get(&format!("{typ}_token")).and_then(token_str) else {
                continue;
            };
            let id = added
                .iter()
                .find(|a| a.get("content").and_then(Json::as_str) == Some(content.as_str()))
                .and_then(|a| a.get("id"))
                .and_then(Json::as_i64);
            if let Some(id) = id {
                self.set_id(&typ, id)?;
            }
        }
        Ok(())
    }

    /// The `TemplateProcessing` branch of `_try_load_read_tokenizer_json`.
    fn template(
        &mut self,
        processor: &Json,
        cfg: &mut Option<Map<String, Json>>,
        bos: &mut Option<String>,
        cls: &mut Option<String>,
        eos: &mut Option<String>,
        sep: &Option<String>,
    ) {
        let list = |k: &str| -> Vec<Json> {
            processor
                .get(k)
                .and_then(Json::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let single = list("single");
        let pair = list("pair");
        let special_id = |item: &Json| -> Option<String> {
            item.get("SpecialToken")
                .and_then(|s| s.get("id"))
                .and_then(Json::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let mut first: Option<String> = None;
        let mut last: Option<String> = None;
        if single.len() > 1 {
            first = special_id(&single[0]);
            if let Some(f) = &first {
                if cfg.is_none() {
                    *bos = Some(f.clone());
                } else if Some(f) != bos.as_ref() && Some(f) != cls.as_ref() {
                    if !truthy(bos) {
                        if let Some(c) = cfg.as_mut() {
                            c.insert("bos_token".into(), Json::String(f.clone()));
                        }
                        *bos = Some(f.clone());
                    }
                    if !truthy(cls) {
                        if let Some(c) = cfg.as_mut() {
                            c.insert("cls_token".into(), Json::String(f.clone()));
                        }
                        *cls = Some(f.clone());
                    }
                }
                let is_bos = Some(f) == bos.as_ref() || Some(f) == cls.as_ref();
                self.set_add("bos", is_bos, true);
            }
            last = single.last().and_then(special_id);
            if let Some(l) = &last {
                if cfg.is_none() {
                    *eos = Some(l.clone());
                } else if Some(l) != eos.as_ref() {
                    if !self.types.iter().any(|t| t == "eot") {
                        self.types.push("eot".into());
                        if let Some(c) = cfg.as_mut() {
                            c.insert(
                                "eot_token".into(),
                                Json::String(eos.clone().unwrap_or_default()),
                            );
                        }
                    } else if !self.types.iter().any(|t| t == "eom") {
                        self.types.push("eom".into());
                        if let Some(c) = cfg.as_mut() {
                            c.insert(
                                "eom_token".into(),
                                Json::String(eos.clone().unwrap_or_default()),
                            );
                        }
                    }
                    if let Some(c) = cfg.as_mut() {
                        c.insert("eos_token".into(), Json::String(l.clone()));
                    }
                    *eos = Some(l.clone());
                }
                self.set_add("eos", Some(l) == eos.as_ref(), true);
            }
        }
        if !pair.is_empty() {
            let seq_start = usize::from(first.is_some() && special_id(&pair[0]) == first);
            let trim_end = last.is_some() && pair.last().and_then(special_id) == last;
            let mut mid: Vec<Json> = pair[seq_start..].to_vec();
            if trim_end {
                mid.pop();
            }
            if !mid.is_empty() {
                let seq_id = |i: &Json| {
                    i.get("Sequence")
                        .and_then(|s| s.get("id"))
                        .and_then(Json::as_str)
                        .map(str::to_string)
                };
                let a = mid.first().and_then(seq_id);
                let b = mid.last().and_then(seq_id);
                if a.as_deref() == Some("A") && b.as_deref() == Some("B") && mid.len() > 2 {
                    let mid = &mid[1..mid.len() - 1];
                    let mut add_sep = false;
                    if let Some(entry) = mid.first().and_then(special_id)
                        && (Some(&entry) == sep.as_ref() || Some(&entry) == eos.as_ref())
                        && last.is_none()
                    {
                        add_sep = true;
                    }
                    if mid.len() == 2
                        && let Some(entry) = special_id(&mid[1])
                        && (Some(&entry) == sep.as_ref() || Some(&entry) == eos.as_ref())
                    {
                        add_sep = true;
                    }
                    self.set_add("sep", add_sep, true);
                    if add_sep
                        && !truthy(sep)
                        && let Some(c) = cfg.as_mut()
                    {
                        c.insert(
                            "sep_token".into(),
                            Json::String(eos.clone().unwrap_or_default()),
                        );
                    }
                }
            }
        }
    }

    /// The metadata `SpecialVocab.add_to_gguf` writes.
    fn write(&self, out: &mut Vec<(String, Value)>) {
        if !self.merges.is_empty() {
            out.push((
                "tokenizer.ggml.merges".into(),
                strings(self.merges.iter().cloned()),
            ));
        }
        for (typ, id) in &self.ids {
            let key = match typ.as_str() {
                "bos" => "bos",
                "eos" => "eos",
                "eot" => "eot",
                "eom" => "eom",
                "unk" => "unknown",
                "sep" => "seperator",
                "pad" => "padding",
                "mask" => "mask",
                // gguf-py has no handler for the others (cls, ...).
                _ => continue,
            };
            out.push((format!("tokenizer.ggml.{key}_token_id"), Value::U32(*id)));
        }
        for (typ, v) in &self.add {
            if matches!(typ.as_str(), "bos" | "eos" | "sep") {
                out.push((format!("tokenizer.ggml.add_{typ}_token"), Value::Bool(*v)));
            }
        }
        if let Some(template) = &self.chat_template {
            chat_template(template, out);
        }
        for (key, v) in [
            ("tokenizer.ggml.normalizer.nfc", self.nfc),
            ("tokenizer.ggml.normalizer.lowercase", self.lowercase),
            (
                "tokenizer.ggml.normalizer.strip_accents",
                self.strip_accents,
            ),
            ("tokenizer.ggml.clean_spaces", self.clean_spaces),
        ] {
            if let Some(v) = v {
                out.push((key.into(), Value::Bool(v)));
            }
        }
        if let Some(s) = &self.metaspace {
            out.push((
                "tokenizer.ggml.metaspace.prepend_scheme".into(),
                string(s.clone()),
            ));
        }
    }
}

/// `GGUFWriter.add_chat_template`.
fn chat_template(value: &Json, out: &mut Vec<(String, Value)>) {
    match value {
        Json::String(s) => out.push(("tokenizer.chat_template".into(), string(s.clone()))),
        Json::Array(choices) => {
            let mut default = None;
            let mut names = Vec::new();
            for c in choices {
                let name: String = c
                    .get("name")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .chars()
                    .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
                    .collect();
                let Some(t) = c.get("template").and_then(Json::as_str) else {
                    continue;
                };
                if name.is_empty() {
                    continue;
                }
                if name == "default" {
                    default = Some(t.to_string());
                } else {
                    out.push((format!("tokenizer.chat_template.{name}"), string(t)));
                    names.push(name);
                }
            }
            if !names.is_empty() {
                out.push(("tokenizer.chat_templates".into(), strings(names)));
            }
            if let Some(d) = default {
                out.push(("tokenizer.chat_template".into(), string(d)));
            }
        }
        _ => {}
    }
}

/// `added_tokens` of `tokenizer.json` only: `(id, content, special)`.
fn json_added_tokens(ck: &CheckpointDir) -> Vec<(usize, String, bool)> {
    ck.tokenizer_json
        .as_ref()
        .and_then(|t| t.get("added_tokens"))
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| {
            Some((
                a.get("id")?.as_u64()? as usize,
                a.get("content")?.as_str()?.to_string(),
                a.get("special").and_then(Json::as_bool).unwrap_or(false),
            ))
        })
        .collect()
}

/// The added tokens `AutoTokenizer` knows: those of `tokenizer.json`, plus
/// the entries of `tokenizer_config.json`'s `added_tokens_decoder` whose id
/// `tokenizer.json` does not use (Qwen 3.5 lists its audio and TTS tokens
/// only there).
fn added_tokens(ck: &CheckpointDir) -> Vec<(usize, String, bool)> {
    let mut out = json_added_tokens(ck);
    if let Some(Json::Object(decoder)) = ck
        .tokenizer_config
        .as_ref()
        .and_then(|c| c.get("added_tokens_decoder"))
    {
        for (id, data) in decoder {
            let (Ok(id), Some(content)) = (
                id.parse::<usize>(),
                data.get("content").and_then(Json::as_str),
            ) else {
                continue;
            };
            if out.iter().any(|(i, _, _)| *i == id) {
                continue;
            }
            let special = data.get("special").and_then(Json::as_bool).unwrap_or(false);
            out.push((id, content.to_string(), special));
        }
    }
    out
}

/// The checkpoint's `tokenizer.json`. A repository that ships its vocabulary
/// only as `tekken.json` is converted by `tekken.rs` for the Llama family
/// (llama.cpp's `_set_vocab_mistral`); another architecture is named.
fn tokenizer_json(ck: &CheckpointDir) -> Result<&Json, CucaError> {
    ck.tokenizer_json.as_ref().ok_or_else(|| {
        if ck.dir.join("tekken.json").is_file() {
            cfg_err(
                "checkpoint ships its vocabulary only as tekken.json, which the converter reads for the Llama family only (llama.cpp's `_set_vocab_mistral`); this architecture needs the repository's tokenizer.json",
            )
        } else {
            cfg_err("checkpoint has no tokenizer.json")
        }
    })
}

/// The base vocabulary of a BPE `tokenizer.json`: token by id, and the
/// number of entries.
fn bpe_vocab(ck: &CheckpointDir) -> Result<BTreeMap<usize, String>, CucaError> {
    let tok = tokenizer_json(ck)?;
    let model = tok
        .get("model")
        .ok_or_else(|| cfg_err("tokenizer.json has no model"))?;
    if model.get("type").and_then(Json::as_str) != Some("BPE") {
        return Err(cfg_err(format!(
            "tokenizer.json model type {:?} is not supported (only BPE)",
            model.get("type")
        )));
    }
    let vocab = model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or_else(|| cfg_err("tokenizer.json has no vocab"))?;
    vocab
        .iter()
        .map(|(t, id)| {
            id.as_u64()
                .map(|i| (i as usize, t.clone()))
                .ok_or_else(|| cfg_err("tokenizer.json vocab holds a non-integer id"))
        })
        .collect()
}

/// `get_vocab_base`: tokens and their types for ids `0..vocab_size`.
fn gpt2_tokens(
    ck: &CheckpointDir,
    vocab_size: usize,
) -> Result<(Vec<String>, Vec<i32>), CucaError> {
    let mut by_id = bpe_vocab(ck)?;
    let added = added_tokens(ck);
    let added_by_content: HashMap<&str, bool> =
        added.iter().map(|(_, c, s)| (c.as_str(), *s)).collect();
    for (id, content, _) in &added {
        by_id.insert(*id, content.clone());
    }
    if let Some((&max, _)) = by_id.iter().next_back()
        && max >= vocab_size
    {
        return Err(cfg_err(format!(
            "token id {max} is outside vocab_size {vocab_size}"
        )));
    }
    let mut tokens = Vec::with_capacity(vocab_size);
    let mut types = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        match by_id.get(&i) {
            None => {
                tokens.push(format!("[PAD{i}]"));
                types.push(UNUSED);
            }
            Some(t) => match added_by_content.get(t.as_str()) {
                Some(&special) => {
                    if special || looks_special(t) {
                        tokens.push(t.clone());
                        types.push(CONTROL);
                    } else {
                        tokens.push(t.replace(SPACE_MARK, " "));
                        types.push(USER_DEFINED);
                    }
                }
                None => {
                    tokens.push(t.clone());
                    types.push(NORMAL);
                }
            },
        }
    }
    Ok((tokens, types))
}

const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const GPT4O: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The `tokenizer.ggml.pre` this checkpoint's pre-tokenizer corresponds to,
/// by `tokenizer.json` structure (the generic name of its chain).
fn structural_pre(tok: &Json) -> Result<&'static str, CucaError> {
    fn flatten<'a>(p: &'a Json, out: &mut Vec<&'a Json>) {
        match p.get("type").and_then(Json::as_str) {
            Some("Sequence") => {
                for sub in p
                    .get("pretokenizers")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                {
                    flatten(sub, out);
                }
            }
            _ => out.push(p),
        }
    }
    let unsupported = |p: &Json| {
        cfg_err(format!(
            "pre-tokenizer of `tokenizer.json` is not supported: {p}"
        ))
    };
    let Some(pre) = tok.get("pre_tokenizer").filter(|p| !p.is_null()) else {
        return Err(cfg_err("tokenizer.json has no pre_tokenizer"));
    };
    let mut parts = Vec::new();
    flatten(pre, &mut parts);
    let ignore_merges = tok
        .get("model")
        .and_then(|m| m.get("ignore_merges"))
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let mut regex: Option<&str> = None;
    let mut digits = false;
    let mut byte_level_regex = false;
    for p in &parts {
        match p.get("type").and_then(Json::as_str) {
            Some("Split") => {
                if regex.is_some() {
                    return Err(unsupported(pre));
                }
                regex = p
                    .get("pattern")
                    .and_then(|x| x.get("Regex"))
                    .and_then(Json::as_str);
                if regex.is_none() {
                    return Err(unsupported(pre));
                }
            }
            Some("Digits") if p.get("individual_digits").and_then(Json::as_bool) == Some(true) => {
                digits = true;
            }
            Some("ByteLevel") => {
                byte_level_regex |= p.get("use_regex").and_then(Json::as_bool).unwrap_or(false);
            }
            _ => return Err(unsupported(pre)),
        }
    }
    match (regex, digits, byte_level_regex) {
        (Some(r), false, false) if r == LLAMA3 => Ok(if ignore_merges {
            "llama-bpe"
        } else {
            "smaug-bpe"
        }),
        (Some(r), false, false) if r == QWEN2 => Ok("qwen2"),
        (Some(r), false, false) if r == QWEN35 => Ok("qwen35"),
        (Some(r), false, false) if r == GPT4O => Ok("gpt-4o"),
        // llama.cpp's `tekken` always sets `ignore_merges`.
        (Some(r), false, false) if r == TEKKEN && ignore_merges => Ok("tekken"),
        (None, true, true) => Ok("starcoder"),
        (None, false, true) => Ok("gpt-2"),
        _ => Err(unsupported(pre)),
    }
}

/// Python's `str(list_of_ints)` hashed as llama.cpp does.
fn chkhsh(ids: &[u32]) -> String {
    let repr = format!(
        "[{}]",
        ids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    Sha256::digest(repr.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The `pre` name for a probe-text fingerprint: the name llama.cpp's table
/// gives it (`refact` for Granite, `dbrx` for OLMo 2, `pixtral` for Mistral
/// Small 4 whose structure also reads as `tekken`) when the engine
/// implements it, an error naming both when it does not; an unknown
/// fingerprint keeps the structural name.
fn pre_for_hash(structural: &'static str, hash: &str) -> Result<String, CucaError> {
    let Some(&(_, name)) = PRE_BY_HASH.iter().rev().find(|(h, _)| *h == hash) else {
        return Ok(structural.to_string());
    };
    if name == structural || crate::provider::local::tokenizer::supports_pre(name) {
        Ok(name.to_string())
    } else {
        Err(cfg_err(format!(
            "the tokenizer matches llama.cpp's `{name}` pre-tokenizer, which this engine does not implement (its pre-tokenizer structure reads as `{structural}`)"
        )))
    }
}

/// Settle the `pre` name: tokenize llama.cpp's probe text with this engine's
/// tokenizer under the structural choice and look the fingerprint up
/// ([`pre_for_hash`]), so a fingerprint llama.cpp knows beats the structure.
fn settle_pre(
    structural: &'static str,
    tokens: &[String],
    types: &[i32],
    merges: &[String],
    bos: Option<u32>,
    eos: Option<u32>,
) -> Result<String, CucaError> {
    let md: HashMap<String, Value> = HashMap::from([
        ("tokenizer.ggml.model".to_string(), string("gpt2")),
        ("tokenizer.ggml.pre".to_string(), string(structural)),
        (
            "tokenizer.ggml.tokens".to_string(),
            strings(tokens.iter().cloned()),
        ),
        ("tokenizer.ggml.token_type".to_string(), token_types(types)),
        (
            "tokenizer.ggml.merges".to_string(),
            strings(merges.iter().cloned()),
        ),
    ]);
    let tokenizer = Tokenizer::from_gguf(&md).map_err(|e| {
        cfg_err(format!(
            "the converted tokenizer does not load with pre `{structural}`: {e}"
        ))
    })?;
    let mut ids = tokenizer.encode(CHKTXT);
    if let Some(b) = bos {
        ids.insert(0, b);
    }
    if let Some(e) = eos {
        ids.push(e);
    }
    pre_for_hash(structural, &chkhsh(&ids))
}

/// The byte-level BPE path (`_set_vocab_gpt2`).
fn gpt2(ck: &CheckpointDir, hp: &Hp) -> Result<VocabMeta, CucaError> {
    let tok = tokenizer_json(ck)?;
    let base = bpe_vocab(ck)?;
    let vocab_size = match hp.uint(&["vocab_size"]) {
        Some(n) => n,
        None => base.keys().next_back().map_or(0, |m| m + 1),
    };
    let (tokens, types) = gpt2_tokens(ck, vocab_size)?;
    let structural = structural_pre(tok)?;
    let special = Special::load(ck, true, None, None)?;
    let id = |t: &str| special.ids.iter().find(|(k, _)| k == t).map(|(_, v)| *v);
    let flag = |t: &str| {
        special
            .add
            .iter()
            .find(|(k, _)| k == t)
            .is_some_and(|(_, v)| *v)
    };
    let pre = settle_pre(
        structural,
        &tokens,
        &types,
        &special.merges,
        if flag("bos") { id("bos") } else { None },
        if flag("eos") { id("eos") } else { None },
    )?;
    let n_vocab = base.len() + added_tokens(ck).len();
    let mut kv = vec![
        ("tokenizer.ggml.model".to_string(), string("gpt2")),
        ("tokenizer.ggml.pre".to_string(), string(pre)),
        ("tokenizer.ggml.tokens".to_string(), strings(tokens)),
        ("tokenizer.ggml.token_type".to_string(), token_types(&types)),
    ];
    special.write(&mut kv);
    Ok(VocabMeta { kv, n_vocab })
}

fn piece_type_value(k: PieceType) -> i32 {
    k as i32
}

/// A `tokenizer.model` read, with the token list as the converter builds it.
struct SpmVocab {
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<i32>,
}

/// `_create_vocab_sentencepiece`.
fn spm_generic(ck: &CheckpointDir, hp: &Hp) -> Result<SpmVocab, CucaError> {
    let path = ck
        .tokenizer_model
        .as_ref()
        .ok_or_else(|| cfg_err("checkpoint has no tokenizer.model"))?;
    let pieces = spm::parse_pieces(&std::fs::read(path)?)?;
    let vocab_size = hp
        .uint(&["vocab_size_per_layer_input", "vocab_size"])
        .unwrap_or(pieces.len());
    let mut v = SpmVocab {
        tokens: (0..vocab_size).map(|i| format!("[PAD{i}]")).collect(),
        scores: vec![-10000.0; vocab_size],
        types: vec![UNUSED; vocab_size],
    };
    for (i, p) in pieces.iter().enumerate() {
        if i >= vocab_size {
            break;
        }
        v.tokens[i] = p.text.clone();
        v.scores[i] = p.score;
        v.types[i] = piece_type_value(p.kind);
    }
    if let Some(Json::Object(added)) = &ck.added_tokens {
        for (key, id) in added {
            let Some(id) = id.as_u64().map(|i| i as usize).filter(|i| *i < vocab_size) else {
                continue;
            };
            v.tokens[id] = key.clone();
            v.scores[id] = -1000.0;
            v.types[id] = USER_DEFINED;
        }
    }
    if let Some(Json::Object(decoder)) = ck
        .tokenizer_config
        .as_ref()
        .and_then(|c| c.get("added_tokens_decoder"))
    {
        for (id, data) in decoder {
            let (Ok(id), Some(content)) = (
                id.parse::<usize>(),
                data.get("content").and_then(Json::as_str),
            ) else {
                continue;
            };
            if id >= vocab_size {
                continue;
            }
            let token = if data.get("special").and_then(Json::as_bool).unwrap_or(false)
                || looks_special(content)
            {
                v.types[id] = CONTROL;
                content.to_string()
            } else {
                v.types[id] = USER_DEFINED;
                content.replace(SPACE_MARK, " ")
            };
            v.scores[id] = -1000.0;
            v.tokens[id] = token;
        }
    }
    Ok(v)
}

fn spm_meta(
    ck: &CheckpointDir,
    v: SpmVocab,
    add_space_prefix: Option<bool>,
) -> Result<VocabMeta, CucaError> {
    let n = v.tokens.len();
    let special = Special::load(ck, false, None, Some(n))?;
    let mut kv = vec![
        ("tokenizer.ggml.model".to_string(), string("llama")),
        ("tokenizer.ggml.pre".to_string(), string("default")),
        ("tokenizer.ggml.tokens".to_string(), strings(v.tokens)),
        (
            "tokenizer.ggml.scores".to_string(),
            Value::Array(v.scores.into_iter().map(Value::F32).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            token_types(&v.types),
        ),
    ];
    special.write(&mut kv);
    if let Some(p) = add_space_prefix {
        kv.push(("tokenizer.ggml.add_space_prefix".into(), Value::Bool(p)));
    }
    Ok(VocabMeta { kv, n_vocab: n })
}

/// `Gemma4Model.set_vocab` (`LlamaHfVocab` over `tokenizer.json`).
fn gemma4(ck: &CheckpointDir) -> Result<VocabMeta, CucaError> {
    const VISIBLE: [&str; 7] = [
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|\"|>",
    ];
    let base = bpe_vocab(ck)?;
    let n_base = base.len();
    let added = added_tokens(ck);
    // Special ids: the tokens of tokenizer_config's special-token entries and
    // every added token flagged special.
    let mut special_texts: Vec<String> = Vec::new();
    if let Some(Json::Object(cfg)) = &ck.tokenizer_config {
        for (k, v) in cfg {
            if k.ends_with("_token")
                || k == "extra_special_tokens"
                || k == "additional_special_tokens"
            {
                match v {
                    Json::Array(a) => special_texts.extend(a.iter().filter_map(token_str)),
                    other => special_texts.extend(token_str(other)),
                }
            }
        }
    }
    let special_ids: std::collections::HashSet<usize> = added
        .iter()
        .filter(|(_, c, s)| *s || special_texts.contains(c))
        .map(|(i, _, _)| *i)
        .chain(
            base.iter()
                .filter(|(_, t)| special_texts.contains(t))
                .map(|(i, _)| *i),
        )
        .collect();
    let mut tokens = Vec::with_capacity(n_base);
    let mut types = Vec::with_capacity(n_base);
    let is_byte = |t: &str| {
        t.len() == 6
            && t.starts_with("<0x")
            && t.ends_with('>')
            && t[3..5].chars().all(|c| c.is_ascii_hexdigit())
    };
    let by_id: BTreeMap<usize, String> = base.clone();
    for id in 0..n_base {
        let t = by_id
            .get(&id)
            .ok_or_else(|| cfg_err(format!("tokenizer.json vocabulary has no token {id}")))?;
        tokens.push(t.clone());
        types.push(if VISIBLE.contains(&t.as_str()) {
            USER_DEFINED
        } else if is_byte(t) {
            BYTE
        } else if special_ids.contains(&id) {
            CONTROL
        } else {
            NORMAL
        });
    }
    // Added tokens beyond the base vocabulary follow, sorted by id.
    let mut extra: Vec<&(usize, String, bool)> =
        added.iter().filter(|(i, _, _)| *i >= n_base).collect();
    extra.sort_by_key(|(i, _, _)| *i);
    for (_, content, special) in extra {
        tokens.push(content.clone());
        types.push(if VISIBLE.contains(&content.as_str()) {
            USER_DEFINED
        } else if *special {
            CONTROL
        } else {
            USER_DEFINED
        });
    }
    let n = tokens.len();
    let special = Special::load(ck, true, None, None)?;
    let mut kv = vec![
        ("tokenizer.ggml.model".to_string(), string("gemma4")),
        ("tokenizer.ggml.tokens".to_string(), strings(tokens)),
        (
            "tokenizer.ggml.scores".to_string(),
            Value::Array(vec![Value::F32(-1000.0); n]),
        ),
        ("tokenizer.ggml.token_type".to_string(), token_types(&types)),
    ];
    special.write(&mut kv);
    kv.push(("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false)));
    kv.retain(|(k, _)| k != "tokenizer.ggml.add_bos_token");
    kv.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
    Ok(VocabMeta {
        kv,
        n_vocab: base.len() + added.len(),
    })
}

/// The id of the vocabulary token `text` (the converter's
/// `tok.convert_tokens_to_ids`).
fn token_id(ck: &CheckpointDir, text: &str) -> Result<u32, CucaError> {
    let added = added_tokens(ck)
        .into_iter()
        .find(|(_, content, _)| content == text)
        .map(|(id, _, _)| id);
    let base = || {
        bpe_vocab(ck)
            .ok()?
            .into_iter()
            .find(|(_, tok)| tok == text)
            .map(|(id, _)| id)
    };
    added
        .or_else(base)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| cfg_err(format!("the tokenizer has no token `{text}`")))
}

fn config_str<'a>(ck: &'a CheckpointDir, key: &str) -> Option<&'a Json> {
    ck.tokenizer_config.as_ref()?.get(key)
}

/// The tokenizer metadata of a checkpoint; `None` for the Gemma 4 drafter,
/// which carries no vocabulary (it shares the target model's).
pub(crate) fn build(
    arch: ConvArch,
    ck: &CheckpointDir,
    hp: &Hp,
) -> Result<Option<VocabMeta>, CucaError> {
    let meta = match arch {
        ConvArch::Gemma4Assistant | ConvArch::Mistral4Eagle => return Ok(None),
        ConvArch::Gemma4 => gemma4(ck)?,
        ConvArch::Gemma3 => {
            if ck.tokenizer_model.is_some() {
                spm_meta(ck, spm_generic(ck, hp)?, Some(false))?
            } else {
                gpt2(ck, hp)?
            }
        }
        ConvArch::Ernie4_5 => {
            let prefix = config_str(ck, "add_prefix_space").and_then(Json::as_bool);
            spm_meta(ck, spm_generic(ck, hp)?, prefix)?
        }
        ConvArch::Phi3 => {
            let gpt2_class =
                config_str(ck, "tokenizer_class").and_then(Json::as_str) == Some("GPT2Tokenizer");
            if !gpt2_class {
                return Err(cfg_err(
                    "Phi 3 SentencePiece checkpoints are not supported (only Phi 4 style GPT2Tokenizer ones)",
                ));
            }
            gpt2(ck, hp)?
        }
        ConvArch::Llama | ConvArch::Smollm3 | ConvArch::Granite | ConvArch::Mistral3 => {
            // `LlamaModel.set_vocab`: a directory with `tekken.json` and no
            // `tokenizer.json` goes through `_set_vocab_mistral`.
            if ck.tokenizer_json.is_none() && ck.dir.join("tekken.json").is_file() {
                return Ok(Some(super::tekken::build(&ck.dir, arch.gguf_name())?));
            }
            let prefix = config_str(ck, "add_prefix_space").and_then(Json::as_bool);
            if ck.tokenizer_model.is_some() {
                spm_meta(ck, spm_generic(ck, hp)?, prefix)?
            } else {
                let mut m = gpt2(ck, hp)?;
                if let Some(p) = prefix {
                    m.kv.push(("tokenizer.ggml.add_space_prefix".into(), Value::Bool(p)));
                }
                m
            }
        }
        ConvArch::Qwen2 | ConvArch::Qwen3 | ConvArch::Qwen3Moe => {
            if ck.tokenizer_model.is_some() {
                spm_meta(ck, spm_generic(ck, hp)?, None)?
            } else {
                gpt2(ck, hp)?
            }
        }
        ConvArch::Qwen35
        | ConvArch::Qwen35Moe
        | ConvArch::Olmo2
        | ConvArch::Olmoe
        | ConvArch::Mistral4 => gpt2(ck, hp)?,
        // `MuseGlimmerModel.set_vocab`: the GPT-2 vocabulary plus `<|eot|>`
        // as the end-of-turn token (`<|eom|>` ends a message, not the turn).
        ConvArch::MuseGlimmer => {
            let mut m = gpt2(ck, hp)?;
            let eot = token_id(ck, "<|eot|>")?;
            m.kv.retain(|(k, _)| k != "tokenizer.ggml.eot_token_id");
            m.kv.push(("tokenizer.ggml.eot_token_id".into(), Value::U32(eot)));
            m
        }
    };
    if !meta.kv.iter().any(|(k, _)| k == "tokenizer.chat_template") {
        return Err(cfg_err("checkpoint has no chat template"));
    }
    Ok(Some(meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tokenizer_structures_map_to_pres() {
        let split = |regex: &str| serde_json::json!({"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": false});
        let byte = |rx: bool| serde_json::json!({"type": "ByteLevel", "use_regex": rx});
        let tok = |pre: Json, ignore: bool| serde_json::json!({"pre_tokenizer": pre, "model": {"type": "BPE", "ignore_merges": ignore}});
        let seq =
            |parts: Vec<Json>| serde_json::json!({"type": "Sequence", "pretokenizers": parts});
        assert_eq!(
            structural_pre(&tok(seq(vec![split(LLAMA3), byte(false)]), true)).unwrap(),
            "llama-bpe"
        );
        assert_eq!(
            structural_pre(&tok(seq(vec![split(LLAMA3), byte(false)]), false)).unwrap(),
            "smaug-bpe"
        );
        assert_eq!(
            structural_pre(&tok(seq(vec![split(QWEN2), byte(false)]), false)).unwrap(),
            "qwen2"
        );
        assert_eq!(
            structural_pre(&tok(seq(vec![split(QWEN35), byte(false)]), false)).unwrap(),
            "qwen35"
        );
        assert_eq!(
            structural_pre(&tok(seq(vec![split(GPT4O), byte(false)]), false)).unwrap(),
            "gpt-4o"
        );
        assert_eq!(
            structural_pre(&tok(seq(vec![split(TEKKEN), byte(false)]), true)).unwrap(),
            "tekken"
        );
        // Tekken always merges whole vocabulary words first.
        assert!(structural_pre(&tok(seq(vec![split(TEKKEN), byte(false)]), false)).is_err());
        assert_eq!(structural_pre(&tok(byte(true), false)).unwrap(), "gpt-2");
        let digits = serde_json::json!({"type": "Digits", "individual_digits": true});
        assert_eq!(
            structural_pre(&tok(seq(vec![digits, byte(true)]), false)).unwrap(),
            "starcoder"
        );
        let err = structural_pre(&tok(seq(vec![split("\\w+"), byte(false)]), false))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not supported"), "{err}");
    }

    #[test]
    fn probe_text_hash_table_is_loaded() {
        assert_eq!(PRE_BY_HASH.len(), 105);
        // llama.cpp's `tekken` and `pixtral` fingerprints resolve to names the
        // engine implements.
        for (hash, name) in [
            (
                "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e",
                "tekken",
            ),
            (
                "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3",
                "pixtral",
            ),
        ] {
            assert!(PRE_BY_HASH.contains(&(hash, name)), "{name}");
            assert!(
                crate::provider::local::tokenizer::supports_pre(name),
                "{name}"
            );
        }
        assert!(PRE_BY_HASH.iter().all(|(h, _)| h.len() == 64));
        assert_eq!(chkhsh(&[1, 2, 3]).len(), 64);
        assert!(CHKTXT.contains("កាន់តែពិសេសអាច"));
    }

    #[test]
    fn a_vocabulary_shipped_only_as_tekken_json_is_refused_by_name() {
        use crate::provider::local::convert::safetensors::tests::{
            f32_bytes, temp_dir, write_file,
        };
        let dir = temp_dir("tekken-only");
        write_file(
            &dir.join("model.safetensors"),
            &[("w", "F32", &[1], f32_bytes(&[1.0]))],
        );
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        let hp = Hp::new(&serde_json::json!({})).unwrap();
        let open = || CheckpointDir::open(&dir).unwrap();
        let plain = gpt2(&open(), &hp).unwrap_err().to_string();
        assert!(plain.contains("no tokenizer.json"), "{plain}");
        std::fs::write(dir.join("tekken.json"), "{}").unwrap();
        let err = gpt2(&open(), &hp).unwrap_err().to_string();
        assert!(err.contains("only as tekken.json"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_known_fingerprint_beats_the_structural_pre() {
        // Mistral Small 4's tokenizer.json structure reads as `tekken`; its
        // fingerprint is llama.cpp's `pixtral`, which is what gets written.
        assert_eq!(
            pre_for_hash(
                "tekken",
                "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3"
            )
            .unwrap(),
            "pixtral"
        );
        // An unknown fingerprint keeps the structure; a known one the engine
        // lacks is refused.
        assert_eq!(pre_for_hash("tekken", &"0".repeat(64)).unwrap(), "tekken");
        let (hash, name) = PRE_BY_HASH
            .iter()
            .find(|(_, n)| !crate::provider::local::tokenizer::supports_pre(n))
            .expect("llama.cpp knows pre-tokenizers the engine lacks");
        let err = pre_for_hash("llama-bpe", hash).unwrap_err().to_string();
        assert!(
            err.contains(&format!("`{name}`")) && err.contains("llama-bpe"),
            "{err}"
        );
    }

    #[test]
    fn template_processing_sets_bos_and_special_ids() {
        let ck = CheckpointDir {
            dir: std::path::PathBuf::new(),
            config: serde_json::json!({"eos_token_id": [1, 2]}),
            generation_config: None,
            tokenizer_json: Some(serde_json::json!({
                "added_tokens": [{"id": 5, "content": "<s>", "special": true}, {"id": 6, "content": "</s>", "special": true}],
                "post_processor": {"type": "TemplateProcessing",
                    "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                    "pair": []}
            })),
            tokenizer_config: Some(serde_json::json!({
                "bos_token": "<s>", "eos_token": "</s>", "add_bos_token": true,
                "chat_template": [{"name": "default", "template": "T"}, {"name": "tool use", "template": "U"}]
            })),
            tokenizer_model: None,
            added_tokens: None,
            chat_template_file: None,
            shards: Vec::new(),
        };
        let sv = Special::load(&ck, false, None, None).unwrap();
        assert_eq!(sv.ids, [("bos".to_string(), 5), ("eos".to_string(), 6)]);
        assert_eq!(sv.add, [("bos".to_string(), true)]);
        let mut kv = Vec::new();
        sv.write(&mut kv);
        let keys: Vec<&str> = kv.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"tokenizer.chat_template"));
        assert!(keys.contains(&"tokenizer.chat_template.tool_use"));
        assert!(keys.contains(&"tokenizer.chat_templates"));
        assert!(keys.contains(&"tokenizer.ggml.add_bos_token"));
    }

    #[test]
    fn unused_ids_are_padded_and_added_tokens_typed() {
        let ck = CheckpointDir {
            dir: std::path::PathBuf::new(),
            config: serde_json::json!({}),
            generation_config: None,
            tokenizer_json: Some(serde_json::json!({
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<|im_end|>": 3, "\u{2581}x": 4}},
                "added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}, {"id": 4, "content": "\u{2581}x", "special": false}]
            })),
            tokenizer_config: None,
            tokenizer_model: None,
            added_tokens: None,
            chat_template_file: None,
            shards: Vec::new(),
        };
        let (tokens, types) = gpt2_tokens(&ck, 6).unwrap();
        assert_eq!(tokens, ["a", "b", "[PAD2]", "<|im_end|>", " x", "[PAD5]"]);
        assert_eq!(types, [1, 1, 5, 3, 4, 5]);
        assert!(gpt2_tokens(&ck, 4).is_err());
    }
}
