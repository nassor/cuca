//! In-process tokenizers built from GGUF metadata, following llama.cpp's
//! `llama-vocab.cpp`.
//!
//! [`Tokenizer::from_gguf`] reads `tokenizer.ggml.model` and selects one of
//! three schemes:
//!
//! - `llama`: SentencePiece ([`spm`]): score-ordered bigram merging over UTF-8
//!   characters, `<0xHH>` byte fallback, an optional leading space
//!   (`tokenizer.ggml.add_space_prefix`);
//! - `gpt2`: byte-level BPE ([`bpe`], [`pretok`]): the text is split into words
//!   by the regex chain `tokenizer.ggml.pre` names (hand-written linear
//!   splitters), each word's bytes are mapped to the GPT-2 alphabet and merged by
//!   rank, with `ignore_merges` for the llama3 family;
//! - `gemma4` (also `gpt2` with `pre == "gemma4"`): space-escaped BPE that
//!   replaces spaces with `▁`, cuts newline runs so a merge never spans a newline
//!   boundary, and falls back to `<0xHH>` byte tokens.
//!
//! Any other model, or a `gpt2` pre-tokenizer this module cannot reproduce, is
//! an error that names the value.
//!
//! [`Tokenizer::encode`] first splits special tokens (control, user-defined and
//! unknown types) out of the text longest-first, exactly as llama.cpp's
//! `tokenizer_st_partition`, then runs the scheme over each remaining fragment.
//! `encode` never adds BOS/EOS; [`Tokenizer::encode_prompt`] adds them per the
//! model's `add_bos`/`add_eos` flags unless the text already starts/ends with
//! them. Special tokens carry llama.cpp's `lstrip`/`rstrip` attributes: when one
//! is cut out of the text, `rstrip` drops the whitespace that follows it and
//! `lstrip` the whitespace that precedes it. The GGUF has no per-token field for
//! them, so they are derived from the metadata as llama.cpp does: Phi-3 family
//! SPM checkpoints (see `Metadata::is_phi3`) get `rstrip` on every special token
//! except `<unk>`, `<s>` and `<|endoftext|>`, matching the HuggingFace
//! `tokenizer.json` added-token flags (`<|user|>\nHi<|end|>\n<|assistant|>\n`
//! tokenizes without the newlines); no other family gets either flag.
//!
//! Memory: the vocabulary is built once into one compact string arena plus an
//! open-addressing table ([`vocab`]) and the merge map holds 16 bytes per rule;
//! all of it is read-only afterwards and shared by cloning the `Arc` the engine
//! keeps. Per `encode` call the transient state is the output vector, one 8 KiB
//! bit set for the special-token prefilter, the fragment list, and one buffer
//! set shared by all fragments (the longest word's symbol list and heap, the
//! word text and two range vectors for the pre-tokenizer): all bounded by the
//! input length and freed on return.

mod bpe;
mod pretok;
mod spm;
mod vocab;

#[cfg(test)]
pub(crate) mod golden;

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use candle_core::quantized::gguf_file::{self, Value};

use self::bpe::{Merges, Scratch, Unknown, byte_to_char, char_to_byte};
use self::pretok::{Pre, Pretokenizer};
use self::spm::Spm;
use self::vocab::{
    TYPE_BYTE, TYPE_CONTROL, TYPE_NORMAL, TYPE_UNDEFINED, TYPE_UNKNOWN, TYPE_UNUSED,
    TYPE_USER_DEFINED, Vocab, parse_byte_token,
};
use crate::error::CucaError;
use crate::types::ProviderEndpoint;

/// The word-boundary marker substituted for ASCII space.
const SPACE_MARK: char = '\u{2581}';

/// Build a local-provider error.
fn local_err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

/// How a BPE vocabulary turns text into words.
#[derive(Debug, Clone)]
enum Words {
    /// GPT-2 byte-level: regex chain, then bytes mapped to the GPT-2 alphabet.
    Byte(Pretokenizer),
    /// Space-escaped (`gemma4`): `▁` for spaces, newline runs cut apart.
    Escaped,
}

/// The scheme-specific encoder.
#[derive(Clone)]
enum Engine {
    /// SentencePiece.
    Spm(Spm),
    /// Merge-rank BPE.
    Bpe {
        /// Merge rules.
        merges: Merges,
        /// Word splitting scheme.
        words: Words,
    },
}

/// Buffers reused across the fragments of one `encode` call.
#[derive(Default)]
struct Buffers {
    /// Word byte ranges the pre-tokenizer produced.
    ranges: Vec<Range<usize>>,
    /// Second range buffer of the pre-tokenizer's stage chain.
    stage_ranges: Vec<Range<usize>>,
    /// Symbol list and heap of the BPE merger.
    scratch: Scratch,
    /// The text being cut into words: the current GPT-2-alphabet word (byte-level
    /// BPE) or the whole space-escaped fragment (`gemma4`).
    text: String,
}

/// A text fragment during special-token partitioning.
enum Fragment<'a> {
    /// Text still to be encoded.
    Raw(&'a str),
    /// An already-resolved special token id.
    Special(u32),
}

/// Special token ids and BOS/EOS flags, before the tokenizer is assembled.
#[derive(Debug, Clone, Copy, Default)]
struct Config {
    /// Beginning-of-sequence id.
    bos: Option<u32>,
    /// End-of-sequence id.
    eos: Option<u32>,
    /// Whether prompts start with BOS.
    add_bos: bool,
    /// Whether prompts end with EOS.
    add_eos: bool,
}

/// A special token with the whitespace-stripping attributes llama.cpp calls
/// `LSTRIP` and `RSTRIP`: when the token is cut out of text, whitespace at the
/// end of the text before it (`lstrip`) or at the start of the text after it
/// (`rstrip`) is dropped.
#[derive(Debug, Clone, Copy)]
struct Special {
    /// Token id.
    id: u32,
    /// Strip trailing whitespace of the preceding text.
    lstrip: bool,
    /// Strip leading whitespace of the following text.
    rstrip: bool,
}

/// C `isspace` in the C locale: space, `\t`, `\n`, `\v`, `\f`, `\r`.
fn is_c_space(c: char) -> bool {
    matches!(c, ' ' | '\t'..='\r')
}

/// A tokenizer built from GGUF metadata.
#[derive(Clone)]
pub(crate) struct Tokenizer {
    /// Token texts, types and lookup.
    vocab: Vocab,
    /// The encoding scheme.
    engine: Engine,
    /// Special tokens with their strip attributes, longest text first.
    specials: Vec<Special>,
    /// Ids and BOS/EOS flags.
    config: Config,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scheme = match &self.engine {
            Engine::Spm(_) => "spm",
            Engine::Bpe {
                words: Words::Byte(_),
                ..
            } => "byte-bpe",
            Engine::Bpe {
                words: Words::Escaped,
                ..
            } => "space-escaped-bpe",
        };
        f.debug_struct("Tokenizer")
            .field("scheme", &scheme)
            .field("vocab_size", &self.vocab.len())
            .field("specials", &self.specials.len())
            .field("config", &self.config)
            .finish()
    }
}

/// A GGUF integer value, whatever its width.
fn value_int(value: &Value) -> Option<i64> {
    match value {
        Value::U8(n) => Some(i64::from(*n)),
        Value::I8(n) => Some(i64::from(*n)),
        Value::U16(n) => Some(i64::from(*n)),
        Value::I16(n) => Some(i64::from(*n)),
        Value::U32(n) => Some(i64::from(*n)),
        Value::I32(n) => Some(i64::from(*n)),
        Value::U64(n) => i64::try_from(*n).ok(),
        Value::I64(n) => Some(*n),
        _ => None,
    }
}

/// Typed access to the tokenizer keys of a GGUF metadata map.
struct Metadata<'a>(&'a HashMap<String, Value>);

impl<'a> Metadata<'a> {
    /// A string value, if present and a string.
    fn string(&self, key: &str) -> Option<&'a str> {
        match self.0.get(key) {
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// A boolean value, if present and a boolean.
    fn boolean(&self, key: &str) -> Option<bool> {
        match self.0.get(key) {
            Some(Value::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// An array value; `None` when the key is absent.
    fn array(&self, key: &str) -> Result<Option<&'a Vec<Value>>, CucaError> {
        match self.0.get(key) {
            Some(Value::Array(items)) => Ok(Some(items)),
            Some(_) => Err(local_err(format!("GGUF `{key}` is not an array"))),
            None => Ok(None),
        }
    }

    /// A required array of strings.
    fn strings(&self, key: &str) -> Result<Vec<&'a str>, CucaError> {
        self.array(key)?
            .ok_or_else(|| local_err(format!("GGUF lacks `{key}`")))?
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.as_str()),
                _ => Err(local_err(format!("GGUF `{key}` holds a non-string entry"))),
            })
            .collect()
    }

    /// A token id (`u32` or `i32`-style value) that is inside the vocabulary.
    fn token_id(&self, key: &str, vocab_size: usize) -> Option<u32> {
        let n = value_int(self.0.get(key)?)?;
        let id = u32::try_from(n).ok()?;
        ((id as usize) < vocab_size).then_some(id)
    }

    /// Whether this is a Phi-3 family SentencePiece checkpoint, for which
    /// llama.cpp (and the HuggingFace `tokenizer.json`) strip the whitespace
    /// after every special token. llama.cpp tests the lower-cased
    /// `general.name` for `phi-3`/`phi3`; the Phi-3.5 GGUF only says
    /// `Phi 3.5 Mini Instruct`, so `general.basename` (`Phi-3.5`) is tested too,
    /// and so is `general.architecture == "phi3"` for a `llama` (SPM) model.
    /// Phi-4-mini shares the architecture but is a `gpt2` model named `Phi-4`,
    /// and gets no stripping.
    fn is_phi3(&self, model: &str) -> bool {
        let named = ["general.name", "general.basename"]
            .iter()
            .filter_map(|key| self.string(key))
            .any(|name| {
                let name = name.to_ascii_lowercase();
                name.contains("phi-3") || name.contains("phi3")
            });
        named || (model == "llama" && self.string("general.architecture") == Some("phi3"))
    }
}

impl Tokenizer {
    /// Build a tokenizer from vocabulary pieces, their GGUF token types and the
    /// textual merge rules (`"left right"`, ranked by position): the
    /// space-escaped BPE scheme of `gemma4`, with no BOS/EOS ids.
    ///
    /// Fails when the piece and type counts differ, a merge rule is malformed, or
    /// a rule names a token (or merged result) missing from the vocabulary.
    #[cfg(test)]
    pub(crate) fn new(
        pieces: Vec<String>,
        token_types: Vec<i32>,
        merges: &[String],
    ) -> Result<Self, CucaError> {
        let vocab = Vocab::new(&pieces, token_types)?;
        let merges = Merges::new(&vocab, merges, true)?;
        Ok(Self::assemble(
            vocab,
            Engine::Bpe {
                merges,
                words: Words::Escaped,
            },
            Config::default(),
            false,
        ))
    }

    /// Build a tokenizer from GGUF metadata.
    ///
    /// Supports `tokenizer.ggml.model` = `llama` (SentencePiece), `gpt2`
    /// (byte-level BPE, pre-tokenizer from `tokenizer.ggml.pre`, missing meaning
    /// llama.cpp's `default`) and `gemma4`. Any other model, or a `gpt2`
    /// pre-tokenizer this module does not implement, is an error naming the
    /// value. BOS/EOS flags default as in llama.cpp (SPM: BOS; byte BPE: BOS only
    /// for the llama3-family pres; `gemma4`: BOS forced) and
    /// `tokenizer.ggml.add_{bos,eos}_token` override them. SPM falls back to
    /// llama.cpp's default ids (BOS 1, EOS 2) when the id keys are absent.
    pub(crate) fn from_gguf(md: &HashMap<String, gguf_file::Value>) -> Result<Self, CucaError> {
        /// Which scheme the metadata selects.
        enum Kind {
            Spm,
            Byte(Pretokenizer),
            Escaped,
        }
        let md = Metadata(md);
        let model = md
            .string("tokenizer.ggml.model")
            .ok_or_else(|| local_err("GGUF lacks `tokenizer.ggml.model`"))?;
        let kind = match model {
            "llama" => Kind::Spm,
            "gpt2" => {
                let pre = md.string("tokenizer.ggml.pre").filter(|p| !p.is_empty());
                match Pre::from_name(pre) {
                    Some(Pre::Words(p)) => Kind::Byte(p),
                    Some(Pre::Gemma4) => Kind::Escaped,
                    None => {
                        return Err(local_err(format!(
                            "unsupported pre-tokenizer `{}`",
                            pre.unwrap_or_default()
                        )));
                    }
                }
            }
            "gemma4" => Kind::Escaped,
            other => return Err(local_err(format!("unsupported tokenizer model `{other}`"))),
        };

        let pieces = md.strings("tokenizer.ggml.tokens")?;
        let n = pieces.len();
        let token_types = match md.array("tokenizer.ggml.token_type")? {
            None => vec![TYPE_NORMAL; n],
            Some(items) => items
                .iter()
                .map(|v| {
                    value_int(v)
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or_else(|| {
                            local_err("GGUF `tokenizer.ggml.token_type` holds a non-integer entry")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let vocab = Vocab::new(&pieces, token_types)?;

        let (default_bos, default_eos) = match kind {
            Kind::Spm => (true, false),
            Kind::Byte(pre) => (pre.add_bos(), false),
            Kind::Escaped => (true, false),
        };
        let mut config = Config {
            bos: md.token_id("tokenizer.ggml.bos_token_id", n),
            eos: md.token_id("tokenizer.ggml.eos_token_id", n),
            add_bos: md
                .boolean("tokenizer.ggml.add_bos_token")
                .unwrap_or(default_bos),
            add_eos: md
                .boolean("tokenizer.ggml.add_eos_token")
                .unwrap_or(default_eos),
        };
        // llama.cpp forces BOS on for gemma4 whatever the metadata says.
        if matches!(kind, Kind::Escaped) {
            config.add_bos = true;
        }

        let engine = match kind {
            Kind::Spm => {
                // llama.cpp's SPM default special ids.
                let default_id = |id: u32| ((id as usize) < n).then_some(id);
                config.bos = config.bos.or_else(|| default_id(1));
                config.eos = config.eos.or_else(|| default_id(2));
                let scores = match md.array("tokenizer.ggml.scores")? {
                    None => vec![0.0; n],
                    Some(items) => {
                        if items.len() < n {
                            return Err(local_err(format!(
                                "GGUF `tokenizer.ggml.scores` has {} entries for {n} tokens",
                                items.len()
                            )));
                        }
                        items
                            .iter()
                            .take(n)
                            .map(|v| match v {
                                Value::F32(s) => Ok(*s),
                                Value::I32(s) => Ok(*s as f32),
                                _ => Err(local_err(
                                    "GGUF `tokenizer.ggml.scores` holds a non-f32/i32 entry",
                                )),
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                };
                let prefix = md
                    .boolean("tokenizer.ggml.add_space_prefix")
                    .unwrap_or(true);
                Engine::Spm(Spm::new(scores, prefix))
            }
            Kind::Byte(pre) => Engine::Bpe {
                merges: Merges::new(&vocab, &md.strings("tokenizer.ggml.merges")?, false)?,
                words: Words::Byte(pre),
            },
            Kind::Escaped => Engine::Bpe {
                merges: Merges::new(&vocab, &md.strings("tokenizer.ggml.merges")?, true)?,
                words: Words::Escaped,
            },
        };
        let rstrip_specials = md.is_phi3(model);
        Ok(Self::assemble(vocab, engine, config, rstrip_specials))
    }

    /// Collect the special tokens and finish construction.
    fn assemble(vocab: Vocab, engine: Engine, config: Config, rstrip_specials: bool) -> Self {
        let mut specials: Vec<Special> = (0..vocab.len() as u32)
            .filter(|&id| {
                !vocab.piece(id).is_empty()
                    && matches!(
                        vocab.token_type(id),
                        Some(TYPE_CONTROL | TYPE_USER_DEFINED | TYPE_UNKNOWN)
                    )
            })
            .map(|id| Special {
                id,
                lstrip: false,
                rstrip: rstrip_specials
                    && !matches!(vocab.piece(id), "<unk>" | "<s>" | "<|endoftext|>"),
            })
            .collect();
        specials.sort_by(|a, b| {
            vocab
                .piece(b.id)
                .len()
                .cmp(&vocab.piece(a.id).len())
                .then(a.id.cmp(&b.id))
        });
        Self {
            vocab,
            engine,
            specials,
            config,
        }
    }

    /// Encode `text` to token ids, recognising special tokens in the text
    /// (longest first, as llama.cpp's `tokenizer_st_partition`) and never adding
    /// BOS/EOS. SentencePiece models prepend a space to the first raw fragment and
    /// to every raw fragment that follows a special token when
    /// `add_space_prefix` is set.
    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        let mut buffers = Buffers::default();
        let mut after_special = true;
        for fragment in self.partition_specials(text) {
            match fragment {
                Fragment::Special(id) => {
                    out.push(id);
                    after_special = true;
                }
                Fragment::Raw(raw) => {
                    self.encode_raw(raw, after_special, &mut buffers, &mut out);
                    after_special = false;
                }
            }
        }
        out
    }

    /// Look up the id of an exact token text.
    pub(crate) fn token_id(&self, text: &str) -> Option<u32> {
        self.vocab.get(text)
    }

    /// The raw vocabulary text of `id` (empty for an out-of-range id).
    pub(crate) fn token_text(&self, id: u32) -> &str {
        self.vocab.piece(id)
    }

    /// Append the bytes `id` renders to onto `out`.
    ///
    /// Normal, unused and undefined pieces render their text decoded per scheme
    /// (`▁` to a space for SentencePiece and `gemma4`, the GPT-2 alphabet back to
    /// bytes for byte-level BPE); `<0xHH>` byte tokens render their single byte;
    /// user-defined tokens render verbatim; control and unknown tokens render
    /// nothing.
    pub(crate) fn decode_into(&self, id: u32, out: &mut Vec<u8>) {
        let Some(ty) = self.vocab.token_type(id) else {
            return;
        };
        let text = self.vocab.piece(id);
        match ty {
            TYPE_NORMAL | TYPE_UNUSED | TYPE_UNDEFINED => {
                let mut buf = [0u8; 4];
                if matches!(
                    self.engine,
                    Engine::Bpe {
                        words: Words::Byte(_),
                        ..
                    }
                ) {
                    for c in text.chars() {
                        match char_to_byte(c) {
                            Some(byte) => out.push(byte),
                            None => out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes()),
                        }
                    }
                } else {
                    for c in text.chars() {
                        if c == SPACE_MARK {
                            out.push(b' ');
                        } else {
                            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            }
            TYPE_BYTE => {
                if let Some(byte) = parse_byte_token(text) {
                    out.push(byte);
                }
            }
            TYPE_USER_DEFINED => out.extend_from_slice(text.as_bytes()),
            _ => {}
        }
    }

    /// Split `text` around every special token, longest special first, each
    /// applied left to right over the still-raw fragments.
    ///
    /// A special can only occur if its first byte, and its first two bytes, occur
    /// in `text`; both are checked against bit sets before any search.
    fn partition_specials<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut fragments = vec![Fragment::Raw(text)];
        if self.specials.is_empty() {
            return fragments;
        }
        let mut first = [false; 256];
        let mut pairs = vec![0u64; 1 << 10];
        let bytes = text.as_bytes();
        for &b in bytes {
            first[b as usize] = true;
        }
        for w in bytes.windows(2) {
            let key = (usize::from(w[0]) << 8) | usize::from(w[1]);
            pairs[key >> 6] |= 1 << (key & 63);
        }
        let mut next: Vec<Fragment<'a>> = Vec::new();
        for &Special { id, lstrip, rstrip } in &self.specials {
            let special = self.vocab.piece(id);
            let sb = special.as_bytes();
            let Some(&lead) = sb.first() else {
                continue;
            };
            if !first[lead as usize] {
                continue;
            }
            if let Some(&second) = sb.get(1) {
                let key = (usize::from(lead) << 8) | usize::from(second);
                if pairs[key >> 6] & (1 << (key & 63)) == 0 {
                    continue;
                }
            }
            if !fragments
                .iter()
                .any(|f| matches!(f, Fragment::Raw(raw) if raw.contains(special)))
            {
                continue;
            }
            next.clear();
            for fragment in fragments.drain(..) {
                let Fragment::Raw(mut rest) = fragment else {
                    next.push(fragment);
                    continue;
                };
                while let Some(pos) = rest.find(special) {
                    let mut left = &rest[..pos];
                    if lstrip {
                        left = left.trim_end_matches(is_c_space);
                    }
                    if !left.is_empty() {
                        next.push(Fragment::Raw(left));
                    }
                    next.push(Fragment::Special(id));
                    rest = &rest[pos + special.len()..];
                    if rstrip {
                        rest = rest.trim_start_matches(is_c_space);
                    }
                }
                if !rest.is_empty() {
                    next.push(Fragment::Raw(rest));
                }
            }
            std::mem::swap(&mut fragments, &mut next);
        }
        fragments
    }

    /// Encode a special-free fragment with the scheme's algorithm.
    fn encode_raw(
        &self,
        raw: &str,
        after_special: bool,
        buffers: &mut Buffers,
        out: &mut Vec<u32>,
    ) {
        match &self.engine {
            Engine::Spm(spm) => spm.encode_fragment(&self.vocab, raw, after_special, out),
            Engine::Bpe {
                merges,
                words: Words::Byte(pre),
            } => {
                let Buffers {
                    ranges,
                    stage_ranges,
                    scratch,
                    text: word,
                } = buffers;
                pre.split(raw, ranges, stage_ranges);
                for range in ranges.iter() {
                    word.clear();
                    word.extend(raw[range.clone()].bytes().map(byte_to_char));
                    if pre.ignore_merges()
                        && let Some(id) = self.vocab.get(word)
                    {
                        out.push(id);
                        continue;
                    }
                    merges.merge_word(&self.vocab, scratch, word, Unknown::Drop, out);
                }
            }
            Engine::Bpe {
                merges,
                words: Words::Escaped,
            } => {
                let Buffers {
                    scratch,
                    text: escaped,
                    ..
                } = buffers;
                escaped.clear();
                escaped.extend(raw.chars().map(|c| if c == ' ' { SPACE_MARK } else { c }));
                let mut rest = escaped.as_str();
                while let Some(first) = rest.chars().next() {
                    let newline = first == '\n';
                    let end = rest
                        .find(|c: char| (c == '\n') != newline)
                        .unwrap_or(rest.len());
                    let (word, tail) = rest.split_at(end);
                    rest = tail;
                    if newline && let Some(id) = self.vocab.get(word) {
                        out.push(id);
                        continue;
                    }
                    merges.merge_word(&self.vocab, scratch, word, Unknown::ByteTokens, out);
                }
            }
        }
    }
}

/// Accessors the engine and chat integration consume.
impl Tokenizer {
    /// Number of vocabulary entries.
    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode a prompt: [`Self::encode`], plus BOS first when [`Self::add_bos`] is
    /// set and the ids do not already start with it, plus EOS last under the same
    /// rule for [`Self::add_eos`].
    pub(crate) fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode(text);
        if self.config.add_bos
            && let Some(bos) = self.config.bos
            && ids.first() != Some(&bos)
        {
            ids.insert(0, bos);
        }
        if self.config.add_eos
            && let Some(eos) = self.config.eos
            && ids.last() != Some(&eos)
        {
            ids.push(eos);
        }
        ids
    }

    /// Whether `id` is a control or unknown token (GGUF types 3 and 2).
    pub(crate) fn is_control(&self, id: u32) -> bool {
        self.vocab.is_control(id)
    }

    /// The BOS id (`tokenizer.ggml.bos_token_id`), if in range.
    #[cfg(test)]
    pub(crate) fn bos_id(&self) -> Option<u32> {
        self.config.bos
    }

    /// The EOS id (`tokenizer.ggml.eos_token_id`), if in range.
    #[cfg(test)]
    pub(crate) fn eos_id(&self) -> Option<u32> {
        self.config.eos
    }

    /// The BOS piece text, or `""`.
    pub(crate) fn bos_token(&self) -> &str {
        self.config.bos.map_or("", |id| self.vocab.piece(id))
    }

    /// The EOS piece text, or `""`.
    pub(crate) fn eos_token(&self) -> &str {
        self.config.eos.map_or("", |id| self.vocab.piece(id))
    }

    /// Whether [`Self::encode_prompt`] ends with EOS.
    #[cfg(test)]
    pub(crate) fn add_eos(&self) -> bool {
        self.config.add_eos
    }

    /// Whether [`Self::encode_prompt`] starts with BOS.
    #[cfg(test)]
    pub(crate) fn add_bos(&self) -> bool {
        self.config.add_bos
    }

    /// True for SentencePiece models with `add_space_prefix`: encoding put a
    /// space in front of the text, so a decoder of generated text drops one
    /// leading space of the first emitted text (llama.cpp's `lstrip` on
    /// detokenize).
    pub(crate) fn strips_leading_space(&self) -> bool {
        matches!(&self.engine, Engine::Spm(spm) if spm.add_space_prefix())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic vocabulary for the space-escaped scheme: `(text, type)` by id.
    const VOCAB: &[(&str, i32)] = &[
        ("<pad>", 3),     // 0
        ("<eos>", 3),     // 1
        ("<bos>", 3),     // 2
        ("<unk>", 2),     // 3
        ("\u{2581}", 1),  // 4
        ("a", 1),         // 5
        ("b", 1),         // 6
        ("c", 1),         // 7
        ("ab", 1),        // 8
        ("bc", 1),        // 9
        ("abc", 1),       // 10
        ("\u{2581}a", 1), // 11
        ("\n", 1),        // 12
        ("\n\n", 1),      // 13
        ("<0xC3>", 6),    // 14
        ("<0xA9>", 6),    // 15
        ("<|x|>", 4),     // 16
        ("<|y|>", 3),     // 17
        ("<unused0>", 5), // 18
        ("a\n", 1),       // 19
    ];

    const MERGES: &[&str] = &["a b", "b c", "ab c", "\u{2581} a", "\n \n", "a \n"];

    fn build(vocab: &[(&str, i32)], merges: &[&str]) -> Result<Tokenizer, CucaError> {
        Tokenizer::new(
            vocab.iter().map(|(t, _)| t.to_string()).collect(),
            vocab.iter().map(|(_, ty)| *ty).collect(),
            &merges.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
        )
    }

    fn synthetic() -> Tokenizer {
        build(VOCAB, MERGES).expect("synthetic vocabulary is valid")
    }

    fn decode(t: &Tokenizer, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            t.decode_into(id, &mut out);
        }
        out
    }

    #[test]
    fn merges_follow_rank_order() {
        let t = synthetic();
        assert_eq!(t.encode("abc"), [10]);
        assert_eq!(t.encode(" a"), [11]);
        // `b c` outranks `a b` here, so `bc` forms first and `a` stays alone.
        let t = build(
            &[("a", 1), ("b", 1), ("c", 1), ("ab", 1), ("bc", 1)],
            &["b c", "a b"],
        )
        .expect("valid");
        assert_eq!(t.encode("abc"), [0, 4]);
    }

    #[test]
    fn equal_rank_ties_merge_leftmost_first() {
        let t = build(&[("a", 1), ("aa", 1)], &["a a"]).expect("valid");
        assert_eq!(t.encode("aaa"), [1, 0]);
        assert_eq!(t.encode("aaaa"), [1, 1]);
    }

    #[test]
    fn first_duplicate_merge_rule_wins() {
        let t = build(&[("a", 1), ("b", 1), ("ab", 1)], &["a b", "a b"]).expect("valid");
        assert_eq!(t.encode("ab"), [2]);
    }

    #[test]
    fn specials_split_out_of_text() {
        let t = synthetic();
        assert_eq!(t.encode("ab<|x|>c"), [8, 16, 7]);
        assert_eq!(t.encode("ab<|x|><|y|>c"), [8, 16, 17, 7]);
        assert_eq!(t.encode("<bos>a"), [2, 5]);
        assert_eq!(t.encode("<|x|><|x|>"), [16, 16]);
        assert_eq!(t.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn longest_special_wins() {
        let t = build(&[("<a>", 3), ("<a><b>", 3), ("<b>", 3)], &[]).expect("valid");
        assert_eq!(t.encode("<a><b>"), [1]);
    }

    #[test]
    fn empty_pieces_never_become_specials() {
        let t = build(&[("", 3), ("a", 1)], &[]).expect("valid");
        assert_eq!(t.encode("a"), [1]);
        assert_eq!(t.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn newline_runs_are_never_merged_across() {
        let t = synthetic();
        // `a \n` is a rule, but the newline run is a separate word.
        assert_eq!(t.encode("a\n"), [5, 12]);
        assert_eq!(t.encode("\n\n"), [13]);
        assert_eq!(t.encode("\n\n\n"), [13, 12]);
        assert_eq!(t.encode("ab\nbc"), [8, 12, 9]);
    }

    #[test]
    fn unknown_characters_fall_back_to_bytes() {
        let t = synthetic();
        assert_eq!(t.encode("\u{e9}"), [14, 15]);
        assert_eq!(t.encode("a\u{e9}b"), [5, 14, 15, 6]);
    }

    #[test]
    fn decode_renders_by_token_type() {
        let t = synthetic();
        let bytes = decode(&t, &[11, 14, 15, 16, 1, 3, 18]);
        assert_eq!(
            String::from_utf8(bytes).expect("valid UTF-8"),
            " a\u{e9}<|x|><unused0>"
        );
        // A lone lead byte stays an incomplete sequence.
        assert_eq!(decode(&t, &[14]), [0xC3]);
        assert!(decode(&t, &[999_999]).is_empty());
    }

    #[test]
    fn encode_decode_round_trip() {
        let t = synthetic();
        let text = "abc a\nbc\u{e9}\n\n";
        let ids = t.encode(text);
        assert_eq!(decode(&t, &ids), text.as_bytes());
    }

    #[test]
    fn lookups_work() {
        let t = synthetic();
        assert_eq!(t.token_id("<bos>"), Some(2));
        assert_eq!(t.token_id("missing"), None);
        assert_eq!(t.token_text(10), "abc");
        assert_eq!(t.token_text(999_999), "");
        assert_eq!(t.vocab_size(), VOCAB.len());
        assert!(t.is_control(0) && t.is_control(3) && !t.is_control(5) && !t.is_control(16));
        assert_eq!(t.bos_id(), None, "the builder sets no BOS/EOS");
        assert_eq!(t.bos_token(), "");
        assert!(!t.add_bos() && !t.add_eos() && !t.strips_leading_space());
        assert_eq!(t.encode_prompt("abc"), [10]);
    }

    #[test]
    fn debug_prints_sizes_not_the_vocabulary() {
        let shown = format!("{:?}", synthetic());
        assert!(shown.contains("vocab_size: 20"), "{shown}");
        assert!(!shown.contains("<unused0>"), "{shown}");
    }

    #[test]
    fn invalid_merges_are_rejected() {
        let outside = build(VOCAB, &["a zzz"]).expect_err("unknown right half");
        assert!(
            outside
                .to_string()
                .contains("merge rule 0 references a token outside the vocabulary")
        );
        let no_result = build(VOCAB, &["a a"]).expect_err("`aa` is not a token");
        assert!(
            no_result
                .to_string()
                .contains("merge rule 0 references a token outside the vocabulary")
        );
        assert!(build(VOCAB, &["nospace"]).is_err());
        assert!(
            Tokenizer::new(vec!["a".into()], vec![], &[]).is_err(),
            "piece/type count mismatch"
        );
    }

    // ---- GGUF metadata construction -------------------------------------

    fn strings(items: &[&str]) -> Value {
        Value::Array(items.iter().map(|s| Value::String((*s).into())).collect())
    }

    fn string_value(s: &str) -> Value {
        Value::String(s.into())
    }

    /// Metadata of a gemma4-style tokenizer over [`VOCAB`].
    fn metadata(model: &str) -> HashMap<String, Value> {
        HashMap::from([
            ("tokenizer.ggml.model".to_string(), string_value(model)),
            (
                "tokenizer.ggml.tokens".to_string(),
                strings(&VOCAB.iter().map(|(t, _)| *t).collect::<Vec<_>>()),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Value::Array(VOCAB.iter().map(|(_, ty)| Value::I32(*ty)).collect()),
            ),
            ("tokenizer.ggml.merges".to_string(), strings(MERGES)),
        ])
    }

    #[test]
    fn from_gguf_reads_metadata_and_checks_model() {
        let t = Tokenizer::from_gguf(&metadata("gemma4")).expect("gemma4 metadata");
        assert_eq!(t.encode("abc"), [10]);
        assert!(t.add_bos(), "gemma4 forces BOS on");
        let err = Tokenizer::from_gguf(&metadata("bert")).expect_err("unsupported model");
        assert!(
            err.to_string()
                .contains("unsupported tokenizer model `bert`"),
            "{err}"
        );
        let mut missing = metadata("gemma4");
        missing.remove("tokenizer.ggml.merges");
        assert!(Tokenizer::from_gguf(&missing).is_err());
        let mut no_model = metadata("gemma4");
        no_model.remove("tokenizer.ggml.model");
        assert!(Tokenizer::from_gguf(&no_model).is_err());
    }

    #[test]
    fn from_gguf_names_an_unsupported_pre_tokenizer() {
        let mut md = metadata("gpt2");
        md.insert("tokenizer.ggml.pre".into(), string_value("no-such-pre"));
        let err = Tokenizer::from_gguf(&md).expect_err("unknown pre");
        assert!(
            err.to_string()
                .contains("unsupported pre-tokenizer `no-such-pre`"),
            "{err}"
        );
        // `gemma4` is reachable as a pre under model `gpt2`.
        md.insert("tokenizer.ggml.pre".into(), string_value("gemma4"));
        let t = Tokenizer::from_gguf(&md).expect("gemma4 pre");
        assert_eq!(t.encode("abc"), [10]);
    }

    // ---- byte-level BPE -------------------------------------------------

    /// Byte-level vocabulary: the 256 alphabet characters (ids 0..=255), then
    /// merged pieces and specials.
    fn byte_level(pre: Option<&str>, extra: &[(&str, i32)], merges: &[&str]) -> Tokenizer {
        let mut tokens: Vec<String> = (0..=255u8).map(|b| byte_to_char(b).to_string()).collect();
        let mut types = vec![1; 256];
        for (text, ty) in extra {
            tokens.push((*text).to_string());
            types.push(*ty);
        }
        let mut md = HashMap::from([
            ("tokenizer.ggml.model".to_string(), string_value("gpt2")),
            (
                "tokenizer.ggml.tokens".to_string(),
                Value::Array(tokens.iter().map(|s| Value::String(s.clone())).collect()),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Value::Array(types.into_iter().map(Value::I32).collect()),
            ),
            ("tokenizer.ggml.merges".to_string(), strings(merges)),
            ("tokenizer.ggml.bos_token_id".to_string(), Value::U32(267)),
            ("tokenizer.ggml.eos_token_id".to_string(), Value::I32(268)),
        ]);
        if let Some(pre) = pre {
            md.insert("tokenizer.ggml.pre".into(), string_value(pre));
        }
        Tokenizer::from_gguf(&md).expect("byte-level metadata")
    }

    /// Ids after the 256 base characters: 256 `he`, 257 `ll`, 258 `hell`,
    /// 259 `hello`, 260 `Ġw`, 261 `or`, 262 `ld`, 263 `Ġwor`, 264 `Ġworld`,
    /// 265 `Ġhello`, 266 `Ġhi`, 267 `<|bos|>`, 268 `<|eos|>`, 269 `<|user|>`.
    const BYTE_EXTRA: &[(&str, i32)] = &[
        ("he", 1),
        ("ll", 1),
        ("hell", 1),
        ("hello", 1),
        ("\u{120}w", 1),
        ("or", 1),
        ("ld", 1),
        ("\u{120}wor", 1),
        ("\u{120}world", 1),
        ("\u{120}hello", 1),
        ("\u{120}hi", 1),
        ("<|bos|>", 3),
        ("<|eos|>", 3),
        ("<|user|>", 4),
    ];

    const BYTE_MERGES: &[&str] = &[
        "h e",
        "l l",
        "he ll",
        "hell o",
        "\u{120} w",
        "o r",
        "l d",
        "\u{120}w or",
        "\u{120}wor ld",
        // `hi` is not a vocabulary token: the rule must be skipped, not crash.
        "h i",
    ];

    #[test]
    fn byte_level_merges_by_rank() {
        let t = byte_level(Some("qwen2"), BYTE_EXTRA, BYTE_MERGES);
        assert_eq!(t.encode("hello world"), [259, 264]);
        assert_eq!(t.encode("hello"), [259]);
        // Merging stops where rules stop: `hi` cannot merge.
        assert_eq!(t.encode("hi"), [u32::from(b'h'), u32::from(b'i')]);
        assert_eq!(t.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn byte_level_ignore_merges_follows_the_pre() {
        // `Ġhi` is a token but no rule builds it: only ignore_merges finds it.
        let llama3 = byte_level(Some("llama-bpe"), BYTE_EXTRA, BYTE_MERGES);
        assert_eq!(llama3.encode(" hi"), [266]);
        assert_eq!(llama3.encode(" hello"), [265]);
        let smaug = byte_level(Some("smaug-bpe"), BYTE_EXTRA, BYTE_MERGES);
        // Without ignore_merges the word falls apart: space (`Ġ`, byte 0x20 = id
        // 32), `h`, `i`.
        assert_eq!(smaug.encode(" hi"), [32, 104, 105]);
    }

    #[test]
    fn byte_level_decodes_the_alphabet_back_to_bytes() {
        let t = byte_level(Some("qwen2"), BYTE_EXTRA, BYTE_MERGES);
        // The 256 base tokens decode to every byte value, in order.
        let all: Vec<u32> = (0..256).collect();
        let bytes = decode(&t, &all);
        assert_eq!(bytes, (0..=255u8).collect::<Vec<_>>());
        assert_eq!(decode(&t, &[264]), b" world");
        // Special kinds: control renders nothing, user-defined verbatim.
        assert!(decode(&t, &[267]).is_empty());
        assert_eq!(decode(&t, &[269]), b"<|user|>");
        // Non-ASCII text round-trips through the byte alphabet.
        let text = "h\u{e9}llo \u{4e16}\u{754c} \u{1f980}";
        assert_eq!(decode(&t, &t.encode(text)), text.as_bytes());
    }

    #[test]
    fn byte_level_specials_and_bos_handling() {
        let t = byte_level(Some("llama-bpe"), BYTE_EXTRA, BYTE_MERGES);
        assert_eq!(t.encode("hello<|user|>hello"), [259, 269, 259]);
        assert_eq!(t.bos_id(), Some(267));
        assert_eq!(t.eos_id(), Some(268));
        assert_eq!(t.bos_token(), "<|bos|>");
        assert_eq!(t.eos_token(), "<|eos|>");
        // The llama3 family adds BOS by default; EOS is off.
        assert!(t.add_bos() && !t.add_eos());
        assert_eq!(t.encode_prompt("hello"), [267, 259]);
        // No double BOS when the text already starts with it.
        assert_eq!(t.encode_prompt("<|bos|>hello"), [267, 259]);
        assert_eq!(t.encode("hello"), [259], "encode never adds BOS");
        // Other pres default to no BOS.
        let qwen = byte_level(Some("qwen2"), BYTE_EXTRA, BYTE_MERGES);
        assert!(!qwen.add_bos());
        assert_eq!(qwen.encode_prompt("hello"), [259]);
    }

    #[test]
    fn missing_pre_is_default_and_flags_can_be_overridden() {
        let t = byte_level(None, BYTE_EXTRA, BYTE_MERGES);
        assert_eq!(t.encode("hello world"), [259, 264]);
        assert!(!t.add_bos());

        let mut md = HashMap::from([
            ("tokenizer.ggml.model".to_string(), string_value("gpt2")),
            ("tokenizer.ggml.tokens".to_string(), strings(&["a", "<e>"])),
            ("tokenizer.ggml.merges".to_string(), strings(&[])),
            ("tokenizer.ggml.pre".to_string(), string_value("llama3")),
            ("tokenizer.ggml.bos_token_id".to_string(), Value::U32(1)),
            ("tokenizer.ggml.eos_token_id".to_string(), Value::U32(1)),
            (
                "tokenizer.ggml.add_bos_token".to_string(),
                Value::Bool(false),
            ),
            (
                "tokenizer.ggml.add_eos_token".to_string(),
                Value::Bool(true),
            ),
        ]);
        let t = Tokenizer::from_gguf(&md).expect("valid");
        assert!(!t.add_bos() && t.add_eos());
        assert_eq!(t.encode_prompt("a"), [0, 1]);
        md.insert("tokenizer.ggml.bos_token_id".into(), Value::I32(-1));
        assert_eq!(Tokenizer::from_gguf(&md).expect("valid").bos_id(), None);
    }

    // ---- SentencePiece --------------------------------------------------

    /// SPM vocabulary: `(text, type, score)` by id.
    const SPM_VOCAB: &[(&str, i32, f32)] = &[
        ("<unk>", 2, 0.0),       // 0
        ("<s>", 3, 0.0),         // 1
        ("</s>", 3, 0.0),        // 2
        ("<0x0A>", 6, 0.0),      // 3
        ("<0xC3>", 6, 0.0),      // 4
        ("<0xA9>", 6, 0.0),      // 5
        ("\u{2581}", 1, -1.0),   // 6
        ("a", 1, -2.0),          // 7
        ("b", 1, -2.0),          // 8
        ("c", 1, -2.0),          // 9
        ("\u{2581}a", 1, -3.0),  // 10
        ("ab", 1, -3.0),         // 11
        ("bc", 1, -2.5),         // 12
        ("\u{2581}ab", 1, -4.0), // 13
        ("<|u|>", 4, 0.0),       // 14
    ];

    /// SPM metadata over [`SPM_VOCAB`] with extra keys.
    fn spm(extra: &[(&str, Value)]) -> Tokenizer {
        let mut md = HashMap::from([
            ("tokenizer.ggml.model".to_string(), string_value("llama")),
            (
                "tokenizer.ggml.tokens".to_string(),
                strings(&SPM_VOCAB.iter().map(|(t, _, _)| *t).collect::<Vec<_>>()),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Value::Array(SPM_VOCAB.iter().map(|(_, ty, _)| Value::I32(*ty)).collect()),
            ),
            (
                "tokenizer.ggml.scores".to_string(),
                Value::Array(SPM_VOCAB.iter().map(|(_, _, s)| Value::F32(*s)).collect()),
            ),
        ]);
        for (key, value) in extra {
            md.insert((*key).to_string(), value.clone());
        }
        Tokenizer::from_gguf(&md).expect("SPM metadata")
    }

    #[test]
    fn spm_merges_by_score_then_leftmost() {
        let t = spm(&[]);
        // "▁abc": `bc` (-2.5) outscores `▁a` and `ab` (-3), then `▁a` forms.
        assert_eq!(t.encode("abc"), [10, 12]);
        assert_eq!(t.encode("ab"), [13], "`▁` `a` `b` cascade into `▁ab`");
        assert_eq!(t.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn spm_prefix_follows_specials_and_the_option() {
        let t = spm(&[]);
        assert!(t.strips_leading_space());
        // The first raw fragment and a fragment after a special get the prefix.
        assert_eq!(t.encode("a<|u|>a"), [10, 14, 10]);
        assert_eq!(t.encode("<|u|>a"), [14, 10]);
        // Inside one fragment only the leading space escapes to `▁`.
        assert_eq!(t.encode("a a"), [10, 10]);
        let t = spm(&[("tokenizer.ggml.add_space_prefix", Value::Bool(false))]);
        assert!(!t.strips_leading_space());
        assert_eq!(t.encode("a<|u|>a"), [7, 14, 7]);
        assert_eq!(t.encode(" a"), [10]);
    }

    #[test]
    fn spm_byte_fallback_and_newline() {
        let t = spm(&[("tokenizer.ggml.add_space_prefix", Value::Bool(false))]);
        assert_eq!(t.encode("\u{e9}"), [4, 5]);
        assert_eq!(t.encode("a\nb"), [7, 3, 8]);
        assert_eq!(
            decode(&t, &t.encode("a\u{e9}\nb c")),
            "a\u{e9}\nb c".as_bytes()
        );
    }

    #[test]
    fn spm_prompt_adds_bos_and_dedupes() {
        let t = spm(&[
            ("tokenizer.ggml.bos_token_id", Value::U32(1)),
            ("tokenizer.ggml.eos_token_id", Value::U32(2)),
        ]);
        assert!(t.add_bos() && !t.add_eos(), "SPM defaults");
        assert_eq!(t.encode_prompt("a"), [1, 10]);
        assert_eq!(t.encode_prompt("<s>a"), [1, 10], "no double BOS");
        assert_eq!(t.encode("a"), [10], "encode never adds BOS");
        let t = spm(&[
            ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
            ("tokenizer.ggml.add_eos_token", Value::Bool(true)),
        ]);
        assert_eq!(t.bos_id(), Some(1), "SPM default ids apply without keys");
        assert_eq!(t.eos_id(), Some(2));
        assert_eq!(t.encode_prompt("a"), [10, 2]);
        assert_eq!(t.encode_prompt("a</s>"), [10, 2], "no double EOS");
        assert_eq!(t.bos_token(), "<s>");
        assert_eq!(t.eos_token(), "</s>");
    }

    #[test]
    fn spm_scores_may_be_integers_and_must_cover_the_vocabulary() {
        let ints = Value::Array((0..SPM_VOCAB.len() as i32).map(Value::I32).collect());
        let t = spm(&[("tokenizer.ggml.scores", ints)]);
        // Scores now grow with the id, so later (longer) tokens merge first.
        assert_eq!(t.encode("ab"), [13]);
        let mut md = HashMap::from([
            ("tokenizer.ggml.model".to_string(), string_value("llama")),
            ("tokenizer.ggml.tokens".to_string(), strings(&["a", "b"])),
            (
                "tokenizer.ggml.scores".to_string(),
                Value::Array(vec![Value::F32(0.0)]),
            ),
        ]);
        assert!(Tokenizer::from_gguf(&md).is_err(), "too few scores");
        md.remove("tokenizer.ggml.scores");
        let t = Tokenizer::from_gguf(&md).expect("scores are optional");
        assert_eq!(t.encode("ab"), [0, 1], "no merges without tokens like `ab`");
    }

    // ---- lstrip / rstrip ------------------------------------------------

    #[test]
    fn phi3_specials_strip_the_whitespace_after_them() {
        let phi3 = spm(&[("general.architecture", string_value("phi3"))]);
        // `<|u|>` (id 14) eats the newline and spaces after it; the fragment
        // after a special still gets the SPM prefix space.
        assert_eq!(phi3.encode("a<|u|>\n  a"), [10, 14, 10]);
        assert_eq!(
            phi3.encode("a<|u|>\n"),
            [10, 14],
            "an all-space remainder vanishes"
        );
        // `<s>` and `<unk>` keep the following whitespace.
        assert_eq!(phi3.encode("<s>\na"), [1, 6, 3, 7]);
        // Only following whitespace is stripped, not preceding.
        assert_eq!(phi3.encode("a \n<|u|>a"), [10, 6, 3, 14, 10]);
        // Any other SPM model keeps everything.
        let plain = spm(&[]);
        assert_eq!(plain.encode("a<|u|>\na"), [10, 14, 6, 3, 7]);
        // `general.basename` alone selects it too.
        let named = spm(&[("general.basename", string_value("Phi-3.5"))]);
        assert_eq!(named.encode("a<|u|>\n  a"), [10, 14, 10]);
    }

    #[test]
    fn phi3_rule_needs_the_name_or_a_llama_phi3_architecture() {
        let md = |entries: &[(&str, &str)]| -> HashMap<String, Value> {
            entries
                .iter()
                .map(|(k, v)| ((*k).to_string(), string_value(v)))
                .collect()
        };
        let hit = |entries: &[(&str, &str)], model: &str| Metadata(&md(entries)).is_phi3(model);
        assert!(hit(&[("general.name", "Phi-3-mini-4k-instruct")], "llama"));
        assert!(hit(&[("general.name", "my phi3 tune")], "gpt2"));
        assert!(hit(&[("general.basename", "Phi-3.5")], "llama"));
        assert!(hit(&[("general.architecture", "phi3")], "llama"));
        // Phi-3.5 name alone has no hyphen; Phi-4-mini is arch phi3 but gpt2.
        assert!(!hit(&[("general.name", "Phi 3.5 Mini Instruct")], "llama"));
        assert!(!hit(
            &[
                ("general.architecture", "phi3"),
                ("general.name", "Phi 4 Mini Instruct"),
                ("general.basename", "Phi-4"),
            ],
            "gpt2"
        ));
        assert!(!hit(&[], "llama"));
    }

    #[test]
    fn lstrip_drops_whitespace_before_a_special() {
        let mut t = spm(&[]);
        for special in &mut t.specials {
            special.lstrip = special.id == 14;
        }
        // Trailing whitespace of the preceding text goes; following text stays.
        assert_eq!(t.encode("a \t<|u|>a"), [10, 14, 10]);
        // A remainder that is only whitespace before the special disappears.
        assert_eq!(t.encode(" <|u|> a"), [14, 6, 10]);
    }
}
