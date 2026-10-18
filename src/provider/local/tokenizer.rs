//! Gemma 4 byte-pair-encoding tokenizer, following llama.cpp's `gemma4` BPE.
//!
//! The vocabulary, token types and merge rules come from the GGUF metadata
//! (`tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`,
//! `tokenizer.ggml.merges`). [`Tokenizer::encode`] works in three stages, as
//! llama.cpp does:
//!
//! 1. special tokens (control, user-defined and unknown types) are split out of
//!    the text longest-first, so `<|turn>` in a prompt becomes one id;
//! 2. each remaining fragment has spaces replaced by `▁` (U+2581) and is cut
//!    into maximal runs of `\n` and runs of everything else, so a merge never
//!    spans a newline boundary (the reference HuggingFace tokenizer would merge
//!    across it);
//! 3. each run is merged with a heap-driven BPE (lowest rank first, ties to the
//!    leftmost pair); characters absent from the vocabulary fall back to one
//!    `<0xHH>` byte token per UTF-8 byte.
//!
//! The tokenizer never adds a BOS token. Allocation: one id vector per call plus
//! transient symbol and heap vectors per word; the tables are built once.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use candle_core::quantized::gguf_file;

use crate::error::CucaError;
use crate::types::ProviderEndpoint;

/// GGUF token type: undefined.
const TYPE_UNDEFINED: i32 = 0;
/// GGUF token type: normal vocabulary piece.
const TYPE_NORMAL: i32 = 1;
/// GGUF token type: unknown (`<unk>`).
const TYPE_UNKNOWN: i32 = 2;
/// GGUF token type: control token (`<bos>`, `<|turn>`, ...).
const TYPE_CONTROL: i32 = 3;
/// GGUF token type: user-defined token, matched verbatim in text.
const TYPE_USER_DEFINED: i32 = 4;
/// GGUF token type: unused slot.
const TYPE_UNUSED: i32 = 5;
/// GGUF token type: single raw byte, spelled `<0xHH>`.
const TYPE_BYTE: i32 = 6;

/// The word-boundary marker substituted for ASCII space.
const SPACE_MARK: char = '\u{2581}';
/// Sentinel for "no neighbour" in the BPE symbol list.
const NONE: usize = usize::MAX;

/// Build a local-provider error.
fn local_err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

/// A text fragment during special-token partitioning.
enum Fragment<'a> {
    /// Text still to be BPE-encoded.
    Raw(&'a str),
    /// An already-resolved special token id.
    Special(u32),
}

/// One symbol of the BPE linked list: a byte range of the word, with a vocab id
/// when the range is a known token.
struct Symbol {
    /// Vocabulary id, or `None` for a character outside the vocabulary.
    id: Option<u32>,
    /// Start byte offset in the word.
    start: usize,
    /// End byte offset (exclusive) in the word.
    end: usize,
    /// Index of the previous live symbol, or [`NONE`].
    prev: usize,
    /// Index of the next live symbol, or [`NONE`].
    next: usize,
    /// False once merged into its left neighbour.
    alive: bool,
}

/// Gemma 4 BPE tokenizer built from GGUF metadata.
#[derive(Debug, Clone)]
pub(crate) struct Tokenizer {
    /// Token text by id.
    pieces: Vec<String>,
    /// GGUF token type by id.
    attrs: Vec<i32>,
    /// Reverse vocabulary lookup (first id wins on duplicate text).
    piece_to_id: HashMap<String, u32>,
    /// `(left id, right id)` to `(rank, merged id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special tokens as `(text, id)`, longest text first.
    specials: Vec<(String, u32)>,
    /// Byte-fallback token for each raw byte value.
    byte_ids: [Option<u32>; 256],
}

impl Tokenizer {
    /// Build a tokenizer from vocabulary pieces, their GGUF token types and the
    /// textual merge rules (`"left right"`, ranked by position).
    ///
    /// Fails when the piece and type counts differ, a merge rule is malformed, or
    /// a rule names a token (or merged result) missing from the vocabulary.
    pub(crate) fn new(
        pieces: Vec<String>,
        token_types: Vec<i32>,
        merges: &[String],
    ) -> Result<Self, CucaError> {
        if pieces.len() != token_types.len() {
            return Err(local_err(format!(
                "tokenizer has {} tokens but {} token types",
                pieces.len(),
                token_types.len()
            )));
        }
        let mut piece_to_id = HashMap::with_capacity(pieces.len());
        for (id, piece) in pieces.iter().enumerate() {
            piece_to_id.entry(piece.clone()).or_insert(id as u32);
        }

        let mut merge_table = HashMap::with_capacity(merges.len());
        for (rank, rule) in merges.iter().enumerate() {
            let split = rule
                .as_bytes()
                .iter()
                .skip(1)
                .position(|&b| b == b' ')
                .map(|p| p + 1)
                .ok_or_else(|| local_err(format!("merge rule {rank} is malformed: `{rule}`")))?;
            let (left, right) = (&rule[..split], &rule[split + 1..]);
            let outside = || {
                local_err(format!(
                    "merge rule {rank} references a token outside the vocabulary"
                ))
            };
            let left_id = *piece_to_id.get(left).ok_or_else(outside)?;
            let right_id = *piece_to_id.get(right).ok_or_else(outside)?;
            let merged_id = *piece_to_id
                .get(rule.replacen(' ', "", 1).as_str())
                .ok_or_else(outside)?;
            merge_table
                .entry((left_id, right_id))
                .or_insert((rank as u32, merged_id));
        }

        let mut specials: Vec<(String, u32)> = pieces
            .iter()
            .zip(&token_types)
            .enumerate()
            .filter(|(_, (piece, ty))| {
                !piece.is_empty() && matches!(**ty, TYPE_CONTROL | TYPE_USER_DEFINED | TYPE_UNKNOWN)
            })
            .map(|(id, (piece, _))| (piece.clone(), id as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let mut byte_ids = [None; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            *slot = piece_to_id.get(format!("<0x{byte:02X}>").as_str()).copied();
        }

        Ok(Self {
            pieces,
            attrs: token_types,
            piece_to_id,
            merges: merge_table,
            specials,
            byte_ids,
        })
    }

    /// Build a tokenizer from GGUF metadata, requiring
    /// `tokenizer.ggml.model == "gemma4"`.
    pub(crate) fn from_gguf(md: &HashMap<String, gguf_file::Value>) -> Result<Self, CucaError> {
        let model = match md.get("tokenizer.ggml.model") {
            Some(gguf_file::Value::String(s)) => s.as_str(),
            _ => return Err(local_err("GGUF lacks `tokenizer.ggml.model`")),
        };
        if model != "gemma4" {
            return Err(local_err(format!("unsupported tokenizer model `{model}`")));
        }
        let array = |key: &str| -> Result<&Vec<gguf_file::Value>, CucaError> {
            match md.get(key) {
                Some(gguf_file::Value::Array(items)) => Ok(items),
                Some(_) => Err(local_err(format!("GGUF `{key}` is not an array"))),
                None => Err(local_err(format!("GGUF lacks `{key}`"))),
            }
        };
        let strings = |key: &str| -> Result<Vec<String>, CucaError> {
            array(key)?
                .iter()
                .map(|v| match v {
                    gguf_file::Value::String(s) => Ok(s.clone()),
                    _ => Err(local_err(format!("GGUF `{key}` holds a non-string entry"))),
                })
                .collect()
        };
        let pieces = strings("tokenizer.ggml.tokens")?;
        let merges = strings("tokenizer.ggml.merges")?;
        let token_types = array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| match v {
                gguf_file::Value::I32(n) => Ok(*n),
                gguf_file::Value::U32(n) => i32::try_from(*n).map_err(|e| local_err(e.to_string())),
                gguf_file::Value::I8(n) => Ok(i32::from(*n)),
                gguf_file::Value::U8(n) => Ok(i32::from(*n)),
                gguf_file::Value::I16(n) => Ok(i32::from(*n)),
                gguf_file::Value::U16(n) => Ok(i32::from(*n)),
                _ => Err(local_err(
                    "GGUF `tokenizer.ggml.token_type` holds a non-integer entry",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(pieces, token_types, &merges)
    }

    /// Number of vocabulary entries.
    #[cfg(test)]
    pub(crate) fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Encode `text` to token ids, recognising special tokens in the text and
    /// never adding a BOS token.
    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        for fragment in self.partition_specials(text) {
            match fragment {
                Fragment::Special(id) => out.push(id),
                Fragment::Raw(raw) => self.encode_raw(raw, &mut out),
            }
        }
        out
    }

    /// Look up the id of an exact token text.
    pub(crate) fn token_id(&self, text: &str) -> Option<u32> {
        self.piece_to_id.get(text).copied()
    }

    /// The raw vocabulary text of `id` (empty for an out-of-range id).
    pub(crate) fn token_text(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map_or("", String::as_str)
    }

    /// Append the bytes `id` renders to onto `out`.
    ///
    /// Normal, unused and undefined pieces render their text with `▁` as a
    /// space; byte tokens render their single byte; user-defined tokens render
    /// verbatim; control and unknown tokens render nothing.
    pub(crate) fn decode_into(&self, id: u32, out: &mut Vec<u8>) {
        let Some(&attr) = self.attrs.get(id as usize) else {
            return;
        };
        let text = &self.pieces[id as usize];
        match attr {
            TYPE_NORMAL | TYPE_UNUSED | TYPE_UNDEFINED => {
                let mut buf = [0u8; 4];
                for c in text.chars() {
                    if c == SPACE_MARK {
                        out.push(b' ');
                    } else {
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            TYPE_BYTE => {
                if let Some(byte) = parse_byte_token(text) {
                    out.push(byte);
                }
            }
            TYPE_USER_DEFINED => out.extend_from_slice(text.as_bytes()),
            TYPE_CONTROL | TYPE_UNKNOWN => {}
            _ => {}
        }
    }

    /// Split `text` around every special token, longest special first, each
    /// applied left to right over the still-raw fragments.
    fn partition_specials<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        let mut fragments = vec![Fragment::Raw(text)];
        if text.is_empty() {
            return Vec::new();
        }
        let mut present = [false; 256];
        for &b in text.as_bytes() {
            present[b as usize] = true;
        }
        for (special, id) in &self.specials {
            if !present[special.as_bytes()[0] as usize] {
                continue;
            }
            let mut next = Vec::with_capacity(fragments.len() + 2);
            for fragment in fragments {
                let Fragment::Raw(mut rest) = fragment else {
                    next.push(fragment);
                    continue;
                };
                while let Some(pos) = rest.find(special.as_str()) {
                    if pos > 0 {
                        next.push(Fragment::Raw(&rest[..pos]));
                    }
                    next.push(Fragment::Special(*id));
                    rest = &rest[pos + special.len()..];
                }
                if !rest.is_empty() {
                    next.push(Fragment::Raw(rest));
                }
            }
            fragments = next;
        }
        fragments
    }

    /// Encode a special-free fragment: map spaces to `▁`, cut into newline and
    /// non-newline runs, and BPE each run.
    fn encode_raw(&self, raw: &str, out: &mut Vec<u32>) {
        let text = raw.replace(' ', "\u{2581}");
        let mut rest = text.as_str();
        while let Some(first) = rest.chars().next() {
            let newline = first == '\n';
            let end = rest
                .find(|c: char| (c == '\n') != newline)
                .unwrap_or(rest.len());
            let (word, tail) = rest.split_at(end);
            rest = tail;
            if newline && let Some(&id) = self.piece_to_id.get(word) {
                out.push(id);
                continue;
            }
            self.bpe_word(word, out);
        }
    }

    /// Push the candidate merge of `left` with its right neighbour, when that
    /// pair has a rule.
    fn push_pair(
        &self,
        symbols: &[Symbol],
        heap: &mut BinaryHeap<Reverse<(u32, usize, usize)>>,
        left: usize,
    ) {
        let right = symbols[left].next;
        if right == NONE {
            return;
        }
        if let (Some(a), Some(b)) = (symbols[left].id, symbols[right].id)
            && let Some(&(rank, _)) = self.merges.get(&(a, b))
        {
            heap.push(Reverse((rank, left, right)));
        }
    }

    /// Run heap-based BPE over one word and append the resulting ids.
    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<Symbol> = Vec::with_capacity(word.len());
        for (start, c) in word.char_indices() {
            let end = start + c.len_utf8();
            let index = symbols.len();
            symbols.push(Symbol {
                id: self.piece_to_id.get(&word[start..end]).copied(),
                start,
                end,
                prev: if index == 0 { NONE } else { index - 1 },
                next: NONE,
                alive: true,
            });
            if index > 0 {
                symbols[index - 1].next = index;
            }
        }

        let mut heap = BinaryHeap::new();
        for left in 0..symbols.len().saturating_sub(1) {
            self.push_pair(&symbols, &mut heap, left);
        }
        while let Some(Reverse((rank, left, right))) = heap.pop() {
            if !symbols[left].alive || !symbols[right].alive || symbols[left].next != right {
                continue;
            }
            let (Some(a), Some(b)) = (symbols[left].id, symbols[right].id) else {
                continue;
            };
            let Some(&(current_rank, merged)) = self.merges.get(&(a, b)) else {
                continue;
            };
            if current_rank != rank {
                continue;
            }
            symbols[left].id = Some(merged);
            symbols[left].end = symbols[right].end;
            symbols[right].alive = false;
            let after = symbols[right].next;
            symbols[left].next = after;
            if after != NONE {
                symbols[after].prev = left;
            }
            let before = symbols[left].prev;
            if before != NONE {
                self.push_pair(&symbols, &mut heap, before);
            }
            self.push_pair(&symbols, &mut heap, left);
        }

        let mut index = 0;
        while index != NONE {
            let symbol = &symbols[index];
            match symbol.id {
                Some(id) => out.push(id),
                None => {
                    for &byte in &word.as_bytes()[symbol.start..symbol.end] {
                        if let Some(id) = self.byte_ids[byte as usize] {
                            out.push(id);
                        }
                    }
                }
            }
            index = symbol.next;
        }
    }
}

/// Parse the byte value from a `<0xHH>` token text.
fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic vocabulary: `(text, type)` by id.
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

    fn metadata(model: &str) -> HashMap<String, gguf_file::Value> {
        use gguf_file::Value;
        let strings = |items: Vec<&str>| {
            Value::Array(items.into_iter().map(|s| Value::String(s.into())).collect())
        };
        HashMap::from([
            (
                "tokenizer.ggml.model".to_string(),
                Value::String(model.into()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                strings(VOCAB.iter().map(|(t, _)| *t).collect()),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Value::Array(VOCAB.iter().map(|(_, ty)| Value::I32(*ty)).collect()),
            ),
            (
                "tokenizer.ggml.merges".to_string(),
                strings(MERGES.to_vec()),
            ),
        ])
    }

    #[test]
    fn from_gguf_reads_metadata_and_checks_model() {
        let t = Tokenizer::from_gguf(&metadata("gemma4")).expect("gemma4 metadata");
        assert_eq!(t.encode("abc"), [10]);
        let err = Tokenizer::from_gguf(&metadata("llama")).expect_err("wrong model");
        assert!(
            err.to_string()
                .contains("unsupported tokenizer model `llama`")
        );
        let mut missing = metadata("gemma4");
        missing.remove("tokenizer.ggml.merges");
        assert!(Tokenizer::from_gguf(&missing).is_err());
    }

    /// Model-gated golden test against ids produced by the reference
    /// HuggingFace tokenizer (`tokenizer.json` of google/gemma-4-E2B-it).
    ///
    /// The reference merges across `\n`/non-`\n` boundaries while llama.cpp (and
    /// this tokenizer) cut newline runs first. If a vector ever differs from the
    /// reference only at a merge spanning such a boundary, the llama.cpp-style
    /// (newline-run split) result is the expected one; all current vectors agree.
    #[test]
    fn golden_ids_match_reference_tokenizer() {
        let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") else {
            eprintln!("SKIP: CUCA_LOCAL_MODEL_PATH not set");
            return;
        };
        let mut file = std::fs::File::open(&path).expect("model file opens");
        let content = gguf_file::Content::read(&mut file).expect("GGUF header parses");
        let t = Tokenizer::from_gguf(&content.metadata).expect("tokenizer builds");
        assert_eq!(t.vocab_size(), 262_144);
        assert_eq!(t.token_id("<bos>"), Some(2));
        assert_eq!(t.token_id("<eos>"), Some(1));
        let cases: &[(&str, &[u32])] = GOLDEN;
        for (text, expected) in cases {
            assert_eq!(t.encode(text), *expected, "encoding of {text:?}");
        }
    }

    /// Expected ids per input (see [`golden_ids_match_reference_tokenizer`]).
    const GOLDEN: &[(&str, &[u32])] = &[
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
}
