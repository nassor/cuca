//! Byte-pair merging shared by the byte-level (`gpt2`) and space-escaped
//! (`gemma4`) schemes, plus the GPT-2 byte-to-unicode alphabet.
//!
//! Merge rules are kept as `(left id, right id) -> (rank, merged id)` in one hash
//! map built once; a word is merged with a heap that always takes the lowest rank
//! and, on equal rank, the leftmost pair, like llama.cpp's `llm_bigram_bpe`
//! work queue. Per word the transient state is one symbol vector and one heap
//! (both reused across every word of an `encode` call), so encoding allocates
//! O(longest word) beyond the output.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use crate::error::CucaError;

use super::local_err;
use super::vocab::Vocab;

/// Sentinel for "no neighbour" in the symbol list.
const NONE: usize = usize::MAX;

/// Multiplicative hasher for the packed `(left, right)` merge keys.
#[derive(Default, Clone)]
pub(super) struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, n: u64) {
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }
}

/// Rank and result of one merge rule.
type MergeEntry = (u32, u32);

/// The merge rules of a vocabulary.
#[derive(Clone)]
pub(super) struct Merges {
    /// `(left << 32 | right)` to `(rank, merged id)`; the first rule wins.
    table: HashMap<u64, MergeEntry, BuildHasherDefault<PairHasher>>,
}

/// Pack a token pair into a map key.
fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

impl Merges {
    /// Parse `"left right"` rules, ranked by position, against `vocab`.
    ///
    /// The split is at the first space after byte 0, as llama.cpp's
    /// `find(' ', 1)`; a rule without one is an error. With `strict`, a rule whose
    /// halves or concatenation are not vocabulary tokens is also an error;
    /// otherwise such a rule is skipped, since it can never produce a token
    /// (real byte-level checkpoints with pruned or extended vocabularies carry
    /// them).
    pub(super) fn new<S: AsRef<str>>(
        vocab: &Vocab,
        rules: &[S],
        strict: bool,
    ) -> Result<Self, CucaError> {
        let mut table = HashMap::with_capacity_and_hasher(rules.len(), Default::default());
        let mut joined = String::new();
        for (rank, rule) in rules.iter().enumerate() {
            let rule = rule.as_ref();
            let split = rule
                .as_bytes()
                .iter()
                .skip(1)
                .position(|&b| b == b' ')
                .map(|p| p + 1)
                .ok_or_else(|| local_err(format!("merge rule {rank} is malformed: `{rule}`")))?;
            let (left, right) = (&rule[..split], &rule[split + 1..]);
            joined.clear();
            joined.push_str(left);
            joined.push_str(right);
            let ids = (vocab.get(left), vocab.get(right), vocab.get(&joined));
            let (Some(left_id), Some(right_id), Some(merged_id)) = ids else {
                if strict {
                    return Err(local_err(format!(
                        "merge rule {rank} references a token outside the vocabulary"
                    )));
                }
                continue;
            };
            table
                .entry(pair_key(left_id, right_id))
                .or_insert((rank as u32, merged_id));
        }
        Ok(Self { table })
    }

    /// Number of distinct merge pairs.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.table.len()
    }

    /// Rank and result of merging `left` and `right`.
    fn get(&self, left: u32, right: u32) -> Option<MergeEntry> {
        self.table.get(&pair_key(left, right)).copied()
    }
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

/// Reusable buffers for merging words.
#[derive(Default)]
pub(super) struct Scratch {
    /// Symbol list of the current word.
    symbols: Vec<Symbol>,
    /// Candidate merges, lowest `(rank, left)` first.
    heap: BinaryHeap<Reverse<(u32, usize, usize)>>,
}

/// What to emit for a character that is not a vocabulary token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Unknown {
    /// One `<0xHH>` byte token per UTF-8 byte (the space-escaped scheme).
    ByteTokens,
    /// Nothing: byte-level vocabularies hold every byte as a character, so a
    /// missing one means the vocabulary is incomplete (llama.cpp drops it too).
    Drop,
}

impl Merges {
    /// Push the candidate merge of `left` with its right neighbour, when that
    /// pair has a rule.
    fn push_pair(&self, scratch: &mut Scratch, left: usize) {
        let right = scratch.symbols[left].next;
        if right == NONE {
            return;
        }
        if let (Some(a), Some(b)) = (scratch.symbols[left].id, scratch.symbols[right].id)
            && let Some((rank, _)) = self.get(a, b)
        {
            scratch.heap.push(Reverse((rank, left, right)));
        }
    }

    /// Merge `word` and append the resulting ids to `out`.
    pub(super) fn merge_word(
        &self,
        vocab: &Vocab,
        scratch: &mut Scratch,
        word: &str,
        unknown: Unknown,
        out: &mut Vec<u32>,
    ) {
        scratch.symbols.clear();
        scratch.heap.clear();
        for (start, c) in word.char_indices() {
            let end = start + c.len_utf8();
            let index = scratch.symbols.len();
            scratch.symbols.push(Symbol {
                id: vocab.get(&word[start..end]),
                start,
                end,
                prev: if index == 0 { NONE } else { index - 1 },
                next: NONE,
                alive: true,
            });
            if index > 0 {
                scratch.symbols[index - 1].next = index;
            }
        }
        for left in 0..scratch.symbols.len().saturating_sub(1) {
            self.push_pair(scratch, left);
        }
        while let Some(Reverse((rank, left, right))) = scratch.heap.pop() {
            let symbols = &mut scratch.symbols;
            if !symbols[left].alive || !symbols[right].alive || symbols[left].next != right {
                continue;
            }
            let (Some(a), Some(b)) = (symbols[left].id, symbols[right].id) else {
                continue;
            };
            let Some((current_rank, merged)) = self.get(a, b) else {
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
                self.push_pair(scratch, before);
            }
            self.push_pair(scratch, left);
        }

        let mut index = 0;
        while index != NONE && index < scratch.symbols.len() {
            let symbol = &scratch.symbols[index];
            match symbol.id {
                Some(id) => out.push(id),
                None if unknown == Unknown::ByteTokens => {
                    vocab.push_byte_tokens(&word[symbol.start..symbol.end], out);
                }
                None => {}
            }
            index = symbol.next;
        }
    }
}

/// Printable-byte ranges that GPT-2 maps to themselves.
const fn is_identity_byte(b: u8) -> bool {
    matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff)
}

/// Number of bytes GPT-2 remaps to code points from U+0100 on.
const REMAPPED: usize = 68;

/// `BYTE_TO_CHAR[b]` is the GPT-2 alphabet character of byte `b`.
const BYTE_TO_CHAR: [char; 256] = {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    let mut b = 0usize;
    while b < 256 {
        table[b] = if is_identity_byte(b as u8) {
            b as u8 as char
        } else {
            let c = match char::from_u32(next) {
                Some(c) => c,
                None => '\0',
            };
            next += 1;
            c
        };
        b += 1;
    }
    table
};

/// `REMAPPED_BYTES[c - 256]` is the byte whose alphabet character is U+0100 + `c - 256`.
const REMAPPED_BYTES: [u8; REMAPPED] = {
    let mut table = [0u8; REMAPPED];
    let mut n = 0usize;
    let mut b = 0usize;
    while b < 256 {
        if !is_identity_byte(b as u8) {
            table[n] = b as u8;
            n += 1;
        }
        b += 1;
    }
    table
};

/// The GPT-2 alphabet character of `byte`.
pub(super) fn byte_to_char(byte: u8) -> char {
    BYTE_TO_CHAR[byte as usize]
}

/// The byte whose GPT-2 alphabet character is `c`, if `c` is in the alphabet.
pub(super) fn char_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    if let Ok(byte) = u8::try_from(code) {
        return is_identity_byte(byte).then_some(byte);
    }
    let index = code.checked_sub(256)? as usize;
    REMAPPED_BYTES.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_round_trips_all_bytes() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let c = byte_to_char(b);
            assert!(seen.insert(c), "alphabet characters are distinct");
            assert_eq!(char_to_byte(c), Some(b), "byte {b:#04x}");
        }
        // Spot checks against the GPT-2 reference table.
        assert_eq!(byte_to_char(b' '), '\u{120}');
        assert_eq!(byte_to_char(b'\n'), '\u{10a}');
        assert_eq!(byte_to_char(b'\t'), '\u{109}');
        assert_eq!(byte_to_char(0), '\u{100}');
        assert_eq!(byte_to_char(0x7f), '\u{121}');
        assert_eq!(byte_to_char(0xa0), '\u{142}');
        assert_eq!(byte_to_char(0xad), '\u{143}');
        assert_eq!(byte_to_char(b'a'), 'a');
        assert_eq!(char_to_byte('\u{144}'), None);
        assert_eq!(char_to_byte('\u{20ac}'), None);
        assert_eq!(char_to_byte('\u{7f}'), None);
    }

    /// A vocabulary with `a`, `b`, `c` and their merges.
    fn vocab(pieces: &[&str]) -> Vocab {
        Vocab::new(pieces, vec![1; pieces.len()]).expect("valid vocabulary")
    }

    fn merge(vocab: &Vocab, merges: &Merges, word: &str, unknown: Unknown) -> Vec<u32> {
        let mut scratch = Scratch::default();
        let mut out = Vec::new();
        merges.merge_word(vocab, &mut scratch, word, unknown, &mut out);
        out
    }

    #[test]
    fn lowest_rank_first_then_leftmost() {
        let v = vocab(&["a", "b", "c", "ab", "bc", "abc"]);
        let rules = ["b c", "a b", "a bc"].map(String::from);
        let m = Merges::new(&v, &rules, true).expect("valid rules");
        assert_eq!(m.len(), 3);
        // `b c` (rank 0) outranks `a b` (rank 1): `bc` forms, then `a bc`.
        assert_eq!(merge(&v, &m, "abc", Unknown::Drop), [5]);
        let rules = ["a b", "b c"].map(String::from);
        let m = Merges::new(&v, &rules, true).expect("valid rules");
        assert_eq!(merge(&v, &m, "abc", Unknown::Drop), [3, 2]);
        // Equal ranks never occur for distinct pairs; identical pairs go leftmost.
        let v = vocab(&["a", "aa"]);
        let m = Merges::new(&v, &["a a".to_string()], true).expect("valid");
        assert_eq!(merge(&v, &m, "aaa", Unknown::Drop), [1, 0]);
        assert_eq!(merge(&v, &m, "aaaa", Unknown::Drop), [1, 1]);
    }

    #[test]
    fn first_duplicate_rule_wins() {
        let v = vocab(&["a", "b", "ab"]);
        let rules = ["a b", "a b"].map(String::from);
        let m = Merges::new(&v, &rules, true).expect("valid");
        assert_eq!(m.len(), 1);
        assert_eq!(merge(&v, &m, "ab", Unknown::Drop), [2]);
    }

    #[test]
    fn lenient_mode_skips_rules_outside_the_vocabulary() {
        let v = vocab(&["a", "b", "ab", "c"]);
        let rules = ["a b", "b c", "a zzz"].map(String::from);
        assert!(Merges::new(&v, &rules, true).is_err());
        let m = Merges::new(&v, &rules, false).expect("lenient");
        // `b c` has no `bc` token and is skipped rather than crashing.
        assert_eq!(m.len(), 1);
        assert_eq!(merge(&v, &m, "abc", Unknown::Drop), [2, 3]);
        assert!(Merges::new(&v, &["nospace".to_string()], false).is_err());
    }

    #[test]
    fn unknown_characters_follow_the_scheme() {
        let v = vocab(&["a", "<0xC3>", "<0xA9>"]);
        let m = Merges::new(&v, &[] as &[&str], true).expect("no rules");
        assert_eq!(merge(&v, &m, "a\u{e9}", Unknown::ByteTokens), [0, 1, 2]);
        assert_eq!(merge(&v, &m, "a\u{e9}", Unknown::Drop), [0]);
    }

    #[test]
    fn rule_split_is_at_first_space_after_byte_zero() {
        // The rule `"  a"` splits at the second space (the first after byte 0)
        // into `" "` and `"a"`, like llama.cpp's `find(' ', 1)`.
        let v = vocab(&[" ", "a", " a"]);
        let m = Merges::new(&v, &["  a".to_string()], true).expect("rule parses");
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(0, 1), Some((0, 2)));
    }
}
