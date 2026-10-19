//! SentencePiece (`tokenizer.ggml.model == "llama"`) encoding, following
//! llama.cpp's `llm_tokenizer_spm_session`.
//!
//! The text of a fragment is escaped (`' '` to `▁`) into one buffer and split
//! into UTF-8 character symbols that are byte ranges of that buffer. A work
//! queue then merges neighbours: the highest token score first, the leftmost
//! pair on a tie, only when the concatenated range is itself a vocabulary token
//! and neither side changed since the pair was queued. Symbols that end up
//! outside the vocabulary become one `<0xHH>` byte token per byte.
//!
//! Allocation: the escaped buffer, the symbol vector and the queue, each
//! O(fragment length) and dropped after the fragment; no per-pair strings.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::vocab::Vocab;

/// Sentinel for "no neighbour" in the symbol list.
const NONE: usize = usize::MAX;

/// SentencePiece scores and options.
#[derive(Clone)]
pub(super) struct Spm {
    /// Token score by id (`tokenizer.ggml.scores`; zero when absent).
    scores: Vec<f32>,
    /// Prepend a space to the first fragment and to any fragment that follows a
    /// special token (`tokenizer.ggml.add_space_prefix`).
    add_space_prefix: bool,
}

/// One symbol: a byte range of the escaped buffer; `len == 0` once merged away.
struct Symbol {
    /// Start byte offset.
    start: usize,
    /// Length in bytes.
    len: usize,
    /// Previous live symbol or [`NONE`].
    prev: usize,
    /// Next live symbol or [`NONE`].
    next: usize,
}

/// A queued merge of two adjacent symbols.
#[derive(Clone, Copy)]
struct Bigram {
    /// Score of the merged token.
    score: f32,
    /// Left symbol index.
    left: usize,
    /// Right symbol index.
    right: usize,
    /// Byte length of the merged text when queued.
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    /// Max-heap order: higher score first, then the smaller left index.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl Spm {
    /// Build from per-token scores (one per vocabulary entry) and the
    /// `add_space_prefix` option.
    pub(super) fn new(scores: Vec<f32>, add_space_prefix: bool) -> Self {
        // `x + 0.0` turns -0.0 into 0.0 so equal scores compare equal.
        let scores = scores.into_iter().map(|s| s + 0.0).collect();
        Self {
            scores,
            add_space_prefix,
        }
    }

    /// Whether fragments get a leading space.
    pub(super) fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    /// Encode one raw fragment; `after_special` is true for the first fragment
    /// and any fragment that follows a special token.
    pub(super) fn encode_fragment(
        &self,
        vocab: &Vocab,
        raw: &str,
        after_special: bool,
        out: &mut Vec<u32>,
    ) {
        let mut text = String::with_capacity(raw.len() + 3);
        if self.add_space_prefix && after_special {
            text.push('\u{2581}');
        }
        for c in raw.chars() {
            text.push(if c == ' ' { '\u{2581}' } else { c });
        }
        self.tokenize(vocab, &text, out);
    }

    /// Queue the merge of `left` and `right` when their concatenation is a token.
    fn try_add_bigram(
        &self,
        vocab: &Vocab,
        text: &str,
        symbols: &[Symbol],
        queue: &mut BinaryHeap<Bigram>,
        left: usize,
        right: usize,
    ) {
        if left == NONE || right == NONE {
            return;
        }
        let size = symbols[left].len + symbols[right].len;
        let start = symbols[left].start;
        let Some(id) = vocab.get(&text[start..start + size]) else {
            return;
        };
        let Some(&score) = self.scores.get(id as usize) else {
            return;
        };
        queue.push(Bigram {
            score,
            left,
            right,
            size,
        });
    }

    /// Run the merge queue over the escaped `text` and append the ids.
    fn tokenize(&self, vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<Symbol> = Vec::with_capacity(text.len());
        for (start, c) in text.char_indices() {
            let index = symbols.len();
            symbols.push(Symbol {
                start,
                len: c.len_utf8(),
                prev: index.wrapping_sub(1),
                next: if start + c.len_utf8() == text.len() {
                    NONE
                } else {
                    index + 1
                },
            });
        }
        if symbols.is_empty() {
            return;
        }
        // `prev` of the first symbol wrapped to `usize::MAX`, which is `NONE`.
        let mut queue = BinaryHeap::with_capacity(symbols.len());
        for right in 1..symbols.len() {
            self.try_add_bigram(vocab, text, &symbols, &mut queue, right - 1, right);
        }
        while let Some(bigram) = queue.pop() {
            let (left, right) = (bigram.left, bigram.right);
            if symbols[left].len == 0
                || symbols[right].len == 0
                || symbols[left].len + symbols[right].len != bigram.size
            {
                continue;
            }
            symbols[left].len += symbols[right].len;
            symbols[right].len = 0;
            let after = symbols[right].next;
            symbols[left].next = after;
            if after != NONE {
                symbols[after].prev = left;
            }
            let before = symbols[left].prev;
            self.try_add_bigram(vocab, text, &symbols, &mut queue, before, left);
            self.try_add_bigram(vocab, text, &symbols, &mut queue, left, after);
        }

        let mut index = 0;
        while index != NONE {
            let symbol = &symbols[index];
            let piece = &text[symbol.start..symbol.start + symbol.len];
            match vocab.get(piece) {
                Some(id) => out.push(id),
                None => vocab.push_byte_tokens(piece, out),
            }
            index = symbol.next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a vocabulary and SPM from `(text, score)` pairs (all normal tokens,
    /// plus `<0xHH>` bytes where listed with type 6 through `byte`).
    fn setup(tokens: &[(&str, f32)], prefix: bool) -> (Vocab, Spm) {
        let pieces: Vec<String> = tokens.iter().map(|(t, _)| t.to_string()).collect();
        let types = pieces
            .iter()
            .map(|p| if p.starts_with("<0x") { 6 } else { 1 })
            .collect();
        let vocab = Vocab::new(&pieces, types).expect("valid vocabulary");
        let spm = Spm::new(tokens.iter().map(|(_, s)| *s).collect(), prefix);
        (vocab, spm)
    }

    fn encode(vocab: &Vocab, spm: &Spm, text: &str, after_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        spm.encode_fragment(vocab, text, after_special, &mut out);
        out
    }

    #[test]
    fn highest_score_merges_first() {
        // `bc` outscores `ab`, so "abc" becomes a + bc even though `ab` is leftmost.
        let (v, s) = setup(
            &[
                ("a", 0.0),
                ("b", 0.0),
                ("c", 0.0),
                ("ab", -2.0),
                ("bc", -1.0),
            ],
            false,
        );
        assert_eq!(encode(&v, &s, "abc", false), [0, 4]);
        let (v, s) = setup(
            &[
                ("a", 0.0),
                ("b", 0.0),
                ("c", 0.0),
                ("ab", -1.0),
                ("bc", -2.0),
            ],
            false,
        );
        assert_eq!(encode(&v, &s, "abc", false), [3, 2]);
    }

    #[test]
    fn equal_scores_merge_leftmost_first() {
        let (v, s) = setup(&[("a", 0.0), ("b", 0.0), ("ab", -1.0), ("ba", -1.0)], false);
        assert_eq!(encode(&v, &s, "aba", false), [2, 0]);
        assert_eq!(encode(&v, &s, "bab", false), [3, 1]);
    }

    #[test]
    fn merges_cascade_through_vocabulary_tokens() {
        let (v, s) = setup(
            &[
                ("a", 0.0),
                ("b", 0.0),
                ("c", 0.0),
                ("ab", -1.0),
                ("abc", -2.0),
            ],
            false,
        );
        assert_eq!(encode(&v, &s, "abc", false), [4]);
        // `bc` is not a token, so `ab` must form first.
        assert_eq!(encode(&v, &s, "abcabc", false), [4, 4]);
    }

    #[test]
    fn escapes_spaces_and_applies_the_prefix() {
        let (v, s) = setup(&[("\u{2581}", 0.0), ("a", 0.0), ("\u{2581}a", -1.0)], true);
        assert_eq!(encode(&v, &s, "a", true), [2]);
        assert_eq!(encode(&v, &s, "a a", true), [2, 2]);
        // No prefix when the fragment follows normal text.
        assert_eq!(encode(&v, &s, "a", false), [1]);
        let (v, s) = setup(&[("\u{2581}", 0.0), ("a", 0.0), ("\u{2581}a", -1.0)], false);
        assert_eq!(encode(&v, &s, "a", true), [1]);
        assert!(!s.add_space_prefix());
    }

    #[test]
    fn unmerged_unknown_characters_become_byte_tokens() {
        let (v, s) = setup(&[("a", 0.0), ("<0xC3>", 0.0), ("<0xA9>", 0.0)], false);
        assert_eq!(encode(&v, &s, "a\u{e9}", false), [0, 1, 2]);
        // A byte without a `<0xHH>` token is dropped rather than crashing.
        assert_eq!(encode(&v, &s, "\u{e8}", false), [1]);
    }

    #[test]
    fn empty_fragment_yields_nothing() {
        let (v, s) = setup(&[("a", 0.0)], false);
        assert!(encode(&v, &s, "", false).is_empty());
    }
}
