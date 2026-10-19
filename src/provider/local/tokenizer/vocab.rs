//! Compact vocabulary storage shared by every tokenizer scheme.
//!
//! All piece texts live in one `String` addressed by an offset table, and the
//! text-to-id lookup is an open-addressing table of `u32` ids that compares
//! against that string. That is about 12 bytes per token plus the text itself,
//! instead of two heap strings per token, which matters for 200k-entry
//! vocabularies. Everything is built once and read-only afterwards.

use crate::error::CucaError;

use super::local_err;

/// GGUF token type: undefined.
pub(super) const TYPE_UNDEFINED: i32 = 0;
/// GGUF token type: normal vocabulary piece.
pub(super) const TYPE_NORMAL: i32 = 1;
/// GGUF token type: unknown (`<unk>`).
pub(super) const TYPE_UNKNOWN: i32 = 2;
/// GGUF token type: control token (`<bos>`, `<|turn>`, ...).
pub(super) const TYPE_CONTROL: i32 = 3;
/// GGUF token type: user-defined token, matched verbatim in text.
pub(super) const TYPE_USER_DEFINED: i32 = 4;
/// GGUF token type: unused slot.
pub(super) const TYPE_UNUSED: i32 = 5;
/// GGUF token type: single raw byte, spelled `<0xHH>`.
pub(super) const TYPE_BYTE: i32 = 6;

/// Marks an unused slot of the lookup table.
const EMPTY: u32 = u32::MAX;

/// Token texts, GGUF token types and the text-to-id lookup.
#[derive(Clone)]
pub(super) struct Vocab {
    /// Every piece text, concatenated.
    text: String,
    /// `offsets[id]..offsets[id + 1]` is the text of `id`.
    offsets: Vec<u32>,
    /// GGUF token type by id.
    types: Vec<i32>,
    /// Open-addressing slots holding ids (first id wins on duplicate text).
    table: Vec<u32>,
    /// Byte-fallback token for each raw byte value (`<0xHH>` pieces).
    byte_ids: [Option<u32>; 256],
}

/// FNV-1a over `bytes`, folded so the low bits carry the whole hash.
fn hash(bytes: &[u8]) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h ^ (h >> 32)) as usize
}

impl Vocab {
    /// Build the vocabulary from piece texts and their GGUF token types.
    ///
    /// Fails when the counts differ or the vocabulary does not fit 32-bit ids and
    /// offsets. Empty pieces are kept by id but are not findable by text.
    pub(super) fn new(pieces: Vec<String>, types: Vec<i32>) -> Result<Self, CucaError> {
        if pieces.len() != types.len() {
            return Err(local_err(format!(
                "tokenizer has {} tokens but {} token types",
                pieces.len(),
                types.len()
            )));
        }
        let total: usize = pieces.iter().map(String::len).sum();
        if u32::try_from(total).is_err() || pieces.len() >= EMPTY as usize {
            return Err(local_err("tokenizer vocabulary is too large"));
        }
        let mut text = String::with_capacity(total);
        let mut offsets = Vec::with_capacity(pieces.len() + 1);
        offsets.push(0);
        for piece in &pieces {
            text.push_str(piece);
            // `total` fits `u32`, checked above.
            offsets.push(text.len() as u32);
        }
        drop(pieces);

        let slots = (offsets.len() * 2).next_power_of_two().max(8);
        let mut vocab = Self {
            text,
            offsets,
            types,
            table: vec![EMPTY; slots],
            byte_ids: [None; 256],
        };
        let mask = slots - 1;
        for id in 0..vocab.types.len() {
            let piece = vocab.piece(id as u32);
            if piece.is_empty() {
                continue;
            }
            let mut slot = hash(piece.as_bytes()) & mask;
            loop {
                match vocab.table[slot] {
                    EMPTY => {
                        vocab.table[slot] = id as u32;
                        break;
                    }
                    other if vocab.piece(other) == piece => break,
                    _ => slot = (slot + 1) & mask,
                }
            }
        }
        for byte in 0..=255u8 {
            vocab.byte_ids[byte as usize] = vocab.get(&format!("<0x{byte:02X}>"));
        }
        Ok(vocab)
    }

    /// Number of tokens.
    pub(super) fn len(&self) -> usize {
        self.types.len()
    }

    /// The text of `id` (empty for an out-of-range id).
    pub(super) fn piece(&self, id: u32) -> &str {
        let id = id as usize;
        match (self.offsets.get(id), self.offsets.get(id + 1)) {
            (Some(&start), Some(&end)) => self.text.get(start as usize..end as usize).unwrap_or(""),
            _ => "",
        }
    }

    /// The GGUF token type of `id`, if in range.
    pub(super) fn token_type(&self, id: u32) -> Option<i32> {
        self.types.get(id as usize).copied()
    }

    /// Find the id of an exact piece text.
    pub(super) fn get(&self, text: &str) -> Option<u32> {
        if text.is_empty() {
            return None;
        }
        let mask = self.table.len() - 1;
        let mut slot = hash(text.as_bytes()) & mask;
        loop {
            match self.table[slot] {
                EMPTY => return None,
                id if self.piece(id) == text => return Some(id),
                _ => slot = (slot + 1) & mask,
            }
        }
    }

    /// The `<0xHH>` token of a raw byte, if the vocabulary has one.
    pub(super) fn byte_token(&self, byte: u8) -> Option<u32> {
        self.byte_ids[byte as usize]
    }

    /// Append the `<0xHH>` token of every byte of `text` that has one.
    pub(super) fn push_byte_tokens(&self, text: &str, out: &mut Vec<u32>) {
        out.extend(text.bytes().filter_map(|b| self.byte_token(b)));
    }

    /// Whether `id` is a control or unknown token.
    pub(super) fn is_control(&self, id: u32) -> bool {
        matches!(self.token_type(id), Some(TYPE_CONTROL) | Some(TYPE_UNKNOWN))
    }
}

/// Parse the byte value from a `<0xHH>` token text.
pub(super) fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_and_duplicates() {
        let pieces = ["a", "b", "a", "", "<0x41>"].map(String::from).to_vec();
        let v = Vocab::new(pieces, vec![1, 1, 1, 1, 6]).expect("valid");
        assert_eq!(v.len(), 5);
        assert_eq!(v.get("a"), Some(0), "first duplicate wins");
        assert_eq!(v.get("b"), Some(1));
        assert_eq!(v.get(""), None);
        assert_eq!(v.get("zzz"), None);
        assert_eq!(v.piece(4), "<0x41>");
        assert_eq!(v.piece(99), "");
        assert_eq!(v.byte_token(0x41), Some(4));
        assert_eq!(v.byte_token(0x42), None);
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0x4>"), None);
        assert_eq!(parse_byte_token("0x41"), None);
    }

    #[test]
    fn count_mismatch_is_an_error() {
        assert!(Vocab::new(vec!["a".into()], vec![]).is_err());
    }
}
