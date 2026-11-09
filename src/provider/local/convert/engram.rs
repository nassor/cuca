//! The hash constants of DeepSeek-V4.1's Engram layers, computed exactly like
//! the reference checkpoint's `inference/engram.py` does.
//!
//! The runtime reads them from the GGUF file, so only the converter needs to
//! reproduce them; each one comes from a library the engine does not have:
//!
//! - [`primes`] and [`head_offsets`]: the bucket moduli of the hash heads
//!   (`EngramLayout.from_args`, `find_next_prime`) and their row offsets;
//! - [`multipliers`]: the per-layer odd hash multipliers, drawn from numpy's
//!   `default_rng(10007 * layer_id).integers(...)`, hence a port of numpy's
//!   `SeedSequence` ([`SeedSequence`]), `PCG64` ([`Pcg64`]) and its bounded
//!   integer generation (`random_bounded_uint64_fill`, Lemire's method)
//!   ([`Pcg64::integers_i64`]), bit for bit;
//! - [`compressed_token_map`]: the token id to compressed id table
//!   (`build_compressed_token_map`): tokens whose decoded text normalizes alike
//!   (NFKC, NFD, accents stripped, lowercased, whitespace collapsed) share an
//!   id. The normalizer is the Hugging Face `tokenizers` crate, the very
//!   implementation the reference runs, used here as a text library only.
//!
//! Everything is a pure function of its arguments.

use std::collections::HashMap;

use serde_json::Value;
use tokenizers::normalizers::{
    Lowercase, NFD, NFKC, NormalizerWrapper, Replace, Sequence, Strip, StripAccents,
};
use tokenizers::tokenizer::{NormalizedString, Normalizer};

use crate::error::CucaError;

// ---------------------------------------------------------------------------
// numpy's SeedSequence
// ---------------------------------------------------------------------------

/// numpy `bit_generator.pyx` constants of the entropy-mixing hash.
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
/// Half the word width: the xor-shift of every mixing step.
const XSHIFT: u32 = 16;
/// The entropy pool of a `SeedSequence` (`DEFAULT_POOL_SIZE`).
const POOL_SIZE: usize = 4;

/// numpy's `np.random.SeedSequence(entropy)` for an integer entropy and no spawn key.
#[derive(Debug, Clone)]
pub(crate) struct SeedSequence {
    /// The mixed entropy pool.
    pool: [u32; POOL_SIZE],
}

/// numpy's `hashmix`: mixes `value` with the running constant, which advances.
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

/// numpy's `mix`: combines two words.
fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

impl SeedSequence {
    /// The sequence of an integer entropy: its little-endian 32-bit words
    /// (a single zero word for 0), mixed into the pool (`mix_entropy`).
    pub(crate) fn new(entropy: u64) -> Self {
        let low = entropy as u32;
        let high = (entropy >> 32) as u32;
        let words: &[u32] = if high == 0 { &[low] } else { &[low, high] };
        let mut hash_const = INIT_A;
        let mut pool = [0u32; POOL_SIZE];
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        // Mix all bits together.
        for src in 0..POOL_SIZE {
            for dst in 0..POOL_SIZE {
                if src != dst {
                    let hashed = hashmix(pool[src], &mut hash_const);
                    pool[dst] = mix(pool[dst], hashed);
                }
            }
        }
        // Any entropy words beyond the pool mix into every pool word.
        for &word in words.iter().skip(POOL_SIZE) {
            for slot in &mut pool {
                *slot = mix(*slot, hashmix(word, &mut hash_const));
            }
        }
        Self { pool }
    }

    /// `generate_state(n_words, np.uint64)`: each 64-bit word is two 32-bit
    /// words of the stream, the earlier one in the low half.
    pub(crate) fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let mut next_u32 = |i: usize| {
            let mut data = self.pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data = data.wrapping_mul(hash_const);
            data ^ (data >> XSHIFT)
        };
        (0..n_words)
            .map(|k| {
                let low = u64::from(next_u32(2 * k));
                let high = u64::from(next_u32(2 * k + 1));
                low | (high << 32)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// numpy's PCG64
// ---------------------------------------------------------------------------

/// `PCG_DEFAULT_MULTIPLIER_128` of the PCG reference implementation.
const PCG_MULTIPLIER: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

/// numpy's `np.random.PCG64` (the generator behind `default_rng`): a 128-bit
/// LCG with an XSL-RR output, plus the 32-bit draw buffer of its bit generator.
#[derive(Debug, Clone)]
pub(crate) struct Pcg64 {
    /// The LCG state.
    state: u128,
    /// The LCG increment (always odd).
    inc: u128,
    /// The unused high half of the last 64-bit draw (`has_uint32`/`uinteger`).
    buffered: Option<u32>,
}

impl Pcg64 {
    /// `np.random.default_rng(seed)`: a `SeedSequence` of `seed`, its first four
    /// 64-bit words as the initial state (high word first) and the stream, seeded
    /// like `pcg_setseq_128_srandom_r`.
    pub(crate) fn new(seed: u64) -> Self {
        let words = SeedSequence::new(seed).generate_state_u64(4);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            buffered: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    /// One LCG step.
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    /// The next 64 random bits (`pcg64_next64`: step, then XSL-RR of the new state).
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step();
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((self.state >> 122) as u32)
    }

    /// The next 32 random bits (`pcg64_next32`): the low half of a fresh 64-bit
    /// draw first, then the cached high half.
    pub(crate) fn next_u32(&mut self) -> u32 {
        if let Some(cached) = self.buffered.take() {
            return cached;
        }
        let next = self.next_u64();
        self.buffered = Some((next >> 32) as u32);
        next as u32
    }

    /// Lemire's bounded draw in `0..=rng` from 32-bit words
    /// (`buffered_bounded_lemire_uint32`); `rng` is below `u32::MAX`.
    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = u64::from(rng) + 1;
        let mut m = u64::from(self.next_u32()) * rng_excl;
        let mut leftover = m & 0xFFFF_FFFF;
        if leftover < rng_excl {
            let threshold = (u64::from(u32::MAX) - u64::from(rng)) % rng_excl;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * rng_excl;
                leftover = m & 0xFFFF_FFFF;
            }
        }
        (m >> 32) as u32
    }

    /// Lemire's bounded draw in `0..=rng` from 64-bit words
    /// (`bounded_lemire_uint64`); `rng` is above `u32::MAX` and below `u64::MAX`.
    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let rng_excl = u128::from(rng) + 1;
        let mut m = u128::from(self.next_u64()) * rng_excl;
        let mut leftover = m as u64;
        if u128::from(leftover) < rng_excl {
            let threshold = u128::from(u64::MAX - rng) % rng_excl;
            while u128::from(leftover) < threshold {
                m = u128::from(self.next_u64()) * rng_excl;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `rng.integers(low=low, high=high, size=(size,), dtype=np.int64)`, the
    /// default (non-masked) path: `high` is exclusive and may be 2^63, which numpy
    /// accepts and `i64` cannot hold.
    ///
    /// numpy draws in the inclusive range `0..=high - 1 - low` and adds `low`:
    /// no draw for a single value, 32-bit words for ranges of at most 2^32 (the
    /// whole word for exactly 2^32 values, Lemire's method otherwise), a whole
    /// 64-bit word for 2^64 values and Lemire's method over 64-bit words in between.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] if the range is empty or `high` exceeds 2^63.
    pub(crate) fn integers_i64(
        &mut self,
        low: i64,
        high: i128,
        size: usize,
    ) -> Result<Vec<i64>, CucaError> {
        if high <= i128::from(low) || high > 1i128 << 63 {
            return Err(CucaError::Config(format!(
                "integers(low={low}, high={high}): empty or out of the int64 range"
            )));
        }
        let rng = (high - 1 - i128::from(low)) as u64;
        let offset = low as u64;
        let draw = |rng_state: &mut Self| -> u64 {
            if rng == 0 {
                0
            } else if rng < u64::from(u32::MAX) {
                u64::from(rng_state.bounded_lemire_u32(rng as u32))
            } else if rng == u64::from(u32::MAX) {
                u64::from(rng_state.next_u32())
            } else if rng == u64::MAX {
                rng_state.next_u64()
            } else {
                rng_state.bounded_lemire_u64(rng)
            }
        };
        Ok((0..size)
            .map(|_| offset.wrapping_add(draw(self)) as i64)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Primes, offsets and multipliers
// ---------------------------------------------------------------------------

/// Whether `n` is prime (trial division; the Engram moduli are around 1.6e7).
fn is_prime(n: u64) -> bool {
    if n < 2 || n.is_multiple_of(2) {
        return n == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= n {
        if n.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The bucket moduli of the hash heads (`EngramLayout.from_args`), indexed
/// `[layer][n-gram order 2..=max_ngram_size][head]`.
///
/// Every order of every layer restarts at `vocab_size - 1` and takes `n_heads`
/// primes, each the next prime above the previous one that no earlier layer,
/// order or head took, so the bucket ranges never overlap.
pub(crate) fn primes(
    n_layers: usize,
    max_ngram_size: usize,
    n_heads: usize,
    vocab_size: u64,
) -> Vec<Vec<Vec<u64>>> {
    let mut seen = std::collections::HashSet::new();
    (0..n_layers)
        .map(|_| {
            (0..max_ngram_size.saturating_sub(1))
                .map(|_| {
                    let mut current = vocab_size.saturating_sub(1);
                    (0..n_heads)
                        .map(|_| {
                            let mut candidate = current + 1;
                            while !is_prime(candidate) || seen.contains(&candidate) {
                                candidate += 1;
                            }
                            seen.insert(candidate);
                            current = candidate;
                            candidate
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// The first row of every hash head in the concatenated tables of all layers:
/// the exclusive prefix sum of every prime, in layer, order, head order (the
/// GGUF's `ple.head_offsets`). A layer's own table starts at the offset of its
/// first head.
pub(crate) fn head_offsets(primes: &[Vec<Vec<u64>>]) -> Vec<u64> {
    let mut total = 0u64;
    primes
        .iter()
        .flatten()
        .flatten()
        .map(|&prime| {
            let offset = total;
            total += prime;
            offset
        })
        .collect()
}

/// The odd hash multipliers, `[layer][lookback]` (`compute_hash_multipliers`):
/// `default_rng(10007 * layer_id).integers(0, bound, size=max_ngram_size)` doubled
/// plus one, where `bound = max(1, (i64::MAX / compressed_vocab_size) / 2)` keeps
/// `token_id * multiplier` below 2^63. (A zero `compressed_vocab_size`, which
/// numpy would divide by, counts as 1; the seed wraps at 2^64, where numpy would
/// hash a longer entropy: neither happens for real layer ids.)
pub(crate) fn multipliers(
    layer_ids: &[u64],
    max_ngram_size: usize,
    compressed_vocab_size: u64,
) -> Vec<Vec<u64>> {
    let bound = ((i64::MAX as u64 / compressed_vocab_size.max(1)) / 2).max(1);
    layer_ids
        .iter()
        .map(|&layer_id| {
            let mut rng = Pcg64::new(layer_id.wrapping_mul(10_007));
            rng.integers_i64(0, i128::from(bound), max_ngram_size)
                .expect("1..=2^62 is a valid, non-empty range")
                .into_iter()
                .map(|v| v as u64 * 2 + 1)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The compressed token map
// ---------------------------------------------------------------------------

/// The GPT-2 byte alphabet reversed: which byte a vocabulary character stands for
/// (printable Latin-1 characters stand for themselves, the other 68 bytes for
/// the characters from U+0100 up, in byte order).
fn byte_of_char() -> HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut map = HashMap::with_capacity(256);
    let mut next = 0x100u32;
    for byte in 0..=u8::MAX {
        if printable(byte) {
            map.insert(char::from(byte), byte);
        } else {
            map.insert(
                char::from_u32(next).expect("U+0100.. are scalar values"),
                byte,
            );
            next += 1;
        }
    }
    map
}

/// `tokenizer.decode([id], skip_special_tokens=False)` of a ByteLevel decoder
/// (`decoders::ByteLevel::decode_chain`): the token's characters as bytes, or,
/// if one is outside the byte alphabet (an added token), the token's own UTF-8
/// bytes; decoded lossily.
fn decode_token(token: &str, bytes: &HashMap<char, u8>) -> String {
    let decoded: Option<Vec<u8>> = token.chars().map(|c| bytes.get(&c).copied()).collect();
    match decoded {
        Some(raw) => String::from_utf8_lossy(&raw).into_owned(),
        None => token.to_owned(),
    }
}

/// The private-use character that keeps a token made of one space from
/// stripping to nothing and merging with unrelated tokens.
const SPACE_SENTINEL: &str = "\u{e000}";

/// The reference normalizer: NFKC, NFD, accent stripping, lowercasing, runs of
/// whitespace to one space, a lone space protected by the sentinel while the
/// ends are stripped.
fn key_normalizer() -> Result<Sequence, CucaError> {
    let replace = |pattern: tokenizers::normalizers::replace::ReplacePattern, content: &str| {
        Replace::new(pattern, content)
            .map(NormalizerWrapper::Replace)
            .map_err(|e| CucaError::Config(format!("engram normalizer: {e}")))
    };
    use tokenizers::normalizers::replace::ReplacePattern::{Regex, String as Literal};
    Ok(Sequence::new(vec![
        NormalizerWrapper::NFKC(NFKC),
        NormalizerWrapper::NFD(NFD),
        NormalizerWrapper::StripAccents(StripAccents),
        NormalizerWrapper::Lowercase(Lowercase),
        replace(Regex(r"[ \t\r\n]+".into()), " ")?,
        replace(Regex("^ $".into()), SPACE_SENTINEL)?,
        NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
        replace(Literal(SPACE_SENTINEL.into()), " ")?,
    ]))
}

/// The token strings of a byte-level BPE `tokenizer.json` by id: the model's
/// vocabulary, overridden by the added tokens (whose `content` is what
/// `id_to_token` returns).
fn token_strings(tokenizer_json: &Value, vocab_size: usize) -> Result<Vec<String>, CucaError> {
    let bad = |what: &str| CucaError::Config(format!("tokenizer.json: {what}"));
    let decoder = tokenizer_json
        .pointer("/decoder/type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("no decoder"))?;
    if decoder != "ByteLevel" {
        return Err(bad(&format!("the decoder is {decoder}, not ByteLevel")));
    }
    let mut tokens: Vec<Option<String>> = vec![None; vocab_size];
    let mut place = |id: u64, token: &str| -> Result<(), CucaError> {
        let slot = usize::try_from(id)
            .ok()
            .and_then(|i| tokens.get_mut(i))
            .ok_or_else(|| {
                bad(&format!(
                    "token id {id} is outside the vocabulary of {vocab_size}"
                ))
            })?;
        *slot = Some(token.to_owned());
        Ok(())
    };
    let vocab = tokenizer_json
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("model.vocab is not a map"))?;
    for (token, id) in vocab {
        place(
            id.as_u64()
                .ok_or_else(|| bad("a vocabulary id is not an integer"))?,
            token,
        )?;
    }
    for added in tokenizer_json
        .get("added_tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = added
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| bad("an added token has no id"))?;
        let content = added
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("an added token has no content"))?;
        place(id, content)?;
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.ok_or_else(|| bad(&format!("no token has id {id}"))))
        .collect()
}

/// The token id to compressed id table (`build_compressed_token_map`) and its
/// number of classes.
///
/// Each id's text is its decoded token. If the text holds U+FFFD (a partial
/// UTF-8 sequence) the key is the raw token spelling; otherwise the key is the
/// normalized text, or the text itself when that is empty. Compressed ids are
/// handed out in the order their keys first appear.
///
/// # Errors
///
/// [`CucaError::Config`] if `tokenizer_json` is not a byte-level `tokenizer.json`
/// that names every id below `vocab_size`.
pub(crate) fn compressed_token_map(
    tokenizer_json: &Value,
    vocab_size: usize,
) -> Result<(Vec<i32>, usize), CucaError> {
    let tokens = token_strings(tokenizer_json, vocab_size)?;
    let bytes = byte_of_char();
    let normalizer = key_normalizer()?;
    let mut classes: HashMap<String, i32> = HashMap::new();
    let mut lookup = Vec::with_capacity(vocab_size);
    for token in &tokens {
        let text = decode_token(token, &bytes);
        let key = if text.contains('\u{fffd}') {
            token.clone()
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(|e| CucaError::Config(format!("engram normalizer: {e}")))?;
            if normalized.get().is_empty() {
                text
            } else {
                normalized.get().to_owned()
            }
        };
        let next = i32::try_from(classes.len())
            .map_err(|_| CucaError::Config("more than i32::MAX compressed tokens".into()))?;
        lookup.push(*classes.entry(key).or_insert(next));
    }
    Ok((lookup, classes.len()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::provider::local::tokenizer::golden::skip;

    /// The two layers' primes of the real checkpoint (`engram_goldens.json`).
    const REAL_PRIMES: [[[u64; 8]; 3]; 2] = [
        [
            [
                16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183,
            ],
            [
                16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321,
            ],
            [
                16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
            ],
        ],
        [
            [
                16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
            ],
            [
                16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
            ],
            [
                16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
            ],
        ],
    ];

    #[test]
    fn the_primes_of_the_real_checkpoint_are_reproduced() {
        let computed = primes(2, 4, 8, 16_000_000);
        let expected: Vec<Vec<Vec<u64>>> = REAL_PRIMES
            .iter()
            .map(|l| l.iter().map(|o| o.to_vec()).collect())
            .collect();
        assert_eq!(computed, expected);
        // Each layer's rows are the config's `engram_num_embeddings`.
        let sums: Vec<u64> = computed.iter().map(|l| l.iter().flatten().sum()).collect();
        assert_eq!(sums, [384_006_168, 384_016_682]);
    }

    #[test]
    fn head_offsets_are_the_exclusive_prefix_sum_over_all_layers() {
        let computed = primes(2, 4, 8, 16_000_000);
        let offsets = head_offsets(&computed);
        assert_eq!(offsets.len(), 48);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 16_000_057);
        assert_eq!(
            offsets[24], 384_006_168,
            "the second layer starts after the first layer's rows"
        );
        assert_eq!(offsets[47] + 16_000_889, 384_006_168 + 384_016_682);
        assert!(head_offsets(&[]).is_empty());
    }

    #[test]
    fn the_multipliers_of_the_real_checkpoint_are_reproduced() {
        let computed = multipliers(&[1, 14], 4, 99_092);
        assert_eq!(
            computed,
            [
                [
                    76632096046245,
                    4839876093313,
                    35959672319349,
                    73987337458391
                ],
                [
                    67716810739261,
                    51510806800915,
                    30921347202721,
                    82619226485591
                ],
            ]
        );
        assert_eq!(
            multipliers(&[1], 2, 0).len(),
            1,
            "an empty vocabulary does not panic"
        );
    }

    #[test]
    fn empty_integer_ranges_are_rejected() {
        let mut rng = Pcg64::new(0);
        assert!(rng.integers_i64(5, 5, 1).is_err());
        assert!(rng.integers_i64(0, (1i128 << 63) + 1, 1).is_err());
        assert!(
            rng.integers_i64(0, 3, 0)
                .expect("size 0 is valid")
                .is_empty()
        );
    }

    /// `numpy_integers_golden.json`: numpy's outputs (see its `numpy` version key).
    #[derive(serde::Deserialize)]
    struct NumpyGolden {
        cases: Vec<Case>,
        states: Vec<State>,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        seed: u64,
        draws: Vec<Draw>,
    }

    #[derive(serde::Deserialize)]
    struct Draw {
        low: i64,
        high: i128,
        size: usize,
        values: Vec<i64>,
    }

    #[derive(serde::Deserialize)]
    struct State {
        seed: u64,
        state: Vec<String>,
    }

    fn numpy_golden() -> NumpyGolden {
        serde_json::from_str(include_str!("../testdata/numpy_integers_golden.json"))
            .expect("the committed numpy golden parses")
    }

    #[test]
    fn seed_sequence_states_match_numpy() {
        let golden = numpy_golden();
        assert!(golden.states.len() >= 20);
        for State { seed, state } in &golden.states {
            let expected: Vec<u64> = state.iter().map(|s| s.parse().expect("a u64")).collect();
            assert_eq!(
                SeedSequence::new(*seed).generate_state_u64(4),
                expected,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn bounded_integers_match_numpy_for_every_branch() {
        let golden = numpy_golden();
        assert!(golden.cases.len() >= 100);
        let mut draws = 0;
        for Case {
            seed,
            draws: triples,
        } in &golden.cases
        {
            for Draw {
                low,
                high,
                size,
                values,
            } in triples
            {
                let got = Pcg64::new(*seed)
                    .integers_i64(*low, *high, *size)
                    .unwrap_or_else(|e| panic!("seed {seed} [{low}, {high}): {e}"));
                assert_eq!(&got, values, "seed {seed}, integers({low}, {high}, {size})");
                draws += 1;
            }
        }
        assert!(draws >= 2000, "{draws} draws");
    }

    /// A tiny byte-level `tokenizer.json`: ids 0..=8 of the vocabulary, 9 added.
    fn synthetic_tokenizer() -> Value {
        json!({
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": {
                "A": 0, "a": 1,
                // `é` is the two bytes C3 A9, spelt `Ã©`.
                "\u{c3}\u{a9}": 2, "e": 3,
                "\u{120}": 4,               // " "
                "\u{120}\u{120}": 5,        // "  "
                "\u{10a}": 6,               // "\n"
                "\u{c3}": 7, "\u{c2}": 8,   // half a UTF-8 sequence each
                "<|x|>": 10
            }},
            "added_tokens": [{"id": 9, "content": "<\u{ff5c}x\u{ff5c}>", "special": true}]
        })
    }

    #[test]
    fn the_normalizer_pipeline_collapses_what_the_reference_collapses() {
        let (lookup, classes) =
            compressed_token_map(&synthetic_tokenizer(), 11).expect("a valid tokenizer");
        assert_eq!(
            lookup,
            [
                0, 0, // `A` and `a`: lowercased
                1, 1, // `é` and `e`: accent stripped
                2, 2, 2, // " ", "  " and "\n" all become one space, kept by the sentinel
                3, 4, // two partial UTF-8 bytes keep their own raw spellings
                5, 5, // an added token with fullwidth bars: NFKC folds them to `<|x|>`
            ]
        );
        assert_eq!(classes, 6);
    }

    #[test]
    fn a_tokenizer_that_cannot_be_decoded_byte_level_is_rejected() {
        let mut not_byte_level = synthetic_tokenizer();
        not_byte_level["decoder"] = json!({"type": "Metaspace"});
        assert!(compressed_token_map(&not_byte_level, 11).is_err());
        // A vocabulary larger than the tokens it names.
        assert!(compressed_token_map(&synthetic_tokenizer(), 12).is_err());
        // Ids past the declared vocabulary.
        assert!(compressed_token_map(&synthetic_tokenizer(), 5).is_err());
    }

    /// The reference tokenizer: `CUCA_DS41_TOKENIZER_JSON`, else the research copy.
    fn real_tokenizer_path() -> PathBuf {
        std::env::var_os("CUCA_DS41_TOKENIZER_JSON").map_or_else(
            || {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join(".omp/local-models/ref/deepseek-v41/tokenizer.json")
            },
            PathBuf::from,
        )
    }

    #[test]
    fn the_token_map_of_the_real_vocabulary_matches_the_reference() {
        let path = real_tokenizer_path();
        let Ok(text) = std::fs::read_to_string(&path) else {
            skip(&format!(
                "the DeepSeek-V4.1 tokenizer.json is absent ({})",
                path.display()
            ));
            return;
        };
        let tokenizer: Value = serde_json::from_str(&text).expect("tokenizer.json parses");
        let (lookup, classes) =
            compressed_token_map(&tokenizer, 129_280).expect("the real vocabulary");
        assert_eq!(classes, 99_092);
        assert_eq!(lookup.len(), 129_280);
        assert_eq!(lookup[2], 2, "the pad token");
        // The reference's first 64 entries (`engram_goldens.json` `token_map_head`).
        assert_eq!(lookup[..64], (0..64).collect::<Vec<i32>>());
        let bytes: Vec<u8> = lookup.iter().flat_map(|v| v.to_le_bytes()).collect();
        let digest: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            digest,
            "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
        );
    }
}
