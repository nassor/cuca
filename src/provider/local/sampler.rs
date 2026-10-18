//! Token sampling for the local engine.
//!
//! Implements the HuggingFace logits-warper order over a raw logits slice:
//! temperature scaling, top-k, softmax, top-p (nucleus), then a single draw
//! from a seedable [`SplitMix64`] generator. Temperature `<= 0` selects the
//! argmax (first maximum wins). The sampler allocates only a candidate vector
//! of at most `top_k` entries (the whole vocabulary when `top_k == 0`).

use std::cmp::Ordering;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::time::SystemTime;

/// SplitMix64 pseudo-random generator: tiny, fast, and fully determined by its
/// 64-bit state, so a fixed seed reproduces a sampling sequence exactly.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    /// Advance the state and return the next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Return a uniform `f32` in `[0, 1)` with 24 bits of precision.
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
}

/// Produce a non-deterministic seed from the std hasher's per-instance random
/// keys mixed with the current wall-clock time.
pub(crate) fn random_seed() -> u64 {
    RandomState::new().hash_one(SystemTime::now())
}

/// Index of the first maximum of `logits` (0 for an empty slice).
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_value {
            best = i;
            best_value = v;
        }
    }
    best as u32
}

/// Descending order on `(logit, id)`; NaN logits sort last.
fn by_logit_desc(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Sample one token id from `logits`.
///
/// Warper order: `temperature <= 0` returns the argmax; otherwise logits are
/// divided by `temperature`, truncated to the `top_k` best (`0` keeps all),
/// softmaxed, truncated to the shortest prefix whose cumulative probability
/// reaches `top_p` (at least one candidate), renormalised, and a token is drawn
/// with `rng`. `logits` is scaled in place.
pub(crate) fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut SplitMix64,
) -> u32 {
    if temperature <= 0.0 || logits.len() <= 1 {
        return argmax(logits);
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }

    let mut candidates: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as u32))
        .collect();
    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, by_logit_desc);
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(by_logit_desc);

    let max = candidates[0].0;
    let mut sum = 0.0f32;
    for c in candidates.iter_mut() {
        c.0 = (c.0 - max).exp();
        sum += c.0;
    }
    for c in candidates.iter_mut() {
        c.0 /= sum;
    }

    let mut cumulative = 0.0f32;
    let mut keep = candidates.len();
    for (i, c) in candidates.iter().enumerate() {
        cumulative += c.0;
        if cumulative >= top_p {
            keep = i + 1;
            break;
        }
    }
    candidates.truncate(keep);

    let total: f32 = candidates.iter().map(|c| c.0).sum();
    let target = rng.next_f32() * total;
    let mut acc = 0.0f32;
    for c in &candidates {
        acc += c.0;
        if target < acc {
            return c.1;
        }
    }
    candidates[candidates.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<f32> {
        vec![0.1, 2.0, 0.5, 1.9, -1.0, 1.95, 0.0]
    }

    #[test]
    fn splitmix64_matches_reference_vector() {
        // Reference outputs for seed 0 from the SplitMix64 specification.
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn next_f32_is_in_unit_interval() {
        let mut rng = SplitMix64(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let mut rng = SplitMix64(1);
        assert_eq!(sample(&mut logits(), 0.0, 0, 1.0, &mut rng), 1);
    }

    #[test]
    fn argmax_first_max_wins() {
        let mut rng = SplitMix64(1);
        assert_eq!(sample(&mut [1.0, 3.0, 3.0, 2.0], 0.0, 0, 1.0, &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_argmax_for_any_seed() {
        for seed in 0..64 {
            let mut rng = SplitMix64(seed);
            assert_eq!(sample(&mut logits(), 1.5, 1, 1.0, &mut rng), 1);
        }
    }

    #[test]
    fn same_seed_same_sequence() {
        let run = |seed| {
            let mut rng = SplitMix64(seed);
            (0..32)
                .map(|_| sample(&mut logits(), 1.0, 5, 0.95, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn tiny_top_p_is_argmax() {
        for seed in 0..64 {
            let mut rng = SplitMix64(seed);
            assert_eq!(sample(&mut logits(), 1.0, 0, 1e-6, &mut rng), 1);
        }
    }

    #[test]
    fn top_k_restricts_candidates() {
        let mut rng = SplitMix64(3);
        for _ in 0..200 {
            let id = sample(&mut logits(), 5.0, 3, 1.0, &mut rng);
            assert!([1, 3, 5].contains(&id), "id {id} outside the top 3");
        }
    }

    #[test]
    fn random_seed_varies() {
        assert_ne!(random_seed(), random_seed());
    }
}
