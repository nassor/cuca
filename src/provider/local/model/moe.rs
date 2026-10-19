//! Mixture-of-experts routing, shared by the CPU and GPU engines.
//!
//! The router logits of every token reach the host (`tokens * n_expert`
//! floats), where [`route`] applies llama.cpp's `LLM_GATING_SOFTMAX` top-k:
//! softmax over all experts, keep the `n_used` largest, and renormalise the
//! kept weights to sum to one when the architecture asks for it. The result
//! is grouped by expert so each expert runs once on exactly its own tokens.

use super::params::MoeSpec;

/// The tokens routed to each expert with their combination weights.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Routing {
    /// Per expert: `(token index, weight)` pairs in token order.
    pub(crate) per_expert: Vec<Vec<(u32, f32)>>,
}

/// Route `t` tokens whose router logits are row-major in `logits`
/// (`t * n_expert` values).
pub(crate) fn route(logits: &[f32], t: usize, spec: &MoeSpec) -> Routing {
    let n = spec.n_expert;
    let mut per_expert = vec![Vec::new(); n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    for (token, row) in logits.chunks_exact(n).take(t).enumerate() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        probs.clear();
        probs.extend(row.iter().map(|&l| (l - max).exp()));
        let total: f32 = probs.iter().sum();
        order.clear();
        order.extend(0..n);
        // Largest probability first; the lower expert index wins ties, so the
        // order is total and an unstable sort gives the stable sort's result.
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
        let kept = &order[..spec.n_used];
        let denominator = if spec.norm_weights {
            kept.iter().map(|&e| probs[e]).sum::<f32>()
        } else {
            total
        };
        for &e in kept {
            per_expert[e].push((token as u32, probs[e] / denominator));
        }
    }
    Routing { per_expert }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(n_expert: usize, n_used: usize, norm: bool) -> MoeSpec {
        MoeSpec {
            n_expert,
            n_used,
            norm_weights: norm,
        }
    }

    #[test]
    fn picks_the_top_k_and_renormalises() {
        // Token 0 prefers experts 2 then 0; token 1 prefers 1 then 3.
        let logits = [1.0, 0.0, 3.0, -1.0, 0.0, 2.0, -2.0, 1.0];
        let r = route(&logits, 2, &spec(4, 2, true));
        let soft0: Vec<f32> = [1.0f32, 0.0, 3.0, -1.0].iter().map(|l| l.exp()).collect();
        let w2 = soft0[2] / (soft0[2] + soft0[0]);
        assert_eq!(r.per_expert[2].len(), 1);
        assert!((r.per_expert[2][0].1 - w2).abs() < 1e-6);
        assert!((r.per_expert[0][0].1 - (1.0 - w2)).abs() < 1e-6);
        assert_eq!(r.per_expert[1].len(), 1);
        assert_eq!(r.per_expert[1][0].0, 1);
        assert_eq!(r.per_expert[3].len(), 1);
        assert!(r.per_expert[3][0].1 < r.per_expert[1][0].1);
        // Every token's kept weights sum to one.
        let sum0: f32 = r
            .per_expert
            .iter()
            .flatten()
            .filter(|p| p.0 == 0)
            .map(|p| p.1)
            .sum();
        assert!((sum0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn without_renormalisation_weights_are_the_full_softmax() {
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let r = route(&logits, 1, &spec(4, 2, false));
        // Equal logits: experts 0 and 1 win the tie, each with weight 1/4.
        assert_eq!(r.per_expert[0], vec![(0, 0.25)]);
        assert_eq!(r.per_expert[1], vec![(0, 0.25)]);
        assert!(r.per_expert[2].is_empty() && r.per_expert[3].is_empty());
    }

    #[test]
    fn tokens_stay_in_order_within_an_expert() {
        let logits = [5.0f32, 0.0, 5.0, 0.0, 5.0, 0.0];
        let r = route(&logits, 3, &spec(2, 1, true));
        assert_eq!(r.per_expert[0], vec![(0, 1.0), (1, 1.0), (2, 1.0)]);
        assert!(r.per_expert[1].is_empty());
    }
}
