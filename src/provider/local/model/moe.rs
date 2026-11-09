//! Mixture-of-experts routing, shared by the CPU and GPU engines.
//!
//! The router logits of every token reach the host (`tokens * n_expert`
//! floats), where [`route`] applies llama.cpp's `LLM_GATING_SOFTMAX` top-k
//! ([`route_with`] the other scoring functions and a selection bias):
//! softmax over all experts, keep the `n_used` largest, optionally renormalise
//! the kept weights to sum to one, then multiply them by `expert_weights_scale`.
//! The result is grouped by expert so each expert runs once on exactly its own
//! tokens. Routing is a pure function of one token's logits: ties go to the
//! lower expert index and tokens stay in order within an expert, so a token's
//! experts and weights do not depend on the batch it travels in.

use super::csa::Gating;
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
    route_with(logits, t, spec, Gating::Softmax, None)
}

/// [`route`] with another scoring function and an optional selection bias.
///
/// - [`Gating::Softmax`]: as [`route`].
/// - [`Gating::Sigmoid`] and [`Gating::SqrtSoftplus`] score every expert on
///   its own (`sigmoid(l)`, `sqrt(softplus(l))` with torch's threshold of 20);
///   with `norm_weights` the kept scores are divided by their sum plus
///   `1e-20`, without it they are used as they are.
///
/// `bias` (one value per expert) steers the selection only: the experts with
/// the largest `score + bias` are kept (lower index on ties), their weights
/// come from the unbiased scores.
pub(crate) fn route_with(
    logits: &[f32],
    t: usize,
    spec: &MoeSpec,
    gating: Gating,
    bias: Option<&[f32]>,
) -> Routing {
    let n = spec.n_expert;
    let mut per_expert = vec![Vec::new(); n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    let mut keys: Vec<f32> = Vec::with_capacity(if bias.is_some() { n } else { 0 });
    for (token, row) in logits.chunks_exact(n).take(t).enumerate() {
        probs.clear();
        let mut total = 0.0;
        match gating {
            Gating::Softmax => {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                probs.extend(row.iter().map(|&l| (l - max).exp()));
                total = probs.iter().sum();
            }
            Gating::Sigmoid => probs.extend(row.iter().map(|&l| 1.0 / (1.0 + (-l).exp()))),
            Gating::SqrtSoftplus => probs.extend(row.iter().map(|&l| softplus(l).sqrt())),
        }
        let selecting: &[f32] = match bias {
            Some(bias) => {
                keys.clear();
                keys.extend(probs.iter().zip(bias).map(|(&p, &b)| p + b));
                &keys
            }
            None => &probs,
        };
        order.clear();
        order.extend(0..n);
        // Largest key first, lower expert index on ties: the order is total,
        // so selecting the top `n_used` and ordering just those equals a full
        // sort.
        let by_key = |&a: &usize, &b: &usize| selecting[b].total_cmp(&selecting[a]).then(a.cmp(&b));
        let k = spec.n_used;
        if 0 < k && k < n {
            order.select_nth_unstable_by(k - 1, by_key);
        }
        order[..k].sort_unstable_by(by_key);
        let kept = &order[..k];
        let denominator = match (gating, spec.norm_weights) {
            (Gating::Softmax, true) => kept.iter().map(|&e| probs[e]).sum::<f32>(),
            (Gating::Softmax, false) => total,
            (_, true) => kept.iter().map(|&e| probs[e]).sum::<f32>() + 1e-20,
            (_, false) => 1.0,
        };
        for &e in kept {
            per_expert[e].push((token as u32, probs[e] / denominator * spec.scale));
        }
    }
    Routing { per_expert }
}

/// `softplus` as `torch.nn.functional.softplus` (beta 1, threshold 20).
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(n_expert: usize, n_used: usize, norm: bool) -> MoeSpec {
        MoeSpec {
            n_expert,
            n_used,
            norm_weights: norm,
            scale: 1.0,
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
    fn the_expert_weights_scale_multiplies_the_final_weights() {
        let logits = [1.0f32, 0.0, 3.0, -1.0];
        let plain = route(&logits, 1, &spec(4, 2, true));
        let scaled = route(
            &logits,
            1,
            &MoeSpec {
                scale: 2.5,
                ..spec(4, 2, true)
            },
        );
        for e in 0..4 {
            for (a, b) in plain.per_expert[e].iter().zip(&scaled.per_expert[e]) {
                assert_eq!(a.0, b.0);
                assert!((a.1 * 2.5 - b.1).abs() < 1e-6);
            }
        }
        // Not renormalised: the scale multiplies the plain softmax weights.
        let raw = route(&logits, 1, &spec(4, 2, false));
        let raw_scaled = route(
            &logits,
            1,
            &MoeSpec {
                scale: 0.5,
                ..spec(4, 2, false)
            },
        );
        assert!((raw.per_expert[2][0].1 * 0.5 - raw_scaled.per_expert[2][0].1).abs() < 1e-7);
    }

    #[test]
    fn tokens_stay_in_order_within_an_expert() {
        let logits = [5.0f32, 0.0, 5.0, 0.0, 5.0, 0.0];
        let r = route(&logits, 3, &spec(2, 1, true));
        assert_eq!(r.per_expert[0], vec![(0, 1.0), (1, 1.0), (2, 1.0)]);
        assert!(r.per_expert[1].is_empty());
    }

    #[test]
    fn sqrt_softplus_scores_each_expert_alone() {
        let logits = [0.0f32, 21.0, -3.0, 1.0];
        let spec = MoeSpec {
            n_expert: 4,
            n_used: 2,
            norm_weights: false,
            scale: 1.0,
        };
        let r = route_with(&logits, 1, &spec, Gating::SqrtSoftplus, None);
        // softplus(0) = ln 2; above the threshold of 20 it is the identity.
        let w = |e: usize| r.per_expert[e][0].1;
        assert!((w(1) - 21f32.sqrt()).abs() < 1e-5);
        assert!((w(3) - 1f32.exp().ln_1p().sqrt()).abs() < 1e-6);
        assert!(r.per_expert[0].is_empty() && r.per_expert[2].is_empty());
    }

    #[test]
    fn a_selection_bias_steers_the_choice_but_not_the_weights() {
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let spec = MoeSpec {
            n_expert: 4,
            n_used: 2,
            norm_weights: true,
            scale: 1.5,
        };
        let r = route_with(
            &logits,
            1,
            &spec,
            Gating::SqrtSoftplus,
            Some(&[0.0, 0.5, 0.0, 0.25]),
        );
        // Experts 1 and 3 win through the bias; equal scores share the
        // routing weight equally (renormalised, times the scale).
        assert!(r.per_expert[0].is_empty() && r.per_expert[2].is_empty());
        assert!((r.per_expert[1][0].1 - 0.75).abs() < 1e-6);
        assert!((r.per_expert[3][0].1 - 0.75).abs() < 1e-6);
    }

    #[test]
    fn equal_keys_go_to_the_lower_expert_and_sigmoid_needs_no_renormalisation() {
        let spec = MoeSpec {
            n_expert: 4,
            n_used: 2,
            norm_weights: false,
            scale: 1.0,
        };
        let r = route_with(&[0.0f32; 4], 1, &spec, Gating::Sigmoid, None);
        assert_eq!(r.per_expert[0], vec![(0, 0.5)]);
        assert_eq!(r.per_expert[1], vec![(0, 0.5)]);
        assert!(r.per_expert[2].is_empty() && r.per_expert[3].is_empty());
    }
}
