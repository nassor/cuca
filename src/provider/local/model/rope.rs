//! RoPE frequency math, shared by the CPU and GPU engines (llama.cpp / ggml
//! `ggml_rope_ext` semantics).
//!
//! A [`RopeSpec`] describes one rotary embedding: how pairs of head values are
//! formed ([`RopeKind`]), how many leading head dimensions rotate
//! (`n_dims`, partial rotary), the frequency base, the optional context
//! scaling (linear, YaRN) and the optional per-pair frequency factors (Llama 3
//! `rope_freqs.weight`, Gemma 4 full-attention layers, LongRoPE short/long
//! lists). [`RopeSpec::tables`] turns it into row-major cosine and sine
//! tables, one row per position and one column per rotated pair.
//!
//! For pair `i` (`i0 = 2 * i`) at position `p`, as ggml computes it:
//!
//! ```text
//! theta_extrap = p * base^(-i0 / n_dims) / factor[i]
//! theta_interp = freq_scale * theta_extrap
//! YaRN:  theta = interp * (1 - mix) + extrap * mix,  mix = ramp(i0) * ext_factor
//!        mscale = attn_factor * (1 + 0.1 * ln(1 / freq_scale))
//! else:  theta = theta_interp,  mscale = attn_factor
//! cos, sin = cos(theta) * mscale, sin(theta) * mscale
//! ```
//!
//! A factor of `1e30` rotates by an angle that rounds to exactly zero, so such
//! a pair passes through unchanged.

/// How the values of a head are paired for rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RopeKind {
    /// Adjacent pairs `(2i, 2i + 1)`: llama.cpp `LLAMA_ROPE_TYPE_NORM`
    /// (Llama, SmolLM3, Granite).
    Norm,
    /// Half-split pairs `(i, i + n_dims / 2)`: `LLAMA_ROPE_TYPE_NEOX` (Gemma,
    /// Qwen, Phi 3).
    Neox,
}

/// YaRN extrapolation-mix parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Yarn {
    /// Blend strength; `1.0` is full YaRN.
    pub(crate) ext_factor: f64,
    /// Rotation count above which a dimension is extrapolated (default 32).
    pub(crate) beta_fast: f64,
    /// Rotation count below which a dimension is interpolated (default 1).
    pub(crate) beta_slow: f64,
    /// Context length the checkpoint was originally trained for.
    pub(crate) n_ctx_orig: usize,
}

/// Where a layer's per-pair frequency factors come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Factors {
    /// None: every pair uses its plain frequency.
    None,
    /// The `rope_freqs.weight` tensor.
    Freqs,
    /// LongRoPE: `rope_factors_long.weight` when the configured context
    /// exceeds `n_ctx_orig`, else `rope_factors_short.weight` (llama.cpp picks
    /// by context size, not by position).
    LongShort,
}

/// One rotary embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RopeSpec {
    pub(crate) kind: RopeKind,
    /// Leading head dimensions that rotate; even, at most the head dimension.
    pub(crate) n_dims: usize,
    pub(crate) base: f64,
    /// `1 / scaling factor` for linear and YaRN scaling, else `1.0`.
    pub(crate) freq_scale: f64,
    pub(crate) yarn: Option<Yarn>,
    /// Multiplier on cos and sin (LongRoPE / YaRN attention factor).
    pub(crate) attn_factor: f64,
    pub(crate) factors: Factors,
}

/// Row-major `positions x half` cosine and sine tables.
#[derive(Debug, Clone)]
pub(crate) struct CosSin {
    /// Rotated pairs per position: `n_dims / 2`.
    pub(crate) half: usize,
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
}

/// ggml `rope_yarn_corr_dim`.
fn corr_dim(n_dims: usize, n_ctx_orig: usize, n_rot: f64, base: f64) -> f64 {
    n_dims as f64 * (n_ctx_orig as f64 / (n_rot * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * base.ln())
}

impl Yarn {
    /// ggml `ggml_rope_yarn_corr_dims`: the pair-index range of the ramp.
    fn corr_dims(&self, n_dims: usize, base: f64) -> (f64, f64) {
        let start = corr_dim(n_dims, self.n_ctx_orig, self.beta_fast, base).floor();
        let end = corr_dim(n_dims, self.n_ctx_orig, self.beta_slow, base).ceil();
        (start.max(0.0), end.min(n_dims as f64 - 1.0))
    }
}

impl RopeSpec {
    /// Plain RoPE: no scaling, no factors, no attention factor.
    pub(crate) fn plain(kind: RopeKind, n_dims: usize, base: f64) -> Self {
        Self {
            kind,
            n_dims,
            base,
            freq_scale: 1.0,
            yarn: None,
            attn_factor: 1.0,
            factors: Factors::None,
        }
    }

    /// The rotation angle of pair `i` at position `pos` and the mscale applied
    /// to its cosine and sine.
    fn angle(&self, pos: usize, i: usize, factor: f64, corr: (f64, f64)) -> (f64, f64) {
        let i0 = 2 * i;
        let extrap = pos as f64 * self.base.powf(-(i0 as f64) / self.n_dims as f64) / factor;
        let interp = self.freq_scale * extrap;
        match &self.yarn {
            Some(y) if y.ext_factor != 0.0 => {
                let ramp = 1.0
                    - (((i0 / 2) as f64 - corr.0) / (corr.1 - corr.0).max(0.001)).clamp(0.0, 1.0);
                let mix = ramp * y.ext_factor;
                let mscale = self.attn_factor * (1.0 + 0.1 * (1.0 / self.freq_scale).ln());
                (interp * (1.0 - mix) + extrap * mix, mscale)
            }
            _ => (interp, self.attn_factor),
        }
    }

    /// Cosine and sine tables for `positions` rows. `factors`, when given,
    /// holds one divisor per pair (`n_dims / 2` entries).
    pub(crate) fn tables(&self, factors: Option<&[f32]>, positions: usize) -> CosSin {
        let half = self.n_dims / 2;
        let corr = self
            .yarn
            .map_or((0.0, 0.0), |y| y.corr_dims(self.n_dims, self.base));
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for pos in 0..positions {
            for i in 0..half {
                let factor = factors.map_or(1.0, |f| f64::from(f[i]));
                let (theta, mscale) = self.angle(pos, i, factor, corr);
                cos.push((theta.cos() * mscale) as f32);
                sin.push((theta.sin() * mscale) as f32);
            }
        }
        CosSin { half, cos, sin }
    }
}

/// Rotate one head vector by the direct pair formula (test reference): the
/// first `2 * cos.len()` values rotate under `kind`, the rest pass through.
#[cfg(test)]
pub(crate) fn rotate_reference(kind: RopeKind, x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let half = cos.len();
    let mut out = x.to_vec();
    for i in 0..half {
        let (a, b) = match kind {
            RopeKind::Norm => (2 * i, 2 * i + 1),
            RopeKind::Neox => (i, i + half),
        };
        out[a] = x[a] * cos[i] - x[b] * sin[i];
        out[b] = x[a] * sin[i] + x[b] * cos[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_tables_match_the_definition() {
        let spec = RopeSpec::plain(RopeKind::Neox, 8, 10_000.0);
        let t = spec.tables(None, 5);
        assert_eq!(t.half, 4);
        for pos in 0..5usize {
            for i in 0..4usize {
                let theta = pos as f64 * 10_000f64.powf(-(2.0 * i as f64) / 8.0);
                assert!((f64::from(t.cos[pos * 4 + i]) - theta.cos()).abs() < 1e-6);
                assert!((f64::from(t.sin[pos * 4 + i]) - theta.sin()).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn unit_factors_leave_pairs_unrotated() {
        // 64 pairs at 1.0, 192 at 1e30 (the E2B full-attention layout).
        let mut factors = vec![1.0f32; 64];
        factors.extend(std::iter::repeat_n(1e30f32, 192));
        let spec = RopeSpec::plain(RopeKind::Neox, 512, 1e6);
        let t = spec.tables(Some(&factors), 4);
        for pos in 0..4 {
            for i in 64..256 {
                assert_eq!(t.cos[pos * 256 + i], 1.0, "pos {pos} pair {i}");
                assert!(t.sin[pos * 256 + i].abs() < 1e-12, "pos {pos} pair {i}");
            }
        }
        assert_ne!(t.sin[256], 0.0, "rotated pair 0 at position 1");
    }

    #[test]
    fn linear_scale_divides_the_angle() {
        let mut spec = RopeSpec::plain(RopeKind::Neox, 4, 100.0);
        let base = spec.tables(None, 3);
        spec.freq_scale = 0.125;
        let scaled = spec.tables(None, 3);
        for pos in 0..3usize {
            for i in 0..2usize {
                let theta = pos as f64 * 100f64.powf(-(2.0 * i as f64) / 4.0) * 0.125;
                assert!((f64::from(scaled.cos[pos * 2 + i]) - theta.cos()).abs() < 1e-6);
            }
        }
        assert_ne!(base.cos, scaled.cos);
    }

    #[test]
    fn attn_factor_scales_cos_and_sin() {
        let mut spec = RopeSpec::plain(RopeKind::Neox, 4, 100.0);
        let base = spec.tables(None, 3);
        spec.attn_factor = 1.5;
        let scaled = spec.tables(None, 3);
        for (a, b) in base.cos.iter().zip(&scaled.cos) {
            assert!((a * 1.5 - b).abs() < 1e-6);
        }
        for (a, b) in base.sin.iter().zip(&scaled.sin) {
            assert!((a * 1.5 - b).abs() < 1e-6);
        }
    }

    /// YaRN: the highest-frequency pairs (below the correction range) keep
    /// their extrapolated angle, the lowest-frequency pairs are interpolated by
    /// `freq_scale`, and mscale gains `1 + 0.1 ln(1 / freq_scale)`.
    #[test]
    fn yarn_blends_between_extrapolation_and_interpolation() {
        let n_dims = 64;
        let factor = 4.0;
        let spec = RopeSpec {
            kind: RopeKind::Neox,
            n_dims,
            base: 10_000.0,
            freq_scale: 1.0 / factor,
            yarn: Some(Yarn {
                ext_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                n_ctx_orig: 4096,
            }),
            attn_factor: 1.0,
            factors: Factors::None,
        };
        let t = spec.tables(None, 100);
        let mscale = 1.0 + 0.1 * factor.ln();
        let pos = 99usize;
        // Pair 0 rotates fastest: fully extrapolated (ramp = 1).
        let theta0 = pos as f64;
        assert!((f64::from(t.cos[pos * 32]) - theta0.cos() * mscale).abs() < 1e-5);
        // Last pair: interpolated (ramp = 0).
        let theta_last = pos as f64 * 10_000f64.powf(-62.0 / 64.0) / factor;
        assert!((f64::from(t.cos[pos * 32 + 31]) - theta_last.cos() * mscale).abs() < 1e-5);
    }
}
