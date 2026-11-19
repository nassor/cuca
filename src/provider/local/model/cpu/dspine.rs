//! The DSpine selected-prefix code bridge and online exact-sender tail.
//!
//! A DSpine drafter shares the trunk layout of [`super::dflash`], adds
//! block-causal attention over `[anchor, mask, ..., mask]` with FlashAttention
//! sliding-window semantics (`window_left = w - 1`), applies a learned code
//! bridge after every draft layer (`dspine.blk.*.early_read.weight` and
//! `dspine.read.weight`, `dspine.codebook.weight`, `dspine.gate.*`,
//! `dspine.pred_gate.weight`, `dspine.prev.weight`, and
//! `dspine.blk.*.write.weight`), and runs a 1-row `exact_sender` refinement
//! walk over the top-`head_topk` target head candidates of each step.

use std::sync::{Arc, Mutex, PoisonError};

use candle_core::{Device, Tensor};

use super::super::layout::tensor_info;
use super::super::params::Keys;
use super::super::quant::TokenEmbedding;
use super::super::{candle_err, err, meta_err};
use super::{Loader, QMatMul, Transformer, rms};
use crate::error::CucaError;

/// Clamp floor matching `torch.finfo(torch.float32).tiny` inside the code
/// bridge RMS denominator (`mean(x^2).clamp_min(_F32_TINY).sqrt()`).
const F32_TINY: f32 = f32::MIN_POSITIVE;

const CODEBOOK: &str = "dspine.codebook.weight";
const READ: &str = "dspine.read.weight";
const GATE_WEIGHT: &str = "dspine.gate.weight";
const GATE_BIAS: &str = "dspine.gate.bias";
const PRED_GATE: &str = "dspine.pred_gate.weight";
const PREV: &str = "dspine.prev.weight";

fn early_read_name(il: usize) -> String {
    format!("dspine.blk.{il}.early_read.weight")
}

fn write_name(il: usize) -> String {
    format!("dspine.blk.{il}.write.weight")
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// DSpine metadata parsed from `dflash.dspine.*`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DspineSpec {
    pub(super) code_size: usize,
    pub(super) rank: usize,
    pub(super) eps: f32,
    pub(super) head_topk: usize,
}

/// Read `dflash.dspine.*` if any DSpine key is present, or `None` for plain
/// DFlash. Validates power-of-two `head_topk` in `2..=64` and `block_causal`.
pub(super) fn read_spec(k: &Keys<'_>, block_causal: bool) -> Result<Option<DspineSpec>, CucaError> {
    let keys = [
        "dspine.code_size",
        "dspine.rank",
        "dspine.eps",
        "dspine.head_topk",
    ];
    let present = keys.iter().any(|name| k.value(name).is_some());
    if !present {
        return Ok(None);
    }
    let code_size = k.usize("dspine.code_size")?;
    let rank = k.usize("dspine.rank")?;
    let eps = k.f64("dspine.eps")? as f32;
    let head_topk = k.usize("dspine.head_topk")?;
    if code_size == 0 || rank == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(meta_err(
            "dflash.dspine",
            format!("invalid bridge parameters (code_size={code_size}, rank={rank}, eps={eps})"),
        ));
    }
    if !(2..=64).contains(&head_topk) || !head_topk.is_power_of_two() {
        return Err(meta_err(
            "dflash.dspine.head_topk",
            format!("must be a power of two in 2..=64, got {head_topk}"),
        ));
    }
    if !block_causal {
        return Err(meta_err(
            "dflash.block_causal",
            "a DSpine drafter requires block-causal attention",
        ));
    }
    Ok(Some(DspineSpec {
        code_size,
        rank,
        eps,
        head_topk,
    }))
}

/// Pre-write state kept from the last bridge layer for the refinement walk,
/// covering block rows `1..=k` (every drafted position after the anchor).
pub(super) struct Stash {
    raw: Vec<f32>,
    rms: Vec<f32>,
    q: Vec<f32>,
}

/// Post-layer DSpine code bridge and refinement weights.
pub(super) struct DspineBridge {
    spec: DspineSpec,
    sqrt_c: f32,
    vocab: usize,
    n_embd: usize,
    codebook: TokenEmbedding,
    early_read: Vec<QMatMul>,
    read: QMatMul,
    gate_q: QMatMul,
    gate_bias: Vec<f32>,
    pred_gate: QMatMul,
    prev: QMatMul,
    writes: Vec<QMatMul>,
    /// Reusable `vocab`-wide `-inf` scratch so `draft_tokens` only writes and
    /// restores the `head_topk` candidate slots per step.
    masked_scratch: Mutex<Vec<f32>>,
}

impl DspineBridge {
    /// Load every DSpine bridge weight from the GGUF file in `loader` and
    /// verify shapes against `spec`, `vocab`, `n_embd`, and `n_layer`.
    pub(super) fn load(
        loader: &mut Loader<'_>,
        spec: DspineSpec,
        vocab: usize,
        n_embd: usize,
        n_layer: usize,
    ) -> Result<Self, CucaError> {
        let cb_info = tensor_info(loader.content, CODEBOOK)?;
        let cb_dims = cb_info.shape.dims();
        if cb_dims.len() != 2 || cb_dims[0] != vocab || cb_dims[1] != spec.code_size {
            return Err(err(format!(
                "tensor `{CODEBOOK}` has shape {cb_dims:?}, expected [{vocab}, {}] (target vocab size {vocab})",
                spec.code_size
            )));
        }
        let (cb_dtype, cb_bytes) = loader.raw(CODEBOOK, &[vocab, spec.code_size])?;
        let codebook = TokenEmbedding::new(cb_dtype, Arc::from(cb_bytes), vocab, spec.code_size)?;

        let mut early_read = Vec::with_capacity(n_layer.saturating_sub(1));
        for il in 0..n_layer.saturating_sub(1) {
            early_read.push(loader.matmul(&early_read_name(il), &[spec.code_size, n_embd])?);
        }
        let read = loader.matmul(READ, &[spec.code_size, n_embd])?;
        let gate_q = loader.matmul(GATE_WEIGHT, &[spec.rank, spec.code_size])?;
        let gate_bias = loader
            .vector(GATE_BIAS, spec.rank)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;
        let pred_gate = loader.matmul(PRED_GATE, &[spec.rank, spec.code_size])?;
        let prev = loader.matmul(PREV, &[spec.rank, spec.code_size])?;
        let mut writes = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            writes.push(loader.matmul(&write_name(il), &[n_embd, spec.rank])?);
        }

        Ok(Self {
            spec,
            sqrt_c: (spec.code_size as f32).sqrt(),
            vocab,
            n_embd,
            codebook,
            early_read,
            read,
            gate_q,
            gate_bias,
            pred_gate,
            prev,
            writes,
            masked_scratch: Mutex::new(vec![f32::NEG_INFINITY; vocab]),
        })
    }

    /// Run one code-bridge step on layer `il`'s post-MLP residual `x`
    /// `(1, rows, n_embd)` (`row 0` is `anchor_id`, kept unchanged). When
    /// `il` is the final draft layer, stores pre-write rows `1..rows` in
    /// `stash`.
    pub(super) fn step(
        &self,
        il: usize,
        x: &Tensor,
        anchor_id: u32,
        stash: &mut Option<Stash>,
    ) -> Result<Tensor, CucaError> {
        let dev = &Device::Cpu;
        let (_b, rows, h) = x.dims3().map_err(candle_err)?;
        let raw = x
            .flatten_all()
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;
        let c = self.spec.code_size;
        let r = self.spec.rank;

        let mut rms_vals = vec![0f32; rows];
        let mut normalized = vec![0f32; rows * h];
        for row in 0..rows {
            let src = &raw[row * h..(row + 1) * h];
            let mean_sq = src.iter().map(|v| v * v).sum::<f32>() / (h as f32);
            let r_val = mean_sq.max(F32_TINY).sqrt();
            rms_vals[row] = r_val;
            let inv = 1.0 / (r_val + self.spec.eps);
            let dst = &mut normalized[row * h..(row + 1) * h];
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = v * inv;
            }
        }

        let read_w = if il + 1 == self.writes.len() {
            &self.read
        } else {
            &self.early_read[il]
        };
        let norm_t = Tensor::from_vec(normalized, (rows, h), dev).map_err(candle_err)?;
        let proj = read_w
            .forward(&norm_t)
            .map_err(candle_err)?
            .flatten_all()
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;

        let mut codes = vec![0f32; rows * c];
        let anchor_code = self.codebook.rows(&[anchor_id], self.sqrt_c)?;
        codes[..c].copy_from_slice(&anchor_code);
        for row in 1..rows {
            let prow = &proj[row * c..(row + 1) * c];
            let pnorm = prow.iter().map(|v| v * v).sum::<f32>().sqrt();
            let scale = self.sqrt_c / (pnorm + self.spec.eps);
            let dst = &mut codes[row * c..(row + 1) * c];
            for (d, &v) in dst.iter_mut().zip(prow) {
                *d = v * scale;
            }
        }

        let codes_t = Tensor::from_vec(codes, (rows, c), dev).map_err(candle_err)?;
        let q = self
            .gate_q
            .forward(&codes_t)
            .map_err(candle_err)?
            .flatten_all()
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;
        let s = self
            .pred_gate
            .forward(&codes_t)
            .map_err(candle_err)?
            .flatten_all()
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;
        let v = self
            .prev
            .forward(&codes_t)
            .map_err(candle_err)?
            .flatten_all()
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;

        if il + 1 == self.writes.len() {
            *stash = Some(Stash {
                raw: raw[h..].to_vec(),
                rms: rms_vals[1..].to_vec(),
                q: q[r..].to_vec(),
            });
        }

        let n_write = rows - 1;
        let mut msg = vec![0f32; n_write * r];
        for row in 1..rows {
            let dst = &mut msg[(row - 1) * r..row * r];
            let qr = &q[row * r..(row + 1) * r];
            let sp = &s[(row - 1) * r..row * r];
            let vp = &v[(row - 1) * r..row * r];
            for j in 0..r {
                dst[j] = sigmoid_f32(qr[j] + sp[j] + self.gate_bias[j]) * vp[j];
            }
        }

        let delta = if n_write > 0 {
            let msg_t = Tensor::from_vec(msg, (n_write, r), dev).map_err(candle_err)?;
            self.writes[il]
                .forward(&msg_t)
                .map_err(candle_err)?
                .flatten_all()
                .map_err(candle_err)?
                .to_vec1::<f32>()
                .map_err(candle_err)?
        } else {
            Vec::new()
        };

        let mut out = raw;
        for row in 1..rows {
            let rms_val = rms_vals[row];
            let src = &delta[(row - 1) * h..row * h];
            let dst = &mut out[row * h..(row + 1) * h];
            for (o, &d) in dst.iter_mut().zip(src) {
                *o += rms_val * d;
            }
        }
        Tensor::from_vec(out, (1, rows, h), dev).map_err(candle_err)
    }

    /// Propose up to `k` tokens by running one batched head over `normed`
    /// (`(1, k, n_embd)`) to pick `head_topk` candidates per step, then walking
    /// `s = 0..k` online with exact sender code `codebook[sender]` and
    /// `Transformer::head_indexed` over just `cand[s]`.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn draft_tokens(
        &self,
        target: &Transformer,
        out_norm: &Tensor,
        norm_eps: f32,
        normed: &Tensor,
        stash: &Stash,
        first: u32,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> Option<u32>,
    ) -> Result<Vec<u32>, CucaError> {
        let h = self.n_embd;
        let r = self.spec.rank;
        let mut coarse = target
            .head(normed)
            .map_err(candle_err)?
            .flatten_all()
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;
        let vocab = self.vocab;
        if coarse.len() != k * vocab {
            return Err(err(format!(
                "DSpine head returned {} logits for {k} rows of vocab {vocab}",
                coarse.len()
            )));
        }

        let suppressed = target.suppressed();
        if !suppressed.is_empty() {
            for row in coarse.chunks_exact_mut(vocab) {
                crate::provider::local::tokenizer::mask_suppressed(row, suppressed);
            }
        }

        let kk = self.spec.head_topk.min(vocab);
        let cands: Vec<Vec<u32>> = coarse
            .chunks_exact(vocab)
            .map(|row| topk_ids(row, kk))
            .collect();

        let last_write = self
            .writes
            .last()
            .ok_or_else(|| err("DSpine bridge has no write layer"))?;
        let mut masked = self
            .masked_scratch
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        masked.fill(f32::NEG_INFINITY);

        let dev = &Device::Cpu;
        let mut sender = first;
        let mut drafts = Vec::with_capacity(k);
        let mut msg = vec![0f32; r];

        for (s, cand_ids) in cands.iter().enumerate() {
            let e = self.codebook.rows(&[sender], self.sqrt_c)?;
            let e_t = Tensor::from_vec(e, (1, self.spec.code_size), dev).map_err(candle_err)?;
            let sv = self
                .pred_gate
                .forward(&e_t)
                .map_err(candle_err)?
                .flatten_all()
                .map_err(candle_err)?
                .to_vec1::<f32>()
                .map_err(candle_err)?;
            let vv = self
                .prev
                .forward(&e_t)
                .map_err(candle_err)?
                .flatten_all()
                .map_err(candle_err)?
                .to_vec1::<f32>()
                .map_err(candle_err)?;
            let qrow = &stash.q[s * r..(s + 1) * r];
            for j in 0..r {
                msg[j] = sigmoid_f32(qrow[j] + sv[j] + self.gate_bias[j]) * vv[j];
            }
            let delta = last_write
                .forward(&Tensor::from_slice(&msg, (1, r), dev).map_err(candle_err)?)
                .map_err(candle_err)?
                .flatten_all()
                .map_err(candle_err)?
                .to_vec1::<f32>()
                .map_err(candle_err)?;

            let scale = stash.rms[s];
            let raw_s = &stash.raw[s * h..(s + 1) * h];
            let refined: Vec<f32> = raw_s
                .iter()
                .zip(&delta)
                .map(|(&base, &d)| base + scale * d)
                .collect();
            let norm_row = rms(
                &Tensor::from_vec(refined, (1, 1, h), dev).map_err(candle_err)?,
                out_norm,
                norm_eps,
            )
            .map_err(candle_err)?;
            let scores = target
                .head_indexed(&norm_row, cand_ids)
                .map_err(candle_err)?
                .flatten_all()
                .map_err(candle_err)?
                .to_vec1::<f32>()
                .map_err(candle_err)?;

            for (&id, &sc) in cand_ids.iter().zip(&scores) {
                masked[id as usize] = sc;
            }
            let chosen = pick(&mut masked[..vocab]);
            for &id in cand_ids {
                masked[id as usize] = f32::NEG_INFINITY;
            }
            let Some(tok) = chosen else { break };
            drafts.push(tok);
            sender = tok;
        }
        Ok(drafts)
    }
}

/// Top `k` vocabulary indices descending by logit, ascending id on ties.
fn topk_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    let mut pairs: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as u32))
        .collect();
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, cmp);
        pairs.truncate(k);
    }
    pairs.sort_unstable_by(cmp);
    pairs.into_iter().map(|(_, id)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::local::gguf::{self, Value};
    use crate::provider::local::model::cpu::dflash::DflashNet;
    use crate::provider::local::model::tiny::{
        TinyDflash, TinyDspine, TinySpec, load, write_spec, write_tiny_dflash,
    };
    use crate::provider::local::sampler::{Dist, Scratch, SplitMix64, greedy, sample};

    const CTX: usize = 64;

    /// Reference `serial_table` matching `dspine_mlx/propose.py::RefinementTail.sampled_table`:
    /// builds `[k, kk, kk]` scores `(step s, sender index j, cand index c)`.
    fn serial_table_scores(
        net: &DflashNet,
        target: &Transformer,
        normed: &Tensor,
        stash: &Stash,
        first: u32,
        k: usize,
    ) -> (Vec<Vec<u32>>, Vec<Vec<Vec<f32>>>) {
        let bridge = net.w.bridge.as_ref().unwrap();
        let out_norm = &net.w.out_norm;
        let norm_eps = net.w.geo.eps;
        let h = bridge.n_embd;
        let r = bridge.spec.rank;
        let coarse = target
            .head(normed)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let kk = bridge.spec.head_topk.min(bridge.vocab);
        let cands: Vec<Vec<u32>> = coarse
            .chunks_exact(bridge.vocab)
            .map(|row| topk_ids(row, kk))
            .collect();
        let last_write = bridge.writes.last().unwrap();
        let dev = &Device::Cpu;
        let mut table = Vec::with_capacity(k);
        for s in 0..k {
            let senders = if s == 0 {
                vec![first; kk]
            } else {
                cands[s - 1].clone()
            };
            let mut per_sender = Vec::with_capacity(senders.len());
            for sender in senders {
                let e = bridge.codebook.rows(&[sender], bridge.sqrt_c).unwrap();
                let e_t = Tensor::from_vec(e, (1, bridge.spec.code_size), dev).unwrap();
                let sv = bridge
                    .pred_gate
                    .forward(&e_t)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let vv = bridge
                    .prev
                    .forward(&e_t)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let qrow = &stash.q[s * r..(s + 1) * r];
                let msg: Vec<f32> = (0..r)
                    .map(|j| sigmoid_f32(qrow[j] + sv[j] + bridge.gate_bias[j]) * vv[j])
                    .collect();
                let delta = last_write
                    .forward(&Tensor::from_vec(msg, (1, r), dev).unwrap())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let refined: Vec<f32> = stash.raw[s * h..(s + 1) * h]
                    .iter()
                    .zip(&delta)
                    .map(|(&base, &d)| base + stash.rms[s] * d)
                    .collect();
                let norm_row = rms(
                    &Tensor::from_vec(refined, (1, 1, h), dev).unwrap(),
                    out_norm,
                    norm_eps,
                )
                .unwrap();
                let scores = target
                    .head_indexed(&norm_row, &cands[s])
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                per_sender.push(scores);
            }
            table.push(per_sender);
        }
        (cands, table)
    }

    fn qwen3_and_dspine() -> (Transformer, DflashNet) {
        let target_model = write_spec(&TinySpec {
            context: CTX,
            ..TinySpec::qwen3()
        });
        let target = load(&target_model, CTX);
        let dspine_model = write_tiny_dflash(&TinyDflash {
            block: 8,
            window: None,
            block_causal: true,
            dspine: Some(TinyDspine::for_vocab(target.vocab)),
            ..TinyDflash::default()
        });
        let content = gguf::Content::open(&dspine_model.path).unwrap();
        let mut net = DflashNet::open(&content, &target, CTX).unwrap();
        let feats =
            crate::provider::local::model::tiny::Lcg(91).values(6 * net.feature_width(), 0.75);
        net.observe(0, &feats).unwrap();
        (target, net)
    }

    /// Online `exact_sender` refinement in `draft_tokens` equals `serial_table`
    /// bit-for-bit on both greedy tokens and warped candidate distributions.
    #[test]
    fn online_exact_sender_matches_serial_table_bit_for_bit() {
        let (target, net) = qwen3_and_dspine();
        let k = 6usize;
        let first = 3u32;
        let len = 6usize;

        let bridge = net.w.bridge.as_ref().unwrap();
        let mut ids = vec![net.w.geo.mask_token; k + 1];
        ids[0] = first;
        let emb = target.embedding_rows(&ids).unwrap();
        let mut x = Tensor::from_vec(emb, (1, k + 1, net.w.geo.n_embd), &Device::Cpu).unwrap();
        let mut stash = None;
        for (il, lw) in net.w.layers.iter().enumerate() {
            x = net.block_layer(lw, il, &x, len).unwrap();
            x = bridge.step(il, &x, first, &mut stash).unwrap();
        }
        let stash = stash.unwrap();
        let normed = rms(&x, &net.w.out_norm, net.w.geo.eps)
            .unwrap()
            .narrow(1, 1, k)
            .unwrap();

        let (cands, table) = serial_table_scores(&net, &target, &normed, &stash, first, k);

        // Greedy chain against serial_table.
        let got_greedy = net
            .draft(&target, first, len, k, &mut |l| Some(greedy(l)))
            .unwrap();
        let mut want_greedy = Vec::new();
        let mut j = 0usize;
        for s in 0..k {
            let mut masked = vec![f32::NEG_INFINITY; bridge.vocab];
            for (&id, &sc) in cands[s].iter().zip(&table[s][j]) {
                masked[id as usize] = sc;
            }
            let tok = greedy(&masked);
            want_greedy.push(tok);
            if s + 1 < k {
                j = cands[s].iter().position(|&id| id == tok).unwrap();
            }
        }
        assert_eq!(got_greedy, want_greedy);

        // Stochastic draws and candidate distributions match bit-for-bit.
        let mut rng_a = SplitMix64(17);
        let mut rng_b = SplitMix64(17);
        let mut scratch_a = Scratch::default();
        let mut scratch_b = Scratch::default();
        let mut dists_a = Vec::new();
        let got_stoch = net
            .draft(&target, first, len, k, &mut |l| {
                let d = Dist::warp(l, 0.7, 4, 0.9, &mut scratch_a);
                let tok = d.draw(&mut rng_a);
                dists_a.push(d);
                Some(tok)
            })
            .unwrap();
        let mut want_stoch = Vec::new();
        let mut j = 0usize;
        for s in 0..k {
            let mut masked = vec![f32::NEG_INFINITY; bridge.vocab];
            for (&id, &sc) in cands[s].iter().zip(&table[s][j]) {
                masked[id as usize] = sc;
            }
            let d = Dist::warp(&mut masked, 0.7, 4, 0.9, &mut scratch_b);
            for id in 0..bridge.vocab as u32 {
                assert_eq!(dists_a[s].prob(id).to_bits(), d.prob(id).to_bits());
            }
            let tok = sample(
                &mut {
                    let mut m = vec![f32::NEG_INFINITY; bridge.vocab];
                    for (&id, &sc) in cands[s].iter().zip(&table[s][j]) {
                        m[id as usize] = sc;
                    }
                    m
                },
                0.7,
                4,
                0.9,
                &mut rng_b,
                &mut scratch_b,
            );
            assert_eq!(got_stoch[s], tok);
            want_stoch.push(tok);
            if s + 1 < k {
                j = cands[s].iter().position(|&id| id == tok).unwrap();
            }
        }
        assert_eq!(got_stoch, want_stoch);
    }

    /// Perturbing any block position `p > i` leaves every block position `<= i`
    /// bit-identical across all draft layers and bridge steps.
    #[test]
    fn block_causal_mask_and_bridge_are_strictly_causal() {
        let (target, net) = qwen3_and_dspine();
        let bridge = net.w.bridge.as_ref().unwrap();
        let k = 5usize;
        let len = 6usize;
        let first = 2u32;

        let mut ids = vec![net.w.geo.mask_token; k + 1];
        ids[0] = first;
        let base_emb = target.embedding_rows(&ids).unwrap();
        let h = net.w.geo.n_embd;

        let run_at = |emb: Vec<f32>| -> (Vec<f32>, Stash) {
            let mut x = Tensor::from_vec(emb, (1, k + 1, h), &Device::Cpu).unwrap();
            let mut stash = None;
            for (il, lw) in net.w.layers.iter().enumerate() {
                x = net.block_layer(lw, il, &x, len).unwrap();
                x = bridge.step(il, &x, first, &mut stash).unwrap();
            }
            let out = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            (out, stash.unwrap())
        };

        let (base_out, base_stash) = run_at(base_emb.clone());
        let mut pert_emb = base_emb;
        // Perturb row 4 (index 3 in stash rows 1..=k); rows 0..4 must be bit-identical.
        for d in 0..h {
            pert_emb[4 * h + d] += 3.5;
        }
        let (pert_out, pert_stash) = run_at(pert_emb);

        assert_eq!(base_out[..4 * h], pert_out[..4 * h]);
        assert_eq!(base_stash.raw[..3 * h], pert_stash.raw[..3 * h]);
        assert_eq!(base_stash.rms[..3], pert_stash.rms[..3]);
        let r = bridge.spec.rank;
        assert_eq!(base_stash.q[..3 * r], pert_stash.q[..3 * r]);
        assert_ne!(base_out[4 * h..5 * h], pert_out[4 * h..5 * h]);
    }

    #[test]
    fn invalid_dspine_metadata_is_refused() {
        let target_model = write_spec(&TinySpec {
            context: CTX,
            ..TinySpec::qwen3()
        });
        let target = load(&target_model, CTX);

        let check = |mut meta: Vec<(String, Value)>, needle: &str| {
            let file = write_tiny_dflash(&TinyDflash {
                block_causal: meta
                    .iter()
                    .find(|(k, _)| k == "dflash.block_causal")
                    .and_then(|(_, v)| match v {
                        Value::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(true),
                meta: {
                    meta.retain(|(k, _)| k != "dflash.block_causal");
                    meta
                },
                dspine: Some(TinyDspine::for_vocab(target.vocab)),
                ..TinyDflash::default()
            });
            let content = gguf::Content::open(&file.path).unwrap();
            let err = DflashNet::open(&content, &target, CTX)
                .err()
                .expect("must reject")
                .to_string();
            assert!(err.contains(needle), "{err}");
        };

        check(
            vec![("dflash.dspine.head_topk".into(), Value::U32(3))],
            "power of two",
        );
        check(
            vec![("dflash.block_causal".into(), Value::Bool(false))],
            "block-causal",
        );

        let bad_vocab = write_tiny_dflash(&TinyDflash {
            block_causal: true,
            dspine: Some(TinyDspine::for_vocab(target.vocab + 4)),
            ..TinyDflash::default()
        });
        let content = gguf::Content::open(&bad_vocab.path).unwrap();
        let err = DflashNet::open(&content, &target, CTX)
            .err()
            .expect("vocab mismatch")
            .to_string();
        assert!(err.contains("dspine.codebook.weight"), "{err}");
    }
}
