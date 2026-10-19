//! The multi-token-prediction head embedded in a trunk checkpoint.
//!
//! Qwen 3.5 `*-MTP` GGUF files carry, after the trunk blocks, one more block
//! (`blk.{n_layer}.*`): a full gated-attention block with its own attention
//! and SwiGLU tensors, plus four tensors that turn it into a head,
//! `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` and `nextn.shared_head_norm`.
//! It predicts the token *after* the one the trunk predicts, from the trunk's
//! hidden state and the token the trunk just predicted:
//!
//! 1. For a trunk position `j` with its final-norm hidden state `h[j]` (the
//!    row that feeds the LM head: [`Hidden::PostNorm`](super::Hidden); the
//!    pre-norm state works much worse, 59 % against 95 % teacher-forced
//!    top-1 on long prompts) and the token `t[j + 1]` that follows it, the
//!    head input is `x = eh_proj(cat(enorm(embed(t[j + 1])), hnorm(h[j])))`
//!    (embedding first; `embed` is the plain `token_embd` row).
//! 2. `x` runs through the block like a trunk attention layer would (same
//!    code: [`Transformer::block`]), with RoPE position `j + 1`, attending
//!    over *every earlier head input* held in the head's own K/V cache.
//! 3. The output goes through `nextn.shared_head_norm` and the shared LM head
//!    (the tied `token_embd`): logits for the token `t[j + 2]`.
//!
//! The head attends over all earlier inputs, so its K/V cache needs an entry
//! for every trunk position, not only for the positions being drafted from.
//! Because the head has a single layer, an entry depends on its own input
//! only (`K`, `V` of `attn_norm(x)`): committing a *pair* `(h[j], t[j + 1])`
//! costs the input projection and two K/V projections, no attention, no FFN
//! and no LM head ([`Transformer::head_observe`]).
//!
//! # Cache
//!
//! [`NextnHead`] holds one K/V slot per committed pair, in ascending pair
//! index `j` (slots are contiguous, pair indices may skip: a skipped pair is a
//! position the head cannot see, which costs a worse proposal and never
//! changes output), plus the *tail*: the hidden row of the last observed
//! position, which still waits for its successor token. Observing
//! `tokens[i]`/`hidden[i]` rows at positions `pos..` ([`Transformer::head_observe`])
//! pairs every row with its successor, the tail with `tokens[0]` when it sits
//! at `pos - 1`, and makes the last row the new tail; it drops every pair
//! with an index at or above its first new one, so re-observing replayed
//! rows replaces what they supersede. [`NextnHead::truncate`] drops the pairs
//! and the tail that a changed token invalidates (see there); [`NextnHead::reset`]
//! empties everything. Drafting writes its steps into slots *past* the
//! committed ones and never commits them: the next `observe` (which the
//! engine calls after every forward pass, drafted or not) overwrites them.
//!
//! # Chain
//!
//! Step 1 of a draft chain uses the pair `(h[len - 1], pending token)` at
//! RoPE position `len`. Step `s > 1` uses the pair (the previous step's
//! output hidden, the token step `s - 1` drafted) at position `len + s - 1`,
//! attending over the committed entries and the earlier steps' temporary
//! ones.
//!
//! The hidden row a step hands to the next is the head's output *after*
//! `nextn.shared_head_norm` (the row its own logits come from, which then
//! goes through `hnorm` like a trunk row does), not the block output before
//! that norm. Measured on Qwen3.5-0.8B over seven texts (Python, Rust, a
//! list, prose, facts and two chat prompts, each continued by 128 greedy
//! tokens: 868 chains of four steps from every position of the
//! continuation), the conditional acceptance by draft position is 88.7 / 84.5
//! / 86.2 / 87.0 % with the normed output and 88.7 / 33.1 / 39.2 / 21.0 %
//! with the output before the norm (3.85 against 2.32 expected tokens per
//! round at four drafts). Step 1 is the same either way: it reads the
//! trunk's own final-norm row.
//!
//! # Proposal vocabulary
//!
//! The head's LM head is the trunk's: 248,320 rows of Q8_0 (270 MB) on the
//! 0.8B model, so a whole draft step measures 5.4 ms with it. A draft is only a proposal, so a
//! step ranks just the first [`PROPOSAL_ROWS`] token ids: byte-level BPE
//! vocabularies number their pieces in merge order, which puts the frequent
//! ones first (a tokenizer-specific heuristic; it only affects proposal quality). With it a step takes 1.05 ms and, on the same 868 chains, 84.1 /
//! 82.9 / 86.0 / 86.0 % conditional acceptance (3.14 against 3.28 expected
//! tokens per round at three drafts). Tokens beyond (rare pieces, the
//! end-of-generation and other special tokens) are never proposed. The logits
//! a step passes to `pick` cover those ids only, so an index is a token id;
//! drafting from a distribution over a subset keeps speculative sampling
//! exact, because the acceptance rule uses the distribution the draft was
//! really drawn from and resamples a rejection from `max(0, p - q)`.
//!
//! # Memory
//!
//! The head's weights stay quantized as in the file: 21.7 MB for
//! Qwen3.5-0.8B Q8_0 (attention 7.8 MB, FFN 11.7 MB, `eh_proj` 2.2 MB); the
//! token embedding and LM head are the trunk's. The K/V cache is preallocated
//! for `context_length` slots of `2 * 4 * n_kv_head * head_dim` bytes (4 KB
//! per position for the 0.8B model: 33.5 MB at 8192 positions) and the
//! bookkeeping is 8 bytes per slot plus one hidden row (4 KB).

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use super::super::layout::{LayerLayout, name};
use super::super::params::LayerSpec;
use super::super::{candle_err, err};
use super::{AttnCall, LayerWeights, Loader, Mixer, QMatMul, Transformer, load_layer, rms};
use crate::error::CucaError;

/// Token ids a draft step ranks (the first ones of the vocabulary): see
/// "Proposal vocabulary" in the [module docs](self).
const PROPOSAL_ROWS: usize = 65_536;

/// A multi-token-prediction head and its K/V cache; see the [module
/// docs](self).
pub(crate) struct NextnHead {
    /// `nextn.eh_proj`: `2 * n_embd -> n_embd`.
    eh_proj: QMatMul,
    enorm: Tensor,
    hnorm: Tensor,
    shared_head_norm: Tensor,
    /// The block's attention and FFN weights.
    block: LayerWeights,
    spec: LayerSpec,
    /// K and V `(1, n_kv_head, context_length, head_dim)`: slot `s` holds the
    /// entry of pair `pairs[s]`; slots past `pairs.len()` are scratch.
    kv: (Tensor, Tensor),
    capacity: usize,
    /// Rows of the LM head the draft steps rank: the first `vocab_rows` token
    /// ids.
    vocab_rows: usize,
    /// The pair index `j` (the pair `(h[j], t[j + 1])`) of every committed
    /// slot, strictly ascending.
    pairs: Vec<usize>,
    /// Position and final-norm hidden row of the last observed position,
    /// awaiting the token that follows it.
    tail: Option<(usize, Vec<f32>)>,
}

impl NextnHead {
    /// Number of committed pairs (K/V slots in use).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Forget every pair and the tail.
    pub(crate) fn reset(&mut self) {
        self.pairs.clear();
        self.tail = None;
    }

    /// The first `shared` tokens of the sequence stay as they were and every
    /// later one may change: keep the pairs `(h[j], t[j + 1])` with
    /// `j + 1 < shared` (the tokens and so the hidden rows they depend on are
    /// unchanged) and the tail when its position is below `shared`.
    pub(crate) fn truncate(&mut self, shared: usize) {
        let keep = self.pairs.partition_point(|&j| j + 1 < shared);
        self.pairs.truncate(keep);
        if self.tail.as_ref().is_some_and(|(at, _)| *at >= shared) {
            self.tail = None;
        }
    }

    /// Drop the pairs with an index of at least `first`.
    fn drop_from(&mut self, first: usize) {
        let keep = self.pairs.partition_point(|&j| j < first);
        self.pairs.truncate(keep);
    }
}

impl Transformer {
    /// Load the multi-token-prediction head of this model from the GGUF file
    /// at `path` (index `content`): the first prediction block, block
    /// `n_layer`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the file declares no prediction block or
    /// lacks one of its tensors.
    pub(crate) fn load_head(
        &self,
        content: &gguf_file::Content,
        path: &Path,
    ) -> Result<NextnHead, CucaError> {
        self.load_head_with(content, path, false)
    }

    /// [`Transformer::load_head`], with every matmul weight dequantized to
    /// f32 when `dequantize` is set (the test reference).
    pub(crate) fn load_head_with(
        &self,
        content: &gguf_file::Content,
        path: &Path,
        dequantize: bool,
    ) -> Result<NextnHead, CucaError> {
        let hp = &self.hp;
        let Some(spec) = &hp.nextn else {
            return Err(err(format!(
                "this `{arch}` file has no in-trunk multi-token-prediction head (`{arch}.nextn_predict_layers` is {n}, or the architecture embeds none); use a separate drafter file instead",
                arch = hp.arch.name(),
                n = hp.nextn_predict_layers
            )));
        };
        let il = hp.n_layer;
        let n_embd = hp.n_embd;
        let layout = LayerLayout::read(hp, il, spec, content)?;
        let mut loader = Loader {
            content,
            reader: BufReader::new(File::open(path)?),
            dequantize,
            requantize: false,
        };
        let block = load_layer(&mut loader, hp, il, spec, &layout)?;
        if !matches!(block.mixer, Mixer::Attention(_)) {
            return Err(err(
                "the multi-token-prediction block must be an attention block",
            ));
        }
        let blk = |part: &str| name::blk(il, part);
        let eh_proj = loader.matmul(&blk(name::NEXTN_EH_PROJ), &[n_embd, 2 * n_embd])?;
        let enorm = loader.vector(&blk(name::NEXTN_ENORM), n_embd)?;
        let hnorm = loader.vector(&blk(name::NEXTN_HNORM), n_embd)?;
        let shared_head_norm = loader.vector(&blk(name::NEXTN_SHARED_HEAD_NORM), n_embd)?;
        let capacity = self.context_length;
        let shape = (1, spec.n_kv_head, capacity, spec.head_dim);
        let zeros = || Tensor::zeros(shape, DType::F32, &Device::Cpu).map_err(candle_err);
        Ok(NextnHead {
            eh_proj,
            enorm,
            hnorm,
            shared_head_norm,
            block,
            spec: spec.clone(),
            kv: (zeros()?, zeros()?),
            capacity,
            vocab_rows: self.vocab.min(PROPOSAL_ROWS),
            pairs: Vec::new(),
            tail: None,
        })
    }

    /// The embedding rows of `ids`, concatenated (scaled as the first trunk
    /// layer sees them).
    fn embedding_rows(&self, ids: &[u32]) -> Result<Vec<f32>, CucaError> {
        self.embed
            .as_ref()
            .ok_or_else(|| err("this model has no token embedding"))?
            .rows(ids, self.hp.embd_scale)
    }

    /// The head inputs `eh_proj(cat(enorm(emb), hnorm(hidden)))` of `m` pairs:
    /// `emb` holds the `m` embedding rows and `hidden` the `m` final-norm
    /// hidden rows, `(1, m, n_embd)` each.
    fn head_input(
        &self,
        head: &NextnHead,
        emb: Vec<f32>,
        hidden: &[f32],
    ) -> candle_core::Result<Tensor> {
        let d = self.hp.n_embd;
        let m = emb.len() / d;
        let eps = self.hp.eps as f32;
        let dev = &Device::Cpu;
        let e = rms(&Tensor::from_vec(emb, (1, m, d), dev)?, &head.enorm, eps)?;
        let h = rms(
            &Tensor::from_vec(hidden.to_vec(), (1, m, d), dev)?,
            &head.hnorm,
            eps,
        )?;
        head.eh_proj.forward(&Tensor::cat(&[&e, &h], 2)?)
    }

    /// Commit the trunk rows `tokens[i]` at positions `pos + i`, whose
    /// final-norm hidden rows are `hidden` (`tokens.len() * n_embd` floats):
    /// every row is paired with its successor token, the tail with
    /// `tokens[0]` when it sits at `pos - 1`, and the last row becomes the
    /// tail. Pairs with an index at or above the first new one are replaced.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for mismatched lengths, a full cache or a
    /// failed projection.
    pub(crate) fn head_observe(
        &self,
        head: &mut NextnHead,
        pos: usize,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), CucaError> {
        let d = self.hp.n_embd;
        let n = tokens.len();
        if n == 0 || hidden.len() != n * d {
            return Err(err(format!(
                "head observe: {n} tokens with {} hidden floats, expected {}",
                hidden.len(),
                n * d
            )));
        }
        // Pair `(h[j], t[j + 1])` for `j` from `first`.
        let (first, rows, ids) = match head.tail.as_ref().filter(|(at, _)| at + 1 == pos) {
            Some((_, prev)) => {
                let mut rows = Vec::with_capacity(n * d);
                rows.extend_from_slice(prev);
                rows.extend_from_slice(&hidden[..(n - 1) * d]);
                (pos - 1, rows, tokens)
            }
            None => (pos, hidden[..(n - 1) * d].to_vec(), &tokens[1..]),
        };
        head.drop_from(first);
        let m = ids.len();
        if m > 0 {
            let slot = head.pairs.len();
            if slot + m > head.capacity {
                return Err(err(format!(
                    "the head cache holds {} positions, {} do not fit",
                    head.capacity,
                    slot + m
                )));
            }
            let emb = self.embedding_rows(ids)?;
            self.head_store(head, emb, &rows, slot, first)
                .map_err(candle_err)?;
            head.pairs.extend(first..first + m);
        }
        head.tail = Some((pos + n - 1, hidden[(n - 1) * d..].to_vec()));
        Ok(())
    }

    /// Store the K/V entries of `emb.len() / n_embd` pairs in slots `slot..`,
    /// the first pair being `first`.
    fn head_store(
        &self,
        head: &NextnHead,
        emb: Vec<f32>,
        hidden: &[f32],
        slot: usize,
        first: usize,
    ) -> candle_core::Result<()> {
        let Mixer::Attention(aw) = &head.block.mixer else {
            return Err(candle_core::Error::msg("the head block is not attention"));
        };
        let spec = &head.spec;
        let m = emb.len() / self.hp.n_embd;
        let x = self.head_input(head, emb, hidden)?;
        let h = match &head.block.attn_norm {
            Some(w) => rms(&x, w, self.hp.eps as f32)?,
            None => x.clone(),
        };
        let (k, v) = aw.project_kv(&h, spec.n_head, spec.n_kv_head, spec.head_dim)?;
        let call = AttnCall {
            spec,
            start: slot,
            rope_start: first + 1,
            t: m,
            own: Some(&head.kv),
            read: Some(&head.kv),
            ext: None,
        };
        self.store_kv(&call, aw, &k, &v)
    }

    /// Draft `k` tokens after the pending token `first`: `hidden` is the
    /// trunk's final-norm hidden row at its last cached position and `len`
    /// the number of cached positions (`first` sits at position `len`, not
    /// cached). Each step's logits go to `pick`, which chooses the token.
    /// Nothing is committed: see the [module docs](self).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a bad `hidden` row, a full cache or a
    /// failed forward step.
    pub(crate) fn head_draft(
        &self,
        head: &NextnHead,
        hidden: &[f32],
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> u32,
    ) -> Result<Vec<u32>, CucaError> {
        if len == 0 || hidden.len() != self.hp.n_embd {
            return Err(err("head draft: needs the hidden row of the last position"));
        }
        let slot = head.pairs.len();
        if slot + k > head.capacity {
            return Err(err(format!(
                "the head cache holds {} positions, {} do not fit",
                head.capacity,
                slot + k
            )));
        }
        let mut drafts = Vec::with_capacity(k);
        let mut hidden = hidden.to_vec();
        let mut token = first;
        for step in 0..k {
            let emb = self.embedding_rows(&[token])?;
            let (mut logits, next) = self
                .head_step(head, emb, &hidden, slot + step, len - 1 + step)
                .map_err(candle_err)?;
            token = pick(&mut logits);
            drafts.push(token);
            hidden = next;
        }
        Ok(drafts)
    }

    /// One draft step: the pair `(hidden, embedded token)` of pair index `j`
    /// goes into slot `slot`; returns the logits of the following token and
    /// the hidden row the next step takes.
    fn head_step(
        &self,
        head: &NextnHead,
        emb: Vec<f32>,
        hidden: &[f32],
        slot: usize,
        j: usize,
    ) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let Mixer::Attention(aw) = &head.block.mixer else {
            return Err(candle_core::Error::msg("the head block is not attention"));
        };
        let x = self.head_input(head, emb, hidden)?;
        let call = AttnCall {
            spec: &head.spec,
            start: slot,
            rope_start: j + 1,
            t: 1,
            own: Some(&head.kv),
            read: Some(&head.kv),
            ext: None,
        };
        let y = self.block(&head.block, &x, None, 1, |h| {
            self.attention_in(&call, aw, h)
        })?;
        let normed = rms(&y, &head.shared_head_norm, self.hp.eps as f32)?;
        let logits = self.head_rows(&normed, head.vocab_rows)?;
        Ok((
            logits.flatten_all()?.to_vec1::<f32>()?,
            normed.flatten_all()?.to_vec1::<f32>()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::hybrid_tests::{content_of, load_dequantized, matvec, tensor};
    use super::*;
    use crate::provider::local::model::Want;
    use crate::provider::local::model::cpu::Hidden;
    use crate::provider::local::model::tiny::{
        CONTEXT, Lcg, N_EMBD, Q35_FF, Q35_HEAD, Q35_N_HEAD, Q35_N_KV, Q35_ROPE_DIMS, TinyModel,
        TinyQwen35, VOCAB, load, write_qwen35,
    };
    use crate::provider::local::sampler::greedy;

    const D: usize = N_EMBD;
    /// The tiny model's prediction block (five trunk layers, then the head).
    const HEAD_BLOCK: usize = 5;

    fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    /// The trunk and its head, every matmul weight dequantized to f32.
    fn dequantized(model: &TinyModel) -> (Transformer, NextnHead) {
        let m = load_dequantized(model);
        let head = m
            .load_head_with(&content_of(model), &model.path, true)
            .unwrap();
        (m, head)
    }

    /// The trunk and its head as the engine loads them (quantized).
    fn production(model: &TinyModel) -> (Transformer, NextnHead) {
        let m = load(model, CONTEXT);
        let head = m.load_head(&content_of(model), &model.path).unwrap();
        (m, head)
    }

    /// `n` token ids inside the tiny vocabulary and `n` synthetic hidden rows.
    fn rows(n: usize, seed: u64) -> (Vec<u32>, Vec<f32>) {
        let tokens = (0..n).map(|i| (i * 5 % 13 + 3) as u32).collect();
        (tokens, Lcg(seed).values(n * D, 1.0))
    }

    /// The logits of a draft chain of `k` steps from the state `head` holds.
    fn chain(
        m: &Transformer,
        head: &NextnHead,
        h: &[f32],
        first: u32,
        len: usize,
        k: usize,
    ) -> Vec<Vec<f32>> {
        let mut seen = Vec::new();
        m.head_draft(head, h, first, len, k, &mut |l: &mut [f32]| {
            seen.push(l.to_vec());
            greedy(l)
        })
        .unwrap();
        seen
    }

    /// Direct f64 evaluation of the head. `pairs[p] = (hidden row, token)` is
    /// the pair with index `first + p` (RoPE position `first + p + 1`); the
    /// block runs on the last pair and attends over all of them. Returns the
    /// logits and the normed output (what the next chain step takes).
    fn reference(
        model: &TinyModel,
        pairs: &[(Vec<f64>, u32)],
        first: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let w = |part: &str| tensor(model, &name::blk(HEAD_BLOCK, part));
        let embd = tensor(model, name::TOKEN_EMBD);
        let (nh, nkv, d, rd) = (Q35_N_HEAD, Q35_N_KV, Q35_HEAD, Q35_ROPE_DIMS);
        let eps = 1e-6f64;
        let rms = |v: &[f64], weight: &[f64]| -> Vec<f64> {
            let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64 + eps).sqrt();
            v.iter().zip(weight).map(|(x, w)| x * inv * w).collect()
        };
        let rotate = |v: &mut [f64], pos: usize| {
            for i in 0..rd / 2 {
                let angle = pos as f64 * 1e7f64.powf(-(2.0 * i as f64) / rd as f64);
                let (c, s) = (angle.cos(), angle.sin());
                let (a, b) = (v[i], v[i + rd / 2]);
                v[i] = a * c - b * s;
                v[i + rd / 2] = b * c + a * s;
            }
        };
        let (enorm, hnorm) = (w(name::NEXTN_ENORM), w(name::NEXTN_HNORM));
        let (attn_norm, ffn_norm) = (w(name::ATTN_NORM), w(name::POST_ATTENTION_NORM));
        let (wq, wk, wv, wo) = (
            w(name::ATTN_Q),
            w(name::ATTN_K),
            w(name::ATTN_V),
            w(name::ATTN_OUTPUT),
        );
        let (qn, kn) = (w(name::ATTN_Q_NORM), w(name::ATTN_K_NORM));
        let eh = w(name::NEXTN_EH_PROJ);

        let (mut xs, mut ks, mut vs) = (Vec::new(), Vec::new(), Vec::new());
        for (p, (h, token)) in pairs.iter().enumerate() {
            let row = &embd[*token as usize * D..(*token as usize + 1) * D];
            let mut cat = rms(row, &enorm);
            cat.extend(rms(h, &hnorm));
            let x = matvec(&eh, 2 * D, &cat);
            let a = rms(&x, &attn_norm);
            let (k, v) = (matvec(&wk, D, &a), matvec(&wv, D, &a));
            let mut key = vec![0f64; nkv * d];
            for head in 0..nkv {
                let mut kh = rms(&k[head * d..(head + 1) * d], &kn);
                rotate(&mut kh, first + p + 1);
                key[head * d..(head + 1) * d].copy_from_slice(&kh);
            }
            xs.push(x);
            ks.push(key);
            vs.push(v);
        }
        let last = pairs.len() - 1;
        let x = &xs[last];
        let packed = matvec(&wq, D, &rms(x, &attn_norm));
        let mut o = vec![0f64; nh * d];
        for head in 0..nh {
            let at = head * 2 * d;
            let mut q = rms(&packed[at..at + d], &qn);
            rotate(&mut q, first + last + 1);
            let kv = head / (nh / nkv);
            let scores: Vec<f64> = (0..=last)
                .map(|j| {
                    let k = &ks[j][kv * d..(kv + 1) * d];
                    q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() / (d as f64).sqrt()
                })
                .collect();
            let top = scores.iter().fold(f64::MIN, |a, &b| a.max(b));
            let exps: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                for c in 0..d {
                    o[head * d + c] += e / sum * vs[j][kv * d + c];
                }
            }
            for c in 0..d {
                let g = packed[at + d + c];
                o[head * d + c] *= 1.0 / (1.0 + (-g).exp());
            }
        }
        let attn = matvec(&wo, nh * d, &o);
        let x1: Vec<f64> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let f = rms(&x1, &ffn_norm);
        let (g, u) = (
            matvec(&w(name::FFN_GATE), D, &f),
            matvec(&w(name::FFN_UP), D, &f),
        );
        let mid: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let ffn = matvec(&w(name::FFN_DOWN), Q35_FF, &mid);
        let y: Vec<f64> = x1.iter().zip(&ffn).map(|(a, b)| a + b).collect();
        let normed = rms(&y, &w(name::NEXTN_SHARED_HEAD_NORM));
        (matvec(&embd, D, &normed), normed)
    }

    /// The head's draft chain equals a direct f64 evaluation of the math: the
    /// input `eh_proj(cat(enorm(embed(token)), hnorm(hidden)))` (embedding
    /// first), the gated-attention block over the K/V of every earlier pair
    /// (RoPE position `j + 1`), the shared head norm, the tied LM head, and
    /// the normed output as the next step's hidden row.
    #[test]
    fn draft_chain_matches_an_f64_reference() {
        let model = write_qwen35(&TinyQwen35::default());
        let (m, mut head) = dequantized(&model);
        let n = 7;
        let (tokens, hidden) = rows(n, 41);
        m.head_observe(&mut head, 0, &tokens, &hidden).unwrap();
        // Pairs 0..n - 1 are committed, the last row is the tail.
        assert_eq!(head.len(), n - 1);
        let first = 11u32;
        let last_hidden = &hidden[(n - 1) * D..];
        let logits = chain(&m, &head, last_hidden, first, n, 3);
        assert_eq!(logits.len(), 3);

        let mut pairs: Vec<(Vec<f64>, u32)> = (0..n - 1)
            .map(|j| (f64s(&hidden[j * D..(j + 1) * D]), tokens[j + 1]))
            .collect();
        let (mut h, mut token) = (f64s(last_hidden), first);
        for (step, got) in logits.iter().enumerate() {
            pairs.push((h, token));
            let (want, normed) = reference(&model, &pairs, 0);
            assert_eq!(got.len(), VOCAB);
            let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(*g) - w).abs() <= 2e-4 * scale,
                    "step {step} logit {i}: {g} vs {w}"
                );
            }
            token = greedy(got);
            let best = want
                .iter()
                .enumerate()
                .fold((0, f64::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            assert_eq!(token as usize, best.0, "step {step}");
            h = normed;
        }
    }

    /// Observing in any chunking commits the same pairs, and truncating to
    /// the shared prefix and re-observing from where the trunk resumes
    /// reproduces a cold head bit for bit.
    #[test]
    fn observe_is_chunk_invariant_and_truncate_plus_replay_equals_cold() {
        let model = write_qwen35(&TinyQwen35::default());
        let (m, mut cold) = production(&model);
        let n = 9;
        let (tokens, hidden) = rows(n, 7);
        let (first, last) = (4u32, &hidden[(n - 1) * D..]);
        m.head_observe(&mut cold, 0, &tokens, &hidden).unwrap();
        assert_eq!(cold.len(), n - 1);
        let want = chain(&m, &cold, last, first, n, 3);

        // Chunks of 2, 3 and 4 rows: the tail carries the pair across calls.
        let (_, mut chunked) = production(&model);
        for (from, to) in [(0, 2), (2, 5), (5, 9)] {
            m.head_observe(
                &mut chunked,
                from,
                &tokens[from..to],
                &hidden[from * D..to * D],
            )
            .unwrap();
        }
        assert_eq!(chunked.len(), n - 1);
        assert_eq!(chain(&m, &chunked, last, first, n, 3), want);

        // The first six tokens stay, the trunk replays from position 4 (it
        // could only return there): pairs 0..=4 hold, the replay re-adds the
        // rest and the result is the cold head.
        let (_, mut replayed) = production(&model);
        m.head_observe(&mut replayed, 0, &tokens, &hidden).unwrap();
        replayed.truncate(6);
        assert_eq!(replayed.len(), 5);
        m.head_observe(&mut replayed, 4, &tokens[4..], &hidden[4 * D..])
            .unwrap();
        assert_eq!(replayed.len(), n - 1);
        assert_eq!(chain(&m, &replayed, last, first, n, 3), want);

        // A trunk that could resume only at the shared length leaves out the
        // pair that needs the hidden row it no longer has: one position
        // fewer for the head to see, nothing else.
        let (_, mut gap) = production(&model);
        m.head_observe(&mut gap, 0, &tokens, &hidden).unwrap();
        gap.truncate(5);
        m.head_observe(&mut gap, 5, &tokens[5..], &hidden[5 * D..])
            .unwrap();
        assert_eq!(gap.len(), n - 2);
        assert_eq!(chain(&m, &gap, last, first, n, 3).len(), 3);

        // The whole sequence is shared: the tail survives and bridges the
        // next call.
        let (_, mut same) = production(&model);
        m.head_observe(&mut same, 0, &tokens, &hidden).unwrap();
        same.truncate(n);
        assert_eq!(same.len(), n - 1);
        m.head_observe(&mut same, n, &tokens[..1], &hidden[..D])
            .unwrap();
        assert_eq!(same.len(), n);

        // Reset forgets the tail as well.
        same.reset();
        assert_eq!(same.len(), 0);
        m.head_observe(&mut same, 3, &tokens[..2], &hidden[..2 * D])
            .unwrap();
        assert_eq!(same.len(), 1);
    }

    /// Drafting is deterministic, commits nothing, and its temporary steps
    /// never leak into what the next observation commits.
    #[test]
    fn drafting_is_deterministic_and_commits_nothing() {
        let model = write_qwen35(&TinyQwen35::default());
        let (m, mut head) = production(&model);
        let n = 6;
        let (tokens, hidden) = rows(n, 3);
        let last = &hidden[(n - 1) * D..];
        m.head_observe(&mut head, 0, &tokens, &hidden).unwrap();
        let a = chain(&m, &head, last, 5, n, 4);
        assert_eq!(head.len(), n - 1);
        assert_eq!(chain(&m, &head, last, 5, n, 4), a);
        // A shorter chain is a prefix of the longer one.
        assert_eq!(chain(&m, &head, last, 5, n, 2), a[..2]);

        // After drafting, commit more true pairs: the head equals one that
        // never drafted.
        let (_, mut clean) = production(&model);
        m.head_observe(&mut clean, 0, &tokens, &hidden).unwrap();
        let (more_tokens, more_hidden) = rows(3, 99);
        for h in [&mut head, &mut clean] {
            m.head_observe(h, n, &more_tokens, &more_hidden).unwrap();
        }
        let next = &more_hidden[2 * D..];
        assert_eq!(
            chain(&m, &head, next, 7, n + 3, 3),
            chain(&m, &clean, next, 7, n + 3, 3)
        );

        // Bad input is an error, not a panic.
        assert!(m.head_observe(&mut head, 0, &[], &[]).is_err());
        assert!(m.head_observe(&mut head, 0, &tokens, &hidden[1..]).is_err());
        let mut pick = |l: &mut [f32]| greedy(l);
        assert!(
            m.head_draft(&head, &hidden[..D - 1], 3, n, 1, &mut pick)
                .is_err()
        );
        assert!(m.head_draft(&head, last, 3, 0, 1, &mut pick).is_err());
    }

    /// A draft step may rank only a prefix of the vocabulary: its logits are
    /// the first entries of the full ones bit for bit, and every token it
    /// proposes comes from the prefix.
    #[test]
    fn a_proposal_prefix_ranks_the_first_token_ids_only() {
        let model = write_qwen35(&TinyQwen35::default());
        let (m, mut full) = production(&model);
        let (_, mut part) = production(&model);
        assert_eq!(full.vocab_rows, VOCAB);
        part.vocab_rows = 5;
        let n = 6;
        let (tokens, hidden) = rows(n, 12);
        for head in [&mut full, &mut part] {
            m.head_observe(head, 0, &tokens, &hidden).unwrap();
        }
        let last = &hidden[(n - 1) * D..];
        let whole = chain(&m, &full, last, 4, n, 1);
        let prefix = chain(&m, &part, last, 4, n, 1);
        assert_eq!(prefix[0].len(), 5);
        assert_eq!(prefix[0][..], whole[0][..5]);
        let mut drafted = Vec::new();
        m.head_draft(&part, last, 4, n, 4, &mut |l: &mut [f32]| {
            let t = greedy(l);
            drafted.push(t);
            t
        })
        .unwrap();
        assert_eq!(drafted.len(), 4);
        assert!(drafted.iter().all(|&t| t < 5), "{drafted:?}");
    }

    /// A cache that cannot hold the entries is an error naming the limit.
    #[test]
    fn a_full_head_cache_is_an_error() {
        let model = write_qwen35(&TinyQwen35::default());
        let (m, mut head) = production(&model);
        let (tokens, hidden) = rows(CONTEXT + 2, 5);
        let e = m
            .head_observe(&mut head, 0, &tokens, &hidden)
            .unwrap_err()
            .to_string();
        assert!(e.contains("do not fit"), "{e}");
    }

    /// A file without a prediction block has no head to load.
    #[test]
    fn a_model_without_a_prediction_block_has_no_head() {
        let model = write_qwen35(&TinyQwen35 {
            nextn_layers: 0,
            ..TinyQwen35::default()
        });
        let m = load(&model, CONTEXT);
        let e = m
            .load_head(&content_of(&model), &model.path)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("no in-trunk multi-token-prediction head"), "{e}");
    }

    /// `Hidden::PostNormAll` returns every row's final-norm output without
    /// changing the logits any `Want` selects, and skips the LM head for
    /// `Want::None`.
    #[test]
    fn all_rows_hidden_leaves_the_logits_alone() {
        let model = write_qwen35(&TinyQwen35::default());
        let tokens = [3u32, 5, 7, 1, 9, 2];
        for want in [Want::None, Want::Last, Want::All] {
            let mut plain = load(&model, CONTEXT);
            let mut all = load(&model, CONTEXT);
            let a = plain
                .forward_ext(&tokens, 0, want, Hidden::PostNorm, true)
                .unwrap();
            let b = all
                .forward_ext(&tokens, 0, want, Hidden::PostNormAll, true)
                .unwrap();
            assert_eq!(a.logits, b.logits, "{want:?}");
            let rows = b.hidden.unwrap();
            assert_eq!(rows.len(), tokens.len() * D, "{want:?}");
            match want {
                Want::None => assert!(a.hidden.is_none() && b.logits.is_none()),
                Want::Last => assert_eq!(a.hidden.unwrap(), rows[(tokens.len() - 1) * D..]),
                Want::All => assert_eq!(a.hidden.unwrap(), rows),
            }
        }
    }
}
