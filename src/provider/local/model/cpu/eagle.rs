//! The EAGLE drafter of Mistral Small 4 (`mistral4-eagle`).
//!
//! `mistralai/Mistral-Small-4-119B-2603-eagle` is a two-layer decoder that
//! predicts the token *after* the one the target predicts, from the target's
//! hidden state and the token the target just chose. It is a separate GGUF
//! file (written by the converter from the published Mistral-format
//! checkpoint) with no embedding table and no LM head: both are the target's.
//! One position of it works like this (vLLM's `EagleMistralLarge3Model`):
//!
//! 1. For a target position `j` with its final-norm hidden state `h[j]` (the
//!    row that feeds the LM head: [`Hidden::PostNormAll`](super::Hidden)) and
//!    the token `t[j + 1]` that follows it, the input is
//!    `x = eh_proj(cat(embed(t[j + 1]), h[j]))`: embedding first, the target's
//!    plain `token_embd` row, no norm on either half.
//! 2. `x` runs through two decoder layers, each with the target's own
//!    latent attention ([`Transformer::mla_in`]) and a *dense* gated FFN
//!    (width 12,288; the target's layers have experts), at RoPE position `j`,
//!    attending over every earlier input held in the drafter's own K/V cache.
//!    (vLLM keeps the position of the hidden state, `j`, where the Qwen 3.5
//!    head of [`NextnHead`](super::NextnHead) uses `j + 1`. The drafter
//!    attends to its own entries only, so the offset changes nothing but the
//!    Llama 4 query scaling, which counts absolute positions.)
//! 3. The output goes through the drafter's final norm (`output_norm`) and the
//!    target's LM head: logits for the token `t[j + 2]`. The normed output is
//!    also the hidden row of the next chain step, as for the Qwen head.
//!
//! The drafter has two layers, so committing a pair `(h[j], t[j + 1])` is a
//! forward pass of the first layer (attention and FFN, because the second
//! layer's input is its output) and only the latent K/V write of the second
//! (its output feeds nothing): no query, attention, FFN or LM head
//! ([`Transformer::eagle_observe`]). The cache is one latent row of
//! `kv_lora + rot` floats per position and layer, 2.5 KB per position for the
//! published drafter.
//!
//! # Cache
//!
//! Exactly the bookkeeping of the Qwen head ([`NextnHead`](super::NextnHead)):
//! one slot per committed pair in ascending pair index (pair indices may skip:
//! a skipped pair is a position the drafter cannot see, which costs a worse
//! proposal and never changes output), plus the *tail*, the hidden row of the
//! last observed position, which still waits for its successor token.
//! [`EagleHead::truncate`] and [`EagleHead::reset`] keep it consistent with the
//! trunk, and drafting writes into slots past the committed ones, which the
//! next observation overwrites.
//!
//! # Chain
//!
//! Step 1 of a draft chain uses the pair `(h[len - 1], pending token)` at
//! position `len - 1`; step `s > 1` uses the pair (the previous step's normed
//! output, the token step `s - 1` drafted) at position `len - 2 + s`.
//!
//! # Proposal vocabulary
//!
//! A step ranks the whole vocabulary through the target's LM head (the
//! drafter was trained against it); unlike the Qwen head it does not restrict
//! itself to a prefix of the token ids.
//!
//! # Memory
//!
//! The 392 MB FP8 checkpoint converts to 0.4 GB at Q8_0 (the converter's
//! default), kept quantized as in the file; the K/V cache is allocated for
//! `context_length` positions.

use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};

use super::super::arch::Arch;
use super::super::layout::{LayerLayout, name};
use super::super::paged::{BLOCK_SIZE, BlockTable};
use super::super::params::Hparams;
use super::super::{candle_err, err, read_hparams};
use super::{
    AttnCall, FfnWeights, KvAccess, KvLayer, LayerWeights, Loader, Mixer, QMatMul, Transformer,
    load_layer, rms,
};
use crate::error::CucaError;

/// An EAGLE drafter and its K/V cache; see the [module docs](self).
pub(crate) struct EagleHead {
    /// `nextn.eh_proj`: `2 * n_embd -> n_embd` over `[embedding | hidden]`.
    eh_proj: QMatMul,
    /// `output_norm`: the drafter's own final norm.
    norm: Tensor,
    layers: Vec<LayerWeights>,
    /// The drafter's own hyperparameters (its layers' geometry); everything
    /// else the layers read is the target's, which [`Transformer::load_eagle`]
    /// checked equal.
    hp: Hparams,
    /// The latent K/V of every layer: slot `s` holds the entry of pair
    /// `pairs[s]`; slots past `pairs.len()` are scratch.
    kv: Vec<Arc<KvLayer>>,
    /// The identity block table over the pools: slot `s` is row `s %
    /// BLOCK_SIZE` of block `s / BLOCK_SIZE`.
    table: BlockTable,
    capacity: usize,
    /// The pair index `j` (the pair `(h[j], t[j + 1])`) of every committed
    /// slot, strictly ascending.
    pairs: Vec<usize>,
    /// Position and final-norm hidden row of the last observed position,
    /// awaiting the token that follows it.
    tail: Option<(usize, Vec<f32>)>,
}

impl EagleHead {
    /// Number of committed pairs (K/V slots in use).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pairs.len()
    }

    /// The latent rows of every layer for the committed slots (tests).
    #[cfg(test)]
    pub(crate) fn kv_prefix(&self) -> Vec<Vec<f32>> {
        self.kv
            .iter()
            .zip(&self.hp.layers)
            .map(|(layer, spec)| layer.lock().k[..self.pairs.len() * spec.head_dim].to_vec())
            .collect()
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

    /// The attention call of layer `il`: `t` rows from slot `slot`, the first
    /// at RoPE position `rope_start`.
    fn call(&self, il: usize, slot: usize, rope_start: usize, t: usize) -> AttnCall<'_> {
        AttnCall {
            spec: &self.hp.layers[il],
            offset: 0,
            start: slot,
            rope_start,
            t,
            kv: KvAccess::Paged {
                layer: self.kv[il].as_ref(),
                write: true,
                table: &self.table,
                read_len: None,
            },
        }
    }
}

impl Transformer {
    /// Load the EAGLE drafter in the GGUF file at `path` (index `content`) for
    /// this model, with a K/V cache for its context length.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the file is not a `mistral4-eagle`
    /// drafter, lacks a tensor, or was trained for another model: a different
    /// hidden width, vocabulary, attention geometry, RoPE or query scaling.
    pub(crate) fn load_eagle(
        &self,
        content: &gguf_file::Content,
        path: &Path,
    ) -> Result<EagleHead, CucaError> {
        self.load_eagle_with(content, path, false)
    }

    /// [`Transformer::load_eagle`], with every matmul weight dequantized to
    /// f32 when `dequantize` is set (the test reference).
    pub(crate) fn load_eagle_with(
        &self,
        content: &gguf_file::Content,
        path: &Path,
        dequantize: bool,
    ) -> Result<EagleHead, CucaError> {
        let hp = read_hparams(content)?;
        if hp.arch != Arch::Mistral4Eagle {
            return Err(err(format!(
                "`{}` is not an EAGLE drafter (`general.architecture` is not `{}`)",
                hp.arch.name(),
                Arch::Mistral4Eagle.name()
            )));
        }
        self.check_eagle(&hp, content)?;
        let n = hp.n_embd;
        let mut loader = Loader {
            content,
            reader: BufReader::new(File::open(path)?),
            dequantize,
            requantize: false,
        };
        let eh_proj = loader.matmul(name::EAGLE_EH_PROJ, &[n, 2 * n])?;
        let norm = loader.vector(name::OUTPUT_NORM, n)?;
        let capacity = self.context_length;
        let blocks = capacity.div_ceil(BLOCK_SIZE);
        let mut layers = Vec::with_capacity(hp.n_layer);
        let mut kv = Vec::with_capacity(hp.n_layer);
        for (il, spec) in hp.layers.iter().enumerate() {
            let layout = LayerLayout::read(&hp, il, spec, content)?;
            let layer = load_layer(&mut loader, &hp, il, spec, &layout, None)?;
            if !matches!(layer.mixer, Mixer::Mla(_))
                || !matches!(layer.ffn, FfnWeights::Dense { .. })
            {
                return Err(err(format!(
                    "EAGLE layer {il} must be latent attention with a dense FFN"
                )));
            }
            layers.push(layer);
            kv.push(KvLayer::zeros(spec, blocks));
        }
        Ok(EagleHead {
            eh_proj,
            norm,
            layers,
            hp,
            kv,
            table: (0..blocks as u32).collect(),
            capacity,
            pairs: Vec::new(),
            tail: None,
        })
    }

    /// Reject a drafter that was not trained for this model. Its layers run
    /// on this model's RoPE tables, query scaling, norm epsilon and
    /// activation, and its input is this model's hidden state and embedding,
    /// so each of those must be the same.
    fn check_eagle(&self, hp: &Hparams, content: &gguf_file::Content) -> Result<(), CucaError> {
        let target = &self.hp;
        let differs = |what: &str, drafter: String, model: String| {
            Err(err(format!(
                "the EAGLE drafter's {what} is {drafter} but the model's is {model}: it was trained for another model"
            )))
        };
        if hp.n_embd != target.n_embd || hp.n_embd_out != target.n_embd {
            return differs(
                "hidden width",
                format!("{} (target side {})", hp.n_embd, hp.n_embd_out),
                target.n_embd.to_string(),
            );
        }
        let key = format!("{}.vocab_size", hp.arch.name());
        let vocab = content
            .metadata
            .get(&key)
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| err(format!("GGUF metadata `{key}`: missing or not an integer")))?;
        if vocab as usize != self.vocab {
            return differs("vocabulary", vocab.to_string(), self.vocab.to_string());
        }
        if (hp.eps - target.eps).abs() > 1e-9 * target.eps.abs() {
            return differs("norm epsilon", hp.eps.to_string(), target.eps.to_string());
        }
        if hp.act() != target.act() || hp.residual_scale != target.residual_scale {
            return differs(
                "activation or residual scale",
                format!("{:?}/{}", hp.act(), hp.residual_scale),
                format!("{:?}/{}", target.act(), target.residual_scale),
            );
        }
        if hp.ropes != target.ropes {
            return differs(
                "RoPE",
                format!("{:?}", hp.ropes),
                format!("{:?}", target.ropes),
            );
        }
        if hp.attn_temp != target.attn_temp {
            return differs(
                "query scaling",
                format!("{:?}", hp.attn_temp),
                format!("{:?}", target.attn_temp),
            );
        }
        // Every layer of the drafter attends like the model's first layer.
        let Some(t) = target.layers.first() else {
            return Err(err("the model has no layers"));
        };
        for (il, l) in hp.layers.iter().enumerate() {
            let same = (l.n_head, l.n_kv_head, l.head_dim, l.window, l.rope, l.mla)
                == (t.n_head, t.n_kv_head, t.head_dim, t.window, t.rope, t.mla)
                && (l.scale - t.scale).abs() <= 1e-6 * t.scale.abs();
            if !same {
                return differs(
                    &format!("layer {il} attention"),
                    format!("{l:?}"),
                    format!("{t:?}"),
                );
            }
        }
        Ok(())
    }

    /// The drafter inputs `eh_proj(cat(embedding, hidden))` of `m` pairs:
    /// `emb` holds the `m` embedding rows and `hidden` the `m` final-norm
    /// hidden rows, `(1, m, n_embd)` each.
    fn eagle_input(
        &self,
        head: &EagleHead,
        emb: Vec<f32>,
        hidden: &[f32],
    ) -> candle_core::Result<Tensor> {
        let d = self.hp.n_embd;
        let m = emb.len() / d;
        let dev = &Device::Cpu;
        let e = Tensor::from_vec(emb, (1, m, d), dev)?;
        let h = Tensor::from_vec(hidden.to_vec(), (1, m, d), dev)?;
        head.eh_proj.forward(&Tensor::cat(&[&e, &h], 2)?)
    }

    /// Layer `il` of the drafter over `x` with the attention call `call`.
    fn eagle_layer(
        &self,
        head: &EagleHead,
        il: usize,
        x: &Tensor,
        call: &AttnCall<'_>,
    ) -> candle_core::Result<Tensor> {
        let lw = &head.layers[il];
        let Mixer::Mla(mw) = &lw.mixer else {
            return Err(candle_core::Error::msg(
                "an EAGLE layer is latent attention",
            ));
        };
        self.block(lw, x, None, call.t, |h| {
            self.mla_in(std::slice::from_ref(call), mw, h)
        })
    }

    /// Commit the target rows `tokens[i]` at positions `pos + i`, whose
    /// final-norm hidden rows are `hidden` (`tokens.len() * n_embd` floats):
    /// every row is paired with its successor token, the tail with
    /// `tokens[0]` when it sits at `pos - 1`, and the last row becomes the
    /// tail. Pairs with an index at or above the first new one are replaced.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for mismatched lengths, a full cache or a
    /// failed forward pass.
    pub(crate) fn eagle_observe(
        &self,
        head: &mut EagleHead,
        pos: usize,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), CucaError> {
        let d = self.hp.n_embd;
        let n = tokens.len();
        if n == 0 || hidden.len() != n * d {
            return Err(err(format!(
                "EAGLE observe: {n} tokens with {} hidden floats, expected {}",
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
                (pos - 1, Cow::Owned(rows), tokens)
            }
            None => (pos, Cow::Borrowed(&hidden[..(n - 1) * d]), &tokens[1..]),
        };
        head.drop_from(first);
        let m = ids.len();
        if m > 0 {
            let slot = head.pairs.len();
            if slot + m > head.capacity {
                return Err(err(format!(
                    "the EAGLE cache holds {} positions, {} do not fit",
                    head.capacity,
                    slot + m
                )));
            }
            let emb = self.embedding_rows(ids)?;
            self.eagle_store(head, emb, &rows, slot, first)
                .map_err(candle_err)?;
            head.pairs.extend(first..first + m);
        }
        head.tail = Some((pos + n - 1, hidden[(n - 1) * d..].to_vec()));
        Ok(())
    }

    /// Store the K/V entries of `emb.len() / n_embd` pairs in slots `slot..`
    /// of every layer, the first pair being `first` (RoPE position `first`).
    /// All layers but the last run in full: the next layer's input is their
    /// output. The last only writes its latent row, so no query, attention,
    /// FFN or output projection runs for it.
    fn eagle_store(
        &self,
        head: &EagleHead,
        emb: Vec<f32>,
        hidden: &[f32],
        slot: usize,
        first: usize,
    ) -> candle_core::Result<()> {
        let m = emb.len() / self.hp.n_embd;
        let Some(last) = head.layers.len().checked_sub(1) else {
            return Err(candle_core::Error::msg("an EAGLE drafter has no layers"));
        };
        let mut x = self.eagle_input(head, emb, hidden)?;
        for il in 0..last {
            x = self.eagle_layer(head, il, &x, &head.call(il, slot, first, m))?;
        }
        let lw = &head.layers[last];
        let Mixer::Mla(mw) = &lw.mixer else {
            return Err(candle_core::Error::msg(
                "an EAGLE layer is latent attention",
            ));
        };
        let h = match &lw.attn_norm {
            Some(w) => rms(&x, w, self.hp.eps as f32)?,
            None => x,
        };
        self.mla_store_in(
            std::slice::from_ref(&head.call(last, slot, first, m)),
            mw,
            &h,
        )
    }

    /// Draft at most `k` tokens after the pending token `first`: `hidden` is
    /// the target's final-norm hidden row at its last cached position and
    /// `len` the number of cached positions (`first` sits at position `len`,
    /// not cached). Each step's logits go to `pick`, which chooses the token
    /// or ends the chain with `None`; the returned drafts are the tokens `pick`
    /// accepted. Nothing is committed: see the [module docs](self).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a bad `hidden` row, a full cache or a
    /// failed forward step.
    pub(crate) fn eagle_draft(
        &self,
        head: &EagleHead,
        hidden: &[f32],
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> Option<u32>,
    ) -> Result<Vec<u32>, CucaError> {
        if len == 0 || hidden.len() != self.hp.n_embd {
            return Err(err(
                "EAGLE draft: needs the hidden row of the last position",
            ));
        }
        let slot = head.pairs.len();
        if slot + k > head.capacity {
            return Err(err(format!(
                "the EAGLE cache holds {} positions, {} do not fit",
                head.capacity,
                slot + k
            )));
        }
        let mut drafts = Vec::with_capacity(k);
        let mut hidden = Cow::Borrowed(hidden);
        let mut token = first;
        for step in 0..k {
            let emb = self.embedding_rows(&[token])?;
            let (mut logits, next_hidden) = self
                .eagle_step(head, emb, &hidden, slot + step, len - 1 + step)
                .map_err(candle_err)?;
            let Some(next) = pick(&mut logits) else { break };
            token = next;
            drafts.push(token);
            hidden = Cow::Owned(next_hidden);
        }
        Ok(drafts)
    }

    /// One draft step: the pair `(hidden, embedded token)` of pair index `j`
    /// goes into slot `slot` of every layer; returns the logits of the
    /// following token and the normed output the next step takes.
    fn eagle_step(
        &self,
        head: &EagleHead,
        emb: Vec<f32>,
        hidden: &[f32],
        slot: usize,
        j: usize,
    ) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let mut x = self.eagle_input(head, emb, hidden)?;
        for il in 0..head.layers.len() {
            x = self.eagle_layer(head, il, &x, &head.call(il, slot, j, 1))?;
        }
        let normed = rms(&x, &head.norm, self.hp.eps as f32)?;
        let logits = self.head_rows(&normed, self.vocab)?;
        Ok((
            logits.flatten_all()?.to_vec1::<f32>()?,
            normed.flatten_all()?.to_vec1::<f32>()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::gguf_file::Value;
    use serde_json::Value as Json;

    use super::super::hybrid_tests::{content_of, load_dequantized};
    use super::*;
    use crate::provider::local::WeightQuant;
    use crate::provider::local::convert::{convert, hf::CheckpointDir};
    use crate::provider::local::model::tiny::{
        CONTEXT, Lcg, N_EMBD, TinyEagle, TinyMistral4, TinyModel, VOCAB, load, write_mistral4,
        write_tiny_eagle,
    };
    use crate::provider::local::sampler::greedy;

    const D: usize = N_EMBD;

    fn target() -> TinyModel {
        write_mistral4(&TinyMistral4::default())
    }

    /// The target and its drafter as the engine loads them (quantized).
    fn production(target: &TinyModel, eagle: &TinyModel) -> (Transformer, EagleHead) {
        let m = load(target, CONTEXT);
        let head = m.load_eagle(&content_of(eagle), &eagle.path).unwrap();
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
        head: &EagleHead,
        h: &[f32],
        first: u32,
        len: usize,
        k: usize,
    ) -> Vec<Vec<f32>> {
        let mut seen = Vec::new();
        m.eagle_draft(head, h, first, len, k, &mut |l: &mut [f32]| {
            seen.push(l.to_vec());
            Some(greedy(l))
        })
        .unwrap();
        seen
    }

    /// Every float of a JSON array of arrays of numbers, in order.
    fn flat(v: &Json) -> Vec<f32> {
        match v {
            Json::Array(items) => items.iter().flat_map(flat).collect(),
            other => vec![other.as_f64().unwrap() as f32],
        }
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    /// The tiny drafter of `testdata/eagle` (Mistral-format FP8 checkpoint,
    /// `make_fixture.py`) against what transformers' `Mistral4Model` computes
    /// from the same weights: the checkpoint goes through the engine's
    /// converter (F16) and the drafter loads in the exact dequantized mode, so
    /// only float rounding separates the two sides. The chain's draft logits
    /// and the first step's normed output match within 5e-4, the greedy tokens
    /// agree, and every convention the reference was also run with changed
    /// (hidden row first, no Llama 4 query scaling, RoPE position `j + 1`) is
    /// far from the engine's logits, so the comparison notices each mistake.
    #[test]
    fn the_converted_drafter_matches_the_transformers_reference() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/provider/local/testdata/eagle");
        let reference: Json =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let gguf =
            std::env::temp_dir().join(format!("cuca-eagle-oracle-{}.gguf", std::process::id()));
        convert(
            &CheckpointDir::open(&dir).unwrap(),
            &gguf,
            WeightQuant::F16,
            None,
        )
        .unwrap();

        // The target the reference's embedding and LM head belong to; its
        // YaRN original context is the checkpoint's (the drafter must scale
        // like it).
        let model = write_mistral4(&TinyMistral4 {
            layers: 1,
            meta: vec![("rope.scaling.original_context_length".into(), Value::U32(8))],
            embd: Some(flat(&reference["embd"])),
            output: Some(flat(&reference["lm_head"])),
            ..TinyMistral4::default()
        });
        let m = load_dequantized(&model);
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&gguf).unwrap())).unwrap();
        let mut head = m.load_eagle_with(&content, &gguf, true).unwrap();

        let tokens: Vec<u32> = flat(&reference["tokens"])
            .iter()
            .map(|&t| t as u32)
            .collect();
        let hidden = flat(&reference["hidden"]);
        let first = reference["first"].as_u64().unwrap() as u32;
        let n = tokens.len();
        assert_eq!(hidden.len(), n * D);
        m.eagle_observe(&mut head, 0, &tokens, &hidden).unwrap();
        assert_eq!(head.len(), n - 1);
        let last = &hidden[(n - 1) * D..];

        // The first step's normed output, which feeds the next step.
        let steps = reference["steps"].as_array().unwrap();
        let emb = m.embedding_rows(&[first]).unwrap();
        let (_, normed) = m.eagle_step(&head, emb, last, head.len(), n - 1).unwrap();
        let want = flat(&steps[0]["normed"]);
        eprintln!(
            "EAGLE oracle: step 1 normed max |d| {}",
            max_diff(&normed, &want)
        );
        assert!(max_diff(&normed, &want) < 5e-4, "{normed:?} vs {want:?}");

        let mut got = Vec::new();
        let mut tokens_out = Vec::new();
        m.eagle_draft(
            &head,
            last,
            first,
            n,
            steps.len(),
            &mut |l: &mut [f32]| {
                got.push(l.to_vec());
                tokens_out.push(greedy(l));
                Some(greedy(l))
            },
        )
        .unwrap();
        assert_eq!(got.len(), steps.len());
        for (s, step) in steps.iter().enumerate() {
            let want = flat(&step["logits"]);
            let d = max_diff(&got[s], &want);
            eprintln!("EAGLE oracle: step {s} logits max |d| {d}");
            assert!(d < 5e-4, "step {s}: max |dlogit| {d}");
            assert_eq!(
                u64::from(tokens_out[s]),
                step["token"].as_u64().unwrap(),
                "step {s}"
            );
        }
        for (name, control) in reference["controls"].as_object().unwrap() {
            let worst = (0..steps.len())
                .map(|s| max_diff(&got[s], &flat(&control[s])))
                .fold(0.0, f32::max);
            eprintln!("EAGLE oracle: control `{name}` max |d| {worst}");
            assert!(worst > 0.1, "control `{name}` is as close as {worst}");
        }
        // Teacher forced: every step fed the reference's own previous output
        // and token, so a float difference cannot compound through the chain,
        // and the first three steps stay within 1e-4.
        let (mut hidden_in, mut token_in) = (last.to_vec(), first);
        for (s, step) in steps.iter().enumerate() {
            let emb = m.embedding_rows(&[token_in]).unwrap();
            let (logits, _) = m
                .eagle_step(&head, emb, &hidden_in, head.len() + s, n - 1 + s)
                .unwrap();
            let d = max_diff(&logits, &flat(&step["logits"]));
            eprintln!("EAGLE oracle: teacher-forced step {s} logits max |d| {d}");
            assert!(d < 1e-4, "teacher-forced step {s}: max |dlogit| {d}");
            hidden_in = flat(&step["normed"]);
            token_in = step["token"].as_u64().unwrap() as u32;
        }
        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_file(gguf.with_extension("gguf.lock"));
    }

    /// Committing a pair writes the same latent rows in every layer as a
    /// full forward pass through both layers would.
    #[test]
    fn a_commit_stores_the_rows_a_full_forward_would() {
        let (model, eagle) = (target(), write_tiny_eagle(&TinyEagle::default()));
        let (m, mut quick) = production(&model, &eagle);
        let (_, full) = production(&model, &eagle);
        let n = 12;
        let (tokens, hidden) = rows(n, 17);
        m.eagle_observe(&mut quick, 0, &tokens, &hidden).unwrap();

        let emb = m.embedding_rows(&tokens[1..]).unwrap();
        let mut x = m.eagle_input(&full, emb, &hidden[..(n - 1) * D]).unwrap();
        for il in 0..full.layers.len() {
            x = m
                .eagle_layer(&full, il, &x, &full.call(il, 0, 0, n - 1))
                .unwrap();
        }
        for il in 0..quick.layers.len() {
            assert_eq!(quick.kv[il].lock().k, full.kv[il].lock().k, "layer {il}");
        }
        assert!(quick.kv[0].lock().k.iter().any(|&v| v != 0.0));
    }

    /// Observing in any chunking commits the same pairs, and truncating to
    /// the shared prefix and re-observing from where the trunk resumes
    /// reproduces a cold drafter bit for bit.
    #[test]
    fn observe_is_chunk_invariant_and_truncate_plus_replay_equals_cold() {
        let (model, eagle) = (target(), write_tiny_eagle(&TinyEagle::default()));
        let (m, mut cold) = production(&model, &eagle);
        let n = 9;
        let (tokens, hidden) = rows(n, 7);
        let (first, last) = (4u32, &hidden[(n - 1) * D..]);
        m.eagle_observe(&mut cold, 0, &tokens, &hidden).unwrap();
        assert_eq!(cold.len(), n - 1);
        let want = chain(&m, &cold, last, first, n, 3);

        // Chunks of 2, 3 and 4 rows: the tail carries the pair across calls.
        let (_, mut chunked) = production(&model, &eagle);
        for (from, to) in [(0, 2), (2, 5), (5, 9)] {
            m.eagle_observe(
                &mut chunked,
                from,
                &tokens[from..to],
                &hidden[from * D..to * D],
            )
            .unwrap();
        }
        assert_eq!(chunked.len(), n - 1);
        assert_eq!(chain(&m, &chunked, last, first, n, 3), want);

        // The first six tokens stay, the trunk replays from position 4: pairs
        // 0..=4 hold, the replay re-adds the rest, and the result is cold.
        let (_, mut replayed) = production(&model, &eagle);
        m.eagle_observe(&mut replayed, 0, &tokens, &hidden).unwrap();
        replayed.truncate(6);
        assert_eq!(replayed.len(), 5);
        m.eagle_observe(&mut replayed, 4, &tokens[4..], &hidden[4 * D..])
            .unwrap();
        assert_eq!(replayed.len(), n - 1);
        assert_eq!(chain(&m, &replayed, last, first, n, 3), want);

        // A trunk that could resume only at the shared length leaves out the
        // pair that needs the hidden row it no longer has: one position
        // fewer for the drafter to see, nothing else.
        let (_, mut gap) = production(&model, &eagle);
        m.eagle_observe(&mut gap, 0, &tokens, &hidden).unwrap();
        gap.truncate(5);
        m.eagle_observe(&mut gap, 5, &tokens[5..], &hidden[5 * D..])
            .unwrap();
        assert_eq!(gap.len(), n - 2);
        assert_eq!(chain(&m, &gap, last, first, n, 3).len(), 3);

        // The whole sequence is shared: the tail survives and bridges the
        // next call.
        let (_, mut same) = production(&model, &eagle);
        m.eagle_observe(&mut same, 0, &tokens, &hidden).unwrap();
        same.truncate(n);
        assert_eq!(same.len(), n - 1);
        m.eagle_observe(&mut same, n, &tokens[..1], &hidden[..D])
            .unwrap();
        assert_eq!(same.len(), n);

        // Reset forgets the tail as well.
        same.reset();
        assert_eq!(same.len(), 0);
        m.eagle_observe(&mut same, 3, &tokens[..2], &hidden[..2 * D])
            .unwrap();
        assert_eq!(same.len(), 1);
    }

    /// Drafting is deterministic, commits nothing, and its temporary steps
    /// never leak into what the next observation commits.
    #[test]
    fn drafting_is_deterministic_and_commits_nothing() {
        let (model, eagle) = (target(), write_tiny_eagle(&TinyEagle::default()));
        let (m, mut head) = production(&model, &eagle);
        let n = 6;
        let (tokens, hidden) = rows(n, 3);
        let last = &hidden[(n - 1) * D..];
        m.eagle_observe(&mut head, 0, &tokens, &hidden).unwrap();
        let a = chain(&m, &head, last, 5, n, 4);
        assert_eq!(head.len(), n - 1);
        assert_eq!(chain(&m, &head, last, 5, n, 4), a);
        // A shorter chain is a prefix of the longer one.
        assert_eq!(chain(&m, &head, last, 5, n, 2), a[..2]);

        // After drafting, commit more true pairs: the drafter equals one that
        // never drafted.
        let (_, mut clean) = production(&model, &eagle);
        m.eagle_observe(&mut clean, 0, &tokens, &hidden).unwrap();
        let (more_tokens, more_hidden) = rows(3, 99);
        for h in [&mut head, &mut clean] {
            m.eagle_observe(h, n, &more_tokens, &more_hidden).unwrap();
        }
        let next = &more_hidden[2 * D..];
        assert_eq!(
            chain(&m, &head, next, 7, n + 3, 3),
            chain(&m, &clean, next, 7, n + 3, 3)
        );

        // Bad input is an error, not a panic.
        assert!(m.eagle_observe(&mut head, 0, &[], &[]).is_err());
        assert!(
            m.eagle_observe(&mut head, 0, &tokens, &hidden[1..])
                .is_err()
        );
        let mut pick = |l: &mut [f32]| Some(greedy(l));
        assert!(
            m.eagle_draft(&head, &hidden[..D - 1], 3, n, 1, &mut pick)
                .is_err()
        );
        assert!(m.eagle_draft(&head, last, 3, 0, 1, &mut pick).is_err());
    }

    /// A step ranks the whole vocabulary, and a pick that ends the chain
    /// stops it with the drafts so far.
    #[test]
    fn a_step_ranks_the_whole_vocabulary_and_the_pick_may_end_the_chain() {
        let (model, eagle) = (target(), write_tiny_eagle(&TinyEagle::default()));
        let (m, mut head) = production(&model, &eagle);
        let (tokens, hidden) = rows(6, 12);
        m.eagle_observe(&mut head, 0, &tokens, &hidden).unwrap();
        let last = &hidden[5 * D..];
        assert!(
            chain(&m, &head, last, 4, 6, 3)
                .iter()
                .all(|l| l.len() == VOCAB)
        );
        let mut steps = 0;
        let drafted = m
            .eagle_draft(&head, last, 4, 6, 4, &mut |l: &mut [f32]| {
                steps += 1;
                (steps < 3).then(|| greedy(l))
            })
            .unwrap();
        assert_eq!((steps, drafted.len()), (3, 2));
    }

    /// A cache that cannot hold the entries is an error naming the limit.
    #[test]
    fn a_full_cache_is_an_error() {
        let (model, eagle) = (target(), write_tiny_eagle(&TinyEagle::default()));
        let (m, mut head) = production(&model, &eagle);
        let (tokens, hidden) = rows(CONTEXT + 2, 5);
        let e = m
            .eagle_observe(&mut head, 0, &tokens, &hidden)
            .unwrap_err()
            .to_string();
        assert!(e.contains("do not fit"), "{e}");
        let (tokens, hidden) = rows(CONTEXT, 5);
        m.eagle_observe(&mut head, 0, &tokens, &hidden).unwrap();
        let mut pick = |l: &mut [f32]| Some(greedy(l));
        let e = m
            .eagle_draft(
                &head,
                &hidden[(CONTEXT - 1) * D..],
                3,
                CONTEXT,
                2,
                &mut pick,
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("do not fit"), "{e}");
    }

    /// A one-layer drafter commits through the cache-only path alone and a
    /// longer one runs the same code.
    #[test]
    fn drafters_of_other_depths_run() {
        let model = target();
        for layers in [1, 2, 3] {
            let eagle = write_tiny_eagle(&TinyEagle {
                layers,
                ..TinyEagle::default()
            });
            let (m, mut head) = production(&model, &eagle);
            let (tokens, hidden) = rows(8, 2);
            m.eagle_observe(&mut head, 0, &tokens, &hidden).unwrap();
            let logits = chain(&m, &head, &hidden[7 * D..], 5, 8, 2);
            assert_eq!(logits.len(), 2, "{layers} layers");
            assert!(logits.iter().flatten().all(|v| v.is_finite()));
        }
    }

    /// A drafter trained for another model, a file that is not a drafter and
    /// one that lacks a tensor are loud errors naming the cause.
    #[test]
    fn a_drafter_for_another_model_is_rejected() {
        let model = target();
        let m = load(&model, CONTEXT);
        let load_with = |meta: Vec<(String, Value)>| {
            let eagle = write_tiny_eagle(&TinyEagle {
                meta,
                ..TinyEagle::default()
            });
            m.load_eagle(&content_of(&eagle), &eagle.path)
                .err()
                .map(|e| e.to_string())
        };
        let key = |k: &str, v: Value| vec![(k.to_string(), v)];
        assert_eq!(load_with(Vec::new()), None);
        for (meta, expect) in [
            (key("embedding_length_out", Value::U32(32)), "hidden width"),
            (
                key("vocab_size", Value::U32(VOCAB as u32 + 1)),
                "vocabulary",
            ),
            (
                key("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
                "norm epsilon",
            ),
            (key("rope.scaling.factor", Value::F32(4.0)), "RoPE"),
            (
                key("attention.temperature_scale", Value::F32(0.5)),
                "query scaling",
            ),
            (key("attention.head_count", Value::U32(2)), "attention"),
            (key("block_count", Value::U32(3)), "blk.2"),
        ] {
            let e = load_with(meta).unwrap();
            assert!(e.contains(expect), "{expect}: {e}");
        }
        // The target's own file is not a drafter.
        let e = m
            .load_eagle(&content_of(&model), &model.path)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("not an EAGLE drafter"), "{e}");
        // And a drafter is not a model, nor a shared-K/V assistant.
        let eagle = write_tiny_eagle(&TinyEagle::default());
        let e = Transformer::load(&content_of(&eagle), &eagle.path, CONTEXT)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("drafter"), "{e}");
        let e = Transformer::load_drafter(&content_of(&eagle), &eagle.path, CONTEXT)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("keeps a K/V cache of its own"), "{e}");
    }
}
