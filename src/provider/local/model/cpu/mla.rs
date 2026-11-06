//! Multi-head latent attention on the CPU (Mistral Small 4, llama.cpp's
//! `mistral4` / `deepseek2` graph with absorbed matrices).
//!
//! Per token the layer computes a low-rank query `q = q_b(norm(q_a(x)))`
//! (`n_head` heads of `nope + rot` values) and `kv = kv_a(x)`, whose first
//! `kv_lora` values, normed, are the *latent* `c` shared by every head and
//! whose last `rot` values are the rotary key `k_rot`, also shared. In the
//! reference (transformers) form the keys and values of head `h` are
//! `k_nope = K_h c` and `v = V_h c`, and the score is
//! `q_nope . k_nope + q_rot . k_rot`. This engine never builds them: it
//! absorbs `K_h` into the query, `q_abs = K_h^T q_nope` (`kv_lora` values),
//! so that the score is `[q_abs | q_rot] . [c | k_rot]` against one cached
//! row `[c | k_rot]` per position, and the attention-weighted sum of the
//! values is `V_h (sum_j p_j c_j)`, taken over the first `kv_lora` columns of
//! the same cached rows. One row of `kv_lora + rot` floats per position and
//! layer is the whole K/V cache (1,280 bytes for Mistral Small 4, against
//! 32,768 for the expanded heads).
//!
//! Every step is row-independent (a row's result depends only on its own
//! input and its own cache range, with a fixed reduction order), so a row is
//! bit-identical alone or in a batch, as for the other attention kernels.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::utils::barrier_pool;
use candle_core::{Device, Tensor};

use super::paged_attn::{attend_latent_rows, dot, store_rows};
use super::{AttnCall, KvAccess, MlaWeights, Transformer, rms};

/// Run `f(job, chunk)` for every `chunk`-sized piece of `out` on candle's
/// barrier pool; the arithmetic of one piece never depends on the split.
fn par_chunks(out: &mut [f32], chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let jobs = out.len() / chunk.max(1);
    let slots: Vec<Mutex<&mut [f32]>> = out.chunks_mut(chunk.max(1)).map(Mutex::new).collect();
    let next = AtomicUsize::new(0);
    let run = |_: usize| {
        loop {
            let job = next.fetch_add(1, Ordering::Relaxed);
            if job >= jobs {
                break;
            }
            let mut slot = slots[job].lock().unwrap_or_else(PoisonError::into_inner);
            f(job, &mut slot);
        }
    };
    if jobs <= 1 {
        run(0);
    } else {
        barrier_pool().execute(run);
    }
}

/// `q_abs[row][h][c] = sum_n k_b[h][c][n] * q_nope[row][h][n]`: the nope part
/// of every query head folded through its key up-projection. `q_nope` is
/// `[t][n_head][nope]`, `k_b` `[n_head][kv_lora][nope]`; the result is
/// `[t][n_head][kv_lora]`.
pub(super) fn absorb_queries(
    q_nope: &[f32],
    t: usize,
    n_head: usize,
    nope: usize,
    kv_lora: usize,
    k_b: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; t * n_head * kv_lora];
    par_chunks(&mut out, kv_lora, |job, o| {
        let h = job % n_head;
        let q = &q_nope[job * nope..(job + 1) * nope];
        let kb = &k_b[h * kv_lora * nope..(h + 1) * kv_lora * nope];
        for (c, v) in o.iter_mut().enumerate() {
            *v = dot(&kb[c * nope..(c + 1) * nope], q);
        }
    });
    out
}

/// `out[row][h][j] = sum_c v_b[h][j][c] * lat[row][h][c]`: the attended
/// latent of every head through its value up-projection. `lat` is
/// `[t][n_head][kv_lora]`, `v_b` `[n_head][v_head][kv_lora]`; the result is
/// `[t][n_head][v_head]`.
pub(super) fn expand_values(
    lat: &[f32],
    t: usize,
    n_head: usize,
    kv_lora: usize,
    v_head: usize,
    v_b: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; t * n_head * v_head];
    par_chunks(&mut out, v_head, |job, o| {
        let h = job % n_head;
        let l = &lat[job * kv_lora..(job + 1) * kv_lora];
        let vb = &v_b[h * v_head * kv_lora..(h + 1) * v_head * kv_lora];
        for (j, v) in o.iter_mut().enumerate() {
            *v = dot(&vb[j * kv_lora..(j + 1) * kv_lora], l);
        }
    });
    out
}

impl Transformer {
    /// The latent attention of one layer over the normed input `h` `(1, T,
    /// n_embd)` of the rows the `calls` describe (all one layer's): the
    /// projections and RoPE run over all rows, the cache write and the
    /// attention per sequence. The result is `(1, T, n_embd)`.
    pub(super) fn mla_in(
        &self,
        calls: &[AttnCall<'_>],
        mw: &MlaWeights,
        h: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hp = &self.hp;
        let first = calls
            .first()
            .ok_or_else(|| candle_core::Error::msg("attention needs a sequence"))?;
        let spec = first.spec;
        let mla = spec
            .mla
            .ok_or_else(|| candle_core::Error::msg("not a latent attention layer"))?;
        let t = h.dim(1)?;
        let eps = hp.eps as f32;
        let (nh, nope, rot, kvl, vh) = (spec.n_head, mla.nope, mla.rot, mla.kv_lora, mla.v_head);
        let width = mla.latent_width();

        // Query: low rank, normed, split per head into `[nope | rot]`.
        let cq = rms(&mw.q_a.forward(h)?, &mw.q_a_norm, eps)?;
        let q = mw.q_b.forward(&cq)?.reshape((1, t, nh, mla.q_head()))?;
        let q_nope = q.narrow(3, 0, nope)?.contiguous()?;
        let q_rot = q.narrow(3, nope, rot)?.transpose(1, 2)?.contiguous()?;
        // Latent and the shared rotary key.
        let kv = mw.kv_a.forward(h)?;
        let c = rms(&kv.narrow(2, 0, kvl)?.contiguous()?, &mw.kv_a_norm, eps)?;
        let k_rot = kv
            .narrow(2, kvl, rot)?
            .contiguous()?
            .reshape((1, 1, t, rot))?;
        // The rotated dimensions are the whole rotary slice, so the adjacent
        // pair rotation applies to it unchanged (the leading `n_dims` of a
        // `rot`-wide row).
        let q_rot = self.rotate_calls(spec, &q_rot, calls)?;
        let k_rot = self.rotate_calls(spec, &k_rot, calls)?;

        let q_nope = q_nope.flatten_all()?.to_vec1::<f32>()?;
        let q_rot = q_rot
            .transpose(1, 2)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let c = c.flatten_all()?.to_vec1::<f32>()?;
        let k_rot = k_rot.flatten_all()?.to_vec1::<f32>()?;

        // Absorbed query rows `[q_abs | q_rot]`, each scaled by the Llama 4
        // factor of its absolute position.
        let q_abs = absorb_queries(&q_nope, t, nh, nope, kvl, &mw.k_b);
        let mut q_full = vec![0f32; t * nh * width];
        for call in calls {
            for i in 0..call.t {
                let row = call.offset + i;
                let s = hp.attn_temp.map_or(1.0, |a| a.at(call.rope_start + i));
                for head in 0..nh {
                    let at = (row * nh + head) * width;
                    let (dst_abs, dst_rot) = q_full[at..at + width].split_at_mut(kvl);
                    let src_abs = &q_abs[(row * nh + head) * kvl..][..kvl];
                    let src_rot = &q_rot[(row * nh + head) * rot..][..rot];
                    for (d, v) in dst_abs.iter_mut().zip(src_abs) {
                        *d = v * s;
                    }
                    for (d, v) in dst_rot.iter_mut().zip(src_rot) {
                        *d = v * s;
                    }
                }
            }
        }

        // Store `[c | k_rot]` of every row in its sequence's blocks.
        for call in calls {
            let mut rows = Vec::with_capacity(call.t * width);
            for i in 0..call.t {
                let row = call.offset + i;
                rows.extend_from_slice(&c[row * kvl..][..kvl]);
                rows.extend_from_slice(&k_rot[row * rot..][..rot]);
            }
            match call.kv {
                KvAccess::Paged {
                    layer,
                    write: true,
                    table,
                    ..
                } => {
                    let mut data = layer.lock();
                    store_rows(table, call.start, call.t, &rows, &mut data.k, layer.shape)?;
                }
                _ => {
                    return Err(candle_core::Error::msg(
                        "a latent attention layer needs a paged cache of its own",
                    ));
                }
            }
        }

        // Attend over the latent rows, then expand the attended latent of
        // every head through `v_b`.
        let mut attended = Vec::with_capacity(t * nh * kvl);
        for call in calls {
            let KvAccess::Paged { layer, table, .. } = call.kv else {
                return Err(candle_core::Error::msg(
                    "a latent layer reads a paged cache",
                ));
            };
            let data = layer.lock();
            let q = &q_full[call.offset * nh * width..(call.offset + call.t) * nh * width];
            let start = call.start;
            attended.extend(attend_latent_rows(
                q,
                call.t,
                |i| (0, start + i + 1),
                table,
                &data.k,
                layer.shape,
                kvl,
                spec.scale,
            )?);
        }
        let attn = expand_values(&attended, t, nh, kvl, vh, &mw.v_b);
        mw.wo
            .forward(&Tensor::from_vec(attn, (1, t, nh * vh), &Device::Cpu)?)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use candle_core::quantized::GgmlDType;
    use candle_core::quantized::gguf_file::{self, Value};
    use candle_core::{Device, Tensor};

    use super::super::Mixer;
    use super::super::hybrid_tests::{
        content_of, load_dequantized, matvec, solo_seg, step, tensor,
    };
    use super::super::tests::{check_batched, check_invariants, greedy_run, load_tiered};
    use crate::provider::local::model::arch::Act;
    use crate::provider::local::model::layout::{Ffn, LayerLayout, name};
    use crate::provider::local::model::paged::KvPools;
    use crate::provider::local::model::tier::ExpertSlice;
    use crate::provider::local::model::tiny::{
        M4_DENSE_FF, M4_FF, M4_KV_LORA, M4_N_EXPERT, M4_N_HEAD, M4_NOPE, M4_ORIG_CTX, M4_ROT,
        M4_V_HEAD, N_EMBD, PROMPT, TinyMistral4, VOCAB, load, write_mistral4,
    };
    use crate::provider::local::model::{Want, read_hparams};

    fn variants() -> Vec<(&'static str, TinyMistral4)> {
        vec![
            ("fused, Q8_0", TinyMistral4::default()),
            (
                "split, Q8_0",
                TinyMistral4 {
                    fused: false,
                    ..TinyMistral4::default()
                },
            ),
            (
                "fused, F32",
                TinyMistral4 {
                    dtype: GgmlDType::F32,
                    ..TinyMistral4::default()
                },
            ),
            (
                "dense first layer",
                TinyMistral4 {
                    dense_layers: 1,
                    ..TinyMistral4::default()
                },
            ),
        ]
    }

    #[test]
    fn mistral4_satisfies_the_cache_and_batching_invariants() {
        for (what, spec) in variants() {
            let model = write_mistral4(&spec);
            check_invariants(&model, what);
            check_batched(&model, what);
        }
    }

    /// Rows are position-dependent (the Llama 4 query scaling and YaRN rotate
    /// by absolute position), so a prefill split anywhere, across the scaling
    /// length (8) and the original YaRN context (16) too, equals one pass.
    #[test]
    fn a_long_prefill_in_any_chunking_equals_one_pass() {
        let model = write_mistral4(&TinyMistral4::default());
        let prompt: Vec<u32> = (0..30u32).map(|i| (i * 5 + 3) % VOCAB as u32).collect();
        let whole = step(&mut load(&model, 32), &prompt, 0, Want::Last);
        for cuts in [[7usize, 8, 14], [3, 5, 17], [1, 1, 1]] {
            let mut m = load(&model, 32);
            let mut at = 0;
            for cut in cuts {
                m.forward(&prompt[at..at + cut], at, Want::None).unwrap();
                at += cut;
            }
            let last = step(&mut m, &prompt[at..], at, Want::Last);
            crate::provider::local::model::tiny::assert_close(&whole, &last);
        }
    }

    /// One shared latent row per position and layer is the whole cache: no
    /// value pool, `kv_lora + rot` floats per position.
    #[test]
    fn the_cache_is_one_latent_row_per_position_and_layer() {
        let model = write_mistral4(&TinyMistral4::default());
        let m = load(&model, 32);
        let row = M4_KV_LORA + M4_ROT;
        // Single sequence: twice the context in blocks of 16 positions.
        assert_eq!(m.kv_pool_bytes(), (3 * 4 * 16 * row * 4) as u64);
        for (il, spec) in m.hp.kv_owners() {
            assert_eq!((spec.n_kv_head, spec.head_dim), (1, row), "layer {il}");
            let layer = m.cache(il).unwrap();
            let data = layer.lock();
            assert_eq!(data.k.len(), 4 * 16 * row);
            assert!(data.v.is_empty());
        }
    }

    /// The same weights stored as one fused gate/up tensor or as two give
    /// the same logits, bit for bit.
    #[test]
    fn fused_and_split_gate_up_experts_agree_bit_for_bit() {
        let fused = write_mistral4(&TinyMistral4::default());
        let split = write_mistral4(&TinyMistral4 {
            fused: false,
            ..TinyMistral4::default()
        });
        assert_eq!(
            greedy_run(&mut load(&fused, 32)),
            greedy_run(&mut load(&split, 32))
        );
    }

    /// Tiered experts (read from the file by offset, the fused tensor in two
    /// strided halves) reproduce the resident model bit for bit, at the
    /// minimum budget and at one that holds a few experts per layer.
    #[test]
    fn tiered_fused_experts_match_resident_bit_for_bit() {
        for dense_layers in [0, 1] {
            tiered_matches_resident(dense_layers);
        }
    }

    fn tiered_matches_resident(dense_layers: usize) {
        let model = write_mistral4(&TinyMistral4 {
            dense_layers,
            ..TinyMistral4::default()
        });
        let content = content_of(&model);
        let hp = read_hparams(&content).unwrap();
        let per_index: u64 = (dense_layers..hp.n_layer)
            .map(|il| {
                let both = ExpertSlice::read(
                    &content,
                    &name::blk(il, name::FFN_GATE_UP_EXPS),
                    M4_N_EXPERT,
                    2 * M4_FF,
                    N_EMBD,
                )
                .unwrap();
                let down = ExpertSlice::read(
                    &content,
                    &name::blk(il, name::FFN_DOWN_EXPS),
                    M4_N_EXPERT,
                    N_EMBD,
                    M4_FF,
                )
                .unwrap();
                (both.file_bytes + down.file_bytes) as u64
            })
            .sum();
        let want = greedy_run(&mut load(&model, 32));
        for experts in [2u64, 5] {
            let budget = experts * per_index;
            let mut tiered = load_tiered(&model, budget, None).unwrap();
            assert_eq!(greedy_run(&mut tiered), want, "{experts} experts");
            let stats = tiered.expert_tiers().unwrap().stats();
            assert!(stats.disk_reads > 0, "{stats:?}");
        }
    }

    /// The per-position query scaling changes the logits only past its
    /// length, and a zero scale switches it off.
    #[test]
    fn query_scaling_applies_by_absolute_position() {
        let default = write_mistral4(&TinyMistral4::default());
        let wide = |key: &str, value: Value| TinyMistral4 {
            meta: vec![(key.to_string(), value)],
            ..TinyMistral4::default()
        };
        let off = write_mistral4(&wide("attention.temperature_scale", Value::F32(0.0)));
        let long = write_mistral4(&wide("attention.temperature_length", Value::U32(1000)));
        // Positions 0..8: the factor is exactly one everywhere.
        let short = |m: &TinyModel| step(&mut load(m, 32), &PROMPT[..8], 0, Want::Last);
        assert_eq!(short(&default), short(&off));
        assert_eq!(short(&default), short(&long));
        // Position 8 and later are scaled in the default model only.
        let full = |m: &TinyModel| step(&mut load(m, 32), &PROMPT, 0, Want::Last);
        assert_eq!(full(&off), full(&long));
        assert!(
            full(&default)
                .iter()
                .zip(full(&off))
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }

    use crate::provider::local::model::tiny::TinyModel;

    /// The tiny model's latent attention layer, run on random rows, equals a
    /// direct f64 implementation of the transformers (expanded) form on the
    /// same dequantized weights: per-head keys and values built from the
    /// latent, rotary key shared by every head, YaRN inverse frequencies from
    /// the published formulas, the softmax scale `mscale^2 / sqrt(128)`-style
    /// and the per-position Llama 4 query scaling.
    #[test]
    fn latent_attention_matches_the_expanded_f64_reference() {
        let model = write_mistral4(&TinyMistral4::default());
        let m = load_dequantized(&model);
        let il = 1;
        let Mixer::Mla(mw) = &m.layers[il].mixer else {
            panic!("a latent layer");
        };
        let (nh, nope, rot, kvl, vh) = (M4_N_HEAD, M4_NOPE, M4_ROT, M4_KV_LORA, M4_V_HEAD);
        let qh = nope + rot;
        let t = 20;
        let h = crate::provider::local::model::tiny::Lcg(9).values(t * N_EMBD, 1.0);
        let w = |part: &str| tensor(&model, &name::blk(il, part));
        let (q_a, q_a_norm, q_b) = (w(name::ATTN_Q_A), w(name::ATTN_Q_A_NORM), w(name::ATTN_Q_B));
        let (kv_a, kv_a_norm) = (w(name::ATTN_KV_A_MQA), w(name::ATTN_KV_A_NORM));
        let (k_b, v_b, wo) = (w(name::ATTN_K_B), w(name::ATTN_V_B), w(name::ATTN_OUTPUT));
        let q_lora = q_a_norm.len();
        let eps = 1e-6f64;
        let rms = |v: &[f64], weight: &[f64]| -> Vec<f64> {
            let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64 + eps).sqrt();
            v.iter().zip(weight).map(|(x, w)| x * inv * w).collect()
        };
        // transformers `_compute_yarn_parameters` (truncate = true).
        let (base, factor, orig, fast, slow) = (10_000f64, 2f64, M4_ORIG_CTX as f64, 32f64, 1f64);
        let dim = rot as f64;
        let corr =
            |n: f64| dim * (orig / (n * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
        let low = corr(fast).floor().max(0.0);
        let high = corr(slow).ceil().min(dim - 1.0);
        let inv: Vec<f64> = (0..rot / 2)
            .map(|i| {
                let p = base.powf(2.0 * i as f64 / dim);
                let (extrapolation, interpolation) = (1.0 / p, 1.0 / (factor * p));
                let ramp = ((i as f64 - low) / if high == low { 0.001 } else { high - low })
                    .clamp(0.0, 1.0);
                interpolation * ramp + extrapolation * (1.0 - ramp)
            })
            .collect();
        // Adjacent pairs, as `rope_interleave` rotates them.
        let rotate = |v: &mut [f64], pos: usize| {
            for (i, f) in inv.iter().enumerate() {
                let (c, s) = ((pos as f64 * f).cos(), (pos as f64 * f).sin());
                let (a, b) = (v[2 * i], v[2 * i + 1]);
                v[2 * i] = a * c - b * s;
                v[2 * i + 1] = a * s + b * c;
            }
        };
        let mscale = 1.0 + 0.1 * factor.ln();
        let scale = mscale * mscale / (qh as f64).sqrt();
        let llama4 = |pos: usize| 1.0 + 0.3 * (1.0 + (pos / 8) as f64).ln();

        let mut q_nope = vec![vec![0f64; nh * nope]; t];
        let mut q_rot = vec![vec![0f64; nh * rot]; t];
        let mut k_nope = vec![vec![0f64; nh * nope]; t];
        let mut k_rot = vec![vec![0f64; rot]; t];
        let mut values = vec![vec![0f64; nh * vh]; t];
        for (pos, row) in h.chunks(N_EMBD).enumerate() {
            let x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let cq = rms(&matvec(&q_a, N_EMBD, &x), &q_a_norm);
            let q = matvec(&q_b, q_lora, &cq);
            for head in 0..nh {
                let at = head * qh;
                q_nope[pos][head * nope..(head + 1) * nope].copy_from_slice(&q[at..at + nope]);
                let mut r = q[at + nope..at + qh].to_vec();
                rotate(&mut r, pos);
                q_rot[pos][head * rot..(head + 1) * rot].copy_from_slice(&r);
            }
            let kv = matvec(&kv_a, N_EMBD, &x);
            let c = rms(&kv[..kvl], &kv_a_norm);
            k_rot[pos].copy_from_slice(&kv[kvl..]);
            let mut r = k_rot[pos].clone();
            rotate(&mut r, pos);
            k_rot[pos] = r;
            for head in 0..nh {
                for n in 0..nope {
                    k_nope[pos][head * nope + n] = (0..kvl)
                        .map(|j| k_b[(head * kvl + j) * nope + n] * c[j])
                        .sum();
                }
                for j in 0..vh {
                    values[pos][head * vh + j] = (0..kvl)
                        .map(|c_| v_b[(head * vh + j) * kvl + c_] * c[c_])
                        .sum();
                }
            }
        }
        let mut want = Vec::new();
        for pos in 0..t {
            let s = llama4(pos);
            let mut o = vec![0f64; nh * vh];
            for head in 0..nh {
                let qn = &q_nope[pos][head * nope..(head + 1) * nope];
                let qr = &q_rot[pos][head * rot..(head + 1) * rot];
                let scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        let kn = &k_nope[j][head * nope..(head + 1) * nope];
                        let a: f64 = qn.iter().zip(kn).map(|(a, b)| a * b).sum();
                        let b: f64 = qr.iter().zip(&k_rot[j]).map(|(a, b)| a * b).sum();
                        (a + b) * s * scale
                    })
                    .collect();
                let top = scores.iter().fold(f64::MIN, |a, &b| a.max(b));
                let exps: Vec<f64> = scores.iter().map(|x| (x - top).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (j, e) in exps.iter().enumerate() {
                    for c in 0..vh {
                        o[head * vh + c] += e / sum * values[j][head * vh + c];
                    }
                }
            }
            want.extend(matvec(&wo, nh * vh, &o));
        }

        let input = Tensor::from_vec(h, (1, t, N_EMBD), &Device::Cpu).unwrap();
        let mut pools = KvPools::new(m.kv_layout());
        let mut seq = pools.seq();
        pools.reserve(&mut seq, t).unwrap();
        let segs = [solo_seg(t, &seq.tables)];
        let calls = m.attn_calls(il, &segs, None).unwrap();
        let got = m
            .mla_in(&calls, mw, &input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let top = want.iter().fold(1.0f64, |a, v| a.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-4 * top,
                "output {i}: {g} vs {w}"
            );
        }
    }

    /// The MoE FFN of the tiny model: softmax routing with the weights
    /// renormalised and scaled by `expert_weights_scale`, the fused gate/up
    /// halves, and a shared expert added without a gate, against a per-token
    /// reference on the same dequantized weights.
    #[test]
    fn the_moe_ffn_scales_routed_weights_and_adds_the_ungated_shared_expert() {
        let model = write_mistral4(&TinyMistral4::default());
        let content = content_of(&model);
        let hp = read_hparams(&content).unwrap();
        let moe = hp.moe.unwrap();
        assert!(moe.norm_weights && moe.scale == 1.5);
        let mut loader = super::super::Loader {
            content: &content,
            reader: BufReader::new(File::open(&model.path).unwrap()),
            dequantize: true,
            requantize: false,
        };
        let layout = LayerLayout::read(&hp, 0, &hp.layers[0], &content).unwrap();
        assert_eq!(
            layout.ffn,
            Ffn::Moe {
                ff: M4_FF,
                shared_ff: Some(32),
                fused: true
            }
        );
        let lw =
            super::super::load_layer(&mut loader, &hp, 0, &hp.layers[0], &layout, None).unwrap();
        let t = 6;
        let values: Vec<f32> = (0..t * N_EMBD)
            .map(|i| ((i * 29 % 19) as f32 - 9.0) * 0.12)
            .collect();
        let f = Tensor::from_vec(values, (1, t, N_EMBD), &Device::Cpu).unwrap();
        let got = lw.ffn.forward(&f, Act::Silu).unwrap();
        let super::super::FfnWeights::Moe {
            spec,
            router,
            experts: super::super::ExpertBank::Resident(experts),
            shared: Some(shared),
        } = &lw.ffn
        else {
            panic!("resident experts plus a shared expert");
        };
        assert!(shared.gate_inp.is_none(), "the shared expert is ungated");
        let flat = f.reshape((t, N_EMBD)).unwrap();
        let vec_of = |x: Tensor| x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let routing = crate::provider::local::model::moe::route(
            &vec_of(router.forward(&flat).unwrap()),
            t,
            spec,
        );
        let mut want = vec![0f64; t * N_EMBD];
        for (e, picks) in routing.per_expert.iter().enumerate() {
            for &(token, weight) in picks {
                let row = flat.narrow(0, token as usize, 1).unwrap();
                let hidden = (experts[e].gate.forward(&row).unwrap().silu().unwrap()
                    * experts[e].up.forward(&row).unwrap())
                .unwrap();
                let y = vec_of(experts[e].down.forward(&hidden).unwrap());
                for (o, v) in want[token as usize * N_EMBD..][..N_EMBD].iter_mut().zip(&y) {
                    *o += f64::from(weight) * f64::from(*v);
                }
            }
        }
        // Renormalised then scaled: a token's weights sum to the scale.
        for token in 0..t as u32 {
            let sum: f32 = routing
                .per_expert
                .iter()
                .flatten()
                .filter(|(tk, _)| *tk == token)
                .map(|(_, w)| *w)
                .sum();
            assert!((sum - 1.5).abs() < 1e-5, "token {token}: {sum}");
        }
        for token in 0..t {
            let row = flat.narrow(0, token, 1).unwrap();
            let hidden = (shared.gate.forward(&row).unwrap().silu().unwrap()
                * shared.up.forward(&row).unwrap())
            .unwrap();
            let y = vec_of(shared.down.forward(&hidden).unwrap());
            for (o, v) in want[token * N_EMBD..][..N_EMBD].iter_mut().zip(&y) {
                *o += f64::from(*v);
            }
        }
        for (a, b) in vec_of(got).iter().zip(&want) {
            assert!((f64::from(*a) - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// Leading dense blocks (`leading_dense_block_count`, DeepSeek style) run
    /// a dense gated FFN in the first layers and experts in the rest.
    #[test]
    fn leading_dense_layers_run_a_dense_ffn() {
        let model = write_mistral4(&TinyMistral4 {
            dense_layers: 2,
            ..TinyMistral4::default()
        });
        let content = content_of(&model);
        let hp = read_hparams(&content).unwrap();
        assert_eq!(hp.leading_dense, 2);
        let ffn = |il| {
            LayerLayout::read(&hp, il, &hp.layers[il], &content)
                .unwrap()
                .ffn
        };
        for il in 0..2 {
            assert_eq!(
                ffn(il),
                Ffn::Dense {
                    ff: M4_DENSE_FF,
                    fused: false
                },
                "layer {il}"
            );
        }
        assert!(matches!(ffn(2), Ffn::Moe { fused: true, .. }));
        let m = load(&model, 32);
        let kinds: Vec<bool> = m
            .layers
            .iter()
            .map(|l| matches!(l.ffn, super::super::FfnWeights::Moe { .. }))
            .collect();
        assert_eq!(kinds, [false, false, true]);
        let out = step(&mut load(&model, 32), &PROMPT, 0, Want::Last);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// The loader names what a latent model lacks.
    #[test]
    fn a_latent_model_missing_a_tensor_fails_the_load_naming_it() {
        let model = write_mistral4(&TinyMistral4::default());
        let content = content_of(&model);
        let hp = read_hparams(&content).unwrap();
        // Same file, but ask for a layer past the written ones.
        let e = LayerLayout::read(&hp, 7, &hp.layers[0], &content)
            .err()
            .unwrap();
        assert!(e.to_string().contains("blk.7."), "{e}");
    }
    /// Env-gated (`CUCA_MISTRAL4_GGUF`): a published `mistral4` GGUF (only
    /// its header is read) resolves to the latent geometry the engine
    /// computes, with every layer's tensors present at the shapes the loader
    /// expects.
    #[test]
    fn a_published_header_resolves_to_the_latent_geometry() {
        let Some(path) = std::env::var_os("CUCA_MISTRAL4_GGUF") else {
            eprintln!("SKIP: set CUCA_MISTRAL4_GGUF");
            return;
        };
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(path).unwrap())).unwrap();
        let hp = read_hparams(&content).unwrap();
        assert_eq!(hp.arch.name(), "mistral4");
        assert_eq!((hp.n_layer, hp.n_embd), (36, 4096));
        let ls = &hp.layers[0];
        assert_eq!((ls.n_head, ls.n_kv_head, ls.head_dim), (32, 1, 320));
        assert!(
            (f64::from(ls.scale) - 0.194_977).abs() < 1e-5,
            "{}",
            ls.scale
        );
        for (il, spec) in hp.layers.iter().enumerate() {
            let layout = LayerLayout::read(&hp, il, spec, &content).unwrap();
            assert!(layout.mla);
            let Ffn::Moe { ff, shared_ff, .. } = layout.ffn else {
                panic!("layer {il} is not a mixture of experts");
            };
            assert_eq!((ff, shared_ff), (2048, Some(2048)), "layer {il}");
        }
        let bytes = crate::provider::local::model::paged::KvLayout::sized(
            &hp,
            8192,
            &crate::provider::local::model::paged::Sizing::single(8192),
        )
        .pool_bytes(&hp);
        eprintln!("MISTRAL4|kv_pool_bytes_at_8192|{bytes}");
    }
}
