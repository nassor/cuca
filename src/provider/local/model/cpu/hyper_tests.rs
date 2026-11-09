//! The hyper-connection engine against the torch oracle (design §9, L1).
//!
//! `CUCA_LOCAL_DS41_REF` names the oracle directory (`make_ref.py`): the tiny
//! release-format checkpoint in `target/` is converted here at `f16` (every
//! tensor, the MXFP4 experts included, dequantized exactly) and the CPU
//! engine's logits are compared with the dumps of the reference run with the
//! simulated quantization switched off (`noquant/`), which is the arithmetic
//! this engine implements. Without the variable each test prints `SKIP:` and
//! passes.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use crate::provider::local::convert::convert_set;
use crate::provider::local::convert::deepseek41::plan::Role;
use crate::provider::local::convert::hf::CheckpointDir;
use crate::provider::local::gguf::Content;
use crate::provider::local::tokenizer::golden::skip;
use crate::provider::local::{ConvertConfig, EngramQuant, WeightQuant};

use super::hyper::without_qdq;
use super::*;

/// The oracle directory, or `None` after a `SKIP:` line.
pub(in super::super) fn oracle_root() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("CUCA_LOCAL_DS41_REF").map(PathBuf::from) else {
        skip("set CUCA_LOCAL_DS41_REF to the ds41 oracle directory");
        return None;
    };
    if !dir.join("target").is_dir() || !dir.join("noquant").is_dir() {
        skip(&format!(
            "`{}` has no target/ and noquant/ (run make_ref.py)",
            dir.display()
        ));
        return None;
    }
    Some(dir)
}

/// The oracle checkpoint converted at `quant` (cached for the process; the
/// files live in the temp directory).
pub(in super::super) fn converted_oracle(root: &Path, quant: WeightQuant) -> PathBuf {
    static SETS: Mutex<Vec<(String, PathBuf)>> = Mutex::new(Vec::new());
    let mut sets = SETS.lock().unwrap_or_else(PoisonError::into_inner);
    let key = format!("{quant:?}");
    if let Some((_, p)) = sets.iter().find(|(k, _)| *k == key) {
        return p.clone();
    }
    let ck = CheckpointDir::open(&root.join("target")).expect("the oracle checkpoint opens");
    let dir = std::env::temp_dir().join(format!(
        "cuca-ds41-oracle-{}-{}",
        std::process::id(),
        key.to_lowercase()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("model.gguf");
    let config = ConvertConfig {
        quant,
        engram_quant: EngramQuant::Bf16,
    };
    let path = convert_set(&ck, &out, config, Role::Main, None).expect("the oracle converts");
    sets.push((key, path.clone()));
    path
}

pub(in super::super) fn oracle_tokens(root: &Path) -> Vec<u32> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("noquant/meta.json")).unwrap())
            .unwrap();
    meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

pub(in super::super) fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The unquantized reference engine over the f16-converted oracle.
fn reference_engine(root: &Path) -> Transformer {
    let path = converted_oracle(root, WeightQuant::F16);
    let content = Content::open(&path).unwrap();
    without_qdq(|| Transformer::load(&content, 64)).unwrap()
}

#[test]
fn one_pass_matches_the_unquantized_reference_logits() {
    let Some(root) = oracle_root() else { return };
    let tokens = oracle_tokens(&root);
    let want = read_f32(&root.join("noquant/logits_full.f32"));
    let mut m = reference_engine(&root);
    let got = m.forward(&tokens, 0, Want::All).unwrap().unwrap();
    assert_eq!(got.len(), want.len());
    let vocab = m.vocab();
    for (pos, (g, w)) in got
        .chunks_exact(vocab)
        .zip(want.chunks_exact(vocab))
        .enumerate()
    {
        let worst = max_abs(g, w);
        assert!(worst < 5e-4, "position {pos}: max |diff| {worst:.3e}");
    }
}

#[test]
fn layer_inputs_match_the_reference_dump() {
    let Some(root) = oracle_root() else { return };
    let tokens = oracle_tokens(&root);
    let dump: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("noquant/debug_layers.json")).unwrap(),
    )
    .unwrap();
    let mut m = reference_engine(&root);
    let n_layer = 6;
    let set = LayerSet::new(0..n_layer).unwrap();
    let out = m
        .forward_ext(&tokens, 0, Want::Last, Hidden::Layers(set), false)
        .unwrap();
    let hidden = out.hidden.unwrap();
    let d = m.n_embd();
    let last = tokens.len() - 1;
    for il in 0..n_layer {
        let got = &hidden[(last * n_layer + il) * d..(last * n_layer + il + 1) * d];
        let want: Vec<f32> = dump["layer_in_mean_hc"][il]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let worst = max_abs(got, &want);
        assert!(worst < 5e-4, "layer {il} input: max |diff| {worst:.3e}");
    }
}

// ---- the tiny model (no oracle needed) ----------------------------------------

use super::super::tiny::{TinyDeepseek41, write_deepseek41};

/// Context of the tiny model's cache.
const CTX: usize = 96;

/// The storage and indexing variants the invariants hold for: Q8_0 experts, MXFP4
/// experts (the checkpoint's native type) and an indexer without candidate
/// blocks.
fn variants() -> Vec<TinyDeepseek41> {
    vec![
        TinyDeepseek41::default(),
        TinyDeepseek41 {
            mxfp4: true,
            ..TinyDeepseek41::default()
        },
        TinyDeepseek41 {
            candidates: None,
            ..TinyDeepseek41::default()
        },
    ]
}

fn tiny_engine(s: &TinyDeepseek41) -> (super::super::tiny::TinyModel, Transformer) {
    let model = write_deepseek41(s);
    let content = Content::open(&model.path).unwrap();
    let m = Transformer::load(&content, CTX).unwrap();
    (model, m)
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 5 + 3 + i / 7) % 16).collect()
}

#[test]
fn the_tiny_model_runs() {
    let (_m, mut e) = tiny_engine(&TinyDeepseek41::default());
    let p = prompt(40);
    let logits = e.forward(&p, 0, Want::All).unwrap().unwrap();
    assert_eq!(logits.len(), 40 * 16);
    assert!(logits.iter().all(|v| v.is_finite()));
    let lo = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("tiny logits in [{lo}, {hi}]");
    assert!(hi - lo > 1e-3);
}

fn all_logits(e: &mut Transformer, tokens: &[u32], start: usize) -> Vec<f32> {
    e.forward_ext(tokens, start, Want::All, Hidden::None, true)
        .unwrap()
        .logits
        .unwrap()
}

/// `tokens` through the engine in chunks of `size` (the last one shorter),
/// the logits of every position.
fn chunked(e: &mut Transformer, tokens: &[u32], size: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < tokens.len() {
        let end = (at + size).min(tokens.len());
        out.extend(all_logits(e, &tokens[at..end], at));
        at = end;
    }
    out
}

fn chunking_never_changes_a_bit(spec: &TinyDeepseek41) {
    let (model, mut e) = tiny_engine(spec);
    let p = prompt(70);
    let want = all_logits(&mut e, &p, 0);
    for size in [1, 2, 3, 7, 16, 33, 64] {
        let content = Content::open(&model.path).unwrap();
        let mut e = Transformer::load(&content, CTX).unwrap();
        assert_eq!(chunked(&mut e, &p, size), want, "chunks of {size}");
    }
}

#[test]
fn the_window_and_the_compressed_rows_both_matter() {
    // A model that sees the whole prompt differs from one that only sees the
    // last window: change a token outside the window but inside the
    // compressed rows' reach and the last logits move.
    let (_m, mut a) = tiny_engine(&TinyDeepseek41::default());
    let (_m2, mut b) = tiny_engine(&TinyDeepseek41::default());
    let p = prompt(60);
    let mut q = p.clone();
    q[10] = (q[10] + 1) % 16;
    let la = all_logits(&mut a, &p, 0);
    let lb = all_logits(&mut b, &q, 0);
    let last = |l: &[f32]| l[l.len() - 16..].to_vec();
    assert_ne!(last(&la), last(&lb));
}

/// How the indexer's scoring is split over the pool (rows per job, floats per
/// group of tokens) changes no bit: one pass and token-by-token decoding give
/// the default split's logits for jobs of 1, 3 and 7 rows and groups of one
/// token or a few.
fn the_indexer_split_never_changes_a_bit(spec: &TinyDeepseek41) {
    use super::hyper::with_index_split;
    let (model, mut e) = tiny_engine(spec);
    let p = prompt(90);
    let want = all_logits(&mut e, &p, 0);
    for (chunk, budget) in [(1, 1), (3, 1), (3, 40), (7, 90), (64, 1 << 22)] {
        let got = with_index_split(chunk, budget, || {
            let content = Content::open(&model.path).unwrap();
            let mut e = Transformer::load(&content, CTX).unwrap();
            let mut got = all_logits(&mut e, &p[..60], 0);
            for (i, t) in p[60..].iter().enumerate() {
                got.extend(all_logits(&mut e, &[*t], 60 + i));
            }
            got
        });
        assert_eq!(got, want, "jobs of {chunk} rows, groups of {budget}");
        let whole = with_index_split(chunk, budget, || {
            let content = Content::open(&model.path).unwrap();
            all_logits(&mut Transformer::load(&content, CTX).unwrap(), &p, 0)
        });
        assert_eq!(
            whole, want,
            "one pass, jobs of {chunk} rows, groups of {budget}"
        );
    }
}

fn rolling_back_into_the_last_call_restores_the_compressor(spec: &TinyDeepseek41) {
    let (model, mut e) = tiny_engine(spec);
    let p = prompt(60);
    let want = all_logits(&mut e, &p, 0);
    // Every position of the last call of 20 rows can be returned to, odd ones
    // included (they hold a half-pooled pair); the rest replays from scratch.
    for keep in [59, 51, 48, 47, 43, 41, 40, 39, 25] {
        let content = Content::open(&model.path).unwrap();
        let mut e = Transformer::load(&content, CTX).unwrap();
        all_logits(&mut e, &p[..40], 0);
        all_logits(&mut e, &p[40..], 40);
        let kept = e.truncate_cache(keep);
        let got = all_logits(&mut e, &p[kept..], kept);
        assert_eq!(got, want[kept * 16..], "rolled back to {keep}, kept {kept}");
    }
}

fn greedy(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn a_decode_continuation_equals_one_pass(spec: &TinyDeepseek41) {
    let (model, mut e) = tiny_engine(spec);
    let p = prompt(55);
    let want = all_logits(&mut e, &p, 0);
    let content = Content::open(&model.path).unwrap();
    let mut s = Transformer::load(&content, CTX).unwrap();
    all_logits(&mut s, &p[..30], 0);
    let mut got = Vec::new();
    for (i, t) in p[30..].iter().enumerate() {
        got.extend(all_logits(&mut s, &[*t], 30 + i));
    }
    assert_eq!(got, want[30 * 16..]);
    let _ = greedy(&got[..16]);
}

fn batched_rows_equal_solo_rows(spec: &TinyDeepseek41) {
    use super::super::paged::KvPools;
    use super::super::tiny::load_sized;
    use super::super::{SeqRows, StateSlot};
    let model = write_deepseek41(spec);
    let seqs_tokens: [Vec<u32>; 3] = [prompt(50), prompt(61)[11..].to_vec(), prompt(37)];
    // (sequence, rows to take) per step.
    let steps: &[&[(usize, usize)]] = &[
        &[(0, 21), (1, 9), (2, 30)],
        &[(0, 1), (1, 1), (2, 1)],
        &[(0, 4), (2, 3)],
        &[(1, 17), (2, 1)],
        &[(0, 24), (1, 20)],
    ];
    let sizing = Sizing {
        max_sequences: 3,
        kv_tokens: 192,
        max_rows: 32,
    };
    let mut batched = load_sized(&model, CTX, &sizing);
    let mut pools = KvPools::new(batched.kv_layout());
    let mut seqs: Vec<_> = (0..3).map(|_| pools.seq()).collect();
    let content = Content::open(&model.path).unwrap();
    let mut solo: Vec<Transformer> = (0..3)
        .map(|_| Transformer::load(&content, CTX).unwrap())
        .collect();
    let mut lens = [0usize; 3];
    for (n, step) in steps.iter().enumerate() {
        for &(i, rows) in *step {
            pools.reserve(&mut seqs[i], lens[i] + rows).unwrap();
        }
        let batch = Batch {
            rows: step
                .iter()
                .map(|&(i, rows)| SeqRows {
                    tokens: &seqs_tokens[i][lens[i]..lens[i] + rows],
                    start_pos: lens[i],
                    history: &seqs_tokens[i][..lens[i]],
                    tables: &seqs[i].tables,
                    state: StateSlot(i),
                    want: Want::All,
                    hidden: Hidden::PostNormAll,
                })
                .collect(),
            exact: true,
        };
        let outs = batched.forward_batch(&batch).unwrap();
        for (&(i, rows), out) in step.iter().zip(&outs) {
            let alone = solo[i]
                .forward_ext(
                    &seqs_tokens[i][lens[i]..lens[i] + rows],
                    lens[i],
                    Want::All,
                    Hidden::PostNormAll,
                    true,
                )
                .unwrap();
            assert_eq!(out.logits, alone.logits, "step {n}, sequence {i}");
            assert_eq!(out.hidden, alone.hidden, "step {n}, sequence {i}");
            lens[i] += rows;
        }
        // The window leaves blocks behind as the engine does.
        for &(i, _) in *step {
            pools.release_window(&mut seqs[i], lens[i], None);
        }
    }
}

fn a_prefix_found_in_the_store_equals_a_cold_prefill(spec: &TinyDeepseek41) {
    use super::super::paged::KvPools;
    use super::super::tiny::load_sized;
    use super::super::{SeqRows, StateSlot};
    let model = write_deepseek41(spec);
    let sizing = Sizing {
        max_sequences: 2,
        kv_tokens: 192,
        max_rows: 32,
    };
    let mut m = load_sized(&model, CTX, &sizing);
    let mut pools = KvPools::new(m.kv_layout());
    let p = prompt(75);
    // The first prompt fills the store, 16 rows at a time as the engine does.
    let mut a = pools.seq();
    let mut at = 0;
    while at < p.len() {
        let end = (at + 16).min(p.len());
        pools.reserve(&mut a, end).unwrap();
        let batch = Batch {
            rows: vec![SeqRows {
                tokens: &p[at..end],
                start_pos: at,
                history: &p[..at],
                tables: &a.tables,
                state: StateSlot(0),
                want: Want::None,
                hidden: Hidden::None,
            }],
            exact: true,
        };
        m.forward_batch(&batch).unwrap();
        a.len = end;
        pools.release_window(&mut a, end, Some(&p));
        pools.publish(&mut a, &p);
        at = end;
    }
    pools.finish(&mut a, &p);
    // A prompt with the same first 70 tokens finds the prefix: a multiple of
    // 32 (the compressed blocks span 32 tokens).
    let mut q = p[..70].to_vec();
    q.extend_from_slice(&[1, 2, 3, 4, 5]);
    let hit = pools.lookup(&q);
    assert_eq!(hit.len, 64, "the prefix is a whole compressed block");
    let mut b = hit.into_seq();
    pools.reserve(&mut b, q.len()).unwrap();
    let batch = Batch {
        rows: vec![SeqRows {
            tokens: &q[64..],
            start_pos: 64,
            history: &q[..64],
            tables: &b.tables,
            state: StateSlot(1),
            want: Want::All,
            hidden: Hidden::None,
        }],
        exact: true,
    };
    let warm = m.forward_batch(&batch).unwrap().remove(0).logits.unwrap();
    let content = Content::open(&model.path).unwrap();
    let mut cold = Transformer::load(&content, CTX).unwrap();
    let cold = all_logits(&mut cold, &q, 0);
    assert_eq!(warm, cold[64 * 16..]);
}

fn load_tiered(model: &super::super::tiny::TinyModel, ram_bytes: u64) -> Transformer {
    use crate::provider::local::{ExpertPlacement, ExpertTiers};
    let content = Content::open(&model.path).unwrap();
    let tiers = ExpertTiers {
        ram_bytes,
        loader_threads: 2,
        ..ExpertTiers::default()
    };
    Transformer::load_placed(
        &content,
        CTX,
        &ExpertPlacement::Tiered(tiers),
        &Sizing::single(CTX),
        &crate::provider::local::EngramCache::default(),
    )
    .unwrap()
}

fn tiered_experts_match_resident_bit_for_bit(spec: &TinyDeepseek41) {
    let model = write_deepseek41(spec);
    let content = Content::open(&model.path).unwrap();
    let mut resident = Transformer::load(&content, CTX).unwrap();
    let p = prompt(50);
    let want = chunked(&mut resident, &p, 16);
    // Room for a handful of experts per layer: the rest is read on demand.
    let mut tiered = load_tiered(&model, 128 * 1024);
    assert_eq!(chunked(&mut tiered, &p, 16), want);
}

#[test]
fn simulated_quantization_matches_the_quantized_reference() {
    // The reference with its quantizers on is chaotic (a rounding decision
    // flipped by one ulp moves a logit by 1e-2), so agreement at this level
    // means the quantizers round the same values, bit for bit.
    let Some(root) = oracle_root() else { return };
    let tokens = oracle_tokens(&root);
    let want = read_f32(&root.join("logits_full.f32"));
    let path = converted_oracle(&root, WeightQuant::F16);
    let content = Content::open(&path).unwrap();
    let mut m = Transformer::load(&content, 64).unwrap();
    let got = m.forward(&tokens, 0, Want::All).unwrap().unwrap();
    let vocab = m.vocab();
    for (pos, (g, w)) in got
        .chunks_exact(vocab)
        .zip(want.chunks_exact(vocab))
        .enumerate()
    {
        assert_eq!(greedy(g), greedy(w), "position {pos}");
    }
    let worst = max_abs(&got, &want);
    assert!(worst < 5e-4, "max |diff| {worst:.3e}");
    // And it is far from the unquantized reference: the quantizers are on.
    let plain = read_f32(&root.join("noquant/logits_full.f32"));
    assert!(max_abs(&got, &plain) > 0.1);
}

#[test]
fn captured_layer_inputs_are_the_drafters_main_hidden_rows() {
    let Some(root) = oracle_root() else { return };
    let tokens = oracle_tokens(&root);
    let want = read_f32(&root.join("noquant/main_hidden.f32"));
    let mut m = reference_engine(&root);
    let set = LayerSet::new([3, 4, 5]).unwrap();
    let out = m
        .forward_ext(&tokens, 0, Want::None, Hidden::Layers(set), false)
        .unwrap();
    let got = out.hidden.unwrap();
    assert_eq!(got.len(), want.len());
    let worst = max_abs(&got, &want);
    assert!(worst < 5e-4, "max |diff| {worst:.3e}");
}

#[test]
fn chunking_never_changes_a_bit_in_every_variant() {
    for spec in variants() {
        chunking_never_changes_a_bit(&spec);
    }
}

#[test]
fn the_indexer_split_never_changes_a_bit_in_every_variant() {
    for spec in variants() {
        the_indexer_split_never_changes_a_bit(&spec);
    }
}

#[test]
fn rolling_back_into_the_last_call_restores_the_compressor_in_every_variant() {
    for spec in variants() {
        rolling_back_into_the_last_call_restores_the_compressor(&spec);
    }
}

#[test]
fn a_decode_continuation_equals_one_pass_in_every_variant() {
    for spec in variants() {
        a_decode_continuation_equals_one_pass(&spec);
    }
}

#[test]
fn batched_rows_equal_solo_rows_in_every_variant() {
    for spec in variants() {
        batched_rows_equal_solo_rows(&spec);
    }
}

#[test]
fn a_prefix_found_in_the_store_equals_a_cold_prefill_in_every_variant() {
    for spec in variants() {
        a_prefix_found_in_the_store_equals_a_cold_prefill(&spec);
    }
}

#[test]
fn tiered_experts_match_resident_bit_for_bit_in_every_variant() {
    for spec in variants() {
        tiered_experts_match_resident_bit_for_bit(&spec);
    }
}

#[test]
fn the_converters_default_output_runs_and_keeps_the_invariants() {
    // Q8_0 dense matrices and the MXFP4 experts, as the converter writes them
    // by default. The rounding of the weights and activations moves this tiny
    // random model's logits by O(1) (its router has near ties), so the check is
    // that the engine is itself: chunking and batching change no bit.
    let Some(root) = oracle_root() else { return };
    let tokens = oracle_tokens(&root);
    let path = converted_oracle(&root, WeightQuant::Q8_0);
    let content = Content::open(&path).unwrap();
    assert_eq!(
        content.info("blk.1.ffn_gate_exps.weight").unwrap().dtype,
        crate::provider::local::gguf::GgufType::Mxfp4
    );
    let mut whole = Transformer::load(&content, 64).unwrap();
    let want = all_logits(&mut whole, &tokens, 0);
    assert!(want.iter().all(|v| v.is_finite()));
    for size in [1, 7, 16] {
        let mut m = Transformer::load(&content, 64).unwrap();
        assert_eq!(chunked(&mut m, &tokens, size), want, "chunks of {size}");
    }
}

#[test]
fn mxfp4_experts_only_add_the_activation_rounding() {
    // One MoE layer over the same inputs: the router and so the routing are
    // identical (its weights are f32), the experts are MXFP4 blocks read by
    // the kernel on one side and their exact f32 values on the other, so the
    // outputs differ by the Q8_0 rounding of the expert inputs alone.
    let spec = TinyDeepseek41 {
        mxfp4: true,
        ..TinyDeepseek41::default()
    };
    let model = write_deepseek41(&spec);
    let content = Content::open(&model.path).unwrap();
    let native = Transformer::load(&content, CTX).unwrap();
    let exact = Transformer::load_dequantized(&content, CTX).unwrap();
    let t = 24;
    let x: Vec<f32> = (0..t * 64)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.04)
        .collect();
    let x = Tensor::from_vec(x, (1, t, 64), &Device::Cpu).unwrap();
    for il in [0, 3] {
        let a = native.layers[il].ffn.forward(&x, native.hp.act()).unwrap();
        let b = exact.layers[il].ffn.forward(&x, exact.hp.act()).unwrap();
        let (a, b) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = max_abs(&a, &b);
        assert!(scale > 0.01, "layer {il}: the layer produces something");
        assert!(
            worst < 0.02 * scale,
            "layer {il}: max |diff| {worst:.3e} of {scale:.3e}"
        );
    }
}

#[test]
fn an_offload_cut_inside_a_group_of_layers_that_share_sources_is_rejected_before_a_device_opens() {
    use crate::provider::local::{EngramCache, ExpertPlacement, LocalDevice};
    let model = write_deepseek41(&TinyDeepseek41::default());
    let content = Content::open(&model.path).unwrap();
    // The tiny model's cache sources are layers 1 and 3: only cuts that start
    // at one of them keep every source with the layers that read it.
    for gpu_layers in [1usize, 2, 4] {
        let err = super::super::Model::load(
            &content,
            CTX,
            LocalDevice::Cuda(0),
            &[],
            &ExpertPlacement::Resident,
            &Sizing::single(CTX),
            Some(gpu_layers),
            &EngramCache::default(),
        )
        .err()
        .expect("the cut is rejected");
        let message = err.to_string();
        assert!(
            message.contains(&format!("gpu_layers {gpu_layers} cuts `deepseek41`"))
                && message.contains("legal values: 0, 3, 5, 6"),
            "{message}"
        );
    }
}

#[test]
fn engram_hash_ids_match_the_reference() {
    use super::super::engram::{MAX_NGRAM, row_ids, window};
    let Some(root) = oracle_root() else { return };
    let tokens = oracle_tokens(&root);
    let dump: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("noquant/debug_layers.json")).unwrap(),
    )
    .unwrap();
    let path = converted_oracle(&root, WeightQuant::F16);
    let hp = super::super::read_hparams(&Content::open(&path).unwrap()).unwrap();
    let es = hp.engram.unwrap();
    let cols = es.cols();
    for (pos, want) in dump["engram_hash_ids"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let w = window(&es, |back| pos.checked_sub(back).map(|p| tokens[p])).unwrap();
        assert!(w.len() == MAX_NGRAM);
        for ordinal in 0..es.layers.len() {
            let mut ids = vec![0u64; cols];
            row_ids(&es, ordinal, &w, &mut ids);
            // The reference numbers the rows of each layer's own table; the
            // converted table holds the layers one after the other.
            let base: u64 = es.head_sizes[..ordinal * cols].iter().sum();
            let want: Vec<u64> = want[ordinal]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let got: Vec<u64> = ids.iter().map(|i| i - base).collect();
            assert_eq!(got, want, "position {pos}, engram layer {ordinal}");
        }
    }
}

#[test]
fn the_metadata_resolves_to_the_layer_roles() {
    use super::super::csa::Candidates;
    let model = write_deepseek41(&TinyDeepseek41::default());
    let hp = super::super::read_hparams(&Content::open(&model.path).unwrap()).unwrap();
    assert_eq!(hp.arch, super::super::arch::Arch::Deepseek41);
    let roles: Vec<_> = hp
        .layers
        .iter()
        .map(|l| {
            l.csa
                .map(|c| (c.ratio, c.kv_source, c.index_source, c.candidates))
        })
        .collect();
    assert_eq!(
        roles,
        vec![
            None,
            Some((2, 1, 1, Candidates::None)),
            Some((2, 1, 1, Candidates::None)),
            Some((1, 3, 3, Candidates::Source)),
            Some((1, 3, 4, Candidates::Consumer)),
            Some((1, 3, 4, Candidates::None)),
        ]
    );
    assert!(
        hp.layers
            .iter()
            .all(|l| l.window == Some(8) && l.kv_is_value)
    );
    // The window-only layer and the compressed ones rotate differently.
    assert_ne!(hp.layers[0].rope, hp.layers[1].rope);
    assert_eq!(hp.layers[1].rope, hp.layers[4].rope);
    let g = hp.csa.unwrap();
    assert_eq!((g.q_lora, g.out_groups, g.out_rank), (32, 2, 16));
    assert_eq!(g.indexer.candidates, Some((3, 2, 4)));
    assert_eq!(hp.hyper.unwrap().streams, 4);
    assert_eq!(hp.engram.unwrap().layers, vec![1, 4]);
}
