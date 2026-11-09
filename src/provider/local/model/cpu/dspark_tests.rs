//! The DSpark drafter against the torch oracle (`mtp.json`, design §9) and
//! against its own invariants.
//!
//! `CUCA_LOCAL_DS41_REF` names the oracle directory: the drafter of its
//! `target/` checkpoint is converted at `f16` and run as the reference does
//! (`forward_spec`: the stages are seeded with the main hidden rows of every
//! prompt position, then the block is drafted from the anchor token at
//! `start_pos = T - 1`), with the simulated quantization off like
//! `noquant/`. Without the variable the oracle tests print `SKIP:` and pass.

use std::path::{Path, PathBuf};

use crate::provider::local::convert::convert_set;
use crate::provider::local::convert::deepseek41::plan::Role;
use crate::provider::local::convert::hf::CheckpointDir;
use crate::provider::local::gguf::Content;
use crate::provider::local::{ConvertConfig, EngramQuant, WeightQuant};

use super::dspark::DsparkNet;
use super::hyper::without_qdq;
use super::hyper_tests::{converted_oracle, oracle_root, oracle_tokens, read_f32};
use super::*;

/// Context both models of the oracle tests are opened with.
const CTX: usize = 64;

fn converted_drafter(root: &Path) -> PathBuf {
    let ck = CheckpointDir::open(&root.join("target")).expect("the oracle checkpoint opens");
    let dir = std::env::temp_dir().join(format!("cuca-ds41-dspark-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = ConvertConfig {
        quant: WeightQuant::F16,
        engram_quant: EngramQuant::Bf16,
    };
    convert_set(&ck, &dir.join("model.gguf"), config, Role::Drafter, None)
        .expect("the oracle's drafter converts")
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

struct Loaded {
    target: Transformer,
    net: DsparkNet,
}

fn load(root: &Path) -> Loaded {
    let main = Content::open(&converted_oracle(root, WeightQuant::F16)).unwrap();
    let drafter = Content::open(&converted_drafter(root)).unwrap();
    without_qdq(|| {
        let target = Transformer::load(&main, CTX).unwrap();
        let net = DsparkNet::open(&drafter, &target, CTX).unwrap();
        Loaded { target, net }
    })
}

/// Draft from the oracle's anchor; returns the draft ids, the biased logits of
/// each block row and the confidence.
fn draft(l: &Loaded, anchor: u32, len: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut logits = Vec::new();
    let (ids, confidence) = l
        .net
        .draft_scored(&l.target, anchor, len, 5, &mut |row: &mut [f32]| {
            logits.extend_from_slice(row);
            Some(argmax(row))
        })
        .unwrap();
    (ids, logits, confidence)
}

fn mtp(root: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(root.join("noquant/mtp.json")).unwrap()).unwrap()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn the_block_matches_the_reference_drafts_logits_and_confidence() {
    let Some(root) = oracle_root() else { return };
    let want = mtp(&root);
    let t = want["n_tokens"].as_u64().unwrap() as usize;
    let anchor = want["anchor_token"].as_u64().unwrap() as u32;
    let mut l = load(&root);
    let hidden = read_f32(&root.join("noquant/main_hidden.f32"));
    assert_eq!(hidden.len(), t * l.net.feature_width());
    l.net.observe(0, &hidden).unwrap();
    assert_eq!(l.net.held().1, t);

    let (got_ids, logits, confidence) = draft(&l, anchor, t);
    let want_logits = read_f32(&root.join("noquant/mtp_logits.f32"));
    let worst = max_abs(&logits, &want_logits);
    eprintln!("DSPARK|block logits max |diff| {worst:.3e}");
    assert_eq!(logits.len(), want_logits.len());
    assert!(worst < 1e-4, "block logits: max |diff| {worst:.3e}");
    assert_eq!(got_ids, ids(&want["draft_ids"]));
    let want_conf: Vec<f32> = want["confidence"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let worst = max_abs(&confidence, &want_conf);
    eprintln!("DSPARK|confidence max |diff| {worst:.3e}");
    assert!(worst < 1e-4, "confidence: max |diff| {worst:.3e}");
}

/// The features the target itself hands over give the same block as the
/// reference's `main_hidden`.
#[test]
fn the_targets_own_features_drive_the_same_block() {
    let Some(root) = oracle_root() else { return };
    let want = mtp(&root);
    let tokens = oracle_tokens(&root);
    let anchor = want["anchor_token"].as_u64().unwrap() as u32;
    let mut l = load(&root);
    let out = l
        .target
        .forward_ext(
            &tokens,
            0,
            Want::Last,
            Hidden::Layers(l.net.features()),
            false,
        )
        .unwrap();
    let hidden = out.hidden.unwrap();
    let reference = read_f32(&root.join("noquant/main_hidden.f32"));
    let worst = max_abs(&hidden, &reference);
    assert!(worst < 5e-4, "features: max |diff| {worst:.3e}");
    l.net.observe(0, &hidden).unwrap();
    let (got_ids, ..) = draft(&l, anchor, tokens.len());
    assert_eq!(got_ids, ids(&want["draft_ids"]));
}

/// Seeding the rings in several steps, or past a truncation, is the same as in
/// one (the ring is addressed by position).
#[test]
fn chunked_and_replayed_observation_matches_one_pass() {
    let Some(root) = oracle_root() else { return };
    let want = mtp(&root);
    let t = want["n_tokens"].as_u64().unwrap() as usize;
    let anchor = want["anchor_token"].as_u64().unwrap() as u32;
    let hidden = read_f32(&root.join("noquant/main_hidden.f32"));
    let mut l = load(&root);
    let w = l.net.feature_width();
    for (a, b) in [(0, 7), (7, 8), (8, 31), (31, t)] {
        l.net.observe(a, &hidden[a * w..b * w]).unwrap();
    }
    let (_, once, _) = draft(&l, anchor, t);
    // Forget the tail and replay it.
    l.net.truncate(20);
    assert_eq!(l.net.held().1, 20);
    let (none, ..) = draft(&l, anchor, t);
    assert!(none.is_empty(), "a ring short of `len` drafts nothing");
    l.net.observe(20, &hidden[20 * w..t * w]).unwrap();
    let (_, replayed, _) = draft(&l, anchor, t);
    assert_eq!(once, replayed);
}
