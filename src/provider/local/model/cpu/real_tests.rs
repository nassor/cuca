//! The converted real DeepSeek-V4.1-Flash-1017 on the CPU against the
//! real-weights streaming oracle (`.omp/local-models/ref/deepseek-v41-real`,
//! see its README).
//!
//! Gated: `CUCA_LOCAL_DS41_REAL` names the oracle directory and
//! `CUCA_CACHE_DIR` the cache holding the converted set
//! (`converted/deepseek-ai--DeepSeek-V4.1-Flash-1017/<revision>/`); without
//! them the test prints `SKIP:` and passes. It loads the 298 GB backbone with
//! tiered experts (`CUCA_LOCAL_DS41_REAL_RAM`, default 90 GiB of expert RAM), so
//! run it in release mode:
//!
//! ```sh
//! CUCA_LOCAL_DS41_REAL=<oracle dir> CUCA_CACHE_DIR=<cache dir> \
//!   RUSTFLAGS="-C target-cpu=native" cargo test --release --lib --features provider-local \
//!   real_model -- --nocapture
//! ```
//!
//! The 16 prompt ids run in one pass twice: simulated quantization off (the
//! smooth target, compared with `*_off`) and on (the reference's behaviour,
//! compared with `*_on` by top-1, top-10 overlap and KL at the last position;
//! the quantization noise floor of the dumps is the README's).

use std::path::{Path, PathBuf};

use crate::provider::local::gguf::Content;
use crate::provider::local::tokenizer::golden::skip;
use crate::provider::local::{EngramCache, ExpertPlacement, ExpertTiers};

use super::hyper::without_qdq;
use super::hyper_tests::read_f32;
use super::*;

const REVISION: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";
/// The largest logit change the oracle's own quantization noise makes (max
/// |dlogit| ON vs OFF at the last position, 1.33), rounded up.
const NOISE: f64 = 1.5;
/// Token the real model answers with at the last prompt position (`Paris`).
const PARIS: usize = 51119;

pub(in crate::provider::local::model) fn setup() -> Option<(PathBuf, PathBuf)> {
    let (Some(dir), Some(cache)) = (
        std::env::var_os("CUCA_LOCAL_DS41_REAL").map(PathBuf::from),
        std::env::var_os("CUCA_CACHE_DIR").map(PathBuf::from),
    ) else {
        skip("set CUCA_LOCAL_DS41_REAL (oracle dir) and CUCA_CACHE_DIR (converted set)");
        return None;
    };
    let model = cache
        .join("converted/deepseek-ai--DeepSeek-V4.1-Flash-1017")
        .join(REVISION)
        .join("model-q8_0-00001-of-00003.gguf");
    if !model.is_file() || !dir.join("prompt.json").is_file() {
        skip(&format!(
            "`{}` or `{}` is missing",
            model.display(),
            dir.display()
        ));
        return None;
    }
    Some((dir, model))
}

pub(in crate::provider::local::model) fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn load(model: &Path) -> Transformer {
    let gib: u64 = std::env::var("CUCA_LOCAL_DS41_REAL_RAM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90);
    let content = Content::open(model).unwrap();
    let experts = ExpertPlacement::Tiered(ExpertTiers {
        ram_bytes: gib << 30,
        ..ExpertTiers::default()
    });
    Transformer::load_placed(
        &content,
        64,
        &experts,
        &Sizing::single(64),
        &EngramCache::default(),
    )
    .unwrap()
}

/// `log softmax`.
fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = x.iter().map(|&v| (v as f64 - m).exp()).sum();
    x.iter().map(|&v| v as f64 - m - sum.ln()).collect()
}

/// `KL(reference || ours)`.
fn kl(reference: &[f32], ours: &[f32]) -> f64 {
    let (p, q) = (log_softmax(reference), log_softmax(ours));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

fn top10(row: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx.truncate(10);
    idx
}

/// The prompt token ids of the oracle in `dir`.
pub(in crate::provider::local::model) fn prompt_ids(dir: &Path) -> Vec<u32> {
    json(&dir.join("prompt.json"))["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

/// Run the prompt in one pass and compare with the `mode` dumps; returns the
/// KL at the last position.
fn check(dir: &Path, model: &Path, mode: &str, qdq: bool) -> f64 {
    let ids = prompt_ids(dir);
    let started = std::time::Instant::now();
    let mut m = if qdq {
        load(model)
    } else {
        without_qdq(|| load(model))
    };
    let loaded = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let logits = m
        .forward_ext(&ids, 0, Want::All, Hidden::None, true)
        .unwrap()
        .logits
        .unwrap();
    let pass = started.elapsed().as_secs_f64();
    let tiers = m.expert_tiers().map(|t| t.stats());
    score(
        dir,
        mode,
        &ids,
        &logits,
        m.vocab(),
        &format!("cpu|load {loaded:.0}s|pass {pass:.0}s"),
        &format!("{tiers:?}"),
        false,
    )
}

/// Compare the logits of every prompt position (`ids.len() * v` values) with
/// the `mode` dumps of the oracle in `dir`: print the `REAL|` line (`timing`
/// and `extra` are appended as given), assert the last position answers Paris
/// and the dumps' noise floor, and return the KL at the last position.
///
/// `chaotic` is for an engine whose rounding differs from the reference's
/// with the simulated quantization on (the quantizers' rounding decisions flip
/// on a one-ulp difference): its top-1 may differ from the dump's only where
/// the dump's own top two are within its noise floor ([`NOISE`]), and the
/// top-10 overlap may drop to 5 at positions before the last.
#[expect(
    clippy::too_many_arguments,
    reason = "the logits, what they are compared with and what the report says"
)]
pub(in crate::provider::local::model) fn score(
    dir: &Path,
    mode: &str,
    ids: &[u32],
    logits: &[f32],
    v: usize,
    timing: &str,
    extra: &str,
    chaotic: bool,
) -> f64 {
    assert_eq!(logits.len(), ids.len() * v);
    // `CUCA_LOCAL_DS41_REAL_DUMP=<dir>` writes every position's logits, for
    // comparing two engines with each other.
    if let Some(out) = std::env::var_os("CUCA_LOCAL_DS41_REAL_DUMP") {
        let engine = timing.split('|').next().unwrap_or("engine");
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(
            PathBuf::from(out).join(format!("{engine}_{mode}.f32")),
            bytes,
        )
        .unwrap();
    }
    let want = json(&dir.join(format!("topk_{mode}.json")));
    let (mut top1_equal, mut worst, mut overlap_min) = (0, 0f32, 10);
    let mut widest_flip = 0f64;
    for (pos, entry) in want.as_array().unwrap().iter().enumerate() {
        let row = &logits[pos * v..(pos + 1) * v];
        let ref_ids: Vec<usize> = entry["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let mine = top10(row);
        top1_equal += usize::from(mine[0] == ref_ids[0]);
        if mine[0] != ref_ids[0] {
            // The reference's own margin between its top two, and how far we
            // put its top-1 below ours: a near tie flips with rounding.
            let ref_logits = entry["logits"].as_array().unwrap();
            let gap = ref_logits[0].as_f64().unwrap() - ref_logits[1].as_f64().unwrap();
            widest_flip = widest_flip.max(gap);
            eprintln!(
                "REAL|{mode}|position {pos}: ours {} vs reference {}, reference gap {gap:.3}, ours at the reference's top-1 {:.3} below our top-1",
                mine[0],
                ref_ids[0],
                row[mine[0]] - row[ref_ids[0]]
            );
        }
        let overlap = mine.iter().filter(|i| ref_ids.contains(i)).count();
        overlap_min = overlap_min.min(overlap);
        for (id, l) in ref_ids.iter().zip(entry["logits"].as_array().unwrap()) {
            worst = worst.max((row[*id] - l.as_f64().unwrap() as f32).abs());
        }
    }
    let last = &logits[(ids.len() - 1) * v..];
    let reference = read_f32(&dir.join(format!("last_logits_{mode}.f32")));
    let max_diff = last
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let kl = kl(&reference, last);
    eprintln!(
        "REAL|{mode}|{timing}|top-1 equal {top1_equal}/{}|min top-10 overlap {overlap_min}|top-10 logits max|diff| {worst:.3}|last logits max|diff| {max_diff:.3}|KL {kl:.2e}|{extra}",
        ids.len()
    );
    assert_eq!(
        top10(last)[0],
        PARIS,
        "{mode}: the last position answers Paris"
    );
    // The dumps themselves agree with each other (ON vs OFF) at 14 of 16
    // positions and 9-10 of the top 10 at the last: that is the noise floor.
    if chaotic {
        assert!(
            widest_flip <= NOISE,
            "{mode}: top-1 differs where the reference leads by {widest_flip:.3}"
        );
        assert!(overlap_min >= 5, "{mode}: top-10 overlap {overlap_min}");
        let last_ids: Vec<usize> = want.as_array().unwrap()[ids.len() - 1]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let at_last = top10(last).iter().filter(|i| last_ids.contains(i)).count();
        assert!(
            at_last >= 8,
            "{mode}: top-10 overlap {at_last} at the last position"
        );
    } else {
        assert!(
            top1_equal >= 14,
            "{mode}: top-1 equal at {top1_equal} of 16"
        );
        assert!(overlap_min >= 8, "{mode}: top-10 overlap {overlap_min}");
    }
    kl
}

#[test]
fn real_model_matches_the_streaming_oracle_with_quantization_off() {
    let Some((dir, model)) = setup() else { return };
    let kl = check(&dir, &model, "off", false);
    assert!(kl < 5e-3, "KL {kl:.3e}");
}

#[test]
fn real_model_matches_the_streaming_oracle_with_quantization_on() {
    let Some((dir, model)) = setup() else { return };
    let kl = check(&dir, &model, "on", true);
    // The reference's quantization noise floor is ON vs OFF (max |dlogit|
    // 1.33 at the last position): bound the KL by a generous multiple.
    assert!(kl < 5e-2, "KL {kl:.3e}");
}

/// Speed with the experts in the RAM tier (`CUCA_LOCAL_DS41_REAL_SPEED=1`):
/// the prompt cold and warm, then one decode step repeated at the same
/// position. Prints `REAL|speed|cpu|...`; asserts nothing about the numbers.
#[test]
fn real_model_speed() {
    if std::env::var_os("CUCA_LOCAL_DS41_REAL_SPEED").is_none() {
        return;
    }
    let Some((dir, model)) = setup() else { return };
    let ids = prompt_ids(&dir);
    let mut m = load(&model);
    let time = |m: &mut Transformer, tokens: &[u32], at: usize| {
        let started = std::time::Instant::now();
        m.forward(tokens, at, Want::Last).unwrap().unwrap();
        started.elapsed().as_secs_f64()
    };
    let cold = time(&mut m, &ids, 0);
    m.truncate_cache(0);
    let warm = time(&mut m, &ids, 0);
    let first = time(&mut m, &[PARIS as u32], ids.len());
    let mut steps = Vec::new();
    for _ in 0..6 {
        m.truncate_cache(ids.len());
        steps.push(time(&mut m, &[PARIS as u32], ids.len()));
    }
    steps.sort_by(f64::total_cmp);
    eprintln!(
        "REAL|speed|cpu|prompt of {} tokens cold {cold:.1} s, warm {warm:.2} s ({:.1} tok/s)|first decode step {first:.2} s|warm decode step median {:.3} s best {:.3} s ({:.2} tok/s)",
        ids.len(),
        ids.len() as f64 / warm,
        steps[steps.len() / 2],
        steps[0],
        1.0 / steps[steps.len() / 2]
    );
}
