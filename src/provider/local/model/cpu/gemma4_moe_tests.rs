//! The Gemma 4 mixture of experts (26B-A4B dialect) on the CPU against
//! engines this crate does not own: transformers on a tiny release-format
//! file, and llama.cpp on the real 26B-A4B Q4_0.
//!
//! Both tests are gated and print `SKIP:` without their cuca-test.kdl keys:
//!
//! - `oracle { g4moe { oracle_dir "<dir>" } }`: the output of
//!   `.omp/local-models/ref/gemma4moe/make_ref.py --out <dir>`: `tiny-q4.gguf`
//!   (Q4_0 matrices, Q8_0 embedding, F32 norms, router and scales, the dtypes
//!   of the real file), `tiny-f32.gguf` (the same weights dequantized) and
//!   `expected.json` (the logits of every position of three prompts, computed
//!   by transformers' `Gemma4ForCausalLM` in f32 over the dequantized weights).
//! - `oracle { g4moe { real "<GGUF>"; real_ref "<json>" } }`: the
//!   real `gemma-4-26B-A4B-it-Q4_0.gguf` and the output of
//!   `.omp/local-models/ref/llamacpp/ref_llama.py all` on it (llama.cpp's
//!   top-20 log-probabilities after each prompt and its greedy continuations).
//!
//! ```kdl
//! // cuca-test.kdl
//! oracle { g4moe { oracle_dir "<dir>" } }
//! ```
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" \
//!   cargo test --release --lib --features provider-local gemma4_moe_matches -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use crate::provider::local::gguf::Content;
use crate::provider::local::tokenizer::golden::skip;
use crate::provider::local::{EngramCache, ExpertPlacement, ExpertTiers};

use super::*;

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

/// `log softmax`.
fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = x.iter().map(|&v| (v as f64 - m).exp()).sum();
    x.iter().map(|&v| v as f64 - m - sum.ln()).collect()
}

fn argmax(x: &[f32]) -> u32 {
    (0..x.len())
        .max_by(|&a, &b| x[a].total_cmp(&x[b]).then(b.cmp(&a)))
        .unwrap() as u32
}

fn oracle_dir() -> Option<PathBuf> {
    let dir = crate::test_config::get().oracle.g4moe.oracle_dir.clone();
    match dir {
        Some(d) if d.join("expected.json").is_file() && d.join("tiny-q4.gguf").is_file() => Some(d),
        _ => {
            skip(
                "set `oracle { g4moe { oracle_dir \"...\" } }` in cuca-test.kdl to the output of make_ref.py (gemma4moe)",
            );
            None
        }
    }
}

/// How a tiny file is loaded.
#[derive(Clone, Copy, Debug)]
enum Load {
    /// The production path: quantized matmuls round their activations to Q8_0.
    Quantized,
    /// Matmul weights held as dequantized f32 (no activation rounding).
    Dequantized,
    /// The production path with the experts tiered at the smallest budget.
    Tiered,
}

fn load_tiny(path: &Path, how: Load) -> Transformer {
    let content = Content::open(path).unwrap();
    match how {
        Load::Quantized => Transformer::load(&content, 64).unwrap(),
        Load::Dequantized => Transformer::load_dequantized(&content, 64).unwrap(),
        Load::Tiered => {
            let hp = read_hparams(&content).unwrap();
            let per_expert: u64 = (0..hp.n_layer)
                .map(|il| {
                    let n_expert = hp.moe.unwrap().n_expert;
                    let part = |part: &str| tensor_info(&content, &name::blk(il, part)).unwrap();
                    let bytes = |info: &gguf::TensorInfo| {
                        (info.dtype.byte_len(info.shape.elem_count()).unwrap() / n_expert) as u64
                    };
                    bytes(part(name::FFN_GATE_UP_EXPS)) + bytes(part(name::FFN_DOWN_EXPS))
                })
                .sum();
            let tiers = ExpertTiers {
                ram_bytes: hp.moe.unwrap().n_used as u64 * per_expert,
                loader_threads: 2,
                ..ExpertTiers::default()
            };
            Transformer::load_placed(
                &content,
                64,
                &ExpertPlacement::Tiered(tiers),
                &Sizing::single(64),
                &EngramCache::default(),
            )
            .unwrap()
        }
    }
}

/// Largest `|ours - expected|` over every position of every prompt, and the
/// number of positions whose top-1 differs.
fn compare(m: &mut Transformer, expected: &serde_json::Value, what: &str) -> (f64, usize) {
    let vocab = expected["vocab"].as_u64().unwrap() as usize;
    let (mut worst, mut flipped) = (0f64, 0usize);
    for (p, (prompt, rows)) in expected["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["logits"].as_array().unwrap())
        .enumerate()
    {
        let prompt = ids(prompt);
        m.truncate_cache(0);
        let ours = m
            .forward_ext(&prompt, 0, Want::All, Hidden::None, true)
            .unwrap()
            .logits
            .unwrap();
        assert_eq!(ours.len(), prompt.len() * vocab, "{what}: prompt {p}");
        for (pos, row) in rows.as_array().unwrap().iter().enumerate() {
            let want: Vec<f32> = row
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let got = &ours[pos * vocab..(pos + 1) * vocab];
            flipped += usize::from(argmax(got) != argmax(&want));
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max(f64::from((g - w).abs()));
            }
        }
    }
    eprintln!("G4MOE|{what}|max |logit diff| {worst:.3e}, {flipped} top-1 flips");
    (worst, flipped)
}

/// The tiny release-format model equals transformers' Gemma 4 mixture of
/// experts: every position of three prompts (one longer than the sliding
/// window), with the weights exactly dequantized (tight), with Q8_0
/// activation rounding (production, loose) and with tiered experts (equal to
/// production bit for bit). The oracle's perturbed references
/// (`expected.<mistake>.json`: no per-expert scale, K as V off, ...) must all
/// differ from the engine by far more than the agreement, so a wrong layer
/// formula could not hide in the tolerance.
#[test]
fn gemma4_moe_matches_the_transformers_oracle() {
    let Some(dir) = oracle_dir() else {
        return;
    };
    let expected = json(&dir.join("expected.json"));
    let scale = expected["meta"]["max_abs_logit"].as_f64().unwrap();
    let q4 = dir.join("tiny-q4.gguf");
    let f32_file = dir.join("tiny-f32.gguf");

    let (exact, flipped) = compare(
        &mut load_tiny(&f32_file, Load::Quantized),
        &expected,
        "f32 file",
    );
    assert!(exact <= 1e-4 * scale && flipped == 0, "f32 file: {exact}");
    let (dequantized, flipped) = compare(
        &mut load_tiny(&q4, Load::Dequantized),
        &expected,
        "q4 dequantized",
    );
    assert!(
        dequantized <= 1e-4 * scale && flipped == 0,
        "q4 dequantized: {dequantized}"
    );
    // The production path rounds every activation to Q8_0 before a Q4_0
    // matmul: a few percent of the logit scale at most, and no position's
    // top-1 moves.
    let (production, flipped) = compare(
        &mut load_tiny(&q4, Load::Quantized),
        &expected,
        "q4 production",
    );
    assert!(production <= 0.05 * scale, "q4 production: {production}");
    assert_eq!(flipped, 0, "q4 production: top-1 flips");
    let (tiered, _) = compare(&mut load_tiny(&q4, Load::Tiered), &expected, "q4 tiered");
    assert_eq!(tiered, production, "tiered experts change no bit");

    let mut perturbed = 0;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    for path in files {
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(mistake) = file
            .strip_prefix("expected.")
            .and_then(|f| f.strip_suffix(".json"))
        else {
            continue;
        };
        let (diff, _) = compare(
            &mut load_tiny(&f32_file, Load::Quantized),
            &json(&path),
            mistake,
        );
        // Renormalising the top-k weights is invisible here by construction:
        // `post_ffw_norm_2` normalises the routed sum, so a common factor
        // of the weights cancels.
        if mistake != "no_topk_renorm" {
            assert!(
                diff > 0.01 * scale,
                "{mistake}: only {diff} from the engine"
            );
            perturbed += 1;
        }
    }
    assert!(
        perturbed >= 5,
        "the oracle's perturbed references are missing"
    );
}

fn real_files() -> Option<(PathBuf, PathBuf)> {
    match (
        crate::test_config::get().oracle.g4moe.real.clone(),
        crate::test_config::get().oracle.g4moe.real_ref.clone(),
    ) {
        (Some(m), Some(r)) if m.is_file() && r.is_file() => Some((m, r)),
        _ => {
            skip(
                "set `oracle { g4moe { real \"...\"; real_ref \"...\" } }` in cuca-test.kdl (the 26B-A4B Q4_0 GGUF and its llama.cpp reference)",
            );
            None
        }
    }
}

/// The real Gemma 4 26B-A4B Q4_0 against llama.cpp on the same file: the
/// top-20 log-probabilities after each prompt (token ids in, so no
/// tokenizer is involved) and the greedy continuations.
#[test]
fn gemma4_moe_real_model_matches_llama_cpp() {
    let Some((model, reference)) = real_files() else {
        return;
    };
    let reference = json(&reference);
    let content = Content::open(&model).unwrap();
    let started = std::time::Instant::now();
    let mut m = Transformer::load(&content, 512).unwrap();
    eprintln!("G4MOE|real|load {:.1} s", started.elapsed().as_secs_f64());

    for (i, entry) in reference["logits"].as_array().unwrap().iter().enumerate() {
        let prompt = ids(&entry["prompt_ids"]);
        m.truncate_cache(0);
        let started = std::time::Instant::now();
        let logits = m.forward(&prompt, 0, Want::Last).unwrap().unwrap();
        let secs = started.elapsed().as_secs_f64();
        let lp = log_softmax(&logits);
        let top = entry["top"].as_array().unwrap();
        let want_top1 = top[0]["id"].as_u64().unwrap() as u32;
        let mut worst = 0f64;
        for t in top.iter().take(5) {
            let id = t["id"].as_u64().unwrap() as usize;
            worst = worst.max((lp[id] - t["logprob"].as_f64().unwrap()).abs());
        }
        eprintln!(
            "G4MOE|real|prompt {i}: {} tokens in {secs:.2} s, top-1 {} (llama.cpp {want_top1}), max |dlogprob| over its top 5: {worst:.4}",
            prompt.len(),
            argmax(&logits),
        );
        // Two engines that round activations alike still differ in the order
        // of their sums (and llama.cpp keeps its K/V cache in f16 by
        // default), and 30 layers amplify it: a top-1 may differ only where
        // llama.cpp itself ranks the engine's pick within 0.5 of its own
        // top-1. The pick must be among llama.cpp's top 20.
        let ours = argmax(&logits);
        if ours != want_top1 {
            let pick = top
                .iter()
                .find(|t| t["id"].as_u64() == Some(u64::from(ours)))
                .unwrap_or_else(|| {
                    panic!("prompt {i}: the engine's pick {ours} is not in llama.cpp's top 20")
                });
            let gap = top[0]["logprob"].as_f64().unwrap() - pick["logprob"].as_f64().unwrap();
            eprintln!(
                "G4MOE|real|prompt {i}: llama.cpp ranks the engine's pick {gap:.4} below its top-1"
            );
            assert!(
                gap < 0.5,
                "prompt {i}: the pick is {gap} below llama.cpp's top-1"
            );
        }
        assert!(worst < 0.5, "prompt {i}: {worst}");
    }

    for (i, entry) in reference["greedy"].as_array().unwrap().iter().enumerate() {
        let prompt = ids(&entry["prompt_ids"]);
        let want = ids(&entry["tokens"]);
        m.truncate_cache(0);
        let mut logits = m.forward(&prompt, 0, Want::Last).unwrap().unwrap();
        let mut at = prompt.len();
        let mut got: Vec<u32> = Vec::new();
        // The first token where the engine picks another than llama.cpp: its
        // index, both tokens and the engine's own log-probability lead of
        // its pick over llama.cpp's.
        let mut split: Option<(usize, u32, u32, f64)> = None;
        let started = std::time::Instant::now();
        while got.len() < want.len() {
            let next = argmax(&logits);
            let theirs = want[got.len()];
            if split.is_none() && next != theirs {
                let lp = log_softmax(&logits);
                split = Some((
                    got.len(),
                    next,
                    theirs,
                    lp[next as usize] - lp[theirs as usize],
                ));
            }
            got.push(next);
            if got.len() == want.len() {
                break;
            }
            logits = m.forward(&[next], at, Want::Last).unwrap().unwrap();
            at += 1;
        }
        let equal = split.map_or(want.len(), |s| s.0);
        eprintln!(
            "G4MOE|real|greedy {i}: {equal} of {} tokens equal llama.cpp's, {:.1} tokens/s decode",
            want.len(),
            (got.len() - 1) as f64 / started.elapsed().as_secs_f64()
        );
        if let Some((index, ours, theirs, lead)) = split {
            let mut prefix = prompt.clone();
            prefix.extend_from_slice(&got[..index]);
            eprintln!(
                "G4MOE|real|greedy {i}: first difference at token {index}: engine {ours}, llama.cpp {theirs}, the engine's lead {lead:.4}; prefix ids {}",
                serde_json::to_string(&prefix).unwrap()
            );
            // A divergence must be a near-tie in the engine's own
            // distribution too.
            assert!(lead < 0.5, "greedy {i}: diverged over a lead of {lead}");
        }
        // The chat answer is short and must be identical.
        if want.len() < 20 {
            assert!(split.is_none(), "greedy {i}");
        }
    }
}

/// The chat template the 26B GGUF carries (Google's 2026-07-09 canonical
/// one: `<|think|>` in a system turn, an empty closed thought channel when
/// thinking is off) renders the prompts llama.cpp renders, and the tokenizer
/// turns them into the same ids.
#[test]
fn gemma4_moe_prompt_ids_match_llama_cpp() {
    use crate::provider::local::chat::{ChatPrompt, ChatTemplate, ThinkingRequest};
    use crate::provider::local::tokenizer::{Tokenizer, golden::read_metadata};

    let Some((model, reference)) = real_files() else {
        return;
    };
    let reference = json(&reference);
    let md = read_metadata(&model);
    let tokenizer = Tokenizer::from_gguf(&md).unwrap();
    let Some(gguf::Value::String(source)) = md.get("tokenizer.chat_template") else {
        panic!("the GGUF carries no chat template");
    };
    let template = ChatTemplate::new(
        source.clone(),
        tokenizer.bos_token().to_string(),
        tokenizer.eos_token().to_string(),
    )
    .unwrap();
    for (key, thinking) in [("thinking_off", false), ("default", true)] {
        let prompt = ChatPrompt {
            messages: serde_json::json!([{
                "role": "user",
                "content": "What is the capital of France? Answer in one word.",
            }]),
            tools: serde_json::Value::Array(Vec::new()),
            enable_thinking: thinking,
            thinking: ThinkingRequest::from_flag(thinking),
        };
        let rendered = template.render(&prompt).unwrap();
        let want = &reference["template"][key];
        // llama.cpp reports the text without the BOS the template starts with.
        assert_eq!(
            rendered.strip_prefix("<bos>").unwrap_or(&rendered),
            want["prompt"].as_str().unwrap(),
            "{key}: text"
        );
        assert_eq!(
            tokenizer.encode_prompt(&rendered),
            ids(&want["ids_add_special"]),
            "{key}: ids"
        );
    }
    eprintln!("G4MOE|real|chat template renders and tokenizes like llama.cpp");
}

/// The published 26B-A4B header resolves to the geometry the engine computes:
/// 30 layers in the 25 sliding / 5 full pattern with 8 and 2 K/V heads, one
/// dual-branch mixture of experts per layer (128 experts of width 704, 8
/// used, a dense FFN of width 2112 beside them), and no `attn_v` on the full
/// layers.
#[test]
fn gemma4_moe_published_header_resolves_to_the_dual_branch_geometry() {
    use super::super::layout::{Ffn, LayerLayout};
    use super::super::paged::KvLayout;

    let Some(model) = crate::test_config::get()
        .oracle
        .g4moe
        .real
        .clone()
        .filter(|p| p.is_file())
    else {
        skip("set `oracle { g4moe { real \"...\" } }` in cuca-test.kdl to the 26B-A4B Q4_0 GGUF");
        return;
    };
    let content = Content::open(&model).unwrap();
    let hp = read_hparams(&content).unwrap();
    assert_eq!(hp.arch.name(), "gemma4");
    assert_eq!((hp.n_layer, hp.n_embd), (30, 2816));
    let moe = hp.moe.unwrap();
    assert_eq!((moe.n_expert, moe.n_used, moe.norm_weights), (128, 8, true));
    for (il, spec) in hp.layers.iter().enumerate() {
        let sliding = il % 6 != 5;
        let (n_kv, head_dim, window) = if sliding {
            (8, 256, Some(1024))
        } else {
            (2, 512, None)
        };
        assert_eq!(
            (spec.n_head, spec.n_kv_head, spec.head_dim, spec.window),
            (16, n_kv, head_dim, window),
            "layer {il}"
        );
        let layout = LayerLayout::read(&hp, il, spec, &content).unwrap();
        assert_eq!(layout.v_from_k, !sliding, "layer {il}");
        assert_eq!(
            layout.ffn,
            Ffn::Moe {
                ff: 704,
                shared_ff: Some(2112),
                fused: true
            },
            "layer {il}"
        );
    }
    for (what, context, sizing) in [
        (
            "preset table: 8192 positions, 4 sequences, 256-row steps",
            8192,
            Sizing {
                max_sequences: 4,
                kv_tokens: 2 * 8192,
                max_rows: 256,
            },
        ),
        (
            "16384 positions, 2 sequences, 256-row steps",
            16384,
            Sizing {
                max_sequences: 2,
                kv_tokens: 16384 + 16,
                max_rows: 256,
            },
        ),
    ] {
        let bytes = KvLayout::sized(&hp, context, &sizing).pool_bytes(&hp);
        eprintln!("G4MOE|real|kv_pool_bytes|{what}|{bytes}");
    }
}

/// A prefix of the real file (its first N layers, tensors copied unchanged by
/// `make_trunc.py`) on the CPU, three ways, against llama.cpp's top-20
/// log-probabilities after each prompt: the production path, the weights
/// held as exact f32 (no activation rounding, the closest this engine gets
/// to the Q4_0 weights' own function), and each against the other. Both
/// engines round activations to Q8_0, so the production path and llama.cpp
/// should sit about as far from the exact function as from each other;
/// llama.cpp ran with its defaults (an f16 K/V cache), the engine keeps f32.
///
/// Gated: `oracle { g4moe { trunc "..." } }` (the truncated GGUF) and
/// `oracle { g4moe { trunc_ref "..." } }` (`ref_llama.py logits` output on it), both in cuca-test.kdl.
#[test]
fn gemma4_moe_truncated_real_file_matches_llama_cpp() {
    let (Some(model), Some(reference)) = (
        crate::test_config::get().oracle.g4moe.trunc.clone(),
        crate::test_config::get().oracle.g4moe.trunc_ref.clone(),
    ) else {
        skip("set `oracle { g4moe { trunc \"...\"; trunc_ref \"...\" } }` in cuca-test.kdl");
        return;
    };
    let reference = json(&reference);
    let content = Content::open(&model).unwrap();
    let mut production = Transformer::load(&content, 128).unwrap();
    let mut exact = Transformer::load_dequantized(&content, 128).unwrap();
    let layers = production.hp.n_layer;
    let entries = reference["logits"].as_array().unwrap();
    let (mut worst_prod, mut worst_exact, mut worst_between) = (0f64, 0f64, 0f64);
    for (i, entry) in entries.iter().enumerate() {
        let prompt = ids(&entry["prompt_ids"]);
        production.truncate_cache(0);
        exact.truncate_cache(0);
        let p = log_softmax(&production.forward(&prompt, 0, Want::Last).unwrap().unwrap());
        let e = log_softmax(&exact.forward(&prompt, 0, Want::Last).unwrap().unwrap());
        let (mut dp, mut de, mut between) = (0f64, 0f64, 0f64);
        for t in entry["top"].as_array().unwrap() {
            let id = t["id"].as_u64().unwrap() as usize;
            let want = t["logprob"].as_f64().unwrap();
            dp = dp.max((p[id] - want).abs());
            de = de.max((e[id] - want).abs());
            between = between.max((p[id] - e[id]).abs());
        }
        eprintln!(
            "G4MOE|trunc L{layers}|prompt {i}: llama.cpp vs production {dp:.4}, llama.cpp vs exact {de:.4}, production vs exact {between:.4} (max |dlogprob|, llama.cpp's top 20)"
        );
        worst_prod = worst_prod.max(dp);
        worst_exact = worst_exact.max(de);
        worst_between = worst_between.max(between);
    }
    eprintln!(
        "G4MOE|trunc L{layers}|worst: production {worst_prod:.4}, exact {worst_exact:.4}, production vs exact {worst_between:.4}"
    );
}
