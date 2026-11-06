//! Numerical parity of a locally converted checkpoint with a transformers
//! f32 reference, for checkpoints too large or too new for the pinned
//! `oracle` set (a layer-truncated real checkpoint, a tiny synthetic one).
//!
//! Gated on `CUCA_LOCAL_REF_CKPT` (the checkpoint directory) or
//! `CUCA_LOCAL_REF_GGUF` (a GGUF file, loaded as it is: a layer-truncated
//! published GGUF whose dequantized weights the reference script loaded too)
//! and `CUCA_LOCAL_REF_PREFIX` (`<prefix>.json` and `<prefix>.bin`, written by
//! the reference script: the prompt ids, the compared positions, the greedy
//! ids and the f32 logits of those positions). Without them the test prints
//! `SKIP:` and passes, unless `CUCA_REQUIRE_LIVE=1`. The checkpoint is
//! converted to F16 (its BF16 values are exact in F16) and loaded in the
//! engine's exact dequantized mode, so only float rounding separates the two
//! sides: `CUCA_LOCAL_REF_TOL` (default `5e-4`) bounds `max |dlogit|`.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use candle_core::quantized::gguf_file;
use serde_json::Value;

use crate::provider::local::model::cpu::{Hidden, Transformer};
use crate::provider::local::model::{Want, prefill_chunk};
use crate::provider::local::{
    ConvertConfig, LocalModelConfig, ModelSource, WeightQuant, resolve_model_path,
};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

#[test]
fn local_checkpoint_matches_the_reference() {
    let source = env("CUCA_LOCAL_REF_CKPT")
        .map(|dir| (true, dir))
        .or_else(|| env("CUCA_LOCAL_REF_GGUF").map(|file| (false, file)));
    let (Some((is_dir, source)), Some(prefix)) = (source, env("CUCA_LOCAL_REF_PREFIX")) else {
        let msg =
            "SKIP: set CUCA_LOCAL_REF_CKPT (or CUCA_LOCAL_REF_GGUF) and CUCA_LOCAL_REF_PREFIX";
        assert!(env("CUCA_REQUIRE_LIVE").is_none(), "{msg}");
        println!("{msg}");
        return;
    };
    let tol: f32 = env("CUCA_LOCAL_REF_TOL").map_or(5e-4, |t| t.parse().unwrap());
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{prefix}.json")).unwrap()).unwrap();
    let prompt = ids(&meta["ids"]);
    let positions: Vec<usize> = ids(&meta["positions"])
        .iter()
        .map(|&p| p as usize)
        .collect();
    let greedy = ids(&meta["greedy"]);
    let vocab = meta["vocab"].as_u64().unwrap() as usize;
    let raw = std::fs::read(format!("{prefix}.bin")).unwrap();
    assert_eq!(raw.len(), positions.len() * vocab * 4, "reference size");
    let reference: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let path = if is_dir {
        let config = LocalModelConfig {
            source: ModelSource::Dir(PathBuf::from(&source)),
            convert: ConvertConfig {
                quant: WeightQuant::F16,
            },
            ..Default::default()
        };
        resolve_model_path(&config, &config.source).expect("conversion")
    } else {
        PathBuf::from(&source)
    };
    let content =
        gguf_file::Content::read(&mut BufReader::new(File::open(&path).unwrap())).unwrap();
    let ctx = prompt.len() + greedy.len() + 8;
    let mut model = Transformer::load_dequantized(&content, &path, ctx).unwrap();

    // Prefill in passes that end at every compared position (the LM head runs
    // there only: a long prompt's logits for every row would be terabytes of
    // arithmetic), split further at the engine's chunk size. Any chunking
    // gives the same bits in exact mode.
    let chunk = prefill_chunk(ctx).min(prompt.len());
    let mut per_pos: Vec<Option<Vec<f32>>> = vec![None; prompt.len()];
    let mut cuts: Vec<usize> = positions.iter().map(|p| p + 1).collect();
    cuts.push(prompt.len());
    cuts.sort_unstable();
    cuts.dedup();
    let mut pos = 0;
    for cut in cuts {
        while pos < cut {
            let end = (pos + chunk).min(cut);
            let at_position = end == cut && positions.contains(&(cut - 1));
            let want = if at_position { Want::Last } else { Want::None };
            let out = model
                .forward_ext(&prompt[pos..end], pos, want, Hidden::None, true)
                .unwrap()
                .logits;
            if let Some(out) = out {
                per_pos[end - 1] = Some(out);
            }
            pos = end;
        }
    }
    let mut worst = 0f32;
    for (n, &p) in positions.iter().enumerate() {
        let mine = per_pos[p].as_ref().unwrap();
        let want = &reference[n * vocab..(n + 1) * vocab];
        let d = mine
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("REF position {p}: max|dlogit| = {d:.2e}");
        worst = worst.max(d);
    }
    assert!(worst < tol, "max|dlogit| {worst:.3e} exceeds {tol:.1e}");

    // Greedy continuation (the last prompt position is always compared).
    let mut cur = per_pos[prompt.len() - 1].clone().unwrap();
    let mut out = Vec::new();
    for _ in 0..greedy.len() {
        let next = argmax(&cur);
        out.push(next);
        cur = model
            .forward_ext(
                &[next],
                prompt.len() + out.len() - 1,
                Want::Last,
                Hidden::None,
                true,
            )
            .unwrap()
            .logits
            .unwrap();
    }
    println!("REF greedy mine {out:?}\nREF greedy ref  {greedy:?}");
    assert_eq!(out, greedy, "greedy ids");
}

/// Converts `CUCA_CONVERT_PROBE=<checkpoint dir>|<output file>|<f16|q8_0|q4_0>`
/// to the given path, for diffing against the llama.cpp converter's output
/// (`.omp/local-models/convert/runcmp.py`). Skips without the variable.
#[test]
fn convert_probe() {
    let Some(spec) = env("CUCA_CONVERT_PROBE") else {
        return;
    };
    let parts: Vec<&str> = spec.split('|').collect();
    let [dir, out, quant] = parts.as_slice() else {
        panic!("CUCA_CONVERT_PROBE is `<dir>|<out>|<quant>`");
    };
    let quant = match *quant {
        "f16" => WeightQuant::F16,
        "q8_0" => WeightQuant::Q8_0,
        "q4_0" => WeightQuant::Q4_0,
        other => panic!("unknown quant `{other}`"),
    };
    let checkpoint = super::hf::CheckpointDir::open(std::path::Path::new(dir)).unwrap();
    super::convert(&checkpoint, std::path::Path::new(out), quant, None).unwrap();
    println!("PROBE wrote {out}");
}
