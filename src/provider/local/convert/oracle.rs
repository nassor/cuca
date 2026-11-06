//! Numerical parity of converted checkpoints with the transformers oracle.
//!
//! Gated on `CUCA_LOCAL_REF_DIR` (the oracle dumps `ref/*.json`, see the
//! repository's `.omp/local-models/ref/README.md`; they are not in the
//! repository) and on the network or a seeded cache: without them the test
//! prints `SKIP:` and passes, unless `CUCA_REQUIRE_LIVE=1` turns that into a
//! failure. For each pinned checkpoint it converts to F16 (the closest to the
//! oracle's f32 weights), loads the result on the CPU, runs the oracle's
//! prompt ids and compares the last-position logits and the greedy
//! continuation with the safetensors oracle of the same revision:
//! top-1 equal, top-k overlap, top-k logits within 2 % relative L2, and the
//! first greedy ids equal.

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

/// `(repo, revision)` of the checkpoints with a safetensors oracle.
const PINNED: &[(&str, &str)] = &[
    (
        "allenai/OLMo-2-0425-1B-Instruct",
        "48d788eca847d4d7548f375ad03d3c9312f6139e",
    ),
    (
        "ibm-granite/granite-3.3-2b-instruct",
        "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
    ),
    (
        "baidu/ERNIE-4.5-0.3B-PT",
        "b565cf6caebdb7a1eadf00100857b1ed5e044f12",
    ),
    (
        "HuggingFaceTB/SmolLM3-3B",
        "a07cc9a04f16550a088caea529712d1d335b0ac1",
    ),
    (
        "unsloth/gemma-3-1b-it",
        "5b11413a10db4e486ef16a20101fd028f8f2499c",
    ),
    (
        "allenai/OLMoE-1B-7B-0125-Instruct",
        "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
    ),
    (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "6a6f4aa4197940add57724a7707d069478df56b1",
    ),
    (
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
    ),
];

fn skip(reason: &str) {
    if std::env::var("CUCA_REQUIRE_LIVE").as_deref() == Ok("1") {
        panic!("{reason}");
    }
    println!("SKIP: {reason}");
}

/// The oracle dump made from `repo@revision`'s own safetensors.
fn dump_for(dir: &std::path::Path, repo: &str, revision: &str) -> Option<Value> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        if entry.path().extension().is_none_or(|e| e != "json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(dump) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let s = &dump["source"];
        if s["type"] == "safetensors" && s["repo"] == repo && s["revision"] == revision {
            return Some(dump);
        }
    }
    None
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect()
        })
        .unwrap_or_default()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |a, (i, &x)| if x > a.1 { (i, x) } else { a },
        )
        .0 as u32
}

#[test]
fn converted_checkpoints_match_the_transformers_oracle() {
    let Some(ref_dir) = std::env::var_os("CUCA_LOCAL_REF_DIR").map(PathBuf::from) else {
        return skip("set CUCA_LOCAL_REF_DIR to the oracle dumps");
    };
    let only = std::env::var("CUCA_LOCAL_CONVERT_ONLY").ok();
    let mut checked = 0;
    for (repo, revision) in PINNED {
        if only.as_ref().is_some_and(|o| !repo.contains(o.as_str())) {
            continue;
        }
        let Some(dump) = dump_for(&ref_dir, repo, revision) else {
            skip(&format!(
                "{repo}: no safetensors oracle dump for {revision}"
            ));
            continue;
        };
        let config = LocalModelConfig {
            source: ModelSource::Checkpoint {
                repo: (*repo).to_string(),
                revision: (*revision).to_string(),
            },
            convert: ConvertConfig {
                quant: WeightQuant::F16,
            },
            ..Default::default()
        };
        let path = match resolve_model_path(&config, &config.source) {
            Ok(p) => p,
            Err(e @ crate::error::CucaError::Transport { .. }) => {
                return skip(&format!("{repo} cannot be downloaded: {e}"));
            }
            Err(e) => panic!("{repo}: conversion failed: {e}"),
        };
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&path).unwrap())).unwrap();
        let prompts: Vec<&Value> = dump["prompts"].as_array().unwrap().iter().collect();
        let ctx = prompts
            .iter()
            .map(|p| ids(&p["ids"]).len() + ids(&p["greedy_ids"]).len() + 8)
            .max()
            .unwrap();
        let mut model = Transformer::load(&content, &path, ctx).unwrap();
        for prompt in prompts {
            let name = prompt["name"].as_str().unwrap();
            let prompt_ids = ids(&prompt["ids"]);
            let greedy = ids(&prompt["greedy_ids"]);
            let top: Vec<(u32, f32)> = prompt
                .get("top10")
                .or_else(|| prompt.get("top5"))
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .map(|e| (e[0].as_u64().unwrap() as u32, e[1].as_f64().unwrap() as f32))
                .collect();
            model.truncate_cache(0);
            let chunk = prefill_chunk(ctx);
            let (mut pos, mut logits) = (0, Vec::new());
            while pos < prompt_ids.len() {
                let end = (pos + chunk).min(prompt_ids.len());
                let want = if end == prompt_ids.len() {
                    Want::Last
                } else {
                    Want::None
                };
                if let Some(l) = model
                    .forward_ext(&prompt_ids[pos..end], pos, want, Hidden::None, true)
                    .unwrap()
                    .logits
                {
                    logits = l;
                }
                pos = end;
            }
            let mut ranked: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mine: Vec<u32> = ranked.iter().take(top.len()).map(|r| r.0).collect();
            let overlap = top.iter().filter(|t| mine.contains(&t.0)).count();
            let (mut num, mut den) = (0f64, 0f64);
            for (id, want) in &top {
                let d = f64::from(logits[*id as usize] - want);
                num += d * d;
                den += f64::from(*want) * f64::from(*want);
            }
            let rel_l2 = (num / den).sqrt();
            let mut cur = logits.clone();
            let mut out = Vec::new();
            for _ in 0..greedy.len() {
                let next = argmax(&cur);
                out.push(next);
                cur = model
                    .forward_ext(
                        &[next],
                        prompt_ids.len() + out.len() - 1,
                        Want::Last,
                        Hidden::None,
                        true,
                    )
                    .unwrap()
                    .logits
                    .unwrap();
            }
            let matched = out.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
            // Margin of the oracle's top-1 over its runner-up: a flip inside
            // a near-tie is not a conversion error.
            let margin = top[0].1 - top[1].1;
            println!(
                "ORACLE {repo}/{name}: top1={} overlap={overlap}/{} rel_l2={rel_l2:.4} greedy={matched}/{} (oracle margin {margin:.3})",
                mine[0] == top[0].0,
                top.len(),
                greedy.len()
            );
            assert!(
                mine[0] == top[0].0 || margin < 0.2,
                "{repo}/{name}: top-1 differs from the oracle"
            );
            assert!(
                overlap * 10 >= top.len() * 9 || overlap + 1 >= top.len(),
                "{repo}/{name}: top-k overlap {overlap}/{}",
                top.len()
            );
            assert!(
                rel_l2 <= 0.02,
                "{repo}/{name}: logits {rel_l2:.4} relative L2 from the oracle"
            );
            let need = greedy.len().min(8);
            assert!(
                matched >= need || margin < 0.2,
                "{repo}/{name}: greedy ids match for {matched} of the first {need}: mine {out:?} oracle {greedy:?}"
            );
        }
        checked += 1;
    }
    if checked == 0 {
        skip("no pinned checkpoint had an oracle dump");
    }
}
