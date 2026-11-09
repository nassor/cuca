//! Converted checkpoints through the public API (`provider-local`).
//!
//! For each pinned Hugging Face checkpoint this loads
//! `ModelSource::Checkpoint { repo, revision }` with `quant = F16`: the files
//! are downloaded (or found in the cache, `CUCA_CACHE_DIR`), converted to a
//! cached GGUF and loaded like any other file. The chat turn of the
//! transformers oracle's `chat_a` prompt must then give the oracle's first
//! greedy tokens, and, where the rendered template holds no date, the
//! oracle's prompt token count.
//!
//! Gated on `CUCA_LOCAL_REF_DIR` (the oracle dumps, not in the repository;
//! `.omp/local-models/ref/`): without it, or without network access for a
//! checkpoint that is not cached, each check prints `SKIP:` and passes, unless
//! `CUCA_REQUIRE_LIVE=1` turns that into a failure. The logits, top-k and
//! relative-L2 comparison with the same oracle runs inside the crate
//! (`convert::oracle`, a unit test with the same gate and the same variable);
//! this file checks what the public API can see.
//!
//! ```sh
//! CUCA_LOCAL_REF_DIR=$PWD/.omp/local-models/ref CUCA_REQUIRE_LIVE=1 \
//!   RUSTFLAGS="-C target-cpu=native" cargo test --release --no-default-features \
//!   --features provider-local --test provider_local_convert -- --test-threads=1 --nocapture
//! ```
//!
//! `CUCA_LOCAL_CONVERT_ONLY=<substring>` restricts the run to the repositories
//! containing it.
//!
//! # Verifying the converter itself
//!
//! The converter is a port of llama.cpp's `convert_hf_to_gguf.py`. Its output
//! was compared, key by key and tensor by tensor, with what that script writes
//! for the same checkpoint (pinned commit
//! `67107ac34f62882d8b53ea41ad68010ea635b9be`, `--outtype f16`): the files were
//! identical for Gemma 3, Gemma 4, the Gemma 4 assistant (but for the
//! tokenizer it does not carry), OLMo 2, Granite 3.3, ERNIE 4.5, SmolLM 3,
//! Phi 4 mini and a `Qwen2ForCausalLM` checkpoint, and for synthetic Qwen 3
//! MoE, OLMoE, Llama (with attention biases), Qwen 3.5 and Qwen 3.5 MoE
//! checkpoints (value heads regrouped). Qwen 3
//! 0.6B at Q8_0 equals the converter's own Q8_0, but not the published
//! `Qwen/Qwen3-0.6B-GGUF` file: that file's layer weights were built from
//! different weights (relative L2 distance 0.9 % from the checkpoint at
//! `c1899de`, against 0.54 % for either converter's Q8_0; `token_embd` is
//! identical to this converter's). The comparison scripts live in
//! `.omp/local-models/convert/` (gitignored).

#![cfg(feature = "provider-local")]

mod local_env;

use cuca::{ConvertConfig, LocalModel, LocalModelConfig, ModelSource, UnifiedRequest, WeightQuant};
use serde_json::Value;

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

/// The oracle dump made from `repo@revision`'s own safetensors.
fn dump_for(dir: &std::path::Path, repo: &str, revision: &str) -> Option<Value> {
    local_env::oracle_dumps(dir).find(|dump| {
        let s = &dump["source"];
        s["type"] == "safetensors" && s["repo"] == repo && s["revision"] == revision
    })
}

/// The length of the common prefix of two strings, in characters.
fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[tokio::test]
async fn converted_checkpoints_answer_like_the_oracle() {
    let Some(ref_dir) = local_env::ref_dir() else {
        return local_env::skip("SKIP: set CUCA_LOCAL_REF_DIR to the oracle dumps");
    };
    let only = local_env::convert_only();
    for (repo, revision) in PINNED {
        if only.as_ref().is_some_and(|o| !repo.contains(o.as_str())) {
            continue;
        }
        let Some(dump) = dump_for(&ref_dir, repo, revision) else {
            local_env::skip(&format!(
                "SKIP: {repo}: no safetensors oracle dump for {revision}"
            ));
            continue;
        };
        let chat = dump["prompts"]
            .as_array()
            .and_then(|p| p.iter().find(|p| p["name"] == "chat_a"))
            .expect("chat_a prompt");
        let model = match LocalModel::load(LocalModelConfig {
            source: ModelSource::Checkpoint {
                repo: (*repo).to_string(),
                revision: (*revision).to_string(),
            },
            convert: ConvertConfig {
                quant: WeightQuant::F16,
                ..ConvertConfig::default()
            },
            ..Default::default()
        })
        .await
        {
            Ok(m) => m,
            Err(cuca::CucaError::Transport { message }) => {
                return local_env::skip(&format!("SKIP: {repo} cannot be downloaded: {message}"));
            }
            Err(e) => panic!("{repo}: the checkpoint must convert and load: {e}"),
        };
        let (blocks, response) = local_env::drain(
            &model,
            UnifiedRequest::new("local")
                .add_user_message(local_env::ORACLE_USER)
                .set_temperature(0.0)
                .set_max_tokens(24),
        )
        .await;
        let text = local_env::written(&blocks);
        let usage = response.usage.expect("usage");

        let want = chat["greedy_text"].as_str().unwrap_or_default().trim();
        let got = text.trim();
        let shared = common_prefix(got, want);
        println!("CONVERT {repo}: reply {got:?} oracle {want:?} (shared prefix {shared} chars)");
        assert!(
            shared >= want.chars().count().min(6),
            "{repo}: reply {got:?} does not start like the oracle's {want:?}"
        );

        // The oracle rendered the GGUF-embedded template; the checkpoint's own
        // template can differ (Granite's adds today's date to its default
        // system prompt), so the prompt token count is reported, and compared
        // only where the oracle's template renders no date at all.
        let has_date = dump["chat_template"]["rendered"]
            .as_str()
            .is_some_and(|r| r.contains("2025"));
        let oracle_tokens = chat["n_prompt_tokens"].as_u64().unwrap();
        println!(
            "CONVERT {repo}: prompt tokens {} (oracle {oracle_tokens}, date in oracle prompt: {has_date})",
            usage.prompt_tokens
        );
        if !has_date {
            assert_eq!(
                u64::from(usage.prompt_tokens),
                oracle_tokens,
                "{repo}: prompt token count differs from the oracle's"
            );
        }
    }
}
