//! Live test of multi-token-prediction speculative decoding in `provider-local`.
//!
//! A target model runs once alone (its outputs are recorded and the model is
//! dropped, so a target that fills the machine's memory never has two copies
//! resident) and then once per draft length with a drafter
//! (`LocalModelConfig::drafter`); greedy output must be identical on the CPU,
//! and the run reports acceptance and decode speed per prompt (code, a list,
//! prose; 128 new tokens each). Two kinds of drafter:
//!
//! - a separate file: a Gemma 4 target with its Gemma 4 assistant,
//!   `CUCA_LOCAL_DRAFTER_PATH` / `CUCA_LOCAL_DRAFTER` (the `gemma4-e2b-mtp-f16`
//!   preset); or
//! - the prediction head inside the target's own file,
//!   `CUCA_LOCAL_DRAFTER=in-trunk`, with a Qwen 3.5 `*-MTP` file as the target
//!   (`CUCA_LOCAL_MODEL_PATH`, for example `Qwen3.5-0.8B-Q8_0.gguf` of
//!   `unsloth/Qwen3.5-0.8B-MTP-GGUF`; the CPU only).
//!
//! The target comes from `CUCA_LOCAL_MODEL_PATH` / `CUCA_LOCAL_MODEL` (default
//! preset `gemma4-e2b-q4_0` when `CUCA_LOCAL_DOWNLOAD=1`).
//! `CUCA_LOCAL_DRAFT_TOKENS` is the `draft_tokens` of the drafter run, or a
//! comma-separated list (`1,2,3,4`) to run one comparison per value (default
//! 3).
//!
//! `CUCA_LOCAL_ALLOW_MISMATCH=1` reports a CPU output that differs from plain
//! decoding (`identical false`) instead of failing the test. It is for timing
//! a target so large (67 GB) that two loads of it on one machine can differ
//! by a flipped bit, which breaks the exactness check without a speculation
//! fault.
//!
//! With the target or the drafter missing the test prints `SKIP` and passes,
//! unless `CUCA_REQUIRE_LIVE=1` turns the skip into a panic. Build for the host
//! CPU:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" CUCA_LOCAL_DOWNLOAD=1 CUCA_LOCAL_DRAFTER=gemma4-e2b-mtp-f16 \
//!   cargo test --release --features provider-local --test provider_local_mtp -- --nocapture
//! RUSTFLAGS="-C target-cpu=native" CUCA_LOCAL_MODEL_PATH=Qwen3.5-0.8B-Q8_0.gguf CUCA_LOCAL_DRAFTER=in-trunk \
//!   CUCA_LOCAL_DRAFT_TOKENS=1,2,3,4 cargo test --release --features provider-local --test provider_local_mtp -- --nocapture
//! ```

#![cfg(feature = "provider-local")]

mod local_env;

use std::time::Instant;

use cuca::{LocalDevice, LocalModel, LocalModelConfig, UnifiedRequest};

const PROMPTS: [&str; 3] = [
    "Write a Python function that returns the n-th Fibonacci number, then a function for the factorial.",
    "List the planets of the solar system in order from the sun, one per line, with one fact each.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
];

async fn run(model: &LocalModel, prompt: &str, max_tokens: u32) -> (String, u64, f64) {
    let request = UnifiedRequest::new("local")
        .add_user_message(prompt)
        .set_temperature(0.0)
        .set_max_tokens(max_tokens);
    let started = Instant::now();
    let (blocks, response) = local_env::drain(model, request).await;
    let elapsed = started.elapsed().as_secs_f64();
    // Reasoning and answer both count: the comparison covers every token.
    let text = local_env::written(&blocks);
    local_env::print_reply(&text);
    (text, response.completion_tokens.into(), elapsed)
}

#[tokio::test]
async fn drafter_output_equals_plain_greedy_decoding() {
    let (Some(target), Some(drafter)) =
        (local_env::model_source(true), local_env::drafter_source())
    else {
        local_env::skip(
            "SKIP: set a target (CUCA_LOCAL_MODEL_PATH, CUCA_LOCAL_MODEL or \
             CUCA_LOCAL_DOWNLOAD=1) and a drafter (CUCA_LOCAL_DRAFTER_PATH, \
             CUCA_LOCAL_DRAFTER, or CUCA_LOCAL_DRAFTER=in-trunk)",
        );
        return;
    };
    let device = local_env::device();
    let exact = device == LocalDevice::Cpu;
    // The plain run goes first and the plain model is dropped before any
    // speculative model loads: two resident copies of a large target do not
    // fit in memory together, and a machine that pages would time both runs
    // by the paging, not by the engines.
    let plain = LocalModel::load(LocalModelConfig {
        source: target.clone(),
        context_length: Some(2048),
        device,
        convert: local_env::convert(),
        ..Default::default()
    })
    .await
    .expect("target loads");
    let mut plain_runs = Vec::new();
    for prompt in PROMPTS {
        // A first pass warms the kernels (GPU autotune) and fills the prefix
        // store; the second pass, which reuses the prefix, is what is
        // compared and timed.
        let (cold, ..) = run(&plain, prompt, 128).await;
        let (text, tokens, secs) = run(&plain, prompt, 128).await;
        if exact {
            assert_eq!(
                text, cold,
                "a warm plain run must equal the cold one for {prompt:?}"
            );
        }
        plain_runs.push((text, tokens, secs));
    }
    drop(plain);
    // The engine thread frees the model once its last handle is gone.
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let allow_mismatch = local_env::allow_mismatch();
    for draft_tokens in local_env::draft_token_counts() {
        let speculative = LocalModel::load(LocalModelConfig {
            source: target.clone(),
            drafter: Some(drafter.clone()),
            draft_tokens,
            context_length: Some(2048),
            device,
            convert: local_env::convert(),
            ..Default::default()
        })
        .await
        .expect("target and drafter load");

        let (mut plain_secs, mut spec_secs, mut tokens) = (0.0, 0.0, 0u64);
        for (prompt, (expected, n_plain, t_plain)) in PROMPTS.into_iter().zip(&plain_runs) {
            // A first pass warms the kernels and the governor.
            run(&speculative, prompt, 128).await;
            let before = speculative.draft_stats();
            let (got, n_spec, t_spec) = run(&speculative, prompt, 128).await;
            let after = speculative.draft_stats();
            // On the CPU the engine is row-exact, so output must be identical.
            // A GPU's batched rows differ from single steps by float rounding,
            // which can flip a near-tie: report identity there instead of
            // asserting it. `CUCA_LOCAL_ALLOW_MISMATCH=1` reports it on the
            // CPU too, for timing a target so large that separate loads of it
            // can differ by a flipped bit.
            if exact && !allow_mismatch {
                assert_eq!(
                    &got, expected,
                    "speculative output must equal plain greedy output for {prompt:?}"
                );
                assert_eq!(n_spec, *n_plain);
            }
            println!(
                "k {draft_tokens} {:?}: plain {:.1} tok/s, speculative {:.1} tok/s ({:.2}x), identical {}, drafted {} accepted {}",
                &prompt[..30],
                *n_plain as f64 / t_plain,
                n_spec as f64 / t_spec,
                (*n_plain as f64 / t_plain).recip() * (n_spec as f64 / t_spec),
                &got == expected,
                after.drafted - before.drafted,
                after.accepted - before.accepted,
            );
            plain_secs += t_plain;
            spec_secs += t_spec;
            tokens += n_plain;
        }
        let stats = speculative.draft_stats();
        assert!(stats.rounds > 0, "the drafter ran");
        println!(
            "{tokens} tokens: plain {:.1} tok/s, speculative {:.1} tok/s ({:.2}x); \
             drafted {} accepted {} (acceptance {:.1}%) over {} rounds, draft_tokens {draft_tokens}",
            tokens as f64 / plain_secs,
            tokens as f64 / spec_secs,
            plain_secs / spec_secs,
            stats.drafted,
            stats.accepted,
            100.0 * stats.acceptance_rate().unwrap_or(0.0),
            stats.rounds,
        );
        // Free this speculative model before the next one loads.
        drop(speculative);
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}
