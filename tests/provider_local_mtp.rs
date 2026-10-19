//! Live test of multi-token-prediction speculative decoding in `provider-local`.
//!
//! A target model runs once alone and once with a drafter
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

use std::time::Instant;

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    CucaClient, DrafterSource, LocalDevice, LocalModel, LocalModelConfig, ModelSource,
    UnifiedRequest,
};
use tokio_stream::StreamExt;

const PROMPTS: [&str; 3] = [
    "Write a Python function that returns the n-th Fibonacci number, then a function for the factorial.",
    "List the planets of the solar system in order from the sun, one per line, with one fact each.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
];

fn source(path_var: &str, name_var: &str, download_default: bool) -> Option<ModelSource> {
    if let Some(path) = std::env::var_os(path_var) {
        return Some(ModelSource::File(path.into()));
    }
    match std::env::var(name_var) {
        Ok(name) if !name.is_empty() => Some(
            name.parse()
                .unwrap_or_else(|e| panic!("{name_var}={name:?}: {e}")),
        ),
        _ if download_default && std::env::var("CUCA_LOCAL_DOWNLOAD").as_deref() == Ok("1") => {
            Some(ModelSource::default_model())
        }
        _ => None,
    }
}

/// The drafter the environment selects: a file by path or preset, or the
/// head inside the target's file (`CUCA_LOCAL_DRAFTER=in-trunk`).
fn drafter_source() -> Option<DrafterSource> {
    if let Some(path) = std::env::var_os("CUCA_LOCAL_DRAFTER_PATH") {
        return Some(ModelSource::File(path.into()).into());
    }
    match std::env::var("CUCA_LOCAL_DRAFTER") {
        Ok(name) if name.eq_ignore_ascii_case("in-trunk") => Some(DrafterSource::InTrunk),
        Ok(name) if !name.is_empty() => Some(
            name.parse::<ModelSource>()
                .unwrap_or_else(|e| panic!("CUCA_LOCAL_DRAFTER={name:?}: {e}"))
                .into(),
        ),
        _ => None,
    }
}

/// The `draft_tokens` values to compare.
fn draft_token_counts() -> Vec<usize> {
    let Ok(value) = std::env::var("CUCA_LOCAL_DRAFT_TOKENS") else {
        return vec![3];
    };
    value
        .split(',')
        .map(|v| {
            v.trim()
                .parse()
                .unwrap_or_else(|e| panic!("CUCA_LOCAL_DRAFT_TOKENS={value:?}: {e}"))
        })
        .collect()
}

async fn run(model: &LocalModel, prompt: &str, max_tokens: u32) -> (String, u64, f64) {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model.clone())
        .build()
        .expect("client builds");
    let request = UnifiedRequest::new("local")
        .add_user_message(prompt)
        .set_temperature(0.0)
        .set_max_tokens(max_tokens);
    let started = Instant::now();
    let (mut stream, handle) = client
        .generate_stream_with_response(request)
        .await
        .expect("request dispatches");
    // Reasoning and answer both count: the comparison covers every token.
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            MessageContentBlock::Text(t) => text.push_str(&t),
            MessageContentBlock::Thinking { reasoning, .. } => text.push_str(&reasoning),
            _ => {}
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let response = handle.take().expect("terminal response");
    (text, response.completion_tokens.into(), elapsed)
}

#[tokio::test]
async fn drafter_output_equals_plain_greedy_decoding() {
    let (Some(target), Some(drafter)) = (
        source("CUCA_LOCAL_MODEL_PATH", "CUCA_LOCAL_MODEL", true),
        drafter_source(),
    ) else {
        let reason = "SKIP: set a target (CUCA_LOCAL_MODEL_PATH, CUCA_LOCAL_MODEL or \
                      CUCA_LOCAL_DOWNLOAD=1) and a drafter (CUCA_LOCAL_DRAFTER_PATH, \
                      CUCA_LOCAL_DRAFTER, or CUCA_LOCAL_DRAFTER=in-trunk)";
        if std::env::var("CUCA_REQUIRE_LIVE").as_deref() == Ok("1") {
            panic!("{reason}");
        }
        println!("{reason}");
        return;
    };
    let device: LocalDevice = std::env::var("CUCA_LOCAL_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|e| panic!("CUCA_LOCAL_DEVICE={s:?}: {e}"))
        })
        .unwrap_or_default();
    let exact = device == LocalDevice::Cpu;
    let plain = LocalModel::load(LocalModelConfig {
        source: target.clone(),
        context_length: Some(2048),
        device,
        ..Default::default()
    })
    .await
    .expect("target loads");

    for draft_tokens in draft_token_counts() {
        let speculative = LocalModel::load(LocalModelConfig {
            source: target.clone(),
            drafter: Some(drafter.clone()),
            draft_tokens,
            context_length: Some(2048),
            device,
            ..Default::default()
        })
        .await
        .expect("target and drafter load");

        let (mut plain_secs, mut spec_secs, mut tokens) = (0.0, 0.0, 0u64);
        for prompt in PROMPTS {
            // A first pass warms the kernels (GPU autotune) and the governor.
            run(&plain, prompt, 128).await;
            run(&speculative, prompt, 128).await;
            let before = speculative.draft_stats();
            let (expected, n_plain, t_plain) = run(&plain, prompt, 128).await;
            let (got, n_spec, t_spec) = run(&speculative, prompt, 128).await;
            let after = speculative.draft_stats();
            // On the CPU the engine is row-exact, so output must be identical.
            // A GPU's batched rows differ from single steps by float rounding,
            // which can flip a near-tie: report identity there instead of
            // asserting it.
            if exact {
                assert_eq!(
                    got, expected,
                    "speculative output must equal plain greedy output for {prompt:?}"
                );
                assert_eq!(n_spec, n_plain);
            }
            println!(
                "k {draft_tokens} {:?}: plain {:.1} tok/s, speculative {:.1} tok/s ({:.2}x), identical {}, drafted {} accepted {}",
                &prompt[..30],
                n_plain as f64 / t_plain,
                n_spec as f64 / t_spec,
                (n_plain as f64 / t_plain).recip() * (n_spec as f64 / t_spec),
                got == expected,
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
    }
}
