//! Live test of tiered expert placement (`ExpertPlacement::Tiered`) in
//! `provider-local`.
//!
//! A mixture-of-experts model answers three fixed prompts greedily (48 new
//! tokens each) three times: (a) with the reference placement, (b) cold with
//! the tiered placement under test, (c) tiered again, warm-started from the
//! heat (b) exported. Placement only decides speed, so (b) and (c) must
//! reproduce (a)'s replies exactly: the reference runs on the same device,
//! with the same kernels, on the same weight bytes. Each run prints
//! `TIERS|<phase>|<tok/s>|<stats>`.
//!
//! - `CUCA_LOCAL_MODEL_PATH` / `CUCA_LOCAL_MODEL`: the model (a preset such as
//!   `olmoe-1b-7b-q4_0`, `qwen3-30b-a3b-q4_0`, `qwen3.6-35b-a3b-q4_0`).
//! - `CUCA_LOCAL_EXPERTS`: the tiered placement under test (`ExpertPlacement`
//!   syntax, e.g. `tiered:ram=4g`, or `tiered:device=6g,ram=12g` on a GPU).
//! - `CUCA_LOCAL_EXPERTS_REF`: the reference placement (default
//!   `resident`); set it to a small tiered placement when the resident model
//!   does not fit the device.
//! - `CUCA_LOCAL_ENGRAM`: `LocalModelConfig::engram` (`EngramCache` syntax,
//!   e.g. `ram=2g,threads=8`) for a model with Engram tables; its counters
//!   print with the tier counters.
//! - `CUCA_LOCAL_DEVICE`: `LocalDevice` syntax; a GPU needs its feature.
//! - `CUCA_LOCAL_GPU_LAYERS`: `LocalModelConfig::gpu_layers` (layer offload:
//!   the last `n` layers on the GPU, the rest on the CPU with the RAM tier).
//!
//! Without a model or `CUCA_LOCAL_EXPERTS` the test prints `SKIP` and passes,
//! unless `CUCA_REQUIRE_LIVE=1` turns the skip into a panic.
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" CUCA_LOCAL_MODEL=olmoe-1b-7b-q4_0 CUCA_LOCAL_EXPERTS=tiered:ram=500m \
//!   cargo test --release --features provider-local --test provider_local_tiers -- --nocapture
//! ```

#![cfg(feature = "provider-local")]

mod local_env;

use std::time::Instant;

use cuca::{
    ExpertPlacement, ExpertTierStats, LocalDevice, LocalModel, LocalModelConfig, ModelSource,
    UnifiedRequest,
};

const PROMPTS: [&str; 3] = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "List the planets of the solar system in order from the sun.",
    "Write the opening of a short story about a lighthouse keeper.",
];

const MAX_TOKENS: u32 = 48;

/// One greedy reply: its text (reasoning included) and completion tokens.
async fn reply(model: &LocalModel, prompt: &str) -> (String, u64) {
    let request = UnifiedRequest::new("local")
        .add_user_message(prompt)
        .set_temperature(0.0)
        .set_max_tokens(MAX_TOKENS);
    let (blocks, response) = local_env::drain(model, request).await;
    let text = local_env::written(&blocks);
    local_env::print_reply(&text);
    (text, response.completion_tokens.into())
}

/// Load `experts`, answer every prompt, print the phase line; returns the
/// replies and the tier counters.
async fn phase(
    label: &str,
    source: &ModelSource,
    device: LocalDevice,
    experts: ExpertPlacement,
) -> (
    Vec<(String, u64)>,
    Option<(ExpertTierStats, ExpertTierStats)>,
    LocalModel,
) {
    let model = LocalModel::load(LocalModelConfig {
        source: source.clone(),
        context_length: Some(2048),
        device,
        gpu_layers: local_env::gpu_layers(),
        experts,
        engram: local_env::engram(),
        convert: local_env::convert(),
        ..Default::default()
    })
    .await
    .unwrap_or_else(|e| panic!("{label}: model must load: {e}"));
    // A warm start reads experts during the load.
    let at_load = model.expert_tier_stats();
    let started = Instant::now();
    let mut replies = Vec::new();
    for prompt in PROMPTS {
        replies.push(reply(&model, prompt).await);
    }
    let secs = started.elapsed().as_secs_f64();
    let tokens: u64 = replies.iter().map(|r| r.1).sum();
    let stats = model.expert_tier_stats();
    if let Some(engram) = model.engram_stats() {
        println!("TIERS|{label}|engram|{engram:?}");
    }
    match (at_load, stats) {
        (Some(l), Some(s)) => println!(
            "TIERS|{label}|{:.1} tok/s|{s:?}|hit rate {:.3}|read at load {}",
            tokens as f64 / secs,
            s.hit_rate().unwrap_or(0.0),
            l.disk_reads
        ),
        _ => println!("TIERS|{label}|{:.1} tok/s", tokens as f64 / secs),
    }
    (replies, at_load.zip(stats), model)
}

/// Drop `model` and give its engine thread time to hand the device memory
/// back before the next load: on a GPU two models of this size do not fit
/// the device together.
fn release(model: LocalModel) {
    drop(model);
    std::thread::sleep(std::time::Duration::from_secs(5));
}

#[tokio::test]
async fn tiered_experts_reproduce_the_reference_replies() {
    let Some(source) = local_env::model_source(false) else {
        return local_env::skip(
            "SKIP: set CUCA_LOCAL_MODEL_PATH or CUCA_LOCAL_MODEL to a MoE model",
        );
    };
    let Some(tested) = local_env::parsed::<ExpertPlacement>("CUCA_LOCAL_EXPERTS") else {
        return local_env::skip("SKIP: set CUCA_LOCAL_EXPERTS to a tiered placement");
    };
    let ExpertPlacement::Tiered(tiers) = tested.clone() else {
        panic!("CUCA_LOCAL_EXPERTS must be a tiered placement");
    };
    let reference =
        local_env::parsed::<ExpertPlacement>("CUCA_LOCAL_EXPERTS_REF").unwrap_or_default();
    let device = local_env::device();

    let (want, _, model) = phase("reference", &source, device, reference).await;
    release(model);

    let (cold, stats, model) = phase("cold", &source, device, tested).await;
    let (_, cold_stats) = stats.expect("a tiered model reports tier stats");
    assert_eq!(cold, want, "cold tiered replies equal the reference");
    assert!(cold_stats.lookups > 0, "{cold_stats:?}");
    assert!(cold_stats.disk_reads > 0, "{cold_stats:?}");
    assert!(
        cold_stats.ram_resident_bytes <= tiers.ram_bytes,
        "{cold_stats:?}"
    );
    assert!(
        cold_stats.device_resident_bytes <= tiers.device_bytes,
        "{cold_stats:?}"
    );
    let heat = model.expert_heat().expect("a tiered model exports heat");
    release(model);

    let warm_tiers = cuca::ExpertTiers {
        warm_from: Some(heat),
        ..tiers.clone()
    };
    let (warm, stats, model) =
        phase("warm", &source, device, ExpertPlacement::Tiered(warm_tiers)).await;
    let (warm_at_load, warm_stats) = stats.expect("a tiered model reports tier stats");
    assert_eq!(warm, want, "warm tiered replies equal the reference");
    assert!(warm_at_load.disk_reads > 0, "the warm start loads experts");
    // The preloaded experts save reads while generating.
    assert!(
        warm_stats.disk_reads - warm_at_load.disk_reads <= cold_stats.disk_reads,
        "warm {warm_stats:?} vs cold {cold_stats:?}"
    );
    assert!(
        warm_stats.ram_resident_bytes <= tiers.ram_bytes,
        "{warm_stats:?}"
    );
    assert!(
        warm_stats.device_resident_bytes <= tiers.device_bytes,
        "{warm_stats:?}"
    );
    drop(model);
}
