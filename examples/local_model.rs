//! Run a GGUF language model in-process on the CPU or a GPU: no server, no HTTP
//! endpoint.
//!
//! One `LocalModel` downloads (once) and loads the model; one client routes a
//! request to it and streams the reply. The default is Gemma 4 E2B Q4_0; the
//! environment selects any verified preset (Gemma 3/4, Qwen 2.5/3, Llama 3.2,
//! SmolLM 3, Phi 3.5/4, DeepSeek-R1-Distill-Qwen, Granite 3 MoE) or any other
//! GGUF file of a supported architecture (see Configuration).
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - Disk for the one-time download of the GGUF file: 2.84 GB for the default
//!   (`ggml-org/gemma-4-E2B-it-GGUF`, SHA-256 verified), from 0.29 GB
//!   (`gemma3-270m-q8_0`) to 8.03 GB (`gemma4-e4b-q8_0`) for the presets. RAM
//!   (device memory on a GPU) is about the file size (Gemma 4 keeps its
//!   per-layer embedding table on disk) plus the KV cache, which
//!   `LocalModel::kv_cache_bytes` reports (the general formula is on
//!   `LocalModel`). See "Choosing a model" in the local-provider docs page
//!   for every preset.
//!
//! # Run
//!
//! candle picks its quantized SIMD kernels at compile time, so build for the
//! host CPU:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local
//! CUCA_LOCAL_MODEL=qwen3-0.6b-q8_0 RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local
//! CUCA_LOCAL_DEVICE=vulkan cargo run --release --example local_model --features provider-local-vulkan
//! ```
//!
//! # Configuration
//!
//! - `CUCA_LOCAL_MODEL_PATH`: path of an already downloaded GGUF file (any
//!   supported architecture); skips the download and wins over
//!   `CUCA_LOCAL_MODEL`.
//! - `CUCA_LOCAL_MODEL`: what to download: a preset name from
//!   `ModelSource::PRESETS` (for example `gemma4-e2b-q4_0`, `qwen3-0.6b-q8_0`,
//!   `llama3.2-1b-q4_0`; ASCII case-insensitive) or an unverified
//!   `repo:revision:file` on Hugging Face. Default: `gemma4-e2b-q4_0`.
//! - `CUCA_LOCAL_DRAFTER`: a preset name, `in-trunk`, or `CUCA_LOCAL_DRAFTER_PATH`,
//!   a path, enabling self-speculative decoding on the CPU; the acceptance
//!   counters print at the end. A preset or path is the separate
//!   multi-token-prediction drafter of a Gemma 4 model (`gemma4-e2b-mtp-f16`);
//!   `in-trunk` uses the prediction head inside the model's own file, which a
//!   Qwen 3.5 `*-MTP` GGUF carries (for example
//!   `unsloth/Qwen3.5-0.8B-MTP-GGUF`; set `CUCA_LOCAL_MODEL_PATH` to it).
//! - `CUCA_LOCAL_DEVICE`: where the model runs: `cpu` (default), `vulkan`,
//!   `vulkan:<n>`, `metal`, `cuda[:<n>]` or `rocm[:<n>]` (see `LocalDevice`).
//!   A GPU needs its feature, e.g. `--features provider-local-vulkan`.
//! - `CUCA_CACHE_DIR`: download cache directory, defaults to the platform cache
//!   directory.
//! - `HF_TOKEN`: Hugging Face token, only needed for gated mirrors.
//!
//! # Output
//!
//! The download progress goes to stderr, then `loaded on <device>`; the reply
//! prints to stdout as its blocks arrive. Reasoning (`Thinking`) blocks print
//! dimmed after a `[thinking]` prefix when the model reasons, then the answer
//! follows as plain text. The reply depends on the model and the sampling
//! seed.
//!
//! The first run waits for the download and the load; later runs only load.
//! On the CPU generation is CPU-bound: figures vary with the model and the
//! machine (see the local-provider docs page). On a GPU the first request is
//! slow while CubeCL autotunes and compiles its kernels.

use std::io::{Write, stdout};
use std::path::PathBuf;

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    CucaClient, DownloadProgressCallback, DrafterSource, LocalModel, LocalModelConfig, ModelSource,
    UnifiedRequest,
};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Stage 1: load the model. A path in the environment skips the download;
    // `CUCA_LOCAL_MODEL` picks a preset or `repo:revision:file` to download;
    // otherwise the default source is the pinned Gemma 4 E2B Q4_0 file. The
    // callback reports download progress on stderr.
    let mut config = LocalModelConfig {
        on_download_progress: Some(DownloadProgressCallback::new(|progress| {
            eprint!(
                "\r{} / {:?} bytes",
                progress.downloaded_bytes, progress.total_bytes
            );
        })),
        ..Default::default()
    };
    if let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") {
        config.source = ModelSource::File(PathBuf::from(path));
    } else if let Ok(name) = std::env::var("CUCA_LOCAL_MODEL")
        && !name.is_empty()
    {
        config.source = name.parse()?;
    }
    if let Some(path) = std::env::var_os("CUCA_LOCAL_DRAFTER_PATH") {
        config.drafter = Some(ModelSource::File(PathBuf::from(path)).into());
    } else if let Ok(name) = std::env::var("CUCA_LOCAL_DRAFTER")
        && !name.is_empty()
    {
        config.drafter = Some(if name.eq_ignore_ascii_case("in-trunk") {
            DrafterSource::InTrunk
        } else {
            name.parse::<ModelSource>()?.into()
        });
    }
    if let Ok(device) = std::env::var("CUCA_LOCAL_DEVICE")
        && !device.is_empty()
    {
        config.device = device.parse()?;
    }
    let device = config.device;
    let model = LocalModel::load(config).await?;
    let stats_handle = model.clone();
    eprintln!();
    eprintln!("loaded on {device:?}");

    // Stage 2: build the client. `Local` needs the model attached; the request's
    // model name is not used for routing.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model)
        .build()?;

    // Stage 3: build the request. Thinking is on, so a model that reasons
    // streams `Thinking` blocks ahead of the answer.
    let request = UnifiedRequest::new("local")
        .add_system_message("You are concise.")
        .add_user_message("Explain CUCA in one sentence.")
        .enable_thinking(None)
        .set_max_tokens(1024);

    // Stage 4: drain the stream. Flushing after every chunk makes the streaming
    // visible even when piped.
    let mut stream = client.generate_stream(request).await?;
    let mut in_thinking = false;
    while let Some(chunk) = stream.next().await {
        match chunk? {
            MessageContentBlock::Text(text) => {
                if in_thinking {
                    println!();
                    in_thinking = false;
                }
                print!("{text}");
            }
            MessageContentBlock::Thinking { reasoning, .. } => {
                // One block arrives per token: the prefix opens a run only.
                let prefix = if in_thinking { "" } else { "[thinking] " };
                in_thinking = true;
                print!("\x1b[2m{prefix}{reasoning}\x1b[0m");
            }
            _ => {}
        }
        stdout().flush()?;
    }
    println!();
    let stats = stats_handle.draft_stats();
    if let Some(rate) = stats.acceptance_rate() {
        eprintln!(
            "drafter: {} of {} drafted tokens accepted ({:.0}%) in {} rounds",
            stats.accepted,
            stats.drafted,
            rate * 100.0,
            stats.rounds
        );
    }
    Ok(())
}
