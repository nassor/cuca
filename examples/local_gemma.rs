//! Run Gemma 4 in-process on the CPU or a GPU: no server, no HTTP endpoint.
//!
//! One `LocalModel` downloads (once) and loads the model; one client routes a
//! request to it and streams the reply. The default is E2B Q4_0; the
//! environment selects another checkpoint (see Configuration).
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - Disk for the one-time download of the GGUF file: 2.84 GB for the default
//!   E2B Q4_0 (`ggml-org/gemma-4-E2B-it-GGUF`, SHA-256 verified), up to
//!   8.03 GB for E4B Q8_0. RAM (device memory on a GPU) is about the file
//!   size minus the per-layer embedding table, plus the KV cache: roughly
//!   1.5 GB plus 302 MB at the
//!   default 8192 context for E2B Q4_0 (`context_length * 36,864` bytes; the
//!   general formula is on `LocalModel`). See "Choosing a model" in the
//!   local-provider docs page for every preset.
//!
//! # Run
//!
//! candle picks its quantized SIMD kernels at compile time, so build for the
//! host CPU:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_gemma --features provider-local
//! CUCA_LOCAL_MODEL=gemma4-e4b-q4_0 RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_gemma --features provider-local
//! CUCA_LOCAL_DEVICE=vulkan cargo run --release --example local_gemma --features provider-local-vulkan
//! ```
//!
//! # Configuration
//!
//! - `CUCA_LOCAL_MODEL_PATH`: path of an already downloaded GGUF file (any
//!   dense `gemma4` checkpoint); skips the download and wins over
//!   `CUCA_LOCAL_MODEL`.
//! - `CUCA_LOCAL_MODEL`: what to download: a preset name (`gemma4-e2b-q4_0`,
//!   `gemma4-e2b-q8_0`, `gemma4-e4b-q4_0`, `gemma4-e4b-q8_0`, ASCII
//!   case-insensitive) or an unverified `repo:revision:file` on Hugging Face.
//!   Default: `gemma4-e2b-q4_0`.
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
//! dimmed after a `[thinking]` prefix, then the answer follows as plain text.
//! The reply depends on the model and the sampling seed.
//!
//! The first run waits for the download and the load; later runs only load.
//! On the CPU generation is CPU-bound: on one development machine (release
//! build, `target-cpu=native`) decoding ran at roughly 17 tokens per second
//! and prefill at roughly 60; expect different figures on other CPUs. On a
//! GPU the first request is slow while CubeCL autotunes and compiles its
//! kernels.

use std::io::{Write, stdout};
use std::path::PathBuf;

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    CucaClient, DownloadProgressCallback, LocalModel, LocalModelConfig, ModelSource, UnifiedRequest,
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
    if let Ok(device) = std::env::var("CUCA_LOCAL_DEVICE")
        && !device.is_empty()
    {
        config.device = device.parse()?;
    }
    let device = config.device;
    let model = LocalModel::load(config).await?;
    eprintln!();
    eprintln!("loaded on {device:?}");

    // Stage 2: build the client. `Local` needs the model attached; the request's
    // model name is not used for routing.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model)
        .build()?;

    // Stage 3: build the request. Thinking is on, so the stream carries
    // `Thinking` blocks ahead of the answer.
    let request = UnifiedRequest::new("gemma-4-it")
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
    Ok(())
}
