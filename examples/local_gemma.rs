//! Run Gemma 4 E2B in-process on the CPU: no server, no HTTP endpoint.
//!
//! One `LocalModel` downloads (once) and loads the model; one client routes a
//! request to it and streams the reply.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - About 2.8 GB of disk for the one-time download of the Q4_0 GGUF file
//!   (`ggml-org/gemma-4-E2B-it-GGUF`, SHA-256 verified), and roughly 1.5 GB of
//!   RAM plus the KV cache (`context_length * 36,864` bytes, 302 MB at the
//!   default 8192).
//!
//! # Run
//!
//! candle picks its Q4_0/Q8_0 SIMD kernels at compile time, so build for the
//! host CPU:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_gemma --features provider-local
//! ```
//!
//! # Configuration
//!
//! - `CUCA_LOCAL_MODEL_PATH`: path of an already downloaded GGUF file; skips
//!   the download.
//! - `CUCA_CACHE_DIR`: download cache directory, defaults to the platform cache
//!   directory.
//! - `HF_TOKEN`: Hugging Face token, only needed for gated mirrors.
//!
//! # Output
//!
//! The download progress goes to stderr; the reply prints to stdout as its
//! blocks arrive. Reasoning (`Thinking`) blocks print dimmed after a
//! `[thinking]` prefix, then the answer follows as plain text. The reply
//! depends on the model and the sampling seed.
//!
//! The first run waits for the download and the load; later runs only load.
//! Generation is CPU-bound: on one development machine (release build,
//! `target-cpu=native`) decoding ran at roughly 17 tokens per second and
//! prefill at roughly 60; expect different figures on other CPUs.

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
    // otherwise the default source is the pinned Gemma 4 E2B Q4_0 file, and the
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
    }
    let model = LocalModel::load(config).await?;
    eprintln!();

    // Stage 2: build the client. `Local` needs the model attached; the request's
    // model name is not used for routing.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model)
        .build()?;

    // Stage 3: build the request. Thinking is on, so the stream carries
    // `Thinking` blocks ahead of the answer.
    let request = UnifiedRequest::new("gemma-4-e2b-it")
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
