//! Run a GGUF language model in-process on the CPU or a GPU: no server, no HTTP
//! endpoint.
//!
//! One `LocalModel` downloads (once) and loads the model; one client routes a
//! streams the reply. The default is Gemma 4 E2B Q4_0; the
//! environment selects any verified preset (Gemma 3/4, Qwen 3/3.5/3.6,
//! SmolLM 3, Phi 4 mini, Granite 3.3, OLMo 2, OLMoE, DeepSeek-R1-Distill,
//! Muse Glimmer, Mistral Small 4) or any other
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
//!   supported architecture), or of a Hugging Face checkpoint directory
//!   (`config.json`, `*.safetensors`, tokenizer files), which is converted to
//!   a cached GGUF once; skips the download and wins over `CUCA_LOCAL_MODEL`.
//! - `CUCA_LOCAL_MODEL`: what to download: a preset name from
//!   `ModelSource::PRESETS` (for example `gemma4-e2b-q4_0`, `qwen3-0.6b-q8_0`,
//!   `phi4-mini-q4_0`; ASCII case-insensitive), an unverified
//!   `repo:revision:file` GGUF on Hugging Face, or `repo:revision`, a
//!   safetensors checkpoint that is downloaded and converted to a GGUF
//!   once. Default: `gemma4-e2b-q4_0`.
//! - `CUCA_LOCAL_CONVERT_QUANT`: the weight precision of a converted
//!   checkpoint, `f16`, `q8_0` (default) or `q4_0`.
//! - `CUCA_LOCAL_DRAFTER`: a preset name, `in-trunk`, a `repo:revision`
//!   checkpoint (DeepSeek-V4.1's DSpark stages live in the target's own
//!   checkpoint: the same `repo:revision` as `CUCA_LOCAL_MODEL` converts to the
//!   drafter's file next to the model's), or `CUCA_LOCAL_DRAFTER_PATH`, a file
//!   or a checkpoint directory, enabling self-speculative decoding on the CPU; the acceptance
//!   counters print at the end. A preset or path is the separate
//!   multi-token-prediction drafter of a Gemma 4 model (`gemma4-e2b-mtp-f16`);
//!   `in-trunk` uses the prediction head inside the model's own file, which a
//!   Qwen 3.5 `*-MTP` GGUF carries (for example
//!   `unsloth/Qwen3.5-0.8B-MTP-GGUF`; set `CUCA_LOCAL_MODEL_PATH` to it).
//! - `CUCA_LOCAL_DEVICE`: where the model runs: `cpu` (default), `vulkan`,
//!   `vulkan:<n>`, `metal`, `cuda[:<n>]` or `rocm[:<n>]` (see `LocalDevice`).
//!   A GPU needs its feature, e.g. `--features provider-local-vulkan`. A
//!   `+`-separated list such as `cuda:0+cuda:1` splits the model over those
//!   GPUs (tensor parallelism, `LocalModelConfig::shards`; a device may
//!   repeat, e.g. `vulkan:0+vulkan:0`); the line `loaded on ...` lists them.
//! - `CUCA_LOCAL_GPU_LAYERS`: layer offload (`LocalModelConfig::gpu_layers`,
//!   llama.cpp's `-ngl`): the last `n` layers run on the GPU device and the
//!   rest on the CPU, for a model that does not fit the device; the line
//!   `loaded on ...` prints the split. Not combinable with `+` device lists
//!   or a drafter.
//! - `CUCA_LOCAL_EXPERTS`: where a mixture-of-experts model's experts live
//!   (see `ExpertPlacement`): `resident` (default) or tiered caches such as
//!   `tiered:ram=600m` (CPU) or `tiered:device=4g,ram=8g` (GPU); the tier
//!   counters print at the end.
//! - `CUCA_LOCAL_ENGRAM`: the row cache and reader threads of a model with
//!   Engram tables (see `EngramCache`), e.g. `ram=2g,threads=8`; the cache
//!   counters print at the end.
//! - `CUCA_LOCAL_PROMPT`, `CUCA_LOCAL_THINKING` (`off` disables reasoning),
//!   `CUCA_LOCAL_MAX_TOKENS` (default 1024) and `CUCA_LOCAL_TEMPERATURE` (0 is
//!   greedy; default: the model's own): the request, for timing runs.
//! - `CUCA_LOCAL_BATCHING`: how concurrent requests share the engine (see
//!   `BatchingConfig`): any subset of `seqs=4,kv=16384,step=256,prefix=off`.
//!   This example sends one request, so it only changes the memory it reserves
//!   (`kv` sizes the shared K/V pool).
//! - `CUCA_LOCAL_CHAT_TEMPLATE`: path of a Jinja chat template (`.jinja`) that
//!   replaces the one in the GGUF (`LocalModelConfig::chat_template`).
//! - `CUCA_CACHE_DIR`: download cache directory, defaults to the platform cache
//!   directory.
//! - `HF_TOKEN`: Hugging Face token, only needed for gated mirrors.
//!
//! # Output
//!
//! The download progress goes to stderr, then `loaded on <devices>` with the
//! load time; the reply
//! prints to stdout as its blocks arrive. Reasoning (`Thinking`) blocks print
//! dimmed after a `[thinking]` prefix when the model reasons, then the answer
//! follows as plain text. The reply depends on the model and the sampling
//! seed.
//!
//! The first run waits for the download and the load; later runs only load.
//! On the CPU generation is CPU-bound: figures vary with the model and the
//! machine (see the local-provider docs page). On a GPU the first request is
//! slow while CubeCL autotunes and compiles its kernels.
//!
//! After the reply, stderr carries a `timing:` line: the prompt tokens and the
//! time to the first block (prompt processing and the first token), the exact
//! completion tokens the engine reports, and the tokens per second after the
//! first block.

use std::io::{Write, stdout};
use std::path::PathBuf;
use std::time::Instant;

use cuca::types::MessageContentBlock;
use cuca::{
    DownloadProgressCallback, DrafterSource, LocalModel, LocalModelConfig, ModelSource,
    UnifiedRequest, WeightQuant,
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
        let path = PathBuf::from(path);
        config.source = if path.is_dir() {
            ModelSource::Dir(path)
        } else {
            ModelSource::File(path)
        };
    } else if let Some(name) = env("CUCA_LOCAL_MODEL") {
        config.source = name.parse()?;
    }
    if let Some(path) = std::env::var_os("CUCA_LOCAL_DRAFTER_PATH") {
        let path = PathBuf::from(path);
        config.drafter = Some(
            if path.is_dir() {
                ModelSource::Dir(path)
            } else {
                ModelSource::File(path)
            }
            .into(),
        );
    } else if let Some(name) = env("CUCA_LOCAL_DRAFTER") {
        config.drafter = Some(if name.eq_ignore_ascii_case("in-trunk") {
            DrafterSource::InTrunk
        } else {
            name.parse::<ModelSource>()?.into()
        });
    }
    if let Some(quant) = env("CUCA_LOCAL_CONVERT_QUANT") {
        config.convert.quant = quant.parse::<WeightQuant>()?;
    }
    if let Some(device) = env("CUCA_LOCAL_DEVICE") {
        config.set_devices(&device)?;
    }
    if let Some(layers) = env("CUCA_LOCAL_GPU_LAYERS") {
        config.gpu_layers = Some(layers.parse()?);
    }
    if let Some(experts) = env("CUCA_LOCAL_EXPERTS") {
        config.experts = experts.parse()?;
    }
    if let Some(engram) = env("CUCA_LOCAL_ENGRAM") {
        config.engram = engram.parse()?;
    }
    if let Some(batching) = env("CUCA_LOCAL_BATCHING") {
        config.batching = batching.parse()?;
    }
    if let Some(path) = env("CUCA_LOCAL_CHAT_TEMPLATE") {
        config.chat_template = Some(std::fs::read_to_string(&path)?);
    }
    let devices = std::iter::once(config.device)
        .chain(config.shards.iter().copied())
        .map(|d| format!("{d:?}"))
        .collect::<Vec<_>>()
        .join(" + ");
    let started = Instant::now();
    let model = LocalModel::load(config).await?;
    eprintln!();
    eprintln!("load: {:.1} s", started.elapsed().as_secs_f64());
    match model.layer_split() {
        Some((gpu, total)) => eprintln!(
            "loaded on {devices}: {gpu} of {total} layers on the GPU (layers {} to {}), layers 0 to {} and the embeddings on the CPU",
            total - gpu,
            total - 1,
            total - gpu - 1
        ),
        None => eprintln!("loaded on {devices}"),
    }

    // Stage 2: build the request. Thinking is on unless `CUCA_LOCAL_THINKING=off`,
    // so a model that reasons streams `Thinking` blocks ahead of the answer.
    let prompt = env("CUCA_LOCAL_PROMPT").unwrap_or_else(|| "Explain CUCA in one sentence.".into());
    let max_tokens = match env("CUCA_LOCAL_MAX_TOKENS") {
        Some(n) => n.parse()?,
        None => 1024,
    };
    let request = UnifiedRequest::new("local")
        .add_system_message("You are concise.")
        .add_user_message(prompt)
        .set_max_tokens(max_tokens);
    let request = match env("CUCA_LOCAL_TEMPERATURE") {
        Some(t) => request.set_temperature(t.parse()?),
        None => request,
    };
    let request = if env("CUCA_LOCAL_THINKING").is_some_and(|v| v.eq_ignore_ascii_case("off")) {
        request.disable_thinking()
    } else {
        request.enable_thinking(None)
    };

    // Stage 3: start the request on the model; `generate` is what a client
    // routes to (`CucaClient` with `with_local_model` reaches the same
    // engine) and also reports the exact token counts when the stream ends.
    // Stage 4: drain the stream. Flushing after every chunk makes the streaming
    // visible even when piped.
    let asked = Instant::now();
    let mut stream = model.generate(request).await?;
    let mut in_thinking = false;
    let (mut first, mut blocks) = (None, 0u64);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if first.is_none() {
            first = Some(asked.elapsed());
        }
        blocks += 1;
        match chunk {
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
    let report = stream.report();
    if let (Some(first), Some(usage)) = (first, report.usage) {
        let rest = asked.elapsed().saturating_sub(first).as_secs_f64();
        eprintln!(
            "timing: {} prompt tokens, first block after {:.2} s (~{:.2} prompt tokens/s); {} completion tokens ({blocks} blocks), then {:.3} tokens/s after the first block",
            usage.prompt_tokens,
            first.as_secs_f64(),
            f64::from(usage.prompt_tokens) / first.as_secs_f64().max(f64::EPSILON),
            usage.completion_tokens,
            f64::from(usage.completion_tokens.saturating_sub(1)) / rest.max(f64::EPSILON)
        );
    }
    let stats = model.draft_stats();
    if let Some(rate) = stats.acceptance_rate() {
        eprintln!(
            "drafter: {} of {} drafted tokens accepted ({:.0}%) in {} rounds",
            stats.accepted,
            stats.drafted,
            rate * 100.0,
            stats.rounds
        );
    }
    if let Some(tiers) = model.expert_tier_stats() {
        eprintln!("expert tiers: {tiers:?}");
    }
    if let Some(engram) = model.engram_stats() {
        eprintln!("engram cache: {engram:?}");
    }
    Ok(())
}

/// A non-empty environment variable.
fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}
