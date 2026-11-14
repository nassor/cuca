//! Configuration and request plumbing shared by the live `provider-local` tests
//! (`provider_local.rs`, `provider_local_batch.rs`, `provider_local_mtp.rs`,
//! `provider_local_tiers.rs`, `provider_local_convert.rs`).
//!
//! Every setting those tests read comes from `cuca-test.kdl` at the repository
//! root ([`test_config`]; `cuca-test.example.kdl` lists every key): the model
//! is `local { model { ... } }`, a [`LocalModelConfig`] written with its own
//! field names, and the other `local` keys steer the suites. Each test file
//! decides which settings it honours (for example only `provider_local.rs` and
//! `provider_local_tiers.rs` apply `experts` and `engram`). An invalid file
//! panics naming the key. `examples/local_model.rs` stays self-contained and
//! does not use this module.
//!
//! `#![allow(dead_code)]`: each test binary compiles this module standalone
//! and none uses every helper (same reasoning as `tests/common/mod.rs`).
#![allow(dead_code)]

use std::fmt::Debug;
use std::path::{Path, PathBuf};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    BatchingConfig, ConvertConfig, CucaClient, DownloadProgressCallback, DrafterSource,
    EngramCache, ExpertPlacement, LocalDevice, LocalModel, LocalModelConfig, ModelSource,
    UnifiedRequest, UnifiedResponse,
};
use tokio_stream::StreamExt;

/// The `cuca-test.kdl` reader shared with `tests/common`.
#[path = "../test_config/mod.rs"]
pub mod test_config;

/// The user turn of the transformers oracle dumps' `chat_a` prompt.
pub const ORACLE_USER: &str = "What is the capital of France? Answer in one word.";

/// The `local { model { ... } }` of the test configuration.
fn model() -> LocalModelConfig {
    test_config::get().local_model()
}

/// Whether `local { model { ... } }` writes `key`.
fn has(key: &str) -> bool {
    test_config::get().local_has(key)
}

/// `require_live #true`: a skip becomes a failure.
pub fn require_live() -> bool {
    test_config::get().require_live
}

/// A test that cannot run here: under `require_live` it panics with
/// `reason`, otherwise it prints `reason` and the test passes.
pub fn skip(reason: &str) {
    assert!(!require_live(), "{reason}");
    println!("{reason}");
}

/// [`skip`] for "no model configured".
pub fn skip_no_model() {
    skip(
        "SKIP: local model not available: set `local { model { source ... } }` \
         or `local { download_default #true }` in cuca-test.kdl",
    );
}

/// Checkpoints (`repo`, `revision`) and the GGUF preset made from the same
/// weights: the harness reports a converted checkpoint under the preset's name
/// so the known limits recorded for the preset apply to it. A checkpoint that
/// is no library preset (`deepseek-v41-flash-1017`) is written `source
/// "repo:revision"`.
pub const CHECKPOINT_PRESETS: &[(&str, &str, &str)] = &[
    (
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "qwen3-0.6b-q8_0",
    ),
    (
        "microsoft/Phi-4-mini-instruct",
        "cfbefacb99257ffa30c83adab238a50856ac3083",
        "phi4-mini-q4_0",
    ),
    (
        "unsloth/gemma-3-1b-it",
        "5b11413a10db4e486ef16a20101fd028f8f2499c",
        "gemma3-1b-q4_0",
    ),
    (
        "google/gemma-4-E2B-it-qat-q4_0-unquantized",
        "6befbaca7398925921802abd1f277b495b78b738",
        "gemma4-e2b-q4_0",
    ),
    (
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "deepseek-v41-flash-1017",
    ),
];

/// The model to load: the written `source`, else (when `download_default`)
/// the default preset under `local { download_default #true }`; `None` when
/// none applies.
pub fn model_source(download_default: bool) -> Option<ModelSource> {
    if has("source") {
        return Some(model().source);
    }
    (download_default && test_config::get().local.download_default).then(ModelSource::default_model)
}

/// The written `device` and `shards` joined by `+` (`vulkan:0+vulkan:1`), for
/// reports; `None` without a `device` key.
pub fn device_list() -> Option<String> {
    test_config::get().local_devices().map(|devices| {
        devices
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("+")
    })
}

/// The written `device` (default CPU).
///
/// # Panics
///
/// When `shards` are written: the calling suite runs on one device.
pub fn device() -> LocalDevice {
    let config = model();
    assert!(
        config.shards.is_empty(),
        "this suite runs on one device: remove `shards` from cuca-test.kdl"
    );
    config.device
}

/// Copy the written `device` and `shards` into `config`.
pub fn set_devices(config: &mut LocalModelConfig) {
    let written = model();
    config.device = written.device;
    config.shards = written.shards;
}

/// `batching` when written.
pub fn batching() -> Option<BatchingConfig> {
    has("batching").then(|| model().batching)
}

/// `gpu_layers` when written.
pub fn gpu_layers() -> Option<usize> {
    model().gpu_layers
}

/// `experts` when written.
pub fn experts() -> Option<ExpertPlacement> {
    has("experts").then(|| model().experts)
}

/// `local { experts_ref ... }`: the placement `provider_local_tiers.rs`
/// compares with (default resident).
pub fn experts_ref() -> ExpertPlacement {
    test_config::get()
        .local
        .experts_ref
        .clone()
        .unwrap_or_default()
}

/// `engram` (the library default when not written).
pub fn engram() -> EngramCache {
    model().engram
}

/// `convert` (default `q8_0`).
pub fn convert() -> ConvertConfig {
    model().convert
}

/// The model cache: `local { model { cache_dir ... } }`, else the top-level
/// `cache_dir`; `None` is the platform cache directory.
pub fn cache_dir() -> Option<PathBuf> {
    model().cache_dir
}

/// `hf_token` when written and not empty.
pub fn hf_token() -> Option<String> {
    model().hf_token.filter(|token| !token.is_empty())
}

/// `chat_template`, else the text of `chat_template_file`.
pub fn chat_template() -> Option<String> {
    let config = model();
    config.chat_template.or_else(|| {
        config.chat_template_file.map(|path| {
            std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("chat_template_file {}: {e}", path.display()))
        })
    })
}

/// A download progress line on stderr.
pub fn download_progress() -> DownloadProgressCallback {
    DownloadProgressCallback::new(|p| {
        eprint!(
            "\rdownloading: {} / {:?} bytes",
            p.downloaded_bytes, p.total_bytes
        );
    })
}

/// `drafter`: a file or preset, or the head inside the target's own file
/// (`drafter "in-trunk"`).
pub fn drafter_source() -> Option<DrafterSource> {
    model().drafter
}

/// The `local { draft_tokens ... }` values (default `[3]`).
pub fn draft_token_counts() -> Vec<usize> {
    let counts = &test_config::get().local.draft_tokens;
    if counts.is_empty() {
        vec![3]
    } else {
        counts.clone()
    }
}

/// `local { allow_mismatch #true }`: report a CPU speculative output that
/// differs from plain decoding instead of failing.
pub fn allow_mismatch() -> bool {
    test_config::get().local.allow_mismatch
}

/// `local { perf #true }`: run the long-prompt speed scenario.
pub fn perf() -> bool {
    test_config::get().local.perf
}

/// `local { perf_cold #true }`: no round of the speed scenario hits the
/// prefix cache.
pub fn perf_cold() -> bool {
    test_config::get().local.perf_cold
}

/// The directory of transformers oracle dumps, `oracle { ref_dir ... }`.
pub fn ref_dir() -> Option<PathBuf> {
    test_config::get().oracle.ref_dir.clone()
}

/// `oracle { convert_only ... }`: only the repositories containing this text.
pub fn convert_only() -> Option<String> {
    test_config::get().oracle.convert_only.clone()
}

/// The oracle dumps (`*.json` files) in `dir`; unreadable or unparsable files
/// are skipped.
pub fn oracle_dumps(dir: &Path) -> impl Iterator<Item = serde_json::Value> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "json"))
        .filter_map(|entry| serde_json::from_str(&std::fs::read_to_string(entry.path()).ok()?).ok())
}

/// Print `reply` as a `REPLY|...` line under `local { print_replies #true }`
/// (the diff of two runs is the regression check for attention or K/V changes).
pub fn print_reply(reply: &impl Debug) {
    if test_config::get().local.print_replies {
        println!("REPLY|{reply:?}");
    }
}

/// A client whose `Local` provider serves `model`.
pub fn client(model: &LocalModel) -> CucaClient {
    CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model.clone())
        .build()
        .expect("client builds")
}

/// Run `request` on `model`: every block of its stream, and the terminal
/// response.
pub async fn drain(
    model: &LocalModel,
    request: UnifiedRequest,
) -> (Vec<MessageContentBlock>, UnifiedResponse) {
    let client = client(model);
    let (mut stream, handle) = client
        .generate_stream_with_response(request)
        .await
        .expect("request dispatches");
    let mut blocks = Vec::new();
    while let Some(item) = stream.next().await {
        blocks.push(item.expect("stream item"));
    }
    (blocks, handle.take().expect("terminal response"))
}

/// Everything the model wrote, reasoning and answer in stream order.
pub fn written(blocks: &[MessageContentBlock]) -> String {
    let mut text = String::new();
    for block in blocks {
        match block {
            MessageContentBlock::Text(t) => text.push_str(t),
            MessageContentBlock::Thinking { reasoning, .. } => text.push_str(reasoning),
            _ => {}
        }
    }
    text
}
