//! Environment and request plumbing shared by the live `provider-local` tests
//! (`provider_local.rs`, `provider_local_batch.rs`, `provider_local_mtp.rs`,
//! `provider_local_tiers.rs`, `provider_local_convert.rs`).
//!
//! Every `CUCA_*` variable those tests read is parsed here, so a name or a
//! default lives in one place; each test file decides which of them it honours
//! (for example only `provider_local.rs` and `provider_local_tiers.rs` read
//! `CUCA_LOCAL_EXPERTS`). The variables, and the files that use them, are
//! listed in `AGENTS.md` (Testing & QA). A value that does not parse panics
//! naming its variable. `examples/local_model.rs` stays self-contained and
//! does not use this module.
//!
//! `#![allow(dead_code)]`: each test binary compiles this module standalone
//! and none uses every helper (same reasoning as `tests/common/mod.rs`).
#![allow(dead_code)]

use std::fmt::{Debug, Display};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    BatchingConfig, ConvertConfig, CucaClient, DownloadProgressCallback, DrafterSource,
    LocalDevice, LocalModel, LocalModelConfig, ModelSource, UnifiedRequest, UnifiedResponse,
    WeightQuant,
};
use tokio_stream::StreamExt;

/// The user turn of the transformers oracle dumps' `chat_a` prompt.
pub const ORACLE_USER: &str = "What is the capital of France? Answer in one word.";

/// A variable's value; unset, empty and non-Unicode count as unset.
pub fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

/// A variable parsed as `T`; unset or empty is `None`.
pub fn parsed<T: FromStr>(key: &str) -> Option<T>
where
    T::Err: Display,
{
    env(key).map(|s| s.parse().unwrap_or_else(|e| panic!("{key}={s:?}: {e}")))
}

/// Whether a variable is set at all, whatever its value.
pub fn flag(key: &str) -> bool {
    std::env::var_os(key).is_some()
}

/// Whether a variable is exactly `1`.
fn is_one(key: &str) -> bool {
    std::env::var(key).as_deref() == Ok("1")
}

/// `CUCA_REQUIRE_LIVE=1`: a skip becomes a failure.
pub fn require_live() -> bool {
    is_one("CUCA_REQUIRE_LIVE")
}

/// A test that cannot run here: under `CUCA_REQUIRE_LIVE=1` it panics with
/// `reason`, otherwise it prints `reason` and the test passes.
pub fn skip(reason: &str) {
    assert!(!require_live(), "{reason}");
    println!("{reason}");
}

/// [`skip`] for "no model configured".
pub fn skip_no_model() {
    skip(
        "SKIP: local model not available: set CUCA_LOCAL_MODEL_PATH, \
         CUCA_LOCAL_MODEL or CUCA_LOCAL_DOWNLOAD=1",
    );
}

/// A path in the environment: a Hugging Face checkpoint directory (converted,
/// `CUCA_LOCAL_CONVERT_QUANT`) or a GGUF file.
pub fn path_source(path: impl Into<PathBuf>) -> ModelSource {
    let path = path.into();
    if path.is_dir() {
        ModelSource::Dir(path)
    } else {
        ModelSource::File(path)
    }
}

/// The model to load, from `CUCA_LOCAL_MODEL_PATH` (wins), else
/// `CUCA_LOCAL_MODEL`, else (when `download_default`) the default preset under
/// `CUCA_LOCAL_DOWNLOAD=1`; `None` when none applies.
pub fn model_source(download_default: bool) -> Option<ModelSource> {
    if let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") {
        return Some(path_source(path));
    }
    if let Some(source) = parsed("CUCA_LOCAL_MODEL") {
        return Some(source);
    }
    (download_default && is_one("CUCA_LOCAL_DOWNLOAD")).then(ModelSource::default_model)
}

/// `CUCA_LOCAL_DEVICE` as written: one device, or a `+`-separated list.
pub fn device_list() -> Option<String> {
    env("CUCA_LOCAL_DEVICE")
}

/// `CUCA_LOCAL_DEVICE` as one device (default CPU).
pub fn device() -> LocalDevice {
    parsed("CUCA_LOCAL_DEVICE").unwrap_or_default()
}

/// Set `config.device` and `config.shards` from `CUCA_LOCAL_DEVICE`, a
/// `+`-separated list; unset leaves the CPU.
pub fn set_devices(config: &mut LocalModelConfig) {
    if let Some(devices) = device_list() {
        config
            .set_devices(&devices)
            .unwrap_or_else(|e| panic!("CUCA_LOCAL_DEVICE={devices:?}: {e}"));
    }
}

/// `CUCA_LOCAL_BATCHING` for `LocalModelConfig::batching` (`BatchingConfig`
/// syntax, e.g. `seqs=4,kv=16384`).
pub fn batching() -> Option<BatchingConfig> {
    parsed("CUCA_LOCAL_BATCHING")
}

/// `CUCA_LOCAL_GPU_LAYERS` for `LocalModelConfig::gpu_layers`.
pub fn gpu_layers() -> Option<usize> {
    parsed("CUCA_LOCAL_GPU_LAYERS")
}

/// `LocalModelConfig::convert` from `CUCA_LOCAL_CONVERT_QUANT` (default q8_0).
pub fn convert() -> ConvertConfig {
    ConvertConfig {
        quant: parsed::<WeightQuant>("CUCA_LOCAL_CONVERT_QUANT").unwrap_or_default(),
    }
}

/// The text of the file `CUCA_LOCAL_CHAT_TEMPLATE` names, for
/// `LocalModelConfig::chat_template`.
pub fn chat_template() -> Option<String> {
    env("CUCA_LOCAL_CHAT_TEMPLATE").map(|path| {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("CUCA_LOCAL_CHAT_TEMPLATE={path:?}: {e}"))
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

/// The drafter: a file by `CUCA_LOCAL_DRAFTER_PATH` or preset in
/// `CUCA_LOCAL_DRAFTER`, or the head inside the target's own file
/// (`CUCA_LOCAL_DRAFTER=in-trunk`).
pub fn drafter_source() -> Option<DrafterSource> {
    if let Some(path) = std::env::var_os("CUCA_LOCAL_DRAFTER_PATH") {
        return Some(path_source(path).into());
    }
    let name = env("CUCA_LOCAL_DRAFTER")?;
    if name.eq_ignore_ascii_case("in-trunk") {
        return Some(DrafterSource::InTrunk);
    }
    Some(
        name.parse::<ModelSource>()
            .unwrap_or_else(|e| panic!("CUCA_LOCAL_DRAFTER={name:?}: {e}"))
            .into(),
    )
}

/// The `draft_tokens` values of `CUCA_LOCAL_DRAFT_TOKENS`, comma-separated
/// (default `[3]`).
pub fn draft_token_counts() -> Vec<usize> {
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

/// `CUCA_LOCAL_ALLOW_MISMATCH`: report a CPU speculative output that differs
/// from plain decoding instead of failing.
pub fn allow_mismatch() -> bool {
    flag("CUCA_LOCAL_ALLOW_MISMATCH")
}

/// `CUCA_LOCAL_PERF`: run the long-prompt speed scenario.
pub fn perf() -> bool {
    flag("CUCA_LOCAL_PERF")
}

/// `CUCA_LOCAL_PERF_COLD`: no round of the speed scenario hits the prefix cache.
pub fn perf_cold() -> bool {
    flag("CUCA_LOCAL_PERF_COLD")
}

/// The directory of transformers oracle dumps, `CUCA_LOCAL_REF_DIR`.
pub fn ref_dir() -> Option<PathBuf> {
    std::env::var_os("CUCA_LOCAL_REF_DIR").map(PathBuf::from)
}

/// `CUCA_LOCAL_CONVERT_ONLY`: only the repositories containing this text.
pub fn convert_only() -> Option<String> {
    std::env::var("CUCA_LOCAL_CONVERT_ONLY").ok()
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

/// Print `reply` as a `REPLY|...` line when `CUCA_LOCAL_PRINT_REPLIES` is set
/// (the diff of two runs is the regression check for attention or K/V changes).
pub fn print_reply(reply: &impl Debug) {
    if flag("CUCA_LOCAL_PRINT_REPLIES") {
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
