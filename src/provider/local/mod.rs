//! In-process Gemma 4 inference (`provider-local`).
//!
//! Pure Rust on the CPU through candle 0.9.x: no llama.cpp, no FFI, and no
//! HTTP endpoint. The model runs inside the application and is consumed
//! through the library API only.
//!
//! `LocalModel::load` resolves and loads the GGUF file and starts one engine
//! thread; `CucaClientBuilder::with_local_model` attaches it to a client for
//! `ProviderEndpoint::Local`. Usage, behavior and build notes live on
//! `LocalModel`. Bounds:
//!
//! - Single-slot FIFO engine: one request runs at a time, at most 16 queue
//!   behind it (further callers wait), 64 output blocks buffered per request.
//! - K/V cache memory is `context_length * 36,864` bytes (302 MB at 8192).
//! - Per-layer embedding rows (5,040 bytes each) are read from disk per input
//!   token instead of being held in RAM (saves 1.32 GB).
//! - Prefill runs in chunks of at most 256 tokens, bounding the score matrix.

mod chat;
mod download;
mod engine;
mod gemma4;
mod output;
mod sampler;
mod tokenizer;

use std::fmt;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures_core::Stream;
use tokio::sync::{mpsc, oneshot};

use self::chat::{ChatTemplate, chat_prompt};
use self::tokenizer::Tokenizer;
use crate::client::{CucaClient, ProviderDispatch, ResponseMetadataHandle};
use crate::error::CucaError;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, ProviderEndpoint};

/// Message shared by `CucaClientBuilder::build` and `dispatch_local` when the
/// selected provider is `Local` but no model was attached.
pub(crate) const MODEL_REQUIRED: &str =
    "ProviderEndpoint::Local requires CucaClientBuilder::with_local_model";

/// Jobs that may wait behind the running one.
const JOB_QUEUE: usize = 16;
/// Output blocks buffered between the engine thread and the stream.
const EVENT_BUFFER: usize = 64;
/// Rendered text that ends a prompt inside an open reasoning channel.
const THINKING_OPEN: &str = "<|channel>thought\n";
/// Connect timeout of the one-off download client.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Where [`LocalModel::load`] gets the GGUF file from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A file hosted on Hugging Face, downloaded once into the cache and
    /// pinned to an exact `revision`.
    HuggingFace {
        /// Repository id, e.g. `ggml-org/gemma-4-E2B-it-GGUF`.
        repo: String,
        /// Commit hash the download is pinned to.
        revision: String,
        /// File name inside the repository.
        file: String,
        /// Expected lowercase hex SHA-256 of the file; `None` skips the check.
        sha256: Option<String>,
    },
    /// A GGUF file already on disk; nothing is downloaded.
    File(PathBuf),
}

impl ModelSource {
    /// Gemma 4 E2B instruction-tuned, Q4_0 quantization, from
    /// `ggml-org/gemma-4-E2B-it-GGUF` at a pinned revision with its SHA-256
    /// (2.84 GB).
    pub fn gemma4_e2b_it_q4_0() -> Self {
        ModelSource::HuggingFace {
            repo: "ggml-org/gemma-4-E2B-it-GGUF".to_string(),
            revision: "b4243c156154b6dca9324415f8c7ccc098b4aed1".to_string(),
            file: "gemma-4-E2B-it-Q4_0.gguf".to_string(),
            sha256: Some(
                "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52".to_string(),
            ),
        }
    }
}

/// Default sampling parameters of a [`LocalModel`].
///
/// `temperature` may be overridden per request with
/// `UnifiedRequest::set_temperature`; `top_k`, `top_p` and `seed` apply to
/// every request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// Softmax temperature; `0.0` selects the greedy argmax.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens; `0` disables the filter.
    pub top_k: usize,
    /// Nucleus threshold in `(0, 1]`; `1.0` disables the filter.
    pub top_p: f32,
    /// Fixed RNG seed: with `Some`, every request restarts the generator from
    /// the seed, so equal prompts sample equal tokens. `None` seeds the
    /// generator from the system once at load.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            seed: None,
        }
    }
}

/// Model download progress, reported to a [`DownloadProgressCallback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    /// Bytes of the file present on disk so far, resumed bytes included.
    pub downloaded_bytes: u64,
    /// Total file size when the server reported it.
    pub total_bytes: Option<u64>,
}

/// Shared callback invoked with [`DownloadProgress`] while a model downloads.
///
/// It runs on the engine thread during [`LocalModel::load`]; keep it cheap.
#[derive(Clone)]
pub struct DownloadProgressCallback(Arc<dyn Fn(DownloadProgress) + Send + Sync>);

impl DownloadProgressCallback {
    /// Wrap `f` as a progress callback.
    pub fn new(f: impl Fn(DownloadProgress) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl fmt::Debug for DownloadProgressCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DownloadProgressCallback(..)")
    }
}

/// Configuration for [`LocalModel::load`].
///
/// `Debug` redacts `hf_token`.
#[derive(Clone)]
pub struct LocalModelConfig {
    /// Where the GGUF file comes from.
    pub source: ModelSource,
    /// Cache directory for downloads; `None` resolves `CUCA_CACHE_DIR`, then
    /// the platform cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Hugging Face access token; `None` falls back to the `HF_TOKEN`
    /// environment variable, then to anonymous access.
    pub hf_token: Option<String>,
    /// Maximum tokens of prompt plus completion per request, also the K/V
    /// cache capacity: it costs `context_length * 36,864` bytes of RAM. Must
    /// be at least 1 and at most the model's own maximum.
    pub context_length: usize,
    /// Default sampling parameters.
    pub sampling: SamplingConfig,
    /// Optional download progress observer.
    pub on_download_progress: Option<DownloadProgressCallback>,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::gemma4_e2b_it_q4_0(),
            cache_dir: None,
            hf_token: None,
            context_length: 8192,
            sampling: SamplingConfig::default(),
            on_download_progress: None,
        }
    }
}

impl fmt::Debug for LocalModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalModelConfig")
            .field("source", &self.source)
            .field("cache_dir", &self.cache_dir)
            .field("hf_token", &self.hf_token.as_ref().map(|_| "<redacted>"))
            .field("context_length", &self.context_length)
            .field("sampling", &self.sampling)
            .field("on_download_progress", &self.on_download_progress)
            .finish()
    }
}

/// What a loaded engine hands back to `load`: the pieces the request path
/// needs without touching the engine thread.
type Ready = Result<(Arc<Tokenizer>, ChatTemplate, usize), CucaError>;

/// State shared by every clone of a [`LocalModel`].
pub(crate) struct LocalModelInner {
    /// Queue into the engine thread; dropping the last sender ends it.
    jobs: mpsc::Sender<engine::Job>,
    tokenizer: Arc<Tokenizer>,
    template: ChatTemplate,
    context_length: usize,
}

/// A loaded in-process model, cheap to clone and share.
///
/// Load it with [`LocalModel::load`], attach it with
/// `CucaClientBuilder::with_local_model`, and select
/// `ProviderEndpoint::Local`. The model runs in this process on the CPU: no
/// HTTP endpoint is involved. The engine thread ends once the last clone is
/// dropped.
///
/// # Concurrency and memory
///
/// - The engine is a single slot: requests run one at a time in FIFO order,
///   with at most 16 queued behind the running one. Past that,
///   `generate_stream` waits for a free queue slot (backpressure, not an
///   error). A running request streams through a bounded 64-block channel, so
///   a slow consumer paces generation, and dropping the stream stops it.
/// - The K/V cache is preallocated for `context_length` positions and costs
///   `context_length * 36,864` bytes (302 MB at the default 8192). It is
///   reused across requests: a prompt that shares a prefix with the previous
///   one only prefills the new tokens, reported as the response's
///   `prompt_cache_usage`.
/// - Weights stay quantized (about 1.5 GB resident). The 1.32 GB per-layer
///   embedding table is not loaded: one 5,040-byte row per input token is
///   read from the GGUF file, usually from the OS page cache.
///
/// # Behavior
///
/// - Thinking is binary: a request with thinking enabled opens the model's
///   reasoning channel, and its effort level is ignored.
/// - The engine is text-only: image blocks are rejected.
/// - `UnifiedRequest::model` is not used for routing, since one client holds
///   one model. `temperature` and `max_tokens` apply per request; `top_k`,
///   `top_p` and `seed` come from [`SamplingConfig`]. Generation also stops
///   at the context limit.
/// - Token usage in the response is exact, not estimated.
///
/// # Build
///
/// candle selects its Q4_0 and Q8_0 SIMD kernels at compile time, so build
/// for the host CPU, e.g. `RUSTFLAGS="-C target-cpu=native" cargo build
/// --release --features provider-local`; generic builds are several times
/// slower.
#[derive(Clone)]
pub struct LocalModel {
    inner: Arc<LocalModelInner>,
}

impl fmt::Debug for LocalModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalModel")
            .field("context_length", &self.inner.context_length)
            .finish_non_exhaustive()
    }
}

impl LocalModel {
    /// Load the model described by `config`: resolve or download the file,
    /// read the weights, tokenizer and chat template, and start the engine
    /// thread.
    ///
    /// The heavy work (download, hashing, parsing 2.8 GB of weights) runs on
    /// the engine thread; this future only awaits its result, so it never
    /// blocks the async runtime.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for an invalid configuration (zero or oversized
    /// `context_length`, non-finite `temperature`, `top_p` outside `(0, 1]`);
    /// [`CucaError::Io`] when the file is missing or unreadable or the thread
    /// cannot start; [`CucaError::Transport`] and [`CucaError::Http`] for
    /// download failures; [`CucaError::Provider`] for an unsupported or
    /// malformed GGUF file or a failed checksum.
    pub async fn load(config: LocalModelConfig) -> Result<Self, CucaError> {
        validate(&config)?;
        let context_length = config.context_length;
        Self::start(context_length, move || {
            let path = resolve_model_path(&config)?;
            let (engine, tokenizer, template) = engine::Engine::open(&path, &config)?;
            Ok((engine, tokenizer, template))
        })
        .await
    }

    /// Start the engine thread, build its engine with `open` there, and wait
    /// for it to report ready.
    async fn start<F>(context_length: usize, open: F) -> Result<Self, CucaError>
    where
        F: FnOnce() -> Result<(engine::Engine, Arc<Tokenizer>, ChatTemplate), CucaError>
            + Send
            + 'static,
    {
        let (jobs_tx, mut jobs_rx) = mpsc::channel::<engine::Job>(JOB_QUEUE);
        let (ready_tx, ready_rx) = oneshot::channel::<Ready>();
        std::thread::Builder::new()
            .name("cuca-local-engine".to_string())
            .spawn(move || {
                let mut engine = match open() {
                    Ok((engine, tokenizer, template)) => {
                        if ready_tx
                            .send(Ok((tokenizer, template, context_length)))
                            .is_err()
                        {
                            return;
                        }
                        engine
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Some(job) = jobs_rx.blocking_recv() {
                    engine.run(job);
                }
            })
            .map_err(|e| CucaError::Io {
                message: format!("failed to start the local engine thread: {e}"),
            })?;

        match ready_rx.await {
            Ok(Ok((tokenizer, template, context_length))) => Ok(LocalModel {
                inner: Arc::new(LocalModelInner {
                    jobs: jobs_tx,
                    tokenizer,
                    template,
                    context_length,
                }),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(CucaError::provider(
                ProviderEndpoint::Local,
                "local engine thread exited during load",
            )),
        }
    }
}

/// Reject configurations the engine cannot honor, before any thread starts.
fn validate(config: &LocalModelConfig) -> Result<(), CucaError> {
    if config.context_length == 0 {
        return Err(CucaError::Config(
            "context_length must be at least 1".into(),
        ));
    }
    let temperature = config.sampling.temperature;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(CucaError::Config(format!(
            "sampling.temperature must be a finite value >= 0, got {temperature}"
        )));
    }
    let top_p = config.sampling.top_p;
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(CucaError::Config(format!(
            "sampling.top_p must be in (0, 1], got {top_p}"
        )));
    }
    Ok(())
}

/// The GGUF path for `config.source`, downloading it first when needed.
///
/// Runs on the engine thread. The download uses a private current-thread
/// runtime and HTTP client (never the `CucaClient` one), both dropped before
/// returning: the engine loop later blocks on a channel, which must not happen
/// inside a runtime.
fn resolve_model_path(config: &LocalModelConfig) -> Result<PathBuf, CucaError> {
    match &config.source {
        ModelSource::File(path) => {
            if !path.is_file() {
                return Err(CucaError::Io {
                    message: format!("model file `{}` does not exist", path.display()),
                });
            }
            Ok(path.clone())
        }
        ModelSource::HuggingFace {
            repo,
            revision,
            file,
            sha256,
        } => {
            let cache_dir = download::resolve_cache_dir(config.cache_dir.as_deref(), &|key| {
                std::env::var_os(key)
            })?;
            let final_path = download::cached_path(&cache_dir, repo, revision, file)?;
            if !final_path.exists() {
                let token = config.hf_token.clone().or_else(|| {
                    std::env::var("HF_TOKEN")
                        .ok()
                        .filter(|token| !token.is_empty())
                });
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let client = reqwest::Client::builder()
                    .connect_timeout(CONNECT_TIMEOUT)
                    .build()?;
                runtime.block_on(download::ensure_downloaded(
                    &client,
                    download::HF_ENDPOINT,
                    repo,
                    revision,
                    file,
                    sha256.as_deref(),
                    &final_path,
                    token.as_deref(),
                    config.on_download_progress.as_ref(),
                ))?;
            }
            Ok(final_path)
        }
    }
}

/// The block stream of one local request: the receiving end of the channel the
/// engine thread writes to. Dropping it tells the engine to stop generating.
struct LocalStream {
    rx: mpsc::Receiver<Result<MessageContentBlock, CucaError>>,
}

impl Stream for LocalStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl CucaClient {
    /// Dispatch a unified request to the in-process engine.
    ///
    /// Renders the chat template, tokenizes the prompt, and queues one job for
    /// the engine thread; the returned stream yields the job's blocks. The
    /// response metadata carries the prefix-reuse counters (as prompt-cache
    /// usage) and the exact token usage once generation ends.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when no model is attached or the request's
    /// `temperature` is invalid; [`CucaError::Provider`] when the request
    /// cannot be rendered (for example an image block), the prompt does not
    /// fit the context, or the engine has shut down.
    pub(crate) async fn dispatch_local(
        &self,
        req: UnifiedRequest,
    ) -> Result<ProviderDispatch, CucaError> {
        let model = self
            .local_model()
            .ok_or_else(|| CucaError::Config(MODEL_REQUIRED.into()))?;
        let inner = &model.inner;
        if let Some(t) = req.temperature
            && (!t.is_finite() || t < 0.0)
        {
            return Err(CucaError::Config(format!(
                "temperature must be a finite value >= 0, got {t}"
            )));
        }

        let prompt = chat_prompt(&req)?;
        let rendered = inner.template.render(&prompt)?;
        let prompt_ids = inner.tokenizer.encode(&rendered);
        if prompt_ids.len() >= inner.context_length {
            return Err(CucaError::provider(
                ProviderEndpoint::Local,
                format!(
                    "prompt has {} tokens; context_length is {}",
                    prompt_ids.len(),
                    inner.context_length
                ),
            ));
        }

        let (events, rx) = mpsc::channel(EVENT_BUFFER);
        let metadata = ResponseMetadataHandle::empty();
        let job = engine::Job {
            prompt_ids,
            start_in_thinking: rendered.ends_with(THINKING_OPEN),
            enable_thinking: prompt.enable_thinking,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            events,
            metadata: metadata.clone(),
        };
        inner.jobs.send(job).await.map_err(|_| {
            CucaError::provider(ProviderEndpoint::Local, "local engine has shut down")
        })?;
        Ok(ProviderDispatch {
            stream: Box::pin(LocalStream { rx }),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use tokio_stream::StreamExt;

    use super::*;
    use crate::provider::local::engine::testing::{engine, tiny_tokenizer};
    use crate::provider::local::gemma4::tiny::{TinyModel, write_tiny_model};

    /// A template that renders the first message's content after `<bos>`.
    const TEMPLATE: &str = "{{ bos_token }}{{ messages[0]['content'] }}";

    async fn tiny_model(model: &TinyModel, context_length: usize) -> LocalModel {
        let path = model.path.clone();
        LocalModel::start(context_length, move || {
            let tokenizer = Arc::new(tiny_tokenizer());
            let template = ChatTemplate::new(TEMPLATE.to_string(), "<bos>".to_string())?;
            // The engine must be built on the engine thread, from the path.
            let tiny = TinyModel { path };
            let engine = engine(&tiny, context_length, 0.0, None);
            // `tiny` would delete the file on drop; the caller still owns it.
            std::mem::forget(tiny);
            Ok((engine, tokenizer, template))
        })
        .await
        .expect("tiny engine starts")
    }

    fn local_client(model: LocalModel) -> CucaClient {
        CucaClient::builder()
            .with_provider(ProviderEndpoint::Local)
            .with_local_model(model)
            .build()
            .expect("client builds")
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let config = LocalModelConfig::default();
        assert_eq!(config.context_length, 8192);
        assert_eq!(config.sampling, SamplingConfig::default());
        assert_eq!(config.sampling.top_k, 64);
        assert_eq!(config.source, ModelSource::gemma4_e2b_it_q4_0());
        let ModelSource::HuggingFace {
            repo,
            revision,
            file,
            sha256,
        } = ModelSource::gemma4_e2b_it_q4_0()
        else {
            panic!("default source is a Hugging Face file");
        };
        assert_eq!(repo, "ggml-org/gemma-4-E2B-it-GGUF");
        assert_eq!(revision.len(), 40);
        assert_eq!(file, "gemma-4-E2B-it-Q4_0.gguf");
        assert_eq!(sha256.map(|s| s.len()), Some(64));
    }

    #[test]
    fn debug_output_hides_the_token_and_names_the_callback() {
        let config = LocalModelConfig {
            hf_token: Some("hf_secret".into()),
            on_download_progress: Some(DownloadProgressCallback::new(|_| {})),
            ..Default::default()
        };
        let text = format!("{config:?}");
        assert!(!text.contains("hf_secret"), "{text}");
        assert!(text.contains("<redacted>"), "{text}");
        assert!(text.contains("DownloadProgressCallback(..)"), "{text}");
    }

    #[tokio::test]
    async fn load_rejects_invalid_configuration() {
        for (config, needle) in [
            (
                LocalModelConfig {
                    context_length: 0,
                    ..Default::default()
                },
                "context_length",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        temperature: f32::NAN,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "temperature",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        temperature: -0.5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "temperature",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        top_p: 0.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "top_p",
            ),
            (
                LocalModelConfig {
                    sampling: SamplingConfig {
                        top_p: 1.5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                "top_p",
            ),
        ] {
            let err = LocalModel::load(config).await.unwrap_err();
            assert!(
                matches!(&err, CucaError::Config(m) if m.contains(needle)),
                "{err}"
            );
        }
    }

    #[tokio::test]
    async fn load_reports_a_missing_model_file() {
        let config = LocalModelConfig {
            source: ModelSource::File(PathBuf::from("definitely/not/here.gguf")),
            ..Default::default()
        };
        let err = LocalModel::load(config).await.unwrap_err();
        assert!(matches!(&err, CucaError::Io { message } if message.contains("does not exist")));
    }

    #[tokio::test]
    async fn load_reports_a_file_that_is_not_gguf() {
        let path = std::env::temp_dir().join(format!("cuca-not-gguf-{}.gguf", std::process::id()));
        std::fs::write(&path, b"this is not a GGUF file").unwrap();
        let config = LocalModelConfig {
            source: ModelSource::File(path.clone()),
            ..Default::default()
        };
        let err = LocalModel::load(config).await.unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, CucaError::Io { .. }), "{err}");
    }

    #[tokio::test]
    async fn dispatch_without_a_model_is_a_config_error() {
        // `build` refuses `Local` without a model, so build for another
        // provider and re-select `Local` is impossible; assert the build check.
        let err = CucaClient::builder()
            .with_provider(ProviderEndpoint::Local)
            .build()
            .err()
            .expect("Local without a model must not build");
        assert!(matches!(&err, CucaError::Config(m) if m == MODEL_REQUIRED));
    }

    #[tokio::test]
    async fn local_requests_stream_blocks_and_report_usage() {
        let file = write_tiny_model();
        let client = local_client(tiny_model(&file, 32).await);

        let request = UnifiedRequest::new("ignored")
            .add_user_message("abcdefg")
            .set_temperature(0.0)
            .set_max_tokens(5);
        let (mut stream, response) = client.generate_stream_with_response(request).await.unwrap();
        while let Some(item) = stream.next().await {
            // Random weights may emit a malformed tool call, a parse error
            // item that does not end the stream.
            let _ = item;
        }
        let response = response.take().expect("terminal response");
        let usage = response.usage.expect("local engine reports usage");
        // `<bos>` + 7 letters.
        assert_eq!(usage.prompt_tokens, 8);
        assert!(usage.completion_tokens <= 5);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(response.provider, ProviderEndpoint::Local);
        let cache = response.prompt_cache_usage.expect("prefix reuse reported");
        assert_eq!((cache.read_tokens, cache.write_tokens), (0, 8));

        // The same prompt again reuses all but the last token.
        let request = UnifiedRequest::new("ignored")
            .add_user_message("abcdefg")
            .set_temperature(0.0)
            .set_max_tokens(2);
        let (mut stream, response) = client.generate_stream_with_response(request).await.unwrap();
        while stream.next().await.is_some() {}
        let cache = response.take().unwrap().prompt_cache_usage.unwrap();
        assert_eq!((cache.read_tokens, cache.write_tokens), (7, 1));
    }

    #[tokio::test]
    async fn queued_requests_all_complete_in_order() {
        let file = write_tiny_model();
        let client = Arc::new(local_client(tiny_model(&file, 32).await));
        let mut streams = Vec::new();
        for i in 0..4u32 {
            let request = UnifiedRequest::new("ignored")
                .add_user_message("abcdef")
                .set_temperature(0.0)
                .set_max_tokens(i + 1);
            streams.push(client.generate_stream_with_response(request).await.unwrap());
        }
        for (i, (mut stream, response)) in streams.into_iter().enumerate() {
            while stream.next().await.is_some() {}
            let usage = response.take().unwrap().usage.unwrap();
            assert!(usage.completion_tokens as usize <= i + 1);
        }
    }

    #[tokio::test]
    async fn invalid_requests_fail_before_queueing() {
        let file = write_tiny_model();
        let client = local_client(tiny_model(&file, 8).await);

        let err = client
            .generate_stream(
                UnifiedRequest::new("m")
                    .add_user_message("abc")
                    .set_temperature(f32::INFINITY),
            )
            .await
            .err()
            .expect("non-finite temperature");
        assert!(matches!(&err, CucaError::Config(m) if m.contains("temperature")));

        // `<bos>` + 7 letters fills the 8-position context.
        let err = client
            .generate_stream(UnifiedRequest::new("m").add_user_message("abcdefg"))
            .await
            .err()
            .expect("prompt fills the context");
        assert!(
            matches!(&err, CucaError::Provider { message, .. } if message.contains("context_length is 8")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn dropping_the_stream_leaves_the_engine_usable() {
        let file = write_tiny_model();
        let client = local_client(tiny_model(&file, 32).await);
        for _ in 0..3 {
            let request = UnifiedRequest::new("m")
                .add_user_message("abcd")
                .set_temperature(0.0)
                .set_max_tokens(20);
            drop(client.generate_stream(request).await.unwrap());
        }
        let request = UnifiedRequest::new("m")
            .add_user_message("abcd")
            .set_temperature(0.0)
            .set_max_tokens(2);
        let mut stream = client.generate_stream(request).await.unwrap();
        while let Some(item) = stream.next().await {
            let _ = item;
        }
    }

    #[tokio::test]
    async fn debug_prints_the_context_length() {
        let file = write_tiny_model();
        let model = tiny_model(&file, 32).await;
        assert_eq!(
            format!("{model:?}"),
            "LocalModel { context_length: 32, .. }"
        );
    }
}
