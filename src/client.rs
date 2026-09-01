//! Client core: [`CucaClientBuilder`], [`CucaClient`], and the
//! plugin-instrumented stream pipeline.
//!
//! # Pipeline contract
//!
//! [`CucaClient::generate_stream`] runs every request through the same stages:
//!
//! 1. `request.provider` is overwritten with the client's selected provider, so
//!    callers cannot mis-route a request by setting it on [`UnifiedRequest`].
//! 2. `on_request` hooks run, in registration order, and may mutate the request
//!    (inject context, enforce policies, count tokens).
//! 3. Under `service-prompt-cache` with a configured cache: the digest of the
//!    now-effective request is looked up. A hit returns a [`CacheHitStream`]
//!    that replays the stored blocks and never reaches provider dispatch,
//!    `execute_local_tool`, or `on_stream_chunk`, but still runs every
//!    `on_response_complete` hook exactly once. A miss falls through to
//!    dispatch below and is written back after a fully successful stream.
//! 4. The provider's `dispatch_*` method, an `impl CucaClient` block living in
//!    that provider's module, produces an [`AgentResponseStream`], or a
//!    [`CucaError::ProviderNotEnabled`] / [`CucaError::Config`] error when the
//!    feature is off or the endpoint is a `Custom` gateway with no registered
//!    adapter.
//! 5. The stream is wrapped in [`PluginStream`], which applies
//!    `on_stream_chunk` per block and `on_response_complete` exactly once when
//!    the stream ends, and (under `service-prompt-cache`) writes one cache
//!    entry after a fully successful completion.
//!
//! Token accounting: `completion_tokens` counts one token per `Text`, `Thinking`,
//! and `ToolCall` block; `prompt_tokens` stays `0`.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::error::{CucaError, PluginError};
use crate::plugin::CucaPlugin;
#[cfg(feature = "provider-anthropic")]
use crate::provider::anthropic::OAuthPkceConfig;
#[cfg(feature = "provider-llamacpp")]
use crate::provider::llamacpp::LlamaCppConfig;
use crate::request::{AgentResponseStream, PromptCacheUsage, UnifiedRequest, UnifiedResponse};
#[cfg(feature = "service-speculative")]
use crate::services::orchestrator::ModelOrchestrator;
#[cfg(feature = "service-prompt-cache")]
use crate::services::prompt_cache::{
    PromptCache, PromptCacheConfig, PromptCacheEntry, PromptCacheError, PromptCacheImportReport,
    PromptCacheSnapshot, digest_request,
};
use crate::types::{MessageContentBlock, ProviderEndpoint};

/// Cloneable, mutex-protected carrier for provider-reported metadata that
/// does not belong in [`AgentResponseStream`]'s per-block item type.
///
/// Every provider stream-construction point builds one alongside its
/// [`AgentResponseStream`] (see [`ProviderDispatch`]); adapters that never
/// report prompt-cache usage build [`Self::empty`] and never call
/// [`Self::set`], so their responses' `prompt_cache_usage` always stays
/// `None`. [`PluginStream`] reads the handle with [`Self::take`] exactly once,
/// when the inner stream reaches `None`, and copies the result into
/// [`UnifiedResponse::prompt_cache_usage`] before terminal hooks run.
///
/// The payload type, [`PromptCacheUsage`], is unconditional (defined in
/// [`crate::request`] with no feature gate), so this handle compiles and
/// behaves identically whether or not `service-prompt-cache` is enabled.
///
/// A poisoned lock is treated as "no usage" rather than propagated: metadata
/// is best-effort and must never turn a successful provider stream into an
/// error. Explicit cache/export APIs remain the place state/serialization
/// errors surface.
#[derive(Clone, Default)]
pub(crate) struct ResponseMetadataHandle(Arc<Mutex<Option<PromptCacheUsage>>>);

impl ResponseMetadataHandle {
    /// Build a handle carrying no usage. Used by every adapter that never
    /// reports provider prompt-cache usage, and by the orchestrator path.
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Record `usage`, overwriting any previously recorded value.
    ///
    /// A poisoned lock silently drops the update rather than panicking or
    /// propagating an error: see the type-level "poisoned lock" note.
    ///
    /// Called by the Anthropic SSE stream translator once it decodes
    /// `message_start` usage, so by `provider-anthropic` and by
    /// `provider-deepseek`, whose bridge reuses that translator. Every other
    /// adapter builds [`Self::empty`] and never calls this, so a build with
    /// neither feature never calls it either.
    #[cfg_attr(
        not(any(test, feature = "provider-anthropic", feature = "provider-deepseek")),
        expect(
            dead_code,
            reason = "only the Anthropic translator records usage; other adapters build `empty()`"
        )
    )]
    pub(crate) fn set(&self, usage: PromptCacheUsage) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = Some(usage);
        }
    }

    /// Take the recorded usage, if any, leaving the handle empty afterward.
    ///
    /// A poisoned lock is treated as "no usage" (`None`), never an error: see
    /// the type-level "poisoned lock" note.
    pub(crate) fn take(&self) -> Option<PromptCacheUsage> {
        self.0.lock().ok().and_then(|mut guard| guard.take())
    }
}

/// A provider's response stream paired with its [`ResponseMetadataHandle`].
///
/// Every `dispatch_*` method and shared provider stream helper
/// (`openai_compat_stream`, `gemini_stream`, `anthropic_stream`,
/// `llamacpp_completion_stream`) returns this internally instead of a bare
/// [`AgentResponseStream`]; [`CucaClient::instrument`] unwraps it into
/// [`PluginStream`]. The public [`AgentResponseStream`] item type is
/// unaffected: this wrapper never crosses a public API boundary.
pub(crate) struct ProviderDispatch {
    pub(crate) stream: AgentResponseStream,
    pub(crate) metadata: ResponseMetadataHandle,
}

/// Map an explicit [`PromptCacheError`] to the existing client-facing error
/// contract, reusing [`CucaError`] variants rather than adding a new one.
///
/// [`PromptCacheError::Json`] carries a JSON serialization failure and maps
/// to [`CucaError::Json`]; every other variant (`Config`, `Validation`,
/// `Lock`) is a configuration/state problem and maps to [`CucaError::Config`]
/// with the original error's message preserved via `Display`.
#[cfg(feature = "service-prompt-cache")]
fn map_prompt_cache_error(err: PromptCacheError) -> CucaError {
    match err {
        PromptCacheError::Json(message) => CucaError::Json { message },
        other => CucaError::Config(other.to_string()),
    }
}

/// Builder for a [`CucaClient`].
///
/// Required: `with_provider`; everything else is optional and defaults to the
/// per-provider default (`base_url`) or none (`api_key`, plugins).
pub struct CucaClientBuilder {
    provider: Option<ProviderEndpoint>,
    base_url: Option<String>,
    api_key: Option<String>,
    #[cfg(feature = "provider-anthropic")]
    bearer_token: Option<String>,
    #[cfg(feature = "provider-anthropic")]
    oauth: Option<OAuthPkceConfig>,
    plugins: Vec<Arc<dyn CucaPlugin>>,
    #[cfg(feature = "provider-llamacpp")]
    llamacpp_config: Option<LlamaCppConfig>,
    #[cfg(feature = "service-speculative")]
    orchestrator: Option<ModelOrchestrator>,
    #[cfg(feature = "service-prompt-cache")]
    prompt_cache_config: Option<PromptCacheConfig>,
    #[cfg(feature = "service-prompt-cache")]
    prompt_cache_service: Option<Arc<PromptCache>>,
}

impl CucaClientBuilder {
    /// Start an empty builder; a provider must be set before [`Self::build`].
    pub fn new() -> Self {
        Self {
            provider: None,
            base_url: None,
            api_key: None,
            #[cfg(feature = "provider-anthropic")]
            bearer_token: None,
            #[cfg(feature = "provider-anthropic")]
            oauth: None,
            plugins: Vec::new(),
            #[cfg(feature = "provider-llamacpp")]
            llamacpp_config: None,
            #[cfg(feature = "service-speculative")]
            orchestrator: None,
            #[cfg(feature = "service-prompt-cache")]
            prompt_cache_config: None,
            #[cfg(feature = "service-prompt-cache")]
            prompt_cache_service: None,
        }
    }

    /// Configure the llama.cpp adapter: the route (OpenAI-compatible
    /// chat vs native completion), the runtime knobs (`n_threads`,
    /// `n_gpu_layers`, `flash_attn`). Stored on
    /// the builder and copied into the client by [`Self::build`].
    #[cfg(feature = "provider-llamacpp")]
    pub fn with_llamacpp_config(mut self, config: LlamaCppConfig) -> Self {
        self.llamacpp_config = Some(config);
        self
    }

    /// Select the provider endpoint requests are routed to.
    pub fn with_provider(mut self, provider: ProviderEndpoint) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Base URL override; when unset, each provider adapter applies its own
    /// default.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// API key; used by adapters that read it from the client.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set an OAuth 2.0 PKCE bearer token for Anthropic.
    ///
    /// Mutually exclusive with [`Self::with_api_key`] at dispatch time: when
    /// both are set the bearer token wins (see `dispatch_anthropic`).
    #[cfg(feature = "provider-anthropic")]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Configure Anthropic OAuth 2.0 PKCE: the client id,
    /// authorize/token endpoints, and requested scopes. The config is stored
    /// and exposed via [`CucaClient::oauth_config`] so callers can drive the
    /// authorization-code flow and hand the exchanged token to
    /// [`Self::with_bearer_token`].
    #[cfg(feature = "provider-anthropic")]
    pub fn with_anthropic_oauth(mut self, oauth: OAuthPkceConfig) -> Self {
        self.oauth = Some(oauth);
        self
    }

    /// Register a plugin, in hook-invocation order.
    pub fn register_plugin(mut self, plugin: Arc<dyn CucaPlugin>) -> Self {
        self.plugins.push(plugin);
        self
    }

    /// Attach a speculative fast/slow orchestrator.
    ///
    /// When set, [`CucaClient::generate_stream`] routes every turn through
    /// [`ModelOrchestrator::execute_adaptive_turn`] instead of the provider
    /// dispatch arms. `on_request` hooks still run on the request first; the
    /// orchestrator's own tier executors use pool clients built without an
    /// orchestrator, so they dispatch to providers directly and never recurse.
    ///
    /// [`Self::with_base_url`] and [`Self::with_api_key`] configure this
    /// client, not those tier clients; the tier endpoint is
    /// [`ModelOrchestrator::with_endpoint`].
    #[cfg(feature = "service-speculative")]
    pub fn with_orchestrator(mut self, orchestrator: ModelOrchestrator) -> Self {
        self.orchestrator = Some(orchestrator);
        self
    }

    /// Configure a client-owned local response cache from a validated
    /// [`PromptCacheConfig`]; [`Self::build`] constructs the [`PromptCache`]
    /// service and reports any construction error there.
    ///
    /// Ignored when [`Self::with_prompt_cache_service`] is also called: an
    /// explicit shared service always wins.
    #[cfg(feature = "service-prompt-cache")]
    pub fn with_prompt_cache_config(mut self, config: PromptCacheConfig) -> Self {
        self.prompt_cache_config = Some(config);
        self
    }

    /// Attach a caller-owned, already-constructed [`PromptCache`] service
    /// (e.g. one shared across multiple clients, or restored from an
    /// imported snapshot). Takes precedence over
    /// [`Self::with_prompt_cache_config`].
    #[cfg(feature = "service-prompt-cache")]
    pub fn with_prompt_cache_service(mut self, service: Arc<PromptCache>) -> Self {
        self.prompt_cache_service = Some(service);
        self
    }

    /// Build the client.
    ///
    /// # Errors
    ///
    /// Returns [`CucaError::Config`] when no provider was selected, or when
    /// an explicit `service-prompt-cache` configuration fails validation.
    pub fn build(self) -> Result<CucaClient, CucaError> {
        let selected_provider = self.provider.ok_or_else(|| {
            CucaError::Config("no provider selected; call with_provider before build".into())
        })?;
        #[cfg(feature = "service-prompt-cache")]
        let prompt_cache = match self.prompt_cache_service {
            Some(service) => Some(service),
            None => match self.prompt_cache_config {
                Some(config) => Some(Arc::new(
                    PromptCache::new(config).map_err(map_prompt_cache_error)?,
                )),
                None => None,
            },
        };
        Ok(CucaClient {
            selected_provider,
            base_url: self.base_url.unwrap_or_default(),
            api_key: self.api_key,
            #[cfg(feature = "provider-anthropic")]
            bearer_token: self.bearer_token,
            #[cfg(feature = "provider-anthropic")]
            oauth: self.oauth,
            #[cfg(any(
                feature = "provider-openai",
                feature = "provider-anthropic",
                feature = "provider-deepseek",
                feature = "provider-gemini",
                feature = "provider-llamacpp",
                feature = "provider-vllm",
                feature = "provider-lmstudio",
            ))]
            http_client: reqwest::Client::new(),
            #[cfg(feature = "provider-llamacpp")]
            llamacpp_config: self.llamacpp_config,
            #[cfg(feature = "service-speculative")]
            orchestrator: self.orchestrator,
            plugins: self.plugins.into(),
            #[cfg(feature = "service-prompt-cache")]
            prompt_cache,
        })
    }
}

impl Default for CucaClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The unified async client: routes requests and runs the plugin pipeline.
///
/// Built via [`CucaClient::builder`]; `generate_stream` is the single entry
/// point that produces a plugin-instrumented [`AgentResponseStream`].
pub struct CucaClient {
    selected_provider: ProviderEndpoint,
    base_url: String,
    api_key: Option<String>,
    #[cfg(feature = "provider-anthropic")]
    bearer_token: Option<String>,
    #[cfg(feature = "provider-anthropic")]
    oauth: Option<OAuthPkceConfig>,
    #[cfg(any(
        feature = "provider-openai",
        feature = "provider-anthropic",
        feature = "provider-deepseek",
        feature = "provider-gemini",
        feature = "provider-llamacpp",
        feature = "provider-vllm",
        feature = "provider-lmstudio",
    ))]
    http_client: reqwest::Client,
    /// Registered plugins in hook-invocation order, shared with every stream
    /// this client builds: an `Arc` slice so instrumenting a request is one
    /// refcount bump instead of a fresh `Vec` allocation plus one refcount
    /// bump per plugin.
    plugins: Arc<[Arc<dyn CucaPlugin>]>,
    #[cfg(feature = "provider-llamacpp")]
    llamacpp_config: Option<LlamaCppConfig>,
    #[cfg(feature = "service-speculative")]
    orchestrator: Option<ModelOrchestrator>,
    #[cfg(feature = "service-prompt-cache")]
    prompt_cache: Option<Arc<PromptCache>>,
}

impl CucaClient {
    /// Start a builder for a new client.
    pub fn builder() -> CucaClientBuilder {
        CucaClientBuilder::new()
    }

    /// The provider endpoint this client routes requests to.
    pub fn selected_provider(&self) -> &ProviderEndpoint {
        &self.selected_provider
    }

    /// The base URL used by the provider adapter (empty until the caller sets
    /// one).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The API key, when one was configured.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// The OAuth bearer token, when one was configured (Anthropic OAuth path).
    #[cfg(feature = "provider-anthropic")]
    pub fn bearer_token(&self) -> Option<&str> {
        self.bearer_token.as_deref()
    }

    /// The configured Anthropic OAuth PKCE config, when one was set.
    #[cfg(feature = "provider-anthropic")]
    pub fn oauth_config(&self) -> Option<&OAuthPkceConfig> {
        self.oauth.as_ref()
    }

    /// The configured llama.cpp adapter config, when one was set.
    ///
    /// `dispatch_llamacpp` reads it to resolve the route and runtime knobs;
    /// when `None` the dispatch uses
    /// [`LlamaCppConfig::default`].
    #[cfg(feature = "provider-llamacpp")]
    pub fn llamacpp_config(&self) -> Option<&LlamaCppConfig> {
        self.llamacpp_config.as_ref()
    }

    /// The speculative fast/slow orchestrator, when one was
    /// attached via [`CucaClientBuilder::with_orchestrator`].
    #[cfg(feature = "service-speculative")]
    pub fn orchestrator(&self) -> Option<&ModelOrchestrator> {
        self.orchestrator.as_ref()
    }

    /// The client-owned local response cache, when one was configured via
    /// [`CucaClientBuilder::with_prompt_cache_config`] or
    /// [`CucaClientBuilder::with_prompt_cache_service`].
    #[cfg(feature = "service-prompt-cache")]
    pub fn prompt_cache(&self) -> Option<Arc<PromptCache>> {
        self.prompt_cache.clone()
    }

    /// Export every live entry of the configured prompt cache.
    ///
    /// **Sensitive full-fidelity export:** `cuca-export` intentionally
    /// includes the complete memory graph and local-cache request/response
    /// values. It may contain confidential system prompts, user messages,
    /// tool arguments and results, base64 image data, model output,
    /// signatures, and graph properties. Treat the JSON as sensitive data; do
    /// not log or publish it. CUCA does not encrypt, redact, or write it. The
    /// caller owns access control, encryption, storage, and deletion.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when no prompt cache is configured, or when the
    /// cache reports an explicit lock/state error (mapped from
    /// [`PromptCacheError`]).
    #[cfg(feature = "service-prompt-cache")]
    pub fn prompt_cache_snapshot(&self) -> Result<PromptCacheSnapshot, CucaError> {
        let cache = self
            .prompt_cache
            .as_ref()
            .ok_or_else(|| CucaError::Config("no prompt cache configured".into()))?;
        cache.snapshot().map_err(map_prompt_cache_error)
    }

    /// Validate and atomically replace the configured prompt cache's state
    /// with `snapshot`.
    ///
    /// The snapshot is validated in full before any live state is touched, so
    /// a rejected import leaves the cache unchanged. `snapshot` carries
    /// full-fidelity request/response values: see
    /// [`PromptCacheSnapshot`] for the sensitive-data warning.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when no prompt cache is configured, or when
    /// `snapshot` fails validation or the cache reports an explicit
    /// lock/state error (mapped from [`PromptCacheError`]).
    #[cfg(feature = "service-prompt-cache")]
    pub fn replace_prompt_cache_snapshot(
        &self,
        snapshot: PromptCacheSnapshot,
    ) -> Result<PromptCacheImportReport, CucaError> {
        let cache = self
            .prompt_cache
            .as_ref()
            .ok_or_else(|| CucaError::Config("no prompt cache configured".into()))?;
        cache
            .replace_snapshot(snapshot)
            .map_err(map_prompt_cache_error)
    }

    #[cfg(any(
        feature = "provider-openai",
        feature = "provider-anthropic",
        feature = "provider-deepseek",
        feature = "provider-gemini",
        feature = "provider-llamacpp",
        feature = "provider-vllm",
        feature = "provider-lmstudio",
    ))]
    /// The shared HTTP client used by provider adapters.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// The registered plugins, in hook-invocation order.
    pub fn plugins(&self) -> &[Arc<dyn CucaPlugin>] {
        &self.plugins
    }

    /// Run the full pipeline: provider selection, `on_request` hooks, provider
    /// dispatch, and plugin instrumentation of the resulting stream.
    ///
    /// When a speculative orchestrator is attached, the pipeline
    /// ends at `on_request`: the turn is handed to
    /// [`ModelOrchestrator::execute_adaptive_turn`] and the provider dispatch
    /// arms below are bypassed. The orchestrator's own stream carries its
    /// own instrumentation and never runs this client's top-level
    /// `on_stream_chunk`/`on_response_complete` hooks, with one asymmetry:
    /// under `service-prompt-cache` with a cache actually configured, a miss
    /// wraps the orchestrator stream in the same `PluginStream` every other
    /// dispatch arm uses, so terminal hooks and the cache write run for that
    /// turn; with no cache configured (or the feature compiled out), the
    /// orchestrator stream is returned unwrapped.
    ///
    /// # Errors
    ///
    /// [`CucaError::Plugin`] when an `on_request` hook fails;
    /// [`CucaError::ProviderNotEnabled`] when the selected provider's feature
    /// flag is not compiled in; [`CucaError::Config`] for `Custom` endpoints
    /// without a registered adapter; plus any provider dispatch error.
    pub async fn generate_stream(
        &self,
        mut request: UnifiedRequest,
    ) -> Result<AgentResponseStream, CucaError> {
        request.provider = self.selected_provider.clone();
        for plugin in self.plugins.iter() {
            plugin.on_request(&mut request)?; // PluginError -> CucaError::Plugin via From
        }

        // Cache lookup: the digest is computed from the request exactly as
        // it will cross the wire (after provider selection and every
        // `on_request` hook). A hit bypasses every dispatch arm below,
        // including the orchestrator; a miss captures a clone of this exact
        // request so a later fully-successful completion can be written back.
        #[cfg(feature = "service-prompt-cache")]
        let cache_write: Option<(Arc<PromptCache>, UnifiedRequest)> = match &self.prompt_cache {
            Some(cache) => {
                let key = digest_request(&request).map_err(map_prompt_cache_error)?;
                match cache.lookup(&key).map_err(map_prompt_cache_error)? {
                    Some(entry) => return Ok(self.instrument_cache_hit(entry)),
                    None => Some((Arc::clone(cache), request.clone())),
                }
            }
            None => None,
        };
        #[cfg(all(
            feature = "service-prompt-cache",
            not(any(
                feature = "provider-openai",
                feature = "provider-anthropic",
                feature = "provider-deepseek",
                feature = "provider-gemini",
                feature = "provider-llamacpp",
                feature = "provider-vllm",
                feature = "provider-lmstudio",
            ))
        ))]
        let _ = &cache_write;
        #[cfg(not(any(
            feature = "provider-openai",
            feature = "provider-anthropic",
            feature = "provider-deepseek",
            feature = "provider-gemini",
            feature = "provider-llamacpp",
            feature = "provider-vllm",
            feature = "provider-lmstudio",
        )))]
        return Err(match &self.selected_provider {
            ProviderEndpoint::LlamaCpp => CucaError::ProviderNotEnabled("provider-llamacpp"),
            ProviderEndpoint::Anthropic => CucaError::ProviderNotEnabled("provider-anthropic"),
            ProviderEndpoint::DeepSeek => CucaError::ProviderNotEnabled("provider-deepseek"),
            ProviderEndpoint::GoogleGemini => CucaError::ProviderNotEnabled("provider-gemini"),
            ProviderEndpoint::OpenAi => CucaError::ProviderNotEnabled("provider-openai"),
            ProviderEndpoint::Vllm => CucaError::ProviderNotEnabled("provider-vllm"),
            ProviderEndpoint::LmStudio => CucaError::ProviderNotEnabled("provider-lmstudio"),
            ProviderEndpoint::Custom(_) => {
                CucaError::Config("custom endpoints require a registered adapter".into())
            }
        });

        // Orchestrator path: when a speculative orchestrator is
        // attached, the turn is routed through `execute_adaptive_turn` instead
        // of the provider dispatch arms below. Plugins still ran on the request
        // above (and the cache lookup, when a cache is configured, already
        // returned a hit before reaching here); the orchestrator's own tier
        // executors use pool clients built without `with_orchestrator`, so
        // they dispatch to providers directly and never recurse back into
        // this branch.
        #[cfg(feature = "service-speculative")]
        if let Some(orchestrator) = &self.orchestrator {
            // Configured-cache/no-cache asymmetry: the orchestrator path
            // never runs top-level per-chunk/terminal plugin hooks.
            // `OrchestratorStream` carries its own instrumentation (draft
            // validation, fallback, session events), and the pool clients
            // backing its tier executors are built without plugins. With no
            // cache configured (or `service-prompt-cache` not compiled in),
            // the orchestrator stream below is returned unwrapped. With a
            // cache actually configured, a miss (`cache_write.is_some()`)
            // needs a completion point to write the entry at, so it wraps
            // the stream in the same `PluginStream` instrumentation every
            // other dispatch arm uses: for this path that is the *first*
            // time top-level `on_stream_chunk`/`on_response_complete` hooks
            // run over an orchestrator turn, not a second time.
            #[cfg(feature = "service-prompt-cache")]
            if cache_write.is_some() {
                let model = request.model.clone();
                let dispatch = ProviderDispatch {
                    stream: orchestrator.execute_adaptive_turn(request).await?,
                    metadata: ResponseMetadataHandle::empty(),
                };
                return Ok(self.instrument(model, dispatch, cache_write));
            }
            let dispatch = ProviderDispatch {
                stream: orchestrator.execute_adaptive_turn(request).await?,
                metadata: ResponseMetadataHandle::empty(),
            };
            return Ok(dispatch.stream);
        }
        // The model name is captured before dispatch: the adapter consumes the
        // request, and the completion hook reports the (possibly plugin-mutated)
        // model that actually ran.
        #[cfg(any(
            feature = "provider-openai",
            feature = "provider-anthropic",
            feature = "provider-deepseek",
            feature = "provider-gemini",
            feature = "provider-llamacpp",
            feature = "provider-vllm",
            feature = "provider-lmstudio",
        ))]
        let model = request.model.clone();
        // The endpoint is matched by reference: ProviderEndpoint is Clone, not
        // Copy, and cloning it for the match would be wasteful.
        #[cfg(any(
            feature = "provider-openai",
            feature = "provider-anthropic",
            feature = "provider-deepseek",
            feature = "provider-gemini",
            feature = "provider-llamacpp",
            feature = "provider-vllm",
            feature = "provider-lmstudio",
        ))]
        let dispatch = match &self.selected_provider {
            #[cfg(feature = "provider-llamacpp")]
            ProviderEndpoint::LlamaCpp => self.dispatch_llamacpp(request).await?,
            #[cfg(feature = "provider-anthropic")]
            ProviderEndpoint::Anthropic => self.dispatch_anthropic(request).await?,
            #[cfg(feature = "provider-deepseek")]
            ProviderEndpoint::DeepSeek => self.dispatch_deepseek(request).await?,
            #[cfg(feature = "provider-gemini")]
            ProviderEndpoint::GoogleGemini => self.dispatch_gemini(request).await?,
            #[cfg(feature = "provider-openai")]
            ProviderEndpoint::OpenAi => self.dispatch_openai(request).await?,
            #[cfg(feature = "provider-vllm")]
            ProviderEndpoint::Vllm => self.dispatch_vllm(request).await?,
            #[cfg(feature = "provider-lmstudio")]
            ProviderEndpoint::LmStudio => self.dispatch_lmstudio(request).await?,
            // Runtime complement of the compile-time provider gate: these arms
            // compile only when the matching feature is off.
            #[cfg(not(feature = "provider-llamacpp"))]
            ProviderEndpoint::LlamaCpp => {
                return Err(CucaError::ProviderNotEnabled("provider-llamacpp"));
            }
            #[cfg(not(feature = "provider-anthropic"))]
            ProviderEndpoint::Anthropic => {
                return Err(CucaError::ProviderNotEnabled("provider-anthropic"));
            }
            #[cfg(not(feature = "provider-deepseek"))]
            ProviderEndpoint::DeepSeek => {
                return Err(CucaError::ProviderNotEnabled("provider-deepseek"));
            }
            #[cfg(not(feature = "provider-gemini"))]
            ProviderEndpoint::GoogleGemini => {
                return Err(CucaError::ProviderNotEnabled("provider-gemini"));
            }
            #[cfg(not(feature = "provider-openai"))]
            ProviderEndpoint::OpenAi => {
                return Err(CucaError::ProviderNotEnabled("provider-openai"));
            }
            #[cfg(not(feature = "provider-vllm"))]
            ProviderEndpoint::Vllm => return Err(CucaError::ProviderNotEnabled("provider-vllm")),
            #[cfg(not(feature = "provider-lmstudio"))]
            ProviderEndpoint::LmStudio => {
                return Err(CucaError::ProviderNotEnabled("provider-lmstudio"));
            }
            ProviderEndpoint::Custom(_) => {
                return Err(CucaError::Config(
                    "custom endpoints require a registered adapter".into(),
                ));
            }
        };
        #[cfg(all(
            feature = "service-prompt-cache",
            any(
                feature = "provider-openai",
                feature = "provider-anthropic",
                feature = "provider-deepseek",
                feature = "provider-gemini",
                feature = "provider-llamacpp",
                feature = "provider-vllm",
                feature = "provider-lmstudio",
            )
        ))]
        return Ok(self.instrument(model, dispatch, cache_write));
        #[cfg(all(
            not(feature = "service-prompt-cache"),
            any(
                feature = "provider-openai",
                feature = "provider-anthropic",
                feature = "provider-deepseek",
                feature = "provider-gemini",
                feature = "provider-llamacpp",
                feature = "provider-vllm",
                feature = "provider-lmstudio",
            )
        ))]
        return Ok(self.instrument(model, dispatch));
    }

    /// Wrap a provider stream with the plugin instrumentation: `on_stream_chunk`
    /// per block, `on_response_complete` once at the end, the aggregated
    /// [`UnifiedResponse`] token/content accounting, and (when `cache_write`
    /// is `Some`) a one-shot cache write after a fully successful completion.
    #[cfg(all(
        feature = "service-prompt-cache",
        any(
            feature = "provider-openai",
            feature = "provider-anthropic",
            feature = "provider-deepseek",
            feature = "provider-gemini",
            feature = "provider-llamacpp",
            feature = "provider-vllm",
            feature = "provider-lmstudio",
        )
    ))]
    fn instrument(
        &self,
        model: String,
        dispatch: ProviderDispatch,
        cache_write: Option<(Arc<PromptCache>, UnifiedRequest)>,
    ) -> AgentResponseStream {
        Box::pin(PluginStream {
            inner: dispatch.stream,
            metadata: dispatch.metadata,
            plugins: Arc::clone(&self.plugins),
            started: std::time::Instant::now(),
            response: UnifiedResponse {
                model,
                provider: self.selected_provider.clone(),
                duration_secs: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: None,
                content: Vec::new(),
                prompt_cache_usage: None,
            },
            done: false,
            cache_write: cache_write.map(|(cache, request)| CacheWriteSeam { cache, request }),
            saw_error: false,
        })
    }

    /// Wrap a provider stream with the plugin instrumentation: `on_stream_chunk`
    /// per block, `on_response_complete` once at the end, and the aggregated
    /// [`UnifiedResponse`] token/content accounting.
    #[cfg(all(
        not(feature = "service-prompt-cache"),
        any(
            feature = "provider-openai",
            feature = "provider-anthropic",
            feature = "provider-deepseek",
            feature = "provider-gemini",
            feature = "provider-llamacpp",
            feature = "provider-vllm",
            feature = "provider-lmstudio",
        )
    ))]
    fn instrument(&self, model: String, dispatch: ProviderDispatch) -> AgentResponseStream {
        Box::pin(PluginStream {
            inner: dispatch.stream,
            metadata: dispatch.metadata,
            plugins: Arc::clone(&self.plugins),
            started: std::time::Instant::now(),
            response: UnifiedResponse {
                model,
                provider: self.selected_provider.clone(),
                duration_secs: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: None,
                content: Vec::new(),
                prompt_cache_usage: None,
            },
            done: false,
        })
    }

    /// Build a [`CacheHitStream`] that replays `entry`'s stored response
    /// without dispatching to a provider, running local-tool execution, or
    /// per-chunk hooks.
    #[cfg(feature = "service-prompt-cache")]
    fn instrument_cache_hit(&self, entry: PromptCacheEntry) -> AgentResponseStream {
        let response = entry.response;
        let blocks = response.content.clone().into_iter();
        Box::pin(CacheHitStream {
            blocks,
            plugins: Arc::clone(&self.plugins),
            started: std::time::Instant::now(),
            response,
            done: false,
        })
    }
}

/// Stream wrapper that runs the per-block and completion plugin hooks and
/// accumulates the [`UnifiedResponse`] handed to `on_response_complete`.
pub struct PluginStream {
    inner: AgentResponseStream,
    /// Provider-reported metadata (e.g. prompt-cache usage), read exactly
    /// once when `inner` reaches `None`.
    metadata: ResponseMetadataHandle,
    /// Shared handle on the client's registered plugins (see
    /// [`CucaClient::plugins`]); cloning the stream's handle never copies the
    /// list.
    plugins: Arc<[Arc<dyn CucaPlugin>]>,
    started: std::time::Instant,
    response: UnifiedResponse,
    // Guards the completion hook against double invocation if a consumer polls
    // again after the inner stream reported `None`.
    done: bool,
    /// Present only on a cache miss with a configured cache: the service and
    /// effective request to write back after a fully successful completion.
    #[cfg(feature = "service-prompt-cache")]
    cache_write: Option<CacheWriteSeam>,
    /// Set once any item-level error (inner stream, local-tool, or
    /// per-chunk hook) is observed; gates the cache write so a partially
    /// failed stream never writes an entry even if a consumer keeps polling
    /// past the error to `None`.
    #[cfg(feature = "service-prompt-cache")]
    saw_error: bool,
}

/// The cache service and effective request a [`PluginStream`] writes back to
/// after a fully successful completion.
#[cfg(feature = "service-prompt-cache")]
struct CacheWriteSeam {
    cache: Arc<PromptCache>,
    request: UnifiedRequest,
}

impl Stream for PluginStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(mut chunk))) => {
                if let MessageContentBlock::ToolCall { id, .. } = &chunk {
                    for plugin in this.plugins.iter() {
                        match plugin.execute_local_tool(&chunk) {
                            Ok(Some(replacement)) => {
                                let valid_replacement = matches!(
                                    &replacement,
                                    MessageContentBlock::ToolResult { tool_call_id, .. }
                                        if tool_call_id == id
                                );
                                if !valid_replacement {
                                    #[cfg(feature = "service-prompt-cache")]
                                    {
                                        this.saw_error = true;
                                    }
                                    return Poll::Ready(Some(Err(CucaError::Plugin(
                                        PluginError::Validation {
                                            schema: "local tool result".into(),
                                            message: "local tool executor must return a ToolResult for the input ToolCall id".into(),
                                        },
                                    ))));
                                }
                                chunk = replacement;
                                break;
                            }
                            Ok(None) => {}
                            Err(e) => {
                                #[cfg(feature = "service-prompt-cache")]
                                {
                                    this.saw_error = true;
                                }
                                return Poll::Ready(Some(Err(CucaError::Plugin(e))));
                            }
                        }
                    }
                }

                for plugin in this.plugins.iter() {
                    if let Err(e) = plugin.on_stream_chunk(&mut chunk) {
                        // First plugin failure wins; the failed block is neither
                        // accumulated nor token-counted, and polling continues.
                        #[cfg(feature = "service-prompt-cache")]
                        {
                            this.saw_error = true;
                        }
                        return Poll::Ready(Some(Err(CucaError::Plugin(e))));
                    }
                }
                this.response.content.push(chunk.clone());
                // One token per text/reasoning/tool-call block; images and tool
                // results carry no generated tokens.
                match &chunk {
                    MessageContentBlock::Text(_)
                    | MessageContentBlock::Thinking { .. }
                    | MessageContentBlock::ToolCall { .. } => {
                        this.response.completion_tokens += 1;
                    }
                    MessageContentBlock::ImageBase64 { .. }
                    | MessageContentBlock::ToolResult { .. } => {}
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                #[cfg(feature = "service-prompt-cache")]
                {
                    this.saw_error = true;
                }
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if !this.done {
                    this.done = true;
                    this.response.duration_secs = this.started.elapsed().as_secs_f64();
                    // Read exactly once, here: a poisoned metadata lock
                    // degrades to `None` rather than failing the stream (see
                    // `ResponseMetadataHandle`'s docs).
                    this.response.prompt_cache_usage = this.metadata.take();
                    // `finish_reason` stays `None`; no provider adapter
                    // populates it.
                    for plugin in this.plugins.iter() {
                        if let Err(e) = plugin.on_response_complete(&this.response) {
                            #[cfg(feature = "plugin-guardrails")]
                            tracing::error!(target: "cuca::client", plugin = plugin.name(), error = %e, "response completion hook failed");
                            #[cfg(not(feature = "plugin-guardrails"))]
                            let _ = e;
                        }
                    }
                    // One write, exactly here: only after a fully successful
                    // completion (never saw an item-level error) and only
                    // when a cache was actually configured for this stream.
                    // Advisory: a write failure never fails the primary
                    // stream, which has already finished successfully.
                    #[cfg(feature = "service-prompt-cache")]
                    if !this.saw_error
                        && let Some(seam) = this.cache_write.take()
                    {
                        let _ = seam.cache.insert(seam.request, this.response.clone());
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Stream that replays a cached [`UnifiedResponse`]'s stored blocks without
/// dispatching to a provider.
///
/// Per the cache-hit hook contract: yields the stored blocks in order
/// without calling `execute_local_tool` or `on_stream_chunk`, then on the
/// first end-of-stream transition invokes every terminal `on_response_complete`
/// hook exactly once against a clone of the stored response whose
/// `duration_secs` is replaced with the elapsed time since this stream was
/// constructed (every other field is the stored value, unchanged: `model`,
/// `provider`, `content`, `prompt_tokens`, `completion_tokens`,
/// `finish_reason`, `prompt_cache_usage`). Terminal hook
/// errors are swallowed/logged exactly as [`PluginStream`] does, and a
/// `done` guard prevents a duplicate completion if a consumer polls again
/// after `None`.
#[cfg(feature = "service-prompt-cache")]
struct CacheHitStream {
    blocks: std::vec::IntoIter<MessageContentBlock>,
    /// Shared handle on the client's registered plugins; see
    /// [`PluginStream::plugins`].
    plugins: Arc<[Arc<dyn CucaPlugin>]>,
    started: std::time::Instant,
    response: UnifiedResponse,
    done: bool,
}

#[cfg(feature = "service-prompt-cache")]
impl Stream for CacheHitStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if let Some(block) = this.blocks.next() {
            return Poll::Ready(Some(Ok(block)));
        }
        if !this.done {
            this.done = true;
            this.response.duration_secs = this.started.elapsed().as_secs_f64();
            for plugin in this.plugins.iter() {
                if let Err(e) = plugin.on_response_complete(&this.response) {
                    #[cfg(feature = "plugin-guardrails")]
                    tracing::error!(target: "cuca::client", plugin = plugin.name(), error = %e, "response completion hook failed");
                    #[cfg(not(feature = "plugin-guardrails"))]
                    let _ = e;
                }
            }
        }
        Poll::Ready(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal plugin whose `name()` distinguishes it in order tests.
    struct TestPlugin(&'static str);

    impl TestPlugin {
        fn new(name: &'static str) -> Self {
            TestPlugin(name)
        }
    }

    impl CucaPlugin for TestPlugin {
        fn name(&self) -> &'static str {
            self.0
        }
    }

    /// Records every `on_response_complete` payload for assertion.
    struct MetadataRecordingPlugin {
        tx: std::sync::mpsc::Sender<UnifiedResponse>,
    }

    impl CucaPlugin for MetadataRecordingPlugin {
        fn name(&self) -> &'static str {
            "metadata-recording"
        }

        fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
            self.tx
                .send(res.clone())
                .map_err(|_| PluginError::Internal("recording channel closed".into()))
        }
    }

    #[tokio::test]
    async fn response_metadata_handle_reaches_terminal_response() {
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = ResponseMetadataHandle::empty();
        handle.set(crate::request::PromptCacheUsage {
            read_tokens: 7,
            write_tokens: 3,
        });

        let stream = PluginStream {
            inner: Box::pin(tokio_stream::iter([Ok(MessageContentBlock::Text(
                "hi".into(),
            ))])),
            plugins: Arc::from([Arc::new(MetadataRecordingPlugin { tx }) as Arc<dyn CucaPlugin>]),
            started: std::time::Instant::now(),
            metadata: handle,
            response: UnifiedResponse {
                model: "model".into(),
                provider: ProviderEndpoint::OpenAi,
                duration_secs: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: None,
                content: Vec::new(),
                prompt_cache_usage: None,
            },
            #[cfg(feature = "service-prompt-cache")]
            cache_write: None,
            #[cfg(feature = "service-prompt-cache")]
            saw_error: false,
            done: false,
        };

        use tokio_stream::StreamExt;
        let _: Vec<_> = stream.collect().await;

        let completed = rx.try_recv().expect("terminal hook must have fired");
        assert_eq!(
            completed.prompt_cache_usage,
            Some(crate::request::PromptCacheUsage {
                read_tokens: 7,
                write_tokens: 3
            })
        );
    }

    #[tokio::test]
    async fn empty_metadata_handle_leaves_response_metadata_none() {
        let stream = PluginStream {
            inner: Box::pin(tokio_stream::iter([Ok(MessageContentBlock::Text(
                "hi".into(),
            ))])),
            plugins: Arc::from([]),
            started: std::time::Instant::now(),
            metadata: ResponseMetadataHandle::empty(),
            response: UnifiedResponse {
                model: "model".into(),
                provider: ProviderEndpoint::OpenAi,
                duration_secs: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: None,
                content: Vec::new(),
                prompt_cache_usage: None,
            },
            #[cfg(feature = "service-prompt-cache")]
            cache_write: None,
            #[cfg(feature = "service-prompt-cache")]
            saw_error: false,
            done: false,
        };

        use tokio_stream::StreamExt;
        let mut stream = Box::pin(stream);
        while stream.next().await.is_some() {}
        // No plugin recorded the response, but the stream itself must have
        // drained to completion without any error from the empty handle.
    }

    #[test]
    fn metadata_handle_take_survives_a_poisoned_lock() {
        let handle = ResponseMetadataHandle::empty();
        let poison_handle = handle.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poison_handle.0.lock().unwrap();
            panic!("intentionally poison the metadata mutex");
        })
        .join();

        // A metadata lock failure must not fail a successful stream: `take`
        // degrades to "no usage" rather than propagating the poison error.
        assert_eq!(handle.take(), None);
    }

    #[tokio::test]
    async fn local_executor_replaces_matching_tool_call_before_stream_hooks() {
        struct LocalExecutor;

        impl CucaPlugin for LocalExecutor {
            fn name(&self) -> &'static str {
                "local-executor"
            }

            fn execute_local_tool(
                &self,
                call: &MessageContentBlock,
            ) -> Result<Option<MessageContentBlock>, crate::error::PluginError> {
                let MessageContentBlock::ToolCall { id, .. } = call else {
                    panic!("local executor received a non-tool call");
                };
                Ok(Some(MessageContentBlock::ToolResult {
                    tool_call_id: id.clone(),
                    output: "local output".into(),
                }))
            }

            fn on_stream_chunk(
                &self,
                chunk: &mut MessageContentBlock,
            ) -> Result<(), crate::error::PluginError> {
                let MessageContentBlock::ToolResult { output, .. } = chunk else {
                    panic!("stream hook received an unreplaced tool call");
                };
                output.push_str(" after hook");
                Ok(())
            }
        }

        let stream = PluginStream {
            inner: Box::pin(tokio_stream::iter([Ok(MessageContentBlock::ToolCall {
                id: "call_1".into(),
                name: "read_tool_result".into(),
                arguments: serde_json::json!({}),
            })])),
            plugins: Arc::from([Arc::new(LocalExecutor) as Arc<dyn CucaPlugin>]),
            metadata: ResponseMetadataHandle::empty(),
            started: std::time::Instant::now(),
            response: UnifiedResponse {
                model: "model".into(),
                provider: ProviderEndpoint::OpenAi,
                duration_secs: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: None,
                content: Vec::new(),
                prompt_cache_usage: None,
            },
            #[cfg(feature = "service-prompt-cache")]
            cache_write: None,
            #[cfg(feature = "service-prompt-cache")]
            saw_error: false,
            done: false,
        };

        use tokio_stream::StreamExt;
        let chunks: Vec<_> = stream
            .map(|chunk| chunk.expect("local execution should succeed"))
            .collect()
            .await;

        assert_eq!(
            chunks,
            vec![MessageContentBlock::ToolResult {
                tool_call_id: "call_1".into(),
                output: "local output after hook".into(),
            }]
        );
    }

    #[test]
    fn build_without_provider_is_config_error() {
        let err = CucaClient::builder()
            .build()
            .err()
            .expect("build without a provider must fail");
        assert!(matches!(err, CucaError::Config(_)));
    }

    #[test]
    fn build_with_provider_succeeds() {
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::OpenAi)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));
        assert_eq!(client.selected_provider(), &ProviderEndpoint::OpenAi);
    }

    #[test]
    fn builder_defaults_base_url_empty_and_no_plugins() {
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::DeepSeek)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));
        assert_eq!(client.base_url(), "");
        assert!(client.api_key().is_none());
        assert!(client.plugins().is_empty());
    }

    #[test]
    fn with_base_url_and_api_key_are_set() {
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::Anthropic)
            .with_base_url("https://api.example.test")
            .with_api_key("sk-test")
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));
        assert_eq!(client.base_url(), "https://api.example.test");
        assert_eq!(client.api_key(), Some("sk-test"));
    }

    #[test]
    fn register_plugin_preserves_order_in_plugins() {
        let a: Arc<dyn CucaPlugin> = Arc::new(TestPlugin::new("a"));
        let b: Arc<dyn CucaPlugin> = Arc::new(TestPlugin::new("b"));
        let c: Arc<dyn CucaPlugin> = Arc::new(TestPlugin::new("c"));
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::LlamaCpp)
            .register_plugin(Arc::clone(&a))
            .register_plugin(Arc::clone(&b))
            .register_plugin(Arc::clone(&c))
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));
        let names: Vec<&str> = client.plugins().iter().map(|p| p.name()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    /// Zero-feature path: the fallback arm fires without any network access.
    /// Under `provider-openai` the adapter is compiled in and dispatch would
    /// hit the real API, so this test only runs when the feature is off.
    #[cfg(not(feature = "provider-openai"))]
    #[tokio::test]
    async fn generate_stream_with_disabled_provider_returns_provider_not_enabled() {
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::OpenAi)
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));
        let err = client
            .generate_stream(UnifiedRequest::new("x"))
            .await
            .err()
            .expect("feature is off, generate_stream must fail");
        assert!(matches!(
            err,
            CucaError::ProviderNotEnabled("provider-openai")
        ));
    }

    /// Custom endpoints have no adapter, so the Config arm fires.
    #[tokio::test]
    async fn custom_provider_requires_registered_adapter() {
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::Custom("gw".into()))
            .build()
            .unwrap_or_else(|e| panic!("provider set, build must succeed: {e}"));
        let err = client
            .generate_stream(UnifiedRequest::new("x"))
            .await
            .err()
            .expect("no adapter registered, generate_stream must fail");
        assert!(matches!(err, CucaError::Config(_)));
    }

    /// Cache lookup/miss-write pipeline semantics.
    #[cfg(feature = "service-prompt-cache")]
    mod prompt_cache_tests {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::time::Duration;

        use tokio_stream::StreamExt;

        use super::*;
        use crate::services::prompt_cache::{PromptCache, PromptCacheConfig};

        /// Per-hook invocation counters, shared with the test via `Arc`.
        #[derive(Default)]
        struct HookCounters {
            on_request: AtomicUsize,
            execute_local_tool: AtomicUsize,
            on_stream_chunk: AtomicUsize,
            on_response_complete: AtomicUsize,
        }

        /// Counts every hook invocation and forwards completed responses.
        struct CountingPlugin {
            counters: Arc<HookCounters>,
            completed: mpsc::Sender<UnifiedResponse>,
        }

        impl CucaPlugin for CountingPlugin {
            fn name(&self) -> &'static str {
                "counting-plugin"
            }
            fn on_request(&self, _req: &mut UnifiedRequest) -> Result<(), PluginError> {
                self.counters.on_request.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn execute_local_tool(
                &self,
                _call: &MessageContentBlock,
            ) -> Result<Option<MessageContentBlock>, PluginError> {
                self.counters
                    .execute_local_tool
                    .fetch_add(1, Ordering::SeqCst);
                Ok(None)
            }
            fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
                self.counters.on_stream_chunk.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
                self.counters
                    .on_response_complete
                    .fetch_add(1, Ordering::SeqCst);
                let _ = self.completed.send(res.clone());
                Ok(())
            }
        }

        /// A stored response covering `Text`, `ToolCall`, and `ImageBase64`
        /// blocks, with a sentinel `duration_secs` that must never survive a
        /// hit (the hit-stream always replaces it with its own elapsed time).
        fn stored_response() -> UnifiedResponse {
            UnifiedResponse {
                model: "cached-model".to_string(),
                provider: ProviderEndpoint::OpenAi,
                duration_secs: 999.0,
                prompt_tokens: 11,
                completion_tokens: 22,
                finish_reason: Some("stop".to_string()),
                content: vec![
                    MessageContentBlock::Text("hello".to_string()),
                    MessageContentBlock::ToolCall {
                        id: "call_1".to_string(),
                        name: "search".to_string(),
                        arguments: serde_json::json!({ "q": "x" }),
                    },
                    MessageContentBlock::ImageBase64 {
                        media_type: "image/png".to_string(),
                        data: "AAAA".to_string(),
                    },
                ],
                prompt_cache_usage: Some(PromptCacheUsage {
                    read_tokens: 5,
                    write_tokens: 0,
                }),
            }
        }

        fn cache(capacity: usize) -> Arc<PromptCache> {
            Arc::new(
                PromptCache::new(
                    PromptCacheConfig::new(capacity, Duration::from_secs(60)).unwrap(),
                )
                .unwrap(),
            )
        }

        // --- cache-hit hook semantics ---

        #[tokio::test]
        async fn cache_hit_bypasses_dispatch_and_runs_only_request_and_terminal_hooks() {
            let cache = cache(10);
            let request = UnifiedRequest::new("gpt-4o").add_user_message("hi");
            // `generate_stream` overwrites `provider` before hooks/lookup run;
            // pre-populate under that exact effective request so the digest
            // computed inside `generate_stream` matches this entry's key.
            let mut effective = request.clone();
            effective.provider = ProviderEndpoint::OpenAi;
            cache.insert(effective, stored_response()).unwrap();

            let counters = Arc::new(HookCounters::default());
            let (tx, rx) = mpsc::channel();
            let plugin: Arc<dyn CucaPlugin> = Arc::new(CountingPlugin {
                counters: counters.clone(),
                completed: tx,
            });

            let client = CucaClient::builder()
                .with_provider(ProviderEndpoint::OpenAi)
                .with_prompt_cache_service(cache)
                .register_plugin(plugin)
                .build()
                .unwrap_or_else(|e| panic!("build must succeed: {e}"));

            let before = std::time::Instant::now();
            let stream = client
                .generate_stream(request)
                .await
                .unwrap_or_else(|e| panic!("cache hit must not fail: {e}"));
            let blocks: Vec<MessageContentBlock> = stream
                .map(|b| b.unwrap_or_else(|e| panic!("hit block must be Ok: {e}")))
                .collect()
                .await;
            let test_elapsed = before.elapsed().as_secs_f64();

            // Ordered content, exactly as stored: text, tool call, image.
            assert_eq!(blocks, stored_response().content);

            // Request/terminal hooks fire exactly once; a hit never reaches
            // local-tool execution or per-chunk hooks.
            assert_eq!(counters.on_request.load(Ordering::SeqCst), 1);
            assert_eq!(counters.on_response_complete.load(Ordering::SeqCst), 1);
            assert_eq!(counters.execute_local_tool.load(Ordering::SeqCst), 0);
            assert_eq!(counters.on_stream_chunk.load(Ordering::SeqCst), 0);

            let completed = rx
                .try_recv()
                .expect("terminal hook must have fired exactly once");
            assert!(
                rx.try_recv().is_err(),
                "terminal hook must fire exactly once"
            );
            assert_eq!(completed.model, "cached-model");
            assert_eq!(completed.provider, ProviderEndpoint::OpenAi);
            assert_eq!(completed.content, stored_response().content);
            assert_eq!(completed.prompt_tokens, 11);
            assert_eq!(completed.completion_tokens, 22);
            assert_eq!(completed.finish_reason, Some("stop".to_string()));
            assert_eq!(
                completed.prompt_cache_usage,
                Some(PromptCacheUsage {
                    read_tokens: 5,
                    write_tokens: 0
                })
            );
            // `duration_secs` is replaced with this call's own elapsed time
            // (measured from cache-stream creation to end), never the stored
            // sentinel from whenever the entry was originally written.
            assert_ne!(completed.duration_secs, 999.0);
            assert!(
                completed.duration_secs <= test_elapsed + 0.25,
                "duration_secs ({}) should reflect cache-stream creation-to-end \
                 elapsed time (test took {}s)",
                completed.duration_secs,
                test_elapsed
            );
        }

        // --- miss / no-partial-write ---

        fn miss_request() -> UnifiedRequest {
            UnifiedRequest::new("m").add_user_message("hi")
        }

        fn base_response() -> UnifiedResponse {
            UnifiedResponse {
                model: "m".to_string(),
                provider: ProviderEndpoint::OpenAi,
                duration_secs: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: None,
                content: Vec::new(),
                prompt_cache_usage: None,
            }
        }

        /// Directly construct a [`PluginStream`] with a cache-write seam
        /// attached, bypassing real provider dispatch. The miss-path write
        /// mechanism (`saw_error` gating, one-shot insertion after `None`,
        /// advisory failure) is entirely dispatch-agnostic, so this is a
        /// faster, equally faithful unit test of that mechanism; the
        /// "effective post-hook request" and "full pipeline" ends of the
        /// contract are covered by the cache-hit test above, which drives
        /// the exact same `PluginStream` through the real
        /// `generate_stream` -> `instrument` seam on a hit.
        fn miss_stream(
            items: Vec<Result<MessageContentBlock, CucaError>>,
            plugins: Arc<[Arc<dyn CucaPlugin>]>,
            cache: Arc<PromptCache>,
            request: UnifiedRequest,
        ) -> Pin<Box<PluginStream>> {
            Box::pin(PluginStream {
                inner: Box::pin(tokio_stream::iter(items)),
                metadata: ResponseMetadataHandle::empty(),
                plugins,
                started: std::time::Instant::now(),
                response: base_response(),
                done: false,
                cache_write: Some(CacheWriteSeam { cache, request }),
                saw_error: false,
            })
        }

        #[tokio::test]
        async fn fully_consumed_success_writes_exactly_one_complete_entry() {
            let cache = cache(10);
            let request = miss_request();
            let mut stream = miss_stream(
                vec![
                    Ok(MessageContentBlock::Text("a".to_string())),
                    Ok(MessageContentBlock::Text("b".to_string())),
                ],
                Arc::from([]),
                cache.clone(),
                request.clone(),
            );
            while stream.next().await.is_some() {}

            let snapshot = cache.snapshot().unwrap();
            assert_eq!(
                snapshot.entries.len(),
                1,
                "insertion must be a single operation, not per block"
            );
            let entry = &snapshot.entries[0];
            assert_eq!(
                entry.request, request,
                "entry must hold the effective post-hook request"
            );
            assert_eq!(
                entry.response.content,
                vec![
                    MessageContentBlock::Text("a".to_string()),
                    MessageContentBlock::Text("b".to_string()),
                ],
                "entry must hold the normalized transformed response"
            );
        }

        #[tokio::test]
        async fn stream_item_error_leaves_cache_unchanged_even_when_drained_to_none() {
            let cache = cache(10);
            let mut stream = miss_stream(
                vec![
                    Ok(MessageContentBlock::Text("a".to_string())),
                    Err(CucaError::Provider {
                        provider: ProviderEndpoint::OpenAi,
                        message: "boom".to_string(),
                    }),
                ],
                Arc::from([]),
                cache.clone(),
                miss_request(),
            );
            // Drain all the way to `None`: proves the write is gated on
            // having observed an item-level error, not merely on "never
            // reached `None`".
            while stream.next().await.is_some() {}

            assert!(cache.snapshot().unwrap().entries.is_empty());
        }

        #[tokio::test]
        async fn failed_per_block_hook_leaves_cache_unchanged() {
            struct FailingHookPlugin;
            impl CucaPlugin for FailingHookPlugin {
                fn name(&self) -> &'static str {
                    "failing-hook"
                }
                fn on_stream_chunk(
                    &self,
                    _chunk: &mut MessageContentBlock,
                ) -> Result<(), PluginError> {
                    Err(PluginError::Internal(
                        "intentional hook failure".to_string(),
                    ))
                }
            }

            let cache = cache(10);
            let mut stream = miss_stream(
                vec![Ok(MessageContentBlock::Text("a".to_string()))],
                Arc::from([Arc::new(FailingHookPlugin) as Arc<dyn CucaPlugin>]),
                cache.clone(),
                miss_request(),
            );
            while stream.next().await.is_some() {}

            assert!(cache.snapshot().unwrap().entries.is_empty());
        }

        #[tokio::test]
        async fn dropping_the_stream_before_none_writes_nothing() {
            let cache = cache(10);
            let mut stream = miss_stream(
                vec![
                    Ok(MessageContentBlock::Text("a".to_string())),
                    Ok(MessageContentBlock::Text("b".to_string())),
                ],
                Arc::from([]),
                cache.clone(),
                miss_request(),
            );
            // Poll exactly once, then drop without ever reaching `None`.
            let first = stream.next().await;
            assert!(first.is_some());
            drop(stream);

            assert!(cache.snapshot().unwrap().entries.is_empty());
        }

        #[tokio::test]
        async fn advisory_write_failure_does_not_fail_the_primary_stream() {
            // A non-finite temperature makes `digest_request` (called inside
            // `PromptCache::insert`) fail; the write must be swallowed rather
            // than surfacing as a stream error, since the primary stream
            // already completed successfully.
            let cache = cache(10);
            let mut failing_request = miss_request();
            failing_request.temperature = Some(f32::NAN);

            let mut stream = miss_stream(
                vec![Ok(MessageContentBlock::Text("a".to_string()))],
                Arc::from([]),
                cache.clone(),
                failing_request,
            );
            let mut saw_only_ok = true;
            while let Some(item) = stream.next().await {
                if item.is_err() {
                    saw_only_ok = false;
                }
            }
            assert!(
                saw_only_ok,
                "a cache write failure must never surface as a stream item error"
            );
            assert!(
                cache.snapshot().unwrap().entries.is_empty(),
                "the failed write must not have inserted anything"
            );
        }

        // --- orchestrator miss gets instrumented + cache write ---

        /// A [`crate::services::orchestrator::TurnExecutor`] that returns one canned
        /// text block, so the orchestrator path can be exercised without
        /// real provider dispatch.
        #[cfg(feature = "service-speculative")]
        struct CannedExecutor(&'static str);

        #[cfg(feature = "service-speculative")]
        impl crate::services::orchestrator::TurnExecutor for CannedExecutor {
            fn tier_name(&self) -> &'static str {
                self.0
            }
            fn execute(&self, _request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
                Ok(Box::pin(tokio_stream::iter([Ok(
                    MessageContentBlock::Text("fast-answer".to_string()),
                )])))
            }
        }

        /// A cache miss on a client with both an orchestrator and a
        /// configured cache is wrapped in the same `PluginStream`
        /// instrumentation every other dispatch arm uses, so terminal hooks
        /// fire and a complete entry is written. The cache-off and
        /// unconfigured cases stay unwrapped.
        #[cfg(feature = "service-speculative")]
        #[tokio::test]
        async fn orchestrator_miss_gets_instrumented_and_writes_cache_entry() {
            let config = crate::services::orchestrator::SwappableModelPair {
                fast_provider: ProviderEndpoint::OpenAi,
                fast_model_id: "fast-model".to_string(),
                slow_provider: ProviderEndpoint::Anthropic,
                slow_model_id: "slow-model".to_string(),
                latency_threshold_ms: 5_000,
                fallback_on_tool_error: false,
            };
            let pool = Arc::new(crate::services::orchestrator::ClientPool::default());
            let orch = crate::services::orchestrator::ModelOrchestrator::with_executors(
                config,
                pool,
                Arc::new(CannedExecutor("fast")),
                Arc::new(CannedExecutor("slow")),
            );

            let cache = cache(10);
            let counters = Arc::new(HookCounters::default());
            let (tx, rx) = mpsc::channel();
            let plugin: Arc<dyn CucaPlugin> = Arc::new(CountingPlugin {
                counters: counters.clone(),
                completed: tx,
            });

            let client = CucaClient::builder()
                .with_provider(ProviderEndpoint::OpenAi)
                .with_orchestrator(orch)
                .with_prompt_cache_service(cache.clone())
                .register_plugin(plugin)
                .build()
                .unwrap_or_else(|e| panic!("build must succeed: {e}"));

            let request = UnifiedRequest::new("gpt-fast").add_user_message("hi");
            let mut stream = client
                .generate_stream(request)
                .await
                .unwrap_or_else(|e| panic!("orchestrator miss must dispatch: {e}"));
            let mut blocks = Vec::new();
            while let Some(block) = stream.next().await {
                blocks.push(block.unwrap_or_else(|e| panic!("block must be Ok: {e}")));
            }

            assert_eq!(
                blocks,
                vec![MessageContentBlock::Text("fast-answer".to_string())]
            );
            assert_eq!(
                counters.on_response_complete.load(Ordering::SeqCst),
                1,
                "the miss must be instrumented: terminal hook fires"
            );
            let completed = rx.try_recv().expect("terminal hook must have fired");
            assert_eq!(completed.content, blocks);

            let snapshot = cache.snapshot().unwrap();
            assert_eq!(
                snapshot.entries.len(),
                1,
                "a successful orchestrator miss must write one cache entry"
            );
            assert_eq!(snapshot.entries[0].response.content, blocks);
        }
    }
}
