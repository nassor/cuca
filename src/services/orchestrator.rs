//! Speculative fast/slow model pairing and deterministic complexity routing.
//!
//! This module implements the fast/slow model-swapping engine: the
//! [`SwappableModelPair`] configuration record, the [`ClientPool`] shared
//! client registry, and the [`ComplexityEvaluator`] that routes a prompt to a
//! fast or slow tier.
//!
//! An explicit-call service, never a [`crate::plugin::CucaPlugin`]
//! ([`crate::services`] owns that contract): no pipeline hook can route a turn
//! across two tiers. Callers either drive
//! [`ModelOrchestrator::execute_adaptive_turn`] directly or attach the
//! orchestrator with
//! [`CucaClientBuilder::with_orchestrator`](crate::CucaClientBuilder::with_orchestrator)
//! so [`CucaClient::generate_stream`](crate::CucaClient::generate_stream) routes
//! through it.
//!
//! # Tier semantics
//!
//! - **Fast tier**: a low-latency routing model used for simple operations
//!   such as intent classification and parameter extraction.
//! - **Slow tier**: a high-capacity model used for deep tool chains, large
//!   inputs, and multi-file context.
//!
//! Routing is deterministic and driven by three indicators:
//!
//! 1. **Tool-call depth**: the number of messages carrying `ToolCall` /
//!    `ToolResult` content blocks.
//! 2. **Input token volume**: approximated from text content.
//! 3. **Multi-file context**: the number of distinct path-like file references
//!    in message text.
//!
//! # Token accounting note
//!
//! This evaluator approximates input tokens with a character-count heuristic
//! (`chars / 4`); it does not pull a tokenizer. The exact counter is
//! plugin-memory's `tiktoken-rs`, which should be used when precise accounting
//! is required. The thresholds here are tunable defaults.
//!
//! # Orchestrator pipeline
//!
//! [`ModelOrchestrator::execute_adaptive_turn`] runs the speculative fast/slow
//! pipeline in three stages:
//!
//! 1. **Complexity routing**: [`ComplexityEvaluator`] decides the tier;
//!    [`Complexity::Slow`] requests go straight to the slow tier with no draft
//!    phase.
//! 2. **Speculative draft**: the fast tier streams blocks through the
//!    [`DraftValidator`]; accepted blocks pass through to the caller, rejected
//!    blocks (malformed tool calls, invalid JSON, low confidence) are captured
//!    as a synthetic `ToolResult` appended to a working copy of the request.
//! 3. **Fallback cascade**: when `fallback_on_tool_error` is set, a rejection
//!    re-routes the turn to the slow tier with the captured error state, up to
//!    two cascades; exhaustion surfaces the last rejection as a
//!    [`CucaError::Provider`].
//!
//! # Latency swap boundary
//!
//! The fast tier gets `latency_threshold_ms` to produce its next block. The
//! guard runs when the fast stream reports `Pending`: once the deadline has
//! passed, the orchestrator swaps to the slow tier at the next natural poll
//! boundary. Blocks already yielded stay delivered, and the swap applies to
//! the remainder of the turn. Every swap (latency or fallback) appends a
//! `SessionEvent::ModelSwap` record when a session store is registered.
//!
//! # Session-log edge (documented-optional)
//!
//! `ModelOrchestrator::with_session_store` and the `SessionEvent::ModelSwap`
//! records it enables exist only under `plugin-session-log`: without that
//! feature the method is compiled out of existence rather than accepting a
//! store and silently dropping every swap. `service-speculative` therefore
//! declares no Cargo edge to it, and the session-log plugin stays ignorant that
//! this service exists.
//!
//! # Recursion guard
//!
//! The real [`PoolTurnExecutor`]s execute through pool clients built without
//! `with_orchestrator`, so their `generate_stream` calls dispatch straight to
//! the provider and never re-enter the orchestrator.
//!
//! # Tier endpoint
//!
//! Pool clients are built from the orchestrator's own configuration, not from
//! the [`crate::client::CucaClient`] that holds it: an orchestrator also runs
//! standalone through [`ModelOrchestrator::execute_adaptive_turn`], and its
//! tiers may sit on providers the enclosing client never names.
//! [`ModelOrchestrator::with_endpoint`] sets that endpoint; unset, each
//! provider adapter falls back to its own default.
//!
//! # Sync trait, async execution
//!
//! [`TurnExecutor::execute`] is deliberately synchronous. The real executor
//! launches `generate_stream` in a `tokio::spawn` task and hands back a stream
//! that resolves the spawned result on its first poll, so the trait needs no
//! async method (and no async-trait dependency). `execute` therefore requires
//! a running tokio runtime: `execute_adaptive_turn`, being async, guarantees
//! one at every call site.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use tokio::sync::oneshot;
use tokio_stream::Stream;

use crate::error::CucaError;
#[cfg(feature = "plugin-session-log")]
use crate::plugin::SessionStorePlugin;
#[cfg(all(test, feature = "service-speculative"))]
use crate::request::PromptCacheDirective;
use crate::request::{AgentResponseStream, UnifiedRequest};
#[cfg(feature = "plugin-session-log")]
use crate::session::{SessionEvent, SessionRecord};
use crate::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};

/// A pair of models the orchestrator can swap between based on request
/// complexity: a fast, low-latency routing model and a slow, high-capacity
/// model, each on its own provider endpoint.
///
/// `latency_threshold_ms` is the ceiling under which the fast tier is expected
/// to respond; when it is exceeded, or a tool step errors, the orchestrator
/// may fall back to the slow tier when `fallback_on_tool_error` is set.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SwappableModelPair {
    /// Provider endpoint serving the fast tier.
    pub fast_provider: ProviderEndpoint,
    /// Model id served by the fast provider.
    pub fast_model_id: String,
    /// Provider endpoint serving the slow tier.
    pub slow_provider: ProviderEndpoint,
    /// Model id served by the slow provider.
    pub slow_model_id: String,
    /// Fast-tier response latency ceiling in milliseconds; beyond this the
    /// orchestrator may fall back to the slow tier.
    pub latency_threshold_ms: u64,
    /// Whether a tool error triggers a fallback to the slow tier.
    pub fallback_on_tool_error: bool,
}

/// Caches one `CucaClient` per `(provider, base_url)` so the orchestrator and
/// user code share connections instead of rebuilding them per turn.
///
/// # Growth
///
/// The map is deliberately uncapped, and its size is a property of the
/// deployment rather than of traffic: one entry per distinct
/// `(provider, base_url)` an application actually talks to (two for a
/// [`SwappableModelPair`] on two endpoints, one when both tiers share a
/// backend), reused by every later turn. Repeated requests never add entries.
/// A caller that mints endpoints per request (a per-tenant base URL, say)
/// owns that bound and should build a client per tenant instead of routing it
/// through one shared pool. [`Self::len`] is the O(1) usage gauge.
pub struct ClientPool {
    clients: Mutex<HashMap<(ProviderEndpoint, String), Arc<crate::client::CucaClient>>>,
}

impl ClientPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// Return the cached client for `(provider, base_url)`, building and
    /// caching a new one on first use.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when the pool lock is poisoned, plus whatever
    /// [`crate::client::CucaClientBuilder::build`] rejects for the requested
    /// provider.
    ///
    /// `api_key` is applied only when `Some`; pass `None` to use a provider's
    /// default auth. A provider feature that is not compiled does not fail
    /// here: the builder succeeds and the missing feature surfaces as
    /// [`CucaError::ProviderNotEnabled`] at `generate_stream` time.
    pub fn get_or_create(
        &self,
        provider: &ProviderEndpoint,
        base_url: &str,
        api_key: Option<&str>,
    ) -> Result<Arc<crate::client::CucaClient>, CucaError> {
        let mut clients = self
            .clients
            .lock()
            .map_err(|_| CucaError::Config("client pool lock poisoned".into()))?;
        // One owned key for both the probe and the insert: `ProviderEndpoint`
        // and `base_url` are heap-backed, so building it twice would allocate
        // the same key twice on every miss.
        let key = (provider.clone(), base_url.to_string());
        if let Some(client) = clients.get(&key) {
            return Ok(client.clone());
        }
        let mut builder = crate::client::CucaClient::builder()
            .with_provider(provider.clone())
            .with_base_url(base_url);
        // `with_api_key` takes `impl Into<String>`; only feed it when the caller
        // actually supplied a key (the builder has no Option-taking variant).
        if let Some(api_key) = api_key {
            builder = builder.with_api_key(api_key);
        }
        let client = Arc::new(builder.build()?);
        clients.insert(key, client.clone());
        Ok(client)
    }

    /// Number of unique `(provider, base_url)` entries cached.
    pub fn len(&self) -> usize {
        self.clients.lock().map(|c| c.len()).unwrap_or_else(|_| 0)
    }
    /// Whether the pool holds no cached clients.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ClientPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Routing decision for a request: serve from the fast or slow tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    /// Low-latency routing model.
    Fast,
    /// High-capacity model.
    Slow,
}

/// Deterministic complexity router.
///
/// The three public thresholds are tunable defaults; see [`Self::evaluate`]
/// for the exact decision rules.
#[derive(Debug, Clone)]
pub struct ComplexityEvaluator {
    /// Tool-call depth (`ToolCall`/`ToolResult` round-trips) at or beyond which
    /// a request is routed to the slow tier.
    pub slow_tool_call_depth: usize,
    /// Approximated input tokens at or beyond which a request is routed to the
    /// slow tier.
    pub slow_input_tokens: usize,
    /// Distinct file references at or beyond which a request is routed to the
    /// slow tier.
    pub slow_multi_file_threshold: usize,
}

impl Default for ComplexityEvaluator {
    fn default() -> Self {
        Self {
            slow_tool_call_depth: 1,
            slow_input_tokens: 2_000,
            slow_multi_file_threshold: 3,
        }
    }
}

impl ComplexityEvaluator {
    /// Route `req` to a tier.
    ///
    /// Simple operations (intent classification, parameter extraction) yield
    /// [`Complexity::Fast`]; deep tool chains, large inputs, and multi-file
    /// context yield [`Complexity::Slow`]. Tool-call depth is checked first,
    /// then input volume, then file references.
    pub fn evaluate(&self, req: &UnifiedRequest) -> Complexity {
        let tokens: usize = req
            .messages
            .iter()
            .map(|m| m.content.iter().map(block_tokens).sum::<usize>())
            .sum();
        let tool_depth = req
            .messages
            .iter()
            .map(|m| usize::from(has_tool_content(m)))
            .sum::<usize>();
        let file_refs = req.messages.iter().map(file_ref_count).sum::<usize>();
        if tool_depth >= self.slow_tool_call_depth
            || tokens >= self.slow_input_tokens
            || file_refs >= self.slow_multi_file_threshold
        {
            Complexity::Slow
        } else {
            Complexity::Fast
        }
    }
}

/// Approximate the input-token cost of a single content block.
///
/// Text, thinking, and tool-call argument text are estimated at `chars / 4`
/// (floor, minimum 1). Image and tool-result blocks contribute a small fixed
/// count. This is a heuristic, not an exact tokenizer.
fn block_tokens(block: &MessageContentBlock) -> usize {
    match block {
        MessageContentBlock::Text(text) => approx_tokens(text.chars().count()),
        MessageContentBlock::Thinking { reasoning, .. } => approx_tokens(reasoning.chars().count()),
        MessageContentBlock::ToolCall { arguments, .. } => {
            // `Value` always serializes; on the (impossible) failure treat it
            // as zero tokens rather than panicking.
            serde_json::to_string(arguments)
                .map(|s| approx_tokens(s.chars().count()))
                .unwrap_or(0)
        }
        // An image contributes no measurable token volume in this heuristic.
        MessageContentBlock::ImageBase64 { .. } => 0,
        // A tool result carries output text; count a single token.
        MessageContentBlock::ToolResult { .. } => 1,
    }
}

/// `chars / 4`, floored, with a minimum of 1.
fn approx_tokens(chars: usize) -> usize {
    (chars / 4).max(1)
}

/// Whether a message carries any `ToolCall` or `ToolResult` block (tool-call
/// depth signal).
fn has_tool_content(msg: &UnifiedMessage) -> bool {
    msg.content.iter().any(|b| {
        matches!(
            b,
            MessageContentBlock::ToolCall { .. } | MessageContentBlock::ToolResult { .. }
        )
    })
}

/// Count distinct path-like file references across a message's text blocks.
///
/// A token is path-like if it starts with `./` or `/`, or contains a `.`
/// followed by a known source/config extension. Distinctness is deduped with a
/// `HashSet` per message; punctuation is stripped from token edges.
fn file_ref_count(msg: &UnifiedMessage) -> usize {
    let mut seen = HashSet::new();
    for block in &msg.content {
        if let MessageContentBlock::Text(text) = block {
            for token in text.split_whitespace() {
                let candidate = token.trim_matches(|c: char| {
                    !(c.is_alphanumeric() || c == '.' || c == '/' || c == '_' || c == '-')
                });
                if is_path_like(candidate) {
                    seen.insert(candidate);
                }
            }
        }
    }
    seen.len()
}

/// Heuristic: a token names a file if it is a relative/absolute path or ends in
/// a known extension.
fn is_path_like(token: &str) -> bool {
    if token.starts_with("./") || token.starts_with('/') {
        return true;
    }
    // Known source/config extensions that make a `word.ext` token a likely
    // file reference (tunable default set).
    const KNOWN_EXTENSIONS: &[&str] = &[
        "rs", "py", "js", "ts", "tsx", "jsx", "go", "c", "cpp", "h", "hpp", "java", "rb", "sh",
        "toml", "json", "yaml", "yml", "md", "txt", "html", "css", "sql", "lock", "mod", "lua",
        "zig", "php", "cs", "ex", "exs", "vue", "svelte",
    ];
    match token.rsplit_once('.') {
        Some((_, ext)) => KNOWN_EXTENSIONS.contains(&ext),
        None => false,
    }
}

/// Validates a speculative draft block from the fast tier.
///
/// A validator rejects malformed tool calls, invalid JSON, or low-confidence
/// output; the `String` error is the rejection reason. A confidence-aware
/// validator is a supported seam: [`MessageContentBlock`] carries no
/// confidence field, so this trait is where such a policy plugs in.
pub trait DraftValidator: Send + Sync {
    /// Validate one draft block, returning the rejection reason on failure.
    fn validate(&self, block: &MessageContentBlock) -> Result<(), String>;
}

/// Default [`DraftValidator`]: structural JSON checks only.
///
/// - `ToolCall`: `id` and `name` must be non-empty and `arguments` must parse
///   as JSON (string payloads are re-parsed; structured values are already
///   valid).
/// - `Text`: a loose check: text that happens to parse as JSON must be an
///   object; anything else passes as ordinary prose.
/// - All other blocks pass.
#[derive(Default)]
pub struct JsonToolDraftValidator;

impl DraftValidator for JsonToolDraftValidator {
    fn validate(&self, block: &MessageContentBlock) -> Result<(), String> {
        match block {
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                if id.is_empty() {
                    return Err("tool call id must be non-empty".into());
                }
                if name.is_empty() {
                    return Err("tool call name must be non-empty".into());
                }
                // `arguments` is already a `Value`; a string payload may still
                // carry raw, unparsed JSON text (some providers emit arguments
                // as strings), so require it to parse.
                if let serde_json::Value::String(raw) = arguments
                    && let Err(e) = serde_json::from_str::<serde_json::Value>(raw)
                {
                    return Err(format!("tool call arguments are not valid JSON: {e}"));
                }
                Ok(())
            }
            MessageContentBlock::Text(text) => {
                match serde_json::from_str::<serde_json::Value>(text) {
                    Ok(serde_json::Value::Object(_)) => Ok(()),
                    Ok(_) => Err("text block is valid JSON but not a JSON object".into()),
                    Err(_) => Ok(()),
                }
            }
            MessageContentBlock::Thinking { .. }
            | MessageContentBlock::ImageBase64 { .. }
            | MessageContentBlock::ToolResult { .. } => Ok(()),
        }
    }
}

/// Executes one turn against a tier of the fast/slow pair.
///
/// The trait is deliberately synchronous: execution is started eagerly and the
/// caller receives an already-running [`AgentResponseStream`]. The real
/// implementation spawns the underlying `generate_stream` call on the tokio
/// runtime and resolves it lazily (see [`PoolTurnExecutor`]); tests inject
/// canned streams. Tier names are `"fast"` and `"slow"`.
pub trait TurnExecutor: Send + Sync {
    /// The tier this executor serves: `"fast"` or `"slow"`.
    fn tier_name(&self) -> &'static str;

    /// Start one turn against this tier, returning its block stream.
    ///
    /// # Errors
    ///
    /// Returns a [`CucaError`] when the tier cannot start (e.g. the pooled
    /// client fails to build).
    fn execute(&self, request: UnifiedRequest) -> Result<AgentResponseStream, CucaError>;
}

/// Real [`TurnExecutor`] that runs a tier through a pooled [`CucaClient`].
///
/// The tier dispatches to the endpoint configured with
/// [`ModelOrchestrator::with_endpoint`]: an empty `base_url` leaves the pooled
/// client on the provider adapter's own default, and `api_key` is sent only
/// when set.
///
/// The pool clients are built without `with_orchestrator`, so their
/// `generate_stream` calls dispatch straight to the provider: this is the
/// recursion guard that keeps an orchestrator from re-entering itself through
/// its own executors.
pub struct PoolTurnExecutor {
    tier: &'static str,
    provider: ProviderEndpoint,
    model_id: String,
    base_url: String,
    api_key: Option<String>,
    pool: Arc<ClientPool>,
}

impl TurnExecutor for PoolTurnExecutor {
    fn tier_name(&self) -> &'static str {
        self.tier
    }

    fn execute(&self, request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
        let mut request = request;
        request.model = self.model_id.clone();
        let provider = self.provider.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let pool = Arc::clone(&self.pool);
        let (tx, rx) = oneshot::channel();
        // Spawn the underlying `generate_stream`; the returned stream resolves
        // the oneshot on its first poll. `tokio::spawn` requires a running
        // runtime: `execute_adaptive_turn` is async, so one is always active
        // here.
        tokio::spawn(async move {
            let outcome = match pool.get_or_create(&provider, &base_url, api_key.as_deref()) {
                Ok(client) => client.generate_stream(request).await,
                Err(e) => Err(e),
            };
            // A receiver dropped before its first poll makes the send fail;
            // that is harmless.
            let _ = tx.send(outcome);
        });
        Ok(Box::pin(SpawnedStream {
            receiver: Some(rx),
            inner: None,
        }))
    }
}

/// The fast and slow [`PoolTurnExecutor`]s for `config`, both bound to one
/// endpoint.
///
/// Shared by [`ModelOrchestrator::new`] and
/// [`ModelOrchestrator::with_endpoint`] so the two tiers can never drift onto
/// different endpoints.
fn pool_executors(
    config: &SwappableModelPair,
    pool: &Arc<ClientPool>,
    base_url: &str,
    api_key: Option<&str>,
) -> (Arc<dyn TurnExecutor>, Arc<dyn TurnExecutor>) {
    let fast = Arc::new(PoolTurnExecutor {
        tier: "fast",
        provider: config.fast_provider.clone(),
        model_id: config.fast_model_id.clone(),
        base_url: base_url.to_string(),
        api_key: api_key.map(str::to_owned),
        pool: Arc::clone(pool),
    });
    let slow = Arc::new(PoolTurnExecutor {
        tier: "slow",
        provider: config.slow_provider.clone(),
        model_id: config.slow_model_id.clone(),
        base_url: base_url.to_string(),
        api_key: api_key.map(str::to_owned),
        pool: Arc::clone(pool),
    });
    (fast, slow)
}

/// Lazily-resolved wrapper around a spawned tier turn.
///
/// The first poll resolves the `oneshot` produced by [`PoolTurnExecutor`];
/// afterwards polls forward to the inner stream. A failed spawn surfaces its
/// error once, then the stream ends.
struct SpawnedStream {
    receiver: Option<oneshot::Receiver<Result<AgentResponseStream, CucaError>>>,
    inner: Option<AgentResponseStream>,
}

impl Stream for SpawnedStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if let Some(receiver) = this.receiver.as_mut() {
            match Pin::new(receiver).poll(cx) {
                Poll::Ready(Ok(Ok(inner))) => {
                    this.receiver = None;
                    this.inner = Some(inner);
                }
                Poll::Ready(Ok(Err(e))) => {
                    // The spawned call failed: surface the error once and end.
                    this.receiver = None;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Err(_)) => {
                    // The spawned task ended without sending a result; treat it
                    // as a transport failure.
                    this.receiver = None;
                    return Poll::Ready(Some(Err(CucaError::Transport {
                        message: "orchestrator tier task ended without a result".into(),
                    })));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
        match this.inner.as_mut() {
            Some(inner) => inner.as_mut().poll_next(cx),
            // Terminal state after an error was already yielded.
            None => Poll::Ready(None),
        }
    }
}

/// Fast/slow orchestrator: routes by complexity, drafts speculatively on the
/// fast tier, and falls back to the slow tier on rejection or latency.
pub struct ModelOrchestrator {
    config: SwappableModelPair,
    client_pool: Arc<ClientPool>,
    evaluator: ComplexityEvaluator,
    validator: Arc<dyn DraftValidator>,
    fast_executor: Arc<dyn TurnExecutor>,
    slow_executor: Arc<dyn TurnExecutor>,
    base_url: String,
    api_key: Option<String>,
    #[cfg(feature = "plugin-session-log")]
    session_store: Option<Arc<dyn SessionStorePlugin>>,
    #[cfg(feature = "plugin-session-log")]
    session_id: Option<String>,
}

impl ModelOrchestrator {
    /// Build an orchestrator over the default validator, default evaluator,
    /// and real tier executors drawn from `pool` and `config`.
    ///
    /// Both tiers start on their provider adapters' default endpoints; see
    /// [`Self::with_endpoint`].
    pub fn new(config: SwappableModelPair, pool: Arc<ClientPool>) -> Self {
        let (fast, slow) = pool_executors(&config, &pool, "", None);
        Self {
            config,
            client_pool: pool,
            evaluator: ComplexityEvaluator::default(),
            validator: Arc::new(JsonToolDraftValidator),
            fast_executor: fast,
            slow_executor: slow,
            base_url: String::new(),
            api_key: None,
            #[cfg(feature = "plugin-session-log")]
            session_store: None,
            #[cfg(feature = "plugin-session-log")]
            session_id: None,
        }
    }

    /// Build an orchestrator with injected tier executors (test seam).
    ///
    /// The validator and evaluator stay at their defaults; no session store is
    /// attached. An injected executor owns its own dispatch, so the endpoint
    /// fields stay unset.
    pub fn with_executors(
        config: SwappableModelPair,
        pool: Arc<ClientPool>,
        fast: Arc<dyn TurnExecutor>,
        slow: Arc<dyn TurnExecutor>,
    ) -> Self {
        Self {
            config,
            client_pool: pool,
            evaluator: ComplexityEvaluator::default(),
            validator: Arc::new(JsonToolDraftValidator),
            fast_executor: fast,
            slow_executor: slow,
            base_url: String::new(),
            api_key: None,
            #[cfg(feature = "plugin-session-log")]
            session_store: None,
            #[cfg(feature = "plugin-session-log")]
            session_id: None,
        }
    }

    /// Point the pooled tier clients at `base_url`, with `api_key` as their
    /// credential when one is given.
    ///
    /// Both tiers share the endpoint. The default is an empty base URL and no
    /// key, which leaves each provider adapter on its own default endpoint.
    /// The call rebuilds the pool-backed tier executors, so it supersedes
    /// executors injected by [`Self::with_executors`].
    pub fn with_endpoint(mut self, base_url: impl Into<String>, api_key: Option<&str>) -> Self {
        self.base_url = base_url.into();
        self.api_key = api_key.map(str::to_owned);
        let (fast, slow) = pool_executors(
            &self.config,
            &self.client_pool,
            &self.base_url,
            self.api_key.as_deref(),
        );
        self.fast_executor = fast;
        self.slow_executor = slow;
        self
    }

    /// Attach a session store for [`SessionEvent::ModelSwap`] traceability
    /// (co-enabled with `plugin-session-log`).
    #[cfg(feature = "plugin-session-log")]
    pub fn with_session_store(
        mut self,
        store: Arc<dyn SessionStorePlugin>,
        session_id: impl Into<String>,
    ) -> Self {
        self.session_store = Some(store);
        self.session_id = Some(session_id.into());
        self
    }

    /// The shared client pool the tier executors draw from.
    pub fn client_pool(&self) -> &ClientPool {
        &self.client_pool
    }

    /// Run one turn through the three-stage pipeline and return the block
    /// stream.
    ///
    /// Stage 1 routes by complexity: [`Complexity::Slow`] requests skip the
    /// draft phase and go straight to the slow tier. Otherwise the fast tier
    /// produces a speculative draft (stage 2), and stage 3 runs inside the
    /// returned `OrchestratorStream`: every block is validated, rejected
    /// blocks trigger the fallback cascade, and the latency guard swaps to the
    /// slow tier once the fast tier exceeds
    /// [`SwappableModelPair::latency_threshold_ms`].
    ///
    /// # Errors
    ///
    /// Returns the executor error when a tier fails to start, or a
    /// [`CucaError::Provider`] rejection once the fallback budget is
    /// exhausted.
    pub async fn execute_adaptive_turn(
        &self,
        request: UnifiedRequest,
    ) -> Result<AgentResponseStream, CucaError> {
        // Stage 1: deterministic complexity routing. A slow request never
        // enters the draft phase. The fast tier would only waste its latency
        // budget.
        if self.evaluator.evaluate(&request) == Complexity::Slow {
            return self.slow_executor.execute(request);
        }
        // Stage 2: speculative draft. The working copy is what fallback
        // cascades mutate (rejected blocks are appended as tool results), so
        // the original request stays the fast tier's input.
        let working_request = request.clone();
        let fast_stream = self.fast_executor.execute(request)?;
        let started = Instant::now();
        Ok(Box::pin(OrchestratorStream {
            state: OrchestratorState::Drafting {
                inner: fast_stream,
                started,
                working_request,
                fallback_attempts_left: FALLBACK_ATTEMPTS,
            },
            config: self.config.clone(),
            validator: Arc::clone(&self.validator),
            slow_executor: Arc::clone(&self.slow_executor),
            #[cfg(feature = "plugin-session-log")]
            session_store: self.session_store.clone(),
            #[cfg(feature = "plugin-session-log")]
            session_id: self.session_id.clone(),
        }))
    }
}

// Maximum slow-tier fallback re-routes per turn. Each cascade
// re-invokes the slow tier with the accumulated rejection state appended to
// the request.
const FALLBACK_ATTEMPTS: u32 = 2;

/// State of the [`OrchestratorStream`] state machine.
enum OrchestratorState {
    /// The fast tier is still producing the speculative draft.
    Drafting {
        inner: AgentResponseStream,
        started: Instant,
        working_request: UnifiedRequest,
        fallback_attempts_left: u32,
    },
    /// A fallback or latency swap is serving from the slow tier.
    Slow {
        inner: AgentResponseStream,
        working_request: UnifiedRequest,
        fallback_attempts_left: u32,
    },
    /// The stream has ended (clean finish, or after surfacing an error).
    Done,
}

/// Stream wrapper implementing the draft-validation and fallback logic.
///
/// See the module docs for the pipeline stages, the latency swap boundary, and
/// the fallback cascade bounds.
struct OrchestratorStream {
    state: OrchestratorState,
    config: SwappableModelPair,
    validator: Arc<dyn DraftValidator>,
    slow_executor: Arc<dyn TurnExecutor>,
    #[cfg(feature = "plugin-session-log")]
    session_store: Option<Arc<dyn SessionStorePlugin>>,
    #[cfg(feature = "plugin-session-log")]
    session_id: Option<String>,
}

impl OrchestratorStream {
    /// Append a `ModelSwap` record for this turn when a session store is
    /// registered. Non-fatal: a failed append is logged, the swap itself stays
    /// observable in the stream.
    fn record_swap(&self, reason: &str) {
        #[cfg(feature = "plugin-session-log")]
        {
            if let (Some(store), Some(session_id)) = (&self.session_store, &self.session_id) {
                let record = SessionRecord::new(
                    session_id.clone(),
                    SessionEvent::ModelSwap {
                        from: self.config.fast_model_id.clone(),
                        to: self.config.slow_model_id.clone(),
                        reason: reason.to_string(),
                    },
                );
                if let Err(e) = store.append_log(session_id, &record) {
                    eprintln!("orchestrator: failed to append model-swap record: {e}");
                }
            }
        }
        #[cfg(not(feature = "plugin-session-log"))]
        let _ = reason;
    }
}

impl Stream for OrchestratorStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            // Take the state out so a transition can move its fields into a
            // new state without fighting the borrow checker; `Done` is the
            // placeholder while the new state is being computed.
            let state = std::mem::replace(&mut this.state, OrchestratorState::Done);
            match state {
                OrchestratorState::Done => return Poll::Ready(None),
                OrchestratorState::Drafting {
                    mut inner,
                    started,
                    mut working_request,
                    mut fallback_attempts_left,
                } => match inner.as_mut().poll_next(cx) {
                    Poll::Pending => {
                        // Latency guard: the fast tier is quiet. Once the
                        // threshold is exceeded the swap happens at this
                        // natural poll boundary; blocks already yielded stay
                        // delivered.
                        if started.elapsed().as_millis() as u64 > this.config.latency_threshold_ms {
                            this.record_swap("latency_threshold");
                            match this.slow_executor.execute(working_request.clone()) {
                                Ok(slow) => {
                                    this.state = OrchestratorState::Slow {
                                        inner: slow,
                                        working_request,
                                        fallback_attempts_left,
                                    };
                                    continue;
                                }
                                Err(e) => return Poll::Ready(Some(Err(e))),
                            }
                        }
                        this.state = OrchestratorState::Drafting {
                            inner,
                            started,
                            working_request,
                            fallback_attempts_left,
                        };
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(block))) => {
                        match this.validator.validate(&block) {
                            Ok(()) => {
                                this.state = OrchestratorState::Drafting {
                                    inner,
                                    started,
                                    working_request,
                                    fallback_attempts_left,
                                };
                                return Poll::Ready(Some(Ok(block)));
                            }
                            Err(reason) => {
                                if fallback_attempts_left > 0 && this.config.fallback_on_tool_error
                                {
                                    working_request
                                        .messages
                                        .push(rejection_message(&block, &reason));
                                    this.record_swap("fallback_validation");
                                    fallback_attempts_left -= 1;
                                    match this.slow_executor.execute(working_request.clone()) {
                                        Ok(slow) => {
                                            this.state = OrchestratorState::Slow {
                                                inner: slow,
                                                working_request,
                                                fallback_attempts_left,
                                            };
                                            continue;
                                        }
                                        Err(e) => return Poll::Ready(Some(Err(e))),
                                    }
                                } else {
                                    // No fallback configured, or the cascade
                                    // budget is exhausted: surface the last
                                    // rejection as a provider error.
                                    return Poll::Ready(Some(Err(CucaError::Provider {
                                        provider: this.config.fast_provider.clone(),
                                        message: reason,
                                    })));
                                }
                            }
                        }
                    }
                    Poll::Ready(Some(Err(e))) => {
                        this.state = OrchestratorState::Drafting {
                            inner,
                            started,
                            working_request,
                            fallback_attempts_left,
                        };
                        return Poll::Ready(Some(Err(e)));
                    }
                    // The fast tier finished cleanly: nothing left to do.
                    Poll::Ready(None) => return Poll::Ready(None),
                },
                OrchestratorState::Slow {
                    mut inner,
                    mut working_request,
                    mut fallback_attempts_left,
                } => match inner.as_mut().poll_next(cx) {
                    Poll::Pending => {
                        this.state = OrchestratorState::Slow {
                            inner,
                            working_request,
                            fallback_attempts_left,
                        };
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(block))) => match this.validator.validate(&block) {
                        Ok(()) => {
                            this.state = OrchestratorState::Slow {
                                inner,
                                working_request,
                                fallback_attempts_left,
                            };
                            return Poll::Ready(Some(Ok(block)));
                        }
                        Err(reason) => {
                            if fallback_attempts_left > 0 && this.config.fallback_on_tool_error {
                                working_request
                                    .messages
                                    .push(rejection_message(&block, &reason));
                                this.record_swap("fallback_validation");
                                fallback_attempts_left -= 1;
                                match this.slow_executor.execute(working_request.clone()) {
                                    Ok(slow) => {
                                        this.state = OrchestratorState::Slow {
                                            inner: slow,
                                            working_request,
                                            fallback_attempts_left,
                                        };
                                        continue;
                                    }
                                    Err(e) => return Poll::Ready(Some(Err(e))),
                                }
                            } else {
                                return Poll::Ready(Some(Err(CucaError::Provider {
                                    provider: this.config.slow_provider.clone(),
                                    message: reason,
                                })));
                            }
                        }
                    },
                    Poll::Ready(Some(Err(e))) => {
                        this.state = OrchestratorState::Slow {
                            inner,
                            working_request,
                            fallback_attempts_left,
                        };
                        return Poll::Ready(Some(Err(e)));
                    }
                    Poll::Ready(None) => return Poll::Ready(None),
                },
            }
        }
    }
}

/// Build the synthetic tool-result message that captures a rejected draft
/// block so the next tier observes what was refused and why.
///
/// A rejected `ToolCall` carries its id on the result; other rejected blocks
/// (e.g. non-object JSON text) have no call to answer, so the id is left
/// empty.
fn rejection_message(block: &MessageContentBlock, reason: &str) -> UnifiedMessage {
    let tool_call_id = match block {
        MessageContentBlock::ToolCall { id, .. } => Some(id.clone()),
        _ => None,
    };
    UnifiedMessage {
        role: MessageRole::Tool,
        content: vec![MessageContentBlock::ToolResult {
            tool_call_id: tool_call_id.clone().unwrap_or_default(),
            output: reason.to_string(),
        }],
        name: None,
        tool_call_id,
    }
}

#[cfg(all(test, feature = "service-speculative"))]
mod tests {
    use super::*;
    use crate::client::CucaClient;
    #[cfg(feature = "plugin-session-log")]
    use crate::error::PluginError;
    #[cfg(feature = "plugin-session-log")]
    use crate::plugin::{CucaPlugin, SessionStorePlugin};
    #[cfg(feature = "plugin-session-log")]
    use crate::session::{SessionEvent, SessionRecord};
    use crate::types::{MessageContentBlock, MessageRole};
    use tokio_stream::StreamExt;

    fn request_with(messages: Vec<UnifiedMessage>) -> UnifiedRequest {
        UnifiedRequest {
            model: "test-model".into(),
            provider: ProviderEndpoint::Custom(String::new()),
            messages,
            temperature: None,
            max_tokens: None,
            stream: true,
            thinking: None,
            tools: Vec::new(),
            prompt_cache: PromptCacheDirective::Disabled,
        }
    }

    fn text_message(role: MessageRole, text: &str) -> UnifiedMessage {
        UnifiedMessage {
            role,
            content: vec![MessageContentBlock::Text(text.to_string())],
            name: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn swappable_model_pair_serde_round_trip() {
        let pair = SwappableModelPair {
            fast_provider: ProviderEndpoint::OpenAi,
            fast_model_id: "gpt-fast".into(),
            slow_provider: ProviderEndpoint::Anthropic,
            slow_model_id: "claude-slow".into(),
            latency_threshold_ms: 500,
            fallback_on_tool_error: true,
        };
        let json = serde_json::to_value(&pair).expect("serialize should succeed");
        let back: SwappableModelPair =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(pair, back);
    }

    #[test]
    fn short_intent_classification_is_fast() {
        let evaluator = ComplexityEvaluator::default();
        let req = request_with(vec![text_message(
            MessageRole::User,
            "classify this query as urgent or routine",
        )]);
        assert_eq!(evaluator.evaluate(&req), Complexity::Fast);
    }

    #[test]
    fn large_input_is_slow() {
        let evaluator = ComplexityEvaluator::default();
        // 2_000 tokens needs ~8_000 chars at chars/4; 9_000 chars guarantees it.
        let big = "x".repeat(9_000);
        let req = request_with(vec![text_message(MessageRole::User, &big)]);
        assert_eq!(evaluator.evaluate(&req), Complexity::Slow);
    }

    #[test]
    fn tool_chain_is_slow() {
        let evaluator = ComplexityEvaluator::default();
        let call = UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "x"}),
            }],
            name: None,
            tool_call_id: None,
        };
        let result = UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "c1".into(),
                output: "result".into(),
            }],
            name: None,
            tool_call_id: Some("c1".into()),
        };
        // Two messages with tool content -> tool_depth = 2 >= 1 -> Slow.
        let req = request_with(vec![call, result]);
        assert_eq!(evaluator.evaluate(&req), Complexity::Slow);
    }

    #[test]
    fn multiple_file_references_are_slow() {
        let evaluator = ComplexityEvaluator::default();
        let text = "please review ./src/main.rs, ./src/lib.rs, and /etc/config.json";
        let req = request_with(vec![text_message(MessageRole::User, text)]);
        assert_eq!(evaluator.evaluate(&req), Complexity::Slow);
    }

    #[test]
    fn threshold_tuning_shifts_decision() {
        let evaluator = ComplexityEvaluator {
            slow_tool_call_depth: 3,
            ..ComplexityEvaluator::default()
        };
        let call = UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "x"}),
            }],
            name: None,
            tool_call_id: None,
        };
        let result = UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "c1".into(),
                output: "result".into(),
            }],
            name: None,
            tool_call_id: Some("c1".into()),
        };
        // Two tool messages but threshold raised to 3 -> Fast.
        let req = request_with(vec![call, result]);
        assert_eq!(evaluator.evaluate(&req), Complexity::Fast);
    }

    #[test]
    fn pool_reuses_same_arc_for_same_endpoint() {
        let pool = ClientPool::new();
        let a = pool
            .get_or_create(&ProviderEndpoint::OpenAi, "http://a", None)
            .expect("build should succeed");
        let b = pool
            .get_or_create(&ProviderEndpoint::OpenAi, "http://a", None)
            .expect("build should succeed");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn pool_separates_different_base_urls() {
        let pool = ClientPool::new();
        let a = pool
            .get_or_create(&ProviderEndpoint::OpenAi, "http://a", None)
            .expect("build should succeed");
        let b = pool
            .get_or_create(&ProviderEndpoint::OpenAi, "http://b", None)
            .expect("build should succeed");
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn pool_separates_different_providers() {
        let pool = ClientPool::new();
        let a = pool
            .get_or_create(&ProviderEndpoint::OpenAi, "http://a", None)
            .expect("build should succeed");
        let b = pool
            .get_or_create(&ProviderEndpoint::Anthropic, "http://a", None)
            .expect("build should succeed");
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(pool.len(), 2);
    }

    // ---- tier endpoint ------------------------------------------

    /// A closed loopback port on a `Custom` provider: a tier dispatch fails
    /// before any socket work, and the pooled client still records the
    /// endpoint it was built with.
    const TIER_BASE: &str = "http://127.0.0.1:1/v1";

    fn custom_pair(fast: &str, slow: &str) -> SwappableModelPair {
        SwappableModelPair {
            fast_provider: ProviderEndpoint::Custom(fast.into()),
            fast_model_id: "fast-tier-id".into(),
            slow_provider: ProviderEndpoint::Custom(slow.into()),
            slow_model_id: "slow-tier-id".into(),
            // High enough that a Pending poll never trips the latency swap.
            latency_threshold_ms: 60_000,
            fallback_on_tool_error: false,
        }
    }

    /// Routes fast: one short user message.
    fn tiny_request() -> UnifiedRequest {
        request_with(vec![text_message(MessageRole::User, "hi")])
    }

    /// Routes slow: past the evaluator's input-volume threshold.
    fn bulky_request() -> UnifiedRequest {
        request_with(vec![text_message(MessageRole::User, &"x".repeat(9_000))])
    }

    /// Run a turn to completion so the spawned tier dispatch has reached
    /// `ClientPool::get_or_create`. The dispatch itself always fails here; the
    /// pool entry it created is what the assertions read.
    async fn run_turn(orch: &ModelOrchestrator, request: UnifiedRequest) {
        let mut stream = orch
            .execute_adaptive_turn(request)
            .await
            .expect("the tier must start");
        while stream.next().await.is_some() {}
    }

    #[tokio::test]
    async fn both_tiers_pool_their_client_at_the_configured_base_url() {
        let pool = Arc::new(ClientPool::new());
        let orch =
            ModelOrchestrator::new(custom_pair("fast-endpoint", "slow-endpoint"), pool.clone())
                .with_endpoint(TIER_BASE, None);

        run_turn(&orch, tiny_request()).await;
        run_turn(&orch, bulky_request()).await;

        assert_eq!(pool.len(), 2, "one pooled client per tier provider");
        for tier in ["fast-endpoint", "slow-endpoint"] {
            let client = pool
                .get_or_create(&ProviderEndpoint::Custom(tier.into()), TIER_BASE, None)
                .expect("build should succeed");
            assert_eq!(client.base_url(), TIER_BASE);
        }
        assert_eq!(
            pool.len(),
            2,
            "probing the configured endpoint must hit the entries the tiers \
             created, not add two more"
        );
    }

    #[tokio::test]
    async fn the_configured_api_key_reaches_the_pooled_tier_client() {
        let pool = Arc::new(ClientPool::new());
        let orch = ModelOrchestrator::new(custom_pair("tier", "tier"), pool.clone())
            .with_endpoint(TIER_BASE, Some("tier-key"));

        run_turn(&orch, tiny_request()).await;

        // The probe passes no key, and a cache hit ignores the argument, so a
        // key here can only come from the tier executor that built the entry.
        let client = pool
            .get_or_create(&ProviderEndpoint::Custom("tier".into()), TIER_BASE, None)
            .expect("build should succeed");
        assert_eq!(client.api_key(), Some("tier-key"));
    }

    #[tokio::test]
    async fn an_unconfigured_orchestrator_leaves_tier_clients_on_the_provider_default() {
        let pool = Arc::new(ClientPool::new());
        let orch = ModelOrchestrator::new(custom_pair("tier", "tier"), pool.clone());

        run_turn(&orch, tiny_request()).await;

        let client = pool
            .get_or_create(&ProviderEndpoint::Custom("tier".into()), "", None)
            .expect("build should succeed");
        assert!(
            client.base_url().is_empty(),
            "an unconfigured tier pools at the empty base URL, which leaves the \
             provider adapter on its own default"
        );
        assert_eq!(pool.len(), 1);
    }

    // ---- draft validator ----------------------------------------

    #[test]
    fn draft_validator_accepts_valid_tool_call() {
        let validator = JsonToolDraftValidator;
        let block = MessageContentBlock::ToolCall {
            id: "c1".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({"q": "x"}),
        };
        assert_eq!(validator.validate(&block), Ok(()));
    }

    #[test]
    fn draft_validator_rejects_non_json_arguments() {
        let validator = JsonToolDraftValidator;
        // A string payload that is not itself parseable JSON is a malformed
        // tool call: the model emitted raw, broken JSON.
        let block = MessageContentBlock::ToolCall {
            id: "c1".into(),
            name: "lookup".into(),
            arguments: serde_json::Value::String("not json".into()),
        };
        let err = validator.validate(&block).expect_err("must reject");
        assert!(
            err.contains("not valid JSON"),
            "unexpected rejection: {err}"
        );
    }

    #[test]
    fn draft_validator_rejects_empty_id_and_name() {
        let validator = JsonToolDraftValidator;
        let no_id = MessageContentBlock::ToolCall {
            id: String::new(),
            name: "lookup".into(),
            arguments: serde_json::json!({}),
        };
        let err = validator
            .validate(&no_id)
            .expect_err("empty id must reject");
        assert!(err.contains("id"), "unexpected rejection: {err}");
        let no_name = MessageContentBlock::ToolCall {
            id: "c1".into(),
            name: String::new(),
            arguments: serde_json::json!({}),
        };
        let err = validator
            .validate(&no_name)
            .expect_err("empty name must reject");
        assert!(err.contains("name"), "unexpected rejection: {err}");
    }

    #[test]
    fn draft_validator_accepts_text_and_loose_json_check() {
        let validator = JsonToolDraftValidator;
        // Ordinary prose is not JSON and passes.
        assert_eq!(
            validator.validate(&MessageContentBlock::Text("hello world".into())),
            Ok(())
        );
        // JSON that is an object passes the loose check.
        assert_eq!(
            validator.validate(&MessageContentBlock::Text("{\"a\":1}".into())),
            Ok(())
        );
        // JSON that is not an object is rejected.
        let err = validator
            .validate(&MessageContentBlock::Text("[1,2]".into()))
            .expect_err("array text must reject");
        assert!(err.contains("object"), "unexpected rejection: {err}");
        // Tool results are never second-guessed by the draft validator.
        assert_eq!(
            validator.validate(&MessageContentBlock::ToolResult {
                tool_call_id: "c1".into(),
                output: "42".into(),
            }),
            Ok(())
        );
    }

    // ---- canned turn executors -----------------------------------

    /// Shared recording state for the two canned executors: which requests each
    /// tier received, in call order.
    #[derive(Default)]
    struct RecordingState {
        fast_calls: Mutex<Vec<UnifiedRequest>>,
        slow_calls: Mutex<Vec<UnifiedRequest>>,
    }

    /// A canned [`TurnExecutor`] that records the requests it receives and
    /// serves a fixed item list on every call.
    struct CannedExecutor {
        tier: &'static str,
        items: Vec<Result<MessageContentBlock, CucaError>>,
        state: Arc<RecordingState>,
    }

    impl CannedExecutor {
        fn new(
            tier: &'static str,
            items: Vec<Result<MessageContentBlock, CucaError>>,
            state: Arc<RecordingState>,
        ) -> Self {
            Self { tier, items, state }
        }
    }

    impl TurnExecutor for CannedExecutor {
        fn tier_name(&self) -> &'static str {
            self.tier
        }

        fn execute(&self, request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
            let calls = if self.tier == "fast" {
                &self.state.fast_calls
            } else {
                &self.state.slow_calls
            };
            calls.lock().expect("test lock").push(request);
            Ok(Box::pin(tokio_stream::iter(self.items.clone())))
        }
    }

    /// A stream that never yields: always `Pending`, wakes nobody. Pins the
    /// fast tier past the latency threshold.
    struct PendingForever;

    impl Stream for PendingForever {
        type Item = Result<MessageContentBlock, CucaError>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    /// A fast-tier executor whose stream pends forever.
    struct PendingExecutor {
        state: Arc<RecordingState>,
    }

    impl TurnExecutor for PendingExecutor {
        fn tier_name(&self) -> &'static str {
            "fast"
        }

        fn execute(&self, request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
            self.state
                .fast_calls
                .lock()
                .expect("test lock")
                .push(request);
            Ok(Box::pin(PendingForever))
        }
    }

    /// Mock session store that records every appended record (used by the
    /// session-log co-enabled swap test).
    #[cfg(feature = "plugin-session-log")]
    #[derive(Default)]
    struct RecordingStore {
        records: Mutex<Vec<SessionRecord>>,
    }

    #[cfg(feature = "plugin-session-log")]
    impl CucaPlugin for RecordingStore {
        fn name(&self) -> &'static str {
            "recording-store"
        }
    }

    #[cfg(feature = "plugin-session-log")]
    impl SessionStorePlugin for RecordingStore {
        fn append_log(&self, _session_id: &str, record: &SessionRecord) -> Result<(), PluginError> {
            self.records.lock().expect("test lock").push(record.clone());
            Ok(())
        }

        fn replay_session(&self, _session_id: &str) -> Result<Vec<SessionRecord>, PluginError> {
            Ok(vec![])
        }

        fn fork_session(&self, _session_id: &str, _point_id: &str) -> Result<String, PluginError> {
            Ok("forked-session".into())
        }
    }

    fn default_config() -> SwappableModelPair {
        SwappableModelPair {
            fast_provider: ProviderEndpoint::OpenAi,
            fast_model_id: "fast-model".into(),
            slow_provider: ProviderEndpoint::Anthropic,
            slow_model_id: "slow-model".into(),
            latency_threshold_ms: 60_000,
            fallback_on_tool_error: true,
        }
    }

    /// Build an orchestrator over two canned executors sharing one recording
    /// state; returns the state so tests can assert what each tier received.
    fn canned_orchestrator(
        config: SwappableModelPair,
        fast_items: Vec<Result<MessageContentBlock, CucaError>>,
        slow_items: Vec<Result<MessageContentBlock, CucaError>>,
    ) -> (ModelOrchestrator, Arc<RecordingState>) {
        let state = Arc::new(RecordingState::default());
        let fast = Arc::new(CannedExecutor::new("fast", fast_items, Arc::clone(&state)));
        let slow = Arc::new(CannedExecutor::new("slow", slow_items, Arc::clone(&state)));
        let orch =
            ModelOrchestrator::with_executors(config, Arc::new(ClientPool::new()), fast, slow);
        (orch, state)
    }

    // ---- complexity routing --------------------------------------

    #[tokio::test]
    async fn complexity_fast_uses_fast_tier_without_swap() {
        let (orch, state) = canned_orchestrator(
            default_config(),
            vec![Ok(MessageContentBlock::Text("fast-answer".into()))],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        let req = request_with(vec![text_message(MessageRole::User, "classify this")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("fast path must build a stream");
        let block = stream.next().await;
        assert!(matches!(
            block,
            Some(Ok(MessageContentBlock::Text(t))) if t == "fast-answer"
        ));
        assert_eq!(state.fast_calls.lock().expect("test lock").len(), 1);
        assert!(state.slow_calls.lock().expect("test lock").is_empty());
    }

    #[tokio::test]
    async fn complexity_slow_routes_directly_to_slow_tier() {
        let (orch, state) = canned_orchestrator(
            default_config(),
            vec![Ok(MessageContentBlock::Text("fast-answer".into()))],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        // One assistant message carrying a tool call -> tool depth 1 >= 1.
        let req = request_with(vec![UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "x"}),
            }],
            name: None,
            tool_call_id: None,
        }]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("slow path must build a stream");
        let block = stream.next().await;
        assert!(matches!(
            block,
            Some(Ok(MessageContentBlock::Text(t))) if t == "slow-answer"
        ));
        assert!(state.fast_calls.lock().expect("test lock").is_empty());
        assert_eq!(state.slow_calls.lock().expect("test lock").len(), 1);
    }

    // ---- fallback cascade ----------------------------------------

    #[tokio::test]
    async fn rejected_draft_falls_back_to_slow_with_error_context() {
        let (orch, state) = canned_orchestrator(
            default_config(),
            vec![Ok(MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "frob".into(),
                arguments: serde_json::Value::String("not json".into()),
            })],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        let req = request_with(vec![text_message(MessageRole::User, "do the thing")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("fallback path must build a stream");
        let block = stream.next().await;
        assert!(matches!(
            block,
            Some(Ok(MessageContentBlock::Text(t))) if t == "slow-answer"
        ));
        // The slow tier must have received the original request plus a
        // synthetic ToolResult carrying the validation error.
        let slow_reqs = state.slow_calls.lock().expect("test lock");
        assert_eq!(slow_reqs.len(), 1);
        let last = slow_reqs[0]
            .messages
            .last()
            .expect("rejection must be appended as the last message");
        assert_eq!(last.role, MessageRole::Tool);
        match &last.content[0] {
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => {
                assert_eq!(tool_call_id, "c1");
                assert!(
                    output.contains("not valid JSON"),
                    "unexpected rejection text: {output}"
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        // The fast tier ran the untouched original request.
        let fast_reqs = state.fast_calls.lock().expect("test lock");
        assert_eq!(fast_reqs.len(), 1);
        assert_eq!(fast_reqs[0].messages.len(), 1);
    }

    #[tokio::test]
    async fn rejected_draft_without_fallback_surfaces_provider_error() {
        let config = SwappableModelPair {
            fallback_on_tool_error: false,
            ..default_config()
        };
        let (orch, state) = canned_orchestrator(
            config,
            vec![Ok(MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "frob".into(),
                arguments: serde_json::Value::String("not json".into()),
            })],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        let req = request_with(vec![text_message(MessageRole::User, "do the thing")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("fast path must build a stream");
        let item = stream.next().await;
        match item {
            Some(Err(CucaError::Provider { provider, message })) => {
                assert_eq!(provider, ProviderEndpoint::OpenAi);
                assert!(
                    message.contains("not valid JSON"),
                    "unexpected rejection: {message}"
                );
            }
            other => panic!("expected provider error, got {other:?}"),
        }
        assert!(stream.next().await.is_none());
        // No fallback happened: the slow tier was never invoked.
        assert!(state.slow_calls.lock().expect("test lock").is_empty());
    }

    #[tokio::test]
    async fn fallback_cascade_exhausts_after_budget() {
        let malformed = Ok(MessageContentBlock::ToolCall {
            id: "c1".into(),
            name: "frob".into(),
            arguments: serde_json::Value::String("not json".into()),
        });
        let (orch, state) =
            canned_orchestrator(default_config(), vec![malformed.clone()], vec![malformed]);
        let req = request_with(vec![text_message(MessageRole::User, "do the thing")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("must build a stream");
        // Fast rejects -> slow #1 rejects -> slow #2 rejects -> budget gone.
        let item = stream.next().await;
        match item {
            Some(Err(CucaError::Provider { provider, message })) => {
                assert_eq!(provider, ProviderEndpoint::Anthropic);
                assert!(
                    message.contains("not valid JSON"),
                    "unexpected rejection: {message}"
                );
            }
            other => panic!("expected provider error, got {other:?}"),
        }
        assert!(stream.next().await.is_none());
        assert_eq!(state.fast_calls.lock().expect("test lock").len(), 1);
        // Two cascades: the slow tier was re-invoked twice.
        assert_eq!(state.slow_calls.lock().expect("test lock").len(), 2);
    }

    // ---- latency guard -------------------------------------------

    #[tokio::test]
    async fn latency_guard_swaps_to_slow_when_fast_pends_past_threshold() {
        let config = SwappableModelPair {
            latency_threshold_ms: 0,
            ..default_config()
        };
        let state = Arc::new(RecordingState::default());
        let slow = Arc::new(CannedExecutor::new(
            "slow",
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
            Arc::clone(&state),
        ));
        let orch = ModelOrchestrator::with_executors(
            config,
            Arc::new(ClientPool::new()),
            Arc::new(PendingExecutor {
                state: Arc::clone(&state),
            }),
            slow,
        );
        let req = request_with(vec![text_message(MessageRole::User, "hi")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("must build a stream");
        // Poll with a timeout: the fast tier pends forever, so eventually the
        // latency guard must swap the turn to the slow tier.
        let mut saw_slow = false;
        for _ in 0..100 {
            match tokio::time::timeout(std::time::Duration::from_millis(5), stream.next()).await {
                Ok(Some(Ok(MessageContentBlock::Text(t)))) if t == "slow-answer" => {
                    saw_slow = true;
                    break;
                }
                // Timeout (still drafting) or some other item: keep polling.
                Ok(_) => {}
                Err(_elapsed) => {}
            }
        }
        assert!(
            saw_slow,
            "latency guard must swap the stream to the slow tier"
        );
        assert_eq!(state.fast_calls.lock().expect("test lock").len(), 1);
        assert_eq!(state.slow_calls.lock().expect("test lock").len(), 1);
    }

    // ---- stream sanity + client wiring ---------------------------

    #[tokio::test]
    async fn orchestrator_stream_yields_blocks_then_ends() {
        let (orch, _state) = canned_orchestrator(
            default_config(),
            vec![
                Ok(MessageContentBlock::Text("first".into())),
                Ok(MessageContentBlock::Text("second".into())),
            ],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        let req = request_with(vec![text_message(MessageRole::User, "hi")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("must build a stream");
        let first = stream.next().await;
        assert!(matches!(
            first,
            Some(Ok(MessageContentBlock::Text(t))) if t == "first"
        ));
        let second = stream.next().await;
        assert!(matches!(
            second,
            Some(Ok(MessageContentBlock::Text(t))) if t == "second"
        ));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn client_wiring_routes_through_orchestrator() {
        let (orch, state) = canned_orchestrator(
            default_config(),
            vec![Ok(MessageContentBlock::Text("fast-answer".into()))],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        let client = CucaClient::builder()
            .with_provider(ProviderEndpoint::OpenAi)
            .with_orchestrator(orch)
            .build()
            .expect("provider set, build must succeed");
        assert!(client.orchestrator().is_some());
        let mut stream = client
            .generate_stream(UnifiedRequest::new("gpt-fast").add_user_message("hi"))
            .await
            .expect("orchestrator path must build a stream");
        let block = stream.next().await;
        assert!(matches!(
            block,
            Some(Ok(MessageContentBlock::Text(t))) if t == "fast-answer"
        ));
        assert_eq!(state.fast_calls.lock().expect("test lock").len(), 1);
        assert!(state.slow_calls.lock().expect("test lock").is_empty());
    }

    #[cfg(feature = "plugin-session-log")]
    #[tokio::test]
    async fn model_swap_is_recorded_in_session_store() {
        let (orch, _state) = canned_orchestrator(
            default_config(),
            vec![Ok(MessageContentBlock::ToolCall {
                id: "c1".into(),
                name: "frob".into(),
                arguments: serde_json::Value::String("not json".into()),
            })],
            vec![Ok(MessageContentBlock::Text("slow-answer".into()))],
        );
        let store = Arc::new(RecordingStore::default());
        let dyn_store: Arc<dyn SessionStorePlugin> = store.clone();
        let orch = orch.with_session_store(dyn_store, "sess-019");
        let req = request_with(vec![text_message(MessageRole::User, "do the thing")]);
        let mut stream = orch
            .execute_adaptive_turn(req)
            .await
            .expect("must build a stream");
        let block = stream.next().await;
        assert!(matches!(
            block,
            Some(Ok(MessageContentBlock::Text(t))) if t == "slow-answer"
        ));
        let records = store.records.lock().expect("test lock");
        assert_eq!(records.len(), 1, "one swap must be recorded");
        match &records[0].event {
            SessionEvent::ModelSwap { from, to, reason } => {
                assert_eq!(from, "fast-model");
                assert_eq!(to, "slow-model");
                assert_eq!(reason, "fallback_validation");
            }
            other => panic!("expected ModelSwap, got {other:?}"),
        }
    }
}
