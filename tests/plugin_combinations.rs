//! Cross-plugin combination tests: behavior that exists only when two plugin
//! features are co-enabled.
//!
//! One file rather than one test per owning plugin file. Seven of the eleven
//! surfaces below (`memory + prompt-cache`, `speculative + prompt-cache`,
//! `memory + prompt-cache` export, `cost + prompt-cache`, `cost + memory`,
//! `cost + telemetry`, `redaction + prompt-cache`) are *core-mediated*: they
//! have no derived plugin to belong to, and AGENTS.md deliberately puts
//! two-plugin workflows in core
//! (`CucaExport::from_live`, `OtelCostObserver`). Keeping the tests together
//! mirrors that and gives one place to audit cross-plugin coupling; each
//! module is gated on exactly the features its surface needs, so no test
//! relies on a peer being co-enabled by accident.
//!
//! Only two tests here need the live server. Dispatch counts, cache-entry
//! counts, hook-invocation counts, and session records cannot be asserted
//! reliably against a real model, so the rest run against
//! `common::spawn_counting_sse_server` or fully in process.
//!
//! Surfaces covered:
//! 1. `entity-extraction → memory` (hard feature dep): extraction delta →
//!    `merge_graph` → memory's graph-context injection reaches the next prompt.
//! 2. `memory + prompt-cache` (core-mediated hook order): all `on_request`
//!    hooks run before the cache key is digested, so memory's injections are
//!    inside the key (`src/plugins/prompt_cache.rs:5-10`, `src/client.rs`
//!    ~513-529).
//! 3. `speculative + session-log`: `ModelOrchestrator::with_session_store`
//!    records `SessionEvent::ModelSwap` on swap.
//! 4. `speculative + prompt-cache` (client-mediated): a configured cache wraps
//!    the orchestrator stream in the standard instrumentation, which is the
//!    only way terminal hooks run over an orchestrator turn
//!    (`src/client.rs` ~576-600).
//! 5. `memory + prompt-cache` export coordinator: `CucaExport::from_live`.
//! 6. `cost + prompt-cache` (client-mediated): a local cache hit skips provider
//!    dispatch but still runs every `on_response_complete` hook, so the ledger
//!    keeps charging and reads as gross, pre-cache spend.
//! 7. `cost + memory` (hook order is observable in the estimate): memory's
//!    graph injection is inside the cost estimate only when memory is
//!    registered first. Neither plugin requires a position; the number differs.
//! 8. `cost + telemetry` (core-mediated bridge): `OtelCostObserver` records
//!    every `CostUsage` reading to the same meter provider
//!    `OpenTelemetryPlugin` reports on, so one export batch carries both the
//!    cost gauges and the request counter.
//! 9. `rate-limit + prompt-cache` (caller-mediated): the caller acquires its
//!    permit before `generate_stream` can short-circuit on the cache, so a
//!    cache hit still spends a limiter token.
//! 10. `redaction + session-log` (hook order is observable in the trajectory):
//!     the store records whatever earlier `on_request` hooks made of the
//!     request, so registering redaction first persists scrubbed content and
//!     registering it after persists the raw value. Neither order is required.
//! 11. `redaction + prompt-cache` (core-mediated hook order): the key is
//!     digested from the post-hook request, so enabling redaction changes every
//!     key, and two requests differing only inside a redacted secret collapse
//!     onto one entry.
#![cfg(feature = "provider-llamacpp")]

mod common;

// ---------------------------------------------------------------------------
// Surface 1: entity-extraction → memory
// ---------------------------------------------------------------------------

/// `plugin-entity-extraction` enables `plugin-memory` in `Cargo.toml`, so this
/// one feature is exactly what the surface needs.
#[cfg(feature = "plugin-entity-extraction")]
mod extraction_into_memory_context {
    use std::sync::{Arc, Mutex};

    use crate::common;
    use crate::common::extraction::{
        LiveExtractionModel, SOURCE, extraction_plugin, pair_candidate,
    };
    use cuca::plugin::CucaPlugin;
    use cuca::types::MessageRole;
    use cuca::{
        EntityExtractionCandidate, GraphContextConfig, MemoryConfig, MemoryPlugin, MergePolicy,
        PluginError, UnifiedRequest,
    };

    /// The marker `MemoryPlugin` prefixes its injected graph message with
    /// (`src/plugins/memory/graph.rs`); not root-exported, so it is pinned here
    /// on purpose: if it changes, this test says so.
    const GRAPH_RENDER_MARKER: &str = "CUCA graph memory:";

    /// Records the request as it exists *after* every earlier `on_request`
    /// hook has run. Registered after `MemoryPlugin` so it observes the
    /// injection, which is what hook-registration order buys the caller.
    #[derive(Default)]
    struct RequestCapture {
        requests: Mutex<Vec<UnifiedRequest>>,
    }

    impl CucaPlugin for RequestCapture {
        fn name(&self) -> &'static str {
            "request-capture"
        }

        fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
            self.requests
                .lock()
                .expect("capture lock must not be poisoned")
                .push(req.clone());
            Ok(())
        }
    }

    fn memory_with_graph_context() -> MemoryPlugin {
        MemoryPlugin::new(MemoryConfig {
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        })
        .expect("memory plugin must build")
    }

    /// The full loop neither per-plugin file owns: a live extraction becomes a
    /// memory graph, and the memory graph becomes prompt context on the next
    /// request.
    #[tokio::test]
    async fn live_extraction_reaches_the_next_prompt_through_graph_context() {
        if let Err(reason) = common::require_live_server() {
            eprintln!("SKIP: llama.cpp not reachable: {reason}");
            return;
        }
        let model_id = common::live_model();
        let model = LiveExtractionModel::new(model_id.clone());
        let extractor = extraction_plugin();

        let report = match extractor.extract(SOURCE, &model).await {
            Ok(report) => report,
            Err(error) if model.produced_no_candidate() => {
                eprintln!(
                    "SKIP: the served model produced no parseable extraction in {} attempts \
                     ({error}); raw replies:\n{}",
                    model.attempts(),
                    model.diagnostics()
                );
                return;
            }
            Err(error) => panic!(
                "the plugin rejected an adapter-built, schema-conformant candidate: {error:?}\n\
                 candidate: {:?}",
                model.candidate()
            ),
        };
        assert!(
            report.nodes_accepted > 0,
            "a produced candidate must yield at least one node"
        );
        let extracted = report.delta.snapshot();

        let memory = Arc::new(memory_with_graph_context());
        memory
            .merge_graph(report.delta, MergePolicy::Keep)
            .expect("merge must not fail");

        let capture = Arc::new(RequestCapture::default());
        let client = common::llamacpp_builder(common::base_url())
            .register_plugin(Arc::clone(&memory) as Arc<dyn CucaPlugin>)
            .register_plugin(Arc::clone(&capture) as Arc<dyn CucaPlugin>)
            .build()
            .expect("client build must succeed");

        let request = UnifiedRequest::new(model_id)
            .add_system_message("Answer only from the CUCA graph memory context.")
            .add_user_message(
                "Which company does Ada Lovelace work at? Answer with the company name only.",
            )
            .set_max_tokens(512);
        let blocks = common::drain_timeout(
            client
                .generate_stream(request)
                .await
                .expect("generate_stream must start"),
            120,
        )
        .await;

        // Contract: the extracted graph is in the outgoing prompt, as a system
        // message placed right after the first System message.
        let captured = capture
            .requests
            .lock()
            .expect("capture lock must not be poisoned");
        assert_eq!(captured.len(), 1, "exactly one request was sent");
        let messages = &captured[0].messages;
        let injected = common::text_of(&messages[1].content);
        assert!(
            injected.starts_with(GRAPH_RENDER_MARKER),
            "the graph message must sit right after the first System message, got: {injected}"
        );
        assert_eq!(
            messages[1].role,
            MessageRole::System,
            "graph context is injected as a System message"
        );
        for node in &extracted.nodes {
            assert!(
                injected.contains(&format!("node {}:", node.id)),
                "the injected context must carry extracted node {:?}, got:\n{injected}",
                node.id
            );
        }
        // The turn itself must have been served: `drain_timeout` already panics
        // on any stream error or a stall, so a non-empty block sequence means
        // the graph-carrying request was accepted and streamed back. Whether a
        // small model spends its budget on `Thinking` before answering is model
        // quality, so the answer text is reported, never asserted.
        assert!(
            !blocks.is_empty(),
            "the graph-carrying request must be served and stream at least one block"
        );
        eprintln!(
            "graph context carried {} node(s); model answered: {:?}",
            extracted.nodes.len(),
            common::text_of(&blocks)
        );
    }

    /// An empty extraction degrades to nothing: no delta, no graph, and no
    /// marker message injected, instead of an empty render in every prompt.
    #[test]
    fn empty_extraction_leaves_no_graph_context() {
        let report = extraction_plugin()
            .validate_candidate(EntityExtractionCandidate {
                entities: Vec::new(),
                relationships: Vec::new(),
            })
            .expect("an empty candidate is valid");
        assert_eq!(report.nodes_accepted, 0);
        assert_eq!(report.relationships_accepted, 0);

        let memory = memory_with_graph_context();
        memory
            .merge_graph(report.delta, MergePolicy::Keep)
            .expect("merging an empty delta must not fail");
        assert!(
            memory
                .snapshot()
                .expect("graph lock must not be poisoned")
                .nodes
                .is_empty()
        );

        let mut request = UnifiedRequest::new("combo-model")
            .add_system_message("primary instruction")
            .add_user_message("hi");
        memory
            .on_request(&mut request)
            .expect("on_request must not fail");
        assert_eq!(
            request.messages.len(),
            2,
            "an empty graph injects no message, got {:?}",
            request.messages
        );

        // A non-empty extraction through the same seam does inject, so the
        // absence above is emptiness, not a disabled config.
        memory
            .merge_graph(
                extraction_plugin()
                    .validate_candidate(pair_candidate("Ada", "Analytical Engines"))
                    .expect("candidate must be accepted")
                    .delta,
                MergePolicy::Keep,
            )
            .expect("merge must not fail");
        memory
            .on_request(&mut request)
            .expect("on_request must not fail");
        assert_eq!(request.messages.len(), 3);
        assert!(
            common::text_of(&request.messages[1].content).starts_with(GRAPH_RENDER_MARKER),
            "a non-empty extracted graph must inject its render"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 2: memory + prompt-cache (hook order is observable in the cache key)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-memory", feature = "plugin-prompt-cache"))]
mod memory_changes_cache_keys {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::common;
    use cuca::plugin::CucaPlugin;
    use cuca::{
        CucaClient, GraphContextConfig, GraphNode, MemoryConfig, MemoryGraph, MemoryPlugin,
        MergePolicy, PromptCache, PromptCacheConfig, UnifiedRequest,
    };

    fn cache() -> Arc<PromptCache> {
        Arc::new(
            PromptCache::new(
                PromptCacheConfig::new(16, Duration::from_secs(60)).expect("config must build"),
            )
            .expect("cache must build"),
        )
    }

    fn node(id: &str) -> GraphNode {
        GraphNode {
            id: id.into(),
            labels: vec!["person".into()],
            properties: serde_json::Map::new(),
        }
    }

    fn graph_with(id: &str) -> MemoryGraph {
        let mut graph = MemoryGraph::new();
        graph.upsert_node(node(id));
        graph
    }

    /// Memory with graph injection enabled and one node in its graph.
    fn memory_with_graph_context() -> Arc<MemoryPlugin> {
        let plugin = MemoryPlugin::new(MemoryConfig {
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        })
        .expect("memory plugin must build");
        plugin
            .merge_graph(graph_with("alice"), MergePolicy::Keep)
            .expect("seed merge must not fail");
        Arc::new(plugin)
    }

    fn client_at(
        addr: &str,
        cache: &Arc<PromptCache>,
        plugins: Vec<Arc<dyn CucaPlugin>>,
    ) -> CucaClient {
        let mut builder =
            common::llamacpp_builder(addr.to_string()).with_prompt_cache_service(Arc::clone(cache));
        for plugin in plugins {
            builder = builder.register_plugin(plugin);
        }
        builder.build().expect("client build must succeed")
    }

    fn request() -> UnifiedRequest {
        UnifiedRequest::new("combo-model")
            .add_system_message("primary instruction")
            .add_user_message("hi")
    }

    async fn run(client: &CucaClient) -> Vec<cuca::types::MessageContentBlock> {
        common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("generate_stream must start"),
            10,
        )
        .await
    }

    /// The injected graph render is part of the digest: identical logical
    /// requests hit while the graph is unchanged and miss once it changes, and
    /// restoring the graph restores the original key.
    #[tokio::test]
    async fn memory_graph_injection_is_inside_the_cache_key() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let cache = cache();
        let memory = memory_with_graph_context();
        let original = memory.snapshot().expect("graph lock must not be poisoned");
        let client = client_at(
            &format!("http://{addr}/v1"),
            &cache,
            vec![Arc::clone(&memory) as Arc<dyn CucaPlugin>],
        );

        run(&client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            1,
            "the first request must miss and dispatch"
        );

        run(&client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            1,
            "memory's injection is deterministic and idempotent, so an unchanged \
             graph must still hit"
        );

        memory
            .merge_graph(graph_with("carol"), MergePolicy::Keep)
            .expect("merge must not fail");
        run(&client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "a changed graph changes the injected render, hence the digest"
        );

        memory
            .replace_snapshot(original)
            .expect("restore must succeed");
        run(&client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "the digest is a pure function of the injected state, not of call order"
        );
    }

    /// Co-registration changes cache routing: two clients sharing one cache do
    /// not share an entry for the same logical request when only one of them
    /// has memory registered.
    #[tokio::test]
    async fn an_unregistered_memory_plugin_yields_a_different_cache_entry() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let addr = format!("http://{addr}/v1");
        let cache = cache();
        let memory = memory_with_graph_context();

        let with_memory = client_at(
            &addr,
            &cache,
            vec![Arc::clone(&memory) as Arc<dyn CucaPlugin>],
        );
        let without_memory = client_at(&addr, &cache, Vec::new());

        run(&with_memory).await;
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);

        run(&without_memory).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "the key is digested from the post-hook request, so an uninjected \
             request is a different entry"
        );

        run(&with_memory).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "the injected variant must still hit its own entry"
        );
    }

    /// The near-limit warning injection is inside the key too, not just the
    /// graph render.
    #[tokio::test]
    async fn near_limit_warning_injection_also_changes_the_key() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let addr = format!("http://{addr}/v1");
        let cache = cache();

        // A 100-token window with compression disabled: a one-line request is
        // well past 5% of the window, so the warning fires, and nothing else
        // mutates the messages.
        let window_config = |warn_fraction: Option<f32>| MemoryConfig {
            context_window_tokens: 100,
            max_fraction: None,
            warn_fraction,
            ..Default::default()
        };
        let warning_memory: Arc<dyn CucaPlugin> = Arc::new(
            MemoryPlugin::new(window_config(Some(0.05))).expect("memory plugin must build"),
        );
        let quiet_memory: Arc<dyn CucaPlugin> =
            Arc::new(MemoryPlugin::new(window_config(None)).expect("memory plugin must build"));

        let warning_client = client_at(&addr, &cache, vec![Arc::clone(&warning_memory)]);
        let quiet_client = client_at(&addr, &cache, vec![quiet_memory]);

        run(&warning_client).await;
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);

        run(&quiet_client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "the injected warning message must be part of the digest"
        );

        run(&warning_client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "the warning injection is deterministic, so its own entry still hits"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 3: speculative + session-log
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-speculative", feature = "plugin-session-log"))]
mod speculative_records_model_swaps {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use crate::common;
    use cuca::plugin::SessionStorePlugin;
    use cuca::request::AgentResponseStream;
    use cuca::types::{MessageContentBlock, ProviderEndpoint};
    use cuca::{
        ClientPool, CucaClient, CucaError, ModelOrchestrator, SessionEvent, SessionLogPlugin,
        SwappableModelPair, TurnExecutor, UnifiedRequest,
    };
    use tokio_stream::{Stream, StreamExt};

    /// A tier that answers immediately with canned blocks.
    struct CannedExecutor {
        tier: &'static str,
        blocks: Vec<MessageContentBlock>,
        calls: Arc<AtomicUsize>,
    }

    impl TurnExecutor for CannedExecutor {
        fn tier_name(&self) -> &'static str {
            self.tier
        }

        fn execute(&self, _request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let items: Vec<Result<MessageContentBlock, CucaError>> =
                self.blocks.iter().cloned().map(Ok).collect();
            Ok(Box::pin(tokio_stream::iter(items)))
        }
    }

    /// A fast tier that never answers.
    ///
    /// Every poll returns `Pending` *and* schedules a wake 2ms out. The
    /// scheduled wake is essential: the latency guard only runs in the
    /// `Poll::Pending` arm, and a stream that never wakes its task would never
    /// be polled a second time, so the guard could never observe elapsed time.
    struct QuietExecutor {
        calls: Arc<AtomicUsize>,
    }

    struct QuietStream;

    impl Stream for QuietStream {
        type Item = Result<MessageContentBlock, CucaError>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let waker = cx.waker().clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(2)).await;
                waker.wake();
            });
            Poll::Pending
        }
    }

    impl TurnExecutor for QuietExecutor {
        fn tier_name(&self) -> &'static str {
            "fast"
        }

        fn execute(&self, _request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::pin(QuietStream))
        }
    }

    /// A tier backed by a real `CucaClient`, counting its calls.
    ///
    /// `TurnExecutor::execute` is synchronous, so the dispatch is spawned and
    /// the returned stream resolves it on first poll: the same shape as the
    /// crate's own `PoolTurnExecutor`, plus the call counter this suite
    /// asserts a single slow-tier swap on.
    struct ClientExecutor {
        tier: &'static str,
        client: Arc<CucaClient>,
        model: String,
        calls: Arc<AtomicUsize>,
    }

    impl TurnExecutor for ClientExecutor {
        fn tier_name(&self) -> &'static str {
            self.tier
        }

        fn execute(&self, request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut request = request;
            request.model = self.model.clone();
            let client = Arc::clone(&self.client);
            let (tx, rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = tx.send(client.generate_stream(request).await);
            });
            Ok(Box::pin(LazyStream {
                receiver: Some(rx),
                inner: None,
            }))
        }
    }

    struct LazyStream {
        receiver: Option<tokio::sync::oneshot::Receiver<Result<AgentResponseStream, CucaError>>>,
        inner: Option<AgentResponseStream>,
    }

    impl Stream for LazyStream {
        type Item = Result<MessageContentBlock, CucaError>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            if let Some(receiver) = this.receiver.as_mut() {
                match Pin::new(receiver).poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(Ok(stream))) => {
                        this.receiver = None;
                        this.inner = Some(stream);
                    }
                    Poll::Ready(Ok(Err(error))) => {
                        this.receiver = None;
                        return Poll::Ready(Some(Err(error)));
                    }
                    Poll::Ready(Err(_)) => {
                        this.receiver = None;
                        return Poll::Ready(None);
                    }
                }
            }
            match this.inner.as_mut() {
                Some(stream) => stream.as_mut().poll_next(cx),
                None => Poll::Ready(None),
            }
        }
    }

    /// A draft block `JsonToolDraftValidator` rejects (empty tool call id).
    fn invalid_draft() -> MessageContentBlock {
        MessageContentBlock::ToolCall {
            id: String::new(),
            name: "noop".into(),
            arguments: serde_json::json!({}),
        }
    }

    fn pair(slow_model: &str, latency_threshold_ms: u64, fallback: bool) -> SwappableModelPair {
        SwappableModelPair {
            fast_provider: ProviderEndpoint::LlamaCpp,
            fast_model_id: "fast-tier-id".into(),
            slow_provider: ProviderEndpoint::LlamaCpp,
            slow_model_id: slow_model.into(),
            latency_threshold_ms,
            fallback_on_tool_error: fallback,
        }
    }

    fn turn(model: &str) -> UnifiedRequest {
        UnifiedRequest::new(model)
            .add_system_message("You are concise.")
            .add_user_message("Reply with the single word: ok")
            .set_max_tokens(128)
    }

    async fn drain(mut stream: AgentResponseStream) -> Vec<MessageContentBlock> {
        let mut blocks = Vec::new();
        tokio::time::timeout(Duration::from_secs(120), async {
            while let Some(item) = stream.next().await {
                blocks.push(item.unwrap_or_else(|e| panic!("stream item must be Ok: {e}")));
            }
        })
        .await
        .expect("orchestrated turn must finish within 120s");
        blocks
    }

    /// A rejected draft swaps to the live slow tier, records exactly one
    /// `ModelSwap`, and never leaks the rejected block to the consumer.
    #[tokio::test]
    async fn fallback_swap_appends_one_model_swap_record_and_serves_from_the_slow_tier() {
        if let Err(reason) = common::require_live_server() {
            eprintln!("SKIP: llama.cpp not reachable: {reason}");
            return;
        }
        let model = common::live_model();
        let fast_calls = Arc::new(AtomicUsize::new(0));
        let slow_calls = Arc::new(AtomicUsize::new(0));
        let store = Arc::new(SessionLogPlugin::new_in_memory());

        let orchestrator = ModelOrchestrator::with_executors(
            pair(&model, 60_000, true),
            Arc::new(ClientPool::default()),
            Arc::new(CannedExecutor {
                tier: "fast",
                blocks: vec![invalid_draft()],
                calls: Arc::clone(&fast_calls),
            }),
            Arc::new(ClientExecutor {
                tier: "slow",
                client: Arc::new(common::client_with_plugins(Vec::new())),
                model: model.clone(),
                calls: Arc::clone(&slow_calls),
            }),
        )
        .with_session_store(
            Arc::clone(&store) as Arc<dyn SessionStorePlugin>,
            "combo-session",
        );

        let blocks = drain(
            orchestrator
                .execute_adaptive_turn(turn(&model))
                .await
                .expect("orchestrated turn must start"),
        )
        .await;

        assert_eq!(fast_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            slow_calls.load(Ordering::SeqCst),
            1,
            "the rejected draft must route to the slow tier exactly once"
        );
        assert!(
            !blocks
                .iter()
                .any(|b| matches!(b, MessageContentBlock::ToolCall { .. })),
            "the rejected draft block must never reach the consumer, got {blocks:?}"
        );
        assert!(
            blocks
                .iter()
                .any(|b| matches!(b, MessageContentBlock::Text(_))),
            "the live slow tier must produce a Text block, got {blocks:?}"
        );

        let records = store
            .replay_session("combo-session")
            .expect("replay must succeed");
        assert_eq!(records.len(), 1, "exactly one swap record, got {records:?}");
        match &records[0].event {
            SessionEvent::ModelSwap { from, to, reason } => {
                assert_eq!(from, "fast-tier-id");
                assert_eq!(to, &model);
                assert_eq!(reason, "fallback_validation");
            }
            other => panic!("expected SessionEvent::ModelSwap, got {other:?}"),
        }
        assert_eq!(records[0].session_id, "combo-session");
    }

    /// The latency guard is a distinct trigger with its own recorded reason.
    #[tokio::test]
    async fn latency_threshold_swap_is_recorded_with_its_own_reason() {
        let fast_calls = Arc::new(AtomicUsize::new(0));
        let slow_calls = Arc::new(AtomicUsize::new(0));
        let store = Arc::new(SessionLogPlugin::new_in_memory());

        let orchestrator = ModelOrchestrator::with_executors(
            pair("slow-tier-id", 0, false),
            Arc::new(ClientPool::default()),
            Arc::new(QuietExecutor {
                calls: Arc::clone(&fast_calls),
            }),
            Arc::new(CannedExecutor {
                tier: "slow",
                blocks: vec![MessageContentBlock::Text("slow answer".into())],
                calls: Arc::clone(&slow_calls),
            }),
        )
        .with_session_store(
            Arc::clone(&store) as Arc<dyn SessionStorePlugin>,
            "latency-session",
        );

        let blocks = drain(
            orchestrator
                .execute_adaptive_turn(turn("combo-model"))
                .await
                .expect("orchestrated turn must start"),
        )
        .await;

        assert_eq!(fast_calls.load(Ordering::SeqCst), 1);
        assert_eq!(slow_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            blocks,
            vec![MessageContentBlock::Text("slow answer".into())],
            "the swap must serve the slow tier's content"
        );

        let records = store
            .replay_session("latency-session")
            .expect("replay must succeed");
        assert_eq!(records.len(), 1, "exactly one swap record, got {records:?}");
        match &records[0].event {
            SessionEvent::ModelSwap { from, to, reason } => {
                assert_eq!(from, "fast-tier-id");
                assert_eq!(to, "slow-tier-id");
                assert_eq!(
                    reason, "latency_threshold",
                    "the latency guard must be distinguishable from a validation fallback"
                );
            }
            other => panic!("expected SessionEvent::ModelSwap, got {other:?}"),
        }
    }

    /// Without an attached store the swap still happens and nothing is
    /// recorded: the coupling is opt-in, and its absence is silent by design
    /// rather than half-wired.
    #[tokio::test]
    async fn no_session_store_means_no_records_and_no_error() {
        let slow_calls = Arc::new(AtomicUsize::new(0));
        let unattached = SessionLogPlugin::new_in_memory();

        let orchestrator = ModelOrchestrator::with_executors(
            pair("slow-tier-id", 60_000, true),
            Arc::new(ClientPool::default()),
            Arc::new(CannedExecutor {
                tier: "fast",
                blocks: vec![invalid_draft()],
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            Arc::new(CannedExecutor {
                tier: "slow",
                blocks: vec![MessageContentBlock::Text("slow answer".into())],
                calls: Arc::clone(&slow_calls),
            }),
        );

        let blocks = drain(
            orchestrator
                .execute_adaptive_turn(turn("combo-model"))
                .await
                .expect("orchestrated turn must start"),
        )
        .await;

        assert_eq!(
            slow_calls.load(Ordering::SeqCst),
            1,
            "the swap must still happen without a store"
        );
        assert_eq!(
            blocks,
            vec![MessageContentBlock::Text("slow answer".into())]
        );
        assert!(
            unattached
                .replay_session("combo-session")
                .expect("replay must succeed")
                .is_empty(),
            "an unattached store must receive nothing"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 4: speculative + prompt-cache
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-speculative", feature = "plugin-prompt-cache"))]
mod speculative_with_cache_instrumentation {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::common;
    use cuca::plugin::CucaPlugin;
    use cuca::request::{AgentResponseStream, UnifiedResponse};
    use cuca::types::{MessageContentBlock, ProviderEndpoint};
    use cuca::{
        ClientPool, CucaClient, CucaError, ModelOrchestrator, PluginError, PromptCache,
        PromptCacheConfig, SwappableModelPair, TurnExecutor, UnifiedRequest,
    };
    use tokio_stream::StreamExt;

    struct CannedExecutor {
        blocks: Vec<MessageContentBlock>,
        calls: Arc<AtomicUsize>,
    }

    impl TurnExecutor for CannedExecutor {
        fn tier_name(&self) -> &'static str {
            "fast"
        }

        fn execute(&self, _request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let items: Vec<Result<MessageContentBlock, CucaError>> =
                self.blocks.iter().cloned().map(Ok).collect();
            Ok(Box::pin(tokio_stream::iter(items)))
        }
    }

    #[derive(Default)]
    struct HookCounters {
        on_request: AtomicUsize,
        on_stream_chunk: AtomicUsize,
        on_response_complete: AtomicUsize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CounterSnapshot {
        on_request: usize,
        on_stream_chunk: usize,
        on_response_complete: usize,
    }

    impl HookCounters {
        fn snapshot(&self) -> CounterSnapshot {
            CounterSnapshot {
                on_request: self.on_request.load(Ordering::SeqCst),
                on_stream_chunk: self.on_stream_chunk.load(Ordering::SeqCst),
                on_response_complete: self.on_response_complete.load(Ordering::SeqCst),
            }
        }
    }

    struct CountingPlugin {
        counters: Arc<HookCounters>,
    }

    impl CucaPlugin for CountingPlugin {
        fn name(&self) -> &'static str {
            "counting"
        }

        fn on_request(&self, _req: &mut UnifiedRequest) -> Result<(), PluginError> {
            self.counters.on_request.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
            self.counters.on_stream_chunk.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn on_response_complete(&self, _res: &UnifiedResponse) -> Result<(), PluginError> {
            self.counters
                .on_response_complete
                .fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn pair() -> SwappableModelPair {
        SwappableModelPair {
            fast_provider: ProviderEndpoint::LlamaCpp,
            fast_model_id: "fast-tier-id".into(),
            slow_provider: ProviderEndpoint::LlamaCpp,
            slow_model_id: "slow-tier-id".into(),
            latency_threshold_ms: 60_000,
            fallback_on_tool_error: false,
        }
    }

    fn orchestrator(
        blocks: Vec<MessageContentBlock>,
        calls: &Arc<AtomicUsize>,
    ) -> ModelOrchestrator {
        ModelOrchestrator::with_executors(
            pair(),
            Arc::new(ClientPool::default()),
            Arc::new(CannedExecutor {
                blocks,
                calls: Arc::clone(calls),
            }),
            Arc::new(CannedExecutor {
                blocks: vec![MessageContentBlock::Text("slow".into())],
                calls: Arc::new(AtomicUsize::new(0)),
            }),
        )
    }

    /// The base URL is a closed port on purpose: the orchestrator arm must
    /// never fall through to provider dispatch.
    const NO_SERVER: &str = "http://127.0.0.1:1/v1";

    fn cache() -> Arc<PromptCache> {
        Arc::new(
            PromptCache::new(
                PromptCacheConfig::new(16, Duration::from_secs(60)).expect("config must build"),
            )
            .expect("cache must build"),
        )
    }

    fn client(
        orchestrator: ModelOrchestrator,
        cache: Option<&Arc<PromptCache>>,
        counters: &Arc<HookCounters>,
    ) -> CucaClient {
        let mut builder = common::llamacpp_builder(NO_SERVER)
            .with_orchestrator(orchestrator)
            .register_plugin(Arc::new(CountingPlugin {
                counters: Arc::clone(counters),
            }) as Arc<dyn CucaPlugin>);
        if let Some(cache) = cache {
            builder = builder.with_prompt_cache_service(Arc::clone(cache));
        }
        builder.build().expect("client build must succeed")
    }

    fn request() -> UnifiedRequest {
        UnifiedRequest::new("combo-model").add_user_message("hi")
    }

    /// With a cache configured, an orchestrator turn is wrapped in the standard
    /// instrumentation: terminal hooks run and exactly one entry is written.
    #[tokio::test]
    async fn cached_orchestrator_turn_runs_terminal_hooks_and_writes_one_entry() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cache = cache();
        let counters = Arc::new(HookCounters::default());
        let client = client(
            orchestrator(vec![MessageContentBlock::Text("ok".into())], &calls),
            Some(&cache),
            &counters,
        );

        let blocks = common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("orchestrated turn must start"),
            10,
        )
        .await;

        assert_eq!(blocks, vec![MessageContentBlock::Text("ok".into())]);
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.on_request, 1);
        assert_eq!(
            snapshot.on_stream_chunk, 1,
            "the cached orchestrator path is the only one that runs per-chunk hooks"
        );
        assert_eq!(snapshot.on_response_complete, 1);
        assert_eq!(
            cache
                .snapshot()
                .expect("cache snapshot must succeed")
                .entries
                .len(),
            1
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "one fast-tier turn");
    }

    /// The second identical turn is served from the cache and never reaches the
    /// orchestrator's tier executors.
    #[tokio::test]
    async fn second_identical_turn_is_served_from_the_cache_without_touching_the_orchestrator() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cache = cache();
        let counters = Arc::new(HookCounters::default());
        let client = client(
            orchestrator(vec![MessageContentBlock::Text("ok".into())], &calls),
            Some(&cache),
            &counters,
        );

        let first = common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("first turn must start"),
            10,
        )
        .await;
        let before = counters.snapshot();

        let second = common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("second turn must be served from the cache"),
            10,
        )
        .await;
        let after = counters.snapshot();

        assert_eq!(second, first, "the replay must be identical and in order");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a hit must not invoke a tier executor again"
        );
        assert_eq!(
            cache
                .snapshot()
                .expect("cache snapshot must succeed")
                .entries
                .len(),
            1,
            "a hit must not add an entry"
        );
        assert_eq!(after.on_request - before.on_request, 1);
        assert_eq!(after.on_response_complete - before.on_response_complete, 1);
        assert_eq!(
            after.on_stream_chunk, before.on_stream_chunk,
            "a hit must never replay on_stream_chunk"
        );
    }

    /// Without a cache the orchestrator stream is returned unwrapped, so
    /// top-level per-chunk and terminal hooks never run. This asymmetry is
    /// documented at `src/client.rs` ~578-592; pinning it makes a silent change
    /// fail loudly.
    #[tokio::test]
    async fn an_uncached_orchestrator_turn_stays_uninstrumented() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counters = Arc::new(HookCounters::default());
        let client = client(
            orchestrator(vec![MessageContentBlock::Text("ok".into())], &calls),
            None,
            &counters,
        );

        let blocks = common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("orchestrated turn must start"),
            10,
        )
        .await;

        assert_eq!(blocks, vec![MessageContentBlock::Text("ok".into())]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            counters.snapshot(),
            CounterSnapshot {
                on_request: 1,
                on_stream_chunk: 0,
                on_response_complete: 0,
            },
            "an uncached orchestrator turn runs on_request only"
        );
    }

    /// The completion-point contract holds on the orchestrator arm too: a
    /// stream dropped before its end writes nothing.
    #[tokio::test]
    async fn a_dropped_orchestrator_stream_writes_nothing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cache = cache();
        let counters = Arc::new(HookCounters::default());
        let client = client(
            orchestrator(
                vec![
                    MessageContentBlock::Text("first".into()),
                    MessageContentBlock::Text("second".into()),
                ],
                &calls,
            ),
            Some(&cache),
            &counters,
        );

        let mut stream = client
            .generate_stream(request())
            .await
            .expect("orchestrated turn must start");
        let first = stream.next().await;
        assert!(first.is_some(), "the first block must be delivered");
        drop(stream);

        assert_eq!(calls.load(Ordering::SeqCst), 1, "the turn did dispatch");
        assert_eq!(
            cache
                .snapshot()
                .expect("cache snapshot must succeed")
                .entries
                .len(),
            0,
            "an incomplete orchestrator stream must not write a cache entry"
        );
        assert_eq!(
            counters.snapshot().on_response_complete,
            0,
            "no terminal hook fires for an abandoned stream"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 5: memory + prompt-cache export coordinator
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-memory", feature = "plugin-prompt-cache"))]
mod export_round_trip {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::common;
    use cuca::{
        CucaClient, CucaExport, GraphSnapshot, MemoryConfig, MemoryPlugin, PromptCache,
        PromptCacheConfig, UnifiedRequest,
    };

    fn new_cache() -> Arc<PromptCache> {
        Arc::new(
            PromptCache::new(
                PromptCacheConfig::new(16, Duration::from_secs(60)).expect("config must build"),
            )
            .expect("cache must build"),
        )
    }

    fn new_memory() -> MemoryPlugin {
        MemoryPlugin::new(MemoryConfig::default()).expect("memory plugin must build")
    }

    /// Real extracted graph state where `plugin-entity-extraction` is compiled
    /// in, an equivalent hand-built graph otherwise.
    fn populated_graph() -> GraphSnapshot {
        #[cfg(feature = "plugin-entity-extraction")]
        {
            common::extraction::extraction_plugin()
                .validate_candidate(common::extraction::pair_candidate(
                    "Ada",
                    "Analytical Engines",
                ))
                .expect("candidate must be accepted")
                .delta
                .snapshot()
        }
        #[cfg(not(feature = "plugin-entity-extraction"))]
        {
            use cuca::{GraphNode, GraphRelationship};

            GraphSnapshot {
                nodes: vec![
                    GraphNode {
                        id: "person:ada".into(),
                        labels: vec!["person".into()],
                        properties: serde_json::Map::new(),
                    },
                    GraphNode {
                        id: "company:analytical-engines".into(),
                        labels: vec!["company".into()],
                        properties: serde_json::Map::new(),
                    },
                ],
                relationships: vec![GraphRelationship {
                    id: "ada-works-at".into(),
                    from: "person:ada".into(),
                    to: "company:analytical-engines".into(),
                    kind: "works_at".into(),
                    weight: 1.0,
                    properties: serde_json::Map::new(),
                }],
            }
        }
    }

    fn requests() -> [UnifiedRequest; 2] {
        [
            UnifiedRequest::new("export-model").add_user_message("first"),
            UnifiedRequest::new("export-model").add_user_message("second"),
        ]
    }

    /// Drive both requests through a cache-configured client so the cache holds
    /// real entries produced by the real digesting path.
    async fn populated_cache(addr: &str, cache: &Arc<PromptCache>) -> CucaClient {
        let client = common::llamacpp_builder(addr.to_string())
            .with_prompt_cache_service(Arc::clone(cache))
            .build()
            .expect("client build must succeed");
        for request in requests() {
            common::drain_timeout(
                client
                    .generate_stream(request)
                    .await
                    .expect("dispatch must succeed"),
                10,
            )
            .await;
        }
        client
    }

    #[tokio::test]
    async fn populated_from_live_round_trips_through_json_into_a_fresh_pair() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "exported").await;
        let cache = new_cache();
        populated_cache(&format!("http://{addr}/v1"), &cache).await;
        assert_eq!(dispatches.load(Ordering::SeqCst), 2);

        let memory = new_memory();
        memory
            .replace_snapshot(populated_graph())
            .expect("graph import must succeed");

        let export = CucaExport::from_live(&memory, &cache).expect("export must succeed");
        let bytes = export.to_json_bytes().expect("encode must succeed");
        assert_eq!(
            CucaExport::from_json_slice(&bytes).expect("decode must succeed"),
            export,
            "the envelope must survive a JSON round trip unchanged"
        );

        let fresh_memory = new_memory();
        let fresh_cache = new_cache();
        let report = export
            .import_into(&fresh_memory, &fresh_cache, common::now_unix_ms())
            .expect("import must succeed");

        let source_graph = memory.snapshot().expect("graph lock must not be poisoned");
        let source_cache = cache.snapshot().expect("cache snapshot must succeed");
        assert_eq!(report.graph_nodes, source_graph.nodes.len());
        assert_eq!(report.graph_relationships, source_graph.relationships.len());
        assert_eq!(report.imported_cache_entries, source_cache.entries.len());
        assert_eq!(report.expired_cache_entries, 0);
        assert_eq!(report.capacity_evictions, 0);
        assert_eq!(
            fresh_memory
                .snapshot()
                .expect("graph lock must not be poisoned"),
            source_graph,
            "the imported graph must equal the exported one"
        );
        assert_eq!(
            fresh_cache.snapshot().expect("cache snapshot must succeed"),
            source_cache,
            "the imported cache must equal the exported one"
        );
    }

    /// The export is only meaningful if the keys still match a freshly digested
    /// request: an imported entry serves a hit with zero dispatches.
    #[tokio::test]
    async fn an_imported_cache_entry_still_serves_a_hit() {
        let source_dispatches = Arc::new(AtomicUsize::new(0));
        let source_addr =
            common::spawn_counting_sse_server(Arc::clone(&source_dispatches), "exported").await;
        let cache = new_cache();
        populated_cache(&format!("http://{source_addr}/v1"), &cache).await;

        let memory = new_memory();
        memory
            .replace_snapshot(populated_graph())
            .expect("graph import must succeed");
        let export = CucaExport::from_live(&memory, &cache).expect("export must succeed");

        let fresh_cache = new_cache();
        export
            .import_into(&new_memory(), &fresh_cache, common::now_unix_ms())
            .expect("import must succeed");

        let replay_dispatches = Arc::new(AtomicUsize::new(0));
        let replay_addr =
            common::spawn_counting_sse_server(Arc::clone(&replay_dispatches), "never").await;
        let client = common::llamacpp_builder(format!("http://{replay_addr}/v1"))
            .with_prompt_cache_service(Arc::clone(&fresh_cache))
            .build()
            .expect("client build must succeed");

        for request in requests() {
            let blocks = common::drain_timeout(
                client
                    .generate_stream(request)
                    .await
                    .expect("the imported entry must serve the request"),
                10,
            )
            .await;
            assert_eq!(common::text_of(&blocks), "exported");
        }
        assert_eq!(
            replay_dispatches.load(Ordering::SeqCst),
            0,
            "imported keys must match freshly digested requests, so nothing dispatches"
        );
    }

    /// Expiry is applied by the importer against its own clock, not baked into
    /// the exported document.
    #[tokio::test]
    async fn expired_entries_are_dropped_by_the_importer_not_the_exporter() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "exported").await;
        let cache = new_cache();
        populated_cache(&format!("http://{addr}/v1"), &cache).await;

        let memory = new_memory();
        memory
            .replace_snapshot(populated_graph())
            .expect("graph import must succeed");
        let export = CucaExport::from_live(&memory, &cache).expect("export must succeed");
        let entries = cache
            .snapshot()
            .expect("cache snapshot must succeed")
            .entries;
        let past_every_ttl = entries
            .iter()
            .map(|entry| entry.expires_at_unix_ms)
            .max()
            .expect("the cache holds entries")
            + 1;

        let fresh_memory = new_memory();
        let fresh_cache = new_cache();
        let report = export
            .import_into(&fresh_memory, &fresh_cache, past_every_ttl)
            .expect("import must succeed");
        assert_eq!(report.imported_cache_entries, 0);
        assert_eq!(report.expired_cache_entries, entries.len());
        assert_eq!(
            report.graph_nodes,
            populated_graph().nodes.len(),
            "cache expiry must not affect the graph section"
        );
        assert!(
            fresh_cache
                .snapshot()
                .expect("cache snapshot must succeed")
                .entries
                .is_empty()
        );

        let replay_dispatches = Arc::new(AtomicUsize::new(0));
        let replay_addr =
            common::spawn_counting_sse_server(Arc::clone(&replay_dispatches), "fresh").await;
        let client = common::llamacpp_builder(format!("http://{replay_addr}/v1"))
            .with_prompt_cache_service(Arc::clone(&fresh_cache))
            .build()
            .expect("client build must succeed");
        common::drain_timeout(
            client
                .generate_stream(requests().into_iter().next().expect("first request"))
                .await
                .expect("dispatch must succeed"),
            10,
        )
        .await;
        assert_eq!(
            replay_dispatches.load(Ordering::SeqCst),
            1,
            "an expired entry must miss and dispatch"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 6: cost + prompt-cache (a local cache hit is still charged)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-cost", feature = "plugin-prompt-cache"))]
mod cost_with_prompt_cache {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::common;
    use cuca::plugin::CucaPlugin;
    use cuca::{
        CostConfig, CostPlugin, CucaClient, ModelRates, PricingTable, PromptCache,
        PromptCacheConfig, UnifiedRequest,
    };

    const MODEL: &str = "combo-cost-model";

    fn cache() -> Arc<PromptCache> {
        Arc::new(
            PromptCache::new(
                PromptCacheConfig::new(16, Duration::from_secs(60)).expect("config must build"),
            )
            .expect("cache must build"),
        )
    }

    /// A cost plugin that prices `MODEL`, so a charged turn is visible in
    /// currency as well as in tokens.
    fn priced_cost() -> Arc<CostPlugin> {
        Arc::new(
            CostPlugin::new(CostConfig {
                pricing: PricingTable::new().with_model(
                    MODEL,
                    ModelRates {
                        input_micros_per_mtok: 3_000_000,
                        output_micros_per_mtok: 15_000_000,
                        ..Default::default()
                    },
                ),
                ..Default::default()
            })
            .expect("cost plugin must build"),
        )
    }

    fn client_at(addr: &str, cache: &Arc<PromptCache>, cost: &Arc<CostPlugin>) -> CucaClient {
        common::llamacpp_builder(addr.to_string())
            .with_prompt_cache_service(Arc::clone(cache))
            .register_plugin(Arc::clone(cost) as Arc<dyn CucaPlugin>)
            .build()
            .expect("client build must succeed")
    }

    fn request() -> UnifiedRequest {
        UnifiedRequest::new(MODEL)
            .add_system_message("primary instruction")
            .add_user_message("hi")
    }

    async fn run(client: &CucaClient) {
        common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("generate_stream must start"),
            10,
        )
        .await;
    }

    /// The ledger is *gross* spend: a replayed cache hit runs every
    /// `on_request` and `on_response_complete` hook, so two identical turns are
    /// charged twice while the provider is dispatched once. The plugin has no
    /// way to tell a replay from a real dispatch, and looking one up would be a
    /// runtime peer lookup AGENTS.md forbids. This test pins the documented
    /// behavior so a future change to it is loud.
    #[tokio::test]
    async fn a_cache_hit_skips_dispatch_but_still_charges_the_ledger() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let cache = cache();
        let cost = priced_cost();
        let client = client_at(&format!("http://{addr}/v1"), &cache, &cost);

        run(&client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            1,
            "the first turn must miss and dispatch"
        );
        let first = cost.usage().expect("ledger lock must not be poisoned");
        assert_eq!(first.turns, 1);
        assert!(first.prompt_tokens > 0 && first.spent_micros > 0);

        run(&client).await;
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            1,
            "the identical turn must be served from the local cache"
        );

        let second = cost.usage().expect("ledger lock must not be poisoned");
        assert_eq!(
            second.turns, 2,
            "the replayed response still runs on_response_complete"
        );
        assert_eq!(
            second.prompt_tokens,
            first.prompt_tokens * 2,
            "on_request charges the cached turn too"
        );
        assert_eq!(second.completion_tokens, first.completion_tokens * 2);
        assert_eq!(
            second.spent_micros,
            first.spent_micros * 2,
            "the ledger reads as gross, pre-cache spend"
        );

        let breakdown = cost.breakdown().expect("ledger lock must not be poisoned");
        assert_eq!(breakdown.len(), 1);
        assert_eq!(breakdown[0].0, MODEL);
        assert_eq!(breakdown[0].1.turns, 2);
    }
}

// ---------------------------------------------------------------------------
// Surface 7: cost + memory (hook order is observable in the estimate)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-cost", feature = "plugin-memory"))]
mod cost_with_memory {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use crate::common;
    use cuca::plugin::CucaPlugin;
    use cuca::{
        CostConfig, CostPlugin, CucaClient, GraphContextConfig, GraphNode, MemoryConfig,
        MemoryGraph, MemoryPlugin, MergePolicy, UnifiedRequest,
    };

    const MODEL: &str = "combo-order-model";

    /// Memory with graph injection enabled and one node in its graph, so every
    /// `on_request` renders a graph system message into the prompt.
    fn memory_with_graph_context() -> Arc<MemoryPlugin> {
        let plugin = MemoryPlugin::new(MemoryConfig {
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        })
        .expect("memory plugin must build");
        let mut graph = MemoryGraph::new();
        graph.upsert_node(GraphNode {
            id: "alice".into(),
            labels: vec!["person".into()],
            properties: serde_json::Map::new(),
        });
        plugin
            .merge_graph(graph, MergePolicy::Keep)
            .expect("seed merge must not fail");
        Arc::new(plugin)
    }

    fn cost() -> Arc<CostPlugin> {
        Arc::new(CostPlugin::new(CostConfig::default()).expect("cost plugin must build"))
    }

    fn client_at(addr: &str, plugins: Vec<Arc<dyn CucaPlugin>>) -> CucaClient {
        let mut builder = common::llamacpp_builder(addr.to_string());
        for plugin in plugins {
            builder = builder.register_plugin(plugin);
        }
        builder.build().expect("client build must succeed")
    }

    fn request() -> UnifiedRequest {
        UnifiedRequest::new(MODEL)
            .add_system_message("primary instruction")
            .add_user_message("hi")
    }

    async fn run(client: &CucaClient) {
        common::drain_timeout(
            client
                .generate_stream(request())
                .await
                .expect("generate_stream must start"),
            10,
        )
        .await;
    }

    /// `on_request` hooks run in registration order over one shared
    /// `UnifiedRequest`, so the prompt the cost plugin estimates depends on
    /// where it sits: after memory it prices the injected graph message, before
    /// memory it prices the caller's request untouched. Neither plugin requires
    /// a position, and neither looks the other up; only the number differs.
    #[tokio::test]
    async fn memory_graph_injection_is_inside_the_estimate_only_when_memory_runs_first() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let addr = format!("http://{addr}/v1");

        // The estimate of the caller's own request, before any injection.
        let baseline = cost()
            .estimate_request_tokens(&request())
            .expect("encoder lock must not be poisoned");
        assert!(baseline > 0);

        let after_memory = cost();
        run(&client_at(
            &addr,
            vec![
                memory_with_graph_context() as Arc<dyn CucaPlugin>,
                Arc::clone(&after_memory) as Arc<dyn CucaPlugin>,
            ],
        ))
        .await;
        let with_injection = after_memory
            .usage()
            .expect("ledger lock must not be poisoned")
            .prompt_tokens;

        let before_memory = cost();
        run(&client_at(
            &addr,
            vec![
                Arc::clone(&before_memory) as Arc<dyn CucaPlugin>,
                memory_with_graph_context() as Arc<dyn CucaPlugin>,
            ],
        ))
        .await;
        let without_injection = before_memory
            .usage()
            .expect("ledger lock must not be poisoned")
            .prompt_tokens;

        assert_eq!(
            without_injection, baseline,
            "registered first, the cost plugin never sees memory's injection"
        );
        assert!(
            with_injection > baseline,
            "registered after memory, the cost plugin prices the injected graph \
             message too: {with_injection} must exceed {baseline}"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 8: cost + telemetry (the core bridge feeds the ledger to OTel)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-cost", feature = "plugin-telemetry"))]
mod cost_with_telemetry {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::common;
    use cuca::plugin::CucaPlugin;
    use cuca::{
        CostConfig, CostPlugin, CucaClient, ModelRates, OpenTelemetryPlugin, OtelCostObserver,
        PricingTable, UnifiedRequest,
    };
    use opentelemetry_sdk::metrics::data::{
        AggregatedMetrics, Metric, MetricData, ResourceMetrics,
    };
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    const MODEL: &str = "combo-otel-cost-model";

    /// One meter provider for both the bridge and the telemetry plugin, with
    /// an in-memory exporter so the test can flush and read the batch back.
    fn provider_with_exporter() -> (SdkMeterProvider, InMemoryMetricExporter) {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        (provider, exporter)
    }

    fn exported_metric<'a>(metrics: &'a [ResourceMetrics], name: &str) -> &'a Metric {
        metrics
            .iter()
            .flat_map(|rm| rm.scope_metrics())
            .flat_map(|sm| sm.metrics())
            .find(|m| m.name() == name)
            .unwrap_or_else(|| panic!("metric `{name}` missing from export"))
    }

    fn gauge_value(metrics: &[ResourceMetrics], name: &str) -> u64 {
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) =
            exported_metric(metrics, name).data()
        else {
            panic!("`{name}` must export a Gauge<u64>");
        };
        gauge
            .data_points()
            .map(|dp| dp.value())
            .next()
            .unwrap_or_else(|| panic!("`{name}` must carry a data point"))
    }

    fn counter_total(metrics: &[ResourceMetrics], name: &str) -> u64 {
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = exported_metric(metrics, name).data()
        else {
            panic!("`{name}` must export a Sum<u64>");
        };
        sum.data_points().map(|dp| dp.value()).sum()
    }

    /// A priced ledger with the bridge attached as its only observer.
    fn cost_with_bridge(provider: &SdkMeterProvider) -> Arc<CostPlugin> {
        Arc::new(
            CostPlugin::new(CostConfig {
                pricing: PricingTable::new().with_model(
                    MODEL,
                    ModelRates {
                        input_micros_per_mtok: 3_000_000,
                        output_micros_per_mtok: 15_000_000,
                        ..Default::default()
                    },
                ),
                observers: vec![Arc::new(OtelCostObserver::new(provider))],
                ..Default::default()
            })
            .expect("cost plugin must build"),
        )
    }

    /// The caller wires the bridge, not the crate: the two plugins never look
    /// each other up, they only share the meter provider handed to both.
    #[tokio::test]
    async fn one_turn_moves_the_cost_gauges_on_the_shared_meter_provider() {
        let (provider, exporter) = provider_with_exporter();
        let cost = cost_with_bridge(&provider);
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;

        let client: CucaClient = common::llamacpp_builder(format!("http://{addr}/v1"))
            .register_plugin(Arc::clone(&cost) as Arc<dyn CucaPlugin>)
            .register_plugin(Arc::new(OpenTelemetryPlugin::new(&provider)))
            .build()
            .expect("client build must succeed");

        common::drain_timeout(
            client
                .generate_stream(
                    UnifiedRequest::new(MODEL)
                        .add_system_message("You are concise.")
                        .add_user_message("hi"),
                )
                .await
                .expect("generate_stream must start"),
            10,
        )
        .await;
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);

        let usage = cost.usage().expect("ledger lock must not be poisoned");
        assert_eq!(usage.turns, 1);
        assert!(usage.prompt_tokens > 0 && usage.spent_micros > 0);

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");

        assert_eq!(
            gauge_value(&metrics, "cuca_cost_prompt_tokens"),
            usage.prompt_tokens,
            "the bridge exports the ledger's own prompt-token total"
        );
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_completion_tokens"),
            usage.completion_tokens
        );
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_spent_micros"),
            usage.spent_micros
        );
        assert_eq!(gauge_value(&metrics, "cuca_cost_turns"), 1);
        assert_eq!(
            counter_total(&metrics, "cuca_requests_total"),
            1,
            "the telemetry plugin's own instrument shares the batch"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 9: rate-limit + prompt-cache (a cache hit still spends a token)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-rate-limit", feature = "plugin-prompt-cache"))]
mod rate_limit_with_prompt_cache {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::common;
    use cuca::{
        CucaClient, PromptCache, PromptCacheConfig, RateLimitConfig, RateLimiter, UnifiedRequest,
    };

    const MODEL: &str = "combo-rate-limit-model";

    fn client_at(addr: &str) -> CucaClient {
        let cache = Arc::new(
            PromptCache::new(
                PromptCacheConfig::new(16, Duration::from_secs(60)).expect("config must build"),
            )
            .expect("cache must build"),
        );
        common::llamacpp_builder(addr.to_string())
            .with_prompt_cache_service(cache)
            .build()
            .expect("client build must succeed")
    }

    /// The limiter is caller-driven, so it cannot know that `generate_stream`
    /// will short-circuit on a local cache hit: the permit is already acquired
    /// and its token already spent. The over-count is conservative (it never
    /// over-admits) and is pinned here so a future core-wired variant flips a
    /// failing test instead of changing behavior silently.
    #[tokio::test]
    async fn caller_side_permit_is_spent_even_on_a_cache_hit() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let client = client_at(&format!("http://{addr}/v1"));
        let limiter = RateLimiter::new(
            RateLimitConfig::new(8, Duration::from_secs(60), 2, 8).expect("config must validate"),
        )
        .expect("limiter must build");
        let before = limiter.usage().expect("usage must read").available_tokens;

        for _ in 0..2 {
            let permit = limiter.acquire().await.expect("permit must be granted");
            let stream = client
                .generate_stream(UnifiedRequest::new(MODEL).add_user_message("hi"))
                .await
                .expect("generate_stream must start");
            common::drain_timeout(stream, 10).await;
            drop(permit);
        }

        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            1,
            "the identical second turn must be served from the local cache"
        );
        assert_eq!(
            limiter.usage().expect("usage must read").available_tokens,
            before - 2,
            "both turns spent a token, including the one the cache served"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 10: redaction + session-log (order decides what gets persisted)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-redaction", feature = "plugin-session-log"))]
mod redaction_order_decides_what_is_logged {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use crate::common;
    use cuca::plugin::{CucaPlugin, SessionStorePlugin};
    use cuca::types::MessageContentBlock;
    use cuca::{
        RedactionConfig, RedactionPlugin, RedactionRule, SessionEvent, SessionLogPlugin,
        UnifiedRequest,
    };

    /// The fake secret both orders carry into the request.
    const SECRET: &str = "sk-live-4242";
    const MODEL: &str = "combo-model";

    fn redaction() -> Arc<RedactionPlugin> {
        Arc::new(
            RedactionPlugin::new(
                RedactionConfig::new(vec![RedactionRule::Literal {
                    kind: "api-key".to_string(),
                    value: SECRET.to_string(),
                }])
                .expect("policy must build"),
            )
            .expect("plugin must build"),
        )
    }

    /// Every `SystemPrompt`/`Message` text the store recorded, joined.
    ///
    /// Inbound `Output`/`Reasoning` records are excluded on purpose: those come
    /// from `on_stream_chunk`, which redaction deliberately does not implement,
    /// so model-output records are unscrubbed at *either* order.
    fn requested_text(store: &SessionLogPlugin) -> String {
        store
            .replay_session("default")
            .expect("replay must succeed")
            .iter()
            .filter_map(|record| match &record.event {
                SessionEvent::SystemPrompt { text } => Some(text.clone()),
                SessionEvent::Message { content, .. } => Some(
                    content
                        .iter()
                        .filter_map(|block| match block {
                            MessageContentBlock::Text(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Drive one turn through a client that registers the two plugins in the
    /// requested order, and return what the store recorded on `on_request`.
    async fn recorded_with(redaction_first: bool) -> String {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let store = Arc::new(SessionLogPlugin::new_in_memory());
        let scrubber = redaction() as Arc<dyn CucaPlugin>;
        let logger = Arc::clone(&store) as Arc<dyn CucaPlugin>;
        let ordered = if redaction_first {
            vec![scrubber, logger]
        } else {
            vec![logger, scrubber]
        };

        let mut builder = common::llamacpp_builder(format!("http://{addr}/v1"));
        for plugin in ordered {
            builder = builder.register_plugin(plugin);
        }
        let client = builder.build().expect("client build must succeed");
        common::drain_timeout(
            client
                .generate_stream(
                    UnifiedRequest::new(MODEL)
                        .add_system_message(format!("system {SECRET}"))
                        .add_user_message(format!("user {SECRET}")),
                )
                .await
                .expect("generate_stream must start"),
            10,
        )
        .await;

        requested_text(&store)
    }

    /// Registered first, redaction decides what the trajectory persists.
    #[tokio::test]
    async fn redaction_before_the_store_records_scrubbed_content() {
        let recorded = recorded_with(true).await;

        assert!(
            recorded.contains("[REDACTED:api-key]"),
            "the store must persist the replacement token; got {recorded:?}"
        );
        assert!(
            !recorded.contains(SECRET),
            "the raw secret must not reach the trajectory; got {recorded:?}"
        );
    }

    /// Registered after, the store persists the raw value — to disk, with a
    /// file backend, whose append-only format never rewrites an existing frame.
    /// Documented, never enforced: neither plugin requires a position.
    #[tokio::test]
    async fn redaction_after_the_store_records_the_raw_value() {
        let recorded = recorded_with(false).await;

        assert!(
            recorded.contains(SECRET),
            "the store ran first, so it recorded the unscrubbed request; got {recorded:?}"
        );
        assert!(
            !recorded.contains("[REDACTED:api-key]"),
            "nothing had rewritten the request yet; got {recorded:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Surface 11: redaction + prompt-cache (redaction is inside the cache key)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "plugin-redaction", feature = "plugin-prompt-cache"))]
mod redaction_changes_cache_keys {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::common;
    use cuca::plugin::CucaPlugin;
    use cuca::plugins::prompt_cache::digest_request;
    use cuca::{
        PromptCache, PromptCacheConfig, RedactionConfig, RedactionPlugin, RedactionRule,
        UnifiedRequest,
    };

    const MODEL: &str = "combo-model";

    /// A prefixed rule, so two *distinct* secrets collapse onto one token and
    /// the key-convergence effect is observable.
    fn redaction() -> Arc<RedactionPlugin> {
        Arc::new(
            RedactionPlugin::new(
                RedactionConfig::new(vec![RedactionRule::Prefixed {
                    kind: "api-key".to_string(),
                    prefix: "sk-".to_string(),
                    min_len: 4,
                    max_len: 32,
                }])
                .expect("policy must build"),
            )
            .expect("plugin must build"),
        )
    }

    fn request(secret: &str) -> UnifiedRequest {
        UnifiedRequest::new(MODEL).add_user_message(format!("deploy with {secret}"))
    }

    /// The digest is taken from the request *after* every `on_request` hook, so
    /// enabling redaction changes every cache key: a snapshot imported from a
    /// pre-redaction run is rejected as a digest mismatch rather than served.
    #[test]
    fn the_post_hook_digest_differs_with_and_without_redaction() {
        let plugin = redaction();
        let raw = request("sk-live-4242");
        let mut scrubbed = raw.clone();
        plugin.on_request(&mut scrubbed).expect("hook must succeed");

        assert_ne!(
            digest_request(&raw).expect("digest must succeed"),
            digest_request(&scrubbed).expect("digest must succeed"),
            "the scrubbed request must key differently from the raw one"
        );
    }

    /// Two requests differing only inside a fully redacted secret converge onto
    /// one key. Accepted on purpose: the provider saw neither secret, so the
    /// response cannot depend on which one it was.
    #[tokio::test]
    async fn requests_differing_only_in_the_secret_share_one_entry() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
        let cache = Arc::new(
            PromptCache::new(
                PromptCacheConfig::new(16, Duration::from_secs(60)).expect("config must build"),
            )
            .expect("cache must build"),
        );
        let client = common::llamacpp_builder(format!("http://{addr}/v1"))
            .with_prompt_cache_service(Arc::clone(&cache))
            .register_plugin(redaction() as Arc<dyn CucaPlugin>)
            .build()
            .expect("client build must succeed");

        for secret in ["sk-live-4242", "sk-live-9999"] {
            common::drain_timeout(
                client
                    .generate_stream(request(secret))
                    .await
                    .expect("generate_stream must start"),
                10,
            )
            .await;
        }

        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            1,
            "both secrets scrub to the same token, hence to the same cache key"
        );
        assert_eq!(
            cache
                .snapshot()
                .expect("snapshot must succeed")
                .entries
                .len(),
            1
        );
    }
}
