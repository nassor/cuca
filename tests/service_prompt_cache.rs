//! Deterministic integration coverage for the client-owned prompt cache
//! (`service-prompt-cache`), driven end-to-end through `CucaClient::generate_stream`
//! and a local, ephemeral, OpenAI-compatible (llama.cpp) SSE mock server.
//!
//! No real provider and no filesystem fixtures: the SSE server is an
//! in-process `tokio::net::TcpListener` on an ephemeral loopback port
//! (`common::spawn_counting_sse_server`), mirroring the existing per-provider
//! end-to-end test pattern (see `src/provider/llamacpp.rs`,
//! `src/provider/openai.rs`) and `tests/plugin_web_search.rs`'s
//! local-mock-server style. No test here reaches the live llama.cpp harness in
//! `tests/common/mod.rs`; only the shared mock server is used.

#![cfg(all(feature = "provider-llamacpp", feature = "service-prompt-cache"))]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::spawn_counting_sse_server;
use cuca::CucaClient;
use cuca::error::PluginError;
use cuca::plugin::CucaPlugin;
use cuca::request::{AgentResponseStream, UnifiedRequest, UnifiedResponse};
use cuca::services::prompt_cache::{PromptCache, PromptCacheConfig};
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use tokio_stream::StreamExt;

/// Drain a stream to completion, panicking on any item-level error.
async fn drain(stream: &mut AgentResponseStream) -> Vec<MessageContentBlock> {
    let mut blocks = Vec::new();
    while let Some(item) = stream.next().await {
        blocks.push(item.unwrap_or_else(|e| panic!("stream item must be Ok: {e}")));
    }
    blocks
}

fn cache(capacity: usize) -> Arc<PromptCache> {
    Arc::new(
        PromptCache::new(PromptCacheConfig::new(capacity, Duration::from_secs(60)).unwrap())
            .unwrap(),
    )
}

/// Per-hook invocation counters, shared with the test via `Arc`.
#[derive(Default)]
struct HookCounters {
    on_request: AtomicUsize,
    execute_local_tool: AtomicUsize,
    on_stream_chunk: AtomicUsize,
    on_response_complete: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CounterSnapshot {
    on_request: usize,
    execute_local_tool: usize,
    on_stream_chunk: usize,
    on_response_complete: usize,
}

impl HookCounters {
    fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            on_request: self.on_request.load(Ordering::SeqCst),
            execute_local_tool: self.execute_local_tool.load(Ordering::SeqCst),
            on_stream_chunk: self.on_stream_chunk.load(Ordering::SeqCst),
            on_response_complete: self.on_response_complete.load(Ordering::SeqCst),
        }
    }
}

/// Counts every hook invocation and unconditionally overwrites `temperature`
/// to a fixed canonical value, so two requests that differ only in
/// `temperature` before this hook runs become identical (and therefore
/// digest identically) after it.
struct NormalizingCountingPlugin {
    counters: Arc<HookCounters>,
}

impl CucaPlugin for NormalizingCountingPlugin {
    fn name(&self) -> &'static str {
        "normalizing-counting"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        self.counters.on_request.fetch_add(1, Ordering::SeqCst);
        req.temperature = Some(0.42);
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

    fn on_response_complete(&self, _res: &UnifiedResponse) -> Result<(), PluginError> {
        self.counters
            .on_response_complete
            .fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Post-hook effective-request identity drives cache routing: a request that
/// differs *before* the hook but converges *after* it hits (one dispatch
/// total for both), while a request that still differs *after* the hook
/// misses (a second dispatch).
#[tokio::test]
async fn converging_requests_hit_after_post_hook_normalization_with_one_dispatch() {
    let dispatch_count = Arc::new(AtomicUsize::new(0));
    let addr = spawn_counting_sse_server(dispatch_count.clone(), "ok").await;

    let counters = Arc::new(HookCounters::default());
    let plugin: Arc<dyn CucaPlugin> = Arc::new(NormalizingCountingPlugin {
        counters: counters.clone(),
    });

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(format!("http://{addr}/v1"))
        .with_prompt_cache_config(PromptCacheConfig::new(16, Duration::from_secs(60)).unwrap())
        .register_plugin(plugin)
        .build()
        .unwrap_or_else(|e| panic!("build must succeed: {e}"));

    // Request A: pre-hook temperature 0.1 (miss: no cache yet).
    let request_a = UnifiedRequest::new("model-x")
        .add_user_message("hi")
        .set_temperature(0.1);
    let mut stream = client
        .generate_stream(request_a)
        .await
        .unwrap_or_else(|e| panic!("request A dispatch must succeed: {e}"));
    drain(&mut stream).await;
    assert_eq!(dispatch_count.load(Ordering::SeqCst), 1);

    // Request B: differs from A *before* the hook (temperature 0.9), but the
    // hook forces both to 0.42, so the effective (post-hook) requests are
    // identical: this must hit, not dispatch again.
    let request_b = UnifiedRequest::new("model-x")
        .add_user_message("hi")
        .set_temperature(0.9);
    let mut stream = client
        .generate_stream(request_b)
        .await
        .unwrap_or_else(|e| panic!("request B must succeed from cache: {e}"));
    let blocks_b = drain(&mut stream).await;
    assert_eq!(
        dispatch_count.load(Ordering::SeqCst),
        1,
        "a post-hook-identical request must hit the cache, not dispatch again"
    );
    assert_eq!(blocks_b, vec![MessageContentBlock::Text("ok".to_string())]);

    // Request C: differs from A even *after* the hook (a different model,
    // which the hook never touches): this must miss and dispatch again.
    let request_c = UnifiedRequest::new("model-y")
        .add_user_message("hi")
        .set_temperature(0.1);
    let mut stream = client
        .generate_stream(request_c)
        .await
        .unwrap_or_else(|e| panic!("request C dispatch must succeed: {e}"));
    drain(&mut stream).await;
    assert_eq!(
        dispatch_count.load(Ordering::SeqCst),
        2,
        "a request that still differs after the hook must miss and dispatch again"
    );
}

/// `UnifiedRequest` carries no client credentials or base URL, so the digest
/// never depends on them: two `CucaClient`s with different `api_key`s and
/// base URLs, sharing one `PromptCache`, converge on the same cache entry for
/// an identical logical request. The second client's base URL refuses every
/// connection outright, so if credentials/base URL leaked into the digest
/// this test would fail loudly (a transport error) instead of silently
/// passing.
#[tokio::test]
async fn credentials_and_base_url_are_not_part_of_the_digest() {
    let dispatch_count = Arc::new(AtomicUsize::new(0));
    let addr = spawn_counting_sse_server(dispatch_count.clone(), "shared").await;

    let shared_cache = cache(16);
    let request = || UnifiedRequest::new("shared-model").add_user_message("hi");

    let client_a = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(format!("http://{addr}/v1"))
        .with_api_key("key-a")
        .with_prompt_cache_service(shared_cache.clone())
        .build()
        .unwrap_or_else(|e| panic!("client_a build must succeed: {e}"));

    let mut stream = client_a
        .generate_stream(request())
        .await
        .unwrap_or_else(|e| panic!("client_a dispatch must succeed: {e}"));
    drain(&mut stream).await;
    assert_eq!(dispatch_count.load(Ordering::SeqCst), 1);

    // Port 1 is a privileged port with nothing listening: any real dispatch
    // attempt here fails fast with a transport error.
    let client_b = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url("http://127.0.0.1:1/v1")
        .with_api_key("key-b-completely-different")
        .with_prompt_cache_service(shared_cache.clone())
        .build()
        .unwrap_or_else(|e| panic!("client_b build must succeed: {e}"));

    let mut stream = client_b
        .generate_stream(request())
        .await
        .unwrap_or_else(|e| {
            panic!(
                "client_b's identical logical request must hit the shared cache \
             without dispatching (base_url/api_key must not affect the \
             digest); got a dispatch error instead: {e}"
            )
        });
    let blocks = drain(&mut stream).await;
    assert_eq!(
        blocks,
        vec![MessageContentBlock::Text("shared".to_string())]
    );
    assert_eq!(
        dispatch_count.load(Ordering::SeqCst),
        1,
        "client_b must not have dispatched to the mock server"
    );
}

/// A cache hit runs `on_request`/`on_response_complete` exactly once each and
/// never replays `execute_local_tool`/`on_stream_chunk`; the replayed content
/// is byte-for-byte identical and in order to what the original miss
/// produced.
#[tokio::test]
async fn cache_hit_runs_only_request_and_terminal_hooks_with_ordered_content() {
    let dispatch_count = Arc::new(AtomicUsize::new(0));
    let addr = spawn_counting_sse_server(dispatch_count.clone(), "hi there").await;

    let counters = Arc::new(HookCounters::default());
    let plugin: Arc<dyn CucaPlugin> = Arc::new(NormalizingCountingPlugin {
        counters: counters.clone(),
    });

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(format!("http://{addr}/v1"))
        .with_prompt_cache_config(PromptCacheConfig::new(16, Duration::from_secs(60)).unwrap())
        .register_plugin(plugin)
        .build()
        .unwrap_or_else(|e| panic!("build must succeed: {e}"));

    let request = || UnifiedRequest::new("hook-model").add_user_message("hi");

    // First call: miss, real dispatch, writes the cache entry.
    let mut stream = client
        .generate_stream(request())
        .await
        .unwrap_or_else(|e| panic!("first dispatch must succeed: {e}"));
    let first_blocks = drain(&mut stream).await;
    assert_eq!(dispatch_count.load(Ordering::SeqCst), 1);

    let before = counters.snapshot();

    // Second call: the identical (post-hook) request must hit.
    let mut stream = client
        .generate_stream(request())
        .await
        .unwrap_or_else(|e| panic!("second call must hit the cache: {e}"));
    let second_blocks = drain(&mut stream).await;
    assert_eq!(
        dispatch_count.load(Ordering::SeqCst),
        1,
        "a hit must not dispatch again"
    );
    assert_eq!(
        second_blocks, first_blocks,
        "cached content must be replayed in the same order"
    );

    let after = counters.snapshot();
    assert_eq!(
        after.on_request - before.on_request,
        1,
        "on_request must fire exactly once for the hit"
    );
    assert_eq!(
        after.on_response_complete - before.on_response_complete,
        1,
        "on_response_complete must fire exactly once for the hit"
    );
    assert_eq!(
        after.execute_local_tool, before.execute_local_tool,
        "a cache hit must never replay execute_local_tool"
    );
    assert_eq!(
        after.on_stream_chunk, before.on_stream_chunk,
        "a cache hit must never replay on_stream_chunk"
    );
}

/// A fully drained successful stream writes exactly one cache entry; an
/// otherwise-identical-shaped stream that the consumer explicitly drops
/// before reaching the end writes nothing, even though the server already
/// answered and the first block was already delivered.
#[tokio::test]
async fn fully_drained_stream_writes_dropped_partial_stream_does_not() {
    let dispatch_count = Arc::new(AtomicUsize::new(0));
    let addr = spawn_counting_sse_server(dispatch_count.clone(), "written").await;

    let cache = cache(16);
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(format!("http://{addr}/v1"))
        .with_prompt_cache_service(cache.clone())
        .build()
        .unwrap_or_else(|e| panic!("build must succeed: {e}"));

    // Fully drained: writes exactly one entry.
    let written_request = UnifiedRequest::new("write-model").add_user_message("hi");
    let mut stream = client
        .generate_stream(written_request)
        .await
        .unwrap_or_else(|e| panic!("write dispatch must succeed: {e}"));
    drain(&mut stream).await;
    let snapshot = cache
        .snapshot()
        .unwrap_or_else(|e| panic!("snapshot must succeed: {e}"));
    assert_eq!(snapshot.entries.len(), 1);
    assert_eq!(dispatch_count.load(Ordering::SeqCst), 1);

    // A distinct request, explicitly dropped after its first (and only)
    // block: must not add a second entry.
    let dropped_request = UnifiedRequest::new("drop-model").add_user_message("hi");
    let mut stream = client
        .generate_stream(dropped_request)
        .await
        .unwrap_or_else(|e| panic!("drop dispatch must succeed: {e}"));
    let first = stream.next().await;
    assert!(first.is_some(), "the first block must have been delivered");
    drop(stream);

    let snapshot = cache
        .snapshot()
        .unwrap_or_else(|e| panic!("snapshot must succeed: {e}"));
    assert_eq!(
        snapshot.entries.len(),
        1,
        "a stream dropped before completion must not write a cache entry"
    );
    assert_eq!(dispatch_count.load(Ordering::SeqCst), 2);
}
