//! Integration coverage for the call scheduler (`service-scheduler`), driven
//! end-to-end through `Dispatcher::call`.
//!
//! The deterministic test runs against two in-process SSE mock servers
//! (`common::spawn_counting_sse_server`), the same choice
//! `tests/service_rate_limit.rs` documents: which backend received a call
//! cannot be asserted against a real model. The last test is the live
//! llama.cpp smoke and skips when the server is unreachable.

#![cfg(all(feature = "provider-llamacpp", feature = "service-scheduler"))]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cuca::request::UnifiedRequest;
use cuca::{
    BackendClient, BackendId, BackendRegistry, BackendSpec, CallMeta, DispatchGate, Dispatcher,
    DispatcherConfig, OpenGate, Priority, SchedulerConfig,
};

/// A spec for the backend at `base`, sending each request's own model.
fn spec(base: &str) -> BackendSpec {
    let client = common::llamacpp_builder(base)
        .build()
        .expect("client build must succeed");
    BackendSpec {
        id: BackendId::parse(base),
        client: BackendClient::Ready(Arc::new(client)),
        model: None,
        thinking: None,
    }
}

fn registry() -> BackendRegistry {
    let gate: Arc<dyn DispatchGate> = Arc::new(OpenGate::default());
    BackendRegistry::new(gate, Arc::new(tokio::sync::watch::Sender::new(0)))
}

fn config() -> DispatcherConfig {
    DispatcherConfig::new(SchedulerConfig::new(1).expect("a valid ceiling"))
}

#[tokio::test]
async fn four_calls_over_two_backends_reach_both_and_are_counted() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let a = common::spawn_counting_sse_server(Arc::clone(&first), "ok").await;
    let b = common::spawn_counting_sse_server(Arc::clone(&second), "ok").await;
    let mut registry = registry();
    let dispatcher = Arc::new(
        Dispatcher::new(
            &mut registry,
            vec![
                spec(&format!("http://{a}/v1")),
                spec(&format!("http://{b}/v1")),
            ],
            config(),
        )
        .expect("two distinct backends must join"),
    );

    let mut handles = Vec::new();
    for index in 0..4 {
        let dispatcher = Arc::clone(&dispatcher);
        handles.push(tokio::spawn(async move {
            let request = UnifiedRequest::new("mock").add_user_message("hello");
            let meta = CallMeta::new("probe", Priority::Medium, format!("call {index}"));
            dispatcher.call(request, meta).await
        }));
    }
    for handle in handles {
        let completion = handle
            .await
            .expect("the task must not panic")
            .expect("the mock backend must answer");
        assert_eq!(completion.text, "ok");
    }

    assert!(first.load(Ordering::SeqCst) >= 1, "backend one got no call");
    assert!(
        second.load(Ordering::SeqCst) >= 1,
        "backend two got no call"
    );
    assert_eq!(
        first.load(Ordering::SeqCst) + second.load(Ordering::SeqCst),
        4
    );
    let counted: u64 = registry
        .schedulers()
        .iter()
        .flat_map(|scheduler| scheduler.snapshot().totals)
        .map(|entry| entry.totals.calls)
        .sum();
    assert_eq!(counted, 4, "every call is in the totals");
}

/// Live smoke: one call through a single real backend.
#[tokio::test]
async fn live_call_returns_text() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let mut registry = registry();
    let dispatcher = Dispatcher::new(&mut registry, vec![spec(&common::base_url())], config())
        .expect("one backend must join");

    let completion = dispatcher
        .call(
            common::live_request("Reply with ok.", &model),
            CallMeta::new("probe", Priority::High, "live smoke"),
        )
        .await
        .expect("the live call must succeed");

    assert!(!completion.text.is_empty(), "a live call must return text");
    assert_eq!(completion.model.as_ref(), model);
}
