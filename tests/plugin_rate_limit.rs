//! Integration coverage for the client-side rate limiter
//! (`plugin-rate-limit`), driven end-to-end through
//! `CucaClient::generate_stream`.
//!
//! Four of the five tests run against the shared in-process SSE mock server
//! (`common::spawn_counting_sse_server`), the same choice
//! `tests/plugin_prompt_cache.rs` documents: dispatch counts, concurrency
//! peaks, and pacing wall times cannot be asserted against a real model. The
//! last test is the live llama.cpp smoke and skips when the server is
//! unreachable.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-rate-limit"))]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cuca::request::UnifiedRequest;
use cuca::{CucaClient, RateLimitConfig, RateLimitError, RateLimiter};
use tokio_stream::StreamExt;

/// A limiter with a generous waiter queue and a 10 s wait budget, so only the
/// bound under test can refuse an acquire.
fn limiter(max_requests: u32, interval: Duration, max_concurrent: usize) -> Arc<RateLimiter> {
    let config = RateLimitConfig::new(max_requests, interval, max_concurrent, 32)
        .expect("config must validate")
        .with_max_wait(Duration::from_secs(10))
        .expect("wait budget must validate");
    Arc::new(RateLimiter::new(config).expect("limiter must build"))
}

fn client_at(addr: &str) -> CucaClient {
    common::llamacpp_builder(format!("http://{addr}/v1"))
        .build()
        .expect("client build must succeed")
}

fn request(prompt: &str) -> UnifiedRequest {
    UnifiedRequest::new("rate-limit-model").add_user_message(prompt)
}

/// Eight turns share one limiter capped at two concurrent slots: every turn
/// dispatches and drains, and a sampling loop never sees a third slot held.
#[tokio::test]
async fn concurrency_cap_is_never_exceeded_across_a_fan_out() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
    let client = Arc::new(client_at(&addr.to_string()));
    // A 64-token bucket keeps the bucket out of this test: only the
    // concurrency cap may throttle.
    let limiter = limiter(64, Duration::from_secs(60), 2);

    let handles: Vec<_> = (0..8)
        .map(|turn| {
            let client = Arc::clone(&client);
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                let _permit = limiter.acquire().await.expect("permit must be granted");
                // Sampled from inside the admitted turn, so it always counts
                // at least this permit and can never be a vacuous reading.
                let in_flight = limiter.usage().expect("usage must read").in_flight;
                assert!(
                    (1..=2).contains(&in_flight),
                    "an admitted turn must see 1 or 2 slots held, saw {in_flight}"
                );
                let stream = client
                    .generate_stream(request(&format!("turn {turn}")))
                    .await
                    .expect("dispatch must succeed");
                let blocks = common::drain_timeout(stream, 10).await;
                assert_eq!(common::text_of(&blocks), "ok");
            })
        })
        .collect();

    let mut peak = 0;
    loop {
        peak = peak.max(limiter.usage().expect("usage must read").in_flight);
        if handles.iter().all(tokio::task::JoinHandle::is_finished) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    for handle in handles {
        handle.await.expect("turn task must not panic");
    }

    assert!(
        peak <= 2,
        "the concurrency cap was exceeded: {peak} in flight"
    );
    assert_eq!(dispatches.load(Ordering::SeqCst), 8);
    assert_eq!(limiter.usage().expect("usage must read").in_flight, 0);
}

/// Six turns through a two-per-200 ms bucket: none is refused, and the run
/// cannot finish faster than the refills it needs.
#[tokio::test]
async fn bucket_paces_a_burst_instead_of_failing_it() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
    let client = client_at(&addr.to_string());
    let limiter = limiter(2, Duration::from_millis(200), 6);

    let started = Instant::now();
    for turn in 0..6 {
        let permit = limiter.acquire().await.expect("permit must be granted");
        let stream = client
            .generate_stream(request(&format!("turn {turn}")))
            .await
            .expect("dispatch must succeed");
        common::drain_timeout(stream, 10).await;
        drop(permit);
    }
    let elapsed = started.elapsed();

    assert_eq!(dispatches.load(Ordering::SeqCst), 6);
    assert!(
        elapsed >= Duration::from_millis(400),
        "a 6-turn burst through a 2-per-200ms bucket must span at least two \
         refill intervals, took {elapsed:?}"
    );
}

/// `try_acquire` is admission control, not dispatch: a refusal carries the
/// wait and never reaches the provider.
#[tokio::test]
async fn try_acquire_refuses_without_dispatching() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
    let client = client_at(&addr.to_string());
    let limiter = limiter(1, Duration::from_secs(60), 4);

    let permit = limiter.try_acquire().expect("the one token admits");
    let stream = client
        .generate_stream(request("first"))
        .await
        .expect("dispatch must succeed");
    common::drain_timeout(stream, 10).await;
    drop(permit);
    assert_eq!(dispatches.load(Ordering::SeqCst), 1);

    match limiter.try_acquire() {
        Err(RateLimitError::Exhausted { retry_after }) => assert!(
            retry_after > Duration::ZERO,
            "the refusal must carry a non-zero retry_after"
        ),
        other => panic!("an empty bucket must refuse, got {other:?}"),
    }
    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        1,
        "a refused acquire must not reach the provider"
    );
}

/// Release is RAII, not a terminal hook: a consumer that abandons the stream
/// mid-drain still returns its slot when the permit drops.
#[tokio::test]
async fn permit_is_released_when_the_caller_drops_the_stream_mid_drain() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
    let client = client_at(&addr.to_string());
    // One slot and a short budget: a leaked slot fails the next acquire
    // instead of hanging the test.
    let config = RateLimitConfig::new(16, Duration::from_secs(60), 1, 4)
        .expect("config must validate")
        .with_max_wait(Duration::from_millis(200))
        .expect("wait budget must validate");
    let limiter = RateLimiter::new(config).expect("limiter must build");

    {
        let _permit = limiter.acquire().await.expect("permit must be granted");
        let mut stream = client
            .generate_stream(request("abandoned"))
            .await
            .expect("dispatch must succeed");
        assert!(
            stream.next().await.is_some(),
            "the first block must be delivered"
        );
        // Dropped before `Poll::Ready(None)`, so no terminal hook ever runs.
        drop(stream);
    }

    let permit = limiter
        .acquire()
        .await
        .expect("the dropped permit must have returned its slot");
    assert_eq!(limiter.usage().expect("usage must read").in_flight, 1);
    drop(permit);
    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
}

/// Live smoke: two paced turns through a single-slot limiter against real
/// llama.cpp.
#[tokio::test]
async fn live_paced_turns_return_text() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let client = common::client();
    let limiter = limiter(2, Duration::from_secs(1), 1);

    for prompt in ["Reply with ok.", "Reply with ok again."] {
        let permit = limiter.acquire().await.expect("permit must be granted");
        let stream = client
            .generate_stream(common::live_request(prompt, &model))
            .await
            .expect("live dispatch must succeed");
        let blocks = common::drain_timeout(stream, 120).await;
        assert!(
            !common::text_of(&blocks).is_empty(),
            "a live paced turn must return text"
        );
        drop(permit);
        assert_eq!(limiter.usage().expect("usage must read").in_flight, 0);
    }
}
