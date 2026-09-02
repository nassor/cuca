//! Pace a fan-out of turns through one client-side rate limiter.
//!
//! Six turns share one `CucaClient` and one `Arc<RateLimiter>` sized at one
//! request per 45 seconds with one turn in flight. Each turn runs the hand-off
//! the limiter's contract requires: acquire a permit, dispatch, drain the
//! whole stream, drop the permit. The permit's lifetime is the turn, and the
//! bucket is slower than the model, so the admission timestamps are the
//! bucket's refill schedule.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example rate_limit --features provider-llamacpp,service-rate-limit
//! ```
//!
//! # Configuration
//!
//! Both values default to a local llama.cpp server; override them to target
//! any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_MODEL`: upstream model id, defaults to `google/gemma-4-e4b`.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example rate_limit --features provider-llamacpp,service-rate-limit`
//!
//! # Output
//!
//! One run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Limiter: 1 requests / 45s, burst 1, 1 concurrent, 8 queued, 600s wait budget
//!   before any turn        tokens=1 in_flight=0 waiting=0
//!
//!   turn 0 replies: turn 0 (+49 thinking blocks)
//!
//! Fanning out 5 turns over the same limiter
//!   turn 1: admitted at 45030 ms (tokens=0 in_flight=1 waiting=4), replied "turn 1" (+59 thinking) at 75968 ms
//!   turn 2: admitted at 90054 ms (tokens=0 in_flight=1 waiting=3), replied "turn 2" (+53 thinking) at 126285 ms
//!   turn 3: admitted at 135061 ms (tokens=0 in_flight=1 waiting=2), replied "turn 3" (+58 thinking) at 166749 ms
//!   turn 4: admitted at 180070 ms (tokens=0 in_flight=1 waiting=1), replied "turn 4" (+50 thinking) at 211396 ms
//!   turn 5: admitted at 225078 ms (tokens=0 in_flight=1 waiting=0), replied "turn 5" (+46 thinking) at 255461 ms
//!
//! Peak while the fan-out ran: in_flight=1 (cap 1), waiting=5 (cap 8)
//! 6 turns took 255479 ms through a 1-per-45s bucket
//!
//! Draining the bucket with try_acquire
//!   refused: the bucket is empty, retry after 14625 ms
//!   after the demo         tokens=0 in_flight=0 waiting=0
//! ```
//!
//! The admission timestamps are the demo: 45 seconds apart, one per refilled
//! token, with `waiting` counting down as each parked caller is admitted.
//! `in_flight` never passes the cap, and the `try_acquire` refusal carries the
//! wait instead of a failed request.
//!
//! The replies and the thinking-block counts depend on the model. The
//! admission schedule does not: the bucket refills one token every 45 seconds
//! whatever the model is doing.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `RateLimiter` is a service, not a `CucaPlugin`, so it is passed around rather
//! than registered on the builder. `CucaPlugin::on_request` is synchronous and
//! could only reject a request, never pace it, and a hook-acquired permit would
//! leak on the dispatch-error and early-stream-drop paths. Dropping
//! `RateLimitPermit` releases the slot on every exit path, which is why the
//! caller holds it across the drain.

use std::io::{Write, stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, RateLimitConfig, RateLimitError, RateLimiter, UnifiedRequest};
use tokio_stream::StreamExt;

/// Turns fanned out concurrently after the first sequential one.
const FANNED_OUT_TURNS: usize = 5;

/// One labelled gauge reading.
fn print_usage(label: &str, limiter: &RateLimiter) {
    match limiter.usage() {
        Ok(usage) => println!(
            "  {label:<22} tokens={} in_flight={} waiting={}",
            usage.available_tokens, usage.in_flight, usage.waiting
        ),
        Err(error) => println!("  {label:<22} usage unavailable: {error}"),
    }
}

/// A short turn. The reply is one word: the pacing is the point, and a
/// reasoning model needs a token budget big enough to finish thinking and
/// still say it.
fn request(model: &str, turn: usize) -> UnifiedRequest {
    UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message(format!("Reply with exactly: turn {turn}"))
        .set_max_tokens(96)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Stage 1: the bounds. Every one of them is validated at construction, so
    // a zero rate or a zero cap fails here instead of being clamped.
    let limiter = Arc::new(RateLimiter::new(
        RateLimitConfig::new(1, Duration::from_secs(45), 1, 8)?
            .with_max_wait(Duration::from_secs(600))?,
    )?);
    let config = limiter.config();
    println!(
        "Limiter: {} requests / {:?}, burst {}, {} concurrent, {} queued, {:?} wait budget",
        config.max_requests,
        config.interval,
        config.burst,
        config.max_concurrent,
        config.max_waiters,
        config.max_wait
    );
    print_usage("before any turn", &limiter);

    // Stage 2: one client for every turn. The limiter stays a separate object:
    // it has no hooks, so `register_plugin(limiter)` would not compile.
    let client = Arc::new(
        CucaClient::builder()
            .with_provider(ProviderEndpoint::LlamaCpp)
            .with_base_url(base_url.clone())
            .build()?,
    );

    let started = Instant::now();

    // Stage 3: one turn end to end, which doubles as the reachability check.
    // The permit is taken before the dispatch and dropped after the drain.
    let permit = limiter.acquire().await?;
    let mut stream = match client.generate_stream(request(&model, 0)).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    print!("\n  turn 0 replies: ");
    let mut thinking = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
            Ok(_) => {}
            Err(error) => {
                print!("[the stream ended early: {error}]");
                break;
            }
        }
    }
    println!(" (+{thinking} thinking blocks)");
    drop(permit);

    // Stage 4: the remaining turns, all dispatched at once. The limiter is
    // what decides when each of them actually reaches the provider.
    println!("\nFanning out {FANNED_OUT_TURNS} turns over the same limiter");
    let mut handles = Vec::with_capacity(FANNED_OUT_TURNS);
    for turn in 1..=FANNED_OUT_TURNS {
        let client = Arc::clone(&client);
        let limiter = Arc::clone(&limiter);
        let model = model.clone();
        handles.push(tokio::spawn(async move {
            let permit = match limiter.acquire().await {
                Ok(permit) => permit,
                // QueueFull past `max_waiters`, Exhausted past `max_wait`.
                Err(error) => {
                    println!("  turn {turn}: refused, {error}");
                    return;
                }
            };
            let admitted_ms = started.elapsed().as_millis();
            let reading = match limiter.usage() {
                Ok(usage) => format!(
                    "tokens={} in_flight={} waiting={}",
                    usage.available_tokens, usage.in_flight, usage.waiting
                ),
                Err(error) => format!("usage unavailable: {error}"),
            };

            let mut reply = String::new();
            let mut thinking = 0usize;
            match client.generate_stream(request(&model, turn)).await {
                Ok(mut stream) => {
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(MessageContentBlock::Text(text)) => reply.push_str(&text),
                            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
                            Ok(_) => {}
                            Err(error) => {
                                println!("  turn {turn}: the stream ended early, {error}");
                                break;
                            }
                        }
                    }
                }
                Err(error) => println!("  turn {turn}: dispatch failed, {error}"),
            }
            println!(
                "  turn {turn}: admitted at {admitted_ms} ms ({reading}), replied {:?} \
                 (+{thinking} thinking) at {} ms",
                reply.trim(),
                started.elapsed().as_millis()
            );
            // Explicit for the demo: the slot comes back here, not when the
            // stream ended.
            drop(permit);
        }));
    }

    // Stage 5: the gauge, sampled while the fan-out runs. The peak is the
    // observable proof that the caps held.
    let mut peak_in_flight = 0;
    let mut peak_waiting = 0;
    loop {
        if let Ok(usage) = limiter.usage() {
            peak_in_flight = peak_in_flight.max(usage.in_flight);
            peak_waiting = peak_waiting.max(usage.waiting);
        }
        if handles.iter().all(tokio::task::JoinHandle::is_finished) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for handle in handles {
        handle.await?;
    }
    println!(
        "\nPeak while the fan-out ran: in_flight={peak_in_flight} (cap {}), \
         waiting={peak_waiting} (cap {})",
        limiter.config().max_concurrent,
        limiter.config().max_waiters
    );
    println!(
        "{} turns took {} ms through a {}-per-{:?} bucket",
        FANNED_OUT_TURNS + 1,
        started.elapsed().as_millis(),
        limiter.config().max_requests,
        limiter.config().interval
    );

    // Stage 6: the non-blocking path. `try_acquire` never waits: it spends a
    // token when one is there and otherwise reports how long until the next
    // one, without ever reaching the provider.
    println!("\nDraining the bucket with try_acquire");
    loop {
        match limiter.try_acquire() {
            Ok(permit) => {
                println!("  admitted: one token spent, nothing dispatched");
                drop(permit);
            }
            Err(RateLimitError::Exhausted { retry_after }) => {
                println!(
                    "  refused: the bucket is empty, retry after {} ms",
                    retry_after.as_millis()
                );
                break;
            }
            // Busy (every slot held) or Lock (poisoned bucket); neither can
            // happen here, since each permit above is dropped immediately.
            Err(error) => {
                println!("  refused: {error}");
                break;
            }
        }
    }
    print_usage("after the demo", &limiter);
    Ok(())
}
