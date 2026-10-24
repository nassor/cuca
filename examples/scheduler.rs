//! Send three prioritised calls through one scheduled backend.
//!
//! One `Dispatcher` over one backend takes three calls at once: a low, a
//! medium and a high priority one. The backend runs one call at a time (the
//! tuner starts every scheduler at level 1), so the first call to arrive runs
//! at once and the other two wait. The waiting call with the bigger value goes
//! next, which the `recent` list of the scheduler's snapshot shows: completed
//! calls, newest first, with the backend id, the priority, and the
//! `waited_ms` and `run_ms` of each.
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
//! cargo run --example scheduler --features provider-llamacpp,service-scheduler
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example scheduler --features provider-llamacpp,service-scheduler`
//!
//! # Output
//!
//! The program prints one line per reply, then the snapshot's `recent` list,
//! one line per completed call: the backend id, the call's subject and
//! priority, the level it ran under, and its `waited_ms` and `run_ms`. The
//! replies and the milliseconds depend on the model and the machine. The
//! order does not: the first call to arrive runs first, then the high call,
//! then the medium one. This is a recorded run against LM Studio serving
//! `google/gemma-4-e2b`:
//!
//! ```text
//! Dispatcher over http://127.0.0.1:1234/v1: ceiling 2, one call at a time to start
//!
//!   low    replied "low" (level 1, waited 0 ms, ran 386 ms)
//!   medium replied "medium" (level 1, waited 705 ms, ran 302 ms)
//!   high   replied "high" (level 1, waited 385 ms, ran 319 ms)
//!
//! Completed calls, newest first
//!   http://loopback:1234/v1 medium medium level=1 waited_ms=705 run_ms=302
//!   http://loopback:1234/v1 high   high   level=1 waited_ms=385 run_ms=319
//!   http://loopback:1234/v1 low    low    level=1 waited_ms=0 run_ms=386
//! ```
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `Dispatcher` is a service, not a `CucaPlugin`, so it wraps the client
//! rather than being registered on the builder. `CucaPlugin::on_request` is
//! synchronous: a hook can inspect a request but cannot park it behind other
//! calls, and parking is the capability. The caller keeps two duties the hooks
//! could not take over: drive `Dispatcher::rebalance` on its own beat when it
//! lists several backends, and finish every `CallSlot` it takes directly
//! (`Dispatcher::call` finishes its own).

use std::sync::Arc;

use cuca::request::UnifiedRequest;
use cuca::types::ProviderEndpoint;
use cuca::{
    BackendClient, BackendId, BackendRegistry, BackendSpec, CallMeta, CucaClient, DispatchError,
    DispatchGate, Dispatcher, DispatcherConfig, OpenGate, Priority, SchedulerConfig,
};

/// A short turn: the scheduling is the point, not the reply.
fn request(model: &str, prompt: &str) -> UnifiedRequest {
    UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message(prompt)
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

    // Stage 1: the registry owns one scheduler, tuner and breaker per backend.
    // The gate lets the caller pause dispatch or mark tasks stale; the open
    // gate never does.
    let gate: Arc<dyn DispatchGate> = Arc::new(OpenGate::default());
    let mut registry = BackendRegistry::new(gate, Arc::new(tokio::sync::watch::Sender::new(0)));

    // Stage 2: one dispatcher over one backend. A single entry in `specs`
    // is enough; add more and each call goes to the backend that would start
    // it soonest.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .build()?;
    let spec = BackendSpec {
        id: BackendId::parse(&base_url),
        client: BackendClient::Ready(Arc::new(client)),
        model: None,
        thinking: None,
    };
    let dispatcher = Arc::new(Dispatcher::new(
        &mut registry,
        vec![spec],
        DispatcherConfig::new(SchedulerConfig::new(2)?),
    )?);
    println!(
        "Dispatcher over {base_url}: ceiling {}, one call at a time to start",
        dispatcher.ceiling()
    );

    // Stage 3: three calls at once, two priorities. The first to arrive runs
    // at once; of the other two the higher priority goes first.
    let calls = [
        ("low", Priority::Low, "Reply with exactly: low"),
        ("medium", Priority::Medium, "Reply with exactly: medium"),
        ("high", Priority::High, "Reply with exactly: high"),
    ];
    let mut handles = Vec::new();
    for (subject, priority, prompt) in calls {
        let dispatcher = Arc::clone(&dispatcher);
        let request = request(&model, prompt);
        handles.push(tokio::spawn(async move {
            let meta = CallMeta::new("demo", priority, subject);
            (subject, dispatcher.call(request, meta).await)
        }));
    }
    println!();
    for handle in handles {
        let (subject, outcome) = handle.await?;
        match outcome {
            Ok(completion) => println!(
                "  {subject:<6} replied {:?} (level {}, waited {} ms, ran {} ms)",
                completion.text.trim(),
                completion.level,
                completion.wait_ms,
                completion.run_ms
            ),
            Err(DispatchError::Unreachable { detail }) => {
                println!("\nNo server answered at {base_url}: {detail}");
                println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
                return Ok(());
            }
            Err(error) => println!("  {subject:<6} failed: {error}"),
        }
    }

    // Stage 4: the scheduler's own record of what it ran, newest first.
    println!("\nCompleted calls, newest first");
    for scheduler in registry.schedulers() {
        let snapshot = scheduler.snapshot();
        for call in &snapshot.recent {
            println!(
                "  {} {:<6} {:<6} level={} waited_ms={} run_ms={}",
                snapshot.backend.as_str(),
                call.subject,
                call.priority.as_str(),
                call.level,
                call.waited_ms,
                call.run_ms
            );
        }
    }
    Ok(())
}
