//! Integration coverage for outbound PII/secret scrubbing (`plugin-redaction`),
//! driven end-to-end through `CucaClient::generate_stream`.
//!
//! The proof this plugin exists for is a claim about *bytes on the wire*, which
//! no unit test can make: it needs a real dispatch. So most of this file runs
//! against a local, ephemeral, OpenAI-compatible SSE server that records the raw
//! request body it received. That server is kept local to this file rather than
//! added to `tests/common/mod.rs`, modelled on `common::spawn_counting_sse_server`
//! — no other suite needs the request body, and the shared helper deliberately
//! reads the socket only to drain it.
//!
//! One test reaches the live llama.cpp harness, with the suite's standard
//! `SKIP: llama.cpp not reachable: ...` behavior.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-redaction"))]

mod common;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cuca::error::CucaError;
use cuca::plugin::CucaPlugin;
use cuca::{CucaClient, RedactionConfig, RedactionPlugin, RedactionRule, UnifiedRequest};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The fake secret every prompt in this file carries.
const SECRET: &str = "sk-live-4242";

/// The one rule under test: an exact known-secret match.
fn api_key_rule() -> RedactionRule {
    RedactionRule::Literal {
        kind: "api-key".to_string(),
        value: SECRET.to_string(),
    }
}

/// A plugin over [`api_key_rule`] with the default bounds.
fn redaction_plugin() -> Arc<RedactionPlugin> {
    Arc::new(
        RedactionPlugin::new(
            RedactionConfig::new(vec![api_key_rule()]).expect("policy must build"),
        )
        .expect("plugin must build"),
    )
}

/// A client pointed at `addr` with `plugins` registered, no API key.
fn client_at(addr: SocketAddr, plugins: Vec<Arc<dyn CucaPlugin>>) -> CucaClient {
    let mut builder = common::llamacpp_builder(format!("http://{addr}/v1"));
    for plugin in plugins {
        builder = builder.register_plugin(plugin);
    }
    builder.build().expect("client build must succeed")
}

/// A one-message streaming request; the model id is fixed so two captured
/// bodies differ only where the prompt does.
fn request(prompt: &str) -> UnifiedRequest {
    UnifiedRequest::new("redaction-model")
        .add_system_message("You are concise.")
        .add_user_message(prompt)
}

/// The single body the server captured.
fn only_captured(bodies: &Arc<Mutex<Vec<String>>>) -> String {
    let captured = bodies.lock().expect("body lock must not be poisoned");
    assert_eq!(captured.len(), 1, "expected exactly one captured request");
    captured[0].clone()
}

/// Spawn an ephemeral loopback OpenAI-compatible SSE server that records the raw
/// JSON body of every request it accepts, counts one dispatch per accepted
/// connection, and answers each one with the same canned single-`Text`-block
/// stream (`"ok"`, `finish_reason: "stop"`, `[DONE]`).
///
/// Returns the address it listens on; the spawned task is aborted when the
/// test's tokio runtime is torn down, so no explicit shutdown is needed.
async fn spawn_body_capturing_sse_server(
    dispatches: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<String>>>,
) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            dispatches.fetch_add(1, Ordering::SeqCst);
            let body = read_request_body(&mut socket).await;
            bodies
                .lock()
                .expect("body lock must not be poisoned")
                .push(body);
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in [
                r#"data: {"choices":[{"delta":{"content":"ok"}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                "data: [DONE]",
            ] {
                let chunk = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{chunk}\r\n", chunk.len()));
            }
            response.push_str("0\r\n\r\n");
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });
    addr
}

/// Read one HTTP request off `socket` and return its body.
///
/// Unlike the shared counting server's single 4KiB read, this honors
/// `content-length`: the assertions here are about the body, so a body split
/// across reads must still be captured whole.
async fn read_request_body(socket: &mut tokio::net::TcpStream) -> String {
    let mut raw = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut body_start = None;
    loop {
        let read = socket
            .read(&mut chunk)
            .await
            .expect("request read must succeed");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&chunk[..read]);
        if body_start.is_none() {
            body_start = raw
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|at| at + 4);
        }
        if let Some(start) = body_start
            && raw.len() - start >= content_length(&raw[..start])
        {
            break;
        }
    }
    let start = body_start.expect("the request must carry a complete header block");
    String::from_utf8(raw[start..].to_vec()).expect("the request body must be UTF-8")
}

/// The `content-length` of a raw header block, or `0` when absent.
fn content_length(headers: &[u8]) -> usize {
    String::from_utf8_lossy(headers)
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse().ok())
        .unwrap_or(0)
}

/// The claim the plugin exists for: the secret never reaches the socket, and the
/// replacement token does.
#[tokio::test]
async fn redacted_prompt_is_what_crosses_the_wire() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let addr = spawn_body_capturing_sse_server(Arc::clone(&dispatches), Arc::clone(&bodies)).await;
    let plugin = redaction_plugin();
    let client = client_at(addr, vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);

    let blocks = common::drain_timeout(
        client
            .generate_stream(request(&format!("deploy with {SECRET} now")))
            .await
            .expect("generate_stream must start"),
        10,
    )
    .await;

    assert_eq!(common::text_of(&blocks), "ok");
    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    let body = only_captured(&bodies);
    assert!(
        body.contains("[REDACTED:api-key]"),
        "the dispatched body must carry the replacement token; got {body}"
    );
    assert!(
        !body.contains(SECRET),
        "the secret must never reach the socket; got {body}"
    );
    assert_eq!(plugin.total_redactions(), 1);
    assert_eq!(
        plugin.last_redaction_event(),
        Some(("api-key".to_string(), "message_text", 1))
    );
}

/// Past the match cap the hook refuses, so `generate_stream` fails with
/// `CucaError::Plugin` and the server is never contacted at all.
#[tokio::test]
async fn over_cap_request_is_refused_before_dispatch() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let addr = spawn_body_capturing_sse_server(Arc::clone(&dispatches), Arc::clone(&bodies)).await;
    let plugin: Arc<dyn CucaPlugin> = Arc::new(
        RedactionPlugin::new(RedactionConfig {
            rules: vec![api_key_rule()],
            max_matches_per_text: 2,
            ..Default::default()
        })
        .expect("plugin must build"),
    );
    let client = client_at(addr, vec![plugin]);

    // `AgentResponseStream` is not `Debug`, so the Ok arm cannot go through
    // `expect_err`.
    let Err(err) = client
        .generate_stream(request(&format!("{SECRET} {SECRET} {SECRET}")))
        .await
    else {
        panic!("an over-cap value must refuse the request");
    };

    assert!(
        matches!(err, CucaError::Plugin(_)),
        "expected a plugin error, got {err:?}"
    );
    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        0,
        "the refusal happens before any provider dispatch"
    );
    assert!(
        bodies
            .lock()
            .expect("body lock must not be poisoned")
            .is_empty()
    );
}

/// With rules that match nothing, the instrumented body is byte-for-byte the
/// body an unregistered client sends: a clean prompt is not rewritten.
#[tokio::test]
async fn clean_prompt_leaves_the_request_untouched() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let addr = spawn_body_capturing_sse_server(Arc::clone(&dispatches), Arc::clone(&bodies)).await;
    let plugin = redaction_plugin();
    let prompt = "nothing sensitive in this prompt at all";

    let uninstrumented = client_at(addr, Vec::new());
    common::drain_timeout(
        uninstrumented
            .generate_stream(request(prompt))
            .await
            .expect("generate_stream must start"),
        10,
    )
    .await;

    let instrumented = client_at(addr, vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);
    common::drain_timeout(
        instrumented
            .generate_stream(request(prompt))
            .await
            .expect("generate_stream must start"),
        10,
    )
    .await;

    assert_eq!(dispatches.load(Ordering::SeqCst), 2);
    let captured = bodies.lock().expect("body lock must not be poisoned");
    assert_eq!(
        captured[0], captured[1],
        "a clean prompt must dispatch identically with and without the plugin"
    );
    assert_eq!(plugin.total_redactions(), 0);
    assert_eq!(plugin.last_request_redactions(), 0);
}

/// A real llama.cpp turn still completes with the plugin registered, and the
/// scrub actually fired on the way out.
#[tokio::test]
async fn live_turn_completes_with_redactions_applied() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let plugin = redaction_plugin();
    let client = common::client_with_plugins(vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);
    let prompt = format!("The key is {SECRET}. Reply with the single word: ok");

    let blocks = common::drain_timeout(
        client
            .generate_stream(common::live_request(&prompt, &common::live_model()))
            .await
            .expect("generate_stream must start"),
        60,
    )
    .await;

    assert!(!blocks.is_empty(), "the live turn must yield blocks");
    assert!(
        plugin.total_redactions() > 0,
        "the seeded secret must have been scrubbed on the way out"
    );
}
