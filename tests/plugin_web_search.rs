//! Integration tests for the web search plugin (`plugin-web-search`).
//!
//! Response parsing, tool-argument validation, and a REAL search against a
//! local mock HTTP server cover the plugin logic without an external API; one
//! live pipeline smoke test registers the plugin (dummy key) against LM
//! Studio, where the model never actually calls `web_search`.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-web-search"))]

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::mpsc;

use cuca::plugin::CucaPlugin;
use cuca::types::MessageContentBlock;
use cuca::{PluginError, SearchResult, WebSearchConfig, WebSearchPlugin, WebSearchProvider};
use serde_json::json;

fn firecrawl_plugin(base_url: Option<String>) -> WebSearchPlugin {
    WebSearchPlugin::new(WebSearchConfig {
        provider: WebSearchProvider::Firecrawl,
        api_key: "test".to_string(),
        base_url,
        max_results: 3,
    })
}

#[test]
fn parse_response_firecrawl_shape() {
    let plugin = firecrawl_plugin(None);
    let body = r#"{
        "data": [
            { "title": "CUCA", "url": "https://cuca.dev", "description": "Compact Universal Client" }
        ]
    }"#;
    let results = plugin.parse_response(body).expect("parse must succeed");
    assert_eq!(
        results,
        vec![SearchResult {
            title: "CUCA".to_string(),
            url: "https://cuca.dev".to_string(),
            snippet: "Compact Universal Client".to_string(),
        }]
    );
}

#[test]
fn parse_response_malformed_is_validation() {
    let plugin = firecrawl_plugin(None);
    match plugin.parse_response("not json") {
        Err(PluginError::Validation { schema, message }) => {
            assert_eq!(schema, "web_search response");
            assert!(
                message.starts_with("malformed JSON response: "),
                "unexpected validation message: {message}"
            );
        }
        other => panic!("expected Validation, got {other:?}"),
    }
}

#[test]
fn web_search_missing_query_becomes_error_tool_result() {
    let plugin = firecrawl_plugin(None);
    let mut chunk = MessageContentBlock::ToolCall {
        id: "call_1".to_string(),
        name: "web_search".to_string(),
        arguments: json!({}),
    };
    // Validation fails before any network I/O, so the hook is safe end-to-end.
    plugin
        .on_stream_chunk(&mut chunk)
        .expect("hook must succeed");
    match chunk {
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => {
            assert_eq!(tool_call_id, "call_1");
            assert_eq!(
                output,
                "validation failed for schema web_search: web_search requires a non-empty string `query` argument"
            );
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[tokio::test]
async fn search_against_local_mock_server() {
    // A tiny canned HTTP server on an ephemeral loopback port records the
    // request line and answers one Firecrawl-shaped JSON response.
    let (port_tx, port_rx) = mpsc::channel();
    let (request_tx, request_rx) = mpsc::channel();
    let server = std::thread::spawn(move || {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        port_tx.send(port).expect("send port");
        let (mut sock, _) = listener.accept().expect("accept one connection");

        let mut buf: Vec<u8> = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = sock.read(&mut tmp).expect("read request");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..pos]).to_string();
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let mut it = line.split(':');
                        if it.next()?.trim().eq_ignore_ascii_case("content-length") {
                            it.next()?.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                while buf.len() < pos + 4 + content_length {
                    let n = sock.read(&mut tmp).expect("read body");
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                break;
            }
        }
        let request = String::from_utf8_lossy(&buf).to_string();
        let request_line = request.lines().next().unwrap_or_default().to_string();
        request_tx.send(request_line).expect("send request line");

        let body = r#"{"data":[{"title":"CUCA docs","url":"https://cuca.dev","description":"Compact Universal Client for Agents"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(response.as_bytes()).expect("write response");
        sock.flush().expect("flush response");
    });

    let port = port_rx.recv().expect("receive port");
    let plugin = firecrawl_plugin(Some(format!("http://127.0.0.1:{port}")));

    // Real HTTP round trip: the plugin's reqwest client against our listener.
    let results = plugin.search("cuca").await.expect("search must succeed");
    assert!(!results.is_empty(), "expected at least one SearchResult");
    assert_eq!(results[0].title, "CUCA docs");

    server.join().expect("mock server thread must exit");
    let request_line = request_rx.recv().expect("receive request line");
    assert!(
        request_line.starts_with("POST /v1/search"),
        "Firecrawl search must POST {{base}}/v1/search, got {request_line:?}"
    );
}

#[tokio::test]
async fn live_pipeline_smoke_with_web_search_plugin() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    // Dummy key: the trivial prompt never emits a web_search tool call, so the
    // plugin only observes the stream.
    let plugin = firecrawl_plugin(None);
    let client = common::client_with_plugins(vec![Arc::new(plugin)]);
    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must succeed");
    let blocks = common::drain_timeout(stream, 60).await;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Text(_))),
        "expected at least one Text block, got {blocks:?}"
    );
}
