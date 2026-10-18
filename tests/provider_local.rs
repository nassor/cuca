//! Live tests for the in-process `provider-local` Gemma 4 engine.
//!
//! These run the real model on the CPU, so they need the GGUF file:
//!
//! - `CUCA_LOCAL_MODEL_PATH=<file.gguf>` loads an existing file, or
//! - `CUCA_LOCAL_DOWNLOAD=1` downloads the pinned Gemma 4 E2B Q4_0 file
//!   (2.84 GB) into the cache directory (`CUCA_CACHE_DIR` overrides it).
//!
//! With neither set, every test prints `SKIP: local model not available: ...`
//! and passes, unless `CUCA_REQUIRE_LIVE=1` turns the skip into a panic. Build
//! for the host CPU, since candle's SIMD kernels are chosen at compile time:
//!
//! ```sh
//! CUCA_LOCAL_DOWNLOAD=1 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local
//! ```
//!
//! One model is loaded for the whole binary and the tests take turns on its
//! single engine, so they run sequentially whatever the test harness does.
//! Every request is greedy (`temperature = 0`).

#![cfg(feature = "provider-local")]

use cuca::types::{
    MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
};
use cuca::{
    CucaClient, DownloadProgressCallback, LocalModel, LocalModelConfig, ModelSource,
    UnifiedRequest, UnifiedResponse,
};
use serde_json::json;
use tokio::sync::{Mutex, OnceCell};
use tokio_stream::StreamExt;

static MODEL: OnceCell<Option<LocalModel>> = OnceCell::const_new();
static ENGINE_TURN: Mutex<()> = Mutex::const_new(());

/// The shared model, loaded on first use; `None` when no source is configured.
async fn model() -> Option<LocalModel> {
    MODEL
        .get_or_init(|| async {
            let source = if let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") {
                ModelSource::File(path.into())
            } else if std::env::var("CUCA_LOCAL_DOWNLOAD").as_deref() == Ok("1") {
                ModelSource::gemma4_e2b_it_q4_0()
            } else {
                return None;
            };
            let config = LocalModelConfig {
                source,
                on_download_progress: Some(DownloadProgressCallback::new(|p| {
                    eprint!(
                        "\rdownloading: {} / {:?} bytes",
                        p.downloaded_bytes, p.total_bytes
                    );
                })),
                ..Default::default()
            };
            Some(
                LocalModel::load(config)
                    .await
                    .unwrap_or_else(|e| panic!("local model must load: {e}")),
            )
        })
        .await
        .clone()
}

/// Skip (or, under `CUCA_REQUIRE_LIVE=1`, fail) when no model is configured.
fn skip_or_fail() {
    let reason =
        "SKIP: local model not available: set CUCA_LOCAL_MODEL_PATH or CUCA_LOCAL_DOWNLOAD=1";
    if std::env::var("CUCA_REQUIRE_LIVE").as_deref() == Ok("1") {
        panic!("{reason}");
    }
    println!("{reason}");
}

/// Bind the shared model and hold the engine turn, or return from the test.
macro_rules! live_model {
    ($model:ident, $turn:ident) => {
        let Some($model) = model().await else {
            skip_or_fail();
            return;
        };
        let $turn = ENGINE_TURN.lock().await;
    };
}

/// Everything one request produced.
struct Reply {
    blocks: Vec<MessageContentBlock>,
    response: UnifiedResponse,
}

impl Reply {
    fn text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MessageContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn position(&self, pred: impl Fn(&MessageContentBlock) -> bool) -> Option<usize> {
        self.blocks.iter().position(pred)
    }
}

/// Run `request` greedily on `model` and drain its stream.
async fn ask(model: &LocalModel, request: UnifiedRequest) -> Reply {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model.clone())
        .build()
        .expect("client builds");
    let (mut stream, handle) = client
        .generate_stream_with_response(request.set_temperature(0.0))
        .await
        .expect("request dispatches");
    let mut blocks = Vec::new();
    while let Some(item) = stream.next().await {
        blocks.push(item.expect("stream item"));
    }
    Reply {
        blocks,
        response: handle.take().expect("terminal response"),
    }
}

fn request() -> UnifiedRequest {
    UnifiedRequest::new("gemma-4-e2b-it")
}

#[tokio::test]
async fn greedy_text_and_usage() {
    live_model!(model, _turn);
    let reply = ask(
        &model,
        request()
            .add_user_message("What is the capital of France? Answer with one word.")
            .set_max_tokens(32),
    )
    .await;
    let text = reply.text();
    assert!(text.to_lowercase().contains("paris"), "got: {text:?}");
    assert!(
        !reply
            .blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Thinking { .. })),
        "thinking is off by default"
    );
    let usage = reply.response.usage.expect("local engine reports usage");
    assert!(usage.completion_tokens > 0);
    assert!(usage.prompt_tokens > 0);
}

#[tokio::test]
async fn thinking_precedes_answer() {
    live_model!(model, _turn);
    let reply = ask(
        &model,
        request()
            .add_user_message("What is 17 * 23?")
            .enable_thinking(None)
            .set_max_tokens(1024),
    )
    .await;
    let first_thinking = reply
        .position(|b| matches!(b, MessageContentBlock::Thinking { .. }))
        .expect("at least one Thinking block");
    let first_text = reply
        .position(|b| matches!(b, MessageContentBlock::Text(_)))
        .expect("an answer follows the reasoning");
    assert!(first_thinking < first_text, "{:?}", reply.blocks);
    let text = reply.text();
    assert!(text.contains("391"), "got: {text:?}");
    let usage = reply.response.usage.expect("usage");
    assert!(usage.reasoning_tokens.is_some_and(|n| n > 0));
}

#[tokio::test]
async fn tool_call_round_trip() {
    live_model!(model, _turn);
    let weather = ToolDefinition {
        name: "get_weather".into(),
        description: "Get the current weather for a city.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["city"]
        }),
    };
    let question = "What's the weather in Paris right now? Use the tool.";

    let first = ask(
        &model,
        request()
            .add_tool(weather.clone())
            .add_user_message(question)
            .set_max_tokens(256),
    )
    .await;
    let (id, name, arguments) = first
        .blocks
        .iter()
        .find_map(|b| match b {
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some((id.clone(), name.clone(), arguments.clone())),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a ToolCall, got {:?}", first.blocks));
    assert_eq!(name, "get_weather");
    let city = arguments["city"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    assert!(city.contains("paris"), "arguments: {arguments}");

    let second = ask(
        &model,
        request()
            .add_tool(weather)
            .add_user_message(question)
            .add_message(UnifiedMessage {
                role: MessageRole::Assistant,
                content: vec![MessageContentBlock::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments,
                }],
                name: None,
                tool_call_id: None,
            })
            .add_message(UnifiedMessage {
                role: MessageRole::Tool,
                content: vec![MessageContentBlock::ToolResult {
                    tool_call_id: id.clone(),
                    output: r#"{"temperature":18,"condition":"cloudy"}"#.into(),
                }],
                name: Some(name),
                tool_call_id: Some(id),
            })
            .set_max_tokens(128),
    )
    .await;
    let text = second.text();
    assert!(text.contains("18"), "got: {text:?}");
}

#[tokio::test]
async fn prefix_reuse_hits_and_matches_cold() {
    live_model!(model, _turn);
    let system = "You are a terse assistant.";
    let first_question = "Name three primary colors.";
    let second_question = "Which of those is warmest?";

    let turn_a = ask(
        &model,
        request()
            .add_system_message(system)
            .add_user_message(first_question)
            .set_max_tokens(64),
    )
    .await;
    let answer_a = turn_a.text();
    assert!(!answer_a.is_empty(), "turn A produced no text");

    let turn_b = |max_tokens: u32| {
        request()
            .add_system_message(system)
            .add_user_message(first_question)
            .add_message(UnifiedMessage::assistant(answer_a.clone()))
            .add_user_message(second_question)
            .set_max_tokens(max_tokens)
    };

    let warm = ask(&model, turn_b(16)).await;
    let warm_read = warm
        .response
        .prompt_cache_usage
        .expect("prefix reuse is reported")
        .read_tokens;
    assert!(
        warm_read >= 20,
        "turn B should reuse turn A's prompt, read {warm_read} tokens"
    );

    // An unrelated conversation evicts the shared prefix from the cache.
    ask(
        &model,
        request()
            .add_system_message("You are a pirate. Reply in one short sentence.")
            .add_user_message("Say hello.")
            .set_max_tokens(8),
    )
    .await;

    let cold = ask(&model, turn_b(16)).await;
    let cold_read = cold
        .response
        .prompt_cache_usage
        .expect("prefix reuse is reported")
        .read_tokens;
    assert!(
        cold_read < warm_read,
        "an unrelated request should shrink the reuse: {cold_read} vs {warm_read}"
    );
    assert_eq!(
        warm.text(),
        cold.text(),
        "a warm and a cold prefix must produce the same greedy text"
    );
}

#[tokio::test]
async fn long_context_crosses_sliding_window() {
    live_model!(model, _turn);
    // About 900 tokens: past the 512-token sliding window, so the code word
    // is only reachable through the full-attention layers.
    let mut prompt = String::from("The secret code word is PELICAN. ");
    prompt.push_str(&"This sentence is filler text. ".repeat(120));
    prompt.push_str("What is the secret code word? Answer with the word only.");

    let reply = ask(
        &model,
        request().add_user_message(prompt).set_max_tokens(24),
    )
    .await;
    let text = reply.text();
    assert!(text.to_lowercase().contains("pelican"), "got: {text:?}");
    let usage = reply.response.usage.expect("usage");
    assert!(
        usage.prompt_tokens > 512,
        "prompt was {} tokens",
        usage.prompt_tokens
    );
}
