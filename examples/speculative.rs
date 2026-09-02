//! Route turns across a fast and a slow model tier, and watch one turn cascade
//! from the fast tier to the slow one.
//!
//! Stage 1 runs the deterministic `ComplexityEvaluator` over three requests,
//! with no network. Stage 2 sends the simple one through the orchestrator, so
//! the fast tier serves it end to end. Stage 3 asks for a reply the default
//! `JsonToolDraftValidator` refuses, which re-routes the turn to the slow tier
//! with the rejection attached, twice, until the cascade budget is spent.
//! Stage 4 sends the multi-file request, which routing sends straight to the
//! slow tier with no draft phase. Stage 5 prints the verdicts behind stage 3.
//!
//! Both tiers run through injected `TurnExecutor`s so every routing decision
//! is printed as it happens; they draw their clients from the same
//! `ClientPool` the pooled executors use.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 serving both tier models.
//!
//! # Run
//!
//! ```sh
//! cargo run --example speculative --features provider-llamacpp,service-speculative
//! ```
//!
//! # Configuration
//!
//! A fast/slow pair needs two model ids, so this demo reads one variable per
//! tier instead of `CUCA_MODEL`. All three default to a local llama.cpp
//! server; override them to target any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_FAST_MODEL`: fast-tier model id, defaults to `google/gemma-4-e4b`.
//! - `CUCA_SLOW_MODEL`: slow-tier model id, defaults to `google/gemma-4-12b-qat`.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_FAST_MODEL=<small-id> CUCA_SLOW_MODEL=<large-id> cargo run --example speculative --features provider-llamacpp,service-speculative`
//!
//! # Output
//!
//! One run with `google/gemma-4-e4b` as the fast tier and
//! `google/gemma-4-12b-qat` as the slow tier:
//!
//! ```text
//! Pair: fast google/gemma-4-e4b, slow google/gemma-4-12b-qat
//! Thresholds: tool depth 1, input tokens 2000, file refs 3
//!
//! Stage 1: routing, with no network
//!   Fast  <- classify one short sentence
//!   Slow  <- four file references
//!   Slow  <- one tool-call round trip
//!
//! Stage 2: the Fast request, latency budget 60000 ms
//!   fast tier -> google/gemma-4-e4b
//!   reply: Positive
//!   346 ms, 1 text blocks, 0 thinking blocks
//!
//! Stage 3: a refused draft and the fallback cascade
//!   fast tier -> google/gemma-4-e4b
//!   slow tier -> google/gemma-4-12b-qat
//!   slow tier -> google/gemma-4-12b-qat
//!   cascade exhausted: provider llamacpp failed: text block is valid JSON but not a JSON object
//!   35635 ms, 0 text blocks, 77 thinking blocks
//!
//! Stage 4: the Slow request, no draft phase
//!   slow tier -> google/gemma-4-12b-qat
//!   reply: src/sse.rs
//!   24975 ms, 5 text blocks, 244 thinking blocks
//!
//! Stage 5: the draft verdicts behind the cascade
//!   accepted: Text("The sentiment is neutral.")
//!   rejected: Text("42") because text block is valid JSON but not a JSON object
//!   rejected: ToolCall(read_file) because tool call id must be non-empty
//!
//! Pooled clients: 1 (both tiers share one provider and base URL)
//! ```
//!
//! Stage 3 is the point of the demo: three tier lines for one turn. The fast
//! tier drafted a bare integer, the validator refused it, the slow tier was
//! asked again with the rejection attached, refused again, and the second
//! cascade spent the budget, so the last rejection surfaced as
//! `CucaError::Provider`. A slow tier that answers in prose ends the same turn
//! with a reply instead.
//!
//! The replies, the block counts and the timings depend on the models. The
//! routing decisions in stage 1 and the verdicts in stage 5 depend on neither:
//! both are pure functions of the request and the block.
//!
//! # The latency guard
//!
//! `SwappableModelPair::latency_threshold_ms` is left generous here on
//! purpose. The guard swaps tiers at the first poll at which the fast stream
//! is still `Pending` *after* the deadline, so a tier stream that is only ever
//! woken when a block is ready, like the channel-backed executor below, never
//! offers the guard that poll. The validation cascade is the fallback a demo
//! can trigger on purpose.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `ModelOrchestrator` is a service, not a `CucaPlugin`. It decides which
//! provider a turn reaches and may re-dispatch that turn to a second one
//! mid-stream. `on_request` can only mutate a request, and `on_stream_chunk`
//! sees one block at a time with no way to replace the stream it came from, so
//! no hook can route a turn across two tiers.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use cuca::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};
use cuca::{
    AgentResponseStream, ClientPool, ComplexityEvaluator, CucaError, DraftValidator,
    JsonToolDraftValidator, ModelOrchestrator, SwappableModelPair, TurnExecutor, UnifiedRequest,
};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// Tier lines in dispatch order, shared by both executors.
type TierLog = Arc<Mutex<Vec<String>>>;

/// A `TurnExecutor` that dispatches one tier through a pooled client and
/// records the dispatch.
///
/// `TurnExecutor::execute` is synchronous while `generate_stream` is not, so
/// the turn is spawned and its blocks are forwarded through a channel. That is
/// also what makes the latency guard observable: the returned stream reports
/// `Pending` until the tier produces its first block.
struct TierExecutor {
    tier: &'static str,
    model_id: String,
    base_url: String,
    pool: Arc<ClientPool>,
    log: TierLog,
}

impl TurnExecutor for TierExecutor {
    fn tier_name(&self) -> &'static str {
        self.tier
    }

    fn execute(&self, mut request: UnifiedRequest) -> Result<AgentResponseStream, CucaError> {
        request.model = self.model_id.clone();
        if let Ok(mut log) = self.log.lock() {
            log.push(format!("{} tier -> {}", self.tier, self.model_id));
        }
        let client = self
            .pool
            .get_or_create(&ProviderEndpoint::LlamaCpp, &self.base_url, None)?;
        let (sender, receiver) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            match client.generate_stream(request).await {
                Ok(mut stream) => {
                    while let Some(item) = stream.next().await {
                        if sender.send(item).await.is_err() {
                            break;
                        }
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                }
            }
        });
        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}

/// One drained turn: text, per-kind block counts, and the first error.
struct Turn {
    text: String,
    text_blocks: usize,
    thinking_blocks: usize,
    error: Option<String>,
}

/// Drain a turn, counting `Thinking` blocks instead of printing them: the slow
/// tier emits one per token and would bury the lines this demo is about.
async fn drain(mut stream: AgentResponseStream) -> Turn {
    let mut turn = Turn {
        text: String::new(),
        text_blocks: 0,
        thinking_blocks: 0,
        error: None,
    };
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                turn.text_blocks += 1;
                turn.text.push_str(&text);
            }
            Ok(MessageContentBlock::Thinking { .. }) => turn.thinking_blocks += 1,
            Ok(_) => {}
            Err(error) => {
                turn.error = Some(error.to_string());
                break;
            }
        }
    }
    turn.text = turn.text.trim().to_string();
    turn
}

/// The three requests stage 1 routes: one per routing indicator plus the
/// simple case.
fn requests(model: &str) -> [(&'static str, UnifiedRequest); 3] {
    let tool_turn = UnifiedRequest::new(model)
        .add_user_message("Summarize what the tool returned.")
        .add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "src/lib.rs" }),
            }],
            name: None,
            tool_call_id: None,
        });
    [
        (
            "classify one short sentence",
            UnifiedRequest::new(model)
                .add_system_message("Reply with one word: positive, negative or neutral.")
                .add_user_message("The build finished.")
                .set_max_tokens(512),
        ),
        (
            "four file references",
            UnifiedRequest::new(model)
                .add_system_message("Reply with one file path and nothing else.")
                .add_user_message(
                    "Which of src/sse.rs, src/client.rs, src/request.rs and src/types.rs parses \
                     server-sent events?",
                )
                .set_max_tokens(512),
        ),
        ("one tool-call round trip", tool_turn),
    ]
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and the two tier models come from the environment so the
    // example runs against any OpenAI-compatible server; the defaults target a
    // local llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let fast_model =
        std::env::var("CUCA_FAST_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());
    let slow_model =
        std::env::var("CUCA_SLOW_MODEL").unwrap_or_else(|_| "google/gemma-4-12b-qat".to_string());
    println!("Pair: fast {fast_model}, slow {slow_model}");

    let evaluator = ComplexityEvaluator::default();
    println!(
        "Thresholds: tool depth {}, input tokens {}, file refs {}",
        evaluator.slow_tool_call_depth,
        evaluator.slow_input_tokens,
        evaluator.slow_multi_file_threshold
    );

    println!("\nStage 1: routing, with no network");
    let [
        (fast_label, fast_request),
        (slow_label, slow_request),
        (tool_label, tool_request),
    ] = requests(&fast_model);
    for (label, request) in [
        (fast_label, &fast_request),
        (slow_label, &slow_request),
        (tool_label, &tool_request),
    ] {
        println!(
            "  {:<5} <- {label}",
            format!("{:?}", evaluator.evaluate(request))
        );
    }

    let pool = Arc::new(ClientPool::new());
    let log: TierLog = Arc::new(Mutex::new(Vec::new()));
    // One orchestrator per latency budget: the budget lives on the immutable
    // pair, so stage 4 needs its own.
    let orchestrator = |latency_threshold_ms: u64| {
        let config = SwappableModelPair {
            fast_provider: ProviderEndpoint::LlamaCpp,
            fast_model_id: fast_model.clone(),
            slow_provider: ProviderEndpoint::LlamaCpp,
            slow_model_id: slow_model.clone(),
            latency_threshold_ms,
            fallback_on_tool_error: true,
        };
        let tier = |name: &'static str, model_id: String| {
            Arc::new(TierExecutor {
                tier: name,
                model_id,
                base_url: base_url.clone(),
                pool: Arc::clone(&pool),
                log: Arc::clone(&log),
            }) as Arc<dyn TurnExecutor>
        };
        ModelOrchestrator::with_executors(
            config.clone(),
            Arc::clone(&pool),
            tier("fast", config.fast_model_id.clone()),
            tier("slow", config.slow_model_id.clone()),
        )
    };

    // Every stage below drains one turn and reports which tiers ran; `log` is
    // cleared first so each stage's lines are its own.
    let run = |orchestrator: ModelOrchestrator, request: UnifiedRequest| {
        let log = Arc::clone(&log);
        async move {
            log.lock().expect("tier log must not be poisoned").clear();
            let started = Instant::now();
            let turn = drain(orchestrator.execute_adaptive_turn(request).await?).await;
            for line in log.lock().expect("tier log must not be poisoned").iter() {
                println!("  {line}");
            }
            Ok::<_, CucaError>((turn, started.elapsed().as_millis()))
        }
    };

    println!("\nStage 2: the Fast request, latency budget 60000 ms");
    let (turn, elapsed) = run(orchestrator(60_000), fast_request).await?;
    if let Some(error) = &turn.error {
        println!("\nNo server answered at {base_url}: {error}");
        println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
        return Ok(());
    }
    println!("  reply: {}", turn.text);
    println!(
        "  {elapsed} ms, {} text blocks, {} thinking blocks",
        turn.text_blocks, turn.thinking_blocks
    );

    // Stage 3: the draft the validator refuses. The default
    // `JsonToolDraftValidator` rejects a text block that parses as JSON but is
    // not an object, so a bare integer reply is refused whether it arrives as
    // one block or as one digit per block.
    println!("\nStage 3: a refused draft and the fallback cascade");
    let arithmetic = UnifiedRequest::new(&fast_model)
        .add_system_message("You output one integer and nothing else: no words, no punctuation.")
        .add_user_message("What is 6 times 7?")
        .set_max_tokens(512);
    let (turn, elapsed) = run(orchestrator(60_000), arithmetic).await?;
    match &turn.error {
        Some(error) => println!("  cascade exhausted: {error}"),
        None => println!("  accepted draft, no cascade: {:?}", turn.text),
    }
    println!(
        "  {elapsed} ms, {} text blocks, {} thinking blocks",
        turn.text_blocks, turn.thinking_blocks
    );

    println!("\nStage 4: the Slow request, no draft phase");
    let (turn, elapsed) = run(orchestrator(60_000), slow_request).await?;
    println!("  reply: {}", turn.text);
    println!(
        "  {elapsed} ms, {} text blocks, {} thinking blocks",
        turn.text_blocks, turn.thinking_blocks
    );

    // Stage 5: the verdicts behind stage 3. Every rejected block is appended
    // to the working request as a synthetic tool result before the turn
    // re-routes, so the next tier sees what was refused and why.
    println!("\nStage 5: the draft verdicts behind the cascade");
    let validator = JsonToolDraftValidator;
    for (label, block) in [
        (
            "Text(\"The sentiment is neutral.\")",
            MessageContentBlock::Text("The sentiment is neutral.".into()),
        ),
        ("Text(\"42\")", MessageContentBlock::Text("42".into())),
        (
            "ToolCall(read_file)",
            MessageContentBlock::ToolCall {
                id: String::new(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "src/lib.rs" }),
            },
        ),
    ] {
        match validator.validate(&block) {
            Ok(()) => println!("  accepted: {label}"),
            Err(reason) => println!("  rejected: {label} because {reason}"),
        }
    }

    println!(
        "\nPooled clients: {} (both tiers share one provider and base URL)",
        pool.len()
    );
    Ok(())
}
