//! Live test of the in-process engine's continuous batching: several clients
//! on one [`LocalModel`] at once.
//!
//! The model comes from the environment like `provider_local.rs`
//! (`CUCA_LOCAL_MODEL_PATH`, `CUCA_LOCAL_MODEL` or `CUCA_LOCAL_DOWNLOAD=1`;
//! `CUCA_LOCAL_DEVICE`; `CUCA_REQUIRE_LIVE=1` turns a skip into a failure),
//! and `CUCA_LOCAL_BATCHING` sets `LocalModelConfig::batching` (`BatchingConfig`
//! syntax, e.g. `seqs=4,kv=16384`; `seqs=4` when unset, so the 8 clients
//! below queue behind 4 slots).
//!
//! ```sh
//! CUCA_LOCAL_MODEL=qwen3-0.6b-q8_0 CUCA_REQUIRE_LIVE=1 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local_batch -- --test-threads=1 --nocapture
//! ```
//!
//! Every request is greedy. The scenarios:
//!
//! 1. `shared_prefix_clients`: four prompts that share a long preamble run
//!    serially (the reference), then as 8 concurrent clients (each prompt
//!    twice). On the CPU every reply must equal the serial one bit for bit
//!    (a row does not depend on its neighbours); on a GPU the rows agree only
//!    up to float rounding, so a flipped near-tie is reported, not failed.
//!    Some forward step must have carried several rows
//!    (`ServingStats::batched_rows > steps`).
//! 2. The same 8 clients again: every prompt now starts from the prefix
//!    store (`read_tokens >= 16 * floor((prompt - 1) / 16)` for an
//!    attention-only model, more than zero for a hybrid one, which resumes
//!    from its snapshot).
//! 3. `aggregate_throughput`: four distinct prompts one after the other and
//!    then concurrently, printed as `BATCH|<scenario>|<tokens per second>`.

#![cfg(feature = "provider-local")]

use std::time::Instant;

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    CucaClient, DownloadProgressCallback, LocalModel, LocalModelConfig, ModelSource, UnifiedRequest,
};
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

static MODEL: OnceCell<Option<(LocalModel, String)>> = OnceCell::const_new();

/// The model to load, from the environment; `None` when none is configured.
fn model_source() -> Option<ModelSource> {
    if let Some(path) = std::env::var_os("CUCA_LOCAL_MODEL_PATH") {
        return Some(ModelSource::File(path.into()));
    }
    match std::env::var("CUCA_LOCAL_MODEL") {
        Ok(name) if !name.is_empty() => Some(
            name.parse()
                .unwrap_or_else(|e| panic!("CUCA_LOCAL_MODEL={name:?}: {e}")),
        ),
        _ if std::env::var("CUCA_LOCAL_DOWNLOAD").as_deref() == Ok("1") => {
            Some(ModelSource::default_model())
        }
        _ => None,
    }
}

fn env_parsed<T: std::str::FromStr>(key: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    std::env::var(key)
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|e| panic!("{key}={s:?}: {e}")))
}

/// The shared model and a label for it, loaded on first use.
async fn loaded() -> Option<(LocalModel, String)> {
    MODEL
        .get_or_init(|| async {
            let source = model_source()?;
            let label = match &source {
                ModelSource::File(path) => path.display().to_string(),
                ModelSource::HuggingFace { repo, file, .. } => format!("{repo}:{file}"),
                other => format!("{other:?}"),
            };
            let mut config = LocalModelConfig {
                source,
                device: env_parsed("CUCA_LOCAL_DEVICE").unwrap_or_default(),
                on_download_progress: Some(DownloadProgressCallback::new(|p| {
                    eprint!(
                        "\rdownloading: {} / {:?} bytes",
                        p.downloaded_bytes, p.total_bytes
                    );
                })),
                ..Default::default()
            };
            if let Some(batching) = env_parsed("CUCA_LOCAL_BATCHING") {
                config.batching = batching;
            }
            let model = LocalModel::load(config)
                .await
                .unwrap_or_else(|e| panic!("local model must load: {e}"));
            Some((model, label))
        })
        .await
        .clone()
}

/// Skip (or, under `CUCA_REQUIRE_LIVE=1`, fail) when no model is configured.
fn skip_or_fail() {
    let reason = "SKIP: local model not available: set CUCA_LOCAL_MODEL_PATH, \
                  CUCA_LOCAL_MODEL or CUCA_LOCAL_DOWNLOAD=1";
    if std::env::var("CUCA_REQUIRE_LIVE").as_deref() == Ok("1") {
        panic!("{reason}");
    }
    println!("{reason}");
}

/// What one request reported.
#[derive(Debug, Clone, PartialEq)]
struct Reply {
    /// Reasoning and text, in order.
    output: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    read_tokens: u32,
}

/// One greedy request: `question` after `preamble`, at most `max_tokens`.
async fn ask(model: &LocalModel, preamble: &str, question: &str, max_tokens: u32) -> Reply {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model.clone())
        .build()
        .expect("client builds");
    let request = UnifiedRequest::new("local")
        .add_user_message(if preamble.is_empty() {
            question.to_string()
        } else {
            format!("{preamble}\n\n{question}")
        })
        .set_max_tokens(max_tokens)
        .set_temperature(0.0);
    let (mut stream, handle) = client
        .generate_stream_with_response(request)
        .await
        .expect("request dispatches");
    let mut output = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            MessageContentBlock::Text(t) => output.push_str(&t),
            MessageContentBlock::Thinking { reasoning, .. } => output.push_str(&reasoning),
            _ => {}
        }
    }
    let response = handle.take().expect("terminal response");
    let usage = response.usage.expect("usage");
    Reply {
        output,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        read_tokens: response
            .prompt_cache_usage
            .expect("prompt-cache usage")
            .read_tokens,
    }
}

/// Run every request at once; the replies in request order.
async fn ask_all(model: &LocalModel, requests: &[(String, String)], max_tokens: u32) -> Vec<Reply> {
    let mut set = tokio::task::JoinSet::new();
    for (i, (preamble, question)) in requests.iter().cloned().enumerate() {
        let model = model.clone();
        set.spawn(async move { (i, ask(&model, &preamble, &question, max_tokens).await) });
    }
    let mut replies = vec![None; requests.len()];
    while let Some(done) = set.join_next().await {
        let (i, reply) = done.expect("client task");
        replies[i] = Some(reply);
    }
    replies.into_iter().map(|r| r.expect("reply")).collect()
}

/// A preamble of some 200 tokens that every prompt of a scenario shares.
fn preamble(nonce: &str) -> String {
    format!(
        "Session {nonce}. You are a careful geography assistant. Answer with a single short \
         sentence and never add explanations. The user may ask about the capital city of a \
         country, about a famous river, or about a mountain range. Keep your answers factual, \
         neutral and brief. If you do not know the answer, say so plainly. Remember that \
         capital cities are not always the largest city of their country: Canberra is not \
         Sydney, Ottawa is not Toronto, Brasilia is not Rio de Janeiro, Bern is not Zurich. \
         Rivers flow to the sea: the Danube flows to the Black Sea, the Rhine to the North \
         Sea, the Nile to the Mediterranean Sea. Mountain ranges form natural borders: the \
         Pyrenees between France and Spain, the Alps across several countries. Use plain \
         words. Do not use lists. Do not use markdown. Do not repeat the question."
    )
}

const QUESTIONS: [&str; 4] = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "Which sea does the Danube flow to?",
    "Which mountains lie between France and Spain?",
];

/// Whether the device computes rows independently of their batch (the CPU).
fn rows_are_exact() -> bool {
    std::env::var("CUCA_LOCAL_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .is_none_or(|d| d.eq_ignore_ascii_case("cpu"))
}

#[tokio::test]
async fn shared_prefix_clients() {
    let Some((model, label)) = loaded().await else {
        skip_or_fail();
        return;
    };
    let lower = label.to_lowercase();
    let hybrid = lower.contains("qwen3.5") || lower.contains("qwen35");
    let shared = preamble("alpha");
    let max_tokens = 24;
    // A warm-up request compiles kernels and touches the weights.
    ask(&model, "warm-up", "Say hi.", 4).await;

    // The serial reference.
    let mut serial = Vec::new();
    for q in QUESTIONS {
        serial.push(ask(&model, &shared, q, max_tokens).await);
    }

    // Eight concurrent clients: every prompt twice.
    let before = model.serving_stats();
    let requests: Vec<(String, String)> = (0..8)
        .map(|i| (shared.clone(), QUESTIONS[i % 4].to_string()))
        .collect();
    let start = Instant::now();
    let replies = ask_all(&model, &requests, max_tokens).await;
    let wall = start.elapsed().as_secs_f64();
    let after = model.serving_stats();
    let rows = after.batched_rows - before.batched_rows;
    let steps = after.steps - before.steps;
    println!(
        "BATCH|shared_prefix_clients|steps {steps} rows {rows} preemptions {} wall {wall:.2}s",
        after.preemptions - before.preemptions
    );
    assert!(
        rows > steps,
        "steps carried several rows: {rows} rows in {steps} steps ({after:?})"
    );

    let mut flipped = 0;
    for (i, reply) in replies.iter().enumerate() {
        let reference = &serial[i % 4];
        if reply.output != reference.output {
            flipped += 1;
            assert!(
                !rows_are_exact(),
                "client {i}: the batched reply differs from the serial one\n  batched: {:?}\n  serial:  {:?}",
                reply.output,
                reference.output
            );
        }
        assert_eq!(reply.prompt_tokens, reference.prompt_tokens, "client {i}");
    }
    for i in 0..4 {
        assert!(
            replies[i].output == replies[i + 4].output || !rows_are_exact(),
            "the pair {i} of identical prompts differs"
        );
    }
    println!(
        "BATCH|shared_prefix_equal|{}/8 equal to serial",
        8 - flipped
    );

    // The second round starts from the prefix store.
    let again = ask_all(&model, &requests, max_tokens).await;
    for (i, reply) in again.iter().enumerate() {
        let whole_blocks = 16 * ((reply.prompt_tokens - 1) / 16);
        if hybrid {
            assert!(reply.read_tokens > 0, "client {i}: {reply:?}");
        } else {
            assert!(
                reply.read_tokens >= whole_blocks,
                "client {i}: read {} of {} prompt tokens, expected at least {whole_blocks}",
                reply.read_tokens,
                reply.prompt_tokens
            );
        }
        if rows_are_exact() {
            assert_eq!(reply.output, replies[i].output, "client {i}");
        }
    }
    println!(
        "BATCH|prefix_hits|{}",
        again
            .iter()
            .map(|r| format!("{}/{}", r.read_tokens, r.prompt_tokens))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("BATCH|stats|{:?}", model.serving_stats());
}

#[tokio::test]
async fn aggregate_throughput() {
    let Some((model, _)) = loaded().await else {
        skip_or_fail();
        return;
    };
    // Short prompts and long replies, so decode (not prefill) dominates.
    let max_tokens = 96;
    ask(&model, "", "Say hi.", 4).await;
    let questions: Vec<String> = [3, 41, 77, 120]
        .iter()
        .map(|n| {
            format!("Count upward from {n} in steps of one, separated by commas, without stopping.")
        })
        .collect();

    // Four distinct prompts, one after the other.
    let start = Instant::now();
    let mut serial_tokens = 0;
    for q in &questions {
        serial_tokens += ask(&model, "", q, max_tokens).await.completion_tokens;
    }
    let serial_rate = f64::from(serial_tokens) / start.elapsed().as_secs_f64();

    // The same four at once.
    let requests: Vec<(String, String)> = questions
        .iter()
        .map(|q| (String::new(), q.clone()))
        .collect();
    let before = model.serving_stats();
    let start = Instant::now();
    let replies = ask_all(&model, &requests, max_tokens).await;
    let tokens: u32 = replies.iter().map(|r| r.completion_tokens).sum();
    let rate = f64::from(tokens) / start.elapsed().as_secs_f64();
    let after = model.serving_stats();
    println!("BATCH|single_sequence|{serial_rate:.2} tok/s ({serial_tokens} tokens)");
    println!(
        "BATCH|aggregate_4|{rate:.2} tok/s ({tokens} tokens, {} rows in {} steps)",
        after.batched_rows - before.batched_rows,
        after.steps - before.steps
    );
    println!("BATCH|speedup|{:.2}x", rate / serial_rate);
}
