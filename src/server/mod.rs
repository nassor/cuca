//! The `cuca` command's implementation (feature `server`, which enables
//! `provider-local`; add `provider-local-vulkan` and friends for a GPU):
//!
//! * `cuca serve` ([`serve`]): an OpenAI-compatible HTTP server over the
//!   in-process local engine, described below.
//! * `cuca balance` ([`balance`], module [`balancer`]): a load balancer over
//!   several servers with prefix-affinity routing, health checks and a stats
//!   API of its own.
//! * `cuca tui` (feature `server-tui`, module `tui`): a terminal view of a
//!   server or a balancer that follows the stats stream.
//!
//! ```text
//! cuca serve --model qwen3-0.6b-q8_0 --stats-api
//! curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
//! ```
//!
//! # What it serves
//!
//! The routes are listed in [`http`]; the request and response wire format,
//! with what is supported and what is rejected, in [`openai`]. The server
//! loads one or more models concurrently at startup ([`ServerConfig`],
//! [`ModelSpec`]) and answers `/health` with 503 until every model is ready
//! (llama.cpp's behaviour); a model that fails to load stops the server.
//! The engine has no raw-prompt entry, so `/v1/completions` is not served.
//!
//! # Telemetry
//!
//! * `GET /metrics`: Prometheus text ([`metrics`]), per model: request counts
//!   by status, time-to-first-token, inter-token, end-to-end and queue
//!   histograms, token counters, the engine's gauges and counters (running,
//!   waiting, K/V usage, prefix hits, preemptions, steps, rows) and the
//!   machine's CPU, memory, disk and GPU gauges ([`sysmon`]).
//! * `GET /cuca/v1/stats` and `/cuca/v1/stats/stream` (opt-in with
//!   `--stats-api`): one [`stats::StatsSnapshot`] as JSON, or one per second
//!   as server-sent events: per-model engine rates, windowed latency
//!   percentiles ([`latency`]), the machine, and a ranked list of reasons the
//!   latency is what it is ([`diagnosis`]). The types in [`stats`] are the
//!   contract a balancer or a terminal UI reads.
//! * Logs ([`telemetry`]): every request has an `x-request-id` (an incoming
//!   one is honoured), a tracing span (`http`) and one `info` line when it
//!   finishes with its model, token counts and timings.
//!
//! # Bounds
//!
//! Nothing grows with traffic: request bodies are capped at
//! [`http::MAX_BODY_BYTES`]; the latency window is a ring of at most
//! [`latency::MAX_RECORDS`] records inside
//! [`ServerConfig::latency_window`] (60 s by default); the stats sampler keeps
//! [`collector::RING`] one-second samples per model and for the machine; the
//! metrics are a fixed set of atomics (the only labels are the model, the
//! status code and the GPU index); at most [`http::MAX_STATS_STREAMS`] stats
//! streams are open; a chat stream buffers [`http::SSE_BUFFER`] events and a
//! slow reader stalls only its own request; at most [`MAX_MODELS`] models are
//! loaded; the engine's own bounds (sequences, K/V pool, 16 queued jobs)
//! apply to requests, whose callers wait when the queue is full.

pub mod balancer;
pub mod collector;
pub mod config;
pub mod diagnosis;
pub mod error;
pub mod http;
pub mod latency;
pub mod metrics;
pub mod openai;
pub mod registry;
pub mod state;
pub mod stats;
pub mod sysmon;
pub mod telemetry;
#[cfg(feature = "server-tui")]
pub mod tui;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;

pub use balancer::{BalancerArgs, BalancerConfig, balance};
pub use config::{
    MAX_MODELS, ModelOptions, ModelSpec, ServerConfig, collect_models, load_config_file,
};
pub use error::ServerError;
pub use telemetry::{LogFormat, init as init_logging};

use self::registry::ModelRegistry;
use self::state::ServerState;
use self::sysmon::SystemSampler;

/// How long open connections get to finish after a shutdown signal.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Resolve when the process is asked to stop: Ctrl-C, or SIGTERM on Unix.
pub(crate) async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                term.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

fn join_error(what: &str, error: &tokio::task::JoinError) -> ServerError {
    ServerError::Io(format!("{what} task failed: {error}"))
}

/// Run the server until it is asked to stop (Ctrl-C, SIGTERM) or a model
/// fails to load.
///
/// Binds `config.listen` first (so a port in use fails at once), loads the
/// models concurrently while already answering `/health` with 503, and shuts
/// down gracefully: open responses get a five second grace period to finish.
///
/// # Errors
///
/// [`ServerError::Config`] for an invalid configuration,
/// [`ServerError::Io`] when the address cannot be bound or the HTTP server
/// fails, [`ServerError::ModelLoad`] when a model fails to load.
pub async fn serve(config: ServerConfig) -> Result<(), ServerError> {
    config.validate()?;
    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|e| ServerError::Io(format!("cannot bind {}: {e}", config.listen)))?;
    let addr = listener
        .local_addr()
        .map_err(|e| ServerError::Io(format!("cannot read the bound address: {e}")))?;

    let system = SystemSampler::start(collector::SAMPLE_INTERVAL);
    let (registry, mut loader) = ModelRegistry::start(config.models.clone());
    let state = Arc::new(ServerState::new(
        &config,
        addr,
        Arc::clone(&registry),
        system,
    ));
    let sampler = collector::run_sampler(Arc::clone(&state));

    tracing::info!(
        listen = %addr,
        models = state.registry.slots().len(),
        stats_api = config.stats_api,
        auth = config.api_key.is_some(),
        version = env!("CARGO_PKG_VERSION"),
        "cuca server listening; /health answers 503 until every model is loaded"
    );

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let app = http::router(Arc::clone(&state));
    let mut server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .await
    });

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    let mut loading = true;
    let result = loop {
        tokio::select! {
            () = &mut shutdown => {
                tracing::info!("shutdown signal received");
                break Ok(());
            }
            joined = &mut server => {
                break match joined {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(ServerError::Io(format!("HTTP server failed: {e}"))),
                    Err(e) => Err(join_error("HTTP server", &e)),
                };
            }
            joined = &mut loader, if loading => {
                loading = false;
                match joined {
                    Ok(Ok(())) => tracing::info!("every model is ready; serving"),
                    Ok(Err(e)) => break Err(e),
                    Err(e) => break Err(join_error("model loader", &e)),
                }
            }
        }
    };

    let _ = stop_tx.send(());
    sampler.abort();
    if !server.is_finished()
        && tokio::time::timeout(SHUTDOWN_GRACE, &mut server)
            .await
            .is_err()
    {
        tracing::warn!("open connections did not finish in time; closing them");
        server.abort();
    }
    result
}
