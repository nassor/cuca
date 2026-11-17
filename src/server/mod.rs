//! The `cuca` command's implementation (feature `server`, which enables
//! `provider-local`; add `provider-local-vulkan` and friends for a GPU):
//!
//! * `cuca run` ([`run`]): runs what the KDL file ([`CucaConfig`], module
//!   [`config`]) holds. A `serve` section is an OpenAI-compatible HTTP server
//!   over the in-process local engine, described below. A `balance` section
//!   is a load balancer over several servers (module [`balancer`]) with
//!   prefix-affinity routing, health checks and a stats API of its own. With
//!   both, the balancer also routes to the local models.
//! * `cuca tui` (feature `server-tui`, module `tui`): a terminal view of a
//!   server or a balancer that follows the stats stream.
//!
//! ```text
//! cuca run --config cuca.kdl      # a `serve` section with `stats_api #true`
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
//!   machine's CPU, memory, disk and GPU gauges ([`crate::sysmon`]).
//! * `GET /cuca/v1/stats` and `/cuca/v1/stats/stream` (opt-in with
//!   `stats_api #true`): one [`stats::StatsSnapshot`] as JSON, or one per second
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
pub(crate) mod kv;
pub mod latency;
pub mod metrics;
pub mod openai;
pub mod registry;
pub(crate) mod socket;
pub mod state;
pub mod stats;
pub mod telemetry;
#[cfg(feature = "server-tui")]
pub mod tui;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{oneshot, watch};

pub use balancer::BalancerConfig;
pub use config::{CucaConfig, MAX_MODELS, ModelSpec, ServerConfig};
pub use error::ServerError;
pub use telemetry::{LogConfig, LogFormat, init as init_logging};

use self::registry::ModelRegistry;
use self::state::ServerState;
use crate::sysmon::SystemSampler;

/// How long open connections get to finish after a shutdown signal.
pub(crate) const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

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

/// `listener` with `TCP_NODELAY` set on every accepted connection. axum
/// leaves Nagle's algorithm on, which holds back the small trailing segment
/// of a response (a layer-stage reply, one SSE event) until the peer
/// acknowledges the one before it.
pub(crate) fn nodelay(
    listener: tokio::net::TcpListener,
) -> impl axum::serve::Listener<Io = tokio::net::TcpStream, Addr = SocketAddr> {
    use axum::serve::ListenerExt as _;
    listener.tap_io(|stream| {
        if let Err(e) = stream.set_nodelay(true) {
            tracing::debug!("cannot set TCP_NODELAY on an accepted connection: {e}");
        }
    })
}

pub(crate) fn join_error(what: &str, error: &tokio::task::JoinError) -> ServerError {
    ServerError::Io(format!("{what} task failed: {error}"))
}

/// The model server until `shutdown` resolves or a model fails to load.
///
/// Binds `config.listen` first (so a port in use fails at once), loads the
/// models concurrently while already answering `/health` with 503, and shuts
/// down gracefully: open responses get a five second grace period to finish.
async fn serve_until(
    config: ServerConfig,
    shutdown: impl Future<Output = ()>,
) -> Result<(), ServerError> {
    config.validate()?;
    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|e| ServerError::Io(format!("cannot bind {}: {e}", config.listen)))?;
    let addr = listener
        .local_addr()
        .map_err(|e| ServerError::Io(format!("cannot read the bound address: {e}")))?;

    let peer_client = state::peer_client()?;
    let system = SystemSampler::start(collector::SAMPLE_INTERVAL);
    let (registry, mut loader) = ModelRegistry::start(config.models.clone());
    let state = Arc::new(ServerState::new(
        &config,
        addr,
        Arc::clone(&registry),
        system,
        peer_client,
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
        let (served, make_service) = socket::serving(listener, app);
        axum::serve(served, make_service)
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .await
    });

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
    stop_models(&registry).await;
    result
}

/// Stop the engine of every model that is ready, in parallel, and wait for it:
/// a model on a GPU gives its device memory back as its engine drops, and a
/// process that exits while that is in flight aborts (ROCm). Handles that
/// outlive the HTTP server (a connection task still streaming a reply, a
/// layer-stage stream's thread) would otherwise keep the engine running, or
/// drop the last clone before its readers are gone, until the process exits.
async fn stop_models(registry: &ModelRegistry) {
    let mut stops = tokio::task::JoinSet::new();
    for slot in registry.slots() {
        if let registry::SlotStatus::Ready(loaded) = slot.status() {
            let name = slot.name.clone();
            stops.spawn(async move {
                if !loaded.model.shutdown().await {
                    tracing::warn!(model = %name, "the model's engine did not stop in time");
                }
            });
        }
    }
    while stops.join_next().await.is_some() {}
}

/// Make `balance` route to the `serve` section of the same process: its
/// listener, on the loopback address when it binds every interface, becomes
/// the first upstream. Validates both sections, so nothing starts on an
/// invalid file.
fn colocate(serve: &ServerConfig, balance: &mut BalancerConfig) -> Result<(), ServerError> {
    if serve.listen.port() == 0 {
        return Err(ServerError::Config(
            "a `serve` section beside a `balance` one needs a fixed port in `serve.listen`: the balancer routes to it".into(),
        ));
    }
    if serve.listen == balance.listen {
        return Err(ServerError::Config(format!(
            "`serve.listen` and `balance.listen` are both {}: give the balancer another port",
            serve.listen
        )));
    }
    if serve.api_key != balance.upstream_api_key {
        return Err(ServerError::Config(
            "`serve.api_key` and `balance.upstream_api_key` differ: the balancer could not reach the local server".into(),
        ));
    }
    let ip = match serve.listen.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    balance.add_local_upstream(SocketAddr::new(ip, serve.listen.port()));
    serve.validate()?;
    balance.resolve_roles()?;
    Ok(())
}

/// Run the sections of the configuration until a shutdown signal (Ctrl-C,
/// SIGTERM):
///
/// * `serve` alone: the model server, which also stops when a model fails
///   to load. It binds `serve.listen` first (a port in use fails at once),
///   loads the models concurrently while answering `/health` with 503, and
///   gives open responses five seconds to finish on shutdown.
/// * `balance` alone: a balancer over `balance.upstreams`; `/health` is 503
///   until an upstream has answered.
/// * both: the two on their own listeners, the server's `listen` address
///   (the loopback address when it binds every interface) first among the
///   balancer's upstreams. When either stops, the other is shut down
///   gracefully and the first error is returned. A cluster runs `cuca run`
///   on every machine, and each machine's `balance.upstreams` lists the
///   other machines' `serve` addresses, never a balancer's, so a request is
///   forwarded at most once.
///
/// The `log` section is not applied here: call [`init_logging`].
///
/// # Errors
///
/// [`ServerError::Config`] without either section, for an invalid section,
/// and with both sections for a `serve.listen` with port 0, the same address
/// for both listeners, or a `balance.upstream_api_key` that differs from
/// `serve.api_key`; [`ServerError::Io`] when an address cannot be bound or
/// an HTTP server fails; [`ServerError::ModelLoad`] when a model fails to
/// load.
pub async fn run(
    serve: Option<ServerConfig>,
    balance: Option<BalancerConfig>,
) -> Result<(), ServerError> {
    match (serve, balance) {
        (None, None) => Err(ServerError::Config(
            "the configuration has neither a `serve` nor a `balance` section".into(),
        )),
        (Some(serve), None) => serve_until(serve, shutdown_signal()).await,
        (None, Some(balance)) => balancer::balance_until(balance, shutdown_signal()).await,
        (Some(serve), Some(mut balance)) => {
            colocate(&serve, &mut balance)?;
            serve_and_balance(serve, balance).await
        }
    }
}

/// Both sections of [`run`], each stopping the other when it ends.
async fn serve_and_balance(
    serve: ServerConfig,
    balance: BalancerConfig,
) -> Result<(), ServerError> {
    let (stop_tx, stop_rx) = watch::channel(false);
    let stopped = |mut rx: watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|stop| *stop).await;
    };
    let serve_fut = serve_until(serve, stopped(stop_rx.clone()));
    let balance_fut = balancer::balance_until(balance, stopped(stop_rx));
    tokio::pin!(serve_fut, balance_fut);
    tokio::select! {
        () = shutdown_signal() => {
            let _ = stop_tx.send(true);
            let (served, balanced) = tokio::join!(serve_fut, balance_fut);
            served.and(balanced)
        }
        served = &mut serve_fut => {
            let _ = stop_tx.send(true);
            served.and(balance_fut.await)
        }
        balanced = &mut balance_fut => {
            let _ = stop_tx.send(true);
            balanced.and(serve_fut.await)
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::serve::Listener as _;

    use super::*;
    use crate::{LocalModelConfig, ModelSource};

    #[tokio::test]
    async fn accepted_connections_have_nodelay() {
        let bound = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = bound.local_addr().unwrap();
        let mut listener = nodelay(bound);
        let _client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (stream, _) = listener.accept().await;
        assert!(stream.nodelay().unwrap());
    }

    fn serve_at(listen: &str) -> ServerConfig {
        let model = ModelSpec::new(
            Some("m".into()),
            LocalModelConfig {
                source: "qwen3-0.6b-q8_0".parse().unwrap(),
                ..Default::default()
            },
        );
        ServerConfig::new(listen.parse().unwrap(), vec![model])
    }

    fn balance_at(listen: &str, upstreams: &[&str]) -> BalancerConfig {
        BalancerConfig::new(
            listen.parse().unwrap(),
            upstreams.iter().map(|u| (*u).to_string()).collect(),
        )
    }

    #[test]
    fn colocate_puts_the_local_server_first_once() {
        let serve = serve_at("0.0.0.0:8000");
        let mut balance = balance_at("0.0.0.0:8080", &["http://10.0.0.2:8000"]);
        colocate(&serve, &mut balance).unwrap();
        assert_eq!(
            balance.upstreams,
            ["http://127.0.0.1:8000", "http://10.0.0.2:8000"]
        );
        let mut listed = balance_at("0.0.0.0:8080", &["http://127.0.0.1:8000/"]);
        colocate(&serve, &mut listed).unwrap();
        assert_eq!(listed.upstreams, ["http://127.0.0.1:8000/"]);
        let mut v6 = balance_at("[::]:8080", &[]);
        colocate(&serve_at("[::]:8000"), &mut v6).unwrap();
        assert_eq!(v6.upstreams, ["http://[::1]:8000"]);
    }

    #[test]
    fn colocate_rejects_what_the_balancer_could_not_reach() {
        let message = |serve: &ServerConfig, balance: &mut BalancerConfig| {
            colocate(serve, balance).unwrap_err().to_string()
        };
        let err = message(
            &serve_at("127.0.0.1:0"),
            &mut balance_at("127.0.0.1:8080", &[]),
        );
        assert!(
            err.contains("needs a fixed port in `serve.listen`"),
            "{err}"
        );
        let err = message(
            &serve_at("127.0.0.1:8000"),
            &mut balance_at("127.0.0.1:8000", &[]),
        );
        assert!(err.contains("are both 127.0.0.1:8000"), "{err}");
        let mut keyed = serve_at("127.0.0.1:8000");
        keyed.api_key = Some("k".into());
        let err = message(&keyed, &mut balance_at("127.0.0.1:8080", &[]));
        assert!(err.contains("`balance.upstream_api_key` differ"), "{err}");
    }

    /// A model source that fails to load at once, with no network.
    fn failing_serve() -> ServerConfig {
        let dir = std::env::temp_dir().join(format!("cuca-run-test-{}", std::process::id()));
        let mut serve = serve_at("127.0.0.1:1");
        serve.models[0].config.source = ModelSource::File(dir.join("missing.gguf"));
        serve.listen.set_port({
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        });
        serve
    }

    async fn run_briefly(
        serve: Option<ServerConfig>,
        balance: Option<BalancerConfig>,
    ) -> Result<(), ServerError> {
        tokio::time::timeout(Duration::from_secs(30), run(serve, balance))
            .await
            .expect("run stops on its own")
    }

    #[tokio::test]
    async fn run_starts_what_the_configuration_holds() {
        let err = run_briefly(None, None).await.unwrap_err().to_string();
        assert!(err.contains("neither a `serve` nor a `balance`"), "{err}");
        match run_briefly(Some(failing_serve()), None).await {
            Err(ServerError::ModelLoad { model, .. }) => assert_eq!(model, "m"),
            other => panic!("`serve` alone runs the server, got {other:?}"),
        }
        let err = run_briefly(None, Some(balance_at("127.0.0.1:0", &[])))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no upstream"),
            "`balance` alone needs upstreams: {err}"
        );
    }

    #[tokio::test]
    async fn a_model_that_fails_to_load_stops_the_balancer_too() {
        let balance = balance_at("127.0.0.1:0", &[]);
        match run_briefly(Some(failing_serve()), Some(balance)).await {
            Err(ServerError::ModelLoad { model, .. }) => assert_eq!(model, "m"),
            other => panic!("expected a model load error, got {other:?}"),
        }
    }
}
