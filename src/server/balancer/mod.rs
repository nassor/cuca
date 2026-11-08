//! `cuca balance`: a load balancer in front of `cuca serve` instances.
//!
//! It speaks the same OpenAI-compatible API, routes each
//! `POST /v1/chat/completions` to one upstream and streams the answer back
//! unchanged. See [`proxy`] for the routes and the proxying rules,
//! [`routing`] for the policies (prefix affinity by default), [`upstream`] for
//! health checking, [`snapshot`] and [`diagnosis`] for the stats API
//! (`role: balancer`, the upstreams' snapshots nested), and [`metrics`] for
//! `/metrics`.
//!
//! ```text
//! cuca balance --upstream http://10.0.0.1:8000 --upstream http://10.0.0.2:8000 --stats-api
//! ```
//!
//! # Prefix affinity
//!
//! Each upstream caches the K/V of the prompts it served. The balancer hashes
//! the model and the prompt's prefix (the system messages and the first user
//! message, at most 4 KiB of text) with rendezvous hashing over the healthy
//! upstreams serving the model, so a conversation keeps reaching the upstream
//! that holds its prefix, and when an upstream joins or leaves only the keys
//! it wins or held move. A target whose load score exceeds
//! `2 * min_score + 2` is passed over for the least-loaded upstream (counted
//! in `cuca_balancer_affinity_fallbacks_total`); the score is the balancer's
//! in-flight requests to the upstream plus the requests it reports waiting
//! plus its running fraction of `max_sequences`.
//!
//! # Bounds
//!
//! At most [`MAX_UPSTREAMS`] upstreams; per upstream one stats snapshot, one
//! model list, a 1,024-record latency ring and fixed counters and
//! histograms; request bodies up to [`proxy::MAX_BODY_BYTES`], held only
//! while the request is routed and retried (at most [`proxy::MAX_RETRIES`]
//! extra attempts); response bodies are never buffered (except a 503 that
//! will be retried); at most [`proxy::MAX_STATS_STREAMS`] stats streams.

pub mod config;
pub mod diagnosis;
pub mod metrics;
pub mod proxy;
pub mod routing;
pub mod snapshot;
pub mod state;
pub mod upstream;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;

pub use config::{BalancerArgs, BalancerConfig, MAX_UPSTREAMS};
pub use routing::Policy;

use self::state::BalancerState;
use self::upstream::{Upstream, spawn_poller};
use super::error::ServerError;
use super::sysmon::SystemSampler;

/// How long open connections get to finish after a shutdown signal.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Run the balancer until it is asked to stop (Ctrl-C, SIGTERM).
///
/// Binds `config.listen`, starts one poller per upstream and answers
/// immediately: `/health` is 503 until an upstream has answered.
///
/// # Errors
///
/// [`ServerError::Config`] for an invalid configuration,
/// [`ServerError::Io`] when the address cannot be bound or the HTTP server
/// fails.
pub async fn balance(config: BalancerConfig) -> Result<(), ServerError> {
    let resolved = config.resolve_upstreams()?;
    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|e| ServerError::Io(format!("cannot bind {}: {e}", config.listen)))?;
    let addr = listener
        .local_addr()
        .map_err(|e| ServerError::Io(format!("cannot read the bound address: {e}")))?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| ServerError::Io(format!("cannot build the HTTP client: {e}")))?;

    let upstreams: Vec<Arc<Upstream>> = resolved
        .into_iter()
        .map(|(id, url)| Arc::new(Upstream::new(&id, url, config.latency_window)))
        .collect();
    let upstream_key: Option<Arc<str>> = config.upstream_api_key.as_deref().map(Arc::from);
    let pollers: Vec<_> = upstreams
        .iter()
        .map(|u| {
            spawn_poller(
                Arc::clone(u),
                client.clone(),
                upstream_key.clone(),
                config.health_interval,
            )
        })
        .collect();
    let state = Arc::new(BalancerState::new(
        &config,
        addr,
        upstreams,
        client,
        SystemSampler::start(Duration::from_secs(1)),
    ));

    tracing::info!(
        listen = %addr,
        upstreams = state.upstreams.len(),
        policy = config.policy.as_str(),
        stats_api = config.stats_api,
        auth = config.api_key.is_some(),
        version = env!("CARGO_PKG_VERSION"),
        "cuca balancer listening; /health answers 503 until an upstream is healthy"
    );

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let app = proxy::router(Arc::clone(&state));
    let mut server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .await
    });
    let result = tokio::select! {
        () = super::shutdown_signal() => {
            tracing::info!("shutdown signal received");
            Ok(())
        }
        joined = &mut server => match joined {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(ServerError::Io(format!("HTTP server failed: {e}"))),
            Err(e) => Err(ServerError::Io(format!("HTTP server task failed: {e}"))),
        },
    };
    let _ = stop_tx.send(());
    for poller in &pollers {
        poller.abort();
    }
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

#[cfg(test)]
mod tests;
