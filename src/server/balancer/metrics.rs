//! The balancer's Prometheus exposition (`GET /metrics`).
//!
//! Labels: the upstream id (at most
//! [`MAX_UPSTREAMS`](super::config::MAX_UPSTREAMS) values), the status code,
//! the routing decision and the GPU index: a fixed set, whatever the
//! traffic.

use std::sync::atomic::Ordering;
use std::time::Instant;

use super::state::{BalancerState, DECISIONS};
use crate::server::metrics::{Exposition, escape, render_system};

/// Render the balancer's series, then its host's.
pub fn render(state: &BalancerState) -> String {
    let now = Instant::now();
    let mut x = Exposition::new();
    x.family(
        "cuca_uptime_seconds",
        "gauge",
        "Seconds since the balancer started.",
    );
    x.sample(
        "cuca_uptime_seconds",
        "",
        state.started.elapsed().as_secs_f64(),
    );
    x.family(
        "cuca_http_requests_total",
        "counter",
        "HTTP responses by status code, counted when the headers are sent.",
    );
    state
        .metrics
        .http
        .render_into(&mut x.out, "cuca_http_requests_total", "");
    x.family(
        "cuca_rejected_requests_total",
        "counter",
        "Chat requests refused before any upstream was tried, by status code.",
    );
    state
        .metrics
        .rejected
        .render_into(&mut x.out, "cuca_rejected_requests_total", "");

    x.family(
        "cuca_balancer_retries_total",
        "counter",
        "Attempts repeated on another upstream.",
    );
    x.sample(
        "cuca_balancer_retries_total",
        "",
        state.metrics.retries.load(Ordering::Relaxed),
    );
    x.family(
        "cuca_balancer_affinity_fallbacks_total",
        "counter",
        "Prefix-affinity targets passed over for being too loaded.",
    );
    x.sample(
        "cuca_balancer_affinity_fallbacks_total",
        "",
        state.metrics.fallbacks.load(Ordering::Relaxed),
    );
    x.family(
        "cuca_balancer_decisions_total",
        "counter",
        "Routing decisions by kind.",
    );
    for (decision, counter) in DECISIONS.iter().zip(&state.metrics.decisions) {
        x.sample(
            "cuca_balancer_decisions_total",
            &format!("decision=\"{}\"", decision.as_str()),
            counter.load(Ordering::Relaxed),
        );
    }
    x.family(
        "cuca_balancer_healthy_upstreams",
        "gauge",
        "Upstreams that take requests.",
    );
    x.sample("cuca_balancer_healthy_upstreams", "", state.healthy_count());

    let label =
        |u: &crate::server::balancer::upstream::Upstream| format!("upstream=\"{}\"", escape(&u.id));
    x.family(
        "cuca_balancer_upstream_healthy",
        "gauge",
        "1 when the upstream takes requests.",
    );
    for u in &state.upstreams {
        x.sample(
            "cuca_balancer_upstream_healthy",
            &label(u),
            u8::from(state.is_healthy(u, now)),
        );
    }
    x.family(
        "cuca_balancer_upstream_in_flight",
        "gauge",
        "Requests outstanding to the upstream.",
    );
    for u in &state.upstreams {
        x.sample(
            "cuca_balancer_upstream_in_flight",
            &label(u),
            u.in_flight.load(Ordering::Relaxed),
        );
    }
    x.family(
        "cuca_balancer_routed_total",
        "counter",
        "Requests routed to the upstream.",
    );
    for u in &state.upstreams {
        x.sample(
            "cuca_balancer_routed_total",
            &label(u),
            u.routed.load(Ordering::Relaxed),
        );
    }
    x.family(
        "cuca_balancer_requests_total",
        "counter",
        "Responses from the upstream by status code (502: the connection failed, 499: the client went away).",
    );
    for u in &state.upstreams {
        u.requests
            .render_into(&mut x.out, "cuca_balancer_requests_total", &label(u));
    }
    x.family(
        "cuca_balancer_request_ttft_seconds",
        "histogram",
        "Time from arrival to the first body byte, measured at the balancer.",
    );
    for u in &state.upstreams {
        u.ttft
            .render_into(&mut x.out, "cuca_balancer_request_ttft_seconds", &label(u));
    }
    x.family(
        "cuca_balancer_request_e2e_seconds",
        "histogram",
        "Time from arrival to the end of the body, measured at the balancer.",
    );
    for u in &state.upstreams {
        u.e2e
            .render_into(&mut x.out, "cuca_balancer_request_e2e_seconds", &label(u));
    }

    render_system(&mut x, &state.system.snapshot());
    x.out
}
