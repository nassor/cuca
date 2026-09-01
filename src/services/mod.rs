//! Feature-gated service implementations.
//!
//! A service is an explicit-call, client-level capability: it is driven by
//! direct method calls, never by pipeline hooks. No service implements
//! [`CucaPlugin`](crate::plugin::CucaPlugin), because no hook fits — nothing to
//! mutate on the outbound request, nothing to annotate on an arriving chunk, and
//! no hook signature able to return what the capability produces. Handing a
//! service to
//! [`CucaClientBuilder::register_plugin`](crate::CucaClientBuilder::register_plugin)
//! is therefore a compile error rather than an inert no-op registration. Every
//! capability behind a `service-*` feature lives in one submodule here; each
//! submodule's `//!` header is the authority on its entry points, its
//! bounded-growth policy, and any mandatory hand-off. Nothing in this module is
//! compiled unless its feature is enabled.
//!
//! # Tier rule: core < plugins < services
//!
//! A service MAY depend on core and on the plugin features it declares in
//! `Cargo.toml`:
//!
//! - `service-entity-extraction = ["plugin-memory"]` — hard: the extraction
//!   delta *is* a `MemoryGraph`, and the caller merges it into the memory
//!   plugin's working graph.
//! - `service-replay = ["plugin-session-log"]` — hard: replay reads a recorded
//!   trajectory through the session log's `SessionBackend` seam.
//! - `service-speculative` — documented-optional on `plugin-session-log`:
//!   `ModelOrchestrator::with_session_store` is compiled out of existence
//!   without it, and records `SessionEvent::ModelSwap` with it.
//!
//! The direction is one-way. A plugin MUST NEVER name a service in any form: no
//! `use crate::services::…`, no `cfg(feature = "service-…")` (including inside
//! a `cfg(all(…))`), no Cargo feature edge, no runtime lookup. A
//! service-to-service edge requires re-tiering the shared part downward first,
//! not a feature line between two services.
//!
//! With this tier in place the plugin tier is flat: no plugin depends on another
//! plugin. Multi-capability workflows still belong to core (`CucaExport` in
//! `crate::export`, `OtelCostObserver` in `crate::cost_otel`) or to the
//! application, never to a sideways edge between capabilities.

// No `///` summary on these declarations, by design; the reason is in
// `src/plugins/mod.rs`, which carries the same shape.
#[cfg(feature = "service-entity-extraction")]
pub mod entity_extraction;
#[cfg(feature = "service-speculative")]
pub mod orchestrator;
#[cfg(feature = "service-prompt-cache")]
pub mod prompt_cache;
#[cfg(feature = "service-rate-limit")]
pub mod rate_limit;
#[cfg(feature = "service-replay")]
pub mod replay;
