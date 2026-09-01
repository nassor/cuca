//! Feature-gated plugin implementations.
//!
//! Every capability behind a `plugin-*` feature lives in one submodule here;
//! each submodule's `//!` header is the authority on its hooks and its
//! bounded-growth policy. Every module here implements
//! [`CucaPlugin`](crate::plugin::CucaPlugin) and is driven by the request
//! pipeline; an explicit-call capability belongs one tier up, in the sibling
//! service tier, whose module docs own the tier rule. This tier is flat: no
//! plugin depends on another plugin, and nothing under this directory names a
//! service in any form — not a feature, not a path, not a doc link — so a
//! directory-wide grep is a valid reverse-edge check. Nothing in this module is
//! compiled unless its feature is enabled.

// Deliberately no `///` summary on these declarations, here or in the sibling
// tier's `mod.rs`. A doc comment written at the declaration site makes rustdoc
// resolve the *whole* merged doc block -- including the target file's own `//!`
// header -- in this parent module's scope, where none of a submodule's items are
// nameable, so every in-module intra-doc link in every header below silently
// degrades to plain text. Each module's first `//!` line already supplies the
// summary the module list renders.
#[cfg(feature = "plugin-cost")]
pub mod cost;
#[cfg(feature = "plugin-guardrails")]
pub mod guardrails;
#[cfg(feature = "plugin-hitl")]
pub mod hitl;
#[cfg(feature = "plugin-mcp")]
pub mod mcp;
#[cfg(feature = "plugin-memory")]
pub mod memory;
#[cfg(feature = "plugin-redaction")]
pub mod redaction;
#[cfg(feature = "plugin-sandbox")]
pub mod sandbox;
#[cfg(feature = "plugin-session-log")]
pub mod session_log;
#[cfg(feature = "plugin-skills")]
pub mod skills;
#[cfg(feature = "plugin-subagent")]
pub mod subagent;
#[cfg(feature = "plugin-telemetry")]
pub mod telemetry;
#[cfg(feature = "plugin-web-search")]
pub mod web_search;
