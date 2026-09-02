# Repository Guidelines

## Project Overview

CUCA (**Compact Universal Client for Agents**) is a high-performance, asynchronous Rust client library intended to sit between autonomous agent orchestrators and LLM backends: an ultra-low-overhead runtime that monitors, parses, and dispatches multi-modal SSE streams. Tagline: *"Legendary vigilant witch; small footprint that constantly watches for responses."*

Current state: **implemented**. The unified client (`CucaClient`/`CucaClientBuilder`), the zero-allocation SSE parser, seven provider adapters, twelve feature-gated plugins, and six explicit-call services, including the speculative fast/slow orchestrator, are all implemented.

## Architecture & Data Flow

**Implemented:** the unified client pipeline, the SSE parser, the provider adapters, the plugins, and the services. The architecture:

- **Unified provider abstraction**: one `AgentRequest` in, one normalized `AgentResponseStream` out, across OpenAI, Anthropic, DeepSeek, Gemini, llama.cpp, vLLM, and LM Studio (`ProviderEndpoint` enum, `UnifiedMessage`, `MessageContentBlock`).
- **Zero-default provider policy**: no provider compiled by default (`default = []`); a `compile_error!` fires if no provider feature is enabled. Providers are opt-in Cargo features (`provider-openai`, `provider-anthropic`, …).
- **Zero-allocation SSE engine.** `SseStreamParser` is a byte state machine over `bytes::BytesMut`; inbound chunks are scanned with `memchr` in contiguous buffers, with no intermediate string allocation.
- **Plugins and services.** A plugin implements `trait CucaPlugin` (`on_request` / `on_stream_chunk` / `on_response_complete`) and observes the request/stream pipeline; it is compile-time feature-gated and carries its own non-core dependencies. A service is an explicit-call, client-level capability driven by direct method calls instead of pipeline hooks; a service MUST NEVER implement `CucaPlugin`, so registering one with `register_plugin` is a compile error rather than an inert no-op.
- **Fast/slow orchestration** (`service-speculative`): `ModelOrchestrator` + `SwappableModelPair`; complexity routing → speculative draft generation → fallback cascades on malformed output.

Data flow: builder selects provider → `UnifiedRequest` passes through registered plugins (`on_request`) → provider adapter translates to vendor protocol → SSE stream parsed into `MessageContentBlock` events → each chunk passes plugins (`on_stream_chunk`) → `on_response_complete`.

## Key Directories

| Path | Purpose |
| --- | --- |
| `src/` | Crate root; library source. |
| `tests/` | Live llama.cpp integration suite: shared harness in `common/`, one file per plugin or service, MCP stdio echo server embedded in `plugin_mcp.rs` (test-binary re-execution). |
| `docs/` | Zola documentation site: `content/` pages, `templates/`, `static/`, and the `[[extra.nav]]` reading order in `config.toml`. Build output `docs/public/` and the regenerated `docs/static/giallo.css` are gitignored. |
| `rust-toolchain.toml` | Pinned toolchain 1.98.0. |
| `target/` | Build output. Gitignored. |

`examples/` holds documented, runnable demos gated on provider features via `required-features`, and `.github/workflows/ci.yml` runs fmt, clippy, test, doc, and the plugin-layering checks (see Runtime/Tooling Preferences). `tests/` holds the integration suite. `benches/` holds one Divan benchmark, `benches/vector_store.rs`, run with `cargo bench --features provider-openai,service-vector-store`; no CI job runs it.

## Development Commands

```sh
cargo build --features provider-openai                    # minimal provider build
cargo test --features provider-openai                     # minimal provider test suite
cargo clippy --all-targets --features provider-openai -- -D warnings
cargo fmt                                                  # format (default rustfmt; no rustfmt.toml)
cargo check --all-targets --no-default-features --features provider-openai,plugin-memory   # one plugin alone: the solo-build check CI runs per plugin or service feature
cargo test --features "provider-llamacpp plugin-mcp plugin-sandbox plugin-memory service-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry service-speculative plugin-session-log service-replay service-prompt-cache plugin-cost service-rate-limit service-vector-store plugin-redaction"   # live llama.cpp integration suite (see Testing & QA)
cargo doc --open --features provider-openai               # minimal provider docs
cd docs && zola build && zola check                       # site build; both fail on a broken @/ link or a nav path with no page
cd docs && python3 build-local.py                         # browsable local site plus its search index
```

`cargo run` does not apply: this is a library crate, no binary target.

`zola build` alone leaves site search empty: `base.html` reads `data-index="search-index.json"`,
which only `search-index.py` writes. `python3 build-local.py` runs both, and rewrites the base URL to
per-page relative paths so `docs/public/index.html` opens from disk. Search itself needs an HTTP
server; `fetch` refuses a `file://` URL. Every `docs/content` page MUST have a `[[extra.nav]]` entry:
the sidebar, the breadcrumb and the previous/next pager all derive from that list. What belongs on a
site page versus `README.md` versus a `///` is defined in
`.agents/skills/writing-cuca-docs/SKILL.md` (*Surfaces*); that directory is gitignored.

## Code Conventions & Common Patterns

Follow Rust defaults plus the conventions below:

- **Naming**: snake_case functions/variables, CamelCase types (`CucaClient`, `CucaPlugin`, `UnifiedRequest`, `MessageContentBlock`); plugin names are `&'static str` (e.g. `"opentelemetry-observability"`).
- **Error handling**: `CucaError` + `PluginError`; plugins return `Result<(), PluginError>`, client methods `Result<AgentResponseStream, CucaError>`. Avoid `unwrap`/`panic` in library code.
- **Async:** provider and async plugins use Tokio; the public stream contract is `futures_core::Stream` and yields `Result<MessageContentBlock, CucaError>`.
- **Feature gating:** `default = []`; `lib.rs` requires at least one `provider-*` feature. Provider, plugin, and service code MUST be `#[cfg(feature = "…")]`-gated, and each non-core dependency MUST be enabled only by its owning feature via `dep:`. Cross-tier feature and code edges additionally obey *Plugins and services* below.
- **Dependency injection**: providers via builder (`CucaClient::builder().with_provider(…).register_plugin(Arc<dyn CucaPlugin>)`); plugins as `Arc<dyn CucaPlugin>` (Send + Sync).
- **State management**: session state is append-only trajectory logs with fork support (`SessionStorePlugin`); no global mutable state.
- **Memory**: state that grows with traffic is capped, evicting or offloading at the bound; see *Memory discipline* below.
- **Data modeling**: `#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]` on all wire types.

### Plugins and services (no cycles)

Core, plugins and services form a DAG at three levels: Cargo features, `#[cfg]`/import edges, and runtime coupling. Cargo rejects feature cycles; the code and runtime levels are on contributors.

**Tiers.** Core (`types`, `request`, `error`, `sse`, `session`, `plugin`, `client`, `provider`, `export`, `canonical`, `tokenize`, `cost_otel`) sits below every plugin and every service. The plugin tier is flat: `memory`, `session-log`, `mcp`, `sandbox`, `guardrails`, `subagent`, `hitl`, `web-search`, `skills`, `telemetry`, `cost`, and `redaction` depend on core only, and no plugin depends on another plugin. Services depend on core and on the plugin features they declare: `entity-extraction → memory` (hard, `service-entity-extraction = ["plugin-memory"]`), `replay → session-log` (hard, `service-replay = ["plugin-session-log"]`), `vector-store → memory` (hard, `service-vector-store = ["plugin-memory", "dep:wide"]`), and `speculative → session-log` (documented-optional: `ModelOrchestrator::with_session_store` records `SessionEvent::ModelSwap`); `prompt-cache` and `rate-limit` declare no plugin dependency at all.

- **Direction is one-way:** a plugin MUST NOT name a service in any form: no `use crate::services::<name>`, no `cfg(feature = "service-<name>")` (including inside `cfg(all(…))`), no runtime lookup. `src/plugins/` stays ignorant that any service exists. A service→service edge requires re-tiering the shared part downward first.
- **Declare every code edge:** any compile-time reference from a service to a plugin it uses REQUIRES `service-<name> = ["plugin-<other>", …]` in `Cargo.toml`; never rely on a peer being co-enabled by accident.
- **Coordinate in core, never downward:** core MAY `#[cfg]`-reference any plugin or service; neither MUST reach a peer through a core module to dodge the two rules above. Multi-capability workflows live in core: `CucaExport::from_live` in `src/export.rs`, and `OtelCostObserver` in `src/cost_otel.rs`, the cost-ledger-to-OpenTelemetry bridge gated `cfg(all(feature = "plugin-cost", feature = "plugin-telemetry"))`.
- **No runtime peer discovery:** plugins and services MUST NOT look each other up. `CucaClient::plugins()` is an inspection accessor, not a service locator: no name matching, no downcasting. Cross-tier data moves through a caller-injected `Arc<dyn Trait>` or an explicit application hand-off (`EntityExtractionReport.delta` → `MemoryPlugin::merge_graph`).
- **Optional cross-tier behavior is documented and loud:** behavior that exists only when a declared peer is co-enabled or a peer handle is attached MUST be named in the owning module's `//!` docs, listed in the tiers above, and degrade loudly: compile-gated out of existence (`ModelOrchestrator::with_session_store`) or an explicit error (`CucaExportError::Unsupported` for a compiled-out export section). Silently doing nothing is a defect.
- **No hidden hook-order dependence:** `on_request` hooks run in registration order over one shared `UnifiedRequest`; a plugin MUST NOT require a position relative to another plugin. Where order is observable, document it at the consuming site: memory's graph/warning injections change the digest `PromptCache::lookup` computes afterward.
- **Explicit-call services:** a service is driven by direct method calls rather than pipeline hooks and MUST NOT implement `CucaPlugin` (`PromptCache`, `EntityExtractor`, `SessionReplay`, `ModelOrchestrator`, `RateLimiter`, `InMemoryVectorStore`), so registering one is a compile error instead of an inert no-op; its `//!` header names the real entry points and any mandatory hand-off.

**Checks.** Every plugin or service feature MUST build alone on one provider (`cargo check --all-targets --no-default-features --features provider-openai,plugin-<name>` or `…,service-<name>`); CI runs that matrix across two jobs, `plugin_solo` and `service_solo`, which is how undeclared and reverse code edges are caught: `--all-features` cannot catch them. CI also greps every plugin directory for `service-` feature names, `crate::services` paths, and the `plugins::` paths the five moved modules left behind, since a `cfg(all(…))` reverse edge compiles in both the solo and all-features builds.

### Memory discipline (bounded by default)

Memory size is a first-order concern: every structure that grows with traffic ships bounded, with a stated policy at the bound.

- **Compact structures first.** Keep stream bytes in one reusable `BytesMut` scanned with `memchr` (`SseStreamParser`); pre-size known growth (`MemoryGraph::with_capacity`/`reserve`) instead of reallocating.
- **Cap every growable collection.** A new cache, log, buffer, or map that outlives one request REQUIRES a validated bound at construction, in the mold of `PromptCacheConfig::new` (rejects zero capacity), `MemoryConfig` (`max_messages`/`max_tokens`/`max_fraction`), `GraphContextConfig` (`max_nodes`/`max_relationships`), and `SandboxConfig::max_memory_bytes`. An uncapped collection is a defect unless its docs state why growth is bounded elsewhere.
- **Evict or offload at the cap.** State the at-cap policy in the owning type's docs and reuse an existing shape: TTL plus deterministic LRU eviction (`PromptCache`, `lru_order`), the ordered `CompactionStrategy` pipeline (`Offload` to a caller-supplied `VectorStore`, `Summarize`, `ClampOversizedMessages`, `SlidingWindow`), or append-only offload to disk (`SessionLogPlugin` with `FileBackend`). Unbounded growth is a defect; so is silent data loss.
- **Expose usage and warn near the cap.** A capped structure surfaces a cheap usage reading. Follow memory's seams: `ContextUsageObserver` receives a `ContextUsage` on every request (the reporting gauge), and `MemoryConfig::warn_fraction` injects a one-shot near-limit warning (idempotent via marker scan). Alerting and export belong on the caller-supplied OTel meter (`OpenTelemetryPlugin`); never install a global provider.
- **Re-verify bounds when touching hot paths.** A change to the SSE engine, compaction, a cache, or the graph MUST keep the owning module's `//!` allocation and bounds story accurate (`src/sse.rs` documents its per-frame allocations; `src/plugins/memory/graph.rs` documents its complexity envelope).

## Important Files

| Path | Why it matters |
| --- | --- |
| `src/lib.rs` | Crate root: public re-exports of the client, request/response contracts, errors, and the feature-gated plugin and service surfaces. The implementation lives in the module tree (`client`, `request`, `types`, `error`, `sse`, `session`, `plugin`, `plugins/`, `services/`, `provider/`). |
| `Cargo.toml` | Package manifest for `cuca` (v0.2.0, edition 2024, rust-version 1.98): crates.io publish metadata (description, repository, homepage, docs.rs all-features build), the 25-feature matrix (7 providers + 12 plugins + 6 services), no default provider, the three hard service feature edges (`service-entity-extraction = ["plugin-memory"]`, `service-replay = ["plugin-session-log"]`, `service-vector-store = ["plugin-memory", "dep:wide"]`), minimal core dependencies, feature-owned transport/runtime dependencies, and dev-deps. |
| `README.md` | Public-facing claims (zero-default providers, unified abstraction, zero-allocation engine, plugin/service architecture); keep in sync as implementation lands. |
| `.gitignore` | Ignores `/target`, `/.agents`, `/graphify*`, `/.omp`, `/docs/public`, and `/docs/static/giallo.css`. |

## Runtime/Tooling Preferences

- **Rust 1.98.0**, pinned by the root `rust-toolchain.toml` (`channel = "1.98.0"`, `profile = "minimal"`, components `clippy` + `rustfmt`); MSRV `rust-version = "1.98"` in package metadata.
- **Cargo** is the only build/package tool; no alternative package managers.
- **Crate CI**: `.github/workflows/ci.yml` runs eight jobs: `fmt`, `clippy` and `test` (each over the `provider-openai`-only and `--all-features` points), `doc`, `no_provider` (asserts the bare build fails), `plugin_solo` (per-plugin solo build, one matrix leg per plugin feature), `service_solo` (the same, one matrix leg per service feature), and `plugin_layering` (flat-tier greps; see *Plugins and services*). Clippy enforces `-D warnings` with `--all-targets`. No rustfmt/clippy config files; defaults apply.
- **Docs CI**: `.github/workflows/docs.yml` runs two jobs, triggered by a push to `main` or a pull request touching `docs/**` or the workflow itself, plus `workflow_dispatch`. `build` installs Zola 0.22.1 from the pinned `getzola/zola` release tarball, verifies its SHA-256, then runs `zola build`, `zola check`, and `python3 search-index.py public` in `docs/`, and uploads `docs/public` as the Pages artifact; `zola check` is the gate on broken `@/` links. `deploy` is gated on `github.event_name != 'pull_request' && github.ref == 'refs/heads/main'`, so a pull request builds and link-checks without deploying, and a `workflow_dispatch` on `main` republishes the committed site without a new commit. Workflow permissions are `contents: read`; `deploy` alone adds `pages: write` and `id-token: write`. The concurrency group is `pages` with `cancel-in-progress: false`. The repository Pages source MUST be set to GitHub Actions, not a branch, or `deploy` fails.
- **Dependencies:** core `bytes`, `futures-core`, `memchr`, `serde`, and `serde_json`; feature-gated `reqwest`, `tokio`, `tokio-stream`, `rmcp`, `wasmtime`, `tiktoken-rs`, `jsonschema`, OpenTelemetry, `tracing`, `sha2`, `getrandom`, `base64`, `postcard`, and `wide`. Test-only Tokio and `tokio-stream` support the crate's test suite, and `divan` is the bench-only harness.

## Testing & QA

- **Unit tests**: `#[cfg(test)]` modules next to the code, one per feature module, gated with the same feature flags. Plain `#[test]` plus `#[tokio::test]` (current-thread flavor only; `rt-multi-thread` is not a declared tokio feature of this crate).
- **Integration suite**: `tests/` holds one file per plugin (`plugin_mcp.rs`, `plugin_subagent.rs`, `plugin_hitl.rs`, `plugin_web_search.rs`, `plugin_cost.rs`, `plugin_redaction.rs`, plus the memory/guardrails/telemetry/session-log/skills/sandbox files) and one file per service (`service_prompt_cache.rs`, `service_entity_extraction.rs`, `service_replay.rs`, `service_speculative.rs`, `service_rate_limit.rs`, `service_vector_store.rs`), `plugin_combinations.rs` for cross-tier combinations, and `public_exports.rs`, sharing the harness in `tests/common/mod.rs`. Each single-feature file compiles only when `provider-llamacpp` and its own feature are on, via a top-level `#![cfg]` gate; `plugin_combinations.rs` instead gates each combination module with its own per-module `cfg(all(feature = "…", feature = "…"))`; `public_exports.rs` is gated `cfg(any(feature = "provider-openai", feature = "provider-llamacpp"))` so it runs in the live suite too.
- **Live llama.cpp**: server-dependent tests probe `GET {base}/models`; an unreachable server prints `SKIP: llama.cpp not reachable: ...` and the test passes, unless `CUCA_REQUIRE_LIVE=1` turns the skip into a panic. `CUCA_BASE_URL` (default `http://127.0.0.1:1234/v1`, not llama-server's own default port 8080) and `CUCA_MODEL` (default: first id the server reports) configure the target. The MCP echo server lives in `plugin_mcp.rs`: the test binary re-executes itself with `--mcp-echo-server` and serves an rmcp stdio echo server before libtest's main (via a `ctor` interceptor).
- **Run**: `cargo test --features provider-llamacpp` for unit tests; the live command in Development Commands for the integration suite.
