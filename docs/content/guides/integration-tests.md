+++
title = "Run the live integration suite"
description = "Start llama-server, run the suite with every plugin and service feature, read the skips, and turn a skip into a failure."
template = "page.html"
weight = 3
+++

# Run the live integration suite

<dl class="page-facts">
<dt>In one line</dt>
<dd>The llama.cpp test files share one <code>llama-server</code>; server-dependent tests skip when it is unreachable unless you set <code>CUCA_REQUIRE_LIVE=1</code></dd>
<dt>You need</dt>
<dd><code>llama-server</code> on port 1234 with a model loaded, per <a href="@/quick-start/first-stream.md">Stream a first reply</a> step 2</dd>
<dt>Read this if</dt>
<dd>You are changing the crate and need the suite to exercise a backend instead of skipping it</dd>
</dl>

Goal: run the integration suite against a live server and know that every server-dependent test ran.

Every llama.cpp test file gates itself on `provider-llamacpp` plus its own feature. A run with the wrong feature list compiles an empty test binary and reports success, so the feature list decides what runs.

## Step 1: start the server

The harness targets `http://127.0.0.1:1234/v1`. `llama-server` listens on 8080 by default, so start it on 1234.

Linux and macOS:

```bash
llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
```

Windows (PowerShell):

```powershell
llama-server -m C:\path\to\model.gguf --port 1234 --alias google/gemma-4-e4b
```

Leave it running and check it from a second terminal.

Linux and macOS:

```bash
curl http://127.0.0.1:1234/v1/models
```

Windows (PowerShell), where `curl` is an alias for `Invoke-WebRequest`:

```powershell
curl.exe http://127.0.0.1:1234/v1/models
```

Result: a JSON list with at least one model `id`. To target another server, set `CUCA_BASE_URL` in step 2.

## Step 2: run the whole suite

Name the provider and every plugin and service feature:

```bash,name=Runs the same on all three platforms
cargo test --features "provider-llamacpp plugin-mcp plugin-sandbox plugin-memory service-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry service-speculative plugin-session-log service-replay service-prompt-cache plugin-cost service-rate-limit service-vector-store plugin-redaction" -- --nocapture --test-threads=1
```

The trailing flags matter for a readable result:

- `--nocapture` shows the skip lines. They are written with `eprintln!`, which libtest captures by default, so without the flag a skipped test looks like a passing one.
- `--test-threads=1` runs the tests in each binary one at a time against the single `llama-server`. The canonical command in the README uses it.

Result: each test binary reports `test result: ok`, and `--nocapture` prints any `SKIP:` lines.

The CI feature set is every feature except `provider-local-cuda` and `provider-local-rocm`, which download a prebuilt LLVM on the first build. See [The feature matrix](@/reference/features.md).

## Step 3: read the skips

A server-dependent test that cannot reach the server prints one line and returns:

```text,name=The skip line every server-dependent test emits
SKIP: llama.cpp not reachable: <reason>
```

The reason comes from a probe of `GET {base_url}/models` with a two second timeout. It tells a refused connection from a server that answered with no model ids.

Two files never call the probe: `tests/service_prompt_cache.rs` drives in-process mock SSE servers on ephemeral loopback ports, and `tests/public_exports.rs` constructs types and never dispatches. Every other llama.cpp file has more tests than live gates, so a subset of each runs with no server up.

`tests/provider_local.rs` and `tests/provider_local_mtp.rs` need no server. They print `SKIP: local model not available: ...` until you select a model; see [Environment variables](@/reference/environment.md).

Result: you can tell which tests skipped and why.

## Step 4: make a skip fail

A silent skip in CI is a test that stopped testing. `CUCA_REQUIRE_LIVE=1` turns the unreachable-server path into a panic:

```text,name=The panic when the variable is set and nothing answers
CUCA_REQUIRE_LIVE=1 but llama.cpp is unreachable: <reason>
```

Linux and macOS:

```bash
CUCA_REQUIRE_LIVE=1 cargo test --features "provider-llamacpp plugin-memory" -- --nocapture --test-threads=1
```

Windows (PowerShell), where each assignment is a statement and `;` joins them:

```powershell
$env:CUCA_REQUIRE_LIVE = "1"; cargo test --features "provider-llamacpp plugin-memory" -- --nocapture --test-threads=1
```

Only the exact value `1` arms it. Any other value behaves as unset. In PowerShell the variable stays set for the session; clear it with `Remove-Item Env:CUCA_REQUIRE_LIVE`.

Result: with the server stopped, the run fails with the panic above instead of printing `SKIP`.

## Step 5: run one file

A single-plugin or single-service file is gated on `provider-llamacpp` plus its own feature, so a single-file run names exactly those two:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_guardrails --features provider-llamacpp,plugin-guardrails -- --nocapture
```

`plugin_combinations.rs` is gated on `provider-llamacpp` alone, and each of its submodules adds its own gate. A run that compiles all of them needs the union of those features:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_combinations --features provider-llamacpp,service-entity-extraction,plugin-memory,service-prompt-cache,service-speculative,plugin-session-log,plugin-cost,plugin-telemetry,service-rate-limit,plugin-redaction,service-replay,service-vector-store -- --nocapture
```

A shorter list does not fail. It compiles fewer submodules and reports success for the rest.

Result: the run lists only that file's tests.

## If the model id is wrong

With `CUCA_MODEL` unset or empty, the harness probes the server and uses the first model id it reports. Set `CUCA_MODEL` to override. With neither available the test panics:

```text,name=The panic when no model id can be resolved
no model id: set CUCA_MODEL or start llama-server at http://127.0.0.1:1234/v1
```

Defaults for all three variables are in [Environment variables](@/reference/environment.md).

## If you are checking one feature builds alone

CI's `plugin_solo` and `service_solo` jobs do this, and they need no server. One leg runs per plugin feature (12) and per service feature (6):

```bash,name=Runs the same on all three platforms
cargo check --all-targets --no-default-features --features provider-openai,plugin-hitl
```

Swap `plugin-hitl` for another feature to check it.

Result: `cargo check` finishes with no error.

Next page: [The feature matrix](@/reference/features.md).
