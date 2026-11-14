+++
title = "Run the live integration suite"
description = "Start llama-server, run the suite with every plugin and service feature, read the skips, and turn a skip into a failure."
template = "page.html"
weight = 3
+++

# Run the live integration suite

<dl class="page-facts">
<dt>In one line</dt>
<dd>The llama.cpp test files share one <code>llama-server</code>; server-dependent tests skip when it is unreachable unless <code>cuca-test.kdl</code> sets <code>require_live #true</code></dd>
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

Result: a JSON list with at least one model `id`. To target another server, write `llamacpp base_url="..."` in `cuca-test.kdl`; see the end of this page.

## Step 2: run the whole suite

Name the provider and every plugin and service feature:

```bash,name=Runs the same on all three platforms
cargo test --features "provider-llamacpp config-kdl plugin-mcp plugin-sandbox plugin-memory service-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry service-speculative plugin-session-log service-replay service-prompt-cache plugin-cost service-rate-limit service-vector-store plugin-redaction service-scheduler" -- --nocapture --test-threads=1
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

`tests/provider_local.rs`, `tests/provider_local_batch.rs` and `tests/provider_local_mtp.rs` need no server. They print ``SKIP: local model not available: set `local { model { source ... } }` or `local { download_default #true }` in cuca-test.kdl`` until you select a model. `tests/provider_local_tiers.rs` prints a `SKIP:` line until you select a model and write `experts { tiered ... }` in it, and `tests/provider_local_convert.rs` prints a `SKIP:` line until you set `oracle { ref_dir ... }`. All five run under `--features provider-local`, which enables `config-kdl`; the keys are in [Configuration files](@/reference/environment.md#the-test-file).

Result: you can tell which tests skipped and why.

## Step 4: make a skip fail

A silent skip in CI is a test that stopped testing. `require_live #true` in `cuca-test.kdl` turns the unreachable-server path into a panic:

```kdl,name=cuca-test.kdl
require_live #true
llamacpp base_url="http://127.0.0.1:1234/v1"
```

```text,name=The panic when require_live is set and nothing answers
require_live is set but llama.cpp is unreachable: <reason>
```

```bash,name=Runs the same on all three platforms
cargo test --features "provider-llamacpp config-kdl plugin-memory" -- --nocapture --test-threads=1
```

Result: with the server stopped, the run fails with the panic above instead of printing `SKIP`.

## Step 5: run one file

A single-plugin or single-service file is gated on `provider-llamacpp` plus its own feature, so a single-file run names exactly those two. Add `config-kdl` whenever `cuca-test.kdl` exists, because the llama.cpp suites read the file only with that feature and panic if the file is present without it:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_guardrails --features provider-llamacpp,config-kdl,plugin-guardrails -- --nocapture
```

`plugin_combinations.rs` is gated on `provider-llamacpp` alone, and each of its submodules adds its own gate. A run that compiles all of them needs the union of those features:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_combinations --features provider-llamacpp,config-kdl,service-entity-extraction,plugin-memory,service-prompt-cache,service-speculative,plugin-session-log,plugin-cost,plugin-telemetry,service-rate-limit,plugin-redaction,service-replay,service-vector-store -- --nocapture
```

A shorter list does not fail. It compiles fewer submodules and reports success for the rest.

Result: the run lists only that file's tests.

## The test file

The suites read one file, `cuca-test.kdl`, at the repository root. It is gitignored. `cuca-test.example.kdl`, next to it, lists every key; the llama.cpp suite uses three:

| Key | Effect | Default |
|---|---|---|
| `llamacpp base_url="..."` | the server base URL, with `/v1` | `http://127.0.0.1:1234/v1` |
| `llamacpp model="..."` | the model id under test | the first id `GET {base_url}/models` reports |
| `require_live` | `#true` turns an unreachable server into a panic | `#false` |

With no model id configured and no server answering, the test panics:

```text,name=The panic when no model id can be resolved
no model id: set `llamacpp model="..."` in cuca-test.kdl or start llama-server at http://127.0.0.1:1234/v1
```

The llama.cpp suites read the file only when `config-kdl` is among the `--features`. With the file present and the feature off, the run panics and says so. The `provider-local` keys are in [Configuration files](@/reference/environment.md#the-test-file).

## If you are checking one feature builds alone

CI's `plugin_solo` and `service_solo` jobs do this, and they need no server. One leg runs per plugin feature (12) and per service feature (7):

```bash,name=Runs the same on all three platforms
cargo check --all-targets --no-default-features --features provider-openai,plugin-hitl
```

Swap `plugin-hitl` for another feature to check it.

Result: `cargo check` finishes with no error.

Next page: [The feature matrix](@/reference/features.md).
