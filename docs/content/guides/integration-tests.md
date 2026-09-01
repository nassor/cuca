+++
title = "Run the live integration suite"
description = "Start llama-server, run the suite with every plugin feature, read the skips, and turn a skip into a failure."
template = "page.html"
weight = 3
+++

# Run the live integration suite

<dl class="page-facts">
<dt>In one line</dt>
<dd>Seventeen test binaries share one <code>llama-server</code>; server-dependent tests skip when it is unreachable unless you tell them not to</dd>
<dt>You need</dt>
<dd><code>llama-server</code> on port 1234 with a model loaded, per <a href="@/quick-start/first-stream.md">Stream a first reply</a> step 2</dd>
<dt>Read this if</dt>
<dd>You are changing the crate and need the suite to actually exercise a backend rather than skipping past it</dd>
</dl>

Every `tests/*.rs` file gates itself on the features it needs, so a run with the
wrong feature list compiles an empty test binary and reports success. The feature
list is the whole game.

## Step 1: start the server

The harness targets `http://127.0.0.1:1234/v1` by default, and `llama-server`
listens on 8080 unless told otherwise:

```bash,name=Runs the same on all three platforms
llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
```

Leave it running. To target something else, set `CUCA_BASE_URL` in step 2.

## Step 2: run the whole suite

Every plugin feature, plus the provider every test file requires:

```bash,name=Runs the same on all three platforms
cargo test --features "provider-llamacpp plugin-mcp plugin-sandbox plugin-memory plugin-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry plugin-speculative plugin-session-log plugin-prompt-cache plugin-cost plugin-rate-limit plugin-redaction" -- --nocapture --test-threads=1
```

Both trailing flags earn their place. `--nocapture` is what makes the skip lines
visible: they are written with `eprintln!`, which libtest swallows by default, so
without it a skipped test is indistinguishable from a passing one.
`--test-threads=1` serialises the run because every live test talks to the same
single `llama-server` instance.

`--all-features` compiles the same set plus the six unused provider features. It
works, and no additional test runs as a result.

## Step 3: read the skips

A test that cannot reach the server prints one line and returns:

```text,name=The skip line every server-dependent test emits
SKIP: llama.cpp not reachable: <reason>
```

The reason comes from a probe of `GET {base_url}/models` with a two second
timeout, so it distinguishes a refused connection from a server that answered
with no model ids.

Two test files never need the server and never skip:

| File | Why |
|---|---|
| `tests/plugin_prompt_cache.rs` | drives in-process mock SSE servers on ephemeral loopback ports |
| `tests/public_exports.rs` | asserts the re-export surface; constructs types and never dispatches |

Five more are partial. `plugin_mcp.rs`, `plugin_speculative.rs`,
`plugin_subagent.rs`, `plugin_combinations.rs` and `plugin_rate_limit.rs` each
hold both mock-backed tests and live ones, so a subset runs with no server up.

## Step 4: make a skip fail instead

A silent skip in CI is a test that stopped testing. Setting
`CUCA_REQUIRE_LIVE=1` turns the unreachable-server path into a panic:

```text,name=The panic when the variable is set and nothing answers
CUCA_REQUIRE_LIVE=1 but llama.cpp is unreachable: <reason>
```

Linux and macOS:

```bash
CUCA_REQUIRE_LIVE=1 cargo test --features "provider-llamacpp plugin-memory" -- --nocapture --test-threads=1
```

Windows (PowerShell), where assignments are separate statements joined by `;`:

```powershell
$env:CUCA_REQUIRE_LIVE = "1"; cargo test --features "provider-llamacpp plugin-memory" -- --nocapture --test-threads=1
```

Only the exact value `1` arms it. Any other value behaves as if unset.

## Step 5: run one file at a time

Each single-plugin file is gated on `provider-llamacpp` plus its own plugin
feature, so a single-file run names exactly those two:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_guardrails --features provider-llamacpp,plugin-guardrails -- --nocapture
```

`plugin_combinations.rs` is the exception. Its crate gate is only
`provider-llamacpp`, and each of its nine submodules adds its own gate, so a run
that compiles all nine needs the union:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_combinations --features provider-llamacpp,plugin-entity-extraction,plugin-memory,plugin-prompt-cache,plugin-speculative,plugin-session-log,plugin-cost,plugin-telemetry,plugin-rate-limit -- --nocapture
```

A shorter feature list here does not fail. It compiles fewer submodules and
reports success for the ones that remain.

## If the model id is wrong

With `CUCA_MODEL` unset or empty, the harness probes the server and uses the
first model id it reports. Set the variable to override that. With neither
available it panics rather than guessing:

```text,name=The panic when no model id can be resolved
no model id: set CUCA_MODEL or start llama-server at http://127.0.0.1:1234/v1
```

All three variables, with their exact defaults and gate behavior, are in
[Environment variables](@/reference/environment.md).

## If you are checking a single plugin compiles alone

That is what CI's `plugin_solo` job does, and it needs no server at all:

```bash,name=Runs the same on all three platforms
cargo check --all-targets --no-default-features --features provider-openai,plugin-hitl
```

Swap the plugin feature to check another. Sixteen runs cover the set.

Next page: [The feature matrix](@/reference/features.md).
