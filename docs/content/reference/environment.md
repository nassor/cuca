+++
title = "Environment variables"
description = "Three variables, all belonging to the examples and the test harness. No file under src/ reads the environment."
template = "page.html"
weight = 3
+++

# Environment variables

<dl class="page-facts">
<dt>In one line</dt>
<dd>Three variables, all read by <code>examples/</code> and <code>tests/</code>; the library itself reads none</dd>
<dt>You need</dt>
<dd>Nothing; every variable has a default or a documented fallback</dd>
<dt>Read this if</dt>
<dd>You are pointing an example or the test suite somewhere other than a local llama.cpp server</dd>
</dl>

## The library reads none

No file under `src/` reads an environment variable. Base URLs, API keys and
bearer tokens reach the client only through `CucaClientBuilder`. The three
`std::env` calls that exist in `src/` are `std::env::temp_dir()` inside test
modules of `plugin-guardrails`, `plugin-session-log` and `plugin-skills`, and
none of them reads a variable.

A library that read `OPENAI_API_KEY` from the process environment would make
credential resolution invisible to its caller. Configuration is passed in.

## The three variables

| Variable | Read by | Default | Effect when unset |
|---|---|---|---|
| `CUCA_BASE_URL` | all three examples, `tests/common/mod.rs` | `http://127.0.0.1:1234/v1` | falls back to the default; never fails |
| `CUCA_MODEL` | all three examples, `tests/common/mod.rs` | examples: `google/gemma-4-e4b`; tests: none | examples use their literal default; tests probe the server and take the first model id it reports |
| `CUCA_REQUIRE_LIVE` | `tests/common/mod.rs` | unset | an unreachable server is a skip rather than a failure |

## `CUCA_BASE_URL`

The base URL an example or the test harness targets. `tests/common/mod.rs`
declares the fallback as a constant:

```rust,name=tests/common/mod.rs line 117
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";
```

The three examples carry the same literal independently. `llama-server` listens
on 8080 by default, so the suite expects it started with `--port 1234`, or this
variable pointed wherever it listens.

The value is used verbatim as the client's base URL, so it must include every
path segment before the route. For the OpenAI-compatible route that means it
normally ends in `/v1`.

## `CUCA_MODEL`

The model id the request carries.

The examples treat it as a plain override with a literal default of
`google/gemma-4-e4b`. The test harness treats an unset or empty value as a
request to discover the id: it issues `GET {base_url}/models` with a two second
timeout and takes the first `data[].id` in the response.

With neither the variable nor a reachable server, `model_name()` panics rather
than sending a guessed id:

```text,name=The panic message with the resolved base URL interpolated
no model id: set CUCA_MODEL or start llama-server at http://127.0.0.1:1234/v1
```

An empty string is treated as unset.

## `CUCA_REQUIRE_LIVE`

Controls what a server-dependent test does when the server is unreachable.

Unset, or set to anything other than `1`, the test prints a skip line and
returns:

```text,name=The skip line
SKIP: llama.cpp not reachable: <reason>
```

Set to exactly `1`, the same condition panics:

```text,name=The panic
CUCA_REQUIRE_LIVE=1 but llama.cpp is unreachable: <reason>
```

The comparison is against the literal string `1`. `true`, `yes` and `0` all
behave as unset.

Skip lines are written with `eprintln!`, which libtest captures, so they are
invisible without `--nocapture`. See
[Run the live integration suite](@/guides/integration-tests.md).

## Setting them

Linux and macOS, prefixed onto one command:

```bash
CUCA_BASE_URL=http://10.0.0.7:8000/v1 CUCA_MODEL=your-model-id cargo run --example llamacpp_gemma --features provider-llamacpp
```

Windows (PowerShell), where each assignment is a statement and `;` joins them:

```powershell
$env:CUCA_BASE_URL = "http://10.0.0.7:8000/v1"; $env:CUCA_MODEL = "your-model-id"; cargo run --example llamacpp_gemma --features provider-llamacpp
```

## What is not an environment variable

`tests/plugin_mcp.rs` reads `std::env::args_os()` looking for the literal flag
`--mcp-echo-server`, and calls `std::env::current_exe()` to re-execute itself as
an MCP echo server over stdio. Both are process argv and executable path, not
environment variables, and neither is settable by the reader.

Next page: [The public API surface](@/reference/public-api.md).
