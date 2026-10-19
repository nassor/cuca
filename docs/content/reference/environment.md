+++
title = "Environment variables"
description = "Every environment variable read by the library, the examples, the integration tests and the unit-test fixtures. The library reads the environment only in provider-local, as a fallback."
template = "page.html"
weight = 3
+++

# Environment variables

<dl class="page-facts">
<dt>In one line</dt>
<dd>The library reads the environment only in <code>provider-local</code>, as a fallback; the examples, the integration tests and the local unit-test fixtures read the rest</dd>
<dt>You need</dt>
<dd>Nothing; every variable has a default or a documented fallback</dd>
<dt>Read this if</dt>
<dd>You are pointing an example or the test suite somewhere other than a local llama.cpp server, or choosing a local model</dd>
</dl>

No HTTP adapter reads an environment variable. Base URLs, API keys and bearer tokens reach the client only through `CucaClientBuilder`.

Under `provider-local`, the library reads `CUCA_CACHE_DIR`, the platform cache variables and `HF_TOKEN`, only when the matching `LocalModelConfig` field (`cache_dir`, `hf_token`) is `None`. See [Local configuration](@/providers/local/configuration.md#environment-variables).

Everything else below is read by `examples/`, `tests/` or unit-test fixtures in `src/`. The `std::env::temp_dir()` calls in other `src/` tests read no variable.

## Variables at a glance

| Variable | Read by | Default |
|---|---|---|
| `CUCA_BASE_URL` | the examples except `local_model`; `tests/common/mod.rs` | `http://127.0.0.1:1234/v1` |
| `CUCA_MODEL` | the same | examples: `google/gemma-4-e4b`; tests: first id the server reports |
| `CUCA_REQUIRE_LIVE` | `tests/common/mod.rs`, the `provider-local` tests, the tokenizer and chat golden unit tests | unset |
| `CUCA_SEARCH_API_KEY` | `examples/web_search.rs` | none; required |
| `CUCA_SEARCH_BASE_URL` | `examples/web_search.rs` | `https://api.firecrawl.dev` |
| `CUCA_CACHE_DIR` | the library (`provider-local`), the tokenizer and chat golden unit tests | the platform cache directory |
| `HF_TOKEN` | the library (`provider-local`) | none; anonymous access |
| `CUCA_LOCAL_MODEL_PATH` | `examples/local_model.rs`, both local test files, the gemma4 golden unit tests | none |
| `CUCA_LOCAL_MODEL` | `examples/local_model.rs`, both local test files | `gemma4-e2b-q4_0` |
| `CUCA_LOCAL_DOWNLOAD` | both local test files | unset |
| `CUCA_LOCAL_DEVICE` | `examples/local_model.rs`, both local test files, the `gpu_` unit tests | `cpu` |
| `CUCA_LOCAL_DRAFTER_PATH` | `examples/local_model.rs`, `tests/provider_local_mtp.rs` | none |
| `CUCA_LOCAL_DRAFTER` | `examples/local_model.rs`, `tests/provider_local_mtp.rs` | none |
| `CUCA_LOCAL_DRAFT_TOKENS` | `tests/provider_local_mtp.rs` | `3` |
| `CUCA_LOCAL_REF_DIR` | `tests/provider_local.rs`, the chat golden unit tests | none |
| `CUCA_LOCAL_CATALOG_DIR` | the tokenizer and chat golden unit tests | `<cache dir>/huggingface` |

## Examples and live llama.cpp tests

### `CUCA_BASE_URL`

The base URL an example or the test harness targets. The harness declares its fallback as a constant, and every example carries the same literal:

```rust,name=tests/common/mod.rs
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";
```

- The value is used as given, so it must include every path segment before the route. For an OpenAI-compatible route it normally ends in `/v1`.
- `llama-server` listens on 8080 by default, so the suite expects it started with `--port 1234`, or this variable pointed at its port.
- Unset means the default. It never fails.

### `CUCA_MODEL`

The model id the request carries.

- Examples treat it as a plain override with the literal default `google/gemma-4-e4b`.
- The test harness treats an unset or empty value as a request to discover the id. It issues `GET {base_url}/models` with a two second timeout and takes the first `data[].id`.
- With neither the variable nor a reachable server, the harness panics instead of sending a guessed id:

```text,name=The panic with the resolved base URL interpolated
no model id: set CUCA_MODEL or start llama-server at http://127.0.0.1:1234/v1
```

### `CUCA_REQUIRE_LIVE`

Controls what a server-dependent or model-dependent test does when its server or model is missing.

- Unset, or any value other than `1`: the test prints a `SKIP` line and returns. `true`, `yes` and `0` behave as unset.
- Exactly `1`: the same condition panics.
- Skip lines use `eprintln!`, which libtest captures, so they are invisible without `--nocapture`.

```text,name=The skip line and the panic for the llama.cpp tests
SKIP: llama.cpp not reachable: <reason>
CUCA_REQUIRE_LIVE=1 but llama.cpp is unreachable: <reason>
```

For the `provider-local` tests the skip line is `SKIP: local model not available: ...`. The tokenizer and chat golden unit tests print `SKIP: ...` and, under `CUCA_REQUIRE_LIVE=1`, fail with `required model unavailable: ...`. The how-to is [Run the live integration suite](@/guides/integration-tests.md).

### `CUCA_SEARCH_API_KEY` and `CUCA_SEARCH_BASE_URL`

Read by `examples/web_search.rs` only.

- `CUCA_SEARCH_API_KEY` is a Firecrawl API key. Without it the example prints one line and exits with code 0.
- `CUCA_SEARCH_BASE_URL` overrides the search endpoint. Unset means `https://api.firecrawl.dev`.

## Local model variables

### Cache and download

- `CUCA_CACHE_DIR` is the model download cache directory, used as given. Empty counts as unset. It is read when `LocalModelConfig::cache_dir` is `None`.
- `HF_TOKEN` is a Hugging Face bearer token, read when `LocalModelConfig::hf_token` is `None`. Empty counts as unset. The pinned default model downloads anonymously.

With `CUCA_CACHE_DIR` unset, the cache directory is:

| Platform | Directory |
|---|---|
| Windows | `%LOCALAPPDATA%\cuca`, else `%USERPROFILE%\AppData\Local\cuca` |
| macOS | `$HOME/Library/Caches/cuca` |
| Linux and others | `$XDG_CACHE_HOME/cuca`, else `$HOME/.cache/cuca` |

With none of them set, `LocalModel::load` fails with `CucaError::Config`. Downloads land at `<cache>/huggingface/<repo>/<revision>/<file>`.

### Model selection

- `CUCA_LOCAL_MODEL_PATH` is the path of an existing GGUF file of any supported architecture. It skips the download and wins over `CUCA_LOCAL_MODEL`.
- `CUCA_LOCAL_MODEL` names what to download: a preset from `ModelSource::PRESETS` (for example `gemma4-e2b-q4_0`, `qwen3-0.6b-q8_0`, `llama3.2-1b-q4_0`; ASCII case-insensitive) or an unverified `repo:revision:file`. It is parsed with `ModelSource::from_str`, and an unknown value fails with the accepted names. Default: `gemma4-e2b-q4_0`.
- `CUCA_LOCAL_DOWNLOAD=1` downloads the default model (Gemma 4 E2B Q4_0) into the cache for a test run. Setting `CUCA_LOCAL_MODEL` downloads that model without it. The tests compare against the literal `1`.

### Device

`CUCA_LOCAL_DEVICE` sets `LocalModelConfig::device`, parsed with `LocalDevice::from_str`. Default: `cpu`. Empty counts as unset in the examples and tests.

- Accepted forms, ASCII case-insensitive: `cpu`, `vulkan`, `vulkan:<n>`, `vulkan:integrated:<n>`, `metal` (same forms), `cuda[:<n>]`, `rocm[:<n>]`.
- A GPU device needs its Cargo feature, for example `provider-local-vulkan`. Without it the load fails with `CucaError::Config`.
- The library never reads this variable. The `gpu_` unit tests read it and panic on an unparseable value.

### Drafter (multi-token prediction)

- `CUCA_LOCAL_DRAFTER_PATH` is the path of a drafter file. It wins over `CUCA_LOCAL_DRAFTER`.
- `CUCA_LOCAL_DRAFTER` is a preset name (`gemma4-e2b-mtp-f16`, the drafter of a Gemma 4 target) or `in-trunk`, the head inside the main model's own file (CPU only).
- Both set `LocalModelConfig::drafter` to `DrafterSource::File` or `DrafterSource::InTrunk`.
- `CUCA_LOCAL_DRAFT_TOKENS` is the `draft_tokens` value of the drafter run in `tests/provider_local_mtp.rs`, or a comma-separated list such as `1,2,3,4` to run one comparison per value. Default: `3`.

### Reference fixtures

- `CUCA_LOCAL_REF_DIR` is a directory of Hugging Face `transformers` oracle dumps, one `<id>.json` each. When a dump matches the loaded file, the live suite also checks that the prompt token count equals the oracle's. The chat golden unit tests check that the rendered prompt and its ids equal the oracle's.
- `CUCA_LOCAL_CATALOG_DIR` is the `huggingface` directory that holds the cached catalog GGUFs. Default: `<cache dir>/huggingface`. A model whose file is absent is skipped.

Each of these unit tests prints `SKIP: ... not set` when its variable is absent.

## Setting them

Linux and macOS, prefixed onto one command:

```bash
CUCA_BASE_URL=http://10.0.0.7:8000/v1 CUCA_MODEL=your-model-id cargo run --example llamacpp_gemma --features provider-llamacpp
```

Windows (PowerShell), where each assignment is a statement and `;` joins them:

```powershell
$env:CUCA_BASE_URL = "http://10.0.0.7:8000/v1"; $env:CUCA_MODEL = "your-model-id"; cargo run --example llamacpp_gemma --features provider-llamacpp
```

A `$env:` assignment lasts for the PowerShell session. `Remove-Item Env:NAME` clears it.

## Not environment variables

- `tests/plugin_mcp.rs` and `examples/mcp.rs` read `std::env::args_os()` for a flag (`--mcp-echo-server`, `--mcp-add-server`) and call `std::env::current_exe()` to re-execute themselves as an MCP server over stdio. Both are process arguments and the executable path.
- `src/plugins/mcp.rs` uses `env!("CARGO_PKG_VERSION")`, which Cargo sets at compile time.
