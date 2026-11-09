+++
title = "Environment variables"
description = "Every environment variable read by the library, the examples, the integration tests and the unit-test fixtures. The library reads the environment only in provider-local, as a fallback."
template = "page.html"
weight = 3
+++

# Environment variables

<dl class="page-facts">
<dt>In one line</dt>
<dd>The library reads the environment only in <code>provider-local</code>, as a fallback; the <code>cuca</code> binary reads two API keys and <code>RUST_LOG</code>; the examples, the integration tests and the local unit-test fixtures read the rest</dd>
<dt>You need</dt>
<dd>Nothing; every variable has a default or a documented fallback</dd>
<dt>Read this if</dt>
<dd>You are pointing an example or the test suite somewhere other than a local llama.cpp server, or choosing a local model</dd>
</dl>

No HTTP adapter reads an environment variable. Base URLs, API keys and bearer tokens reach the client only through `CucaClientBuilder`.

Under `provider-local`, the library reads `CUCA_CACHE_DIR`, the platform cache variables and `HF_TOKEN`, only when the matching `LocalModelConfig` field (`cache_dir`, `hf_token`) is `None`. See [Local configuration](@/providers/local/configuration.md#environment-variables).

Everything else below is read by `examples/`, `tests/` or unit-test fixtures in `src/`. The `provider-local` test files (`tests/provider_local*.rs`) parse their variables in one shared module, `tests/local_env/mod.rs`; `examples/local_model.rs` reads its own. The `std::env::temp_dir()` calls in other `src/` tests read no variable.

## Variables at a glance

| Variable | Read by | Default |
|---|---|---|
| `CUCA_BASE_URL` | the examples except `local_model`; `tests/common/mod.rs` | `http://127.0.0.1:1234/v1` |
| `CUCA_MODEL` | the examples except `local_model` and `speculative`; `tests/common/mod.rs` | examples: `google/gemma-4-e4b`; tests: first id the server reports |
| `CUCA_FAST_MODEL` | `examples/speculative.rs` | `google/gemma-4-e4b` |
| `CUCA_SLOW_MODEL` | `examples/speculative.rs` | `google/gemma-4-12b-qat` |
| `CUCA_REQUIRE_LIVE` | `tests/common/mod.rs`, `tests/local_env/mod.rs`, the oracle, tokenizer and chat golden unit tests | unset |
| `CUCA_SEARCH_API_KEY` | `examples/web_search.rs` | none; required |
| `CUCA_SEARCH_BASE_URL` | `examples/web_search.rs` | `https://api.firecrawl.dev` |
| `CUCA_CACHE_DIR` | the library (`provider-local`), the tokenizer and chat golden unit tests | the platform cache directory |
| `HF_TOKEN` | the library (`provider-local`) | none; anonymous access |
| `LOCALAPPDATA`, `USERPROFILE`, `HOME`, `XDG_CACHE_HOME` | the library (`provider-local`), as the platform cache directory | see below |
| `CUCA_API_KEY` | the `cuca` binary: `serve --api-key`, `balance --api-key`, `tui --api-key` | none |
| `CUCA_UPSTREAM_API_KEY` | the `cuca` binary: `balance --upstream-api-key` | none |
| `RUST_LOG` | the `cuca` binary, as the log filter when `--log-level` is absent | `info` |
| `CUCA_LOCAL_MODEL_PATH` | `examples/local_model.rs`, `tests/provider_local.rs`, `_batch.rs`, `_mtp.rs`, `_tiers.rs`, the tokenizer, chat golden and eagle unit tests | none |
| `CUCA_LOCAL_MODEL` | `examples/local_model.rs`, `tests/provider_local.rs`, `_batch.rs`, `_mtp.rs`, `_tiers.rs` (the tests also accept a checkpoint alias such as `deepseek-v41-flash-1017`) | `gemma4-e2b-q4_0`; `_tiers.rs` has no default and skips without a model |
| `CUCA_LOCAL_DOWNLOAD` | `tests/provider_local.rs`, `_batch.rs`, `_mtp.rs` | unset |
| `CUCA_LOCAL_CONVERT_QUANT` | `examples/local_model.rs`, `tests/provider_local.rs`, `_mtp.rs`, `_tiers.rs` | `q8_0` |
| `CUCA_LOCAL_DEVICE` | `examples/local_model.rs`, `tests/provider_local.rs`, `_batch.rs`, `_mtp.rs`, `_tiers.rs`, the `gpu` unit tests | `cpu` |
| `CUCA_LOCAL_GPU_LAYERS` | `examples/local_model.rs`, `tests/provider_local.rs`, `_batch.rs`, `_tiers.rs` | unset |
| `CUCA_LOCAL_EXPERTS` | `examples/local_model.rs`, `tests/provider_local.rs`, `_tiers.rs` | `resident` |
| `CUCA_LOCAL_EXPERTS_REF` | `tests/provider_local_tiers.rs` | `resident` |
| `CUCA_LOCAL_BATCHING` | `examples/local_model.rs`, `tests/provider_local_batch.rs` | `BatchingConfig::default()` |
| `CUCA_LOCAL_ENGRAM` | `examples/local_model.rs`, `tests/provider_local.rs`, `_batch.rs`, `_tiers.rs` (through `tests/local_env/mod.rs`) | `EngramCache::default()` |
| `CUCA_LOCAL_PROMPT`, `CUCA_LOCAL_THINKING`, `CUCA_LOCAL_MAX_TOKENS`, `CUCA_LOCAL_TEMPERATURE` | `examples/local_model.rs` | `Explain CUCA in one sentence.`; thinking on; `1024`; the model's own temperature |
| `CUCA_LOCAL_CHAT_TEMPLATE` | `examples/local_model.rs`, `tests/provider_local.rs`, `_batch.rs` | none |
| `CUCA_LOCAL_DRAFTER_PATH` | `examples/local_model.rs`, `tests/provider_local_mtp.rs`, the eagle unit test | none |
| `CUCA_LOCAL_DRAFTER` | `examples/local_model.rs`, `tests/provider_local_mtp.rs` | none |
| `CUCA_LOCAL_DRAFT_TOKENS` | `tests/provider_local_mtp.rs` | `3` |
| `CUCA_LOCAL_ALLOW_MISMATCH` | `tests/provider_local_mtp.rs` | unset |
| `CUCA_LOCAL_PERF`, `CUCA_LOCAL_PERF_COLD` | `tests/provider_local.rs` | unset |
| `CUCA_LOCAL_PRINT_REPLIES` | `tests/provider_local.rs`, `_mtp.rs`, `_tiers.rs` | unset |
| `CUCA_LOCAL_REF_DIR` | `tests/provider_local.rs`, `tests/provider_local_convert.rs`, the oracle and chat golden unit tests | none |
| `CUCA_LOCAL_CONVERT_ONLY` | `tests/provider_local_convert.rs`, the oracle unit test | none |
| `CUCA_LOCAL_CATALOG_DIR` | the tokenizer and chat golden unit tests | `<cache dir>/huggingface` |
| `CUCA_LOCAL_REF_CKPT`, `CUCA_LOCAL_REF_GGUF`, `CUCA_LOCAL_REF_PREFIX`, `CUCA_LOCAL_REF_TOL` | the `oracle_local` unit test (`CUCA_LOCAL_REF_TOL` also `oracle_dflash`) | none; tolerance `5e-4` |
| `CUCA_LOCAL_DFLASH_REF` | the `oracle_dflash` unit test | none |
| `CUCA_LOCAL_DS41_REF` | `tests/local_env/mod.rs` (`ds41_ref`, no test calls it), the DeepSeek-V4.1 oracle, hyper-connection and DSpark unit tests | none |
| `CUCA_LOCAL_DS41_REAL`, `CUCA_LOCAL_DS41_REAL_RAM`, `CUCA_LOCAL_DS41_REAL_GPU_LAYERS`, `CUCA_LOCAL_DS41_REAL_SPEED`, `CUCA_LOCAL_DS41_REAL_DUMP` | the `real_tests` unit test of the CPU engine and the `real_model_on_the_gpu_*` tests of the GPU engine | none; RAM `90` GiB on the CPU, `85` on a GPU; the others unset |
| `CUCA_DS41_REF_DIR`, `CUCA_DS41_TOKENIZER_JSON` | the DeepSeek-V4.1 tokenizer conversion, Engram token map and tokenizer golden unit tests | the reference copy under `.omp/local-models/ref/deepseek-v41/` |
| `CUCA_CONVERT_PROBE` | the `convert_probe` unit test | none |
| `CUCA_MISTRAL4_GGUF` | a `mla` unit test | none |
| `CUCA_TOK_GGUF`, `CUCA_TOK_CASES`, `CUCA_TOK_MAX_DIFF` | the tokenizer corpus unit test | none; `CUCA_TOK_MAX_DIFF` defaults to `0` |

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

- Examples treat it as a plain override with the literal default `google/gemma-4-e4b`. `examples/speculative.rs` reads `CUCA_FAST_MODEL` and `CUCA_SLOW_MODEL` instead.
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

For the `provider-local` tests the skip line is `SKIP: local model not available: set CUCA_LOCAL_MODEL_PATH, CUCA_LOCAL_MODEL or CUCA_LOCAL_DOWNLOAD=1`. The tokenizer and chat golden unit tests print `SKIP: ...` and, under `CUCA_REQUIRE_LIVE=1`, fail with `required model unavailable: ...`. The how-to is [Run the live integration suite](@/guides/integration-tests.md).

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

- `CUCA_LOCAL_MODEL_PATH` is the path of an existing file or directory. A directory is a Hugging Face checkpoint (`ModelSource::Dir`, converted once to a cached GGUF); anything else is a GGUF file (`ModelSource::File`). It skips the download and wins over `CUCA_LOCAL_MODEL`.
- `CUCA_LOCAL_MODEL` names what to download, parsed with `ModelSource::from_str`: a preset from `ModelSource::PRESETS` (for example `gemma4-e2b-q4_0`, `qwen3-0.6b-q8_0`, `phi4-mini-q4_0`; ASCII case-insensitive), an unverified `repo:revision:file` GGUF, or `repo:revision`, a checkpoint that is downloaded and converted. An unknown value fails with the accepted names. Default: `gemma4-e2b-q4_0`.
- `CUCA_LOCAL_DOWNLOAD=1` selects the default model (Gemma 4 E2B Q4_0) when neither of the two above is set. The tests compare against the literal `1`.
- `CUCA_LOCAL_CONVERT_QUANT` is the weight precision of a converted checkpoint: `f16`, `q8_0` or `q4_0`. Default: `q8_0`.
- `CUCA_LOCAL_CHAT_TEMPLATE` is the path of a Jinja template file whose text becomes `LocalModelConfig::chat_template`. An unreadable file fails the example or panics the test.
- `CUCA_LOCAL_MODEL` also takes, in the tests, a checkpoint alias of `local_env::CHECKPOINT_PRESETS` that is no library preset, such as `deepseek-v41-flash-1017` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, converted once to a cached split GGUF set).

### Device and memory

- `CUCA_LOCAL_DEVICE` sets `LocalModelConfig::device`, parsed with `LocalDevice::from_str`. Default: `cpu`. Empty counts as unset. Accepted forms, ASCII case-insensitive: `cpu`, `vulkan[:<n>|:integrated:<n>]`, `metal[:<n>|:integrated:<n>]`, `cuda[:<n>]`, `rocm[:<n>]`.
- A `+`-separated list such as `vulkan:0+vulkan:1` goes through `LocalModelConfig::set_devices` and splits the model over those devices (tensor parallelism, `LocalModelConfig::shards`). The example, `tests/provider_local.rs` and `tests/provider_local_batch.rs` accept a list. `tests/provider_local_mtp.rs` and `tests/provider_local_tiers.rs` parse one device. The `gpu` unit tests read the list and skip when it is unset or starts with `cpu`.
- A GPU device needs its Cargo feature, for example `provider-local-vulkan`. Without it the load fails with `CucaError::Config`.
- `CUCA_LOCAL_GPU_LAYERS` is `LocalModelConfig::gpu_layers`, a `usize`: the last `n` layers run on the GPU device, the rest on the CPU.
- `CUCA_LOCAL_BATCHING` is `LocalModelConfig::batching`, parsed with `BatchingConfig::from_str`: any subset of `seqs=<n>`, `kv=<tokens>`, `step=<tokens>`, `prefix=on|off`, comma separated.
- The library never reads these variables. The values mean what [Local configuration](@/providers/local/configuration.md) and [GPU](@/providers/local/gpu.md) say.

### Expert placement

- `CUCA_LOCAL_EXPERTS` sets `LocalModelConfig::experts`, parsed with `ExpertPlacement::from_str`: `resident`, `host[:layers=<n>]`, or `tiered:` with `ram=<size>`, `device=<size>` and `threads=<n>`, where `<size>` takes a `k`, `m` or `g` suffix. Empty counts as unset. `tests/provider_local_tiers.rs` needs it and prints `SKIP` without it; it panics when the value is not tiered.
- `CUCA_LOCAL_EXPERTS_REF` is the placement `tests/provider_local_tiers.rs` compares the tiered run against. Default: `resident`.
- The library never reads these variables. Details are on [Tiered expert placement](@/providers/local/expert-tiers.md).

### Engram cache

- `CUCA_LOCAL_ENGRAM` sets `LocalModelConfig::engram`, parsed with `EngramCache::from_str`: any subset of `ram=<size>` (default `4g`, 1 MiB to 1 TiB) and `threads=<n>` (default `16`, 1 to 64), comma separated, for example `ram=8g,threads=32`. A model without Engram tables ignores it.
- The library never reads this variable. Details are on [Local configuration](@/providers/local/configuration.md).

### Drafter

- `CUCA_LOCAL_DRAFTER_PATH` is the path of a drafter file or checkpoint directory. It wins over `CUCA_LOCAL_DRAFTER`.
- `CUCA_LOCAL_DRAFTER` is a preset name (`gemma4-e2b-mtp-f16`, `muse-glimmer-30b-dflash-q4_0`) or `in-trunk`, the head inside the main model's own file (CPU only). The tests also accept `repo:revision:file`.
- Both set `LocalModelConfig::drafter`, to a `DrafterSource` from a `ModelSource` or to `DrafterSource::InTrunk`. See [Drafting](@/providers/local/drafting.md).
- `CUCA_LOCAL_DRAFT_TOKENS` is the `draft_tokens` value of the drafter run in `tests/provider_local_mtp.rs`, or a comma-separated list such as `1,2,3,4` to run one comparison per value. Default: `3`.
- `CUCA_LOCAL_ALLOW_MISMATCH`, set to any value, makes `tests/provider_local_mtp.rs` report a CPU speculative output that differs from plain decoding instead of failing.

### Test switches

- `CUCA_LOCAL_PERF`, set to any value (even empty), enables the long-prompt speed scenario of `tests/provider_local.rs`. `CUCA_LOCAL_PERF_COLD`, set to any value, changes the first words of every round so no round hits the prefix cache.
- `CUCA_LOCAL_PRINT_REPLIES`, set to any value, prints each reply as a `REPLY|...` line.
- `CUCA_LOCAL_CONVERT_ONLY` limits the converted-checkpoint oracle test to repositories whose id contains the text.
- `CUCA_LOCAL_PROMPT` is the request text of `examples/local_model.rs`, `CUCA_LOCAL_THINKING=off` (any case) turns reasoning off in that request, `CUCA_LOCAL_MAX_TOKENS` is its token limit and `CUCA_LOCAL_TEMPERATURE` its temperature (a number; `0` is greedy).

### Reference fixtures

- `CUCA_LOCAL_REF_DIR` is a directory of Hugging Face `transformers` oracle dumps, one `<id>.json` each. When a dump matches the loaded file, `tests/provider_local.rs` also checks that the prompt token count equals the oracle's. The chat golden unit tests check that the rendered prompt and its ids equal the oracle's, and the converted-checkpoint oracle test compares last-position logits and the greedy continuation.
- `CUCA_LOCAL_CATALOG_DIR` is the `huggingface` directory that holds the cached catalog GGUFs. Default: `<cache dir>/huggingface`. A model whose file is absent is skipped.
- `CUCA_LOCAL_REF_CKPT` (a checkpoint directory) or `CUCA_LOCAL_REF_GGUF` (a GGUF file), with `CUCA_LOCAL_REF_PREFIX` (`<prefix>.json` and `<prefix>.bin` written by the reference script), gate the `oracle_local` parity test. `CUCA_LOCAL_REF_TOL` bounds `max |dlogit|`; default `5e-4`.
- `CUCA_LOCAL_DFLASH_REF` is the directory the `oracle_dflash` parity test reads (`target/`, `assistant/`, `ref.json`, `ref.bin`), and it uses `CUCA_LOCAL_REF_TOL` as well.
- `CUCA_LOCAL_DS41_REF` is the directory of the DeepSeek-V4.1 oracle dump: a tiny release-format checkpoint (`target/`), the exact float tensors and metadata (`expected/`) and the torch port's logits. The converter, hyper-connection and DSpark unit tests compare against it and print `SKIP:` without it.
- `CUCA_LOCAL_DS41_REAL` is the directory of the real-weights streaming oracle; with `CUCA_CACHE_DIR` naming the cache that holds the converted DeepSeek-V4.1-Flash-1017 set, it gates the `real_tests` unit test of the CPU engine and the `real_model_on_the_gpu_*` tests (which also need a GPU `CUCA_LOCAL_DEVICE`). `CUCA_LOCAL_DS41_REAL_RAM` is the tiered expert RAM in GiB, default `90` on the CPU and `85` on a GPU. `CUCA_LOCAL_DS41_REAL_GPU_LAYERS=<n>` runs the last `n` layers on the GPU with a 3 GiB device tier. `CUCA_LOCAL_DS41_REAL_SPEED`, set to any value, enables the speed probes, and `CUCA_LOCAL_DS41_REAL_DUMP=<dir>` writes every position's logits.
- `CUCA_DS41_REF_DIR` is the directory of the DeepSeek-V4.1 tokenizer files, and `CUCA_DS41_TOKENIZER_JSON` the path of its `tokenizer.json`. Both default to the reference copy under `.omp/local-models/ref/deepseek-v41/`.
- `CUCA_CONVERT_PROBE` is `<checkpoint dir>|<output file>|<f16|q8_0|q4_0>`: the `convert_probe` unit test converts the checkpoint to that file. A malformed value panics.
- `CUCA_MISTRAL4_GGUF` is the path of a published `mistral4` GGUF; one `mla` unit test reads its header. The eagle unit test reads `CUCA_LOCAL_MODEL_PATH` (that GGUF) and `CUCA_LOCAL_DRAFTER_PATH` (the drafter checkpoint directory).
- `CUCA_TOK_GGUF` is a GGUF whose header supplies the tokenizer and `CUCA_TOK_CASES` a JSON array of `{"text", "ids"}` reference cases. Both are required. `CUCA_TOK_MAX_DIFF` bounds the accepted mismatches; default `0`.

Except `convert_probe`, each of these unit tests prints `SKIP: ...` when its variables are absent.

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
