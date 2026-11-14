+++
title = "Configuration files"
description = "How cuca is configured: no environment variables, the KDL mapping and its placeholders, the cuca.kdl service file, the model keys, the cuca-test.kdl test file and the flags of the examples."
template = "page.html"
weight = 3
+++

# Configuration files

<dl class="page-facts">
<dt>In one line</dt>
<dd>cuca reads no environment variable for its settings: the <code>cuca</code> binary reads one KDL file, a library user fills Rust structs, the tests read <code>cuca-test.kdl</code> and the examples take flags</dd>
<dt>You need</dt>
<dd>The <code>config-kdl</code> feature to read a KDL file; <code>provider-local</code> and <code>server</code> enable it</dd>
<dt>Read this if</dt>
<dd>You are writing a <code>cuca.kdl</code>, a <code>LocalModelConfig</code> file or a <code>cuca-test.kdl</code>, or pointing an example at another server</dd>
</dl>

## No environment variables

No setting of the library, the binary, the examples or the tests comes from an environment variable. Two reads remain:

- A `${VAR}` placeholder that a KDL file writes itself resolves against the file's `variables` block, then the process environment. See [Placeholders](#placeholders-and-the-variables-block).
- With `LocalModelConfig::cache_dir` set to `None`, the model cache is the platform cache directory. The library asks the operating system for it (`LOCALAPPDATA`, `USERPROFILE`, `HOME`, `XDG_CACHE_HOME`).

No HTTP adapter reads anything. Base URLs, API keys and bearer tokens reach the client through `CucaClientBuilder`. A library user configures the local model with Rust structs, and the KDL keys are the field names of the same structs, so a setting is spelled the same in both.

## The KDL mapping

The reader is `cuca::config`, behind the `config-kdl` feature. It knows no key names, so seven rules apply to every section:

1. A document and every children block is a table; each node contributes one entry keyed by its name.
2. Properties (`key=value`) are scalar entries of the node's own table. Writing one property twice on a node is an error.
3. A node with one argument is a scalar, and a node with several arguments is an array.
4. A node with properties or children is a table. One leading argument is stored under the key `id`.
5. Sibling nodes with the same name collapse into an array, in document order.
6. A node with nothing on it is an empty table.
7. Numbers map to integers or floats. `#null` and type annotations are errors.

A list written once is a single value, so every list field accepts one value or several. A key that is not a field of the struct is an error that names the key.

Two value forms need more than a plain scalar:

- A size is an integer, or a string of digits with an optional `k`, `m` or `g` suffix, in powers of 1024: `"512m"`, `"4g"`.
- A duration is a string of digits with `ms`, `s`, `m` or `h`: `"500ms"`, `"60s"`. A bare integer is an error.

The entry points are `cuca::config::load_kdl_file::<T>(path)` and `load_kdl_str::<T>(text)` for any `Deserialize` type, and `from_kdl_str` for the raw value tree. Every error is a `CucaError::Config` that names the file when there is one.

## Placeholders and the variables block

A placeholder is `${NAME}` or `${NAME:-default}`. The reader substitutes placeholders in the raw text before it parses the file, comments included. A name resolves against the top-level `variables` block first, then the process environment. A name that is neither declared nor set, and has no default, is an error that names it.

```kdl,name=A secret from the environment, empty when unset
variables {
    API_KEY "change-me"
}

serve {
    api_key "${API_KEY}"
    model "chat" {
        source "qwen3-0.6b-q8_0"
        hf_token "${HF_TOKEN:-}"
    }
}
```

- A declared value may reference other names. A cycle is an error.
- A value is inserted verbatim, so one that holds a backslash or a double quote ends the KDL string it lands in. The error names the variables involved and never prints a value.
- The `variables` block is removed before the rest of the file deserializes.

Delete the `API_KEY` entry above to read `API_KEY` from the environment instead.

## The cuca file

`cuca serve` and `cuca balance` take one option, `--config <FILE>` (`-c`), default `cuca.kdl`. The file deserializes into `cuca::server::CucaConfig` and has three sections. Relative paths inside the file resolve against the working directory. `examples/configs/cuca.kdl` in the repository is the whole file:

```kdl,name=examples/configs/cuca.kdl
variables {
    API_KEY "change-me"          // delete the entry to read API_KEY from the environment
}

log format="pretty" level="info"

serve {
    listen "127.0.0.1:8000"
    api_key "${API_KEY}"
    stats_api #true
    latency_window "60s"

    model "chat" {
        source "qwen3-0.6b-q8_0"
        batching max_sequences=8
    }
    model {
        source file="/models/gemma-4-E4B-it-Q4_0.gguf"
        device "vulkan:0"
        gpu_layers 20
    }
}

balance {
    listen "127.0.0.1:8080"
    upstreams "http://10.0.0.1:8000" "http://10.0.0.2:8000"
    upstream_api_key "${API_KEY}"
    policy "prefix-affinity"
    health_interval "1s"
    stats_api #true
}
```

| Section | Keys | Page |
|---|---|---|
| `log` | `format` (`pretty` or `json`), `level` (a level or `target=level,...`, default `info`) | [Stats and metrics](@/server/stats-and-metrics.md) |
| `serve` | `listen`, `api_key`, `stats_api`, `latency_window`, `model` | [The model server](@/server/model-server.md) |
| `balance` | `listen`, `upstreams`, `upstream_api_key`, `api_key`, `policy`, `health_interval`, `latency_window`, `stats_api` | [The balancer](@/server/balancer.md) |

`cuca tui` does not read the file. It takes `--url`, `--api-key` and `--interval-ms`; see [The terminal UI](@/server/tui.md).

## The model keys

A `model` node in `serve`, the file of the `local_model` example and `local { model { ... } }` in the test file are all a `LocalModelConfig`. The keys are its field names. `examples/configs/local_model.kdl` in the repository lists every key.

| Key | Form |
|---|---|
| `source` | `"preset"`, `"repo:revision:file"`, `"repo:revision"`, `file="..."`, `dir="..."`, `{ checkpoint repo=".." revision=".." }`, or `{ hugging_face repo=.. revision=.. file=.. sha256=.. { part file=.. sha256=..; template repo=.. revision=.. file=.. } }` |
| `device` | `"cpu"`, `"vulkan:0"`, `"metal"`, `"cuda:0"`, `"rocm:0"` |
| `shards` | further devices that split the model: `"vulkan:1"` |
| `gpu_layers`, `context_length` | integers |
| `cache_dir`, `hf_token` | a path, a Hugging Face token; `shard_dirs` takes several paths |
| `chat_template`, `chat_template_file` | the template text, or the path of a Jinja file |
| `sampling` | `temperature=.. top_k=.. top_p=.. seed=..` |
| `batching` | `max_sequences=8 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false` |
| `experts` | `"resident"`, `{ tiered ram_bytes="4g" device_bytes="6g" loader_threads=8 }`, `{ host layers=4 }`, `{ host }` |
| `engram` | `ram_bytes="2g" reader_threads=8` |
| `convert` | `quant="q4_0" engram_quant="q8_0"` |
| `drafter`, `draft_tokens` | `"in-trunk"` or any `source` form; an integer |

- `cache_dir` unset means the platform cache directory. A `hf_token` that is unset or empty means anonymous access, so `"${HF_TOKEN:-}"` is a safe spelling.
- `chat_template` wins over `chat_template_file`, which wins over the template a preset pins. An unreadable file is a `CucaError::Io`.
- A model of a `serve` section is named by its argument. Without one the name is the preset's name, else the file stem, the checkpoint repository name or the directory name.
- Field defaults and ranges are on [Local configuration](@/providers/local/configuration.md).

Without a `cache_dir`, the cache directory is:

| Platform | Directory |
|---|---|
| Windows | `%LOCALAPPDATA%\cuca`, else `%USERPROFILE%\AppData\Local\cuca` |
| macOS | `$HOME/Library/Caches/cuca` |
| Linux and others | `$XDG_CACHE_HOME/cuca`, else `$HOME/.cache/cuca` |

With none of them available, `LocalModel::load` fails with `CucaError::Config` and asks for `LocalModelConfig::cache_dir`. Downloads land at `<cache>/huggingface/<repo>/<revision>/<file>`.

## The test file

Every live and oracle test reads one file, `cuca-test.kdl`, at the repository root. The file is gitignored. Copy `cuca-test.example.kdl`, next to it, which lists every key, and uncomment what you need. Without the file every live test prints `SKIP:` and passes. The how-to is [Run the live integration suite](@/guides/integration-tests.md).

An invalid file panics and names the key. The llama.cpp suites read the file only when `config-kdl` is among the `--features`; an existing file with the feature off panics and says so.

### General keys

| Key | Meaning |
|---|---|
| `require_live` | `#true` turns every skip into a failure: an unreachable server, a missing model or a missing oracle panics. Default `#false` |
| `cache_dir` | the model cache of every test that downloads or converts; the platform directory when absent |
| `llamacpp` | `base_url="..."` (default `http://127.0.0.1:1234/v1`) and `model="..."` (default: the first id `GET {base_url}/models` reports) of the llama.cpp suites |

With no model id configured and no server answering, the harness panics with ``no model id: set `llamacpp model="..."` in cuca-test.kdl or start llama-server at <base_url>``. The skip line of the llama.cpp suites is `SKIP: llama.cpp not reachable: <reason>`.

### The local model section

`local` configures `tests/provider_local*.rs`. Without a `model { source ... }` node and without `download_default`, the suites print `` SKIP: local model not available: set `local { model { source ... } }` or `local { download_default #true }` in cuca-test.kdl ``.

| Key | Meaning |
|---|---|
| `model { ... }` | a `LocalModelConfig`; see [The model keys](#the-model-keys). `cache_dir` falls back to the top-level one |
| `download_default` | `#true` selects the default preset (Gemma 4 E2B Q4_0) when `model` has no `source` |
| `experts_ref` | the placement `provider_local_tiers.rs` compares the tiered run against; default `resident` |
| `draft_tokens` | one draft length, or several (`draft_tokens 1 2 3`) for one comparison each in `provider_local_mtp.rs`; default `3` |
| `allow_mismatch` | `provider_local_mtp.rs` reports a CPU speculative output that differs from plain decoding instead of failing |
| `print_replies` | prints each reply as a `REPLY|...` line |
| `perf`, `perf_cold` | the long-prompt speed scenario of `provider_local.rs`; `perf_cold` changes the first words of every round so none hits the prefix cache |

`tests/provider_local_tiers.rs` needs `model { experts { tiered ... } }` and skips without it. The `gpu` unit tests read `device` and `shards` and skip when there is no `device` or it is `cpu`. `provider_local_mtp.rs` and `provider_local_tiers.rs` accept one device and no shards.

### The oracle section

`oracle` holds the reference dumps and real-model files of the local engine's unit tests. A test whose keys are absent prints `SKIP:` and passes.

| Key | Meaning |
|---|---|
| `ref_dir` | a directory of Hugging Face `transformers` oracle dumps, one `<id>.json` each. Checks the prompt token count, the rendered chat prompt and its ids, and the converted-checkpoint logits |
| `convert_only` | limits the converted-checkpoint oracle test to repositories whose id contains the text |
| `catalog_dir` | the `huggingface` directory of cached catalog GGUFs; default `<cache dir>/huggingface` |
| `ref_ckpt`, `ref_gguf`, `ref_prefix` | a checkpoint directory or a GGUF, and the `<prefix>.json` and `<prefix>.bin` the reference script wrote; gate the `oracle_local` parity test |
| `ref_tol` | bound on `max \|dlogit\|` for `oracle_local` and `oracle_dflash`; default `5e-4` |
| `dflash_ref` | the directory `oracle_dflash` reads (`target/`, `assistant/`, `ref.json`, `ref.bin`) |
| `convert_probe` | `<checkpoint dir>\|<output file>\|<f16\|q8_0\|q4_0>`: the `convert_probe` test converts the checkpoint to that file. A malformed value panics |
| `tok_gguf`, `tok_cases` | a GGUF whose header supplies the tokenizer, and a JSON array of `{"text", "ids"}` cases; both are required by the tokenizer corpus test |
| `tok_max_diff` | the accepted mismatches of that test; default `0` |
| `mistral4_gguf` | a published `mistral4` GGUF; one `mla` unit test reads its header |

`oracle { ds41 { ... } }` serves DeepSeek-V4.1:

| Key | Meaning |
|---|---|
| `oracle_dir` | the oracle dump: a tiny release-format checkpoint (`target/`), the exact float tensors and metadata (`expected/`) and the torch port's logits |
| `tokenizer_ref_dir`, `tokenizer_json` | the tokenizer files directory and the path of its `tokenizer.json`; they default to the reference copy under `.omp/local-models/ref/deepseek-v41/` |
| `real_dir` | the real-weights streaming oracle; with `cache_dir` naming the cache that holds the converted DeepSeek-V4.1-Flash set, it gates the `real_tests` unit test of the CPU engine and the `real_model_on_the_gpu_*` tests |
| `real_ram_gib` | the tiered expert RAM in GiB; default 90 on the CPU and 85 on a GPU |
| `real_gpu_layers` | runs the last `n` layers on the GPU with a 3 GiB device tier |
| `real_ctx`, `real_seqs`, `real_rows` | context length, sequence count and row count of the real-model runs |
| `real_speed`, `real_paths`, `real_qdq` | `#true` enables the speed probes, the GPU engine's production-path check and the quantize-dequantize check |
| `real_dump_dir` | a directory that receives every position's logits |

`oracle { g4moe { ... } }` serves Gemma 4 MoE:

| Key | Meaning |
|---|---|
| `oracle_dir` | the directory `make_ref.py --out` writes: `tiny-q4.gguf`, `tiny-f32.gguf`, `expected.json` and the perturbed `expected.<mistake>.json` |
| `real`, `real_ref` | the real `gemma-4-26B-A4B-it-Q4_0.gguf` and the `ref_llama.py all` output on it; the header test needs only `real` |
| `trunc`, `trunc_ref` | a prefix of the real file written by `make_trunc.py` and the `ref_llama.py logits` output on it |

A real-model GPU test needs a GPU `device` in `local { model { ... } }` and the matching Cargo feature, for example `provider-local-vulkan`.

## Example flags

Every example takes flags and reads no environment variable. `--base-url` defaults to `http://127.0.0.1:1234/v1` and `--model` to `google/gemma-4-e4b`. The value is used as given, so it includes every path segment before the route, normally `/v1`.

| Example | Flags |
|---|---|
| `llamacpp_gemma`, `stream_all_blocks`, `custom_plugin`, `cost`, `cost_otel`, `entity_extraction`, `guardrails`, `hitl`, `mcp`, `memory`, `prompt_cache`, `rate_limit`, `redaction`, `replay`, `sandbox`, `scheduler`, `session_log`, `skills`, `subagent`, `telemetry`, `vector_store` | `--base-url`, `--model` |
| `speculative` | `--base-url`, `--fast-model` (default `google/gemma-4-e4b`), `--slow-model` (default `google/gemma-4-12b-qat`) |
| `web_search` | `--base-url`, `--model`, `--search-api-key` (required; without it the example prints one line and exits with code 0), `--search-base-url` (default `https://api.firecrawl.dev`) |
| `local_model` | `--config <FILE>` (a `LocalModelConfig`; the default configuration without it), `--prompt` (default `Explain CUCA in one sentence.`), `--max-tokens` (default 1024), `--temperature` (0 is greedy; default the model's own), `--thinking off` |

Flags go after `--`:

```bash,name=Runs the same on all three platforms
cargo run --example llamacpp_gemma --features provider-llamacpp -- --base-url http://10.0.0.7:8000/v1 --model your-model-id
```

## Not configuration

- `tests/plugin_mcp.rs` and `examples/mcp.rs` read the process arguments for a flag (`--mcp-echo-server`, `--mcp-add-server`) and call `std::env::current_exe()` to re-execute themselves as an MCP server over stdio.
- `src/plugins/mcp.rs` uses `env!("CARGO_PKG_VERSION")`, which Cargo sets at compile time.
- `RUSTFLAGS` is a cargo build flag. Local-model builds use `RUSTFLAGS="-C target-cpu=native"` (PowerShell: `$env:RUSTFLAGS = "-C target-cpu=native"`).
