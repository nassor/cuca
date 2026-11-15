+++
title = "Configuration files"
description = "How cuca is configured: no environment variables, the KDL file format and its placeholders, the cuca.kdl service file, the model keys, the cuca-test.kdl test file and the flags of the examples."
template = "page.html"
weight = 3
+++

# Configuration files

<dl class="page-facts">
<dt>In one line</dt>
<dd>cuca reads no environment variable for its settings: the <code>cuca</code> binary reads one KDL file, library code fills Rust structs, tests read <code>cuca-test.kdl</code> and the examples take flags</dd>
<dt>You need</dt>
<dd>The <code>config-kdl</code> feature to read a KDL file; <code>provider-local</code> and <code>server</code> turn it on</dd>
<dt>Read this if</dt>
<dd>You are writing a <code>cuca.kdl</code>, a model file or a <code>cuca-test.kdl</code>, or pointing an example at another server</dd>
</dl>

KDL is a small text format for settings. A line is a node with a name, then values. This page shows how cuca reads it.

## Where settings come from

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 200" role="img" aria-labelledby="cfgx-title cfgx-desc">
  <title id="cfgx-title">Four places settings come from, and what reads each one.</title>
  <desc id="cfgx-desc">A cuca.kdl file is read by the cuca binary. Rust structs are filled by library code. A cuca-test.kdl file is read by the tests. Flags after two dashes are read by the examples. The KDL files all go through the same reader.</desc>
  <defs>
    <marker id="cfgx-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Settings and who reads them</text>
  <rect class="blk blk-ctl" x="6" y="40" width="190" height="50" rx="6"/>
  <text class="t-lbl t-ctl" x="18" y="62">cuca.kdl</text>
  <text class="t-sm" x="18" y="78">cuca serve, cuca balance</text>
  <rect class="blk blk-ctl" x="6" y="104" width="190" height="50" rx="6"/>
  <text class="t-lbl t-ctl" x="18" y="126">cuca-test.kdl</text>
  <text class="t-sm" x="18" y="142">live tests</text>
  <rect class="blk blk-data" x="262" y="72" width="140" height="50" rx="6"/>
  <text class="t-lbl t-data" x="274" y="94">KDL reader</text>
  <text class="t-sm" x="274" y="110">cuca::config</text>
  <rect class="blk" x="468" y="40" width="186" height="50" rx="6"/>
  <text class="t-lbl" x="480" y="62">Rust structs</text>
  <text class="t-sm" x="480" y="78">same field names</text>
  <rect class="blk" x="468" y="104" width="186" height="50" rx="6"/>
  <text class="t-lbl" x="480" y="126">example flags</text>
  <text class="t-sm" x="480" y="142">after -- on the command</text>
  <path class="arw arw-ctl" d="M198 65 H260" marker-end="url(#cfgx-arrow)"/>
  <path class="arw arw-ctl" d="M198 129 H230 V108 H260" marker-end="url(#cfgx-arrow)"/>
  <path class="arw arw-data" d="M404 97 H466" marker-end="url(#cfgx-arrow)"/>
  <text class="t-sm" x="6" y="188">Environment variables are read only through ${NAME} placeholders written in a file.</text>
</svg>
</div>
<p class="dgm-cap"><b>One reader, one set of names.</b> The KDL reader fills the same structs a library user fills in Rust, so a setting is spelled the same in a file and in code. Example flags are separate command-line options.</p>
</div>

## No environment variables

No setting of the library, the binary, the examples or the tests comes from an environment variable. Two reads remain:

- A `${VAR}` placeholder that a KDL file writes itself. It resolves against the file's `variables` block, then the process environment. See [Placeholders](#placeholders-and-the-variables-block).
- With `cache_dir` unset, the model cache is the platform cache directory. The library asks the operating system for it (`LOCALAPPDATA`, `USERPROFILE`, `HOME`, `XDG_CACHE_HOME`).

No HTTP adapter reads anything either. Base URLs, API keys and bearer tokens reach the client through `CucaClientBuilder`.

## The KDL mapping

The reader is `cuca::config`, behind the `config-kdl` feature. It knows no key names. These seven rules apply to every section:

1. The document and every children block is a table. Each node adds one entry named after the node.
2. Properties (`key=value`) are entries of the node's own table. Writing one property twice on a node is an error.
3. A node with one argument is a single value. A node with several arguments is a list.
4. A node with properties or children is a table. One leading argument is stored under the key `id`.
5. Sibling nodes with the same name collapse into a list, in document order.
6. A node with nothing on it is an empty table.
7. Numbers become integers or floats. `#null` and type annotations are errors.

A key that is not a field of the struct is an error that names the key. A list written once is a single value, so every list field accepts one value or several.

```kdl,name=Rules 3 and 5: one value, a list, a list again
device "vulkan:0"                // one value
shards "vulkan:1" "vulkan:2"     // a list of two
model "a" { source "qwen3-0.6b-q8_0" }
model "b" { source "qwen3-0.6b-q8_0" }   // two sibling nodes: a list of two models
```

Two value forms need more than a plain number:

| Form | Written as | Examples |
|---|---|---|
| Size | an integer, or digits with a `k`, `m` or `g` suffix (powers of 1024) | `4294967296`, `"512m"`, `"4g"` |
| Duration | digits with `ms`, `s`, `m` or `h`; a bare integer is an error | `"500ms"`, `"60s"`, `"5m"`, `"1h"` |

Entry points: `cuca::config::load_kdl_file::<T>(path)` and `load_kdl_str::<T>(text)` for any `Deserialize` type, and `from_kdl_str` for the raw value tree. Every failure is a `CucaError::Config` that names the file when there is one. See [Error types](@/reference/errors.md).

## Placeholders and the variables block

A placeholder is `${NAME}` or `${NAME:-default}`. The reader replaces placeholders in the text before it parses the file, comments included. It looks the name up in the top-level `variables` block first, then in the process environment. A name that is in neither and has no default is an error that names it.

```kdl,name=A secret with a fallback, and an empty default
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

- A declared value may refer to other names. A cycle is an error.
- A value is inserted as it is, so one that holds a backslash or a double quote ends the KDL string it lands in. The error names the variables involved and never prints a value.
- The `variables` block is removed before the rest of the file is read.
- Delete the `API_KEY` entry above to read `API_KEY` from the environment instead.

## The cuca.kdl file

`cuca serve` and `cuca balance` take one option, `--config <FILE>` (`-c`), default `cuca.kdl`. Relative paths inside the file resolve against the working directory. The file has three sections, all optional, and unknown keys are rejected. `examples/configs/cuca.kdl` in the repository is a full example:

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

| Section | Key | Default | Meaning |
|---|---|---|---|
| `log` | `format` | `pretty` | `pretty` or `json` |
| `log` | `level` | `info` | a level, or a filter such as `info,cuca=debug` |
| `serve` | `listen` | `127.0.0.1:8000` | address of the HTTP listener |
| `serve` | `model` | none | one node per model, 1 to 16, unique names |
| `serve` | `api_key` | none | when set, every route except `GET /health` needs `Authorization: Bearer <key>` |
| `serve` | `stats_api` | `#false` | serve `/cuca/v1/stats` and `/cuca/v1/stats/stream` |
| `serve` | `latency_window` | `"60s"` | window of the latency figures |
| `balance` | `listen` | `127.0.0.1:8000` | address of the HTTP listener |
| `balance` | `upstreams` | none | base URLs of the model servers |
| `balance` | `upstream_api_key` | none | key sent to the upstreams |
| `balance` | `api_key` | none | key clients must send, as for `serve` |
| `balance` | `policy` | `prefix-affinity` | `prefix-affinity`, `least-loaded` or `round-robin` |
| `balance` | `health_interval` | `"1s"` | time between polls of each upstream, 100 ms to 60 s |
| `balance` | `latency_window` | `"60s"` | window of the latency figures |
| `balance` | `stats_api` | `#false` | serve the stats API |

Details: [The model server](@/server/model-server.md), [The balancer](@/server/balancer.md) and [Stats and metrics](@/server/stats-and-metrics.md).

`cuca tui` does not read the file. It takes `--url` (default `http://127.0.0.1:8000`), `--api-key` and `--interval-ms` (default 1000). See [The terminal UI](@/server/tui.md).

## The model keys

A `model` node in `serve`, the file of the `local_model` example and `local { model { ... } }` in the test file are all a `LocalModelConfig`. The keys are its field names, and `examples/configs/local_model.kdl` lists every one. [Configuration](@/providers/local/configuration.md) has the meaning, default and range of each key.

```kdl,name=A model with a few common keys
source "qwen3-0.6b-q8_0"
device "vulkan:0"
context_length 8192
batching max_sequences=4
```

| Key | Written as |
|---|---|
| `source` | `"preset"`, `"repo:revision:file"`, `"repo:revision"`, `file="..."` or `dir="..."` |
| `device`, `shards` | `"cpu"`, `"vulkan:0"`, `"metal"`, `"cuda:0"`, `"rocm:0"`; `shards` lists further devices |
| `gpu_layers`, `context_length` | integers |
| `cache_dir`, `shard_dirs`, `hf_token` | paths and a Hugging Face token |
| `chat_template`, `chat_template_file` | the template text, or the path of a Jinja file |
| `sampling` | `temperature=.. top_k=.. top_p=.. seed=..` |
| `batching` | `max_sequences=8 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false` |
| `experts` | `"resident"`, `{ tiered ram_bytes="4g" device_bytes="6g" loader_threads=8 }`, `{ host layers=4 }` or `{ host }` |
| `engram` | `ram_bytes="2g" reader_threads=8` |
| `convert` | `quant="q4_0" engram_quant="q8_0"` |
| `drafter`, `draft_tokens` | `"in-trunk"` or any `source` form; an integer |

- An unset `cache_dir` means the platform cache directory (next table). An unset or empty `hf_token` means anonymous access, so `"${HF_TOKEN:-}"` is a safe spelling.
- `chat_template` wins over `chat_template_file`, which wins over the template a preset pins. An unreadable file is a `CucaError::Io`.
- A `serve` model is named by its argument. Without one, the name is the preset name, else the file stem, the checkpoint repository name or the directory name.

| Platform | Model cache directory |
|---|---|
| Windows | `%LOCALAPPDATA%\cuca`, else `%USERPROFILE%\AppData\Local\cuca` |
| macOS | `$HOME/Library/Caches/cuca` |
| Linux and others | `$XDG_CACHE_HOME/cuca`, else `$HOME/.cache/cuca` |

With none of these available, `LocalModel::load` fails with `CucaError::Config` and asks for `LocalModelConfig::cache_dir`. Downloads land at `<cache>/huggingface/<repo>/<revision>/<file>`.

## The test file

The live tests read one file, `cuca-test.kdl`, at the repository root. It is not in git. Copy `cuca-test.example.kdl`, which lists every key, and uncomment what you need. Without the file, each live test prints `SKIP:` and passes. [Run the live integration suite](@/guides/integration-tests.md) shows how.

An invalid file panics and names the key. The llama.cpp tests read the file only when `config-kdl` is in `--features`; an existing file with the feature off panics and says so.

| Key | Meaning |
|---|---|
| `require_live` | `#true` turns every skip into a failure. Default `#false` |
| `cache_dir` | the model cache of every test that downloads or converts |
| `llamacpp` | `base_url="..."` (default `http://127.0.0.1:1234/v1`) and `model="..."` (default: the first id that `GET {base_url}/models` reports) |
| `local { model { ... } }` | a `LocalModelConfig` for the `provider_local*` tests; see [The model keys](#the-model-keys) |
| `local { download_default }` | `#true` runs the default preset (Gemma 4 E2B Q4_0) when `model` has no `source` |
| `local { experts_ref }` | the placement that `provider_local_tiers.rs` compares against; default `resident` |
| `local { draft_tokens }` | one draft length, or several such as `draft_tokens 1 2 3`; default `3` |
| `local { allow_mismatch }` | report a CPU speculative output that differs from plain decoding instead of failing |
| `local { print_replies }` | print each reply on a line that starts with `REPLY` |
| `local { perf, perf_cold }` | the long-prompt speed scenario; `perf_cold` avoids prefix-cache hits |
| `oracle { ... }` | reference dumps and real-model files for the local engine's unit tests; the keys are in `cuca-test.example.kdl` |

- With no model id and no server answering, the llama.cpp tests panic with ``no model id: set `llamacpp model="..."` in cuca-test.kdl or start llama-server at <base_url>``. Their skip line is `SKIP: llama.cpp not reachable: <reason>`.
- Without a `model { source ... }` node and without `download_default`, the local tests print `` SKIP: local model not available: set `local { model { source ... } }` or `local { download_default #true }` in cuca-test.kdl ``.
- A test that needs a GPU needs a GPU `device` in the model and the matching Cargo feature, for example `provider-local-vulkan`.
- An `oracle` test whose keys are absent prints `SKIP:` and passes.

## Example flags

Every example takes flags. `--base-url` defaults to `http://127.0.0.1:1234/v1` and `--model` to `google/gemma-4-e4b`. The value is used as given, so it includes every path segment before the route, normally `/v1`.

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

## Build flags

`RUSTFLAGS` is a cargo build setting, not a cuca setting. Local-model builds use `-C target-cpu=native`:

```bash,name=Linux and macOS
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features server --bin cuca
```

```powershell,name=Windows (PowerShell)
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release --no-default-features --features server --bin cuca
```
