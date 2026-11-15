+++
title = "Run the live integration suites"
description = "Run the tests that use a real model: the llama.cpp suite and the local model suites. Write cuca-test.kdl, run one command, read the SKIP, PASS and KNOWN-LIMIT lines."
template = "page.html"
weight = 3
+++

# Run the live integration suites

<dl class="page-facts">
<dt>In one line</dt>
<dd>Write <code>cuca-test.kdl</code> at the repository root, run one <code>cargo test</code> command per suite, and read the lines it prints</dd>
<dt>You need</dt>
<dd>A checkout of this repository and Rust 1.98 or newer. The llama.cpp suite also needs <code>llama-server</code>. The local suites need a model, which they download on the first run</dd>
<dt>Read this if</dt>
<dd>You change the crate and need the tests to exercise a real model instead of skipping</dd>
</dl>

Goal: run a live suite and know that every test in it ran. A *live* test talks to a real model. Without a configured model it prints `SKIP` and still passes, so a green run alone does not prove the model was used.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="itest-title itest-desc">
  <title id="itest-title">Which test suite reads which part of cuca-test.kdl</title>
  <desc id="itest-desc">The file cuca-test.kdl at the repository root feeds two kinds of suite. The llama.cpp suite, one test file per plugin or service, reads the llamacpp key and talks to a llama-server on port 1234. The local suites, provider_local and its batch, tiers, mtp and convert siblings, read the local key and run a model inside the test process.</desc>
  <defs>
    <marker id="itest-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One config file, two kinds of suite</text>
  <rect class="blk blk-ctl" x="6" y="64" width="130" height="60" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="71" y="88">cuca-test.kdl</text>
  <text class="t-sm t-mid" x="71" y="102">repository root</text>
  <rect class="blk blk-data" x="196" y="36" width="210" height="52" rx="6"/>
  <text class="t-sm t-mid t-data" x="301" y="58">llama.cpp suite</text>
  <text class="t-sm t-mid" x="301" y="72">plugin_*, service_* files</text>
  <text class="t-sm t-mid" x="301" y="84">reads the llamacpp key</text>
  <rect class="blk blk-bnd" x="466" y="36" width="188" height="52" rx="6"/>
  <text class="t-sm t-mid" x="560" y="58">llama-server</text>
  <text class="t-sm t-mid" x="560" y="72">http://127.0.0.1:1234/v1</text>
  <rect class="blk blk-data" x="196" y="104" width="210" height="52" rx="6"/>
  <text class="t-sm t-mid t-data" x="301" y="126">local model suites</text>
  <text class="t-sm t-mid" x="301" y="140">provider_local, _batch, ...</text>
  <text class="t-sm t-mid" x="301" y="152">reads the local key</text>
  <rect class="blk" x="466" y="104" width="188" height="52" rx="6"/>
  <text class="t-sm t-mid" x="560" y="126">model in the test process</text>
  <text class="t-sm t-mid" x="560" y="140">no server needed</text>
  <path class="arw arw-ctl" d="M138 84 L192 66" marker-end="url(#itest-arrow)"/>
  <path class="arw arw-ctl" d="M138 104 L192 128" marker-end="url(#itest-arrow)"/>
  <path class="arw arw-bnd" d="M408 62 H462" marker-end="url(#itest-arrow)"/>
  <path class="arw arw-data" d="M408 130 H462" marker-end="url(#itest-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>The llama.cpp suite needs a server. The local suites need none.</b> Both read their settings from the same file.</p>
</div>

## Read the output first

Add `--nocapture` to every command. The test harness hides what a passing test prints, and the lines below are what tell you what happened.

| Line | Meaning |
|---|---|
| `SKIP: ...` | the test had no server, model or reference files to use, so it did nothing and passed. The text names the missing setting. |
| `MATRIX\|<scenario>\|PASS` | the scenario ran against the model and its checks held |
| `MATRIX\|<scenario>\|SKIP\|<reason>` | the scenario does not apply to this model, for example thinking tests on a model that never thinks |
| `MATRIX\|<scenario>\|KNOWN-LIMIT\|<evidence>` | the scenario failed in a way that is a known limit of this model, not a fault of the engine. The evidence says what happened. |
| `MATRIX\|<scenario>\|DECODING\|...` | the scenario used the model's own sampling settings, because greedy decoding makes that model loop |
| `BATCH\|...`, `TIERS\|...` | measurements: steps, rows, tokens per second, cache hits |
| `test result: ok.` | cargo's summary. It says `ok` when tests skipped. |

A *known limit* is a model weakness that the Hugging Face reference implementation shows too, or a near-tie in quantized arithmetic. Everything else that fails is a real failure.

To make a skip fail, put `require_live #true` in `cuca-test.kdl`. The run then stops with a panic that names what is missing. Use it in CI.

## The test file

All live suites read one file, `cuca-test.kdl`, at the repository root. It is not tracked by git. Copy the template, which lists every key:

```bash,name=Linux and macOS
cp cuca-test.example.kdl cuca-test.kdl
```

```powershell,name=Windows (PowerShell)
Copy-Item cuca-test.example.kdl cuca-test.kdl
```

A key you mistype stops the run with a panic that names it. The keys the suites below use:

| Key | Effect | Default |
|---|---|---|
| `require_live` | `#true` turns every skip into a failure | `#false` |
| `cache_dir` | where tests download and convert models | the platform cache directory |
| `llamacpp base_url="..."` | the llama.cpp server, with `/v1` | `http://127.0.0.1:1234/v1` |
| `llamacpp model="..."` | the model id under test | the first id the server reports |
| `local { model { ... } }` | the in-process model. It holds the keys of a `LocalModelConfig` (see [Configuration](@/providers/local/configuration.md)): `source`, `device`, `gpu_layers`, `experts`, `drafter` and so on | none |
| `local { download_default #true }` | with no `source`, download and use the default preset | `#false` |
| `local { experts_ref ... }` | the placement the tiered run is compared with | `"resident"` |
| `local { draft_tokens 1 2 3 4 }` | one drafter run per value | `3` |
| `local { allow_mismatch #true }` | report a drafter output that differs from plain decoding instead of failing | `#false` |
| `local { print_replies #true }` | print each greedy reply as a `REPLY|...` line | `#false` |
| `local { perf #true }`, `perf_cold` | run the long-prompt speed scenario, optionally with a cold prefix | `#false` |
| `oracle { ref_dir "..." }`, `convert_only "..."` | the reference dumps for the conversion suite, and a substring that limits it to some repositories | none |

Flags are KDL booleans, `#true` or `#false`. The oracle keys beyond these belong to contributors and are explained in `AGENTS.md`.

## Suite 1: llama.cpp (plugins and services)

Every plugin and service has its own test file. They send real requests to `llama-server`.

1. Start `llama-server` on port 1234. It listens on 8080 by default, but the tests expect 1234.

```bash,name=Linux and macOS
llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
```

```powershell,name=Windows (PowerShell)
llama-server -m C:\path\to\model.gguf --port 1234 --alias google/gemma-4-e4b
```

2. From a second terminal, check that it answers.

```bash,name=Linux and macOS
curl http://127.0.0.1:1234/v1/models
```

```powershell,name=Windows (PowerShell)
curl.exe http://127.0.0.1:1234/v1/models
```

Result: a JSON list with at least one model `id`. To use another server, write `llamacpp base_url="..."` in `cuca-test.kdl`.

3. Run the whole suite. Name the provider and every plugin and service feature:

```bash,name=Runs the same on all three platforms
cargo test --features "provider-llamacpp config-kdl plugin-mcp plugin-sandbox plugin-memory service-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry service-speculative plugin-session-log service-replay service-prompt-cache plugin-cost service-rate-limit service-vector-store plugin-redaction service-scheduler" -- --nocapture --test-threads=1
```

`--test-threads=1` runs the tests one at a time against the single `llama-server`. `config-kdl` lets the tests read `cuca-test.kdl`.

```text,name=Expected output with the server running
test result: ok. ...
```

With the server stopped, the same run prints this line for each test that needs it, and still passes:

```text,name=The skip line
SKIP: llama.cpp not reachable: <reason>
```

The reason comes from a probe of `GET {base_url}/models` with a two second timeout. It tells a refused connection from a server that answered with no model ids.

4. To make that skip a failure, add this to `cuca-test.kdl` and run again:

```kdl,name=cuca-test.kdl
require_live #true
llamacpp base_url="http://127.0.0.1:1234/v1"
```

```text,name=The panic when require_live is set and nothing answers
require_live is set but llama.cpp is unreachable: <reason>
```

### Run one file

A file for one plugin or service needs `provider-llamacpp`, `config-kdl` and its own feature. A shorter feature list does not fail. It compiles fewer tests and reports success.

```bash,name=Runs the same on all three platforms
cargo test --test plugin_guardrails --features provider-llamacpp,config-kdl,plugin-guardrails -- --nocapture
```

The file `plugin_combinations.rs` tests plugins together and needs the union of their features:

```bash,name=Runs the same on all three platforms
cargo test --test plugin_combinations --features provider-llamacpp,config-kdl,service-entity-extraction,plugin-memory,service-prompt-cache,service-speculative,plugin-session-log,plugin-cost,plugin-telemetry,service-rate-limit,plugin-redaction,service-replay,service-vector-store -- --nocapture
```

Two files never call the server: `service_prompt_cache.rs` runs against mock servers inside the test, and `public_exports.rs` only builds types.

If `cuca-test.kdl` exists but `config-kdl` is not in `--features`, the llama.cpp tests panic and tell you so.

## Suite 2: local models

The local suites run a real model inside the test process. They need no server. Build for your CPU first, because the CPU kernels pick their instructions at compile time. Set this once per terminal:

```bash,name=Linux and macOS
export RUSTFLAGS="-C target-cpu=native"
```

```powershell,name=Windows (PowerShell)
$env:RUSTFLAGS = "-C target-cpu=native"
```

Every command below is then the same on all three platforms. The first run downloads the model into the cache folder, which takes a while. Later runs reuse it.

### Chat behaviour: provider_local

It loads one model and runs a set of scenarios: a factual answer with exact token counts, thinking on and off, a tool call and its follow-up, a long prompt with a needle to find, prefix cache reuse, and a stop string.

```kdl,name=cuca-test.kdl
require_live #true
local {
    model { source "qwen3-0.6b-q8_0" }
}
```

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local --test provider_local -- --test-threads=1 --nocapture
```

```text,name=Expected output, shortened
MATRIX|model|...
MATRIX|greedy_text_and_usage|PASS
MATRIX|thinking_precedes_answer|PASS
...
test result: ok. ...
```

Which scenarios apply depends on what the model's own chat template supports, so a model without tools prints `MATRIX|tool_call_round_trip|SKIP|...`. Swap the `source` line for any other preset, a `file="..."` GGUF or a checkpoint to test another model.

### Batching: provider_local_batch

It sends eight clients at once to one model and checks that every reply equals the one-at-a-time reply. On the CPU the replies must be identical. On a GPU they agree up to float rounding. It also prints the speed gain of batching. It uses the same `cuca-test.kdl` as `provider_local`, and `batching max_sequences=4` in the `model` node sets the slots.

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local --test provider_local_batch -- --test-threads=1 --nocapture
```

```text,name=Expected output, shortened
BATCH|shared_prefix_clients|steps ... rows ...
BATCH|shared_prefix_equal|8/8 equal to serial
BATCH|prefix_hits|...
BATCH|speedup|...x
test result: ok. ...
```

### Tiered experts: provider_local_tiers

A mixture of experts model has many small expert sub-networks and uses only a few per token. A tiered run keeps the hot experts in RAM and reads the rest from disk. This suite runs three prompts with a normal load, then tiered cold, then tiered warm, and requires the same replies each time. The model must be a mixture of experts and `experts` must be tiered. Otherwise the test panics.

```kdl,name=cuca-test.kdl
require_live #true
local {
    model {
        source "olmoe-1b-7b-q4_0"
        experts { tiered ram_bytes="500m"; }
    }
}
```

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local --test provider_local_tiers -- --nocapture
```

```text,name=Expected output, shortened
TIERS|reference|... tok/s
TIERS|cold|... tok/s|...|hit rate ...|read at load ...
TIERS|warm|... tok/s|...|hit rate ...|read at load ...
test result: ok. ...
```

The reference placement is `resident` unless you set `local { experts_ref ... }`. [Tiered experts](@/providers/local/expert-tiers.md) explains the placement.

### Drafters: provider_local_mtp

Speculative decoding lets a small *drafter* guess several tokens that the main model then checks in one pass. This suite runs plain decoding once, then decoding with the drafter for each `draft_tokens` value, and compares the output. On the CPU the output must be identical.

```kdl,name=cuca-test.kdl
require_live #true
local {
    download_default #true
    model { drafter "gemma4-e2b-mtp-f16" }
    draft_tokens 1 2 3
}
```

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local --test provider_local_mtp -- --nocapture
```

```text,name=Expected output, shortened
k 3 "Write a Python function that ...": plain ... tok/s, speculative ... tok/s (...x), identical true, drafted ... accepted ...
...
test result: ok. ...
```

`identical false` fails the test on the CPU. `local { allow_mismatch #true }` reports it instead. [Speculative decoding](@/providers/local/drafting.md) lists the drafter types.

### Checkpoint conversion: provider_local_convert

This suite converts Hugging Face checkpoints to GGUF and compares the first greedy tokens with reference dumps made by the Hugging Face `transformers` library. The dumps are not in the repository. Without `oracle { ref_dir ... }` the suite prints `SKIP:` and passes, so only a contributor who has the dumps can run it. `AGENTS.md` (Testing & QA) lists the oracle keys.

```kdl,name=cuca-test.kdl
require_live #true
oracle { ref_dir ".omp/local-models/ref" }
```

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local --test provider_local_convert -- --test-threads=1 --nocapture
```

```text,name=Expected output per checkpoint, shortened
CONVERT <repo>: reply "..." oracle "..." (shared prefix ... chars)
test result: ok. ...
```

### Run on a GPU

Add the backend feature and name the device in `cuca-test.kdl`:

```kdl,name=cuca-test.kdl
local {
    model { device "vulkan" }
    download_default #true
}
```

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local-vulkan --test provider_local
```

Result: the output includes `MATRIX|devices|...` with the device that was used. The first request on a GPU takes longer while the GPU kernels compile. [Run on a GPU](@/providers/local/gpu.md) covers the backends.

## Check that one feature builds alone

CI does this for every plugin and service feature. It needs no server and no model:

```bash,name=Runs the same on all three platforms
cargo check --all-targets --no-default-features --features provider-openai,plugin-hitl
```

Swap `plugin-hitl` for the feature you changed. Result: `cargo check` finishes with no error.

The full list of CI features is every feature except `provider-local-cuda` and `provider-local-rocm`, which download a prebuilt LLVM on the first build. See [the feature matrix](@/reference/features.md).

Oracle checks against reference dumps and the real-weights DeepSeek runs are gated in `cuca-test.kdl` too, and `AGENTS.md` (Testing & QA) lists their keys.

Next page: [The feature matrix](@/reference/features.md).
