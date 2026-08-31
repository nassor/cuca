# CUCA

**Compact Universal Client for Agents**

> Legendary vigilant witch; small footprint that constantly watches for responses.

[![Website](https://img.shields.io/badge/docs-nassor.github.io%2Fcuca-2f81f7)](https://nassor.github.io/cuca/)
[![Crates.io](https://img.shields.io/crates/v/cuca.svg)](https://crates.io/crates/cuca)
[![CI](https://github.com/nassor/cuca/actions/workflows/ci.yml/badge.svg)](https://github.com/nassor/cuca/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.98%2B-orange)](https://www.rust-lang.org)

CUCA is an asynchronous Rust library for LLM backends. It parses multi-modal SSE streams and normalizes every chunk into typed blocks. One `UnifiedRequest` and one `AgentResponseStream` work across every provider.

## Design

- **Explicit provider selection.** `default = []`; compilation fails until you enable at least one `provider-*` feature. Pick `provider-openai` (also for OpenAI-compatible endpoints such as Ollama via a base-URL override), `provider-anthropic`, `provider-deepseek`, `provider-gemini`, `provider-llamacpp`, `provider-vllm`, or `provider-lmstudio`.
- **Lean dependency boundary.** The core carries wire types and the SSE parser. HTTP, Tokio stream adapters, provider SDK support, and plugin dependencies compile only when their owning provider or plugin feature is enabled.
- **Optional per-request thinking.** One effort level, `minimal` to `xhigh`, maps onto each provider's native controls: OpenAI-compatible `reasoning_effort`, Anthropic budget and adaptive modes, Gemini budgets and levels, DeepSeek thinking mode. Providers without a knob ignore it.
- **Everything is a plugin.** MCP connectors, WASM sandboxing, memory compression and in-memory graph memory, output guardrails, subagent delegation, human approval, web search, skills, telemetry, session logging, local response caching, token and cost accounting with budget caps, speculative fast/slow routing, and schema-guided entity extraction are compile-time feature flags.

## Quick start

Prerequisites: Rust 1.98 or newer, a checkout of this repository, and a running llama.cpp server (`llama-server`) on port 1234 with the demo model loaded. The demo streams one reply from Gemma 4 E4B.

Every `cargo` command below runs the same on Linux, macOS and Windows (PowerShell).

1. Make sure llama.cpp's server is up with a GGUF build of Gemma 4 E4B loaded. The `google/gemma-4-e4b` repository on Hugging Face publishes safetensors, not GGUF, so `llama-server -hf google/gemma-4-e4b` cannot load it directly; point `-m` at a GGUF file for the model, substituting its path for `/path/to/model.gguf` below.

   ```
   llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
   ```

   `llama-server` keeps a local server on port 1234 in the foreground; leave it running in its own terminal. Confirm it is serving the model.

   Linux/macOS:

   ```
   curl http://127.0.0.1:1234/v1/models
   ```

   Windows (PowerShell):

   ```
   curl.exe http://127.0.0.1:1234/v1/models
   ```

   You see a JSON object whose `data` array contains an entry with `"id": "google/gemma-4-e4b"`.

2. Create the demo crate and enter it.

   ```
   cargo new cuca-demo
   cd cuca-demo
   ```

   You see ``Creating binary (application) `cuca-demo` package``.

3. Add CUCA from your checkout, with the llama.cpp provider enabled. Substitute the path of this repository for `/path/to/cuca`.

   ```
   cargo add cuca-core --path /path/to/cuca --features provider-llamacpp
   ```

   You see `Adding cuca-core (local) to dependencies`.

   llama-server's chat route speaks the OpenAI-compatible `/v1/chat/completions` protocol, but the llama.cpp adapter's own default base URL is `http://127.0.0.1:8080`; the demo below overrides it to reach the server on port 1234, and no API key is needed.

4. Add the async runtime and stream extension trait used by the demo.

   ```
   cargo add tokio --features macros,rt-multi-thread
   cargo add futures-util
   ```

   Each command prints `Adding <name> v<version> to dependencies`.

5. Replace `src/main.rs` with the demo below.

   ```rust
   use cuca::types::{MessageContentBlock, ProviderEndpoint};
   use cuca::{CucaClient, UnifiedRequest};
   use futures_util::StreamExt;

   #[tokio::main]
   async fn main() -> Result<(), Box<dyn std::error::Error>> {
       let client = CucaClient::builder()
           .with_provider(ProviderEndpoint::LlamaCpp)
           .with_base_url("http://127.0.0.1:1234/v1")
           .build()?;

       let request = UnifiedRequest::new("google/gemma-4-e4b")
           .add_system_message("You are concise.")
           .add_user_message("Explain CUCA in one sentence.");

       let mut stream = client.generate_stream(request).await?;
       while let Some(chunk) = stream.next().await {
           if let Ok(MessageContentBlock::Text(text)) = chunk {
               print!("{text}");
           }
       }
       Ok(())
   }
   ```

   The program builds a client pointed at the local llama.cpp server, streams the model's reply, and prints each text chunk as it arrives.

6. Run the demo.

   ```
   cargo run
   ```

   You see Gemma 4 E4B's reply stream to your terminal, one chunk at a time; the process exits once the stream ends.

## After the first run

- Enable only the provider features you use.
- The same demo works against any OpenAI-compatible endpoint, such as Ollama, vLLM, or a remote gateway, by switching to `provider-openai` and setting the base URL in `with_base_url` and the model name in `UnifiedRequest::new`; Ollama serves this protocol at `http://localhost:11434/v1` once `ollama serve` is running with a model pulled.
- LM Studio speaks the same protocol through its own dedicated adapter, `provider-lmstudio`, which already defaults to LM Studio's local server at `http://127.0.0.1:1234/v1`.
- The repository's `examples/` folder demonstrates the same pattern in four flavors: `cargo run --example llamacpp_gemma --features provider-llamacpp` (plain reply), `stream_all_blocks` (every block type), `custom_plugin` (a counting plugin), and `cost_otel` (a priced turn exported to OpenTelemetry, which also needs `plugin-cost,plugin-telemetry`). The examples read `CUCA_BASE_URL` and `CUCA_MODEL` from the environment (defaults target a local llama.cpp server), e.g. `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example llamacpp_gemma --features provider-llamacpp` to point at a vLLM server instead.

### Integration tests (live llama.cpp)

The suite in `tests/` runs server-dependent plugins against a live model server: one file per plugin, sharing the harness in `tests/common/mod.rs`. Each file compiles only when its plugin feature is on, on top of `provider-llamacpp`.

Prerequisites: an OpenAI-compatible server (llama.cpp's `llama-server` serves this natively) with a chat model loaded at `http://127.0.0.1:1234`, no API key. The MCP echo server is embedded in `plugin_mcp.rs`: the test binary re-executes itself with `--mcp-echo-server` and serves an rmcp stdio echo server (no Python needed). On WSL2, `127.0.0.1` may not reach a server on the Windows host; set `CUCA_BASE_URL` to the host address, for example `http://172.25.0.1:1234/v1`.

Runs the same on Linux, macOS and Windows (PowerShell):

```
cargo test --features "provider-llamacpp plugin-mcp plugin-sandbox plugin-memory plugin-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry plugin-speculative plugin-session-log plugin-prompt-cache plugin-cost" -- --nocapture --test-threads=1
```

Environment variables:

- `CUCA_BASE_URL`: the OpenAI-compatible base URL; defaults to `http://127.0.0.1:1234/v1`.
- `CUCA_MODEL`: the model id to exercise; defaults to the first id the server reports.
- `CUCA_REQUIRE_LIVE`: set to `1` to fail when the server is unreachable; otherwise server-dependent tests print `SKIP: llama.cpp not reachable: ...` and pass.
- `plugin-speculative`'s live orchestrated-turn test does not read `CUCA_BASE_URL`: its tier executors draw clients from a pool with no base URL configured, so `dispatch_llamacpp` falls back to llama.cpp's own default, `http://127.0.0.1:8080`. That test needs a server reachable there, independent of wherever `CUCA_BASE_URL` points the rest of the suite.

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Copyright 2026 Nassor Frazier-Silva <nassor@gmail.com>
