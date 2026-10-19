# CUCA

**Compact Universal Client for Agents**

> Legendary vigilant witch; small footprint that constantly watches for responses.

[![Website](https://img.shields.io/badge/docs-nassor.github.io%2Fcuca-2f81f7)](https://nassor.github.io/cuca/)
[![Crates.io](https://img.shields.io/crates/v/cuca.svg)](https://crates.io/crates/cuca)
[![CI](https://github.com/nassor/cuca/actions/workflows/ci.yml/badge.svg)](https://github.com/nassor/cuca/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.98%2B-orange)](https://www.rust-lang.org)

CUCA is an asynchronous Rust library for talking to LLM backends. One `UnifiedRequest` goes in, and one `AgentResponseStream` of typed blocks (`Text`, `Thinking`, `ToolCall` and others) comes out, whichever provider answers. The streaming parser reuses one buffer and copies a frame only when it emits an event.

## What you get

Nothing is compiled by default. You pick Cargo features, and the build fails until you pick at least one provider. The full list is on the [features page](https://nassor.github.io/cuca/reference/features/).

- [Providers](https://nassor.github.io/cuca/providers/): `provider-openai` (also Ollama and other OpenAI-compatible servers), `provider-anthropic`, `provider-deepseek`, `provider-gemini`, `provider-llamacpp`, `provider-vllm` and `provider-lmstudio` over HTTP, and `provider-local`, which runs a GGUF model inside your process.
- [Plugins](https://nassor.github.io/cuca/plugins/) watch the request and the stream: MCP, WASM sandbox, memory, guardrails, subagents, human approval, web search, skills, telemetry, session log, cost accounting and redaction.
- [Services](https://nassor.github.io/cuca/services/) are called directly: prompt cache, entity extraction, session replay, speculative fast/slow routing, rate limiting and a vector store.
- Thinking: one effort level, `minimal` to `xhigh`, maps onto each provider's native reasoning control. Providers without one ignore it.

Start with the [Quick start](#quick-start) below, or read the [documentation site](https://nassor.github.io/cuca/).

## Quick start

Prerequisites: Rust 1.98 or newer, a checkout of this repository, and `llama-server` from llama.cpp with a Gemma 4 E4B GGUF file. About 15 minutes end to end.

You will start the model on port 1234, create a small crate, and stream one reply into your terminal. Every `cargo` command below runs the same on Linux, macOS and Windows (PowerShell).

1. Start the model server in its own terminal and leave it running. Replace `/path/to/model.gguf` with your GGUF file. The `google/gemma-4-e4b` repository on Hugging Face holds safetensors, so `llama-server -hf` cannot load it.

   ```
   llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
   ```

   From a second terminal, check that it serves the model.

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

3. Add CUCA from your checkout with the llama.cpp provider enabled. Replace `/path/to/cuca` with the path of this repository.

   ```
   cargo add cuca --path /path/to/cuca --features provider-llamacpp
   ```

   You see `Adding cuca (local) to dependencies`, followed by the feature list, with `+ provider-llamacpp` enabled and every other feature listed as disabled.

   The llama.cpp adapter defaults to `http://127.0.0.1:8080`, so the demo below sets the base URL to port 1234. No API key is needed.

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

   You see Gemma 4 E4B's reply stream to your terminal, one chunk at a time; the process exits once the stream ends. A reasoning model pauses before the first word: the demo prints `Text` blocks and drops `Thinking` ones, so nothing appears until the reasoning is done.

## After the first run

- Enable only the provider features you use. The [Providers](https://nassor.github.io/cuca/providers/) pages list what each one talks to.
- Any OpenAI-compatible server (Ollama, vLLM, a remote gateway) works with `provider-openai`, a different `with_base_url` and your model name. Ollama serves at `http://localhost:11434/v1`. LM Studio has its own `provider-lmstudio`, which defaults to `http://127.0.0.1:1234/v1`.
- The `examples/` folder has a runnable demo for the core client, every plugin and every service. All except `local_model` read `CUCA_BASE_URL` and `CUCA_MODEL`. Each plugin and service page on the site shows the `cargo run` command and its output.

### Run a model with no server

`provider-local` runs a GGUF model inside your process, on the CPU. It downloads the default model, Gemma 4 E2B Q4_0 (2.84 GB), on first use. Inside an async function:

```rust
use cuca::types::ProviderEndpoint;
use cuca::{CucaClient, LocalModel, LocalModelConfig};

let model = LocalModel::load(LocalModelConfig::default()).await?;
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::Local)
    .with_local_model(model)
    .build()?;
```

`generate_stream` then works as in the Quick start. The [Local page](https://nassor.github.io/cuca/providers/local/) has the full example, the presets, GPU features and the build command. [Local models and formats](https://nassor.github.io/cuca/providers/local/models/) lists the supported architectures.

### Run the live tests

With a chat model on `http://127.0.0.1:1234` (a llama.cpp server works), run this on Linux, macOS and Windows (PowerShell):

```
cargo test --features "provider-llamacpp plugin-mcp plugin-sandbox plugin-memory service-entity-extraction plugin-guardrails plugin-subagent plugin-hitl plugin-web-search plugin-skills plugin-telemetry service-speculative plugin-session-log service-replay service-prompt-cache plugin-cost service-rate-limit service-vector-store plugin-redaction" -- --nocapture --test-threads=1
```

Without a reachable server, the tests print `SKIP` and pass. The environment variables are in the [integration tests guide](https://nassor.github.io/cuca/guides/integration-tests/).

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Copyright 2026 Nassor Frazier-Silva <nassor@gmail.com>
