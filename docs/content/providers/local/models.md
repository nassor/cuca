+++
title = "Choose a model"
description = "Pick a local model preset by machine size, see the supported architectures and weight formats, and load any GGUF file or safetensors checkpoint that is not a preset."
template = "page.html"
weight = 2
+++

# Choose a model

<dl class="page-facts">
<dt>In one line</dt>
<dd>Put a preset name in <code>source</code> and the model downloads on first use. Any other GGUF file or safetensors checkpoint of a supported architecture loads too.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, disk for the download, and RAM (or VRAM on a GPU) for the model. The Memory column below says how much.</dd>
<dt>Read this if</dt>
<dd>You are choosing a model, or loading a model that is not a preset.</dd>
</dl>

A preset is a model with a short name such as `qwen3-0.6b-q8_0`. CUCA downloads its file once, checks its SHA-256 and runs it. A preset file is a GGUF file: one file that holds a model's weights, its tokenizer and its chat template. Your own model can also be a Hugging Face safetensors checkpoint, which CUCA converts once to a cached GGUF.

The smallest working setup is one line of KDL:

```kdl,name=model.kdl
source "qwen3-0.6b-q8_0"
```

```bash,name=Runs the same on all three platforms (from a checkout of this repository)
cargo run --release --example local_model --features provider-local -- --config model.kdl --prompt "Say hi"
```

The first run downloads 0.64 GB and prints the progress on stderr. Later runs skip the download.

```text,name=Expected output (the reply varies)
load: <seconds> s
loaded on cpu
<the reply, streamed>
timing: <n> prompt tokens, first block after <s> s (~<r> prompt tokens/s); <n> completion tokens (<n> blocks), then <t> tokens/s after the first block
```

`load:` is the load time. A model that reasons prints its thinking dimmed after `[thinking]`, then the answer. How to load and configure a model in your own code is on [Local (in-process)](@/providers/local/_index.md).

## Pick a preset

Start from the memory you can spare. A model needs about its file size plus a K/V cache, the memory it keeps for the tokens it has already read.

- A small laptop: `gemma3-1b-q4_0` (1.4 GB, plain chat) or `qwen3-0.6b-q8_0` (reasons, calls tools). Qwen 3 0.6B reserves a large K/V cache for its size, 3.76 GB at the default context. `context_length 2048` cuts it to about a quarter, so the model then needs about 1.6 GB.
- A normal laptop: the default, `gemma4-e2b-q4_0` (3.3 GB). It reasons and calls tools. `gemma4-e2b-q8_0` is more accurate and needs 5.5 GB.
- 8 GB or more: `gemma4-e4b-q4_0` (6.1 GB) or `qwen3-1.7b-q8_0` (5.6 GB).
- 20 GB or more: `qwen3.6-35b-a3b-q4_0`, `qwen3-30b-a3b-q4_0`, `gemma4-26b-a4b-q4_0` or `muse-glimmer-30b-q4_0`.
- A mixture of experts bigger than your RAM: keep its experts on disk with `experts { tiered ram_bytes="4g"; }`. A mixture of experts is a model with many small expert networks that uses only a few for each token. With this setting `qwen3.6-35b-a3b-q4_0` peaked at 5.95 GiB of RAM instead of 18.8 GiB and decoded at about half the speed (CPU, Ryzen 9 9950X3D2, release build). See [Tiered experts](@/providers/local/expert-tiers.md).
- Reasoning and tool calls: the Gemma 4, Qwen 3, Qwen 3.5, Qwen 3.6, SmolLM 3, Granite 3.3, Muse Glimmer, Ministral 3 Reasoning and Mistral Small 4 presets do both. Ministral 3 Instruct calls tools and does not reason. The DeepSeek-R1 distills reason but cannot call tools. Phi 4 mini, OLMo 2, OLMoE and Gemma 3 do neither.
- A model larger than your GPU: see [Fit a large model](@/providers/local/placement.md).

## The presets

`ModelSource::PRESETS` has 24 models and 2 drafters. Each is pinned to one repository commit and one SHA-256, so every download is the same file. A name is ASCII case-insensitive.

- Download is the file size.
- Memory is the file plus the K/V cache at the default `context_length` of 8,192 positions with 4 parallel requests. Lower `context_length` to shrink the cache. A mixture of experts with tiered experts needs less.
- Thinking is `yes` when a request can switch reasoning on or off, `always` when the model reasons on every request, and `no` when it has no reasoning mode.
- Tools is `yes` when the model can call tools you declare. Details on calls and reasoning are on [Requests, thinking and tool calls](@/providers/local/requests.md).

| Preset | Family | Download | Memory | Thinking | Tools |
|---|---|---|---|---|---|
| `gemma4-e2b-q4_0` (default) | Gemma 4 E2B | 2.84 GB | 3.3 GB | yes | yes |
| `gemma4-e2b-q8_0` | Gemma 4 E2B | 4.97 GB | 5.5 GB | yes | yes |
| `gemma4-e4b-q4_0` | Gemma 4 E4B | 4.59 GB | 6.1 GB | yes | yes |
| `gemma4-e4b-q8_0` | Gemma 4 E4B | 8.03 GB | 9.5 GB | yes | yes |
| `gemma4-26b-a4b-q4_0` | Gemma 4 26B-A4B | 14.62 GB | 20.8 GB | yes | yes |
| `gemma3-270m-q8_0` | Gemma 3 270M | 0.29 GB | 0.7 GB | no | no |
| `gemma3-1b-q4_0` | Gemma 3 1B | 0.72 GB | 1.4 GB | no | no |
| `qwen3-0.6b-q8_0` | Qwen 3 0.6B | 0.64 GB | 4.4 GB | yes | yes |
| `qwen3-1.7b-q8_0` | Qwen 3 1.7B | 1.83 GB | 5.6 GB | yes | yes |
| `qwen3-30b-a3b-q4_0` | Qwen 3 30B-A3B | 17.38 GB | 20.6 GB | yes | yes |
| `qwen3.5-0.8b-q8_0` | Qwen 3.5 0.8B | 0.83 GB | 1.3 GB | yes | yes |
| `qwen3.6-35b-a3b-q4_0` | Qwen 3.6 35B-A3B | 20.84 GB | 21.6 GB | yes | yes |
| `r1-distill-qwen-1.5b-q4_0` | DeepSeek-R1 distill, Qwen 1.5B | 1.07 GB | 2.0 GB | always | no |
| `r1-distill-llama-8b-q4_0` | DeepSeek-R1 distill, Llama 8B | 4.68 GB | 9.0 GB | always | no |
| `smollm3-3b-q4_0` | SmolLM 3 3B | 1.81 GB | 4.2 GB | yes | yes |
| `phi4-mini-q4_0` | Phi 4 mini | 2.33 GB | 6.6 GB | no | no |
| `phi4-mini-q4_k_m` | Phi 4 mini | 2.49 GB | 6.8 GB | no | no |
| `granite3.3-2b-q4_0` | Granite 3.3 2B | 1.45 GB | 4.1 GB | yes | yes |
| `olmo2-1b-q4_0` | OLMo 2 1B | 0.89 GB | 3.0 GB | no | no |
| `olmoe-1b-7b-q4_0` | OLMoE 1B-7B | 3.93 GB | 6.1 GB | no | no |
| `muse-glimmer-30b-q4_0` | Muse Glimmer 30B | 16.27 GB | 18.0 GB | always | yes |
| `ministral3-3b-instruct-q4_0` | Ministral 3 3B Instruct | 2.05 GB | 5.5 GB | no | yes |
| `ministral3-3b-reasoning-q4_0` | Ministral 3 3B Reasoning | 2.05 GB | 5.5 GB | always | yes |
| `mistral-small-4-119b-q4_0` | Mistral Small 4 119B | 67.71 GB | 68.5 GB | yes | yes |

The number after `q` is the bits per weight. This is quantization: Q4_0 keeps each weight in about 4 bits and Q8_0 in about 8, so a Q4_0 file is roughly half the size of its Q8_0 sibling and a little less accurate.

Notes that change what you do:

- `gemma4-26b-a4b-q4_0` runs on the CPU only. The GPU engine rejects it. Its experts can stay in RAM (resident) or be tiered.
- `mistral-small-4-119b-q4_0` is two files, 39.92 GB and 27.79 GB. They download together and open as one model, see [Model files and downloads](@/providers/local/model-sources.md). Resident it needs 67.7 GB of RAM plus 0.76 GB of K/V cache. Tiered experts need less.
- `qwen3.5-0.8b-q8_0` has a draft head inside its file, so `drafter "in-trunk"` works with it. See [Speculative decoding](@/providers/local/drafting.md).
- `olmo2-1b-q4_0` and `olmoe-1b-7b-q4_0` were trained on 4,096 positions. They load at 4,096 by default, and a larger `context_length` fails with `context_length N exceeds the model maximum M`.
- Reasoning models spend many tokens thinking, so give a request a generous `max_tokens`. This matters most for `r1-distill-*`, `ministral3-3b-reasoning-q4_0`, `muse-glimmer-30b-q4_0` and `smollm3-3b-q4_0`.
- A model that always reasons can use its whole token budget thinking at temperature 0 and never write the tool call. `ministral3-3b-reasoning-q4_0` does. When you need tool calls from such a model, leave `temperature` at the model's own default instead of setting 0.
- `gemma4-e4b-q8_0` answers an easy question at once without a reasoning block, even with thinking on. A harder question makes it reason. `gemma3-270m-q8_0` is tiny and can end its answer at once on prompts of about 750 tokens or more.

What was run on which model is on [Support status and known limits](@/providers/local/verification.md).

## Drafter presets

A drafter is a small model that guesses the next few tokens so the main model can check them in one pass. This is speculative decoding. A drafter is used through `drafter`, never as `source`.

- `gemma4-e2b-mtp-f16` (0.17 GB) drafts for the Gemma 4 E2B presets.
- `muse-glimmer-30b-dflash-q4_0` (1.45 GB) drafts for `muse-glimmer-30b-q4_0`, on the CPU only.

## Supported architectures

The architecture is the model's design, stored in the file as `general.architecture`. The loader picks the forward pass from it. Every main-model architecture runs on the CPU and on a GPU, except the Gemma 4 mixture of experts (26B-A4B), which is CPU only.

| Architecture | Models | Notes |
|---|---|---|
| `gemma4` | Gemma 4 (E2B, E4B, 26B-A4B) | 26B-A4B is a mixture of experts, CPU only |
| `gemma3` | Gemma 3 (270M, 1B) | |
| `llama` | Llama-style models, such as DeepSeek-R1-Distill-Llama and Mistral-style models | |
| `qwen2` | DeepSeek-R1-Distill-Qwen and other Qwen 2.5 models | |
| `qwen3` | Qwen 3 | |
| `qwen3moe` | Qwen 3 MoE (Qwen3-30B-A3B) | mixture of experts |
| `qwen35` | Qwen 3.5 (0.8B) | has a draft head in `*-MTP` files |
| `qwen35moe` | Qwen 3.6 MoE (Qwen3.6-35B-A3B) | mixture of experts |
| `phi3` | Phi 4 mini | |
| `smollm3` | SmolLM 3 | |
| `granite` | Granite 3.3 (dense) | |
| `olmo2` | OLMo 2 | |
| `olmoe` | OLMoE | mixture of experts |
| `ernie4_5` | ERNIE 4.5 (dense) | no preset |
| `muse-glimmer` | Muse Glimmer 30B | |
| `mistral3` | Ministral 3 (3B, 8B, 14B), Devstral Small 2 | |
| `mistral4` | Mistral Small 4 (119B) | mixture of experts, not split over several GPUs |
| `deepseek41` | DeepSeek-V4.1 Flash | mixture of experts, no preset, not split over several GPUs; see [Run DeepSeek-V4.1](@/providers/local/deepseek-v41.md) |
| `gemma4-assistant` | the Gemma 4 drafter | through `drafter` only |
| `mistral4-eagle` | the Mistral Small 4 drafter | through `drafter` only, CPU only |

A file with another architecture fails the load with an error that names it and lists the supported ones:

```text,name=Error for an unknown architecture
unsupported GGUF architecture `NAME`; supported: gemma4, gemma3, llama, qwen2, qwen3, phi3, smollm3, granite, qwen3moe, olmoe, qwen35, qwen35moe, olmo2, ernie4_5, muse-glimmer, mistral3, mistral4, gemma4-assistant, mistral4-eagle, deepseek41
```

The `dflash` architecture is a drafter only: it loads through `drafter` as the Muse Glimmer drafter or as the DeepSeek-V4.1 drafter. As a main model it fails with an error that says so.

## Load your own model

A model that is not a preset loads from disk or from Hugging Face, as a GGUF file or as a safetensors checkpoint. A checkpoint (`source "repo:revision"` or `source dir="/ckpt"`) is converted once in process to a cached GGUF, at the precision `convert quant` names; see [Model files and downloads](@/providers/local/model-sources.md). The rest of this section covers a GGUF file:

```kdl,name=A GGUF file you already have
source file="/models/x.gguf"
```

```kdl,name=A GGUF file on Hugging Face (use a commit hash as the revision)
source "owner/name:COMMIT:model-Q4_0.gguf"
```

```rust,name=Load a GGUF file (provider-local)
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        source: ModelSource::File("/models/x.gguf".into()),
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

The loader reads everything about the model from the file, so no config says what the model is. Four checks decide whether it loads.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 214" role="img" aria-labelledby="mdl-title mdl-desc">
  <title id="mdl-title">A model file, read directly or converted from a checkpoint, feeds four parts of the engine and each part checks it</title>
  <desc id="mdl-desc">One box on the left is the model file, a single file or a split set. Four arrows lead to four boxes on the right. The forward pass needs general.architecture to be in the supported list. The tokenizer needs tokenizer.ggml.model and tokenizer.ggml.pre to be implemented. The prompt format needs tokenizer.chat_template, or a template you supply. The weights need every tensor type to be one the engine runs. A failed check stops the load with an error that names the cause.</desc>
  <defs>
    <marker id="mdl-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">What the loader reads from one model file</text>
  <rect class="blk blk-data" x="6" y="36" width="124" height="166" rx="6"/>
  <text class="t-lbl t-mid t-data" x="68" y="114">model file</text>
  <text class="t-sm t-mid" x="68" y="132">one file or</text>
  <text class="t-sm t-mid" x="68" y="146">a split set</text>
  <rect class="blk" x="190" y="36" width="464" height="34" rx="6"/>
  <text class="t-lbl" x="204" y="57">Forward pass</text>
  <text class="t-sm" x="330" y="57">general.architecture is in the supported list</text>
  <rect class="blk" x="190" y="80" width="464" height="34" rx="6"/>
  <text class="t-lbl" x="204" y="101">Tokenizer</text>
  <text class="t-sm" x="330" y="101">tokenizer.ggml.model and .pre are implemented</text>
  <rect class="blk" x="190" y="124" width="464" height="34" rx="6"/>
  <text class="t-lbl" x="204" y="145">Prompt format</text>
  <text class="t-sm" x="330" y="145">tokenizer.chat_template, or a template you supply</text>
  <rect class="blk" x="190" y="168" width="464" height="34" rx="6"/>
  <text class="t-lbl" x="204" y="189">Weights</text>
  <text class="t-sm" x="330" y="189">every tensor type is one the engine runs</text>
  <path class="arw arw-data" d="M132 119 H158 V53 H186" marker-end="url(#mdl-arrow)"/>
  <path class="arw arw-data" d="M132 119 H158 V97 H186" marker-end="url(#mdl-arrow)"/>
  <path class="arw arw-data" d="M132 119 H158 V141 H186" marker-end="url(#mdl-arrow)"/>
  <path class="arw arw-data" d="M132 119 H158 V185 H186" marker-end="url(#mdl-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>A file loads only if all four checks pass.</b> A failed check stops the load with an error that names the cause. The common ones are below.</p>
</div>

Two more things must hold. The file names an end-of-generation token (the token that stops the reply), and its attention keys and values have the same size. A split set (`<base>-00001-of-0000N.gguf`) loads from any of its parts, see [Model files and downloads](@/providers/local/model-sources.md).

A file that loads is not a verified model. Check that its rendered prompt and token ids equal the reference tokenizer's before you rely on it.

### Common load errors

All of them are a `CucaError::Provider` that names the cause.

- ``unsupported GGUF architecture `NAME`; supported: ...``: the architecture is not in the table above. Use a supported model or a preset.
- ``unsupported tokenizer model `NAME` `` or ``unsupported pre-tokenizer `NAME` ``: the tokenizer scheme, or its rule for cutting text into words, is not implemented. Use a model of a supported family.
- ``GGUF metadata `tokenizer.chat_template`: missing``: the file has no chat template. Give one with `chat_template_file "templates/x.jinja"`.
- ``tensor `NAME` has type TYPE, which this engine cannot run``: a weight uses a format the engine does not run. Pick a file in a format from the next section.
- ``GGUF metadata `ARCH.attention.value_length`: ... only equal key and value lengths are supported``: the model's attention keys and values differ in size. Pick another model.
- `cannot determine the end-of-generation token`: the file names no end token and the vocabulary holds no known one. Pick another file of the model.

A `context_length` above the model's maximum fails with `context_length N exceeds the model maximum M`, a `CucaError::Config`. Device, configuration and download errors are on [Limits and errors](@/providers/local/limits-and-errors.md).

## Weight formats

The weight format is how the file stores the numbers of the model. The engine runs these:

| Format | On the CPU | On a GPU |
|---|---|---|
| Q4_0, Q4_1, Q8_0 | stays quantized | stays quantized |
| Q5_0, Q5_1 | stays quantized | converted to Q8_0 at load |
| K-quants (Q2_K to Q6_K) | stays quantized | converted to Q8_0 at load |
| Q8_1 | stays quantized | converted to Q8_0 at load |
| MXFP4 | stays quantized | stays quantized |
| F32, F16, BF16 | held as 32-bit floats, 4 bytes per element | held as 32-bit floats, 4 bytes per element |
| IQ4_NL | read only as an embedding table; a weight matrix fails the load | the same |
| other IQ types | the file opens and the load fails when it needs such a tensor | the same |

- Memory is about the file size for the quantized formats. F16 and BF16 files take about twice their file size, because they are held at 4 bytes per element.
- A GPU converts the formats it does not run natively to Q8_0, one row at a time. The change is smaller than the error a Q4_K or Q6_K file already carries, but GPU replies can differ a little from CPU replies for those files.
- A GPU file that mixes Q4_0, Q4_1 and Q8_0 weights keeps each weight in its own format, so the Q4_0 presets with a Q8_0 embedding load as they are.
- A mixture of experts is split once at load into one weight per expert. `experts { tiered ... }` leaves the experts in the file instead, see [Tiered experts](@/providers/local/expert-tiers.md).

## Tokenizers

A tokenizer cuts text into numbered tokens. The engine builds it at load from the file's `tokenizer.ggml.*` keys, in pure Rust. `tokenizer.ggml.model` picks the scheme:

| Value | Scheme | Used by |
|---|---|---|
| `llama` | SentencePiece: merges characters by score and falls back to `<0xHH>` byte tokens | Gemma 3, ERNIE 4.5 |
| `gpt2` | byte-level BPE (merging common byte pairs), with the word-cutting rule that `tokenizer.ggml.pre` names | Qwen, SmolLM, Phi, Granite, OLMo, Mistral Small 4, Muse Glimmer, DeepSeek |
| `gemma4` | Gemma 4's own scheme, which writes spaces as `▁`; also `gpt2` with `pre` set to `gemma4` | Gemma 4 |

Any other value fails the load with ``unsupported tokenizer model `NAME` ``. A `gpt2` file whose `tokenizer.ggml.pre` is not on this list fails with ``unsupported pre-tokenizer `NAME` ``:

`default`, `llama3`, `llama-v3`, `llama-bpe`, `falcon3`, `falcon-h1`, `midm-2.0`, `lfm2`, `jina-v5-nano`, `smaug-bpe`, `dbrx`, `qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2`, `hunyuan`, `solar-open`, `qwen35`, `gpt-2`, `gigachat`, `a.x-4.0`, `mellum`, `modern-bert`, `olmo`, `trillion`, `granite-docling`, `exaone4`, `starcoder`, `refact`, `command-r`, `exaone`, `mellum2`, `tekken`, `pixtral`, `gpt-4o`, `llama4`, `kanana2`, `talkie`, `minimax-m2`, `deepseek-v3`, `joyai-llm`, `hunyuan-dense`, `hy_v4`, `gemma4`.

- Special tokens such as `<|im_start|>` count as one token wherever a rendered prompt contains them.
- A BOS token (the token that marks the start of a prompt) is added when the file says so, and never twice when the rendered prompt already starts with one. Gemma 4 always gets it.
- A SentencePiece model with `tokenizer.ggml.add_space_prefix` puts a space token in front of text that follows a special token, as llama.cpp does. Its prompt can then have one more token than Hugging Face's tokenizer counts.
