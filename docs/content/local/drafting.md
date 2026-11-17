+++
title = "Speculative decoding"
description = "Speed up the local model with a drafter that guesses tokens the main model then checks: how a round works, which drafter pairs with which model, how to turn it on, and what speedup to expect."
template = "page.html"
aliases = ["providers/local/drafting/"]
weight = 9
+++

# Speculative decoding

<dl class="page-facts">
<dt>In one line</dt>
<dd>A small drafter guesses the next few tokens and the main model (the target) checks them all in one pass. At temperature 0 the text is the same as without a drafter, and decoding can be faster.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, and a drafter that fits your model: a preset, a checkpoint source or a prediction head inside the model file.</dd>
<dt>Read this if</dt>
<dd>You run one request at a time on the CPU and want faster decoding, or you need the pairings, the errors and the measured speedups.</dd>
</dl>

A model normally writes one token per pass over its weights. That pass is the slow part, because the weights must be read from memory each time. Speculative decoding gets more than one token out of each pass of the big model.

A **drafter** is a small, fast helper. It proposes the next few tokens, and the **target** (your main model) checks all of them in a single pass. The target keeps the ones it agrees with. Drafting is off by default.

## How a round works

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 196" role="img" aria-labelledby="rnd-title rnd-desc">
  <title id="rnd-title">One speculative round with four drafted tokens</title>
  <desc id="rnd-desc">Three steps in a row. The drafter proposes k tokens. The target checks all of them in one pass. The round keeps the longest prefix of drafts the target agrees with, plus one token of the target's own. Below, an example of four drafted tokens: the first two match the target and are kept, the third differs and is replaced by the target's token, and the fourth is dropped. The round yields three tokens from one pass of the target.</desc>
  <defs>
    <marker id="rnd-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One round with 4 drafts</text>
  <rect class="blk blk-ctl" x="6" y="36" width="190" height="52" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="101" y="58">1. Drafter proposes</text>
  <text class="t-sm t-mid" x="101" y="74">k tokens, one cheap pass</text>
  <rect class="blk blk-data" x="235" y="36" width="190" height="52" rx="6"/>
  <text class="t-lbl t-mid t-data" x="330" y="58">2. Target verifies</text>
  <text class="t-sm t-mid" x="330" y="74">all k in one pass</text>
  <rect class="blk" x="464" y="36" width="190" height="52" rx="6"/>
  <text class="t-lbl t-mid" x="559" y="58">3. Keep the prefix</text>
  <text class="t-sm t-mid" x="559" y="74">plus 1 token of the target</text>
  <path class="arw" d="M198 62 H231" marker-end="url(#rnd-arrow)"/>
  <path class="arw" d="M427 62 H460" marker-end="url(#rnd-arrow)"/>
  <text class="t-ax" x="6" y="118">DRAFTED TOKENS</text>
  <rect class="blk blk-ctl" x="6" y="126" width="70" height="30" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="41" y="145">t1</text>
  <rect class="blk blk-ctl" x="84" y="126" width="70" height="30" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="119" y="145">t2</text>
  <rect class="blk" x="162" y="126" width="70" height="30" rx="6"/>
  <text class="t-sm t-mid" x="197" y="145">t3</text>
  <rect class="blk" x="240" y="126" width="70" height="30" rx="6"/>
  <text class="t-sm t-mid" x="275" y="145">t4</text>
  <text class="t-ax t-mid" x="41" y="172">kept</text>
  <text class="t-ax t-mid" x="119" y="172">kept</text>
  <text class="t-ax t-mid" x="197" y="172">replaced</text>
  <text class="t-ax t-mid" x="275" y="172">dropped</text>
  <text class="t-sm" x="336" y="136">t1 and t2 equal the target's own choice.</text>
  <text class="t-sm" x="336" y="152">t3 does not: the target's token takes its place.</text>
  <text class="t-sm" x="336" y="168">One pass of the target gave 3 tokens.</text>
</svg>
</div>
<p class="dgm-cap"><b>Several tokens can come out of one pass of the target.</b> The target checks every draft at once, keeps the longest run it agrees with, and adds one token of its own.</p>
</div>

### Why the output stays the target's

The target decides every token. At temperature 0, a draft is accepted only when it equals the token the target would pick on its own, so the text is identical to plain decoding. Above 0, a draft is accepted with a probability that preserves the target's sampling distribution, and a rejected draft is replaced by a corrected draw, so the text follows the same distribution but is not the same text. Only the speed changes.

- On the CPU, the greedy output equals plain decoding bit for bit.
- On a GPU, a batch of rows equals single steps only to float rounding, so a greedy run can differ from plain decoding where two choices are that close.

### When it pays

Speculation pays when the target is limited by reading its weights, which is the case when one request runs alone. With several requests running, the engine already batches their decode rows, which shares the weight reads, so drafting gains less there. Each running request drafts and verifies in the shared step. The engine measures speculation separately for each number of requests decoding together (1, 2, 3 to 4, 5 to 8, up to 33 to 64) and drafts only in the sizes where it measures a gain. A step in which a request prefills does not draft.

## Turn it on

This pairs Gemma 4 E2B with its 170 MB assistant drafter on the CPU. The drafter downloads on the first run, like a model.

```kdl,name=gemma4-drafter.kdl
source "gemma4-e2b-q4_0"
drafter "gemma4-e2b-mtp-f16"
draft_tokens 5
```

```bash,name=Linux and macOS: the example with a drafter
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local -- --config gemma4-drafter.kdl
```

```powershell,name=Windows (PowerShell): the example with a drafter
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local -- --config gemma4-drafter.kdl
```

After the reply, the example prints the draft counters on stderr:

```text,name=Expected on stderr after the reply
drafter: <accepted> of <drafted> drafted tokens accepted (<percent>%) in <n> rounds
```

The line is missing when no round drafted a token. The governor (see below) turns drafting off while plain decoding is faster.

In Rust, set `drafter` and `draft_tokens` in `LocalModelConfig`:

```rust,name=A Gemma 4 target with its drafter (provider-local)
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        source: "gemma4-e2b-q4_0".parse::<ModelSource>()?,
        drafter: Some("gemma4-e2b-mtp-f16".parse::<ModelSource>()?.into()),
        draft_tokens: 5,
        ..Default::default()
    })
    .await?;
    // ... send requests; afterwards:
    println!("{:?}", model.draft_stats());
    Ok(())
}
```

`LocalModelConfig::drafter` is an `Option<DrafterSource>`. `DrafterSource::File(ModelSource)` is a separate drafter file, found and downloaded like any model source. `DrafterSource::InTrunk` is the prediction head inside the main model's own file. In KDL the first is `drafter "<source>"` and the second is `drafter "in-trunk"`. The engine picks the drafter kind from the drafter file itself.

## Pick a drafter

Each drafter works with one family of target. Use the pair in the same bullet.

- Gemma 4 assistant: set `drafter` to the preset `gemma4-e2b-mtp-f16`. Target: Gemma 4 E2B (a `gemma4` model of the same hidden width and vocabulary). Runs on the CPU or a GPU.
- Qwen 3.5 head: set `drafter` to `"in-trunk"`. Target: a dense Qwen 3.5 `*-MTP` file, such as the preset `qwen3.5-0.8b-q8_0`. CPU only.
- EAGLE: set `drafter` to the checkpoint `mistralai/Mistral-Small-4-119B-2603-eagle:9a8ea22dca0161ff7af7879f4a0000f65314797d`. Target: Mistral Small 4. CPU only.
- DFlash: set `drafter` to the preset `muse-glimmer-30b-dflash-q4_0`, or the checkpoint `meta-models/Muse-Glimmer-30B-assistant:c676cce91658f9568aaf264a416211e2782735b7`. Target: Muse Glimmer 30B. CPU only.
- DSpark: set `drafter` to the target's own checkpoint, `deepseek-ai/DeepSeek-V4.1-Flash-1017:<revision>`. Target: DeepSeek-V4.1. CPU only.

A checkpoint source is `repo:revision`, a Hugging Face safetensors checkpoint. The engine converts it once, in process, and caches the result. DSpark and DeepSeek-V4.1 are on [Run DeepSeek-V4.1](@/local/deepseek-v41.md). The checkpoint classes are on [Model files and downloads](@/local/model-sources.md).

- The Gemma 4 assistant is a four-layer decoder that reads the target's K/V cache (the model's memory of past tokens), so it adds no cache memory.
- The Qwen 3.5 head is a prediction head already inside the model file. Nothing more is downloaded. It keeps a K/V cache of its own per request slot, 4 KB per context position for the 0.8B model.
- EAGLE is converted from a 392 MB FP8 download to a 0.4 GB GGUF at the default `convert.quant` of `q8_0`. It keeps 2.5 KB of K/V cache per context position and request slot.
- DFlash proposes a whole block of tokens in one pass instead of one pass per token. The checkpoint converts to about 1.4 GB at `q4_0` or 2.7 GB at the default `q8_0`. It keeps a ring of the last 2,048 positions per request slot, 82 MB at any context length. Muse Glimmer takes at most 15 drafts.
- DSpark is a block drafter like DFlash, with stages of the DeepSeek-V4.1 kind. Its converted file is 7.91 GB at the default precision.
- The Qwen head, EAGLE, DFlash and DSpark follow every position of their sequence, so each request slot keeps a cache of its own over the drafter's shared weights, and the cache memory above is multiplied by `batching.max_sequences`. A new request takes the free slot whose cache shares the longest prefix with its prompt.

## Layer stages

A model whose first layers run on workers (see [Layer stages](@/server/model-server.md#layer-stages)) drafts on the coordinator, which holds the last layers, the output head and the drafter. A round sends one verification pass of the pending token and the drafts over the network, the same single call per worker as a plain step, so an accepted draft saves a whole round trip. What a chain accepts:

| Drafter | On a chain |
|---|---|
| Gemma 4 assistant | yes, it reads the K/V of the coordinator's own layers |
| in-trunk head | yes, also on a target with recurrent state (Qwen 3.5): the workers roll a rejected draft back |
| EAGLE | yes (Mistral Small 4 is attention only) |
| DFlash, DSpark | no: they read the hidden rows of several layers of the target, and a worker returns none |

- The drafter and the coordinator's layers run on the CPU. The last stage keeps the token embedding table for the drafter even when the model's output head is not tied to it.
- A target with recurrent state (Qwen 3.5 and every `*-MTP` file the in-trunk head comes from) rolls a rejected draft back on the workers too: the next call carries the position to go back to, and a worker returns to it from the record its last verification pass kept (up to 17 rows per pass, as on one node). A worker whose delta-net layers run on a GPU keeps no such record and offers one row, so a drafter is refused there. The coordinator learns what each worker offers when it opens the session, and the wire version is 4. DeepSeek-V4.1 chains (compressor state) roll back the same way, but the only drafter for that model, DSpark, does not pair with a chain.
- Measured on Qwen 3.5 0.8B Q8_0 with the in-trunk head and `draft_tokens 3`, one worker with 12 layers on the same machine over the frame stream, all on the CPU, in three runs: counting from 1 to 120 (300 tokens) ran at 52 tokens per second without a drafter and at 107 to 123 with it (one node: 53 without, 114 to 127 with), with 59% to 61% fewer worker calls over the 800 tokens of a run. A story (200 tokens) went from 48 to 52 tokens per second to 55 to 56 (one node: 52 to 53, then 49 to 55). The replies equal plain greedy decoding's.
- A failed call during a round fails that request, as any failed stage call does; the drafter is reset and the next request starts a new session.

## Draft length and counters

- `draft_tokens` is how many tokens the drafter may propose per round. The default is 5, and the range is `1` to `16`. A different value fails the load with `draft_tokens must be in 1..=16, got N`. It is ignored without a drafter.
- It is a cap, not a fixed length. A round proposes fewer near `max_tokens` and the context limit, and the engine drops drafts that are not expected to pay off.
- A request with a fixed `seed` and a temperature above 0 always drafts exactly `draft_tokens`, so equal prompts give equal tokens.
- `LocalModel::draft_stats()` returns `DraftStats { rounds, drafted, accepted }`, summed over every request. `DraftStats::acceptance_rate()` returns the accepted fraction, and `None` before the first drafted token.

## The governor

Drafting costs work, so an engine-side governor drafts only while it is measurably faster than plain decoding. It measures plain decoding first and then speculative rounds, keeps drafting while speculation is at least 1.02 times as fast, and remembers its state across requests. While plain decoding wins, it tests drafting again from time to time, and waits longer after each failed test.

## Errors when a pair does not fit

A drafter is built for one target. A wrong pair fails the load. The messages and what to do:

- ``the `gemma4-assistant` drafter pairs with a `gemma4` model, not `<arch>` ``: the target is not a Gemma 4 model. Use a Gemma 4 target.
- `drafter hidden width N differs from the target's M: it was trained for another model size`: the drafter was made for a bigger or smaller model. Use the drafter made for your model size.
- `drafter vocabulary N differs from the target's M`: the two models use different token lists. Use the drafter made for your model.
- ``the `mistral4-eagle` drafter pairs with a `mistral4` model, not `<arch>` ``: EAGLE only pairs with Mistral Small 4. Use a `mistral4` target.
- `the EAGLE drafter's <what> is X but the model's is Y: it was trained for another model`: a setting of the drafter differs from the target's. Use the drafter published for this model.
- ``the dflash drafter reads the hidden rows of several layers of the target, and a stage call returns none of a worker's: it cannot be combined with `remote_stage` ``: DFlash and DSpark do not pair with layer stages. Run the model on one machine, or without a drafter.
- ``<drafter> cannot be combined with `remote_stage` here: a stage of this chain holds recurrent or compressor state and keeps no verification record, so a rejected draft cannot be rolled back on it``: a worker (or the coordinator's own layers) holds delta-net layers on a GPU, which keep no verification record. Run the stage on the CPU, run without a drafter, or run on one machine.
- ``<drafter> starts from a token's embedding, which the last stage of this chain does not hold``: the coordinator could not load the token embedding table for the drafter.
- `... runs on the CPU only ...` for the in-trunk head, EAGLE, DFlash or DSpark, and `DrafterSource::InTrunk runs on the CPU only, not on <device>`: these drafters cannot run on a GPU. Set `device "cpu"`, and drop `gpu_layers` and `shards`.
- ``this `<arch>` file has no in-trunk multi-token-prediction head (...); use a separate drafter file instead``: the model file has no built-in head. Use a Qwen 3.5 `*-MTP` file, or a separate drafter.
- ``the in-trunk multi-token-prediction head of `qwen35moe` has a mixture-of-experts FFN, which the head does not support; run without a drafter``: Qwen3.6-35B-A3B has a head that is not supported. Run without a drafter.
- `draft_tokens must be in 1..=16, got N`: `draft_tokens` is out of range. Set it from 1 to 16.

DFlash and DSpark also fail to load when their width differs from the target's, when they read a layer the target does not have, or when their mask token is outside the target's vocabulary. Use the pair from the list above.

## What to expect

Speedup is speculative tokens per second divided by plain tokens per second, for the same model on the same device. Method: the CPU of a Ryzen 9 desktop, release build with native CPU flags, greedy sampling, 128-token replies, three prompts (a Python function, a list of the planets, a short story). The greedy output equaled plain decoding in every run.

| Target and drafter | `draft_tokens` | Code | List | Story |
|---|---|---|---|---|
| Gemma 4 E2B Q4_0, Gemma 4 assistant | 5 | 1.32x | 1.31x | 1.01x |
| Qwen 3.5 0.8B Q8_0, in-trunk head (plain 55 tokens/s) | 3 | 1.86x | 1.37x | 0.99x |
| Muse Glimmer 30B Q4_0, DFlash (plain 3.4 tokens/s) | 3 | 1.76x | 1.69x | 1.13x |
| Mistral Small 4 Q4_0, EAGLE (plain 11.2 tokens/s) | 2 | 1.0x | 1.0x | 1.0x |

- Predictable text gains most. Code and lists gain, and a story gains little, because the drafter's guesses are accepted less often.
- EAGLE gives no gain on Mistral Small 4. It is a mixture-of-experts model, and every extra row routes to its own experts, so checking more rows costs almost as much as decoding them.
- A longer draft is not always faster. DFlash on Muse Glimmer gave 1.47x over all three prompts at 3 and at 7 drafts, and 1.15x at 15, because verifying a long block costs more.
- DSpark with DeepSeek-V4.1 gave the same output as plain decoding and no speedup in the one run made: both runs are limited by reading experts from disk.
- The Gemma 4 pair on a GPU ran at about 1.0x.
