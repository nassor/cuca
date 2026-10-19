+++
title = "Speculative decoding with a drafter"
description = "DrafterSource, the draft and accept rounds, the governor that turns drafting off, measured speedups and what was verified."
template = "page.html"
weight = 6
+++

# Speculative decoding with a drafter

<dl class="page-facts">
<dt>In one line</dt>
<dd>A multi-token-prediction drafter proposes tokens that the target verifies in one batched pass; a governor drafts only while it is faster.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, a drafter preset or an in-trunk prediction head.</dd>
<dt>Read this if</dt>
<dd>You want faster decoding on a model with a drafter, or need the acceptance rule and the measured speedups.</dd>
</dl>

## Speculative decoding with a drafter

A multi-token-prediction drafter proposes tokens that the target model verifies in one batched forward pass. Drafting is off by default. `LocalModelConfig::drafter` is an `Option<DrafterSource>`:

- `DrafterSource::File(ModelSource)` is a separate drafter GGUF, resolved and downloaded like any model source. `ModelSource::into()` converts. The preset `gemma4-e2b-mtp-f16` (170 MB, architecture `gemma4-assistant`) pairs with the Gemma 4 E2B presets. It is a four-layer decoder that reads the target's K/V cache, so it adds no cache memory. On the CPU the loader stores the drafter's float matrices as Q8_0, because the drafter only proposes tokens and its rounding cannot change the output. A separate drafter file runs on the CPU and on a GPU device.
- `DrafterSource::InTrunk` is the prediction head inside the main model's own GGUF file, for a Qwen 3.5 `*-MTP` file such as `unsloth/Qwen3.5-0.8B-MTP-GGUF` (the preset `qwen3.5-0.8b-q8_0`). Nothing more is downloaded. The head is one gated-attention block with a K/V cache of its own (4 KB per context position and 21.7 MB of weights for the 0.8B model), and it runs on the CPU only. A GPU device with `InTrunk` is a `CucaError::Config`, and a file without a head fails to load with an error that says so.
- A drafter must match its target: the `gemma4-assistant` drafter pairs with a `gemma4` model of the same hidden width and vocabulary, or `load` fails with a `CucaError::Provider`.

```rust,name=A Gemma 4 target with its drafter
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        source: "gemma4-e2b-q4_0".parse::<ModelSource>()?,
        drafter: Some("gemma4-e2b-mtp-f16".parse::<ModelSource>()?.into()),
        draft_tokens: 3,
        ..Default::default()
    })
    .await?;
    // ... send requests; afterwards:
    println!("{:?}", model.draft_stats());
    Ok(())
}
```

### Rounds and acceptance

- Each round the drafter proposes `draft_tokens` tokens, fewer near `max_tokens`, near the context limit and, for a hybrid model, past the rollback limit of 16 drafts per verification call.
- At temperature `0.0` a draft is accepted when it equals the target's greedy token. On the CPU the output equals plain decoding bit for bit.
- Above `0.0` a draft is accepted with probability `min(1, p/q)`, where `p` and `q` are the target's and the drafter's distributions after temperature, `top_k` and `top_p`. A rejected draft is replaced by a token drawn from the residual distribution `max(0, p - q)`, which preserves the target's sampling distribution.
- `LocalModel::draft_stats()` returns `DraftStats { rounds, drafted, accepted }`, summed over every request. `DraftStats::acceptance_rate()` returns the accepted fraction, `None` before the first round.
- A longer round gains more on predictable text and wastes more verification on unpredictable text, so `draft_tokens` is worth tuning on your own prompts.
- The in-trunk head ranks only the first 65,536 token ids in a draft step, which is about five times cheaper than ranking the whole vocabulary for 4 percent fewer accepted tokens per round (Qwen 3.5 0.8B, whose vocabulary has 248,320 ids).

### Governor

An engine-side governor drafts only while drafting is measurably faster than plain decoding:

- It measures plain decoding first, then speculative rounds, as moving averages of tokens per second.
- It keeps drafting while speculation is at least 1.02 times as fast as plain decoding, and refreshes the plain estimate with two plain rounds every 64 rounds.
- While plain decoding is faster, it re-probes with two speculative rounds after 32 plain rounds. Each failed probe doubles the interval, up to 512 rounds.
- A run with a fixed `seed` and a temperature above `0.0` always drafts, so equal prompts still reproduce equal tokens.
- The governor state is kept across requests.

### Measured speedups

Speedup is speculative throughput over plain decoding on the same device, in release builds, CPU builds with `-C target-cpu=native`, 128-token replies, on a 32-core CPU and an AMD RX 9070 XT (Windows).

- Gemma 4 E2B Q4_0 with `gemma4-e2b-mtp-f16`: on the CPU 1.17x on a code prompt, 1.37x on a list and 1.00x on prose; on Vulkan 0.95x to 0.97x, because single-token GPU decoding is bound by the host and the governor keeps the run near plain decoding. The drafter pays on the CPU.
- Qwen 3.5 0.8B Q8_0 with the in-trunk head, CPU, three drafts per round: 1.62x on code, 1.18x on a list and 0.98x on prose. Prose text is hard to predict, so the governor mostly turns drafting off there.

### Verified

`tests/provider_local_mtp.rs` loads a target alone and with its drafter and compares greedy output. It asserts identical output and token counts on the CPU, and reports identity without asserting it on a GPU. Recorded CPU runs matched plain greedy decoding for the Gemma 4 E2B pair and for the Qwen 3.5 in-trunk head, and recorded Vulkan runs of the Gemma 4 pair produced identical output.
