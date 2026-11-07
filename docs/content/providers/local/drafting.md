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
- `DrafterSource::InTrunk` is the prediction head inside the main model's own GGUF file, for a Qwen 3.5 `*-MTP` file such as `unsloth/Qwen3.5-0.8B-MTP-GGUF` (the preset `qwen3.5-0.8b-q8_0`). Nothing more is downloaded. The head is one gated-attention block with a K/V cache of its own (4 KB per context position and 21.7 MB of weights for the 0.8B model), and it runs on the CPU only. A GPU device with `InTrunk` is a `CucaError::Config`, and a file without a head fails to load with an error that says so. A `qwen35moe` file (Qwen3.6-35B-A3B) has a prediction block with a mixture-of-experts FFN, which the head does not support: `InTrunk` on it is a `CucaError::Config` that names the mixture-of-experts FFN.
- The Mistral Small 4 EAGLE drafter (`mistralai/Mistral-Small-4-119B-2603-eagle`, 392 MB, published in Mistral's own format: `params.json` and an FP8 `consolidated.safetensors`, no `config.json`) is a `DrafterSource::File` with a checkpoint source, `"mistralai/Mistral-Small-4-119B-2603-eagle:9a8ea22dca0161ff7af7879f4a0000f65314797d".parse::<ModelSource>()?.into()`. It is converted once, in process, to a GGUF of architecture `mistral4-eagle` at `convert.quant` (Q8_0 by default, 0.4 GB): the FP8 weights times their `qscale_weight`, the activation scales dropped. It is two latent-attention layers over a dense FFN of width 12,288 whose input is `fc(cat(embed(t[j+1]), h[j]))`, the target's token embedding first and its final-norm hidden row second; its embedding and LM head are the target's, and a draft step ranks the whole vocabulary. It keeps a K/V cache of its own (2.5 KB per context position), observes every position (so the engine runs one request at a time with it), pairs with a `mistral4` model on the CPU only (a GPU device or another target is an error), and refuses a drafter whose hidden width, vocabulary, RoPE, query scaling or attention geometry differ from its target's. GGUF layout: `mistral4-eagle.*` keys as `mistral4` (plus `embedding_length_out`, `vocab_size`, explicit `attention.temperature_scale` and `temperature_length`), tensors `nextn.eh_proj.weight` `[dim, 2*dim]`, `output_norm.weight`, and per layer `blk.N.{attn_norm, attn_q_a, attn_q_a_norm, attn_q_b, attn_kv_a_mqa, attn_kv_a_norm, attn_k_b, attn_v_b, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down}.weight`; no token embedding and no output head. llama.cpp has no counterpart: its `eagle3` architecture is a different design (three fused target layers, a vocabulary map).
- A drafter must match its target: the `gemma4-assistant` drafter pairs with a `gemma4` model of the same hidden width and vocabulary, or `load` fails with a `CucaError::Provider`.
- A DFlash block drafter (GGUF architecture `dflash`) pairs with a Muse Glimmer target on the CPU: see *DFlash block drafter* below.

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

- Each round the drafter proposes at most `draft_tokens` tokens (default 5), fewer near `max_tokens`, near the context limit and, for a hybrid model, past the rollback limit of 16 drafts per verification call.
- Within that cap, a speculative round always verifies its first draft and adds each further one only while it pays. The engine predicts that draft `j` is accepted with probability `a_j`, the product of the drafter's own probabilities of its proposals so far, scaled by the ratio of accepted to predicted drafts it has observed. It keeps draft `j` while `a_j * T_j > E_{j-1} * dT`: `T_j` is the measured cost of a round verifying `j` drafts, `dT` what one more draft adds to it (a verification row and a drafter step), and `E_{j-1}` the expected tokens of the round without it.
- At temperature `0.0` a draft is accepted when it equals the target's greedy token. On the CPU the output equals plain decoding bit for bit.
- Above `0.0` a draft is accepted with probability `min(1, p/q)`, where `p` and `q` are the target's and the drafter's distributions after temperature, `top_k` and `top_p`. A rejected draft is replaced by a token drawn from the residual distribution `max(0, p - q)`, which preserves the target's sampling distribution.
- `LocalModel::draft_stats()` returns `DraftStats { rounds, drafted, accepted }`, summed over every request. `DraftStats::acceptance_rate()` returns the accepted fraction, `None` before the first round.
- A drafter speculates only while one request runs alone and none waits: with several running requests the engine batches their decode rows instead, which already shares the weight reads that speculation saves. The in-trunk head observes every position of its one sequence, so with it the engine runs one request at a time (`batching.max_sequences` is treated as 1).
- `draft_tokens` is only the cap: the rule above sets the length per round from the drafter's confidence and the verification cost it measures on the running device.
- The in-trunk head ranks only the first 65,536 token ids in a draft step, which is about five times cheaper than ranking the whole vocabulary for 4 percent fewer accepted tokens per round (Qwen 3.5 0.8B, whose vocabulary has 248,320 ids).

### DFlash block drafter

A DFlash drafter proposes a whole block of tokens in one pass instead of one pass per token. `DrafterSource::File` takes either the GGUF llama.cpp's converter writes (`dflash-Muse-Glimmer-30B-Q4_0.gguf` in `ggml-org/Muse-Glimmer-30B-GGUF`, as `repo:revision:file`) or the safetensors checkpoint `meta-models/Muse-Glimmer-30B-assistant:c676cce91658f9568aaf264a416211e2782735b7` as a `ModelSource::Checkpoint`, converted at `LocalModelConfig::convert.quant` (about 1.4 GB at `q4_0`, 2.7 GB at the default `q8_0`; the converter writes the same tensors and keys as llama.cpp, without the target's vocabulary).

- It reads the residual stream after five target layers (the outputs of layers 1, 13, 25, 37 and 49 of 52; the file stores llama.cpp's `target_layers` `[2, 14, 26, 38, 50]`, the inputs of the next layer) at every accepted position, and keeps its own K/V of them: a ring of the last 2,048 positions, 82 MB, whatever the context length.
- A round runs five sliding-window layers once over a block of `1 + draft_tokens` rows (at most 16: the pending token and mask tokens, embedded with the target's raw table; the rows attend to each other in both directions and to the context) and ranks them with the target's LM head, logit multiplier and soft-cap included (through a Q8_0 copy of the head when the target's head is a K-quant: see *Measured speedups*). The cost is one pass per round, not per draft, and the engine's cost model prices it that way; it prices the drafts by the acceptance it has observed (`rho^j`) rather than by the drafter's confidence, which is flat on the mask rows. `draft_tokens` therefore also sets the block length; the DFlash paper (Table 8) measures 6.33 against 5.09 average accepted tokens for blocks of 16 against 8.
- It is CPU only and observes every position of one sequence, so the engine runs one request at a time. The context cache follows the trunk's prefix reuse, truncation and failed requests; a gap in it costs proposal quality, never output.
- Published numbers (not reproduced here, except the last): average accepted length 4.35 to 7.84 on Qwen3-4B (paper Table 1, H200), 2.4 to 5.0 for `RedHatAI/Mistral-Small-4-119B-2603.dflash` at seven drafts, and for Muse Glimmer 30B 3.1x on an RTX 5090 (llama.cpp), 1.5x and 1.8x on Apple M4 and M5 Max (the assistant's model card); measured on the CPU of this project: 3.1 to 3.9 tokens per drafting round and 1.15x to 1.47x.
- Verified by unit tests (greedy output equals plain for several blocks, windows and caps; warm equals cold; failed-request recovery; a plain-loop reference of the block; the Q8_0 copy of a K-quant head against the original) and, with `CUCA_LOCAL_DFLASH_REF`, by a transformers oracle on a tiny checkpoint (max |dlogit| below 2e-4, equal drafts); the live Muse run is `tests/provider_local_mtp.rs` with `CUCA_LOCAL_MODEL=muse-glimmer-30b-q4_0 CUCA_LOCAL_DRAFTER=muse-glimmer-30b-dflash-q4_0 CUCA_LOCAL_DRAFT_TOKENS=3,7,15`, 1.46x, 1.47x and 1.15x (see *Measured speedups*).

### Governor

An engine-side governor drafts only while drafting is measurably faster than plain decoding:

- It measures plain decoding first, then speculative rounds, as moving averages of tokens per second.
- It keeps drafting while speculation is at least 1.02 times as fast as plain decoding, and refreshes the plain estimate with two plain rounds every 64 rounds.
- While plain decoding is faster, it re-probes with two speculative rounds after 32 plain rounds. Each failed probe doubles the interval, up to 512 rounds.
- A run with a fixed `seed` and a temperature above `0.0` always drafts the full `draft_tokens`, so equal prompts still reproduce equal tokens.
- The governor state is kept across requests.
- A drafter that keeps a cache of its own (the EAGLE drafter, the DFlash block drafter and the Qwen 3.5 head) does not pay for it in plain rounds. A round that drafts nothing only hands its hidden rows (the features, for DFlash) to the drafter, which holds them and the range of positions it has not seen, at most 1,024 positions and 16 MiB. When a round drafts again (a probe or a resumed drafting), the drafter first takes the held rows into its cache in one batched pass, in chunks of the trunk's prefill size; at the bound it catches up first. The CPU kernels are row-exact, so the cache after catching up equals the one a per-round upkeep leaves, bit for bit: the proposals and the output do not change, only where the cost falls. The pass is a cost of switching and the governor charges it to neither mode. A new prompt drops the held rows it does not share, and a failed request forgets them. Measured per token (plain round, then catching up per row): EAGLE 4.4 ms of 86.6 ms and 0.89 ms, DFlash 3.5 ms of 288 ms and 1.2 ms, the Qwen 3.5 head 0.09 ms of 17.3 ms and 0.05 ms. The Gemma 4 assistant reads the target's K/V and has no upkeep.

### Measured speedups

Speedup is speculative throughput over plain decoding on the same device, in release builds, CPU builds with `-C target-cpu=native`, 128-token replies, at the default `draft_tokens` of 5, on a 32-core CPU and an AMD RX 9070 XT (Windows).

- Gemma 4 E2B Q4_0 with `gemma4-e2b-mtp-f16`: on the CPU 1.32x on a code prompt, 1.31x on a list and 1.01x on prose; on Vulkan 0.97x to 1.00x, because single-token GPU decoding is bound by the host and the governor keeps the run near plain decoding. The drafter pays on the CPU.
- Qwen 3.5 0.8B Q8_0 with the in-trunk head, CPU: 1.59x on code, 1.40x on a list and 0.98x on prose; with `draft_tokens` 3, 1.86x, 1.37x and 0.99x. Prose text is hard to predict, so the governor mostly turns drafting off there. Its upkeep is 0.09 ms of a 17.3 ms token (0.5 %), so lazy upkeep changes little: 1.25x over the three prompts at `draft_tokens` 3 with per-round upkeep and 1.27x with it, in two traced runs.

#### EAGLE on Mistral Small 4 (CPU)

`mistral-small-4-119b-q4_0` with the EAGLE drafter, release build with `-C target-cpu=native`, 128-token replies, three prompts, `tests/provider_local_mtp.rs`, before lazy upkeep (plain 11.7 tok/s; greedy output equal to plain in all twelve runs):

| `draft_tokens` | code | list | prose | all three | drafts accepted | drafting rounds |
|---|---|---|---|---|---|---|
| 1 | 0.96x | 0.81x | 0.85x | 0.87x | 76 % (31 of 41) | 41 |
| 2 | 0.95x | 0.87x | 0.85x | 0.89x | 90 % (156 of 173) | 106 |
| 3 | 0.94x | 0.83x | 0.85x | 0.87x | 92 % (129 of 141) | 66 |
| 4 | 0.91x | 0.84x | 0.84x | 0.86x | 88 % (126 of 143) | 62 |

Two further runs at `draft_tokens` 2 (one with the round trace) measured 1.05x and 0.98x on code, 0.98x and 0.95x on the list and 0.97x on the story: run to run the same prompt moves by about 10 %. The drafter does not pay on this CPU, for these reasons (measured per round with the trace, `draft_tokens` 2):

- A verification forward over one row takes 90 ms (11 tok/s); over 2 rows 153 ms and over 3 rows 201 ms. The model is mixture-of-experts with 4 of 128 experts per token, and every row routes to its own experts, so extra rows share no expert weights: each costs 48 to 63 ms, 53 to 70 % of a decode step, where a dense model's extra row costs a few percent.
- A draft pass costs 15 ms (two layers and the vocabulary ranking). By the engine's cost model a first extra draft pays only above an acceptance of about 0.74 (the 55 ms of its row plus the 15 ms pass against a 95 ms token), and the governor drafts only then.
- The drafter had to take every committed position into its own cache, 4.4 ms per token (5 %), plain rounds included, so a prompt where the governor drafts nothing (the list and the story) ran below plain decoding: 0.95x to 0.98x in the traced runs and 0.81x to 0.87x in the four-length run above. Plain rounds no longer pay it: see *Lazy upkeep* below.
- Over the traced run a round that kept two drafts averaged 2.85 tokens (92 % of the drafts accepted) and the drafting rounds together 2.5, against 2.0 to 2.3 in community SGLang runs. At 237 ms for 2.85 tokens that is 83 ms a token against 90 ms plain, a 1.08x round before the 4.4 ms of upkeep (still paid in a round that drafts) and the rounds that keep a single draft (186 ms for 1.77 tokens, slower than plain), which bring the run back to about 1.0x.
- The 65,536-id proposal cap of the in-trunk head does nothing here: it only shortens the product on weights the row-batched kernel serves, and this model's output head is Q6_K (candle computes the whole product). A draft pass measured 14.9 ms with the cap and 15.1 ms without.

##### Lazy upkeep

Plain rounds no longer take their rows into the drafter's cache (see *Governor*). `draft_tokens` 2: the engine before the change and the engine after it, run back to back (old, new, old, new) on one machine state with `CUCA_LOCAL_ALLOW_MISMATCH=1`, then the traced runs of the new engine (plain 11.2 to 11.3 tok/s overall):

| engine | code | list | story | all three | drafted tokens |
|---|---|---|---|---|---|
| old, run 1 | 1.06x | 1.05x | 0.96x | 1.02x | 221 |
| new, run 1 | 0.99x | 1.00x | 0.98x | 0.99x | 90 |
| old, run 2 | 1.08x | 1.00x | 0.96x | 1.01x | 189 |
| new, run 2 | 1.02x | 0.99x | 1.00x | 1.00x | 86 |
| new, traced | 1.03x | 0.99x | 1.00x | 1.00x | 118 |
| new, traced, three loads of one process | 0.99x, 1.00x, 0.99x | 0.99x, 0.99x, 1.00x | 0.99x, 0.98x, 0.99x | 0.99x each | 45, 54, 53 |

- **The worst case is plain decoding.** The story prompt, where the new engine drafted 0 to 2 tokens in every run, measured 0.98x, 1.00x, 1.00x, 0.99x, 0.98x and 0.99x (six runs), the list 1.00x, 0.99x, 0.99x, 0.98x, 0.99x, 0.99x and 1.00x (seven runs); the old engine measured 0.96x on the story in both runs. A plain round of the new engine runs 86.4 to 86.8 ms (11.5 tok/s) in the four traced loads (5th to 95th percentile 85.6 to 88.9 ms in the first), with 0.00 ms of upkeep where the old engine paid 4.4 ms.
- **Catching up costs 0.89 ms a row**, against 4.4 ms for a row observed on a round of its own (11 passes of 2 to 63 rows in the first traced run, 302 rows, 268 ms, 0.4 % of the run; 0.65 % in the three-load run), because the two layers' weights are read once for all the rows. A round that drafts still pays its own upkeep (4.7 to 5.0 ms in a verification of 140 to 190 ms). The probes (stretches of at most four drafting rounds) cost 0.5 % to 0.6 % of the traced runs.
- **It drafts less, and the overall number is 1.0x.** The old engine measured plain decoding 5 % slow (its upkeep) and drafted 189 to 221 tokens a run; the new one measures the true plain rate, and over the first traced run the drafting rounds gain only 2 % on it (168 tokens in 14.2 s, 84.7 ms a token against 86.6), under the governor's 1.02 margin once its estimate, a moving average of per-round rates that weighs the last round by 0.3, takes a rejected round: in one traced probe three rounds that gained 16 % to 21 % (215 to 224 ms for three tokens) were followed by one fully rejected round (216 ms for one token), which dropped the estimate under plain decoding. So the new engine drafts 86 to 90 tokens a run and the code prompt falls from 1.06x and 1.08x to 0.99x and 1.02x. The drafter does not pay on this CPU: its gain is the 2 %, within the spread of the plain figure.
- **The earlier 0.81x to 0.87x did not reproduce.** In the session above the old engine measured 0.96x on the story prompt twice, which is its 5 % upkeep. Ten loads of a speculative engine (seven in ordinary runs of the live test, three in one traced process after the plain load) showed no load-order or load-to-load slowdown: the four traced loads ran plain rounds at 86.4 to 86.8 ms, the speed of the plain engine, and no load of the old or the new engine measured below 0.96x on a prompt where it drafts nothing. The plain engine's own figure moved by 4 % between sessions (11.7 tok/s in the four-length run, 11.2 to 11.3 here). Upkeep (5 %) and probing plus catching up (about 1 %) cannot make 13 % to 19 %, so [INFERENCE] the gap was that run's machine state, as its machine note says (memory that misbehaved), not the engine; it cannot be settled now.

#### DFlash on Muse Glimmer 30B (CPU)

`muse-glimmer-30b-q4_0` with `muse-glimmer-30b-dflash-q4_0`, same conditions (plain 3.4 tok/s; greedy output equal to plain in all runs of every row below):

| `draft_tokens` | code | list | prose | all three | accepted | tokens per drafting round |
|---|---|---|---|---|---|---|
| 3, first version | 1.38x | 1.32x | 0.98x | 1.20x | 90 % | 1.9 |
| 7, first version | 1.01x | 1.00x | 0.97x | 1.00x | 92 % | 2.0 |
| 15, first version | 0.97x | 0.97x | 0.97x | 0.97x | 62 % | |
| 3 | 1.74x | 1.69x | 1.12x | 1.46x | 70 % (424 of 607) | 3.1 |
| 3, lazy upkeep | 1.76x | 1.69x | 1.13x | 1.47x | 68 % (446 of 655) | 3.0 |
| 7 | 1.64x | 1.66x | 1.20x | 1.47x | 69 % (425 of 617) | 3.2 |
| 15 | 1.18x | 1.14x | 1.15x | 1.15x | 64 % (324 of 506) | 3.9 |

The first-version rows are the engine before two changes that only affect which tokens are proposed and how many are verified, never the output:

- **Ranking through a Q8_0 copy of the head.** The block's rows go through the target's LM head, which in this file is Q6_K (1.1 GB, 202,048 rows). candle's kernel for K-quants reads the whole head once per row, 24 ms a row: 194 ms for a block of 7 and 404 ms for a block of 15, a pass that should have cost a fraction of a plain token (287 ms). The drafter now ranks through a Q8_0 copy of the head (1.4 GB, built when it opens) that the row-batched kernel reads once for all rows: 65 ms a round at `draft_tokens` 3 and 7, 206 ms at 15.
- **Pricing drafts by observed acceptance.** The keep rule priced draft `j` by the drafter's own chained confidence, which on a block drafter's mask rows is flat while the drafts are right about nine times in ten; every round kept one draft (336 drafted in 332 rounds at `draft_tokens` 3). Block drafters now price draft `j` as `rho^j`, `rho` the accepted fraction of the drafts judged over the last twenty rounds, and a round keeps three drafts where they pay.

Cost per round now (milliseconds, medians over the run): one plain token 287 forward; 2 rows 352, 4 rows 457, 8 rows 853, 16 rows 1,667 to verify (65 ms an extra row for the first rows, 80 to 92 for 8 to 16); at 3 drafts 63 for the draft pass, 456 for the verification, 525 for the round, 3.1 tokens, 170 ms a token (1.7x a plain token). With the head copy alone (the old keep rule) the three rows read 1.22x, 1.12x and 0.98x. The prose prompt stays at 1.12 to 1.20x: its drafts are accepted less often. For comparison the paper reports 4.35 to 7.84 accepted tokens per round on Qwen3-4B, `RedHatAI/Mistral-Small-4-119B-2603.dflash` 2.4 to 5.0 at seven drafts, and the assistant's model card 3.1x on an RTX 5090 and 1.5x on an Apple M4 Max; 1.5x on this CPU matches the last. At `draft_tokens` 15 the block pass alone costs 206 ms and a 16-row verification 1.7 s, so the rule keeps 4.6 drafts on average and the run reaches 1.15x.

#### Machine note

Two loads of the same 67.7 GB `mistral-small-4-119b-q4_0` file on the measurement machine sometimes differ by one bit in one tensor (5 of 579 tensors across three loads of one run, found by hashing each tensor after `read_exact` and again after an independent positional re-read), which moves first-token logits by up to 1 %. The read paths are audited and tested; a ten-minute RAM and file-read probe did not reproduce it, so hardware or OS memory is suspected. `tests/provider_local_mtp.rs` loads the plain and the speculative model separately, so on that machine a mismatch between them has to be repeated before it is believed: speculative output was equal to plain in all twelve runs of the table above, and different in one extra run, whose plain model alone produced a story that the six other loads whose stories were compared all agreed on.

In the session that measured lazy upkeep, 7 of 30 prompt runs (nine runs of the live test) printed `identical false`, in four of the nine runs: three of the nine prompt runs of the old engine and four of the 21 of the new one, each with a story or a list whose first sentence or two matched plain decoding's. The old engine differs as often, so the change is not the cause; the unit tests (below) assert bit-exact output, and `CUCA_LOCAL_ALLOW_MISMATCH=1` makes the live test report such a run (`identical false`) and go on instead of aborting before its timing line.

### Verified

`tests/provider_local_mtp.rs` loads a target alone, records its greedy outputs (cold and warm, which must agree), drops it, and then loads the target once per draft length with its drafter and compares greedy output (one resident copy at a time, so a target that fills the machine's memory is never held twice). It asserts identical output and token counts on the CPU, and reports identity without asserting it on a GPU. Recorded CPU runs matched plain greedy decoding for the Gemma 4 E2B pair, the Qwen 3.5 in-trunk head, the Mistral Small 4 EAGLE drafter (draft lengths 1 to 4) and the Muse Glimmer DFlash drafter (3, 7 and 15), and recorded Vulkan runs of the Gemma 4 pair produced identical output.

Lazy upkeep is covered by unit tests: for each of the EAGLE drafter, the in-trunk head and the DFlash drafter an engine test forces the governor plain for 0, 1, 3, 10, 30 and 1,000 rounds and then drafts, and asserts greedy output equal to plain, the drafter holding the rows back while the rounds are plain, no round drafting before the drafter has caught up, the cache after catching up equal bit for bit to the cache of an engine that drafted all along (K/V of every committed slot; the whole context ring for DFlash), the same proposals and statistics as the same schedule with eager upkeep, and a following request that replays the cached ids from the rows still held; a drafter-level test checks chunking, the row and byte caps, truncation, replay at an earlier position, gaps and reset. Each was checked to fail when the catch-up positions, the replay trim, the truncation trim, the cap flush, the deferral or the catch-up before a draft is removed.
