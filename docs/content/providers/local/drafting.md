+++
title = "Speculative decoding with a drafter"
description = "DrafterSource, the four drafters, the draft and accept rounds, the governor that turns drafting off, measured speedups and what was verified."
template = "page.html"
weight = 10
+++

# Speculative decoding with a drafter

<dl class="page-facts">
<dt>In one line</dt>
<dd>A multi-token-prediction drafter proposes tokens that the target verifies in one batched pass; a governor drafts only while that is faster.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, and a drafter preset, a checkpoint source or an in-trunk prediction head.</dd>
<dt>Read this if</dt>
<dd>You want faster decoding on a model that has a drafter, or need the acceptance rule, the pairing errors and the measured speedups.</dd>
</dl>

## Drafters

A drafter proposes tokens that the target model verifies in one batched forward pass, and the output is the target's. Drafting is off by default. `LocalModelConfig::drafter` is an `Option<DrafterSource>`. `DrafterSource::File(ModelSource)` is a separate drafter file, resolved and downloaded like any model source (`ModelSource::into()` converts). `DrafterSource::InTrunk` is the head inside the main model's own GGUF. The engine picks the drafter kind from the `general.architecture` of the drafter file.

| Drafter | Source | Target | Device | K/V cache of its own | Requests |
|---|---|---|---|---|---|
| Gemma 4 assistant (`gemma4-assistant`) | `File`, preset `gemma4-e2b-mtp-f16` (170 MB) | a `gemma4` model of the same hidden width and vocabulary | the target's: CPU or GPU | none, it reads the target's | batched; it speculates only while one request runs alone |
| Qwen 3.5 head | `InTrunk` | a dense `qwen35` `*-MTP` file | CPU | 4 KB per context position (0.8B model) | one at a time |
| EAGLE (`mistral4-eagle`) | `File` with a checkpoint source | `mistral4` (Mistral Small 4) | CPU | 2.5 KB per context position | one at a time |
| DFlash (`dflash`) | `File`: preset `muse-glimmer-30b-dflash-q4_0` or a checkpoint source | Muse Glimmer | CPU | a ring of the last 2,048 positions, 82 MB | one at a time |

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

### Gemma 4 assistant

A four-layer decoder of width 256 that reads the target's K/V cache, so it adds no cache memory. On the CPU the loader stores its float matrices as Q8_0, because a drafter only proposes and its rounding cannot change the output. A target that is not `gemma4`, or a `gemma4` target of another hidden width or vocabulary, fails the load with `CucaError::Provider`:

- `` the `gemma4-assistant` drafter pairs with a `gemma4` model, not `<arch>` ``
- `drafter hidden width N differs from the target's M: it was trained for another model size`
- `drafter vocabulary N differs from the target's M`

### Qwen 3.5 head

`DrafterSource::InTrunk` uses the prediction head inside a Qwen 3.5 `*-MTP` file such as `unsloth/Qwen3.5-0.8B-MTP-GGUF` (preset `qwen3.5-0.8b-q8_0`). Nothing more is downloaded. The head is one gated attention block with a K/V cache of its own (4 KB per context position and 21.7 MB of weights for the 0.8B model). A draft step ranks only the first 65,536 token ids, which is about five times cheaper than ranking all 248,320 ids of Qwen 3.5 0.8B for 4 percent fewer accepted tokens per round. It fails for:

- a GPU device: `CucaError::Config`, `DrafterSource::InTrunk runs on the CPU only, not on <device>`;
- a file without a head: `CucaError::Provider`, ``this `<arch>` file has no in-trunk multi-token-prediction head (...); use a separate drafter file instead``;
- a `qwen35moe` file (Qwen3.6-35B-A3B), whose prediction block has a mixture of experts FFN: `CucaError::Config`, ``the in-trunk multi-token-prediction head of `qwen35moe` has a mixture-of-experts FFN, which the head does not support; run without a drafter``.

### EAGLE for Mistral Small 4

`mistralai/Mistral-Small-4-119B-2603-eagle` (392 MB) is published in Mistral's own format: `params.json` and an FP8 `consolidated.safetensors`, no `config.json`. It is a `DrafterSource::File` with a checkpoint source, converted once, in process, to a GGUF of architecture `mistral4-eagle` at `convert.quant` (Q8_0 by default, 0.4 GB). The converter multiplies the FP8 weights by their `qscale_weight` and drops the activation scales.

```rust,name=The EAGLE drafter source
// Inside LocalModelConfig { drafter: Some(...), ..Default::default() }
"mistralai/Mistral-Small-4-119B-2603-eagle:9a8ea22dca0161ff7af7879f4a0000f65314797d"
    .parse::<ModelSource>()?
    .into()
```

- Two latent attention layers over a dense FFN of width 12,288. Its input is `fc(cat(embed(t[j+1]), h[j]))`: the target's token embedding first, the target's final-norm hidden row second. Its embedding and LM head are the target's, and a draft step ranks the whole vocabulary.
- It keeps a K/V cache of its own (2.5 KB per context position) and observes every position, so the engine runs one request at a time with it.
- It pairs with a `mistral4` model on the CPU only. On a GPU device the load fails with `CucaError::Config`, `the EAGLE drafter runs on the CPU only, not on <device>`.
- It refuses a drafter whose hidden width, vocabulary, norm epsilon, activation or residual scale, RoPE, query scaling or attention geometry differs from its target's: `the EAGLE drafter's <what> is X but the model's is Y: it was trained for another model`.

### DFlash for Muse Glimmer

A DFlash drafter proposes a whole block of tokens in one pass instead of one pass per token. `DrafterSource::File` takes the GGUF that llama.cpp's converter writes (`dflash-Muse-Glimmer-30B-Q4_0.gguf` in `ggml-org/Muse-Glimmer-30B-GGUF`, the preset `muse-glimmer-30b-dflash-q4_0`) or the safetensors checkpoint `meta-models/Muse-Glimmer-30B-assistant:c676cce91658f9568aaf264a416211e2782735b7` as a `ModelSource::Checkpoint`, converted at `LocalModelConfig::convert.quant` (about 1.4 GB at `q4_0`, 2.7 GB at the default `q8_0`). The converter writes the tensors and keys llama.cpp writes, without the target's vocabulary.

- It reads the residual stream after five target layers (the outputs of layers 1, 13, 25, 37 and 49 of 52; the file stores llama.cpp's `target_layers` `[2, 14, 26, 38, 50]`, the inputs of the next layer) at every accepted position, and keeps its own K/V of them in a ring of the last 2,048 positions, 82 MB whatever the context length.
- A round runs five sliding-window layers once over a block of `1 + draft_tokens` rows: the pending token and mask tokens, embedded with the target's raw table. The rows attend to each other in both directions and to the context. A block holds at most the model's block size, so Muse Glimmer takes at most 15 drafts (16 rows).
- The rows are ranked with the target's LM head, logit multiplier and soft-cap included. Muse Glimmer's head is Q6_K (1.1 GB, 202,048 rows), which candle's kernel reads once per row, so the drafter ranks through a Q8_0 copy of the head (1.4 GB, built when the drafter opens) that the row-batched kernel reads once for all rows. The copy only changes which tokens are proposed.
- It is CPU only and observes every position of one sequence, so the engine runs one request at a time. The context cache follows the trunk's prefix reuse, truncation and failed requests; a gap in it costs proposal quality, never output.
- It fails to load when its width differs from the target's, when it reads a layer the target does not have, when its mask token is outside the target's vocabulary, or on a GPU device (`CucaError::Provider`, `` the DFlash drafter runs on the CPU only: use `LocalDevice::Cpu` ``). Muse Glimmer 30B is the one target that has been run with a DFlash drafter.

## Rounds and acceptance

- Each round the drafter proposes at most `draft_tokens` tokens (default 5, `1..=16`), fewer near `max_tokens` and the context limit. `draft_tokens` is ignored without a drafter and is only the cap: the rule below sets the length per round.
- A round always verifies its first draft. It adds draft `j`, counting from 1, only while that raises the round's expected tokens per second:

```text,name=Keep rule for draft j of a chained drafter
a_j * (t_j + (j - 1) * d)  >  e * (t_(j+1) - t_j + d)
```

- `a_j` is the predicted acceptance of draft `j`, and `e` the expected tokens of the round without it (`1 + a_1 + ... + a_(j-1)`). `t_j` and `t_(j+1)` are the measured seconds of a verification pass over `j` and `j + 1` rows, rollback and drafter upkeep included, and `d` is the seconds of one drafter step. Before any measurement every row count costs the same, so the first rounds draft the full cap.
- For a chained drafter (the Gemma 4 assistant, the Qwen 3.5 head, EAGLE), `a_j` is the product of the drafter's own probabilities of its proposals so far, times the ratio of accepted to predicted drafts the engine observed, clamped to 0.25 to 4.
- A block drafter (DFlash) costs one pass of `f` seconds per round however many drafts are kept, so the rule is `a_j * (t_j + f) > e * (t_(j+1) - t_j)`. Its own confidence is flat on the mask rows, so `a_j` is `rho^j`, with `rho` the share of judged drafts that were accepted over about the last 20 rounds (an accepted draft or the first rejected one of a round is judged). `rho` is 1 before the first round.
- At temperature `0.0` a draft is accepted when it equals the target's greedy token. On the CPU the output equals plain decoding bit for bit. On a GPU, batched rows equal single steps only to float rounding, so a speculative greedy run can differ from plain decoding where two logits are that close.
- Above `0.0` a draft is accepted with probability `min(1, p/q)`, where `p` and `q` are the target's and the drafter's distributions after temperature, `top_k` and `top_p`. A rejected draft is replaced by a token drawn from the residual `max(0, p - q)`, which preserves the target's sampling distribution.
- A run with a fixed `seed` and a temperature above `0.0` always drafts exactly `draft_tokens`, so equal prompts reproduce equal tokens: neither the keep rule nor the governor applies.
- A drafter speculates only while one request runs alone and none waits. With several running requests the engine batches their decode rows instead, which already shares the weight reads that speculation saves. The in-trunk head, EAGLE and DFlash observe every position of their one sequence, so with them `batching.max_sequences` is treated as 1.
- `LocalModel::draft_stats()` returns `DraftStats { rounds, drafted, accepted }`, summed over every request. `rounds` counts verification rounds that carried drafts. `DraftStats::acceptance_rate()` returns the accepted fraction, `None` before the first drafted token.

## Governor

An engine side governor drafts only while drafting is measurably faster than plain decoding. Its state is kept across requests.

- It measures plain decoding first (the first round is a discarded warm-up, then two rounds), then speculative rounds, as moving averages of tokens per second with weight 0.3 on the newest round.
- It keeps drafting while speculation is at least 1.02 times as fast as plain decoding, and refreshes the plain estimate with two plain rounds every 64 rounds.
- While plain decoding is faster, it probes with two speculative rounds after 32 plain rounds. Each failed probe doubles the interval, up to 512 rounds.
- While fewer than two speculative rounds are recorded, a round more than 6 times slower than plain decoding counts as a kernel compile (GPU autotune), not a measurement, and is left out of the estimate.

## Lazy upkeep

A drafter that keeps a cache of its own (EAGLE, DFlash, the Qwen 3.5 head) does not pay for it in plain rounds. A round that drafts nothing only hands its hidden rows (the features, for DFlash) to the drafter, which holds them with the range of positions it has not seen, at most 1,024 positions and 16 MiB. When a round drafts again, because of a probe or because drafting resumes, the drafter first takes the held rows into its cache in one batched pass, in chunks of the trunk's prefill size. At the bound it catches up first.

The CPU kernels are row exact, so the cache after catching up equals the one that per round upkeep leaves, bit for bit: proposals and output do not change, only where the cost falls. The catch up is a cost of switching and the governor charges it to neither mode. A new prompt drops the held rows it does not share, and a failed request forgets them.

Per token, per round upkeep would cost 4.4 ms of an 86.6 ms plain round for EAGLE, 3.5 ms of 288 ms for DFlash and 0.09 ms of 17.3 ms for the Qwen 3.5 head. Catching up costs 0.89, 1.2 and 0.05 ms per row. The Gemma 4 assistant reads the target's K/V and has no upkeep.

## Measured speedups

Speedup is speculative tokens per second over plain decoding of the same model on the same device. Method: greedy, 128-token replies, the three prompts of `tests/provider_local_mtp.rs` (a Python function, a list of the planets, a short story), release build with `-C target-cpu=native`, on the development machine (Ryzen 9 9950X3D2). Greedy output equaled plain decoding in every run.

| Target and drafter | `draft_tokens` | code | list | story | all three |
|---|---|---|---|---|---|
| Gemma 4 E2B Q4_0, Gemma 4 assistant | 5 | 1.32x | 1.31x | 1.01x | not recorded |
| Qwen 3.5 0.8B Q8_0, in-trunk head (plain 55 tokens/s) | 5 | 1.59x | 1.40x | 0.98x | 1.26x |
| the same | 3 | 1.86x | 1.37x | 0.99x | 1.32x |
| Mistral Small 4 Q4_0, EAGLE (plain 11.2 to 11.3 tokens/s) | 2 | 0.99x to 1.03x | 0.99x to 1.00x | 0.98x to 1.00x | 0.99x to 1.00x |
| Muse Glimmer 30B Q4_0, DFlash (plain 3.4 tokens/s) | 3 | 1.76x | 1.69x | 1.13x | 1.47x |
| the same | 7 | 1.64x | 1.66x | 1.20x | 1.47x |
| the same | 15 | 1.18x | 1.14x | 1.15x | 1.15x |

The EAGLE row is three runs of the live test on one machine state. The DFlash rows at 7 and 15 drafts were not repeated after lazy upkeep, which moves about 1 percent of a DFlash token. On Vulkan (RX 9070 XT) the Gemma 4 pair ran with a speedup of 0.97x to 1.00x on the same prompts; that was not measured again with the fused dense kernels. The story prompt gains least because its drafts are accepted less often.

### Mistral Small 4 and EAGLE

The drafter does not pay on this CPU. Measured per round with the round trace at 2 drafts:

- A verification pass takes 90 ms over one row, 153 ms over two and 201 ms over three. The model is a mixture of experts with 4 of 128 experts per token, and every row routes to its own experts, so extra rows share no expert weights: each costs 48 to 63 ms, 53 to 70 percent of a decode step. A dense model's extra row costs a few percent.
- A draft pass costs 15 ms (two layers and the vocabulary ranking). By the cost model a first extra draft pays only above an acceptance of about 0.74 (55 ms for its row plus the 15 ms pass against a 95 ms token).
- Rounds that keep two drafts average 2.85 tokens (92 percent of the drafts accepted) in 237 ms, which is 83 ms a token against 90 ms plain. Rounds that keep one draft are slower than plain (186 ms for 1.77 tokens). Drafting rounds gain about 2 percent over plain, under the governor's 1.02 margin once a rejected round enters its average, so the engine drafts 86 to 90 tokens per run of three prompts.

### Muse Glimmer and DFlash

- Accepted drafts: 68 to 70 percent at 3 and 7 drafts, 64 percent at 15. Tokens per drafting round: 3.0 to 3.2 at 3 and 7 drafts, 3.9 at 15.
- Median cost per round at 3 drafts: a plain token 287 ms; a verification pass over 2, 4, 8 and 16 rows 352, 457, 853 and 1,667 ms (65 ms an extra row for the first rows, 80 to 92 ms for 8 to 16); the draft pass 63 ms; the verification 456 ms. A round is 525 ms for 3.1 tokens, 170 ms a token, 1.7 times a plain token.
- At 15 drafts the block pass costs 206 ms and a 16-row verification 1.7 s, so the rule keeps 4.6 drafts on average and the run reaches 1.15x.
- Published figures, not reproduced here: average accepted length 4.35 to 7.84 on Qwen3-4B at blocks of 16 (DFlash paper, Table 1), 2.4 to 5.0 for `RedHatAI/Mistral-Small-4-119B-2603.dflash` at seven drafts, and for Muse Glimmer 30B 3.1x on an RTX 5090 with llama.cpp and 1.5x and 1.8x on Apple M4 and M5 Max (the assistant's model card).

## Verified

`tests/provider_local_mtp.rs` loads the target alone and records its greedy outputs, cold and warm, which must agree. It drops that model, waits 5 seconds, then loads the target once per draft length with its drafter and compares greedy output, so a target that fills the machine's memory is never held twice. On the CPU it asserts identical output and token counts and that the drafter ran. On a GPU it reports `identical` without asserting it. It uses `context_length` 2,048 and one device.

- `CUCA_LOCAL_MODEL_PATH`, `CUCA_LOCAL_MODEL` or `CUCA_LOCAL_DOWNLOAD=1` names the target (the default preset `gemma4-e2b-q4_0` needs `CUCA_LOCAL_DOWNLOAD=1`).
- `CUCA_LOCAL_DRAFTER` is a preset, a checkpoint source `repo:revision` or `in-trunk`; `CUCA_LOCAL_DRAFTER_PATH` is a path.
- `CUCA_LOCAL_DRAFT_TOKENS` is one `draft_tokens` value or a comma separated list, one comparison each (default 3).
- `CUCA_LOCAL_DEVICE` is one device, `CUCA_LOCAL_CONVERT_QUANT` the precision of a converted checkpoint, `CUCA_REQUIRE_LIVE=1` turns a skip into a failure.
- `CUCA_LOCAL_ALLOW_MISMATCH`, set to any value, makes the CPU run report `identical false` and go on instead of failing.

```bash,name=Linux and macOS: the DFlash check
export RUSTFLAGS="-C target-cpu=native"
export CUCA_LOCAL_MODEL="muse-glimmer-30b-q4_0"
export CUCA_LOCAL_DRAFTER="muse-glimmer-30b-dflash-q4_0"
export CUCA_LOCAL_DRAFT_TOKENS="3,7,15"
cargo test --release --features provider-local --test provider_local_mtp -- --nocapture
```

```powershell,name=Windows (PowerShell): the DFlash check
$env:RUSTFLAGS = "-C target-cpu=native"
$env:CUCA_LOCAL_MODEL = "muse-glimmer-30b-q4_0"
$env:CUCA_LOCAL_DRAFTER = "muse-glimmer-30b-dflash-q4_0"
$env:CUCA_LOCAL_DRAFT_TOKENS = "3,7,15"
cargo test --release --features provider-local --test provider_local_mtp -- --nocapture
```

```text,name=Output format, one line per prompt and one per draft length
k <draft_tokens> "<first 30 characters of the prompt>": plain <n> tok/s, speculative <n> tok/s (<n>x), identical true, drafted <n> accepted <n>
<tokens> tokens: plain <n> tok/s, speculative <n> tok/s (<n>x); drafted <n> accepted <n> (acceptance <n>%) over <n> rounds, draft_tokens <draft_tokens>
```

Recorded CPU runs matched plain greedy decoding for the Gemma 4 pair, the Qwen 3.5 in-trunk head, the EAGLE drafter at draft lengths 1 to 4 and the DFlash drafter at 3, 7 and 15. Recorded Vulkan runs of the Gemma 4 pair produced identical output.

Unit tests on tiny models, on the CPU unless noted:

- For each drafter (`drafter_leaves_greedy_output_unchanged`, `in_trunk_head_leaves_greedy_output_unchanged`, `eagle_leaves_greedy_output_unchanged`, `dflash_drafter_leaves_greedy_output_unchanged`): greedy output equals plain decoding. Matching tests cover `max_tokens`, the context limit and reproducible sampling for a seed. The in-trunk head, EAGLE and DFlash also have tests for a warm prefix equal to a cold one and for recovery from a failed request.
- `plain_rounds_defer_the_upkeep_and_the_next_draft_catches_up` covers lazy upkeep for EAGLE, the Qwen head and DFlash. It forces the governor plain for 0, 1, 3, 10, 30 and 1,000 rounds and then drafts. It asserts greedy output equal to plain, the rows held back while the rounds are plain, no round drafting before the drafter has caught up, a cache after catching up equal bit for bit to the cache of an engine that drafted all along (the K/V of every committed slot; the whole context ring for DFlash), the same proposals and statistics as the same schedule with eager upkeep, and a following request that replays the cached ids from the rows still held.
- `eagle_deferred_rows_catch_up_to_the_eager_cache`, `the_lag_holds_contiguous_rows_and_trims_from_a_position` and `the_lag_is_bounded_by_rows_and_by_bytes` check at the drafter level that rows handed over round by round and taken in one pass leave the cache that observing them as they came leaves, whatever the chunking and the caps, and that held rows follow truncation, replay at an earlier position and reset.
- `gpu_drafter_matches_cpu_drafter` compares the GPU Gemma 4 assistant with the CPU one when `CUCA_LOCAL_DEVICE` names a GPU.
- `dflash_drafter_matches_the_transformers_reference` runs when `CUCA_LOCAL_DFLASH_REF` names a directory of a tiny checkpoint pair and reference rounds (written by `make_tiny_dflash.py`, which is not in the repository). It bounds `max |dlogit|` by `CUCA_LOCAL_REF_TOL` (default 5e-4) and requires equal drafts; the recorded run measured below 2e-4.
