+++
title = "How local inference works"
description = "The words a newcomer needs, one request step by step, how several requests share the engine, and how a repeated prompt is reused instead of recomputed."
template = "page.html"
weight = 1
+++

# How local inference works

<dl class="page-facts">
<dt>In one line</dt>
<dd>A request becomes tokens, waits in a queue, runs in shared steps on one engine thread, and streams its reply back; a prompt start the engine has seen before is not computed twice.</dd>
<dt>You need</dt>
<dd>Nothing running. This page explains and has no code to run.</dd>
<dt>Read this if</dt>
<dd>You are new to running a model inside your own program and want the words and the picture before you configure anything.</dd>
</dl>

## Words you will see

Every local page uses these words the same way.

- A *GGUF* file is one model in one file. It holds the weights, the tokenizer and the chat template. A *safetensors checkpoint* is the model as Hugging Face publishes it, as `config.json`, one or more `.safetensors` files and the tokenizer files. CUCA converts a checkpoint once, in process, to a cached GGUF.
- *Weights* are the numbers the model learned. A bigger model has more of them and needs more memory.
- *Quantization* stores each weight in fewer bits. Q8_0 uses about 8 bits per weight and Q4_0 about 4. A smaller file needs less disk and memory, at a small cost in quality.
- A *token* is a piece of text: a word, part of a word or a mark. The model reads and writes tokens, each one an integer id. The *tokenizer* turns text into ids and ids back into text.
- A *chat template* is a short Jinja program stored in the GGUF, or in the checkpoint before conversion. It turns your messages (system, user, assistant, tool results) into the one prompt text the model was trained on, with the model's own markers.
- The *context length* is the most tokens one request can hold, prompt and reply together. `context_length` defaults to 8,192 or the model's own maximum, whichever is smaller.
- *Prefill* reads the prompt: its tokens go through the model many at a time. *Decode* writes the reply one token per step. Prefill decides how long a long prompt takes to start answering. Decode decides how fast the reply streams.
- A *forward pass* is one run of a batch of tokens through all the layers of the model. The engine works in steps, and each step is one forward pass.
- The *K/V cache* keeps what the model computed for the tokens it has already read, so it never reads them again. The engine stores it in *blocks* of 16 positions, in one pool that every request shares. `LocalModel::kv_cache_bytes` reports the pool size.
- A *mixture of experts* model has many small expert networks in each layer, and each token uses only a few. Every expert is in memory by default, but a step computes only the experts its tokens pick.
- *Thinking* is reasoning that some models write before the answer. The stream carries it as `Thinking` content blocks, next to `Text` and `ToolCall` blocks.

## One request, step by step

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 140" role="img" aria-labelledby="hiw-req-title hiw-req-desc">
  <title id="hiw-req-title">One request from your code to the reply stream</title>
  <desc id="hiw-req-desc">Six stages, left to right. Your request goes through the chat template, which writes the prompt text, and the tokenizer, which turns it into token ids. The job waits in a queue of up to 16 requests. The engine thread runs up to max_sequences requests together in batched steps. Text, Thinking and ToolCall blocks come back over a channel that holds 64 blocks per request. A full queue makes the caller wait, and dropping the stream stops the request.</desc>
  <defs>
    <marker id="hiw-req-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One request, from your code to the stream</text>
  <rect class="blk" x="6" y="40" width="92" height="54" rx="6"/>
  <text class="t-sm t-mid" x="52" y="63">your request</text>
  <text class="t-sm t-mid" x="52" y="79">messages</text>
  <rect class="blk" x="118" y="40" width="92" height="54" rx="6"/>
  <text class="t-sm t-mid" x="164" y="63">chat template</text>
  <text class="t-sm t-mid" x="164" y="79">text prompt</text>
  <rect class="blk" x="230" y="40" width="92" height="54" rx="6"/>
  <text class="t-sm t-mid" x="276" y="63">tokenizer</text>
  <text class="t-sm t-mid" x="276" y="79">text to ids</text>
  <rect class="blk" x="342" y="40" width="92" height="54" rx="6"/>
  <text class="t-sm t-mid" x="388" y="63">job queue</text>
  <text class="t-sm t-mid" x="388" y="79">16 waiting</text>
  <rect class="blk blk-ctl" x="454" y="40" width="92" height="54" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="500" y="63">engine thread</text>
  <text class="t-sm t-mid" x="500" y="79">batched steps</text>
  <rect class="blk blk-data" x="566" y="40" width="92" height="54" rx="6"/>
  <text class="t-sm t-mid t-data" x="612" y="63">block channel</text>
  <text class="t-sm t-mid" x="612" y="79">64 blocks</text>
  <path class="arw" d="M100 67 H116" marker-end="url(#hiw-req-arrow)"/>
  <path class="arw" d="M212 67 H228" marker-end="url(#hiw-req-arrow)"/>
  <path class="arw" d="M324 67 H340" marker-end="url(#hiw-req-arrow)"/>
  <path class="arw" d="M436 67 H452" marker-end="url(#hiw-req-arrow)"/>
  <path class="arw arw-data" d="M548 67 H564" marker-end="url(#hiw-req-arrow)"/>
  <text class="t-ax" x="6" y="116">a full queue makes the caller wait</text>
  <text class="t-ax" x="6" y="132">dropping the stream stops the request</text>
  <text class="t-ax t-end" x="658" y="116">Text, Thinking, ToolCall</text>
</svg>
</div>
<p class="dgm-cap"><b>The caller prepares the prompt, and one engine thread does the model work.</b> The template and the tokenizer run in the caller's task. The thread runs every admitted request together and sends content blocks back.</p>
</div>

1. The chat template turns the messages, the tools and the thinking setting into one prompt text.
2. The tokenizer turns that text into token ids. A prompt with `context_length` tokens or more fails here: `generate_stream` returns `CucaError::Provider`.
3. The request joins the job queue. Up to 16 requests wait. When the queue is full, `generate_stream` waits for room and returns no error.
4. The engine thread admits waiting requests in arrival order while fewer than `max_sequences` run and the K/V pool has room for their prompts.
5. Each step reads prompt tokens (prefill) or writes one token per request (decode). The model scores every possible next token and the sampler picks one. The output parser sorts the text into `Text`, `Thinking` and `ToolCall` blocks.
6. The blocks go to the caller over a channel that holds 64 blocks per request. A reader that falls behind stalls only its own request.
7. The reply ends at the end-of-reply token, a `stop` string, `max_tokens` or the context limit. Token usage in the response is exact.

Dropping the stream stops the request at the next step and keeps its work in the cache. The request's `model` field is not used: one client holds one model.

## Many requests at once

The engine does not finish one request before it starts the next. Each step is one forward pass over rows from several requests: one row for every request that is writing its reply, and chunks of the prompts that are still being read. The weights are read once per step, not once per request, so total throughput rises as more requests run together.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 196" role="img" aria-labelledby="hiw-batch-title hiw-batch-desc">
  <title id="hiw-batch-title">One engine step mixing decode rows and a prefill chunk</title>
  <desc id="hiw-batch-desc">Three requests A, B and C are writing their reply, so each puts one decode row into the step. A fourth request D is still reading its prompt, so it adds a chunk of its prompt with the rows that are left in the step. All rows go through one forward pass, which reads the weights once. Requests A, B and C then get their next token, and D has read its prompt a chunk further.</desc>
  <defs>
    <marker id="hiw-batch-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One step: one forward pass over several requests</text>
  <text class="t-ax" x="6" y="40">decode rows</text>
  <text class="t-ax" x="176" y="40">prefill chunk</text>
  <rect class="blk blk-data" x="6" y="48" width="48" height="44" rx="6"/>
  <text class="t-lbl t-mid t-data" x="30" y="66">A</text>
  <text class="t-sm t-mid" x="30" y="82">1 row</text>
  <rect class="blk blk-data" x="60" y="48" width="48" height="44" rx="6"/>
  <text class="t-lbl t-mid t-data" x="84" y="66">B</text>
  <text class="t-sm t-mid" x="84" y="82">1 row</text>
  <rect class="blk blk-data" x="114" y="48" width="48" height="44" rx="6"/>
  <text class="t-lbl t-mid t-data" x="138" y="66">C</text>
  <text class="t-sm t-mid" x="138" y="82">1 row</text>
  <rect class="blk blk-data" x="176" y="48" width="244" height="44" rx="6"/>
  <text class="t-lbl t-mid t-data" x="298" y="66">D</text>
  <text class="t-sm t-mid" x="298" y="82">a chunk of its prompt</text>
  <path class="arw" d="M84 94 V122" marker-end="url(#hiw-batch-arrow)"/>
  <path class="arw" d="M298 94 V122" marker-end="url(#hiw-batch-arrow)"/>
  <rect class="blk blk-ctl" x="6" y="124" width="414" height="40" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="213" y="141">one forward pass</text>
  <text class="t-sm t-mid" x="213" y="156">the weights are read once for all rows</text>
  <path class="arw arw-data" d="M422 144 H452" marker-end="url(#hiw-batch-arrow)"/>
  <rect class="blk blk-data" x="454" y="124" width="200" height="40" rx="6"/>
  <text class="t-sm t-mid" x="554" y="141">A, B, C: next token each</text>
  <text class="t-sm t-mid" x="554" y="156">D: prompt read further</text>
  <text class="t-ax" x="6" y="186">decode rows come first; prompt chunks take the rows that are left</text>
</svg>
</div>
<p class="dgm-cap"><b>A new request joins at the next step.</b> It does not wait for the others to finish, and a long prompt is read in chunks so the other replies keep streaming.</p>
</div>

- `max_sequences` requests run at once: 4 by default, 1 to 64. The rest wait in the queue.
- A step has at most `max_step_tokens` rows. The default is the prefill chunk: 256 rows at 8,192 positions of context.
- The K/V pool does not grow with `max_sequences`. When it is full, the engine evicts cached blocks that no request uses, least recently used first. Then it preempts the youngest running request, which goes back to the front of the queue and reads its prompt again, mostly from the blocks it left in the cache. The oldest running request is never preempted.
- A drafter that reads every position of its sequence (see [Speculative decoding](@/providers/local/drafting.md)) limits the engine to one request at a time.

The `batching` keys are in [Configuration](@/providers/local/configuration.md). The caps and what happens at each are in [Limits and errors](@/providers/local/limits-and-errors.md).

## Reusing a prompt

Chat programs send the same start again and again: the system prompt, the tools and the earlier turns. The engine keeps the K/V blocks of finished requests. It names each block by the 16 token ids it holds and by the blocks before it, so only a shared start of a prompt matches.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 196" role="img" aria-labelledby="hiw-cache-title hiw-cache-desc">
  <title id="hiw-cache-title">Two prompts that start the same share K/V blocks</title>
  <desc id="hiw-cache-desc">Two rows of four boxes. The first request has a prompt of 69 tokens: three blocks of 16 tokens that the engine stores, then 21 more tokens, of which one block is stored and 5 are left over in a partial block. The second request has 71 tokens and starts with the same three blocks, which are read from the cache with arrows from the first row. Its last 23 tokens differ, so the engine computes them. The second request reads 48 tokens and computes 23.</desc>
  <defs>
    <marker id="hiw-cache-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Two prompts that start the same share blocks</text>
  <text class="t-lbl" x="6" y="70">first request</text>
  <rect class="blk blk-data" x="120" y="44" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="166" y="63">16 tokens</text>
  <text class="t-sm t-mid" x="166" y="79">stored</text>
  <rect class="blk blk-data" x="220" y="44" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="266" y="63">16 tokens</text>
  <text class="t-sm t-mid" x="266" y="79">stored</text>
  <rect class="blk blk-data" x="320" y="44" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="366" y="63">16 tokens</text>
  <text class="t-sm t-mid" x="366" y="79">stored</text>
  <rect class="blk" x="420" y="44" width="192" height="44" rx="6"/>
  <text class="t-sm t-mid" x="516" y="63">21 more tokens</text>
  <text class="t-sm t-mid" x="516" y="79">1 block stored, 5 not</text>
  <path class="arw arw-data" d="M166 90 V122" marker-end="url(#hiw-cache-arrow)"/>
  <path class="arw arw-data" d="M266 90 V122" marker-end="url(#hiw-cache-arrow)"/>
  <path class="arw arw-data" d="M366 90 V122" marker-end="url(#hiw-cache-arrow)"/>
  <text class="t-ax t-end" x="160" y="110">reused, not recomputed</text>
  <text class="t-lbl" x="6" y="150">second request</text>
  <rect class="blk blk-data" x="120" y="124" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="166" y="143">16 tokens</text>
  <text class="t-sm t-mid" x="166" y="159">from cache</text>
  <rect class="blk blk-data" x="220" y="124" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="266" y="143">16 tokens</text>
  <text class="t-sm t-mid" x="266" y="159">from cache</text>
  <rect class="blk blk-data" x="320" y="124" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="366" y="143">16 tokens</text>
  <text class="t-sm t-mid" x="366" y="159">from cache</text>
  <rect class="blk" x="420" y="124" width="192" height="44" rx="6"/>
  <text class="t-sm t-mid" x="516" y="143">23 more tokens</text>
  <text class="t-sm t-mid" x="516" y="159">computed</text>
  <text class="t-ax" x="6" y="188">second request: read_tokens 48, write_tokens 23</text>
</svg>
</div>
<p class="dgm-cap"><b>A prompt reuses the whole blocks it shares from its start.</b> The fourth block of the second prompt differs, so it and everything after it are computed.</p>
</div>

- `prompt_cache_usage.read_tokens` counts the prompt tokens read from the cache and `write_tokens` the ones computed. Read it per call with `CucaClient::generate_stream_with_response`. `read_tokens` is always a multiple of 16.
- The last prompt token and the trailing partial block (at most 15 tokens) are always computed.
- A request that arrives while another with the same prompt runs reuses the blocks as soon as they are filled.
- When the pool is full, the least recently used cached block goes first, so an unrelated request does not evict a prefix that fits.
- On the CPU a request that reuses a prefix generates exactly what a cold request generates.
- A hybrid model such as Qwen 3.5 also keeps a saved state at the block boundary before the last end-of-turn token, so its `read_tokens` stops there at most; see [Choose a model](@/providers/local/models.md).
- `batching prefix_cache=#false` in a KDL file turns the cache off, and every prompt is then read in full.

The request side is in [Requests, thinking and tool calls](@/providers/local/requests.md).

Next page: [Choose a model](@/providers/local/models.md).
