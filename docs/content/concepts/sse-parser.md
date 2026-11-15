+++
title = "The SSE parser"
description = "How CUCA turns streamed bytes into events, why the parser returns SseEvent and not content blocks, and what its buffer can grow to."
template = "page.html"
weight = 2
+++

# The SSE parser

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SseStreamParser</code> cuts a byte stream into Server-Sent Events and knows nothing about what a vendor puts inside them</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You expected the parser to hand you content blocks, or you want to know how much memory it can use</dd>
</dl>

Most LLM services stream their answer as SSE, short for Server-Sent Events. SSE is a plain text format: the server keeps one HTTP response open and writes small messages into it as the model produces text. Each message is called a frame, and frames are separated by a blank line.

```text,name=Two SSE frames, as the server writes them
data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

```

`SseStreamParser` takes the bytes as they arrive from the network and returns the frames they completed. Its one public method:

```rust,name=src/sse.rs
pub fn feed_chunk(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, CucaError>
```

## From bytes to blocks

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 140" role="img" aria-labelledby="sse-title sse-desc">
  <title id="sse-title">From network bytes to content blocks</title>
  <desc id="sse-desc">Five stages, left to right. The network delivers bytes in chunks. SseStreamParser cuts them into frames and returns SseEvent values. A translator, one per wire shape, reads each event's JSON. The translator emits MessageContentBlock values. The parser and the SseEvent have no vendor knowledge. The translator reads the vendor's JSON.</desc>
  <defs>
    <marker id="sse-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Where the vendor-specific work starts</text>
  <rect class="blk blk-bnd" x="6" y="36" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="61" y="60">bytes</text>
  <text class="t-sm t-mid" x="61" y="76">network chunks</text>
  <rect class="blk blk-data" x="141" y="36" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-data" x="196" y="60">SseStream</text>
  <text class="t-sm t-mid" x="196" y="76">Parser</text>
  <rect class="blk blk-data" x="276" y="36" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-data" x="331" y="60">SseEvent</text>
  <text class="t-sm t-mid" x="331" y="76">event + data</text>
  <rect class="blk blk-bnd" x="411" y="36" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="466" y="60">translator</text>
  <text class="t-sm t-mid" x="466" y="76">per wire shape</text>
  <rect class="blk blk-data" x="546" y="36" width="108" height="56" rx="6"/>
  <text class="t-sm t-mid t-data" x="600" y="60">Message</text>
  <text class="t-sm t-mid" x="600" y="76">ContentBlock</text>
  <path class="arw arw-bnd" d="M118 64 H137" marker-end="url(#sse-arrow)"/>
  <path class="arw arw-data" d="M253 64 H272" marker-end="url(#sse-arrow)"/>
  <path class="arw arw-data" d="M388 64 H407" marker-end="url(#sse-arrow)"/>
  <path class="arw arw-data" d="M523 64 H542" marker-end="url(#sse-arrow)"/>
  <path class="ln" d="M141 106 H386"/>
  <path class="ln" d="M411 106 H654"/>
  <text class="t-ax" x="141" y="124">protocol only: no vendor knowledge</text>
  <text class="t-ax" x="411" y="124">reads the vendor's JSON</text>
</svg>
</div>
<p class="dgm-cap"><b>The parser stops at the event.</b> It never looks inside <code>data</code>. A translator that belongs to the provider adapter does that.</p>
</div>

## Why events and not blocks

The vendors disagree about what is inside a frame:

- Anthropic sends `content_block_delta` frames between an explicit start frame and a stop frame.
- OpenAI-compatible servers send `choices[0].delta`. A tool call's arguments arrive as pieces of a string that CUCA joins and then parses.
- Gemini sends `parts`, some flagged `"thought": true`, and never sends a `[DONE]` marker.

A parser that understood all three would hold three vocabularies, and a fourth backend would mean editing it. So the parser produces `SseEvent`, which uses the SSE format's own words:

```rust,name=src/sse.rs
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}
```

The translation sits in the adapter that already builds that vendor's request. A new backend changes nothing else.

The cost is one translator per wire shape. There are four:

| Translator | Serves |
|---|---|
| `ChatCompletionTranslator` | OpenAI, vLLM, LM Studio, DeepSeek's native route and llama.cpp's chat route |
| `AnthropicTranslator` | Anthropic and DeepSeek's bridge route |
| `GeminiTranslator` | Gemini |
| `CompletionTranslator` | llama.cpp's native `/completion` route |

The in-process `provider-local` engine produces blocks directly and does not use the parser.

## What the parser does with a chunk

Network chunks end anywhere, including in the middle of a line or a frame. The parser keeps the unfinished part and waits for more.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 184" role="img" aria-labelledby="chunk-title chunk-desc">
  <title id="chunk-title">A frame split across two network chunks</title>
  <desc id="chunk-desc">Chunk 1 holds the start of a data line, with no blank line yet. The parser keeps it in its buffer and emits nothing. Chunk 2 holds the rest of the line and the blank line that ends the frame. The parser then emits one SseEvent.</desc>
  <defs>
    <marker id="chunk-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Chunk boundaries do not matter</text>
  <rect class="blk blk-bnd" x="6" y="36" width="190" height="56" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="101" y="54">chunk 1</text>
  <text class="t-sm t-mid" x="101" y="72">data: {"text":"He</text>
  <rect class="blk blk-bnd" x="226" y="36" width="190" height="56" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="321" y="54">chunk 2</text>
  <text class="t-sm t-mid" x="321" y="68">llo"}</text>
  <text class="t-sm t-mid" x="321" y="82">then a blank line</text>
  <rect class="blk blk-data" x="6" y="116" width="410" height="48" rx="6"/>
  <text class="t-lbl t-mid t-data" x="211" y="136">SseStreamParser</text>
  <text class="t-sm t-mid" x="211" y="152">keeps the unfinished line until the blank line arrives</text>
  <path class="arw arw-bnd" d="M101 94 V112" marker-end="url(#chunk-arrow)"/>
  <path class="arw arw-bnd" d="M321 94 V112" marker-end="url(#chunk-arrow)"/>
  <rect class="blk blk-data" x="466" y="116" width="188" height="48" rx="6"/>
  <text class="t-lbl t-mid t-data" x="560" y="136">SseEvent</text>
  <text class="t-sm t-mid" x="560" y="152">data: {"text":"Hello"}</text>
  <path class="arw arw-data" d="M418 140 H462" marker-end="url(#chunk-arrow)"/>
  <text class="t-ax t-mid" x="560" y="178">emitted only after chunk 2</text>
</svg>
</div>
<p class="dgm-cap"><b>One frame can arrive in many chunks, and one chunk can hold many frames.</b> The parser returns every frame a chunk completed, and none for a chunk that completed no frame.</p>
</div>

The rules:

- A line is `field: value`. The colon is required, and one space after it is dropped. The fields are `event`, `data`, `id` and `retry`.
- A line that starts with a colon is a comment. Unknown fields are ignored.
- A blank line ends the frame. Several `data` lines in one frame are joined with newlines.
- Line endings can be `\n` or `\r\n`.
- A frame with no `event` field gets the event name `message`.
- A frame without its own `id` inherits the last `id` the stream sent. An `id` that is empty or holds a NUL byte does not update it.
- A malformed `retry` becomes `None`. It never fails the stream.

## What its memory depends on

The parser keeps one reusable buffer. Its size does not depend on how many chunks arrive or how long the stream is. A finished frame is removed from the buffer, so the largest it gets is the biggest single frame plus one unfinished line. It starts at 8192 bytes, and `capacity()` reports the current size.

There is no maximum. A server that opens a frame and never sends the blank line makes the buffer grow without a limit, because the frame cannot be handed over until it ends. That is how the SSE format works.

Cutting the frame short would give the adapter half a JSON body. Failing at a size limit would break a server that sends a large frame on purpose. This is the one growing structure whose limit the other side sets, so [Memory discipline](@/concepts/memory-discipline.md) lists it as uncapped.

## The one error

`feed_chunk` fails in one case: a finished frame whose `data`, `event` or `id` bytes are not valid UTF-8. The error is `CucaError::SseParse`. An unfinished line is never checked, so a character split across two chunks is not an error.

Next page: [Plugins and services](@/concepts/plugin-layering.md).
