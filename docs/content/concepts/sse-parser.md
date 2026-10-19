+++
title = "The SSE parser"
description = "Why feed_chunk returns Vec<SseEvent> rather than content blocks, and what the single reusable buffer bounds."
template = "page.html"
weight = 2
+++

# The SSE parser

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SseStreamParser</code> does protocol work only: it frames Server-Sent Events and knows nothing about what a vendor puts inside them</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You expected the parser to hand you content blocks, or you want to know what its memory depends on</dd>
</dl>

`SseStreamParser` turns bytes into SSE frames. Its one public method takes a chunk of bytes and returns the frames that chunk completed.

```rust,name=src/sse.rs
pub fn feed_chunk(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, CucaError>
```

The return type is `SseEvent`, not `MessageContentBlock`. That boundary is the most consequential one in the crate.

## Why events and not blocks

A parser that returned `MessageContentBlock` would have to decide what each vendor's JSON means, and the vendors disagree.

- Anthropic sends `content_block_delta` frames between an explicit `content_block_start` and `content_block_stop`.
- OpenAI-compatible servers send `choices[0].delta`, with tool-call arguments arriving as string fragments that must be joined and then parsed.
- Gemini sends `parts`, some flagged `"thought": true`, and never sends a `[DONE]` marker.

A parser that understood all three would carry three vendor vocabularies, and a fourth backend would mean editing it. The parser instead produces `SseEvent`, which uses the wire format's own vocabulary.

```rust,name=src/sse.rs
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}
```

Translation lives in the adapter that already owns that vendor's request body. Adding a backend changes nothing else, and the state machine is tested against SSE framing alone.

The cost is one translator per wire shape. Four exist:

- `ChatCompletionTranslator` serves OpenAI, vLLM, LM Studio, DeepSeek's native route and llama.cpp's chat route.
- `AnthropicTranslator` serves Anthropic and DeepSeek's bridge route.
- `GeminiTranslator` serves Gemini.
- `CompletionTranslator` serves llama.cpp's native `/completion` route.

The in-process `provider-local` engine produces blocks directly and does not use the parser.

## What the parser does with a chunk

The parser keeps one reusable `bytes::BytesMut` buffer. A chunk is copied in once, then a cursor walks it.

- The cursor advances line by line with `memchr`. A preceding carriage return is stripped, so CRLF and LF both work.
- A line is `field: value`. The colon is required and one following space is trimmed. Four fields are recognised: `event`, `data`, `id` and `retry`.
- A line starting with a colon is a comment and is ignored. Unknown fields are ignored.
- `data` lines accumulate in the buffer. A blank line ends the frame, which is parsed from the buffer slice, joined with newlines and advanced away.
- A partial trailing line stays in the buffer until a later chunk completes it, so arbitrary TCP chunk boundaries need no special handling.

A frame with no `event` field gets the event name `message`. A non-empty `id` without a NUL byte updates `last_event_id`, and a frame without its own `id` inherits that value. A malformed `retry` stays `None` and never fails the stream.

Allocation per completed frame is the `data` string, the `event` string, and the `id` string only on a stream that carries `id:` fields.

## What its memory depends on

The buffer's size does not depend on the number of chunks or the length of the stream. Each completed frame is advanced away, and the buffer reclaims that offset in place instead of reallocating. The high-water mark is the largest single frame plus one partial trailing line.

The buffer starts at 8192 bytes, and `capacity()` reports the current figure. There is no maximum. A server that opens a frame and never ends it with a blank line makes the buffer grow without limit, because the frame cannot be dispatched until the blank line arrives. That is the SSE wire format's own rule.

Truncating the frame would hand the adapter half a JSON body. Failing the stream at a threshold would break any server that legitimately sends a large frame. This is the one growable structure whose bound the peer sets instead of configuration, so [Memory discipline](@/concepts/memory-discipline.md) lists it as an uncapped structure.

## The one error

`feed_chunk` fails on exactly one condition: a completed frame whose `data`, `event` or `id` bytes are not valid UTF-8. The error is `CucaError::SseParse`. Partial lines are never validated, so a multi-byte character split across two chunks is not an error.

Next page: [Plugins and services](@/concepts/plugin-layering.md).
