+++
title = "The SSE parser"
description = "Why feed_chunk returns Vec<SseEvent> rather than content blocks, and what the single reusable buffer bounds."
template = "page.html"
weight = 2
+++

# The SSE parser

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SseStreamParser</code> is protocol work only: it frames Server-Sent Events and refuses to know what any vendor puts inside them</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You expected the parser to hand you content blocks, or you want to know what its memory is a function of</dd>
</dl>

The design specification gives the parser this signature:

```rust,name=The specified signature
fn feed_chunk(&mut self, chunk: &[u8]) -> Vec<MessageContentBlock>;
```

The implementation returns something else:

```rust,name=src/sse.rs line 113
pub fn feed_chunk(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, CucaError>
```

The deviation is deliberate and it is the most consequential boundary in the
crate.

## Why events and not blocks

Returning `MessageContentBlock` would mean the parser decides what a chunk of
vendor JSON means. Those meanings do not agree. Anthropic sends
`content_block_delta` frames inside an explicit
`content_block_start` and `content_block_stop` envelope. OpenAI-compatible
servers send `choices[0].delta`, with tool-call arguments arriving as string
fragments that must be concatenated and then parsed. Gemini sends `parts`, some
flagged `"thought": true`, and never sends a `[DONE]` marker at all.

A parser that understood all three would carry three vendor vocabularies, and
adding a fourth backend would mean editing the parser. Instead the parser
produces `SseEvent`, which is exactly the wire format's own vocabulary:

```rust,name=src/sse.rs lines 71 to 81
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}
```

Translation lives in the adapter that already owns that vendor's request body.
Nothing else has to change to add a backend, and the state machine stays
testable on its own, against SSE framing rather than against a provider.

The cost is one extra hop: every adapter carries a translator. Five of the seven
share one, `openai_compat`, so the real count is three translators for seven
backends.

## What the parser does with a chunk

One reusable `bytes::BytesMut` buffer, one copy in, then a cursor walk. A scan
cursor advances line by line with `memchr`, a preceding carriage return is
stripped for CRLF tolerance, and a line is `field: value` with the colon
required and one following space trimmed. Four fields are recognised: `event`,
`data`, `id`, `retry`.

`data` lines accumulate inside that same buffer. A blank line ends the frame,
which is then parsed from the buffer slice, joined with newlines, and advanced
away. A partial trailing line stays put until a later chunk completes it, so
chunk boundaries never need special handling: arbitrary TCP splitting is the
normal case, not an edge case.

Unknown fields are ignored, for forward compatibility. A line starting with a
colon is a comment and ignored. A malformed `retry` value stays `None` and is
never fatal, because a reconnection hint is not worth failing a stream over.

## What its memory is a function of

Not the number of chunks, and not the length of the stream. Each completed frame
is advanced away, and the buffer reclaims that offset in place rather than
reallocating. The high-water mark is the largest single frame plus one partial
trailing line.

The buffer starts at 8192 bytes of capacity and `capacity()` reports the current
figure. There is no maximum, and one case makes it grow without limit: a server
that opens a frame and never terminates it with a blank line. That frame cannot
be dispatched until the blank line arrives, which is the SSE wire format's own
rule rather than a decision the parser is free to make. Truncating the frame
would hand the adapter a half-parsed JSON body, and failing the stream on a
threshold would break any server that legitimately sends a large frame.

This is the one growable structure in the crate whose bound is set by the peer
rather than by configuration, which is why it is called out here and in
[Memory discipline](@/concepts/memory-discipline.md) rather than left implicit.

## The one error

`feed_chunk` returns `Result`, and there is exactly one thing it fails on: a
completed frame whose `data`, `event` or `id` bytes are not valid UTF-8, which
becomes `CucaError::SseParse`. Partial lines are never validated, because a
multi-byte character split across two chunks is not an error, it is Tuesday.

Next page: [Plugins and services](@/concepts/plugin-layering.md), which covers
the two stages of the pipeline the parser never sees.
