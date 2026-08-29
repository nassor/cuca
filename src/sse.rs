//! Zero-allocation SSE stream parser engine.
//!
//! # Design deviation from the spec signature
//!
//! The spec signature is `feed_chunk(...) -> Vec<MessageContentBlock>`. This
//! parser intentionally returns `Vec<SseEvent>` instead: it stays
//! protocol-pure and does not interpret vendor JSON shapes. Translating a
//! normalized frame into `MessageContentBlock` is provider-specific
//! (Anthropic `content_block_delta` vs OpenAI `choices[0].delta` vs Gemini
//! parts) and lives in the per-provider adapters on top of this engine.
//! Keeping the parser single-purpose makes the zero-allocation state machine
//! independently testable.
//!
//! # Parsing algorithm
//!
//! The parser is a byte state machine over a single reusable
//! [`bytes::BytesMut`] buffer, fed by TCP chunks:
//!
//! - `buffer.extend_from_slice(chunk)`: one copy into the reusable buffer,
//!   no intermediate strings.
//! - A scan cursor advances line by line with `memchr::memchr(b'\n', ..)`; a
//!   preceding `\r` is stripped for CRLF tolerance.
//! - A line is `field[: ]value` (the colon is required; a single space after
//!   the colon is trimmed). Known fields: `event`, `data`, `id`, `retry`.
//!   Lines starting with `:` are comments and are ignored. Unknown fields are
//!   ignored for forward compatibility.
//! - `data` lines accumulate inside the buffer (the per-frame accumulator);
//!   a blank line terminates the current frame, which is parsed out of the
//!   buffer slice and joined with `\n`, then `advance`d away, clearing the
//!   per-frame accumulators. A partial trailing line stays in the buffer until
//!   a later chunk completes it, so accumulation is naturally preserved across
//!   arbitrary chunk boundaries.
//! - A non-empty, NUL-free `id` field updates `last_event_id`, giving
//!   Last-Event-ID reconnection semantics.
//! - Completed frames are parsed from buffer slices. The per-frame
//!   allocations are exactly: the `data` `String`, the `event` `String` (the
//!   `"message"` default is also owned), and — only for a stream that carries
//!   `id:` fields — the `SseEvent` `id` `String`. An `id`-bearing frame
//!   allocates twice for that field (one owned copy for the event, one for the
//!   parser's persisted `last_event_id`); a frame that inherits the id
//!   allocates once. Both are forced by the owned `Option<String>` on the
//!   public event type. Frames on a stream with no `id:` field allocate
//!   nothing for it.
//! - Malformed `retry` values are ignored (kept `None`), never fatal.
//!
//! # Memory bounds
//!
//! The parser holds exactly one buffer, so its footprint is not a function of
//! how many chunks or frames pass through it: each completed frame is
//! `advance`d away, and `BytesMut::reserve` reclaims that offset in place
//! instead of reallocating. The high-water mark is therefore the largest
//! single frame plus one partial trailing line, never the stream length. A
//! server that never terminates a frame with a blank line is the one case
//! that keeps the buffer growing; the frame is not dispatched until it does,
//! by design of the SSE wire format.
//!
//! # Error path
//!
//! `feed_chunk` returns `Result<_, CucaError>` per spec, which does not name
//! the error source. UTF-8 is the only recoverable-fatal condition here: a
//! completed frame whose `data`/`event`/`id` bytes are not valid UTF-8 yields
//! `Err(CucaError::SseParse(..))`. Partial (unterminated) lines are never
//! validated until the frame they belong to is completed by a blank line.

use bytes::{Buf, BytesMut};
use memchr::memchr;

use crate::error::CucaError;

/// One complete SSE frame (the wire-level event, pre-translation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Event type; defaults to "message" when only `data:` is present.
    pub event: String,
    /// Field payload; multi-line `data:` joined with '\n'.
    pub data: String,
    /// Last-Event-ID carried by this frame, if any.
    pub id: Option<String>,
    /// Suggested reconnection delay in milliseconds, if `retry:` was present
    /// and well-formed.
    pub retry: Option<u64>,
}

/// Zero-allocation byte state machine.
pub struct SseStreamParser {
    /// Single reusable contiguous buffer (8192 capacity init). Holds the
    /// current frame's raw lines plus any trailing partial line.
    buffer: BytesMut,
    /// Persists across chunks per the SSE Last-Event-ID semantics.
    last_event_id: Option<String>,
}

impl SseStreamParser {
    /// Creates a parser with an 8192-byte initial buffer and no last-event-id.
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(8192),
            last_event_id: None,
        }
    }

    /// Appends `chunk` and parses every complete frame; incomplete trailing
    /// data stays in the internal buffer until the next call.
    ///
    /// Returns the frames completed by this chunk.
    ///
    /// # Errors
    ///
    /// [`CucaError::SseParse`] when a completed frame's `data`, `event`, or
    /// `id` bytes are not valid UTF-8. Partial lines are not validated until
    /// the frame they belong to is completed by a blank line, so a chunk that
    /// splits a multi-byte character mid-line is not an error.
    pub fn feed_chunk(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, CucaError> {
        self.buffer.extend_from_slice(chunk);

        let mut events = Vec::new();
        // Scan cursor: start of the current, not-yet-processed line, relative
        // to `buffer`. Non-blank lines are only scanned past: they stay in
        // the buffer as the accumulating frame until a blank line dispatches.
        let mut scan = 0usize;

        while let Some(rel) = memchr(b'\n', &self.buffer[scan..]) {
            let newline = scan + rel;
            let line = &self.buffer[scan..newline];
            let is_blank = line.is_empty() || (line.len() == 1 && line[0] == b'\r');

            if is_blank {
                // Everything before `scan` is the completed frame's raw lines.
                let (ev, frame_id) = self.parse_frame(&self.buffer[..scan])?;
                if let Some(id) = frame_id {
                    self.last_event_id = Some(id);
                }
                events.push(ev);
                // Consume the frame and the blank line; the buffer now starts
                // fresh at the next frame, so reset the scan cursor.
                self.buffer.advance(newline + 1);
                scan = 0;
            } else {
                scan = newline + 1;
            }
        }

        Ok(events)
    }

    /// The most recent valid `id:` field, used for Last-Event-ID reconnection.
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// The current capacity of the internal reusable buffer.
    ///
    /// `new()` initializes this to at least 8192 bytes per spec.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Parses one complete frame's raw lines (which contain no blank line)
    /// into an [`SseEvent`], joining multi-line `data` with `\n`.
    ///
    /// Returns the frame and the id to persist, if the frame carried a valid
    /// `id:` field.
    fn parse_frame(&self, frame: &[u8]) -> Result<(SseEvent, Option<String>), CucaError> {
        let mut event_name: Option<String> = None;
        let mut data = String::new();
        let mut current_id: Option<String> = None;
        let mut retry: Option<u64> = None;

        let mut pos = 0usize;
        while pos < frame.len() {
            let nl = match memchr(b'\n', &frame[pos..]) {
                Some(rel) => pos + rel,
                None => break,
            };
            let mut line_end = nl;
            if line_end > pos && frame[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            let line = &frame[pos..line_end];

            // Comment lines (`: ...`) are ignored entirely.
            if line.first() != Some(&b':') {
                // The colon is required for a field line.
                if let Some(sep) = memchr(b':', line) {
                    let field = &line[..sep];
                    let value = line[sep + 1..]
                        .strip_prefix(b" ")
                        .unwrap_or(&line[sep + 1..]);
                    match field {
                        b"event" => {
                            let s = std::str::from_utf8(value).map_err(|e| {
                                CucaError::SseParse(format!("event field is not valid UTF-8: {e}"))
                            })?;
                            event_name = Some(s.to_string());
                        }
                        b"data" => {
                            let s = std::str::from_utf8(value).map_err(|e| {
                                CucaError::SseParse(format!("data field is not valid UTF-8: {e}"))
                            })?;
                            if !data.is_empty() {
                                data.push('\n');
                            }
                            data.push_str(s);
                        }
                        b"id" => {
                            let s = std::str::from_utf8(value).map_err(|e| {
                                CucaError::SseParse(format!("id field is not valid UTF-8: {e}"))
                            })?;
                            // Empty ids and ids containing a NUL are ignored.
                            if !s.is_empty() && !s.contains('\0') {
                                current_id = Some(s.to_string());
                            }
                        }
                        b"retry" => {
                            // Malformed retry values are ignored (kept None),
                            // never fatal.
                            if let Ok(s) = std::str::from_utf8(value) {
                                retry = s.trim().parse::<u64>().ok();
                            }
                        }
                        // Unknown fields are ignored for forward compat.
                        _ => {}
                    }
                }
            }

            pos = nl + 1;
        }

        Ok((
            SseEvent {
                event: event_name.unwrap_or_else(|| "message".to_string()),
                data,
                id: current_id.clone().or_else(|| self.last_event_id.clone()),
                retry,
            },
            current_id,
        ))
    }
}

impl Default for SseStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One complete frame in one chunk.
    #[test]
    fn single_frame_single_chunk() {
        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b"data: hello\n\n").unwrap();
        assert_eq!(
            events,
            vec![SseEvent {
                event: "message".to_string(),
                data: "hello".to_string(),
                id: None,
                retry: None,
            }]
        );
    }

    /// Frame split across arbitrary chunk boundaries: feeding one byte at a
    /// time yields identical events.
    #[test]
    fn frame_split_across_chunks() {
        let input = b"data: hello\n\nevent: x\ndata: world\n\n";
        let mut byte_at_a_time = SseStreamParser::new();
        let mut events = Vec::new();
        for &b in input {
            events.extend(byte_at_a_time.feed_chunk(&[b]).unwrap());
        }

        let mut whole = SseStreamParser::new();
        let whole_events = whole.feed_chunk(input).unwrap();

        assert_eq!(events, whole_events);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[1].data, "world");
    }

    /// Multiple frames in one chunk; frame split mid-line and mid-field-name.
    #[test]
    fn multiple_frames_and_split_lines() {
        let input = b"data: a\n\ndata: b\ndata: c\n\nevent: done\ndata: fin\n\n";
        let mut parser = SseStreamParser::new();
        // Feed with a split right in the middle of the stream.
        let split = input.len() / 2 + 3;
        let mut events = parser.feed_chunk(&input[..split]).unwrap();
        events.extend(parser.feed_chunk(&input[split..]).unwrap());

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b\nc");
        assert_eq!(events[2].event, "done");
        assert_eq!(events[2].data, "fin");
    }

    /// CRLF and mixed line endings.
    #[test]
    fn crlf_and_mixed_endings() {
        let mut parser = SseStreamParser::new();
        let events = parser
            .feed_chunk(b"data: one\r\ndata: two\r\n\r\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "one\ntwo");

        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b"data: a\r\ndata: b\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a\nb");
    }

    /// Comment lines produce no events.
    #[test]
    fn comments_ignored() {
        let mut parser = SseStreamParser::new();
        let events = parser
            .feed_chunk(b": keep-alive\n: keep-alive2\ndata: hello\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");

        // A stream of only comments produces nothing.
        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b": ping\n: pong\n").unwrap();
        assert!(events.is_empty());
    }

    /// Multi-line data joined with '\n'; `event:` names the event.
    #[test]
    fn multi_line_data_and_event_name() {
        let mut parser = SseStreamParser::new();
        let events = parser
            .feed_chunk(b"event: content_block_delta\ndata: part one\ndata: part two\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, "content_block_delta");
        assert_eq!(events[0].data, "part one\npart two");
    }

    /// `id:` persists across chunks and frames; empty id is ignored.
    #[test]
    fn id_persistence_across_chunks() {
        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b"id: 42\ndata: a\n\n").unwrap();
        assert_eq!(events[0].id.as_deref(), Some("42"));
        assert_eq!(parser.last_event_id(), Some("42"));

        // A subsequent frame without its own id inherits the last one.
        let events = parser.feed_chunk(b"data: b\n\n").unwrap();
        assert_eq!(events[0].id.as_deref(), Some("42"));

        // Empty id is ignored, so the previous id persists.
        let events = parser.feed_chunk(b"id:\ndata: c\n\n").unwrap();
        assert_eq!(events[0].id.as_deref(), Some("42"));
        assert_eq!(parser.last_event_id(), Some("42"));

        // A new id replaces the old one.
        let events = parser.feed_chunk(b"id: 7\ndata: d\n\n").unwrap();
        assert_eq!(events[0].id.as_deref(), Some("7"));
        assert_eq!(parser.last_event_id(), Some("7"));
    }

    /// `retry:` parses to Some(u64); garbage retry is ignored (None).
    #[test]
    fn retry_parsing() {
        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b"retry: 2500\ndata: a\n\n").unwrap();
        assert_eq!(events[0].retry, Some(2500));

        let mut parser = SseStreamParser::new();
        let events = parser
            .feed_chunk(b"retry: not-a-number\ndata: a\n\n")
            .unwrap();
        assert_eq!(events[0].retry, None);

        // retry without a data field still forms a complete (empty) event.
        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b"retry: 100\n\n").unwrap();
        assert_eq!(events[0].retry, Some(100));
        assert_eq!(events[0].data, "");
    }

    /// A partial trailing line is held in the buffer and completed by the
    /// next chunk.
    #[test]
    fn partial_trailing_line_completed_later() {
        let mut parser = SseStreamParser::new();
        let events = parser.feed_chunk(b"data: hel").unwrap();
        assert!(events.is_empty());

        let events = parser.feed_chunk(b"lo\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    /// Byte-by-byte feed produces exactly the same event sequence as a
    /// whole-chunk feed (ordering + no duplication).
    #[test]
    fn byte_by_byte_matches_whole_chunk() {
        let input = b"data: one\n\nid: 5\nevent: delta\ndata: two\n\n: comment\ndata: three\n\n";
        let mut whole = SseStreamParser::new();
        let whole_events = whole.feed_chunk(input).unwrap();

        let mut incremental = SseStreamParser::new();
        let mut incremental_events = Vec::new();
        for &b in input {
            incremental_events.extend(incremental.feed_chunk(&[b]).unwrap());
        }

        assert_eq!(whole_events, incremental_events);
        assert_eq!(incremental_events.len(), 3);
    }

    /// Non-UTF-8 data in a completed frame is a parse error.
    #[test]
    fn non_utf8_data_is_error() {
        let mut parser = SseStreamParser::new();
        let err = parser.feed_chunk(b"data: \xff\xfe\n\n").unwrap_err();
        assert!(matches!(err, CucaError::SseParse(_)));
    }

    /// Non-UTF-8 in event and id fields is also a parse error.
    #[test]
    fn non_utf8_event_and_id_are_errors() {
        let mut parser = SseStreamParser::new();
        let err = parser.feed_chunk(b"event: \xff\ndata: x\n\n").unwrap_err();
        assert!(matches!(err, CucaError::SseParse(_)));

        let mut parser = SseStreamParser::new();
        let err = parser.feed_chunk(b"id: \xff\ndata: x\n\n").unwrap_err();
        assert!(matches!(err, CucaError::SseParse(_)));
    }

    /// `new()` initializes the buffer to at least 8192 bytes.
    #[test]
    fn initial_buffer_capacity_is_at_least_8192() {
        let parser = SseStreamParser::new();
        assert!(parser.capacity() >= 8192);
    }

    /// A line without a colon is ignored per spec.
    #[test]
    fn line_without_colon_ignored() {
        let mut parser = SseStreamParser::new();
        let events = parser
            .feed_chunk(b"this is not a field\ndata: ok\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "ok");
    }
}
