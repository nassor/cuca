//! The plumbing shared by the HTTP providers: the checked request send, and the
//! stream adapter (reqwest byte stream -> per-adapter chunk translator ->
//! blocks, one block per poll).

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use tokio_stream::Stream;

use crate::error::CucaError;
use crate::types::MessageContentBlock;

/// Send `request`; a non-2xx response becomes [`CucaError::Http`] carrying its
/// body.
pub(crate) async fn send_checked(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, CucaError> {
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await?;
        return Err(CucaError::Http {
            status: status.as_u16(),
            body,
        });
    }
    Ok(response)
}

/// One adapter's SSE parser and block translator, fed a byte chunk at a time.
pub(crate) trait ChunkTranslator {
    /// Translate one chunk, pushing its blocks to `out`. `Ok(true)` means the
    /// protocol's terminator was seen: the stream drains `out` and ends. An
    /// error poisons the stream: it is reported once and no more chunks are read.
    fn chunk(
        &mut self,
        bytes: &[u8],
        out: &mut VecDeque<MessageContentBlock>,
    ) -> Result<bool, CucaError>;

    /// The byte stream ended without `chunk` returning `Ok(true)`: push any
    /// blocks still held back.
    fn finish(&mut self, _out: &mut VecDeque<MessageContentBlock>) {}
}

/// Yields at most one block per poll; the stream ends after the terminator or
/// the byte stream's end, once the blocks already translated are drained.
pub(crate) struct SseBlockStream<T> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    translator: T,
    /// Blocks awaiting emission within the current chunk.
    buffer: VecDeque<MessageContentBlock>,
    /// True once the stream ended; only what is left in `buffer` is emitted.
    ended: bool,
}

impl<T> SseBlockStream<T> {
    pub(crate) fn new(
        inner: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        translator: T,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            translator,
            buffer: VecDeque::new(),
            ended: false,
        }
    }
}

impl<T: ChunkTranslator + Unpin> Stream for SseBlockStream<T> {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(block) = this.buffer.pop_front() {
                return Poll::Ready(Some(Ok(block)));
            }
            if this.ended {
                return Poll::Ready(None);
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    match this.translator.chunk(&bytes, &mut this.buffer) {
                        Ok(terminated) => this.ended = terminated,
                        Err(e) => {
                            this.ended = true;
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    this.ended = true;
                    return Poll::Ready(Some(Err(CucaError::Transport {
                        message: e.to_string(),
                    })));
                }
                Poll::Ready(None) => {
                    this.ended = true;
                    this.translator.finish(&mut this.buffer);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
