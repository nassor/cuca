//! The frame stream of a layer stage: the worker's side, and the framing both
//! sides share.
//!
//! A coordinator that reaches a worker over plain `http://` upgrades one
//! connection to a stream (`GET /cuca/v1/stage-stream[/{model}]` with
//! `Connection: Upgrade` and `Upgrade: cuca-stage/1`, through the server's
//! authentication) and then exchanges the frames of
//! [`stageframe`](super::stageframe) on it, one request and one reply at a
//! time, with no HTTP around them:
//!
//! - request: `len u64`, then the frame;
//! - reply: `status u8`, `compute u64` (nanoseconds the engine thread took),
//!   `len u64`, then the reply frame, or for a failure its message in UTF-8.
//!   Status 0 is an answer, 1 a mismatch of configuration (a
//!   [`CucaError::Config`]: the `409` of the `POST` route), 2 any other
//!   failure (its `500`). A failure leaves the stream usable.
//!
//! Why: a call over HTTP crosses the async runtimes of both servers and costs
//! several thread wake-ups in each direction; here a worker's reader thread
//! hands the frame to the engine thread, which runs it and writes the reply
//! to the socket itself, and the coordinator's engine thread blocks on its
//! own socket. A frame is one write each way.
//!
//! # Bounds
//!
//! A model serves at most [`MAX_STAGE_STREAMS`] streams (a thread each); a
//! frame is at most the limit the caller names (the `POST` route's 1 GiB), and
//! a connection holds one frame at a time: the reader waits for the engine to
//! finish before it reads the next. A connection idle for
//! [`STAGE_STREAM_IDLE`] is closed (a coordinator reconnects and sends again,
//! see [`stageframe`](super::stageframe)), and a peer that does not read a
//! reply for [`STAGE_STREAM_WRITE`] fails it.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::LocalModel;
use super::engine::Msg;
use crate::error::CucaError;
use crate::types::ProviderEndpoint;

/// The `Upgrade` token of the stream; the number is the version of the
/// framing above.
pub const STAGE_STREAM_UPGRADE: &str = "cuca-stage/1";

/// The largest frame either side takes (1 GiB, what one forward pass can
/// carry, bounded by `max_step_tokens`).
pub const MAX_STAGE_FRAME_BYTES: usize = 1 << 30;

/// Streams one model serves at once.
pub const MAX_STAGE_STREAMS: usize = 8;

/// How long a stream may sit between two frames before the worker closes it.
pub const STAGE_STREAM_IDLE: Duration = Duration::from_secs(300);

/// How long a reply may wait for the peer to read it.
pub const STAGE_STREAM_WRITE: Duration = Duration::from_secs(60);

/// Reply status: an answer.
pub(crate) const REPLY_OK: u8 = 0;
/// Reply status: the worker does not match the coordinator.
pub(crate) const REPLY_MISMATCH: u8 = 1;
/// Reply status: any other failure.
pub(crate) const REPLY_FAILED: u8 = 2;
/// Bytes of a reply before its payload.
pub(crate) const REPLY_HEAD: usize = 1 + 8 + 8;

/// A place for one more stream on a model, from
/// [`LocalModel::reserve_stage_stream`]; it frees the place when it is
/// dropped, with the stream.
#[derive(Debug)]
pub struct StageStreamSlot {
    taken: Arc<AtomicUsize>,
}

impl Drop for StageStreamSlot {
    fn drop(&mut self) {
        self.taken.fetch_sub(1, Ordering::AcqRel);
    }
}

/// `payload` written after `head` in as few writes as possible: a small frame
/// goes out as one segment, a large one is not copied.
pub(crate) fn write_framed(
    mut out: impl Write,
    head: &[u8],
    payload: &[u8],
) -> std::io::Result<()> {
    if payload.len() <= 1 << 20 {
        let mut buf = Vec::with_capacity(head.len() + payload.len());
        buf.extend_from_slice(head);
        buf.extend_from_slice(payload);
        out.write_all(&buf)
    } else {
        out.write_all(head)?;
        out.write_all(payload)
    }
}

/// Write one reply: `status`, the nanoseconds the engine thread took, then
/// `payload`.
fn write_reply(
    out: impl Write,
    status: u8,
    compute: Duration,
    payload: &[u8],
) -> std::io::Result<()> {
    let mut head = [0u8; REPLY_HEAD];
    head[0] = status;
    let ns = u64::try_from(compute.as_nanos()).unwrap_or(u64::MAX);
    head[1..9].copy_from_slice(&ns.to_le_bytes());
    head[9..].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    write_framed(out, &head, payload)
}

impl LocalModel {
    /// Reserve a place for one more frame stream on this layer-stage worker.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for a model that is not a layer stage;
    /// [`CucaError::Provider`] when [`MAX_STAGE_STREAMS`] streams are open.
    pub fn reserve_stage_stream(&self) -> Result<StageStreamSlot, CucaError> {
        if !self.inner.is_stage {
            return Err(CucaError::Config("this model is not a layer stage".into()));
        }
        let taken = &self.inner.stage_streams;
        let reserved = taken.try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < MAX_STAGE_STREAMS).then_some(n + 1)
        });
        if reserved.is_err() {
            return Err(CucaError::provider(
                ProviderEndpoint::Local,
                format!("this stage worker serves {MAX_STAGE_STREAMS} streams already"),
            ));
        }
        Ok(StageStreamSlot {
            taken: Arc::clone(taken),
        })
    }

    /// Serve a coordinator's frame stream on `stream` (blocking mode, already
    /// upgraded) from a thread of its own until the peer closes it, it idles
    /// out or a frame exceeds `max_frame`. Every frame runs as
    /// [`LocalModel::stage_call`] does; the engine thread writes the reply.
    ///
    /// # Errors
    ///
    /// [`CucaError::Io`] when the socket cannot be set up or the thread
    /// cannot start.
    pub fn serve_stage_stream(
        &self,
        slot: StageStreamSlot,
        stream: TcpStream,
        max_frame: usize,
    ) -> Result<(), CucaError> {
        let io = |e: std::io::Error| CucaError::Io {
            message: format!("stage stream: {e}"),
        };
        stream.set_nodelay(true).map_err(io)?;
        stream
            .set_read_timeout(Some(STAGE_STREAM_IDLE))
            .map_err(io)?;
        stream
            .set_write_timeout(Some(STAGE_STREAM_WRITE))
            .map_err(io)?;
        let model = self.clone();
        std::thread::Builder::new()
            .name("cuca-stage-stream".to_string())
            .spawn(move || {
                let socket = Arc::new(stream);
                model.stream_loop(&socket, max_frame);
                // The place is free before the peer can see the stream end,
                // so a coordinator that reconnects at once is not refused.
                drop(slot);
                let _ = socket.shutdown(Shutdown::Both);
            })
            .map_err(|e| CucaError::Io {
                message: format!("failed to start a stage stream thread: {e}"),
            })?;
        Ok(())
    }

    /// The reader thread of one stream.
    fn stream_loop(&self, socket: &Arc<TcpStream>, max_frame: usize) {
        loop {
            let mut head = [0u8; 8];
            if (&**socket).read_exact(&mut head).is_err() {
                return;
            }
            let Ok(len) = usize::try_from(u64::from_le_bytes(head)) else {
                return;
            };
            if len > max_frame {
                let message = format!("a frame of {len} bytes exceeds the limit of {max_frame}");
                let _ = write_reply(&**socket, REPLY_FAILED, Duration::ZERO, message.as_bytes());
                return;
            }
            let mut frame = vec![0u8; len];
            if (&**socket).read_exact(&mut frame).is_err() {
                return;
            }
            let (done_tx, done_rx) = std::sync::mpsc::sync_channel::<bool>(1);
            let out = Arc::clone(socket);
            let call = Msg::Call(Box::new(move |engine| {
                let started = Instant::now();
                let reply = engine.stage_call(&frame);
                let compute = started.elapsed();
                let written = match &reply {
                    Ok(reply) => write_reply(&*out, REPLY_OK, compute, reply),
                    Err(CucaError::Config(m)) => {
                        write_reply(&*out, REPLY_MISMATCH, compute, m.as_bytes())
                    }
                    Err(e) => write_reply(&*out, REPLY_FAILED, compute, e.to_string().as_bytes()),
                };
                let _ = done_tx.send(written.is_ok());
            }));
            if self.inner.jobs.blocking_send(call).is_err() {
                return;
            }
            // A closure that panicked drops its sender: the stream ends.
            if !done_rx.recv().unwrap_or(false) {
                return;
            }
        }
    }
}
