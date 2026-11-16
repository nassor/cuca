//! K/V transfer between two `LocalModel`s of the same file: the prefill node
//! calls [`LocalModel::kv_export`] and streams the frames of
//! [`KvExport::next_frame`]; the decode node calls
//! [`LocalModel::kv_import`] and writes every frame to [`KvImport`]. The
//! frames are described in [`kvframe`](super::kvframe); the engine side is
//! `engine/scheduler/transfer.rs`.
//!
//! Dropping a [`KvExport`] or a [`KvImport`] ends the transfer on the engine
//! thread: its end is the drop of a `oneshot::Receiver` the engine's sender
//! sees closed, so a crashed or cancelled peer never leaks blocks.

use tokio::sync::oneshot;

use super::engine::ExportReady;
use super::kvframe::{self, Header};
use super::model::err as local_err;
use super::wire::WIRE_VERSION;
use super::{LocalModel, shutdown};
use crate::error::CucaError;

impl LocalModel {
    /// What identifies this model's file and cache layout (a 64-bit hash of
    /// the architecture, the file's size, every tensor's name, shape and
    /// type, and the K/V groups): two nodes exchange K/V only when theirs are
    /// equal. The size of the K/V pools is not part of it.
    pub fn fingerprint(&self) -> u64 {
        self.inner.fingerprint
    }

    /// Whether the engine takes imported K/V: false when its drafter
    /// observes every position (it would replay the imported prompt to feed
    /// the drafter, so the rows would be computed twice) or the prefix store
    /// is off.
    pub fn imports_kv(&self) -> bool {
        self.inner.imports_kv
    }

    /// How many leading tokens of `ids` the prefix store already holds: an
    /// import starts after them. The references are given back at once.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the engine has shut down.
    pub async fn kv_hit(&self, ids: Vec<u32>) -> Result<usize, CucaError> {
        self.call(move |e| e.kv_hit(&ids)).await
    }

    /// Export the K/V of the prompt `ids` from position `from` (a multiple of
    /// the block size, what the importer already holds): the engine
    /// prefills the prompt as a request would, reusing its prefix store, up
    /// to the last block boundary below its last token, and keeps those
    /// blocks until the returned handle is dropped.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when the model is not configured to export
    /// (`kv_export`), [`CucaError::Provider`] for a prompt the context cannot
    /// hold, an empty or unaligned range, all export slots busy, a failed
    /// prefill, or an engine that has shut down.
    pub async fn kv_export(&self, ids: Vec<u32>, from: usize) -> Result<KvExport, CucaError> {
        let (alive_tx, alive_rx) = oneshot::channel();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.call(move |e| e.export_begin(ids, from, alive_tx, reply_tx))
            .await??;
        let ready = reply_rx.await.map_err(|_| shutdown())??;
        Ok(KvExport {
            model: self.clone(),
            ready,
            stage: ExportStage::Header,
            _alive: alive_rx,
        })
    }

    /// Start importing the K/V of `ids[from..to]` (`from` tokens are in the
    /// prefix store already, see [`LocalModel::kv_hit`]): write every frame
    /// of the peer's [`KvExport`] to the returned handle. When the end frame
    /// is in, the prefix is in the store and a request for this prompt reads
    /// `to` tokens from it. Dropping the handle early gives the blocks back.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the engine cannot take imported K/V
    /// ([`LocalModel::imports_kv`]), the range is not whole blocks of `ids`,
    /// the store no longer holds the first `from` tokens, the import slots
    /// are busy, the pools have no room, or the engine has shut down.
    pub async fn kv_import(
        &self,
        ids: Vec<u32>,
        from: usize,
        to: usize,
    ) -> Result<KvImport, CucaError> {
        let (alive_tx, alive_rx) = oneshot::channel();
        let id = self
            .call(move |e| e.import_begin(ids, from, to, alive_tx))
            .await??;
        Ok(KvImport {
            model: self.clone(),
            id,
            from,
            to,
            header: false,
            done: false,
            _alive: Some(alive_rx),
        })
    }
}

/// Where a [`KvExport`] is in its frames.
#[derive(Debug, Clone, Copy)]
enum ExportStage {
    Header,
    /// The rows from this position on.
    Rows(usize),
    Snapshot,
    End,
    Done,
}

/// The K/V of a prompt prefix on its way out, from
/// [`LocalModel::kv_export`]. Its blocks stay in the engine's pools until the
/// handle is dropped.
pub struct KvExport {
    model: LocalModel,
    ready: ExportReady,
    stage: ExportStage,
    /// Dropping it tells the engine the peer is done.
    _alive: oneshot::Receiver<()>,
}

impl std::fmt::Debug for KvExport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvExport")
            .field("range", &self.range())
            .field("stage", &self.stage)
            .finish_non_exhaustive()
    }
}

impl KvExport {
    /// The positions the export carries: `from` is what the importer holds,
    /// `to` where the carried prefix ends.
    pub fn range(&self) -> (usize, usize) {
        (self.ready.from, self.ready.to)
    }

    /// The next frame: the header, the rows of at most `chunk` tokens each,
    /// the recurrent-state snapshot (a hybrid model), the end frame, then
    /// `None`. A failed frame ends the export: `Some(Err)` once, then `None`.
    pub async fn next_frame(&mut self) -> Option<Result<Vec<u8>, CucaError>> {
        let ready = self.ready;
        let (frame, next) = match self.stage {
            ExportStage::Header => {
                let header = Header {
                    version: WIRE_VERSION,
                    fingerprint: self.model.fingerprint(),
                    from: ready.from,
                    to: ready.to,
                    chunk: ready.chunk,
                    snapshot: ready.snapshot,
                };
                (
                    Ok(kvframe::header_frame(&header)),
                    ExportStage::Rows(ready.from),
                )
            }
            ExportStage::Rows(a) => {
                let b = (a + ready.chunk).min(ready.to);
                let id = ready.id;
                let frame = self.model.call(move |e| e.export_rows(id, a, b)).await;
                let next = if b >= ready.to {
                    if ready.snapshot {
                        ExportStage::Snapshot
                    } else {
                        ExportStage::End
                    }
                } else {
                    ExportStage::Rows(b)
                };
                (frame.and_then(|f| f), next)
            }
            ExportStage::Snapshot => {
                let id = ready.id;
                let frame = self.model.call(move |e| e.export_snapshot(id)).await;
                (frame.and_then(|f| f), ExportStage::End)
            }
            ExportStage::End => (Ok(kvframe::end_frame()), ExportStage::Done),
            ExportStage::Done => return None,
        };
        self.stage = if frame.is_ok() {
            next
        } else {
            ExportStage::Done
        };
        Some(frame)
    }
}

/// The K/V of a prompt prefix on its way in, from
/// [`LocalModel::kv_import`]. Dropping it before the end frame gives the
/// imported blocks back.
pub struct KvImport {
    model: LocalModel,
    id: u64,
    from: usize,
    to: usize,
    /// The header frame was accepted.
    header: bool,
    done: bool,
    /// Dropping it tells the engine the import is over; it is dropped as soon
    /// as the import ends, filed or failed.
    _alive: Option<oneshot::Receiver<()>>,
}

impl std::fmt::Debug for KvImport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvImport")
            .field("range", &(self.from, self.to))
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl KvImport {
    /// The positions the import carries.
    pub fn range(&self) -> (usize, usize) {
        (self.from, self.to)
    }

    /// Take one frame of the peer's export, in order. Returns `true` once
    /// the end frame filed the prefix in the prefix store.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a header whose version or fingerprint
    /// differs (the peer's model is not the same file or cache layout) or
    /// whose range is not this import's, a frame before the header or after
    /// the end, a malformed or out-of-order frame, or rows that do not fit
    /// the model. A failed frame ends the import.
    pub async fn write_frame(&mut self, frame: &[u8]) -> Result<bool, CucaError> {
        if self.done {
            return Err(local_err("the K/V import is over"));
        }
        let result = self.take_frame(frame).await;
        if !matches!(result, Ok(false)) {
            // Filed or failed: the engine side is done with it either way.
            self.done = true;
            self._alive = None;
        }
        result
    }

    /// [`KvImport::write_frame`] without the bookkeeping of an ended import.
    async fn take_frame(&mut self, frame: &[u8]) -> Result<bool, CucaError> {
        let (kind, payload) = kvframe::parse(frame)?;
        if kind == kvframe::HEADER {
            let header = kvframe::decode_header(payload)?;
            if self.header
                || header.version != WIRE_VERSION
                || header.fingerprint != self.model.fingerprint()
            {
                return Err(local_err(
                    "the peer's model is not the same file or cache layout",
                ));
            }
            if (header.from, header.to) != (self.from, self.to) {
                return Err(local_err(format!(
                    "the peer exports {}..{}, this import is {}..{}",
                    header.from, header.to, self.from, self.to
                )));
            }
            self.header = true;
            return Ok(false);
        }
        if !self.header {
            return Err(local_err("a K/V frame came before the header"));
        }
        let (id, frame) = (self.id, frame.to_vec());
        self.model
            .call(move |e| e.import_frame(id, &frame))
            .await
            .and_then(|ended| ended)
    }
}
