//! K/V transfer between nodes: exporting the K/V blocks of a prompt prefix
//! (a prefill node) and importing them into the prefix store (a decode node).
//!
//! # Export
//!
//! An export is a sequence like any other, queued with the whole prompt and
//! prefilled through the same steps (reusing what the prefix store holds),
//! except that it stops at `to` (the last block boundary below the prompt's
//! last token, [`Engine::transfer_end`]), computes no logits and generates
//! nothing. When it reaches `to` the engine keeps its blocks as a *lease*
//! (`Export`): the blocks stay referenced, so no allocation evicts them,
//! while the peer reads them chunk by chunk ([`Engine::export_rows`]). A
//! lease ends, its blocks filed in the prefix store, when the peer's handle is
//! dropped; a failed pass ends them all. At most `max_exports` sequences and
//! leases exist at once.
//!
//! # Import
//!
//! An import adopts blocks for `from..to` ([`KvPools::adopt`]: named by no
//! hash, so no lookup finds them while rows are missing), writes each rows
//! frame into them, and files the prefix in the prefix store when the end
//! frame arrives (with the recurrent-state snapshot of a hybrid model). The
//! next request for that prompt reads `to` tokens from the store. An import
//! that fails or whose handle is dropped gives its blocks back. At most
//! [`MAX_IMPORTS`] run at once.

use tokio::sync::{mpsc, oneshot};

use super::{Engine, Item, Job, Sequence, add_owed, local_err};
use crate::client::ResponseMetadataHandle;
use crate::error::CucaError;
use crate::provider::local::kvframe;
use crate::provider::local::model::RecurrentSnapshot;
use crate::provider::local::model::paged::{BLOCK_SIZE, SeqBlocks, block_keys};
use crate::provider::local::model::transfer::DeltaRows;

/// Imports running at once on one engine.
const MAX_IMPORTS: usize = 4;

/// Tokens of one rows frame, at least.
const CHUNK_TOKENS: usize = 256;

/// The request an export sequence carries.
pub(crate) struct ExportTicket {
    id: u64,
    from: usize,
    pub(super) to: usize,
    reply: oneshot::Sender<Result<ExportReady, CucaError>>,
    /// Closed when the peer's handle is dropped.
    alive: oneshot::Sender<()>,
    /// Keeps the sequence's event channel open: nothing is ever sent on it.
    _events: mpsc::Receiver<Item>,
}

/// What an export reports once its prefix is cached and leased.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExportReady {
    /// The lease: the id of every later [`Engine::export_rows`].
    pub(crate) id: u64,
    pub(crate) from: usize,
    pub(crate) to: usize,
    /// Tokens of one rows frame.
    pub(crate) chunk: usize,
    /// A recurrent-state snapshot follows the rows.
    pub(crate) snapshot: bool,
}

/// A prefix whose blocks a peer is reading.
pub(crate) struct Export {
    id: u64,
    /// The `to` tokens the blocks hold.
    ids: Vec<u32>,
    blocks: SeqBlocks,
    to: usize,
    snapshot: Option<Vec<DeltaRows>>,
    alive: oneshot::Sender<()>,
}

/// A prefix being imported.
pub(crate) struct Import {
    id: u64,
    ids: Vec<u32>,
    to: usize,
    /// The positions written so far.
    written: usize,
    blocks: SeqBlocks,
    /// The recurrent-state snapshot, once its frame arrived and fit the
    /// model.
    snapshot: Option<RecurrentSnapshot>,
    alive: oneshot::Sender<()>,
}

impl Sequence {
    /// Whether nobody waits for the sequence any more: its stream was
    /// dropped, or the peer of its export went away.
    pub(super) fn closed(&self) -> bool {
        self.events.is_closed() || self.export.as_ref().is_some_and(|t| t.alive.is_closed())
    }
}

impl Engine {
    /// The position an export of `ids` ends at: the last block boundary (of
    /// every group) below the last token, and for a hybrid model not past the
    /// block boundary its recurrent state is snapshotted at.
    pub(crate) fn transfer_end(&self, ids: &[u32]) -> usize {
        let unit = BLOCK_SIZE * self.pools.step();
        let end = ids.len().saturating_sub(1) / unit * unit;
        if self.model.is_hybrid() {
            end.min(self.snapshot_position(ids) / unit * unit)
        } else {
            end
        }
    }

    /// Whether the engine can take blocks it did not compute: the prefix
    /// store is on, and no drafter that observes every position would
    /// replay the prompt over them.
    pub(crate) fn imports_kv(&self) -> bool {
        self.batching.prefix_cache && !self.observes()
    }

    /// How many leading tokens of `ids` a request would read from the prefix
    /// store (without the bound of an observing drafter) and an import need
    /// not carry: the references are given back at once.
    pub(crate) fn usable_prefix(&mut self, ids: &[u32]) -> usize {
        if !self.batching.prefix_cache {
            return 0;
        }
        let mut blocks = self.pools.lookup(ids).into_seq();
        self.snapshot_cut(ids, &mut blocks);
        let held = blocks.len;
        self.pools.release(&mut blocks);
        held
    }

    /// The tokens of `ids` an import into this engine starts after: the
    /// prefix it holds, at most where an export of `ids` ends.
    pub(crate) fn kv_hit(&mut self, ids: &[u32]) -> usize {
        self.usable_prefix(ids).min(self.transfer_end(ids))
    }

    // ---- export ----

    /// Queue the export of `ids[from..to]` (`to` is [`Engine::transfer_end`]).
    /// The prompt prefills like a request; `reply` receives the lease when it
    /// is cached, and the lease ends when `alive`'s receiver is dropped.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when the model is not configured to export,
    /// [`CucaError::Provider`] for a prompt the context cannot hold, an empty
    /// or unaligned range, or when `max_exports` exports are running.
    pub(crate) fn export_begin(
        &mut self,
        ids: Vec<u32>,
        from: usize,
        alive: oneshot::Sender<()>,
        reply: oneshot::Sender<Result<ExportReady, CucaError>>,
    ) -> Result<(), CucaError> {
        let Some(max) = self.max_exports else {
            return Err(CucaError::Config(
                "this model does not export K/V: add `kv_export` to its configuration".into(),
            ));
        };
        let prompt_len = ids.len();
        if prompt_len == 0 || prompt_len >= self.context_length {
            return Err(local_err(format!(
                "prompt has {prompt_len} tokens; context_length is {}",
                self.context_length
            )));
        }
        let to = self.transfer_end(&ids);
        if !(from.is_multiple_of(BLOCK_SIZE * self.pools.step()) && from < to) {
            return Err(local_err(format!(
                "nothing to export: {from}..{to} is empty or not block-aligned"
            )));
        }
        let running = self
            .waiting
            .iter()
            .chain(&self.running)
            .filter(|s| s.export.is_some())
            .count();
        if running + self.exports.len() >= max {
            return Err(local_err("K/V export slots are busy"));
        }
        let id = self.next_transfer;
        self.next_transfer += 1;
        let (events, rx) = mpsc::channel(1);
        let mut seq = Sequence::new(
            self,
            Job {
                prompt_ids: ids,
                start_in_thinking: false,
                enable_thinking: false,
                temperature: None,
                max_tokens: Some(0),
                top_p: None,
                top_k: None,
                seed: None,
                stop: Vec::new(),
                events,
                metadata: ResponseMetadataHandle::empty(),
                queued_at: std::time::Instant::now(),
            },
        );
        seq.export = Some(ExportTicket {
            id,
            from,
            to,
            reply,
            alive,
            _events: rx,
        });
        self.waiting.push_back(seq);
        Ok(())
    }

    /// An export sequence that reached `to` or ended: lease its blocks and
    /// answer the request (or fail it). The sequence has left `running`.
    pub(super) fn retire_export(&mut self, mut seq: Sequence, failed: bool) {
        let Some(ticket) = seq.export.take() else {
            return;
        };
        let outcome = if failed {
            Err(seq
                .pending
                .iter()
                .find_map(|item| item.as_ref().err().cloned())
                .unwrap_or_else(|| local_err("the K/V export failed")))
        } else {
            self.lease(&mut seq, &ticket)
        };
        // The sequence is done either way: its slot goes back, its blocks
        // are the lease's or are released.
        let blocks = std::mem::take(&mut seq.blocks);
        self.release_seq(&mut seq, false);
        let reply = match outcome {
            Ok((snapshot, ready)) => {
                self.exports.push(Export {
                    id: ticket.id,
                    ids: std::mem::take(&mut seq.ids),
                    blocks,
                    to: ticket.to,
                    snapshot,
                    alive: ticket.alive,
                });
                Ok(ready)
            }
            Err(e) => {
                let mut blocks = blocks;
                self.pools.release(&mut blocks);
                Err(e)
            }
        };
        let _ = ticket.reply.send(reply);
    }

    /// The snapshot and the reply of an export whose sequence `seq` reached
    /// `ticket.to`.
    fn lease(
        &mut self,
        seq: &mut Sequence,
        ticket: &ExportTicket,
    ) -> Result<(Option<Vec<DeltaRows>>, ExportReady), CucaError> {
        let to = ticket.to;
        if seq.blocks.len != to || seq.ids.len() != to {
            return Err(local_err(format!(
                "the cached prefix holds {} positions, not {to}",
                seq.blocks.len
            )));
        }
        let snapshot = if self.model.is_hybrid() {
            let found = block_keys(&seq.ids, to / BLOCK_SIZE)
                .pop()
                .and_then(|key| self.snapshots.get(&key));
            let Some(found) = found else {
                return Err(local_err(format!(
                    "no recurrent-state snapshot at position {to}"
                )));
            };
            Some(self.model.snapshot_export(found)?)
        } else {
            None
        };
        let unit = BLOCK_SIZE * self.pools.step();
        let ready = ExportReady {
            id: ticket.id,
            from: ticket.from,
            to,
            chunk: CHUNK_TOKENS.max(unit) / unit * unit,
            snapshot: snapshot.is_some(),
        };
        Ok((snapshot, ready))
    }

    /// The rows frame of the positions `a..b` of lease `id`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for an unknown lease, positions outside it or
    /// a failed read.
    pub(crate) fn export_rows(
        &mut self,
        id: u64,
        a: usize,
        b: usize,
    ) -> Result<Vec<u8>, CucaError> {
        let lease = self
            .exports
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| local_err("unknown K/V export"))?;
        if !(a < b && b <= lease.to) {
            return Err(local_err(format!(
                "positions {a}..{b} are outside the export of {}",
                lease.to
            )));
        }
        let rows = self
            .model
            .export_rows(&lease.blocks.tables, a, b, lease.to)?;
        Ok(kvframe::rows_frame(a, b, &rows))
    }

    /// The snapshot frame of lease `id`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for an unknown lease or one without a
    /// snapshot (not a hybrid model).
    pub(crate) fn export_snapshot(&self, id: u64) -> Result<Vec<u8>, CucaError> {
        let lease = self
            .exports
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| local_err("unknown K/V export"))?;
        let snapshot = lease
            .snapshot
            .as_ref()
            .ok_or_else(|| local_err("the export carries no recurrent-state snapshot"))?;
        Ok(kvframe::snapshot_frame(snapshot))
    }

    // ---- import ----

    /// Start importing `ids[from..to]`: `from` tokens are in the prefix
    /// store already. Returns the id of the frames that follow
    /// ([`Engine::import_frame`]). The import ends when its frames are all in
    /// or `alive`'s receiver is dropped.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the engine cannot take imported blocks,
    /// the range is not whole blocks of `ids`, the store no longer holds the
    /// first `from` tokens, `MAX_IMPORTS` imports run, or the pools have no
    /// room.
    pub(crate) fn import_begin(
        &mut self,
        ids: Vec<u32>,
        from: usize,
        to: usize,
        alive: oneshot::Sender<()>,
    ) -> Result<u64, CucaError> {
        if !self.imports_kv() {
            return Err(local_err(
                "this model replays a prompt for its drafter or keeps no prefix store: it takes no imported K/V",
            ));
        }
        if self.imports.len() >= MAX_IMPORTS {
            return Err(local_err("K/V import slots are busy"));
        }
        if from >= to || to >= ids.len() {
            return Err(local_err(format!(
                "an import of {from}..{to} of {} tokens leaves nothing to import or no token to compute",
                ids.len()
            )));
        }
        // The blocks `to` needs beyond the ones the store holds, next to what
        // the prompts being prefilled still owe.
        let mut owed = self.committed();
        let per_group: Vec<usize> = self
            .model
            .kv_layout()
            .groups
            .iter()
            .map(|g| to / g.span() - from / g.span())
            .collect();
        add_owed(&mut owed, &per_group);
        if !self.pools.can_give(&owed) {
            return Err(local_err("no room in the K/V pool for the import"));
        }
        let blocks = self.pools.adopt(&ids, from, to)?;
        let id = self.next_transfer;
        self.next_transfer += 1;
        let mut ids = ids;
        ids.truncate(to);
        self.imports.push(Import {
            id,
            ids,
            to,
            written: from,
            blocks,
            snapshot: None,
            alive,
        });
        Ok(id)
    }

    /// Take a rows, snapshot or end frame of import `id`; `true` once the
    /// end frame filed the prefix. A failed frame ends the import and gives
    /// its blocks back.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for an unknown import, a malformed frame, rows
    /// that do not follow the ones written or do not fit the model, an end
    /// before every position arrived, or a hybrid model's transfer without a
    /// snapshot.
    pub(crate) fn import_frame(&mut self, id: u64, frame: &[u8]) -> Result<bool, CucaError> {
        let at = self
            .imports
            .iter()
            .position(|i| i.id == id)
            .ok_or_else(|| local_err("unknown K/V import"))?;
        match self.import_step(at, frame) {
            Ok(false) => Ok(false),
            Ok(true) => {
                let import = self.imports.swap_remove(at);
                self.file_import(import);
                Ok(true)
            }
            Err(e) => {
                // A failed frame ends the import.
                let mut import = self.imports.swap_remove(at);
                self.pools.release(&mut import.blocks);
                Err(e)
            }
        }
    }

    /// Apply `frame` to the import at `at`; `true` for the end frame.
    fn import_step(&mut self, at: usize, frame: &[u8]) -> Result<bool, CucaError> {
        let (kind, payload) = kvframe::parse(frame)?;
        match kind {
            kvframe::ROWS => {
                let (a, b, rows) = kvframe::decode_rows(payload)?;
                let import = &mut self.imports[at];
                if a != import.written || b <= a || b > import.to {
                    return Err(local_err(format!(
                        "rows {a}..{b} do not follow the {} positions written of {}",
                        import.written, import.to
                    )));
                }
                self.model
                    .import_rows(&import.blocks.tables, a, b, import.to, &rows)?;
                import.written = b;
                Ok(false)
            }
            kvframe::SNAPSHOT => {
                let rows = kvframe::decode_snapshot(payload)?;
                let snapshot = self.model.snapshot_import(rows)?;
                self.imports[at].snapshot = Some(snapshot);
                Ok(false)
            }
            kvframe::END => {
                let import = &self.imports[at];
                if import.written != import.to {
                    return Err(local_err(format!(
                        "the transfer ended at position {} of {}",
                        import.written, import.to
                    )));
                }
                if self.model.is_hybrid() && import.snapshot.is_none() {
                    return Err(local_err(
                        "the transfer carries no recurrent-state snapshot",
                    ));
                }
                Ok(true)
            }
            other => Err(local_err(format!("unexpected K/V frame of kind {other}"))),
        }
    }

    /// File a completed import in the prefix store, with the snapshot of a
    /// hybrid model (checked when its frame arrived).
    fn file_import(&mut self, mut import: Import) {
        if let Some(snapshot) = import.snapshot.take()
            && let Some(key) = block_keys(&import.ids, import.to / BLOCK_SIZE).pop()
        {
            self.snapshots.insert(key, snapshot);
        }
        import.blocks.len = import.to;
        self.pools.finish(&mut import.blocks, &import.ids);
    }

    /// Drop the leases and imports nobody waits for any more; whether any
    /// went. A lease's blocks are filed in the prefix store, an import's
    /// are given back.
    pub(super) fn collect_transfers(&mut self) -> bool {
        let before = self.exports.len() + self.imports.len();
        let mut i = 0;
        while i < self.exports.len() {
            if self.exports[i].alive.is_closed() {
                let mut lease = self.exports.swap_remove(i);
                if self.batching.prefix_cache {
                    self.pools.finish(&mut lease.blocks, &lease.ids);
                } else {
                    self.pools.release(&mut lease.blocks);
                }
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.imports.len() {
            if self.imports[i].alive.is_closed() {
                let mut import = self.imports.swap_remove(i);
                self.pools.release(&mut import.blocks);
            } else {
                i += 1;
            }
        }
        before != self.exports.len() + self.imports.len()
    }

    /// Give back every lease and import: the pools were reset after a failed
    /// pass.
    pub(super) fn forget_transfers(&mut self) {
        self.exports.clear();
        self.imports.clear();
    }
}
