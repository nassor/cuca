//! A model whose first layers run on other machines (layer stages).
//!
//! The coordinator ([`ChainModel`]) runs the last layers and the head
//! itself, on its own device, and holds one [`StageTransport`] to the worker
//! of each earlier range of layers. A forward pass goes through the remote
//! stages in order, the coordinator relaying every stage's
//! [`Carry`](super::split::Carry) to the next, then runs its own layers over
//! the last carry: logits never cross the network, and the workers never talk
//! to each other. The paged K/V cache is one [`KvLayout`] everywhere: the
//! coordinator's engine allocates the blocks of a sequence once and sends the
//! same block tables to every stage, whose pools hold the layers it runs.
//!
//! # State on the workers
//!
//! A worker keeps what a whole model keeps for the layers it holds: K/V rows
//! in its pools, the recurrent state of its delta-net layers and the
//! compressor state of its hyper-connection layers, per state slot. The
//! coordinator mirrors only what it needs to keep them in step:
//!
//! - the tokens each remote holds per slot (`sent`), so a call resends only
//!   the history the worker does not have (the Engram layers hash n-grams of
//!   it);
//! - the position of each slot (`pos`), for the rollback it can answer;
//! - the ids of the recurrent-state snapshots the workers hold for it, freed
//!   when the [`ChainSnapshot`] is dropped;
//! - the restores still to be told to the workers (`restores`) and the
//!   snapshots their last call took (`taken`): both ride in forward calls,
//!   so a step costs one call per stage.
//!
//! # Failure
//!
//! Any failed call leaves the workers in an unknown state. The model is then
//! `dirty`: it resets its own slots and the next call opens a new session on
//! every worker (a reset of all their state, under a fresh session number so
//! that a stale coordinator cannot write into it). The engine, which fails
//! every running sequence and empties its pools after any failed pass, then
//! starts from nothing on every node.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use super::cpu::Forward;
use super::paged::{BlockTable, KvLayout};
use super::split::{Carry, LayerRange, StageOut};
use super::transfer::{DeltaRows, PoolRows};
use super::{Batch, Model, RecurrentSnapshot, StateSlot, StepHint, err};
use crate::error::CucaError;
use crate::provider::local::stageframe::{
    ANOTHER_SESSION, ForwardCall, Open, Restore, StageSeq, decode_forward_reply, forward_frame,
    open_frame,
};

mod http;
mod stats;
mod stream;

pub(crate) use http::HttpTransport;
pub(crate) use stats::StageCounters;
pub use stats::StageStats;

/// How the coordinator reaches one remote stage: one frame out, one frame
/// back (the protocol of [`stageframe`](crate::provider::local::stageframe)).
/// A call blocks the calling (engine) thread until the stage answers.
pub(crate) trait StageTransport: Send {
    /// Send `frame` and return the stage's answer. A transport that cannot
    /// tell whether a frame arrived may send the same frame again, once: a
    /// call of a session carries its number and the worker runs it once
    /// (see [`stageframe`](crate::provider::local::stageframe)), answering
    /// the second arrival with the first's reply.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when the stage refuses the frame as a mismatch
    /// (another file, range, sizing or session); [`CucaError::Provider`] for
    /// any other failure, including an unreachable stage.
    fn call(&self, frame: Vec<u8>) -> Result<Vec<u8>, CucaError>;

    /// What messages call the stage: `stage {url}` for an HTTP stage.
    fn name(&self) -> String;

    /// The counters of this stage's calls: the transport records every call
    /// it makes (wall time, the compute time the worker reports, bytes).
    fn counters(&self) -> Arc<StageCounters>;
}

/// What the stages must agree on with the coordinator, as the first call of
/// a session carries it (the range and the session vary per stage).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Agreement {
    /// [`model_fingerprint`](crate::provider::local::engine) of the files and
    /// the cache layout.
    pub(crate) fingerprint: u64,
    pub(crate) context_length: usize,
    pub(crate) max_sequences: usize,
    pub(crate) kv_tokens: usize,
    pub(crate) max_rows: usize,
}

/// One remote stage.
struct Remote {
    range: LayerRange,
    transport: Box<dyn StageTransport>,
    /// Per state slot, the tokens (history, then the rows of the last call)
    /// the worker holds.
    sent: Vec<Vec<u32>>,
    /// The range holds an Engram layer, which reads the token history.
    engram: bool,
    /// Snapshot ids to release on the worker with its next call.
    pending: Mutex<Vec<u64>>,
    /// The calls made in the session, which numbers the next one (see
    /// [`stageframe`](crate::provider::local::stageframe): a call sent twice
    /// runs once).
    calls: AtomicU64,
}

impl Remote {
    /// The number of the next call of the session, from 1.
    fn next_seq(&self) -> u64 {
        self.calls.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// The model of a coordinator: remote stages, then a local stage that ends
/// the model. See the module docs.
pub(crate) struct ChainModel {
    remotes: Vec<Remote>,
    /// The last stage: a CPU or GPU model over the layers from the end of the
    /// last remote range.
    local: Box<Model>,
    hybrid: bool,
    /// The model has hyper-connections, whose compressor state is not
    /// positional either.
    stateful: bool,
    max_sequences: usize,
    /// Per state slot, the positions the stages' state sits at.
    pos: Vec<usize>,
    agreement: Agreement,
    session: u64,
    /// Ids of the snapshots dropped since the last call that told the
    /// workers.
    drops: Arc<Mutex<Vec<u64>>>,
    next_id: AtomicU64,
    /// A call failed: the workers' state is unknown until the next open.
    dirty: AtomicBool,
    /// Why the stages were last marked dirty: what a sequence whose state
    /// they lost is told.
    cause: Mutex<Option<CucaError>>,
    /// Per state slot, the restore the next forward call of the slot tells
    /// the workers to make before its pass (the local stage's was made at
    /// once).
    restores: Vec<Option<Restore>>,
    /// Per state slot, the snapshot the last forward call took on the
    /// workers, until [`ChainModel::snapshot`] adds the local stage's.
    taken: Mutex<Vec<Option<Taken>>>,
}

/// A snapshot the workers took during a forward call, which the snapshot of
/// the local stage completes.
#[derive(Debug, Clone, Copy)]
struct Taken {
    id: u64,
    /// The position the slot sat at after the pass.
    pos: usize,
    /// The bytes the workers hold for it.
    remote_bytes: usize,
}

/// The recurrent state of a [`ChainModel`] slot at one position: the local
/// stage's snapshot, and the id under which each worker keeps its own.
pub(crate) struct ChainSnapshot {
    id: u64,
    /// The session of the workers that hold the snapshots.
    session: u64,
    local: Box<RecurrentSnapshot>,
    /// The local snapshot and the workers' together.
    bytes: usize,
    drops: Arc<Mutex<Vec<u64>>>,
}

impl ChainSnapshot {
    /// Bytes the snapshot holds, on every node.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for ChainSnapshot {
    fn drop(&mut self) {
        lock(&self.drops).push(self.id);
    }
}

/// What a sequence is told when the stages lost the recurrent state it was
/// running on.
const LOST_STATE: &str = "the stages lost this sequence's state";

/// What a call of an open session is told when the worker answers that the
/// session is not the one it serves: the worker restarted (or another
/// coordinator opened it) and what it held of the sequences is gone.
const RESTARTED: &str = "restarted and lost this sequence's state";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A session number no earlier session used: a worker that restarted holds
/// session 0, so a coordinator never matches a worker it did not open.
fn fresh_session() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    (clock.wrapping_add(COUNTER.fetch_add(1, Ordering::Relaxed)) << 1) | 1
}

/// The length of the longest common prefix of `a` and `b`.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl ChainModel {
    /// A chain over `local` (the stage that ends the model) and the remote
    /// stages `remotes`, in layer order from layer 0, with a session open on
    /// each.
    ///
    /// # Errors
    ///
    /// The error of the first stage that refuses to open (naming it):
    /// unreachable, or another file, range or sizing.
    pub(crate) fn connect(
        local: Model,
        remotes: Vec<(LayerRange, Box<dyn StageTransport>, bool)>,
        stateful: bool,
        agreement: Agreement,
    ) -> Result<Self, CucaError> {
        let max_sequences = agreement.max_sequences.max(1);
        let mut chain = Self {
            hybrid: local.is_hybrid(),
            stateful,
            remotes: remotes
                .into_iter()
                .map(|(range, transport, engram)| Remote {
                    range,
                    transport,
                    sent: vec![Vec::new(); max_sequences],
                    engram,
                    pending: Mutex::new(Vec::new()),
                    calls: AtomicU64::new(0),
                })
                .collect(),
            local: Box::new(local),
            max_sequences,
            pos: vec![0; max_sequences],
            agreement,
            session: 0,
            drops: Arc::default(),
            next_id: AtomicU64::new(1),
            dirty: AtomicBool::new(true),
            cause: Mutex::new(None),
            restores: vec![None; max_sequences],
            taken: Mutex::new(vec![None; max_sequences]),
        };
        chain.reopen()?;
        Ok(chain)
    }

    /// Open a new session on every worker, which resets their state, and
    /// empty the local stage's.
    fn reopen(&mut self) -> Result<(), CucaError> {
        self.dirty.store(true, Ordering::Relaxed);
        self.session = fresh_session();
        let a = self.agreement;
        for remote in &mut self.remotes {
            let frame = open_frame(&Open {
                fingerprint: a.fingerprint,
                first: remote.range.first as u32,
                end: remote.range.end as u32,
                context_length: a.context_length as u64,
                max_sequences: a.max_sequences as u64,
                kv_tokens: a.kv_tokens as u64,
                max_rows: a.max_rows as u64,
                session: self.session,
            });
            remote.transport.call(frame)?;
            remote.sent.iter_mut().for_each(Vec::clear);
            lock(&remote.pending).clear();
            remote.calls.store(0, Ordering::Relaxed);
        }
        lock(&self.drops).clear();
        for slot in 0..self.max_sequences {
            self.local.reset_slot(StateSlot(slot));
        }
        self.pos.fill(0);
        self.restores.fill(None);
        lock(&self.taken).fill(None);
        self.dirty.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// The snapshot ids to tell `remote` to drop: what was dropped since any
    /// stage was told, shared out to every stage's list. An id a pending
    /// restore names waits until the restore has been made: the snapshot may
    /// be let go of before the call that restores it.
    fn take_drops(&self, remote: &Remote) -> Vec<u64> {
        let mut fresh = std::mem::take(&mut *lock(&self.drops));
        let wanted = |id: &u64| {
            self.restores
                .iter()
                .flatten()
                .any(|r| r.id.as_ref() == Some(id))
        };
        if fresh.iter().any(wanted) {
            let (held, free): (Vec<u64>, Vec<u64>) = fresh.into_iter().partition(wanted);
            lock(&self.drops).extend(held);
            fresh = free;
        }
        if !fresh.is_empty() {
            for r in &self.remotes {
                lock(&r.pending).extend_from_slice(&fresh);
            }
        }
        std::mem::take(&mut *lock(&remote.pending))
    }

    /// [`Model::forward_batch`]: the remote stages in order, then the local
    /// one.
    ///
    /// # Errors
    ///
    /// As [`ChainModel::forward_batch_hinting`].
    pub(crate) fn forward_batch(&mut self, batch: &Batch<'_>) -> Result<Vec<Forward>, CucaError> {
        self.forward_batch_hinting(batch, &[])
    }

    /// [`Model::forward_batch_hinting`]: one call per remote stage carries
    /// the step's rows, the restores made since the last call of each slot,
    /// the snapshots the hints ask for and the Engram rows to read ahead.
    ///
    /// # Errors
    ///
    /// A failed stage call, a refused frame or a failed local pass. The model
    /// then resets its own state of every sequence of the step and reopens
    /// the stages on its next call.
    pub(crate) fn forward_batch_hinting(
        &mut self,
        batch: &Batch<'_>,
        hints: &[StepHint<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let result = self.run(batch, hints);
        if let Err(e) = &result {
            self.break_with(e);
            for row in &batch.rows {
                if row.state.0 < self.max_sequences {
                    self.local.reset_slot(row.state);
                    self.pos[row.state.0] = 0;
                }
            }
        }
        result
    }

    /// Mark the stages' state unknown: the next pass reopens them. `cause`
    /// is what a sequence that lost its state is told, unless it is itself
    /// such a refusal (the first cause stays).
    fn break_with(&self, cause: &CucaError) {
        if !cause.to_string().contains(LOST_STATE) {
            *lock(&self.cause) = Some(cause.clone());
        }
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// The error of a sequence whose recurrent state is gone with a
    /// reopened session.
    fn lost_state(&self) -> CucaError {
        match &*lock(&self.cause) {
            Some(cause) => err(format!("{LOST_STATE}: {cause}")),
            None => err(LOST_STATE),
        }
    }

    /// One call of an open session. A worker that answers `another session`
    /// has restarted (or another coordinator opened it) since this session
    /// began: what it held of the sequences is gone, and that is no
    /// configuration error. A mismatch of the first call (`open`) stays one:
    /// `reopen` calls the transport directly.
    fn session_call(remote: &Remote, frame: Vec<u8>) -> Result<Vec<u8>, CucaError> {
        remote.transport.call(frame).map_err(|e| match e {
            CucaError::Config(m) if m.contains(ANOTHER_SESSION) => {
                err(format!("{} {RESTARTED}", remote.transport.name()))
            }
            other => other,
        })
    }

    fn run(
        &mut self,
        batch: &Batch<'_>,
        hints: &[StepHint<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        // After a break, a sequence that is under way has lost what the
        // workers held of it (recurrent state, and the K/V rows of a worker
        // that restarted): it fails with the cause instead of running on a
        // session reopened under it.
        if *self.dirty.get_mut() && batch.rows.iter().any(|r| r.start_pos > 0) {
            return Err(self.lost_state());
        }
        if *self.dirty.get_mut() {
            self.reopen()?;
        }
        if self.hybrid {
            for r in &batch.rows {
                if self.pos.get(r.state.0).is_some_and(|&p| p != r.start_pos) {
                    return Err(self.lost_state());
                }
            }
        }
        for row in &batch.rows {
            if row.state.0 >= self.max_sequences {
                return Err(err(format!(
                    "state slot {} is out of range (the model has {})",
                    row.state.0, self.max_sequences
                )));
            }
        }
        let hint_of = |slot: StateSlot| hints.iter().find(|h| h.slot == slot);
        // The ids the workers keep this step's snapshots under: only a model
        // with recurrent state has any to take.
        let snapshot_ids: Vec<Option<u64>> = batch
            .rows
            .iter()
            .map(|r| {
                hint_of(r.state)
                    .filter(|h| h.snapshot && self.hybrid)
                    .map(|_| self.next_id.fetch_add(1, Ordering::Relaxed))
            })
            .collect();
        let mut remote_bytes = vec![0usize; batch.rows.len()];
        let mut carry: Option<Carry> = None;
        for k in 0..self.remotes.len() {
            let drops = self.take_drops(&self.remotes[k]);
            let remote = &mut self.remotes[k];
            let seqs = batch
                .rows
                .iter()
                .zip(&snapshot_ids)
                .map(|(r, &snapshot)| {
                    let from = common_prefix(&remote.sent[r.state.0], r.history);
                    StageSeq {
                        slot: r.state.0 as u32,
                        start_pos: r.start_pos as u64,
                        hist_from: from as u64,
                        hist: r.history[from..].to_vec(),
                        tokens: r.tokens.to_vec(),
                        want: r.want,
                        hidden: r.hidden,
                        tables: r.tables.to_vec(),
                        restore: self.restores[r.state.0],
                        snapshot,
                        prefetch: match hint_of(r.state) {
                            Some(h) if remote.engram => h.next.to_vec(),
                            _ => Vec::new(),
                        },
                    }
                })
                .collect();
            let frame = forward_frame(&ForwardCall {
                session: self.session,
                seq: remote.next_seq(),
                drops,
                exact: batch.exact,
                seqs,
                carry: carry.take(),
            })?;
            let reply = Self::session_call(remote, frame)?;
            let (next, sizes) = decode_forward_reply(&reply)?;
            let asked = snapshot_ids.iter().flatten().count();
            if sizes.len() != asked {
                return Err(err(format!(
                    "{} took {} snapshots, {asked} were asked",
                    remote.transport.name(),
                    sizes.len()
                )));
            }
            let rows_with_snapshot = snapshot_ids
                .iter()
                .enumerate()
                .filter_map(|(i, id)| id.map(|_| i));
            for (i, bytes) in rows_with_snapshot.zip(sizes) {
                remote_bytes[i] += usize::try_from(bytes).unwrap_or(usize::MAX);
            }
            carry = Some(next);
            for r in &batch.rows {
                let held = &mut remote.sent[r.state.0];
                held.clear();
                held.extend_from_slice(r.history);
                held.extend_from_slice(r.tokens);
            }
        }
        let outs = match self.local.forward_stage(batch, carry)? {
            StageOut::Done(outs) => outs,
            StageOut::Carry(_) => return Err(err("the last stage returned no logits")),
        };
        for h in hints.iter().filter(|h| !h.next.is_empty()) {
            self.local.engram_prefetch(h.history, h.next);
        }
        for (i, r) in batch.rows.iter().enumerate() {
            let end = r.start_pos + r.tokens.len();
            self.pos[r.state.0] = end;
            self.restores[r.state.0] = None;
            if let Some(id) = snapshot_ids[i] {
                let taken = Taken {
                    id,
                    pos: end,
                    remote_bytes: remote_bytes[i],
                };
                if let Some(old) = lock(&self.taken)[r.state.0].replace(taken) {
                    lock(&self.drops).push(old.id);
                }
            }
        }
        Ok(outs)
    }

    /// The architecture.
    pub(crate) fn arch(&self) -> super::arch::Arch {
        self.local.arch()
    }

    /// The block pools' layout (the whole model's, on every stage).
    pub(crate) fn kv_layout(&self) -> &KvLayout {
        self.local.kv_layout()
    }

    /// Counters and heat of the local stage's expert tiers.
    pub(crate) fn expert_tiers(&self) -> Option<Arc<super::tier::TierShared>> {
        self.local.expert_tiers()
    }

    /// The counters of the local stage's Engram table.
    pub(crate) fn engram_counters(&self) -> Option<Arc<super::engram_io::EngramCounters>> {
        self.local.engram_counters()
    }

    /// Bytes of K/V data the local stage's pools hold.
    pub(crate) fn kv_pool_bytes(&self) -> u64 {
        self.local.kv_pool_bytes()
    }

    /// Whether the model has recurrent layers.
    pub(crate) fn is_hybrid(&self) -> bool {
        self.hybrid
    }

    /// The call counters of every remote stage, in layer order, with the
    /// stage's name.
    pub(crate) fn stage_counters(&self) -> Vec<(String, Arc<StageCounters>)> {
        self.remotes
            .iter()
            .map(|r| (r.transport.name(), r.transport.counters()))
            .collect()
    }

    /// One row for a hybrid or hyper-connection model: nothing drafts on a
    /// chain, so no rollback call would have to cross the network. The local
    /// stage's figure otherwise (K/V is positional).
    pub(crate) fn rollback_rows(&self) -> usize {
        if self.hybrid || self.stateful {
            1
        } else {
            self.local.rollback_rows()
        }
    }

    /// Take `slot` back to `len` positions: an attention-only model keeps
    /// only positional K/V and returns `len`; a model with a recurrent state
    /// can stay where it is and nothing else (0 otherwise, after which the
    /// engine recomputes or fails the sequence: no state of a worker can be
    /// rewound).
    pub(crate) fn rewind(&mut self, slot: usize, len: usize) -> usize {
        if !self.hybrid && !self.stateful {
            return len;
        }
        if self.pos.get(slot) == Some(&len) {
            return len;
        }
        if slot < self.max_sequences {
            self.local.reset_slot(StateSlot(slot));
            self.pos[slot] = 0;
        }
        0
    }

    /// The zero state at position 0 for `slot` on the local stage; the
    /// workers' slots start over with the next call at position 0 or after a
    /// restore.
    pub(crate) fn reset_slot(&mut self, slot: usize) {
        self.local.reset_slot(StateSlot(slot));
        if let Some(p) = self.pos.get_mut(slot) {
            *p = 0;
        }
    }

    /// A copy of the recurrent state of `slot` on every stage: the workers
    /// took theirs, under a new id, during the forward call whose hint asked
    /// for it ([`StepHint::snapshot`]), and this adds the local stage's.
    /// `None` for an attention-only model, a broken session, a slot whose
    /// last call asked for no snapshot or one that is not at the position it
    /// took it at, and a stage that cannot copy.
    pub(crate) fn snapshot(&self, slot: usize) -> Option<RecurrentSnapshot> {
        if !self.hybrid || self.dirty.load(Ordering::Relaxed) {
            return None;
        }
        let taken = lock(&self.taken).get_mut(slot)?.take()?;
        let local = (self.pos.get(slot) == Some(&taken.pos))
            .then(|| self.local.snapshot(StateSlot(slot)))
            .flatten();
        let Some(local) = local else {
            // The workers hold a copy nobody will use.
            lock(&self.drops).push(taken.id);
            return None;
        };
        let bytes = local.bytes() + taken.remote_bytes;
        Some(RecurrentSnapshot::Chain(ChainSnapshot {
            id: taken.id,
            session: self.session,
            local: Box::new(local),
            bytes,
            drops: Arc::clone(&self.drops),
        }))
    }

    /// Make `snapshot` (or, for `None`, the zero state) the state of `slot`
    /// at position `len`: the local stage's at once, the workers' with the
    /// slot's next forward call.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a snapshot of another session or kind, a
    /// stage that fails or one that does not hold the snapshot.
    pub(crate) fn restore(
        &mut self,
        slot: usize,
        snapshot: Option<&RecurrentSnapshot>,
        len: usize,
    ) -> Result<(), CucaError> {
        if slot >= self.max_sequences {
            return Err(err(format!(
                "state slot {slot} is out of range (the model has {})",
                self.max_sequences
            )));
        }
        let held = match snapshot {
            None => None,
            Some(RecurrentSnapshot::Chain(s)) => Some(s),
            Some(_) => return Err(err("the snapshot comes from another kind of model")),
        };
        if *self.dirty.get_mut() {
            self.reopen()?;
        }
        if held.is_some_and(|s| s.session != self.session) {
            return Err(err(
                "the snapshot predates the last reconnection of the stages",
            ));
        }
        let outcome = self
            .local
            .restore(StateSlot(slot), held.map(|s| &*s.local), len);
        match &outcome {
            Ok(()) => {
                self.restores[slot] = Some(Restore {
                    id: held.map(|s| s.id),
                    len: len as u64,
                });
                self.pos[slot] = len;
            }
            Err(e) => self.break_with(e),
        }
        outcome
    }

    /// The layers of the local stage (the last), for the drafter.
    pub(crate) fn local(&self) -> &Model {
        &self.local
    }

    /// K/V rows do not move between the nodes of a chain.
    ///
    /// # Errors
    ///
    /// Always [`CucaError::Config`].
    pub(crate) fn export_rows(
        &self,
        _tables: &[BlockTable],
        _from: usize,
        _to: usize,
        _end: usize,
    ) -> Result<Vec<PoolRows>, CucaError> {
        Err(no_transfer())
    }

    /// See [`ChainModel::export_rows`].
    ///
    /// # Errors
    ///
    /// Always [`CucaError::Config`].
    pub(crate) fn import_rows(
        &mut self,
        _tables: &[BlockTable],
        _from: usize,
        _to: usize,
        _end: usize,
        _rows: &[PoolRows],
    ) -> Result<Vec<usize>, CucaError> {
        Err(no_transfer())
    }

    /// See [`ChainModel::export_rows`].
    ///
    /// # Errors
    ///
    /// Always [`CucaError::Config`].
    pub(crate) fn snapshot_export(
        &self,
        _snapshot: &ChainSnapshot,
    ) -> Result<Vec<DeltaRows>, CucaError> {
        Err(no_transfer())
    }
}

fn no_transfer() -> CucaError {
    CucaError::Config("K/V transfer is not available with layer stages".into())
}
