//! The worker side of layer stages: an engine whose model holds only the
//! layers `first..end` of a file ([`LocalModelConfig::stage`]) and answers the
//! frames of a coordinator ([`stageframe`]) instead of serving requests.
//!
//! A worker has no scheduler work: the coordinator's engine owns the block
//! tables, the prefix store and the sequences, and sends the tables of each
//! sequence with every forward call. The worker keeps what it must to run
//! the same calls the whole model would: the K/V rows of its layers in its own
//! pools (at the block ids the coordinator names), the recurrent and
//! compressor state of its layers per state slot, the tokens before each
//! slot's next position (for the Engram layers that hash n-grams of them) and
//! the recurrent-state snapshots the coordinator asked for.
//!
//! A session ties the worker to one coordinator: the first call (`open`)
//! checks that both hold the same file, the same layers and the same sizing,
//! resets everything and names the session; every other call must carry it,
//! so a coordinator that lost its connection (and opened a new session
//! elsewhere) cannot write into state it no longer owns.
//!
//! Every call after `open` carries its number in the session. A transport
//! that does not know whether a frame arrived sends it again, and the worker
//! runs a call once: it keeps the reply to the last call and answers the same
//! number with it, without running anything (`Engine::once`). A pass is not
//! repeatable otherwise: the recurrent state of a hybrid model has moved on
//! (the worker refuses the second run with a position mismatch), and so has
//! the compressor state of a hyper-connection model (refused at an odd
//! position). The kept reply is at most one frame (what one pass carries,
//! bounded by `max_step_tokens`) and is replaced by the next call.
//!
//! [`LocalModelConfig::stage`]: crate::provider::local::LocalModelConfig::stage
//! [`stageframe`]: crate::provider::local::stageframe

use bytes::Bytes;

use super::Engine;
use crate::error::CucaError;
use crate::provider::local::model::arch::Arch;
use crate::provider::local::model::cpu::dflash;
use crate::provider::local::model::err as local_err;
use crate::provider::local::model::params::Hparams;
use crate::provider::local::model::split::{LayerRange, StageOut, legal_cuts};
use crate::provider::local::model::{Batch, RecurrentSnapshot, SeqRows, StateSlot};
use crate::provider::local::stageframe::{self, Call, ForwardCall, Open};
use crate::provider::local::{LocalModelConfig, StageRange};

/// What a stage worker keeps between calls, see the module docs.
pub(super) struct StageHost {
    range: LayerRange,
    /// The session the worker serves; `0` before the first `open`.
    session: u64,
    /// Per state slot, the tokens before the slot's next position.
    history: Vec<Vec<u32>>,
    /// Snapshots a coordinator asked for, oldest first, by the id it chose.
    snapshots: Vec<(u64, RecurrentSnapshot)>,
    /// The number of the last call the session ran (`0` before any).
    last_seq: u64,
    /// The reply to that call, kept to answer it again if it comes again.
    last_reply: Option<Bytes>,
}

impl StageHost {
    pub(super) fn new(range: LayerRange, max_sequences: usize) -> Self {
        Self {
            range,
            session: 0,
            history: vec![Vec::new(); max_sequences.max(1)],
            snapshots: Vec::new(),
            last_seq: 0,
            last_reply: None,
        }
    }

    /// The host if `session` is the one it serves, with the snapshots of
    /// `drops` released.
    fn enter<'a>(
        host: &'a mut Option<Self>,
        session: u64,
        drops: &[u64],
    ) -> Result<&'a mut Self, CucaError> {
        let host = host
            .as_mut()
            .ok_or_else(|| CucaError::Config("this model is not a layer stage".into()))?;
        if host.session != session {
            return Err(CucaError::Config(stageframe::ANOTHER_SESSION.into()));
        }
        if !drops.is_empty() {
            host.snapshots.retain(|(id, _)| !drops.contains(id));
        }
        Ok(host)
    }

    fn slot(&self, slot: u32) -> Result<usize, CucaError> {
        let slot = slot as usize;
        if slot >= self.history.len() {
            return Err(local_err(format!(
                "state slot {slot} is out of range (the stage has {})",
                self.history.len()
            )));
        }
        Ok(slot)
    }
}

/// The checks of a stage configuration that need the model's hyperparameters:
/// the layers the stages hold, whether every boundary can be cut, and which
/// drafter files cannot pair with a chain.
///
/// # Errors
///
/// [`CucaError::Config`] naming the cause.
pub(super) fn check_stages(
    hp: &Hparams,
    config: &LocalModelConfig,
    drafter_arch: &str,
) -> Result<(), CucaError> {
    let (arch, n) = (hp.arch.name(), hp.n_layer);
    let mut boundaries = Vec::new();
    if let Some(StageRange { first, layers }) = config.stage {
        let end = first + layers;
        if end >= n {
            return Err(CucaError::Config(format!(
                "stage {first}..{end} must end before the last layer of `{arch}` ({n} layers)"
            )));
        }
        if first > 0 {
            boundaries.push(first);
        }
        boundaries.push(end);
    } else if !config.remote_stages.is_empty() {
        let held: usize = config.remote_stages.iter().map(|s| s.layers).sum();
        if held >= n {
            return Err(CucaError::Config(format!(
                "the remote stages hold {held} of the {n} layers of `{arch}`: the last stage needs at least one"
            )));
        }
        let mut at = 0;
        for stage in &config.remote_stages {
            at += stage.layers;
            boundaries.push(at);
        }
        if drafter_arch == Arch::Mistral4Eagle.name() || drafter_arch == dflash::ARCH {
            return Err(CucaError::Config(format!(
                "the {drafter_arch} drafter cannot be combined with `remote_stage`"
            )));
        }
    }
    let legal = legal_cuts(hp);
    if let Some(b) = boundaries.iter().find(|b| !legal.contains(b)) {
        return Err(CucaError::Config(format!(
            "a stage boundary at layer {b} cuts `{arch}` between layers that share compressed rows or indexer picks; legal boundaries: {}",
            legal
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(())
}

impl Engine {
    /// Whether this engine is a layer-stage worker: it serves
    /// [`Engine::stage_call`] only.
    pub(crate) fn is_stage(&self) -> bool {
        self.stage.is_some()
    }

    /// An `open` frame that matches this worker and names `session`: what a
    /// coordinator other than the one it serves sends (tests).
    #[cfg(test)]
    pub(super) fn foreign_open(&self, session: u64) -> Vec<u8> {
        let range = self
            .stage
            .as_ref()
            .map(|h| h.range)
            .unwrap_or(LayerRange { first: 0, end: 0 });
        stageframe::open_frame(&Open {
            fingerprint: self.fingerprint,
            first: range.first as u32,
            end: range.end as u32,
            context_length: self.context_length as u64,
            max_sequences: self.sizing.max_sequences as u64,
            kv_tokens: self.sizing.kv_tokens as u64,
            max_rows: self.sizing.max_rows as u64,
            session,
        })
    }

    /// Answer one frame of a coordinator ([`stageframe`]).
    ///
    /// A call of the session runs once however often it arrives (see
    /// [`Engine::once`]).
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for a coordinator that does not match this
    /// worker (another file, layers or sizing, or another session), and for a
    /// call on an engine that is not a stage; [`CucaError::Provider`] for a
    /// malformed frame, an unknown slot or snapshot, a call older than the
    /// last one or a failed pass.
    pub(crate) fn stage_call(&mut self, frame: &[u8]) -> Result<Bytes, CucaError> {
        if self.stage.is_none() {
            return Err(CucaError::Config("this model is not a layer stage".into()));
        }
        match stageframe::decode(frame)? {
            Call::Open(open) => self.stage_open(&open).map(Bytes::from),
            Call::Forward(call) => {
                let (session, seq) = (call.session, call.seq);
                self.once(session, seq, |engine| engine.stage_forward(call))
            }
            Call::Snapshot {
                session,
                seq,
                drops,
                slot,
                id,
            } => self.once(session, seq, |engine| {
                engine.stage_snapshot(session, &drops, slot, id)
            }),
            Call::Restore {
                session,
                seq,
                drops,
                slot,
                id,
                len,
            } => self.once(session, seq, |engine| {
                engine.stage_restore(session, &drops, slot, id, len)
            }),
            Call::Prefetch {
                session,
                seq,
                slot,
                tokens,
            } => self.once(session, seq, |engine| {
                engine.stage_prefetch(session, slot, &tokens)
            }),
        }
    }

    /// Run call number `seq` of `session` with `run`, unless the session ran
    /// it already: then answer with the reply it kept, and run nothing. A
    /// transport sends a frame again when it does not know whether the first
    /// attempt arrived, so a call may come twice, and a pass cannot run twice
    /// everywhere (see the module docs). Calls come in order, one at a time,
    /// so remembering the last is enough; a call older than that is refused
    /// (a stray frame of a connection that was given up on, when the
    /// coordinator has gone on). The kept reply is one frame, at most what
    /// the call answered.
    fn once(
        &mut self,
        session: u64,
        seq: u64,
        run: impl FnOnce(&mut Self) -> Result<Vec<u8>, CucaError>,
    ) -> Result<Bytes, CucaError> {
        let Some(host) = self.stage.as_ref() else {
            return Err(CucaError::Config("this model is not a layer stage".into()));
        };
        if host.session != session {
            return Err(CucaError::Config(stageframe::ANOTHER_SESSION.into()));
        }
        if seq <= host.last_seq {
            return match (&host.last_reply, seq == host.last_seq) {
                (Some(reply), true) => Ok(reply.clone()),
                _ => Err(local_err(format!(
                    "call {seq} is not newer than the stage's last call {}",
                    host.last_seq
                ))),
            };
        }
        let reply = Bytes::from(run(self)?);
        if let Some(host) = self.stage.as_mut() {
            host.last_seq = seq;
            host.last_reply = Some(reply.clone());
        }
        Ok(reply)
    }

    /// `snapshot`: copy the recurrent state of a slot under the caller's id.
    fn stage_snapshot(
        &mut self,
        session: u64,
        drops: &[u64],
        slot: u32,
        id: u64,
    ) -> Result<Vec<u8>, CucaError> {
        let host = StageHost::enter(&mut self.stage, session, drops)?;
        let slot = host.slot(slot)?;
        let snapshot = self
            .model
            .snapshot(StateSlot(slot))
            .ok_or_else(|| local_err("the stage holds no recurrent state to copy"))?;
        let bytes = snapshot.bytes() as u64;
        host.snapshots.retain(|(held, _)| *held != id);
        host.snapshots.push((id, snapshot));
        // The coordinator keeps at most a snapshot per slot and per
        // prefix-store entry, and tells the worker when it lets go
        // of one; the cap bounds a coordinator that never does.
        let cap = self.sizing.max_sequences + 5;
        if host.snapshots.len() > cap {
            host.snapshots.drain(..host.snapshots.len() - cap);
        }
        Ok(stageframe::bytes_reply(bytes))
    }

    /// `restore`: make a snapshot (or the zero state) the state of a slot,
    /// at position `len`.
    fn stage_restore(
        &mut self,
        session: u64,
        drops: &[u64],
        slot: u32,
        id: Option<u64>,
        len: u64,
    ) -> Result<Vec<u8>, CucaError> {
        let host = StageHost::enter(&mut self.stage, session, drops)?;
        let slot = host.slot(slot)?;
        let snapshot = match id {
            None => None,
            Some(id) => Some(
                &host
                    .snapshots
                    .iter()
                    .find(|(held, _)| *held == id)
                    .ok_or_else(|| local_err(format!("the stage holds no snapshot {id}")))?
                    .1,
            ),
        };
        let len = usize::try_from(len)
            .map_err(|_| local_err(format!("position {len} is out of range")))?;
        self.model.restore(StateSlot(slot), snapshot, len)?;
        Ok(stageframe::ok_reply())
    }

    /// `prefetch`: start reading the Engram rows of the next tokens of a
    /// slot.
    fn stage_prefetch(
        &mut self,
        session: u64,
        slot: u32,
        tokens: &[u32],
    ) -> Result<Vec<u8>, CucaError> {
        let host = StageHost::enter(&mut self.stage, session, &[])?;
        let slot = host.slot(slot)?;
        self.model
            .engram_prefetch(StateSlot(slot), &host.history[slot], tokens);
        Ok(stageframe::ok_reply())
    }

    /// `open`: check the coordinator against this worker, then start the
    /// session over: every slot, history and snapshot is forgotten, and so
    /// is the last call. Opening twice with the same arguments leaves the
    /// same state, so a repeated `open` needs no number.
    fn stage_open(&mut self, open: &Open) -> Result<Vec<u8>, CucaError> {
        let Some(host) = self.stage.as_mut() else {
            return Err(CucaError::Config("this model is not a layer stage".into()));
        };
        let LayerRange { first, end } = host.range;
        if open.first as usize != first || open.end as usize != end {
            return Err(CucaError::Config(format!(
                "this stage holds layers {first}..{end}, the caller wants {}..{}",
                open.first, open.end
            )));
        }
        let sizing = self.sizing;
        for (what, here, there) in [
            ("context_length", self.context_length, open.context_length),
            ("max_sequences", sizing.max_sequences, open.max_sequences),
            ("kv_tokens", sizing.kv_tokens, open.kv_tokens),
            ("max_rows", sizing.max_rows, open.max_rows),
        ] {
            if here as u64 != there {
                return Err(CucaError::Config(format!(
                    "{what} is {here} here and {there} at the caller"
                )));
            }
        }
        if open.fingerprint != self.fingerprint {
            return Err(CucaError::Config(
                "the caller's model is not the same file".into(),
            ));
        }
        if open.session == 0 {
            return Err(local_err("session 0 is reserved"));
        }
        for (slot, held) in host.history.iter_mut().enumerate() {
            held.clear();
            self.model.reset_slot(StateSlot(slot));
        }
        host.snapshots.clear();
        host.session = open.session;
        host.last_seq = 0;
        host.last_reply = None;
        Ok(stageframe::ok_reply())
    }

    /// `forward`: bring each slot's token history to the call's start, run
    /// the layers over the carry of the stage before, answer with the carry
    /// for the stage after.
    fn stage_forward(&mut self, call: ForwardCall) -> Result<Vec<u8>, CucaError> {
        let host = StageHost::enter(&mut self.stage, call.session, &call.drops)?;
        for seq in &call.seqs {
            let slot = host.slot(seq.slot)?;
            let held = &mut host.history[slot];
            let from = usize::try_from(seq.hist_from).unwrap_or(usize::MAX);
            if from > held.len() {
                return Err(local_err(format!(
                    "the stage holds {} tokens of slot {slot}, the caller resumes at {from}",
                    held.len()
                )));
            }
            held.truncate(from);
            held.extend_from_slice(&seq.hist);
            if held.len() as u64 != seq.start_pos {
                return Err(local_err(format!(
                    "slot {slot}: the stage holds {} tokens before position {}",
                    held.len(),
                    seq.start_pos
                )));
            }
        }
        let host = &*host;
        let batch = Batch {
            rows: call
                .seqs
                .iter()
                .map(|s| SeqRows {
                    tokens: &s.tokens,
                    start_pos: s.start_pos as usize,
                    history: &host.history[s.slot as usize],
                    tables: &s.tables,
                    state: StateSlot(s.slot as usize),
                    want: s.want,
                    hidden: s.hidden,
                })
                .collect(),
            exact: call.exact,
        };
        let out = self.model.forward_stage(&batch, call.carry)?;
        drop(batch);
        if let Some(host) = self.stage.as_mut() {
            for seq in &call.seqs {
                host.history[seq.slot as usize].extend_from_slice(&seq.tokens);
            }
        }
        match out {
            StageOut::Carry(carry) => Ok(stageframe::carry_bytes(&carry)),
            StageOut::Done(_) => Err(local_err("a stage worker holds no head")),
        }
    }
}
