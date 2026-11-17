//! The frames of the layer-stage protocol, on top of the codec of
//! [`wire`](super::wire).
//!
//! A coordinator posts one frame per call to a stage worker and reads one
//! frame back. Every request frame starts with a `u8` op; the answer has no
//! op (its shape follows from the request). There are two calls: `open`,
//! once per session, and `forward`, once per step. Everything else a step
//! needs from a worker rides in the step's forward frame, so a step costs
//! one round trip per stage:
//!
//! - open (1): `fingerprint u64`, `first u32`, `end u32`, `context_length
//!   u64`, `max_sequences u64`, `kv_tokens u64`, `max_rows u64`, `session u64`,
//!   `carry u8` (the [`CarryDtype`] the session's carries travel in: 0 `f32`,
//!   1 `bf16`, 2 `f16`); answered by [`ok_reply`];
//! - forward (2): `session u64`, `seq u64`, `drops` (`count u32`, then `u64`
//!   each: recurrent-state snapshots the worker releases), `exact u8`,
//!   `count u32` sequences, then per sequence `slot u32`, `start_pos u64`,
//!   `hist_from u64`, `hist` and `tokens` as `u32` slices, `want u8` (0
//!   none, 1 last, 2 all), `hidden u8` (0 none, 1 post-norm, 2 post-norm of
//!   every row), `groups u32` and each block table as a `u32` slice, then
//!   what the sequence asks of the worker around the pass: `restore u8`
//!   (1: before the pass, make the slot's recurrent state a snapshot or the
//!   zero state, `has_id u8`, `id u64` (0 when absent), `len u64`),
//!   `snapshot u8` and `id u64` (1: after the pass, keep a copy of the
//!   slot's recurrent state under that id), `prefetch` as a `u32` slice
//!   (after the pass, start reading the Engram rows of these next tokens of
//!   the slot); then `has_carry u8` and, when set, the carry inline (a `dtype
//!   u8`, then the residual stream, the optional per-layer and pre inputs, and
//!   the mirrored K/V rows, every float in that type); answered by the carry
//!   in the same encoding, then `count u32` and one `u64` per snapshot taken,
//!   in sequence order: the bytes it holds ([`forward_reply`]).
//!
//! # The number type of a carry
//!
//! A worker answers in the type its session's `open` named, and refuses a
//! carry written in another. `f32` is bit-exact. `bf16` and `f16` write each
//! float in two bytes, rounded to nearest even, so a chain's logits differ a
//! little from the whole model's; `f16` refuses a finite value above 65504
//! (the encoder fails the call) rather than send infinity. Integers (token
//! ids, block tables, positions) are never rounded.
//!
//! # Repeating a call
//!
//! A transport may send a frame again when it cannot tell whether the first
//! attempt arrived (a connection that died before any answer came back). The
//! `seq` of every call after `open` makes that safe: the coordinator numbers
//! them 1, 2, 3, ... within a session, and a worker answers a call it already
//! ran with the reply it kept, without running it again. That matters
//! because a forward call is not repeatable on its own: a recurrent layer
//! has advanced past the pass's start, so a hybrid model refuses a second
//! run, and so does a hyper-connection model at an odd position (its
//! compressor state sits at the pass's end). `open` is repeatable as it is:
//! it resets the worker to the same state.
//!
//! Every count a decoder reads is checked against the bytes the frame still
//! holds before anything is allocated for it.

use super::CarryDtype;
use super::model::Want;
use super::model::cpu::Hidden;
use super::model::err as local_err;
use super::model::split::{Carry, MirrorRows};
use super::wire::{Reader, Writer};
use crate::error::CucaError;

pub(crate) const OP_OPEN: u8 = 1;
pub(crate) const OP_FORWARD: u8 = 2;

/// What a worker answers (a [`CucaError::Config`]) to a call of another
/// session than the one it serves: a worker that restarted, or one a newer
/// coordinator opened. The coordinator tells a restart from a mismatch at
/// `open` by the call it was a reply to.
pub(crate) const ANOTHER_SESSION: &str = "the stage belongs to another session";

/// The first call of a session: what the worker must be and hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Open {
    /// Fingerprint of the model files the coordinator loaded.
    pub fingerprint: u64,
    /// The first layer the worker runs.
    pub first: u32,
    /// One past the last layer the worker runs.
    pub end: u32,
    pub context_length: u64,
    pub max_sequences: u64,
    pub kv_tokens: u64,
    pub max_rows: u64,
    pub session: u64,
    /// The number type the session's carries travel in, both directions.
    pub carry: CarryDtype,
}

/// One sequence of a forward call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StageSeq {
    pub slot: u32,
    pub start_pos: u64,
    /// Position of `hist[0]`: the worker's cache of the slot is rebuilt from
    /// there when it differs from the coordinator's.
    pub hist_from: u64,
    pub hist: Vec<u32>,
    pub tokens: Vec<u32>,
    pub want: Want,
    pub hidden: Hidden,
    /// Per attention group, the block table of the sequence.
    pub tables: Vec<Vec<u32>>,
    /// Before the pass: the recurrent state the slot starts from.
    pub restore: Option<Restore>,
    /// After the pass: keep a copy of the slot's recurrent state under this
    /// id.
    pub snapshot: Option<u64>,
    /// After the pass: the next tokens of the slot, whose Engram rows the
    /// worker starts reading.
    pub prefetch: Vec<u32>,
}

/// The recurrent state a slot starts a pass from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Restore {
    /// The snapshot the worker holds under this id; `None` is the zero
    /// state.
    pub id: Option<u64>,
    /// The position the state sits at.
    pub len: u64,
}

/// A forward pass of a stage.
pub(crate) struct ForwardCall {
    pub session: u64,
    /// The call's number in the session, from 1.
    pub seq: u64,
    /// Slots whose state the worker releases before the pass.
    pub drops: Vec<u64>,
    pub exact: bool,
    pub seqs: Vec<StageSeq>,
    /// What the previous stage handed over; `None` for the first stage.
    pub carry: Option<Carry>,
    /// The type `carry` is written in; `F32` when there is none.
    pub carry_dtype: CarryDtype,
}

/// A decoded request frame.
pub(crate) enum Call {
    Open(Open),
    Forward(ForwardCall),
}

/// A length as the `u32` a count field holds. A frame never holds 2^32
/// entries of anything; saturating keeps the encoder infallible and the
/// decoder refuses the frame it makes.
fn count32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// A decoded count of entries of at least `min` bytes each, refused when the
/// frame cannot hold that many.
fn bounded(r: &Reader<'_>, count: u32, min: usize, what: &str) -> Result<usize, CucaError> {
    let n = usize::try_from(count).unwrap_or(usize::MAX);
    if n.saturating_mul(min) > r.remaining() {
        return Err(local_err(format!(
            "stage frame claims {n} {what} but holds {} bytes",
            r.remaining()
        )));
    }
    Ok(n)
}

fn put_drops(w: &mut Writer, drops: &[u64]) {
    w.put_u32(count32(drops.len()));
    for d in drops {
        w.put_u64(*d);
    }
}

fn get_drops(r: &mut Reader<'_>) -> Result<Vec<u64>, CucaError> {
    let count = r.get_u32()?;
    let n = bounded(r, count, 8, "drops")?;
    (0..n).map(|_| r.get_u64()).collect()
}

fn want_byte(want: Want) -> u8 {
    match want {
        Want::None => 0,
        Want::Last => 1,
        Want::All => 2,
    }
}

fn want_from(byte: u8) -> Result<Want, CucaError> {
    match byte {
        0 => Ok(Want::None),
        1 => Ok(Want::Last),
        2 => Ok(Want::All),
        other => Err(local_err(format!(
            "unknown want byte {other} in stage frame"
        ))),
    }
}

fn hidden_byte(hidden: Hidden) -> Result<u8, CucaError> {
    match hidden {
        Hidden::None => Ok(0),
        Hidden::PostNorm => Ok(1),
        Hidden::PostNormAll => Ok(2),
        Hidden::Layers(_) => Err(CucaError::Config(
            "a layer stage cannot return per-layer hidden rows (Hidden::Layers)".to_owned(),
        )),
    }
}

fn hidden_from(byte: u8) -> Result<Hidden, CucaError> {
    match byte {
        0 => Ok(Hidden::None),
        1 => Ok(Hidden::PostNorm),
        2 => Ok(Hidden::PostNormAll),
        other => Err(local_err(format!(
            "unknown hidden byte {other} in stage frame"
        ))),
    }
}

fn put_opt_floats(w: &mut Writer, v: Option<&[f32]>, dtype: CarryDtype) -> Result<(), CucaError> {
    match v {
        Some(v) => {
            w.put_u8(1);
            w.put_floats(v, dtype)
        }
        None => {
            w.put_u8(0);
            Ok(())
        }
    }
}

fn get_opt_floats(
    r: &mut Reader<'_>,
    dtype: CarryDtype,
    what: &str,
) -> Result<Option<Vec<f32>>, CucaError> {
    match r.get_u8()? {
        0 => Ok(None),
        1 => Ok(Some(r.get_floats(dtype)?)),
        other => Err(local_err(format!(
            "stage frame {what} flag is {other}, not 0 or 1"
        ))),
    }
}

fn put_carry(w: &mut Writer, carry: &Carry, dtype: CarryDtype) -> Result<(), CucaError> {
    w.put_u8(dtype.tag());
    w.put_floats(&carry.hidden, dtype)?;
    put_opt_floats(w, carry.per_layer.as_deref(), dtype)?;
    put_opt_floats(w, carry.pre.as_deref(), dtype)?;
    w.put_u32(count32(carry.kv.len()));
    for m in &carry.kv {
        w.put_u32(count32(m.layer));
        w.put_u32(count32(m.seqs.len()));
        for (k, v) in &m.seqs {
            w.put_floats(k, dtype)?;
            w.put_floats(v, dtype)?;
        }
    }
    Ok(())
}

fn get_carry(r: &mut Reader<'_>) -> Result<(CarryDtype, Carry), CucaError> {
    let dtype = CarryDtype::from_tag(r.get_u8()?)?;
    let hidden = r.get_floats(dtype)?;
    let per_layer = get_opt_floats(r, dtype, "per-layer")?;
    let pre = get_opt_floats(r, dtype, "pre")?;
    let layers = r.get_u32()?;
    // A layer is at least its `layer` and `seqs` counts.
    let n_layers = bounded(r, layers, 8, "mirror layers")?;
    let mut kv = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let layer = usize::try_from(r.get_u32()?).unwrap_or(usize::MAX);
        let seqs = r.get_u32()?;
        // A sequence is at least its two empty slices.
        let n_seqs = bounded(r, seqs, 16, "mirror sequences")?;
        let mut rows = Vec::with_capacity(n_seqs);
        for _ in 0..n_seqs {
            let k = r.get_floats(dtype)?;
            let v = r.get_floats(dtype)?;
            rows.push((k, v));
        }
        kv.push(MirrorRows { layer, seqs: rows });
    }
    Ok((
        dtype,
        Carry {
            hidden,
            per_layer,
            pre,
            kv,
        },
    ))
}

/// The standalone encoding of a carry in `dtype`: what a forward frame
/// carries inline, and the first part of its answer.
///
/// # Errors
///
/// [`CucaError::Provider`] for a value `F16` cannot hold.
#[cfg(test)]
pub(crate) fn carry_bytes_as(carry: &Carry, dtype: CarryDtype) -> Result<Vec<u8>, CucaError> {
    let mut w = Writer::new();
    put_carry(&mut w, carry, dtype)?;
    Ok(w.into_bytes())
}

/// [`carry_bytes_as`] in `F32`.
#[cfg(test)]
pub(crate) fn carry_bytes(carry: &Carry) -> Vec<u8> {
    carry_bytes_as(carry, CarryDtype::F32).unwrap_or_default()
}

/// The carry of a standalone carry encoding, whatever type it is written
/// in, and that type.
///
/// # Errors
///
/// [`CucaError::Provider`] for a truncated or malformed body, or one with
/// bytes left over.
#[cfg(test)]
pub(crate) fn decode_carry_as(bytes: &[u8]) -> Result<(CarryDtype, Carry), CucaError> {
    let mut r = Reader::new(bytes);
    let got = get_carry(&mut r)?;
    finish(&r)?;
    Ok(got)
}

/// The carry of a standalone carry encoding.
///
/// # Errors
///
/// As [`decode_carry_as`].
#[cfg(test)]
pub(crate) fn decode_carry(bytes: &[u8]) -> Result<Carry, CucaError> {
    decode_carry_as(bytes).map(|(_, carry)| carry)
}

/// The answer of a forward call: the carry for the stage after, and the
/// bytes of each snapshot the call took (one per sequence that asked for
/// one, in sequence order).
///
/// # Errors
///
/// [`CucaError::Provider`] for a value `dtype` cannot hold (an `F16` carry
/// with a finite value above 65504).
pub(crate) fn forward_reply(
    carry: &Carry,
    snapshot_bytes: &[u64],
    dtype: CarryDtype,
) -> Result<Vec<u8>, CucaError> {
    let mut w = Writer::new();
    put_carry(&mut w, carry, dtype)?;
    w.put_u32(count32(snapshot_bytes.len()));
    for b in snapshot_bytes {
        w.put_u64(*b);
    }
    Ok(w.into_bytes())
}

/// The carry and the snapshot sizes of a forward answer.
///
/// # Errors
///
/// [`CucaError::Provider`] for a truncated or malformed body, or one with
/// bytes left over.
pub(crate) fn decode_forward_reply(
    bytes: &[u8],
    expect: CarryDtype,
) -> Result<(Carry, Vec<u64>), CucaError> {
    let mut r = Reader::new(bytes);
    let (dtype, carry) = get_carry(&mut r)?;
    if dtype != expect {
        return Err(local_err(format!(
            "the stage answered in {}, the session carries {}",
            dtype.name(),
            expect.name()
        )));
    }
    let count = r.get_u32()?;
    let n = bounded(&r, count, 8, "snapshot sizes")?;
    let sizes = (0..n).map(|_| r.get_u64()).collect::<Result<Vec<_>, _>>()?;
    finish(&r)?;
    Ok((carry, sizes))
}

/// The (empty) answer of open.
pub(crate) fn ok_reply() -> Vec<u8> {
    Vec::new()
}

fn finish(r: &Reader<'_>) -> Result<(), CucaError> {
    if r.is_empty() {
        Ok(())
    } else {
        Err(local_err(format!(
            "stage frame has {} bytes after its end",
            r.remaining()
        )))
    }
}

pub(crate) fn open_frame(open: &Open) -> Vec<u8> {
    let mut w = Writer::with_capacity(1 + 8 + 4 + 4 + 8 * 5 + 1);
    w.put_u8(OP_OPEN);
    w.put_u64(open.fingerprint);
    w.put_u32(open.first);
    w.put_u32(open.end);
    w.put_u64(open.context_length);
    w.put_u64(open.max_sequences);
    w.put_u64(open.kv_tokens);
    w.put_u64(open.max_rows);
    w.put_u64(open.session);
    w.put_u8(open.carry.tag());
    w.into_bytes()
}

/// A forward call as a frame.
///
/// # Errors
///
/// [`CucaError::Config`] for a sequence that asks for [`Hidden::Layers`],
/// which the wire cannot carry.
pub(crate) fn forward_frame(call: &ForwardCall) -> Result<Vec<u8>, CucaError> {
    let mut w = Writer::new();
    w.put_u8(OP_FORWARD);
    w.put_u64(call.session);
    w.put_u64(call.seq);
    put_drops(&mut w, &call.drops);
    w.put_u8(u8::from(call.exact));
    w.put_u32(count32(call.seqs.len()));
    for s in &call.seqs {
        w.put_u32(s.slot);
        w.put_u64(s.start_pos);
        w.put_u64(s.hist_from);
        w.put_u32s(&s.hist);
        w.put_u32s(&s.tokens);
        w.put_u8(want_byte(s.want));
        w.put_u8(hidden_byte(s.hidden)?);
        w.put_u32(count32(s.tables.len()));
        for t in &s.tables {
            w.put_u32s(t);
        }
        match s.restore {
            Some(r) => {
                w.put_u8(1);
                w.put_u8(u8::from(r.id.is_some()));
                w.put_u64(r.id.unwrap_or(0));
                w.put_u64(r.len);
            }
            None => w.put_u8(0),
        }
        w.put_u8(u8::from(s.snapshot.is_some()));
        w.put_u64(s.snapshot.unwrap_or(0));
        w.put_u32s(&s.prefetch);
    }
    match &call.carry {
        Some(carry) => {
            w.put_u8(1);
            put_carry(&mut w, carry, call.carry_dtype)?;
        }
        None => w.put_u8(0),
    }
    Ok(w.into_bytes())
}

fn decode_open(r: &mut Reader<'_>) -> Result<Open, CucaError> {
    Ok(Open {
        fingerprint: r.get_u64()?,
        first: r.get_u32()?,
        end: r.get_u32()?,
        context_length: r.get_u64()?,
        max_sequences: r.get_u64()?,
        kv_tokens: r.get_u64()?,
        max_rows: r.get_u64()?,
        session: r.get_u64()?,
        carry: CarryDtype::from_tag(r.get_u8()?)?,
    })
}

fn decode_seq(r: &mut Reader<'_>) -> Result<StageSeq, CucaError> {
    let slot = r.get_u32()?;
    let start_pos = r.get_u64()?;
    let hist_from = r.get_u64()?;
    let hist = r.get_u32s()?;
    let tokens = r.get_u32s()?;
    let want = want_from(r.get_u8()?)?;
    let hidden = hidden_from(r.get_u8()?)?;
    let groups = r.get_u32()?;
    // A table is at least its count.
    let n_groups = bounded(r, groups, 8, "block tables")?;
    let mut tables = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        tables.push(r.get_u32s()?);
    }
    let restore = match r.get_u8()? {
        0 => None,
        1 => {
            let has_id = match r.get_u8()? {
                0 => false,
                1 => true,
                other => {
                    return Err(local_err(format!(
                        "stage frame has_id flag is {other}, not 0 or 1"
                    )));
                }
            };
            let id = r.get_u64()?;
            let len = r.get_u64()?;
            Some(Restore {
                id: has_id.then_some(id),
                len,
            })
        }
        other => {
            return Err(local_err(format!(
                "stage frame restore flag is {other}, not 0 or 1"
            )));
        }
    };
    let snapshot = match r.get_u8()? {
        0 => {
            r.get_u64()?;
            None
        }
        1 => Some(r.get_u64()?),
        other => {
            return Err(local_err(format!(
                "stage frame snapshot flag is {other}, not 0 or 1"
            )));
        }
    };
    let prefetch = r.get_u32s()?;
    Ok(StageSeq {
        slot,
        start_pos,
        hist_from,
        hist,
        tokens,
        want,
        hidden,
        tables,
        restore,
        snapshot,
        prefetch,
    })
}

fn decode_forward(r: &mut Reader<'_>) -> Result<ForwardCall, CucaError> {
    let session = r.get_u64()?;
    let seq = r.get_u64()?;
    let drops = get_drops(r)?;
    let exact = match r.get_u8()? {
        0 => false,
        1 => true,
        other => {
            return Err(local_err(format!(
                "stage frame exact flag is {other}, not 0 or 1"
            )));
        }
    };
    let count = r.get_u32()?;
    // A sequence is at least its fixed fields, six empty counts and flags.
    let n_seqs = bounded(r, count, 60, "sequences")?;
    let mut seqs = Vec::with_capacity(n_seqs);
    for _ in 0..n_seqs {
        seqs.push(decode_seq(r)?);
    }
    let (carry, carry_dtype) = match r.get_u8()? {
        0 => (None, CarryDtype::F32),
        1 => {
            let (dtype, carry) = get_carry(r)?;
            (Some(carry), dtype)
        }
        other => {
            return Err(local_err(format!(
                "stage frame carry flag is {other}, not 0 or 1"
            )));
        }
    };
    Ok(ForwardCall {
        session,
        seq,
        drops,
        exact,
        seqs,
        carry,
        carry_dtype,
    })
}

/// A request frame as the call it carries.
///
/// # Errors
///
/// [`CucaError::Provider`] for an unknown op, a truncated frame, a count the
/// frame cannot hold, an unknown want or hidden byte, or bytes after the end.
pub(crate) fn decode(frame: &[u8]) -> Result<Call, CucaError> {
    let mut r = Reader::new(frame);
    let call = match r.get_u8()? {
        OP_OPEN => Call::Open(decode_open(&mut r)?),
        OP_FORWARD => Call::Forward(decode_forward(&mut r)?),
        other => return Err(local_err(format!("unknown stage frame op {other}"))),
    };
    finish(&r)?;
    Ok(call)
}

#[cfg(test)]
mod tests {
    use super::super::model::cpu::LayerSet;
    use super::*;

    fn open() -> Open {
        Open {
            fingerprint: 0xfeed_beef_cafe,
            first: 0,
            end: 14,
            context_length: 8192,
            max_sequences: 4,
            kv_tokens: 16384,
            max_rows: 512,
            session: 77,
            carry: CarryDtype::Bf16,
        }
    }

    fn seq(slot: u32, want: Want, hidden: Hidden) -> StageSeq {
        StageSeq {
            slot,
            start_pos: 40 + u64::from(slot),
            hist_from: 32,
            hist: vec![1, 2, 3],
            tokens: vec![9, 8],
            want,
            hidden,
            tables: vec![vec![0, 1, 2], vec![], vec![7]],
            // Slot 0 restores the zero state and snapshots; slot 2 restores
            // a snapshot and reads ahead; the others ask for nothing.
            restore: match slot {
                0 => Some(Restore { id: None, len: 0 }),
                2 => Some(Restore {
                    id: Some(12),
                    len: 40,
                }),
                _ => None,
            },
            snapshot: (slot == 0).then_some(77),
            prefetch: if slot == 2 { vec![4, 5, 6] } else { Vec::new() },
        }
    }

    fn carry(per_layer: bool, pre: bool) -> Carry {
        Carry {
            hidden: vec![0.5, -1.0, 2.5, 3.0],
            per_layer: per_layer.then(|| vec![1.0, 2.0]),
            pre: pre.then(|| vec![0.25]),
            kv: vec![
                MirrorRows {
                    layer: 3,
                    seqs: vec![(vec![1.0, 2.0], vec![3.0, 4.0]), (vec![5.0], vec![])],
                },
                MirrorRows {
                    layer: 11,
                    seqs: vec![(vec![], vec![6.0]), (vec![7.0, 8.0], vec![9.0, 10.0])],
                },
            ],
        }
    }

    fn assert_carry_eq(a: &Carry, b: &Carry) {
        assert_eq!(a.hidden, b.hidden);
        assert_eq!(a.per_layer, b.per_layer);
        assert_eq!(a.pre, b.pre);
        assert_eq!(a.kv.len(), b.kv.len());
        for (x, y) in a.kv.iter().zip(&b.kv) {
            assert_eq!(x.layer, y.layer);
            assert_eq!(x.seqs, y.seqs);
        }
    }

    fn forward(carry: Option<Carry>) -> ForwardCall {
        ForwardCall {
            session: 77,
            seq: 41,
            drops: vec![3, 5],
            exact: true,
            seqs: vec![
                seq(0, Want::Last, Hidden::PostNorm),
                seq(2, Want::All, Hidden::PostNormAll),
            ],
            carry,
            carry_dtype: CarryDtype::F32,
        }
    }

    fn decoded_forward(frame: &[u8]) -> ForwardCall {
        match decode(frame) {
            Ok(Call::Forward(f)) => f,
            Ok(_) => panic!("not a forward frame"),
            Err(e) => panic!("forward frame refused: {e}"),
        }
    }

    #[test]
    fn open_roundtrips() {
        let frame = open_frame(&open());
        assert_eq!(frame[0], OP_OPEN);
        match decode(&frame) {
            Ok(Call::Open(o)) => assert_eq!(o, open()),
            _ => panic!("open frame did not decode as open"),
        }
    }

    #[test]
    fn forward_roundtrips_with_and_without_a_carry() {
        for (per_layer, pre) in [(true, true), (false, false), (true, false), (false, true)] {
            let call = forward(Some(carry(per_layer, pre)));
            let frame = forward_frame(&call).unwrap();
            assert_eq!(frame[0], OP_FORWARD);
            let got = decoded_forward(&frame);
            assert_eq!((got.session, got.seq), (77, 41));
            assert_eq!(got.drops, vec![3, 5]);
            assert!(got.exact);
            assert_eq!(got.seqs, call.seqs);
            assert_carry_eq(got.carry.as_ref().unwrap(), call.carry.as_ref().unwrap());
        }

        let call = ForwardCall {
            exact: false,
            drops: Vec::new(),
            ..forward(None)
        };
        let got = decoded_forward(&forward_frame(&call).unwrap());
        assert!(!got.exact);
        assert!(got.drops.is_empty());
        assert_eq!(got.seqs, call.seqs);
        assert!(got.carry.is_none());
    }

    #[test]
    fn a_carry_in_a_forward_frame_is_the_carry_bytes() {
        let c = carry(true, true);
        let frame = forward_frame(&forward(Some(c.clone_for_test()))).unwrap();
        let bytes = carry_bytes(&c);
        assert!(frame.ends_with(&bytes));
        // The byte before the carry is the `has_carry` flag.
        assert_eq!(frame[frame.len() - bytes.len() - 1], 1);
    }

    #[test]
    fn a_carry_roundtrips_standalone() {
        for (per_layer, pre) in [(true, true), (false, false)] {
            let c = carry(per_layer, pre);
            assert_carry_eq(&decode_carry(&carry_bytes(&c)).unwrap(), &c);
        }
        let empty = Carry {
            hidden: Vec::new(),
            per_layer: None,
            pre: None,
            kv: Vec::new(),
        };
        assert_carry_eq(&decode_carry(&carry_bytes(&empty)).unwrap(), &empty);
    }

    #[test]
    fn a_forward_answer_roundtrips_its_carry_and_snapshot_sizes() {
        let c = carry(true, false);
        for sizes in [vec![], vec![5u64], vec![u64::MAX, 0, 7]] {
            let reply = forward_reply(&c, &sizes, CarryDtype::F32).unwrap();
            let (got, back) = decode_forward_reply(&reply, CarryDtype::F32).unwrap();
            assert_carry_eq(&got, &c);
            assert_eq!(back, sizes);
        }
        assert!(ok_reply().is_empty());
        // A carry alone, a truncated size list and trailing bytes are refused.
        assert!(decode_forward_reply(&carry_bytes(&c), CarryDtype::F32).is_err());
        let reply = forward_reply(&c, &[1, 2], CarryDtype::F32).unwrap();
        assert!(decode_forward_reply(&reply[..reply.len() - 1], CarryDtype::F32).is_err());
        let mut long = reply;
        long.push(0);
        assert!(decode_forward_reply(&long, CarryDtype::F32).is_err());
    }

    /// A carry in `bf16` or `f16` is a little over half the bytes, comes
    /// back within the rounding of the type for every part of it (hidden,
    /// per-layer, pre and the mirrored rows), names its type, and a reply
    /// in another type than the session's is refused.
    #[test]
    fn a_reduced_carry_is_half_the_bytes_and_within_rounding() {
        let c = carry(true, true);
        let exact = carry_bytes(&c);
        for (dtype, bits) in [(CarryDtype::Bf16, 8), (CarryDtype::F16, 11)] {
            let bytes = carry_bytes_as(&c, dtype).unwrap();
            let floats = |c: &Carry| -> usize {
                c.hidden.len()
                    + c.per_layer.as_ref().map_or(0, Vec::len)
                    + c.pre.as_ref().map_or(0, Vec::len)
                    + c.kv
                        .iter()
                        .flat_map(|m| &m.seqs)
                        .map(|(k, v)| k.len() + v.len())
                        .sum::<usize>()
            };
            assert_eq!(
                exact.len() - bytes.len(),
                2 * floats(&c),
                "{dtype:?}: two bytes saved per float"
            );
            let (got_dtype, got) = decode_carry_as(&bytes).unwrap();
            assert_eq!(got_dtype, dtype);
            let near = |a: &[f32], b: &[f32]| {
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b) {
                    assert!(
                        (x - y).abs() <= y.abs() * 2f32.powi(-bits) + 3.0e-8,
                        "{x} {y}"
                    );
                }
            };
            near(&got.hidden, &c.hidden);
            near(
                got.per_layer.as_ref().unwrap(),
                c.per_layer.as_ref().unwrap(),
            );
            near(got.pre.as_ref().unwrap(), c.pre.as_ref().unwrap());
            assert_eq!(got.kv.len(), c.kv.len());
            for (g, w) in got.kv.iter().zip(&c.kv) {
                assert_eq!(g.layer, w.layer);
                for ((gk, gv), (wk, wv)) in g.seqs.iter().zip(&w.seqs) {
                    near(gk, wk);
                    near(gv, wv);
                }
            }
            let reply = forward_reply(&c, &[3], dtype).unwrap();
            let (_, sizes) = decode_forward_reply(&reply, dtype).unwrap();
            assert_eq!(sizes, [3]);
            let e = decode_forward_reply(&reply, CarryDtype::F32).err().unwrap();
            assert!(e.to_string().contains(dtype.name()), "{e}");
        }
    }

    /// `open` carries the session's type, and a forward frame's carry names
    /// the type it is written in; a value `f16` cannot hold fails the
    /// encoder.
    #[test]
    fn the_carry_type_rides_in_open_and_in_forward_frames() {
        for dtype in [CarryDtype::F32, CarryDtype::Bf16, CarryDtype::F16] {
            let frame = open_frame(&Open {
                carry: dtype,
                ..open()
            });
            match decode(&frame).unwrap() {
                Call::Open(got) => assert_eq!(got.carry, dtype),
                Call::Forward(_) => panic!("not an open frame"),
            }
            let call = ForwardCall {
                carry_dtype: dtype,
                ..forward(Some(carry(true, true)))
            };
            let got = decoded_forward(&forward_frame(&call).unwrap());
            assert_eq!(got.carry_dtype, dtype);
            assert!(got.carry.is_some());
        }
        let mut big = carry(false, false);
        big.hidden[0] = 1.0e6;
        let call = ForwardCall {
            carry_dtype: CarryDtype::F16,
            ..forward(Some(big))
        };
        assert!(forward_frame(&call).is_err());
        let mut bad = open_frame(&open());
        *bad.last_mut().unwrap() = 9;
        assert!(decode(&bad).is_err(), "an unknown carry tag");
    }

    #[test]
    fn hidden_layers_is_refused() {
        let mut call = forward(None);
        call.seqs[1].hidden = Hidden::Layers(LayerSet::new([3, 4]).unwrap());
        assert!(matches!(forward_frame(&call), Err(CucaError::Config(_))));
    }

    fn all_frames() -> Vec<Vec<u8>> {
        vec![
            open_frame(&open()),
            forward_frame(&forward(Some(carry(true, true)))).unwrap(),
            forward_frame(&forward(None)).unwrap(),
        ]
    }

    #[test]
    fn every_truncation_is_an_error() {
        for frame in all_frames() {
            assert!(decode(&frame).is_ok());
            for cut in 0..frame.len() {
                assert!(
                    decode(&frame[..cut]).is_err(),
                    "frame of op {} decoded from {cut} of {} bytes",
                    frame.first().copied().unwrap_or(0),
                    frame.len()
                );
            }
        }
        let c = carry_bytes(&carry(true, true));
        for cut in 0..c.len() {
            assert!(decode_carry(&c[..cut]).is_err());
        }
    }

    #[test]
    fn trailing_bytes_are_refused() {
        for mut frame in all_frames() {
            frame.push(0);
            assert!(decode(&frame).is_err());
        }
        let mut c = carry_bytes(&carry(false, false));
        c.push(0);
        assert!(decode_carry(&c).is_err());
    }

    #[test]
    fn an_unknown_op_or_flag_is_refused() {
        assert!(decode(&[]).is_err());
        assert!(decode(&[0]).is_err());
        assert!(decode(&[OP_FORWARD + 1]).is_err());
        assert!(decode(&[255]).is_err());

        // The `want` byte of the first sequence of a forward frame.
        let frame = forward_frame(&forward(None)).unwrap();
        // op, session, seq, drops (count + 2), exact, count, slot, pos, from,
        // hist (count + 3), tokens (count + 2).
        let want_at = 1 + 8 + 8 + (4 + 16) + 1 + 4 + 4 + 8 + 8 + (8 + 12) + (8 + 8);
        assert_eq!(frame[want_at], 1);
        let mut bad = frame.clone();
        bad[want_at] = 3;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad[want_at + 1] = 3;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad[1 + 8 + 8 + (4 + 16)] = 2;
        assert!(decode(&bad).is_err());

        // The last sequence (slot 2) ends with: restore (flag, has_id, id,
        // len: 18 bytes), snapshot (flag, id: 9), prefetch (count + 3: 20)
        // and the `has_carry` byte.
        let tail = frame.len() - 1 - 20 - 9;
        assert_eq!(frame[tail], 0, "slot 2 asks for no snapshot");
        assert_eq!(frame[tail - 18], 1, "slot 2 restores");
        let mut bad = frame.clone();
        bad[tail] = 2;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad[tail - 18] = 2;
        assert!(decode(&bad).is_err());
        let mut bad = frame;
        bad[tail - 17] = 2;
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn a_hostile_count_fails_without_allocating() {
        // A forward frame that claims u32::MAX sequences and holds none.
        let mut w = Writer::new();
        w.put_u8(OP_FORWARD);
        w.put_u64(1);
        w.put_u64(1);
        w.put_u32(0);
        w.put_u8(0);
        w.put_u32(u32::MAX);
        let e = decode(&w.into_bytes()).err().unwrap();
        assert!(e.to_string().contains("sequences"), "{e}");

        // Snapshot sizes of an answer likewise.
        let mut w = Writer::new();
        w.put_u8(0);
        w.put_f32s(&[]);
        w.put_u8(0);
        w.put_u8(0);
        w.put_u32(0);
        w.put_u32(u32::MAX);
        let e = decode_forward_reply(&w.into_bytes(), CarryDtype::F32)
            .err()
            .unwrap();
        assert!(e.to_string().contains("snapshot sizes"), "{e}");

        let mut w = Writer::new();
        w.put_u8(0);
        w.put_f32s(&[]);
        w.put_u8(0);
        w.put_u8(0);
        w.put_u32(u32::MAX);
        let e = decode_carry(&w.into_bytes()).err().unwrap();
        assert!(e.to_string().contains("mirror layers"), "{e}");

        let mut w = Writer::new();
        w.put_u8(0);
        w.put_f32s(&[]);
        w.put_u8(0);
        w.put_u8(0);
        w.put_u32(1);
        w.put_u32(0);
        w.put_u32(u32::MAX);
        let e = decode_carry(&w.into_bytes()).err().unwrap();
        assert!(e.to_string().contains("mirror sequences"), "{e}");

        // A two-byte float slice claiming more than the frame holds.
        let mut w = Writer::new();
        w.put_u8(CarryDtype::Bf16.tag());
        w.put_u64(1 << 40);
        let e = decode_carry(&w.into_bytes()).err().unwrap();
        assert!(e.to_string().contains("truncated frame"), "{e}");

        // A forward frame whose one sequence claims u32::MAX block tables,
        // with just enough bytes after the count to pass the sequence bound.
        let mut w = Writer::new();
        w.put_u8(OP_FORWARD);
        w.put_u64(1);
        w.put_u64(1);
        w.put_u32(0);
        w.put_u8(0);
        w.put_u32(1);
        w.put_u32(0);
        w.put_u64(0);
        w.put_u64(0);
        w.put_u32s(&[]);
        w.put_u32s(&[]);
        w.put_u8(0);
        w.put_u8(0);
        w.put_u32(u32::MAX);
        w.put_u32s(&[0; 7]);
        let e = decode(&w.into_bytes()).err().unwrap();
        assert!(e.to_string().contains("block tables"), "{e}");

        // And one that claims u32::MAX dropped snapshots.
        let mut w = Writer::new();
        w.put_u8(OP_FORWARD);
        w.put_u64(1);
        w.put_u64(1);
        w.put_u32(u32::MAX);
        let e = decode(&w.into_bytes()).err().unwrap();
        assert!(e.to_string().contains("drops"), "{e}");
    }

    trait CloneForTest {
        fn clone_for_test(&self) -> Self;
    }

    impl CloneForTest for Carry {
        fn clone_for_test(&self) -> Self {
            Carry {
                hidden: self.hidden.clone(),
                per_layer: self.per_layer.clone(),
                pre: self.pre.clone(),
                kv: self
                    .kv
                    .iter()
                    .map(|m| MirrorRows {
                        layer: m.layer,
                        seqs: m.seqs.clone(),
                    })
                    .collect(),
            }
        }
    }
}
