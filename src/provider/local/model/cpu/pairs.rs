//! The pair bookkeeping of the drafters that keep a K/V cache of their own
//! over the trunk's positions: the Qwen 3.5 head ([`NextnHead`](super::NextnHead))
//! and the EAGLE drafter ([`EagleHead`](super::EagleHead)).
//!
//! Both commit one cache slot per pair `(h[j], t[j + 1])` (the trunk's
//! final-norm hidden row at position `j` and the token that follows it), in
//! ascending pair index `j` (an index may be skipped: a position the drafter
//! cannot see costs a worse proposal and never changes output), and keep the
//! *tail*, the hidden row of the last observed position, until its successor
//! token arrives. [`PairLog`] holds that state and the rules that keep it
//! consistent with the trunk; the drafters own the weights, the K/V and the
//! observe and draft passes.

use std::borrow::Cow;

use super::super::err;
use crate::error::CucaError;

/// The pairs one observe pass adds: the first pair index, their hidden rows
/// and their successor tokens.
type NewPairs<'a> = (usize, Cow<'a, [f32]>, &'a [u32]);

/// The committed pairs and the tail of one drafter.
pub(super) struct PairLog {
    /// Slots the drafter's K/V cache holds.
    pub(super) capacity: usize,
    /// The pair index `j` of every committed slot, strictly ascending.
    pub(super) pairs: Vec<usize>,
    /// Position and final-norm hidden row of the last observed position,
    /// awaiting the token that follows it.
    pub(super) tail: Option<(usize, Vec<f32>)>,
}

impl PairLog {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            pairs: Vec::new(),
            tail: None,
        }
    }

    /// The first `shared` tokens of the sequence stay as they were and every
    /// later one may change: keep the pairs `(h[j], t[j + 1])` with
    /// `j + 1 < shared` (the tokens and so the hidden rows they depend on are
    /// unchanged) and the tail when its position is below `shared`.
    /// `truncate(0)` forgets everything.
    pub(super) fn truncate(&mut self, shared: usize) {
        let keep = self.pairs.partition_point(|&j| j + 1 < shared);
        self.pairs.truncate(keep);
        if self.tail.as_ref().is_some_and(|(at, _)| *at >= shared) {
            self.tail = None;
        }
    }

    /// Drop the pairs with an index of at least `first`.
    pub(super) fn drop_from(&mut self, first: usize) {
        let keep = self.pairs.partition_point(|&j| j < first);
        self.pairs.truncate(keep);
    }

    /// `Ok(slot)`, the first free slot, when `m` more pairs fit.
    pub(super) fn reserve(&self, what: &str, m: usize) -> Result<usize, CucaError> {
        let slot = self.pairs.len();
        if slot + m > self.capacity {
            return Err(err(format!(
                "the {what} cache holds {} positions, {} do not fit",
                self.capacity,
                slot + m
            )));
        }
        Ok(slot)
    }

    /// Start observing the trunk rows `tokens[i]` at positions `pos + i`
    /// whose final-norm hidden rows (`d` floats each) are `hidden`: pair
    /// every row with its successor token, the tail with `tokens[0]` when it
    /// sits at `pos - 1`, and drop the pairs the new ones replace. Returns
    /// the first new pair index, the hidden rows and the successor tokens of
    /// the new pairs; the caller stores them in the slots from
    /// `pairs.len()`, then calls [`PairLog::finish`].
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for mismatched lengths or a full cache.
    pub(super) fn begin<'a>(
        &mut self,
        what: &str,
        d: usize,
        pos: usize,
        tokens: &'a [u32],
        hidden: &'a [f32],
    ) -> Result<NewPairs<'a>, CucaError> {
        let n = tokens.len();
        if n == 0 || hidden.len() != n * d {
            return Err(err(format!(
                "{what} observe: {n} tokens with {} hidden floats, expected {}",
                hidden.len(),
                n * d
            )));
        }
        // Pair `(h[j], t[j + 1])` for `j` from `first`.
        let (first, rows, ids) = match self.tail.as_ref().filter(|(at, _)| at + 1 == pos) {
            Some((_, prev)) => {
                let mut rows = Vec::with_capacity(n * d);
                rows.extend_from_slice(prev);
                rows.extend_from_slice(&hidden[..(n - 1) * d]);
                (pos - 1, Cow::Owned(rows), tokens)
            }
            None => (pos, Cow::Borrowed(&hidden[..(n - 1) * d]), &tokens[1..]),
        };
        self.drop_from(first);
        self.reserve(what, ids.len())?;
        Ok((first, rows, ids))
    }

    /// Commit the `m` pairs from `first` that [`PairLog::begin`] returned and
    /// make the last of the `hidden.len() / d` rows the tail.
    pub(super) fn finish(&mut self, first: usize, m: usize, d: usize, pos: usize, hidden: &[f32]) {
        let n = hidden.len() / d;
        self.pairs.extend(first..first + m);
        self.tail = Some((pos + n - 1, hidden[(n - 1) * d..].to_vec()));
    }
}
