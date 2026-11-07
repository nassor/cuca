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
}
