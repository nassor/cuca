//! The hashed per-layer table lookup ("engram") of a hyper-connection model.
//!
//! Every token position is hashed together with the tokens before it into
//! `(ngram - 1) * heads` row indices per engram layer: the token ids go
//! through the model's compressed token map (so that tokens that normalise
//! alike collapse), the n-gram of order `k + 1` XORs the products of the
//! `k + 1` most recent ids with the layer's odd multipliers, and the hash of
//! order `k + 1` and head `h` is that value modulo the prime bucket count of
//! `(layer, k, h)` plus the first row of those buckets in the one table that
//! holds every layer's rows. Look-back stops at the start of the sequence: the
//! positions before it hold the pad token.
//!
//! The rows come from a [`RowSource`]: [`EngramTable`](super::engram_io::EngramTable)
//! reads them from the GGUF file with a pool of threads through a bounded row
//! cache, since a real model's table (209 GB) does not fit in RAM.

use std::sync::Arc;

use super::csa::EngramSpec;
use super::engram_io::EngramCounters;
use super::err;
use crate::error::CucaError;

/// Longest n-gram supported (`ple.ngram_size`).
pub(crate) const MAX_NGRAM: usize = 8;

/// A table of rows addressed by absolute index.
pub(crate) trait RowSource: Send + Sync {
    /// The rows `ids`, dequantized and concatenated, into `out`
    /// (`ids.len() * row_dim` values).
    ///
    /// # Errors
    ///
    /// [`CucaError`] for an index outside the table or a failed read.
    fn rows(&self, ids: &[u64], out: &mut [f32]) -> Result<(), CucaError>;

    /// Start reading `ids` in the background, so that a later
    /// [`RowSource::rows`] that asks for them finds them: a hint that never
    /// fails and changes no value.
    fn prefetch(&self, ids: &[u64]);

    /// The counters of the source, when it keeps any.
    fn counters(&self) -> Option<Arc<EngramCounters>>;
}

/// Open the Engram table of the model `hp` describes (the tensor
/// `per_layer_token_embd` of `content`) with the budget of `cache`; `None`
/// for a model without one. One table serves every layer that reads it,
/// across both halves of a split model, so that its row cache and its
/// counters are single.
///
/// # Errors
///
/// As [`EngramTable::open`](super::engram_io::EngramTable::open).
pub(crate) fn open_table(
    content: &crate::provider::local::gguf::Content,
    hp: &super::Hparams,
    cache: &crate::provider::local::EngramCache,
) -> Result<Option<Arc<dyn RowSource>>, CucaError> {
    let Some(spec) = &hp.engram else {
        return Ok(None);
    };
    let table = super::engram_io::EngramTable::open(
        content,
        super::layout::name::PER_LAYER_TOKEN_EMBD,
        spec.row_dim,
        cache,
    )?;
    Ok(Some(Arc::new(table)))
}

/// The compressed ids of the `ngram` most recent tokens at a position:
/// `window[s]` is the token `s` positions back. `token_at(back)` returns the
/// raw id `back` positions before the position, `None` before the start of the
/// sequence; once one is missing every older one is the pad token.
pub(crate) fn window(
    spec: &EngramSpec,
    token_at: impl Fn(usize) -> Option<u32>,
) -> Result<[u32; MAX_NGRAM], CucaError> {
    let map = |raw: u32| {
        spec.token_map
            .get(raw as usize)
            .copied()
            .ok_or_else(|| err(format!("token {raw} outside the engram token map")))
    };
    let pad = map(spec.pad_id)?;
    let mut out = [pad; MAX_NGRAM];
    for (back, slot) in out.iter_mut().enumerate().take(spec.ngram) {
        match token_at(back) {
            Some(raw) => *slot = map(raw)?,
            None => break,
        }
    }
    Ok(out)
}

/// The table rows (absolute indices) of one position at the engram layer with
/// ordinal `layer` among the engram layers: `cols()` of them, order `2` first,
/// every head of an order together.
pub(crate) fn row_ids(spec: &EngramSpec, layer: usize, window: &[u32; MAX_NGRAM], out: &mut [u64]) {
    let mult = &spec.multipliers[layer * spec.ngram..(layer + 1) * spec.ngram];
    let per_layer = spec.cols();
    let sizes = &spec.head_sizes[layer * per_layer..(layer + 1) * per_layer];
    let offsets = &spec.head_offsets[layer * per_layer..(layer + 1) * per_layer];
    let mut rolling = u64::from(window[0]).wrapping_mul(mult[0]);
    for order in 1..spec.ngram {
        rolling ^= u64::from(window[order]).wrapping_mul(mult[order]);
        for head in 0..spec.heads {
            let at = (order - 1) * spec.heads + head;
            out[at] = rolling % sizes[at] + offsets[at];
        }
    }
}

/// The table rows of the engram layer with ordinal `ordinal` for `tokens` of
/// one sequence, `spec.cols()` per token in `out`: the n-grams end at each
/// token and reach back into `history` (the tokens before the first one) and,
/// past its start, into the pad token.
pub(crate) fn sequence_row_ids(
    spec: &EngramSpec,
    ordinal: usize,
    tokens: &[u32],
    history: &[u32],
    out: &mut [u64],
) -> Result<(), CucaError> {
    let cols = spec.cols();
    for (i, ids) in out.chunks_exact_mut(cols).enumerate().take(tokens.len()) {
        let window = window(spec, |back| {
            if i >= back {
                Some(tokens[i - back])
            } else {
                history.len().checked_sub(back - i).map(|at| history[at])
            }
        })?;
        row_ids(spec, ordinal, &window, ids);
    }
    Ok(())
}
