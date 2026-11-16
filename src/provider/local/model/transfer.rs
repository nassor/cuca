//! K/V rows and recurrent state in transit between two engines: a prefill
//! node exports the blocks of a prompt's prefix, a decode node files them in
//! its own pools ([`Model::export_rows`](super::Model::export_rows),
//! [`Model::import_rows`](super::Model::import_rows)).
//!
//! A transfer covers the token positions `from..to` of a prefix that ends at
//! `end` (a transfer in chunks has one `end` and many `from..to`). All three
//! are multiples of `BLOCK_SIZE * step`, `step` the least common multiple of
//! the groups' strides ([`KvLayout::step`]), so a chunk holds whole blocks of
//! every group.
//!
//! # Entries
//!
//! One [`PoolRows`] per K/V-owning layer of a group, in the order `for g in
//! groups { for il in g.layers }`. The rows of a group in a chunk are
//! [`group_rows`]: `from / stride .. to / stride`, except that a group with a
//! window of `w` positions needs only the rows from `end - (w - 1)` on (the
//! blocks before were released while the prompt was prefilled), so its first
//! row is the block boundary at or below that position, and it carries no rows
//! while the chunk ends before it.
//!
//! # Canonical layout
//!
//! The layout the GPU mirror rows of a split model already use: `k` and `v`
//! are `[n_kv_head][rows][head_dim]` row-major; `v` is empty for a layer
//! whose value is the leading part of its key row (latent attention) or its
//! key itself (`kv_is_value`); a compressed entry holds `[rows][row_width]`
//! in `k`.

use std::ops::Range;

use super::paged::{BLOCK_SIZE, BlockTable, KvGroup, KvLayout};
use super::{Model, err};
use crate::error::CucaError;

/// The rows of one K/V-owning layer of a transfer chunk, see the module docs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PoolRows {
    /// The layer that owns the rows.
    pub(crate) layer: usize,
    /// The rows are compressed rows of the layer's compressed cache, not its
    /// K/V positions.
    pub(crate) compressed: bool,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
}

/// The state of one delta-net layer: the convolution window and the
/// recurrent matrix, flattened.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeltaRows {
    pub(crate) conv: Vec<f32>,
    pub(crate) s: Vec<f32>,
}

/// The `(group, layer)` of every entry of a transfer, in order.
pub(crate) fn entries(layout: &KvLayout) -> impl Iterator<Item = (usize, usize)> + '_ {
    layout
        .groups
        .iter()
        .enumerate()
        .flat_map(|(gi, g)| g.layers.iter().map(move |&il| (gi, il)))
}

/// The rows of group `g` that the chunk `from..to` of a transfer ending at
/// `end` carries, as row indexes of the group's pool (empty when the group
/// has none).
pub(crate) fn group_rows(g: &KvGroup, from: usize, to: usize, end: usize) -> Range<usize> {
    let first = match g.window {
        Some(w) => from.max(end.saturating_sub(w.saturating_sub(1)) / BLOCK_SIZE * BLOCK_SIZE),
        None => from,
    };
    if first >= to {
        0..0
    } else {
        first / g.stride..to / g.stride
    }
}

/// Check that `from..to` of a transfer ending at `end` lies on block
/// boundaries of every group and that `tables` is one table per group.
///
/// # Errors
///
/// [`CucaError::Provider`] naming what is wrong.
pub(crate) fn check_span(
    layout: &KvLayout,
    tables: &[BlockTable],
    from: usize,
    to: usize,
    end: usize,
) -> Result<(), CucaError> {
    let unit = BLOCK_SIZE * layout.step();
    if tables.len() != layout.groups.len() {
        return Err(err(format!(
            "{} block tables for {} K/V groups",
            tables.len(),
            layout.groups.len()
        )));
    }
    if !(from <= to && to <= end && from.is_multiple_of(unit) && to.is_multiple_of(unit)) {
        return Err(err(format!(
            "K/V rows {from}..{to} of {end} are not whole blocks of {unit} positions"
        )));
    }
    if !end.is_multiple_of(unit) {
        return Err(err(format!(
            "a K/V transfer ends at a multiple of {unit} positions, not {end}"
        )));
    }
    Ok(())
}

/// The message of a transfer whose rows do not fit the model.
pub(crate) const MISMATCH: &str = "K/V rows do not match this model's cache layout";

impl Model {
    /// The pool rows of the positions `from..to` of the sequence whose block
    /// tables are `tables`, in the canonical layout (see the module docs):
    /// one entry per K/V-owning layer, `end` being where the whole transfer
    /// stops.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for positions off the block boundaries, a
    /// table that lacks a block of the rows, a model on several devices
    /// (tensor parallelism) or a failed device read.
    pub(crate) fn export_rows(
        &self,
        tables: &[BlockTable],
        from: usize,
        to: usize,
        end: usize,
    ) -> Result<Vec<PoolRows>, CucaError> {
        check_span(self.kv_layout(), tables, from, to, end)?;
        on_model!(self, m => m.export_rows(tables, from, to, end))
    }

    /// Write `rows` (what [`Model::export_rows`] returned on a model of the
    /// same file) into the blocks of `tables`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `rows` is not exactly the sequence of
    /// entries of this model's layout with the lengths of its pools, a block
    /// is missing from `tables`, the model runs on several devices (tensor
    /// parallelism), or no engine of the model wrote an entry.
    pub(crate) fn import_rows(
        &mut self,
        tables: &[BlockTable],
        from: usize,
        to: usize,
        end: usize,
        rows: &[PoolRows],
    ) -> Result<(), CucaError> {
        let layout = self.kv_layout();
        check_span(layout, tables, from, to, end)?;
        let wanted: Vec<(usize, bool)> = entries(layout)
            .map(|(gi, il)| (il, layout.groups[gi].compressed))
            .collect();
        if rows.len() != wanted.len()
            || rows
                .iter()
                .zip(&wanted)
                .any(|(r, &(il, compressed))| r.layer != il || r.compressed != compressed)
        {
            return Err(err(MISMATCH));
        }
        let written = on_model!(self, m => m.import_rows(tables, from, to, end, rows))?;
        if (0..rows.len()).any(|i| !written.contains(&i)) {
            return Err(err("no engine of the model holds a K/V pool for some rows"));
        }
        Ok(())
    }

    /// The state of every delta-net layer of `snapshot`, ascending.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a snapshot from another device or a failed
    /// device read.
    pub(crate) fn snapshot_export(
        &self,
        snapshot: &super::RecurrentSnapshot,
    ) -> Result<Vec<DeltaRows>, CucaError> {
        use super::RecurrentSnapshot as S;
        match (self, snapshot) {
            (Model::Cpu(m), S::Cpu(layers)) => m.snapshot_export(layers),
            #[cfg(feature = "provider-local-gpu")]
            (Model::Gpu(m), S::Gpu(snapshot)) => m.snapshot_export(snapshot),
            #[cfg(feature = "provider-local-gpu")]
            (Model::Split(m), S::Split(snapshot)) => m.snapshot_export(snapshot),
            (Model::Chain(m), S::Chain(snapshot)) => m.snapshot_export(snapshot),
            _ => Err(err("the snapshot comes from another device")),
        }
    }

    /// A snapshot of this model's device holding `rows` (what
    /// [`Model::snapshot_export`] returned on a model of the same file).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `rows` do not fit the model's recurrent
    /// layers or a device upload fails.
    pub(crate) fn snapshot_import(
        &self,
        rows: Vec<DeltaRows>,
    ) -> Result<super::RecurrentSnapshot, CucaError> {
        use super::RecurrentSnapshot as S;
        match self {
            Model::Cpu(m) => m.snapshot_import(rows).map(S::Cpu),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.snapshot_import(rows).map(S::Gpu),
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(m) => m.snapshot_import(rows).map(S::Split),
            Model::Chain(_) => Err(CucaError::Config(
                "K/V transfer is not available with layer stages".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(window: Option<usize>, stride: usize) -> KvGroup {
        KvGroup {
            window,
            stride,
            compressed: stride > 1,
            layers: vec![0],
        }
    }

    #[test]
    fn a_full_group_carries_every_row_of_the_chunk() {
        assert_eq!(group_rows(&group(None, 1), 32, 96, 160), 32..96);
        // A compressed group has one row per `stride` positions.
        assert_eq!(group_rows(&group(None, 4), 64, 128, 128), 16..32);
    }

    #[test]
    fn a_window_group_carries_only_the_rows_the_window_needs() {
        let g = group(Some(40), 1);
        // The window of a transfer ending at 160 starts at 160 - 39 = 121,
        // in the block at 112.
        assert_eq!(group_rows(&g, 0, 64, 160), 0..0);
        assert_eq!(group_rows(&g, 64, 112, 160), 0..0);
        assert_eq!(group_rows(&g, 96, 128, 160), 112..128);
        assert_eq!(group_rows(&g, 128, 160, 160), 128..160);
        // A window longer than the prefix leaves everything in.
        assert_eq!(group_rows(&group(Some(1000), 1), 0, 160, 160), 0..160);
    }
}
