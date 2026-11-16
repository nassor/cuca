//! K/V rows and recurrent state of the CPU engine in transit: the pool rows
//! of a block-aligned range of a sequence, read out and written back in the
//! canonical layout of [`transfer`](super::super::transfer).
//!
//! The row I/O is [`gather`] and [`store_rows`] over the layer's pool (the
//! window or full-attention pool of [`Transformer::kv`], or, for a group of
//! compressed rows, the compressed pool of `HyperModel::ckv`, whose rows sit
//! at the addresses `hyper::pool_row` computes).

use super::super::delta::DeltaState;
use super::super::paged::BlockTable;
use super::super::transfer::{DeltaRows, MISMATCH, PoolRows, entries, group_rows};
use super::super::{candle_err, err};
use super::paged_attn::{gather, store_rows};
use super::{KvLayer, Transformer};
use crate::error::CucaError;

impl Transformer {
    /// The pool pair of the layer of entry `(compressed, il)`, if this
    /// engine holds one.
    fn transfer_layer(&self, compressed: bool, il: usize) -> Option<&KvLayer> {
        if compressed {
            self.hyper.as_ref()?.ckv.get(il)?.as_deref()
        } else {
            self.kv.get(il)?.as_deref()
        }
    }

    /// See [`Model::export_rows`](super::super::Model::export_rows): the
    /// entries of the layers this engine holds a pool for, in layout order.
    pub(crate) fn export_rows(
        &self,
        tables: &[BlockTable],
        from: usize,
        to: usize,
        end: usize,
    ) -> Result<Vec<PoolRows>, CucaError> {
        let layout = &self.layout;
        let mut out = Vec::new();
        for (gi, il) in entries(layout) {
            let g = &layout.groups[gi];
            let Some(layer) = self.transfer_layer(g.compressed, il) else {
                continue;
            };
            let range = group_rows(g, from, to, end);
            let data = layer.lock();
            let read = |pool: &[f32]| -> Result<Vec<f32>, CucaError> {
                if range.is_empty() || pool.is_empty() {
                    return Ok(Vec::new());
                }
                gather(&tables[gi], range.start, range.end, pool, layer.shape).map_err(candle_err)
            };
            out.push(PoolRows {
                layer: il,
                compressed: g.compressed,
                k: read(&data.k)?,
                v: read(&data.v)?,
            });
        }
        Ok(out)
    }

    /// See [`Model::import_rows`](super::super::Model::import_rows): writes
    /// the entries of `rows` (one per entry of the layout, checked by the
    /// caller) whose layer this engine holds a pool for, and returns their
    /// indexes in `rows`.
    pub(crate) fn import_rows(
        &mut self,
        tables: &[BlockTable],
        from: usize,
        to: usize,
        end: usize,
        rows: &[PoolRows],
    ) -> Result<Vec<usize>, CucaError> {
        let layout = &self.layout;
        // Check every entry before the first write: a rejected transfer
        // leaves the pools as they were.
        let mut plan = Vec::new();
        for (i, (gi, il)) in entries(layout).enumerate() {
            let g = &layout.groups[gi];
            let Some(layer) = self.transfer_layer(g.compressed, il) else {
                continue;
            };
            let range = group_rows(g, from, to, end);
            let floats = layer.shape.n_kv_head * range.len() * layer.shape.head_dim;
            let has_values = !layer.lock().v.is_empty();
            let row = rows.get(i).ok_or_else(|| err(MISMATCH))?;
            if row.k.len() != floats || row.v.len() != if has_values { floats } else { 0 } {
                return Err(err(MISMATCH));
            }
            plan.push((i, gi, il, g.compressed, range));
        }
        let mut written = Vec::with_capacity(plan.len());
        for (i, gi, il, compressed, range) in plan {
            if let Some(layer) = self.transfer_layer(compressed, il)
                && !range.is_empty()
            {
                let mut data = layer.lock();
                let shape = layer.shape;
                store_rows(
                    &tables[gi],
                    range.start,
                    range.len(),
                    &rows[i].k,
                    &mut data.k,
                    shape,
                )
                .map_err(candle_err)?;
                if !rows[i].v.is_empty() {
                    store_rows(
                        &tables[gi],
                        range.start,
                        range.len(),
                        &rows[i].v,
                        &mut data.v,
                        shape,
                    )
                    .map_err(candle_err)?;
                }
            }
            written.push(i);
        }
        Ok(written)
    }

    /// The state of every delta-net layer of `snapshot`, ascending.
    pub(crate) fn snapshot_export(
        &self,
        snapshot: &[DeltaState],
    ) -> Result<Vec<DeltaRows>, CucaError> {
        if snapshot.len() != self.delta_layers().count() {
            return Err(err(
                "the snapshot does not fit this model's recurrent state",
            ));
        }
        Ok(snapshot.iter().map(DeltaState::to_rows).collect())
    }

    /// A snapshot holding `rows`, one per delta-net layer, ascending.
    pub(crate) fn snapshot_import(
        &self,
        rows: Vec<DeltaRows>,
    ) -> Result<Vec<DeltaState>, CucaError> {
        if rows.len() != self.delta_layers().count() {
            return Err(err(
                "the snapshot does not fit this model's recurrent state",
            ));
        }
        let mut states = Vec::with_capacity(rows.len());
        for (dw, row) in self.delta_layers().zip(rows) {
            let zero = DeltaState::zeros(dw.kernel.spec());
            let zero = zero.to_rows();
            if row.conv.len() != zero.conv.len() || row.s.len() != zero.s.len() {
                return Err(err(
                    "the snapshot does not fit this model's recurrent state",
                ));
            }
            states.push(DeltaState::from_rows(row));
        }
        Ok(states)
    }
}

#[cfg(test)]
mod tests {
    use super::super::hyper::pool_row;
    use super::*;
    use crate::provider::local::model::paged::BLOCK_SIZE;
    use crate::provider::local::model::tiny::Lcg;

    /// `gather` on a compressed pool, whose geometry is one head of the row
    /// width, returns the rows `pool_row` addresses.
    #[test]
    fn gather_reads_the_rows_of_a_compressed_pool() {
        let width = 6;
        let blocks = 4;
        let layer = KvLayer::compressed(width, blocks);
        let values = Lcg(5).values(blocks * BLOCK_SIZE * width, 1.0);
        layer.lock().k.copy_from_slice(&values);
        // Scattered blocks, the way a sequence's table names them.
        let table: BlockTable = vec![3, 1, 2];
        let (lo, hi) = (5, 40);
        let got = gather(&table, lo, hi, &layer.lock().k, layer.shape).unwrap();
        let want: Vec<f32> = (lo..hi)
            .flat_map(|i| {
                let at = pool_row(&table, i, width).unwrap();
                values[at..at + width].to_vec()
            })
            .collect();
        assert_eq!(got, want);
    }
}
