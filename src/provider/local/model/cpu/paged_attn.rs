//! Attention over a paged K/V cache on the CPU.
//!
//! A layer's keys and values live in two flat `f32` pools of shape
//! `[n_blocks][n_kv_head][BLOCK_SIZE][head_dim]`; a sequence's *block table*
//! maps position `p` to the block `table[p / BLOCK_SIZE]`, row `p %
//! BLOCK_SIZE`. [`attend_paged_rows`] attends query rows over their own key
//! ranges straight out of the pools, block by block with an online softmax
//! (running maximum, running sum, rescaled value accumulator), so no key or
//! value is copied, whatever the block ids are, and the cost per row is
//! independent of how many other rows the call holds: a row's result depends
//! only on its query, its table and its `[lo, hi)` range, whichever call or
//! thread computes it, so a row is bit-identical whether it is computed alone
//! or in a batch. That is the property the shape-stable mode
//! ([`RowExact`](super::RowExact)) and batched decoding need.
//!
//! The work splits over (row, query head) pairs on candle's barrier pool, one
//! pooled call per layer; the arithmetic of one pair never depends on the
//! split. [`gather`] copies a key range into one contiguous tensor for the
//! batched (non-exact) path, which multiplies it with candle's matmul.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::utils::barrier_pool;

use super::super::paged::{BLOCK_SIZE, NO_BLOCK};

/// The geometry of one layer's pools.
#[derive(Clone, Copy)]
pub(crate) struct PoolShape {
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
}

/// Where position `pos`'s row of KV head `kvh` starts in a pool.
fn row_offset(table: &[u32], pos: usize, kvh: usize, shape: PoolShape) -> Option<usize> {
    let block = *table.get(pos / BLOCK_SIZE)?;
    (block != NO_BLOCK).then(|| {
        ((block as usize * shape.n_kv_head + kvh) * BLOCK_SIZE + pos % BLOCK_SIZE) * shape.head_dim
    })
}

/// Check that `table` holds a block for every position of `lo..hi` and that
/// the pools are big enough for it.
fn check_range(
    table: &[u32],
    lo: usize,
    hi: usize,
    pool_len: usize,
    shape: PoolShape,
) -> candle_core::Result<()> {
    if lo >= hi {
        return Err(candle_core::Error::msg("attention over an empty key range"));
    }
    let block_floats = shape.n_kv_head * BLOCK_SIZE * shape.head_dim;
    for b in lo / BLOCK_SIZE..=(hi - 1) / BLOCK_SIZE {
        match table.get(b) {
            Some(&id) if id != NO_BLOCK && (id as usize + 1) * block_floats <= pool_len => {}
            _ => {
                return Err(candle_core::Error::msg(format!(
                    "position {} has no K/V block (it left the window or was never allocated)",
                    b * BLOCK_SIZE
                )));
            }
        }
    }
    Ok(())
}

/// Dot product in eight independent lanes: fixed order, vectorisable.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let ((a8, ar), (b8, br)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    let mut sum = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    for (x, y) in ar.iter().zip(br) {
        sum += x * y;
    }
    sum
}

/// One query head over the keys `lo..hi`: `q` is its `head_dim` query values,
/// `kvh` the KV head it reads, `out` receives `head_dim` values. The ranges
/// were checked by [`check_range`].
#[expect(
    clippy::too_many_arguments,
    reason = "one attention head's inputs; a struct would only rename them"
)]
fn attend_head(
    q: &[f32],
    kvh: usize,
    table: &[u32],
    lo: usize,
    hi: usize,
    pool_k: &[f32],
    pool_v: &[f32],
    shape: PoolShape,
    scale: f32,
    out: &mut [f32],
) {
    let d = shape.head_dim;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0f32;
    let mut acc = vec![0f32; d];
    let mut scores = [0f32; BLOCK_SIZE];
    let mut pos = lo;
    while pos < hi {
        let end = hi.min((pos / BLOCK_SIZE + 1) * BLOCK_SIZE);
        let n = end - pos;
        let Some(first) = row_offset(table, pos, kvh, shape) else {
            return;
        };
        let mut block_max = f32::NEG_INFINITY;
        for (j, score) in scores.iter_mut().take(n).enumerate() {
            let at = first + j * d;
            let s = dot(q, &pool_k[at..at + d]) * scale;
            *score = s;
            if s > block_max {
                block_max = s;
            }
        }
        if block_max > max {
            // Rescale what was accumulated under the old maximum.
            let correction = (max - block_max).exp();
            sum *= correction;
            for a in &mut acc {
                *a *= correction;
            }
            max = block_max;
        }
        for (j, &s) in scores.iter().take(n).enumerate() {
            let p = (s - max).exp();
            sum += p;
            let at = first + j * d;
            for (a, &v) in acc.iter_mut().zip(&pool_v[at..at + d]) {
                *a += p * v;
            }
        }
        pos = end;
    }
    for (o, a) in out.iter_mut().zip(&acc) {
        *o = a / sum;
    }
}

/// Attend `t` query rows over a paged cache. `q` holds the rows as
/// `[t][n_head][head_dim]`; row `i` attends to the positions `range(i)`
/// (`lo..hi`) of `table`. Returns `[t][n_head][head_dim]`. Query head `h`
/// reads KV head `h / (n_head / n_kv_head)`; the scores are scaled by
/// `scale` before the softmax.
///
/// Runs on candle's barrier pool, so it must not be called from inside one of
/// its jobs.
///
/// # Errors
///
/// An empty range, or a position of a range without a block in `table`.
#[expect(
    clippy::too_many_arguments,
    reason = "one attention call's inputs; a struct would only rename them"
)]
pub(crate) fn attend_paged_rows(
    q: &[f32],
    t: usize,
    range: impl Fn(usize) -> (usize, usize) + Sync,
    table: &[u32],
    pool_k: &[f32],
    pool_v: &[f32],
    shape: PoolShape,
    scale: f32,
) -> candle_core::Result<Vec<f32>> {
    let PoolShape {
        n_head,
        n_kv_head,
        head_dim: d,
    } = shape;
    if q.len() != t * n_head * d || n_kv_head == 0 || n_head % n_kv_head != 0 {
        return Err(candle_core::Error::msg("attention: query shape mismatch"));
    }
    for i in 0..t {
        let (lo, hi) = range(i);
        check_range(table, lo, hi, pool_k.len().min(pool_v.len()), shape)?;
    }
    let group = n_head / n_kv_head;
    let mut out = vec![0f32; t * n_head * d];
    let jobs = t * n_head;
    {
        let slots: Vec<Mutex<&mut [f32]>> = out.chunks_mut(d).map(Mutex::new).collect();
        let next = AtomicUsize::new(0);
        let run = |_: usize| {
            loop {
                let job = next.fetch_add(1, Ordering::Relaxed);
                if job >= jobs {
                    break;
                }
                let (row, head) = (job / n_head, job % n_head);
                let (lo, hi) = range(row);
                let mut slot = slots[job].lock().unwrap_or_else(PoisonError::into_inner);
                attend_head(
                    &q[job * d..(job + 1) * d],
                    head / group,
                    table,
                    lo,
                    hi,
                    pool_k,
                    pool_v,
                    shape,
                    scale,
                    &mut slot,
                );
            }
        };
        if jobs == 1 {
            run(0);
        } else {
            barrier_pool().execute(run);
        }
    }
    Ok(out)
}

/// [`attend_paged_rows`] for one query row over the keys `lo..hi`: `q_row`
/// is `n_head * head_dim` values, the result as many.
///
/// # Errors
///
/// As [`attend_paged_rows`].
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "one attention call's inputs; a struct would only rename them"
)]
pub(crate) fn attend_paged_row(
    q_row: &[f32],
    table: &[u32],
    lo: usize,
    hi: usize,
    pool_k: &[f32],
    pool_v: &[f32],
    shape: PoolShape,
    scale: f32,
) -> candle_core::Result<Vec<f32>> {
    attend_paged_rows(q_row, 1, |_| (lo, hi), table, pool_k, pool_v, shape, scale)
}

/// The keys (or values) of positions `lo..hi` as `[n_kv_head][hi - lo][head_dim]`.
///
/// # Errors
///
/// A position without a block in `table`.
pub(crate) fn gather(
    table: &[u32],
    lo: usize,
    hi: usize,
    pool: &[f32],
    shape: PoolShape,
) -> candle_core::Result<Vec<f32>> {
    check_range(table, lo, hi, pool.len(), shape)?;
    let (l, d) = (hi - lo, shape.head_dim);
    let mut out = Vec::with_capacity(shape.n_kv_head * l * d);
    for kvh in 0..shape.n_kv_head {
        let mut pos = lo;
        while pos < hi {
            let end = hi.min((pos / BLOCK_SIZE + 1) * BLOCK_SIZE);
            let Some(at) = row_offset(table, pos, kvh, shape) else {
                return Err(candle_core::Error::msg("gather: a block went missing"));
            };
            out.extend_from_slice(&pool[at..at + (end - pos) * d]);
            pos = end;
        }
    }
    Ok(out)
}

/// Write `rows` positions starting at `start` into the pool: `src` holds
/// `[n_kv_head][rows][head_dim]`.
///
/// # Errors
///
/// A position without a block in `table`, or a short `src`.
pub(crate) fn store_rows(
    table: &[u32],
    start: usize,
    rows: usize,
    src: &[f32],
    pool: &mut [f32],
    shape: PoolShape,
) -> candle_core::Result<()> {
    let d = shape.head_dim;
    if src.len() != shape.n_kv_head * rows * d {
        return Err(candle_core::Error::msg("store: row shape mismatch"));
    }
    if rows == 0 {
        return Ok(());
    }
    check_range(table, start, start + rows, pool.len(), shape)?;
    for kvh in 0..shape.n_kv_head {
        let mut pos = start;
        while pos < start + rows {
            let end = (start + rows).min((pos / BLOCK_SIZE + 1) * BLOCK_SIZE);
            let Some(at) = row_offset(table, pos, kvh, shape) else {
                return Err(candle_core::Error::msg("store: a block went missing"));
            };
            let from = (kvh * rows + (pos - start)) * d;
            let n = (end - pos) * d;
            pool[at..at + n].copy_from_slice(&src[from..from + n]);
            pos = end;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::super::super::tiny::Lcg;
    use super::super::attend;
    use super::*;

    /// Paged attention over scattered blocks equals candle's contiguous
    /// attention (within float rounding), for every key count around a block
    /// boundary, a window and grouped heads.
    #[test]
    fn paged_attention_matches_contiguous_attention() {
        let mut rng = Lcg(42);
        for (n_head, n_kv_head, d) in [(4usize, 2usize, 8usize), (8, 1, 16), (3, 3, 5)] {
            let shape = PoolShape {
                n_head,
                n_kv_head,
                head_dim: d,
            };
            for l in [1usize, 2, 15, 16, 17, 33, 50] {
                for lo in [0, l / 3, l / 2] {
                    // Blocks in a scrambled order inside a larger pool.
                    let blocks = l.div_ceil(BLOCK_SIZE);
                    let table: Vec<u32> = (0..blocks as u32).map(|b| (b * 5 + 3) % 11).collect();
                    let mut pool_k = vec![0f32; 11 * n_kv_head * BLOCK_SIZE * d];
                    let mut pool_v = pool_k.clone();
                    let keys = rng.values(n_kv_head * l * d, 1.0);
                    let values = rng.values(n_kv_head * l * d, 1.0);
                    store_rows(&table, 0, l, &keys, &mut pool_k, shape).unwrap();
                    store_rows(&table, 0, l, &values, &mut pool_v, shape).unwrap();
                    let q = rng.values(n_head * d, 2.0);
                    let scale = 1.0 / (d as f32).sqrt();
                    let got = attend_paged_row(&q, &table, lo, l, &pool_k, &pool_v, shape, scale)
                        .unwrap();

                    // Reference: the same keys as one contiguous tensor.
                    let dev = &Device::Cpu;
                    let slice = |all: &[f32]| -> Tensor {
                        Tensor::from_vec(all.to_vec(), (1, n_kv_head, l, d), dev)
                            .unwrap()
                            .narrow(2, lo, l - lo)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                    };
                    let want = attend(
                        &Tensor::from_vec(q.clone(), (1, n_head, 1, d), dev).unwrap(),
                        &slice(&keys),
                        &slice(&values),
                        None,
                        scale,
                    )
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                    assert_eq!(got.len(), want.len());
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-5,
                            "heads {n_head}/{n_kv_head}, d {d}, l {l}, lo {lo}, \
                             output {i}: {g} vs {w}"
                        );
                    }
                    // The gathered range equals the contiguous slice.
                    let gathered = gather(&table, lo, l, &pool_k, shape).unwrap();
                    let direct = slice(&keys)
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    assert_eq!(gathered, direct);
                }
            }
        }
    }

    /// A row is bit-identical alone and inside a batch of rows with other
    /// ranges, whichever thread runs it and whatever the block ids are.
    #[test]
    fn a_row_is_bit_identical_alone_and_in_a_batch() {
        let mut rng = Lcg(7);
        let shape = PoolShape {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 8,
        };
        let l = 40;
        for table in [[2u32, 0, 1], [0, 1, 2]] {
            let mut pool_k = vec![0f32; 3 * 2 * BLOCK_SIZE * 8];
            let mut pool_v = pool_k.clone();
            store_rows(
                &table,
                0,
                l,
                &rng.values(2 * l * 8, 1.0),
                &mut pool_k,
                shape,
            )
            .unwrap();
            store_rows(
                &table,
                0,
                l,
                &rng.values(2 * l * 8, 1.0),
                &mut pool_v,
                shape,
            )
            .unwrap();
            let t = 9;
            let q = rng.values(t * 4 * 8, 1.0);
            let range = |i: usize| (i / 3, l - t + i + 1);
            let all = attend_paged_rows(
                &q,
                t,
                range,
                &table,
                &pool_k,
                &pool_v,
                shape,
                0.35,
            )
            .unwrap();
            for i in 0..t {
                let one = attend_paged_rows(
                    &q[i * 32..(i + 1) * 32],
                    1,
                    |_| range(i),
                    &table,
                    &pool_k,
                    &pool_v,
                    shape,
                    0.35,
                )
                .unwrap();
                assert_eq!(
                    &all[i * 32..(i + 1) * 32],
                    &one[..],
                    "row {i}, table {table:?}"
                );
            }
        }
    }

    /// A range over a position without a block is an error, not garbage.
    #[test]
    fn a_missing_block_is_an_error() {
        let shape = PoolShape {
            n_head: 1,
            n_kv_head: 1,
            head_dim: 4,
        };
        let pool = vec![0f32; 2 * BLOCK_SIZE * 4];
        let q = [0.0; 4];
        let table = [NO_BLOCK, 1];
        assert!(attend_paged_row(&q, &table, 0, 20, &pool, &pool, shape, 1.0).is_err());
        // Only the second block is read: fine.
        assert!(attend_paged_row(&q, &table, 16, 20, &pool, &pool, shape, 1.0).is_ok());
        // A block id past the pool.
        let table = [5];
        assert!(attend_paged_row(&q, &table, 0, 4, &pool, &pool, shape, 1.0).is_err());
    }
}
