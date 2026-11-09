//! The indexer of the compressed sparse attention on the GPU (CubeCL): score
//! the compressed rows a query can see and keep the best `top_k`, exactly as
//! the CPU engine's `index` does (see the module docs of `cpu::hyper`).
//!
//! A query at position `p` sees the compressed rows `j < (p + 1) / ratio`.
//! The score of row `j` is `sum_h relu(iq_h . key_j) * w_h`, accumulated over
//! the heads in order, every dot product with the CPU's eight running partial
//! sums ([`hc::dot`](super::hc::dot)); the key of row `j` is the columns
//! `key_col..key_col + dim` of the compressed pool row the block table maps
//! it to (`table[j / 16] * 16 + j % 16`). Three kernels do the work, each
//! row independent (a row's result does not depend on which other rows share
//! the launch):
//!
//! 1. [`TopkKernels::index_scores`]: writes `scores[row, j]` of a `[rows,
//!    width]` matrix, one cube of 64 units per (row, tile of 64 slots), one
//!    unit per slot. The cube stages the row's queries (`heads * dim`
//!    floats) in shared memory once and each unit holds its key in registers,
//!    so a multiply-add costs one broadcast shared read. A slot that is not
//!    scored (past the visible rows, a block the candidate list of a
//!    *consumer* layer does not hold, a table entry without a block) is
//!    `-inf`, which is what the CPU fills its score vector with.
//! 2. [`TopkKernels::index_block_scores`] (candidate source layers): the
//!    score of a block of `cand_size` rows is the best score in it; the block
//!    holding the newest visible row `(vis - 1) / cand_size` is pinned with
//!    `+inf`; blocks without a visible row are `-inf`.
//! 3. [`TopkKernels::index_select`]: the `min(k, eligible)` best entries of
//!    every row, where an entry is eligible when its score is above `-inf`
//!    (NaN is not), ties between equal scores going to the LOWER index
//!    (`total_cmp` on the score, then the index), returned sorted ascending
//!    and padded with `-1`. One cube of 256 units per row maps every score to
//!    an order-preserving `u32` key (the bits of `total_cmp`) and runs a radix
//!    select, four bits per round: each round counts, among the keys that
//!    share the digits settled so far, those at or above each of the 15
//!    cutoffs and takes the largest digit that still reaches the wanted rank,
//!    so after eight rounds the `k`-th largest key and the number of keys
//!    equal to it that are still wanted are known. The gather then walks the
//!    row in tiles of 2048 keys (loaded coalesced into shared memory) and an
//!    inclusive scan of the units' counts places every winner, so the winners
//!    come out in ascending index order without a sort: every key above the
//!    threshold plus the first keys at it, the lowest indices first. No
//!    atomics and no readback: the winners stay on the device.
//!
//! [`index_topk`] runs them over tiles of rows so that the score matrix of a
//! tile stays under [`MAX_TILE_SCORES`] floats (64 MB), whatever the context
//! length. The selection is native (device-side) for the rows and for the
//! candidate blocks, there is no host readback in the indexer; the single
//! cube per row is what bounds a long context (about 1 ms for one row of
//! 130k scores, linear in the width).
//!
//! The kernels are safe Rust (`#[cube(launch)]`, the checked `launch`) reached
//! through burn's `backend_extension`, like [`moe_kernel`](super::moe_kernel)
//! and [`paged_kernel`](super::paged_kernel).

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values (`tid - tid` is a runtime zero where a
// literal would be evaluated by rustc), plain `x = x / 2` and nested `if`s.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]

use burn::backend::{
    Dispatch, TensorMetadata, backend_extension, tensor::FloatTensor, tensor::IntTensor,
};
use burn::cubecl::prelude::*;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Int, Shape, Tensor};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};

use self::kernels::{block_scores_kernel, index_scores_kernel, select_kernel};
use super::csa::Candidates;
use super::paged::BLOCK_SIZE;

/// Most scores one tile of rows holds: 16 Mi floats, 64 MB (one storage
/// buffer, far under wgpu's 1 GiB limit).
pub(crate) const MAX_TILE_SCORES: usize = 16 << 20;

/// Most rows of one tile (one selection cube per row, 65535 at most).
const MAX_TILE_ROWS: usize = 32_768;

/// Units of a scoring cube (slots of the score matrix per cube).
const SCORE_TILE: usize = 64;

/// Most query floats (`heads * dim`) a scoring cube stages in shared memory
/// (32 KiB, the workgroup storage the device guarantees here).
const MAX_STAGED_QUERY: usize = 8192;

/// Most cubes along one grid dimension.
const MAX_GRID: usize = 65_535;

/// Units of a selection cube and the depth of its reduction tree.
const SELECT_DIM: usize = 256;
const SELECT_STEPS: usize = 8;

/// Strided loads one selection unit issues per iteration of a counting round.
const SELECT_UNROLL: usize = 8;

/// What one indexer call needs to know beyond its tensors.
#[derive(Clone, Debug)]
pub(crate) struct IndexCall {
    /// Indexer heads.
    pub(crate) heads: usize,
    /// Width of a head's query and of the index key.
    pub(crate) dim: usize,
    /// First column of the index key in a pool row (the attention head dim).
    pub(crate) key_col: usize,
    /// Floats of one pool row.
    pub(crate) row_width: usize,
    /// Tokens pooled into one compressed row.
    pub(crate) ratio: usize,
    /// Rows to keep per query.
    pub(crate) top_k: usize,
    /// Whether the layer publishes (`Source`) or narrows to (`Consumer`) the
    /// candidate blocks, or neither.
    pub(crate) cand: Candidates,
    /// Blocks a source keeps (output width of its candidate list).
    pub(crate) cand_blocks: usize,
    /// Compressed rows per candidate block.
    pub(crate) cand_size: usize,
    /// An upper bound on the visible rows `(p + 1) / ratio` of every row of
    /// the call (the host knows the positions): sizes the score matrix. A row
    /// with more visible rows would silently lose the rows past it.
    pub(crate) max_visible: usize,
}

impl IndexCall {
    /// Columns of the score matrix.
    fn width(&self) -> usize {
        self.max_visible.max(1)
    }
}

/// What an indexer call returns.
pub(crate) struct IndexOut {
    /// `[rows, top_k]` visible compressed rows to attend, ascending, `-1`
    /// padded.
    pub(crate) picks: Tensor<2, Int>,
    /// `[rows, cand_blocks]` candidate blocks, ascending, `-1` padded
    /// (`Source` only).
    pub(crate) cand: Option<Tensor<2, Int>>,
}

/// The scalars of one scoring launch and the shapes the fusion layer needs.
#[derive(Clone, Debug)]
pub(crate) struct ScoreCall {
    rows: usize,
    width: usize,
    heads: usize,
    dim: usize,
    key_col: usize,
    row_width: usize,
    ratio: usize,
    cand_blocks: usize,
    cand_size: usize,
    consumer: bool,
}

/// The scalars of one block-score launch.
#[derive(Clone, Debug)]
pub(crate) struct BlockCall {
    rows: usize,
    width: usize,
    blocks: usize,
    ratio: usize,
    cand_size: usize,
}

/// The scalars of one selection launch.
#[derive(Clone, Debug)]
pub(crate) struct SelectCall {
    rows: usize,
    width: usize,
    k: usize,
}

// ------------------------------------------------------------------ kernels

/// The device code. It lives in a module of its own because the CubeCL
/// prelude's `Tensor<T>` (a kernel argument) and burn's `Tensor<D>` share a name.
mod kernels {
    use burn::cubecl::prelude::*;

    /// The key of a score: `u32` order equals `f32::total_cmp` order, and every
    /// score that is not above `-inf` (`-inf`, negative NaNs, positive NaNs) is
    /// `0`, below every eligible one (eligible keys are `0x0080_0000..=
    /// 0xFF80_0000`).
    #[cube]
    fn eff_key(score: f32) -> u32 {
        let bits = u32::reinterpret(score);
        let mut key = bits | 0x8000_0000u32;
        if (bits >> 31u32) != 0u32 {
            key = bits ^ 0xFFFF_FFFFu32;
        }
        let mut eff = key;
        if key <= 0x007F_FFFFu32 || key > 0xFF80_0000u32 {
            eff = key - key;
        }
        eff
    }

    /// One cube of `tile` units per (row, tile of `tile` slots of the score
    /// matrix), one unit per slot. The row's queries (`heads * dim` floats) are
    /// staged in shared memory once per cube (every unit reads them in the same
    /// order, a broadcast) and each unit holds its key in registers (loaded once,
    /// the dot products then cost one shared read per multiply-add).
    #[cube(launch)]
    pub(super) fn index_scores_kernel(
        iq: &Tensor<f32>,
        wts: &Tensor<f32>,
        pool: &Tensor<f32>,
        tables: &Tensor<i32>,
        row_seq: &Tensor<i32>,
        row_pos: &Tensor<i32>,
        cand: &Tensor<u32>,
        scores: &mut Tensor<f32>,
        width: u32,
        key_col: u32,
        row_width: u32,
        ratio: u32,
        max_blocks: u32,
        block_rows: u32,
        cand_blocks: u32,
        cand_size: u32,
        consumer: u32,
        steps: u32,
        #[comptime] heads: usize,
        #[comptime] dim: usize,
        #[comptime] tile: usize,
    ) {
        let r = CUBE_POS_X as usize;
        let tid = UNIT_POS_X as usize;
        let j = CUBE_POS_Y as usize * tile + tid;
        let zero = f32::cast_from(r - r);

        let mut q = Shared::<[f32]>::new_slice(heads * dim);
        let staged = (heads * dim + tile - 1usize) / tile;
        for it in 0..staged {
            let i = it * tile + tid;
            if i < heads * dim {
                q[i] = iq[r * heads * dim + i];
            }
        }
        sync_cube();

        let vis = (usize::cast_from(row_pos[r]) + 1usize) / usize::cast_from(ratio);
        let mut score = f32::reinterpret(0xFF80_0000u32);
        if j < vis {
            // A consumer layer scores the rows of its candidate blocks only: a
            // binary search of the ascending list (the `-1` padding is the
            // largest unsigned value, so the list stays ascending).
            let mut scored = u32::cast_from(r - r) + 1u32;
            if consumer != 0u32 {
                let b = u32::cast_from(j / usize::cast_from(cand_size));
                let at = r * usize::cast_from(cand_blocks);
                let mut lo = r - r;
                let mut hi = usize::cast_from(cand_blocks);
                for _ in 0..steps {
                    if lo < hi {
                        let mid = (lo + hi) / 2usize;
                        if cand[at + mid] < b {
                            lo = mid + 1usize;
                        } else {
                            hi = mid;
                        }
                    }
                }
                scored = u32::cast_from(r - r);
                if lo < usize::cast_from(cand_blocks) {
                    if cand[at + lo] == b {
                        scored = u32::cast_from(r - r) + 1u32;
                    }
                }
            }
            if scored != 0u32 {
                // The row of the compressed pool this slot lives in.
                let seq = usize::cast_from(row_seq[r]);
                let jb = j / usize::cast_from(block_rows);
                let mut block = row_seq[r] - row_seq[r] - 1i32;
                if jb < usize::cast_from(max_blocks) {
                    block = tables[seq * usize::cast_from(max_blocks) + jb];
                }
                if block >= 0i32 {
                    let key = (usize::cast_from(block) * usize::cast_from(block_rows)
                        + j % usize::cast_from(block_rows))
                        * usize::cast_from(row_width)
                        + usize::cast_from(key_col);
                    let mut k = Array::<f32>::new(dim);
                    #[unroll]
                    for i in 0..dim {
                        k[i] = pool[key + i];
                    }
                    let mut acc = zero;
                    for h in 0..heads {
                        let at = h * dim;
                        let mut s0 = zero;
                        let mut s1 = zero;
                        let mut s2 = zero;
                        let mut s3 = zero;
                        let mut s4 = zero;
                        let mut s5 = zero;
                        let mut s6 = zero;
                        let mut s7 = zero;
                        // Eight running partial sums over the dimensions `i % 8`,
                        // combined in the CPU's fixed order (`dim` is a multiple
                        // of 8: no tail).
                        #[unroll]
                        for c in 0..dim / 8usize {
                            let i = c * 8usize;
                            s0 += q[at + i] * k[i];
                            s1 += q[at + i + 1usize] * k[i + 1usize];
                            s2 += q[at + i + 2usize] * k[i + 2usize];
                            s3 += q[at + i + 3usize] * k[i + 3usize];
                            s4 += q[at + i + 4usize] * k[i + 4usize];
                            s5 += q[at + i + 5usize] * k[i + 5usize];
                            s6 += q[at + i + 6usize] * k[i + 6usize];
                            s7 += q[at + i + 7usize] * k[i + 7usize];
                        }
                        let x = ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7));
                        if x > zero {
                            acc += x * wts[r * heads + h];
                        }
                    }
                    score = acc;
                }
            }
        }
        if j < usize::cast_from(width) {
            scores[r * usize::cast_from(width) + j] = score;
        }
    }

    /// One unit per (row, block): the best score of the block's visible rows,
    /// `+inf` for the block of the newest visible row, `-inf` for a block without
    /// an eligible score.
    #[cube(launch)]
    pub(super) fn block_scores_kernel(
        scores: &Tensor<f32>,
        row_pos: &Tensor<i32>,
        out: &mut Tensor<f32>,
        total: u32,
        width: u32,
        blocks: u32,
        ratio: u32,
        cand_size: u32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx < usize::cast_from(total) {
            let nb_all = usize::cast_from(blocks);
            let r = idx / nb_all;
            let b = idx % nb_all;
            let vis = (usize::cast_from(row_pos[r]) + 1usize) / usize::cast_from(ratio);
            let bs = usize::cast_from(cand_size);
            let mut value = f32::reinterpret(0xFF80_0000u32);
            if vis > 0usize {
                let nb = (vis + bs - 1usize) / bs;
                let last = (vis - 1usize) / bs;
                if b == last {
                    value = f32::reinterpret(0x7F80_0000u32);
                } else if b < nb {
                    let from = b * bs;
                    let mut to = from + bs;
                    if to > vis {
                        to = vis;
                    }
                    let at = r * usize::cast_from(width);
                    let mut best = u32::cast_from(r - r);
                    for i in from..to {
                        let e = eff_key(scores[at + i]);
                        if e > best {
                            best = e;
                        }
                    }
                    if best != 0u32 {
                        // The inverse of the key map.
                        let mut bits = best ^ 0xFFFF_FFFFu32;
                        if (best >> 31u32) != 0u32 {
                            bits = best ^ 0x8000_0000u32;
                        }
                        value = f32::reinterpret(bits);
                    }
                }
            }
            out[idx] = value;
        }
    }

    /// One cube of `dim` units per row: the `min(k, eligible)` best entries of
    /// the row, ascending by index, lower index first on equal scores, `-1`
    /// padded (see the module docs).
    #[cube(launch)]
    pub(super) fn select_kernel(
        scores: &Tensor<f32>,
        out: &mut Tensor<i32>,
        width: u32,
        k: u32,
        #[comptime] dim: usize,
        #[comptime] steps: usize,
        #[comptime] unroll: usize,
    ) {
        let row = CUBE_POS_X as usize;
        let tid = UNIT_POS_X as usize;
        let n = usize::cast_from(width);
        let kk = usize::cast_from(k);
        let base = row * n;
        let zero = u32::cast_from(tid - tid);
        let mut red = Shared::<[u32]>::new_slice(16usize * dim);
        // The keys of the tile the gather works on and their running counts.
        let mut keys = Shared::<[u32]>::new_slice(dim * unroll);
        let mut scan = Shared::<[u32]>::new_slice(dim);

        // The threshold search: the `want`-th largest key, four bits per round
        // from the top (a radix select). `thresh` holds the digits settled so
        // far (zeros below them) and `want` the rank still wanted among the keys
        // that share them. A round counts, per unit, the matching keys at or
        // above each of the 15 cutoffs `thresh | digit << shift` (and, in round
        // 0, the eligible keys, those of at least 1, which fixes `want`),
        // reduces the counters over the cube, takes the largest digit whose
        // count still reaches `want` and, the keys with a larger digit being
        // above the threshold, removes them from the rank. After the last round
        // `thresh` is the key itself and `want` is how many keys equal to it
        // are still wanted. The strided loads are independent of each other
        // (`unroll` per unit and iteration) so that their latency overlaps.
        let chunk = dim * unroll;
        let iters = (n + chunk - 1usize) / chunk;
        let mut thresh = zero;
        let mut want = zero;
        // How many entries win in all (`want` shrinks as keys above the threshold
        // are found).
        let mut winners = zero;
        for round in 0..8usize {
            let shift = 28u32 - 4u32 * u32::cast_from(round);
            // The bits a key shares with the settled digits (none in round 0).
            let mut mask = zero;
            if round > 0usize {
                mask = 0xFFFF_FFFFu32 << (shift + 4u32);
            }
            let mut cut = Array::<u32>::new(16usize);
            let mut cnt = Array::<u32>::new(16usize);
            #[unroll]
            for d in 0..16usize {
                cut[d] = thresh | (u32::cast_from(d) << shift);
                cnt[d] = zero;
            }
            for it in 0..iters {
                // All loads of the iteration first (an index past the row reads
                // its last entry and is masked below): none waits for a branch.
                let mut v = Array::<f32>::new(unroll);
                #[unroll]
                for u in 0..unroll {
                    let i = it * chunk + u * dim + tid;
                    let mut at = i;
                    if i >= n {
                        at = n - 1usize;
                    }
                    v[u] = scores[base + at];
                }
                #[unroll]
                for u in 0..unroll {
                    let i = it * chunk + u * dim + tid;
                    let mut e = eff_key(v[u]);
                    if i >= n {
                        e = zero;
                    }
                    if e >= 1u32 {
                        cnt[0] += 1u32;
                    }
                    if (e & mask) == thresh {
                        #[unroll]
                        for d in 1..16usize {
                            if e >= cut[d] {
                                cnt[d] += 1u32;
                            }
                        }
                    }
                }
            }
            #[unroll]
            for d in 0..16usize {
                red[d * dim + tid] = cnt[d];
            }
            sync_cube();
            let mut s = (tid - tid) + dim / 2usize;
            for _ in 0..steps {
                if tid < s {
                    #[unroll]
                    for d in 0..16usize {
                        red[d * dim + tid] = red[d * dim + tid] + red[d * dim + tid + s];
                    }
                }
                sync_cube();
                s = s / 2usize;
            }
            // Every unit reads the totals (a broadcast) and reaches the same
            // threshold.
            if round == 0usize {
                want = red[0];
                if k < want {
                    want = k;
                }
                winners = want;
            }
            let mut digit = zero;
            #[unroll]
            for d in 1..16usize {
                if red[d * dim] >= want {
                    digit = u32::cast_from(d);
                }
            }
            let mut above = zero;
            #[unroll]
            for d in 0..15usize {
                if digit == u32::cast_from(d) {
                    above = red[(d + 1usize) * dim];
                }
            }
            want = want - above;
            thresh = thresh | (digit << shift);
            sync_cube();
        }
        let quota = want;

        // The gather, a tile of `chunk` consecutive keys at a time: the tile is
        // loaded coalesced into shared memory, each unit takes `unroll`
        // consecutive keys of it, an inclusive scan of the units' counts (keys
        // above the threshold in the low half word, keys at it in the high one)
        // places every unit's winners after those of the units before it, and
        // the winners keep ascending index order without a sort. Of the keys at
        // the threshold the first `qrem` still wanted win.
        let mut out_at = zero;
        let mut qrem = quota;
        for tile in 0..iters {
            let mut v = Array::<f32>::new(unroll);
            #[unroll]
            for u in 0..unroll {
                let i = tile * chunk + u * dim + tid;
                let mut at = i;
                if i >= n {
                    at = n - 1usize;
                }
                v[u] = scores[base + at];
            }
            #[unroll]
            for u in 0..unroll {
                let i = tile * chunk + u * dim + tid;
                let mut e = eff_key(v[u]);
                if i >= n {
                    e = zero;
                }
                keys[u * dim + tid] = e;
            }
            sync_cube();
            let mut g = zero;
            let mut at_it = zero;
            #[unroll]
            for u in 0..unroll {
                let e = keys[tid * unroll + u];
                if e > thresh {
                    g += 1u32;
                } else if e == thresh {
                    at_it += 1u32;
                }
            }
            let mine = g | (at_it << 16u32);
            scan[tid] = mine;
            sync_cube();
            let mut s = (tid - tid) + 1usize;
            for _ in 0..steps {
                let mut add = zero;
                if tid >= s {
                    add = scan[tid - s];
                }
                sync_cube();
                scan[tid] = scan[tid] + add;
                sync_cube();
                s = s * 2usize;
            }
            let total = scan[dim - 1usize];
            let before = scan[tid] - mine;
            let above_before = before & 0xFFFFu32;
            let at_before = before >> 16u32;
            let mut taken_before = at_before;
            if qrem < at_before {
                taken_before = qrem;
            }
            let mut pos = out_at + above_before + taken_before;
            let mut seen = at_before;
            #[unroll]
            for u in 0..unroll {
                let e = keys[tid * unroll + u];
                let mut keep = zero;
                if e > thresh {
                    keep = zero + 1u32;
                } else if e == thresh {
                    if seen < qrem {
                        keep = zero + 1u32;
                    }
                    seen += 1u32;
                }
                if keep != 0u32 {
                    out[row * kk + usize::cast_from(pos)] =
                        i32::cast_from(tile * chunk + tid * unroll + u);
                    pos += 1u32;
                }
            }
            let tile_above = total & 0xFFFFu32;
            let tile_at = total >> 16u32;
            let mut take = tile_at;
            if qrem < tile_at {
                take = qrem;
            }
            out_at += tile_above + take;
            qrem -= take;
            sync_cube();
        }

        // Padding.
        let pad = i32::cast_from(zero) - 1i32;
        let fills = (kk + dim - 1usize) / dim;
        let first = usize::cast_from(winners);
        for it in 0..fills {
            let slot = first + it * dim + tid;
            if slot < kk {
                out[row * kk + slot] = pad;
            }
        }
    }
}

// --------------------------------------------------------------- extension

/// Backend operations of the indexer (see the module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait TopkKernels: burn::backend::Backend {
    /// Scores `[rows, width]` of the visible compressed rows of every query
    /// row. `iq` is `[rows, heads * dim]`, `wts` `[rows, heads]`, `pool`
    /// `[pool_rows, row_width]`, `tables` `[n_seq, max_blocks]` (`-1` = no
    /// block), `row_seq`/`row_pos` `[rows]` and `cand` `[rows, cand_blocks]`
    /// (read only when `call.consumer`; any int tensor otherwise).
    #[fusion(dtype = iq, shape = Shape::new([call.rows, call.width]))]
    fn index_scores(
        iq: FloatTensor<Self>,
        wts: FloatTensor<Self>,
        pool: FloatTensor<Self>,
        tables: IntTensor<Self>,
        row_seq: IntTensor<Self>,
        row_pos: IntTensor<Self>,
        cand: IntTensor<Self>,
        call: ScoreCall,
    ) -> FloatTensor<Self>;

    /// The block scores `[rows, blocks]` of a score matrix.
    #[fusion(dtype = scores, shape = Shape::new([call.rows, call.blocks]))]
    fn index_block_scores(
        scores: FloatTensor<Self>,
        row_pos: IntTensor<Self>,
        call: BlockCall,
    ) -> FloatTensor<Self>;

    /// The best `call.k` eligible entries of every row of `scores`:
    /// `[rows, k]` indices, ascending, `-1` padded.
    #[fusion(dtype = DType::I32, shape = Shape::new([call.rows, call.k]))]
    fn index_select(scores: FloatTensor<Self>, call: SelectCall) -> IntTensor<Self>;
}

impl TopkKernels for CubeBackend {
    fn index_scores(
        iq: CubeTensor,
        wts: CubeTensor,
        pool: CubeTensor,
        tables: CubeTensor,
        row_seq: CubeTensor,
        row_pos: CubeTensor,
        cand: CubeTensor,
        call: ScoreCall,
    ) -> CubeTensor {
        // The kernels index the buffers linearly.
        let iq = into_contiguous(iq);
        let wts = into_contiguous(wts);
        let pool = into_contiguous(pool);
        let tables = into_contiguous(tables);
        let row_seq = into_contiguous(row_seq);
        let row_pos = into_contiguous(row_pos);
        let cand = into_contiguous(cand);
        let max_blocks = tables.shape().dims::<2>()[1];
        let client = iq.client.clone();
        let out = empty_device_dtype(
            client.clone(),
            iq.device.clone(),
            Shape::new([call.rows, call.width]),
            DType::F32,
        );
        assert!(
            call.dim.is_multiple_of(8) && call.heads * call.dim <= MAX_STAGED_QUERY,
            "the indexer kernel needs dim % 8 == 0 and heads * dim <= {MAX_STAGED_QUERY} (got {} x {})",
            call.heads,
            call.dim
        );
        let tiles = call.width.div_ceil(SCORE_TILE);
        assert!(
            call.rows <= MAX_GRID && tiles <= MAX_GRID,
            "a scoring launch of {} rows x {} slots exceeds the grid",
            call.rows,
            call.width
        );
        // Steps of the binary search of the candidate list.
        let steps = usize::BITS - call.cand_blocks.leading_zeros();
        index_scores_kernel::launch(
            &client,
            CubeCount::Static(call.rows as u32, tiles as u32, 1),
            CubeDim::new_1d(SCORE_TILE as u32),
            iq.into_tensor_arg(),
            wts.into_tensor_arg(),
            pool.into_tensor_arg(),
            tables.into_tensor_arg(),
            row_seq.into_tensor_arg(),
            row_pos.into_tensor_arg(),
            cand.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            call.width as u32,
            call.key_col as u32,
            call.row_width as u32,
            call.ratio as u32,
            max_blocks as u32,
            BLOCK_SIZE as u32,
            call.cand_blocks as u32,
            call.cand_size as u32,
            u32::from(call.consumer),
            steps,
            call.heads,
            call.dim,
            SCORE_TILE,
        );
        out
    }

    fn index_block_scores(scores: CubeTensor, row_pos: CubeTensor, call: BlockCall) -> CubeTensor {
        let scores = into_contiguous(scores);
        let row_pos = into_contiguous(row_pos);
        let client = scores.client.clone();
        let out = empty_device_dtype(
            client.clone(),
            scores.device.clone(),
            Shape::new([call.rows, call.blocks]),
            DType::F32,
        );
        let total = call.rows * call.blocks;
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        block_scores_kernel::launch(
            &client,
            count,
            dim,
            scores.into_tensor_arg(),
            row_pos.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            call.width as u32,
            call.blocks as u32,
            call.ratio as u32,
            call.cand_size as u32,
        );
        out
    }

    fn index_select(scores: CubeTensor, call: SelectCall) -> CubeTensor {
        let scores = into_contiguous(scores);
        let client = scores.client.clone();
        let out = empty_device_dtype(
            client.clone(),
            scores.device.clone(),
            Shape::new([call.rows, call.k]),
            DType::I32,
        );
        select_kernel::launch(
            &client,
            CubeCount::Static(call.rows as u32, 1, 1),
            CubeDim::new_1d(SELECT_DIM as u32),
            scores.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            call.width as u32,
            call.k as u32,
            SELECT_DIM,
            SELECT_STEPS,
            SELECT_UNROLL,
        );
        out
    }
}

// ---------------------------------------------------------- tensor wrappers

/// The score matrix `[rows, call.max_visible]` of `rows` query rows (no
/// tiling: the caller keeps `rows * max_visible` under [`MAX_TILE_SCORES`]).
/// `cand_in` is `[rows, cand_blocks]` for a `Consumer` call, else ignored.
pub(crate) fn scores(
    call: &IndexCall,
    iq: Tensor<2>,
    wts: Tensor<2>,
    pool: &Tensor<2>,
    tables: &Tensor<2, Int>,
    row_seq: &Tensor<1, Int>,
    row_pos: &Tensor<1, Int>,
    cand_in: Option<&Tensor<2, Int>>,
) -> Tensor<2> {
    let rows = iq.dims()[0];
    let consumer = call.cand == Candidates::Consumer;
    // The candidate list is read by a consumer only; any int tensor stands in
    // for it otherwise.
    let cand = if consumer {
        cand_in
            .expect("a consumer layer needs the candidate blocks")
            .clone()
            .into_dispatch()
    } else {
        row_pos.clone().into_dispatch()
    };
    let launch = ScoreCall {
        rows,
        width: call.width(),
        heads: call.heads,
        dim: call.dim,
        key_col: call.key_col,
        row_width: call.row_width,
        ratio: call.ratio,
        cand_blocks: call.cand_blocks,
        cand_size: call.cand_size.max(1),
        consumer,
    };
    Tensor::from_dispatch(<Dispatch as TopkKernels>::index_scores(
        iq.into_dispatch(),
        wts.into_dispatch(),
        pool.clone().into_dispatch(),
        tables.clone().into_dispatch(),
        row_seq.clone().into_dispatch(),
        row_pos.clone().into_dispatch(),
        cand,
        launch,
    ))
}

/// The block scores `[rows, blocks]` of the score matrix `scores`.
pub(crate) fn block_scores(
    call: &IndexCall,
    scores: &Tensor<2>,
    row_pos: &Tensor<1, Int>,
) -> Tensor<2> {
    let [rows, width] = scores.dims();
    let launch = BlockCall {
        rows,
        width,
        blocks: width.div_ceil(call.cand_size.max(1)),
        ratio: call.ratio,
        cand_size: call.cand_size.max(1),
    };
    Tensor::from_dispatch(<Dispatch as TopkKernels>::index_block_scores(
        scores.clone().into_dispatch(),
        row_pos.clone().into_dispatch(),
        launch,
    ))
}

/// The best `k` entries of every row of `scores`: `[rows, k]` indices of the
/// entries above `-inf`, highest scores first by selection but returned
/// ascending by index, lower index first on equal scores, `-1` padded.
pub(crate) fn select_topk(scores: Tensor<2>, k: usize) -> Tensor<2, Int> {
    let [rows, width] = scores.dims();
    Tensor::from_dispatch(<Dispatch as TopkKernels>::index_select(
        scores.into_dispatch(),
        SelectCall { rows, width, k },
    ))
}

/// The tile `r0..r0 + n` of `rows` rows of a tensor (the tensor itself when
/// the tile is all of it).
fn tile<const D: usize, K: burn::tensor::kind::Basic>(
    t: &Tensor<D, K>,
    r0: usize,
    n: usize,
    rows: usize,
) -> Tensor<D, K> {
    if n == rows {
        t.clone()
    } else {
        t.clone().narrow(0, r0, n)
    }
}

/// Score the compressed rows each query row can see and keep the best (see
/// the module docs). Per row, `picks` is the `min(top_k, visible)` best
/// visible rows ascending by row id and `-1` padded; for a `Source` layer
/// `cand` is the best `cand_blocks` blocks (the block of the newest visible
/// row always among them) ascending and `-1` padded; a `Consumer` layer
/// scores only the rows of the blocks of `cand_in` (`[rows, cand_blocks]`).
///
/// `iq` is `[rows, heads * dim]` (rotated and FP4-simulated), `wts`
/// `[rows, heads]` (scaled), `pool` the compressed pool `[pool_rows,
/// row_width]` (contiguous), `tables` `[n_seq, max_blocks]` block ids (`-1` =
/// none: a slot in such a block is not scored, where the CPU errors),
/// `row_seq`/`row_pos` `[rows]`. A row's result never depends on the other
/// rows: tiles of rows share nothing.
pub(crate) fn index_topk(
    call: &IndexCall,
    iq: Tensor<2>,
    wts: Tensor<2>,
    pool: &Tensor<2>,
    tables: &Tensor<2, Int>,
    row_seq: Tensor<1, Int>,
    row_pos: Tensor<1, Int>,
    cand_in: Option<&Tensor<2, Int>>,
) -> IndexOut {
    index_topk_tiled(
        call,
        MAX_TILE_SCORES,
        iq,
        wts,
        pool,
        tables,
        row_seq,
        row_pos,
        cand_in,
    )
}

/// [`index_topk`] with tiles of at most `max_tile_scores` scores.
fn index_topk_tiled(
    call: &IndexCall,
    max_tile_scores: usize,
    iq: Tensor<2>,
    wts: Tensor<2>,
    pool: &Tensor<2>,
    tables: &Tensor<2, Int>,
    row_seq: Tensor<1, Int>,
    row_pos: Tensor<1, Int>,
    cand_in: Option<&Tensor<2, Int>>,
) -> IndexOut {
    let rows = iq.dims()[0];
    let source = call.cand == Candidates::Source;
    assert!(
        rows > 0 && call.top_k > 0 && (!source || call.cand_blocks > 0),
        "an indexer call needs at least one row, top_k and (for a source) candidate block"
    );
    let rows_per_tile = (max_tile_scores / call.width()).clamp(1, MAX_TILE_ROWS);
    let mut picks = Vec::new();
    let mut cands = Vec::new();
    let mut r0 = 0;
    while r0 < rows {
        let n = rows_per_tile.min(rows - r0);
        let (q, w) = (tile(&iq, r0, n, rows), tile(&wts, r0, n, rows));
        let (rs, rp) = (tile(&row_seq, r0, n, rows), tile(&row_pos, r0, n, rows));
        let cin = cand_in.map(|c| tile(c, r0, n, rows));
        let s = scores(call, q, w, pool, tables, &rs, &rp, cin.as_ref());
        if source {
            let b = block_scores(call, &s, &rp);
            cands.push(select_topk(b, call.cand_blocks));
        }
        picks.push(select_topk(s, call.top_k));
        r0 += n;
    }
    let join = |mut parts: Vec<Tensor<2, Int>>| {
        if parts.len() == 1 {
            parts.swap_remove(0)
        } else {
            Tensor::cat(parts, 0)
        }
    };
    IndexOut {
        picks: join(picks),
        cand: source.then(|| join(cands)),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use burn::tensor::{Device, TensorData};

    use super::super::gpu::{kernel_test_device, read_f32};
    use super::super::hc::dot;
    use super::*;

    /// A tiny xorshift generator: the tests must not depend on a crate.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A float in `[-1, 1)`.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }

        /// An integer in `lo..=hi`.
        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }

        fn shuffle<T>(&mut self, v: &mut [T]) {
            for i in (1..v.len()).rev() {
                v.swap(i, (self.next() % (i as u64 + 1)) as usize);
            }
        }
    }

    fn ints<const D: usize>(t: Tensor<D, Int>) -> Vec<i32> {
        t.try_into_data()
            .expect("reading ints")
            .try_into_vec::<i32>()
            .expect("int values")
    }

    fn int_tensor<const D: usize>(
        v: Vec<i32>,
        dims: [usize; D],
        device: &Device,
    ) -> Tensor<D, Int> {
        Tensor::from_data(TensorData::new(v, dims), device)
    }

    fn float_tensor<const D: usize>(v: Vec<f32>, dims: [usize; D], device: &Device) -> Tensor<D> {
        Tensor::from_data(TensorData::new(v, dims), device)
    }

    /// One indexer call with the host copies of its inputs.
    struct World {
        call: IndexCall,
        iq: Vec<f32>,
        wts: Vec<f32>,
        pool: Vec<f32>,
        pool_rows: usize,
        tables: Vec<i32>,
        max_blocks: usize,
        row_seq: Vec<i32>,
        row_pos: Vec<i32>,
    }

    impl World {
        fn rows(&self) -> usize {
            self.row_pos.len()
        }

        fn vis(&self, r: usize) -> usize {
            (self.row_pos[r] as usize + 1) / self.call.ratio
        }
    }

    /// How the values of a world are drawn.
    #[derive(Clone, Copy)]
    enum Fill {
        /// Small integers and quarter weights: every sum is exact, so the
        /// device and the host agree bit for bit and many scores tie.
        Exact,
        /// Floats: scores agree to rounding.
        Float,
        /// All queries zero: every score is exactly zero, every row ties.
        Zero,
    }

    /// A world with `n_seq` sequences of `seq_rows` compressed rows each (the
    /// block ids shuffled over the pool, one spare block per sequence left
    /// out of the table as `-1`), and one query row per entry of `queries`
    /// `(seq, pos)`.
    fn world(
        call: IndexCall,
        n_seq: usize,
        seq_rows: usize,
        queries: &[(usize, usize)],
        fill: Fill,
        seed: u64,
    ) -> World {
        let mut rng = Rng(seed);
        let per_seq = seq_rows.div_ceil(BLOCK_SIZE);
        let max_blocks = per_seq + 1;
        let pool_blocks = n_seq * per_seq + 3;
        let mut ids: Vec<i32> = (0..pool_blocks as i32).collect();
        rng.shuffle(&mut ids);
        let mut tables = vec![-1i32; n_seq * max_blocks];
        for s in 0..n_seq {
            for b in 0..per_seq {
                tables[s * max_blocks + b] = ids[s * per_seq + b];
            }
        }
        let pool_rows = pool_blocks * BLOCK_SIZE;
        let draw = |rng: &mut Rng| match fill {
            Fill::Float => rng.unit(),
            _ => rng.int(-2, 2) as f32,
        };
        let pool: Vec<f32> = (0..pool_rows * call.row_width)
            .map(|_| draw(&mut rng))
            .collect();
        let rows = queries.len();
        let iq: Vec<f32> = (0..rows * call.heads * call.dim)
            .map(|_| {
                if matches!(fill, Fill::Zero) {
                    0.0
                } else {
                    draw(&mut rng)
                }
            })
            .collect();
        let wts: Vec<f32> = (0..rows * call.heads)
            .map(|_| match fill {
                // Positive: the score is a sum of non-negative terms, so a
                // relative tolerance is meaningful (a signed sum of terms
                // much larger than itself is not).
                Fill::Float => rng.unit() * 0.0125 + 0.0375,
                _ => rng.int(-1, 3) as f32 * 0.25,
            })
            .collect();
        World {
            call,
            iq,
            wts,
            pool,
            pool_rows,
            tables,
            max_blocks,
            row_seq: queries.iter().map(|q| q.0 as i32).collect(),
            row_pos: queries.iter().map(|q| q.1 as i32).collect(),
        }
    }

    fn with_max_visible(mut w: World) -> World {
        let max = (0..w.rows()).map(|r| w.vis(r)).max().unwrap_or(0);
        w.call.max_visible = max;
        w
    }

    /// The scores of the CPU `index` of the query rows `rows`, `-inf` where it
    /// leaves them. `cand_in` (`[rows.len(), cand_blocks]`) makes it a
    /// consumer layer: only the rows of those blocks are scored.
    fn host_scores(w: &World, rows: std::ops::Range<usize>, cand_in: Option<&[i32]>) -> Vec<f32> {
        let call = &w.call;
        let width = call.width();
        let consumer = cand_in.is_some();
        let mut out = vec![f32::NEG_INFINITY; rows.len() * width];
        for r in rows.clone() {
            let local = r - rows.start;
            let seq = w.row_seq[r] as usize;
            let q = &w.iq[r * call.heads * call.dim..(r + 1) * call.heads * call.dim];
            let wt = &w.wts[r * call.heads..(r + 1) * call.heads];
            let own: Vec<i32> = cand_in
                .map(|c| {
                    c[local * call.cand_blocks..(local + 1) * call.cand_blocks]
                        .iter()
                        .copied()
                        .filter(|&b| b >= 0)
                        .collect()
                })
                .unwrap_or_default();
            for j in 0..w.vis(r) {
                if consumer && !own.contains(&((j / call.cand_size) as i32)) {
                    continue;
                }
                // `pool_row`: no entry (past the table) or `NO_BLOCK`.
                let Some(&block) =
                    w.tables[seq * w.max_blocks..(seq + 1) * w.max_blocks].get(j / BLOCK_SIZE)
                else {
                    continue;
                };
                if block < 0 {
                    continue;
                }
                let at =
                    (block as usize * BLOCK_SIZE + j % BLOCK_SIZE) * call.row_width + call.key_col;
                let key = &w.pool[at..at + call.dim];
                let mut score = 0.0;
                for (h, &wh) in wt.iter().enumerate() {
                    let x = dot(&q[h * call.dim..(h + 1) * call.dim], key);
                    if x > 0.0 {
                        score += x * wh;
                    }
                }
                out[local * width + j] = score;
            }
        }
        out
    }

    /// The selection of the CPU `index` from the scores of one query row:
    /// `(picks, candidate blocks)`, unpadded.
    fn host_pick(scores: &[f32], vis: usize, call: &IndexCall) -> (Vec<u32>, Vec<u32>) {
        let scores = &scores[..vis];
        let by = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        let mut cand = Vec::new();
        if call.cand == Candidates::Source && vis > 0 {
            let bs = call.cand_size;
            let nb = vis.div_ceil(bs);
            let last = (vis - 1) / bs;
            let mut blocks: Vec<(f32, u32)> = (0..nb)
                .map(|b| {
                    let best = if b == last {
                        f32::INFINITY
                    } else {
                        scores[b * bs..((b + 1) * bs).min(vis)]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max)
                    };
                    (best, b as u32)
                })
                .collect();
            let n = call.cand_blocks.min(nb);
            if 0 < n && n < nb {
                blocks.select_nth_unstable_by(n - 1, by);
            }
            blocks.truncate(n);
            cand = blocks
                .into_iter()
                .filter(|b| b.0 > f32::NEG_INFINITY)
                .map(|b| b.1)
                .collect();
            cand.sort_unstable();
        }
        let mut order: Vec<(f32, u32)> = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > f32::NEG_INFINITY)
            .map(|(j, &s)| (s, j as u32))
            .collect();
        let k = call.top_k.min(order.len());
        if 0 < k && k < order.len() {
            order.select_nth_unstable_by(k - 1, by);
        }
        order.truncate(k);
        let mut chosen: Vec<u32> = order.iter().map(|o| o.1).collect();
        chosen.sort_unstable();
        (chosen, cand)
    }

    fn padded(v: &[u32], len: usize) -> Vec<i32> {
        let mut out: Vec<i32> = v.iter().map(|&x| x as i32).collect();
        out.resize(len, -1);
        out
    }

    /// The device tensors of a world.
    struct Dev {
        iq: Tensor<2>,
        wts: Tensor<2>,
        pool: Tensor<2>,
        tables: Tensor<2, Int>,
        row_seq: Tensor<1, Int>,
        row_pos: Tensor<1, Int>,
    }

    fn upload(w: &World, device: &Device) -> Dev {
        let call = &w.call;
        let rows = w.rows();
        Dev {
            iq: float_tensor(w.iq.clone(), [rows, call.heads * call.dim], device),
            wts: float_tensor(w.wts.clone(), [rows, call.heads], device),
            pool: float_tensor(w.pool.clone(), [w.pool_rows, call.row_width], device),
            tables: int_tensor(
                w.tables.clone(),
                [w.tables.len() / w.max_blocks, w.max_blocks],
                device,
            ),
            row_seq: int_tensor(w.row_seq.clone(), [rows], device),
            row_pos: int_tensor(w.row_pos.clone(), [rows], device),
        }
    }

    fn assert_scores_close(gpu: &[f32], host: &[f32], tolerance: f32, what: &str) {
        assert_eq!(gpu.len(), host.len(), "{what}: length");
        for (i, (&g, &h)) in gpu.iter().zip(host).enumerate() {
            if g == h {
                continue;
            }
            assert!(
                g.is_finite() && h.is_finite(),
                "{what}: score {i} is {g} on the device and {h} on the host"
            );
            let bound = tolerance * g.abs().max(h.abs()) + 1e-7;
            assert!(
                (g - h).abs() <= bound,
                "{what}: score {i} is {g} on the device and {h} on the host"
            );
        }
    }

    /// Run the call on the device and check it against the host: the scores
    /// (`tolerance` relative; `0.0` demands bit equality), then the picks and
    /// the candidate blocks against the host selection made from the SAME
    /// device scores. Returns the device candidate list of a `Source` call.
    fn check(
        w: &World,
        device: &Device,
        tolerance: f32,
        cand_in: Option<&[i32]>,
        what: &str,
    ) -> Option<Vec<i32>> {
        let call = &w.call;
        let d = upload(w, device);
        let rows = w.rows();
        let width = call.width();
        let cin_t = cand_in.map(|c| int_tensor(c.to_vec(), [rows, call.cand_blocks], device));
        let gpu = read_f32(
            scores(
                call,
                d.iq.clone(),
                d.wts.clone(),
                &d.pool,
                &d.tables,
                &d.row_seq,
                &d.row_pos,
                cin_t.as_ref(),
            ),
            "scores",
        )
        .unwrap();
        let host = host_scores(w, 0..rows, cand_in);
        assert_scores_close(&gpu, &host, tolerance, what);

        let out = index_topk(
            call,
            d.iq.clone(),
            d.wts.clone(),
            &d.pool,
            &d.tables,
            d.row_seq.clone(),
            d.row_pos.clone(),
            cin_t.as_ref(),
        );
        // Tiles of three rows give the same result as one.
        let tiled = index_topk_tiled(
            call,
            3 * width,
            d.iq,
            d.wts,
            &d.pool,
            &d.tables,
            d.row_seq,
            d.row_pos,
            cin_t.as_ref(),
        );
        assert_eq!(
            ints(tiled.picks),
            ints(out.picks.clone()),
            "{what}: tiled picks"
        );
        assert_eq!(
            tiled.cand.map(ints),
            out.cand.clone().map(ints),
            "{what}: tiled candidates"
        );
        let picks = ints(out.picks);
        let cands = out.cand.map(ints);
        for r in 0..rows {
            let (chosen, cand) = host_pick(&gpu[r * width..(r + 1) * width], w.vis(r), call);
            assert_eq!(
                &picks[r * call.top_k..(r + 1) * call.top_k],
                &padded(&chosen, call.top_k)[..],
                "{what}: picks of row {r} (position {}, {} visible)",
                w.row_pos[r],
                w.vis(r)
            );
            if let Some(cands) = &cands {
                assert_eq!(
                    &cands[r * call.cand_blocks..(r + 1) * call.cand_blocks],
                    &padded(&cand, call.cand_blocks)[..],
                    "{what}: candidate blocks of row {r}"
                );
            }
        }
        cands
    }

    fn tiny_call(ratio: usize, cand: Candidates) -> IndexCall {
        IndexCall {
            heads: 4,
            dim: 32,
            key_col: 32,
            row_width: 64,
            ratio,
            top_k: 4,
            cand,
            cand_blocks: 2,
            cand_size: 4,
            max_visible: 0,
        }
    }

    const POSITIONS: [usize; 8] = [0, 1, 2, 7, 15, 16, 17, 40];

    fn tiny_queries(n_seq: usize) -> Vec<(usize, usize)> {
        (0..n_seq)
            .flat_map(|s| POSITIONS.iter().map(move |&p| (s, p)))
            .collect()
    }

    #[test]
    fn the_tiny_indexer_matches_the_host_in_every_mode() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        for ratio in [1, 2] {
            for fill in [Fill::Exact, Fill::Zero, Fill::Float] {
                let tolerance = if matches!(fill, Fill::Float) {
                    1e-5
                } else {
                    0.0
                };
                for (i, cand) in [Candidates::None, Candidates::Source]
                    .into_iter()
                    .enumerate()
                {
                    let w = with_max_visible(world(
                        tiny_call(ratio, cand),
                        3,
                        41,
                        &tiny_queries(3),
                        fill,
                        0x9E37_79B9 + (ratio * 10 + i) as u64,
                    ));
                    let what = format!("ratio {ratio}, {cand:?}");
                    let cands = check(&w, &device, tolerance, None, &what);
                    if cand != Candidates::Source {
                        continue;
                    }
                    // A consumer of the source's own candidates.
                    let cands = cands.expect("a source returns candidates");
                    let mut c = w;
                    c.call.cand = Candidates::Consumer;
                    check(
                        &c,
                        &device,
                        tolerance,
                        Some(&cands),
                        &format!("ratio {ratio}, consumer"),
                    );
                    // A consumer whose list is shorter than the blocks it may
                    // hold (the padding is `-1`) and an empty one.
                    let rows = c.rows();
                    let mut short = vec![-1i32; rows * 2];
                    for r in 0..rows {
                        short[r * 2] = (r % 3) as i32;
                    }
                    check(
                        &c,
                        &device,
                        tolerance,
                        Some(&short),
                        &format!("ratio {ratio}, short list"),
                    );
                    check(
                        &c,
                        &device,
                        tolerance,
                        Some(&vec![-1; rows * 2]),
                        &format!("ratio {ratio}, empty list"),
                    );
                }
            }
        }
    }

    #[test]
    fn a_table_hole_is_not_scored() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        // Ratio 1, position 70: 71 visible rows over 5 blocks, but the table
        // holds 4 (3 of the sequence's rows and the `-1`).
        let mut call = tiny_call(1, Candidates::None);
        call.max_visible = 71;
        let w = world(call, 2, 41, &[(0, 70), (1, 20)], Fill::Exact, 5);
        let d = upload(&w, &device);
        let gpu = read_f32(
            scores(
                &w.call, d.iq, d.wts, &d.pool, &d.tables, &d.row_seq, &d.row_pos, None,
            ),
            "scores",
        )
        .unwrap();
        let host = host_scores(&w, 0..2, None);
        assert_scores_close(&gpu, &host, 0.0, "holes");
        // Rows 48.. have no block: not scored.
        assert!(gpu[48..71].iter().all(|s| *s == f32::NEG_INFINITY));
        assert!(gpu[..48].iter().any(|s| *s > f32::NEG_INFINITY));
    }

    /// Scores drawn from a few values (many exact ties, `-0.0`, `+inf`,
    /// NaNs and holes) for the selection kernel.
    fn tie_scores(rng: &mut Rng, n: usize, holes: u64) -> Vec<f32> {
        const VALUES: [f32; 9] = [
            f32::NEG_INFINITY,
            -2.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.0,
            3.0,
            f32::INFINITY,
        ];
        (0..n)
            .map(|_| match rng.next() % 64 {
                0 => f32::NAN,
                1 => -f32::NAN,
                x if x < 2 + holes => f32::NEG_INFINITY,
                _ => VALUES[(rng.next() % VALUES.len() as u64) as usize],
            })
            .collect()
    }

    #[test]
    fn selection_matches_the_host_on_ties_and_holes() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xD00D);
        for n in [1usize, 2, 5, 255, 256, 257, 1000, 20_000, 130_000] {
            for k in [1usize, 4, 512, 2048] {
                let mut data = Vec::new();
                // Distinct floats (no ties), many ties, mostly holes (fewer
                // eligible rows than `k`), all equal.
                data.extend((0..n).map(|_| rng.unit()));
                data.extend(tie_scores(&mut rng, n, 0));
                data.extend(tie_scores(&mut rng, n, 60));
                data.extend(std::iter::repeat_n(0.25f32, n));
                let rows = 4;
                let t = float_tensor(data.clone(), [rows, n], &device);
                let got = ints(select_topk(t, k));
                for r in 0..rows {
                    let row = &data[r * n..(r + 1) * n];
                    let mut order: Vec<(f32, u32)> = row
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| **s > f32::NEG_INFINITY)
                        .map(|(j, &s)| (s, j as u32))
                        .collect();
                    let by =
                        |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                    order.sort_by(by);
                    order.truncate(k);
                    let mut want: Vec<u32> = order.iter().map(|o| o.1).collect();
                    want.sort_unstable();
                    assert_eq!(
                        &got[r * k..(r + 1) * k],
                        &padded(&want, k)[..],
                        "n {n}, k {k}, row {r}"
                    );
                }
            }
        }
    }

    /// The real model's indexer at a real context (32 heads of 128, top 512,
    /// 2048 candidate blocks of 8, ratio 2): scores against the host, picks
    /// and candidates against the host selection of the same scores, batches
    /// against single rows bit for bit, and timings (each `TIME` line includes
    /// the readback of the picks, the one sync a decode step pays).
    #[test]
    fn the_indexer_at_real_shapes() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        // Floats (distinct scores, tolerance) and small integers (every score
        // exact and tied with many others: the tie rule at the real scale).
        real_shapes(&device, Fill::Float, 1e-5);
        real_shapes(&device, Fill::Exact, 0.0);
    }

    /// Half a million visible compressed rows (a context of a million tokens
    /// at ratio 2): the score matrix is 2 MB per row, and the scores, the
    /// picks and the candidate blocks still equal the host's. (The pool rows
    /// hold the key only: a real 640-float row at this length would not fit
    /// one 1 GiB storage buffer.)
    #[test]
    fn the_indexer_scales_to_half_a_million_rows() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let visible = 500_000usize;
        let call = IndexCall {
            heads: 32,
            dim: 128,
            key_col: 0,
            row_width: 128,
            ratio: 2,
            top_k: 512,
            cand: Candidates::Source,
            cand_blocks: 2048,
            cand_size: 8,
            max_visible: visible,
        };
        let w = world(
            call,
            1,
            visible,
            &[(0, 2 * visible - 1), (0, 2 * 400_000 - 1)],
            Fill::Float,
            0x5EED,
        );
        let c = &w.call;
        let width = c.width();
        let d = upload(&w, &device);
        let gpu = read_f32(
            scores(
                c,
                d.iq.clone(),
                d.wts.clone(),
                &d.pool,
                &d.tables,
                &d.row_seq,
                &d.row_pos,
                None,
            ),
            "scores",
        )
        .unwrap();
        assert_scores_close(&gpu, &host_scores(&w, 0..2, None), 1e-5, "500k scores");
        let run = || {
            let out = index_topk(
                c,
                d.iq.clone(),
                d.wts.clone(),
                &d.pool,
                &d.tables,
                d.row_seq.clone(),
                d.row_pos.clone(),
                None,
            );
            (ints(out.picks), ints(out.cand.expect("a source")))
        };
        let (picks, cands) = run();
        let started = Instant::now();
        for _ in 0..3 {
            let _ = run();
        }
        println!(
            "TIME t=2 500k Source: {:.2} ms per call of 2 row(s)",
            started.elapsed().as_secs_f64() * 1e3 / 3.0
        );
        for r in 0..2 {
            let (chosen, blocks) = host_pick(&gpu[r * width..(r + 1) * width], w.vis(r), c);
            assert_eq!(
                picks[r * c.top_k..(r + 1) * c.top_k],
                padded(&chosen, c.top_k)[..],
                "picks of row {r}"
            );
            assert_eq!(
                cands[r * c.cand_blocks..(r + 1) * c.cand_blocks],
                padded(&blocks, c.cand_blocks)[..],
                "candidates of row {r}"
            );
        }
    }

    fn real_shapes(device: &Device, fill: Fill, tolerance: f32) {
        let real = |cand| IndexCall {
            heads: 32,
            dim: 128,
            key_col: 512,
            row_width: 640,
            ratio: 2,
            top_k: 512,
            cand,
            cand_blocks: 2048,
            cand_size: 8,
            max_visible: 0,
        };
        // Row 0 sees 130k compressed rows; rows 1..=33 see about 20k.
        let (long, short) = (130_000usize, 20_000usize);
        let queries: Vec<(usize, usize)> = std::iter::once((0, 2 * long - 1))
            .chain((0..33).map(|i| (0, 2 * short - 1 - 7 * i)))
            .collect();
        let mut w = world(real(Candidates::Source), 1, long, &queries, fill, 0xABCDEF);
        w.call.max_visible = long;
        let dev = upload(&w, device);
        let slice = |rows: std::ops::Range<usize>| {
            (
                dev.iq.clone().narrow(0, rows.start, rows.len()),
                dev.wts.clone().narrow(0, rows.start, rows.len()),
                dev.row_seq.clone().narrow(0, rows.start, rows.len()),
                dev.row_pos.clone().narrow(0, rows.start, rows.len()),
            )
        };
        let run = |c: &IndexCall, rows: std::ops::Range<usize>| {
            let (q, wt, rs, rp) = slice(rows);
            let out = index_topk(c, q, wt, &dev.pool, &dev.tables, rs, rp, None);
            let cand = out.cand.map(ints);
            (ints(out.picks), cand)
        };
        let timed = |label: &str, c: &IndexCall, rows: std::ops::Range<usize>| {
            let n = rows.len();
            let _ = run(c, rows.clone());
            let started = Instant::now();
            let reps = 5;
            let mut last = run(c, rows.clone());
            for _ in 1..reps {
                last = run(c, rows.clone());
            }
            println!(
                "TIME {} {label}: {:.2} ms per call of {n} row(s)",
                if matches!(fill, Fill::Float) {
                    "float"
                } else {
                    "exact"
                },
                started.elapsed().as_secs_f64() * 1e3 / reps as f64
            );
            last
        };

        for cand in [Candidates::None, Candidates::Source] {
            let mut c_long = w.call.clone();
            c_long.cand = cand;
            c_long.max_visible = long;
            let mut c_short = c_long.clone();
            c_short.max_visible = short;
            let (k, nb) = (c_long.top_k, c_long.cand_blocks);

            let (picks_130k, cand_130k) = timed(&format!("t=1 130k {cand:?}"), &c_long, 0..1);
            let (picks_20k, cand_20k) = timed(&format!("t=1 20k {cand:?}"), &c_short, 1..2);
            let (picks_33, cand_33) = timed(&format!("t=33 20k {cand:?}"), &c_short, 1..34);
            let (picks_mix, cand_mix) = timed(&format!("t=34 mixed {cand:?}"), &c_long, 0..34);

            // Where the time of the long row goes (synced stage by stage).
            if cand == Candidates::Source && matches!(fill, Fill::Float) {
                for pass in 0..3 {
                    let (q, wt, rs, rp) = slice(0..1);
                    let t0 = Instant::now();
                    let s = scores(&c_long, q, wt, &dev.pool, &dev.tables, &rs, &rp, None);
                    let _ = device.sync();
                    let t1 = Instant::now();
                    let b = block_scores(&c_long, &s, &rp);
                    let _ = device.sync();
                    let t2 = Instant::now();
                    let picks = select_topk(s, k);
                    let _ = device.sync();
                    let t3 = Instant::now();
                    let blocks = select_topk(b, nb);
                    let _ = device.sync();
                    let t4 = Instant::now();
                    drop((picks, blocks));
                    if pass == 2 {
                        println!(
                            "TIME stages t=1 130k: scores {:.2} ms, block scores {:.2} ms, select rows {:.2} ms, select blocks {:.2} ms",
                            (t1 - t0).as_secs_f64() * 1e3,
                            (t2 - t1).as_secs_f64() * 1e3,
                            (t3 - t2).as_secs_f64() * 1e3,
                            (t4 - t3).as_secs_f64() * 1e3,
                        );
                    }
                }
            }

            // The scores and the selection of the 130k row and of the 33
            // short rows: the scores of three rows against the host, every
            // pick and candidate list against the host selection of the same
            // device scores.
            for (rows, call, picks, cands) in [
                (0..1, &c_long, &picks_130k, &cand_130k),
                (1..34, &c_short, &picks_33, &cand_33),
            ] {
                let (q, wt, rs, rp) = slice(rows.clone());
                let gpu = read_f32(
                    scores(call, q, wt, &dev.pool, &dev.tables, &rs, &rp, None),
                    "scores",
                )
                .unwrap();
                let width = call.width();
                for r in rows.clone() {
                    let local = r - rows.start;
                    let (chosen, blocks) =
                        host_pick(&gpu[local * width..(local + 1) * width], w.vis(r), call);
                    assert_eq!(
                        picks[local * k..(local + 1) * k],
                        padded(&chosen, k)[..],
                        "picks of row {r}"
                    );
                    if let Some(cands) = cands {
                        assert_eq!(
                            cands[local * nb..(local + 1) * nb],
                            padded(&blocks, nb)[..],
                            "candidates of row {r}"
                        );
                    }
                }
                for r in [rows.start, rows.end - 1] {
                    let local = r - rows.start;
                    let host = host_scores(&w, r..r + 1, None);
                    let device_row = &gpu[local * width..(local + 1) * width];
                    assert_scores_close(
                        device_row,
                        &host[..width],
                        tolerance,
                        &format!("scores of row {r}"),
                    );
                }
            }

            // Row independence: a row's picks do not depend on the rows it
            // runs with, nor on the width of the score matrix.
            assert_eq!(picks_20k[..], picks_33[..k], "row 1 alone and in 33 rows");
            assert_eq!(picks_mix[..k], picks_130k[..], "row 0 alone and in 34 rows");
            assert_eq!(
                picks_mix[k..],
                picks_33[..],
                "rows 1..=33 at 130k width and at 20k width"
            );
            if let (Some(a), Some(b), Some(c), Some(d)) =
                (cand_mix.clone(), cand_33, cand_20k, cand_130k)
            {
                assert_eq!(c[..], b[..nb]);
                assert_eq!(a[..nb], d[..]);
                assert_eq!(a[nb..], b[..]);
            }
            for r in [1usize, 17, 33] {
                let (single, _) = run(&c_short, r..r + 1);
                assert_eq!(
                    single[..],
                    picks_33[(r - 1) * k..r * k],
                    "row {r} alone and in 33 rows"
                );
            }

            // A consumer layer scores only the rows of the blocks the source
            // kept (2048 blocks of 8 of the up to 16k blocks): scores against
            // the host for the first and the last row, picks against the host
            // selection of the same scores for every row.
            if let Some(cin) = cand_mix {
                let mut c = c_long.clone();
                c.cand = Candidates::Consumer;
                let width = c.width();
                let cin_t = int_tensor(cin.clone(), [34, nb], device);
                let (q, wt, rs, rp) = slice(0..34);
                let gpu = read_f32(
                    scores(&c, q, wt, &dev.pool, &dev.tables, &rs, &rp, Some(&cin_t)),
                    "scores",
                )
                .unwrap();
                for r in [0usize, 33] {
                    let host = host_scores(&w, r..r + 1, Some(&cin[r * nb..(r + 1) * nb]));
                    assert_scores_close(
                        &gpu[r * width..(r + 1) * width],
                        &host[..width],
                        tolerance,
                        &format!("consumer scores of row {r}"),
                    );
                }
                let (q, wt, rs, rp) = slice(0..34);
                let out = index_topk(&c, q, wt, &dev.pool, &dev.tables, rs, rp, Some(&cin_t));
                let picks = ints(out.picks);
                for r in 0..34 {
                    let (chosen, _) = host_pick(&gpu[r * width..(r + 1) * width], w.vis(r), &c);
                    assert_eq!(
                        picks[r * k..(r + 1) * k],
                        padded(&chosen, k)[..],
                        "consumer picks of row {r}"
                    );
                }
                let started = Instant::now();
                for _ in 0..5 {
                    let (q, wt, rs, rp) = slice(0..1);
                    let out = index_topk(
                        &c,
                        q,
                        wt,
                        &dev.pool,
                        &dev.tables,
                        rs,
                        rp,
                        Some(&cin_t.clone().narrow(0, 0, 1)),
                    );
                    let _ = ints(out.picks);
                }
                println!(
                    "TIME {} t=1 130k Consumer: {:.2} ms per call of 1 row(s)",
                    if matches!(fill, Fill::Float) {
                        "float"
                    } else {
                        "exact"
                    },
                    started.elapsed().as_secs_f64() * 1e3 / 5.0
                );
            }
        }
    }
}
