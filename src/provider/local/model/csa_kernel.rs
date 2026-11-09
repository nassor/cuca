//! Device kernels of the latent attention with compressed sparse attention
//! (DeepSeek-V4.1) of the GPU engine (CubeCL). The reference is the CPU
//! engine's `csa_attention`, `compress` and `rotate` (`cpu/hyper.rs`).
//!
//! Every pool is a 2-D f32 tensor whose ROWS are addressed through block
//! tables exactly like the CPU pools: row (position) `i` of a sequence lives
//! at pool row `table[i / 16] * 16 + i % 16` ([`BLOCK_SIZE`]). A window pool
//! row is `head_dim` wide (key == value, one vector per position); a
//! compressed pool row is `row_width` wide (`[latent | index key]`), of which
//! only the first `head_dim` floats are the attention key and value. A table
//! entry that holds no block is `-1` (`u32::MAX`): it is never read. The
//! tables of a pass are `[n_seq, max_blocks]` (shorter rows padded with -1)
//! and every query row carries its sequence (`row_seq`) and position
//! (`row_pos`).
//!
//! The kernels:
//!
//! 1. [`rope_tail`]: rotate the last `rot` values of every `width`-wide vector
//!    of a row by the angles of the row's position (or the inverse rotation);
//! 2. [`scatter_rows`]: write rows into pool rows IN PLACE (the pool is taken
//!    by value; the host works the destination rows out of the block tables);
//! 3. [`pair_pool`]: the ratio-2 compressor's softmax pooling of row pairs;
//! 4. [`csa_attention`]: per (row, head) the scores `q . key * scale` over
//!    the window positions `max(0, p + 1 - window) ..= p` (ascending) and then
//!    over the compressed rows the row's `picks` list (ascending ids, `-1` =
//!    unused slot), a softmax with the head's attention sink (`m = max(scores)`
//!    clamped to `>= -1e30`, `denom = sum(exp(s - m)) + exp(sink - m)`) and the
//!    weighted sum of the keys over `denom`. The output is neither rotated back
//!    nor quantized.
//!
//! # Attention kernel
//!
//! A cube is 64 lanes. The key slots of a row are the window slots `0 ..
//! window` followed, from the next multiple of 64, by the pick slots `0 ..
//! K`; a chunk is 64 slots of one kind (so a chunk reads one pool), the
//! window's `ceil(window / 64)` chunks first. The chunks depend on the call's
//! constants alone, never on the number of rows. A cube serves `G` heads of
//! one row (`G = 4`; the heads share every key read) and one chunk:
//!
//! - phase 1: lane `b` resolves key slot `b` of the chunk (the block table
//!   lookup, an invalid slot is skipped) and computes its `G` scores against
//!   the group's queries, kept in shared memory;
//! - phase 2: every lane computes the chunk maxima, the weights `exp(s - m)`
//!   go to shared memory and lane `l` accumulates `G x ceil(head_dim / 64)`
//!   outputs (dimensions `l + 64 j`) over the chunk's keys: the key reads are
//!   coalesced.
//!
//! A chunk's partial result is `(max, sum, accumulators)`. The partials of a
//! row's chunks are folded in chunk order (an empty chunk is skipped) by the
//! one merge expression both launch shapes use, so a row's result never
//! depends on how many rows share the launch or on the shape:
//!
//! - few rows (decode): one cube per (chunk, group, row) writes its partial,
//!   a second kernel folds the partials and applies the sink;
//! - many rows (prefill): one cube per (group, row) loops over its chunks and
//!   folds in registers.
//!
//! The kernels are written in safe Rust (`#[cube(launch)]` with the checked
//! `launch`) and reach the backend through burn's `backend_extension`, as the
//! [`moe_kernel`](super::moe_kernel) and [`paged_kernel`](super::paged_kernel)
//! ones do. The caller checks the block tables on the host: a table entry
//! that holds no block under a position the row reads is skipped silently
//! here (the CPU engine reports an error).

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values (`scale - scale` is a runtime zero where a
// literal would be evaluated by rustc), plain `x = x * c` and nested `if`s.
// The operations are reached by the engine integration (`gpu.rs`) and by the
// tests of this module; until the engine calls them the lib build sees them
// unused.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::too_many_arguments
)]

use burn::backend::{TensorMetadata, backend_extension, tensor::FloatTensor, tensor::IntTensor};
use burn::cubecl::prelude::*;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Shape};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};

use super::paged::BLOCK_SIZE;

// The kernels spell the block size as a literal.
const _: () = assert!(BLOCK_SIZE == 16);

/// Keys per chunk of the attention kernel and lanes per cube (the kernels
/// spell it as a literal).
const LANES: usize = 64;

/// Query heads one attention cube serves.
const GROUP: usize = 4;

/// Bytes of partial results the split attention launch may allocate (one
/// (row, head, chunk) partial is `head_dim + 2` floats): above it the fused
/// launch runs. It stays well under wgpu's limit of 1 GiB per storage buffer.
const SPLIT_SCRATCH_BYTES: usize = 512 << 20;

/// Elements of a key the score loop loads together before using them.
const DOT_UNROLL: usize = 16;

/// Keys the weighted-sum loop loads together before using them (a divisor of
/// the 64 slots of a chunk).
const KEY_UNROLL: usize = 8;

// The kernels read a table entry or pick without a block (-1) as 0xFFFF_FFFF.

// ---------------------------------------------------------------- kernels

/// `out` = `x` with the last `rot` values of every `width`-wide vector
/// rotated pairwise by the angles of the position of the vector's row:
/// `(a, b) -> (a cos - b sin, a sin + b cos)` with `sign` multiplying `sin`
/// (`-1` for the inverse rotation). One thread per element; `row_len` is the
/// floats of a row (`n_vec * width`), the tables `[positions, rot / 2]`.
#[cube(launch)]
fn rope_tail_kernel(
    x: &Tensor<f32>,
    cos: &Tensor<f32>,
    sin: &Tensor<f32>,
    pos: &Tensor<u32>,
    out: &mut Tensor<f32>,
    total: u32,
    row_len: u32,
    width: u32,
    rot: u32,
    sign: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(total) {
        let width = usize::cast_from(width);
        let rot = usize::cast_from(rot);
        let c = i % width;
        let keep = width - rot;
        if c < keep {
            out[i] = x[i];
        } else {
            let k = c - keep;
            let row = i / usize::cast_from(row_len);
            let first = i - k % 2usize;
            let a = x[first];
            let b = x[first + 1usize];
            let at = usize::cast_from(pos[row]) * (rot / 2usize) + k / 2usize;
            let cs = cos[at];
            let sn = sin[at] * sign;
            if k % 2usize == 0usize {
                out[i] = a * cs - b * sn;
            } else {
                out[i] = a * sn + b * cs;
            }
        }
    }
}

/// `pool[dest[r], :] = rows[r, :]` for every row with `dest[r]` a row of the
/// pool (`-1` and anything past the pool are skipped): one thread per
/// element of `rows`.
#[cube(launch)]
fn scatter_rows_kernel(
    pool: &mut Tensor<f32>,
    rows: &Tensor<f32>,
    dest: &Tensor<u32>,
    total: u32,
    width: u32,
    pool_rows: u32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(total) {
        let width = usize::cast_from(width);
        let d = dest[i / width];
        if d < pool_rows {
            pool[usize::cast_from(d) * width + i % width] = rows[i];
        }
    }
}

/// The ratio-2 compressor's pooling: `out[r, c]` mixes row `left[r]` and
/// row `right[r]` of `kv` by the softmax of their `score` channels:
/// `m = max(s0, s1)`, `e_i = exp(s_i - m)`, `kv0 e0 / (e0 + e1) + kv1 e1 /
/// (e0 + e1)`. One thread per output element.
#[cube(launch)]
fn pair_pool_kernel(
    kv: &Tensor<f32>,
    score: &Tensor<f32>,
    left: &Tensor<u32>,
    right: &Tensor<u32>,
    out: &mut Tensor<f32>,
    total: u32,
    width: u32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(total) {
        let width = usize::cast_from(width);
        let r = i / width;
        let c = i % width;
        let a = usize::cast_from(left[r]) * width + c;
        let b = usize::cast_from(right[r]) * width + c;
        let s0 = score[a];
        let s1 = score[b];
        let mut m = s0;
        if s1 > s0 {
            m = s1;
        }
        let e0 = (s0 - m).exp();
        let e1 = (s1 - m).exp();
        let total = e0 + e1;
        out[i] = kv[a] * (e0 / total) + kv[b] * (e1 / total);
    }
}

/// `dots[i] += q[i] . key` for the `g` query rows in shared memory (`hd`
/// floats each) and the key at element `at` of `pool`: `DOT_UNROLL` loads of
/// the key are issued together, then used.
#[cube]
fn key_dots(
    pool: &Tensor<f32>,
    at: usize,
    q: &Shared<[f32]>,
    dots: &mut Array<f32>,
    #[comptime] g: usize,
    #[comptime] hd: usize,
) {
    let blocks = comptime![hd / DOT_UNROLL];
    let tail = comptime![hd / DOT_UNROLL * DOT_UNROLL];
    for block in 0..blocks {
        let d0 = block * DOT_UNROLL;
        let mut ks = Array::<f32>::new(DOT_UNROLL);
        #[unroll]
        for u in 0..DOT_UNROLL {
            ks[u] = pool[at + d0 + u];
        }
        #[unroll]
        for u in 0..DOT_UNROLL {
            #[unroll]
            for i in 0..g {
                dots[i] += q[i * hd + d0 + u] * ks[u];
            }
        }
    }
    for d in tail..hd {
        let k = pool[at + d];
        #[unroll]
        for i in 0..g {
            dots[i] += q[i * hd + d] * k;
        }
    }
}

/// The lane's dimensions of the weighted sum of the 64 keys of a chunk, in
/// slot order: `acc[i * per + j] += sum_b pw[i * 64 + b] * key_b[lane + 64
/// j]` over the slots with a key (`kind[b] != 0`; the others read row 0 and
/// contribute nothing), `KEY_UNROLL` keys at a time. All the loads of a step
/// are issued before the first is used: a load whose value is consumed (or
/// selected) right away, or that sits behind a branch, stalls the wave for
/// its latency.
#[cube]
fn weighted_keys(
    pool: &Tensor<f32>,
    kind: &mut Shared<[u32]>,
    base: &mut Shared<[u32]>,
    pw: &mut Shared<[f32]>,
    acc: &mut Array<f32>,
    lane: usize,
    zero: f32,
    #[comptime] g: usize,
    #[comptime] per: usize,
    #[comptime] hd: usize,
) {
    let mut w = Array::<f32>::new(g * KEY_UNROLL);
    let mut at = Array::<usize>::new(KEY_UNROLL);
    let mut ok = Array::<u32>::new(KEY_UNROLL);
    let mut v = Array::<f32>::new(per * KEY_UNROLL);
    for b0 in 0..(64usize / KEY_UNROLL) {
        #[unroll]
        for u in 0..KEY_UNROLL {
            let b = b0 * KEY_UNROLL + u;
            at[u] = usize::cast_from(base[b]);
            ok[u] = kind[b];
            #[unroll]
            for i in 0..g {
                w[u * g + i] = pw[i * 64usize + b];
            }
        }
        #[unroll]
        for u in 0..KEY_UNROLL {
            #[unroll]
            for j in 0..per {
                let d = lane + j * 64usize;
                // A dimension past the head (heads that are not a multiple of
                // 64 wide) reads the last one and is never used.
                let e = select(d < hd, d, lane - lane + hd - 1usize);
                v[u * per + j] = pool[at[u] + e];
            }
        }
        #[unroll]
        for u in 0..KEY_UNROLL {
            #[unroll]
            for j in 0..per {
                let y = select(ok[u] != 0u32, v[u * per + j], zero);
                #[unroll]
                for i in 0..g {
                    acc[i * per + j] += w[u * g + i] * y;
                }
            }
        }
    }
}

/// The partial softmax of one chunk of 64 key slots of a row for the `g`
/// heads whose queries are in `q` (see the module docs). Chunks `0 ..
/// ceil(window / 64)` hold window slots, the others pick slots. Returns 0
/// when no slot of the chunk holds a key (nothing written but `stats`' empty
/// state), else 1 with `stats[i]` the chunk maximum, `stats[g + i]` the sum
/// of the weights `exp(s - max)` and `acc[i * per + j]` the weighted keys'
/// dimension `lane + 64 j` of head `i`. Every barrier is reached by all
/// lanes.
#[cube]
fn chunk_pass(
    win: &Tensor<f32>,
    win_tables: &Tensor<u32>,
    cmp: &Tensor<f32>,
    cmp_tables: &Tensor<u32>,
    picks: &Tensor<u32>,
    q: &Shared<[f32]>,
    sc: &mut Shared<[f32]>,
    pw: &mut Shared<[f32]>,
    kind: &mut Shared<[u32]>,
    base: &mut Shared<[u32]>,
    acc: &mut Array<f32>,
    stats: &mut Array<f32>,
    lane: usize,
    row: usize,
    seq: usize,
    p: usize,
    chunk: usize,
    scale: f32,
    window: usize,
    n_picks: usize,
    win_blocks: usize,
    cmp_blocks: usize,
    cmp_width: usize,
    #[comptime] g: usize,
    #[comptime] per: usize,
    #[comptime] hd: usize,
) -> u32 {
    let zero = scale - scale;
    let win_chunks = (window + 63usize) / 64usize;
    let in_window = chunk < win_chunks;

    // Phase 1: lane `lane` resolves its slot of the chunk (key `kd` = 1, the
    // pool element `at` of its first float) and scores it.
    let mut at = lane - lane;
    let mut kd = u32::cast_from(lane - lane);
    if in_window {
        let slot = chunk * 64usize + lane;
        if slot < window {
            let mut lo = lane - lane;
            if p + 1usize > window {
                lo = p + 1usize - window;
            }
            let pos = lo + slot;
            if pos <= p {
                let bi = pos / 16usize;
                if bi < win_blocks {
                    let blk = win_tables[seq * win_blocks + bi];
                    if blk != 0xFFFF_FFFFu32 {
                        at = (usize::cast_from(blk) * 16usize + pos % 16usize) * hd;
                        kd = 1u32;
                    }
                }
            }
        }
    } else {
        let k = (chunk - win_chunks) * 64usize + lane;
        if k < n_picks {
            let j = picks[row * n_picks + k];
            if j != 0xFFFF_FFFFu32 {
                let j = usize::cast_from(j);
                let bi = j / 16usize;
                if bi < cmp_blocks {
                    let blk = cmp_tables[seq * cmp_blocks + bi];
                    if blk != 0xFFFF_FFFFu32 {
                        at = (usize::cast_from(blk) * 16usize + j % 16usize) * cmp_width;
                        kd = 1u32;
                    }
                }
            }
        }
    }
    let mut dots = Array::<f32>::new(g);
    #[unroll]
    for i in 0..g {
        dots[i] = zero;
    }
    if kd != 0u32 {
        if in_window {
            key_dots(win, at, q, &mut dots, g, hd);
        } else {
            key_dots(cmp, at, q, &mut dots, g, hd);
        }
    }
    #[unroll]
    for i in 0..g {
        let mut s = zero - 1.0e30f32;
        if kd != 0u32 {
            s = dots[i] * scale;
        }
        sc[i * 64usize + lane] = s;
    }
    kind[lane] = kd;
    base[lane] = u32::cast_from(at);
    sync_cube();

    // The chunk's maxima (every lane the same ones) and the weights. A slot
    // without a key scores -1e30, below every score, and has weight 0.
    let mut any = u32::cast_from(lane - lane);
    #[unroll]
    for i in 0..g {
        stats[i] = zero - 1.0e30f32;
        stats[g + i] = zero;
    }
    for b in 0..64usize {
        let kb = kind[b];
        if kb > any {
            any = kb;
        }
        #[unroll]
        for i in 0..g {
            let s = sc[i * 64usize + b];
            if s > stats[i] {
                stats[i] = s;
            }
        }
    }
    #[unroll]
    for i in 0..g {
        let mut w = zero;
        if kd != 0u32 {
            w = (sc[i * 64usize + lane] - stats[i]).exp();
        }
        pw[i * 64usize + lane] = w;
    }
    sync_cube();

    // Phase 2: the lane's dimensions of the weighted keys, in slot order.
    #[unroll]
    for i in 0..g {
        #[unroll]
        for j in 0..per {
            acc[i * per + j] = zero;
        }
    }
    if any != 0u32 {
        if in_window {
            weighted_keys(win, kind, base, pw, acc, lane, zero, g, per, hd);
        } else {
            weighted_keys(cmp, kind, base, pw, acc, lane, zero, g, per, hd);
        }
        for b in 0..64usize {
            #[unroll]
            for i in 0..g {
                stats[g + i] += pw[i * 64usize + b];
            }
        }
    }
    // Nobody may overwrite the shared slots of this chunk before all are done.
    sync_cube();
    any
}

/// Fold a non-empty chunk (`stats`, `acc` of [`chunk_pass`]) into the running
/// softmax `run` (`run[i]` maximum, `run[g + i]` sum) and weighted sum
/// `acc_run`. The one expression of the merge: the fused kernel and the merge
/// kernel use it in chunk order.
#[cube]
fn merge_chunk(
    run: &mut Array<f32>,
    acc_run: &mut Array<f32>,
    stats: &Array<f32>,
    acc: &Array<f32>,
    #[comptime] g: usize,
    #[comptime] per: usize,
) {
    #[unroll]
    for i in 0..g {
        let m0 = run[i];
        let mc = stats[i];
        let mut m1 = m0;
        if mc > m0 {
            m1 = mc;
        }
        let a = (m0 - m1).exp();
        let c = (mc - m1).exp();
        run[g + i] = run[g + i] * a + stats[g + i] * c;
        run[i] = m1;
        #[unroll]
        for j in 0..per {
            acc_run[i * per + j] = acc_run[i * per + j] * a + acc[i * per + j] * c;
        }
    }
}

/// The group's queries into shared memory: `q[row, (head0 + i) * hd ..]` for
/// the heads that exist, zeros for the others. The loads are issued together
/// (a head or dimension that does not exist reads element 0 of the row's
/// first query, unused). The caller syncs.
#[cube]
fn load_queries(
    q: &Tensor<f32>,
    qs: &mut Shared<[f32]>,
    lane: usize,
    row: usize,
    head0: usize,
    n_head: usize,
    zero: f32,
    #[comptime] g: usize,
    #[comptime] per: usize,
    #[comptime] hd: usize,
) {
    let mut raw = Array::<f32>::new(g * per);
    #[unroll]
    for i in 0..g {
        let head = head0 + i;
        let start = select(head < n_head, (row * n_head + head) * hd, row * n_head * hd);
        #[unroll]
        for j in 0..per {
            let d = lane + j * 64usize;
            let e = select(d < hd, d, lane - lane);
            raw[i * per + j] = q[start + e];
        }
    }
    #[unroll]
    for i in 0..g {
        #[unroll]
        for j in 0..per {
            let d = lane + j * 64usize;
            if d < hd {
                qs[i * hd + d] = select(head0 + i < n_head, raw[i * per + j], zero);
            }
        }
    }
}

/// Fused attention (many rows): cube `(group, row)` folds all the chunks of
/// its row in registers and writes `out[row, head, :]` for its heads.
#[cube(launch)]
fn csa_fused_kernel(
    q: &Tensor<f32>,
    win: &Tensor<f32>,
    win_tables: &Tensor<u32>,
    cmp: &Tensor<f32>,
    cmp_tables: &Tensor<u32>,
    picks: &Tensor<u32>,
    row_seq: &Tensor<u32>,
    row_pos: &Tensor<u32>,
    sinks: &Tensor<f32>,
    out: &mut Tensor<f32>,
    scale: f32,
    n_head: u32,
    window: u32,
    n_picks: u32,
    n_chunks: u32,
    win_blocks: u32,
    cmp_blocks: u32,
    cmp_width: u32,
    #[comptime] g: usize,
    #[comptime] per: usize,
    #[comptime] hd: usize,
) {
    let lane = UNIT_POS_X as usize;
    let row = CUBE_POS_Y as usize;
    let head0 = CUBE_POS_X as usize * g;
    let seq = usize::cast_from(row_seq[row]);
    let p = usize::cast_from(row_pos[row]);
    let n_head = usize::cast_from(n_head);
    let zero = scale - scale;

    let qn = comptime![g * hd];
    let sn = comptime![g * 64usize];
    let mut qs = Shared::<[f32]>::new_slice(qn);
    let mut sc = Shared::<[f32]>::new_slice(sn);
    let mut pw = Shared::<[f32]>::new_slice(sn);
    let mut kind = Shared::<[u32]>::new_slice(64usize);
    let mut base = Shared::<[u32]>::new_slice(64usize);
    load_queries(q, &mut qs, lane, row, head0, n_head, zero, g, per, hd);
    sync_cube();

    let mut run = Array::<f32>::new(g * 2usize);
    let mut acc_run = Array::<f32>::new(g * per);
    let mut acc = Array::<f32>::new(g * per);
    let mut stats = Array::<f32>::new(g * 2usize);
    #[unroll]
    for i in 0..g {
        run[i] = zero - 1.0e30f32;
        run[g + i] = zero;
        #[unroll]
        for j in 0..per {
            acc_run[i * per + j] = zero;
        }
    }
    for chunk in 0..usize::cast_from(n_chunks) {
        let any = chunk_pass(
            win,
            win_tables,
            cmp,
            cmp_tables,
            picks,
            &qs,
            &mut sc,
            &mut pw,
            &mut kind,
            &mut base,
            &mut acc,
            &mut stats,
            lane,
            row,
            seq,
            p,
            chunk,
            scale,
            usize::cast_from(window),
            usize::cast_from(n_picks),
            usize::cast_from(win_blocks),
            usize::cast_from(cmp_blocks),
            usize::cast_from(cmp_width),
            g,
            per,
            hd,
        );
        if any != 0u32 {
            merge_chunk(&mut run, &mut acc_run, &stats, &acc, g, per);
        }
    }

    #[unroll]
    for i in 0..g {
        let head = head0 + i;
        if head < n_head {
            let denom = run[g + i] + (sinks[head] - run[i]).exp();
            #[unroll]
            for j in 0..per {
                let d = lane + j * 64usize;
                if d < hd {
                    out[(row * n_head + head) * hd + d] = acc_run[i * per + j] / denom;
                }
            }
        }
    }
}

/// First pass of the split attention: cube `(group, row, chunk)` writes the
/// partial result of its chunk for its heads: `part[((row * n_head + head) *
/// n_chunks + chunk) * (hd + 2)]` holds the weighted keys (`hd`), then the
/// maximum and the sum (an empty chunk: `-1e30` and 0).
#[cube(launch)]
fn csa_split_kernel(
    q: &Tensor<f32>,
    win: &Tensor<f32>,
    win_tables: &Tensor<u32>,
    cmp: &Tensor<f32>,
    cmp_tables: &Tensor<u32>,
    picks: &Tensor<u32>,
    row_seq: &Tensor<u32>,
    row_pos: &Tensor<u32>,
    part: &mut Tensor<f32>,
    scale: f32,
    n_head: u32,
    window: u32,
    n_picks: u32,
    n_chunks: u32,
    win_blocks: u32,
    cmp_blocks: u32,
    cmp_width: u32,
    #[comptime] g: usize,
    #[comptime] per: usize,
    #[comptime] hd: usize,
) {
    let lane = UNIT_POS_X as usize;
    let row = CUBE_POS_Y as usize;
    let chunk = CUBE_POS_Z as usize;
    let head0 = CUBE_POS_X as usize * g;
    let seq = usize::cast_from(row_seq[row]);
    let p = usize::cast_from(row_pos[row]);
    let n_head = usize::cast_from(n_head);
    let zero = scale - scale;

    let qn = comptime![g * hd];
    let sn = comptime![g * 64usize];
    let mut qs = Shared::<[f32]>::new_slice(qn);
    let mut sc = Shared::<[f32]>::new_slice(sn);
    let mut pw = Shared::<[f32]>::new_slice(sn);
    let mut kind = Shared::<[u32]>::new_slice(64usize);
    let mut base = Shared::<[u32]>::new_slice(64usize);
    load_queries(q, &mut qs, lane, row, head0, n_head, zero, g, per, hd);
    sync_cube();

    let mut acc = Array::<f32>::new(g * per);
    let mut stats = Array::<f32>::new(g * 2usize);
    let _any = chunk_pass(
        win,
        win_tables,
        cmp,
        cmp_tables,
        picks,
        &qs,
        &mut sc,
        &mut pw,
        &mut kind,
        &mut base,
        &mut acc,
        &mut stats,
        lane,
        row,
        seq,
        p,
        chunk,
        scale,
        usize::cast_from(window),
        usize::cast_from(n_picks),
        usize::cast_from(win_blocks),
        usize::cast_from(cmp_blocks),
        usize::cast_from(cmp_width),
        g,
        per,
        hd,
    );

    #[unroll]
    for i in 0..g {
        let head = head0 + i;
        if head < n_head {
            let at = ((row * n_head + head) * usize::cast_from(n_chunks) + chunk) * (hd + 2usize);
            #[unroll]
            for j in 0..per {
                let d = lane + j * 64usize;
                if d < hd {
                    part[at + d] = acc[i * per + j];
                }
            }
            if lane == 0usize {
                part[at + hd] = stats[i];
                part[at + hd + 1usize] = stats[g + i];
            }
        }
    }
}

/// Second pass: one thread per (row, head, dimension) folds the `n_chunks`
/// partials of its (row, head) in chunk order (an empty chunk, sum 0, is
/// skipped) with the expression of [`merge_chunk`] and applies the sink.
#[cube(launch)]
fn csa_merge_kernel(
    part: &Tensor<f32>,
    sinks: &Tensor<f32>,
    out: &mut Tensor<f32>,
    total: u32,
    n_head: u32,
    n_chunks: u32,
    #[comptime] hd: usize,
) {
    let job = ABSOLUTE_POS;
    if job < usize::cast_from(total) {
        let rh = job / hd;
        let d = job % hd;
        let head = rh % usize::cast_from(n_head);
        let zero = f32::cast_from(job - job);
        let mut m = zero - 1.0e30f32;
        let mut l = zero;
        let mut a = zero;
        for chunk in 0..usize::cast_from(n_chunks) {
            let at = (rh * usize::cast_from(n_chunks) + chunk) * (hd + 2usize);
            let lc = part[at + hd + 1usize];
            if lc > zero {
                let mc = part[at + hd];
                let mut m1 = m;
                if mc > m {
                    m1 = mc;
                }
                let x = (m - m1).exp();
                let c = (mc - m1).exp();
                l = l * x + lc * c;
                m = m1;
                a = a * x + part[at + d] * c;
            }
        }
        out[job] = a / (l + (sinks[head] - m).exp());
    }
}

// --------------------------------------------------------------- extension

/// The scalars of a [`rope_tail`] launch.
#[derive(Clone, Debug)]
pub(crate) struct RopeLaunch {
    /// Floats of a vector.
    width: usize,
    /// Rotated trailing floats of a vector.
    rot: usize,
    inverse: bool,
}

/// The scalars of a [`pair_pool`] launch.
#[derive(Clone, Debug)]
pub(crate) struct PairLaunch {
    /// Pairs pooled.
    rows: usize,
    /// Channels of a row.
    width: usize,
}

/// The scalars and shapes of an attention launch.
#[derive(Clone, Debug)]
pub(crate) struct CsaLaunch {
    scale: f32,
    n_head: usize,
    head_dim: usize,
    window: usize,
    /// Pick slots per row (0 without compressed rows).
    n_picks: usize,
    /// Query heads one cube serves.
    group: usize,
    /// Split launch (a partial per chunk, then a merge kernel) or fused.
    split: bool,
}

/// Backend operations of the compressed-sparse-attention layers (see the
/// module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait CsaKernels: burn::backend::Backend {
    /// `x [t, n_vec * width]` with the last `rot` values of every vector
    /// rotated by the angles of `pos [t]` in the tables `[positions, rot / 2]`.
    #[fusion(dtype = x, shape = x)]
    fn csa_rope_tail(
        x: FloatTensor<Self>,
        cos: FloatTensor<Self>,
        sin: FloatTensor<Self>,
        pos: IntTensor<Self>,
        call: RopeLaunch,
    ) -> FloatTensor<Self>;

    /// `pool [n, w]` with `rows [m, w]` written at the pool rows `dest [m]`
    /// (`-1`: skipped), in place when nothing else holds the pool.
    #[fusion(dtype = pool, shape = pool)]
    fn csa_scatter_rows(
        pool: FloatTensor<Self>,
        rows: FloatTensor<Self>,
        dest: IntTensor<Self>,
    ) -> FloatTensor<Self>;

    /// The softmax pooling of the row pairs `(left[r], right[r])` of `kv` by
    /// `score` (both `[n, width]`): `[rows, width]`.
    #[fusion(dtype = kv, shape = Shape::new([call.rows, call.width]))]
    fn csa_pair_pool(
        kv: FloatTensor<Self>,
        score: FloatTensor<Self>,
        left: IntTensor<Self>,
        right: IntTensor<Self>,
        call: PairLaunch,
    ) -> FloatTensor<Self>;

    /// The attention of the query rows `q [t, n_head * hd]` (see the module
    /// docs): `[t, n_head * hd]`.
    #[fusion(dtype = q, shape = q)]
    fn csa_attention(
        q: FloatTensor<Self>,
        win: FloatTensor<Self>,
        win_tables: IntTensor<Self>,
        cmp: FloatTensor<Self>,
        cmp_tables: IntTensor<Self>,
        picks: IntTensor<Self>,
        row_seq: IntTensor<Self>,
        row_pos: IntTensor<Self>,
        sinks: FloatTensor<Self>,
        call: CsaLaunch,
    ) -> FloatTensor<Self>;
}

impl CsaKernels for CubeBackend {
    fn csa_rope_tail(
        x: CubeTensor,
        cos: CubeTensor,
        sin: CubeTensor,
        pos: CubeTensor,
        call: RopeLaunch,
    ) -> CubeTensor {
        // The kernels index the buffers linearly.
        let x = into_contiguous(x);
        let cos = into_contiguous(cos);
        let sin = into_contiguous(sin);
        let pos = into_contiguous(pos);
        let [rows, row_len] = x.shape().dims::<2>();
        let total = rows * row_len;
        let client = x.client.clone();
        let out = empty_device_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new([rows, row_len]),
            DType::F32,
        );
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        rope_tail_kernel::launch(
            &client,
            count,
            dim,
            x.into_tensor_arg(),
            cos.into_tensor_arg(),
            sin.into_tensor_arg(),
            pos.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            row_len as u32,
            call.width as u32,
            call.rot as u32,
            if call.inverse { -1.0f32 } else { 1.0f32 },
        );
        out
    }

    fn csa_scatter_rows(pool: CubeTensor, rows: CubeTensor, dest: CubeTensor) -> CubeTensor {
        // In place when nothing else holds the pool, else into a copy.
        let pool = if pool.can_mut() { pool } else { pool.copy() };
        let rows = into_contiguous(rows);
        let dest = into_contiguous(dest);
        let [pool_rows, _] = pool.shape().dims::<2>();
        let [m, width] = rows.shape().dims::<2>();
        let total = m * width;
        let client = pool.client.clone();
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        scatter_rows_kernel::launch(
            &client,
            count,
            dim,
            pool.clone().into_tensor_arg(),
            rows.into_tensor_arg(),
            dest.into_tensor_arg(),
            total as u32,
            width as u32,
            pool_rows as u32,
        );
        pool
    }

    fn csa_pair_pool(
        kv: CubeTensor,
        score: CubeTensor,
        left: CubeTensor,
        right: CubeTensor,
        call: PairLaunch,
    ) -> CubeTensor {
        let kv = into_contiguous(kv);
        let score = into_contiguous(score);
        let left = into_contiguous(left);
        let right = into_contiguous(right);
        let total = call.rows * call.width;
        let client = kv.client.clone();
        let out = empty_device_dtype(
            client.clone(),
            kv.device.clone(),
            Shape::new([call.rows, call.width]),
            DType::F32,
        );
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        pair_pool_kernel::launch(
            &client,
            count,
            dim,
            kv.into_tensor_arg(),
            score.into_tensor_arg(),
            left.into_tensor_arg(),
            right.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            call.width as u32,
        );
        out
    }

    fn csa_attention(
        q: CubeTensor,
        win: CubeTensor,
        win_tables: CubeTensor,
        cmp: CubeTensor,
        cmp_tables: CubeTensor,
        picks: CubeTensor,
        row_seq: CubeTensor,
        row_pos: CubeTensor,
        sinks: CubeTensor,
        call: CsaLaunch,
    ) -> CubeTensor {
        let q = into_contiguous(q);
        let win = into_contiguous(win);
        let win_tables = into_contiguous(win_tables);
        let cmp = into_contiguous(cmp);
        let cmp_tables = into_contiguous(cmp_tables);
        let picks = into_contiguous(picks);
        let row_seq = into_contiguous(row_seq);
        let row_pos = into_contiguous(row_pos);
        let sinks = into_contiguous(sinks);
        let [rows, _] = q.shape().dims::<2>();
        let (n_head, hd, g) = (call.n_head, call.head_dim, call.group);
        let per = hd.div_ceil(LANES);
        let groups = n_head.div_ceil(g);
        let n_chunks = call.window.div_ceil(LANES) + call.n_picks.div_ceil(LANES);
        let win_blocks = win_tables.shape().dims::<2>()[1];
        let cmp_blocks = cmp_tables.shape().dims::<2>()[1];
        let cmp_width = cmp.shape().dims::<2>()[1];
        let client = q.client.clone();
        let out = empty_device_dtype(
            client.clone(),
            q.device.clone(),
            Shape::new([rows, n_head * hd]),
            DType::F32,
        );
        let dim = CubeDim::new_1d(LANES as u32);
        if call.split {
            let part = empty_device_dtype(
                client.clone(),
                q.device.clone(),
                Shape::new([rows * n_head * n_chunks * (hd + 2)]),
                DType::F32,
            );
            csa_split_kernel::launch(
                &client,
                CubeCount::Static(groups as u32, rows as u32, n_chunks as u32),
                dim,
                q.into_tensor_arg(),
                win.into_tensor_arg(),
                win_tables.into_tensor_arg(),
                cmp.into_tensor_arg(),
                cmp_tables.into_tensor_arg(),
                picks.into_tensor_arg(),
                row_seq.into_tensor_arg(),
                row_pos.into_tensor_arg(),
                part.clone().into_tensor_arg(),
                call.scale,
                n_head as u32,
                call.window as u32,
                call.n_picks as u32,
                n_chunks as u32,
                win_blocks as u32,
                cmp_blocks as u32,
                cmp_width as u32,
                g,
                per,
                hd,
            );
            let total = rows * n_head * hd;
            let merge_dim = CubeDim::new(&client, total);
            let count = calculate_cube_count_elemwise(&client, total, merge_dim);
            csa_merge_kernel::launch(
                &client,
                count,
                merge_dim,
                part.into_tensor_arg(),
                sinks.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                total as u32,
                n_head as u32,
                n_chunks as u32,
                hd,
            );
        } else {
            csa_fused_kernel::launch(
                &client,
                CubeCount::Static(groups as u32, rows as u32, 1),
                dim,
                q.into_tensor_arg(),
                win.into_tensor_arg(),
                win_tables.into_tensor_arg(),
                cmp.into_tensor_arg(),
                cmp_tables.into_tensor_arg(),
                picks.into_tensor_arg(),
                row_seq.into_tensor_arg(),
                row_pos.into_tensor_arg(),
                sinks.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                call.scale,
                n_head as u32,
                call.window as u32,
                call.n_picks as u32,
                n_chunks as u32,
                win_blocks as u32,
                cmp_blocks as u32,
                cmp_width as u32,
                g,
                per,
                hd,
            );
        }
        out
    }
}

// ---------------------------------------------------------- tensor wrappers

use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor as BTensor};

/// [`CsaKernels::csa_rope_tail`] on tensors: `x` is `[t, n_vec * width]`,
/// `cos` and `sin` the layer's tables `[positions, rot / 2]`, `pos` `[t]` the
/// position of every row (the compressed rows' `j * ratio`). The last `rot`
/// values of every `width`-wide vector are rotated pairwise `(a, b) -> (a c -
/// b s, a s + b c)`; `inverse` negates `s`.
pub(crate) fn rope_tail(
    x: BTensor<2>,
    cos: &BTensor<2>,
    sin: &BTensor<2>,
    pos: BTensor<1, Int>,
    width: usize,
    rot: usize,
    inverse: bool,
) -> BTensor<2> {
    let [rows, row_len] = x.dims();
    assert!(
        rows > 0 && width > 0 && row_len.is_multiple_of(width),
        "rope_tail: bad shape"
    );
    assert!(
        rot.is_multiple_of(2) && rot <= width,
        "rope_tail: bad rotary width"
    );
    assert_eq!(cos.dims()[1], rot / 2, "rope_tail: table width");
    assert_eq!(pos.dims()[0], rows, "rope_tail: one position per row");
    BTensor::from_dispatch(<Dispatch as CsaKernels>::csa_rope_tail(
        x.into_dispatch(),
        cos.clone().into_dispatch(),
        sin.clone().into_dispatch(),
        pos.into_dispatch(),
        RopeLaunch {
            width,
            rot,
            inverse,
        },
    ))
}

/// [`CsaKernels::csa_scatter_rows`] on tensors: `rows` `[m, w]` are written
/// into the pool `[n, w]` at the pool rows `dest` `[m]` (`-1` skips a row).
/// In place when the caller holds no other handle to `pool` (else the pool is
/// copied first); no two `dest` entries may name the same row.
pub(crate) fn scatter_rows(
    pool: BTensor<2>,
    rows: BTensor<2>,
    dest: BTensor<1, Int>,
) -> BTensor<2> {
    let [m, width] = rows.dims();
    assert_eq!(pool.dims()[1], width, "scatter_rows: row width");
    assert_eq!(dest.dims()[0], m, "scatter_rows: one destination per row");
    if m == 0 {
        return pool;
    }
    BTensor::from_dispatch(<Dispatch as CsaKernels>::csa_scatter_rows(
        pool.into_dispatch(),
        rows.into_dispatch(),
        dest.into_dispatch(),
    ))
}

/// [`CsaKernels::csa_pair_pool`] on tensors: `kv` and `score` are `[n, hd]`,
/// `left` and `right` `[m]` row indices; the result is `[m, hd]`, per channel
/// `kv0 e0 / (e0 + e1) + kv1 e1 / (e0 + e1)` with `e_i = exp(s_i - max(s0,
/// s1))` (`m >= 1`).
pub(crate) fn pair_pool(
    kv: BTensor<2>,
    score: BTensor<2>,
    left: BTensor<1, Int>,
    right: BTensor<1, Int>,
) -> BTensor<2> {
    let width = kv.dims()[1];
    let rows = left.dims()[0];
    assert!(rows > 0, "pair_pool: no pairs");
    assert_eq!(score.dims(), kv.dims(), "pair_pool: score shape");
    assert_eq!(right.dims()[0], rows, "pair_pool: pair lists");
    BTensor::from_dispatch(<Dispatch as CsaKernels>::csa_pair_pool(
        kv.into_dispatch(),
        score.into_dispatch(),
        left.into_dispatch(),
        right.into_dispatch(),
        PairLaunch { rows, width },
    ))
}

/// The scalars of one attention call.
#[derive(Clone, Debug)]
pub(crate) struct CsaCall {
    /// Query heads.
    pub(crate) heads: usize,
    /// Floats of a key (== value) and of a head's query/output.
    pub(crate) head_dim: usize,
    /// Window positions a row attends to.
    pub(crate) window: usize,
    /// Factor of every `q . key` score.
    pub(crate) scale: f32,
}

/// The compressed rows an attention call reads.
#[derive(Clone)]
pub(crate) struct CmpRead {
    /// The compressed pool `[rows, row_width]`: the first `head_dim` floats of
    /// a row are the key and the value.
    pub(crate) pool: BTensor<2>,
    /// Block tables `[n_seq, max_blocks]` of the compressed pool (-1: no block).
    pub(crate) tables: BTensor<2, Int>,
    /// The compressed rows of every query row `[t, K]`: ascending row ids of
    /// the row's sequence, -1 = unused slot.
    pub(crate) picks: BTensor<2, Int>,
}

/// How an attention call is launched (see the module docs).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    /// Split launch (a partial per chunk, then a merge kernel) or fused.
    pub(crate) split: bool,
    /// Query heads one cube serves.
    pub(crate) group: usize,
}

impl Plan {
    /// The plan for `rows` query rows of `call` (its window no wider than the
    /// block tables, see [`bounded`]) with `n_picks` pick slots: the split
    /// launch (many cubes, a scratch of partial results) unless the scratch
    /// outgrows [`SPLIT_SCRATCH_BYTES`], then the fused one. Both give the
    /// same results bit for bit.
    pub(crate) fn auto(call: &CsaCall, rows: usize, n_picks: usize) -> Plan {
        let n_chunks = call.window.div_ceil(LANES) + n_picks.div_ceil(LANES);
        let scratch = rows * call.heads * n_chunks * (call.head_dim + 2) * size_of::<f32>();
        Plan {
            split: scratch <= SPLIT_SCRATCH_BYTES,
            group: GROUP,
        }
    }
}

/// `call` with its window cut to the positions the window block tables can
/// name (`usize::MAX` stands for "every position of the tables"): a row never
/// reads further back, and the key slots are sized by the window.
fn bounded(call: &CsaCall, win_tables: &BTensor<2, Int>) -> CsaCall {
    CsaCall {
        window: call.window.min(win_tables.dims()[1] * BLOCK_SIZE),
        ..call.clone()
    }
}

/// [`CsaKernels::csa_attention`] on tensors (the semantics are in the module
/// docs): `q` is `[t, heads * head_dim]` with the rotation applied, `win_pool`
/// `[rows, head_dim]` read through `win_tables` `[n_seq, max_blocks]`, `cmp`
/// the compressed rows (`None` or no pick slots: the window alone), `row_seq`
/// and `row_pos` `[t]` the sequence and position of every query row, `sinks`
/// `[heads]`. The result is `[t, heads * head_dim]`, neither rotated back nor
/// quantized. The window may be `usize::MAX` (all the positions the tables
/// hold).
pub(crate) fn csa_attention(
    q: BTensor<2>,
    win_pool: &BTensor<2>,
    win_tables: &BTensor<2, Int>,
    cmp: Option<&CmpRead>,
    row_seq: BTensor<1, Int>,
    row_pos: BTensor<1, Int>,
    sinks: &BTensor<1>,
    call: &CsaCall,
) -> BTensor<2> {
    let call = bounded(call, win_tables);
    let n_picks = cmp.map_or(0, |c| c.picks.dims()[1]);
    let plan = Plan::auto(&call, q.dims()[0], n_picks);
    csa_attention_with(
        q, win_pool, win_tables, cmp, row_seq, row_pos, sinks, &call, plan,
    )
}

/// [`csa_attention`] with an explicit launch plan.
pub(crate) fn csa_attention_with(
    q: BTensor<2>,
    win_pool: &BTensor<2>,
    win_tables: &BTensor<2, Int>,
    cmp: Option<&CmpRead>,
    row_seq: BTensor<1, Int>,
    row_pos: BTensor<1, Int>,
    sinks: &BTensor<1>,
    call: &CsaCall,
    plan: Plan,
) -> BTensor<2> {
    let call = bounded(call, win_tables);
    let [rows, width] = q.dims();
    assert!(rows > 0, "csa_attention: no query rows");
    assert_eq!(
        width,
        call.heads * call.head_dim,
        "csa_attention: query width"
    );
    assert_eq!(
        win_pool.dims()[1],
        call.head_dim,
        "csa_attention: window row width"
    );
    assert_eq!(
        row_seq.dims()[0],
        rows,
        "csa_attention: one sequence per row"
    );
    assert_eq!(
        row_pos.dims()[0],
        rows,
        "csa_attention: one position per row"
    );
    assert_eq!(
        sinks.dims()[0],
        call.heads,
        "csa_attention: one sink per head"
    );
    assert!(plan.group > 0, "csa_attention: empty head group");
    // Without compressed rows the compressed arguments are never read: the
    // window's own tensors stand in for them.
    let (cmp_pool, cmp_tables, picks, n_picks) = match cmp.filter(|c| c.picks.dims()[1] > 0) {
        Some(c) => {
            let [pick_rows, k] = c.picks.dims();
            assert_eq!(pick_rows, rows, "csa_attention: picks per row");
            assert!(
                c.pool.dims()[1] >= call.head_dim,
                "csa_attention: compressed row width"
            );
            (c.pool.clone(), c.tables.clone(), c.picks.clone(), k)
        }
        None => (win_pool.clone(), win_tables.clone(), win_tables.clone(), 0),
    };
    let launch = CsaLaunch {
        scale: call.scale,
        n_head: call.heads,
        head_dim: call.head_dim,
        window: call.window,
        n_picks,
        group: plan.group,
        split: plan.split,
    };
    BTensor::from_dispatch(<Dispatch as CsaKernels>::csa_attention(
        q.into_dispatch(),
        win_pool.clone().into_dispatch(),
        win_tables.clone().into_dispatch(),
        cmp_pool.into_dispatch(),
        cmp_tables.into_dispatch(),
        picks.into_dispatch(),
        row_seq.into_dispatch(),
        row_pos.into_dispatch(),
        sinks.clone().into_dispatch(),
        launch,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::Instant;

    use burn::tensor::{Device, TensorData};

    use super::super::gpu::{kernel_test_device, read_f32};
    use super::*;

    /// A small deterministic generator (xorshift).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 11) as usize % n
        }

        /// Uniform in `[-1, 1)`.
        fn unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn floats<const D: usize>(data: &[f32], shape: [usize; D], device: &Device) -> BTensor<D> {
        BTensor::from_data(TensorData::new(data.to_vec(), shape), device)
    }

    fn ints<const D: usize>(data: &[i32], shape: [usize; D], device: &Device) -> BTensor<D, Int> {
        BTensor::from_data(TensorData::new(data.to_vec(), shape), device)
    }

    fn read<const D: usize>(tensor: BTensor<D>) -> Vec<f32> {
        read_f32(tensor, "a csa kernel result").unwrap()
    }

    /// Every `len`-wide vector of `got` is within `tol` of `want`, relative to
    /// the largest magnitude of the vector (NaN never passes).
    fn check(got: &[f32], want: &[f32], len: usize, tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (v, (g, w)) in got.chunks(len).zip(want.chunks(len)).enumerate() {
            let mag = w.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let err = g.iter().zip(w).fold(0f32, |m, (a, b)| {
                let e = (a - b).abs();
                m.max(if e.is_nan() { f32::INFINITY } else { e })
            });
            assert!(
                err <= tol * mag,
                "{what}: vector {v} is off by {err} (magnitude {mag})"
            );
        }
    }

    // ------------------------------------------------------- attention case

    /// What an attention case is made of.
    struct Spec {
        heads: usize,
        hd: usize,
        window: usize,
        /// Pick slots per row.
        k: usize,
        cmp_width: usize,
        /// Tokens per compressed row: a row at position `p` sees `(p + 1) /
        /// ratio` of them.
        ratio: usize,
        with_cmp: bool,
        /// `(sequence, position)` of every query row.
        rows: Vec<(usize, usize)>,
        /// Pool blocks no table names (poisoned like all unread rows).
        spare_win: usize,
        spare_cmp: usize,
    }

    /// An attention call's host data. Every pool row the call must not read is
    /// NaN: the window rows before a row's window or past its position, the
    /// blocks no table names, the index-key tail of a compressed row.
    struct Case {
        heads: usize,
        hd: usize,
        window: usize,
        k: usize,
        cmp_width: usize,
        scale: f32,
        with_cmp: bool,
        q: Vec<f32>,
        sinks: Vec<f32>,
        win_pool: Vec<f32>,
        win_tables: Vec<i32>,
        win_blocks: usize,
        cmp_pool: Vec<f32>,
        cmp_tables: Vec<i32>,
        cmp_blocks: usize,
        picks: Vec<i32>,
        row_seq: Vec<i32>,
        row_pos: Vec<i32>,
    }

    /// Block tables for the items `need[seq]` (positions or compressed row
    /// ids) of every sequence: blocks holding a needed item get random pool
    /// blocks, the rest of the table is -1. Returns `(tables, columns, pool
    /// blocks, pool row of every needed item in (sequence, item) order)`.
    fn place(
        rng: &mut Rng,
        need: &[BTreeSet<usize>],
        spare: usize,
        pad: usize,
    ) -> (Vec<i32>, usize, usize, Vec<usize>) {
        let cols = need
            .iter()
            .filter_map(|s| s.iter().next_back())
            .map(|&i| i / BLOCK_SIZE + 1)
            .max()
            .unwrap_or(0)
            + pad;
        let used: Vec<Vec<usize>> = need
            .iter()
            .map(|s| {
                s.iter()
                    .map(|i| i / BLOCK_SIZE)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            })
            .collect();
        let total = used.iter().map(Vec::len).sum::<usize>() + spare;
        let mut perm: Vec<usize> = (0..total).collect();
        for i in (1..total).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        let mut tables = vec![-1i32; need.len() * cols];
        let mut next = 0;
        for (seq, blocks) in used.iter().enumerate() {
            for &b in blocks {
                tables[seq * cols + b] = perm[next] as i32;
                next += 1;
            }
        }
        let mut rows = Vec::new();
        for (seq, set) in need.iter().enumerate() {
            for &i in set {
                rows.push(
                    tables[seq * cols + i / BLOCK_SIZE] as usize * BLOCK_SIZE + i % BLOCK_SIZE,
                );
            }
        }
        (tables, cols, total, rows)
    }

    fn build_case(rng: &mut Rng, s: &Spec) -> Case {
        let t = s.rows.len();
        let n_seq = s.rows.iter().map(|r| r.0).max().unwrap() + 1;
        let hd = s.hd;
        let k = if s.with_cmp { s.k } else { 0 };

        let mut win_need = vec![BTreeSet::new(); n_seq];
        for &(seq, p) in &s.rows {
            for pos in (p + 1).saturating_sub(s.window)..=p {
                win_need[seq].insert(pos);
            }
        }

        // The picks: a random ascending subset of the visible compressed rows.
        let mut picks = vec![-1i32; t * k];
        for (r, &(_, p)) in s.rows.iter().enumerate() {
            let visible = (p + 1) / s.ratio;
            let take = visible.min(k);
            let mut ids: Vec<usize> = (0..visible).collect();
            for i in 0..take {
                let j = i + rng.below(visible - i);
                ids.swap(i, j);
            }
            let mut chosen = ids[..take].to_vec();
            chosen.sort_unstable();
            for (slot, id) in chosen.into_iter().enumerate() {
                picks[r * k + slot] = id as i32;
            }
        }
        // An unused slot among used ones in one row.
        if let Some(r) = (0..t).find(|&r| k >= 3 && picks[r * k + 2] >= 0) {
            picks[r * k + 1] = -1;
        }
        let mut cmp_need = vec![BTreeSet::new(); n_seq];
        for (r, &(seq, _)) in s.rows.iter().enumerate() {
            for &j in &picks[r * k..(r + 1) * k] {
                if j >= 0 {
                    cmp_need[seq].insert(j as usize);
                }
            }
        }

        let (win_tables, win_blocks, wtotal, wrows) = place(rng, &win_need, s.spare_win, 2);
        let mut win_pool = vec![f32::NAN; wtotal * BLOCK_SIZE * hd];
        for &row in &wrows {
            for v in &mut win_pool[row * hd..(row + 1) * hd] {
                *v = rng.unit();
            }
        }
        let (cmp_tables, cmp_blocks, ctotal, crows) = if s.with_cmp {
            place(rng, &cmp_need, s.spare_cmp, 2)
        } else {
            (Vec::new(), 0, 0, Vec::new())
        };
        let mut cmp_pool = vec![f32::NAN; ctotal * BLOCK_SIZE * s.cmp_width];
        for &row in &crows {
            for v in &mut cmp_pool[row * s.cmp_width..row * s.cmp_width + hd] {
                *v = rng.unit();
            }
        }
        Case {
            heads: s.heads,
            hd,
            window: s.window,
            k,
            cmp_width: s.cmp_width,
            scale: 1.5 / (hd as f32).sqrt(),
            with_cmp: s.with_cmp,
            q: (0..t * s.heads * hd).map(|_| rng.unit()).collect(),
            sinks: (0..s.heads).map(|_| rng.unit() * 2.0).collect(),
            win_pool,
            win_tables,
            win_blocks,
            cmp_pool,
            cmp_tables,
            cmp_blocks,
            picks,
            row_seq: s.rows.iter().map(|r| r.0 as i32).collect(),
            row_pos: s.rows.iter().map(|r| r.1 as i32).collect(),
        }
    }

    /// The pool row (times `width`) of item `i` of a table row, if it has a block.
    fn pool_at(table: &[i32], i: usize, width: usize) -> Option<usize> {
        let b = *table.get(i / BLOCK_SIZE)?;
        (b >= 0).then(|| (b as usize * BLOCK_SIZE + i % BLOCK_SIZE) * width)
    }

    /// The straightforward attention: `csa_attention`'s loops of the CPU
    /// engine in f64 (the window, then the picks; sink; `m` clamped).
    fn host_attention(c: &Case) -> Vec<f32> {
        let (h, hd, k) = (c.heads, c.hd, c.k);
        let t = c.row_pos.len();
        let mut out = vec![0f32; t * h * hd];
        for r in 0..t {
            let seq = c.row_seq[r] as usize;
            let p = c.row_pos[r] as usize;
            let lo = (p + 1).saturating_sub(c.window);
            let wtable = &c.win_tables[seq * c.win_blocks..(seq + 1) * c.win_blocks];
            let mut keys: Vec<&[f32]> = Vec::new();
            for pos in lo..=p {
                let at = pool_at(wtable, pos, hd).expect("a window position has a block");
                keys.push(&c.win_pool[at..at + hd]);
            }
            if c.with_cmp {
                let ctable = &c.cmp_tables[seq * c.cmp_blocks..(seq + 1) * c.cmp_blocks];
                for &j in &c.picks[r * k..(r + 1) * k] {
                    if j >= 0 {
                        let at =
                            pool_at(ctable, j as usize, c.cmp_width).expect("a pick has a block");
                        keys.push(&c.cmp_pool[at..at + hd]);
                    }
                }
            }
            for head in 0..h {
                let qv = &c.q[(r * h + head) * hd..(r * h + head + 1) * hd];
                let scores: Vec<f64> = keys
                    .iter()
                    .map(|key| {
                        qv.iter()
                            .zip(*key)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                            * f64::from(c.scale)
                    })
                    .collect();
                let m = scores
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(-1e30);
                let weights: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let denom = weights.iter().sum::<f64>() + (f64::from(c.sinks[head]) - m).exp();
                let mut acc = vec![0f64; hd];
                for (w, key) in weights.iter().zip(&keys) {
                    for (a, v) in acc.iter_mut().zip(*key) {
                        *a += w * f64::from(*v);
                    }
                }
                for (o, a) in out[(r * h + head) * hd..(r * h + head + 1) * hd]
                    .iter_mut()
                    .zip(&acc)
                {
                    *o = (a / denom) as f32;
                }
            }
        }
        out
    }

    /// The device tensors of an attention call over some rows of a case.
    struct Uploaded {
        q: BTensor<2>,
        win: BTensor<2>,
        win_tables: BTensor<2, Int>,
        cmp: Option<CmpRead>,
        row_seq: BTensor<1, Int>,
        row_pos: BTensor<1, Int>,
        sinks: BTensor<1>,
        call: CsaCall,
    }

    impl Uploaded {
        fn new(device: &Device, c: &Case, rows: &[usize]) -> Uploaded {
            let (h, hd, k) = (c.heads, c.hd, c.k);
            let t = rows.len();
            let mut q = Vec::new();
            let (mut seq, mut pos, mut picks) = (Vec::new(), Vec::new(), Vec::new());
            for &r in rows {
                q.extend_from_slice(&c.q[r * h * hd..(r + 1) * h * hd]);
                seq.push(c.row_seq[r]);
                pos.push(c.row_pos[r]);
                picks.extend_from_slice(&c.picks[r * k..(r + 1) * k]);
            }
            let n_seq = c.win_tables.len() / c.win_blocks;
            let cmp = c.with_cmp.then(|| CmpRead {
                pool: floats(
                    &c.cmp_pool,
                    [c.cmp_pool.len() / c.cmp_width, c.cmp_width],
                    device,
                ),
                tables: ints(&c.cmp_tables, [n_seq, c.cmp_blocks], device),
                picks: ints(&picks, [t, k], device),
            });
            Uploaded {
                q: floats(&q, [t, h * hd], device),
                win: floats(&c.win_pool, [c.win_pool.len() / hd, hd], device),
                win_tables: ints(&c.win_tables, [n_seq, c.win_blocks], device),
                cmp,
                row_seq: ints(&seq, [t], device),
                row_pos: ints(&pos, [t], device),
                sinks: floats(&c.sinks, [h], device),
                call: CsaCall {
                    heads: h,
                    head_dim: hd,
                    window: c.window,
                    scale: c.scale,
                },
            }
        }

        fn attend(&self, plan: Plan) -> BTensor<2> {
            csa_attention_with(
                self.q.clone(),
                &self.win,
                &self.win_tables,
                self.cmp.as_ref(),
                self.row_seq.clone(),
                self.row_pos.clone(),
                &self.sinks,
                &self.call,
                plan,
            )
        }
    }

    fn plans(groups: &[usize]) -> Vec<Plan> {
        groups
            .iter()
            .flat_map(|&group| [true, false].map(|split| Plan { split, group }))
            .collect()
    }

    fn tiny_spec(with_cmp: bool) -> Spec {
        // Several sequences in one launch, rows out of order.
        let rows = vec![
            (0, 8),
            (1, 40),
            (0, 0),
            (2, 41),
            (0, 40),
            (1, 3),
            (0, 1),
            (2, 40),
            (0, 7),
            (1, 17),
        ];
        Spec {
            heads: 4,
            hd: 32,
            window: 8,
            k: 4,
            cmp_width: 64,
            ratio: 4,
            with_cmp,
            rows,
            spare_win: 3,
            spare_cmp: 2,
        }
    }

    #[test]
    fn csa_attention_matches_host_on_tiny_shapes() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut shapes: Vec<Spec> = vec![tiny_spec(true), tiny_spec(false)];
        // A layer without a window (`usize::MAX`) attends to every position
        // the block tables hold.
        shapes.push(Spec {
            window: usize::MAX,
            ..tiny_spec(true)
        });
        // Two dimensions per lane (the second ones past the head for half the
        // lanes), a window and a pick list that are not a multiple of a chunk
        // (two chunks of each, the last of each mostly empty).
        shapes.push(Spec {
            heads: 3,
            hd: 96,
            window: 70,
            k: 70,
            cmp_width: 120,
            ratio: 2,
            with_cmp: true,
            rows: vec![(0, 5), (1, 69), (1, 70), (0, 150), (2, 300), (1, 301)],
            spare_win: 2,
            spare_cmp: 1,
        });
        for (si, spec) in shapes.iter().enumerate() {
            let case = build_case(&mut rng, spec);
            let want = host_attention(&case);
            let all: Vec<usize> = (0..spec.rows.len()).collect();
            let up = Uploaded::new(&device, &case, &all);
            // Groups that divide the heads, do not, and exceed them.
            for plan in plans(&[1, 2, 3, 4, 8]) {
                let got = read(up.attend(plan));
                check(&got, &want, spec.hd, 1e-4, &format!("shape {si} {plan:?}"));
            }
            // The automatic plan of the shape.
            let got = read(csa_attention(
                up.q.clone(),
                &up.win,
                &up.win_tables,
                up.cmp.as_ref(),
                up.row_seq.clone(),
                up.row_pos.clone(),
                &up.sinks,
                &up.call,
            ));
            check(&got, &want, spec.hd, 1e-4, &format!("shape {si} auto"));
        }
    }

    fn real_spec(rows: Vec<(usize, usize)>) -> Spec {
        Spec {
            heads: 64,
            hd: 512,
            window: 128,
            k: 512,
            cmp_width: 640,
            ratio: 2,
            with_cmp: true,
            rows,
            spare_win: 200,
            spare_cmp: 100,
        }
    }

    fn real_rows(t: usize) -> Vec<(usize, usize)> {
        match t {
            1 => vec![(0, 1999)],
            4 => vec![(0, 2000), (1, 130), (0, 2001), (1, 131)],
            33 => {
                let mut rows: Vec<(usize, usize)> = (1900..1916).map(|p| (0, p)).collect();
                rows.extend((700..709).map(|p| (1, p)));
                rows.extend((0..8).map(|p| (2, p)));
                rows
            }
            t => (0..t).map(|i| (0, 2000 + i)).collect(),
        }
    }

    #[test]
    fn csa_attention_matches_host_on_real_shapes() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for t in [1usize, 4, 33] {
            let case = build_case(&mut rng, &real_spec(real_rows(t)));
            let want = host_attention(&case);
            let all: Vec<usize> = (0..t).collect();
            let up = Uploaded::new(&device, &case, &all);
            let started = Instant::now();
            for plan in [
                Plan::auto(&up.call, t, 512),
                Plan {
                    split: true,
                    group: 4,
                },
                Plan {
                    split: false,
                    group: 4,
                },
            ] {
                let got = read(up.attend(plan));
                check(&got, &want, 512, 1e-4, &format!("t {t} {plan:?}"));
            }
            println!(
                "real shapes t={t}: ok ({:.2?} with compiles)",
                started.elapsed()
            );
        }
    }

    /// The harness sees a stray read: a window one position too wide reads
    /// pool rows that are NaN, and the comparison must fail.
    #[test]
    fn csa_attention_check_sees_a_stray_read() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
        let spec = tiny_spec(true);
        let case = build_case(&mut rng, &spec);
        let want = host_attention(&case);
        let all: Vec<usize> = (0..spec.rows.len()).collect();
        let mut up = Uploaded::new(&device, &case, &all);
        let honest = read(up.attend(Plan {
            split: false,
            group: 4,
        }));
        check(&honest, &want, spec.hd, 1e-4, "the honest call");
        up.call.window += 1;
        let stray = read(up.attend(Plan {
            split: false,
            group: 4,
        }));
        let caught = std::panic::catch_unwind(|| check(&stray, &want, spec.hd, 1e-4, "stray"));
        assert!(caught.is_err(), "a read past the window must be caught");
    }

    /// A decode step splits, a launch whose partials would outgrow the
    /// scratch limit fuses.
    #[test]
    fn csa_attention_plan_follows_the_scratch_size() {
        let call = CsaCall {
            heads: 64,
            head_dim: 512,
            window: 128,
            scale: 1.0,
        };
        for rows in [1usize, 4, 33, 256] {
            assert!(Plan::auto(&call, rows, 512).split, "{rows} rows split");
        }
        assert!(!Plan::auto(&call, 4096, 512).split, "4096 rows fuse");
        assert_eq!(Plan::auto(&call, 1, 0).group, GROUP);
    }

    #[test]
    fn csa_attention_rows_do_not_depend_on_the_launch() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let case = build_case(&mut rng, &real_spec(real_rows(33)));
        let width = case.heads * case.hd;
        let all: Vec<usize> = (0..33).collect();
        let up_all = Uploaded::new(&device, &case, &all);
        for plan in [
            Plan {
                split: true,
                group: 4,
            },
            Plan {
                split: false,
                group: 4,
            },
        ] {
            let together = read(up_all.attend(plan));
            for rows in [vec![5usize], vec![20, 31], vec![0, 17, 24, 32]] {
                let up = Uploaded::new(&device, &case, &rows);
                let alone = read(up.attend(plan));
                for (i, &r) in rows.iter().enumerate() {
                    assert_eq!(
                        alone[i * width..(i + 1) * width],
                        together[r * width..(r + 1) * width],
                        "row {r} {plan:?} changes with the launch"
                    );
                }
            }
        }
        // The two launch shapes fold the same chunks the same way.
        let split = read(up_all.attend(Plan {
            split: true,
            group: 4,
        }));
        let fused = read(up_all.attend(Plan {
            split: false,
            group: 4,
        }));
        let equal = split.iter().zip(&fused).filter(|(a, b)| a == b).count();
        println!("split == fused bit for bit: {equal} of {}", split.len());
        check(&split, &fused, case.hd, 1e-6, "split against fused");
    }

    // ------------------------------------------------------ element kernels

    #[test]
    fn csa_rope_tail_matches_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xA076_1D64_78BD_642F);
        // (rows, vectors per row, width, rotated, positions)
        for (t, n_vec, width, rot, positions) in [
            (5usize, 4usize, 32usize, 8usize, 64usize),
            (3, 2, 16, 16, 32),
            (33, 64, 512, 64, 4096),
            (33, 1, 512, 64, 4096),
        ] {
            let half = rot / 2;
            let angle = |p: usize, i: usize| p as f32 * 0.37 * (i + 1) as f32 / half as f32;
            let cos: Vec<f32> = (0..positions * half)
                .map(|k| angle(k / half, k % half).cos())
                .collect();
            let sin: Vec<f32> = (0..positions * half)
                .map(|k| angle(k / half, k % half).sin())
                .collect();
            let x: Vec<f32> = (0..t * n_vec * width).map(|_| rng.unit()).collect();
            let mut pos: Vec<i32> = (0..t).map(|_| rng.below(positions) as i32).collect();
            pos[0] = 0;
            pos[t - 1] = positions as i32 - 1;
            let ct = floats(&cos, [positions, half], &device);
            let st = floats(&sin, [positions, half], &device);
            let pt = ints(&pos, [t], &device);
            for inverse in [false, true] {
                let mut want = x.clone();
                for (r, row) in want.chunks_mut(n_vec * width).enumerate() {
                    let p = pos[r] as usize;
                    for v in row.chunks_mut(width) {
                        for (i, pair) in v[width - rot..]
                            .as_chunks_mut::<2>()
                            .0
                            .iter_mut()
                            .enumerate()
                        {
                            let (a, b) = (pair[0], pair[1]);
                            let s = if inverse {
                                -sin[p * half + i]
                            } else {
                                sin[p * half + i]
                            };
                            let c = cos[p * half + i];
                            pair[0] = a * c - b * s;
                            pair[1] = a * s + b * c;
                        }
                    }
                }
                let got = read(rope_tail(
                    floats(&x, [t, n_vec * width], &device),
                    &ct,
                    &st,
                    pt.clone(),
                    width,
                    rot,
                    inverse,
                ));
                check(
                    &got,
                    &want,
                    width,
                    1e-5,
                    &format!("rope t {t} vectors {n_vec} width {width} inverse {inverse}"),
                );
            }
        }
    }

    #[test]
    fn csa_scatter_rows_matches_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xE703_7ED1_A0B4_28DB);
        for (n_pool, w, m, skipped) in [
            (20usize, 8usize, 6usize, 2usize),
            (4800, 512, 33, 5),
            (4800, 640, 33, 5),
        ] {
            let pool: Vec<f32> = (0..n_pool * w).map(|_| rng.unit()).collect();
            let rows: Vec<f32> = (0..m * w).map(|_| rng.unit()).collect();
            let mut ids: Vec<usize> = (0..n_pool).collect();
            for i in 0..m {
                let j = i + rng.below(n_pool - i);
                ids.swap(i, j);
            }
            let dest: Vec<i32> = (0..m)
                .map(|r| {
                    if r % 7 < skipped.min(2) || r == m - 1 {
                        -1
                    } else {
                        ids[r] as i32
                    }
                })
                .collect();
            let mut want = pool.clone();
            for (r, &d) in dest.iter().enumerate() {
                if d >= 0 {
                    want[d as usize * w..(d as usize + 1) * w]
                        .copy_from_slice(&rows[r * w..(r + 1) * w]);
                }
            }
            assert!(dest.contains(&-1) && dest.iter().any(|&d| d >= 0));
            let rows_t = floats(&rows, [m, w], &device);
            let dest_t = ints(&dest, [m], &device);
            // Another handle to the pool: the pool is copied, the first stays.
            let held = floats(&pool, [n_pool, w], &device);
            let kept = held.clone();
            let out = scatter_rows(held, rows_t.clone(), dest_t.clone());
            assert!(
                read(out) == want,
                "scatter into a shared pool ({n_pool} x {w})"
            );
            assert!(read(kept) == pool, "a shared pool must not change");
            // The only handle: written in place.
            let sole = floats(&pool, [n_pool, w], &device);
            let out = scatter_rows(sole, rows_t, dest_t);
            assert!(read(out) == want, "scatter in place ({n_pool} x {w})");
        }
    }

    #[test]
    fn csa_pair_pool_matches_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x8CB9_2BA7_2F3D_8DD7);
        for (n, hd, m) in [(10usize, 32usize, 5usize), (40, 512, 20)] {
            let kv: Vec<f32> = (0..n * hd).map(|_| rng.unit()).collect();
            let score: Vec<f32> = (0..n * hd).map(|_| rng.unit() * 6.0).collect();
            // The consecutive pairs of a call, then arbitrary ones (a row may
            // pair with itself).
            let mut left: Vec<i32> = (0..m).map(|i| (i * 2 % n) as i32).collect();
            let mut right: Vec<i32> = (0..m).map(|i| (i * 2 % n + 1) as i32).collect();
            left[m - 1] = rng.below(n) as i32;
            right[m - 1] = left[m - 1];
            left[m - 2] = rng.below(n) as i32;
            right[m - 2] = rng.below(n) as i32;
            let mut want = vec![0f32; m * hd];
            for r in 0..m {
                let (l, rr) = (left[r] as usize, right[r] as usize);
                for c in 0..hd {
                    let (s0, s1) = (score[l * hd + c], score[rr * hd + c]);
                    let mx = s0.max(s1);
                    let (e0, e1) = ((s0 - mx).exp(), (s1 - mx).exp());
                    let total = e0 + e1;
                    want[r * hd + c] =
                        kv[l * hd + c] * (e0 / total) + kv[rr * hd + c] * (e1 / total);
                }
            }
            let got = read(pair_pool(
                floats(&kv, [n, hd], &device),
                floats(&score, [n, hd], &device),
                ints(&left, [m], &device),
                ints(&right, [m], &device),
            ));
            check(&got, &want, hd, 1e-5, &format!("pair pool {n} x {hd}"));
        }
    }

    // ---------------------------------------------------------------- timing

    /// Milliseconds per call of `run`: the minimum and the median of seven
    /// rounds of `iters` calls, each round after a flush.
    fn time_ms(device: &Device, mut run: impl FnMut() -> BTensor<2>, iters: usize) -> (f64, f64) {
        for _ in 0..3 {
            let _ = run();
        }
        device.sync().unwrap();
        let mut rounds: Vec<f64> = (0..7)
            .map(|_| {
                let started = Instant::now();
                for _ in 0..iters {
                    let _ = run();
                }
                device.sync().unwrap();
                started.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .collect();
        rounds.sort_by(f64::total_cmp);
        (rounds[0], rounds[3])
    }

    #[test]
    fn csa_attention_timings() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x5851_F42D_4C95_7F2D);
        for t in [1usize, 4, 8, 16, 33, 64, 128, 256] {
            let case = build_case(&mut rng, &real_spec(real_rows(t)));
            let all: Vec<usize> = (0..t).collect();
            let up = Uploaded::new(&device, &case, &all);
            for plan in plans(&[2, 4]) {
                let (min, median) = time_ms(&device, || up.attend(plan), 20);
                println!(
                    "TIMING csa_attention t={t:>2} K=512 {plan:?}: min {min:.3} median {median:.3} ms"
                );
            }
        }
    }

    /// The element kernels at the real model's shapes (the launch floor of an
    /// op is the cost of the smallest ones).
    #[test]
    fn csa_element_kernel_timings() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let (positions, half, width, rot) = (4096usize, 32usize, 512usize, 64usize);
        let table: Vec<f32> = (0..positions * half).map(|_| rng.unit()).collect();
        let cos = floats(&table, [positions, half], &device);
        let sin = floats(&table, [positions, half], &device);
        for (t, n_vec) in [(1usize, 64usize), (33, 64), (33, 1)] {
            let x: Vec<f32> = (0..t * n_vec * width).map(|_| rng.unit()).collect();
            let xt = floats(&x, [t, n_vec * width], &device);
            let pos = ints(&vec![7i32; t], [t], &device);
            let (min, median) = time_ms(
                &device,
                || rope_tail(xt.clone(), &cos, &sin, pos.clone(), width, rot, false),
                50,
            );
            println!(
                "TIMING rope_tail t={t:>2} vectors={n_vec:>2} width 512: min {min:.3} median {median:.3} ms"
            );
        }
        for m in [1usize, 33] {
            let n = 2 * m;
            let kv: Vec<f32> = (0..n * width).map(|_| rng.unit()).collect();
            let kvt = floats(&kv, [n, width], &device);
            let left: Vec<i32> = (0..m).map(|i| 2 * i as i32).collect();
            let right: Vec<i32> = (0..m).map(|i| 2 * i as i32 + 1).collect();
            let (lt, rt) = (ints(&left, [m], &device), ints(&right, [m], &device));
            let (min, median) = time_ms(
                &device,
                || pair_pool(kvt.clone(), kvt.clone(), lt.clone(), rt.clone()),
                50,
            );
            println!("TIMING pair_pool m={m:>2} hd 512: min {min:.3} median {median:.3} ms");
        }
        // In place: the pool is the only handle of every call's result.
        let (n_pool, m) = (4800usize, 33usize);
        let pool: Vec<f32> = (0..n_pool * width).map(|_| rng.unit()).collect();
        let rows: Vec<f32> = (0..m * width).map(|_| rng.unit()).collect();
        let dest: Vec<i32> = (0..m as i32).map(|i| i * 3).collect();
        let (rows_t, dest_t) = (
            floats(&rows, [m, width], &device),
            ints(&dest, [m], &device),
        );
        let mut pool_t = floats(&pool, [n_pool, width], &device);
        for _ in 0..3 {
            pool_t = scatter_rows(pool_t, rows_t.clone(), dest_t.clone());
        }
        device.sync().unwrap();
        let started = Instant::now();
        for _ in 0..100 {
            pool_t = scatter_rows(pool_t, rows_t.clone(), dest_t.clone());
        }
        device.sync().unwrap();
        println!(
            "TIMING scatter_rows m={m} into 4800 x 512 (in place): {:.3} ms",
            started.elapsed().as_secs_f64() * 1e1
        );
    }
}
