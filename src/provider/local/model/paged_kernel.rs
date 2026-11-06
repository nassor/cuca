//! The decode-attention kernel of the GPU engine's paged K/V cache (CubeCL).
//!
//! A layer's keys and values live in two pools `[n_blocks, n_kv_head,
//! BLOCK_SIZE, head_dim]`; a sequence's block table maps position `p` to the
//! block `table[p / BLOCK_SIZE]`, row `p % BLOCK_SIZE` (see
//! [`paged`](super::paged)). A prefill or verification call gathers the
//! blocks it reads into contiguous tensors and uses burn's matmuls; a single
//! query row cannot afford the gather, so [`PagedKernels::paged_attention_decode`]
//! reads the blocks in place:
//!
//! - the row's keys are split into chunks of one block (at most 128
//!   chunks, a function of the table width only), and one cube per (KV head,
//!   chunk, query row) runs the first pass, one unit per query head of that
//!   KV head (the grouped heads share every key and value read from the
//!   pool): an online softmax over its chunk, a running maximum and sum, a
//!   `head_dim`-wide accumulator rescaled whenever the maximum grows, f32
//!   throughout, the score being `q . k * scale`;
//! - a second pass merges the chunks' partial results of each (row, head) in
//!   chunk order (the global maximum, every partial rescaled to it), so the
//!   result does not depend on how the work was scheduled. One serial chunk
//!   per head took 17 ms per layer for a 1,100-key decode on the RX 9070 XT:
//!   the split is what makes the kernel fast.
//!
//! The kernel is written in safe Rust (`#[cube(launch)]` with the checked
//! `launch`) and reaches the backend through burn's `backend_extension`, as
//! the [`moe_kernel`](super::moe_kernel) ones do. The caller checks the block
//! table on the host: every entry the range touches must hold a block.

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values (`scale - scale` is a runtime zero where a
// literal would be evaluated by rustc), plain `x = x * c` and nested `if`s.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
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

/// Most chunks one row's keys are split into.
const MAX_SPLITS: usize = 128;

/// The scalars of one decode-attention call.
#[derive(Clone, Debug)]
pub(crate) struct PagedCall {
    /// Factor of every `q . k` score.
    pub(crate) scale: f32,
}

/// First pass of the split decode attention: cube `(kvh * splits + s, row)`
/// runs the online softmax of its query heads (one unit each) over chunk `s`
/// of the row's keys `lo..hi` and writes the partial result `[acc (d) | max |
/// sum]` to `part[((row * n_head + head) * splits + s) * (d + 2)]`. A chunk
/// past the end writes an empty partial (max `-1e30`, sum 0). `tables` is
/// `[rows, max_blocks]` block ids and `ranges` `[rows * 2]` (`lo`, `hi`)
/// positions relative to the table's first block.
#[cube(launch)]
fn paged_decode_kernel(
    q: &Tensor<f32>,
    pool_k: &Tensor<f32>,
    pool_v: &Tensor<f32>,
    tables: &Tensor<u32>,
    ranges: &Tensor<u32>,
    part: &mut Tensor<f32>,
    scale: f32,
    n_head: u32,
    n_kv: u32,
    max_blocks: u32,
    splits: u32,
    #[comptime] d: usize,
) {
    let kvh = CUBE_POS_X as usize / usize::cast_from(splits);
    let s = CUBE_POS_X as usize % usize::cast_from(splits);
    let row = CUBE_POS_Y as usize;
    let head = kvh * (usize::cast_from(n_head) / usize::cast_from(n_kv)) + UNIT_POS_X as usize;
    let q_at = (row * usize::cast_from(n_head) + head) * d;
    let lo = usize::cast_from(ranges[row * 2usize]);
    let hi = usize::cast_from(ranges[row * 2usize + 1usize]);
    let table_at = row * usize::cast_from(max_blocks);
    let chunk = (hi - lo + usize::cast_from(splits) - 1usize) / usize::cast_from(splits);
    let from = lo + s * chunk;
    let mut to = from + chunk;
    if to > hi {
        to = hi;
    }

    let mut acc = Array::<f32>::new(d);
    for i in 0..d {
        acc[i] = scale - scale;
    }
    let mut top = (scale - scale) - 1.0e30f32;
    let mut total = scale - scale;
    for p in from..to {
        let block = usize::cast_from(tables[table_at + p / 16usize]);
        let at = ((block * usize::cast_from(n_kv) + kvh) * 16usize + p % 16usize) * d;
        let mut score = scale - scale;
        for i in 0..d {
            score += q[q_at + i] * pool_k[at + i];
        }
        score = score * scale;
        if score > top {
            let c = (top - score).exp();
            total = total * c;
            for i in 0..d {
                acc[i] = acc[i] * c;
            }
            top = score;
        }
        let w = (score - top).exp();
        total += w;
        for i in 0..d {
            acc[i] += w * pool_v[at + i];
        }
    }
    let out_at =
        ((row * usize::cast_from(n_head) + head) * usize::cast_from(splits) + s) * (d + 2usize);
    for i in 0..d {
        part[out_at + i] = acc[i];
    }
    part[out_at + d] = top;
    part[out_at + d + 1usize] = total;
}

/// Second pass: one unit per (row, head, element) merges the `splits`
/// partials in split order: the global maximum, then every partial rescaled
/// by `exp(max_s - max)`; the result is `sum acc_s / sum sum_s`.
#[cube(launch)]
fn paged_merge_kernel(
    part: &Tensor<f32>,
    out: &mut Tensor<f32>,
    jobs: u32,
    splits: u32,
    #[comptime] d: usize,
) {
    let job = ABSOLUTE_POS;
    if job < usize::cast_from(jobs) * d {
        let head = job / d;
        let i = job % d;
        let base = head * usize::cast_from(splits) * (d + 2usize);
        let mut top = part[base + d];
        for s in 1..usize::cast_from(splits) {
            let m = part[base + s * (d + 2usize) + d];
            if m > top {
                top = m;
            }
        }
        let mut total = top - top;
        let mut v = top - top;
        for s in 0..usize::cast_from(splits) {
            let at = base + s * (d + 2usize);
            let w = (part[at + d] - top).exp();
            total += part[at + d + 1usize] * w;
            v += part[at + i] * w;
        }
        out[job] = v / total;
    }
}

// --------------------------------------------------------------- extension

/// Backend operations of the paged K/V cache (see the module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait PagedKernels: burn::backend::Backend {
    /// Attention of the query rows `q` `[rows, n_head, head_dim]` over the
    /// pools `[n_blocks, n_kv_head, 16, head_dim]` through the block `tables`
    /// `[rows, max_blocks]` and the `ranges` `[rows * 2]`: `[rows, n_head,
    /// head_dim]`.
    #[fusion(dtype = q, shape = q)]
    fn paged_attention_decode(
        q: FloatTensor<Self>,
        pool_k: FloatTensor<Self>,
        pool_v: FloatTensor<Self>,
        tables: IntTensor<Self>,
        ranges: IntTensor<Self>,
        call: PagedCall,
    ) -> FloatTensor<Self>;
}

impl PagedKernels for CubeBackend {
    fn paged_attention_decode(
        q: CubeTensor,
        pool_k: CubeTensor,
        pool_v: CubeTensor,
        tables: CubeTensor,
        ranges: CubeTensor,
        call: PagedCall,
    ) -> CubeTensor {
        // The kernels index the buffers linearly.
        let q = into_contiguous(q);
        let pool_k = into_contiguous(pool_k);
        let pool_v = into_contiguous(pool_v);
        let tables = into_contiguous(tables);
        let ranges = into_contiguous(ranges);
        let [rows, n_head, d] = q.shape().dims::<3>();
        let [_, n_kv, block, _] = pool_k.shape().dims::<4>();
        debug_assert_eq!(block, BLOCK_SIZE);
        let max_blocks = tables.shape().dims::<2>()[1];
        // One block per split, so a long context spreads
        // over many cubes (the unit count of one cube is only the query heads
        // of a KV head). The count depends on the table width alone.
        let splits = max_blocks.clamp(1, MAX_SPLITS);
        let client = q.client.clone();
        let part = empty_device_dtype(
            client.clone(),
            q.device.clone(),
            Shape::new([rows * n_head * splits * (d + 2)]),
            DType::F32,
        );
        let out = empty_device_dtype(
            client.clone(),
            q.device.clone(),
            Shape::new([rows, n_head, d]),
            DType::F32,
        );
        paged_decode_kernel::launch(
            &client,
            CubeCount::Static((n_kv * splits) as u32, rows as u32, 1),
            CubeDim::new_1d((n_head / n_kv) as u32),
            q.into_tensor_arg(),
            pool_k.into_tensor_arg(),
            pool_v.into_tensor_arg(),
            tables.into_tensor_arg(),
            ranges.into_tensor_arg(),
            part.clone().into_tensor_arg(),
            call.scale,
            n_head as u32,
            n_kv as u32,
            max_blocks as u32,
            splits as u32,
            d,
        );
        let jobs = rows * n_head;
        let dim = CubeDim::new(&client, jobs * d);
        let count = calculate_cube_count_elemwise(&client, jobs * d, dim);
        paged_merge_kernel::launch(
            &client,
            count,
            dim,
            part.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            jobs as u32,
            splits as u32,
            d,
        );
        out
    }
}

// ---------------------------------------------------------- tensor wrappers

use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor as BTensor};

/// [`PagedKernels::paged_attention_decode`] on tensors: `q` is `[rows,
/// n_head, head_dim]`, the pools `[n_blocks, n_kv_head, 16, head_dim]`,
/// `tables` `[rows, max_blocks]` and `ranges` `[rows * 2]`.
pub(crate) fn decode(
    q: BTensor<3>,
    pool_k: BTensor<4>,
    pool_v: BTensor<4>,
    tables: BTensor<2, Int>,
    ranges: BTensor<1, Int>,
    call: &PagedCall,
) -> BTensor<3> {
    BTensor::from_dispatch(<Dispatch as PagedKernels>::paged_attention_decode(
        q.into_dispatch(),
        pool_k.into_dispatch(),
        pool_v.into_dispatch(),
        tables.into_dispatch(),
        ranges.into_dispatch(),
        call.clone(),
    ))
}
