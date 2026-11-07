//! Dynamic job loops on candle's barrier pool, shared by the CPU kernels.
//!
//! Every kernel that splits its output over the pool (the row-batched matmul,
//! paged attention, latent attention's projections, the delta-net heads) runs
//! the same loop: each worker claims the next job index from one counter
//! until none is left. The arithmetic of one job never depends on which
//! worker runs it or in which order, so the split changes no bits.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::utils::barrier_pool;

/// Run `job(&mut state, i)` for every `i` in `0..jobs`. With `pooled` every
/// worker of candle's barrier pool (and the caller) claims indices
/// dynamically and owns one `state`, built by `init` when it claims its first
/// index (a worker that finds none left builds nothing); without it the
/// calling thread runs them all with one.
///
/// Must not be called from inside one of the pool's jobs.
pub(super) fn for_each<S>(
    jobs: usize,
    pooled: bool,
    init: impl Fn() -> S + Sync,
    job: impl Fn(&mut S, usize) + Sync,
) {
    let next = AtomicUsize::new(0);
    let work = |_: usize| {
        let mut state = None;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= jobs {
                break;
            }
            job(state.get_or_insert_with(&init), i);
        }
    };
    if pooled {
        barrier_pool().execute(work);
    } else {
        work(0);
    }
}

/// Run `f(i, piece)` for every `chunk`-sized piece `i` of `out`, on the pool
/// when there is more than one.
pub(super) fn par_chunks(out: &mut [f32], chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    par_chunks_with(out, chunk, || (), |(), i, piece| f(i, piece));
}

/// [`par_chunks`] where every worker owns one `state` built by `init`. A job
/// that sums into a scratch there and writes its piece once never shares a
/// cache line with the worker next door for longer than that write; summing
/// in the piece itself does, because pieces are not cache-line aligned and the
/// neighbours are claimed at the same time by other cores.
pub(super) fn par_chunks_with<S>(
    out: &mut [f32],
    chunk: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize, &mut [f32]) + Sync,
) {
    let slots: Vec<Mutex<&mut [f32]>> = out.chunks_mut(chunk.max(1)).map(Mutex::new).collect();
    for_each(slots.len(), slots.len() > 1, init, |state, i| {
        f(
            state,
            i,
            &mut slots[i].lock().unwrap_or_else(PoisonError::into_inner),
        )
    });
}
