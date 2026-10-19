//! Gated delta-net (linear attention) layers of hybrid models: the recurrent
//! state, the scan kernel and the snapshots cache rollback needs.
//!
//! A delta-net layer replaces the K/V cache of softmax attention with a
//! constant-size state per sequence: the last `conv_kernel - 1` inputs of a
//! causal depthwise convolution and, per value head, a `head_k_dim x
//! head_v_dim` f32 matrix `S`. For every token `t` and value head (query/key
//! head `h / (n_v_heads / n_k_heads)`, the key heads repeat-interleaved):
//!
//! ```text
//! x      = silu(sum_i conv[i] * mixed[t - (K-1) + i])      per channel
//! q, k, v = split(x);  q = l2norm(q) / sqrt(head_k_dim);  k = l2norm(k)
//! S     *= exp(g_t)                                         decay
//! delta  = (v - S^T k) * beta_t
//! S     += k delta^T                                        delta rule
//! o      = S^T q
//! y      = rms(o) * norm_weight * silu(z)                   gated norm
//! ```
//!
//! # Exactness
//!
//! The recurrence is sequential in `t` and runs the same f32 operations in
//! the same order whatever the batch: [`DeltaKernel::scan`] over `n` rows
//! gives the same output rows and the same final state, bit for bit, as `n`
//! scans of one row, or any split in between. Only the value heads run
//! concurrently (they never interact), so the thread count changes nothing
//! either. No f32 reduction is reassociated and no fused multiply-add is
//! used.
//!
//! # Snapshots
//!
//! The state is not positional: unlike K/V rows it cannot be truncated, and
//! the recurrence cannot be undone. [`DeltaCell`] holds, next to the live
//! state, what cache rollback restores from:
//!
//! - a **checkpoint**: a copy of the state at one chosen position;
//! - the **verify record** of the last forward call over several rows: a copy
//!   of the state before the call and the per-row inputs of the recurrence
//!   (the raw convolution inputs, `beta`, `exp(g)`: `conv_dim + 2 * n_v_heads`
//!   floats per row, not a state copy per row). The state after any prefix of
//!   those rows is rebuilt by restoring the pre-call copy and re-running
//!   [`DeltaKernel::scan`] on the first rows, which reproduces the original
//!   run bit for bit.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::utils::barrier_pool;

use super::params::DeltaSpec;

/// The l2 normalisation epsilon of the query and key (the reference
/// implementation's fixed value, independent of the RMSNorm epsilon).
const L2_EPS: f32 = 1e-6;

/// Token-heads below this many state elements per scan run on the calling
/// thread: the pool round trip would cost more than the work.
const SERIAL_BELOW: usize = 1 << 16;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The recurrent state of one delta-net layer for one sequence.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeltaState {
    /// The last `conv_kernel - 1` convolution inputs, oldest first:
    /// `(conv_kernel - 1)` rows of `conv_dim`.
    conv: Vec<f32>,
    /// `n_v_heads` matrices `[head_k_dim][head_v_dim]`, row-major, one after
    /// the other.
    s: Vec<f32>,
}

impl DeltaState {
    /// The all-zero state of a sequence start.
    pub(crate) fn zeros(spec: &DeltaSpec) -> Self {
        Self {
            conv: vec![0.0; (spec.conv_kernel - 1) * spec.conv_dim()],
            s: vec![0.0; spec.n_v_heads * spec.head_k_dim * spec.head_v_dim],
        }
    }

    /// Back to the sequence-start state.
    fn clear(&mut self) {
        self.conv.fill(0.0);
        self.s.fill(0.0);
    }

    /// Overwrite with `other`, reusing this state's allocation.
    fn copy_from(&mut self, other: &Self) {
        self.conv.clear();
        self.conv.extend_from_slice(&other.conv);
        self.s.clear();
        self.s.extend_from_slice(&other.s);
    }

    /// The matrix state, for tests.
    #[cfg(test)]
    pub(crate) fn matrices(&self) -> &[f32] {
        &self.s
    }

    /// The convolution history, for tests.
    #[cfg(test)]
    pub(crate) fn history(&self) -> &[f32] {
        &self.conv
    }
}

/// The per-token inputs of the recurrence of a run of `t` consecutive rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub(crate) t: usize,
    /// `t` rows of `conv_dim` raw convolution inputs (the output of
    /// `attn_qkv`, before the convolution).
    pub(crate) mixed: &'a [f32],
    /// `t` rows of `n_v_heads` write strengths `sigmoid(b)`.
    pub(crate) beta: &'a [f32],
    /// `t` rows of `n_v_heads` state decays `exp(g)`.
    pub(crate) decay: &'a [f32],
}

/// The convolution input rows a scan reads: the history, then the new rows.
#[derive(Clone, Copy)]
struct Window<'a> {
    history: &'a [f32],
    mixed: &'a [f32],
    history_rows: usize,
    conv_dim: usize,
}

impl<'a> Window<'a> {
    /// Row `r` of `history ++ mixed`.
    #[inline]
    fn row(&self, r: usize) -> &'a [f32] {
        let (src, at) = if r < self.history_rows {
            (self.history, r)
        } else {
            (self.mixed, r - self.history_rows)
        };
        &src[at * self.conv_dim..(at + 1) * self.conv_dim]
    }
}

/// Parameters of one delta-net layer's recurrence (everything but the
/// matmul weights).
#[derive(Debug, Clone)]
pub(crate) struct DeltaKernel {
    spec: DeltaSpec,
    /// Convolution taps, `[conv_kernel][conv_dim]`: tap `i` of channel `c`
    /// multiplies the input `conv_kernel - 1 - i` rows back.
    conv: Vec<f32>,
    /// Gated-norm weight, `head_v_dim` (multiplies directly).
    norm: Vec<f32>,
    /// Gated-norm epsilon.
    eps: f32,
    /// `ssm_a` per value head: `-exp(A_log)`.
    a: Vec<f32>,
    /// `ssm_dt.bias` per value head.
    dt_bias: Vec<f32>,
}

impl DeltaKernel {
    /// `conv` is the tensor as stored, `[conv_dim][conv_kernel]` (each
    /// channel's taps contiguous); it is transposed here so the scan reads
    /// the taps of neighbouring channels contiguously.
    pub(crate) fn new(
        spec: DeltaSpec,
        conv_stored: &[f32],
        norm: Vec<f32>,
        eps: f32,
        a: Vec<f32>,
        dt_bias: Vec<f32>,
    ) -> Self {
        let (cd, taps) = (spec.conv_dim(), spec.conv_kernel);
        let mut conv = vec![0.0; taps * cd];
        for c in 0..cd {
            for i in 0..taps {
                conv[i * cd + c] = conv_stored[c * taps + i];
            }
        }
        Self {
            spec,
            conv,
            norm,
            eps,
            a,
            dt_bias,
        }
    }

    pub(crate) fn spec(&self) -> &DeltaSpec {
        &self.spec
    }

    /// The write strengths `sigmoid(b)` and state decays
    /// `exp(ssm_a * softplus(a + dt_bias))` of `t` rows of the `ssm_beta` and
    /// `ssm_alpha` projections (`t` rows of `n_v_heads` each).
    pub(crate) fn gates(&self, b: &[f32], a: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let nv = self.spec.n_v_heads;
        let beta = b.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let decay = a
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let h = i % nv;
                let pre = x + self.dt_bias[h];
                // softplus with the reference's linear threshold.
                let softplus = if pre > 20.0 { pre } else { pre.exp().ln_1p() };
                (self.a[h] * softplus).exp()
            })
            .collect();
        (beta, decay)
    }

    /// Run the recurrence over `rows`, advancing `state`. With `z` (`t` rows
    /// of `value_dim`, the `attn_gate` projection) also returns the layer's
    /// `t x value_dim` output rows before `ssm_out`; without it only the state
    /// advances (the same state, bit for bit).
    pub(crate) fn scan(
        &self,
        state: &mut DeltaState,
        rows: &Rows<'_>,
        z: Option<&[f32]>,
    ) -> Option<Vec<f32>> {
        let d = &self.spec;
        let serial = rows.t * d.n_v_heads * d.head_k_dim * d.head_v_dim < SERIAL_BELOW;
        self.scan_with(state, rows, z, !serial)
    }

    /// [`DeltaKernel::scan`], choosing whether the heads run on the pool.
    fn scan_with(
        &self,
        state: &mut DeltaState,
        rows: &Rows<'_>,
        z: Option<&[f32]>,
        pooled: bool,
    ) -> Option<Vec<f32>> {
        let d = &self.spec;
        let (cd, dk, dv) = (d.conv_dim(), d.head_k_dim, d.head_v_dim);
        let history_rows = d.conv_kernel - 1;
        let t = rows.t;
        let DeltaState { conv, s } = state;
        let window = Window {
            history: conv,
            mixed: rows.mixed,
            history_rows,
            conv_dim: cd,
        };
        // One cell per value head: its state matrix and its output rows. The
        // heads never touch each other's cells, so the locks are uncontended.
        let cells: Vec<Mutex<(&mut [f32], Vec<f32>)>> = s
            .chunks_mut(dk * dv)
            .map(|m| Mutex::new((m, Vec::new())))
            .collect();
        let next = AtomicUsize::new(0);
        let work = |_: usize| {
            let mut scratch = HeadScratch::new(d);
            loop {
                let h = next.fetch_add(1, Ordering::Relaxed);
                let Some(cell) = cells.get(h) else {
                    break;
                };
                let mut guard = cell.lock().unwrap_or_else(PoisonError::into_inner);
                let (matrix, out) = &mut *guard;
                self.scan_head(h, matrix, &window, rows, z, out, &mut scratch);
            }
        };
        if pooled {
            barrier_pool().execute(work);
        } else {
            work(0);
        }

        // The new convolution history is the last rows of `history ++ mixed`.
        let mut next_history = vec![0.0; history_rows * cd];
        for (i, r) in (t..t + history_rows).enumerate() {
            next_history[i * cd..(i + 1) * cd].copy_from_slice(window.row(r));
        }
        let outs: Vec<Vec<f32>> = cells
            .into_iter()
            .map(|c| c.into_inner().unwrap_or_else(PoisonError::into_inner).1)
            .collect();
        conv.copy_from_slice(&next_history);

        z.map(|_| {
            let vd = d.value_dim();
            let mut out = vec![0.0; t * vd];
            for (h, head) in outs.iter().enumerate() {
                for ti in 0..t {
                    out[ti * vd + h * dv..ti * vd + (h + 1) * dv]
                        .copy_from_slice(&head[ti * dv..(ti + 1) * dv]);
                }
            }
            out
        })
    }

    /// The convolution of channels `off..off + dst.len()` at row `ti` of the
    /// new rows, with `silu`.
    fn conv_slice(&self, window: &Window<'_>, ti: usize, off: usize, dst: &mut [f32]) {
        let cd = self.spec.conv_dim();
        dst.fill(0.0);
        for i in 0..self.spec.conv_kernel {
            let x = &window.row(ti + i)[off..off + dst.len()];
            let w = &self.conv[i * cd + off..i * cd + off + dst.len()];
            for ((o, &w), &x) in dst.iter_mut().zip(w).zip(x) {
                *o += w * x;
            }
        }
        for o in dst {
            *o = silu(*o);
        }
    }

    /// One value head over all `rows`: `matrix` is its `[head_k_dim][head_v_dim]`
    /// state; `out` receives its `t x head_v_dim` gated-norm output when `z`
    /// is given.
    #[expect(
        clippy::too_many_arguments,
        reason = "one head's recurrence needs its state, its inputs and its scratch"
    )]
    fn scan_head(
        &self,
        h: usize,
        matrix: &mut [f32],
        window: &Window<'_>,
        rows: &Rows<'_>,
        z: Option<&[f32]>,
        out: &mut Vec<f32>,
        scratch: &mut HeadScratch,
    ) {
        let d = &self.spec;
        let (nv, dk, dv) = (d.n_v_heads, d.head_k_dim, d.head_v_dim);
        let kd = d.key_dim();
        // Repeat-interleave: consecutive value heads share a key head.
        let hk = h / (nv / d.n_k_heads);
        let (q_at, k_at, v_at) = (hk * dk, kd + hk * dk, 2 * kd + h * dv);
        let vd = d.value_dim();
        let sqrt_dk = (dk as f32).sqrt();

        let HeadScratch {
            q,
            k,
            v,
            kv_mem,
            delta,
            o,
        } = scratch;
        let (q, k, v) = (q.as_mut_slice(), k.as_mut_slice(), v.as_mut_slice());
        let (kv_mem, delta, o) = (
            kv_mem.as_mut_slice(),
            delta.as_mut_slice(),
            o.as_mut_slice(),
        );
        if z.is_some() {
            out.clear();
            out.reserve(rows.t * dv);
        }

        for ti in 0..rows.t {
            self.conv_slice(window, ti, k_at, k);
            self.conv_slice(window, ti, v_at, v);
            l2_normalise(k);
            let (beta, decay) = (rows.beta[ti * nv + h], rows.decay[ti * nv + h]);

            // Decay the state and read what it predicts for `k`.
            kv_mem.fill(0.0);
            for (i, &ki) in k.iter().enumerate() {
                let row = &mut matrix[i * dv..(i + 1) * dv];
                for (sv, m) in row.iter_mut().zip(kv_mem.iter_mut()) {
                    let decayed = *sv * decay;
                    *sv = decayed;
                    *m += decayed * ki;
                }
            }
            for ((dj, &vj), &mj) in delta.iter_mut().zip(v.iter()).zip(kv_mem.iter()) {
                *dj = (vj - mj) * beta;
            }

            // Write the correction and, when asked, read the output.
            if let Some(z) = z {
                self.conv_slice(window, ti, q_at, q);
                l2_normalise(q);
                for x in q.iter_mut() {
                    *x /= sqrt_dk;
                }
                o.fill(0.0);
                for (i, (&ki, &qi)) in k.iter().zip(q.iter()).enumerate() {
                    let row = &mut matrix[i * dv..(i + 1) * dv];
                    for ((sv, &dj), oj) in row.iter_mut().zip(delta.iter()).zip(o.iter_mut()) {
                        let written = *sv + ki * dj;
                        *sv = written;
                        *oj += written * qi;
                    }
                }
                // Gated RMSNorm per head: rms(o) * weight * silu(z).
                let mean_sq = o.iter().map(|x| x * x).sum::<f32>() / dv as f32;
                let scale = 1.0 / (mean_sq + self.eps).sqrt();
                let gate = &z[ti * vd + h * dv..ti * vd + (h + 1) * dv];
                out.extend(
                    o.iter()
                        .zip(&self.norm)
                        .zip(gate)
                        .map(|((&oj, &w), &g)| oj * scale * w * silu(g)),
                );
            } else {
                for (i, &ki) in k.iter().enumerate() {
                    let row = &mut matrix[i * dv..(i + 1) * dv];
                    for (sv, &dj) in row.iter_mut().zip(delta.iter()) {
                        *sv += ki * dj;
                    }
                }
            }
        }
    }
}

/// One worker's buffers for a head's recurrence, reused across the heads it
/// claims: each is fully overwritten before it is read.
struct HeadScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    kv_mem: Vec<f32>,
    delta: Vec<f32>,
    o: Vec<f32>,
}

impl HeadScratch {
    fn new(d: &DeltaSpec) -> Self {
        let (dk, dv) = (d.head_k_dim, d.head_v_dim);
        Self {
            q: vec![0.0; dk],
            k: vec![0.0; dk],
            v: vec![0.0; dv],
            kv_mem: vec![0.0; dv],
            delta: vec![0.0; dv],
            o: vec![0.0; dv],
        }
    }
}

/// `x / sqrt(sum(x^2) + eps)` in place.
fn l2_normalise(x: &mut [f32]) {
    let sum_sq = x.iter().map(|v| v * v).sum::<f32>();
    let inv = 1.0 / (sum_sq + L2_EPS).sqrt();
    for v in x {
        *v *= inv;
    }
}

/// The raw inputs of the rows of a recorded call.
#[derive(Debug, Default)]
struct RowLog {
    rows: usize,
    mixed: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
}

impl RowLog {
    fn store(&mut self, rows: &Rows<'_>) {
        self.rows = rows.t;
        self.mixed.clear();
        self.mixed.extend_from_slice(rows.mixed);
        self.beta.clear();
        self.beta.extend_from_slice(rows.beta);
        self.decay.clear();
        self.decay.extend_from_slice(rows.decay);
    }

    /// The first `n <= rows` logged rows.
    fn prefix(&self, n: usize, spec: &DeltaSpec) -> Rows<'_> {
        let (cd, nv) = (spec.conv_dim(), spec.n_v_heads);
        Rows {
            t: n,
            mixed: &self.mixed[..n * cd],
            beta: &self.beta[..n * nv],
            decay: &self.decay[..n * nv],
        }
    }
}

/// The live state of one layer with its snapshots.
///
/// The live state exists from construction. The checkpoint and the verify
/// record allocate on first use and keep their allocation, so a model that
/// never speculates pays only the live state and the checkpoint.
#[derive(Debug)]
pub(crate) struct DeltaCell {
    live: DeltaState,
    /// The state before the call recorded in `log`.
    pre: DeltaState,
    log: RowLog,
    /// The state at the checkpoint position.
    checkpoint: DeltaState,
}

impl DeltaCell {
    pub(crate) fn new(spec: &DeltaSpec) -> Self {
        Self {
            live: DeltaState::zeros(spec),
            pre: DeltaState::default(),
            log: RowLog::default(),
            checkpoint: DeltaState::default(),
        }
    }

    pub(crate) fn live_mut(&mut self) -> &mut DeltaState {
        &mut self.live
    }

    /// Copy the live state aside as the checkpoint.
    pub(crate) fn save_checkpoint(&mut self) {
        self.checkpoint.copy_from(&self.live);
    }

    /// Make the checkpoint the live state again.
    pub(crate) fn restore_checkpoint(&mut self) {
        self.live.copy_from(&self.checkpoint);
    }

    /// Back to the sequence-start state.
    pub(crate) fn reset(&mut self) {
        self.live.clear();
    }

    /// Record a call about to scan `rows`: the state before it and the rows.
    pub(crate) fn begin_record(&mut self, rows: &Rows<'_>) {
        self.pre.copy_from(&self.live);
        self.log.store(rows);
    }

    /// Rebuild the state after the first `n` rows of the recorded call:
    /// restore the pre-call copy and re-run the recurrence on those rows.
    ///
    /// `n` must not exceed the recorded rows.
    pub(crate) fn restore_prefix(&mut self, kernel: &DeltaKernel, n: usize) {
        self.live.copy_from(&self.pre);
        if n > 0 {
            let rows = self.log.prefix(n, kernel.spec());
            kernel.scan(&mut self.live, &rows, None);
        }
    }

    /// Floats held by the live state, the checkpoint, the pre-call copy and
    /// the row log (for the memory documentation and tests).
    #[cfg(test)]
    pub(crate) fn held_floats(&self) -> usize {
        let state = |s: &DeltaState| s.conv.len() + s.s.len();
        state(&self.live)
            + state(&self.pre)
            + state(&self.checkpoint)
            + self.log.mixed.len()
            + self.log.beta.len()
            + self.log.decay.len()
    }

    #[cfg(test)]
    pub(crate) fn live(&self) -> &DeltaState {
        &self.live
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tiny::Lcg;
    use super::*;

    /// n_v = 8 over 4 key heads, head widths 16 / 24, 4 taps: the shapes a
    /// real model has (value heads repeat the key heads, `dk != dv`).
    fn spec() -> DeltaSpec {
        DeltaSpec {
            conv_kernel: 4,
            n_k_heads: 4,
            n_v_heads: 8,
            head_k_dim: 16,
            head_v_dim: 24,
        }
    }

    fn kernel(spec: DeltaSpec, seed: u64) -> (DeltaKernel, Vec<f32>) {
        let mut rng = Lcg(seed);
        let (cd, nv) = (spec.conv_dim(), spec.n_v_heads);
        let stored = rng.values(cd * spec.conv_kernel, 0.8);
        let norm = rng
            .values(spec.head_v_dim, 0.3)
            .iter()
            .map(|x| 1.0 + x)
            .collect();
        let a = rng
            .values(nv, 1.0)
            .iter()
            .map(|x| -(x.abs() + 0.05))
            .collect();
        let dt = rng.values(nv, 1.0);
        (DeltaKernel::new(spec, &stored, norm, 1e-6, a, dt), stored)
    }

    pub(crate) struct Inputs {
        pub(crate) mixed: Vec<f32>,
        pub(crate) z: Vec<f32>,
        pub(crate) beta: Vec<f32>,
        pub(crate) decay: Vec<f32>,
    }

    fn inputs(spec: &DeltaSpec, t: usize, seed: u64) -> Inputs {
        let mut rng = Lcg(seed);
        let nv = spec.n_v_heads;
        Inputs {
            mixed: rng.values(t * spec.conv_dim(), 2.0),
            z: rng.values(t * spec.value_dim(), 2.0),
            beta: rng
                .values(t * nv, 1.0)
                .iter()
                .map(|x| 0.5 + 0.5 * x)
                .collect(),
            decay: rng.values(t * nv, 0.4).iter().map(|x| 0.6 + x).collect(),
        }
    }

    fn rows<'a>(spec: &DeltaSpec, i: &'a Inputs, from: usize, to: usize) -> Rows<'a> {
        let (cd, nv) = (spec.conv_dim(), spec.n_v_heads);
        Rows {
            t: to - from,
            mixed: &i.mixed[from * cd..to * cd],
            beta: &i.beta[from * nv..to * nv],
            decay: &i.decay[from * nv..to * nv],
        }
    }

    /// The recurrence written out directly in f64 from the module docs.
    /// Returns the output rows (before `ssm_out`), the final matrices and
    /// the final history.
    pub(crate) fn reference(
        spec: &DeltaSpec,
        kernel: &DeltaKernel,
        stored_conv: &[f32],
        i: &Inputs,
        t: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (cd, nv, dk, dv, taps) = (
            spec.conv_dim(),
            spec.n_v_heads,
            spec.head_k_dim,
            spec.head_v_dim,
            spec.conv_kernel,
        );
        let kd = spec.key_dim();
        let rep = nv / spec.n_k_heads;
        let silu = |x: f64| x / (1.0 + (-x).exp());
        // x_t[c], zero before the start.
        let x = |ti: isize, c: usize| -> f64 {
            if ti < 0 {
                0.0
            } else {
                f64::from(i.mixed[ti as usize * cd + c])
            }
        };
        let conv = |ti: usize, c: usize| -> f64 {
            let sum: f64 = (0..taps)
                .map(|tap| {
                    f64::from(stored_conv[c * taps + tap])
                        * x(ti as isize - (taps as isize - 1) + tap as isize, c)
                })
                .sum();
            silu(sum)
        };
        let l2 = |v: &[f64]| -> Vec<f64> {
            let n: f64 = v.iter().map(|x| x * x).sum();
            let inv = 1.0 / (n + 1e-6).sqrt();
            v.iter().map(|x| x * inv).collect()
        };
        let mut s = vec![0f64; nv * dk * dv];
        let mut out = vec![0f64; t * spec.value_dim()];
        for ti in 0..t {
            for h in 0..nv {
                let hk = h / rep;
                let q: Vec<f64> = (0..dk).map(|j| conv(ti, hk * dk + j)).collect();
                let k: Vec<f64> = (0..dk).map(|j| conv(ti, kd + hk * dk + j)).collect();
                let v: Vec<f64> = (0..dv).map(|j| conv(ti, 2 * kd + h * dv + j)).collect();
                let q: Vec<f64> = l2(&q).iter().map(|x| x / (dk as f64).sqrt()).collect();
                let k = l2(&k);
                let beta = f64::from(i.beta[ti * nv + h]);
                let decay = f64::from(i.decay[ti * nv + h]);
                let m = &mut s[h * dk * dv..(h + 1) * dk * dv];
                for e in m.iter_mut() {
                    *e *= decay;
                }
                let kv_mem: Vec<f64> = (0..dv)
                    .map(|j| (0..dk).map(|r| m[r * dv + j] * k[r]).sum())
                    .collect();
                let delta: Vec<f64> = (0..dv).map(|j| (v[j] - kv_mem[j]) * beta).collect();
                for r in 0..dk {
                    for j in 0..dv {
                        m[r * dv + j] += k[r] * delta[j];
                    }
                }
                let o: Vec<f64> = (0..dv)
                    .map(|j| (0..dk).map(|r| m[r * dv + j] * q[r]).sum())
                    .collect();
                let ms: f64 = o.iter().map(|x| x * x).sum::<f64>() / dv as f64;
                let inv = 1.0 / (ms + f64::from(kernel.eps)).sqrt();
                for j in 0..dv {
                    out[ti * spec.value_dim() + h * dv + j] = o[j]
                        * inv
                        * f64::from(kernel.norm[j])
                        * silu(f64::from(i.z[ti * spec.value_dim() + h * dv + j]));
                }
            }
        }
        let history: Vec<f64> = (0..taps - 1)
            .flat_map(|r| {
                let ti = t as isize - (taps as isize - 1) + r as isize;
                (0..cd).map(move |c| (ti, c))
            })
            .map(|(ti, c)| x(ti, c))
            .collect();
        (out, s, history)
    }

    fn assert_near(got: &[f32], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 2e-4 * w.abs().max(1.0);
            assert!((f64::from(*g) - w).abs() <= tol, "{what}[{i}]: {g} vs {w}");
        }
    }

    #[test]
    fn scan_matches_the_f64_reference() {
        let spec = spec();
        let (kernel, stored) = kernel(spec, 3);
        for t in [1usize, 2, 3, 9] {
            let i = inputs(&spec, t, 10 + t as u64);
            let mut state = DeltaState::zeros(&spec);
            let out = kernel
                .scan(&mut state, &rows(&spec, &i, 0, t), Some(&i.z))
                .unwrap();
            let (want_out, want_s, want_hist) = reference(&spec, &kernel, &stored, &i, t);
            assert_near(&out, &want_out, "output");
            assert_near(state.matrices(), &want_s, "state");
            assert_near(state.history(), &want_hist, "history");
        }
    }

    #[test]
    fn any_split_of_the_rows_is_bit_identical() {
        let spec = spec();
        let (kernel, _) = kernel(spec, 4);
        let t = 11;
        let i = inputs(&spec, t, 5);
        let vd = spec.value_dim();
        let mut whole = DeltaState::zeros(&spec);
        let want = kernel
            .scan(&mut whole, &rows(&spec, &i, 0, t), Some(&i.z))
            .unwrap();
        for cuts in [
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![4],
            vec![2, 9],
            vec![10],
        ] {
            let mut state = DeltaState::zeros(&spec);
            let mut got = Vec::new();
            let mut from = 0;
            for to in cuts.iter().copied().chain([t]) {
                let z = &i.z[from * vd..to * vd];
                got.extend(
                    kernel
                        .scan(&mut state, &rows(&spec, &i, from, to), Some(z))
                        .unwrap(),
                );
                from = to;
            }
            assert_eq!(got, want, "cuts {cuts:?}");
            assert_eq!(state.matrices(), whole.matrices(), "cuts {cuts:?}");
            assert_eq!(state.history(), whole.history(), "cuts {cuts:?}");
        }
    }

    #[test]
    fn pooled_and_serial_scans_are_bit_identical() {
        // Large enough that `scan` itself takes the pool.
        let spec = DeltaSpec {
            conv_kernel: 4,
            n_k_heads: 2,
            n_v_heads: 8,
            head_k_dim: 32,
            head_v_dim: 32,
        };
        let (kernel, _) = kernel(spec, 6);
        let t = 70;
        assert!(t * 8 * 32 * 32 >= SERIAL_BELOW);
        let i = inputs(&spec, t, 7);
        let (mut a, mut b, mut c) = (
            DeltaState::zeros(&spec),
            DeltaState::zeros(&spec),
            DeltaState::zeros(&spec),
        );
        let r = rows(&spec, &i, 0, t);
        let serial = kernel.scan_with(&mut a, &r, Some(&i.z), false).unwrap();
        let pooled = kernel.scan_with(&mut b, &r, Some(&i.z), true).unwrap();
        let auto = kernel.scan(&mut c, &r, Some(&i.z)).unwrap();
        assert_eq!(serial, pooled);
        assert_eq!(serial, auto);
        assert_eq!(a.matrices(), b.matrices());
        assert_eq!(a.matrices(), c.matrices());
    }

    #[test]
    fn state_only_scan_gives_the_same_state_as_a_scan_with_output() {
        let spec = spec();
        let (kernel, _) = kernel(spec, 8);
        let t = 6;
        let i = inputs(&spec, t, 9);
        let (mut with, mut without) = (DeltaState::zeros(&spec), DeltaState::zeros(&spec));
        kernel.scan(&mut with, &rows(&spec, &i, 0, t), Some(&i.z));
        assert!(
            kernel
                .scan(&mut without, &rows(&spec, &i, 0, t), None)
                .is_none()
        );
        assert_eq!(with.matrices(), without.matrices());
        assert_eq!(with.history(), without.history());
    }

    #[test]
    fn a_history_shorter_than_the_taps_slides_correctly() {
        // One row at a time through more rows than taps equals one pass.
        let spec = DeltaSpec {
            conv_kernel: 5,
            ..spec()
        };
        let (kernel, _) = kernel(spec, 12);
        let t = 8;
        let i = inputs(&spec, t, 13);
        let mut whole = DeltaState::zeros(&spec);
        kernel.scan(&mut whole, &rows(&spec, &i, 0, t), None);
        let mut stepped = DeltaState::zeros(&spec);
        for ti in 0..t {
            kernel.scan(&mut stepped, &rows(&spec, &i, ti, ti + 1), None);
        }
        assert_eq!(stepped.history(), whole.history());
        assert_eq!(stepped.matrices(), whole.matrices());
        // The history holds the last four raw rows.
        let cd = spec.conv_dim();
        assert_eq!(whole.history(), &i.mixed[(t - 4) * cd..t * cd]);
    }

    #[test]
    fn a_single_tap_convolution_has_no_history() {
        let spec = DeltaSpec {
            conv_kernel: 1,
            ..spec()
        };
        let (kernel, stored) = kernel(spec, 14);
        let i = inputs(&spec, 3, 15);
        let mut state = DeltaState::zeros(&spec);
        let out = kernel
            .scan(&mut state, &rows(&spec, &i, 0, 3), Some(&i.z))
            .unwrap();
        assert!(state.history().is_empty());
        let (want, _, _) = reference(&spec, &kernel, &stored, &i, 3);
        assert_near(&out, &want, "output");
    }

    #[test]
    fn record_and_restore_rebuild_every_prefix_bit_for_bit() {
        let spec = spec();
        let (kernel, _) = kernel(spec, 16);
        let warmup = inputs(&spec, 5, 17);
        let call = inputs(&spec, 6, 18);
        let mut cell = DeltaCell::new(&spec);
        kernel.scan(cell.live_mut(), &rows(&spec, &warmup, 0, 5), None);

        // The reference states after every prefix of the call.
        let mut expected = Vec::new();
        let mut probe = cell.live().clone();
        for n in 0..=6 {
            let mut s = cell.live().clone();
            kernel.scan(&mut s, &rows(&spec, &call, 0, n), None);
            expected.push(s);
        }
        let r = rows(&spec, &call, 0, 6);
        cell.begin_record(&r);
        kernel.scan(&mut probe, &r, None);
        *cell.live_mut() = probe;
        assert_eq!(cell.live().matrices(), expected[6].matrices());

        // Restoring in any order, including a longer prefix after a shorter
        // one, reproduces the cold states.
        for n in [3usize, 0, 5, 1, 6, 2] {
            cell.restore_prefix(&kernel, n);
            assert_eq!(cell.live().matrices(), expected[n].matrices(), "n {n}");
            assert_eq!(cell.live().history(), expected[n].history(), "n {n}");
        }
    }

    #[test]
    fn checkpoint_round_trips_and_reset_clears() {
        let spec = spec();
        let (kernel, _) = kernel(spec, 19);
        let i = inputs(&spec, 4, 20);
        let mut cell = DeltaCell::new(&spec);
        kernel.scan(cell.live_mut(), &rows(&spec, &i, 0, 2), None);
        let at_two = cell.live().clone();
        cell.save_checkpoint();
        kernel.scan(cell.live_mut(), &rows(&spec, &i, 2, 4), None);
        assert_ne!(cell.live().matrices(), at_two.matrices());
        cell.restore_checkpoint();
        assert_eq!(cell.live().matrices(), at_two.matrices());
        assert_eq!(cell.live().history(), at_two.history());
        cell.reset();
        assert!(cell.live().matrices().iter().all(|&v| v == 0.0));
        assert!(cell.live().history().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gates_follow_sigmoid_and_softplus() {
        let spec = spec();
        let (kernel, _) = kernel(spec, 21);
        let nv = spec.n_v_heads;
        // The softplus threshold branch and a negative input.
        let a: Vec<f32> = (0..2 * nv).map(|i| [-3.0, 0.5, 30.0, 1.0][i % 4]).collect();
        let b: Vec<f32> = (0..2 * nv).map(|i| i as f32 - 4.0).collect();
        let (beta, decay) = kernel.gates(&b, &a);
        for (i, (&be, &de)) in beta.iter().zip(&decay).enumerate() {
            let h = i % nv;
            let x = f64::from(b[i]);
            assert!((f64::from(be) - 1.0 / (1.0 + (-x).exp())).abs() < 1e-6);
            let pre = f64::from(a[i]) + f64::from(kernel.dt_bias[h]);
            let softplus = if pre > 20.0 {
                pre
            } else {
                (1.0 + pre.exp()).ln()
            };
            let want = (f64::from(kernel.a[h]) * softplus).exp();
            assert!((f64::from(de) - want).abs() < 1e-6 * want.max(1e-3), "{i}");
            assert!(
                (0.0..=1.0).contains(&de),
                "ssm_a < 0 keeps the decay in (0, 1]"
            );
        }
    }
}
