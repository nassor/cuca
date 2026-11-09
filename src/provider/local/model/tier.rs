//! Tiered placement of a mixture-of-experts model's routed experts
//! (`ExpertPlacement::Tiered`): the dense weights and the routers stay
//! resident, the experts stay in the GGUF file and are fetched on demand
//! through bounded caches. Placement only decides speed: the experts are
//! built from the same file bytes the resident mode reads, so the output is
//! the resident mode's (bit for bit on the CPU).
//!
//! An expert is the triple `(ffn_gate_exps[e], ffn_up_exps[e],
//! ffn_down_exps[e])` of one layer: three contiguous byte ranges of the three
//! stacked tensors ([`ExpertSlice`]). The tiers, fastest first:
//!
//! - device (GPU devices only, owned by the GPU engine): built device
//!   weights, fed from the RAM tier or directly from disk;
//! - RAM ([`ExpertStore`]): built experts (CPU: quantized matmul weights;
//!   GPU: packed, upload-ready host bytes);
//! - disk: the GGUF file, read with positional reads ([`read_exact_at`])
//!   by a pool of loader threads, so one file handle serves every thread.
//!   On Windows the handle is opened unbuffered (`FILE_FLAG_NO_BUFFERING`, see
//!   [`ExpertFile`]) so the expert sweep does not fill the OS cache; elsewhere
//!   reads stay buffered, because `O_DIRECT` cannot be requested without
//!   `unsafe` code or a new dependency.
//!
//! # Bounds
//!
//! A tier budget in bytes becomes the same number of slots in every MoE
//! layer ([`tier_slots`]): `min(n_expert, budget / per_expert)` where
//! `per_expert` is what one built expert of every MoE layer occupies in that
//! tier. A tier must hold `n_used` experts per layer, else the load fails with
//! [`CucaError::Config`] naming the minimum budget. A tier table never holds
//! more than its slots, and reads in flight count as slots; the expert a pass
//! is computing may outlive its eviction while it is computed (the pass holds
//! its handle), so one expert per MoE layer pass can transiently exceed the
//! RAM table (the GPU device tier states its own transient in `gpu`). The
//! heat table is `4 * n_layer * n_expert` bytes, fixed at load. Outside the
//! budgets, each loader thread keeps three read buffers (one per matrix) that
//! grow to the largest run it read, at most [`MAX_RUN`] matrices each.
//!
//! # Policy
//!
//! - Heat: every routing adds each expert's pick count to its heat
//!   (saturating); after every [`HEAT_DECAY_ROWS`] routed rows of a layer all
//!   its counts halve, so the heat follows the recent traffic.
//! - Eviction: the victim is the resident expert not needed later in the
//!   current layer pass with the lowest `(heat, last use)`, ties to the lower
//!   expert id; if every resident expert is needed later, the one needed
//!   farthest ahead (the largest id, since a pass computes its experts in
//!   ascending order), and only when it is needed after the expert being
//!   fetched. A prefill that scans many cold experts evicts them among
//!   themselves and keeps the hot ones.
//! - Pass ([`Pass`]): the routing gives the ascending list of experts a
//!   layer needs. Missing experts are read in that order while the table plus
//!   the reads in flight fit the slots (at most twice the loader threads in
//!   flight; with no RAM tier, at most the loader threads); compute walks the
//!   list in the same ascending order and waits
//!   for each expert's read, so the accumulation order, and with it the
//!   arithmetic, is the resident mode's. Missing experts that are adjacent
//!   in the file (same layer, consecutive ids, matrices stored one expert
//!   after the other) are read as one run of at most [`MAX_RUN`] experts, one
//!   read per matrix: a longer read takes the drive nearer its sequential
//!   rate. Each expert of a run is built, counted and slotted on its own, so
//!   the bytes, the budgets and the output are the single-expert reads'.
//! - Warm start: heat exported from an earlier session
//!   ([`TierShared::heat`], `LocalModel::expert_heat`) seeds the counters and
//!   loads the hottest experts of every layer at startup.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

use crate::provider::local::gguf::{self, GgufType};

use super::err;
use super::layout::tensor_info;
use super::moe::Routing;
use crate::error::CucaError;

/// Routed rows of one layer after which its heat counts halve.
pub(crate) const HEAT_DECAY_ROWS: u32 = 4096;

/// Fill `buf` with the bytes of a source starting at `offset`, calling
/// `read_some(rest, at)` (one positional read of at most `rest.len()` bytes at
/// source offset `at`, returning the count read) until the buffer is full.
///
/// Operating systems return short reads (Windows `ReadFile` caps one call at
/// `u32::MAX` bytes, any platform may stop early), so the loop advances the
/// buffer slice and the source offset together by the bytes actually read.
///
/// # Errors
///
/// The I/O error of `read_some` (a retry on [`std::io::ErrorKind::Interrupted`]),
/// or [`std::io::ErrorKind::UnexpectedEof`] when `read_some` returns 0 bytes.
fn read_fully_at(
    buf: &mut [u8],
    offset: u64,
    mut read_some: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
) -> std::io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        match read_some(&mut buf[done..], offset + done as u64) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// One positional read of `file` at `offset` into `buf`: the number of bytes
/// read, possibly fewer than `buf.len()`.
fn read_some_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positional file reads are not available on this platform",
        ))
    }
}

/// Read exactly `buf.len()` bytes of `file` at `offset`, without relying on
/// the handle's cursor: one handle serves every loader thread. (On Windows
/// `seek_read` does move the cursor, so a handle used with this function must
/// never also be used with `Read`/`Seek`.)
///
/// # Errors
///
/// The I/O error; [`std::io::ErrorKind::UnexpectedEof`] when the file ends first.
pub(crate) fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    read_fully_at(buf, offset, |rest, at| read_some_at(file, rest, at))
}

/// Alignment of unbuffered reads: buffer address, file offset and length must
/// be multiples of the volume's sector size (512 or 4096 bytes).
const DIRECT_ALIGN: usize = 4096;

/// `FILE_FLAG_NO_BUFFERING` of `CreateFile`.
#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

/// A reusable buffer for [`ExpertFile::read`]: the bytes asked for are the
/// slice [`ReadBuf::bytes`], somewhere inside the allocation (an unbuffered
/// read needs an aligned start and an aligned length).
#[derive(Default)]
struct ReadBuf {
    raw: Vec<u8>,
    from: usize,
    len: usize,
}

impl ReadBuf {
    fn bytes(&self) -> &[u8] {
        &self.raw[self.from..self.from + self.len]
    }
}

/// A file of the GGUF set read for experts: with the OS cache bypassed where
/// the platform allows it from safe code (Windows: `FILE_FLAG_NO_BUFFERING`),
/// else buffered.
///
/// Experts stream through the RAM tier, which is the cache that matters:
/// buffered reads would duplicate every expert in the OS cache (hundreds of
/// gigabytes of mostly cold experts that evict the engine's own working set
/// and fill Windows' standby list). The unbuffered path reads the sector
/// aligned superset of the range (at most one alignment unit of waste at each
/// end) into an over-allocated buffer, whose aligned window is found from the
/// buffer's address (no `unsafe`). Unix would need `O_DIRECT`, whose open flag
/// is not exposed by the standard library (`libc` would be a new dependency),
/// so there reads stay buffered, as they do where the platform refuses the
/// flag (for example a network share).
struct ExpertFile {
    file: File,
    direct: bool,
}

impl ExpertFile {
    /// Open `path`, unbuffered when the platform allows it.
    fn open(path: &Path) -> std::io::Result<Self> {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let direct = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(FILE_FLAG_NO_BUFFERING)
                .open(path);
            if let Ok(file) = direct {
                return Ok(Self { file, direct: true });
            }
        }
        Ok(Self {
            file: File::open(path)?,
            direct: false,
        })
    }

    /// Read the `len` bytes at `offset` into `buf`.
    ///
    /// # Errors
    ///
    /// The I/O error; [`std::io::ErrorKind::UnexpectedEof`] when the file ends first.
    fn read(&self, offset: u64, len: usize, buf: &mut ReadBuf) -> std::io::Result<()> {
        if !self.direct {
            // Grow only: a smaller read reuses the larger allocation, and the
            // old bytes need no copy when it grows.
            if buf.raw.len() < len {
                buf.raw.clear();
                buf.raw.resize(len, 0);
            }
            buf.from = 0;
            buf.len = len;
            return read_exact_at(&self.file, offset, &mut buf.raw[..len]);
        }
        let head = (offset % DIRECT_ALIGN as u64) as usize;
        let span = (head + len).next_multiple_of(DIRECT_ALIGN);
        if buf.raw.len() < span + DIRECT_ALIGN {
            buf.raw.clear();
            buf.raw.resize(span + DIRECT_ALIGN, 0);
        }
        let from = buf.raw.as_ptr().align_offset(DIRECT_ALIGN);
        let window = buf
            .raw
            .get_mut(from..from + span)
            .ok_or_else(|| std::io::Error::other("no aligned window in the read buffer"))?;
        // The file may end inside the last unit: accept a short read once the
        // range is covered (whole units are asked for, as the flag requires).
        let need = head + len;
        let at = offset - head as u64;
        let mut done = 0;
        while done < need {
            match read_some_at(&self.file, &mut window[done..], at + done as u64) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    done += n;
                    // Only the end of the file ends a read inside a unit, and
                    // the next read could not be unit aligned anyway.
                    if done < need && !done.is_multiple_of(DIRECT_ALIGN) {
                        return Err(std::io::ErrorKind::UnexpectedEof.into());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        buf.from = from + head;
        buf.len = len;
        Ok(())
    }
}

/// Where one expert's `[rows, cols]` matrix of a stacked `[n_expert, rows,
/// cols]` tensor lives in the file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertSlice {
    pub(crate) dtype: GgufType,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// The file of the set that holds the tensor.
    file: usize,
    /// File offset of expert 0.
    start: u64,
    /// Bytes of one expert's matrix in the file.
    pub(crate) file_bytes: usize,
    /// Bytes from expert `e`'s matrix to expert `e + 1`'s: `file_bytes`, or
    /// twice that for the gate or the up half of a fused gate/up tensor.
    pub(crate) stride: usize,
}

impl ExpertSlice {
    /// The expert layout of `tensor_name`, which must have the shape
    /// `[n_expert, rows, cols]` and whole blocks per row.
    pub(crate) fn read(
        content: &gguf::Content,
        tensor_name: &str,
        n_expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CucaError> {
        let info = tensor_info(content, tensor_name)?;
        if info.shape.dims() != [n_expert, rows, cols] {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {:?}, found {:?}",
                [n_expert, rows, cols],
                info.shape.dims()
            )));
        }
        let dtype = info.dtype;
        let Some(row_bytes) = dtype.byte_len(cols) else {
            return Err(err(format!(
                "tensor `{tensor_name}`: row length {cols} is not a whole number of {} blocks",
                dtype.name()
            )));
        };
        let (_, start) = content.location(info);
        Ok(Self {
            dtype,
            rows,
            cols,
            file: info.file,
            start,
            file_bytes: rows * row_bytes,
            stride: rows * row_bytes,
        })
    }

    /// The gate and up halves of the fused expert tensor `tensor_name`, of
    /// shape `[n_expert, 2 * ff, cols]`: for every expert the first `ff` rows
    /// are the gate matrix and the next `ff` rows the up matrix.
    pub(crate) fn read_fused(
        content: &gguf::Content,
        tensor_name: &str,
        n_expert: usize,
        ff: usize,
        cols: usize,
    ) -> Result<(Self, Self), CucaError> {
        let both = Self::read(content, tensor_name, n_expert, 2 * ff, cols)?;
        let half = both.file_bytes / 2;
        let gate = Self {
            rows: ff,
            file_bytes: half,
            stride: both.file_bytes,
            ..both
        };
        let up = Self {
            start: both.start + half as u64,
            ..gate
        };
        Ok((gate, up))
    }

    /// The candle type of the experts' blocks.
    ///
    /// # Errors
    ///
    /// A type candle's block layouts do not cover (MXFP4 has its own kernels,
    /// not a candle type).
    pub(crate) fn ggml(&self) -> Result<candle_core::quantized::GgmlDType, CucaError> {
        self.dtype.ggml().ok_or_else(|| {
            err(format!(
                "experts of type {} cannot run on this engine",
                self.dtype.name()
            ))
        })
    }

    /// The file of the set that holds the experts.
    pub(crate) fn file(&self) -> usize {
        self.file
    }

    /// File offset of expert `e`'s matrix.
    pub(crate) fn offset(&self, e: usize) -> u64 {
        self.start + e as u64 * self.stride as u64
    }
}

/// The experts of one MoE layer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerExperts {
    pub(crate) il: usize,
    pub(crate) gate: ExpertSlice,
    pub(crate) up: ExpertSlice,
    pub(crate) down: ExpertSlice,
    /// Bytes one built expert of this layer occupies in the RAM tier.
    pub(crate) stored_bytes: usize,
}

/// The most adjacent experts one loader job reads with a single read per
/// matrix. A longer read takes the drive nearer its sequential rate, and a
/// loader thread holds up to three read buffers of this many matrices.
const MAX_RUN: usize = 4;

impl LayerExperts {
    /// Whether the matrices of adjacent experts are adjacent in the file, so
    /// that a run of experts is one range per matrix (a fused gate/up tensor
    /// interleaves its halves and has no such range).
    fn contiguous(&self) -> bool {
        [&self.gate, &self.up, &self.down]
            .iter()
            .all(|s| s.stride == s.file_bytes)
    }
}

/// Slots per layer a tier of `budget` bytes holds when one expert of every
/// MoE layer occupies `per_expert` bytes.
///
/// # Errors
///
/// [`CucaError::Config`] when the tier holds fewer than `n_used` experts per
/// layer, naming the minimum budget.
pub(crate) fn tier_slots(
    tier: &str,
    budget: u64,
    per_expert: usize,
    n_expert: usize,
    n_used: usize,
) -> Result<usize, CucaError> {
    let per_expert = per_expert.max(1) as u64;
    let slots = usize::try_from(budget / per_expert)
        .unwrap_or(usize::MAX)
        .min(n_expert);
    if slots < n_used {
        return Err(CucaError::Config(format!(
            "expert {tier} tier of {budget} bytes holds {slots} experts per layer; this model needs at least {n_used} ({} bytes)",
            n_used as u64 * per_expert
        )));
    }
    Ok(slots)
}

/// Learned expert heat: one count per `(layer, expert)`, exported by
/// `LocalModel::expert_heat` and fed back through `ExpertTiers::warm_from`,
/// which loads the hottest experts of every layer at startup. It serializes
/// with serde (for example to JSON) so it can outlive the process.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "HeatRepr")]
pub struct ExpertHeat {
    n_layer: usize,
    n_expert: usize,
    /// Row-major `[layer][expert]`.
    counts: Vec<u32>,
}

/// The unchecked serialized form of [`ExpertHeat`].
#[derive(serde::Deserialize)]
struct HeatRepr {
    n_layer: usize,
    n_expert: usize,
    counts: Vec<u32>,
}

impl TryFrom<HeatRepr> for ExpertHeat {
    type Error = String;

    fn try_from(r: HeatRepr) -> Result<Self, Self::Error> {
        if r.n_layer.checked_mul(r.n_expert) != Some(r.counts.len()) {
            return Err(format!(
                "{} counts do not cover {} layers of {} experts",
                r.counts.len(),
                r.n_layer,
                r.n_expert
            ));
        }
        Ok(Self {
            n_layer: r.n_layer,
            n_expert: r.n_expert,
            counts: r.counts,
        })
    }
}

impl ExpertHeat {
    /// Layers of the model the heat was learned on (every layer, MoE or not).
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// Experts per layer.
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// The heat of `expert` in `layer`; `None` outside the shape.
    pub fn count(&self, layer: usize, expert: usize) -> Option<u32> {
        if layer >= self.n_layer || expert >= self.n_expert {
            return None;
        }
        self.counts.get(layer * self.n_expert + expert).copied()
    }

    /// The experts of `layer` with a heat above zero, hottest first (ties
    /// to the lower id).
    pub(crate) fn ranked(&self, layer: usize) -> Vec<u32> {
        let row = &self.counts[layer * self.n_expert..(layer + 1) * self.n_expert];
        let mut ids: Vec<u32> = (0..self.n_expert as u32)
            .filter(|&e| row[e as usize] > 0)
            .collect();
        ids.sort_by(|&a, &b| row[b as usize].cmp(&row[a as usize]).then(a.cmp(&b)));
        ids
    }
}

/// Expert tier counters (`LocalModel::expert_tier_stats`), cumulative since
/// load except the two resident gauges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertTierStats {
    /// Expert computations the routers asked for (one per expert per layer
    /// pass).
    pub lookups: u64,
    /// Lookups served by the device tier (GPU devices only).
    pub device_hits: u64,
    /// Lookups served by the RAM tier.
    pub ram_hits: u64,
    /// Experts read from the GGUF file (warm-start loads included).
    pub disk_reads: u64,
    /// Bytes read from the GGUF file for experts.
    pub bytes_read: u64,
    /// Bytes of experts the RAM tier holds now.
    pub ram_resident_bytes: u64,
    /// Bytes of experts the device tier holds now.
    pub device_resident_bytes: u64,
}

impl ExpertTierStats {
    /// `(device_hits + ram_hits) / lookups`; `None` before the first lookup.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.lookups > 0).then(|| (self.device_hits + self.ram_hits) as f64 / self.lookups as f64)
    }
}

/// Heat of one layer.
struct LayerHeat {
    counts: Vec<u32>,
    rows_since_decay: u32,
}

/// Counters and heat shared by the engine thread (which writes them) and the
/// `LocalModel` handle (which reads them).
pub(crate) struct TierShared {
    pub(crate) lookups: AtomicU64,
    pub(crate) device_hits: AtomicU64,
    pub(crate) ram_hits: AtomicU64,
    pub(crate) disk_reads: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) ram_resident_bytes: AtomicU64,
    pub(crate) device_resident_bytes: AtomicU64,
    n_layer: usize,
    n_expert: usize,
    heat: Vec<Mutex<LayerHeat>>,
}

/// `counter += n` (relaxed: the counters are statistics).
pub(crate) fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

/// `counter -= n` (relaxed).
pub(crate) fn sub(counter: &AtomicU64, n: u64) {
    counter.fetch_sub(n, Ordering::Relaxed);
}

impl TierShared {
    pub(crate) fn new(n_layer: usize, n_expert: usize) -> Self {
        Self {
            lookups: AtomicU64::new(0),
            device_hits: AtomicU64::new(0),
            ram_hits: AtomicU64::new(0),
            disk_reads: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            ram_resident_bytes: AtomicU64::new(0),
            device_resident_bytes: AtomicU64::new(0),
            n_layer,
            n_expert,
            heat: (0..n_layer)
                .map(|_| {
                    Mutex::new(LayerHeat {
                        counts: vec![0; n_expert],
                        rows_since_decay: 0,
                    })
                })
                .collect(),
        }
    }

    fn layer(&self, il: usize) -> std::sync::MutexGuard<'_, LayerHeat> {
        self.heat[il].lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Learn from one routing of `t` rows in layer `il`.
    pub(crate) fn record(&self, il: usize, routing: &Routing, t: usize) {
        let mut heat = self.layer(il);
        for (count, picks) in heat.counts.iter_mut().zip(&routing.per_expert) {
            *count = count.saturating_add(u32::try_from(picks.len()).unwrap_or(u32::MAX));
        }
        heat.rows_since_decay = heat
            .rows_since_decay
            .saturating_add(u32::try_from(t).unwrap_or(u32::MAX));
        if heat.rows_since_decay >= HEAT_DECAY_ROWS {
            heat.counts.iter_mut().for_each(|c| *c >>= 1);
            heat.rows_since_decay = 0;
        }
    }

    /// The heat of layer `il` now.
    pub(crate) fn heat_snapshot(&self, il: usize) -> Vec<u32> {
        self.layer(il).counts.clone()
    }

    /// Check that `heat` was learned on a model of this shape.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] naming both shapes.
    pub(crate) fn check(&self, heat: &ExpertHeat) -> Result<(), CucaError> {
        if (heat.n_layer, heat.n_expert) != (self.n_layer, self.n_expert) {
            return Err(CucaError::Config(format!(
                "experts.warm_from is heat for {}x{} experts; this model has {}x{}",
                heat.n_layer, heat.n_expert, self.n_layer, self.n_expert
            )));
        }
        Ok(())
    }

    /// Continue learning from `heat` (shape-checked by [`TierShared::check`]).
    pub(crate) fn seed(&self, heat: &ExpertHeat) {
        for il in 0..self.n_layer.min(heat.n_layer) {
            let row = &heat.counts[il * heat.n_expert..(il + 1) * heat.n_expert];
            let mut layer = self.layer(il);
            for (dst, &src) in layer.counts.iter_mut().zip(row) {
                *dst = src;
            }
        }
    }

    /// The heat learned so far.
    pub(crate) fn heat(&self) -> ExpertHeat {
        let mut counts = Vec::with_capacity(self.n_layer * self.n_expert);
        for il in 0..self.n_layer {
            counts.extend_from_slice(&self.layer(il).counts);
        }
        ExpertHeat {
            n_layer: self.n_layer,
            n_expert: self.n_expert,
            counts,
        }
    }

    pub(crate) fn stats(&self) -> ExpertTierStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        ExpertTierStats {
            lookups: get(&self.lookups),
            device_hits: get(&self.device_hits),
            ram_hits: get(&self.ram_hits),
            disk_reads: get(&self.disk_reads),
            bytes_read: get(&self.bytes_read),
            ram_resident_bytes: get(&self.ram_resident_bytes),
            device_resident_bytes: get(&self.device_resident_bytes),
        }
    }
}

struct Entry<T> {
    value: Arc<T>,
    last_use: u64,
}

/// The resident experts of one layer in one tier; the owner keeps it within
/// its slots.
pub(crate) struct SlotTable<T> {
    entries: HashMap<u32, Entry<T>>,
    tick: u64,
}

impl<T> SlotTable<T> {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(slots),
            tick: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn contains(&self, e: u32) -> bool {
        self.entries.contains_key(&e)
    }

    /// The resident expert `e`, marked used.
    pub(crate) fn get(&mut self, e: u32) -> Option<Arc<T>> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(&e).map(|entry| {
            entry.last_use = tick;
            Arc::clone(&entry.value)
        })
    }

    /// Make `e` resident; the caller made room.
    pub(crate) fn insert(&mut self, e: u32, value: Arc<T>) {
        self.tick += 1;
        self.entries.insert(
            e,
            Entry {
                value,
                last_use: self.tick,
            },
        );
    }

    pub(crate) fn remove(&mut self, e: u32) -> Option<Arc<T>> {
        self.entries.remove(&e).map(|entry| entry.value)
    }

    /// The expert to evict given the layer's `heat` and the ascending experts
    /// the current pass still needs (see the module docs); `None` when the
    /// table is empty.
    pub(crate) fn victim(&self, heat: &[u32], needed_later: &[u32]) -> Option<u32> {
        let cold = self
            .entries
            .iter()
            .filter(|(e, _)| needed_later.binary_search(e).is_err())
            .min_by_key(|&(&e, entry)| {
                (
                    heat.get(e as usize).copied().unwrap_or(0),
                    entry.last_use,
                    e,
                )
            })
            .map(|(&e, _)| e);
        cold.or_else(|| self.entries.keys().copied().max())
    }
}

/// What a loader thread builds one expert from: the layer and the three
/// matrices' file bytes.
pub(crate) type BuildFn<T> =
    dyn Fn(&LayerExperts, &[u8], &[u8], &[u8]) -> Result<T, CucaError> + Send + Sync;

/// A loader's reply: the built expert and the file bytes read.
type Reply<T> = Result<(Arc<T>, usize), CucaError>;

/// A run of adjacent experts being fetched: the positions of its experts in
/// the caller's list and their reply channels.
type Run<'a, T> = (&'a [usize], Vec<Receiver<Reply<T>>>);

/// The experts `first..first + replies.len()` of `layer`, read together (one
/// expert unless the layer is [`LayerExperts::contiguous`]).
struct Job<T> {
    layer: usize,
    first: u32,
    /// One channel per expert of the run, in order.
    replies: Vec<Sender<Reply<T>>>,
}

/// What every loader thread reads.
struct LoaderCtx<T> {
    /// The files of the GGUF set, indexed like [`ExpertSlice::file`].
    files: Vec<ExpertFile>,
    layers: Vec<Option<LayerExperts>>,
    build: Arc<BuildFn<T>>,
    /// Reads issued, one per matrix of a job (tests).
    #[cfg(test)]
    reads: AtomicU64,
}

impl<T> LoaderCtx<T> {
    /// Read the experts `first..first + replies.len()` of `job_layer` with
    /// one read per matrix, build each and send it down its own channel as
    /// soon as it is built, so a pass computes the first while the rest are
    /// built. A run of more than one expert needs a contiguous layer. A failed
    /// read fails every expert of the run.
    fn load_run(
        &self,
        job_layer: usize,
        first: u32,
        replies: &[Sender<Reply<T>>],
        bufs: &mut [ReadBuf; 3],
    ) {
        let Some(layer) = self.layers.get(job_layer).and_then(Option::as_ref) else {
            for reply in replies {
                let _ = reply.send(Err(err(format!("layer {job_layer} has no experts"))));
            }
            return;
        };
        debug_assert!(replies.len() == 1 || layer.contiguous());
        for (buf, slice) in bufs.iter_mut().zip([&layer.gate, &layer.up, &layer.down]) {
            #[cfg(test)]
            self.reads.fetch_add(1, Ordering::Relaxed);
            let at = slice.offset(first as usize);
            if let Err(io) = self.files[slice.file].read(at, replies.len() * slice.file_bytes, buf)
            {
                for (i, reply) in replies.iter().enumerate() {
                    let _ = reply.send(Err(err(format!(
                        "layer {} expert {}: {io} (truncated GGUF?)",
                        layer.il,
                        first as usize + i
                    ))));
                }
                return;
            }
        }
        let read = layer.gate.file_bytes + layer.up.file_bytes + layer.down.file_bytes;
        for (i, reply) in replies.iter().enumerate() {
            let range = |slice: &ExpertSlice| i * slice.file_bytes..(i + 1) * slice.file_bytes;
            let built = (self.build)(
                layer,
                &bufs[0].bytes()[range(&layer.gate)],
                &bufs[1].bytes()[range(&layer.up)],
                &bufs[2].bytes()[range(&layer.down)],
            );
            // The requester may have given up (a failed pass).
            let _ = reply.send(built.map(|value| (Arc::new(value), read)));
        }
    }
}

/// A pool of threads reading and building experts.
struct Loader<T> {
    jobs: Option<Sender<Job<T>>>,
    threads: Vec<JoinHandle<()>>,
}

impl<T: Send + Sync + 'static> Loader<T> {
    fn spawn(ctx: Arc<LoaderCtx<T>>, n: usize) -> Result<Self, CucaError> {
        let (jobs, queue) = mpsc::channel::<Job<T>>();
        let queue = Arc::new(Mutex::new(queue));
        let mut loader = Self {
            jobs: Some(jobs),
            threads: Vec::with_capacity(n),
        };
        for i in 0..n {
            let ctx = Arc::clone(&ctx);
            let queue = Arc::clone(&queue);
            let handle = std::thread::Builder::new()
                .name(format!("cuca-expert-loader-{i}"))
                .spawn(move || {
                    let mut bufs: [ReadBuf; 3] = Default::default();
                    loop {
                        let job = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
                        let Ok(job) = job else { return };
                        ctx.load_run(job.layer, job.first, &job.replies, &mut bufs);
                    }
                })
                .map_err(|e| err(format!("cannot start an expert loader thread: {e}")))?;
            loader.threads.push(handle);
        }
        Ok(loader)
    }

    /// Ask for the `n` adjacent experts from `first` of `layer` as one job:
    /// one receiver per expert, in order.
    fn request(
        &self,
        layer: usize,
        first: u32,
        n: usize,
    ) -> Result<Vec<Receiver<Reply<T>>>, CucaError> {
        let (replies, receivers): (Vec<_>, Vec<_>) = (0..n).map(|_| mpsc::channel()).unzip();
        self.jobs
            .as_ref()
            .ok_or_else(|| err("expert loader thread stopped"))?
            .send(Job {
                layer,
                first,
                replies,
            })
            .map_err(|_| err("expert loader thread stopped"))?;
        Ok(receivers)
    }
}

impl<T> Drop for Loader<T> {
    fn drop(&mut self) {
        self.jobs = None;
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Wait for one loader reply.
fn receive<T>(rx: &Receiver<Reply<T>>) -> Reply<T> {
    rx.recv()
        .map_err(|_| err("expert loader thread stopped"))
        .and_then(|r| r)
}

/// The RAM tier and the disk reader of a tiered model; see the module docs.
pub(crate) struct ExpertStore<T> {
    ctx: Arc<LoaderCtx<T>>,
    /// Slots per layer of the RAM tier; 0 = no RAM tier (GPU devices only).
    ram_slots: usize,
    /// By layer; the tables of layers without experts stay empty.
    ram: Vec<Mutex<SlotTable<T>>>,
    loader_threads: usize,
    loader: Loader<T>,
    shared: Arc<TierShared>,
}

impl<T: Send + Sync + 'static> ExpertStore<T> {
    /// Open the `files` of the GGUF set for the experts of `layers` (every MoE layer
    /// of a model with `n_layer` layers) behind a RAM tier of `ram_bytes`,
    /// with `loader_threads` readers that build each expert with `build`.
    /// `allow_no_ram` accepts `ram_bytes == 0` (no RAM tier).
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when the RAM tier holds fewer than `n_used`
    /// experts per layer; [`CucaError::Io`] when the file cannot be opened.
    #[expect(
        clippy::too_many_arguments,
        reason = "the file, the model shape, the budget and the loader are independent inputs"
    )]
    pub(crate) fn open(
        files: &[&Path],
        n_layer: usize,
        layers: Vec<LayerExperts>,
        n_expert: usize,
        n_used: usize,
        ram_bytes: u64,
        allow_no_ram: bool,
        loader_threads: usize,
        build: Arc<BuildFn<T>>,
        shared: Arc<TierShared>,
    ) -> Result<Self, CucaError> {
        let per_expert: usize = layers.iter().map(|l| l.stored_bytes).sum();
        let ram_slots = if ram_bytes == 0 && allow_no_ram {
            0
        } else {
            tier_slots("RAM", ram_bytes, per_expert, n_expert, n_used)?
        };
        let mut by_layer: Vec<Option<LayerExperts>> = vec![None; n_layer];
        for l in layers {
            if let Some(slot) = by_layer.get_mut(l.il) {
                *slot = Some(l);
            }
        }
        let ctx = Arc::new(LoaderCtx {
            files: files
                .iter()
                .map(|path| ExpertFile::open(path))
                .collect::<Result<_, _>>()?,
            layers: by_layer,
            build,
            #[cfg(test)]
            reads: AtomicU64::new(0),
        });
        let loader_threads = loader_threads.max(1);
        Ok(Self {
            loader: Loader::spawn(Arc::clone(&ctx), loader_threads)?,
            ctx,
            ram_slots,
            ram: (0..n_layer)
                .map(|_| Mutex::new(SlotTable::new(ram_slots)))
                .collect(),
            loader_threads,
            shared,
        })
    }

    pub(crate) fn shared(&self) -> &Arc<TierShared> {
        &self.shared
    }

    pub(crate) fn ram_slots(&self) -> usize {
        self.ram_slots
    }

    /// The experts of layer `il`, if it has any.
    pub(crate) fn layer(&self, il: usize) -> Option<&LayerExperts> {
        self.ctx.layers.get(il).and_then(Option::as_ref)
    }

    fn stored_bytes(&self, il: usize) -> u64 {
        self.layer(il).map_or(0, |l| l.stored_bytes as u64)
    }

    fn table(&self, il: usize) -> std::sync::MutexGuard<'_, SlotTable<T>> {
        self.ram[il].lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Count a finished read.
    fn count_read(&self, bytes: usize) {
        add(&self.shared.disk_reads, 1);
        add(&self.shared.bytes_read, bytes as u64);
    }

    /// Read and build `experts` of layer `il` in parallel, without caching
    /// them (the GPU device tier's warm start).
    ///
    /// # Errors
    ///
    /// The first read or build error.
    pub(crate) fn fetch(&self, il: usize, experts: &[u32]) -> Result<Vec<Arc<T>>, CucaError> {
        // Adjacent ids are read as one run; the answers come back in the
        // order asked.
        let mut order: Vec<usize> = (0..experts.len()).collect();
        order.sort_unstable_by_key(|&i| experts[i]);
        let max_run = self.max_run(il);
        let mut out: Vec<Option<Arc<T>>> = vec![None; experts.len()];
        let mut batch: Vec<Run<'_, T>> = Vec::new();
        let (mut queued, mut at) = (0, 0);
        while at < order.len() {
            let mut n = 1;
            while n < max_run
                && at + n < order.len()
                && experts[order[at + n - 1]].checked_add(1) == Some(experts[order[at + n]])
            {
                n += 1;
            }
            let run = &order[at..at + n];
            batch.push((run, self.loader.request(il, experts[run[0]], n)?));
            queued += n;
            at += n;
            if queued >= 2 * self.loader_threads || at == order.len() {
                for (run, receivers) in batch.drain(..) {
                    for (&i, rx) in run.iter().zip(&receivers) {
                        let (value, bytes) = receive(rx)?;
                        self.count_read(bytes);
                        out[i] = Some(value);
                    }
                }
                queued = 0;
            }
        }
        out.into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| err("an expert was not loaded"))
    }

    /// The most adjacent experts of layer `il` one job reads together: 1
    /// unless the layer's matrices are contiguous (see [`MAX_RUN`]).
    fn max_run(&self, il: usize) -> usize {
        if self.layer(il).is_some_and(LayerExperts::contiguous) {
            MAX_RUN
        } else {
            1
        }
    }

    /// Load into the RAM tier, per MoE layer, the hottest experts of `heat`
    /// after skipping the `skip` hottest (those a device tier takes), at
    /// most `take` and at most the RAM slots.
    ///
    /// # Errors
    ///
    /// The first read or build error.
    pub(crate) fn warm(
        &self,
        heat: &ExpertHeat,
        skip: usize,
        take: usize,
    ) -> Result<(), CucaError> {
        let take = take.min(self.ram_slots);
        if take == 0 {
            return Ok(());
        }
        for il in 0..self.ram.len().min(heat.n_layer) {
            if self.layer(il).is_none() {
                continue;
            }
            let ids: Vec<u32> = heat.ranked(il).into_iter().skip(skip).take(take).collect();
            let built = self.fetch(il, &ids)?;
            let mut table = self.table(il);
            for (e, value) in ids.into_iter().zip(built) {
                table.insert(e, value);
                add(&self.shared.ram_resident_bytes, self.stored_bytes(il));
            }
        }
        Ok(())
    }

    /// Start the pass of layer `il` over the ascending experts `needed`.
    pub(crate) fn pass<'a>(&'a self, il: usize, needed: &[u32]) -> Pass<'a, T> {
        let pending = if self.ram_slots == 0 {
            needed.to_vec()
        } else {
            let table = self.table(il);
            needed
                .iter()
                .copied()
                .filter(|&e| !table.contains(e))
                .collect()
        };
        let mut pass = Pass {
            store: self,
            il,
            needed: needed.to_vec(),
            cursor: 0,
            pending,
            in_flight: Vec::new(),
            failed: None,
        };
        pass.issue();
        pass
    }
}

/// One layer's walk over the experts it needs; see the module docs.
pub(crate) struct Pass<'a, T> {
    store: &'a ExpertStore<T>,
    il: usize,
    /// Ascending.
    needed: Vec<u32>,
    /// Index into `needed` of the next expert to compute.
    cursor: usize,
    /// Missing experts not yet requested, ascending.
    pending: Vec<u32>,
    /// Requested experts with their reply channels, in request order.
    in_flight: Vec<(u32, Receiver<Reply<T>>)>,
    /// A request that could not be sent.
    failed: Option<CucaError>,
}

impl<T: Send + Sync + 'static> Pass<'_, T> {
    /// Request missing experts while the slots allow it (see the module
    /// docs), evicting for room. Adjacent ids are requested as one job (see
    /// [`MAX_RUN`]); each expert of a run still takes its own slot.
    fn issue(&mut self) {
        let store = self.store;
        let max_in_flight = 2 * store.loader_threads;
        let max_run = store.max_run(self.il);
        if store.ram_slots == 0 {
            let cap = store.loader_threads.min(max_in_flight);
            while self.in_flight.len() < cap && !self.pending.is_empty() {
                let n = self.leading_run(max_run.min(cap - self.in_flight.len()));
                self.request_run(n);
            }
            return;
        }
        if self.pending.is_empty() {
            return;
        }
        let mut table = store.table(self.il);
        let mut heat: Option<Vec<u32>> = None;
        loop {
            // Make room, expert by expert, for the run at the head of
            // `pending`.
            let mut run = 0;
            while run < max_run && run < self.pending.len() {
                let next = self.pending[run];
                if run > 0 && self.pending[run - 1].checked_add(1) != Some(next) {
                    break;
                }
                if self.in_flight.len() + run >= max_in_flight {
                    break;
                }
                if table.len() + self.in_flight.len() + run >= store.ram_slots {
                    let heat = heat.get_or_insert_with(|| store.shared.heat_snapshot(self.il));
                    let later = &self.needed[self.cursor..];
                    let Some(victim) = table.victim(heat, later) else {
                        break;
                    };
                    let victim_needed = later.binary_search(&victim).is_ok();
                    if victim_needed && victim < next {
                        // Every resident expert is needed before the one to
                        // fetch: wait for compute to move past them.
                        break;
                    }
                    table.remove(victim);
                    sub(
                        &store.shared.ram_resident_bytes,
                        store.stored_bytes(self.il),
                    );
                    if victim_needed {
                        let at = self.pending.partition_point(|&p| p < victim);
                        self.pending.insert(at, victim);
                    }
                }
                run += 1;
            }
            if run == 0 {
                break;
            }
            self.request_run(run);
        }
    }

    /// The length of the run of adjacent ids at the head of `pending`, at
    /// most `most`.
    fn leading_run(&self, most: usize) -> usize {
        let mut n = 1;
        while n < most
            && n < self.pending.len()
            && self.pending[n - 1].checked_add(1) == Some(self.pending[n])
        {
            n += 1;
        }
        n
    }

    /// Request the first `n` experts of `pending` as one job.
    fn request_run(&mut self, n: usize) {
        let experts: Vec<u32> = self.pending.drain(..n).collect();
        match self.store.loader.request(self.il, experts[0], n) {
            Ok(receivers) => self.in_flight.extend(experts.into_iter().zip(receivers)),
            Err(error) => self.failed = Some(error),
        }
    }

    /// The built expert `e`, the next one of the pass (experts come in
    /// ascending order of `needed`).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `e` is not the pass's next expert or its
    /// read or build failed.
    pub(crate) fn next(&mut self, e: u32) -> Result<Arc<T>, CucaError> {
        if let Some(error) = self.failed.take() {
            return Err(error);
        }
        let at = self.needed[self.cursor..]
            .iter()
            .position(|&n| n == e)
            .ok_or_else(|| err(format!("expert {e} is not next in layer {}", self.il)))?;
        self.cursor += at + 1;
        let store = self.store;
        // Bound first: the table guard must be released before `issue`
        // locks it again.
        let hit = if store.ram_slots > 0 {
            store.table(self.il).get(e)
        } else {
            None
        };
        if let Some(value) = hit {
            add(&store.shared.ram_hits, 1);
            self.issue();
            return Ok(value);
        }
        if !self.in_flight.iter().any(|(id, _)| *id == e) {
            // Evicted for a fetch after the pass started: request it now
            // (it heads `pending`, nothing after it is in flight).
            if let Some(i) = self.pending.iter().position(|&p| p == e) {
                self.pending.remove(i);
            }
            self.pending.insert(0, e);
            self.issue();
            if let Some(error) = self.failed.take() {
                return Err(error);
            }
        }
        let i = self
            .in_flight
            .iter()
            .position(|(id, _)| *id == e)
            .ok_or_else(|| {
                err(format!(
                    "expert {e} of layer {} was never requested",
                    self.il
                ))
            })?;
        let (_, rx) = self.in_flight.remove(i);
        let (value, bytes) = receive(&rx)?;
        store.count_read(bytes);
        if store.ram_slots > 0 {
            store.table(self.il).insert(e, Arc::clone(&value));
            add(
                &store.shared.ram_resident_bytes,
                store.stored_bytes(self.il),
            );
        }
        self.issue();
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::GgmlDType;

    use super::*;
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn new(tag: &str, bytes: &[u8]) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "cuca-tier-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            File::create(&path).unwrap().write_all(bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn read_exact_at_reads_the_requested_range_and_rejects_eof() {
        let bytes: Vec<u8> = (0..=255).collect();
        let tmp = TempFile::new("read", &bytes);
        let file = File::open(&tmp.0).unwrap();
        let mut buf = [0u8; 16];
        read_exact_at(&file, 100, &mut buf).unwrap();
        assert_eq!(buf.as_slice(), &bytes[100..116]);
        read_exact_at(&file, 3, &mut buf[..2]).unwrap();
        assert_eq!(&buf[..2], &[3, 4]);
        let e = read_exact_at(&file, 250, &mut buf).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn aligned_reads_return_the_requested_range_and_stop_at_the_end() {
        let bytes: Vec<u8> = (0..20_000u32).map(|i| (i * 7 + i / 251) as u8).collect();
        let tmp = TempFile::new("direct", &bytes);
        let direct = ExpertFile {
            file: File::open(&tmp.0).unwrap(),
            direct: true,
        };
        let mut buf = ReadBuf::default();
        let ranges = [
            (0, 1),
            (0, 4096),
            (1, 4096),
            (4095, 2),
            (4096, 8192),
            (5000, 9000),
            (12_345, 7_655),
            (19_999, 1),
            (16_384, 3_616),
        ];
        for (offset, len) in ranges {
            direct.read(offset, len, &mut buf).unwrap();
            let want = &bytes[offset as usize..offset as usize + len];
            assert_eq!(buf.bytes(), want, "offset {offset} len {len}");
        }
        // Past the end: the file stops before the range does.
        let e = direct.read(19_990, 20, &mut buf).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
        let e = direct.read(20_000, 1, &mut buf).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
        // The platform's own open reads the same bytes.
        let platform = ExpertFile::open(&tmp.0).unwrap();
        platform.read(777, 15_000, &mut buf).unwrap();
        assert_eq!(buf.bytes(), &bytes[777..15_777]);
        platform.read(19_990, 10, &mut buf).unwrap();
        assert_eq!(buf.bytes(), &bytes[19_990..]);
        let e = platform.read(19_990, 11, &mut buf).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    /// xorshift64: a deterministic stream for the short-read injector.
    struct Xorshift(u64);

    impl Xorshift {
        fn step(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.step() % n.max(1) as u64) as usize
        }
    }

    /// The byte a position-keyed source holds at `at`.
    fn byte_at(at: u64) -> u8 {
        (at.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8
    }

    /// Wrap a positional `source` so that every call returns a random number
    /// of bytes (1 up to what was asked) and sometimes `Interrupted`, the way
    /// a loaded operating system may.
    fn flaky<'a>(
        rng: &'a mut Xorshift,
        mut source: impl FnMut(&mut [u8], u64) -> std::io::Result<usize> + 'a,
    ) -> impl FnMut(&mut [u8], u64) -> std::io::Result<usize> + 'a {
        move |rest, at| {
            if rng.below(8) == 0 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let n = 1 + rng.below(rest.len());
            source(&mut rest[..n], at)
        }
    }

    #[test]
    fn concurrent_positional_reads_with_random_short_reads_return_the_file_bytes() {
        const LEN: usize = 1 << 20;
        let bytes: Vec<u8> = (0..LEN as u64).map(byte_at).collect();
        let tmp = TempFile::new("short", &bytes);
        // One handle, shared by every thread, as the expert loaders share it.
        let file = File::open(&tmp.0).unwrap();
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let (file, bytes) = (&file, &bytes);
                s.spawn(move || {
                    let mut rng = Xorshift(0x9E37_79B9 + t);
                    for _ in 0..400 {
                        let len = 1 + rng.below(1 << 16);
                        let offset = rng.below(LEN - len + 1);
                        let mut buf = vec![0u8; len];
                        read_fully_at(
                            &mut buf,
                            offset as u64,
                            flaky(&mut rng, |rest, at| read_some_at(file, rest, at)),
                        )
                        .unwrap();
                        assert!(
                            buf == bytes[offset..offset + len],
                            "thread {t}: {len} bytes at {offset} differ from the file"
                        );
                        read_exact_at(file, offset as u64, &mut buf).unwrap();
                        assert!(
                            buf == bytes[offset..offset + len],
                            "thread {t}: read_exact_at of {len} bytes at {offset} differs"
                        );
                    }
                });
            }
        });
    }

    #[test]
    fn short_reads_advance_buffer_and_offset_together_beyond_4_gib() {
        let mut rng = Xorshift(7);
        for base in [
            0u64,
            u64::from(u32::MAX) - 100,
            u64::from(u32::MAX) + 1,
            5 << 32,
            (1 << 40) + 12345,
        ] {
            let mut buf = vec![0u8; 300];
            let mut calls: Vec<(u64, usize)> = Vec::new();
            let source = |rest: &mut [u8], at: u64| {
                for (i, b) in rest.iter_mut().enumerate() {
                    *b = byte_at(at + i as u64);
                }
                calls.push((at, rest.len()));
                Ok(rest.len())
            };
            read_fully_at(&mut buf, base, flaky(&mut rng, source)).unwrap();
            assert!(
                buf.iter()
                    .enumerate()
                    .all(|(i, &b)| b == byte_at(base + i as u64)),
                "bytes at {base} differ"
            );
            assert_eq!(calls.first().map(|c| c.0), Some(base));
            for w in calls.windows(2) {
                assert_eq!(w[1].0, w[0].0 + w[0].1 as u64, "gap or overlap at {base}");
            }
            assert_eq!(calls.iter().map(|c| c.1).sum::<usize>(), buf.len());
        }
    }

    #[test]
    fn read_fully_at_stops_at_the_end_of_the_source_and_surfaces_errors() {
        let mut rng = Xorshift(3);
        let end = 1000u64;
        let source = |rest: &mut [u8], at: u64| {
            let n = rest.len().min(end.saturating_sub(at) as usize);
            for (i, b) in rest[..n].iter_mut().enumerate() {
                *b = byte_at(at + i as u64);
            }
            Ok(n)
        };
        let mut buf = [0u8; 64];
        read_fully_at(&mut buf, end - 64, flaky(&mut rng, source)).unwrap();
        let e = read_fully_at(&mut buf, end - 63, flaky(&mut rng, source)).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
        let mut served = 0;
        let e = read_fully_at(&mut buf, 0, |rest, _| {
            served += 1;
            if served == 3 {
                Err(std::io::Error::other("boom"))
            } else {
                Ok(rest.len().min(5))
            }
        })
        .unwrap_err();
        assert_eq!(e.to_string(), "boom");
    }

    #[test]
    fn victim_prefers_cold_then_lru_and_protects_the_pass() {
        let mut t = SlotTable::new(3);
        for e in [1u32, 4, 6] {
            t.insert(e, Arc::new(()));
        }
        // Coldest wins.
        assert_eq!(t.victim(&[0, 5, 0, 0, 2, 0, 9], &[]), Some(4));
        // Equal heat: least recently used.
        t.get(1);
        assert_eq!(t.victim(&[0; 7], &[]), Some(4));
        t.get(4);
        assert_eq!(t.victim(&[0; 7], &[]), Some(6));
        // Experts the pass still needs are protected.
        assert_eq!(t.victim(&[0; 7], &[6]), Some(1));
        // All needed later: the one needed farthest ahead.
        assert_eq!(t.victim(&[0; 7], &[1, 4, 6]), Some(6));
        assert_eq!(SlotTable::<()>::new(2).victim(&[], &[]), None);
    }

    /// A built expert that counts how many are alive.
    struct Built {
        il: usize,
        e: u32,
        alive: Arc<AtomicUsize>,
    }

    impl Drop for Built {
        fn drop(&mut self) {
            self.alive.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// A file with `n_expert` experts of 4 f32 per matrix in each of
    /// `n_layer` layers, whose bytes encode `(layer, expert)`.
    fn store(
        n_layer: usize,
        n_expert: usize,
        slots: usize,
        threads: usize,
        alive: &Arc<AtomicUsize>,
        peak: &Arc<AtomicUsize>,
    ) -> (TempFile, ExpertStore<Built>) {
        store_shaped(n_layer, n_expert, slots, threads, alive, peak, 0, 0)
    }

    /// [`store`] with `gap` padding bytes after every expert's matrix (a
    /// stride above the matrix size, as a fused gate/up tensor has) and the
    /// last `cut` bytes of the file missing.
    #[expect(
        clippy::too_many_arguments,
        reason = "a test fixture's independent knobs"
    )]
    fn store_shaped(
        n_layer: usize,
        n_expert: usize,
        slots: usize,
        threads: usize,
        alive: &Arc<AtomicUsize>,
        peak: &Arc<AtomicUsize>,
        gap: usize,
        cut: usize,
    ) -> (TempFile, ExpertStore<Built>) {
        let file_bytes = 16;
        let mut bytes = Vec::new();
        let mut layers = Vec::new();
        for il in 0..n_layer {
            let slice = |bytes: &mut Vec<u8>| {
                let start = bytes.len() as u64;
                for e in 0..n_expert {
                    bytes.extend(std::iter::repeat_n((il * 16 + e) as u8, file_bytes));
                    bytes.extend(std::iter::repeat_n(0xEE, gap));
                }
                ExpertSlice {
                    dtype: GgufType::Ggml(GgmlDType::F32),
                    rows: 1,
                    file: 0,
                    cols: 4,
                    start,
                    file_bytes,
                    stride: file_bytes + gap,
                }
            };
            let (gate, up, down) = (slice(&mut bytes), slice(&mut bytes), slice(&mut bytes));
            layers.push(LayerExperts {
                il,
                gate,
                up,
                down,
                stored_bytes: 100,
            });
        }
        bytes.truncate(bytes.len() - cut);
        let tmp = TempFile::new("store", &bytes);
        let (alive2, peak2) = (Arc::clone(alive), Arc::clone(peak));
        let build: Arc<BuildFn<Built>> = Arc::new(move |l, g, u, d| {
            assert!(g == u && u == d && g.iter().all(|&b| b == g[0]));
            let now = alive2.fetch_add(1, Ordering::SeqCst) + 1;
            peak2.fetch_max(now, Ordering::SeqCst);
            Ok(Built {
                il: l.il,
                e: u32::from(g[0]) - (l.il * 16) as u32,
                alive: Arc::clone(&alive2),
            })
        });
        let shared = Arc::new(TierShared::new(n_layer, n_expert));
        let store = ExpertStore::open(
            &[tmp.0.as_path()],
            n_layer,
            layers,
            n_expert,
            2,
            slots as u64 * 100 * n_layer as u64,
            slots == 0,
            threads,
            build,
            shared,
        )
        .unwrap();
        (tmp, store)
    }

    /// Reads issued so far, one per matrix of a job.
    fn reads(store: &ExpertStore<Built>) -> u64 {
        store.ctx.reads.load(Ordering::Relaxed)
    }

    #[test]
    fn adjacent_missing_experts_read_as_runs_with_the_same_bytes_and_counters() {
        let (alive, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (_tmp, store) = store(1, 16, 16, 4, &alive, &peak);
        let needed = [0u32, 1, 2, 3, 4, 5, 9, 10, 12];
        let mut pass = store.pass(0, &needed);
        for e in needed {
            let x = pass.next(e).unwrap();
            assert_eq!((x.il, x.e), (0, e));
        }
        // [0..4), [4, 5], [9, 10] and [12]: four jobs of three reads, not
        // nine experts of three.
        assert_eq!(reads(&store), 12);
        let stats = store.shared().stats();
        assert_eq!((stats.disk_reads, stats.bytes_read), (9, 9 * 48));
        assert_eq!(stats.ram_resident_bytes, 900);
        // A second pass over the same experts reads nothing.
        let before = reads(&store);
        let mut pass = store.pass(0, &needed);
        for e in needed {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
        assert_eq!(reads(&store), before);
        assert_eq!(store.shared().stats().ram_hits, 9);
    }

    #[test]
    fn a_run_is_cut_at_max_run_and_without_a_ram_tier_at_the_loader_threads() {
        let (alive, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (_tmp, store) = store(1, 16, 16, 8, &alive, &peak);
        let needed: Vec<u32> = (0..10).collect();
        let mut pass = store.pass(0, &needed);
        for &e in &needed {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
        // Ten adjacent experts: runs of 4, 4 and 2.
        assert_eq!(reads(&store), 9);

        // No RAM tier: at most `threads` experts in flight. Two threads read
        // the pair [2, 3] at once; each consumed expert then frees one place,
        // so 4 and 5 follow alone.
        let (alive, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (_tmp, store) = store_shaped(1, 16, 0, 2, &alive, &peak, 0, 0);
        let mut pass = store.pass(0, &[2, 3, 4, 5]);
        for e in [2, 3, 4, 5] {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
        assert_eq!(reads(&store), 9);
        assert!(peak.load(Ordering::SeqCst) <= 3, "{peak:?}");
    }

    #[test]
    fn a_layer_whose_experts_are_not_adjacent_reads_expert_by_expert() {
        let (alive, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (_tmp, store) = store_shaped(1, 8, 8, 4, &alive, &peak, 16, 0);
        assert!(!store.layer(0).unwrap().contiguous());
        let needed = [0u32, 1, 2, 3, 4];
        let mut pass = store.pass(0, &needed);
        for e in needed {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
        assert_eq!(reads(&store), 15);
    }

    #[test]
    fn fetch_returns_the_experts_in_the_order_asked_and_reads_adjacent_ones_together() {
        let (alive, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (_tmp, store) = store(1, 16, 16, 4, &alive, &peak);
        let asked = [9u32, 3, 4, 5, 0, 10];
        let got = store.fetch(0, &asked).unwrap();
        assert_eq!(got.iter().map(|x| x.e).collect::<Vec<_>>(), asked);
        // [0], [3, 4, 5] and [9, 10].
        assert_eq!(reads(&store), 9);
        assert_eq!(store.shared().stats().disk_reads, 6);
        assert!(store.fetch(0, &[]).unwrap().is_empty());
    }

    #[test]
    fn a_failed_read_fails_every_expert_of_its_run() {
        let (alive, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        // The last matrix of the layer loses its final 20 bytes: experts 6 and
        // 7 of `down` are cut, so the run [4, 5, 6, 7] cannot be read as one.
        let (_tmp, store) = store_shaped(1, 8, 8, 4, &alive, &peak, 0, 20);
        let mut pass = store.pass(0, &[4, 5, 6, 7]);
        for e in [4, 5, 6, 7] {
            let error = pass.next(e).err().expect("the run fails").to_string();
            assert!(
                error.contains(&format!("expert {e}")) && error.contains("truncated GGUF"),
                "{error}"
            );
        }
        // The experts before the cut read fine.
        let mut pass = store.pass(0, &[0, 1]);
        for e in [0, 1] {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
    }

    #[test]
    fn pass_keeps_resident_plus_in_flight_within_slots() {
        let alive = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (_tmp, store) = store(1, 8, 3, 4, &alive, &peak);
        assert_eq!(store.ram_slots(), 3);
        let needed: Vec<u32> = (0..8).collect();
        for round in 0..3 {
            let mut pass = store.pass(0, &needed);
            for &e in &needed {
                let x = pass.next(e).unwrap();
                assert_eq!((x.il, x.e), (0, e), "round {round}");
                // The table plus the reads in flight hold at most the three
                // slots; the expert being computed may outlive its eviction.
                assert!(alive.load(Ordering::SeqCst) <= 4);
            }
        }
        let stats = store.shared().stats();
        assert!(peak.load(Ordering::SeqCst) <= 4, "{peak:?}");
        assert_eq!(stats.ram_resident_bytes, 300);
        assert_eq!(stats.disk_reads + stats.ram_hits, 24);
        assert!(stats.disk_reads >= 8, "{stats:?}");
        assert_eq!(stats.bytes_read, stats.disk_reads * 48);

        // A small pass after the scan hits what the scan left resident.
        let before = store.shared().stats();
        let mut pass = store.pass(0, &[5, 6, 7]);
        for e in [5, 6, 7] {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
        let after = store.shared().stats();
        assert_eq!(after.ram_hits - before.ram_hits, 3);

        // A hit while a read is still pending (expert 7 was evicted to make
        // room for 0, which is needed first) must not deadlock.
        let mut pass = store.pass(0, &[0, 5, 6, 7]);
        for e in [0, 5, 6, 7] {
            assert_eq!(pass.next(e).unwrap().e, e);
        }
        assert!(alive.load(Ordering::SeqCst) <= 4);
    }

    #[test]
    fn pass_without_a_ram_tier_caches_nothing() {
        let alive = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (_tmp, store) = store(2, 8, 0, 2, &alive, &peak);
        assert_eq!(store.ram_slots(), 0);
        let mut pass = store.pass(1, &[1, 3, 4, 7]);
        for e in [1, 3, 4, 7] {
            let x = pass.next(e).unwrap();
            assert_eq!((x.il, x.e), (1, e));
        }
        drop(pass);
        assert_eq!(alive.load(Ordering::SeqCst), 0);
        assert!(peak.load(Ordering::SeqCst) <= 3);
        let stats = store.shared().stats();
        assert_eq!((stats.disk_reads, stats.ram_resident_bytes), (4, 0));
        let mut pass = store.pass(1, &[2, 5]);
        assert!(pass.next(5).is_err() || pass.next(2).is_err());
    }

    #[test]
    fn ram_tier_below_n_used_is_a_config_error() {
        let e = tier_slots("RAM", 199, 100, 8, 2).unwrap_err();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("at least 2 (200 bytes)")),
            "{e:?}"
        );
        assert_eq!(tier_slots("RAM", 200, 100, 8, 2).unwrap(), 2);
        assert_eq!(tier_slots("RAM", 1 << 40, 100, 8, 2).unwrap(), 8);
    }

    #[test]
    fn heat_decays_after_4096_rows() {
        let shared = TierShared::new(2, 3);
        let routing = Routing {
            per_expert: vec![vec![(0, 0.5); 8], Vec::new(), vec![(0, 0.5); 2]],
        };
        shared.record(1, &routing, 4095);
        assert_eq!(shared.heat_snapshot(1), [8, 0, 2]);
        assert_eq!(shared.heat_snapshot(0), [0, 0, 0]);
        shared.record(1, &routing, 1);
        assert_eq!(shared.heat_snapshot(1), [8, 0, 2]);
        shared.record(1, &routing, 10);
        assert_eq!(shared.heat_snapshot(1), [16, 0, 4]);
        let heat = shared.heat();
        assert_eq!(heat.count(1, 0), Some(16));
        assert_eq!(heat.count(2, 0), None);
        assert_eq!(heat.ranked(1), [0, 2]);
    }

    #[test]
    fn expert_heat_round_trips_through_serde_json() {
        let shared = TierShared::new(2, 2);
        shared.record(
            0,
            &Routing {
                per_expert: vec![vec![(0, 1.0)], vec![(0, 1.0); 3]],
            },
            1,
        );
        let heat = shared.heat();
        let json = serde_json::to_string(&heat).unwrap();
        let back: ExpertHeat = serde_json::from_str(&json).unwrap();
        assert_eq!(back, heat);
        assert_eq!(back.ranked(0), [1, 0]);
        let bad = r#"{"n_layer":2,"n_expert":2,"counts":[1,2,3]}"#;
        assert!(serde_json::from_str::<ExpertHeat>(bad).is_err());

        let other = TierShared::new(2, 2);
        other.check(&back).unwrap();
        other.seed(&back);
        assert_eq!(other.heat(), heat);
        let e = TierShared::new(3, 2).check(&back).unwrap_err();
        assert!(matches!(&e, CucaError::Config(m) if m.contains("2x2") && m.contains("3x2")));
    }
}
