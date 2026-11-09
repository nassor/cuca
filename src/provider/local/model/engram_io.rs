//! The reader of an Engram table that does not fit in RAM
//! ([`EngramTable`], a [`RowSource`]).
//!
//! DeepSeek-V4.1's `per_layer_token_embd` holds 768 M rows of 256 values
//! (209 GB at Q8_0); a token position needs 24 rows per engram layer, chosen
//! by hashing the n-gram that ends there, so the reads are random, tiny (272
//! bytes at Q8_0) and nearly independent of each other. Three things make that
//! tolerable:
//!
//! - **Parallel, sorted reads.** A pass knows every row id it needs before it
//!   computes anything (they depend on token ids only). [`RowSource::prefetch`]
//!   sorts and de-duplicates them, drops the ones in the cache or already
//!   being read, and hands the rest, in file order and in chunks, to
//!   [`EngramCache::reader_threads`] reader threads that read with positional
//!   reads ([`read_exact_at`], one handle for all threads) and merge
//!   neighbouring rows into one read. A disk answers many outstanding random
//!   reads much faster than one at a time (NCQ on a hard disk, queue depth on
//!   an SSD), and the reads of the second engram layer overlap the compute of
//!   the first.
//! - **A bounded row cache.** The most recently used rows stay in RAM as the
//!   file's own (quantized) bytes, [`EngramCache::ram_bytes`] at most: n-gram
//!   statistics are heavy tailed, so a repeated phrase, a repeated prompt or a
//!   conversation that appends a turn finds its rows again. The cache is an
//!   LRU over fixed-size slots whose storage is allocated in segments as it
//!   fills, so a short session does not pay for the bound. A cache smaller
//!   than the distinct rows of one pass still gives correct rows: the ones
//!   evicted before they are used are read again.
//! - **Counters.** [`EngramStats`] reports the lookups, the share served
//!   without a new read, the rows and bytes read and the resident rows
//!   ([`LocalModel::engram_stats`](crate::LocalModel::engram_stats)).
//!
//! # Bounds
//!
//! Growth with traffic is bounded: the cache holds at most
//! `ram_bytes / (row_bytes + ENTRY_OVERHEAD)` rows (the overhead is the slot's
//! key, links and index entry, `ENTRY_OVERHEAD` = 48 bytes, so the figure
//! covers the whole structure), evicting the least recently used row; the reads
//! in flight are the distinct rows of the pass that asked, at most the rows of
//! one forward step (`max_step_tokens * 24` per engram layer at the largest
//! step); a pass's scratch is one raw copy of its distinct rows.
//!
//! # Exactness
//!
//! A row's values are the dequantization of its file bytes, whoever read them
//! and whenever, so the cache and the threads change timing only: the output of
//! [`RowSource::rows`] equals reading each row alone.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use super::engram::RowSource;
use super::err;
use super::layout::tensor_info;
use super::pool::par_chunks;
use super::quant::dequant_row;
use super::tier::read_exact_at;
use crate::error::CucaError;
use crate::provider::local::EngramCache;
use crate::provider::local::gguf::{self, GgufType};

/// Bytes the cache accounts per row besides the row's own bytes: the key, the
/// two list links and the index entry.
pub(crate) const ENTRY_OVERHEAD: usize = 48;
/// Rows read by one reader job.
const JOB_ROWS: usize = 64;
/// Rows of one storage segment of the cache.
const SEGMENT_ROWS: usize = 4096;
/// The end of a slot list.
const NIL: u32 = u32::MAX;

/// Engram table counters ([`LocalModel::engram_stats`](crate::LocalModel::engram_stats)),
/// cumulative since load except the two resident gauges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct EngramStats {
    /// Rows the passes asked for, a row asked several times in one pass
    /// counted once.
    pub lookups: u64,
    /// Lookups that cost no read of their own: `lookups - disk_reads` (zero
    /// when more rows were read than asked for, which a read-ahead of rows a
    /// later pass has not asked for yet can cause).
    pub cache_hits: u64,
    /// Rows read from the GGUF file, prefetches included.
    pub disk_reads: u64,
    /// Bytes read from the GGUF file for rows.
    pub bytes_read: u64,
    /// Rows the cache holds now.
    pub resident_rows: u64,
    /// Bytes of rows the cache holds now (their file bytes).
    pub resident_bytes: u64,
    /// Rows the cache can hold.
    pub capacity_rows: u64,
}

impl EngramStats {
    /// `cache_hits / lookups`; `None` before the first lookup.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.lookups > 0).then(|| self.cache_hits as f64 / self.lookups as f64)
    }
}

/// The counters of one table, shared by the reader (which writes them) and
/// the `LocalModel` handle (which reads them).
#[derive(Debug, Default)]
pub(crate) struct EngramCounters {
    lookups: AtomicU64,
    disk_reads: AtomicU64,
    bytes_read: AtomicU64,
    resident_rows: AtomicU64,
    capacity_rows: AtomicU64,
    row_bytes: AtomicU64,
}

impl EngramCounters {
    pub(crate) fn snapshot(&self) -> EngramStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let resident_rows = get(&self.resident_rows);
        EngramStats {
            lookups: get(&self.lookups),
            cache_hits: get(&self.lookups).saturating_sub(get(&self.disk_reads)),
            disk_reads: get(&self.disk_reads),
            bytes_read: get(&self.bytes_read),
            resident_rows,
            resident_bytes: resident_rows * get(&self.row_bytes),
            capacity_rows: get(&self.capacity_rows),
        }
    }
}

fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

/// An LRU of fixed-size rows over segmented storage.
struct RowCache {
    row_bytes: usize,
    /// Most rows held.
    capacity: usize,
    /// Row storage, `SEGMENT_ROWS` rows per segment (the last may be
    /// shorter), allocated when its first slot is used.
    segments: Vec<Vec<u8>>,
    slot_of: HashMap<u64, u32>,
    /// Per used slot: the row id and the neighbours in recency order.
    key: Vec<u64>,
    newer: Vec<u32>,
    older: Vec<u32>,
    /// The most and the least recently used slot.
    head: u32,
    tail: u32,
}

impl RowCache {
    fn new(row_bytes: usize, capacity: usize) -> Self {
        Self {
            row_bytes,
            capacity,
            segments: Vec::new(),
            slot_of: HashMap::new(),
            key: Vec::new(),
            newer: Vec::new(),
            older: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    fn len(&self) -> usize {
        self.key.len()
    }

    fn bytes(&self, slot: u32) -> std::ops::Range<usize> {
        let at = slot as usize % SEGMENT_ROWS * self.row_bytes;
        at..at + self.row_bytes
    }

    fn detach(&mut self, slot: u32) {
        let (newer, older) = (self.newer[slot as usize], self.older[slot as usize]);
        match newer {
            NIL => self.head = older,
            n => self.older[n as usize] = older,
        }
        match older {
            NIL => self.tail = newer,
            o => self.newer[o as usize] = newer,
        }
    }

    fn attach_front(&mut self, slot: u32) {
        self.newer[slot as usize] = NIL;
        self.older[slot as usize] = self.head;
        match self.head {
            NIL => self.tail = slot,
            h => self.newer[h as usize] = slot,
        }
        self.head = slot;
    }

    /// Copy the row `id` into `out` and mark it most recently used; whether
    /// the cache holds it.
    fn get(&mut self, id: u64, out: &mut [u8]) -> bool {
        let Some(&slot) = self.slot_of.get(&id) else {
            return false;
        };
        let range = self.bytes(slot);
        out.copy_from_slice(&self.segments[slot as usize / SEGMENT_ROWS][range]);
        if self.head != slot {
            self.detach(slot);
            self.attach_front(slot);
        }
        true
    }

    /// Make the row `id` (`bytes`) the most recently used, evicting the least
    /// recently used one when the cache is full.
    fn insert(&mut self, id: u64, bytes: &[u8]) {
        if let Some(&slot) = self.slot_of.get(&id) {
            if self.head != slot {
                self.detach(slot);
                self.attach_front(slot);
            }
            return;
        }
        let slot = if self.len() < self.capacity {
            let slot = self.len() as u32;
            self.key.push(id);
            self.newer.push(NIL);
            self.older.push(NIL);
            if slot as usize / SEGMENT_ROWS >= self.segments.len() {
                let first = self.segments.len() * SEGMENT_ROWS;
                let rows = SEGMENT_ROWS.min(self.capacity - first);
                self.segments.push(vec![0u8; rows * self.row_bytes]);
            }
            slot
        } else {
            let slot = self.tail;
            self.detach(slot);
            self.slot_of.remove(&self.key[slot as usize]);
            self.key[slot as usize] = id;
            slot
        };
        let range = self.bytes(slot);
        self.segments[slot as usize / SEGMENT_ROWS][range].copy_from_slice(bytes);
        self.slot_of.insert(id, slot);
        self.attach_front(slot);
    }
}

/// What the cache and the reads in flight hold.
struct State {
    cache: RowCache,
    /// Rows some thread is reading and will put in the cache.
    inflight: HashSet<u64>,
}

/// What a reader job answers: its index among the caller's jobs and the
/// rows' bytes in job order.
type Reply = (usize, Result<Vec<u8>, CucaError>);

/// Rows to read, ascending and distinct.
struct Job {
    ids: Vec<u64>,
    /// The caller waiting for the bytes (a lookup); a prefetch only fills the
    /// cache.
    reply: Option<(usize, Sender<Reply>)>,
}

struct Shared {
    tensor: &'static str,
    file: File,
    /// File offset of row 0.
    base: u64,
    dtype: GgufType,
    rows: u64,
    row_dim: usize,
    row_bytes: usize,
    state: Mutex<State>,
    /// Signalled whenever rows leave `inflight`.
    ready: Condvar,
    counters: Arc<EngramCounters>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The file bytes of the ascending, distinct rows `ids`, neighbouring
    /// rows merged into one read.
    fn read(&self, ids: &[u64]) -> Result<Vec<u8>, CucaError> {
        let rb = self.row_bytes;
        let mut buf = vec![0u8; ids.len() * rb];
        let mut i = 0;
        while i < ids.len() {
            let mut run = 1;
            while i + run < ids.len() && ids[i + run] == ids[i] + run as u64 {
                run += 1;
            }
            read_exact_at(
                &self.file,
                self.base + ids[i] * rb as u64,
                &mut buf[i * rb..(i + run) * rb],
            )
            .map_err(|e| CucaError::Io {
                message: format!(
                    "reading `{}` rows {}..{}: {e}",
                    self.tensor,
                    ids[i],
                    ids[i] + run as u64
                ),
            })?;
            i += run;
        }
        add(&self.counters.disk_reads, ids.len() as u64);
        add(&self.counters.bytes_read, buf.len() as u64);
        Ok(buf)
    }

    /// Put the rows `ids` (`bytes`) in the cache and wake whoever waits for
    /// them.
    fn install(&self, ids: &[u64], bytes: &[u8]) {
        let mut st = self.state();
        for (&id, row) in ids.iter().zip(bytes.chunks_exact(self.row_bytes)) {
            st.cache.insert(id, row);
            st.inflight.remove(&id);
        }
        self.counters
            .resident_rows
            .store(st.cache.len() as u64, Ordering::Relaxed);
        drop(st);
        self.ready.notify_all();
    }

    /// Stop expecting the rows `ids` (their read failed or never started).
    fn abandon(&self, ids: &[u64]) {
        let mut st = self.state();
        for id in ids {
            st.inflight.remove(id);
        }
        drop(st);
        self.ready.notify_all();
    }

    fn run(&self, job: Job) {
        /// Stops expecting the job's rows if the job unwinds before it
        /// installed or abandoned them, so no lookup waits for them forever.
        struct Release<'a> {
            shared: &'a Shared,
            ids: &'a [u64],
            done: bool,
        }
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                if !self.done {
                    self.shared.abandon(self.ids);
                }
            }
        }
        let mut release = Release {
            shared: self,
            ids: &job.ids,
            done: false,
        };
        let result = self.read(&job.ids);
        match &result {
            Ok(bytes) => self.install(&job.ids, bytes),
            Err(_) => self.abandon(&job.ids),
        }
        release.done = true;
        if let Some((index, reply)) = job.reply {
            // The caller may have given up (an earlier job failed).
            let _ = reply.send((index, result));
        }
    }
}

/// A [`RowSource`] over a 2-D tensor of a GGUF file, read by a pool of
/// threads through an LRU row cache; see the [module docs](self).
pub(crate) struct EngramTable {
    shared: Arc<Shared>,
    jobs: Option<Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

/// How big the row cache is.
enum Budget {
    /// A RAM budget in bytes, the per-row overhead included.
    Bytes(u64),
    /// A row count (tests).
    #[cfg(test)]
    Rows(usize),
}

impl EngramTable {
    /// Index the `[rows, row_dim]` tensor `tensor` of the set `content`
    /// describes and start the readers; no row is read yet.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for a `cache` outside its bounds
    /// ([`EngramCache::validate`]) or smaller than one row;
    /// [`CucaError::Provider`] for a missing or mis-shaped tensor, a row
    /// that is not a whole number of blocks, a type the engine cannot
    /// dequantize or a truncated file; [`CucaError::Io`] when the file cannot
    /// be opened or a thread cannot start.
    pub(crate) fn open(
        content: &gguf::Content,
        tensor: &'static str,
        row_dim: usize,
        cache: &EngramCache,
    ) -> Result<Self, CucaError> {
        cache.validate()?;
        Self::open_sized(
            content,
            tensor,
            row_dim,
            cache.reader_threads,
            Budget::Bytes(cache.ram_bytes),
        )
    }

    /// [`EngramTable::open`] with a cache of `rows` rows (the unit tests'
    /// small caches are below the public bound).
    #[cfg(test)]
    pub(crate) fn open_rows(
        content: &gguf::Content,
        tensor: &'static str,
        row_dim: usize,
        reader_threads: usize,
        rows: usize,
    ) -> Result<Self, CucaError> {
        Self::open_sized(content, tensor, row_dim, reader_threads, Budget::Rows(rows))
    }

    fn open_sized(
        content: &gguf::Content,
        tensor: &'static str,
        row_dim: usize,
        reader_threads: usize,
        budget: Budget,
    ) -> Result<Self, CucaError> {
        let info = tensor_info(content, tensor)?;
        let dims = info.shape.dims();
        if dims.len() != 2 || dims[1] != row_dim {
            return Err(err(format!(
                "tensor `{tensor}`: expected shape [rows, {row_dim}], found {dims:?}"
            )));
        }
        let dtype = info.dtype;
        let row_bytes = dtype.byte_len(row_dim).ok_or_else(|| {
            err(format!(
                "tensor `{tensor}`: row length {row_dim} is not a whole number of {} blocks",
                dtype.name()
            ))
        })?;
        // A type the engine cannot dequantize fails here, not mid-pass.
        dequant_row(dtype, &vec![0u8; row_bytes], 1.0, &mut vec![0f32; row_dim])
            .map_err(|e| err(format!("tensor `{tensor}`: {e}")))?;
        let (path, base) = content.location(info);
        let file = File::open(path)?;
        if file.metadata()?.len() < base + dims[0] as u64 * row_bytes as u64 {
            return Err(err(format!(
                "tensor `{tensor}` extends past the end of the file: truncated GGUF"
            )));
        }
        let fits = match budget {
            Budget::Bytes(bytes) => {
                usize::try_from(bytes / (row_bytes + ENTRY_OVERHEAD) as u64).unwrap_or(usize::MAX)
            }
            #[cfg(test)]
            Budget::Rows(rows) => rows,
        };
        let capacity = fits.min(u32::MAX as usize - 1).min(dims[0]);
        if capacity == 0 {
            return Err(CucaError::Config(format!(
                "the engram cache holds no row of `{tensor}` ({} bytes each with its bookkeeping)",
                row_bytes + ENTRY_OVERHEAD
            )));
        }
        let counters = Arc::new(EngramCounters::default());
        counters
            .capacity_rows
            .store(capacity as u64, Ordering::Relaxed);
        counters
            .row_bytes
            .store(row_bytes as u64, Ordering::Relaxed);
        let shared = Arc::new(Shared {
            tensor,
            file,
            base,
            dtype,
            rows: dims[0] as u64,
            row_dim,
            row_bytes,
            state: Mutex::new(State {
                cache: RowCache::new(row_bytes, capacity),
                inflight: HashSet::new(),
            }),
            ready: Condvar::new(),
            counters,
        });
        let (tx, rx) = mpsc::channel::<Job>();
        let queue = Arc::new(Mutex::new(rx));
        let mut table = Self {
            shared: Arc::clone(&shared),
            jobs: Some(tx),
            threads: Vec::with_capacity(reader_threads),
        };
        for i in 0..reader_threads {
            table
                .threads
                .push(spawn_reader(i, Arc::clone(&shared), Arc::clone(&queue))?);
        }
        Ok(table)
    }

    /// Queue the reads of `ids` (ascending and distinct, valid) in chunks.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the readers stopped; the rows are no
    /// longer expected then.
    fn dispatch(
        &self,
        ids: &[u64],
        reply: Option<&Sender<Reply>>,
        first_job: usize,
    ) -> Result<usize, CucaError> {
        let Some(jobs) = &self.jobs else {
            self.shared.abandon(ids);
            return Err(err("the engram readers stopped"));
        };
        let mut sent = 0;
        for chunk in ids.chunks(JOB_ROWS) {
            let job = Job {
                ids: chunk.to_vec(),
                reply: reply.map(|r| (first_job + sent, r.clone())),
            };
            if jobs.send(job).is_err() {
                // The chunks not sent are not coming.
                self.shared.abandon(&ids[sent * JOB_ROWS..]);
                return Err(err("the engram readers stopped"));
            }
            sent += 1;
        }
        Ok(sent)
    }

    /// The distinct rows of `ids`, ascending; an id past the table is an error.
    fn distinct(&self, ids: &[u64]) -> Result<Vec<u64>, CucaError> {
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        match unique.last() {
            Some(&id) if id >= self.shared.rows => Err(err(format!(
                "engram row {id} outside the {} rows of `{}`",
                self.shared.rows, self.shared.tensor
            ))),
            _ => Ok(unique),
        }
    }
}

fn spawn_reader(
    i: usize,
    shared: Arc<Shared>,
    queue: Arc<Mutex<Receiver<Job>>>,
) -> Result<JoinHandle<()>, CucaError> {
    std::thread::Builder::new()
        .name(format!("cuca-engram-reader-{i}"))
        .spawn(move || {
            loop {
                let job = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
                let Ok(job) = job else { return };
                // A panicking job releases its rows (`Shared::run`) and costs
                // that lookup an error; the thread keeps serving.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    shared.run(job);
                }));
            }
        })
        .map_err(|e| CucaError::Io {
            message: format!("cannot start an engram reader thread: {e}"),
        })
}

impl Drop for EngramTable {
    fn drop(&mut self) {
        self.jobs = None;
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl RowSource for EngramTable {
    fn prefetch(&self, ids: &[u64]) {
        let Ok(unique) = self.distinct(ids) else {
            // `rows` reports the bad id when the pass asks for it.
            return;
        };
        let mut wanted = Vec::new();
        {
            let mut st = self.shared.state();
            for id in unique {
                if !st.cache.slot_of.contains_key(&id) && st.inflight.insert(id) {
                    wanted.push(id);
                }
            }
        }
        // A prefetch only warms the cache: if the readers are gone the lookup
        // that needs the rows says so.
        let _ = self.dispatch(&wanted, None, 0);
    }

    fn rows(&self, ids: &[u64], out: &mut [f32]) -> Result<(), CucaError> {
        let sh = &*self.shared;
        let (rb, dim) = (sh.row_bytes, sh.row_dim);
        if out.len() != ids.len() * dim {
            return Err(err(format!(
                "engram rows: {} ids need {} values, the output holds {}",
                ids.len(),
                ids.len() * dim,
                out.len()
            )));
        }
        let unique = self.distinct(ids)?;
        let mut raw = vec![0u8; unique.len() * rb];
        let mut have = vec![false; unique.len()];
        let mut mine: Vec<usize> = Vec::new();
        let mut waiting: Vec<usize> = Vec::new();
        {
            let mut st = sh.state();
            for (u, &id) in unique.iter().enumerate() {
                if st.cache.get(id, &mut raw[u * rb..(u + 1) * rb]) {
                    have[u] = true;
                } else if st.inflight.contains(&id) {
                    waiting.push(u);
                } else {
                    st.inflight.insert(id);
                    mine.push(u);
                }
            }
        }
        add(&sh.counters.lookups, unique.len() as u64);

        // The rows nobody reads yet: read them on the pool, in file order.
        if !mine.is_empty() {
            let wanted: Vec<u64> = mine.iter().map(|&u| unique[u]).collect();
            let (tx, rx) = mpsc::channel();
            let sent = self.dispatch(&wanted, Some(&tx), 0)?;
            drop(tx);
            for _ in 0..sent {
                let (job, bytes) = rx.recv().map_err(|_| err("the engram readers stopped"))?;
                let bytes = bytes?;
                for (k, row) in bytes.chunks_exact(rb).enumerate() {
                    let u = mine[job * JOB_ROWS + k];
                    raw[u * rb..(u + 1) * rb].copy_from_slice(row);
                    have[u] = true;
                }
            }
        }

        // The rows a prefetch is reading: wait for them, and read again the
        // ones the cache evicted before this lookup got to them.
        let mut stale = Vec::new();
        if !waiting.is_empty() {
            let mut st = sh.state();
            loop {
                waiting.retain(|&u| {
                    let id = unique[u];
                    if st.cache.get(id, &mut raw[u * rb..(u + 1) * rb]) {
                        have[u] = true;
                        false
                    } else if st.inflight.contains(&id) {
                        true
                    } else {
                        stale.push(u);
                        false
                    }
                });
                if waiting.is_empty() {
                    break;
                }
                st = sh.ready.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        }
        if !stale.is_empty() {
            let wanted: Vec<u64> = stale.iter().map(|&u| unique[u]).collect();
            let bytes = sh.read(&wanted)?;
            for (&u, row) in stale.iter().zip(bytes.chunks_exact(rb)) {
                raw[u * rb..(u + 1) * rb].copy_from_slice(row);
                have[u] = true;
            }
            sh.install(&wanted, &bytes);
        }
        if let Some(u) = have.iter().position(|h| !h) {
            return Err(err(format!(
                "engram row {} of `{}` was not read",
                unique[u], sh.tensor
            )));
        }

        let failed = AtomicBool::new(false);
        par_chunks(out, dim, |i, dst| {
            // `unique` holds every id: the search finds it.
            let Ok(u) = unique.binary_search(&ids[i]) else {
                failed.store(true, Ordering::Relaxed);
                return;
            };
            if dequant_row(sh.dtype, &raw[u * rb..(u + 1) * rb], 1.0, dst).is_err() {
                failed.store(true, Ordering::Relaxed);
            }
        });
        if failed.load(Ordering::Relaxed) {
            return Err(err(format!("cannot dequantize rows of `{}`", sh.tensor)));
        }
        Ok(())
    }

    fn counters(&self) -> Option<Arc<EngramCounters>> {
        Some(Arc::clone(&self.shared.counters))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::atomic::AtomicU32;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    use super::*;
    use crate::provider::local::gguf::{self, Value};
    use crate::provider::local::model::tiny::Lcg;

    const TENSOR: &str = "per_layer_token_embd.weight";
    const DIM: usize = 64;
    const ROWS: usize = 300;

    /// A GGUF file with one Q8_0 `[ROWS, DIM]` table, removed on drop.
    struct Table {
        path: std::path::PathBuf,
        content: gguf::Content,
        /// The file's own bytes of every row.
        raw: Vec<u8>,
        row_bytes: usize,
    }

    impl Drop for Table {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn table(dtype: GgmlDType) -> Table {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "cuca-engram-{}-{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let values = Lcg(5).values(ROWS * DIM, 2.0);
        let t = Tensor::from_vec(values, (ROWS, DIM), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, dtype).unwrap();
        let arch = Value::String("deepseek41".into());
        gguf::write(
            &mut File::create(&path).unwrap(),
            &[("general.architecture", &arch)],
            &[(TENSOR, &qt)],
        )
        .unwrap();
        let content = gguf::Content::open(&path).unwrap();
        let info = content.info(TENSOR).unwrap();
        let (file, base) = content.location(info);
        let bytes = std::fs::read(file).unwrap();
        let row_bytes = info.dtype.byte_len(DIM).unwrap();
        let raw = bytes[base as usize..base as usize + ROWS * row_bytes].to_vec();
        Table {
            path,
            content,
            raw,
            row_bytes,
        }
    }

    /// The rows `ids` read straight from the file bytes.
    fn reference(t: &Table, ids: &[u64]) -> Vec<f32> {
        let mut out = vec![0f32; ids.len() * DIM];
        let dtype = t.content.info(TENSOR).unwrap().dtype;
        let (rows, _) = out.as_chunks_mut::<DIM>();
        for (&id, dst) in ids.iter().zip(rows) {
            let at = id as usize * t.row_bytes;
            dequant_row(dtype, &t.raw[at..at + t.row_bytes], 1.0, dst).unwrap();
        }
        out
    }

    fn open(t: &Table, threads: usize, rows: usize) -> EngramTable {
        EngramTable::open_rows(&t.content, TENSOR, DIM, threads, rows).unwrap()
    }

    fn stats(table: &EngramTable) -> EngramStats {
        table.counters().unwrap().snapshot()
    }

    /// `n` ids from a seeded generator, repeats included.
    fn ids(seed: u64, n: usize, below: u64) -> Vec<u64> {
        Lcg(seed)
            .values(n, 1.0)
            .iter()
            .map(|v| ((v + 1.0) * 0.5 * below as f32) as u64 % below)
            .collect()
    }

    fn distinct(ids: &[u64]) -> usize {
        let mut u = ids.to_vec();
        u.sort_unstable();
        u.dedup();
        u.len()
    }

    #[test]
    fn rows_equal_the_file_rows_in_any_order_with_repeats_and_a_tiny_cache() {
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4_0, GgmlDType::F32] {
            let t = table(dtype);
            // A cache of 16 rows against ~100 distinct rows per call: most
            // rows are evicted before the next call.
            let e = open(&t, 4, 16);
            for seed in 1..6 {
                let ask = ids(seed, 400, ROWS as u64);
                let mut got = vec![0f32; ask.len() * DIM];
                e.rows(&ask, &mut got).unwrap();
                assert_eq!(got, reference(&t, &ask), "{dtype:?} seed {seed}");
            }
            let s = stats(&e);
            assert!(s.resident_rows <= 16, "{s:?}");
            assert_eq!(s.capacity_rows, 16);
            assert!(s.disk_reads >= s.resident_rows);
        }
    }

    #[test]
    fn a_second_pass_over_the_same_rows_reads_nothing() {
        let t = table(GgmlDType::Q8_0);
        let e = open(&t, 4, ROWS);
        let ask = ids(9, 500, ROWS as u64);
        let n = distinct(&ask) as u64;
        let mut a = vec![0f32; ask.len() * DIM];
        e.rows(&ask, &mut a).unwrap();
        let first = stats(&e);
        assert_eq!(
            (first.lookups, first.cache_hits, first.disk_reads),
            (n, 0, n)
        );
        assert_eq!(first.bytes_read, n * t.row_bytes as u64);
        assert_eq!(first.resident_bytes, n * t.row_bytes as u64);
        let mut b = vec![0f32; ask.len() * DIM];
        e.rows(&ask, &mut b).unwrap();
        let second = stats(&e);
        assert_eq!(second.disk_reads, first.disk_reads, "no row was read again");
        assert_eq!((second.lookups, second.cache_hits), (2 * n, n));
        assert_eq!(second.hit_rate(), Some(0.5));
        assert_eq!(a, b);
    }

    #[test]
    fn a_prefetch_reads_each_row_once_and_the_lookup_waits_for_it() {
        let t = table(GgmlDType::Q8_0);
        let e = open(&t, 4, ROWS);
        let ask = ids(3, 600, ROWS as u64);
        let n = distinct(&ask) as u64;
        // Twice: the second finds every row already being read.
        e.prefetch(&ask);
        e.prefetch(&ask);
        let mut got = vec![0f32; ask.len() * DIM];
        e.rows(&ask, &mut got).unwrap();
        assert_eq!(got, reference(&t, &ask));
        let s = stats(&e);
        assert_eq!(s.disk_reads, n, "{s:?}");
        // Each row was read once, by the prefetch, and asked for once.
        assert_eq!((s.lookups, s.cache_hits), (n, 0));
    }

    #[test]
    fn neighbouring_rows_are_read_together_and_counted_per_row() {
        let t = table(GgmlDType::Q8_0);
        let e = open(&t, 1, ROWS);
        let ask: Vec<u64> = (10..90).chain(200..230).collect();
        let mut got = vec![0f32; ask.len() * DIM];
        e.rows(&ask, &mut got).unwrap();
        assert_eq!(got, reference(&t, &ask));
        assert_eq!(stats(&e).disk_reads, 110);
    }

    #[test]
    fn the_least_recently_used_row_leaves_first() {
        let t = table(GgmlDType::Q8_0);
        let e = open(&t, 2, 8);
        let mut one = vec![0f32; DIM];
        let mut get = |id: u64| e.rows(&[id], &mut one).unwrap();
        for id in 0..8 {
            get(id);
        }
        get(0); // touch: row 1 is now the oldest
        let before = stats(&e).disk_reads;
        get(100); // evicts row 1
        assert_eq!(stats(&e).disk_reads, before + 1);
        get(0);
        get(2);
        assert_eq!(stats(&e).disk_reads, before + 1, "rows 0 and 2 stayed");
        get(1);
        assert_eq!(stats(&e).disk_reads, before + 2, "row 1 was evicted");
        assert_eq!(stats(&e).resident_rows, 8);
    }

    #[test]
    fn threads_asking_at_once_all_get_the_file_rows() {
        let t = table(GgmlDType::Q8_0);
        let e = open(&t, 4, 40);
        std::thread::scope(|s| {
            for seed in 0..8u64 {
                let (e, t) = (&e, &t);
                s.spawn(move || {
                    for round in 0..20 {
                        let ask = ids(seed * 100 + round, 120, ROWS as u64);
                        if round % 3 == 0 {
                            e.prefetch(&ask);
                        }
                        let mut got = vec![0f32; ask.len() * DIM];
                        e.rows(&ask, &mut got).unwrap();
                        assert_eq!(got, reference(t, &ask), "thread {seed} round {round}");
                    }
                });
            }
        });
        let s = stats(&e);
        assert!(s.resident_rows <= 40);
        assert!(s.lookups >= s.cache_hits && s.disk_reads > 0, "{s:?}");
    }

    #[test]
    fn a_row_past_the_table_and_a_wrong_output_size_are_errors() {
        let t = table(GgmlDType::Q8_0);
        let e = open(&t, 2, 8);
        let mut out = vec![0f32; 2 * DIM];
        let err = e.rows(&[1, ROWS as u64], &mut out).unwrap_err();
        assert!(err.to_string().contains("outside the 300 rows"), "{err}");
        let err = e.rows(&[1, 2], &mut out[..DIM]).unwrap_err();
        assert!(err.to_string().contains("need 128 values"), "{err}");
        // A bad id in a prefetch is ignored; the table still works.
        e.prefetch(&[ROWS as u64 + 5, 3]);
        e.rows(&[3], &mut out[..DIM]).unwrap();
        assert_eq!(out[..DIM], reference(&t, &[3])[..]);
    }

    #[test]
    fn a_truncated_table_and_a_wrong_shape_fail_to_open() {
        let t = table(GgmlDType::Q8_0);
        let cache = EngramCache::default();
        let e = EngramTable::open(&t.content, TENSOR, DIM + 32, &cache)
            .err()
            .unwrap();
        assert!(e.to_string().contains("expected shape"), "{e}");
        let e = EngramTable::open(&t.content, "nope.weight", DIM, &cache)
            .err()
            .unwrap();
        assert!(e.to_string().contains("nope.weight"), "{e}");
        let cut = std::env::temp_dir().join(format!("cuca-engram-cut-{}.gguf", std::process::id()));
        let bytes = std::fs::read(&t.path).unwrap();
        std::fs::write(&cut, &bytes[..bytes.len() - t.row_bytes * 3]).unwrap();
        let content = gguf::Content::open(&cut).unwrap();
        let e = EngramTable::open(&content, TENSOR, DIM, &cache)
            .err()
            .unwrap();
        let _ = std::fs::remove_file(&cut);
        assert!(e.to_string().contains("truncated GGUF"), "{e}");
    }

    #[test]
    fn the_public_bounds_are_validated_at_open() {
        let t = table(GgmlDType::Q8_0);
        for (cache, what) in [
            (
                EngramCache {
                    ram_bytes: 1000,
                    ..EngramCache::default()
                },
                "engram.ram_bytes",
            ),
            (
                EngramCache {
                    reader_threads: 0,
                    ..EngramCache::default()
                },
                "engram.reader_threads",
            ),
            (
                EngramCache {
                    reader_threads: 65,
                    ..EngramCache::default()
                },
                "engram.reader_threads",
            ),
        ] {
            let e = EngramTable::open(&t.content, TENSOR, DIM, &cache)
                .err()
                .unwrap();
            assert!(
                matches!(&e, CucaError::Config(m) if m.contains(what)),
                "{e:?}"
            );
        }
        // The smallest budget holds thousands of rows; a model-sized table
        // is capped at its own row count.
        let e = EngramTable::open(
            &t.content,
            TENSOR,
            DIM,
            &EngramCache {
                ram_bytes: 1 << 20,
                reader_threads: 2,
            },
        )
        .unwrap();
        assert_eq!(stats(&e).capacity_rows, ROWS as u64);
    }

    #[test]
    fn the_lru_keeps_its_links_consistent_under_churn() {
        let mut c = RowCache::new(4, 5);
        let mut model: Vec<u64> = Vec::new(); // most recent last
        let mut rng = Lcg(77);
        let mut buf = [0u8; 4];
        for step in 0..2000 {
            let id = ((rng.values(1, 1.0)[0] + 1.0) * 10.0) as u64;
            if step % 3 == 0 {
                let hit = c.get(id, &mut buf);
                assert_eq!(hit, model.contains(&id), "step {step}");
                if hit {
                    assert_eq!(buf, (id as u32).to_le_bytes());
                    model.retain(|&m| m != id);
                    model.push(id);
                }
            } else {
                c.insert(id, &(id as u32).to_le_bytes());
                model.retain(|&m| m != id);
                model.push(id);
                if model.len() > 5 {
                    model.remove(0);
                }
            }
            assert_eq!(c.len(), model.len());
        }
    }
}
