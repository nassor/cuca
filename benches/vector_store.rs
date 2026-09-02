//! Retrieval benchmark for `service-vector-store`.
//!
//! The similarity scan is the crate's hottest explicit-call path, so it is
//! pinned against a baseline rather than trusted. Each `retrieve_simd_*` bench
//! drives [`InMemoryVectorStore::retrieve_embedding`], which runs the crate's
//! `wide::f32x8` kernel over the slot arena; each matching `scan_scalar_*`
//! bench runs a plain `for`-loop dot product over identical vectors plus the
//! same `select_nth_unstable_by` ranking. The scalar loop is a real baseline,
//! not a handicapped one: LLVM cannot auto-vectorize a serial `f32`
//! accumulation without fast-math, which is exactly the code the SIMD kernel
//! replaces. It is also slightly *cheaper* than the store path, which clones
//! the winning messages, so the comparison understates the kernel's advantage.
//!
//! # Run
//!
//! ```sh
//! cargo bench --features provider-openai,service-vector-store
//! RUSTFLAGS="-C target-cpu=native" cargo bench --features provider-openai,service-vector-store
//! ```
//!
//! The second form is the interesting one: `wide` selects its implementation at
//! compile time, so a baseline `x86-64` build lowers `f32x8` to two `f32x4`
//! halves, and only `+avx,+fma` unlocks the 256-bit single-rounding path.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cuca::types::UnifiedMessage;
use cuca::{Embedder, InMemoryVectorStore, PluginError, VectorStoreConfig};

fn main() {
    divan::main();
}

/// Hits returned per query; the realistic recall width, and small enough that
/// ranking cost stays dominated by the scan.
const K: usize = 8;

/// Deterministic filler, identical to the kernel's unit test: no `rand`
/// dependency, and the same bytes on every machine and every run.
fn sample(index: usize) -> f32 {
    ((index * 31 + 7) % 17) as f32 / 17.0 - 0.5
}

/// `count` vectors of width `dimensions`, unit-normalized, laid out
/// contiguously exactly as the store's arena lays them out.
fn arena(count: usize, dimensions: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count * dimensions];
    for slot in 0..count {
        let row = &mut out[slot * dimensions..(slot + 1) * dimensions];
        for (index, value) in row.iter_mut().enumerate() {
            *value = sample(slot * dimensions + index);
        }
        normalize(row);
    }
    out
}

fn unit_query(dimensions: usize) -> Vec<f32> {
    let mut query: Vec<f32> = (0..dimensions).map(|i| sample(i + 11)).collect();
    normalize(&mut query);
    query
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inverse = 1.0 / norm;
        for x in vector.iter_mut() {
            *x *= inverse;
        }
    }
}

/// Hands back staged vectors in insertion order, so the store's arena holds
/// exactly the bytes the scalar baseline scans.
struct StagedEmbedder {
    vectors: Vec<Vec<f32>>,
    cursor: AtomicUsize,
}

impl Embedder for StagedEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, PluginError> {
        let index = self.cursor.fetch_add(1, Ordering::Relaxed) % self.vectors.len();
        Ok(self.vectors[index].clone())
    }
}

/// A store holding `count` entries of width `dimensions`.
fn filled_store(count: usize, dimensions: usize) -> InMemoryVectorStore {
    let rows = arena(count, dimensions);
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|slot| rows[slot * dimensions..(slot + 1) * dimensions].to_vec())
        .collect();
    let store = InMemoryVectorStore::new(
        VectorStoreConfig::new(count, dimensions, 64 * 1024).expect("config must build"),
        Arc::new(StagedEmbedder {
            vectors,
            cursor: AtomicUsize::new(0),
        }),
    )
    .expect("store must build");
    let turns: Vec<UnifiedMessage> = (0..count)
        .map(|slot| UnifiedMessage::user(format!("turn {slot}")))
        .collect();
    cuca::VectorStore::store_turns(&store, "bench", &turns).expect("the batch must be accepted");
    store
}

/// The scalar counterpart of the store's scan: serial dot products plus the
/// identical exact top-k ranking.
fn scalar_scan(rows: &[f32], query: &[f32], dimensions: usize, k: usize) -> Vec<(f32, u64, usize)> {
    let count = rows.len() / dimensions;
    let mut scored: Vec<(f32, u64, usize)> = Vec::with_capacity(count);
    for slot in 0..count {
        let row = &rows[slot * dimensions..(slot + 1) * dimensions];
        let mut sum = 0.0f32;
        for (x, y) in query.iter().zip(row) {
            sum += x * y;
        }
        scored.push((sum, slot as u64, slot));
    }
    let rank =
        |a: &(f32, u64, usize), b: &(f32, u64, usize)| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1));
    if scored.len() > k {
        scored.select_nth_unstable_by(k - 1, rank);
        scored.truncate(k);
    }
    scored.sort_unstable_by(rank);
    scored
}

fn bench_retrieve(bencher: divan::Bencher, count: usize, dimensions: usize) {
    let store = filled_store(count, dimensions);
    let query = unit_query(dimensions);
    bencher.bench(|| {
        store
            .retrieve_embedding(None, divan::black_box(&query), K)
            .expect("query must run")
    });
}

fn bench_scalar(bencher: divan::Bencher, count: usize, dimensions: usize) {
    let rows = arena(count, dimensions);
    let query = unit_query(dimensions);
    bencher.bench(|| {
        scalar_scan(
            divan::black_box(&rows),
            divan::black_box(&query),
            dimensions,
            K,
        )
    });
}

#[divan::bench]
fn retrieve_simd_n1024_d384(bencher: divan::Bencher) {
    bench_retrieve(bencher, 1024, 384);
}

#[divan::bench]
fn scan_scalar_n1024_d384(bencher: divan::Bencher) {
    bench_scalar(bencher, 1024, 384);
}

#[divan::bench]
fn retrieve_simd_n8192_d768(bencher: divan::Bencher) {
    bench_retrieve(bencher, 8192, 768);
}

#[divan::bench]
fn scan_scalar_n8192_d768(bencher: divan::Bencher) {
    bench_scalar(bencher, 8192, 768);
}

#[divan::bench]
fn retrieve_simd_n8192_d1536(bencher: divan::Bencher) {
    bench_retrieve(bencher, 8192, 1536);
}

#[divan::bench]
fn scan_scalar_n8192_d1536(bencher: divan::Bencher) {
    bench_scalar(bencher, 8192, 1536);
}

/// Insert throughput, including projection, embedding, normalization, and the
/// commit under the lock.
#[divan::bench]
fn store_turns_batch64_d768(bencher: divan::Bencher) {
    const BATCH: usize = 64;
    const DIMENSIONS: usize = 768;
    let store = filled_store(BATCH, DIMENSIONS);
    let turns: Vec<UnifiedMessage> = (0..BATCH)
        .map(|slot| UnifiedMessage::user(format!("inserted turn {slot}")))
        .collect();
    bencher.bench(|| {
        cuca::VectorStore::store_turns(&store, "bench", divan::black_box(&turns))
            .expect("the batch must be accepted")
    });
}
