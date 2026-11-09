//! The expert tiers and the Engram table's row cache on the tiny
//! hyper-connection model (MXFP4 experts, an engram table of 180 rows).

use super::super::tiny::{TinyDeepseek41, TinyModel, write_deepseek41};
use super::*;
use crate::provider::local::gguf::Content;
use crate::provider::local::{EngramCache, ExpertHeat, ExpertPlacement, ExpertTiers};

/// Context of the tiny model's cache.
const CTX: usize = 96;

/// Bytes of one expert of the MXFP4 tiny model: three matrices of 1,088
/// bytes (`[32, 64]`, `[32, 64]` and `[64, 32]` in blocks of 17 bytes per 32).
const EXPERT_BYTES: u64 = 3 * 1088;
/// Layers of the tiny model, all with experts.
const LAYERS: u64 = 6;

fn mxfp4() -> TinyModel {
    write_deepseek41(&TinyDeepseek41 {
        mxfp4: true,
        ..TinyDeepseek41::default()
    })
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 5 + 3 + i / 7) % 16).collect()
}

/// `tokens` in chunks of `size`, the logits of every position.
fn chunked(m: &mut Transformer, tokens: &[u32], size: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, chunk) in tokens.chunks(size).enumerate() {
        out.extend(
            m.forward_ext(chunk, i * size, Want::All, Hidden::None, true)
                .unwrap()
                .logits
                .unwrap(),
        );
    }
    out
}

fn load(model: &TinyModel, experts: &ExpertPlacement, engram: &EngramCache) -> Transformer {
    let content = Content::open(&model.path).unwrap();
    Transformer::load_placed(&content, CTX, experts, &Sizing::single(CTX), engram).unwrap()
}

fn tiered(slots: u64, warm_from: Option<ExpertHeat>) -> ExpertPlacement {
    ExpertPlacement::Tiered(ExpertTiers {
        ram_bytes: slots * LAYERS * EXPERT_BYTES,
        loader_threads: 2,
        warm_from,
        ..ExpertTiers::default()
    })
}

#[test]
fn a_warm_start_serves_the_first_pass_from_the_experts_it_loaded() {
    let model = mxfp4();
    let p = prompt(50);
    let resident = chunked(
        &mut load(&model, &ExpertPlacement::Resident, &EngramCache::default()),
        &p,
        16,
    );

    // Five of the eight experts of every layer fit.
    let mut cold = load(&model, &tiered(5, None), &EngramCache::default());
    let tiers = cold.expert_tiers().unwrap();
    assert_eq!(chunked(&mut cold, &p, 16), resident, "cold == resident");
    let cold_stats = tiers.stats();
    assert!(cold_stats.disk_reads > 0, "{cold_stats:?}");
    let heat = tiers.heat();

    let mut warm = load(&model, &tiered(5, Some(heat)), &EngramCache::default());
    let warm_tiers = warm.expert_tiers().unwrap();
    let loaded = warm_tiers.stats();
    assert!(
        loaded.disk_reads > 0,
        "the warm start read the hottest experts"
    );
    assert!(loaded.ram_resident_bytes <= 5 * LAYERS * EXPERT_BYTES);
    assert_eq!(chunked(&mut warm, &p, 16), resident, "warm == resident");
    let after = warm_tiers.stats();
    let during = after.disk_reads - loaded.disk_reads;
    assert!(
        during < cold_stats.disk_reads,
        "the first pass read {during} experts after a warm start, {} cold",
        cold_stats.disk_reads
    );
    assert!(after.ram_hits - loaded.ram_hits > cold_stats.ram_hits);
    assert_eq!(after.lookups - loaded.lookups, cold_stats.lookups);
}

#[test]
fn a_second_pass_over_the_same_prompt_reads_fewer_engram_rows_with_equal_output() {
    let model = mxfp4();
    let p = prompt(50);
    let mut m = load(&model, &ExpertPlacement::Resident, &EngramCache::default());
    let counters = m.engram_counters().expect("the model has an engram table");
    let first = chunked(&mut m, &p, 16);
    let one = counters.snapshot();
    assert!(
        one.disk_reads > 0 && one.lookups >= one.disk_reads,
        "{one:?}"
    );
    assert_eq!(one.resident_rows, one.disk_reads);

    assert_eq!(m.truncate_cache(0), 0);
    let second = chunked(&mut m, &p, 16);
    let two = counters.snapshot();
    assert_eq!(second, first, "the cache changes no bit");
    assert!(
        two.disk_reads - one.disk_reads < one.disk_reads,
        "{one:?} then {two:?}"
    );
    assert_eq!(two.disk_reads, one.disk_reads, "every row was cached");
    assert_eq!(two.lookups, 2 * one.lookups);
    assert!(two.hit_rate().unwrap() > one.hit_rate().unwrap());
}

#[test]
fn a_prefetch_of_the_next_chunk_reads_no_row_twice_and_changes_no_bit() {
    let model = mxfp4();
    let p = prompt(50);
    let mut plain = load(&model, &ExpertPlacement::Resident, &EngramCache::default());
    let want = chunked(&mut plain, &p, 16);
    let plain_reads = plain.engram_counters().unwrap().snapshot().disk_reads;

    let mut m = load(&model, &ExpertPlacement::Resident, &EngramCache::default());
    for (i, chunk) in p.chunks(16).enumerate() {
        // The next chunk's rows are requested while this one runs.
        let end = (i + 1) * 16;
        if end < p.len() {
            m.engram_prefetch(&p[..end], &p[end..(end + 16).min(p.len())]);
        }
        let got = m
            .forward_ext(chunk, i * 16, Want::All, Hidden::None, true)
            .unwrap()
            .logits
            .unwrap();
        assert_eq!(
            got,
            want[i * 16 * 16..(i * 16 + chunk.len()) * 16],
            "chunk {i}"
        );
    }
    let s = m.engram_counters().unwrap().snapshot();
    assert_eq!(s.disk_reads, plain_reads, "{s:?}");
    assert!(s.cache_hits >= plain.engram_counters().unwrap().snapshot().cache_hits);
}

#[test]
fn the_engram_cache_bounds_are_checked_at_load() {
    let model = mxfp4();
    let content = Content::open(&model.path).unwrap();
    for (cache, what) in [
        (
            EngramCache {
                ram_bytes: 0,
                ..EngramCache::default()
            },
            "engram.ram_bytes",
        ),
        (
            EngramCache {
                reader_threads: 65,
                ..EngramCache::default()
            },
            "engram.reader_threads",
        ),
    ] {
        let e = Transformer::load_placed(
            &content,
            CTX,
            &ExpertPlacement::Resident,
            &Sizing::single(CTX),
            &cache,
        )
        .err()
        .unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains(what)),
            "{e:?}"
        );
    }
}
