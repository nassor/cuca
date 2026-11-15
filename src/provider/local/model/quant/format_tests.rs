//! Tests of the block decoders against gguf-py (`testdata/quants/make_fixture.py`
//! regenerates the fixtures), of the integer view of every block
//! ([`Unpacked`]) against the f32 decoders, and of the block geometry of
//! [`GgufType`] against `GGML_QUANT_SIZES`.

use super::block::{MAX_BLOCK, Unpacked};
use super::iq_tables::*;
use super::{dequant_row, unpack_block};
use crate::provider::local::gguf::GgufType;
use candle_core::quantized::GgmlDType;

const QUANTS: &[u8] = include_bytes!("../../testdata/quants/quants.bin");
const SIZES: &str = include_str!("../../testdata/quants/sizes.txt");
const GRIDS: &str = include_str!("../../testdata/quants/grids.txt");

/// What gguf-py returned for a row at one scale.
enum Expected {
    /// The values' bits (scale 1.0: llama.cpp's own values).
    Values(Vec<u32>),
    /// The FNV-1a 64-bit hash of the values' little-endian bytes.
    Hash(u64),
}

/// One fixture entry: whole rows of blocks and the expected values per scale.
struct Entry {
    ty: GgufType,
    blocks: usize,
    bytes: &'static [u8],
    scales: Vec<(f32, Expected)>,
}

/// FNV-1a, 64 bits.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

fn fixture() -> Vec<Entry> {
    struct Cursor(&'static [u8]);
    impl Cursor {
        fn take(&mut self, n: usize) -> &'static [u8] {
            let (head, rest) = self.0.split_at(n);
            self.0 = rest;
            head
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take(4).try_into().unwrap())
        }
    }
    let mut c = Cursor(QUANTS);
    assert_eq!(c.take(4), b"CQF1");
    let n = c.u32();
    (0..n)
        .map(|_| {
            let ty = GgufType::from_id(c.u32());
            let blocks = c.u32() as usize;
            let n_scales = c.u32();
            let bytes = c.take(blocks * ty.type_size().unwrap());
            let scales = (0..n_scales)
                .map(|_| {
                    let scale = f32::from_bits(c.u32());
                    let kind = c.take(1)[0];
                    let expected = if kind == 0 {
                        let raw = c.take(blocks * ty.block_size().unwrap() * 4);
                        Expected::Values(
                            raw.as_chunks::<4>()
                                .0
                                .iter()
                                .map(|w| u32::from_le_bytes(*w))
                                .collect(),
                        )
                    } else {
                        Expected::Hash(u64::from_le_bytes(c.take(8).try_into().unwrap()))
                    };
                    (scale, expected)
                })
                .collect();
            Entry {
                ty,
                blocks,
                bytes,
                scales,
            }
        })
        .collect()
}

/// A deterministic byte generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn block(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| (self.next() >> 56) as u8).collect()
    }
}

#[test]
fn every_type_matches_gguf_py_bit_for_bit() {
    let entries = fixture();
    let ids: Vec<u32> = entries.iter().map(|e| e.ty.id()).collect();
    assert_eq!(
        ids,
        [
            2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 22, 18, 21, 19, 29, 20, 23, 34, 35, 39,
            40
        ]
    );
    for e in &entries {
        let name = e.ty.name();
        let len = e.blocks * e.ty.block_size().unwrap();
        assert_eq!(
            e.bytes.len(),
            e.blocks * e.ty.type_size().unwrap(),
            "{name}"
        );
        let mut out = vec![0f32; len];
        for (scale, expected) in &e.scales {
            out.fill(f32::NAN);
            dequant_row(e.ty, e.bytes, *scale, &mut out).unwrap();
            match expected {
                Expected::Values(bits) => {
                    assert_eq!(*scale, 1.0);
                    let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
                    if let Some(i) = got.iter().zip(bits).position(|(a, b)| a != b) {
                        panic!(
                            "{name}: element {i} (block {}) is {:e} ({:#010x}), gguf-py has {:e} ({:#010x})",
                            i / e.ty.block_size().unwrap(),
                            out[i],
                            got[i],
                            f32::from_bits(bits[i]),
                            bits[i]
                        );
                    }
                    assert_eq!(got.len(), bits.len(), "{name}");
                }
                Expected::Hash(hash) => {
                    let got = fnv1a(out.iter().flat_map(|v| v.to_le_bytes()));
                    assert_eq!(got, *hash, "{name} at scale {scale}");
                }
            }
        }
    }
}

#[test]
fn the_fixture_has_every_type_with_a_scale_check_where_it_applies() {
    for e in fixture() {
        let scaled = e.scales.iter().filter(|(s, _)| *s != 1.0).count();
        let legacy = matches!(
            e.ty,
            GgufType::Ggml(
                GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q8_0
            )
        );
        assert_eq!(scaled, usize::from(!legacy), "{}", e.ty.name());
        assert!(e.blocks >= 6, "{}", e.ty.name());
    }
}

/// The integer view of every fixture block rebuilds the decoded values, and its
/// folded quants stay inside the documented bound.
#[test]
fn unpacked_blocks_rebuild_the_decoded_values() {
    let mut unpacked = Unpacked::default();
    let mut tested = 0;
    for e in fixture() {
        let Some(&(_, bound)) = BOUNDS.iter().find(|(t, _)| *t == e.ty) else {
            continue;
        };
        let (bs, ts) = (e.ty.block_size().unwrap(), e.ty.type_size().unwrap());
        let mut decoded = vec![0f32; e.blocks * bs];
        dequant_row(e.ty, e.bytes, 1.0, &mut decoded).unwrap();
        for (i, block) in e.bytes.chunks_exact(ts).enumerate() {
            unpack_block(e.ty, block, &mut unpacked).unwrap();
            assert_eq!(
                (unpacked.len, bs % unpacked.group),
                (bs, 0),
                "{}",
                e.ty.name()
            );
            let mut folded = [0i16; MAX_BLOCK];
            unpacked.fold_into(&mut folded);
            let worst = folded[..bs]
                .iter()
                .map(|f| i32::from(f.unsigned_abs()))
                .max();
            assert!(worst.unwrap() <= bound, "{} block {i}", e.ty.name());
            let mut rebuilt = [0f32; MAX_BLOCK];
            unpacked.reconstruct(&mut rebuilt);
            // The f32 values are products rounded once, so the integer form
            // gives the same value; only the sign of a zero may differ.
            for (j, (&a, &b)) in rebuilt[..bs].iter().zip(&decoded[i * bs..]).enumerate() {
                assert!(
                    a == b,
                    "{} block {i} value {j}: {a:e} vs {b:e}",
                    e.ty.name()
                );
            }
        }
        tested += 1;
    }
    assert_eq!(tested, BOUNDS.len());
}

/// `(type, max |sc * q|)`: the table of `block.rs`.
const BOUNDS: [(GgufType, i32); 19] = [
    (GgufType::Ggml(GgmlDType::Q2K), 45),
    (GgufType::Ggml(GgmlDType::Q3K), 128),
    (GgufType::Ggml(GgmlDType::Q4K), 945),
    (GgufType::Ggml(GgmlDType::Q5K), 1953),
    (GgufType::Ggml(GgmlDType::Q6K), 4096),
    (GgufType::Ggml(GgmlDType::Q8K), 128),
    (GgufType::Iq2Xxs, 1333),
    (GgufType::Iq2Xs, 1333),
    (GgufType::Iq2S, 1333),
    (GgufType::Iq3Xxs, 1922),
    (GgufType::Iq3S, 465),
    (GgufType::Iq1S, 135),
    (GgufType::Iq1M, 135),
    (GgufType::Iq4Xs, 4064),
    (GgufType::Iq4Nl, 127),
    (GgufType::Tq1_0, 1),
    (GgufType::Tq2_0, 2),
    (GgufType::Mxfp4, 12),
    (GgufType::Nvfp4, 12),
];

/// The ranges of every integer field of every type (`q`, `sc`, `mn` as
/// `(min, max)`): what `block.rs` documents, reached by random blocks and
/// never exceeded.
#[test]
fn unpacked_fields_cover_exactly_the_documented_ranges() {
    type Range = (i32, i32);
    // (type, q, sc, mn)
    let ranges: [(GgufType, Range, Range, Range); 19] = [
        (GgufType::Ggml(GgmlDType::Q2K), (0, 3), (0, 15), (0, 15)),
        (GgufType::Ggml(GgmlDType::Q3K), (-4, 3), (-32, 31), (0, 0)),
        (GgufType::Ggml(GgmlDType::Q4K), (0, 15), (0, 63), (0, 63)),
        (GgufType::Ggml(GgmlDType::Q5K), (0, 31), (0, 63), (0, 63)),
        (
            GgufType::Ggml(GgmlDType::Q6K),
            (-32, 31),
            (-128, 127),
            (0, 0),
        ),
        (GgufType::Ggml(GgmlDType::Q8K), (-128, 127), (1, 1), (0, 0)),
        (GgufType::Iq2Xxs, (-43, 43), (1, 31), (0, 0)),
        (GgufType::Iq2Xs, (-43, 43), (1, 31), (0, 0)),
        (GgufType::Iq2S, (-43, 43), (1, 31), (0, 0)),
        (GgufType::Iq3Xxs, (-62, 62), (1, 31), (0, 0)),
        (GgufType::Iq3S, (-15, 15), (1, 31), (0, 0)),
        (GgufType::Iq1S, (-9, 9), (1, 15), (0, 0)),
        (GgufType::Iq1M, (-9, 9), (1, 15), (0, 0)),
        (GgufType::Iq4Xs, (-127, 113), (-32, 31), (0, 0)),
        (GgufType::Iq4Nl, (-127, 113), (1, 1), (0, 0)),
        (GgufType::Tq1_0, (-1, 1), (1, 1), (0, 0)),
        (GgufType::Tq2_0, (-1, 2), (1, 1), (0, 0)),
        (GgufType::Mxfp4, (-12, 12), (1, 1), (0, 0)),
        (GgufType::Nvfp4, (-12, 12), (1, 1), (0, 0)),
    ];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut unpacked = Unpacked::default();
    for (ty, q, sc, mn) in ranges {
        let name = ty.name();
        let ts = ty.type_size().unwrap();
        let mut seen = [(i32::MAX, i32::MIN); 3];
        let mut note = |slot: usize, v: i32| {
            seen[slot] = (seen[slot].0.min(v), seen[slot].1.max(v));
        };
        let mut blocks = vec![vec![0u8; ts], vec![0xFF; ts]];
        blocks.extend((0..3000).map(|_| rng.block(ts)));
        for block in &blocks {
            unpack_block(ty, block, &mut unpacked).unwrap();
            for &v in &unpacked.q[..unpacked.len] {
                note(0, i32::from(v));
            }
            for g in 0..unpacked.groups() {
                note(1, i32::from(unpacked.sc[g]));
                note(2, i32::from(unpacked.mn[g]));
            }
        }
        assert_eq!(seen, [q, sc, mn], "{name}");
        let corners = [q.0 * sc.0, q.0 * sc.1, q.1 * sc.0, q.1 * sc.1];
        let worst = corners.iter().map(|c| c.abs()).max().unwrap();
        let (_, bound) = BOUNDS.iter().find(|(t, _)| *t == ty).unwrap();
        assert_eq!(worst, *bound, "{name}");
        // A kernel accumulates `i16 * i8` products in `i32`: 256 values.
        assert!(i64::from(worst) * 128 * 256 < i64::from(i32::MAX), "{name}");
    }
}

#[test]
fn the_grids_equal_gguf_pys_entry_for_entry() {
    let bytes64 = |g: &[u64]| fnv1a(g.iter().flat_map(|v| v.to_le_bytes()));
    let bytes32 = |g: &[u32]| fnv1a(g.iter().flat_map(|v| v.to_le_bytes()));
    let hashes = [
        ("iq2xxs_grid", 256, bytes64(&IQ2XXS_GRID)),
        ("iq2xs_grid", 512, bytes64(&IQ2XS_GRID)),
        ("iq2s_grid", 1024, bytes64(&IQ2S_GRID)),
        ("iq3xxs_grid", 256, bytes32(&IQ3XXS_GRID)),
        ("iq3s_grid", 512, bytes32(&IQ3S_GRID)),
        ("iq1s_grid", 2048, bytes64(&IQ1S_GRID)),
        ("ksigns_iq2xs", 128, fnv1a(KSIGNS_IQ2XS)),
    ];
    for (name, count, hash) in hashes {
        let line = GRIDS
            .lines()
            .find(|l| l.starts_with(name))
            .unwrap_or_else(|| panic!("grids.txt has no {name}"));
        let fields: Vec<&str> = line.split(' ').collect();
        assert_eq!(fields[1].parse::<usize>().unwrap(), count, "{name}");
        assert_eq!(u64::from_str_radix(fields[2], 16).unwrap(), hash, "{name}");
    }
}

#[test]
fn the_tables_have_llama_cpps_structure() {
    // 7 sign bits plus an odd-parity bit; bit `j` of a sign byte is mask `j`.
    for (i, &s) in KSIGNS_IQ2XS.iter().enumerate() {
        assert_eq!(usize::from(s & 0x7F), i);
        assert_eq!(u32::from(s >> 7), (i as u32).count_ones() % 2, "{i}");
    }
    assert_eq!(KMASK_IQ2XS, [1, 2, 4, 8, 16, 32, 64, 128]);
    let values = |grid: &[u64]| -> Vec<u8> {
        let mut v: Vec<u8> = grid.iter().flat_map(|g| g.to_le_bytes()).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    for grid in [&IQ2XXS_GRID[..], &IQ2XS_GRID[..], &IQ2S_GRID[..]] {
        assert_eq!(values(grid), [0x08, 0x19, 0x2b]);
    }
    assert_eq!(values(&IQ1S_GRID), [0x00, 0x01, 0xFF]);
    let values32 = |grid: &[u32]| -> Vec<u8> {
        let mut v: Vec<u8> = grid.iter().flat_map(|g| g.to_le_bytes()).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    assert_eq!(values32(&IQ3XXS_GRID), [4, 12, 20, 28, 36, 44, 52, 62]);
    assert_eq!(values32(&IQ3S_GRID), [1, 3, 5, 7, 9, 11, 13, 15]);
    // No two entries of a grid are equal: an index names one 8-value group.
    for grid in [
        &IQ2XXS_GRID[..],
        &IQ2XS_GRID[..],
        &IQ2S_GRID[..],
        &IQ1S_GRID[..],
    ] {
        let mut sorted = grid.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), grid.len());
    }
    assert_eq!(KVALUES_IQ4NL[0], -127);
    assert_eq!(KVALUES_IQ4NL[15], 113);
    assert_eq!(KVALUES_FP4[7], 12);
    assert_eq!(KVALUES_FP4[15], -12);
}

/// `GgufType`'s geometry equals `GGML_QUANT_SIZES` of gguf-py for every type it
/// has, and every other type id gguf-py lists is one the engine does not read.
#[test]
fn block_geometry_equals_gguf_pys_quant_sizes() {
    let mut known = 0;
    for line in SIZES.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let (id, name) = (f[0].parse::<u32>().unwrap(), f[1]);
        let (block, size) = (
            f[2].parse::<usize>().unwrap(),
            f[3].parse::<usize>().unwrap(),
        );
        let ty = GgufType::from_id(id);
        if let GgufType::Unknown(_) = ty {
            // I16, I32, I64, F64 and Q1_0: no engine path reads them.
            assert!(
                [25, 26, 27, 28, 41].contains(&id),
                "{name} ({id}) is not read"
            );
            continue;
        }
        known += 1;
        // gguf-py lists Q8_1 at 40 bytes; ggml-common.h's `block_q8_1` is two
        // f16 and 32 quants, 36 bytes, which is candle's size (an activation
        // format: no file stores it).
        let size = if id == 9 { 36 } else { size };
        assert_eq!(ty.block_size(), Some(block), "{name}");
        assert_eq!(ty.type_size(), Some(size), "{name}");
        assert_eq!(ty.byte_len(block * 3), Some(size * 3), "{name}");
        assert_eq!(ty.byte_len(block * 3 + 1).is_some(), block == 1, "{name}");
        assert_eq!(ty.id(), id, "{name}");
    }
    assert_eq!(known, 29);
}

#[test]
fn new_types_reject_rows_that_are_not_whole_blocks() {
    let mut rng = Rng(7);
    for (ty, _) in BOUNDS {
        let (bs, ts) = (ty.block_size().unwrap(), ty.type_size().unwrap());
        let bytes = rng.block(2 * ts);
        let mut out = vec![0f32; 2 * bs];
        dequant_row(ty, &bytes, 1.0, &mut out).unwrap();
        let name = ty.name();
        assert!(
            dequant_row(ty, &bytes[..2 * ts - 1], 1.0, &mut out).is_err(),
            "{name}"
        );
        assert!(
            dequant_row(ty, &bytes[ts..], 1.0, &mut out).is_err(),
            "{name}"
        );
        assert!(
            dequant_row(ty, &bytes, 1.0, &mut out[..2 * bs - 1]).is_err(),
            "{name}"
        );
        let mut unpacked = Unpacked::default();
        assert!(
            unpack_block(ty, &bytes[..ts], &mut unpacked).is_ok(),
            "{name}"
        );
        assert!(
            unpack_block(ty, &bytes[..ts - 1], &mut unpacked).is_err(),
            "{name}"
        );
        assert!(unpack_block(ty, &bytes, &mut unpacked).is_err(), "{name}");
    }
    let mut unpacked = Unpacked::default();
    assert!(unpack_block(GgmlDType::Q4_0, &[0; 18], &mut unpacked).is_err());
    assert!(unpack_block(GgufType::Unknown(31), &[0; 18], &mut unpacked).is_err());
}

/// A power-of-two scale changes no rounding: every type decodes a row at 4.0 to
/// exactly four times its values at 1.0, whichever way the type applies the
/// scale (to the block scale or to the result).
#[test]
fn a_power_of_two_scale_is_exact_in_every_type() {
    for e in fixture() {
        let n = e.blocks * e.ty.block_size().unwrap();
        let (mut one, mut four) = (vec![0f32; n], vec![0f32; n]);
        dequant_row(e.ty, e.bytes, 1.0, &mut one).unwrap();
        dequant_row(e.ty, e.bytes, 4.0, &mut four).unwrap();
        for (i, (a, b)) in one.iter().zip(&four).enumerate() {
            // The largest E8M0 scales overflow to infinity on both sides.
            assert!(
                *b == a * 4.0,
                "{} value {i}: {a:e} x 4 is not {b:e}",
                e.ty.name()
            );
        }
    }
}
