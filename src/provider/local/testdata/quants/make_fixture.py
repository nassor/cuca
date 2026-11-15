"""Regenerates the quantization fixtures next to this file.

    python make_fixture.py

Requires gguf-py (`pip install gguf`; tested with 2.5.4) and numpy. Writes:

- `quants.bin`: for every block type the engine decodes, whole rows of blocks
  and the f32 values gguf-py's `quants.dequantize` (which mirrors llama.cpp's
  `dequantize_row_*`) returns for them. The blocks are valid by construction:
  every field is an index or a scale, so random bytes are valid blocks as long
  as the f16 / f32 scale fields are finite, which this script writes
  explicitly. Each type has a block of zero data, a block of 0xFF data and
  random blocks, from a fixed seed.
- `sizes.txt`: `GGML_QUANT_SIZES` (type id, name, values per block, bytes per
  block) of the installed gguf-py.
- `grids.txt`: an FNV-1a 64-bit hash of every grid and the sign table as
  gguf-py has them (its own copies of llama.cpp's tables, transcribed
  independently of `make_tables.py`'s source), to be compared with the Rust
  tables entry by entry.

The Rust side is `model/quant/format_tests.rs`. `quants.bin` layout (little
endian): magic `CQF1`, `u32` entries; per entry `u32` ggml type id, `u32`
blocks, `u32` scales, the blocks' bytes, then per scale an `f32` scale, a
`u8` kind (0: the `blocks * block_size` expected `f32` follow; 1: the FNV-1a
64-bit hash of those values' little-endian bytes follows) and the payload.
Kind 0 is written for scale 1.0 (bit-exact llama.cpp values), kind 1 for
the other scales.

The scale argument of `dequant_row` multiplies the value llama.cpp computes,
except for IQ4_NL and MXFP4 (like the legacy blocks) where it multiplies the
block's f32 scale first; the expected values here follow that.
"""

import pathlib
import struct

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as T, quants

HERE = pathlib.Path(__file__).resolve().parent
SEED = 0xC0CA
SCALES = [1.0, 1.37]
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

# Offsets of the f16 scale fields of each block (values: d, then dmin / m).
F16_FIELDS = {
    T.Q4_0: [0],
    T.Q4_1: [0, 2],
    T.Q5_0: [0],
    T.Q5_1: [0, 2],
    T.Q8_0: [0],
    T.Q2_K: [80, 82],
    T.Q3_K: [108],
    T.Q4_K: [0, 2],
    T.Q5_K: [0, 2],
    T.Q6_K: [208],
    T.IQ2_XXS: [0],
    T.IQ2_XS: [0],
    T.IQ2_S: [0],
    T.IQ3_XXS: [0],
    T.IQ3_S: [0],
    T.IQ1_S: [0],
    T.IQ4_NL: [0],
    T.IQ4_XS: [0],
    T.TQ1_0: [52],
    T.TQ2_0: [64],
}
# Q8_K's `d` is an f32 at offset 0.
F32_FIELDS = {T.Q8_K: [0]}
# Types whose scale argument folds into the block scale (llama.cpp's value
# times the scale is not what dequant_row returns): only these two are
# checked at a second scale; the legacy blocks only at 1.0.
FOLDED = {T.IQ4_NL, T.MXFP4}
LEGACY = {T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0}

TYPES = [
    T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0,
    T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.Q8_K,
    T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S,
    T.IQ1_S, T.IQ1_M, T.IQ4_NL, T.IQ4_XS,
    T.TQ1_0, T.TQ2_0, T.MXFP4, T.NVFP4,
]  # fmt: skip


def fnv1a(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def f16(v: float) -> bytes:
    return np.float16(v).tobytes()


def set_scales(t: T, block: bytearray, d: float, dmin: float) -> None:
    for i, off in enumerate(F16_FIELDS.get(t, [])):
        block[off : off + 2] = f16(d if i == 0 else dmin)
    for off in F32_FIELDS.get(t, []):
        block[off : off + 4] = np.float32(d).tobytes()
    if t == T.IQ1_M:
        # The f16 scale is spread over the top nibbles of the four 16-bit
        # scale words at bytes 48..56.
        bits = int(np.float16(d).view(np.uint16))
        for k in range(4):
            word = int.from_bytes(block[48 + 2 * k : 50 + 2 * k], "little")
            word = (word & 0x0FFF) | (((bits >> (4 * k)) & 0xF) << 12)
            block[48 + 2 * k : 50 + 2 * k] = word.to_bytes(2, "little")


def make_blocks(t: T, rng: np.random.Generator, n_random: int) -> list[bytes]:
    _, size = GGML_QUANT_SIZES[t]
    blocks = []
    zeros = bytearray(size)
    set_scales(t, zeros, 1.0, 0.5)
    blocks.append(bytes(zeros))
    ones = bytearray(b"\xff" * size)
    set_scales(t, ones, -0.75, 0.25)
    blocks.append(bytes(ones))
    for _ in range(n_random):
        block = bytearray(rng.integers(0, 256, size, dtype=np.uint8).tobytes())
        d, dmin = rng.normal(0.0, 0.3, 2)
        set_scales(t, block, float(d), float(dmin))
        blocks.append(bytes(block))
    return blocks


def expected(t: T, row: bytes, scale: float) -> np.ndarray:
    data = np.frombuffer(row, dtype=np.uint8).reshape(1, -1)
    if t == T.Q8_K:
        # gguf-py has no Q8_K dequantizer: ggml's is `d * qs[j]` per block
        # (f32 `d`, 256 `i8`, then 16 `i16` sums that decoding ignores).
        blocks = data.reshape(-1, 292)
        d = blocks[:, :4].copy().view(np.float32)
        qs = blocks[:, 4:260].copy().view(np.int8).astype(np.float32)
        values = (d * qs).reshape(-1).astype(np.float32)
    else:
        values = quants.dequantize(data, t).reshape(-1).astype(np.float32)
    if scale == 1.0:
        return values
    s = np.float32(scale)
    if t in FOLDED:
        n_blocks = data.shape[1] // GGML_QUANT_SIZES[t][1]
        blocks = data.reshape(n_blocks, -1)
        if t == T.IQ4_NL:
            d = blocks[:, :2].copy().view(np.float16).astype(np.float32) * s
            kv = np.array(quants.IQ4_NL.kvalues, dtype=np.int8)
            qs = blocks[:, 2:]
            vals = np.concatenate([kv[qs & 0xF], kv[qs >> 4]], axis=1)
        else:
            d = quants.MXFP4.e8m0_to_fp32_half(blocks[:, :1]).astype(np.float32) * s
            kv = np.array(quants.MXFP4.kvalues, dtype=np.int8)
            qs = blocks[:, 1:]
            vals = np.concatenate([kv[qs & 0xF], kv[qs >> 4]], axis=1)
        return (vals.astype(np.float32) * d).reshape(-1).astype(np.float32)
    return (values * s).astype(np.float32)


def write_quants() -> None:
    rng = np.random.default_rng(SEED)
    out = bytearray(b"CQF1")
    out += struct.pack("<I", len(TYPES))
    for t in TYPES:
        block_size, type_size = GGML_QUANT_SIZES[t]
        # About 1,500 values per type: more small blocks, fewer big ones.
        n_random = {256: 4, 64: 14, 32: 30}[block_size]
        blocks = make_blocks(t, rng, n_random)
        row = b"".join(blocks)
        scales = [1.0] if t in LEGACY else SCALES
        out += struct.pack("<III", int(t), len(blocks), len(scales))
        out += row
        for s in scales:
            values = expected(t, row, s)
            assert values.shape == (len(blocks) * block_size,), (t, values.shape)
            # An E8M0 scale of 255 times a quant of 12 overflows to inf in
            # both implementations; a NaN would not have one bit pattern.
            assert not np.isnan(values).any(), t
            payload = values.astype("<f4").tobytes()
            out += struct.pack("<fB", s, 0 if s == 1.0 else 1)
            out += payload if s == 1.0 else struct.pack("<Q", fnv1a(payload))
    (HERE / "quants.bin").write_bytes(bytes(out))


def write_sizes() -> None:
    lines = [
        f"{int(t)} {t.name} {block} {size}"
        for t, (block, size) in sorted(GGML_QUANT_SIZES.items(), key=lambda kv: int(kv[0]))
    ]
    (HERE / "sizes.txt").write_text("\n".join(lines) + "\n")


def write_grids() -> None:
    lines = []
    for name, cls in [
        ("iq2xxs_grid", quants.IQ2_XXS),
        ("iq2xs_grid", quants.IQ2_XS),
        ("iq2s_grid", quants.IQ2_S),
        ("iq3xxs_grid", quants.IQ3_XXS),
        ("iq3s_grid", quants.IQ3_S),
        ("iq1s_grid", quants.IQ1_S),
    ]:
        cls.init_grid()
        grid = cls.grid.reshape(-1, cls.grid_shape[1])
        data = grid.astype(np.int8).view(np.uint8).tobytes()
        lines.append(f"{name} {grid.shape[0]} {fnv1a(data):016x}")
    lines.append(f"ksigns_iq2xs 128 {fnv1a(quants.IQ2_XXS.ksigns):016x}")
    (HERE / "grids.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_quants()
    write_sizes()
    write_grids()
