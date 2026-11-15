"""Regenerates `model/quant/iq_tables.rs` from llama.cpp's `ggml-common.h`.

    python make_tables.py [path/to/ggml-common.h]

Without an argument the header is fetched from llama.cpp at the pinned commit
below. The grids are parsed out of the header and written as Rust `const`
arrays; nothing is typed by hand. When gguf-py is importable the script also
checks the three 2-bit grids, both 3-bit grids and the IQ1 grid against
gguf-py's own (independently transcribed) copies, and stops on any mismatch.
"""

import pathlib
import re
import sys
import urllib.request

COMMIT = "e574a17b062be51b98e239ccd3d3743e3d902333"
URL = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{COMMIT}/ggml/src/ggml-common.h"
OUT = pathlib.Path(__file__).resolve().parents[2] / "model" / "quant" / "iq_tables.rs"

# (header name, Rust name, element type, count, doc)
TABLES = [
    ("kmask_iq2xs", "KMASK_IQ2XS", "u8", 8, "Bit `j` of a sign byte negates element `j` of an 8-value group."),
    ("ksigns_iq2xs", "KSIGNS_IQ2XS", "u8", 128, "The 8 sign bits of a 7-bit sign index: the 7 bits plus an odd-parity bit in bit 7."),
    ("iq2xxs_grid", "IQ2XXS_GRID", "u64", 256, "IQ2_XXS grid: 8 values per entry, entry byte `j` is element `j` (little endian), each one of 0x08, 0x19, 0x2b."),
    ("iq2xs_grid", "IQ2XS_GRID", "u64", 512, "IQ2_XS grid: 8 values per entry, entry byte `j` is element `j` (little endian), each one of 0x08, 0x19, 0x2b."),
    ("iq2s_grid", "IQ2S_GRID", "u64", 1024, "IQ2_S grid: 8 values per entry, entry byte `j` is element `j` (little endian), each one of 0x08, 0x19, 0x2b."),
    ("iq3xxs_grid", "IQ3XXS_GRID", "u32", 256, "IQ3_XXS grid: 4 values per entry, entry byte `j` is element `j` (little endian), each one of 4, 12, 20, 28, 36, 44, 52, 62."),
    ("iq3s_grid", "IQ3S_GRID", "u32", 512, "IQ3_S grid: 4 values per entry, entry byte `j` is element `j` (little endian), each odd, 1 to 15."),
    ("iq1s_grid", "IQ1S_GRID", "u64", 2048, "IQ1_S and IQ1_M grid: 8 values per entry, entry byte `j` is element `j` as an `i8` (little endian), each one of -1, 0, 1."),
    ("kvalues_iq4nl", "KVALUES_IQ4NL", "i8", 16, "IQ4_NL and IQ4_XS code table."),
    ("kvalues_fp4", "KVALUES_FP4", "i8", 16, "E2M1 values doubled, shared by MXFP4 and NVFP4."),
]


def parse(text: str, name: str, count: int) -> list[int]:
    m = re.search(
        r"GGML_TABLE_BEGIN\(\w+, " + re.escape(name) + r", [^)]*\)(.*?)GGML_TABLE_END\(\)",
        text,
        re.S,
    )
    assert m, name
    body = re.sub(r"//[^\n]*", "", m.group(1))
    vals = [int(tok, 0) for tok in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", body)]
    assert len(vals) == count, (name, len(vals), count)
    return vals


def crosscheck(tables: dict[str, list[int]]) -> None:
    try:
        import numpy as np
        from gguf import quants
    except ImportError:
        print("gguf-py not importable: grids not cross-checked", file=sys.stderr)
        return
    pairs = [
        ("iq2xxs_grid", quants.IQ2_XXS, 8, "u64"),
        ("iq2xs_grid", quants.IQ2_XS, 8, "u64"),
        ("iq2s_grid", quants.IQ2_S, 8, "u64"),
        ("iq3xxs_grid", quants.IQ3_XXS, 4, "u32"),
        ("iq3s_grid", quants.IQ3_S, 4, "u32"),
        ("iq1s_grid", quants.IQ1_S, 8, "u64"),
    ]
    for name, cls, width, _ in pairs:
        cls.init_grid()
        grid = cls.grid.reshape(-1, width)
        mine = np.array(
            [[(v >> (8 * j)) & 0xFF for j in range(width)] for v in tables[name]],
            dtype=np.uint8,
        )
        if name == "iq1s_grid":
            mine = mine.view(np.int8).astype(np.float32)
        else:
            mine = mine.astype(np.float32)
        assert grid.shape == mine.shape, (name, grid.shape, mine.shape)
        assert np.array_equal(grid, mine), f"{name} differs from gguf-py"
    signs = np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8)
    assert list(signs) == tables["ksigns_iq2xs"], "ksigns differ from gguf-py"
    assert tables["kvalues_iq4nl"] == [int(v) for v in quants.IQ4_NL.kvalues]
    assert tables["kvalues_fp4"] == [int(v) for v in quants.MXFP4.kvalues]
    print("grids match gguf-py", file=sys.stderr)


def main() -> None:
    if len(sys.argv) > 1:
        text = pathlib.Path(sys.argv[1]).read_text()
    else:
        text = urllib.request.urlopen(URL).read().decode()
    tables = {}
    for name, _, ty, count, _ in TABLES:
        vals = parse(text, name, count)
        if ty == "i8":
            vals = [v if v < 128 else v - 256 for v in vals]
        tables[name] = vals
    crosscheck(tables)

    out = [
        "//! Lookup tables of llama.cpp's I-quants, ported by `testdata/quants/make_tables.py`.\n",
        "//!\n",
        f"//! Source: `ggml/src/ggml-common.h` of ggml-org/llama.cpp at commit `{COMMIT}`.\n",
        "//! The grids are the packed `uint64_t` / `uint32_t` entries of that header\n",
        "//! unchanged; a grid entry's byte `j` (little endian) is element `j`. Do not edit:\n",
        "//! regenerate with the script.\n\n",
    ]
    for name, rust, ty, count, doc in TABLES:
        out.append(f"/// {doc} (`{name}`)\n")
        out.append(f"pub(crate) const {rust}: [{ty}; {count}] = [\n")
        vals = tables[name]
        per = 16 if ty in ("u8", "i8") else 4
        for i in range(0, len(vals), per):
            row = vals[i : i + per]
            if ty == "u64":
                cells = ", ".join(f"0x{v:016x}" for v in row)
            elif ty == "u32":
                cells = ", ".join(f"0x{v:08x}" for v in row)
            else:
                cells = ", ".join(str(v) for v in row)
            out.append(f"    {cells},\n")
        out.append("];\n\n")
    OUT.write_text("".join(out).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
