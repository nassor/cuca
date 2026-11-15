"""Regenerates the K-quant parity fixtures next to this file.

    python make_fixture.py

writes `input.f32` (4096 values: sixteen blocks of 256 chosen to hit every
branch of the reference quantizers) and, for each of `q5_0`, `q5_1`, `q4_K`,
`q5_K` and `q6_K`, `<type>.bin`: the bytes llama.cpp's reference row quantizer
(`quantize_row_<type>_ref`, which `llama-quantize` runs without an importance
matrix) writes for `input.f32`. The Rust side (`convert/refquant.rs`, test
`blocks_equal_llama_cpps_reference_quantizers`) quantizes the same input with
the converter's port and compares the bytes.

The reference functions are cut verbatim out of `ggml/src/ggml-quants.c` at the
pinned llama.cpp commit below (fetched from GitHub, cached beside this file in
`.cache/`), wrapped by `ref_harness.c` and compiled with the first C compiler
found (`cc`, `gcc`, `clang`, or MSVC's `cl` through `vcvars64.bat`).
gguf-py (`gguf.quants.quantize`) has no K-quant quantizer, so it cannot be the
oracle for them; where it has one (`q4_0`, `q5_0`, `q5_1`) the script checks the
compiled harness against it, which validates the harness itself (the FP16
conversion and the build) before its K-quant output is trusted.

The bytes are those of a build without floating-point contraction (the x86-64
baseline, and `-ffp-contract=off` here): a ggml build that fuses `a*b+c` into FMA
can differ in the last bit of some scales, and the Rust port does not fuse.

Requires numpy; gguf is optional.
"""

import glob
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).parent
CACHE = HERE / ".cache"

# llama.cpp commit the reference code is taken from.
COMMIT = "6d3568e31d4e4b1e4db4acb8f6ceaad0084932f3"
SHA256 = {
    "ggml-quants.c": "bcda33625707fdfcb1367cdd2bb1a2f017430162ec9623abc84ad17459698f73",
    "ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
}
TYPES = ["q5_0", "q5_1", "q4_K", "q5_K", "q6_K"]


def fetch(name):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{COMMIT}/ggml/src/{name}"
        path.write_bytes(urllib.request.urlopen(url).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        sys.exit(f"{name}: sha256 {digest} is not the pinned commit's {SHA256[name]}")
    return path


def extract(source):
    """The reference functions of ggml-quants.c, verbatim."""
    lines = source.read_text(encoding="utf8").split("\n")

    def find(prefix, start=0):
        for i in range(start, len(lines)):
            if lines[i].startswith(prefix):
                return i
        sys.exit(f"ggml-quants.c has no line starting {prefix!r}")

    def span(first, nxt):
        a = find(first)
        return "\n".join(lines[a : find(nxt, a + 1)])

    parts = [
        span("void quantize_row_q4_0_ref", "void quantize_row_q4_1_ref"),
        span("void quantize_row_q5_0_ref", "void quantize_row_q5_1_ref"),
        span("void quantize_row_q5_1_ref", "// reference implementation for deterministic"),
        # nearest_int, make_qx_quants, make_q3_quants, make_qkx1_quants, make_qkx2_quants,
        # get_scale_min_k4 (the K-quant helpers precede the 2-bit section)
        span("static inline int nearest_int", "//========================- 2-bit"),
        span("void quantize_row_q4_K_ref", "void dequantize_row_q4_K"),
        span("void quantize_row_q5_K_ref", "void dequantize_row_q5_K"),
        span("void quantize_row_q6_K_ref", "void dequantize_row_q6_K"),
    ]
    return "\n".join(parts)


def compiler_command(out_exe):
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            command = [cc, "-O2", "-std=c11", "-ffp-contract=off", "-I.", "ref_harness.c"]
            return command + ["-lm", "-o", str(out_exe)]
    pattern = r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"
    vcvars = sorted(glob.glob(pattern))
    if vcvars:
        return (
            f'call "{vcvars[0]}" >nul && cl /nologo /O2 /std:c11 /I. ref_harness.c /Fe:{out_exe}'
        )
    sys.exit("no C compiler found (cc, gcc, clang, or Visual Studio's cl)")


def build():
    work = CACHE / "build"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(fetch("ggml-common.h"), work / "ggml-common.h")
    (work / "extract.inc").write_text(extract(fetch("ggml-quants.c")), encoding="utf8")
    shutil.copy(HERE / "ref_harness.c", work / "ref_harness.c")
    exe = work / ("ref_harness.exe" if os.name == "nt" else "ref_harness")
    command = compiler_command(exe.name)
    if isinstance(command, str):
        subprocess.run(command, cwd=work, shell=True, check=True)
    else:
        subprocess.run(command, cwd=work, check=True)
    return exe


def make_input():
    rng = np.random.default_rng(20260101)
    blocks = [
        rng.standard_normal(256) * 0.05,  # a typical weight row
        rng.standard_normal(256) * 3.0,  # large values
        np.zeros(256),  # all zero: the scale search's early outs
        np.full(256, 0.5),  # constant, positive
        rng.uniform(0.1, 2.0, 256),  # strictly positive: the minimum clamps to 0
        rng.standard_t(2, 256) * 0.1,  # heavy tail
        np.concatenate([rng.standard_normal(255) * 0.02, [0.8]]),  # one outlier
        np.concatenate(  # sub-blocks of 32 at very different scales, a zero one
            [
                rng.standard_normal(32) * s
                for s in (1e-3, 1e-1, 1.0, 1e2, 0.0, 5.0, 1e-2, 10.0)
            ]
        ),
        np.full(256, 1e-20) * rng.uniform(-1, 1, 256),  # below GROUP_MAX_EPS
        -np.abs(rng.standard_normal(256)),  # strictly negative
        rng.uniform(-1, 1, 256),  # uniform
    ]
    blocks += [rng.standard_normal(256) * 10 ** rng.uniform(-3, 1) for _ in range(5)]
    # `+ 0.0` turns every -0.0 into +0.0: which zero a block's minimum keeps is
    # unspecified (numpy and C pick differently), and it changes `q5_1`'s `m`.
    x = (np.concatenate(blocks) + 0.0).astype(np.float32)
    assert x.size == 4096
    return x


def main():
    exe = build()
    x = make_input()
    x.tofile(HERE / "input.f32")
    try:
        import gguf
    except ImportError:
        gguf = None
    for t in TYPES:
        out = HERE / f"{t}.bin"
        subprocess.run([str(exe), t, str(HERE / "input.f32"), str(out)], check=True)
        got = np.fromfile(out, dtype=np.uint8)
        print(f"{t}: {got.size} bytes")
    if gguf is not None:
        # The harness against gguf-py on the types gguf-py can quantize.
        for t, q in (("q4_0", "Q4_0"), ("q5_0", "Q5_0"), ("q5_1", "Q5_1")):
            tmp = CACHE / f"{t}.check"
            subprocess.run([str(exe), t, str(HERE / "input.f32"), str(tmp)], check=True)
            want = gguf.quants.quantize(x.reshape(1, -1), getattr(gguf.GGMLQuantizationType, q))
            assert np.array_equal(np.fromfile(tmp, dtype=np.uint8), want.reshape(-1)), t
            print(f"{t}: harness equals gguf-py")


if __name__ == "__main__":
    main()
