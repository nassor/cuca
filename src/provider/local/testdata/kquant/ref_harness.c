/* Runs llama.cpp's reference row quantizers (ggml/src/ggml-quants.c, extracted
 * verbatim by make_fixture.py into extract.inc) over a file of f32 values.
 *
 *   ref_harness <type> <in.f32> <out.bin>      type: q4_0 q5_0 q5_1 q4_K q5_K q6_K
 *
 * ggml's own FP16 conversion is hardware or table based; the two helpers below
 * are an exact round-to-nearest-even replacement, so no other ggml source is
 * needed. Nothing here is part of the crate; it only produces the fixtures. */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#define GGML_RESTRICT
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define GROUP_MAX_EPS 1e-15f

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, exp = (x >> 23) & 0xff, man = x & 0x7fffff;
    if (exp == 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
    int e = (int)exp - 127 + 15;
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        man |= 0x800000;
        int shift = 14 - e;
        uint32_t half = man >> shift, rem = man & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return sign | half;
    }
    uint32_t h = sign | (e << 10) | (man >> 13), rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return h;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, x;
    if (exp == 0) {
        if (man == 0) x = sign;
        else {
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x400));
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) x = sign | 0x7f800000 | (man << 13);
    else x = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f; memcpy(&f, &x, 4); return f;
}

#define GGML_FP32_TO_FP16(x) f32_to_f16(x)
#define GGML_FP16_TO_FP32(x) f16_to_f32(x)

#include "extract.inc"

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *f = fopen(argv[2], "rb");
    if (!f) return 3;
    fseek(f, 0, SEEK_END);
    long n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    float *x = malloc(n * 4);
    if (fread(x, 4, n, f) != (size_t)n) return 4;
    fclose(f);
    void *y = calloc(1, n * 4 + 1024);
    size_t out = 0;
#define GO(name, T, qk) \
    if (!strcmp(argv[1], #name)) { quantize_row_##name##_ref(x, (T *)y, n); out = (n / qk) * sizeof(T); }
    GO(q4_0, block_q4_0, 32)
    GO(q5_0, block_q5_0, 32)
    GO(q5_1, block_q5_1, 32)
    GO(q4_K, block_q4_K, 256)
    GO(q5_K, block_q5_K, 256)
    GO(q6_K, block_q6_K, 256)
    if (!out) return 5;
    f = fopen(argv[3], "wb");
    fwrite(y, 1, out, f);
    fclose(f);
    return 0;
}
