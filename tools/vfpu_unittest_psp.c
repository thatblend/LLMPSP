/* Standalone PSP EBOOT: compares every VFPU kernel against a scalar
 * reference on deterministic pseudo-random data and logs the maximum
 * relative error. Run in PPSSPP; errors should be ~1e-6 (summation
 * order); anything near 1 means a lane or addressing bug. Exact checks
 * (scale-code decoding, float16 state bits) must report 0 mismatches.
 * PPSSPP does not reproduce the hardware's vf2h/vh2f rounding: the
 * float16 path is built so both store the same bits (see
 * falcon_vfpu_mamba_head_f16), which only hardware can finally confirm. */
#include "falcon_h1.h"

#include <pspkernel.h>
#include <pspdebug.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

PSP_MODULE_INFO("VfpuUnit", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(4096);

typedef void (*TileKernel)(const uint8_t *group, const float *xt,
                           const float *block_sums, int blocks, float *out,
                           const float *bias);
extern void falcon_q4_tile4_m3(const uint8_t *, const float *, const float *,
                               int, float *, const float *);
extern void falcon_q4_tile4_m4(const uint8_t *, const float *, const float *,
                               int, float *, const float *);
extern void falcon_q4_tile4_m5(const uint8_t *, const float *, const float *,
                               int, float *, const float *);
extern void falcon_vfpu_attn_score(const float *q, const int8_t *keys,
                                   int stride, int count, float *scores);
extern void falcon_vfpu_attn_accum(float *out, const int8_t *values,
                                   int stride, int count,
                                   const float *weights);
extern void falcon_vfpu_mamba_head(float *state, const float *hidden,
                                   const float *b, const float *cc,
                                   const float *decay_dt, float *acc);
extern void falcon_vfpu_mamba_head_f16(uint16_t *state, const float *hidden,
                                       const float *b, const float *cc,
                                       const float *decay_dt, float *acc);

static uint32_t rng_state = 0x12345678u;
static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float frand(void) { return (float)(int)(rng() % 2001u - 1000u) / 500.0f; }

static float relerr(float a, float b) {
    float d = fabsf(a - b);
    float m = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return m > 1e-6f ? d / m : d;
}

/* FHQ4 version-3 scale code -> float, written independently of
 * scale_code_value() in src/falcon_h1.c. */
static float code_value(int code, int m, int k) {
    uint32_t bits;
    float f;
    if (code == 0) return 0.0f;
    bits = ((uint32_t)(code < 0 ? -code : code) + (uint32_t)k) << (23 - m);
    memcpy(&f, &bits, 4);
    return code < 0 ? -f : f;
}

/* Up to 24 blocks; the kernels need a multiple of 4. */
#define MAXB 24
#define T 7

static float xv[MAXB * 32] __attribute__((aligned(16)));
static float xt[MAXB * 32] __attribute__((aligned(16)));
static float lowsums[MAXB] __attribute__((aligned(16)));
static int8_t codes[4][MAXB];
static uint8_t nib[4][MAXB * 32];
static uint8_t group[4 * MAXB * 17] __attribute__((aligned(16)));
static float out4[4] __attribute__((aligned(16)));
static float bias4[4] __attribute__((aligned(16)));
static float q[64] __attribute__((aligned(16)));
static int8_t keys[T * 128] __attribute__((aligned(16)));
static float scores_v[T] __attribute__((aligned(16)));
static float weights[T] __attribute__((aligned(16)));
static float outv[64] __attribute__((aligned(16)));
static float state_v[32 * 64] __attribute__((aligned(16)));
static float state_r[32 * 64];
static uint16_t state16_v[32 * 64] __attribute__((aligned(16)));
static uint16_t state16_r[32 * 64];
static float bb[64] __attribute__((aligned(16)));
static float ccv[64] __attribute__((aligned(16)));
static float hidden[32] __attribute__((aligned(16)));
static float accv[32] __attribute__((aligned(16)));
static float dd[2] __attribute__((aligned(16)));

/* Packs codes/nibbles into one version-3 group the way
 * tools/convert_falcon_h1.py (pack_groups_v3) does, and builds the
 * activation tables the way q4_prepare_x does. */
static void build_group(int blocks) {
    int b, r, i, k;
    for (b = 0; b < blocks; ++b) {
        for (r = 0; r < 4; ++r) {
            group[b * 4 + r] = (uint8_t)codes[r][b];
            for (i = 0; i < 4; ++i)
                for (k = 0; k < 4; ++k) {
                    int j = i * 4 + k;
                    uint8_t packed = (uint8_t)(nib[r][b * 32 + j] |
                                               (nib[r][b * 32 + j + 16] << 4));
                    group[blocks * 4 + b * 64 + i * 16 + r * 4 + k] =
                        (uint8_t)(packed ^ 0x80u);
                }
        }
        lowsums[b] = 0.0f;
        for (i = 0; i < 16; ++i) lowsums[b] += xv[b * 32 + i];
        for (i = 0; i < 4; ++i)
            for (k = 0; k < 4; ++k) {
                float lo16 = 16.0f * xv[b * 32 + i * 4 + k];
                xt[b * 32 + i * 8 + k] = lo16;
                xt[b * 32 + i * 8 + 4 + k] = xv[b * 32 + 16 + i * 4 + k] - lo16;
            }
    }
}

/* float16 store model: clamp, Veltkamp rounding to 11 bits, flush below
 * 2^-14, exact conversion (state_store() in src/falcon_h1.c). */
static uint16_t f16_store(float x) {
    volatile float c, t;
    float hi;
    uint32_t bits, sign, e;
    if (x > 65504.0f) x = 65504.0f;
    if (x < -65504.0f) x = -65504.0f;
    c = x * 8193.0f;
    t = c - x;
    hi = c - t;
    if (fabsf(hi) < 6.103515625e-05f) hi = 0.0f;
    memcpy(&bits, &hi, 4);
    sign = (bits >> 16) & 0x8000u;
    e = (bits >> 23) & 0xffu;
    if (e < 113u) return (uint16_t)sign;
    return (uint16_t)(sign | ((e - 112u) << 10) | ((bits >> 13) & 0x3ffu));
}
static float f16_load(uint16_t h) {
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t bits = ((uint32_t)(h & 0x8000u) << 16) |
                    (e ? ((e + 112u) << 23) | ((uint32_t)(h & 0x3ffu) << 13)
                       : 0u);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

int main(void) {
    static const TileKernel kernels[3] = {
        falcon_q4_tile4_m3, falcon_q4_tile4_m4, falcon_q4_tile4_m5,
    };
    FILE *log;
    int i, b, t, m;
    float worst;
    pspDebugScreenInit();
    log = fopen("vfpu_unit_log.txt", "w");
    if (!log) { sceKernelExitGame(); return 1; }

    /* --- Q4 tile kernels (FHQ4 version 3), m = 3, 4, 5. --- */
    for (m = 3; m <= 5; ++m) {
        TileKernel kernel = kernels[m - 3];
        /* Bias placing the codes around 2^-7 .. 2^-1 (typical scales). */
        int k = (120 << m) - 64;
        int mismatches = 0, code;
        bias4[0] = bias4[1] = bias4[2] = bias4[3] =
            (float)((uint32_t)k << (23 - m));

        /* 1. Exact decode: every code -128..127. Block 0 alone carries
         * x = (1, 0, ...) and a low nibble 9 (weight +1) in column 0, so
         * each row's result is exactly 9d - 8d = its decoded scale d. */
        memset(xv, 0, sizeof(xv));
        xv[0] = 1.0f;
        memset(nib, 8, sizeof(nib));
        for (code = -128; code < 128; code += 4) {
            int r;
            for (r = 0; r < 4; ++r) {
                nib[r][0] = 9;
                for (b = 0; b < 4; ++b)
                    codes[r][b] = (int8_t)(b == 0 ? code + r :
                                           (int)(rng() % 255u) - 127);
            }
            build_group(4);
            kernel(group, xt, lowsums, 4, out4, bias4);
            for (r = 0; r < 4; ++r) {
                float want = code_value(code + r, m, k);
                if (memcmp(&want, &out4[r], 4)) {
                    if (mismatches < 4)
                        fprintf(log, "q4 m%d code %d: want %.9g got %.9g\n",
                                m, code + r, want, out4[r]);
                    ++mismatches;
                }
            }
        }
        fprintf(log, "q4 m%d scale decode: 256 codes, %d mismatches\n", m,
                mismatches);

        /* 2. Random groups (16 and 24 blocks, like the model) against a
         * scalar reference: random codes including 0, +-1, +-127 and
         * -128, every nibble extreme. */
        worst = 0;
        {
            int blocks;
            for (blocks = 16; blocks <= 24; blocks += 8) {
                int r;
                for (i = 0; i < blocks * 32; ++i) xv[i] = frand();
                for (r = 0; r < 4; ++r)
                    for (b = 0; b < blocks; ++b) {
                        codes[r][b] = (int8_t)(rng() & 0xFF);
                        for (i = 0; i < 32; ++i)
                            nib[r][b * 32 + i] = (uint8_t)(rng() & 15u);
                    }
                codes[0][0] = 0; codes[1][1] = 127; codes[2][2] = -127;
                codes[3][3] = -128; codes[0][4] = 1; codes[1][5] = -1;
                nib[0][0] = 0; nib[0][1] = 15; nib[0][16] = 0;
                nib[0][17] = 15;
                build_group(blocks);
                kernel(group, xt, lowsums, blocks, out4, bias4);
                for (r = 0; r < 4; ++r) {
                    float ref = 0;
                    for (b = 0; b < blocks; ++b) {
                        float d = code_value(codes[r][b], m, k), s = 0;
                        for (i = 0; i < 32; ++i)
                            s += (float)((int)nib[r][b * 32 + i] - 8) *
                                 xv[b * 32 + i];
                        ref += d * s;
                    }
                    if (relerr(ref, out4[r]) > worst)
                        worst = relerr(ref, out4[r]);
                }
            }
        }
        fprintf(log, "q4 m%d tile: worst relerr %.2e\n", m, worst);
    }

    /* --- attention score --- */
    for (i = 0; i < 64; ++i) q[i] = frand();
    for (i = 0; i < T * 128; ++i) keys[i] = (int8_t)(rng() % 255u - 127u);
    falcon_vfpu_attn_score(q, keys, 128, T, scores_v);
    worst = 0;
    for (t = 0; t < T; ++t) {
        float ref = 0;
        for (i = 0; i < 64; ++i) ref += q[i] * (float)keys[t * 128 + i];
        if (relerr(ref, scores_v[t]) > worst) worst = relerr(ref, scores_v[t]);
    }
    fprintf(log, "attn_score: worst relerr %.2e\n", worst);

    /* --- attention accum --- */
    for (t = 0; t < T; ++t) weights[t] = frand();
    falcon_vfpu_attn_accum(outv, keys, 128, T, weights);
    worst = 0;
    for (i = 0; i < 64; ++i) {
        float ref = 0;
        for (t = 0; t < T; ++t) ref += weights[t] * (float)keys[t * 128 + i];
        if (relerr(ref, outv[i]) > worst) worst = relerr(ref, outv[i]);
    }
    fprintf(log, "attn_accum: worst relerr %.2e\n", worst);

    /* --- mamba head, float32 state --- */
    for (i = 0; i < 32 * 64; ++i) state_v[i] = frand() * 0.1f;
    memcpy(state_r, state_v, sizeof(state_r));
    for (i = 0; i < 64; ++i) { bb[i] = frand(); ccv[i] = frand(); }
    for (i = 0; i < 32; ++i) hidden[i] = frand();
    dd[0] = 0.9f; dd[1] = 0.3f;
    falcon_vfpu_mamba_head(state_v, hidden, bb, ccv, dd, accv);
    worst = 0;
    for (i = 0; i < 32; ++i) {
        float ref = 0;
        int s;
        for (s = 0; s < 64; ++s) {
            state_r[i * 64 + s] = state_r[i * 64 + s] * dd[0] +
                                  dd[1] * bb[s] * hidden[i];
            ref += state_r[i * 64 + s] * ccv[s];
        }
        if (relerr(ref, accv[i]) > worst) worst = relerr(ref, accv[i]);
    }
    fprintf(log, "mamba acc: worst relerr %.2e\n", worst);
    worst = 0;
    for (i = 0; i < 32 * 64; ++i)
        if (relerr(state_r[i], state_v[i]) > worst)
            worst = relerr(state_r[i], state_v[i]);
    fprintf(log, "mamba state: worst relerr %.2e\n", worst);

    /* --- mamba head, float16 state: three rounds over states spanning
     * the whole float16 range, including values that must clamp (near
     * 65504 with a large increment), flush (below 2^-14) and round. The
     * stored bits must equal the scalar model computed with the kernel's
     * association, (state * decay) + b * (dt * hidden). --- */
    {
        int round, mismatches = 0, clamped = 0, flushed = 0;
        for (i = 0; i < 32 * 64; ++i) {
            float v = frand();
            int e = (int)(rng() % 40u) - 22;              /* 2^-22 .. 2^17 */
            v = ldexpf(v, e);
            if (i % 97 == 0) v = 65500.0f;
            if (i % 89 == 0) v = -65500.0f;
            state16_v[i] = f16_store(v);
        }
        memcpy(state16_r, state16_v, sizeof(state16_r));
        for (round = 0; round < 3; ++round) {
            for (i = 0; i < 64; ++i) { bb[i] = frand() * 200.0f; ccv[i] = frand(); }
            for (i = 0; i < 32; ++i)
                hidden[i] = (i % 5 == 0) ? 1.0e-9f * frand() : frand() * 50.0f;
            dd[0] = round == 0 ? 1.0f : round == 1 ? 0.5f : 1.0e-6f;
            dd[1] = round == 2 ? 1.0e-4f : 0.7f;
            falcon_vfpu_mamba_head_f16(state16_v, hidden, bb, ccv, dd, accv);
            worst = 0;
            for (i = 0; i < 32; ++i) {
                float ref = 0, coeff = dd[1] * hidden[i];
                int s;
                for (s = 0; s < 64; ++s) {
                    float value = f16_load(state16_r[i * 64 + s]) * dd[0] +
                                  bb[s] * coeff;
                    uint16_t h = f16_store(value);
                    ref += value * ccv[s];
                    if (fabsf(value) > 65504.0f) ++clamped;
                    if (value != 0.0f && fabsf(value) < 6.103515625e-05f)
                        ++flushed;
                    state16_r[i * 64 + s] = h;
                    if (h != state16_v[i * 64 + s]) {
                        if (mismatches < 4)
                            fprintf(log, "f16 state %d.%d: value %.9g want "
                                    "0x%04x got 0x%04x\n", i, s, value, h,
                                    state16_v[i * 64 + s]);
                        ++mismatches;
                    }
                }
                if (relerr(ref, accv[i]) > worst) worst = relerr(ref, accv[i]);
            }
            fprintf(log, "mamba f16 round %d acc: worst relerr %.2e\n", round,
                    worst);
        }
        fprintf(log, "mamba f16 state: %d values, %d clamped, %d flushed, "
                "%d bit mismatches\n", 3 * 32 * 64, clamped, flushed,
                mismatches);
    }

    fprintf(log, "done\n");
    fclose(log);
    sceKernelExitGame();
    return 0;
}
