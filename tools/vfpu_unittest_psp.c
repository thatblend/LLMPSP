/* Standalone PSP EBOOT: compares every VFPU kernel against a scalar
 * reference on deterministic pseudo-random data and logs the maximum
 * relative error. Run in PPSSPP; errors should be ~1e-6 (summation
 * order); anything near 1 means a lane or addressing bug. */
#include "falcon_h1.h"

#include <pspkernel.h>
#include <pspdebug.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

PSP_MODULE_INFO("VfpuUnit", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(4096);

extern float falcon_dot_q4_vfpu(const uint8_t *row, const float *x,
                                int blocks, const float *block_sums);
extern void falcon_vfpu_attn_score(const float *q, const int8_t *keys,
                                   int stride, int count, float *scores);
extern void falcon_vfpu_attn_accum(float *out, const int8_t *values,
                                   int stride, int count,
                                   const float *weights);
extern void falcon_vfpu_mamba_head(float *state, const float *hidden,
                                   const float *b, const float *cc,
                                   const float *decay_dt, float *acc);

static uint32_t rng_state = 0x12345678u;
static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float frand(void) { return (float)(int)(rng() % 2001u - 1000u) / 500.0f; }

static float half_from_float_bits(float f) {
    /* Crude float->half for test scales (normal range only). */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    {
        uint32_t sign = (bits >> 16) & 0x8000u;
        int e = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t m = (bits >> 13) & 0x3FFu;
        if (e <= 0 || e >= 31) return 0; /* out of test range */
        return (float)(sign | ((uint32_t)e << 10) | m);
    }
}

static float relerr(float a, float b) {
    float d = fabsf(a - b);
    float m = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return m > 1e-6f ? d / m : d;
}

#define COLS 96
#define BLOCKS (COLS / 32)
#define T 7

static float xv[COLS] __attribute__((aligned(16)));
static float bsums[BLOCKS] __attribute__((aligned(16)));
static uint8_t rowbuf[BLOCKS * 18 + 4];
static float q[64] __attribute__((aligned(16)));
static int8_t keys[T * 128] __attribute__((aligned(16)));
static float scores_v[T] __attribute__((aligned(16)));
static float weights[T] __attribute__((aligned(16)));
static float outv[64] __attribute__((aligned(16)));
static float state_v[32 * 64] __attribute__((aligned(16)));
static float state_r[32 * 64];
static float bb[64] __attribute__((aligned(16)));
static float ccv[64] __attribute__((aligned(16)));
static float hidden[32] __attribute__((aligned(16)));
static float accv[32] __attribute__((aligned(16)));
static float dd[2] __attribute__((aligned(16)));

int main(void) {
    FILE *log;
    int i, b, t;
    float worst;
    pspDebugScreenInit();
    log = fopen("vfpu_unit_log.txt", "w");
    if (!log) { sceKernelExitGame(); return 1; }

    /* --- dot_q4: includes a zero scale and a denormal scale block --- */
    for (i = 0; i < COLS; ++i) xv[i] = frand();
    for (b = 0; b < BLOCKS; ++b) {
        float s = 0;
        uint8_t *blk = rowbuf + b * 18;
        uint16_t h;
        if (b == 0) h = 0;                      /* zero scale */
        else if (b == 1) h = 0x0123;            /* denormal scale */
        else h = (uint16_t)half_from_float_bits(0.011f);
        blk[0] = (uint8_t)h; blk[1] = (uint8_t)(h >> 8);
        for (i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)(rng() & 0xFF);
        for (i = 0; i < 32; ++i) s += xv[b * 32 + i];
        bsums[b] = s;
    }
    {
        float ref = 0, got;
        for (b = 0; b < BLOCKS; ++b) {
            const uint8_t *blk = rowbuf + b * 18;
            uint16_t h = (uint16_t)(blk[0] | (blk[1] << 8));
            int e = (h >> 10) & 0x1F;
            /* reference uses the kernel's contract: denormals -> 0 */
            float d;
            uint32_t fb = ((uint32_t)(h & 0x8000u) << 16) |
                          (((uint32_t)e + 112u) << 23) |
                          (((uint32_t)h & 0x3FFu) << 13);
            if (e == 0) d = 0.0f; else memcpy(&d, &fb, 4);
            for (i = 0; i < 16; ++i) {
                ref += d * (float)((blk[2 + i] & 15) - 8) * xv[b * 32 + i];
                ref += d * (float)((blk[2 + i] >> 4) - 8) * xv[b * 32 + i + 16];
            }
        }
        got = falcon_dot_q4_vfpu(rowbuf, xv, BLOCKS, bsums);
        fprintf(log, "dot_q4: ref %.6f got %.6f relerr %.2e\n",
                ref, got, relerr(ref, got));
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

    /* --- mamba head --- */
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

    fprintf(log, "done\n");
    fclose(log);
    sceKernelExitGame();
    return 0;
}
