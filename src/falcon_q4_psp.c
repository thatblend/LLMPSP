/* PSP-only VFPU kernels: the four-row Q4 tile matvec kernel, int8
 * attention, and the Mamba state update.
 *
 * Shared conventions:
 * - vc2i.s expands 4 signed bytes of one word into 4 ints (value << 24)
 *   with byte 0 (LSB) in lane 0; vi2f.q with scale 24 turns them into
 *   the plain byte values as floats. Validated end to end in PPSSPP.
 * - Everything handed to lv.q/sv.q is 16-byte aligned (alloc_array in
 *   falcon_h1.c guarantees this for runtime arrays; tile groups are
 *   aligned by the FHQ4 version-2 layout and the aligned cache blocks
 *   and buffers; int8 KV rows are 64 bytes at 64-byte strides from
 *   aligned bases). Real hardware faults otherwise; PPSSPP does not.
 * - Everything runs on the single inference thread; static scratch is
 *   fine. Results are not bit-identical to the scalar host kernels
 *   (summation order), verified end to end against the host build.
 *
 * The Q4 kernels decode each block's four 8-bit scale codes with exact
 * integer-valued float operations only (falcon_q4_tile_psp.h), so the
 * decoded scales are bit-identical on hardware, in PPSSPP and on the host
 * (scale_code_value in falcon_h1.c). No denormal scale can occur: the
 * loader rejects code parameters whose magnitudes are not normal. */
#include "falcon_h1.h"

#if defined(_PSP_FW_VERSION) && !defined(FALCON_SCALAR_ONLY)

#include "falcon_q4_tile_psp.h"

/* group: one tile group (16-byte aligned); xt: per block 4 x {x16 quad,
 * xd quad}; block_sums: per block the sum of x over its 16 low-nibble
 * columns; out: four results (16-byte aligned); blocks a multiple of 4;
 * bias: four copies of k * 2^(23 - m). One kernel per scale-code mantissa
 * width m (the vi2f immediate is m + 1). */
DEFINE_TILE_KERNEL(falcon_q4_tile4_m3, 4)
DEFINE_TILE_KERNEL(falcon_q4_tile4_m4, 5)
DEFINE_TILE_KERNEL(falcon_q4_tile4_m5, 6)

/* Raw int8 attention score dots: scores[t] = sum(q[i] * key_t[i]) for
 * t in [0, count). keys advances by stride bytes per step; the caller
 * applies the per-position dequantization scale afterwards. q and each
 * key row are 64 elements, 16-byte aligned. */
void falcon_vfpu_attn_score(const float *q, const int8_t *keys, int stride,
                            int count, float *scores) {
    const int8_t *k = keys;
    float *s = scores;
    int n = count;
    __asm__ volatile (
        ".set push\n"
        ".set reorder\n"
        ".set macro\n"
        "lv.q  C400, 0(%[q])\n"     /* q resident in M400-M730 */
        "lv.q  C410, 16(%[q])\n"
        "lv.q  C420, 32(%[q])\n"
        "lv.q  C430, 48(%[q])\n"
        "lv.q  C500, 64(%[q])\n"
        "lv.q  C510, 80(%[q])\n"
        "lv.q  C520, 96(%[q])\n"
        "lv.q  C530, 112(%[q])\n"
        "lv.q  C600, 128(%[q])\n"
        "lv.q  C610, 144(%[q])\n"
        "lv.q  C620, 160(%[q])\n"
        "lv.q  C630, 176(%[q])\n"
        "lv.q  C700, 192(%[q])\n"
        "lv.q  C710, 208(%[q])\n"
        "lv.q  C720, 224(%[q])\n"
        "lv.q  C730, 240(%[q])\n"
        "1:\n"
        "lv.q  C100, 0(%[k])\n"     /* 64 int8 = 16 words */
        "lv.q  C110, 16(%[k])\n"
        "lv.q  C120, 32(%[k])\n"
        "lv.q  C130, 48(%[k])\n"
        "vc2i.s C200, S100\n"
        "vc2i.s C210, S101\n"
        "vc2i.s C220, S102\n"
        "vc2i.s C230, S103\n"
        "vc2i.s C300, S110\n"
        "vc2i.s C310, S111\n"
        "vc2i.s C320, S112\n"
        "vc2i.s C330, S113\n"
        "vi2f.q C200, C200, 24\n"
        "vi2f.q C210, C210, 24\n"
        "vi2f.q C220, C220, 24\n"
        "vi2f.q C230, C230, 24\n"
        "vi2f.q C300, C300, 24\n"
        "vi2f.q C310, C310, 24\n"
        "vi2f.q C320, C320, 24\n"
        "vi2f.q C330, C330, 24\n"
        "vmul.q C200, C200, C400\n" /* elements 0-31 */
        "vmul.q C210, C210, C410\n"
        "vmul.q C220, C220, C420\n"
        "vmul.q C230, C230, C430\n"
        "vmul.q C300, C300, C500\n"
        "vmul.q C310, C310, C510\n"
        "vmul.q C320, C320, C520\n"
        "vmul.q C330, C330, C530\n"
        "vadd.q C200, C200, C210\n"
        "vadd.q C220, C220, C230\n"
        "vadd.q C300, C300, C310\n"
        "vadd.q C320, C320, C330\n"
        "vadd.q C200, C200, C220\n"
        "vadd.q C300, C300, C320\n"
        "vadd.q C000, C200, C300\n"
        "vc2i.s C200, S120\n"       /* elements 32-63 */
        "vc2i.s C210, S121\n"
        "vc2i.s C220, S122\n"
        "vc2i.s C230, S123\n"
        "vc2i.s C300, S130\n"
        "vc2i.s C310, S131\n"
        "vc2i.s C320, S132\n"
        "vc2i.s C330, S133\n"
        "vi2f.q C200, C200, 24\n"
        "vi2f.q C210, C210, 24\n"
        "vi2f.q C220, C220, 24\n"
        "vi2f.q C230, C230, 24\n"
        "vi2f.q C300, C300, 24\n"
        "vi2f.q C310, C310, 24\n"
        "vi2f.q C320, C320, 24\n"
        "vi2f.q C330, C330, 24\n"
        "vmul.q C200, C200, C600\n"
        "vmul.q C210, C210, C610\n"
        "vmul.q C220, C220, C620\n"
        "vmul.q C230, C230, C630\n"
        "vmul.q C300, C300, C700\n"
        "vmul.q C310, C310, C710\n"
        "vmul.q C320, C320, C720\n"
        "vmul.q C330, C330, C730\n"
        "vadd.q C200, C200, C210\n"
        "vadd.q C220, C220, C230\n"
        "vadd.q C300, C300, C310\n"
        "vadd.q C320, C320, C330\n"
        "vadd.q C200, C200, C220\n"
        "vadd.q C300, C300, C320\n"
        "vadd.q C200, C200, C300\n"
        "vadd.q C000, C000, C200\n"
        "vfad.q S010, C000\n"
        "sv.s  S010, 0(%[s])\n"
        "addu  %[k], %[k], %[stride]\n"
        "addiu %[s], %[s], 4\n"
        "addiu %[n], %[n], -1\n"
        "bnez  %[n], 1b\n"
        ".set pop\n"
        : [k] "+r"(k), [s] "+r"(s), [n] "+r"(n)
        : [q] "r"(q), [stride] "r"(stride)
        : "memory");
}

/* out[i] = sum_t weights[t] * value_t[i], i in [0, 64), overwriting out.
 * The 16 accumulator quads stay in registers for the whole loop. */
void falcon_vfpu_attn_accum(float *out, const int8_t *values, int stride,
                            int count, const float *weights) {
    const int8_t *v = values;
    const float *w = weights;
    int n = count;
    __asm__ volatile (
        ".set push\n"
        ".set reorder\n"
        ".set macro\n"
        "vzero.q C400\n"
        "vzero.q C410\n"
        "vzero.q C420\n"
        "vzero.q C430\n"
        "vzero.q C500\n"
        "vzero.q C510\n"
        "vzero.q C520\n"
        "vzero.q C530\n"
        "vzero.q C600\n"
        "vzero.q C610\n"
        "vzero.q C620\n"
        "vzero.q C630\n"
        "vzero.q C700\n"
        "vzero.q C710\n"
        "vzero.q C720\n"
        "vzero.q C730\n"
        "1:\n"
        "lv.s  S001, 0(%[w])\n"
        "lv.q  C100, 0(%[v])\n"
        "lv.q  C110, 16(%[v])\n"
        "lv.q  C120, 32(%[v])\n"
        "lv.q  C130, 48(%[v])\n"
        "vc2i.s C200, S100\n"
        "vc2i.s C210, S101\n"
        "vc2i.s C220, S102\n"
        "vc2i.s C230, S103\n"
        "vc2i.s C300, S110\n"
        "vc2i.s C310, S111\n"
        "vc2i.s C320, S112\n"
        "vc2i.s C330, S113\n"
        "vi2f.q C200, C200, 24\n"
        "vi2f.q C210, C210, 24\n"
        "vi2f.q C220, C220, 24\n"
        "vi2f.q C230, C230, 24\n"
        "vi2f.q C300, C300, 24\n"
        "vi2f.q C310, C310, 24\n"
        "vi2f.q C320, C320, 24\n"
        "vi2f.q C330, C330, 24\n"
        "vscl.q C200, C200, S001\n"
        "vscl.q C210, C210, S001\n"
        "vscl.q C220, C220, S001\n"
        "vscl.q C230, C230, S001\n"
        "vscl.q C300, C300, S001\n"
        "vscl.q C310, C310, S001\n"
        "vscl.q C320, C320, S001\n"
        "vscl.q C330, C330, S001\n"
        "vadd.q C400, C400, C200\n"
        "vadd.q C410, C410, C210\n"
        "vadd.q C420, C420, C220\n"
        "vadd.q C430, C430, C230\n"
        "vadd.q C500, C500, C300\n"
        "vadd.q C510, C510, C310\n"
        "vadd.q C520, C520, C320\n"
        "vadd.q C530, C530, C330\n"
        "vc2i.s C200, S120\n"
        "vc2i.s C210, S121\n"
        "vc2i.s C220, S122\n"
        "vc2i.s C230, S123\n"
        "vc2i.s C300, S130\n"
        "vc2i.s C310, S131\n"
        "vc2i.s C320, S132\n"
        "vc2i.s C330, S133\n"
        "vi2f.q C200, C200, 24\n"
        "vi2f.q C210, C210, 24\n"
        "vi2f.q C220, C220, 24\n"
        "vi2f.q C230, C230, 24\n"
        "vi2f.q C300, C300, 24\n"
        "vi2f.q C310, C310, 24\n"
        "vi2f.q C320, C320, 24\n"
        "vi2f.q C330, C330, 24\n"
        "vscl.q C200, C200, S001\n"
        "vscl.q C210, C210, S001\n"
        "vscl.q C220, C220, S001\n"
        "vscl.q C230, C230, S001\n"
        "vscl.q C300, C300, S001\n"
        "vscl.q C310, C310, S001\n"
        "vscl.q C320, C320, S001\n"
        "vscl.q C330, C330, S001\n"
        "vadd.q C600, C600, C200\n"
        "vadd.q C610, C610, C210\n"
        "vadd.q C620, C620, C220\n"
        "vadd.q C630, C630, C230\n"
        "vadd.q C700, C700, C300\n"
        "vadd.q C710, C710, C310\n"
        "vadd.q C720, C720, C320\n"
        "vadd.q C730, C730, C330\n"
        "addu  %[v], %[v], %[stride]\n"
        "addiu %[w], %[w], 4\n"
        "addiu %[n], %[n], -1\n"
        "bnez  %[n], 1b\n"
        "sv.q  C400, 0(%[out])\n"
        "sv.q  C410, 16(%[out])\n"
        "sv.q  C420, 32(%[out])\n"
        "sv.q  C430, 48(%[out])\n"
        "sv.q  C500, 64(%[out])\n"
        "sv.q  C510, 80(%[out])\n"
        "sv.q  C520, 96(%[out])\n"
        "sv.q  C530, 112(%[out])\n"
        "sv.q  C600, 128(%[out])\n"
        "sv.q  C610, 144(%[out])\n"
        "sv.q  C620, 160(%[out])\n"
        "sv.q  C630, 176(%[out])\n"
        "sv.q  C700, 192(%[out])\n"
        "sv.q  C710, 208(%[out])\n"
        "sv.q  C720, 224(%[out])\n"
        "sv.q  C730, 240(%[out])\n"
        ".set pop\n"
        : [v] "+r"(v), [w] "+r"(w), [n] "+r"(n)
        : [out] "r"(out), [stride] "r"(stride)
        : "memory");
}

/* One Mamba head: for each of the 32 channels, updates its 64-element
 * state (state = state * decay + dt * hidden * b) and returns the
 * per-channel dot with cc in acc[32]. b stays resident in registers.
 * decay_dt points at {decay, dt}; all float pointers 16-byte aligned
 * except hidden (lv.s only). */
void falcon_vfpu_mamba_head(float *state, const float *hidden,
                            const float *b, const float *cc,
                            const float *decay_dt, float *acc) {
    float *st = state;
    const float *h = hidden;
    float *a = acc;
    int n = 32;
    __asm__ volatile (
        ".set push\n"
        ".set reorder\n"
        ".set macro\n"
        "lv.s  S010, 0(%[dd])\n"    /* decay */
        "lv.s  S011, 4(%[dd])\n"    /* dt */
        "lv.q  C400, 0(%[b])\n"     /* b resident in M400-M730 */
        "lv.q  C410, 16(%[b])\n"
        "lv.q  C420, 32(%[b])\n"
        "lv.q  C430, 48(%[b])\n"
        "lv.q  C500, 64(%[b])\n"
        "lv.q  C510, 80(%[b])\n"
        "lv.q  C520, 96(%[b])\n"
        "lv.q  C530, 112(%[b])\n"
        "lv.q  C600, 128(%[b])\n"
        "lv.q  C610, 144(%[b])\n"
        "lv.q  C620, 160(%[b])\n"
        "lv.q  C630, 176(%[b])\n"
        "lv.q  C700, 192(%[b])\n"
        "lv.q  C710, 208(%[b])\n"
        "lv.q  C720, 224(%[b])\n"
        "lv.q  C730, 240(%[b])\n"
        "1:\n"
        "lv.s  S012, 0(%[h])\n"
        "vmul.s S013, S011, S012\n" /* coeff = dt * hidden */
        "vzero.q C020\n"            /* channel accumulator */
        /* 4 groups of 4 quads: state chunk, matching b block, cc. */
#define FALCON_MAMBA_GROUP(SOFF, B0, B1, B2, B3)                         \
        "lv.q  C100, " #SOFF "+0(%[st])\n"                               \
        "lv.q  C110, " #SOFF "+16(%[st])\n"                              \
        "lv.q  C120, " #SOFF "+32(%[st])\n"                              \
        "lv.q  C130, " #SOFF "+48(%[st])\n"                              \
        "vscl.q C100, C100, S010\n"                                      \
        "vscl.q C110, C110, S010\n"                                      \
        "vscl.q C120, C120, S010\n"                                      \
        "vscl.q C130, C130, S010\n"                                      \
        "vscl.q C200, " B0 ", S013\n"                                    \
        "vscl.q C210, " B1 ", S013\n"                                    \
        "vscl.q C220, " B2 ", S013\n"                                    \
        "vscl.q C230, " B3 ", S013\n"                                    \
        "vadd.q C100, C100, C200\n"                                      \
        "vadd.q C110, C110, C210\n"                                      \
        "vadd.q C120, C120, C220\n"                                      \
        "vadd.q C130, C130, C230\n"                                      \
        "sv.q  C100, " #SOFF "+0(%[st])\n"                               \
        "sv.q  C110, " #SOFF "+16(%[st])\n"                              \
        "sv.q  C120, " #SOFF "+32(%[st])\n"                              \
        "sv.q  C130, " #SOFF "+48(%[st])\n"                              \
        "lv.q  C300, " #SOFF "+0(%[cc])\n"                               \
        "lv.q  C310, " #SOFF "+16(%[cc])\n"                              \
        "lv.q  C320, " #SOFF "+32(%[cc])\n"                              \
        "lv.q  C330, " #SOFF "+48(%[cc])\n"                              \
        "vmul.q C300, C100, C300\n"                                      \
        "vmul.q C310, C110, C310\n"                                      \
        "vmul.q C320, C120, C320\n"                                      \
        "vmul.q C330, C130, C330\n"                                      \
        "vadd.q C300, C300, C310\n"                                      \
        "vadd.q C320, C320, C330\n"                                      \
        "vadd.q C300, C300, C320\n"                                      \
        "vadd.q C020, C020, C300\n"
        FALCON_MAMBA_GROUP(0,   "C400", "C410", "C420", "C430")
        FALCON_MAMBA_GROUP(64,  "C500", "C510", "C520", "C530")
        FALCON_MAMBA_GROUP(128, "C600", "C610", "C620", "C630")
        FALCON_MAMBA_GROUP(192, "C700", "C710", "C720", "C730")
#undef FALCON_MAMBA_GROUP
        "vfad.q S001, C020\n"
        "sv.s  S001, 0(%[a])\n"
        "addiu %[st], %[st], 256\n"
        "addiu %[h], %[h], 4\n"
        "addiu %[a], %[a], 4\n"
        "addiu %[n], %[n], -1\n"
        "bnez  %[n], 1b\n"
        ".set pop\n"
        : [st] "+r"(st), [h] "+r"(h), [a] "+r"(a), [n] "+r"(n)
        : [b] "r"(b), [cc] "r"(cc), [dd] "r"(decay_dt)
        : "memory");
}

/* The same update with the state stored as float16 (FALCON_STATE_F16,
 * PROJECT.md section 2.10): 64 halves = 128 bytes per channel. The new
 * value of each element (kept in float for the dot with cc) is clamped to
 * +-65504, rounded to 11 significant bits with exact arithmetic (Veltkamp:
 * c = x * 8193, hi = c - (c - x)) and flushed to zero below 2^-14 before
 * vf2h, so vf2h converts exactly and hardware (which truncates and
 * flushes) and PPSSPP (which rounds and keeps denormals) store the same
 * bits; state_store() in falcon_h1.c is the host model of exactly this.
 * Constants: C100 = +65504, C110 = -65504, C120 = 2^-14, C130 = 0 (all
 * four lanes), S030 = 8193. state 16-byte aligned (channel stride 128). */
static const float f16_consts[20] __attribute__((aligned(16))) = {
    65504.0f, 65504.0f, 65504.0f, 65504.0f,
    -65504.0f, -65504.0f, -65504.0f, -65504.0f,
    6.103515625e-05f, 6.103515625e-05f, 6.103515625e-05f, 6.103515625e-05f,
    0.0f, 0.0f, 0.0f, 0.0f,
    8193.0f, 0.0f, 0.0f, 0.0f,
};

void falcon_vfpu_mamba_head_f16(uint16_t *state, const float *hidden,
                                const float *b, const float *cc,
                                const float *decay_dt, float *acc) {
    uint16_t *st = state;
    const float *h = hidden;
    float *a = acc;
    int n = 32;
    __asm__ volatile (
        ".set push\n"
        ".set reorder\n"
        ".set macro\n"
        "lv.s  S010, 0(%[dd])\n"    /* decay */
        "lv.s  S011, 4(%[dd])\n"    /* dt */
        "lv.q  C100, 0(%[k])\n"     /* clamp, flush and Veltkamp constants */
        "lv.q  C110, 16(%[k])\n"
        "lv.q  C120, 32(%[k])\n"
        "lv.q  C130, 48(%[k])\n"
        "lv.s  S030, 64(%[k])\n"
        "lv.q  C400, 0(%[b])\n"     /* b resident in M400-M730 */
        "lv.q  C410, 16(%[b])\n"
        "lv.q  C420, 32(%[b])\n"
        "lv.q  C430, 48(%[b])\n"
        "lv.q  C500, 64(%[b])\n"
        "lv.q  C510, 80(%[b])\n"
        "lv.q  C520, 96(%[b])\n"
        "lv.q  C530, 112(%[b])\n"
        "lv.q  C600, 128(%[b])\n"
        "lv.q  C610, 144(%[b])\n"
        "lv.q  C620, 160(%[b])\n"
        "lv.q  C630, 176(%[b])\n"
        "lv.q  C700, 192(%[b])\n"
        "lv.q  C710, 208(%[b])\n"
        "lv.q  C720, 224(%[b])\n"
        "lv.q  C730, 240(%[b])\n"
        "1:\n"
        "lv.s  S012, 0(%[h])\n"
        "vmul.s S013, S011, S012\n" /* coeff = dt * hidden */
        "vzero.q C020\n"            /* channel accumulator */
        /* 4 groups of 16 elements: 32 bytes of halves, the matching b
         * block and 64 bytes of cc. */
#define FALCON_MAMBA16_GROUP(HOFF, COFF, B0, B1, B2, B3)                  \
        "lv.q  C300, " #HOFF "+0(%[st])\n"                               \
        "lv.q  C310, " #HOFF "+16(%[st])\n"                              \
        "vh2f.p C200, C300\n"                                            \
        "vh2f.p C210, C302\n"                                            \
        "vh2f.p C220, C310\n"                                            \
        "vh2f.p C230, C312\n"                                            \
        "vscl.q C200, C200, S010\n"                                      \
        "vscl.q C210, C210, S010\n"                                      \
        "vscl.q C220, C220, S010\n"                                      \
        "vscl.q C230, C230, S010\n"                                      \
        "vscl.q C300, " B0 ", S013\n"                                    \
        "vscl.q C310, " B1 ", S013\n"                                    \
        "vscl.q C320, " B2 ", S013\n"                                    \
        "vscl.q C330, " B3 ", S013\n"                                    \
        "vadd.q C200, C200, C300\n"                                      \
        "vadd.q C210, C210, C310\n"                                      \
        "vadd.q C220, C220, C320\n"                                      \
        "vadd.q C230, C230, C330\n"                                      \
        "lv.q  C300, " #COFF "+0(%[cc])\n"                               \
        "lv.q  C310, " #COFF "+16(%[cc])\n"                              \
        "lv.q  C320, " #COFF "+32(%[cc])\n"                              \
        "lv.q  C330, " #COFF "+48(%[cc])\n"                              \
        "vmul.q C300, C200, C300\n"                                      \
        "vmul.q C310, C210, C310\n"                                      \
        "vmul.q C320, C220, C320\n"                                      \
        "vmul.q C330, C230, C330\n"                                      \
        "vadd.q C300, C300, C310\n"                                      \
        "vadd.q C320, C320, C330\n"                                      \
        "vadd.q C300, C300, C320\n"                                      \
        "vadd.q C020, C020, C300\n"                                      \
        "vmin.q C200, C200, C100\n"       /* clamp to +-65504 */         \
        "vmin.q C210, C210, C100\n"                                      \
        "vmin.q C220, C220, C100\n"                                      \
        "vmin.q C230, C230, C100\n"                                      \
        "vmax.q C200, C200, C110\n"                                      \
        "vmax.q C210, C210, C110\n"                                      \
        "vmax.q C220, C220, C110\n"                                      \
        "vmax.q C230, C230, C110\n"                                      \
        "vscl.q C300, C200, S030\n"       /* c = x * 8193 */             \
        "vscl.q C310, C210, S030\n"                                      \
        "vscl.q C320, C220, S030\n"                                      \
        "vscl.q C330, C230, S030\n"                                      \
        "vsub.q C200, C300, C200\n"       /* t = c - x */                \
        "vsub.q C210, C310, C210\n"                                      \
        "vsub.q C220, C320, C220\n"                                      \
        "vsub.q C230, C330, C230\n"                                      \
        "vsub.q C200, C300, C200\n"       /* hi = c - t */               \
        "vsub.q C210, C310, C210\n"                                      \
        "vsub.q C220, C320, C220\n"                                      \
        "vsub.q C230, C330, C230\n"                                      \
        "vabs.q C300, C200\n"             /* flush below 2^-14 */        \
        "vabs.q C310, C210\n"                                            \
        "vabs.q C320, C220\n"                                            \
        "vabs.q C330, C230\n"                                            \
        "vcmp.q LT, C300, C120\n"                                        \
        "vcmovt.q C200, C130, 6\n"                                       \
        "vcmp.q LT, C310, C120\n"                                        \
        "vcmovt.q C210, C130, 6\n"                                       \
        "vcmp.q LT, C320, C120\n"                                        \
        "vcmovt.q C220, C130, 6\n"                                       \
        "vcmp.q LT, C330, C120\n"                                        \
        "vcmovt.q C230, C130, 6\n"                                       \
        "vf2h.q C300, C200\n"                                            \
        "vf2h.q C302, C210\n"                                            \
        "vf2h.q C310, C220\n"                                            \
        "vf2h.q C312, C230\n"                                            \
        "sv.q  C300, " #HOFF "+0(%[st])\n"                               \
        "sv.q  C310, " #HOFF "+16(%[st])\n"
        FALCON_MAMBA16_GROUP(0,  0,   "C400", "C410", "C420", "C430")
        FALCON_MAMBA16_GROUP(32, 64,  "C500", "C510", "C520", "C530")
        FALCON_MAMBA16_GROUP(64, 128, "C600", "C610", "C620", "C630")
        FALCON_MAMBA16_GROUP(96, 192, "C700", "C710", "C720", "C730")
#undef FALCON_MAMBA16_GROUP
        "vfad.q S001, C020\n"
        "sv.s  S001, 0(%[a])\n"
        "addiu %[st], %[st], 128\n"
        "addiu %[h], %[h], 4\n"
        "addiu %[a], %[a], 4\n"
        "addiu %[n], %[n], -1\n"
        "bnez  %[n], 1b\n"
        ".set pop\n"
        : [st] "+r"(st), [h] "+r"(h), [a] "+r"(a), [n] "+r"(n)
        : [b] "r"(b), [cc] "r"(cc), [dd] "r"(decay_dt), [k] "r"(f16_consts)
        : "memory");
}

#else
/* ISO C forbids an empty translation unit, hence the typedef. */
typedef int falcon_q4_psp_unused;
#endif
