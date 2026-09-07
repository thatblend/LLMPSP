/* PSP-only VFPU kernels: Q4 row dot product, int8 attention, and the
 * Mamba state update.
 *
 * Shared conventions:
 * - vc2i.s expands 4 signed bytes of one word into 4 ints (value << 24)
 *   with byte 0 (LSB) in lane 0; vi2f.q with scale 24 turns them into
 *   the plain byte values as floats. Validated end to end in PPSSPP.
 * - All float buffers handed to lv.q are 16-byte aligned (alloc_array
 *   in falcon_h1.c guarantees this for runtime arrays; int8 KV rows are
 *   64 bytes at 64-byte strides from aligned bases).
 * - Everything runs on the single inference thread; static scratch is
 *   fine. Results are not bit-identical to the scalar host kernels
 *   (summation order), verified end to end against the host build.
 *
 * The Q4 kernel converts each block's fp16 scale with integer ops
 * inline (sign/exponent/mantissa relocation). Denormal fp16 scales are
 * flushed to zero: the shipped model has 5 of them among 2.8 million
 * scales, and a denormal scale (< 6.1e-5) makes every weight in its
 * block numerically irrelevant anyway. Zero scales convert exactly. */
#include "falcon_h1.h"

#if defined(_PSP_FW_VERSION) && !defined(FALCON_SCALAR_ONLY)

float falcon_dot_q4_vfpu(const uint8_t *row, const float *x,
                         int blocks, const float *block_sums) {
    static uint32_t scratch[12] __attribute__((aligned(16)));
    float result;
    const uint8_t *r = row;
    const float *xp = x;
    const float *bp = block_sums;
    int n = blocks;
    __asm__ volatile (
        ".set push\n"
        ".set reorder\n"
        ".set macro\n"
        "vzero.q C000\n"            /* row accumulator (d-scaled dots) */
        "vzero.s S010\n"            /* sum of d_b * block_sum_b */
        "lui   $24, 0x0F0F\n"
        "ori   $24, $24, 0x0F0F\n"
        "1:\n"
        "ulw   $8, 2(%[row])\n"     /* packed bytes 0-3 */
        "ulw   $9, 6(%[row])\n"
        "ulw   $10, 10(%[row])\n"
        "ulw   $11, 14(%[row])\n"
        "and   $12, $8, $24\n"      /* low nibbles: weights 0-15 */
        "srl   $8, $8, 4\n"
        "and   $8, $8, $24\n"       /* high nibbles: weights 16-31 */
        "and   $13, $9, $24\n"
        "srl   $9, $9, 4\n"
        "and   $9, $9, $24\n"
        "and   $14, $10, $24\n"
        "srl   $10, $10, 4\n"
        "and   $10, $10, $24\n"
        "and   $15, $11, $24\n"
        "srl   $11, $11, 4\n"
        "and   $11, $11, $24\n"
        "sw    $12, 0(%[sc])\n"
        "sw    $13, 4(%[sc])\n"
        "sw    $14, 8(%[sc])\n"
        "sw    $15, 12(%[sc])\n"
        "sw    $8, 16(%[sc])\n"
        "sw    $9, 20(%[sc])\n"
        "sw    $10, 24(%[sc])\n"
        "sw    $11, 28(%[sc])\n"
        /* fp16 block scale -> f32 bits with integer ops. Denormals
         * (exponent field 0) flush to zero, inf/nan never occur. */
        "lhu   $25, 0(%[row])\n"
        "andi  $2, $25, 0x3FF\n"
        "sll   $2, $2, 13\n"        /* mantissa << 13 */
        "srl   $3, $25, 10\n"
        "andi  $3, $3, 0x1F\n"
        "addiu $3, $3, 112\n"
        "sll   $3, $3, 23\n"        /* (e + 112) << 23 */
        "addu  $2, $2, $3\n"        /* disjoint bit fields */
        "srl   $3, $25, 15\n"
        "sll   $3, $3, 31\n"        /* sign */
        "or    $2, $2, $3\n"
        "andi  $3, $25, 0x7C00\n"
        "sltu  $3, $0, $3\n"        /* 1 when exponent != 0 */
        "negu  $3, $3\n"
        "and   $2, $2, $3\n"
        "sw    $2, 32(%[sc])\n"
        "lv.s  S030, 32(%[sc])\n"   /* d for this block */
        "lv.s  S031, 0(%[bs])\n"    /* block sum of x */
        "lv.q  C100, 0(%[sc])\n"    /* nibble bytes, weights 0-15 */
        "lv.q  C110, 16(%[sc])\n"   /* nibble bytes, weights 16-31 */
        "vc2i.s C200, S100\n"       /* 4 signed bytes -> 4 ints << 24 */
        "vc2i.s C210, S101\n"
        "vc2i.s C220, S102\n"
        "vc2i.s C230, S103\n"
        "vc2i.s C300, S110\n"
        "vc2i.s C310, S111\n"
        "vc2i.s C320, S112\n"
        "vc2i.s C330, S113\n"
        "vi2f.q C200, C200, 24\n"   /* ints -> float nibble values 0..15 */
        "vi2f.q C210, C210, 24\n"
        "vi2f.q C220, C220, 24\n"
        "vi2f.q C230, C230, 24\n"
        "vi2f.q C300, C300, 24\n"
        "vi2f.q C310, C310, 24\n"
        "vi2f.q C320, C320, 24\n"
        "vi2f.q C330, C330, 24\n"
        "lv.q  C400, 0(%[x])\n"
        "lv.q  C410, 16(%[x])\n"
        "lv.q  C420, 32(%[x])\n"
        "lv.q  C430, 48(%[x])\n"
        "lv.q  C500, 64(%[x])\n"
        "lv.q  C510, 80(%[x])\n"
        "lv.q  C520, 96(%[x])\n"
        "lv.q  C530, 112(%[x])\n"
        "vmul.q C600, C200, C400\n"
        "vmul.q C610, C210, C410\n"
        "vmul.q C620, C220, C420\n"
        "vmul.q C630, C230, C430\n"
        "vmul.q C700, C300, C500\n"
        "vmul.q C710, C310, C510\n"
        "vmul.q C720, C320, C520\n"
        "vmul.q C730, C330, C530\n"
        "vadd.q C600, C600, C610\n"
        "vadd.q C620, C620, C630\n"
        "vadd.q C700, C700, C710\n"
        "vadd.q C720, C720, C730\n"
        "vadd.q C600, C600, C620\n"
        "vadd.q C700, C700, C720\n"
        "vadd.q C600, C600, C700\n" /* raw block dot in 4 lanes */
        "vscl.q C600, C600, S030\n" /* scale by d */
        "vadd.q C000, C000, C600\n"
        "vmul.s S032, S030, S031\n" /* d * block_sum */
        "vadd.s S010, S010, S032\n"
        "addiu %[row], %[row], 18\n"
        "addiu %[x], %[x], 128\n"
        "addiu %[bs], %[bs], 4\n"
        "addiu %[n], %[n], -1\n"
        "bnez  %[n], 1b\n"
        "vfad.q S020, C000\n"       /* horizontal sum of the 4 lanes */
        "vfim.s S021, 8.0\n"
        "vmul.s S022, S010, S021\n"
        "vsub.s S020, S020, S022\n"
        "sv.s  S020, 0(%[out])\n"
        ".set pop\n"
        : [row] "+r"(r), [x] "+r"(xp), [bs] "+r"(bp), [n] "+r"(n)
        : [sc] "r"(scratch), [out] "r"(&result)
        : "$2", "$3", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15",
          "$24", "$25", "memory");
    return result;
}

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

#else
/* ISO C forbids an empty translation unit, hence the typedef. */
typedef int falcon_q4_psp_unused;
#endif
