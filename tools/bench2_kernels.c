/* Part B of LLMPSPBench2: alternative schedules of the four-row Q4 tile
 * kernel. Every variant computes exactly what falcon_q4_tile4_vtfm
 * (src/falcon_q4_tile_psp.h, copied here so nothing links against src/)
 * computes on the version-2 tile format, with the same arguments:
 *
 *   group       blocks * 8 bytes of fp16 scales (rows 0..3 per block),
 *               then blocks * 64 bytes of tiles; 16-byte aligned
 *   xt          per block 4 x {16 x_lo quad, x_hi - 16 x_lo quad}
 *   block_sums  per block, sum of x over its 16 low-nibble columns
 *   out         four results, 16-byte aligned
 *
 * Tile t of block b is 16 bytes; word r is row r; vc2i.s turns a word
 * into four signed bytes << 24, vi2f.q 28 gives G = byte / 16, vf2id
 * floors it to H (the high nibble), and a row's block dot is
 * sum(G * x16) + sum(H * xd) - 8 * sum(x_lo).
 *
 * The block-end scale handling is kept in the SCALE_* macros so an 8-bit
 * scale code format can replace fp16 by changing only them.
 *
 * Every kernel respects the VFPU overlap rules: vtfm4's output never
 * shares a register with its matrix or vector input. */
#include <stdint.h>

#include "bench2.h"

#if defined(_PSP_FW_VERSION)

/* ---- block-end pieces (fp16 scales today) ------------------------- */

#define SCALE_LOAD                                                          \
        "lv.s   S010, 0(%[sc])\n"                                           \
        "lv.s   S011, 4(%[sc])\n"
#define SCALE_CVT  "vh2f.p C020, C010\n"
#define SCALE_STEP "8"

#define KERNEL_ARGS                                                         \
    (const uint8_t *group, const float *xt, const float *block_sums,        \
     int blocks, float *out)
#define KERNEL_SETUP                                                        \
    const uint8_t *sc = group;                                              \
    const uint8_t *tp = group + (unsigned)blocks * 8u;                      \
    const float *xp = xt;                                                   \
    const float *bs = block_sums;                                           \
    int n = blocks;
#define KERNEL_OPERANDS                                                     \
        : [tp] "+r"(tp), [xt] "+r"(xp), [sc] "+r"(sc), [bs] "+r"(bs),       \
          [n] "+r"(n)                                                       \
        : [out] "r"(out)                                                    \
        : "memory", "$8"

/* ---- 1/2: base, the shipped schedule (verbatim) --------------------- */

#define TILE(TOFF, XOFF, SLOT_A, SLOT_B, MUL)                               \
        "lv.q   C100, " #TOFF "(%[tp])\n"                                   \
        "vc2i.s C200, S100\n"                                               \
        "vc2i.s C210, S101\n"                                               \
        "vc2i.s C220, S102\n"                                               \
        "vc2i.s C230, S103\n"                                               \
        SLOT_A                                                              \
        "vi2f.q C200, C200, 28\n"                                           \
        "vi2f.q C210, C210, 28\n"                                           \
        "vi2f.q C220, C220, 28\n"                                           \
        "vi2f.q C230, C230, 28\n"                                           \
        SLOT_B                                                              \
        "vf2id.q C300, C200, 0\n"                                           \
        "vf2id.q C310, C210, 0\n"                                           \
        "vf2id.q C320, C220, 0\n"                                           \
        "vf2id.q C330, C230, 0\n"                                           \
        "lv.q   C400, " #XOFF "(%[xt])\n"                                   \
        "vi2f.q C300, C300, 0\n"                                            \
        "vi2f.q C310, C310, 0\n"                                            \
        "vi2f.q C320, C320, 0\n"                                            \
        "vi2f.q C330, C330, 0\n"                                            \
        "lv.q   C410, " #XOFF "+16(%[xt])\n"                                \
        MUL

#define MUL_VTFM                                                            \
        "vtfm4.q C500, M200, C400\n"                                        \
        "vtfm4.q C510, M300, C410\n"

#define MUL_VDOT                                                            \
        "vdot.q S500, C200, C400\n"                                         \
        "vdot.q S501, C210, C400\n"                                         \
        "vdot.q S502, C220, C400\n"                                         \
        "vdot.q S503, C230, C400\n"                                         \
        "vdot.q S510, C300, C410\n"                                         \
        "vdot.q S511, C310, C410\n"                                         \
        "vdot.q S512, C320, C410\n"                                         \
        "vdot.q S513, C330, C410\n"

/* Shared epilogue of base/base_pre: the last block's deferred scale. */
#define BASE_FINISH                                                         \
        "vmul.q C620, C620, C020\n"                                         \
        "vfim.s S000, 8.0\n"                                                \
        "vadd.q C700, C700, C620\n"                                         \
        "vscl.q C710, C710, S000\n"                                         \
        "vsub.q C700, C700, C710\n"                                         \
        "sv.q   C700, 0(%[out])\n"

#define DEFINE_BASE(NAME, MUL)                                              \
static void NAME KERNEL_ARGS {                                              \
    KERNEL_SETUP                                                            \
    __asm__ volatile (                                                      \
        ".set push\n"                                                       \
        ".set noreorder\n"                                                  \
        "vzero.q C700\n"                                                    \
        "vzero.q C710\n"                                                    \
        "vzero.q C620\n"                                                    \
        "vzero.q C020\n"                                                    \
        "1:\n"                                                              \
        TILE(0, 0, "vmul.q C620, C620, C020\n",                             \
                   "vadd.q C700, C700, C620\n", MUL)                        \
        SCALE_LOAD                                                          \
        TILE(16, 32, SCALE_CVT, "vadd.q C600, C500, C510\n", MUL)           \
        TILE(32, 64, "vadd.q C600, C600, C500\n",                           \
                     "vadd.q C600, C600, C510\n", MUL)                      \
        "lv.s   S030, 0(%[bs])\n"                                           \
        TILE(48, 96, "vadd.q C600, C600, C500\n",                           \
                     "vadd.q C600, C600, C510\n", MUL)                      \
        "vscl.q C610, C020, S030\n"                                         \
        "addiu  %[tp], %[tp], 64\n"                                         \
        "vadd.q C620, C600, C500\n"                                         \
        "addiu  %[xt], %[xt], 128\n"                                        \
        "addiu  %[sc], %[sc], " SCALE_STEP "\n"                             \
        "vadd.q C710, C710, C610\n"                                         \
        "addiu  %[bs], %[bs], 4\n"                                          \
        "addiu  %[n], %[n], -1\n"                                           \
        "bnez   %[n], 1b\n"                                                 \
        "vadd.q C620, C620, C510\n"                                         \
        BASE_FINISH                                                         \
        ".set pop\n"                                                        \
        KERNEL_OPERANDS);                                                   \
}

DEFINE_BASE(k_base, MUL_VTFM)
DEFINE_BASE(k_base_vdot, MUL_VDOT)

/* ---- 3: nopipe, no software pipelining ------------------------------
 * Each tile's adds come straight after its multiplies and the block's
 * scale is applied at once, so every result waits out its full latency.
 * The difference to base is what the slots buy. */

#define NP_TILE(TOFF, XOFF, ADDS)                                           \
        "lv.q   C100, " #TOFF "(%[tp])\n"                                   \
        "vc2i.s C200, S100\n"                                               \
        "vc2i.s C210, S101\n"                                               \
        "vc2i.s C220, S102\n"                                               \
        "vc2i.s C230, S103\n"                                               \
        "vi2f.q C200, C200, 28\n"                                           \
        "vi2f.q C210, C210, 28\n"                                           \
        "vi2f.q C220, C220, 28\n"                                           \
        "vi2f.q C230, C230, 28\n"                                           \
        "vf2id.q C300, C200, 0\n"                                           \
        "vf2id.q C310, C210, 0\n"                                           \
        "vf2id.q C320, C220, 0\n"                                           \
        "vf2id.q C330, C230, 0\n"                                           \
        "lv.q   C400, " #XOFF "(%[xt])\n"                                   \
        "vi2f.q C300, C300, 0\n"                                            \
        "vi2f.q C310, C310, 0\n"                                            \
        "vi2f.q C320, C320, 0\n"                                            \
        "vi2f.q C330, C330, 0\n"                                            \
        "lv.q   C410, " #XOFF "+16(%[xt])\n"                                \
        MUL_VTFM                                                            \
        ADDS

#define NP_ADD2 "vadd.q C600, C600, C500\n" "vadd.q C600, C600, C510\n"

static void k_nopipe KERNEL_ARGS {
    KERNEL_SETUP
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "vzero.q C700\n"
        "vzero.q C710\n"
        "1:\n"
        NP_TILE(0, 0, "vadd.q C600, C500, C510\n")
        NP_TILE(16, 32, NP_ADD2)
        NP_TILE(32, 64, NP_ADD2)
        NP_TILE(48, 96, NP_ADD2)
        SCALE_LOAD
        SCALE_CVT
        "lv.s   S030, 0(%[bs])\n"
        "vmul.q C620, C600, C020\n"
        "vadd.q C700, C700, C620\n"
        "vscl.q C610, C020, S030\n"
        "vadd.q C710, C710, C610\n"
        "addiu  %[tp], %[tp], 64\n"
        "addiu  %[xt], %[xt], 128\n"
        "addiu  %[sc], %[sc], " SCALE_STEP "\n"
        "addiu  %[bs], %[bs], 4\n"
        "addiu  %[n], %[n], -1\n"
        "bnez   %[n], 1b\n"
        "nop\n"
        "vfim.s S000, 8.0\n"
        "vscl.q C710, C710, S000\n"
        "vsub.q C700, C700, C710\n"
        "sv.q   C700, 0(%[out])\n"
        ".set pop\n"
        KERNEL_OPERANDS);
}

/* ---- 4: base_pre, raw tile loads one tile ahead ---------------------
 * base issues vc2i.s right after the lv.q of its tile, so any lv.q
 * load-use latency stalls four times per block. Here each raw tile is
 * loaded into the other of C100/C110 one tile earlier; the next block's
 * tile 0 comes from $8, clamped to the current block on the last pass so
 * nothing past the group is read. Otherwise identical to base. */

#define PRE_TILE(R, NEXTLOAD, XOFF, SLOT_A, SLOT_B, MUL)                    \
        "vc2i.s C200, S1" #R "0\n"                                          \
        "vc2i.s C210, S1" #R "1\n"                                          \
        "vc2i.s C220, S1" #R "2\n"                                          \
        "vc2i.s C230, S1" #R "3\n"                                          \
        NEXTLOAD                                                            \
        SLOT_A                                                              \
        "vi2f.q C200, C200, 28\n"                                           \
        "vi2f.q C210, C210, 28\n"                                           \
        "vi2f.q C220, C220, 28\n"                                           \
        "vi2f.q C230, C230, 28\n"                                           \
        SLOT_B                                                              \
        "vf2id.q C300, C200, 0\n"                                           \
        "vf2id.q C310, C210, 0\n"                                           \
        "vf2id.q C320, C220, 0\n"                                           \
        "vf2id.q C330, C230, 0\n"                                           \
        "lv.q   C400, " #XOFF "(%[xt])\n"                                   \
        "vi2f.q C300, C300, 0\n"                                            \
        "vi2f.q C310, C310, 0\n"                                            \
        "vi2f.q C320, C320, 0\n"                                            \
        "vi2f.q C330, C330, 0\n"                                            \
        "lv.q   C410, " #XOFF "+16(%[xt])\n"                                \
        MUL

static void k_base_pre KERNEL_ARGS {
    KERNEL_SETUP
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "vzero.q C700\n"
        "vzero.q C710\n"
        "vzero.q C620\n"
        "vzero.q C020\n"
        "lv.q   C100, 0(%[tp])\n"
        "1:\n"
        "addiu  %[n], %[n], -1\n"
        "addiu  $8, %[tp], 64\n"
        "movz   $8, %[tp], %[n]\n"          /* last block: re-read own */
        PRE_TILE(0, "lv.q   C110, 16(%[tp])\n", 0,
                 "vmul.q C620, C620, C020\n",
                 "vadd.q C700, C700, C620\n", MUL_VTFM)
        SCALE_LOAD
        PRE_TILE(1, "lv.q   C100, 32(%[tp])\n", 32,
                 SCALE_CVT, "vadd.q C600, C500, C510\n", MUL_VTFM)
        PRE_TILE(0, "lv.q   C110, 48(%[tp])\n", 64,
                 "vadd.q C600, C600, C500\n",
                 "vadd.q C600, C600, C510\n", MUL_VTFM)
        "lv.s   S030, 0(%[bs])\n"
        PRE_TILE(1, "lv.q   C100, 0($8)\n", 96,
                 "vadd.q C600, C600, C500\n",
                 "vadd.q C600, C600, C510\n", MUL_VTFM)
        "vscl.q C610, C020, S030\n"
        "addiu  %[tp], %[tp], 64\n"
        "vadd.q C620, C600, C500\n"
        "addiu  %[xt], %[xt], 128\n"
        "addiu  %[sc], %[sc], " SCALE_STEP "\n"
        "vadd.q C710, C710, C610\n"
        "addiu  %[bs], %[bs], 4\n"
        "bnez   %[n], 1b\n"
        "vadd.q C620, C620, C510\n"
        BASE_FINISH
        ".set pop\n"
        KERNEL_OPERANDS);
}

/* ---- 5/6: pair, two tiles' conversion chains interleaved ------------
 * Two tiles go through each stage together (8 vc2i, 8 vi2f, 8 vf2id,
 * 8 vi2f), so every producer -> consumer distance is 8+ instructions
 * instead of 4-5: robust against any latency up to ~8 and against
 * write-port conflicts between stages. Same instruction count as base.
 *
 * Registers: M200 G(a), M300 H(a), M400 G(b), M500 H(b); C100/C110 raw
 * tiles a/b, then (once vc2i consumed them) x16(a)/xd(a); C120/C130
 * x16(b)/xd(b); C600-C630 the four partials; C720 block total T, C730
 * T * scales, C000 scales * block sum, C020 scales, S010/S011 raw scales,
 * S012 block sum, C700 rows, C710 low-nibble correction.
 *
 * Pair 0 of block b finishes block b-1 in its slots (adds the partials
 * of tiles 2,3, scales, accumulates), pair 1 starts block b's total. */

#define PAIR(TOFF, XOFF, S1, S2, S3, S4, S5, MUL4)                          \
        "lv.q   C100, " #TOFF "(%[tp])\n"                                   \
        "lv.q   C110, " #TOFF "+16(%[tp])\n"                                \
        S1                                                                  \
        "vc2i.s C200, S100\n"                                               \
        "vc2i.s C210, S101\n"                                               \
        "vc2i.s C220, S102\n"                                               \
        "vc2i.s C230, S103\n"                                               \
        "vc2i.s C400, S110\n"                                               \
        "vc2i.s C410, S111\n"                                               \
        "vc2i.s C420, S112\n"                                               \
        "vc2i.s C430, S113\n"                                               \
        S2                                                                  \
        "vi2f.q C200, C200, 28\n"                                           \
        "vi2f.q C210, C210, 28\n"                                           \
        "vi2f.q C220, C220, 28\n"                                           \
        "vi2f.q C230, C230, 28\n"                                           \
        "vi2f.q C400, C400, 28\n"                                           \
        "vi2f.q C410, C410, 28\n"                                           \
        "vi2f.q C420, C420, 28\n"                                           \
        "vi2f.q C430, C430, 28\n"                                           \
        S3                                                                  \
        "vf2id.q C300, C200, 0\n"                                           \
        "vf2id.q C310, C210, 0\n"                                           \
        "vf2id.q C320, C220, 0\n"                                           \
        "vf2id.q C330, C230, 0\n"                                           \
        "vf2id.q C500, C400, 0\n"                                           \
        "vf2id.q C510, C410, 0\n"                                           \
        "vf2id.q C520, C420, 0\n"                                           \
        "vf2id.q C530, C430, 0\n"                                           \
        "lv.q   C100, " #XOFF "(%[xt])\n"                                   \
        S4                                                                  \
        "vi2f.q C300, C300, 0\n"                                            \
        "vi2f.q C310, C310, 0\n"                                            \
        "vi2f.q C320, C320, 0\n"                                            \
        "vi2f.q C330, C330, 0\n"                                            \
        "lv.q   C110, " #XOFF "+16(%[xt])\n"                                \
        "vi2f.q C500, C500, 0\n"                                            \
        "vi2f.q C510, C510, 0\n"                                            \
        "vi2f.q C520, C520, 0\n"                                            \
        "vi2f.q C530, C530, 0\n"                                            \
        "lv.q   C120, " #XOFF "+32(%[xt])\n"                                \
        "lv.q   C130, " #XOFF "+48(%[xt])\n"                                \
        S5                                                                  \
        MUL4

#define PAIR_VTFM                                                           \
        "vtfm4.q C600, M200, C100\n"                                        \
        "vtfm4.q C610, M300, C110\n"                                        \
        "vtfm4.q C620, M400, C120\n"                                        \
        "vtfm4.q C630, M500, C130\n"

#define PAIR_VDOT                                                           \
        "vdot.q S600, C200, C100\n"                                         \
        "vdot.q S601, C210, C100\n"                                         \
        "vdot.q S602, C220, C100\n"                                         \
        "vdot.q S603, C230, C100\n"                                         \
        "vdot.q S610, C300, C110\n"                                         \
        "vdot.q S611, C310, C110\n"                                         \
        "vdot.q S612, C320, C110\n"                                         \
        "vdot.q S613, C330, C110\n"                                         \
        "vdot.q S620, C400, C120\n"                                         \
        "vdot.q S621, C410, C120\n"                                         \
        "vdot.q S622, C420, C120\n"                                         \
        "vdot.q S623, C430, C120\n"                                         \
        "vdot.q S630, C500, C130\n"                                         \
        "vdot.q S631, C510, C130\n"                                         \
        "vdot.q S632, C520, C130\n"                                         \
        "vdot.q S633, C530, C130\n"

/* T * scales -> rows, scales * block sum -> correction, then the end. */
#define FINISH_T                                                            \
        "vmul.q C730, C720, C020\n"                                         \
        "vscl.q C000, C020, S012\n"                                         \
        "vadd.q C700, C700, C730\n"                                         \
        "vadd.q C710, C710, C000\n"                                         \
        "vfim.s S013, 8.0\n"                                                \
        "vscl.q C710, C710, S013\n"                                         \
        "vsub.q C700, C700, C710\n"                                         \
        "sv.q   C700, 0(%[out])\n"

#define DEFINE_PAIR(NAME, MUL4)                                             \
static void NAME KERNEL_ARGS {                                              \
    KERNEL_SETUP                                                            \
    __asm__ volatile (                                                      \
        ".set push\n"                                                       \
        ".set noreorder\n"                                                  \
        "vzero.q C600\n"                                                    \
        "vzero.q C610\n"                                                    \
        "vzero.q C620\n"                                                    \
        "vzero.q C630\n"                                                    \
        "vzero.q C720\n"                                                    \
        "vzero.q C020\n"                                                    \
        "vzero.q C010\n"                                                    \
        "vzero.q C700\n"                                                    \
        "vzero.q C710\n"                                                    \
        "1:\n"                                                              \
        PAIR(0, 0,                                                          \
             "vadd.q C600, C600, C610\n",                                   \
             "vadd.q C620, C620, C630\n"                                    \
             "vadd.q C720, C720, C600\n",                                   \
             "vadd.q C720, C720, C620\n",                                   \
             "vmul.q C730, C720, C020\n"                                    \
             "vscl.q C000, C020, S012\n",                                   \
             "vadd.q C700, C700, C730\n"                                    \
             "vadd.q C710, C710, C000\n"                                    \
             SCALE_LOAD                                                     \
             "lv.s   S012, 0(%[bs])\n",                                     \
             MUL4)                                                          \
        PAIR(32, 64,                                                        \
             SCALE_CVT                                                      \
             "vadd.q C600, C600, C610\n",                                   \
             "vadd.q C620, C620, C630\n",                                   \
             "vadd.q C720, C600, C620\n",                                   \
             "addiu  %[sc], %[sc], " SCALE_STEP "\n",                       \
             "addiu  %[tp], %[tp], 64\n"                                    \
             "addiu  %[xt], %[xt], 128\n"                                   \
             "addiu  %[n], %[n], -1\n",                                     \
             MUL4)                                                          \
        "bnez   %[n], 1b\n"                                                 \
        "addiu  %[bs], %[bs], 4\n"                                          \
        "vadd.q C600, C600, C610\n"                                         \
        "vadd.q C620, C620, C630\n"                                         \
        "vadd.q C720, C720, C600\n"                                         \
        "vadd.q C720, C720, C620\n"                                         \
        FINISH_T                                                            \
        ".set pop\n"                                                        \
        KERNEL_OPERANDS);                                                   \
}

DEFINE_PAIR(k_pair_vtfm, PAIR_VTFM)
DEFINE_PAIR(k_pair_vdot, PAIR_VDOT)

/* ---- 7/8: rot, the stage boundary moved ------------------------------
 * Iteration j runs the back half of tile j-1 (vf2id, vi2f, x loads,
 * multiply) interleaved one-for-one with the front half of tile j (vc2i,
 * vi2f 28), and loads the raw tile j+1. Every dependency is 8+
 * instructions apart, vc2i (latency 3) never completes in the same cycle
 * as a vf2id/vi2f (latency 5) issued next to it, and the raw tile is
 * loaded a whole iteration before vc2i reads it. Same instruction count
 * as base (+3 integer).
 *
 * Registers: raw tiles C100 (even) / C110 (odd); G of even tiles M200,
 * of odd tiles M400; H M300; x quads C120/C130; partials C500/C510;
 * T C720, T * scales C730, scales * block sum C000, scales C020, raw
 * scales S010/S011, block sum S012, rows C700, correction C710.
 *
 * Partials of tile t are added one iteration after they are made, so a
 * block's total is complete in iteration 1 of the next block, scaled in
 * iteration 2 and accumulated in iteration 3 (the last block in the
 * epilogue). For the first block, the "previous block" is all zeros.
 * The x quads of tile 3 are read through xl (xt + 96 of the block before;
 * xt itself on the first pass, harmless since G is zero then). The next
 * block's tile 0 comes from $8, clamped on the last pass. */

#define ROT_ITER(NEXTLOAD, R, G, P, XBASE, XOFF, SA, SB, MUL)               \
        NEXTLOAD                                                            \
        "vf2id.q C300, C" #P "00, 0\n"                                      \
        "vc2i.s  C" #G "00, S1" #R "0\n"                                    \
        "vf2id.q C310, C" #P "10, 0\n"                                      \
        "vc2i.s  C" #G "10, S1" #R "1\n"                                    \
        "vf2id.q C320, C" #P "20, 0\n"                                      \
        "vc2i.s  C" #G "20, S1" #R "2\n"                                    \
        "vf2id.q C330, C" #P "30, 0\n"                                      \
        "vc2i.s  C" #G "30, S1" #R "3\n"                                    \
        SA                                                                  \
        "vi2f.q  C300, C300, 0\n"                                           \
        "vi2f.q  C" #G "00, C" #G "00, 28\n"                                \
        "vi2f.q  C310, C310, 0\n"                                           \
        "vi2f.q  C" #G "10, C" #G "10, 28\n"                                \
        "vi2f.q  C320, C320, 0\n"                                           \
        "vi2f.q  C" #G "20, C" #G "20, 28\n"                                \
        "vi2f.q  C330, C330, 0\n"                                           \
        "lv.q    C120, " #XOFF "(" XBASE ")\n"                              \
        "vi2f.q  C" #G "30, C" #G "30, 28\n"                                \
        "lv.q    C130, " #XOFF "+16(" XBASE ")\n"                           \
        SB                                                                  \
        MUL(P)

#define ROT_VTFM(P)                                                         \
        "vtfm4.q C500, M" #P "00, C120\n"                                   \
        "vtfm4.q C510, M300, C130\n"

#define ROT_VDOT(P)                                                         \
        "vdot.q S500, C" #P "00, C120\n"                                    \
        "vdot.q S501, C" #P "10, C120\n"                                    \
        "vdot.q S502, C" #P "20, C120\n"                                    \
        "vdot.q S503, C" #P "30, C120\n"                                    \
        "vdot.q S510, C300, C130\n"                                         \
        "vdot.q S511, C310, C130\n"                                         \
        "vdot.q S512, C320, C130\n"                                         \
        "vdot.q S513, C330, C130\n"

#define DEFINE_ROT(NAME, MUL)                                               \
static void NAME KERNEL_ARGS {                                              \
    KERNEL_SETUP                                                            \
    const float *xl;                                                        \
    __asm__ volatile (                                                      \
        ".set push\n"                                                       \
        ".set noreorder\n"                                                  \
        "lv.q   C100, 0(%[tp])\n"                                           \
        "move   %[xl], %[xt]\n"                                             \
        "vzero.q C400\n"                                                    \
        "vzero.q C410\n"                                                    \
        "vzero.q C420\n"                                                    \
        "vzero.q C430\n"                                                    \
        "vzero.q C500\n"                                                    \
        "vzero.q C510\n"                                                    \
        "vzero.q C720\n"                                                    \
        "vzero.q C020\n"                                                    \
        "vzero.q C010\n"                                                    \
        "vzero.q C700\n"                                                    \
        "vzero.q C710\n"                                                    \
        "1:\n"                                                              \
        /* j=0: back of tile 3 (previous block), front of tile 0 */         \
        ROT_ITER("lv.q   C110, 16(%[tp])\n", 0, 2, 4, "%[xl]", 0,           \
                 "vadd.q C720, C720, C500\n",                               \
                 "vadd.q C720, C720, C510\n"                                \
                 "addiu  %[n], %[n], -1\n"                                  \
                 "addiu  $8, %[tp], 64\n"                                   \
                 "movz   $8, %[tp], %[n]\n", MUL)                           \
        /* j=1: back of tile 0, front of tile 1 */                          \
        ROT_ITER("lv.q   C100, 32(%[tp])\n", 1, 4, 2, "%[xt]", 0,           \
                 "vadd.q C720, C720, C500\n"                                \
                 "addiu  %[xl], %[xt], 96\n",                               \
                 "vadd.q C720, C720, C510\n", MUL)                          \
        /* j=2: back of tile 1, front of tile 2; previous block done */     \
        ROT_ITER("lv.q   C110, 48(%[tp])\n", 0, 2, 4, "%[xt]", 32,          \
                 "vmul.q C730, C720, C020\n"                                \
                 "vscl.q C000, C020, S012\n",                               \
                 "vadd.q C720, C500, C510\n"                                \
                 SCALE_LOAD                                                 \
                 "lv.s   S012, 0(%[bs])\n", MUL)                            \
        /* j=3: back of tile 2, front of tile 3 */                          \
        ROT_ITER("lv.q   C100, 0($8)\n", 1, 4, 2, "%[xt]", 64,              \
                 "vadd.q C700, C700, C730\n"                                \
                 "vadd.q C710, C710, C000\n"                                \
                 SCALE_CVT                                                  \
                 "vadd.q C720, C720, C500\n",                               \
                 "vadd.q C720, C720, C510\n"                                \
                 "addiu  %[xt], %[xt], 128\n"                               \
                 "addiu  %[tp], %[tp], 64\n"                                \
                 "addiu  %[sc], %[sc], " SCALE_STEP "\n", MUL)              \
        "bnez   %[n], 1b\n"                                                 \
        "addiu  %[bs], %[bs], 4\n"                                          \
        /* epilogue: back of the last block's tile 3 */                     \
        "vf2id.q C300, C400, 0\n"                                           \
        "vf2id.q C310, C410, 0\n"                                           \
        "vf2id.q C320, C420, 0\n"                                           \
        "vf2id.q C330, C430, 0\n"                                           \
        "vadd.q C720, C720, C500\n"                                         \
        "vi2f.q  C300, C300, 0\n"                                           \
        "vi2f.q  C310, C310, 0\n"                                           \
        "vi2f.q  C320, C320, 0\n"                                           \
        "vi2f.q  C330, C330, 0\n"                                           \
        "lv.q    C120, 0(%[xl])\n"                                          \
        "lv.q    C130, 16(%[xl])\n"                                         \
        "vadd.q C720, C720, C510\n"                                         \
        MUL(4)                                                              \
        "vadd.q C720, C720, C500\n"                                         \
        "vadd.q C720, C720, C510\n"                                         \
        FINISH_T                                                            \
        ".set pop\n"                                                        \
        : [tp] "+r"(tp), [xt] "+r"(xp), [sc] "+r"(sc), [bs] "+r"(bs),       \
          [n] "+r"(n), [xl] "=&r"(xl)                                       \
        : [out] "r"(out)                                                    \
        : "memory", "$8");                                                  \
}

DEFINE_ROT(k_rot_vtfm, ROT_VTFM)
DEFINE_ROT(k_rot_vdot, ROT_VDOT)

const KernelInfo bench2_kernels[] = {
    { "base", k_base, 105,
      "shipped falcon_q4_tile4_vtfm, verbatim schedule" },
    { "base_vdot", k_base_vdot, 129,
      "base schedule, 8 vdot.q per tile instead of 2 vtfm4.q" },
    { "nopipe", k_nopipe, 106,
      "no slots: adds right after the multiplies (what pipelining buys)" },
    { "base_pre", k_base_pre, 107,
      "base, raw tile lv.q one tile ahead (lv.q->vc2i load-use)" },
    { "pair_vtfm", k_pair_vtfm, 105,
      "two tiles' chains interleaved, dependencies 8+ apart" },
    { "pair_vdot", k_pair_vdot, 129,
      "pair schedule, vdot.q multiply" },
    { "rot_vtfm", k_rot_vtfm, 108,
      "next tile's vc2i/vi2f interleaved with this tile's vf2id/vi2f" },
    { "rot_vdot", k_rot_vdot, 132,
      "rot schedule, vdot.q multiply" },
};
const int bench2_kernel_count =
    (int)(sizeof(bench2_kernels) / sizeof(bench2_kernels[0]));

#else
typedef int bench2_kernels_unused;
#endif
