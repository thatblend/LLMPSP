/* Benchmark copy (tools/q4_tile_psp.c only) of the v1.3 four-row VFPU Q4
 * kernel for the FHQ4 version-2 "tile" layout with fp16 block scales:
 * falcon_q4_tile4_vtfm (the v1.3 app kernel) and falcon_q4_tile4_vdot (a
 * cross-check of the documented vtfm4 timing). The app now uses the
 * version-3 kernel in src/falcon_q4_tile_psp.h. Include only from PSP
 * builds.
 *
 * Why (PROJECT.md sections 2.8/2.9): the version-1 row kernel spent about a
 * third of its ~96 instructions per 32 weights unpacking nibbles with
 * integer ops through a scratch buffer; this one does the whole job in the
 * VFPU and handles four rows per pass. Measured on a PSP-2000: 1.75 cycles
 * per weight against 4.83 for the row kernel.
 *
 * Tile format. Four consecutive rows form a group:
 *
 *     blocks * 8 bytes   fp16 scales, rows 0..3 of block 0, then block 1...
 *     blocks * 64 bytes  tiles, four per 32-column block
 *
 * Tile t of block b is 16 bytes: word r belongs to row r, and its byte k
 * packs two weights of that row,
 *
 *     low nibble   w[32b + 4t + k] + 8        (offset binary, as in FHQ4)
 *     high nibble  w[32b + 16 + 4t + k]       (two's complement)
 *
 * which is the FHQ4 byte XOR 0x80, rearranged. The size is unchanged
 * (18 bytes per 32 weights) and every tile is 16-byte aligned.
 *
 * Nibble split without integer ops. vc2i.s turns the four bytes of a word
 * into signed bytes << 24; vi2f.q with scale 28 gives G = byte / 16
 * exactly; vf2id.q floors it, which is the high nibble as a signed value
 * H; and the low nibble is 16 * (G - H). So with
 *
 *     x16[j] = 16 * x_lo[j]         xd[j] = x_hi[j] - 16 * x_lo[j]
 *
 * a row's block dot is  sum(G * x16) + sum(H * xd) - 8 * sum(x_lo).
 *
 * vc2i.s leaves each row's four weights in one column register, so the
 * four rows of a tile are the four columns of a matrix register and
 * vtfm4.q (rd[i] = column i . rt) yields all four row partials at once:
 * 16 multiply-adds in 4 cycles, with one load of x shared by four rows.
 *
 * Scheduling follows the documented timings (throughput 1 and latency 5
 * for vi2f/vf2id/vadd/vmul, latency 3 for vc2i, throughput 4 and latency
 * 10 for vtfm4.q): every tile leaves two free slots that carry the
 * previous tile's adds, and a block's final scale-multiply is deferred
 * into the next block's first tile, so nothing should stall. 105
 * instructions, roughly 129 cycles, per 128 weights by that model.
 *
 * Register use: M200 = G, M300 = H, C100 raw tile, C400/C410 x quads,
 * C500/C510 row partials, C600 block total, C620 finished block total
 * waiting for its scale, C020 scales, C700 row accumulators, C710 the
 * low-nibble correction. vtfm4 allows no overlap between its output and
 * inputs, which this layout respects.
 *
 * Alignment contract. group, xt and out are read/written with lv.q/sv.q
 * and MUST be 16-byte aligned: real hardware raises an address error
 * otherwise, and PPSSPP cannot be relied on to show it. blocks must be
 * even, or the tiles (blocks * 8 bytes into the group) are misaligned:
 * a 3-block test group hung PPSSPP. The scales and block_sums are read
 * with lv.s (4-byte alignment). q4_matvec in src/falcon_h1.c guarantees
 * all of this and keeps a counted copy guard for the group pointer. */
#ifndef FALCON_Q4_TILE_V2_PSP_H
#define FALCON_Q4_TILE_V2_PSP_H

#include <stdint.h>

/* One tile. SLOT_A and SLOT_B are single instructions executed between
 * the conversion stages; MUL is the multiply stage. */
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

/* Two matrix-vector transforms: 4 cycles each. */
#define MUL_VTFM                                                            \
        "vtfm4.q C500, M200, C400\n"                                        \
        "vtfm4.q C510, M300, C410\n"

/* The same arithmetic as eight dot products, to cross-check the
 * documented vtfm4 timing on hardware (also 8 cycles by the model). */
#define MUL_VDOT                                                            \
        "vdot.q S500, C200, C400\n"                                         \
        "vdot.q S501, C210, C400\n"                                         \
        "vdot.q S502, C220, C400\n"                                         \
        "vdot.q S503, C230, C400\n"                                         \
        "vdot.q S510, C300, C410\n"                                         \
        "vdot.q S511, C310, C410\n"                                         \
        "vdot.q S512, C320, C410\n"                                         \
        "vdot.q S513, C330, C410\n"

/* group: one tile-format group (16-byte aligned).
 * xt: per block 4 x {x16 quad, xd quad}, i.e. 32 floats per block.
 * block_sums: per block the sum of x over its 16 low-nibble columns.
 * out: four results, 16-byte aligned. blocks must be >= 1. */
#define DEFINE_TILE_KERNEL(NAME, MUL)                                       \
void NAME(const uint8_t *group, const float *xt,                            \
          const float *block_sums, int blocks, float *out) {                \
    const uint8_t *sc = group;                                              \
    const uint8_t *tp = group + (unsigned)blocks * 8u;                      \
    const float *xp = xt;                                                   \
    const float *bs = block_sums;                                           \
    int n = blocks;                                                         \
    __asm__ volatile (                                                      \
        ".set push\n"                                                       \
        ".set noreorder\n"                                                  \
        "vzero.q C700\n"                                                    \
        "vzero.q C710\n"                                                    \
        "vzero.q C620\n"                                                    \
        "vzero.q C020\n"                                                    \
        "1:\n"                                                              \
        /* Slots finish the previous block: total * scales -> rows. */      \
        TILE(0, 0, "vmul.q C620, C620, C020\n",                             \
                   "vadd.q C700, C700, C620\n", MUL)                        \
        "lv.s   S010, 0(%[sc])\n"                                           \
        "lv.s   S011, 4(%[sc])\n"                                           \
        TILE(16, 32, "vh2f.p C020, C010\n",                                 \
                     "vadd.q C600, C500, C510\n", MUL)                      \
        TILE(32, 64, "vadd.q C600, C600, C500\n",                           \
                     "vadd.q C600, C600, C510\n", MUL)                      \
        "lv.s   S030, 0(%[bs])\n"                                           \
        TILE(48, 96, "vadd.q C600, C600, C500\n",                           \
                     "vadd.q C600, C600, C510\n", MUL)                      \
        "vscl.q C610, C020, S030\n"       /* scales * block sum */          \
        "addiu  %[tp], %[tp], 64\n"                                         \
        "vadd.q C620, C600, C500\n"                                         \
        "addiu  %[xt], %[xt], 128\n"                                        \
        "addiu  %[sc], %[sc], 8\n"                                          \
        "vadd.q C710, C710, C610\n"                                         \
        "addiu  %[bs], %[bs], 4\n"                                          \
        "addiu  %[n], %[n], -1\n"                                           \
        "bnez   %[n], 1b\n"                                                 \
        "vadd.q C620, C620, C510\n"       /* branch delay slot */           \
        "vmul.q C620, C620, C020\n"                                         \
        "vfim.s S000, 8.0\n"                                                \
        "vadd.q C700, C700, C620\n"                                         \
        "vscl.q C710, C710, S000\n"                                         \
        "vsub.q C700, C700, C710\n"                                         \
        "sv.q   C700, 0(%[out])\n"                                          \
        ".set pop\n"                                                        \
        : [tp] "+r"(tp), [xt] "+r"(xp), [sc] "+r"(sc), [bs] "+r"(bs),       \
          [n] "+r"(n)                                                       \
        : [out] "r"(out)                                                    \
        : "memory");                                                        \
}

#endif
