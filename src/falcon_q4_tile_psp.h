/* Four-row VFPU Q4 kernel (FHQ4 version 3 "tile" layout with 8-bit block
 * scale codes), used by src/falcon_q4_psp.c (falcon_q4_tile4_m3/m4/m5,
 * the kernels the app uses). Include only from PSP builds. The version-2
 * (fp16 scale) kernel lives on in tools/q4_tile_v2_psp.h for the old
 * benchmark only.
 *
 * Why (PROJECT.md sections 2.8-2.10): the version-1 row kernel spent about a
 * third of its ~96 instructions per 32 weights unpacking nibbles with
 * integer ops through a scratch buffer; this one does the whole job in the
 * VFPU and handles four rows per pass. Measured on a PSP-2000 (version-2
 * kernel): 1.75 cycles per weight against 4.83 for the row kernel.
 *
 * Tile format. Four consecutive rows form a group:
 *
 *     blocks * 4 bytes   scale codes, rows 0..3 of block 0, then block 1...
 *     blocks * 64 bytes  tiles, four per 32-column block
 *
 * Tile t of block b is 16 bytes: word r belongs to row r, and its byte k
 * packs two weights of that row,
 *
 *     low nibble   w[32b + 4t + k] + 8        (offset binary, as in FHQ4)
 *     high nibble  w[32b + 16 + 4t + k]       (two's complement)
 *
 * which is the FHQ4 byte XOR 0x80, rearranged. 17 bytes per 32 weights;
 * blocks is a multiple of 4, so every tile and group is 16-byte aligned.
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
 * Scale codes. The four code bytes of a block (one per row) are one word.
 * Code c (signed byte) decodes to 0 for c == 0, otherwise to
 * sign(c) * float_from_bits((|c| + k) << (23 - m)), a minifloat with m
 * mantissa bits whose placement k is per matrix (PROJECT.md 2.10). In the
 * VFPU, for all four rows at once:
 *
 *     lv.s   S010            the four codes
 *     vc2i.s C110, S010      c << 24
 *     vi2f.q C110, C110, M+1 c * 2^(23 - m)       (exact)
 *     vabs.q C120, C110      |c| * 2^(23 - m)
 *     vsgn.q C130, C110      sign(c): -1, 0, +1
 *     vadd.q C120, C120, C730  (|c| + k) * 2^(23 - m), an integer < 2^31
 *     vf2iz.q C120, C120, 0  ... as an int: the float's bit pattern
 *     vmul.q C020, C120, C130  signed scale (0 for c == 0)
 *
 * Every value is an integer below 2^24 times a power of two, so each step
 * is exact and the decoded bits equal scale_code_value() in
 * src/falcon_h1.c on any IEEE implementation (PPSSPP and hardware alike).
 * The loader guarantees the magnitudes are normal floats. That is 8
 * instructions per block against 3 for the fp16 scales (lv.s x2 + vh2f.p);
 * the loop now handles two blocks per iteration, which saves 3 integer
 * instructions per block, so the net cost is +2 (107 instead of 105 per
 * 128 weights).
 *
 * Scheduling follows the documented timings (throughput 1 and latency 5
 * for vi2f/vf2id/vadd/vmul, latency 3 for vc2i, throughput 4 and latency
 * 10 for vtfm4.q): every tile has four slots (after the vc2i, vi2f and
 * vf2id groups and after the vtfm4 pair) that carry the previous tile's
 * adds and the scale decode chain, and a block's final scale-multiply is
 * deferred into the next block's first tile.
 *
 * Register use: M200 = G, M300 = H, C100 raw tile, C400/C410 x quads,
 * C500/C510 row partials, C600 block total, C620 finished block total
 * waiting for its scale, C020 scales, C110/C120/C130 scale decode, C730
 * the bias quad, C700 row accumulators, C710 the low-nibble correction.
 * vtfm4 allows no overlap between its output and inputs, which this
 * layout respects.
 *
 * Alignment contract. group, xt, out and bias are read/written with
 * lv.q/sv.q and MUST be 16-byte aligned: real hardware raises an address
 * error otherwise, and PPSSPP cannot be relied on to show it. blocks must
 * be a multiple of 4 (tiles start blocks * 4 bytes into the group; the
 * two-block loop also needs it even). The codes and block_sums are read
 * with lv.s (4-byte alignment). q4_matvec in src/falcon_h1.c guarantees
 * all of this and keeps a counted copy guard for the group pointer. */
#ifndef FALCON_Q4_TILE_PSP_H
#define FALCON_Q4_TILE_PSP_H

#include <stdint.h>

/* One tile. SLOT_A..SLOT_D are instruction strings (possibly empty)
 * executed between the conversion stages. */
#define TILE(TOFF, XOFF, SLOT_A, SLOT_B, SLOT_C, SLOT_D)                    \
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
        SLOT_C                                                              \
        "lv.q   C400, " #XOFF "(%[xt])\n"                                   \
        "vi2f.q C300, C300, 0\n"                                            \
        "vi2f.q C310, C310, 0\n"                                            \
        "vi2f.q C320, C320, 0\n"                                            \
        "vi2f.q C330, C330, 0\n"                                            \
        "lv.q   C410, " #XOFF "+16(%[xt])\n"                                \
        "vtfm4.q C500, M200, C400\n"                                        \
        "vtfm4.q C510, M300, C410\n"                                        \
        SLOT_D

/* One 32-column block: four tiles at tile offset TB, x offset XB, the
 * block's code word at SCOFF(sc) and its low-nibble sum at BSOFF(bs).
 * SCALE is the vi2f immediate, m + 1. TAIL1/TAIL2 are the integer
 * instructions (pointer and loop bookkeeping) interleaved with the
 * block's closing adds. */
#define Q4_BLOCK(TB, XB, SCOFF, BSOFF, SCALE, TAIL1, TAIL2)                 \
        /* Slots A/B finish the previous block: total * scales -> rows; */  \
        /* slots C/D start decoding this block's scale codes. */            \
        TILE(TB, XB,                                                        \
             "vmul.q C620, C620, C020\n",                                   \
             "vadd.q C700, C700, C620\n",                                   \
             "lv.s   S010, " #SCOFF "(%[sc])\n",                            \
             "vc2i.s C110, S010\n")                                         \
        TILE(TB+16, XB+32,                                                  \
             "vi2f.q C110, C110, " #SCALE "\n",                             \
             "vadd.q C600, C500, C510\n",                                   \
             "vabs.q C120, C110\n",                                         \
             "vsgn.q C130, C110\n")                                         \
        TILE(TB+32, XB+64,                                                  \
             "vadd.q C600, C600, C500\n",                                   \
             "vadd.q C600, C600, C510\n",                                   \
             "vadd.q C120, C120, C730\n",                                   \
             "lv.s   S030, " #BSOFF "(%[bs])\n")                            \
        TILE(TB+48, XB+96,                                                  \
             "vadd.q C600, C600, C500\n"                                    \
             "vf2iz.q C120, C120, 0\n",                                     \
             "vadd.q C600, C600, C510\n",                                   \
             "vmul.q C020, C120, C130\n",                                   \
             "")                                                            \
        "vscl.q C610, C020, S030\n"       /* scales * block sum */          \
        TAIL1                                                               \
        "vadd.q C620, C600, C500\n"                                         \
        TAIL2                                                               \
        "vadd.q C710, C710, C610\n"

/* group: one tile-format group (16-byte aligned).
 * xt: per block 4 x {x16 quad, xd quad}, i.e. 32 floats per block.
 * block_sums: per block the sum of x over its 16 low-nibble columns.
 * out: four results, 16-byte aligned. blocks: a multiple of 4.
 * bias: four copies of k * 2^(23 - m) (16-byte aligned). */
#define DEFINE_TILE_KERNEL(NAME, SCALE)                                     \
void NAME(const uint8_t *group, const float *xt,                            \
          const float *block_sums, int blocks, float *out,                  \
          const float *bias) {                                              \
    const uint8_t *sc = group;                                              \
    const uint8_t *tp = group + (unsigned)blocks * 4u;                      \
    const float *xp = xt;                                                   \
    const float *bs = block_sums;                                           \
    int n = blocks;                                                         \
    __asm__ volatile (                                                      \
        ".set push\n"                                                       \
        ".set noreorder\n"                                                  \
        "lv.q   C730, 0(%[bias])\n"                                         \
        "vzero.q C700\n"                                                    \
        "vzero.q C710\n"                                                    \
        "vzero.q C620\n"                                                    \
        "vzero.q C020\n"                                                    \
        "1:\n"                                                              \
        /* First block of the pair; bs advances here, so the second */      \
        /* block finds its sum at -4. */                                    \
        Q4_BLOCK(0, 0, 0, 0, SCALE,                                         \
                 "addiu  %[n], %[n], -2\n",                                 \
                 "addiu  %[bs], %[bs], 8\n")                                \
        "vadd.q C620, C620, C510\n"                                         \
        Q4_BLOCK(64, 128, 4, -4, SCALE,                                     \
                 "addiu  %[tp], %[tp], 128\n",                              \
                 "addiu  %[xt], %[xt], 256\n"                               \
                 "addiu  %[sc], %[sc], 8\n")                                \
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
        : [out] "r"(out), [bias] "r"(bias)                                  \
        : "memory");                                                        \
}

#endif
