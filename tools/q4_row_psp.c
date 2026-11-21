/* The FHQ4 version-1 single-row VFPU Q4 kernel, as shipped through v1.11.
 * The app no longer uses it (version 2 stores tile groups for
 * falcon_q4_tile4_vtfm); it is kept here only so tools/bench_psp.c can
 * keep timing the old kernel against the new one on the same weights.
 *
 * It accumulates raw 0..15 nibbles and corrects with -8 * (per-32-column
 * block sums of x). Denormal fp16 scales flush to zero. */
#include <stdint.h>

#if defined(_PSP_FW_VERSION)

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

#else
/* ISO C forbids an empty translation unit, hence the typedef. */
typedef int falcon_q4_row_psp_unused;
#endif
