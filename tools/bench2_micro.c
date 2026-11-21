/* Part A of LLMPSPBench2: instruction and short-sequence microtimings.
 *
 * Every test is a loop whose body is one "unit" repeated 2 times (lo) or
 * 6 times (hi). The harness runs both with the same iteration count, so
 * (t_hi - t_lo) / (4 * items * n) is the cost of one item with the loop
 * overhead (and any overlap of it with VFPU latency) cancelled exactly.
 *
 * Register conventions (bench2_vfpu_load sets them before every test):
 *   M000-M300  0.0   destination area
 *   M400-M600  1.0   read-only sources (bit pattern 0x3f800000: as fp16
 *                    halves 0x0000/0x3f80, as bytes 00 00 80 3f)
 *   M700       0.0   read-only zero source
 * Throughput tests write eight different quads (C000 C100 C200 C300 C010
 * C110 C210 C310), so a destination is reused only 8 instructions later.
 * Latency chains run on zeros (or on 1.0 multiplied/scaled by 1.0), so no
 * test can reach an infinity or NaN. Every overlap rule from the VFPU
 * docs is respected: vtfm4 outputs never touch its matrix or vector
 * input, vexp2 overlaps only identically.
 *
 * p: 1 KiB scratch of 1.0f (64-byte aligned, reset before every test);
 * q: a 256-float table. All lv.q/sv.q offsets are multiples of 16. */
#include <stdint.h>

#include "bench2.h"

#if defined(_PSP_FW_VERSION)

#define CLOB "memory", "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15"

#define LOOPFN(fname, SETUP, BODY)                                          \
static void fname(unsigned int n, uint8_t *p, float *q) {                   \
    __asm__ volatile (                                                      \
        ".set push\n"                                                       \
        ".set noreorder\n"                                                  \
        SETUP                                                               \
        "1:\n"                                                              \
        BODY                                                                \
        "addiu  %[n], %[n], -1\n"                                           \
        "bnez   %[n], 1b\n"                                                 \
        "nop\n"                                                             \
        ".set pop\n"                                                        \
        : [n] "+r"(n) : [p] "r"(p), [q] "r"(q) : CLOB);                     \
}

#define X2(u) u u
#define X6(u) u u u u u u
#define MICRO(id, SETUP, UNIT)                                              \
    LOOPFN(id##_lo, SETUP, X2(UNIT))                                        \
    LOOPFN(id##_hi, SETUP, X6(UNIT))
#define M(id, name, items, doc) { name, id##_lo, id##_hi, items, doc }
#define HEAD(text) { text, 0, 0, 0, 0 }

void bench2_loop_empty(unsigned int n, uint8_t *p, float *q) {
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "1:\n"
        "addiu  %[n], %[n], -1\n"
        "bnez   %[n], 1b\n"
        "nop\n"
        ".set pop\n"
        : [n] "+r"(n) : [p] "r"(p), [q] "r"(q) : "memory");
}

void bench2_vfpu_load(const float *t) {
    __asm__ volatile (
        "lv.q C000,   0(%[t])\n" "lv.q C010,  16(%[t])\n"
        "lv.q C020,  32(%[t])\n" "lv.q C030,  48(%[t])\n"
        "lv.q C100,  64(%[t])\n" "lv.q C110,  80(%[t])\n"
        "lv.q C120,  96(%[t])\n" "lv.q C130, 112(%[t])\n"
        "lv.q C200, 128(%[t])\n" "lv.q C210, 144(%[t])\n"
        "lv.q C220, 160(%[t])\n" "lv.q C230, 176(%[t])\n"
        "lv.q C300, 192(%[t])\n" "lv.q C310, 208(%[t])\n"
        "lv.q C320, 224(%[t])\n" "lv.q C330, 240(%[t])\n"
        "lv.q C400, 256(%[t])\n" "lv.q C410, 272(%[t])\n"
        "lv.q C420, 288(%[t])\n" "lv.q C430, 304(%[t])\n"
        "lv.q C500, 320(%[t])\n" "lv.q C510, 336(%[t])\n"
        "lv.q C520, 352(%[t])\n" "lv.q C530, 368(%[t])\n"
        "lv.q C600, 384(%[t])\n" "lv.q C610, 400(%[t])\n"
        "lv.q C620, 416(%[t])\n" "lv.q C630, 432(%[t])\n"
        "lv.q C700, 448(%[t])\n" "lv.q C710, 464(%[t])\n"
        "lv.q C720, 480(%[t])\n" "lv.q C730, 496(%[t])\n"
        : : [t] "r"(t) : "memory");
}

/* ------------------------------------------------------------------ */
/* Integer                                                             */
/* ------------------------------------------------------------------ */

MICRO(i_nop, "",
      "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n")
MICRO(i_addiu_thr, "",
      "addiu $8, $8, 1\n"   "addiu $9, $9, 1\n"
      "addiu $10, $10, 1\n" "addiu $11, $11, 1\n"
      "addiu $12, $12, 1\n" "addiu $13, $13, 1\n"
      "addiu $14, $14, 1\n" "addiu $15, $15, 1\n")
MICRO(i_addiu_lat, "",
      "addiu $8, $8, 1\n" "addiu $8, $8, 1\n" "addiu $8, $8, 1\n"
      "addiu $8, $8, 1\n" "addiu $8, $8, 1\n" "addiu $8, $8, 1\n"
      "addiu $8, $8, 1\n" "addiu $8, $8, 1\n")
MICRO(i_lw_thr, "",
      "lw $8, 0(%[p])\n"   "lw $9, 4(%[p])\n"
      "lw $10, 8(%[p])\n"  "lw $11, 12(%[p])\n"
      "lw $12, 16(%[p])\n" "lw $13, 20(%[p])\n"
      "lw $14, 24(%[p])\n" "lw $15, 28(%[p])\n")
/* Pointer chase: word 0 of the scratch holds its own address. */
MICRO(i_lw_lat, "sw %[p], 0(%[p])\n" "move $8, %[p]\n",
      "lw $8, 0($8)\n" "lw $8, 0($8)\n" "lw $8, 0($8)\n" "lw $8, 0($8)\n"
      "lw $8, 0($8)\n" "lw $8, 0($8)\n" "lw $8, 0($8)\n" "lw $8, 0($8)\n")
MICRO(i_sw_thr, "",
      "sw $0, 64(%[p])\n" "sw $0, 68(%[p])\n" "sw $0, 72(%[p])\n"
      "sw $0, 76(%[p])\n" "sw $0, 80(%[p])\n" "sw $0, 84(%[p])\n"
      "sw $0, 88(%[p])\n" "sw $0, 92(%[p])\n")
MICRO(i_sw_lw, "",
      "sw $0, 64(%[p])\n" "lw $8, 64(%[p])\n"
      "sw $0, 68(%[p])\n" "lw $9, 68(%[p])\n"
      "sw $0, 72(%[p])\n" "lw $10, 72(%[p])\n"
      "sw $0, 76(%[p])\n" "lw $11, 76(%[p])\n")
/* A byte code indexing a float table: lbu, sll, addu, lv.s. Scratch
 * bytes are 00 00 80 3f, all valid indices into the 256-entry table. */
MICRO(i_lbu_seq, "",
      "lbu $8, 0(%[p])\n" "sll $8, $8, 2\n" "addu $8, $8, %[q]\n"
      "lv.s S000, 0($8)\n"
      "lbu $9, 3(%[p])\n" "sll $9, $9, 2\n" "addu $9, $9, %[q]\n"
      "lv.s S100, 0($9)\n")
MICRO(i_lbu_ilv, "",
      "lbu $8, 0(%[p])\n" "lbu $9, 3(%[p])\n"
      "sll $8, $8, 2\n" "sll $9, $9, 2\n"
      "addu $8, $8, %[q]\n" "addu $9, $9, %[q]\n"
      "lv.s S000, 0($8)\n" "lv.s S100, 0($9)\n")

/* ------------------------------------------------------------------ */
/* VFPU loads, stores, GPR moves                                       */
/* ------------------------------------------------------------------ */

MICRO(l_lvq_thr, "",
      "lv.q C000, 0(%[p])\n"  "lv.q C100, 16(%[p])\n"
      "lv.q C200, 32(%[p])\n" "lv.q C300, 48(%[p])\n"
      "lv.q C010, 64(%[p])\n" "lv.q C110, 80(%[p])\n"
      "lv.q C210, 96(%[p])\n" "lv.q C310, 112(%[p])\n")
MICRO(l_lvs_thr, "",
      "lv.s S000, 0(%[p])\n"  "lv.s S100, 4(%[p])\n"
      "lv.s S200, 8(%[p])\n"  "lv.s S300, 12(%[p])\n"
      "lv.s S010, 16(%[p])\n" "lv.s S110, 20(%[p])\n"
      "lv.s S210, 24(%[p])\n" "lv.s S310, 28(%[p])\n")
MICRO(l_svq_thr, "",
      "sv.q C400, 128(%[p])\n" "sv.q C410, 144(%[p])\n"
      "sv.q C420, 160(%[p])\n" "sv.q C430, 176(%[p])\n"
      "sv.q C500, 192(%[p])\n" "sv.q C510, 208(%[p])\n"
      "sv.q C520, 224(%[p])\n" "sv.q C530, 240(%[p])\n")
MICRO(l_svs_thr, "",
      "sv.s S400, 128(%[p])\n" "sv.s S410, 132(%[p])\n"
      "sv.s S420, 136(%[p])\n" "sv.s S430, 140(%[p])\n"
      "sv.s S500, 144(%[p])\n" "sv.s S510, 148(%[p])\n"
      "sv.s S520, 152(%[p])\n" "sv.s S530, 156(%[p])\n")
/* Load followed at once by a consumer: pair cost = load-use latency +
 * consumer issue. Four independent pairs per unit. */
MICRO(l_lvq_vadd, "",
      "lv.q C000, 0(%[p])\n"  "vadd.q C100, C000, C400\n"
      "lv.q C010, 16(%[p])\n" "vadd.q C110, C010, C400\n"
      "lv.q C020, 32(%[p])\n" "vadd.q C120, C020, C400\n"
      "lv.q C030, 48(%[p])\n" "vadd.q C130, C030, C400\n")
MICRO(l_lvs_vadds, "",
      "lv.s S000, 0(%[p])\n"  "vadd.s S100, S000, S400\n"
      "lv.s S010, 4(%[p])\n"  "vadd.s S110, S010, S400\n"
      "lv.s S020, 8(%[p])\n"  "vadd.s S120, S020, S400\n"
      "lv.s S030, 12(%[p])\n" "vadd.s S130, S030, S400\n")
MICRO(l_lvq_vc2i, "",
      "lv.q C000, 0(%[p])\n"  "vc2i.s C100, S000\n"
      "lv.q C010, 16(%[p])\n" "vc2i.s C110, S010\n"
      "lv.q C020, 32(%[p])\n" "vc2i.s C120, S020\n"
      "lv.q C030, 48(%[p])\n" "vc2i.s C130, S030\n")
MICRO(l_lvq_vtfm, "",
      "lv.q C000, 0(%[p])\n"  "vtfm4.q C100, M400, C000\n"
      "lv.q C010, 16(%[p])\n" "vtfm4.q C110, M400, C010\n"
      "lv.q C020, 32(%[p])\n" "vtfm4.q C120, M400, C020\n"
      "lv.q C030, 48(%[p])\n" "vtfm4.q C130, M400, C030\n")
MICRO(l_lvs_vh2f, "",
      "lv.s S000, 0(%[p])\n"  "lv.s S001, 4(%[p])\n"
      "vh2f.p C100, C000\n"
      "lv.s S010, 8(%[p])\n"  "lv.s S011, 12(%[p])\n"
      "vh2f.p C110, C010\n"
      "lv.s S020, 16(%[p])\n" "lv.s S021, 20(%[p])\n"
      "vh2f.p C120, C020\n"
      "lv.s S030, 24(%[p])\n" "lv.s S031, 28(%[p])\n"
      "vh2f.p C130, C030\n")
/* Store then load of the same (or a different) address. */
MICRO(l_sw_lvq_same, "",
      "sw $0, 0(%[p])\n"  "lv.q C000, 0(%[p])\n"
      "sw $0, 16(%[p])\n" "lv.q C010, 16(%[p])\n"
      "sw $0, 32(%[p])\n" "lv.q C020, 32(%[p])\n"
      "sw $0, 48(%[p])\n" "lv.q C030, 48(%[p])\n")
MICRO(l_sw_lvq_diff, "",
      "sw $0, 128(%[p])\n" "lv.q C000, 0(%[p])\n"
      "sw $0, 144(%[p])\n" "lv.q C010, 16(%[p])\n"
      "sw $0, 160(%[p])\n" "lv.q C020, 32(%[p])\n"
      "sw $0, 176(%[p])\n" "lv.q C030, 48(%[p])\n")
MICRO(l_svq_lvq_same, "",
      "sv.q C400, 0(%[p])\n"  "lv.q C000, 0(%[p])\n"
      "sv.q C410, 16(%[p])\n" "lv.q C010, 16(%[p])\n"
      "sv.q C420, 32(%[p])\n" "lv.q C020, 32(%[p])\n"
      "sv.q C430, 48(%[p])\n" "lv.q C030, 48(%[p])\n")
MICRO(l_svs_lvs_same, "",
      "sv.s S400, 0(%[p])\n"  "lv.s S000, 0(%[p])\n"
      "sv.s S410, 4(%[p])\n"  "lv.s S010, 4(%[p])\n"
      "sv.s S420, 8(%[p])\n"  "lv.s S020, 8(%[p])\n"
      "sv.s S430, 12(%[p])\n" "lv.s S030, 12(%[p])\n")
MICRO(l_mtv_thr, "move $8, $0\n",
      "mtv $8, S000\n" "mtv $8, S100\n" "mtv $8, S200\n" "mtv $8, S300\n"
      "mtv $8, S010\n" "mtv $8, S110\n" "mtv $8, S210\n" "mtv $8, S310\n")
MICRO(l_mfv_thr, "",
      "mfv $8, S400\n"  "mfv $9, S401\n"  "mfv $10, S402\n" "mfv $11, S403\n"
      "mfv $12, S410\n" "mfv $13, S411\n" "mfv $14, S412\n" "mfv $15, S413\n")
MICRO(l_mtv_mfv, "move $8, $0\n",
      "mtv $8, S000\n" "mfv $8, S000\n" "mtv $8, S000\n" "mfv $8, S000\n"
      "mtv $8, S000\n" "mfv $8, S000\n" "mtv $8, S000\n" "mfv $8, S000\n")
MICRO(l_mfv_addiu, "",
      "mfv $8, S400\n"  "addiu $9, $8, 1\n"
      "mfv $10, S401\n" "addiu $11, $10, 1\n"
      "mfv $12, S402\n" "addiu $13, $12, 1\n"
      "mfv $14, S403\n" "addiu $15, $14, 1\n")

/* ------------------------------------------------------------------ */
/* VFPU throughput: eight independent destinations                     */
/* ------------------------------------------------------------------ */

#define T8(OP, A)                                                           \
    OP " C000, " A "\n" OP " C100, " A "\n" OP " C200, " A "\n"             \
    OP " C300, " A "\n" OP " C010, " A "\n" OP " C110, " A "\n"             \
    OP " C210, " A "\n" OP " C310, " A "\n"

MICRO(t_vc2i, "",
      "vc2i.s C000, S400\n" "vc2i.s C100, S401\n" "vc2i.s C200, S402\n"
      "vc2i.s C300, S403\n" "vc2i.s C010, S410\n" "vc2i.s C110, S411\n"
      "vc2i.s C210, S412\n" "vc2i.s C310, S413\n")
MICRO(t_vi2f28, "", T8("vi2f.q", "C400, 28"))
MICRO(t_vi2f0, "", T8("vi2f.q", "C400, 0"))
MICRO(t_vf2id, "", T8("vf2id.q", "C400, 0"))
MICRO(t_vf2iz, "", T8("vf2iz.q", "C400, 0"))
MICRO(t_vf2in, "", T8("vf2in.q", "C400, 0"))
MICRO(t_vadd, "", T8("vadd.q", "C400, C410"))
MICRO(t_vmul, "", T8("vmul.q", "C400, C410"))
MICRO(t_vscl, "", T8("vscl.q", "C400, S410"))
MICRO(t_vdot, "",
      "vdot.q S000, C400, C410\n" "vdot.q S100, C400, C410\n"
      "vdot.q S200, C400, C410\n" "vdot.q S300, C400, C410\n"
      "vdot.q S010, C400, C410\n" "vdot.q S110, C400, C410\n"
      "vdot.q S210, C400, C410\n" "vdot.q S310, C400, C410\n")
MICRO(t_vtfm4, "", T8("vtfm4.q", "M400, C500"))
MICRO(t_vfad, "",
      "vfad.q S000, C400\n" "vfad.q S100, C400\n" "vfad.q S200, C400\n"
      "vfad.q S300, C400\n" "vfad.q S010, C400\n" "vfad.q S110, C400\n"
      "vfad.q S210, C400\n" "vfad.q S310, C400\n")
MICRO(t_vh2f, "", T8("vh2f.p", "C400"))
MICRO(t_vf2h, "", T8("vf2h.q", "C400"))
MICRO(t_vmov, "", T8("vmov.q", "C400"))
MICRO(t_vmovs, "",
      "vmov.s S000, S400\n" "vmov.s S100, S400\n" "vmov.s S200, S400\n"
      "vmov.s S300, S400\n" "vmov.s S010, S400\n" "vmov.s S110, S400\n"
      "vmov.s S210, S400\n" "vmov.s S310, S400\n")
MICRO(t_vzero, "",
      "vzero.q C000\n" "vzero.q C100\n" "vzero.q C200\n" "vzero.q C300\n"
      "vzero.q C010\n" "vzero.q C110\n" "vzero.q C210\n" "vzero.q C310\n")
MICRO(t_vexp2, "", T8("vexp2.q", "C400"))
/* Same instruction, destinations four columns of one matrix at a time
 * (as the kernel's conversion stages do). */
MICRO(t_vi2f_samemat, "",
      "vi2f.q C000, C400, 28\n" "vi2f.q C010, C400, 28\n"
      "vi2f.q C020, C400, 28\n" "vi2f.q C030, C400, 28\n"
      "vi2f.q C100, C400, 28\n" "vi2f.q C110, C400, 28\n"
      "vi2f.q C120, C400, 28\n" "vi2f.q C130, C400, 28\n")

/* ------------------------------------------------------------------ */
/* VFPU latency: dependent chains                                      */
/* ------------------------------------------------------------------ */

#define R8(s) s s s s s s s s
#define R4(s) s s s s

MICRO(c_vadd, "", R8("vadd.q C000, C000, C700\n"))
MICRO(c_vmul, "", R8("vmul.q C000, C000, C400\n"))
MICRO(c_vscl, "", R8("vscl.q C000, C000, S400\n"))
MICRO(c_vdot, "", R4("vdot.q S010, C000, C400\n" "vdot.q S000, C010, C400\n"))
MICRO(c_vtfm4, "",
      R4("vtfm4.q C100, M400, C000\n" "vtfm4.q C000, M400, C100\n"))
MICRO(c_vfad, "", R4("vfad.q S010, C000\n" "vfad.q S000, C010\n"))
MICRO(c_vi2f, "", R8("vi2f.q C000, C000, 28\n"))
MICRO(c_vf2id, "", R8("vf2id.q C000, C000, 0\n"))
MICRO(c_vc2i, "", R4("vc2i.s C010, S000\n" "vc2i.s C000, S010\n"))
MICRO(c_vh2f, "", R4("vh2f.p C010, C000\n" "vh2f.p C000, C010\n"))
MICRO(c_vf2h, "", R4("vf2h.q C010, C000\n" "vf2h.q C000, C010\n"))
MICRO(c_vmov, "", R4("vmov.q C010, C000\n" "vmov.q C000, C010\n"))
MICRO(c_vmovs, "", R4("vmov.s S010, S000\n" "vmov.s S000, S010\n"))
MICRO(c_vexp2_vmul, "",
      R4("vexp2.q C010, C000\n" "vmul.q C000, C010, C700\n"))

/* ------------------------------------------------------------------ */
/* Producer -> consumer pairs from the kernel                          */
/* Four independent pairs per unit; pair cost = effective latency of   */
/* the producer as seen by that consumer + the consumer's issue time.  */
/* ------------------------------------------------------------------ */

MICRO(p_vc2i_vi2f, "",
      "vc2i.s C000, S400\n" "vi2f.q C100, C000, 28\n"
      "vc2i.s C010, S410\n" "vi2f.q C110, C010, 28\n"
      "vc2i.s C020, S420\n" "vi2f.q C120, C020, 28\n"
      "vc2i.s C030, S430\n" "vi2f.q C130, C030, 28\n")
MICRO(p_vi2f_vf2id, "",
      "vi2f.q C000, C400, 28\n" "vf2id.q C100, C000, 0\n"
      "vi2f.q C010, C410, 28\n" "vf2id.q C110, C010, 0\n"
      "vi2f.q C020, C420, 28\n" "vf2id.q C120, C020, 0\n"
      "vi2f.q C030, C430, 28\n" "vf2id.q C130, C030, 0\n")
MICRO(p_vf2id_vi2f, "",
      "vf2id.q C000, C400, 0\n" "vi2f.q C100, C000, 0\n"
      "vf2id.q C010, C410, 0\n" "vi2f.q C110, C010, 0\n"
      "vf2id.q C020, C420, 0\n" "vi2f.q C120, C020, 0\n"
      "vf2id.q C030, C430, 0\n" "vi2f.q C130, C030, 0\n")
/* vi2f writes the last column of a matrix that vtfm4 then reads. */
MICRO(p_vi2f_vtfm, "",
      "vi2f.q C030, C400, 28\n" "vtfm4.q C600, M000, C500\n"
      "vi2f.q C130, C400, 28\n" "vtfm4.q C610, M100, C500\n"
      "vi2f.q C230, C400, 28\n" "vtfm4.q C620, M200, C500\n"
      "vi2f.q C330, C400, 28\n" "vtfm4.q C630, M300, C500\n")
MICRO(p_vtfm_vadd, "",
      "vtfm4.q C000, M400, C500\n" "vadd.q C100, C000, C500\n"
      "vtfm4.q C010, M400, C500\n" "vadd.q C110, C010, C500\n"
      "vtfm4.q C020, M400, C500\n" "vadd.q C120, C020, C500\n"
      "vtfm4.q C030, M400, C500\n" "vadd.q C130, C030, C500\n")
MICRO(p_vdot_vadd, "",
      "vdot.q S000, C400, C500\n" "vadd.q C100, C000, C500\n"
      "vdot.q S010, C400, C500\n" "vadd.q C110, C010, C500\n"
      "vdot.q S020, C400, C500\n" "vadd.q C120, C020, C500\n"
      "vdot.q S030, C400, C500\n" "vadd.q C130, C030, C500\n")
MICRO(p_vadd_vadd, "",
      "vadd.q C000, C400, C410\n" "vadd.q C100, C000, C400\n"
      "vadd.q C010, C400, C410\n" "vadd.q C110, C010, C400\n"
      "vadd.q C020, C400, C410\n" "vadd.q C120, C020, C400\n"
      "vadd.q C030, C400, C410\n" "vadd.q C130, C030, C400\n")
MICRO(p_vmul_vadd, "",
      "vmul.q C000, C400, C410\n" "vadd.q C100, C000, C400\n"
      "vmul.q C010, C400, C410\n" "vadd.q C110, C010, C400\n"
      "vmul.q C020, C400, C410\n" "vadd.q C120, C020, C400\n"
      "vmul.q C030, C400, C410\n" "vadd.q C130, C030, C400\n")
MICRO(p_vh2f_vmul, "",
      "vh2f.p C000, C400\n" "vmul.q C100, C000, C500\n"
      "vh2f.p C010, C410\n" "vmul.q C110, C010, C500\n"
      "vh2f.p C020, C420\n" "vmul.q C120, C020, C500\n"
      "vh2f.p C030, C430\n" "vmul.q C130, C030, C500\n")
MICRO(p_vzero_vadd, "",
      "vzero.q C000\n" "vadd.q C100, C000, C400\n"
      "vzero.q C010\n" "vadd.q C110, C010, C400\n"
      "vzero.q C020\n" "vadd.q C120, C020, C400\n"
      "vzero.q C030\n" "vadd.q C130, C030, C400\n")

/* ------------------------------------------------------------------ */
/* Probes: register-tracking granularity, write ports, issue mixing    */
/* ------------------------------------------------------------------ */

/* Single written, its whole column read (and the reverse). */
MICRO(g_wS_rC, "",
      "vmov.s S000, S400\n" "vadd.q C100, C000, C500\n"
      "vmov.s S010, S400\n" "vadd.q C110, C010, C500\n"
      "vmov.s S020, S400\n" "vadd.q C120, C020, C500\n"
      "vmov.s S030, S400\n" "vadd.q C130, C030, C500\n")
MICRO(g_wC_rC, "",
      "vmov.q C000, C400\n" "vadd.q C100, C000, C500\n"
      "vmov.q C010, C400\n" "vadd.q C110, C010, C500\n"
      "vmov.q C020, C400\n" "vadd.q C120, C020, C500\n"
      "vmov.q C030, C400\n" "vadd.q C130, C030, C500\n")
MICRO(g_wC_rS, "",
      "vmov.q C000, C400\n" "vadd.s S100, S000, S500\n"
      "vmov.q C010, C400\n" "vadd.s S110, S010, S500\n"
      "vmov.q C020, C400\n" "vadd.s S120, S020, S500\n"
      "vmov.q C030, C400\n" "vadd.s S130, S030, S500\n")
/* Independent pairs that only share a column / a matrix: 1 per instr
 * if the VFPU tracks registers per element / per column. */
MICRO(g_wS_rS_samecol, "",
      "vmov.s S001, S400\n" "vadd.s S100, S000, S500\n"
      "vmov.s S011, S400\n" "vadd.s S110, S010, S500\n"
      "vmov.s S021, S400\n" "vadd.s S120, S020, S500\n"
      "vmov.s S031, S400\n" "vadd.s S130, S030, S500\n")
MICRO(g_wC_rC_samemat, "",
      "vmov.q C010, C400\n" "vadd.q C020, C000, C500\n"
      "vmov.q C110, C400\n" "vadd.q C120, C100, C500\n"
      "vmov.q C210, C400\n" "vadd.q C220, C200, C500\n"
      "vmov.q C310, C400\n" "vadd.q C320, C300, C500\n")
MICRO(g_wC_rC_othermat, "",
      "vmov.q C010, C400\n" "vadd.q C020, C100, C500\n"
      "vmov.q C110, C400\n" "vadd.q C120, C200, C500\n"
      "vmov.q C210, C400\n" "vadd.q C220, C300, C500\n"
      "vmov.q C310, C400\n" "vadd.q C320, C000, C500\n")
/* Independent instructions of different latency. Alternating 3/5 (or
 * 5/7) never finish in the same cycle; in pairs they do, every time.
 * Equal times mean no write-port conflict. */
MICRO(w_35_alt, "",
      "vadd.q C000, C400, C410\n" "vc2i.s C100, S400\n"
      "vadd.q C010, C400, C410\n" "vc2i.s C110, S401\n"
      "vadd.q C020, C400, C410\n" "vc2i.s C120, S402\n"
      "vadd.q C030, C400, C410\n" "vc2i.s C130, S403\n")
MICRO(w_35_pair, "",
      "vadd.q C000, C400, C410\n" "vadd.q C010, C400, C410\n"
      "vc2i.s C100, S400\n" "vc2i.s C110, S401\n"
      "vadd.q C020, C400, C410\n" "vadd.q C030, C400, C410\n"
      "vc2i.s C120, S402\n" "vc2i.s C130, S403\n")
MICRO(w_57_alt, "",
      "vadd.q C000, C400, C410\n" "vdot.q S100, C400, C410\n"
      "vadd.q C010, C400, C410\n" "vdot.q S110, C400, C410\n"
      "vadd.q C020, C400, C410\n" "vdot.q S120, C400, C410\n"
      "vadd.q C030, C400, C410\n" "vdot.q S130, C400, C410\n")
MICRO(w_57_pair, "",
      "vdot.q S100, C400, C410\n" "vdot.q S110, C400, C410\n"
      "vadd.q C000, C400, C410\n" "vadd.q C010, C400, C410\n"
      "vdot.q S120, C400, C410\n" "vdot.q S130, C400, C410\n"
      "vadd.q C020, C400, C410\n" "vadd.q C030, C400, C410\n")
MICRO(x_vadd_lvq_alt, "",
      "vadd.q C000, C400, C410\n" "lv.q C100, 0(%[p])\n"
      "vadd.q C010, C400, C410\n" "lv.q C110, 16(%[p])\n"
      "vadd.q C020, C400, C410\n" "lv.q C120, 32(%[p])\n"
      "vadd.q C030, C400, C410\n" "lv.q C130, 48(%[p])\n")
MICRO(x_vadd_lvq_pair, "",
      "vadd.q C000, C400, C410\n" "vadd.q C010, C400, C410\n"
      "lv.q C100, 0(%[p])\n" "lv.q C110, 16(%[p])\n"
      "vadd.q C020, C400, C410\n" "vadd.q C030, C400, C410\n"
      "lv.q C120, 32(%[p])\n" "lv.q C130, 48(%[p])\n")
MICRO(x_vadd_lvs_alt, "",
      "vadd.q C000, C400, C410\n" "lv.s S100, 0(%[p])\n"
      "vadd.q C010, C400, C410\n" "lv.s S110, 4(%[p])\n"
      "vadd.q C020, C400, C410\n" "lv.s S120, 8(%[p])\n"
      "vadd.q C030, C400, C410\n" "lv.s S130, 12(%[p])\n")
MICRO(x_vadd_addiu, "",
      "vadd.q C000, C400, C410\n" "addiu $8, $8, 1\n"
      "vadd.q C010, C400, C410\n" "addiu $9, $9, 1\n"
      "vadd.q C020, C400, C410\n" "addiu $10, $10, 1\n"
      "vadd.q C030, C400, C410\n" "addiu $11, $11, 1\n")
/* Does anything issue during vtfm4's 4 blocked cycles? 4 per pair if
 * it does, 5 if not. */
MICRO(x_vtfm_lvq, "",
      "vtfm4.q C000, M400, C500\n" "lv.q C100, 0(%[p])\n"
      "vtfm4.q C010, M400, C500\n" "lv.q C110, 16(%[p])\n"
      "vtfm4.q C020, M400, C500\n" "lv.q C120, 32(%[p])\n"
      "vtfm4.q C030, M400, C500\n" "lv.q C130, 48(%[p])\n")
MICRO(x_vtfm_addiu, "",
      "vtfm4.q C000, M400, C500\n" "addiu $8, $8, 1\n"
      "vtfm4.q C010, M400, C500\n" "addiu $9, $9, 1\n"
      "vtfm4.q C020, M400, C500\n" "addiu $10, $10, 1\n"
      "vtfm4.q C030, M400, C500\n" "addiu $11, $11, 1\n")
MICRO(x_vtfm_vadd, "",
      "vtfm4.q C000, M400, C500\n" "vadd.q C100, C400, C410\n"
      "vtfm4.q C010, M400, C500\n" "vadd.q C110, C400, C410\n"
      "vtfm4.q C020, M400, C500\n" "vadd.q C120, C400, C410\n"
      "vtfm4.q C030, M400, C500\n" "vadd.q C130, C400, C410\n")

/* ------------------------------------------------------------------ */
/* One kernel tile as a loop body, and ablations of it. Same registers */
/* and order as falcon_q4_tile4_vtfm's TILE(); tile and x read from the */
/* scratch (1.0f bit patterns), so everything stays in the L1 cache.   */
/* ------------------------------------------------------------------ */

#define MT_TILE(RAW, MID, SA, SB, XA, XB, MUL)                              \
    RAW                                                                     \
    "vc2i.s C200, S100\n" "vc2i.s C210, S101\n"                             \
    "vc2i.s C220, S102\n" "vc2i.s C230, S103\n"                             \
    MID SA                                                                  \
    "vi2f.q C200, C200, 28\n" "vi2f.q C210, C210, 28\n"                     \
    "vi2f.q C220, C220, 28\n" "vi2f.q C230, C230, 28\n"                     \
    SB                                                                      \
    "vf2id.q C300, C200, 0\n" "vf2id.q C310, C210, 0\n"                     \
    "vf2id.q C320, C220, 0\n" "vf2id.q C330, C230, 0\n"                     \
    XA                                                                      \
    "vi2f.q C300, C300, 0\n" "vi2f.q C310, C310, 0\n"                       \
    "vi2f.q C320, C320, 0\n" "vi2f.q C330, C330, 0\n"                       \
    XB MUL

#define MT_RAW "lv.q C100, 0(%[p])\n"
#define MT_XA "lv.q C400, 0(%[p])\n"
#define MT_XB "lv.q C410, 16(%[p])\n"
#define MT_SA "vadd.q C600, C600, C500\n"
#define MT_SB "vadd.q C600, C600, C510\n"
#define MT_VTFM "vtfm4.q C500, M200, C400\n" "vtfm4.q C510, M300, C410\n"
#define MT_VDOT                                                             \
    "vdot.q S500, C200, C400\n" "vdot.q S501, C210, C400\n"                 \
    "vdot.q S502, C220, C400\n" "vdot.q S503, C230, C400\n"                 \
    "vdot.q S510, C300, C410\n" "vdot.q S511, C310, C410\n"                 \
    "vdot.q S512, C320, C410\n" "vdot.q S513, C330, C410\n"

MICRO(k_full, "", MT_TILE(MT_RAW, "", MT_SA, MT_SB, MT_XA, MT_XB, MT_VTFM))
MICRO(k_vdot, "", MT_TILE(MT_RAW, "", MT_SA, MT_SB, MT_XA, MT_XB, MT_VDOT))
MICRO(k_noslot, "", MT_TILE(MT_RAW, "", "", "", MT_XA, MT_XB, MT_VTFM))
MICRO(k_nomul, "", MT_TILE(MT_RAW, "", MT_SA, MT_SB, MT_XA, MT_XB, ""))
MICRO(k_noload, "", MT_TILE("", "", MT_SA, MT_SB, "", "", MT_VTFM))
/* raw tile loaded after the vc2i stage, for the next tile */
MICRO(k_pre, "", MT_TILE("", MT_RAW, MT_SA, MT_SB, MT_XA, MT_XB, MT_VTFM))
/* x quads loaded right after the vc2i stage */
MICRO(k_xearly, "",
      MT_TILE(MT_RAW, MT_XA MT_XB, MT_SA, MT_SB, "", "", MT_VTFM))
MICRO(k_allearly, "",
      MT_TILE("", MT_RAW MT_XA MT_XB, MT_SA, MT_SB, "", "", MT_VTFM))
MICRO(k_conv, "", MT_TILE("", "", "", "", "", "", ""))
MICRO(k_conv_slots, "", MT_TILE("", "", MT_SA, MT_SB, "", "", ""))

/* ------------------------------------------------------------------ */
/* Block scales: four codes -> four floats in a quad                   */
/* ------------------------------------------------------------------ */

/* Today's format: two fp16 words. Two blocks per unit. */
MICRO(s_fp16, "",
      "lv.s S010, 0(%[p])\n" "lv.s S011, 4(%[p])\n" "vh2f.p C020, C010\n"
      "lv.s S110, 8(%[p])\n" "lv.s S111, 12(%[p])\n" "vh2f.p C120, C110\n")
MICRO(s_fp16_use, "",
      "lv.s S010, 0(%[p])\n" "lv.s S011, 4(%[p])\n" "vh2f.p C020, C010\n"
      "vmul.q C600, C020, C500\n"
      "lv.s S110, 8(%[p])\n" "lv.s S111, 12(%[p])\n" "vh2f.p C120, C110\n"
      "vmul.q C610, C120, C500\n")
/* Four lbu + table lv.s (codes index a 256-float table). */
#define S_LBU                                                               \
    "lbu $8, 0(%[p])\n" "lbu $9, 1(%[p])\n"                                 \
    "lbu $10, 2(%[p])\n" "lbu $11, 3(%[p])\n"                               \
    "sll $8, $8, 2\n" "sll $9, $9, 2\n"                                     \
    "sll $10, $10, 2\n" "sll $11, $11, 2\n"                                 \
    "addu $8, $8, %[q]\n" "addu $9, $9, %[q]\n"                             \
    "addu $10, $10, %[q]\n" "addu $11, $11, %[q]\n"                         \
    "lv.s S020, 0($8)\n" "lv.s S021, 0($9)\n"                               \
    "lv.s S022, 0($10)\n" "lv.s S023, 0($11)\n"
MICRO(s_lbu, "", S_LBU)
MICRO(s_lbu_use, "", S_LBU "vmul.q C600, C020, C500\n")
/* One lw, shifts and masks give code * 4, then table lv.s. */
#define S_LW                                                                \
    "lw $8, 0(%[p])\n"                                                      \
    "sll $9, $8, 2\n" "srl $10, $8, 6\n"                                    \
    "srl $11, $8, 14\n" "srl $12, $8, 22\n"                                 \
    "andi $9, $9, 0x3fc\n" "andi $10, $10, 0x3fc\n"                         \
    "andi $11, $11, 0x3fc\n" "andi $12, $12, 0x3fc\n"                       \
    "addu $9, $9, %[q]\n" "addu $10, $10, %[q]\n"                           \
    "addu $11, $11, %[q]\n" "addu $12, $12, %[q]\n"                         \
    "lv.s S020, 0($9)\n" "lv.s S021, 0($10)\n"                              \
    "lv.s S022, 0($11)\n" "lv.s S023, 0($12)\n"
MICRO(s_lw, "", S_LW)
MICRO(s_lw_use, "", S_LW "vmul.q C600, C020, C500\n")
/* Log-domain codes in the VFPU: one lv.s, vc2i.s (signed byte << 24),
 * vi2f.q 28 (= code / 16, a step of 2^(1/16)), vexp2.q. Two blocks per
 * unit, each run as an isolated sequence. */
#define S_EXP2(SRC, D, OFF)                                                 \
    "lv.s " SRC ", " OFF "(%[p])\n" "vc2i.s " D ", " SRC "\n"                \
    "vi2f.q " D ", " D ", 28\n" "vexp2.q " D ", " D "\n"
MICRO(s_exp2, "", S_EXP2("S010", "C020", "0") S_EXP2("S110", "C120", "4"))
MICRO(s_exp2_use, "",
      S_EXP2("S010", "C020", "0") "vmul.q C600, C020, C500\n"
      S_EXP2("S110", "C120", "4") "vmul.q C610, C120, C500\n")
/* Four blocks interleaved stage by stage: the latency-hidden cost. */
MICRO(s_exp2_ilv, "",
      "lv.s S010, 0(%[p])\n" "lv.s S110, 4(%[p])\n"
      "lv.s S210, 8(%[p])\n" "lv.s S310, 12(%[p])\n"
      "vc2i.s C020, S010\n" "vc2i.s C120, S110\n"
      "vc2i.s C220, S210\n" "vc2i.s C320, S310\n"
      "vi2f.q C020, C020, 28\n" "vi2f.q C120, C120, 28\n"
      "vi2f.q C220, C220, 28\n" "vi2f.q C320, C320, 28\n"
      "vexp2.q C020, C020\n" "vexp2.q C120, C120\n"
      "vexp2.q C220, C220\n" "vexp2.q C320, C320\n")
/* Any step and offset: vi2f 24, vscl by the step, vadd the offset. */
#define S_EXP2G(SRC, D, OFF)                                                \
    "lv.s " SRC ", " OFF "(%[p])\n" "vc2i.s " D ", " SRC "\n"                \
    "vi2f.q " D ", " D ", 24\n" "vscl.q " D ", " D ", S400\n"               \
    "vadd.q " D ", " D ", C410\n" "vexp2.q " D ", " D "\n"
MICRO(s_exp2g, "", S_EXP2G("S010", "C020", "0") S_EXP2G("S110", "C120", "4"))

/* ------------------------------------------------------------------ */

const MicroInfo bench2_micro[] = {
    HEAD("integer (CPU)"),
    M(i_nop, "nop thr", 8, "1"),
    M(i_addiu_thr, "addiu thr", 8, "1"),
    M(i_addiu_lat, "addiu lat", 8, "1"),
    M(i_lw_thr, "lw thr", 8, "1"),
    M(i_lw_lat, "lw lat (ptr chase)", 8, "?"),
    M(i_sw_thr, "sw thr", 8, "1"),
    M(i_sw_lw, "sw>lw same addr", 4, "?"),
    M(i_lbu_seq, "lbu>sll>addu>lv.s", 2, "4"),
    M(i_lbu_ilv, "lbu..lv.s x2 ilv", 2, "4"),
    HEAD("VFPU load/store, GPR moves"),
    M(l_lvq_thr, "lv.q thr", 8, "?"),
    M(l_lvs_thr, "lv.s thr", 8, "?"),
    M(l_svq_thr, "sv.q thr", 8, "?"),
    M(l_svs_thr, "sv.s thr", 8, "?"),
    M(l_lvq_vadd, "lv.q>vadd.q pair", 4, "?"),
    M(l_lvs_vadds, "lv.s>vadd.s pair", 4, "?"),
    M(l_lvq_vc2i, "lv.q>vc2i.s pair", 4, "?"),
    M(l_lvq_vtfm, "lv.q>vtfm4 pair", 4, "?"),
    M(l_lvs_vh2f, "2 lv.s>vh2f.p", 4, "?"),
    M(l_sw_lvq_same, "sw>lv.q same addr", 4, "?"),
    M(l_sw_lvq_diff, "sw>lv.q other addr", 4, "?"),
    M(l_svq_lvq_same, "sv.q>lv.q same addr", 4, "?"),
    M(l_svs_lvs_same, "sv.s>lv.s same addr", 4, "?"),
    M(l_mtv_thr, "mtv thr", 8, "?"),
    M(l_mfv_thr, "mfv thr", 8, "?"),
    M(l_mtv_mfv, "mtv>mfv round trip", 4, "?"),
    M(l_mfv_addiu, "mfv>addiu pair", 4, "?"),
    HEAD("VFPU throughput (8 independent dests)"),
    M(t_vc2i, "vc2i.s thr", 8, "1"),
    M(t_vi2f28, "vi2f.q 28 thr", 8, "1"),
    M(t_vi2f0, "vi2f.q 0 thr", 8, "1"),
    M(t_vf2id, "vf2id.q thr", 8, "1"),
    M(t_vf2iz, "vf2iz.q thr", 8, "1"),
    M(t_vf2in, "vf2in.q thr", 8, "1"),
    M(t_vadd, "vadd.q thr", 8, "1"),
    M(t_vmul, "vmul.q thr", 8, "1"),
    M(t_vscl, "vscl.q thr", 8, "1"),
    M(t_vdot, "vdot.q thr", 8, "1"),
    M(t_vtfm4, "vtfm4.q thr", 8, "4"),
    M(t_vfad, "vfad.q thr", 8, "1"),
    M(t_vh2f, "vh2f.p thr", 8, "1"),
    M(t_vf2h, "vf2h.q thr", 8, "1"),
    M(t_vmov, "vmov.q thr", 8, "1"),
    M(t_vmovs, "vmov.s thr", 8, "1"),
    M(t_vzero, "vzero.q thr", 8, "1"),
    M(t_vexp2, "vexp2.q thr", 8, "4"),
    M(t_vi2f_samemat, "vi2f.q thr same mtx", 8, "1"),
    HEAD("VFPU latency (dependent chains)"),
    M(c_vadd, "vadd.q lat", 8, "5"),
    M(c_vmul, "vmul.q lat", 8, "5"),
    M(c_vscl, "vscl.q lat", 8, "5"),
    M(c_vdot, "vdot.q lat", 8, "7"),
    M(c_vtfm4, "vtfm4.q lat", 8, "10"),
    M(c_vfad, "vfad.q lat", 8, "7"),
    M(c_vi2f, "vi2f.q lat", 8, "5"),
    M(c_vf2id, "vf2id.q lat", 8, "5"),
    M(c_vc2i, "vc2i.s lat", 8, "3"),
    M(c_vh2f, "vh2f.p lat", 8, "5"),
    M(c_vf2h, "vf2h.q lat", 8, "5"),
    M(c_vmov, "vmov.q lat", 8, "3"),
    M(c_vmovs, "vmov.s lat", 8, "3"),
    M(c_vexp2_vmul, "vexp2>vmul chain", 4, "15"),
    HEAD("kernel pairs (producer>consumer)"),
    M(p_vc2i_vi2f, "vc2i>vi2f pair", 4, "4"),
    M(p_vi2f_vf2id, "vi2f>vf2id pair", 4, "6"),
    M(p_vf2id_vi2f, "vf2id>vi2f pair", 4, "6"),
    M(p_vi2f_vtfm, "vi2f>vtfm4(mtx) pair", 4, "9"),
    M(p_vtfm_vadd, "vtfm4>vadd pair", 4, "11"),
    M(p_vdot_vadd, "vdot(S)>vadd(C) pair", 4, "8"),
    M(p_vadd_vadd, "vadd>vadd pair", 4, "6"),
    M(p_vmul_vadd, "vmul>vadd pair", 4, "6"),
    M(p_vh2f_vmul, "vh2f>vmul pair", 4, "6"),
    M(p_vzero_vadd, "vzero>vadd pair", 4, "4"),
    HEAD("probes: granularity, write port, mixing"),
    M(g_wS_rC, "write S, read its C", 4, "4"),
    M(g_wC_rC, "write C, read C", 4, "4"),
    M(g_wC_rS, "write C, read its S", 4, "4"),
    M(g_wS_rS_samecol, "indep S, same column", 4, "2"),
    M(g_wC_rC_samemat, "indep C, same matrix", 4, "2"),
    M(g_wC_rC_othermat, "indep C, other matrix", 4, "2"),
    M(w_35_alt, "lat3/5 alternating", 8, "1"),
    M(w_35_pair, "lat3/5 in pairs", 8, "1"),
    M(w_57_alt, "lat5/7 alternating", 8, "1"),
    M(w_57_pair, "lat5/7 in pairs", 8, "1"),
    M(x_vadd_lvq_alt, "vadd/lv.q alternating", 8, "1"),
    M(x_vadd_lvq_pair, "vadd/lv.q in pairs", 8, "1"),
    M(x_vadd_lvs_alt, "vadd/lv.s alternating", 8, "1"),
    M(x_vadd_addiu, "vadd/addiu alternating", 8, "1"),
    M(x_vtfm_lvq, "vtfm4 + lv.q pair", 4, "5"),
    M(x_vtfm_addiu, "vtfm4 + addiu pair", 4, "5"),
    M(x_vtfm_vadd, "vtfm4 + vadd pair", 4, "5"),
    HEAD("one kernel tile as loop body (per tile)"),
    M(k_full, "tile: as shipped", 1, "29"),
    M(k_vdot, "tile: vdot multiply", 1, "29"),
    M(k_noslot, "tile: no slot adds", 1, "27"),
    M(k_nomul, "tile: no vtfm4", 1, "21"),
    M(k_noload, "tile: no lv.q", 1, "26"),
    M(k_pre, "tile: raw lv.q early", 1, "29"),
    M(k_xearly, "tile: x lv.q early", 1, "29"),
    M(k_allearly, "tile: all lv.q early", 1, "29"),
    M(k_conv, "tile: 16 conversions", 1, "18"),
    M(k_conv_slots, "tile: conv + 2 slots", 1, "18"),
    HEAD("block scales: 4 codes -> quad (per block)"),
    M(s_fp16, "fp16: 2 lv.s+vh2f", 2, "3"),
    M(s_fp16_use, "fp16 + use", 2, "?"),
    M(s_lbu, "4 lbu+table lv.s", 1, "16"),
    M(s_lbu_use, "4 lbu+table + use", 1, "?"),
    M(s_lw, "lw+shifts+table", 1, "17"),
    M(s_lw_use, "lw+shifts+table + use", 1, "?"),
    M(s_exp2, "vexp2 isolated", 2, "7"),
    M(s_exp2_use, "vexp2 isolated + use", 2, "?"),
    M(s_exp2_ilv, "vexp2 4 interleaved", 4, "7"),
    M(s_exp2g, "vexp2 any step/offset", 2, "9"),
};
const int bench2_micro_count =
    (int)(sizeof(bench2_micro) / sizeof(bench2_micro[0]));

/* ------------------------------------------------------------------ */
/* Scale routes run once, for the accuracy check                       */
/* ------------------------------------------------------------------ */

void bench2_route_lbu(const uint32_t *codes, const float *table, float *out) {
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "lbu $8, 0(%[c])\n" "lbu $9, 1(%[c])\n"
        "lbu $10, 2(%[c])\n" "lbu $11, 3(%[c])\n"
        "sll $8, $8, 2\n" "sll $9, $9, 2\n"
        "sll $10, $10, 2\n" "sll $11, $11, 2\n"
        "addu $8, $8, %[t]\n" "addu $9, $9, %[t]\n"
        "addu $10, $10, %[t]\n" "addu $11, $11, %[t]\n"
        "lv.s S020, 0($8)\n" "lv.s S021, 0($9)\n"
        "lv.s S022, 0($10)\n" "lv.s S023, 0($11)\n"
        "sv.q C020, 0(%[o])\n"
        ".set pop\n"
        : : [c] "r"(codes), [t] "r"(table), [o] "r"(out)
        : "memory", "$8", "$9", "$10", "$11");
}

void bench2_route_lw(const uint32_t *codes, const float *table, float *out) {
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "lw $8, 0(%[c])\n"
        "sll $9, $8, 2\n" "srl $10, $8, 6\n"
        "srl $11, $8, 14\n" "srl $12, $8, 22\n"
        "andi $9, $9, 0x3fc\n" "andi $10, $10, 0x3fc\n"
        "andi $11, $11, 0x3fc\n" "andi $12, $12, 0x3fc\n"
        "addu $9, $9, %[t]\n" "addu $10, $10, %[t]\n"
        "addu $11, $11, %[t]\n" "addu $12, $12, %[t]\n"
        "lv.s S020, 0($9)\n" "lv.s S021, 0($10)\n"
        "lv.s S022, 0($11)\n" "lv.s S023, 0($12)\n"
        "sv.q C020, 0(%[o])\n"
        ".set pop\n"
        : : [c] "r"(codes), [t] "r"(table), [o] "r"(out)
        : "memory", "$8", "$9", "$10", "$11", "$12");
}

void bench2_route_exp2(const uint32_t *codes, float *out) {
    __asm__ volatile (
        "lv.s    S010, 0(%[c])\n"
        "vc2i.s  C020, S010\n"
        "vi2f.q  C020, C020, 28\n"
        "vexp2.q C020, C020\n"
        "sv.q    C020, 0(%[o])\n"
        : : [c] "r"(codes), [o] "r"(out) : "memory");
}

void bench2_route_exp2g(const uint32_t *codes, const float *sb, float *out) {
    __asm__ volatile (
        "lv.s    S010, 0(%[c])\n"
        "lv.q    C120, 0(%[s])\n"
        "lv.s    S030, 16(%[s])\n"
        "vc2i.s  C020, S010\n"
        "vi2f.q  C020, C020, 24\n"
        "vscl.q  C020, C020, S030\n"
        "vadd.q  C020, C020, C120\n"
        "vexp2.q C020, C020\n"
        "sv.q    C020, 0(%[o])\n"
        : : [c] "r"(codes), [s] "r"(sb), [o] "r"(out) : "memory");
}

#else
typedef int bench2_micro_unused;
#endif
