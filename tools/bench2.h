/* Shared declarations of the second on-device benchmark (LLMPSPBench2):
 * tools/bench2_psp.c (harness), tools/bench2_micro.c (Part A instruction
 * microtimings), tools/bench2_kernels.c (Part B kernel variants).
 * Self-contained: nothing here links against src/. */
#ifndef BENCH2_H
#define BENCH2_H

#include <stdint.h>

/* Part A. Each test is one loop; "lo" repeats its unit 2 times per
 * iteration, "hi" 6 times, so (t_hi - t_lo) / (4 * items * n) is the
 * cost of one item with the loop overhead cancelled exactly.
 * p: 1 KiB scratch, 64-byte aligned, filled with 1.0f before every test.
 * q: 256-float table (1 KiB), 64-byte aligned. */
typedef void (*MicroFn)(unsigned int n, uint8_t *p, float *q);

typedef struct {
    const char *name;   /* lo == 0: a section heading */
    MicroFn lo, hi;
    int items;          /* measured items per unit */
    const char *doc;    /* expected cycles per item from the VFPU docs */
} MicroInfo;

extern const MicroInfo bench2_micro[];
extern const int bench2_micro_count;

/* Plain 3-instruction loop (addiu, bnez, nop): the loop overhead. */
void bench2_loop_empty(unsigned int n, uint8_t *p, float *q);

/* Loads all 32 VFPU quads from a 512-byte, 16-aligned table. */
void bench2_vfpu_load(const float *table);

/* Scale-code routes run once on real data, for the accuracy check.
 * codes: one 4-byte aligned word holding four 8-bit codes; out: four
 * floats, 16-byte aligned. */
void bench2_route_lbu(const uint32_t *codes, const float *table, float *out);
void bench2_route_lw(const uint32_t *codes, const float *table, float *out);
/* out[i] = 2^((int8_t)code_i / 16) via vc2i.s, vi2f.q 28, vexp2.q. */
void bench2_route_exp2(const uint32_t *codes, float *out);
/* out[i] = 2^((int8_t)code_i * sb[4] + sb[i]); sb: 16-aligned, 5 floats
 * (a bias quad, then the step). */
void bench2_route_exp2g(const uint32_t *codes, const float *sb, float *out);

/* Part B. Same contract as falcon_q4_tile4_vtfm (version-2 tile format):
 * group and out 16-byte aligned, blocks even and >= 2 in this benchmark. */
typedef void (*TileKernel)(const uint8_t *group, const float *xt,
                           const float *block_sums, int blocks, float *out);

typedef struct {
    const char *name;
    TileKernel fn;
    int instr;          /* instructions per 32-column block (4 rows) */
    const char *why;
} KernelInfo;

extern const KernelInfo bench2_kernels[];
extern const int bench2_kernel_count;

#endif
