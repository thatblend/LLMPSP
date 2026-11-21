/* Second on-device benchmark (LLMPSPBench2). The vtfm4 tile kernel runs
 * at 1.35 cycles per weight with the data in L1, ~44 cycles per 128-weight
 * block more than the documented VFPU timings predict, and PPSSPP does not
 * model instruction timing. One run of this EBOOT on a real PSP:
 *
 *   Part A  times every instruction and short sequence the kernel is made
 *           of (throughput, latency, producer->consumer pairs, register
 *           tracking and write-port probes, the kernel tile itself with
 *           parts removed, and candidate 8-bit scale-code routes);
 *   Part B  verifies eight kernel schedules against a scalar reference and
 *           times them with the data in L1 and streaming from 8 MiB, then
 *           ranks them.
 *
 * Everything goes to bench2_log.txt beside the EBOOT; the screen shows a
 * summary. A file bench2_quick.txt beside the EBOOT shortens every timing
 * ~40x (for the emulator only; the timings are meaningless there).
 *
 * Build: tools/bench2_build.sh (tools/Makefile.bench2). Self-contained:
 * nothing links against src/. No input needed; X or HOME exits at the
 * end, or it leaves by itself after a minute. */
#include <pspctrl.h>
#include <pspdebug.h>
#include <pspdisplay.h>
#include <pspiofilemgr.h>
#include <pspkernel.h>
#include <psppower.h>

#include <malloc.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench2.h"

PSP_MODULE_INFO("LLMPSPBench2", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(-2048);

#define BENCH_VERSION "1"

#define COLS 512
#define BLOCKS (COLS / 32)
#define GROUP_BYTES (BLOCKS * 72)        /* 4 rows x 18 bytes per block */
#define L1_GROUPS 4                      /* 4.5 KiB: stays in the D-cache */
#define VERIFY_GROUPS 256                /* 1024 rows */

static volatile int exiting;
static int quick;

/* ------------------------------------------------------------------ */
/* Output                                                              */
/* ------------------------------------------------------------------ */

static SceUID log_fd = -1;
static char log_path[512];
static char exe_dir[512];

static void log_sync(void) {
    if (log_fd < 0) return;
    sceIoClose(log_fd);
    log_fd = sceIoOpen(log_path, PSP_O_WRONLY | PSP_O_CREAT | PSP_O_APPEND,
                       0777);
}

static void emit(int on_screen, const char *format, va_list args) {
    char line[256];
    int length = vsnprintf(line, sizeof(line) - 2, format, args);
    if (length < 0) return;
    if (length > (int)sizeof(line) - 2) length = (int)sizeof(line) - 2;
    line[length++] = '\n';
    line[length] = '\0';
    if (log_fd >= 0) sceIoWrite(log_fd, line, length);
    if (on_screen) pspDebugScreenPrintf("%s", line);
}

/* File and screen: keep each under 68 columns, ~25 lines in all. */
static void say(const char *format, ...) {
    va_list args;
    va_start(args, format);
    emit(1, format, args);
    va_end(args);
}

/* File only. */
static void note(const char *format, ...) {
    va_list args;
    va_start(args, format);
    emit(0, format, args);
    va_end(args);
}

static void open_log(int argc, char **argv) {
    char *slash;
    char path[600];
    SceUID fd;
    exe_dir[0] = '\0';
    if (argc > 0 && argv[0] && strlen(argv[0]) + 24 < sizeof(exe_dir)) {
        strcpy(exe_dir, argv[0]);
        slash = strrchr(exe_dir, '/');
        if (slash) slash[1] = '\0';
        else exe_dir[0] = '\0';
    }
    snprintf(log_path, sizeof(log_path), "%sbench2_log.txt", exe_dir);
    log_fd = sceIoOpen(log_path, PSP_O_WRONLY | PSP_O_CREAT | PSP_O_TRUNC,
                       0777);
    snprintf(path, sizeof(path), "%sbench2_quick.txt", exe_dir);
    fd = sceIoOpen(path, PSP_O_RDONLY, 0777);
    if (fd >= 0) {
        quick = 1;
        sceIoClose(fd);
    }
}

/* ------------------------------------------------------------------ */
/* System                                                              */
/* ------------------------------------------------------------------ */

/* Mirror the app: no FPU exception enables, flush denormals. */
static void setup_fpu(void) {
    unsigned int fcr;
    __asm__ volatile ("cfc1 %0, $31" : "=r"(fcr));
    fcr &= ~0x00000f80u;
    fcr |= 0x01000000u;
    __asm__ volatile ("ctc1 %0, $31" : : "r"(fcr));
}

static int exit_callback(int arg1, int arg2, void *common) {
    (void)arg1; (void)arg2; (void)common;
    exiting = 1;
    return 0;
}
static int callback_thread(SceSize args, void *argp) {
    int callback;
    (void)args; (void)argp;
    callback = sceKernelCreateCallback("Exit Callback", exit_callback, NULL);
    if (callback >= 0) sceKernelRegisterExitCallback(callback);
    sceKernelSleepThreadCB();
    return 0;
}
static void setup_callbacks(void) {
    int thread = sceKernelCreateThread("update_thread", callback_thread,
                                       0x11, 0x2000, PSP_THREAD_ATTR_USER,
                                       NULL);
    if (thread >= 0) sceKernelStartThread(thread, 0, NULL);
}

static unsigned int now_us(void) { return sceKernelGetSystemTimeLow(); }

static uint32_t rng_state = 0x2545f491u;
static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static int mhz = 333;

/* ------------------------------------------------------------------ */
/* Part A                                                              */
/* ------------------------------------------------------------------ */

static uint8_t scratch[1024] __attribute__((aligned(64)));
static float table[256] __attribute__((aligned(64)));
static float vinit[128] __attribute__((aligned(64)));

static void reset_state(void) {
    float *f = (float *)(void *)scratch;
    int i;
    for (i = 0; i < 256; ++i) {
        f[i] = 1.0f;
        table[i] = 1.0f;
    }
    bench2_vfpu_load(vinit);
}

static unsigned int run_us(MicroFn fn, unsigned int n) {
    unsigned int started;
    reset_state();
    started = now_us();
    fn(n, scratch, table);
    return now_us() - started;
}

/* Iterations for about target_us of fn. */
static unsigned int calibrate(MicroFn fn, unsigned int target_us) {
    unsigned int n = 64, t = 0, floor_us = quick ? 500u : 8000u;
    for (;;) {
        t = run_us(fn, n);
        if (t >= floor_us || n >= (1u << 26)) break;
        n *= 4u;
    }
    if (!t) t = 1;
    n = (unsigned int)((uint64_t)n * target_us / t);
    return n < 16u ? 16u : n;
}

static double ovh_us_per_iter;           /* empty 3-instruction loop */

typedef struct {
    double cyc;      /* cycles per item, overhead cancelled (hi - lo) */
    double ovh_cyc;  /* cycles per item, hi minus measured loop overhead */
    double ps;       /* picoseconds per item */
} MicroResult;

static MicroResult micro_res[160];

static void measure_micro(const MicroInfo *m, MicroResult *r) {
    unsigned int target = quick ? 3000u : 150000u;
    unsigned int n = calibrate(m->hi, target);
    unsigned int t_lo = run_us(m->lo, n);
    unsigned int t_hi = run_us(m->hi, n);
    double items4 = 4.0 * m->items * (double)n;
    double items6 = 6.0 * m->items * (double)n;
    double diff_us = (double)t_hi - (double)t_lo;
    r->ps = diff_us * 1.0e6 / items4;
    r->cyc = diff_us * mhz / items4;
    r->ovh_cyc = ((double)t_hi - ovh_us_per_iter * n) * mhz / items6;
}

static double micro_cyc(const char *name) {
    int i;
    for (i = 0; i < bench2_micro_count; ++i)
        if (bench2_micro[i].lo && !strcmp(bench2_micro[i].name, name))
            return micro_res[i].cyc;
    return -1.0;
}

static void section_clock(void) {
    unsigned int target = quick ? 3000u : 150000u;
    unsigned int n = calibrate(bench2_loop_empty, target);
    unsigned int t = run_us(bench2_loop_empty, n);
    ovh_us_per_iter = (double)t / (double)n;
    say("[clock] %d/%d MHz; empty loop (addiu,bnez,nop) %.2f cyc/iter",
        scePowerGetCpuClockFrequency(), scePowerGetBusClockFrequency(),
        ovh_us_per_iter * mhz);
    note("[clock] %u iterations in %u us%s", n, t,
         quick ? " (QUICK MODE: emulator check only)" : "");
}

static int section_micro(void) {
    int i, done = 0;
    note("");
    note("== Part A: instruction microtimings ==");
    note("cyc = cycles per item from (6-unit loop - 2-unit loop), loop "
         "overhead cancelled exactly;");
    note("ovh = same from the 6-unit loop minus the measured empty-loop "
         "overhead; doc = expected from the VFPU docs.");
    note("Pairs: producer then dependent consumer, 4 independent pairs per "
         "unit; cost = producer latency + consumer issue.");
    for (i = 0; i < bench2_micro_count && !exiting; ++i) {
        const MicroInfo *m = &bench2_micro[i];
        MicroResult *r = &micro_res[i];
        if (!m->lo) {
            log_sync();
            note("-- %s --", m->name);
            continue;
        }
        measure_micro(m, r);
        note("[A] %-24s cyc %6.2f  ovh %6.2f  %8.0f ps  doc %s",
             m->name, r->cyc, r->ovh_cyc, r->ps, m->doc);
        scePowerTick(PSP_POWER_TICK_ALL);
        ++done;
    }
    log_sync();
    return done;
}

static void screen_micro(int done) {
    say("[A] %d timings (cycles). lv.q>vc2i %.1f lv.q>vtfm4 %.1f",
        done, micro_cyc("lv.q>vc2i.s pair"), micro_cyc("lv.q>vtfm4 pair"));
    say("[A] lat: vc2i %.1f vi2f %.1f vf2id %.1f vadd %.1f vtfm4 %.1f",
        micro_cyc("vc2i.s lat"), micro_cyc("vi2f.q lat"),
        micro_cyc("vf2id.q lat"), micro_cyc("vadd.q lat"),
        micro_cyc("vtfm4.q lat"));
    say("[A] thr: lv.q %.2f vtfm4 %.2f vdot %.2f vc2i %.2f vi2f %.2f",
        micro_cyc("lv.q thr"), micro_cyc("vtfm4.q thr"),
        micro_cyc("vdot.q thr"), micro_cyc("vc2i.s thr"),
        micro_cyc("vi2f.q 28 thr"));
    say("[A] pairs: c2i>i2f %.1f i2f>f2id %.1f f2id>i2f %.1f i2f>tfm %.1f",
        micro_cyc("vc2i>vi2f pair"), micro_cyc("vi2f>vf2id pair"),
        micro_cyc("vf2id>vi2f pair"), micro_cyc("vi2f>vtfm4(mtx) pair"));
    say("[A] wport 3/5 alt %.2f pair %.2f; 5/7 alt %.2f pair %.2f",
        micro_cyc("lat3/5 alternating"), micro_cyc("lat3/5 in pairs"),
        micro_cyc("lat5/7 alternating"), micro_cyc("lat5/7 in pairs"));
    say("[A] wS>rC %.1f samecol %.1f samemtx %.1f; tfm4+lv %.1f +addiu %.1f",
        micro_cyc("write S, read its C"), micro_cyc("indep S, same column"),
        micro_cyc("indep C, same matrix"), micro_cyc("vtfm4 + lv.q pair"),
        micro_cyc("vtfm4 + addiu pair"));
    say("[A] tile %.1f noload %.1f nomul %.1f pre %.1f conv %.1f",
        micro_cyc("tile: as shipped"), micro_cyc("tile: no lv.q"),
        micro_cyc("tile: no vtfm4"), micro_cyc("tile: raw lv.q early"),
        micro_cyc("tile: 16 conversions"));
    say("[A] scales/blk: fp16 %.1f lbu %.1f lw %.1f exp2 %.1f ilv %.1f",
        micro_cyc("fp16: 2 lv.s+vh2f"), micro_cyc("4 lbu+table lv.s"),
        micro_cyc("lw+shifts+table"), micro_cyc("vexp2 isolated"),
        micro_cyc("vexp2 4 interleaved"));
}

/* ------------------------------------------------------------------ */
/* Scale routes: correctness and vexp2 accuracy                        */
/* ------------------------------------------------------------------ */

static void section_routes(void) {
    static uint32_t word[4] __attribute__((aligned(16)));
    static float got[4] __attribute__((aligned(16)));
    static float sb[8] __attribute__((aligned(16)));
    const float step = 0.0731f, bias = -3.3f;
    double worst_exp2 = 0.0, worst_g = 0.0;
    int bad_lbu = 0, bad_lw = 0, i, k;
    for (i = 0; i < 256; ++i)
        table[i] = (float)pow(2.0, (double)(int8_t)(uint8_t)i / 16.0);
    for (k = 0; k < 4; ++k) sb[k] = bias;
    sb[4] = step;
    for (i = 0; i < 256; i += 4) {
        word[0] = (uint32_t)i | ((uint32_t)(i + 1) << 8) |
                  ((uint32_t)(i + 2) << 16) | ((uint32_t)(i + 3) << 24);
        bench2_route_lbu(word, table, got);
        for (k = 0; k < 4; ++k) if (got[k] != table[i + k]) ++bad_lbu;
        bench2_route_lw(word, table, got);
        for (k = 0; k < 4; ++k) if (got[k] != table[i + k]) ++bad_lw;
        bench2_route_exp2(word, got);
        for (k = 0; k < 4; ++k) {
            double want = pow(2.0, (double)(int8_t)(uint8_t)(i + k) / 16.0);
            double err = fabs((double)got[k] - want) / want;
            if (err > worst_exp2) worst_exp2 = err;
        }
        bench2_route_exp2g(word, sb, got);
        for (k = 0; k < 4; ++k) {
            double want = pow(2.0, (double)(int8_t)(uint8_t)(i + k) *
                                   (double)step + (double)bias);
            double err = fabs((double)got[k] - want) / want;
            if (err > worst_g) worst_g = err;
        }
    }
    say("[scale] lbu %s, lw %s; vexp2 err %.1e, any step %.1e",
        bad_lbu ? "FAIL" : "ok", bad_lw ? "FAIL" : "ok",
        worst_exp2, worst_g);
    note("[scale] 256 codes: lbu mismatches %d, lw mismatches %d; vexp2 "
         "2^(c/16) worst rel err %.3e; general (step %.4f, offset %.2f) "
         "%.3e (docs: < 7.2e-7)", bad_lbu, bad_lw, worst_exp2,
         (double)step, (double)bias, worst_g);
    log_sync();
}

/* ------------------------------------------------------------------ */
/* Part B                                                              */
/* ------------------------------------------------------------------ */

static float x_vec[COLS] __attribute__((aligned(16)));
static float x_tiles[COLS] __attribute__((aligned(16)));
static float sums_low[BLOCKS] __attribute__((aligned(16)));

static float half_to_float(uint16_t h) {
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t bits;
    float value;
    if (exponent == 0) return 0.0f;
    bits = (((uint32_t)h & 0x8000u) << 16) | ((exponent + 112u) << 23) |
           (((uint32_t)h & 0x3ffu) << 13);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Random tile groups: normal fp16 scales between 2^-10 and 2^-3 (no
 * denormals, so the hardware/emulator vh2f difference cannot matter),
 * random tile bytes. */
static void fill_groups(uint8_t *groups, size_t count) {
    size_t g;
    int i;
    for (g = 0; g < count; ++g) {
        uint8_t *grp = groups + g * GROUP_BYTES;
        for (i = 0; i < BLOCKS * 4; ++i) {
            uint32_t r = rng();
            uint16_t h = (uint16_t)(((r >> 16) & 0x8000u) |
                                    ((5u + ((r >> 28) & 7u)) << 10) |
                                    ((r >> 12) & 0x3ffu));
            grp[i * 2] = (uint8_t)h;
            grp[i * 2 + 1] = (uint8_t)(h >> 8);
        }
        for (i = BLOCKS * 8; i < GROUP_BYTES; i += 4) {
            uint32_t r = rng();
            memcpy(grp + i, &r, 4);
        }
    }
}

static void prepare_x(void) {
    int block, t, k, j;
    for (j = 0; j < COLS; ++j)
        x_vec[j] = (float)((int)((rng() >> 8) % 4001u) - 2000) / 1000.0f;
    for (block = 0; block < BLOCKS; ++block) {
        const float *v = x_vec + block * 32;
        float low = 0.0f;
        for (j = 0; j < 16; ++j) low += v[j];
        sums_low[block] = low;
        for (t = 0; t < 4; ++t) {
            float *q = x_tiles + block * 32 + t * 8;
            for (k = 0; k < 4; ++k) {
                float lo16 = 16.0f * v[t * 4 + k];
                q[k] = lo16;
                q[4 + k] = v[16 + t * 4 + k] - lo16;
            }
        }
    }
}

/* Decodes the tile format independently (src/falcon_q4_tile_psp.h): tile
 * byte ^ 0x80 is the FHQ4 byte, low nibble = column 32b + 4t + k, high
 * nibble = column 32b + 16 + 4t + k, both offset by 8. */
static double reference_row(const uint8_t *group, int r) {
    const uint8_t *tiles = group + BLOCKS * 8;
    double sum = 0.0;
    int b, t, k;
    for (b = 0; b < BLOCKS; ++b) {
        const uint8_t *s = group + b * 8 + r * 2;
        double d = half_to_float((uint16_t)(s[0] | (s[1] << 8)));
        double inner = 0.0;
        for (t = 0; t < 4; ++t)
            for (k = 0; k < 4; ++k) {
                int byte = tiles[b * 64 + t * 16 + r * 4 + k] ^ 0x80;
                int col = b * 32 + t * 4 + k;
                inner += (double)((byte & 15) - 8) * x_vec[col];
                inner += (double)((byte >> 4) - 8) * x_vec[col + 16];
            }
        sum += d * inner;
    }
    return sum;
}

typedef struct {
    unsigned int ram_ps, l1_ps;   /* per weight */
    double worst;                 /* worst error / rms */
    int pass;
} KernelResult;

static KernelResult kres[16];

static void time_kernel(TileKernel kernel, const uint8_t *groups,
                        size_t count, float *out, KernelResult *res) {
    unsigned int started, elapsed, best = 0xffffffffu;
    int passes = quick ? 1 : 3, pass, round;
    size_t g, rep, reps = count / L1_GROUPS;
    if (quick && reps > 64) reps = 64;
    started = now_us();
    for (pass = 0; pass < passes; ++pass)
        for (g = 0; g < count; ++g)
            kernel(groups + g * GROUP_BYTES, x_tiles, sums_low, BLOCKS,
                   out + g * 4);
    elapsed = now_us() - started;
    res->ram_ps = (unsigned)((uint64_t)elapsed * 1000000u /
                             ((uint64_t)count * 4u * COLS * passes));
    /* Two L1 rounds, the faster one counts. */
    for (round = 0; round < 2; ++round) {
        started = now_us();
        for (pass = 0; pass < passes; ++pass)
            for (rep = 0; rep < reps; ++rep)
                for (g = 0; g < L1_GROUPS; ++g)
                    kernel(groups + g * GROUP_BYTES, x_tiles, sums_low,
                           BLOCKS, out + g * 4);
        elapsed = now_us() - started;
        if (elapsed < best) best = elapsed;
    }
    res->l1_ps = (unsigned)((uint64_t)best * 1000000u /
                            ((uint64_t)reps * L1_GROUPS * 4u * COLS *
                             passes));
}

static double cyc_per_w(unsigned int ps) {
    return (double)ps * mhz / 1.0e6;
}

static int section_kernels(void) {
    static float out4[4] __attribute__((aligned(16)));
    static double ref[VERIFY_GROUPS * 4];
    static int order[16];
    size_t bytes = 8u * 1024u * 1024u, count, g;
    uint8_t *groups = NULL;
    float *out = NULL;
    double rms = 0.0;
    int all_ok = 1, k, i, j;

    for (;;) {
        groups = (uint8_t *)memalign(64, bytes);
        if (groups || bytes <= 1024u * 1024u) break;
        bytes /= 2u;
    }
    count = groups ? bytes / GROUP_BYTES : 0;
    out = count ? (float *)memalign(64, count * 4u * sizeof(float)) : NULL;
    if (!groups || !out || count < VERIFY_GROUPS) {
        say("[fail] not enough memory for the kernel buffers");
        free(groups);
        free(out);
        return 0;
    }
    fill_groups(groups, count);
    prepare_x();
    note("");
    note("== Part B: kernel variants (%u KiB of tile groups, %u groups of "
         "4 rows x %d columns) ==", (unsigned)(bytes / 1024u),
         (unsigned)count, COLS);
    for (g = 0; g < VERIFY_GROUPS; ++g)
        for (i = 0; i < 4; ++i) {
            double v = reference_row(groups + g * GROUP_BYTES, i);
            ref[g * 4 + i] = v;
            rms += v * v;
        }
    rms = sqrt(rms / (VERIFY_GROUPS * 4.0));
    if (rms <= 0.0) rms = 1.0;
    note("[B] reference rms %.4f over %d rows", rms, VERIFY_GROUPS * 4);
    scePowerTick(PSP_POWER_TICK_ALL);

    for (k = 0; k < bench2_kernel_count && !exiting; ++k) {
        const KernelInfo *ki = &bench2_kernels[k];
        KernelResult *r = &kres[k];
        double worst = 0.0;
        for (g = 0; g < VERIFY_GROUPS; ++g) {
            ki->fn(groups + g * GROUP_BYTES, x_tiles, sums_low, BLOCKS,
                   out4);
            for (i = 0; i < 4; ++i) {
                double e = fabs((double)out4[i] - ref[g * 4 + i]);
                if (!(e <= worst)) worst = e;     /* NaN counts as worst */
            }
        }
        r->worst = worst / rms;
        r->pass = r->worst < 1e-4;
        if (!r->pass) all_ok = 0;
        note("[B] %-10s verify %s (worst err/rms %.2e)  %d instr/block: %s",
             ki->name, r->pass ? "PASS" : "FAIL", r->worst, ki->instr,
             ki->why);
        log_sync();
        time_kernel(ki->fn, groups, count, out, r);
        note("[B] %-10s cached %.3f cyc/w (%u ps), RAM %.3f cyc/w (%u ps), "
             "cached %.1f cyc/block", ki->name, cyc_per_w(r->l1_ps),
             r->l1_ps, cyc_per_w(r->ram_ps), r->ram_ps,
             cyc_per_w(r->l1_ps) * 128.0);
        log_sync();
        scePowerTick(PSP_POWER_TICK_ALL);
    }
    say("[verify] %d kernels x %d rows vs scalar: %s", bench2_kernel_count,
        VERIFY_GROUPS * 4, all_ok ? "PASS" : "FAIL (see log)");

    /* Ranking by cached speed. */
    for (k = 0; k < bench2_kernel_count; ++k) order[k] = k;
    for (i = 1; i < bench2_kernel_count; ++i)
        for (j = i; j > 0 && kres[order[j]].l1_ps < kres[order[j - 1]].l1_ps;
             --j) {
            int t = order[j];
            order[j] = order[j - 1];
            order[j - 1] = t;
        }
    note("");
    note("== Ranking by cached cycles per weight (RAM = streaming from "
         "8 MiB) ==");
    for (i = 0; i < bench2_kernel_count; ++i) {
        const KernelResult *r = &kres[order[i]];
        say("#%d %-10s cached %.3f RAM %.3f cyc/w %s", i + 1,
            bench2_kernels[order[i]].name, cyc_per_w(r->l1_ps),
            cyc_per_w(r->ram_ps), r->pass ? "" : "FAIL");
    }
    log_sync();
    free(groups);
    free(out);
    return all_ok;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    SceCtrlData pad;
    int ok, done, i;
    unsigned int started;

    setup_fpu();
    setup_callbacks();
    pspDebugScreenInit();
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_DIGITAL);
    scePowerSetClockFrequency(333, 333, 166);
    open_log(argc, argv);
    started = now_us();
    for (i = 0; i < 128; ++i) vinit[i] = (i >= 64 && i < 112) ? 1.0f : 0.0f;
    say("LLMPSP Bench 2 v" BENCH_VERSION " - about 35 s, no input needed%s",
        quick ? " (QUICK)" : "");
    mhz = scePowerGetCpuClockFrequency();
    if (mhz <= 0) mhz = 333;

    section_clock();
    log_sync();
    done = section_micro();
    screen_micro(done);
    section_routes();
    ok = exiting ? 0 : section_kernels();
    note("");
    note("[done] %u ms total", (now_us() - started) / 1000u);
    say("[done] %s - details in bench2_log.txt",
        exiting ? "interrupted" : ok ? "all checks passed" : "A CHECK FAILED");
    say("Press X or HOME to exit.");
    if (log_fd >= 0) sceIoClose(log_fd);
    log_fd = -1;

    /* Leaves by itself after a minute so an unattended run still ends. */
    for (i = 0; i < 3600 && !exiting; ++i) {
        sceCtrlPeekBufferPositive(&pad, 1);
        if (pad.Buttons & PSP_CTRL_CROSS) break;
        sceDisplayWaitVblankStart();
    }
    sceKernelExitGame();
    return 0;
}
