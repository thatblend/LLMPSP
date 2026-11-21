/* Standalone on-device benchmark (LLMPSPBench). It answers the questions
 * that decide the next speed pass and that PPSSPP cannot answer, because
 * the emulator does not model instruction or memory timing:
 *
 *   1. How fast can the PSP read RAM that is not in the data cache? Every
 *      token scans ~49 MiB of weights, so this bounds any kernel.
 *   2. How fast is the version-1 row Q4 kernel (tools/q4_row_psp.c) on
 *      its own, and how fast is the four-row vtfm4 tile kernel the app
 *      now uses (src/falcon_q4_psp.c; vdot variant in tools/q4_tile_psp.c)?
 *   3. Does reading the memory stick with raw sceIoRead beat the stdio
 *      path the app streamed through up to v1.11?
 *   4. How much heap is there, and how much is left outside it?
 *   5. Do vf2id / vh2f / vf2h behave on hardware as documented?
 *
 * Everything is written to bench_log.txt beside the EBOOT; the screen
 * shows a short version. Kernels are also checked against a scalar
 * reference, which is the part that is meaningful in PPSSPP.
 *
 * Build: see tools/Makefile.bench. No input is needed; press X or HOME
 * when it says it is done. */
#include <pspctrl.h>
#include <pspdebug.h>
#include <pspdisplay.h>
#include <pspiofilemgr.h>
#include <pspkernel.h>
#include <psppower.h>
#include <pspsysmem.h>

#include <malloc.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

PSP_MODULE_INFO("LLMPSPBench", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
/* Same heap policy as the app, so the memory figures carry over. */
PSP_HEAP_SIZE_KB(-2048);

#define BENCH_VERSION "1"

/* The matrices timed here are 512 columns wide, like most of the model. */
#define COLS 512
#define BLOCKS (COLS / 32)
#define ROW_BYTES (BLOCKS * 18)          /* FHQ4 row */
#define GROUP_BYTES (ROW_BYTES * 4)      /* tile group, same density */
#define L1_ROWS 16                       /* 4.5 KiB: stays in the D-cache */
#define L1_WINDOW 8192

/* Weights touched per token, for the projections at the end. */
#define LAYER_WEIGHTS 74219520.0
#define HEAD_WEIGHTS 16777216.0

extern float falcon_dot_q4_vfpu(const uint8_t *row, const float *x,
                                int blocks, const float *block_sums);
extern void falcon_q4_tile4_vtfm(const uint8_t *group, const float *xt,
                                 const float *block_sums, int blocks,
                                 float *out);
extern void falcon_q4_tile4_vdot(const uint8_t *group, const float *xt,
                                 const float *block_sums, int blocks,
                                 float *out);

typedef void (*TileKernel)(const uint8_t *, const float *, const float *,
                           int, float *);

static volatile int exiting;

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

/* File and screen. Keep these to about 30 lines: the debug screen wraps. */
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
    exe_dir[0] = '\0';
    if (argc > 0 && argv[0] && strlen(argv[0]) + 24 < sizeof(exe_dir)) {
        strcpy(exe_dir, argv[0]);
        slash = strrchr(exe_dir, '/');
        if (slash) slash[1] = '\0';
        else exe_dir[0] = '\0';
    }
    snprintf(log_path, sizeof(log_path), "%sbench_log.txt", exe_dir);
    log_fd = sceIoOpen(log_path, PSP_O_WRONLY | PSP_O_CREAT | PSP_O_TRUNC,
                       0777);
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

/* ------------------------------------------------------------------ */
/* Small VFPU probes                                                   */
/* ------------------------------------------------------------------ */

static void probe_floor_trick(const uint32_t *word, float *g, float *h) {
    __asm__ volatile (
        "lv.s   S000, 0(%[w])\n"
        "vc2i.s C010, S000\n"
        "vi2f.q C010, C010, 28\n"
        "vf2id.q C020, C010, 0\n"
        "vi2f.q C020, C020, 0\n"
        "sv.q   C010, 0(%[g])\n"
        "sv.q   C020, 0(%[h])\n"
        : : [w] "r"(word), [g] "r"(g), [h] "r"(h) : "memory");
}

static void probe_vh2f(const uint32_t *halves, float *out) {
    __asm__ volatile (
        "lv.s   S000, 0(%[in])\n"
        "lv.s   S001, 4(%[in])\n"
        "vh2f.p C010, C000\n"
        "sv.q   C010, 0(%[out])\n"
        : : [in] "r"(halves), [out] "r"(out) : "memory");
}

static void probe_vf2h(const float *in, uint32_t *out) {
    __asm__ volatile (
        "lv.q   C000, 0(%[in])\n"
        "vzero.q C010\n"
        "vf2h.q C010, C000\n"
        "sv.q   C010, 0(%[out])\n"
        : : [in] "r"(in), [out] "r"(out) : "memory");
}

/* Three instructions per iteration, nothing but integer ops. */
static void spin(unsigned int iterations) {
    unsigned int n = iterations;
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "1:\n"
        "addiu  %[n], %[n], -1\n"
        "bnez   %[n], 1b\n"
        "nop\n"
        ".set pop\n"
        : [n] "+r"(n));
}

/* Loads every byte of a buffer through the VFPU, four quads per cache
 * line, without doing arithmetic on it. */
static void scan_lvq(const uint8_t *data, unsigned int lines) {
    const uint8_t *p = data;
    unsigned int n = lines;
    __asm__ volatile (
        ".set push\n"
        ".set noreorder\n"
        "1:\n"
        "lv.q   C000, 0(%[p])\n"
        "lv.q   C010, 16(%[p])\n"
        "lv.q   C020, 32(%[p])\n"
        "lv.q   C030, 48(%[p])\n"
        "addiu  %[n], %[n], -1\n"
        "bnez   %[n], 1b\n"
        "addiu  %[p], %[p], 64\n"
        ".set pop\n"
        : [p] "+r"(p), [n] "+r"(n) : : "memory");
}

/* One integer load per 64-byte cache line. */
static uint32_t scan_touch(const uint8_t *data, unsigned int lines) {
    const uint32_t *p = (const uint32_t *)(const void *)data;
    uint32_t sum = 0;
    unsigned int i;
    for (i = 0; i < lines; ++i) sum += p[(size_t)i * 16u];
    return sum;
}

/* ------------------------------------------------------------------ */
/* Test data                                                           */
/* ------------------------------------------------------------------ */

static float half_to_float(uint16_t h) {
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t bits;
    float value;
    if (exponent == 0) return 0.0f;      /* the kernels flush denormals */
    bits = (((uint32_t)h & 0x8000u) << 16) | ((exponent + 112u) << 23) |
           (((uint32_t)h & 0x3ffu) << 13);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* FHQ4 rows with random nibbles and scales between 2^-10 and 2^-3. */
static void fill_rows(uint8_t *rows, size_t count) {
    size_t row, i;
    int block;
    for (row = 0; row < count; ++row) {
        for (block = 0; block < BLOCKS; ++block) {
            uint8_t *b = rows + row * ROW_BYTES + (size_t)block * 18u;
            uint32_t r = rng();
            uint16_t h = (uint16_t)(((r >> 16) & 0x8000u) |
                                    ((5u + ((r >> 28) & 7u)) << 10) |
                                    ((r >> 12) & 0x3ffu));
            b[0] = (uint8_t)h;
            b[1] = (uint8_t)(h >> 8);
            for (i = 0; i < 16; ++i) b[2 + i] = (uint8_t)(rng() >> 24);
        }
    }
}

/* The lossless FHQ4 -> tile transcode a converter would do (see
 * tools/q4_tile_psp.c for the layout). */
static void transcode_group(uint8_t *group, const uint8_t *rows) {
    uint8_t *scales = group;
    uint8_t *tiles = group + BLOCKS * 8;
    int block, r, t, k;
    for (block = 0; block < BLOCKS; ++block) {
        for (r = 0; r < 4; ++r) {
            const uint8_t *src = rows + (size_t)r * ROW_BYTES +
                                 (size_t)block * 18u;
            scales[block * 8 + r * 2] = src[0];
            scales[block * 8 + r * 2 + 1] = src[1];
            for (t = 0; t < 4; ++t)
                for (k = 0; k < 4; ++k)
                    tiles[block * 64 + t * 16 + r * 4 + k] =
                        (uint8_t)(src[2 + t * 4 + k] ^ 0x80u);
        }
    }
}

/* Activation tables for both kernels. */
static float x_vec[COLS] __attribute__((aligned(16)));
static float x_tiles[COLS] __attribute__((aligned(16)));
static float sums_full[BLOCKS] __attribute__((aligned(16)));
static float sums_low[BLOCKS] __attribute__((aligned(16)));

static void prepare_x(void) {
    int block, t, k, j;
    for (j = 0; j < COLS; ++j)
        x_vec[j] = (float)((int)((rng() >> 8) % 4001u) - 2000) / 1000.0f;
    for (block = 0; block < BLOCKS; ++block) {
        const float *v = x_vec + block * 32;
        float full = 0.0f, low = 0.0f;
        for (j = 0; j < 32; ++j) full += v[j];
        for (j = 0; j < 16; ++j) low += v[j];
        sums_full[block] = full;
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

static double reference_row(const uint8_t *row) {
    double sum = 0.0;
    int block, j;
    for (block = 0; block < BLOCKS; ++block) {
        const uint8_t *b = row + block * 18;
        const float *v = x_vec + block * 32;
        double d = half_to_float((uint16_t)(b[0] | (b[1] << 8)));
        double inner = 0.0;
        for (j = 0; j < 16; ++j) {
            inner += (double)((b[2 + j] & 15) - 8) * v[j];
            inner += (double)((b[2 + j] >> 4) - 8) * v[j + 16];
        }
        sum += d * inner;
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/* Sections                                                            */
/* ------------------------------------------------------------------ */

/* The heap is claimed before main() runs, so the partition's free figure
 * is what PSP_HEAP_SIZE_KB(-2048) leaves outside it once the stacks are
 * taken: RAM the app could still hand to the weight cache. The block
 * count is the app's own cache ceiling before its runtime state. The
 * volatile partition is not probed: the app's trace already reports it. */
static void section_memory(unsigned int free_boot, unsigned int max_boot,
                           unsigned int free_heap, unsigned int max_heap) {
    static void *blocks[256];
    int count = 0, i;
    while (count < 256) {
        void *p = malloc(256u * 1024u);
        if (!p) break;
        blocks[count++] = p;
    }
    for (i = 0; i < count; ++i) free(blocks[i]);
    say("[mem] outside the heap: %u KiB free, largest block %u KiB",
        free_boot / 1024u, max_boot / 1024u);
    say("[mem] heap holds %d x 256 KiB = %d.%02d MiB",
        count, count / 4, (count % 4) * 25);
    note("[mem] after the first malloc: %u KiB free, largest %u KiB",
         free_heap / 1024u, max_heap / 1024u);
}

static void section_probes(void) {
    static uint32_t word[4] __attribute__((aligned(16)));
    static float g[4] __attribute__((aligned(16)));
    static float h[4] __attribute__((aligned(16)));
    static float fin[4] __attribute__((aligned(16)));
    static uint32_t bits[4] __attribute__((aligned(16)));
    static const float want_g[4] = { 0.9375f, -8.0f, -0.0625f, 7.9375f };
    static const float want_h[4] = { 0.0f, -8.0f, -1.0f, 7.0f };
    uint32_t raw[4];
    int ok = 1, i;
    /* Bytes 0x0F, 0x80, 0xFF, 0x7F: the nibble-split corner cases. */
    word[0] = 0x7FFF800Fu;
    probe_floor_trick(word, g, h);
    for (i = 0; i < 4; ++i)
        if (g[i] != want_g[i] || h[i] != want_h[i]) ok = 0;
    say("[probe] nibble split via vf2id: %s", ok ? "PASS" : "FAIL");
    note("[probe] G %.4f %.4f %.4f %.4f  H %.1f %.1f %.1f %.1f",
         (double)g[0], (double)g[1], (double)g[2], (double)g[3],
         (double)h[0], (double)h[1], (double)h[2], (double)h[3]);
    /* Halves: smallest denormal, largest denormal, smallest normal, -1. */
    word[0] = 0x03FF0001u;
    word[1] = 0xBC000400u;
    probe_vh2f(word, g);
    memcpy(raw, g, sizeof(raw));
    note("[probe] vh2f 0001 03FF 0400 BC00 -> %08x %08x %08x %08x "
         "(docs: 0 0 38800000 bf800000)",
         (unsigned)raw[0], (unsigned)raw[1], (unsigned)raw[2],
         (unsigned)raw[3]);
    /* 1 + 0.25, 0.75 and 0.5 of a half-float step, then a value below
     * the smallest normal half: shows rounding and underflow. */
    fin[0] = 1.0f + 0.25f / 1024.0f;
    fin[1] = 1.0f + 0.75f / 1024.0f;
    fin[2] = 1.0f + 0.5f / 1024.0f;
    fin[3] = 3.0e-5f;
    probe_vf2h(fin, bits);
    note("[probe] vf2h 1+.25u 1+.75u 1+.5u 3e-5 -> %04x %04x %04x %04x "
         "(nearest: 3c00 3c01 3c00, truncation: 3c00 3c00 3c00)",
         (unsigned)(bits[0] & 0xffffu), (unsigned)(bits[0] >> 16),
         (unsigned)(bits[1] & 0xffffu), (unsigned)(bits[1] >> 16));
}

static void section_clock(void) {
    const unsigned int iterations = 20000000u;
    unsigned int started, elapsed;
    started = now_us();
    spin(iterations);
    elapsed = now_us() - started;
    if (!elapsed) elapsed = 1;
    say("[clock] %d/%d MHz set; 3-instr loop %u ps -> %u MHz-equivalent",
        scePowerGetCpuClockFrequency(), scePowerGetBusClockFrequency(),
        (unsigned)((uint64_t)elapsed * 1000000u / iterations),
        (unsigned)((uint64_t)iterations * 3u / elapsed));
}

/* Returns the cost of one 64-byte line read from RAM, in picoseconds. */
static unsigned int section_ram(const uint8_t *buffer, size_t bytes) {
    unsigned int lines = (unsigned int)(bytes / 64u);
    unsigned int window_lines = L1_WINDOW / 64u;
    unsigned int reps = lines / window_lines, i;
    unsigned int started, ram_lvq, l1_lvq, ram_touch, l1_touch;
    uint32_t sink = 0;
    started = now_us();
    scan_lvq(buffer, lines);
    scan_lvq(buffer, lines);
    ram_lvq = now_us() - started;
    started = now_us();
    for (i = 0; i < reps * 2u; ++i) scan_lvq(buffer, window_lines);
    l1_lvq = now_us() - started;
    started = now_us();
    sink += scan_touch(buffer, lines);
    sink += scan_touch(buffer, lines);
    ram_touch = now_us() - started;
    started = now_us();
    for (i = 0; i < reps * 2u; ++i) sink += scan_touch(buffer, window_lines);
    l1_touch = now_us() - started;
    {
        /* picoseconds per line; two passes each */
        unsigned int n = lines * 2u, m = reps * window_lines * 2u;
        unsigned int ram_ps = (unsigned)((uint64_t)ram_lvq * 1000000u / n);
        unsigned int l1_ps = (unsigned)((uint64_t)l1_lvq * 1000000u / m);
        unsigned int ramt_ps = (unsigned)((uint64_t)ram_touch * 1000000u / n);
        unsigned int l1t_ps = (unsigned)((uint64_t)l1_touch * 1000000u / m);
        unsigned int mib_s = ram_lvq ?
            (unsigned)((uint64_t)bytes * 2u * 1000000u / ram_lvq / 1048576u) :
            0;
        say("[ram] lv.q scan: %u ps/line from RAM, %u cached -> %u MiB/s",
            ram_ps, l1_ps, mib_s);
        say("[ram] 1 load/line: %u ps/line from RAM, %u cached",
            ramt_ps, l1t_ps);
        note("[ram] raw us: lvq %u / %u, touch %u / %u, sink %08x",
             ram_lvq, l1_lvq, ram_touch, l1_touch, (unsigned)sink);
        return ram_ps > l1_ps ? ram_ps - l1_ps : 0;
    }
}

typedef struct {
    unsigned int ram_ps;     /* picoseconds per weight, streaming from RAM */
    unsigned int l1_ps;      /* same, data resident in the D-cache */
    double checksum;
} KernelTiming;

#define PASSES 3

static KernelTiming time_current(const uint8_t *rows, size_t count,
                                 float *out) {
    KernelTiming result;
    unsigned int started, elapsed;
    size_t row, rep, reps = count / L1_ROWS;
    double sum = 0.0;
    int pass;
    started = now_us();
    for (pass = 0; pass < PASSES; ++pass)
        for (row = 0; row < count; ++row)
            out[row] = falcon_dot_q4_vfpu(rows + row * ROW_BYTES, x_vec,
                                          BLOCKS, sums_full);
    elapsed = now_us() - started;
    result.ram_ps = (unsigned)((uint64_t)elapsed * 1000000u /
                               ((uint64_t)count * COLS * PASSES));
    for (row = 0; row < count; ++row) sum += out[row];
    result.checksum = sum;
    started = now_us();
    for (pass = 0; pass < PASSES; ++pass)
        for (rep = 0; rep < reps; ++rep)
            for (row = 0; row < L1_ROWS; ++row)
                out[row] = falcon_dot_q4_vfpu(rows + row * ROW_BYTES, x_vec,
                                              BLOCKS, sums_full);
    elapsed = now_us() - started;
    result.l1_ps = (unsigned)((uint64_t)elapsed * 1000000u /
                              ((uint64_t)reps * L1_ROWS * COLS * PASSES));
    return result;
}

static KernelTiming time_tile(TileKernel kernel, const uint8_t *groups,
                              size_t count, float *out) {
    KernelTiming result;
    unsigned int started, elapsed;
    size_t group, rep, l1_groups = L1_ROWS / 4, reps = count / l1_groups;
    double sum = 0.0;
    int pass;
    started = now_us();
    for (pass = 0; pass < PASSES; ++pass)
        for (group = 0; group < count; ++group)
            kernel(groups + group * GROUP_BYTES, x_tiles, sums_low, BLOCKS,
                   out + group * 4);
    elapsed = now_us() - started;
    result.ram_ps = (unsigned)((uint64_t)elapsed * 1000000u /
                               ((uint64_t)count * 4u * COLS * PASSES));
    for (group = 0; group < count * 4; ++group) sum += out[group];
    result.checksum = sum;
    started = now_us();
    for (pass = 0; pass < PASSES; ++pass)
        for (rep = 0; rep < reps; ++rep)
            for (group = 0; group < l1_groups; ++group)
                kernel(groups + group * GROUP_BYTES, x_tiles, sums_low,
                       BLOCKS, out + group * 4);
    elapsed = now_us() - started;
    result.l1_ps = (unsigned)((uint64_t)elapsed * 1000000u /
                              ((uint64_t)reps * l1_groups * 4u * COLS *
                               PASSES));
    return result;
}

/* Every kernel against the scalar reference on the same logical rows.
 * Errors are reported relative to the RMS of the reference outputs. */
static int section_verify(const uint8_t *rows, const uint8_t *groups,
                          size_t count) {
    static float out4[4] __attribute__((aligned(16)));
    size_t check = count < 1024 ? count : 1024, row;
    double rms = 0.0, worst_cur = 0.0, worst_vtfm = 0.0, worst_vdot = 0.0;
    int ok;
    for (row = 0; row < check; ++row) {
        double ref = reference_row(rows + row * ROW_BYTES);
        rms += ref * ref;
    }
    rms = sqrt(rms / (double)check);
    if (rms <= 0.0) rms = 1.0;
    for (row = 0; row < check; row += 4) {
        int r;
        double ref[4];
        for (r = 0; r < 4; ++r) {
            double got;
            ref[r] = reference_row(rows + (row + r) * ROW_BYTES);
            got = falcon_dot_q4_vfpu(rows + (row + r) * ROW_BYTES, x_vec,
                                     BLOCKS, sums_full);
            if (fabs(got - ref[r]) > worst_cur)
                worst_cur = fabs(got - ref[r]);
        }
        falcon_q4_tile4_vtfm(groups + (row / 4) * GROUP_BYTES, x_tiles,
                             sums_low, BLOCKS, out4);
        for (r = 0; r < 4; ++r)
            if (fabs((double)out4[r] - ref[r]) > worst_vtfm)
                worst_vtfm = fabs((double)out4[r] - ref[r]);
        falcon_q4_tile4_vdot(groups + (row / 4) * GROUP_BYTES, x_tiles,
                             sums_low, BLOCKS, out4);
        for (r = 0; r < 4; ++r)
            if (fabs((double)out4[r] - ref[r]) > worst_vdot)
                worst_vdot = fabs((double)out4[r] - ref[r]);
    }
    worst_cur /= rms;
    worst_vtfm /= rms;
    worst_vdot /= rms;
    ok = worst_cur < 1e-4 && worst_vtfm < 1e-4 && worst_vdot < 1e-4;
    say("[verify] %u rows vs scalar: %s", (unsigned)check,
        ok ? "PASS" : "FAIL");
    note("[verify] worst error / rms: shipped %.2e, vtfm4 %.2e, vdot %.2e "
         "(rms %.4f)", worst_cur, worst_vtfm, worst_vdot, rms);
    return ok;
}

static void report_kernel(const char *name, const KernelTiming *timing,
                          int mhz) {
    /* cycles per weight in hundredths, at the reported CPU clock */
    unsigned int ram_c = (unsigned)((uint64_t)timing->ram_ps * (unsigned)mhz /
                                    10000u);
    unsigned int l1_c = (unsigned)((uint64_t)timing->l1_ps * (unsigned)mhz /
                                   10000u);
    say("[kernel] %s: RAM %u.%02u cyc/w (%u ps), cached %u.%02u (%u)",
        name, ram_c / 100u, ram_c % 100u, timing->ram_ps,
        l1_c / 100u, l1_c % 100u, timing->l1_ps);
    note("[kernel] %s checksum %.4f", name, timing->checksum);
}

/* Reads 4 MiB with each method from its own part of the model file. */
static void section_card(void) {
    static const char *names[3] = {
        "model.fhq4",
        "ms0:/PSP/GAME/LLMPSP/model.fhq4",
        "ef0:/PSP/GAME/LLMPSP/model.fhq4",
    };
    const unsigned int region = 4u * 1024u * 1024u;
    char path[600];
    uint8_t *buffer;
    SceUID fd = -1;
    int i, size;
    for (i = 0; i < 3 && fd < 0; ++i) {
        snprintf(path, sizeof(path), "%s%s", i == 0 ? exe_dir : "", names[i]);
        fd = sceIoOpen(path, PSP_O_RDONLY, 0777);
    }
    if (fd < 0) {
        say("[card] model.fhq4 not found: card test skipped");
        return;
    }
    size = sceIoLseek32(fd, 0, PSP_SEEK_END);
    buffer = (uint8_t *)memalign(64, 1024u * 1024u + 64u);
    if (!buffer || size < 40 * 1024 * 1024) {
        say("[card] no buffer or file too small: card test skipped");
        sceIoClose(fd);
        free(buffer);
        return;
    }
    note("[card] reading %s (%d bytes)", path, size);
    sceIoLseek32(fd, 0, PSP_SEEK_SET);
    sceIoRead(fd, buffer, 65536);            /* warm-up */
    {
        /* The v1.11 app's path: stdio with a 64 KiB buffer, 32 KiB
         * requests (the app now streams with raw 128 KiB sceIoReads). */
        FILE *file = fopen(path, "rb");
        char *vbuf = (char *)malloc(65536);
        if (file && vbuf) {
            unsigned int started, elapsed, done = 0;
            setvbuf(file, vbuf, _IOFBF, 65536);
            fseek(file, 8L * 1024L * 1024L, SEEK_SET);
            started = now_us();
            while (done < region &&
                   fread(buffer, 1, 32768, file) == 32768)
                done += 32768;
            elapsed = now_us() - started;
            if (!elapsed) elapsed = 1;
            say("[card] stdio fread 32K (v1.11 app path): %u KiB/s",
                (unsigned)((uint64_t)done * 1000000u / elapsed / 1024u));
        }
        if (file) fclose(file);
        free(vbuf);
    }
    {
        static const struct {
            const char *name;
            unsigned int chunk, file_skew, buffer_skew;
        } tests[5] = {
            { "sceIoRead 32K", 32768u, 0u, 0u },
            { "sceIoRead 256K", 262144u, 0u, 0u },
            { "sceIoRead 1M", 1048576u, 0u, 0u },
            { "sceIoRead 256K, file offset +100", 262144u, 100u, 0u },
            { "sceIoRead 256K, buffer +4", 262144u, 0u, 4u },
        };
        for (i = 0; i < 5; ++i) {
            unsigned int offset = (13u + 5u * (unsigned)i) * 1024u * 1024u +
                                  tests[i].file_skew;
            unsigned int started, elapsed, done = 0;
            int got = (int)tests[i].chunk;
            sceIoLseek32(fd, (int)offset, PSP_SEEK_SET);
            started = now_us();
            while (done < region && got == (int)tests[i].chunk) {
                got = sceIoRead(fd, buffer + tests[i].buffer_skew,
                                tests[i].chunk);
                if (got > 0) done += (unsigned)got;
            }
            elapsed = now_us() - started;
            if (!elapsed) elapsed = 1;
            say("[card] %s: %u KiB/s", tests[i].name,
                (unsigned)((uint64_t)done * 1000000u / elapsed / 1024u));
            scePowerTick(PSP_POWER_TICK_ALL);
        }
    }
    sceIoClose(fd);
    free(buffer);
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    /* Read the partition figures before anything can touch the heap. */
    unsigned int free_boot = sceKernelTotalFreeMemSize();
    unsigned int max_boot = sceKernelMaxFreeMemSize();
    unsigned int free_heap, max_heap;
    void *first = malloc(16);
    uint8_t *rows = NULL, *groups = NULL;
    float *out = NULL;
    size_t bytes = 8u * 1024u * 1024u, row_count, group_count, i;
    KernelTiming current, vtfm, vdot;
    unsigned int miss_ps;
    int mhz, verified;
    SceCtrlData pad;

    free_heap = sceKernelTotalFreeMemSize();
    max_heap = sceKernelMaxFreeMemSize();
    free(first);
    setup_fpu();
    setup_callbacks();
    pspDebugScreenInit();
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_DIGITAL);
    scePowerSetClockFrequency(333, 333, 166);
    open_log(argc, argv);
    say("LLMPSP bench v" BENCH_VERSION " - about 15 seconds, no input needed");
    mhz = scePowerGetCpuClockFrequency();
    if (mhz <= 0) mhz = 333;

    section_memory(free_boot, max_boot, free_heap, max_heap);
    section_probes();
    section_clock();
    log_sync();

    /* Two equal buffers: FHQ4 rows and the same weights as tile groups.
     * Halve the size until both fit (the emulator has less RAM). */
    for (;;) {
        rows = (uint8_t *)memalign(64, bytes);
        groups = (uint8_t *)memalign(64, bytes);
        if (rows && groups) break;
        free(rows);
        free(groups);
        rows = groups = NULL;
        if (bytes <= 1024u * 1024u) break;
        bytes /= 2u;
    }
    row_count = rows ? (bytes / ROW_BYTES) & ~(size_t)3 : 0;
    group_count = row_count / 4;
    out = row_count ? (float *)memalign(64, row_count * sizeof(float)) : NULL;
    if (!rows || !groups || !out) {
        say("[fail] not enough memory for the test buffers");
        log_sync();
        sceKernelDelayThread(5000000);
        sceKernelExitGame();
        return 1;
    }
    fill_rows(rows, row_count);
    for (i = 0; i < group_count; ++i)
        transcode_group(groups + i * GROUP_BYTES,
                        rows + i * 4 * ROW_BYTES);
    prepare_x();
    note("[data] 2 x %u KiB, %u rows of %d columns",
         (unsigned)(bytes / 1024u), (unsigned)row_count, COLS);
    scePowerTick(PSP_POWER_TICK_ALL);

    verified = section_verify(rows, groups, row_count);
    log_sync();
    miss_ps = section_ram(rows, row_count * ROW_BYTES);
    note("[ram] cache-miss cost about %u ps per 64-byte line "
         "(%u cycles at %d MHz)", miss_ps,
         (unsigned)((uint64_t)miss_ps * (unsigned)mhz / 1000000u), mhz);
    log_sync();
    scePowerTick(PSP_POWER_TICK_ALL);

    current = time_current(rows, row_count, out);
    report_kernel("shipped", &current, mhz);
    scePowerTick(PSP_POWER_TICK_ALL);
    vtfm = time_tile(falcon_q4_tile4_vtfm, groups, group_count, out);
    report_kernel("vtfm4", &vtfm, mhz);
    scePowerTick(PSP_POWER_TICK_ALL);
    vdot = time_tile(falcon_q4_tile4_vdot, groups, group_count, out);
    report_kernel("vdot", &vdot, mhz);
    log_sync();
    {
        /* Matvec time per reply token if every weight were in RAM. */
        unsigned int now_ms = (unsigned)((LAYER_WEIGHTS + HEAD_WEIGHTS) *
                                         (double)current.ram_ps / 1.0e9);
        unsigned int new_ms = (unsigned)((LAYER_WEIGHTS + HEAD_WEIGHTS) *
                                         (double)vtfm.ram_ps / 1.0e9);
        say("[project] matvecs per reply token: shipped %u ms, vtfm4 %u ms",
            now_ms, new_ms);
        if (vtfm.ram_ps)
            say("[project] kernel speedup x%u.%02u (RAM), x%u.%02u (cached)",
                current.ram_ps / vtfm.ram_ps,
                current.ram_ps % vtfm.ram_ps * 100u / vtfm.ram_ps,
                vtfm.l1_ps ? current.l1_ps / vtfm.l1_ps : 0u,
                vtfm.l1_ps ?
                    current.l1_ps % vtfm.l1_ps * 100u / vtfm.l1_ps : 0u);
    }
    free(rows);
    free(groups);
    free(out);
    log_sync();
    scePowerTick(PSP_POWER_TICK_ALL);

    section_card();
    say("[done] %s - results are in bench_log.txt",
        verified ? "all checks passed" : "A CHECK FAILED");
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
