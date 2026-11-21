/* Headless autotest frontend: identical runtime path to src/main_psp.c but
 * drives the prompt "hi" with no controller input, logging checkpoints to
 * stdout so PPSSPPHeadless (or psplink) can capture them.
 *
 * Optional files beside the EBOOT (all absent = the app's defaults):
 *   cache_mb.txt          heap cache target in MiB (default 32)
 *   context.txt           requested context, 128..512 (default 384, as
 *                         the app)
 *   prefetch.txt          1 = async prefetcher with a striped cache
 *                         (default 0, as the app)
 *   autotest_free_kb.txt  test hook: before the runtime is allocated,
 *                         fill the heap so only about this many KiB stay
 *                         free, to exercise the startup context back-off
 *   autotest_heap_blocks.txt  test hook: at boot, before anything else
 *                         is allocated, fill the heap so that exactly this
 *                         many 256 KiB cache blocks still fit (the
 *                         reference PSP-2000 holds 199), to model a
 *                         console's memory budget
 *   fit.txt               0 = no fit policy (default 1, as the app)
 *   state_f16.txt         0 = float32 Mamba state (default 1, as the app) */
#include "falcon_h1.h"

#include <pspctrl.h>
#include <pspdebug.h>
#include <pspkernel.h>
#include <psppower.h>
#include <pspdisplay.h>
#include <pspsuspend.h>

#include <pspiofilemgr.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

PSP_MODULE_INFO("LLMPSPTest", PSP_MODULE_USER, 1, 1);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(-2048);

#define OUTPUT_BYTES 8192
#define TOKEN_CAPACITY 512

static volatile int exiting;
static FILE *log_file;

/* Mirror the main app: no FPU exception enables, flush denormals. */
static void setup_fpu(void) {
    unsigned int fcr;
    __asm__ volatile ("cfc1 %0, $31" : "=r"(fcr));
    fcr &= ~0x00000f80u;
    fcr |= 0x01000000u;
    __asm__ volatile ("ctc1 %0, $31" : : "r"(fcr));
}

static void log_line(const char *format, ...) {
    char line[512];
    int length;
    va_list args;
    va_start(args, format);
    length = vsnprintf(line, sizeof(line) - 1, format, args);
    va_end(args);
    if (length < 0) return;
    sceIoWrite(1, line, (unsigned)length);
    if (log_file) {
        fwrite(line, 1, (unsigned)length, log_file);
        fflush(log_file);
    }
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
    sceKernelRegisterExitCallback(callback);
    sceKernelSleepThreadCB();
    return 0;
}
static void setup_callbacks(void) {
    int thread = sceKernelCreateThread("update_thread", callback_thread,
                                       0x11, 0xFA0, PSP_THREAD_ATTR_USER, NULL);
    if (thread >= 0) sceKernelStartThread(thread, 0, NULL);
}

static int power_guard_thread(SceSize args, void *argp) {
    (void)args; (void)argp;
    while (!exiting) {
        scePowerTick(PSP_POWER_TICK_ALL);
        sceKernelDelayThread(500000);
    }
    return 0;
}
static void setup_power_guard(void) {
    int thread = sceKernelCreateThread("power_guard", power_guard_thread,
                                       0x18, 0x800, PSP_THREAD_ATTR_USER, NULL);
    if (thread >= 0) sceKernelStartThread(thread, 0, NULL);
}

typedef struct {
    char *text;
    size_t length, capacity;
    int token_count;
} OutputState;

static void inference_progress(int position, int layer,
                               int total_layers, void *user) {
    (void)user;
    scePowerTick(PSP_POWER_TICK_ALL);
    pspDebugScreenSetXY(0, 0);
    if (layer < total_layers)
        pspDebugScreenPrintf("Falcon-H1: token %d, layer %d/%d               ",
                             position + 1, layer + 1, total_layers);
    else
        pspDebugScreenPrintf("Falcon-H1: token %d, output head              ",
                             position + 1);
    log_line("[progress] token %d layer %d/%d\n",
             position + 1, layer, total_layers);
}

static int output_token(int token, const uint8_t *piece,
                        size_t piece_len, void *user) {
    OutputState *state = (OutputState *)user;
    SceCtrlData pad;
    size_t available;
    available = state->capacity - state->length - 1;
    if (piece_len > available) piece_len = available;
    if (piece && piece_len) {
        memcpy(state->text + state->length, piece, piece_len);
        state->length += piece_len;
        state->text[state->length] = '\0';
    }
    ++state->token_count;
    log_line("[token] id=%d count=%d text='%s'\n",
           token, state->token_count, state->text);
    sceCtrlPeekBufferPositive(&pad, 1);
    return !(pad.Buttons & PSP_CTRL_CIRCLE) && !exiting;
}

static int read_int_file(const char *name, int fallback, int low, int high) {
    FILE *file = fopen(name, "r");
    int value = fallback;
    if (file) {
        int configured;
        if (fscanf(file, "%d", &configured) == 1) value = configured;
        fclose(file);
    }
    if (value < low) value = low;
    if (value > high) value = high;
    return value;
}

/* Ballast for the back-off test: 64 KiB chunks taken until malloc fails,
 * then the most recent ones returned (they sit at the top of the heap and
 * coalesce) until about keep_kib are free again. */
#define BALLAST_CHUNK (64 * 1024)
#define BALLAST_MAX 1024
static void *ballast[BALLAST_MAX];
static int ballast_count;

static void ballast_fill(int keep_kib) {
    int release = keep_kib / (BALLAST_CHUNK / 1024);
    while (ballast_count < BALLAST_MAX) {
        void *p = malloc(BALLAST_CHUNK);
        if (!p) break;
        ballast[ballast_count++] = p;
    }
    while (release-- > 0 && ballast_count > 0)
        free(ballast[--ballast_count]);
}
static void ballast_free(void) {
    while (ballast_count > 0) free(ballast[--ballast_count]);
}

/* How many weight-cache blocks (256 KiB plus the runtime's alignment
 * overhead) the heap can hold right now; everything is freed again. */
#define PROBE_BLOCK (256 * 1024 + 32)
static void *probe_blocks[512];
static int heap_blocks(int limit) {
    int count = 0;
    if (limit > 512) limit = 512;
    while (count < limit) {
        void *p = malloc(PROBE_BLOCK);
        if (!p) break;
        probe_blocks[count++] = p;
    }
    {
        int n = count;
        while (n > 0) free(probe_blocks[--n]);
    }
    return count;
}

/* Budget ballast: 16 KiB chunks until malloc fails, then chunks from the
 * top of the heap are returned until exactly `blocks` cache blocks fit
 * again (the slack left is under one chunk). */
#define BUDGET_CHUNK (16 * 1024)
#define BUDGET_MAX 4096
static void *budget[BUDGET_MAX];
static int budget_count;
static void budget_fill(int blocks) {
    while (budget_count < BUDGET_MAX) {
        void *p = malloc(BUDGET_CHUNK);
        if (!p) break;
        budget[budget_count++] = p;
    }
    /* Coarse: whole blocks' worth at a time, then back one step and
     * release single chunks. */
    while (budget_count > 0 && heap_blocks(blocks) < blocks) {
        int n = 16;
        while (n-- > 0 && budget_count > 0) free(budget[--budget_count]);
    }
    {
        int n = 16;
        while (n-- > 0 && budget_count < BUDGET_MAX) {
            void *p = malloc(BUDGET_CHUNK);
            if (!p) break;
            budget[budget_count++] = p;
        }
    }
    while (budget_count > 0 && heap_blocks(blocks) < blocks)
        free(budget[--budget_count]);
}
static void budget_free(void) {
    while (budget_count > 0) free(budget[--budget_count]);
}

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    char error[192];
    char output[OUTPUT_BYTES];
    int tokens[TOKEN_CAPACITY];
    int prompt_tokens, max_new, generated = 0, ok;
    int requested_context, granted_context, use_prefetch, free_kib, use_fit;
    const char *model_path = NULL;
    OutputState state;
    (void)argc; (void)argv;
    setvbuf(stdout, NULL, _IONBF, 0);
    setup_fpu();
    setup_callbacks();
    setup_power_guard();
    pspDebugScreenInit();
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
    scePowerSetClockFrequency(333, 333, 166);
    log_file = fopen("falcon_autotest_log.txt", "w");
    log_line("[stage] start autotest\n");
    pspDebugScreenPrintf("Falcon autotest\n");
    {
        int want = read_int_file("autotest_heap_blocks.txt", -1, -1, 512);
        log_line("[mem] heap holds %d cache blocks at boot\n",
                 heap_blocks(512));
        if (want >= 0) {
            budget_fill(want);
            log_line("[mem] budget ballast %d x 16 KiB: heap now holds %d "
                     "cache blocks (%d requested)\n", budget_count,
                     heap_blocks(512), want);
        }
    }
    {
        static const char *candidates[] = {
            "model.fhq4",
            "ms0:/PSP/GAME/LLMPSP/model.fhq4",
            "ef0:/PSP/GAME/LLMPSP/model.fhq4",
        };
        int found = 0;
        unsigned i;
        for (i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
            if (falcon_model_open(&model, candidates[i],
                                  error, sizeof(error))) {
                model_path = candidates[i];
                found = 1;
                break;
            }
        }
        if (!found) {
            log_line("[fail] model open: %s\n", error);
            sceKernelExitGame();
            return 1;
        }
    }
    log_line("[stage] model open ok, tables %.2f MiB, raw stream %d\n",
           (double)falcon_model_resident_bytes(&model) / 1048576.0,
           model.stream_fd >= 0);
    use_fit = read_int_file("fit.txt", 1, 0, 1);
    model.state_mode = read_int_file("state_f16.txt", 1, 0, 1) ?
                       FALCON_STATE_F16 : FALCON_STATE_F32;
    log_line("[stage] fit %d, state_f16 %d\n", use_fit,
             model.state_mode == FALCON_STATE_F16);
    /* Mirror the app: default context 384 (the header holds the 512
     * maximum), then the same graceful back-off. */
    requested_context = read_int_file("context.txt", 384,
                                      FALCON_MIN_CONTEXT, FALCON_MAX_CONTEXT);
    model.config.context = requested_context;
    requested_context = model.config.context;
    free_kib = read_int_file("autotest_free_kb.txt", -1, -1, 65536);
    if (free_kib >= 0) {
        ballast_fill(free_kib);
        log_line("[stage] ballast %d x 64 KiB, about %d KiB left free\n",
                 ballast_count, free_kib);
    }
    granted_context = falcon_runtime_init_fit(&runtime, &model,
                                              FALCON_MIN_CONTEXT, 16,
                                              error, sizeof(error));
    log_line("[stage] context requested %d, granted %d\n",
             requested_context, granted_context);
    if (!granted_context) {
        log_line("[fail] runtime init: %s\n", error);
        ballast_free();
        falcon_model_close(&model);
        sceKernelExitGame();
        return 1;
    }
    log_line("[stage] runtime init ok, %.2f MiB\n",
           (double)falcon_runtime_resident_bytes(&runtime) / 1048576.0);
    falcon_runtime_set_progress(&runtime, inference_progress, NULL);
    runtime.clock_us = (FalconClockCallback)sceKernelGetSystemTimeLow;
    {
        /* Mirror the main app: volatile memory first, heap on top. */
        int cache_mib = read_int_file("cache_mb.txt", 32, 0, 44);
        size_t target_cache = (size_t)cache_mib * 1024u * 1024u;
        void *volatile_mem = NULL;
        int volatile_size = 0;
        int volatile_rc = cache_mib > 0 ?
            sceKernelVolatileMemTryLock(0, &volatile_mem, &volatile_size) :
            -1;
        size_t cached;
        if (volatile_rc < 0 || !volatile_mem || volatile_size <= 0) {
            volatile_mem = NULL;
            volatile_size = 0;
        }
        int prefetch_ready, holes = 0, i;
        log_line("[stage] volatile mem rc 0x%08x, %d KiB\n",
                 volatile_rc, volatile_size / 1024);
        use_prefetch = read_int_file("prefetch.txt", 0, 0, 1);
        prefetch_ready = use_prefetch ? falcon_prefetch_open(model_path) : 0;
        cached = falcon_model_cache_weights_extra(&model, target_cache,
                                                  (uint8_t *)volatile_mem,
                                                  (size_t)volatile_size,
                                                  prefetch_ready ? 1 : 0,
                                                  error, sizeof(error));
        for (i = 0; i < model.weight_cache_blocks; ++i)
            if (!model.weight_cache[i]) ++holes;
        log_line("[stage] cache requested %d MiB got %.2f MiB, "
                 "%d holes (%s)\n",
               cache_mib, (double)cached / 1048576.0, holes, error);
        if (cache_mib > 0) {
            /* Mirror the app's fit policy (section 2.10). */
            char reason[128];
            int floor_context = requested_context - requested_context / 4;
            int before = model.config.context;
            if (floor_context < FALCON_MIN_CONTEXT)
                floor_context = FALCON_MIN_CONTEXT;
            if (use_fit) {
                granted_context = falcon_fit_context(&model, &runtime,
                                                     before - floor_context,
                                                     FALCON_MIN_CONTEXT,
                                                     reason, sizeof(reason));
            } else {
                granted_context = before;
                snprintf(reason, sizeof(reason), "fit = 0");
            }
            log_line("[stage] fit: context requested %d, granted %d (%s)\n",
                     requested_context, granted_context, reason);
            if (!granted_context) {
                log_line("[fail] runtime restore after fit\n");
                sceKernelExitGame();
                return 1;
            }
        }
        log_line("[stage] raw read window %s\n",
                 falcon_model_stream_prepare(&model) ?
                     (model.stream_ready ? "allocated" : "not needed") :
                     "unavailable, stdio streaming");
        holes = falcon_model_uncached_blocks(&model);
        log_line("[mem] runtime %u KiB at context %d, weight cache %u KiB "
                 "(%d volatile + %d heap blocks), %d blocks uncached; heap "
                 "left %d cache blocks\n",
                 (unsigned)(falcon_runtime_resident_bytes(&runtime) / 1024),
                 model.config.context,
                 (unsigned)(model.weight_cache_bytes / 1024),
                 (int)(model.cache_extra_bytes / (256u * 1024u)),
                 model.weight_cache ?
                     model.weight_cache_blocks - holes -
                     (int)(model.cache_extra_bytes / (256u * 1024u)) : 0,
                 holes, heap_blocks(512));
        if (prefetch_ready) {
            falcon_prefetch_configure(&model);
            log_line("[stage] prefetch on\n");
        } else {
            falcon_prefetch_close();
            log_line("[stage] prefetch off\n");
        }
        falcon_prefetch_stats_reset();
    }
    prompt_tokens = falcon_build_chat_prompt(&model, "hi",
                                             tokens, TOKEN_CAPACITY);
    log_line("[stage] prompt tokens %d\n", prompt_tokens);
    if (prompt_tokens <= 0 || prompt_tokens >= model.config.context - 1) {
        log_line("[fail] prompt tokenization\n");
        falcon_runtime_free(&runtime);
        falcon_model_close(&model);
        sceKernelExitGame();
        return 1;
    }
    max_new = model.config.context - prompt_tokens;
    if (max_new > 8) max_new = 8;
    output[0] = '\0';
    state.text = output; state.length = 0; state.capacity = sizeof(output);
    state.token_count = 0;
    falcon_sampler_init(&sampler, 0.0f, 0.9f, 32, 1.08f,
                        (uint64_t)time(NULL));
    log_line("[stage] generation begin, max_new %d\n", max_new);
    ok = falcon_generate(&model, &runtime, &sampler, tokens,
                         prompt_tokens, max_new, output_token,
                         &state, &generated);
    log_line("[stage] generation %s, generated %d, io_error %d\n",
           ok ? "ok" : "FAILED", generated, runtime.io_error);
    log_line("[perf] forward %u ms, matvec %u ms, streamed %u KiB\n",
             runtime.forward_us / 1000u, runtime.matvec_us / 1000u,
             (unsigned)(runtime.streamed_bytes / 1024u));
    /* The PSP alignment guard in q4_matvec must never fire: every tile
     * group is 16-byte aligned by construction. Nonzero means a layout
     * or allocator bug that real hardware would have faulted on. */
    log_line("[check] realigned groups %u (%s)\n",
             (unsigned)runtime.realigned_groups,
             runtime.realigned_groups ? "FAIL: must be 0" : "ok");
    {
        unsigned int pf_kib, pf_wait;
        int pf_alive;
        falcon_prefetch_stats(&pf_kib, &pf_wait, &pf_alive);
        log_line("[perf] prefetch served %u KiB, waited %u ms, alive %d\n",
                 pf_kib, pf_wait, pf_alive);
    }
    log_line("[result] '%s'\n", output);
    log_line("[stage] autotest done\n");
    falcon_prefetch_close();
    falcon_runtime_free(&runtime);
    ballast_free();
    falcon_model_close(&model);
    budget_free();
    sceKernelExitGame();
    return 0;
}
