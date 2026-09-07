/* Headless autotest frontend: identical runtime path to src/main_psp.c but
 * drives the prompt "hi" with no controller input, logging checkpoints to
 * stdout so PPSSPPHeadless (or psplink) can capture them. */
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

static int read_cache_mebibytes(void) {
    FILE *file = fopen("cache_mb.txt", "r");
    int value = 32;
    if (file) {
        int configured;
        if (fscanf(file, "%d", &configured) == 1) value = configured;
        fclose(file);
    }
    if (value < 0) value = 0;
    if (value > 44) value = 44;
    return value;
}

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    char error[192];
    char output[OUTPUT_BYTES];
    int tokens[TOKEN_CAPACITY];
    int prompt_tokens, max_new, generated = 0, ok;
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
    log_line("[stage] model open ok, tables %.2f MiB\n",
           (double)falcon_model_resident_bytes(&model) / 1048576.0);
    if (!falcon_runtime_init(&runtime, &model, error, sizeof(error))) {
        log_line("[fail] runtime init: %s\n", error);
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
        int cache_mib = read_cache_mebibytes();
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
        prefetch_ready = falcon_prefetch_open(model_path);
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
        if (prefetch_ready) {
            falcon_prefetch_configure(&model);
            log_line("[stage] prefetch on\n");
        } else {
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
    falcon_model_close(&model);
    sceKernelExitGame();
    return 0;
}
