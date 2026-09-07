#include "falcon_h1.h"
#include "falcon_ui.h"

#include <pspctrl.h>
#include <pspdebug.h>
#include <pspkernel.h>
#include <psppower.h>
#include <pspdisplay.h>
#include <pspiofilemgr.h>
#include <pspsysmem.h>
#include <pspsuspend.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

PSP_MODULE_INFO("LLMPSP", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
PSP_HEAP_SIZE_KB(-2048);

#define PROMPT_BYTES 768
#define TOKEN_CAPACITY 512

static volatile int exiting;
/* Latched by the power-guard thread so a short Circle tap between
 * token boundaries still cancels the reply; cleared at turn start. */
static volatile int stop_requested;
static int volatile_locked;

/* ------------------------------------------------------------------ */
/* Crash-forensics trace                                               */
/* ------------------------------------------------------------------ */

/* Written with raw sceIo calls and reopened after each flush so the last
 * line always reaches the card, even if the console dies immediately
 * afterwards. llmpsp_trace.txt sits next to EBOOT.PBP. */
static SceUID trace_fd = -1;
static char trace_path[512];

static void trace_line(const char *format, ...) {
    char line[256];
    int length;
    va_list args;
    if (trace_fd < 0) return;
    va_start(args, format);
    length = vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (length < 0) return;
    if (length > (int)sizeof(line) - 2) length = (int)sizeof(line) - 2;
    line[length++] = '\n';
    sceIoWrite(trace_fd, line, length);
}

static void trace_sync(void) {
    if (trace_fd < 0) return;
    sceIoClose(trace_fd);
    trace_fd = sceIoOpen(trace_path,
                         PSP_O_WRONLY | PSP_O_CREAT | PSP_O_APPEND, 0777);
}

static void trace_open(int argc, char **argv) {
    /* Raw sceIo paths must be absolute (device-prefixed): derive the
     * EBOOT directory from argv[0], falling back to the newlib cwd. */
    char *slash;
    trace_path[0] = '\0';
    if (argc > 0 && argv[0] && strlen(argv[0]) + 24 < sizeof(trace_path)) {
        strcpy(trace_path, argv[0]);
        slash = strrchr(trace_path, '/');
        if (!slash) slash = strrchr(trace_path, '\\');
        if (slash) strcpy(slash + 1, "llmpsp_trace.txt");
        else trace_path[0] = '\0';
    }
    if (!trace_path[0]) {
        if (!getcwd(trace_path, sizeof(trace_path) - 24)) return;
        strcat(trace_path, "/llmpsp_trace.txt");
    }
    trace_fd = sceIoOpen(trace_path,
                         PSP_O_WRONLY | PSP_O_CREAT | PSP_O_APPEND, 0777);
}

static void trace_power_state(const char *when) {
    trace_line("[battery] %s: percent %d, ac %d, low %d", when,
               scePowerGetBatteryLifePercent(), scePowerIsPowerOnline(),
               scePowerIsLowBattery());
}

/* ------------------------------------------------------------------ */
/* System setup                                                        */
/* ------------------------------------------------------------------ */

/* Disable every FPU exception enable bit and flush denormals to zero.
 * The real fix for the on-device overflow crash is the numerically
 * stable silu() in falcon_h1.c; this removes the remaining trap
 * surface. */
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

static int power_callback(int unknown, int power_info, void *common) {
    (void)unknown; (void)common;
    trace_line("[power] event 0x%08x%s%s%s%s%s", power_info,
               (power_info & PSP_POWER_CB_SUSPENDING) ? " SUSPENDING" : "",
               (power_info & PSP_POWER_CB_STANDBY) ? " STANDBY" : "",
               (power_info & PSP_POWER_CB_RESUMING) ? " RESUMING" : "",
               (power_info & PSP_POWER_CB_BATTERY_LOW) ? " BATTERY_LOW" : "",
               (power_info & PSP_POWER_CB_POWER_SWITCH) ? " POWER_SWITCH" : "");
    trace_sync();
    return 0;
}

static int callback_thread(SceSize args, void *argp) {
    int callback;
    (void)args; (void)argp;
    callback = sceKernelCreateCallback("Exit Callback", exit_callback, NULL);
    if (callback >= 0) sceKernelRegisterExitCallback(callback);
    callback = sceKernelCreateCallback("Power Callback", power_callback, NULL);
    if (callback >= 0) scePowerRegisterCallback(0, callback);
    sceKernelSleepThreadCB();
    return 0;
}
static void setup_callbacks(void) {
    int thread = sceKernelCreateThread("update_thread", callback_thread,
                                       0x11, 0x2000, PSP_THREAD_ATTR_USER, NULL);
    if (thread >= 0) sceKernelStartThread(thread, 0, NULL);
}

static int power_guard_thread(SceSize args, void *argp) {
    int ticks = 0;
    (void)args; (void)argp;
    while (!exiting) {
        SceCtrlData pad;
        sceCtrlPeekBufferPositive(&pad, 1);
        if (pad.Buttons & PSP_CTRL_CIRCLE) stop_requested = 1;
        if (++ticks >= 25) {
            scePowerTick(PSP_POWER_TICK_ALL);
            ticks = 0;
        }
        sceKernelDelayThread(20000);
    }
    return 0;
}
static int setup_power_guard(void) {
    int thread = sceKernelCreateThread("power_guard", power_guard_thread,
                                       0x18, 0x4000, PSP_THREAD_ATTR_USER, NULL);
    if (thread >= 0) sceKernelStartThread(thread, 0, NULL);
    return thread;
}

/* ------------------------------------------------------------------ */
/* Screen                                                              */
/* ------------------------------------------------------------------ */

/* White, the colour pspDebugScreenPrintf uses by default. */
#define UI_TEXT_COLOR 0xFFFFFFFFu

static char ui_grid[UI_ROWS][UI_COLS + 1];
static char ui_shadow[UI_ROWS][UI_COLS + 1];
static int ui_shadow_valid;
static FalconUiState ui_state;
static char status_text[UI_COLS + 1];

/* Composes the screen and blits only the rows that actually changed,
 * which keeps redraws during generation down to a line or two.
 *
 * Rows sit at the explicit pixel positions falcon_ui_row_y() gives them
 * (chat lines carry leading; see falcon_ui.h), not on the debug screen's
 * own 8-pixel row pitch, so they cannot go through the cursor-based
 * pspDebugScreenPrintf. pspDebugScreenPutChar takes pixel coordinates
 * and paints the whole 7x8 cell, background included, so a redrawn row
 * still needs no separate clear and the leading between rows keeps the
 * background pspDebugScreenClear() left there. */
static void ui_refresh(void) {
    int row;
    ui_state.status = status_text;
    falcon_ui_compose(&ui_state, ui_grid);
    for (row = 0; row < UI_ROWS; ++row) {
        int column, y;
        if (ui_shadow_valid &&
            memcmp(ui_shadow[row], ui_grid[row], UI_COLS) == 0)
            continue;
        y = falcon_ui_row_y(row);
        for (column = 0; column < UI_COLS; ++column)
            pspDebugScreenPutChar(column * UI_FONT_WIDTH, y, UI_TEXT_COLOR,
                                  (unsigned char)ui_grid[row][column]);
        memcpy(ui_shadow[row], ui_grid[row], UI_COLS + 1);
    }
    ui_shadow_valid = 1;
}

static void set_status(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(status_text, sizeof(status_text), format, args);
    va_end(args);
}

static void scroll_to_end(void) {
    ui_state.scroll = falcon_ui_max_scroll();
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

static int turn_start_position;
static int turn_prompt_tokens;
static unsigned int turn_started_us;
static unsigned int reply_started_us;
static int reply_tokens;

/* Runs for every layer of every token. The power tick and Circle poll
 * stay per-layer (cheap syscalls, and PROJECT.md requires the tick), but
 * the screen compose/blit and the synced trace write are throttled: the
 * UI redraws four times per token and the heartbeat reaches the card
 * every eighth token, which keeps the per-token card-sync cost off the
 * generation loop now that on-device stability is confirmed. */
#define HEARTBEAT_TOKENS 8

static void inference_progress(int position, int layer,
                               int total_layers, void *user) {
    int relative = position - turn_start_position;
    SceCtrlData pad;
    (void)user;
    scePowerTick(PSP_POWER_TICK_ALL);
    sceCtrlPeekBufferPositive(&pad, 1);
    if (pad.Buttons & PSP_CTRL_CIRCLE) stop_requested = 1;
    if (layer % 6 != 0) return;
    if (relative < turn_prompt_tokens) {
        int total_steps = turn_prompt_tokens * total_layers;
        int done_steps = relative * total_layers +
                         (layer < total_layers ? layer : total_layers);
        set_status("Prompt processing... %d%%",
                   total_steps ? 100 * done_steps / total_steps : 0);
    } else if (reply_tokens > 1) {
        /* The clock starts at the FIRST emitted token, so the rate is
         * intervals (tokens - 1) over elapsed; counting the first token
         * inflated the early readings up to 2x. */
        unsigned int elapsed = sceKernelGetSystemTimeLow() - reply_started_us;
        unsigned int tps_tenths = elapsed ?
            (unsigned)(reply_tokens - 1) * 10000000u / elapsed : 0;
        set_status("Generating reply... %d words, %u.%u t/s",
                   reply_tokens, tps_tenths / 10u, tps_tenths % 10u);
    } else {
        set_status("Generating reply...");
    }
    ui_refresh();
    if (layer == 0 && relative % HEARTBEAT_TOKENS == 0) {
        trace_line("[heartbeat] pos %d (%s), bat %d%%", position + 1,
                   relative < turn_prompt_tokens ? "prompt" : "reply",
                   scePowerGetBatteryLifePercent());
        trace_sync();
    }
}

static unsigned int clock_us_hook(void) {
    return sceKernelGetSystemTimeLow();
}

static int output_token(int token, const uint8_t *piece,
                        size_t piece_len, void *user) {
    SceCtrlData pad;
    (void)token; (void)user;
    if (reply_tokens == 0) reply_started_us = sceKernelGetSystemTimeLow();
    ++reply_tokens;
    if (piece && piece_len) falcon_ui_append((const char *)piece, piece_len);
    scroll_to_end();
    ui_refresh();
    sceCtrlPeekBufferPositive(&pad, 1);
    if (pad.Buttons & PSP_CTRL_CIRCLE) stop_requested = 1;
    return !stop_requested && !exiting;
}

/* ------------------------------------------------------------------ */
/* Configuration and model discovery                                   */
/* ------------------------------------------------------------------ */

static int read_config_int(const char *model_path, const char *filename,
                           int fallback) {
    char path[512];
    char *slash;
    FILE *file;
    int value = fallback;
    if (strlen(model_path) + strlen(filename) + 4 >= sizeof(path)) return value;
    strcpy(path, model_path);
    slash = strrchr(path, '/');
    if (!slash) slash = strrchr(path, '\\');
    if (slash) strcpy(slash + 1, filename);
    else strcpy(path, filename);
    file = fopen(path, "r");
    if (file) {
        int configured;
        if (fscanf(file, "%d", &configured) == 1) value = configured;
        fclose(file);
    }
    return value;
}

/* ------------------------------------------------------------------ */
/* llmpsp.cfg                                                          */
/* ------------------------------------------------------------------ */

/* All user-tunable settings live in llmpsp_config.cfg beside the EBOOT
 * as simple "key = value" lines ('#' starts a comment). The legacy
 * cache_mb.txt and cpu_mhz.txt files are still honored when present but
 * the config file overrides them. Every value is clamped to a safe
 * range. */
typedef struct {
    int cache_mb;        /* heap weight-cache target, 0..46 MiB */
    int cpu_mhz;         /* 100..333 */
    int volatile_mem;    /* 1 = borrow the firmware's 4 MiB volatile RAM */
    int prefetch;        /* 1 = async weight reads; measured useless on the
                          * PSP-2000 memory stick (CPU-bound driver), so
                          * the default is 0 */
    int stripe;          /* 1 = spread cache holes across layers; only
                          * sensible together with prefetch on hardware
                          * whose card reads truly overlap compute */
    int context;         /* 128..512; smaller frees KV RAM for the cache */
    int top_k;           /* 1..64 */
    int max_reply_tokens;/* 8..256 per reply */
    float temperature;   /* 0 = deterministic greedy */
    float top_p;
    float repetition_penalty;
} AppConfig;

static float clamp_float(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}
static int clamp_int(int value, int low, int high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static void load_config(const char *model_path, AppConfig *cfg) {
    char path[512];
    char *slash;
    FILE *file;
    cfg->cache_mb = read_config_int(model_path, "cache_mb.txt", 44);
    cfg->cpu_mhz = read_config_int(model_path, "cpu_mhz.txt", 333);
    cfg->volatile_mem = 1;
    cfg->prefetch = 0;
    cfg->stripe = 0;
    cfg->context = 512;
    cfg->top_k = 32;
    cfg->max_reply_tokens = 128;
    cfg->temperature = 0.0f;
    cfg->top_p = 0.9f;
    cfg->repetition_penalty = 1.08f;
    if (strlen(model_path) + 24 < sizeof(path)) {
        strcpy(path, model_path);
        slash = strrchr(path, '/');
        if (!slash) slash = strrchr(path, '\\');
        if (slash) strcpy(slash + 1, "llmpsp_config.cfg");
        else strcpy(path, "llmpsp_config.cfg");
        file = fopen(path, "r");
        if (file) {
            char line[128], key[32], value[32];
            while (fgets(line, sizeof(line), file)) {
                if (sscanf(line, " %31[A-Za-z_] = %31s", key, value) != 2)
                    continue;
                if (key[0] == '#') continue;
                if (!strcmp(key, "cache_mb")) cfg->cache_mb = atoi(value);
                else if (!strcmp(key, "cpu_mhz")) cfg->cpu_mhz = atoi(value);
                else if (!strcmp(key, "volatile_mem"))
                    cfg->volatile_mem = atoi(value) != 0;
                else if (!strcmp(key, "prefetch"))
                    cfg->prefetch = atoi(value) != 0;
                else if (!strcmp(key, "stripe"))
                    cfg->stripe = atoi(value) != 0;
                else if (!strcmp(key, "context")) cfg->context = atoi(value);
                else if (!strcmp(key, "top_k")) cfg->top_k = atoi(value);
                else if (!strcmp(key, "max_reply_tokens"))
                    cfg->max_reply_tokens = atoi(value);
                else if (!strcmp(key, "temperature"))
                    cfg->temperature = (float)atof(value);
                else if (!strcmp(key, "top_p"))
                    cfg->top_p = (float)atof(value);
                else if (!strcmp(key, "repetition_penalty"))
                    cfg->repetition_penalty = (float)atof(value);
            }
            fclose(file);
        }
    }
    cfg->cache_mb = clamp_int(cfg->cache_mb, 0, 46);
    cfg->cpu_mhz = clamp_int(cfg->cpu_mhz, 100, 333);
    cfg->context = clamp_int(cfg->context, 128, 512);
    cfg->top_k = clamp_int(cfg->top_k, 1, 64);
    cfg->max_reply_tokens = clamp_int(cfg->max_reply_tokens, 8, 256);
    cfg->temperature = clamp_float(cfg->temperature, 0.0f, 2.0f);
    cfg->top_p = clamp_float(cfg->top_p, 0.05f, 1.0f);
    cfg->repetition_penalty =
        clamp_float(cfg->repetition_penalty, 1.0f, 2.0f);
}

static int append_tokens(const FalconModel *model, const char *text,
                         int *tokens, int count) {
    int made;
    if (count < 0) return -1;
    made = falcon_tokenize(&model->tokenizer, text, tokens + count,
                           TOKEN_CAPACITY - count);
    return made < 0 ? -1 : count + made;
}

static int locate_model(char *path, size_t size, int argc, char **argv) {
    FILE *probe = fopen("model.fhq4", "rb");
    if (probe) {
        fclose(probe);
        strncpy(path, "model.fhq4", size - 1);
        path[size - 1] = '\0';
        return 1;
    }
    if (argc > 0 && argv[0] && strlen(argv[0]) + 16 < size) {
        char *slash;
        strncpy(path, argv[0], size - 1);
        path[size - 1] = '\0';
        slash = strrchr(path, '/');
        if (!slash) slash = strrchr(path, '\\');
        if (slash) {
            strcpy(slash + 1, "model.fhq4");
            probe = fopen(path, "rb");
            if (probe) {
                fclose(probe);
                return 1;
            }
        }
    }
    return 0;
}

/* Shows an unrecoverable startup problem and waits for the HOME menu. */
static void fatal_screen(const char *title, const char *detail) {
    SceCtrlData pad;
    pspDebugScreenClear();
    pspDebugScreenSetXY(0, 0);
    pspDebugScreenPrintf("Falcon-H1 Tiny 90M\n\n%s\n\n%s\n\n", title, detail);
    pspDebugScreenPrintf("Press the PS/HOME button to quit.\n");
    while (!exiting) {
        sceCtrlReadBufferPositive(&pad, 1);
        sceDisplayWaitVblankStart();
    }
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    AppConfig config;
    char error[192], model_path[512];
    char prompt[PROMPT_BYTES] = "";
    int tokens[TOKEN_CAPACITY];
    int conv_position = 0, conv_pending = -1;
    int lock_rc, guard_rc, scroll_hold = 0, draft_changed = 0;
    unsigned int previous = 0;
    SceCtrlData pad;

    setup_fpu();
    setup_callbacks();
    guard_rc = setup_power_guard();
    pspDebugScreenInit();
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
    scePowerSetClockFrequency(333, 333, 166);
    /* Block automatic and switch-initiated suspend for the whole session:
     * suspending with CFW extra RAM in use is not survivable. */
    lock_rc = scePowerLock(0);

    pspDebugScreenPrintf("Falcon-H1 Tiny 90M  v1.1\nStarting...\n");
    trace_open(argc, argv);
    if (!locate_model(model_path, sizeof(model_path), argc, argv)) {
        trace_line("[fail] model.fhq4 not found");
        trace_sync();
        fatal_screen("model.fhq4 was not found next to EBOOT.PBP.",
                     "Copy the whole LLMPSP folder to ms0:/PSP/GAME/.");
        sceKernelExitGame();
        return 1;
    }
    trace_line("=== boot LLMPSP v1.1 ===");
    trace_line("[stage] power lock rc 0x%08x, guard thread 0x%08x",
               lock_rc, guard_rc);
    load_config(model_path, &config);
    trace_line("[cfg] cache %d MiB, %d MHz, volatile %d, prefetch %d, "
               "stripe %d, ctx %d, temp %d.%02d, max reply %d",
               config.cache_mb, config.cpu_mhz, config.volatile_mem,
               config.prefetch, config.stripe, config.context,
               (int)config.temperature,
               (int)(config.temperature * 100.0f) % 100,
               config.max_reply_tokens);
    {
        int mhz = config.cpu_mhz;
        if (mhz != 333) scePowerSetClockFrequency(mhz, mhz, mhz / 2);
        trace_line("[stage] clock %d/%d MHz, free partition %u KiB",
                   scePowerGetCpuClockFrequency(),
                   scePowerGetBusClockFrequency(),
                   (unsigned)(sceKernelTotalFreeMemSize() / 1024));
    }
    trace_power_state("boot");
    trace_sync();

    if (!falcon_model_open(&model, model_path, error, sizeof(error))) {
        trace_line("[fail] model open: %s", error);
        trace_sync();
        fatal_screen("The model file could not be opened.", error);
        sceKernelExitGame();
        return 1;
    }
    trace_line("[stage] model open ok");
    /* A smaller context shrinks the KV cache before runtime_init, which
     * frees heap that the weight cache then claims automatically. */
    if (config.context < model.config.context)
        model.config.context = config.context;
    pspDebugScreenPrintf("Model tables: %.2f MiB\n",
                         (double)falcon_model_resident_bytes(&model) / 1048576.0);
    pspDebugScreenPrintf("Allocating recurrent state and 512-token cache...\n");
    if (!falcon_runtime_init(&runtime, &model, error, sizeof(error))) {
        trace_line("[fail] runtime init: %s", error);
        trace_sync();
        falcon_model_close(&model);
        fatal_screen("There is not enough memory to start the model.", error);
        sceKernelExitGame();
        return 1;
    }
    trace_line("[stage] runtime init ok, %u KiB",
               (unsigned)(falcon_runtime_resident_bytes(&runtime) / 1024));
    falcon_runtime_set_progress(&runtime, inference_progress, NULL);
    runtime.clock_us = clock_us_hook;

    {
        int cache_mib = config.cache_mb;
        size_t cached;
        int holes = 0, i;
        void *volatile_mem = NULL;
        int volatile_size = 0, volatile_rc;
        int prefetch_ready = 0;
        /* The 4 MiB volatile partition (idle UMD cache) is a retail user-
         * mode API on every firmware and CFW; when the lock fails the app
         * simply runs with the heap cache alone. scePowerLock is already
         * held for the whole session, so suspend cannot wipe it mid-run.
         * cache_mb 0 keeps its documented meaning of "no cache at all",
         * the minimum-memory troubleshooting mode. */
        volatile_rc = config.volatile_mem && cache_mib > 0 ?
            sceKernelVolatileMemTryLock(0, &volatile_mem, &volatile_size) :
            -1;
        if (volatile_rc < 0 || !volatile_mem || volatile_size <= 0) {
            volatile_mem = NULL;
            volatile_size = 0;
        }
        volatile_locked = volatile_mem != NULL;
        trace_line("[stage] volatile mem rc 0x%08x, %d KiB",
                   volatile_rc, volatile_size / 1024);
        /* The prefetch double buffer must be allocated before the cache
         * greedily takes the rest of the heap. */
        if (config.prefetch)
            prefetch_ready = falcon_prefetch_open(model_path);
        pspDebugScreenPrintf("Loading weights into RAM (%d+%d MiB)...\n",
                             cache_mib, volatile_size / 1048576);
        trace_line("[stage] caching weights, target %d MiB", cache_mib);
        trace_sync();
        /* The default prefix layout keeps the streamed tail one
         * sequential scan, which is what the memory stick reads
         * fastest. Striping only pays off with a prefetcher whose reads
         * genuinely overlap compute, which the stock memory-stick
         * driver does not provide (section 2.6 of PROJECT.md). */
        cached = falcon_model_cache_weights_extra(&model,
                                           (size_t)cache_mib * 1024u * 1024u,
                                           (uint8_t *)volatile_mem,
                                           (size_t)volatile_size,
                                           (prefetch_ready && config.stripe) ?
                                               1 : 0,
                                           error, sizeof(error));
        for (i = 0; i < model.weight_cache_blocks; ++i)
            if (!model.weight_cache[i]) ++holes;
        if (prefetch_ready) {
            falcon_prefetch_configure(&model);
            trace_line("[stage] prefetch on, streaming %u KiB in %d holes",
                       (unsigned)((model.file_size - model.weights_offset -
                                   cached) / 1024u), holes);
        } else {
            falcon_prefetch_close();
            trace_line("[stage] prefetch off");
        }
        trace_line("[stage] cache ready, %u KiB in %d blocks, %d uncached",
                   (unsigned)(cached / 1024),
                   model.weight_cache_blocks - holes, holes);
        trace_sync();
        set_status("Ready. %u MiB cached, %u MiB streaming.",
                   (unsigned)(cached / 1048576),
                   (unsigned)((model.file_size - model.weights_offset -
                               cached) / 1048576));
    }

    falcon_ui_reset();
    falcon_ui_set_draft(prompt);
    ui_state.cursor = 0;
    ui_state.upper = 0;
    ui_state.context_total = model.config.context;
    ui_state.context_used = 0;
    ui_state.scroll = 0;
    ui_state.busy = 0;
    pspDebugScreenClear();
    ui_shadow_valid = 0;
    ui_refresh();

    while (!exiting) {
        unsigned int pressed;
        size_t prompt_length;
        int row, column;
        sceCtrlReadBufferPositive(&pad, 1);
        pressed = pad.Buttons & ~previous;
        previous = pad.Buttons;

        row = ui_state.cursor / UI_KEY_COLUMNS;
        column = ui_state.cursor % UI_KEY_COLUMNS;
        if (pressed & PSP_CTRL_LEFT)
            ui_state.cursor = row * UI_KEY_COLUMNS +
                              (column + UI_KEY_COLUMNS - 1) % UI_KEY_COLUMNS;
        if (pressed & PSP_CTRL_RIGHT)
            ui_state.cursor = row * UI_KEY_COLUMNS +
                              (column + 1) % UI_KEY_COLUMNS;
        if (pressed & PSP_CTRL_UP)
            ui_state.cursor = (ui_state.cursor + UI_KEY_COUNT -
                               UI_KEY_COLUMNS) % UI_KEY_COUNT;
        if (pressed & PSP_CTRL_DOWN)
            ui_state.cursor = (ui_state.cursor + UI_KEY_COLUMNS) % UI_KEY_COUNT;
        if (pressed & (PSP_CTRL_LTRIGGER | PSP_CTRL_RTRIGGER))
            ui_state.upper = !ui_state.upper;

        /* Analog stick scrolls the transcript, repeating while held. */
        if (pad.Ly < 64 || pad.Ly > 192) {
            if (scroll_hold == 0) {
                int maximum = falcon_ui_max_scroll();
                ui_state.scroll += pad.Ly < 64 ? -1 : 1;
                if (ui_state.scroll < 0) ui_state.scroll = 0;
                if (ui_state.scroll > maximum) ui_state.scroll = maximum;
            }
            scroll_hold = (scroll_hold + 1) % 4;
        } else {
            scroll_hold = 0;
        }

        prompt_length = strlen(prompt);
        if ((pressed & PSP_CTRL_CROSS) && prompt_length + 1 < sizeof(prompt)) {
            prompt[prompt_length] =
                falcon_ui_keys(ui_state.upper)[ui_state.cursor];
            prompt[prompt_length + 1] = '\0';
            draft_changed = 1;
        }
        if ((pressed & PSP_CTRL_SQUARE) && prompt_length + 1 < sizeof(prompt)) {
            prompt[prompt_length] = ' ';
            prompt[prompt_length + 1] = '\0';
            draft_changed = 1;
        }
        if ((pressed & PSP_CTRL_TRIANGLE) && prompt_length) {
            prompt[prompt_length - 1] = '\0';
            draft_changed = 1;
        }
        if (pressed & PSP_CTRL_SELECT) {
            prompt[0] = '\0';
            conv_position = 0;
            conv_pending = -1;
            ui_state.context_used = 0;
            ui_state.scroll = 0;
            falcon_ui_reset();
            draft_changed = 1;
            set_status("New conversation started.");
        }
        if (draft_changed) {
            falcon_ui_set_draft(prompt);
            scroll_to_end();
            draft_changed = 0;
        }

        if (pressed & PSP_CTRL_START) {
            int turn_tokens, max_new, generated = 0, ok;
            int next_position = 0, pending = -1;
            unsigned int elapsed, reply_elapsed;
            if (!prompt[0]) {
                set_status("Type a message first, then press START.");
                ui_refresh();
                sceDisplayWaitVblankStart();
                continue;
            }
            if (conv_position == 0) {
                turn_tokens = falcon_build_chat_prompt(&model, prompt,
                                                       tokens, TOKEN_CAPACITY);
            } else {
                /* Continue the conversation: close the previous assistant
                 * turn and append the new user turn. Only these delta
                 * tokens are processed; accumulated state is kept. */
                turn_tokens = 0;
                if (conv_pending >= 0) tokens[turn_tokens++] = conv_pending;
                tokens[turn_tokens++] = model.config.im_end_id;
                turn_tokens = append_tokens(&model, "\n", tokens, turn_tokens);
                if (turn_tokens > 0 && turn_tokens < TOKEN_CAPACITY)
                    tokens[turn_tokens++] = model.config.im_start_id;
                turn_tokens = append_tokens(&model, "user\n", tokens, turn_tokens);
                turn_tokens = append_tokens(&model, prompt, tokens, turn_tokens);
                if (turn_tokens > 0 && turn_tokens < TOKEN_CAPACITY)
                    tokens[turn_tokens++] = model.config.im_end_id;
                turn_tokens = append_tokens(&model, "\n", tokens, turn_tokens);
                if (turn_tokens > 0 && turn_tokens < TOKEN_CAPACITY)
                    tokens[turn_tokens++] = model.config.im_start_id;
                turn_tokens = append_tokens(&model, "assistant\n",
                                            tokens, turn_tokens);
            }
            if (turn_tokens <= 0 || turn_tokens >= TOKEN_CAPACITY) {
                set_status("That message is too long for the 512-token space.");
                ui_refresh();
                continue;
            }
            max_new = model.config.context - conv_position - turn_tokens;
            if (max_new > config.max_reply_tokens)
                max_new = config.max_reply_tokens;
            if (max_new < 8) {
                set_status("The conversation is full. SELECT starts a new one.");
                ui_refresh();
                continue;
            }

            /* Freeze the typed line into history and clear the editor
             * right away, so the message can never be on screen twice. */
            falcon_ui_commit_draft(prompt);
            prompt[0] = '\0';
            turn_start_position = conv_position;
            turn_prompt_tokens = turn_tokens;
            turn_started_us = sceKernelGetSystemTimeLow();
            reply_tokens = 0;
            stop_requested = 0;
            ui_state.busy = 1;
            scroll_to_end();
            set_status("Prompt processing... 0%%");
            ui_refresh();

            trace_line("[stage] turn begin at %d, %d turn tokens, max_new %d",
                       conv_position, turn_tokens, max_new);
            trace_power_state("turn begin");
            trace_sync();
            runtime.forward_us = 0;
            runtime.matvec_us = 0;
            runtime.streamed_bytes = 0;
            falcon_prefetch_stats_reset();
            falcon_sampler_init(&sampler, config.temperature, config.top_p,
                                config.top_k, config.repetition_penalty,
                                (uint64_t)time(NULL));
            ok = falcon_generate_turn(&model, &runtime, &sampler, tokens,
                                      turn_tokens, conv_position, max_new,
                                      output_token, NULL, &generated,
                                      &next_position, &pending);
            elapsed = sceKernelGetSystemTimeLow() - turn_started_us;
            if (ok) {
                conv_position = next_position;
                conv_pending = pending;
            } else {
                conv_position = 0;
                conv_pending = -1;
                falcon_ui_reset();
            }
            falcon_ui_end_turn();
            falcon_ui_set_draft(prompt);
            ui_state.busy = 0;
            ui_state.context_used = conv_position;
            scroll_to_end();
            /* elapsed spans the whole turn (prompt + reply); the reply
             * clock starts at the first generated token, matching the
             * live t/s shown during generation. */
            reply_elapsed = reply_tokens > 0 ?
                sceKernelGetSystemTimeLow() - reply_started_us : 0;
            trace_line("[stage] turn %s, %d tokens, %u ms, io_error %d, ctx %d/%d",
                       ok ? "ok" : "FAILED", generated, elapsed / 1000u,
                       runtime.io_error, conv_position, model.config.context);
            /* Perf split for tuning: matvec time includes its file I/O;
             * (forward - matvec) is attention/mamba/norm/misc; the rest of
             * elapsed is sampling, UI, and trace overhead. */
            trace_line("[perf] forward %u ms, matvec %u ms, streamed %u KiB, "
                       "prompt %u ms, reply %u ms",
                       runtime.forward_us / 1000u, runtime.matvec_us / 1000u,
                       (unsigned)(runtime.streamed_bytes / 1024u),
                       (elapsed - reply_elapsed) / 1000u,
                       reply_elapsed / 1000u);
            {
                unsigned int pf_kib, pf_wait;
                int pf_alive;
                falcon_prefetch_stats(&pf_kib, &pf_wait, &pf_alive);
                trace_line("[perf] prefetch served %u KiB, waited %u ms, "
                           "alive %d", pf_kib, pf_wait, pf_alive);
            }
            trace_power_state("turn end");
            trace_sync();
            if (!ok)
                set_status("Could not read the model file. Conversation reset.");
            else if (generated > 1) {
                unsigned int tps_tenths = reply_elapsed ?
                    (unsigned)(generated - 1) * 10000000u / reply_elapsed : 0;
                set_status("Done: %d words in %u s (%u.%u t/s). START sends next.",
                           generated, reply_elapsed / 1000000u,
                           tps_tenths / 10u, tps_tenths % 10u);
            }
            else if (generated == 1)
                set_status("Done: 1 word. START sends the next message.");
            else
                set_status("The model had nothing to add. START sends next.");
            previous = pad.Buttons;
        }

        ui_refresh();
        sceDisplayWaitVblankStart();
    }

    trace_line("[stage] clean exit");
    trace_sync();
    if (trace_fd >= 0) sceIoClose(trace_fd);
    falcon_prefetch_close();
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    if (volatile_locked) sceKernelVolatileMemUnlock(0);
    if (lock_rc >= 0) scePowerUnlock(0);
    sceKernelExitGame();
    return 0;
}
