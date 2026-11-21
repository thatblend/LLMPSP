/* Host test for the startup context back-off (falcon_runtime_init_fit)
 * and for running a conversation up to a backed-off context.
 *
 * 1. Measures the runtime size at every context the back-off can grant
 *    (512, 496, ..., 128), then caps the allocator with the test hook
 *    falcon_runtime_set_alloc_limit at several budgets and checks that
 *    init_fit grants exactly the largest context that fits, stores it in
 *    model.config.context, and fails cleanly (context restored) when not
 *    even 128 fits.
 * 2. With a budget that grants a context below the requested one, runs a
 *    multi-turn greedy chat the way src/main_psp.c does until the
 *    conversation is full, and checks that every turn succeeds and stops
 *    at the granted limit.
 *
 * Usage: context_backoff_test MODEL.fhq4   (prints "backoff: ok" last) */
#include "falcon_h1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STEP 16
#define CANDIDATES ((512 - FALCON_MIN_CONTEXT) / STEP + 1)

static size_t size_at(FalconModel *model, int context) {
    FalconRuntime runtime;
    char error[128];
    size_t bytes;
    model->config.context = context;
    falcon_runtime_set_alloc_limit(0);
    if (!falcon_runtime_init(&runtime, model, error, sizeof(error))) {
        printf("backoff: FAIL cannot allocate context %d: %s\n", context,
               error);
        exit(1);
    }
    bytes = runtime.bytes_allocated;
    falcon_runtime_free(&runtime);
    if (bytes != falcon_runtime_plan_bytes(model, context)) {
        printf("backoff: FAIL plan %lu != allocated %lu at context %d\n",
               (unsigned long)falcon_runtime_plan_bytes(model, context),
               (unsigned long)bytes, context);
        exit(1);
    }
    return bytes;
}

static int collect(int token, const uint8_t *piece, size_t piece_len,
                   void *user) {
    int *list = (int *)user;
    (void)piece; (void)piece_len;
    if (list[0] < 63) list[1 + list[0]++] = token;
    return 1;
}

/* The startup fit policy under a modelled heap (falcon_set_heap_budget):
 * runtime arrays and cache blocks share one budget, plus a 4 MiB foreign
 * region like the PSP's volatile memory. Budget A holds context 384, every
 * block and the policy's 16 KiB headroom exactly; B is A minus five tokens
 * of runtime, so the fit must grant exactly 379; C is 3 MiB short, beyond 25% of the context, so the requested
 * context stays and the remainder streams; D is B with the policy off.
 * Every case must give the same greedy tokens as the uncapped run. */
static int test_fit(const char *path, int state_mode) {
    static uint8_t extra[4u << 20] __attribute__((aligned(16)));
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    char error[192], reason[160];
    int tokens[64], reference[64] = {0}, got[64];
    size_t region, blocks_bytes = 0, full;
    int requested = 384, step, count;
    if (!falcon_model_open(&model, path, error, sizeof(error))) {
        printf("fit: FAIL model: %s\n", error);
        return 0;
    }
    region = model.file_size - model.weights_offset;
    {
        size_t block = 256u * 1024u, off;
        for (off = sizeof(extra); off < region; off += block)
            blocks_bytes += (region - off < block ? region - off : block) +
                            16u + 2u * sizeof(void *);
    }
    model.state_mode = state_mode;
    /* Exactly the runtime, every block and the fit headroom (16 KiB). */
    full = falcon_runtime_plan_bytes(&model, requested) + blocks_bytes +
           16u * 1024u;
    count = falcon_build_chat_prompt(&model, "What is 2+2?", tokens, 64);
    falcon_model_close(&model);
    for (step = 0; step < 5; ++step) {
        size_t budget = 0;
        int expect_context = requested, expect_holes = 0, granted, holes;
        int use_fit = 1;
        if (!falcon_model_open(&model, path, error, sizeof(error))) return 0;
        model.state_mode = state_mode;
        if (step == 1) budget = full;
        if (step == 2 || step == 4) {
            budget = full - (falcon_runtime_plan_bytes(&model, requested) -
                             falcon_runtime_plan_bytes(&model, requested - 5));
            expect_context = step == 2 ? requested - 5 : requested;
            expect_holes = step == 4 ? -1 : 0;
            use_fit = step == 2;
        }
        if (step == 3) {
            budget = full - (3u << 20);
            expect_holes = -1;
        }
        model.config.context = requested;
        falcon_set_heap_budget(budget);
        if (!falcon_runtime_init_fit(&runtime, &model, FALCON_MIN_CONTEXT, 16,
                                     error, sizeof(error))) {
            printf("fit: FAIL runtime: %s\n", error);
            return 0;
        }
        falcon_model_cache_weights_extra(&model, 64u << 20, extra,
                                         sizeof(extra), 0, error,
                                         sizeof(error));
        granted = use_fit ?
            falcon_fit_context(&model, &runtime, requested / 4,
                               FALCON_MIN_CONTEXT, reason, sizeof(reason)) :
            model.config.context;
        if (!use_fit) snprintf(reason, sizeof(reason), "policy off");
        holes = falcon_model_uncached_blocks(&model);
        printf("fit[%d] budget %lu: granted %d, %d blocks uncached (%s)\n",
               state_mode, (unsigned long)budget, granted, holes, reason);
        if (granted != expect_context || model.config.context != granted ||
            (expect_holes == 0 ? holes != 0 : holes == 0)) {
            printf("fit: FAIL expected context %d, %s\n", expect_context,
                   expect_holes ? "holes" : "no holes");
            return 0;
        }
        got[0] = 0;
        falcon_sampler_init(&sampler, 0.0f, 1.0f, 1, 1.0f, 1);
        if (!falcon_generate(&model, &runtime, &sampler, tokens, count, 24,
                             collect, got, NULL)) {
            printf("fit: FAIL generation\n");
            return 0;
        }
        if (step == 0) memcpy(reference, got, sizeof(got));
        else if (memcmp(reference, got, sizeof(got))) {
            printf("fit: FAIL tokens differ from the uncapped run\n");
            return 0;
        }
        falcon_runtime_free(&runtime);
        falcon_model_close(&model);
        falcon_set_heap_budget(0);
    }
    return 1;
}

static int append(const FalconModel *model, const char *text, int *tokens,
                  int count, int capacity) {
    int made;
    if (count < 0) return -1;
    made = falcon_tokenize(&model->tokenizer, text, tokens + count,
                           capacity - count);
    return made < 0 ? -1 : count + made;
}

int main(int argc, char **argv) {
    static const char *prompts[] = {
        "hi", "Tell me about the moon.", "What is 2+2?",
        "Write a short poem about the sea.", "Why is the sky blue?",
    };
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    char error[192];
    size_t sizes[CANDIDATES];
    int contexts[CANDIDATES], i, n;
    int tokens[512];
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.fhq4\n", argv[0]);
        return 2;
    }
    if (!falcon_model_open(&model, argv[1], error, sizeof(error))) {
        printf("backoff: FAIL model: %s\n", error);
        return 1;
    }
    for (n = 0; n < CANDIDATES; ++n) {
        contexts[n] = 512 - n * STEP;
        sizes[n] = size_at(&model, contexts[n]);
    }
    printf("runtime bytes: ctx 512 %lu, ctx 128 %lu (%.0f bytes/token)\n",
           (unsigned long)sizes[0], (unsigned long)sizes[CANDIDATES - 1],
           (double)(sizes[0] - sizes[CANDIDATES - 1]) / (512 - 128));
    {
        /* Budgets: everything, between candidates, exact fits, just below
         * the minimum. */
        size_t budgets[8];
        budgets[0] = 0;                                   /* no cap */
        budgets[1] = sizes[0];                            /* exact 512 */
        budgets[2] = sizes[0] - 1;                        /* -> 496 */
        budgets[3] = (sizes[0] + sizes[CANDIDATES - 1]) / 2;
        budgets[4] = sizes[CANDIDATES - 2] + 100;         /* -> 144 */
        budgets[5] = sizes[CANDIDATES - 1];               /* exact 128 */
        budgets[6] = sizes[CANDIDATES - 1] - 1;           /* nothing fits */
        budgets[7] = 1;
        for (i = 0; i < 8; ++i) {
            int expected = 0, granted;
            for (n = 0; n < CANDIDATES; ++n)
                if (!budgets[i] || sizes[n] <= budgets[i]) {
                    expected = contexts[n];
                    break;
                }
            model.config.context = 512;
            falcon_runtime_set_alloc_limit(budgets[i]);
            granted = falcon_runtime_init_fit(&runtime, &model,
                                              FALCON_MIN_CONTEXT, STEP,
                                              error, sizeof(error));
            printf("budget %lu: requested 512, granted %d (expected %d), "
                   "config.context %d%s%s\n", (unsigned long)budgets[i],
                   granted, expected, model.config.context,
                   granted ? "" : ", error: ", granted ? "" : error);
            if (granted != expected ||
                model.config.context != (granted ? granted : 512)) {
                printf("backoff: FAIL\n");
                return 1;
            }
            if (granted) falcon_runtime_free(&runtime);
        }
    }

    /* A conversation at a backed-off context, run to the limit. */
    {
        int budget_index = (int)(CANDIDATES * 3 / 4); /* about 224 */
        int granted, position = 0, pending = -1, turn = 0, full = 0;
        model.config.context = 512;
        falcon_runtime_set_alloc_limit(sizes[budget_index] + 64);
        granted = falcon_runtime_init_fit(&runtime, &model,
                                          FALCON_MIN_CONTEXT, STEP,
                                          error, sizeof(error));
        falcon_runtime_set_alloc_limit(0);
        printf("chat: requested 512, granted %d\n", granted);
        if (granted != contexts[budget_index]) {
            printf("backoff: FAIL wrong grant\n");
            return 1;
        }
        falcon_model_cache_weights(&model, 64u * 1024u * 1024u, error,
                                   sizeof(error));
        falcon_sampler_init(&sampler, 0.0f, 0.9f, 32, 1.08f, 1);
        while (!full) {
            const char *prompt = prompts[turn % 5];
            int count = 0, max_new, generated = 0, next = 0, ok;
            if (position == 0) {
                count = falcon_build_chat_prompt(&model, prompt, tokens, 512);
            } else {
                if (pending >= 0) tokens[count++] = pending;
                tokens[count++] = model.config.im_end_id;
                count = append(&model, "\n", tokens, count, 512);
                if (count > 0) tokens[count++] = model.config.im_start_id;
                count = append(&model, "user\n", tokens, count, 512);
                count = append(&model, prompt, tokens, count, 512);
                if (count > 0) tokens[count++] = model.config.im_end_id;
                count = append(&model, "\n", tokens, count, 512);
                if (count > 0) tokens[count++] = model.config.im_start_id;
                count = append(&model, "assistant\n", tokens, count, 512);
            }
            max_new = model.config.context - position - count;
            if (max_new > 128) max_new = 128;
            if (count <= 0 || max_new < 8) {
                printf("chat: full at position %d of %d after %d turns "
                       "(next turn needs %d tokens)\n", position,
                       model.config.context, turn, count);
                full = 1;
                break;
            }
            ok = falcon_generate_turn(&model, &runtime, &sampler, tokens,
                                      count, position, max_new, NULL, NULL,
                                      &generated, &next, &pending);
            printf("chat: turn %d at %d, %d prompt + %d reply tokens -> %d "
                   "(%s)\n", turn, position, count, generated, next,
                   ok ? "ok" : "FAILED");
            if (!ok || next > model.config.context) {
                printf("backoff: FAIL\n");
                return 1;
            }
            position = next;
            ++turn;
        }
        /* One position past the granted context must be refused, not
         * crash. */
        if (falcon_forward(&model, &runtime, 0, model.config.context)) {
            printf("backoff: FAIL forward past the context accepted\n");
            return 1;
        }
        falcon_runtime_free(&runtime);
    }
    /* The plan must match the allocation for the float16 state too. */
    model.state_mode = FALCON_STATE_F16;
    size_at(&model, 384);
    size_at(&model, 379);
    falcon_model_close(&model);
    if (!test_fit(argv[1], FALCON_STATE_F16) ||
        !test_fit(argv[1], FALCON_STATE_F32))
        return 1;
    printf("fit: ok\n");
    printf("backoff: ok\n");
    return 0;
}
