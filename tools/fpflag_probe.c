/* Host-only probe: run the chat scenario and report, per token position and
 * layer, when the FPU first raises underflow/overflow/invalid flags.
 * A PSP (MIPS) FPU raises an unmaskable exception where x86 merely sets a
 * flag, so the first flagged (position, layer) here predicts the exact
 * on-device crash location. */
#include "falcon_h1.h"

#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int last_position = -1, last_layer = -1;
static int reported = 0;

static void progress(int position, int layer, int total_layers, void *user) {
    int raised = fetestexcept(FE_UNDERFLOW | FE_OVERFLOW |
                              FE_INVALID | FE_DIVBYZERO);
    (void)user;
    if (raised && reported < 60) {
        printf("flags%s%s%s%s raised during pos %d (screen token %d) layer %d "
               "(checked at pos %d layer %d/%d)\n",
               (raised & FE_UNDERFLOW) ? " UNDERFLOW" : "",
               (raised & FE_OVERFLOW) ? " OVERFLOW" : "",
               (raised & FE_INVALID) ? " INVALID" : "",
               (raised & FE_DIVBYZERO) ? " DIVBYZERO" : "",
               last_position, last_position + 1, last_layer,
               position, layer, total_layers);
        ++reported;
    }
    feclearexcept(FE_ALL_EXCEPT);
    last_position = position;
    last_layer = layer;
}

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime rt;
    char error[256];
    int tokens[512];
    int count, pos, i, token = 0;
    float *logits;
    if (argc < 2) { fprintf(stderr, "usage: %s model [prompt]\n", argv[0]); return 2; }
    if (!falcon_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "model: %s\n", error); return 1;
    }
    if (!falcon_runtime_init(&rt, &model, error, sizeof(error))) {
        fprintf(stderr, "runtime: %s\n", error); return 1;
    }
    count = falcon_build_chat_prompt(&model, argc > 2 ? argv[2] : "hi",
                                     tokens, 512);
    fprintf(stderr, "prompt tokens: %d\n", count);
    falcon_runtime_set_progress(&rt, progress, NULL);
    falcon_runtime_reset(&rt, &model);
    feclearexcept(FE_ALL_EXCEPT);
    for (pos = 0; pos < count + 12; ++pos) {
        int feed = pos < count ? tokens[pos] : token;
        logits = falcon_forward(&model, &rt, feed, pos);
        if (!logits) { fprintf(stderr, "forward failed\n"); return 1; }
        token = 0;
        for (i = 1; i < model.config.vocab_size; ++i)
            if (logits[i] > logits[token]) token = i;
    }
    falcon_runtime_free(&rt);
    falcon_model_close(&model);
    return 0;
}
