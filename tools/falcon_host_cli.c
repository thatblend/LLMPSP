#include "falcon_h1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int count;
} ConsoleOutput;

static int print_token(int token, const uint8_t *piece,
                       size_t piece_len, void *user) {
    ConsoleOutput *output = (ConsoleOutput *)user;
    (void)token;
    if (piece && piece_len) fwrite(piece, 1, piece_len, stdout);
    fflush(stdout);
    ++output->count;
    return 1;
}

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    ConsoleOutput output = {0};
    char error[256];
    int *tokens;
    int token_count, max_new = 16, generated = 0, stripe = 0;
    size_t cache_bytes = 0;
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.fhq4 \"prompt\" [max-new] [cache-MiB]"
                " [stripe]\n", argv[0]);
        return 2;
    }
    if (argc > 3) max_new = atoi(argv[3]);
    if (argc > 4) cache_bytes = (size_t)atoi(argv[4]) * 1024u * 1024u;
    if (argc > 5) stripe = atoi(argv[5]) != 0;
    if (!falcon_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "model: %s\n", error);
        return 1;
    }
    if (!falcon_runtime_init(&runtime, &model, error, sizeof(error))) {
        fprintf(stderr, "runtime: %s\n", error);
        falcon_model_close(&model);
        return 1;
    }
    if (cache_bytes)
        falcon_model_cache_weights_extra(&model, cache_bytes, NULL, 0,
                                         stripe, error, sizeof(error));
    tokens = (int *)malloc((size_t)model.config.context * sizeof(int));
    if (!tokens) {
        fprintf(stderr, "token allocation failed\n");
        falcon_runtime_free(&runtime);
        falcon_model_close(&model);
        return 1;
    }
    token_count = falcon_build_chat_prompt(&model, argv[2], tokens,
                                           model.config.context);
    if (token_count <= 0 || token_count + max_new > model.config.context) {
        fprintf(stderr, "prompt tokenization failed or context exceeded\n");
        free(tokens);
        falcon_runtime_free(&runtime);
        falcon_model_close(&model);
        return 1;
    }
    fprintf(stderr, "prompt tokens: %d; runtime: %.2f MiB; cached: %.2f MiB\n",
            token_count,
            (double)falcon_runtime_resident_bytes(&runtime) / 1048576.0,
            (double)model.weight_cache_bytes / 1048576.0);
    falcon_sampler_init(&sampler, 0.0f, 1.0f, 32, 1.0f,
                        (uint64_t)time(NULL));
    if (!falcon_generate(&model, &runtime, &sampler, tokens, token_count,
                         max_new, print_token, &output, &generated)) {
        fprintf(stderr, "\ngeneration failed (I/O=%d)\n", runtime.io_error);
        free(tokens);
        falcon_runtime_free(&runtime);
        falcon_model_close(&model);
        return 1;
    }
    fputc('\n', stdout);
    fprintf(stderr, "generated: %d\n", generated);
    free(tokens);
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    return 0;
}
