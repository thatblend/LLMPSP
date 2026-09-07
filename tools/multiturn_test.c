/* Host test: a second conversation turn generated incrementally with
 * falcon_generate_turn must produce exactly the same tokens as feeding
 * the whole two-turn conversation from scratch. Greedy sampling with
 * repetition penalty 1.0 so history windows cannot differ. */
#include "falcon_h1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int ids[512];
    int count;
} TokenList;

static int collect_token(int token, const uint8_t *piece,
                         size_t piece_len, void *user) {
    TokenList *list = (TokenList *)user;
    (void)piece; (void)piece_len;
    if (list->count < 512) list->ids[list->count++] = token;
    return 1;
}

static int append_text(const FalconModel *model, const char *text,
                       int *tokens, int count) {
    int made;
    if (count < 0) return -1;
    made = falcon_tokenize(&model->tokenizer, text, tokens + count,
                           512 - count);
    return made < 0 ? -1 : count + made;
}

int main(int argc, char **argv) {
    FalconModel model;
    FalconRuntime runtime;
    FalconSampler sampler;
    char error[256];
    int turn1[512], turn2[512], full[512];
    int turn1_count, turn2_count, full_count;
    int generated, next_position, pending, i, ok;
    TokenList reply1, reply2_inc, reply2_full;
    const char *user1 = "hi", *user2 = "what can you do";
    if (argc < 2) { fprintf(stderr, "usage: %s model\n", argv[0]); return 2; }
    if (!falcon_model_open(&model, argv[1], error, sizeof(error)) ||
        !falcon_runtime_init(&runtime, &model, error, sizeof(error))) {
        fprintf(stderr, "setup: %s\n", error);
        return 1;
    }
    memset(&reply1, 0, sizeof(reply1));
    memset(&reply2_inc, 0, sizeof(reply2_inc));
    memset(&reply2_full, 0, sizeof(reply2_full));

    /* Turn 1 (fresh). */
    turn1_count = falcon_build_chat_prompt(&model, user1, turn1, 512);
    falcon_sampler_init(&sampler, 0.0f, 1.0f, 1, 1.0f, 1);
    ok = falcon_generate_turn(&model, &runtime, &sampler, turn1, turn1_count,
                              0, 32, collect_token, &reply1, &generated,
                              &next_position, &pending);
    if (!ok || generated <= 0) { fprintf(stderr, "turn1 failed\n"); return 1; }
    printf("turn1: %d prompt, %d reply, next %d, pending %d\n",
           turn1_count, reply1.count, next_position, pending);

    /* Turn 2 incrementally (exactly the PSP frontend's delta). */
    turn2_count = 0;
    if (pending >= 0) turn2[turn2_count++] = pending;
    turn2[turn2_count++] = model.config.im_end_id;
    turn2_count = append_text(&model, "\n", turn2, turn2_count);
    turn2[turn2_count++] = model.config.im_start_id;
    turn2_count = append_text(&model, "user\n", turn2, turn2_count);
    turn2_count = append_text(&model, user2, turn2, turn2_count);
    turn2[turn2_count++] = model.config.im_end_id;
    turn2_count = append_text(&model, "\n", turn2, turn2_count);
    turn2[turn2_count++] = model.config.im_start_id;
    turn2_count = append_text(&model, "assistant\n", turn2, turn2_count);
    if (turn2_count <= 0) { fprintf(stderr, "turn2 build failed\n"); return 1; }
    falcon_sampler_init(&sampler, 0.0f, 1.0f, 1, 1.0f, 1);
    ok = falcon_generate_turn(&model, &runtime, &sampler, turn2, turn2_count,
                              next_position, 32, collect_token, &reply2_inc,
                              &generated, &next_position, &pending);
    if (!ok) { fprintf(stderr, "turn2 failed\n"); return 1; }
    printf("turn2 incremental: %d delta tokens, %d reply tokens\n",
           turn2_count, reply2_inc.count);

    /* Full re-evaluation: turn1 tokens + emitted reply1 + turn2 delta.
     * reply1 tokens the model actually saw are all emitted ones; the
     * pending token is already the first entry of turn2. */
    full_count = 0;
    for (i = 0; i < turn1_count; ++i) full[full_count++] = turn1[i];
    {
        /* Append only the reply1 tokens that were fed to the model; if a
         * pending token exists it is already the first entry of turn2. */
        int fed_reply = reply1.count - (pending >= 0 ? 1 : 0);
        for (i = 0; i < fed_reply; ++i) full[full_count++] = reply1.ids[i];
    }
    for (i = 0; i < turn2_count; ++i) full[full_count++] = turn2[i];
    falcon_sampler_init(&sampler, 0.0f, 1.0f, 1, 1.0f, 1);
    ok = falcon_generate(&model, &runtime, &sampler, full, full_count,
                         32, collect_token, &reply2_full, &generated);
    if (!ok) { fprintf(stderr, "full re-eval failed\n"); return 1; }
    printf("turn2 full re-eval: %d prompt tokens, %d reply tokens\n",
           full_count, reply2_full.count);

    if (reply2_inc.count != reply2_full.count ||
        memcmp(reply2_inc.ids, reply2_full.ids,
               (size_t)reply2_inc.count * sizeof(int)) != 0) {
        printf("MISMATCH\n");
        for (i = 0; i < reply2_inc.count || i < reply2_full.count; ++i)
            printf("  %d: inc %d full %d\n", i,
                   i < reply2_inc.count ? reply2_inc.ids[i] : -1,
                   i < reply2_full.count ? reply2_full.ids[i] : -1);
        return 1;
    }
    printf("multiturn: ok (%d matching reply tokens)\n", reply2_inc.count);
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    return 0;
}
