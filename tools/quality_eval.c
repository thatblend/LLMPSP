/* Host quality gate for numeric changes (PROJECT.md section 2.10).
 *
 * The reference is the v1.3 host build with the FHQ4 version-2 model. A
 * reference run teacher-forces nothing: it builds 17 fixed conversations
 * (short chat, factual, instructions, arithmetic, a multi-turn one and a
 * ~300-token one), extends each with 64 greedy tokens (plain argmax, stop
 * tokens ignored so every prompt yields the same number of positions) and
 * dumps the token sequence plus the log-softmax of the next-token
 * distribution at up to 80 positions (the last 16 prompt positions, or all of
 * a shorter prompt, and the 64 reply positions). A candidate run feeds exactly those tokens through
 * another build/model/configuration and reports, against the dump:
 *
 *   top-1   argmax agreement over all compared positions
 *   KL      mean and maximum KL(reference || candidate) in nats
 *   greedy  how many leading reply tokens free-running greedy decoding
 *           would reproduce (equal to the teacher-forced count, because the
 *           prefixes are identical up to the first disagreement), per
 *           prompt, and how many prompts keep their whole natural reply
 *           (up to the reference's first stop token)
 *
 * The same source builds against the v1.3 sources (reference) and the
 * current ones; features that only exist in the current runtime are under
 * FALCON_HAS_STATE_F16.
 *
 *   quality_eval ref   MODEL DUMP
 *   quality_eval cand  MODEL DUMP [STATE_MODE]
 *   quality_eval state MODEL [REPLY_TOKENS] [STATE_MODE]
 *
 * "state" reports the range of the float32 Mamba recurrent state over
 * long greedy generations (every prompt, REPLY_TOKENS more tokens,
 * default 400): maximum magnitude, the share of nonzero values below the
 * smallest normal float16 (2^-14), and percentiles of log2|s|.
 *
 * STATE_MODE (current runtime only): 0 float32 state, otherwise the
 * float16 state with FALCON_STATE_* rounding mode STATE_MODE. */
#include "falcon_h1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPLY 64
#define PROMPT_TAIL 16
#define MAX_TOKENS 512
#define DUMP_MAGIC 0x31455146u /* "FQE1" */

typedef struct {
    const char *user[3];
    const char *assistant[2];
} Conversation;

static const char long_text[] =
    "Read the following text and answer the question after it.\n\n"
    "The first machines that people would recognise as bicycles appeared in "
    "Europe in the early nineteenth century. In 1817 the German inventor Karl "
    "Drais built a two-wheeled running machine with a wooden frame and no "
    "pedals: the rider sat on a saddle and pushed against the ground with "
    "both feet. It was popular for a short time in Germany, France and "
    "Britain. About forty years later French craftsmen attached "
    "cranks and pedals directly to the front wheel, creating the velocipede, "
    "which was nicknamed the boneshaker because of its iron tyres. To go "
    "faster, makers enlarged the front wheel, and by the 1870s the high "
    "wheeler, with a front wheel up to one and a half metres across, was the "
    "fastest machine on the road, although falls were common and dangerous. "
    "The safety bicycle of the 1880s solved this problem with two wheels of "
    "similar size and a chain that drove the rear wheel. Soon afterwards "
    "John Boyd Dunlop's air-filled tyre made riding far more comfortable, "
    "and bicycles became cheap enough for workers, students and women, who "
    "gained a new kind of freedom to travel on their own.\n\n"
    "Question: Why was the safety bicycle safer than the high wheeler, and "
    "what invention made riding more comfortable soon afterwards?";

static const Conversation conversations[] = {
    {{"hi"}, {0}},
    {{"What is the capital of France?"}, {0}},
    {{"What is 17 + 25?"}, {0}},
    {{"If I have 3 apples and buy 5 more, how many apples do I have?"}, {0}},
    {{"Write a short poem about the sea."}, {0}},
    {{"Explain how a rainbow forms."}, {0}},
    {{"List three tips for staying healthy."}, {0}},
    {{"Translate 'good morning' into Spanish."}, {0}},
    {{"Who wrote Romeo and Juliet?"}, {0}},
    {{"Give me step-by-step instructions to boil an egg."}, {0}},
    {{"Tell me a joke."}, {0}},
    {{"What is the difference between a virus and a bacterium?"}, {0}},
    {{"Summarize the story of Cinderella in two sentences."}, {0}},
    {{"Write a Python function that returns the sum of two numbers."}, {0}},
    {{"Why is the sky blue?"}, {0}},
    {{"My name is Sam and I like astronomy.",
      "Mostly the planets. Which planet is the largest one?"},
     {"Nice to meet you, Sam! Astronomy is a wonderful hobby. What do you "
      "enjoy observing the most?"}},
    {{long_text}, {0}},
};
#define CONVERSATIONS ((int)(sizeof(conversations) / sizeof(conversations[0])))

static int append_text(const FalconModel *model, const char *text,
                       int *tokens, int count) {
    int made;
    if (count < 0) return -1;
    made = falcon_tokenize(&model->tokenizer, text, tokens + count,
                           MAX_TOKENS - count);
    return made < 0 ? -1 : count + made;
}

/* The official template: <|im_start|>user\n...<|im_end|>\n per turn,
 * ending with <|im_start|>assistant\n. */
static int build_conversation(const FalconModel *model, const Conversation *c,
                              int *tokens) {
    int count = 0, turn;
    for (turn = 0; turn < 3 && c->user[turn]; ++turn) {
        if (turn > 0) {
            tokens[count++] = model->config.im_start_id;
            count = append_text(model, "assistant\n", tokens, count);
            count = append_text(model, c->assistant[turn - 1], tokens, count);
            if (count < 0) return -1;
            tokens[count++] = model->config.im_end_id;
            count = append_text(model, "\n", tokens, count);
            if (count < 0) return -1;
        }
        tokens[count++] = model->config.im_start_id;
        count = append_text(model, "user\n", tokens, count);
        count = append_text(model, c->user[turn], tokens, count);
        if (count < 0) return -1;
        tokens[count++] = model->config.im_end_id;
        count = append_text(model, "\n", tokens, count);
        if (count < 0) return -1;
    }
    tokens[count++] = model->config.im_start_id;
    count = append_text(model, "assistant\n", tokens, count);
    return count;
}

/* Positions compared: the last PROMPT_TAIL prompt positions (all of a
 * shorter prompt) and every reply position. */
static int eval_start(int prompt) {
    return prompt > PROMPT_TAIL ? prompt - PROMPT_TAIL : 0;
}

static int argmax(const float *values, int count) {
    int best = 0, i;
    for (i = 1; i < count; ++i)
        if (values[i] > values[best]) best = i;
    return best;
}

static void log_softmax(const float *logits, int count, double *out) {
    double maximum = logits[0], sum = 0.0;
    int i;
    for (i = 1; i < count; ++i)
        if (logits[i] > maximum) maximum = logits[i];
    for (i = 0; i < count; ++i) sum += exp((double)logits[i] - maximum);
    sum = log(sum) + maximum;
    for (i = 0; i < count; ++i) out[i] = (double)logits[i] - sum;
}

static int is_stop(const FalconModel *model, int token) {
    const FalconConfig *c = &model->config;
    return token == c->eos_id || token == c->im_end_id ||
           token == c->eom_id || token == c->eot_id;
}

/* ---- float32 Mamba state statistics -------------------------------- */

typedef struct {
    double maximum;
    unsigned long long values, nonzero, below_half_normal, below_2m24;
    unsigned long long histogram[64]; /* log2|s| + 40, clamped */
    double layer_max[FALCON_MAX_LAYERS];
} StateStats;

static void state_scan(const FalconModel *model, const FalconRuntime *r,
                       StateStats *s) {
    const FalconConfig *c = &model->config;
    size_t per_layer = (size_t)c->mamba_dim * c->mamba_state, i;
    int layer;
#ifdef FALCON_HAS_STATE_F16
    if (!r->mamba_state) return;
#endif
    for (layer = 0; layer < c->n_layers; ++layer) {
        const float *p = r->mamba_state + (size_t)layer * per_layer;
        for (i = 0; i < per_layer; ++i) {
            double a = fabs((double)p[i]);
            ++s->values;
            if (a > s->maximum) s->maximum = a;
            if (a > s->layer_max[layer]) s->layer_max[layer] = a;
            if (a == 0.0) continue;
            ++s->nonzero;
            if (a < 6.103515625e-05) ++s->below_half_normal;
            if (a < 5.9604644775390625e-08) ++s->below_2m24;
            {
                int bin = (int)floor(log2(a)) + 40;
                if (bin < 0) bin = 0;
                if (bin > 63) bin = 63;
                ++s->histogram[bin];
            }
        }
    }
}

static void state_report(const StateStats *s, int layers) {
    unsigned long long running = 0;
    int bin, layer;
    printf("state: %llu samples, max |s| %.4g (float16 max 65504), "
           "%.3f%% nonzero values below 2^-14, %.3f%% below 2^-24\n",
           s->values, s->maximum,
           s->nonzero ? 100.0 * (double)s->below_half_normal /
                            (double)s->nonzero : 0.0,
           s->nonzero ? 100.0 * (double)s->below_2m24 / (double)s->nonzero
                      : 0.0);
    printf("state: log2|s| histogram (nonzero):");
    for (bin = 0; bin < 64; ++bin) {
        running += s->histogram[bin];
        if (s->histogram[bin])
            printf(" [%d]%.2f%%", bin - 40,
                   100.0 * (double)s->histogram[bin] / (double)s->nonzero);
    }
    printf("\nstate: per-layer max |s|:");
    for (layer = 0; layer < layers; ++layer)
        printf(" %.3g", s->layer_max[layer]);
    printf("\n");
    (void)running;
}

/* -------------------------------------------------------------------- */

/* The state mode is a model setting read by falcon_runtime_init. */
static int set_state_mode(FalconModel *model, int mode) {
#ifdef FALCON_HAS_STATE_F16
    model->state_mode = mode;
    return 1;
#else
    (void)model;
    if (mode) {
        fprintf(stderr, "this build has no float16 state\n");
        return 0;
    }
    return 1;
#endif
}

static int open_all(FalconModel *model, FalconRuntime *runtime,
                    const char *path, int state_mode) {
    char error[256];
    if (!falcon_model_open(model, path, error, sizeof(error))) {
        fprintf(stderr, "model: %s\n", error);
        return 0;
    }
    if (!set_state_mode(model, state_mode)) return 0;
    if (!falcon_runtime_init(runtime, model, error, sizeof(error))) {
        fprintf(stderr, "runtime: %s\n", error);
        return 0;
    }
    /* Whole model in RAM: cached and streamed reads give identical
     * results, and streaming through a slow file system dominates. */
    falcon_model_cache_weights(model, (size_t)64 << 20, error, sizeof(error));
    return 1;
}

static int run_reference(const char *model_path, const char *dump_path) {
    FalconModel model;
    FalconRuntime runtime;
    FILE *dump;
    static int tokens[MAX_TOKENS];
    static double logp[32768];
    static float logp32[32768];
    int conv;
    if (!open_all(&model, &runtime, model_path, 0)) return 1;
    dump = fopen(dump_path, "wb");
    if (!dump) { perror(dump_path); return 1; }
    {
        unsigned header[3] = {DUMP_MAGIC, (unsigned)CONVERSATIONS,
                              (unsigned)model.config.vocab_size};
        fwrite(header, sizeof(header), 1, dump);
    }
    for (conv = 0; conv < CONVERSATIONS; ++conv) {
        int prompt = build_conversation(&model, &conversations[conv], tokens);
        int total = prompt + REPLY, position, natural = REPLY;
        unsigned meta[3];
        long logits_at;
        if (prompt < 2 || total > model.config.context) {
            fprintf(stderr, "conversation %d: bad length %d\n", conv, prompt);
            return 1;
        }
        meta[0] = (unsigned)prompt;
        meta[1] = (unsigned)total;
        meta[2] = 0; /* natural reply length, patched below */
        fwrite(meta, sizeof(meta), 1, dump);
        fwrite(tokens, sizeof(int), (size_t)total, dump); /* patched below */
        logits_at = ftell(dump);
        falcon_runtime_reset(&runtime, &model);
        for (position = 0; position < total; ++position) {
            float *logits = falcon_forward(&model, &runtime, tokens[position],
                                           position);
            int i;
            if (!logits) { fprintf(stderr, "forward failed\n"); return 1; }
            if (position + 1 >= prompt && position + 1 < total) {
                int next = argmax(logits, model.config.vocab_size);
                tokens[position + 1] = next;
                if (natural == REPLY && is_stop(&model, next))
                    natural = position + 1 - prompt;
            }
            if (position >= eval_start(prompt)) {
                log_softmax(logits, model.config.vocab_size, logp);
                for (i = 0; i < model.config.vocab_size; ++i)
                    logp32[i] = (float)logp[i];
                fwrite(logp32, sizeof(float),
                       (size_t)model.config.vocab_size, dump);
            }
        }
        {
            long end = ftell(dump);
            meta[2] = (unsigned)natural;
            fseek(dump, logits_at - (long)(sizeof(meta) +
                                           sizeof(int) * (size_t)total),
                  SEEK_SET);
            fwrite(meta, sizeof(meta), 1, dump);
            fwrite(tokens, sizeof(int), (size_t)total, dump);
            fseek(dump, end, SEEK_SET);
        }
        printf("ref %2d: %3d prompt tokens, natural reply %2d tokens\n", conv,
               prompt, natural);
    }
    fclose(dump);
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    return 0;
}

static int run_candidate(const char *model_path, const char *dump_path,
                         int state_mode) {
    FalconModel model;
    FalconRuntime runtime;
    FILE *dump;
    static int tokens[MAX_TOKENS];
    static float ref32[32768];
    static double cand[32768];
    unsigned header[3];
    int conv, vocab;
    long positions = 0, agree = 0;
    double kl_sum = 0.0, kl_max = 0.0;
    int greedy_sum = 0, natural_kept = 0, full_kept = 0;
    if (!open_all(&model, &runtime, model_path, state_mode)) return 1;
    dump = fopen(dump_path, "rb");
    if (!dump || fread(header, sizeof(header), 1, dump) != 1 ||
        header[0] != DUMP_MAGIC || header[1] != (unsigned)CONVERSATIONS ||
        header[2] != (unsigned)model.config.vocab_size) {
        fprintf(stderr, "bad reference dump %s\n", dump_path);
        return 1;
    }
    vocab = model.config.vocab_size;
    for (conv = 0; conv < CONVERSATIONS; ++conv) {
        unsigned meta[3];
        int prompt, total, natural, position, greedy = -1;
        double conv_kl = 0.0, conv_kl_max = 0.0;
        int conv_agree = 0;
        if (fread(meta, sizeof(meta), 1, dump) != 1) return 1;
        prompt = (int)meta[0];
        total = (int)meta[1];
        natural = (int)meta[2];
        if (total > MAX_TOKENS ||
            fread(tokens, sizeof(int), (size_t)total, dump) != (size_t)total)
            return 1;
        falcon_runtime_reset(&runtime, &model);
        for (position = 0; position < total; ++position) {
            float *logits = falcon_forward(&model, &runtime, tokens[position],
                                           position);
            if (!logits) { fprintf(stderr, "forward failed\n"); return 1; }
            if (position >= prompt - 1 && position + 1 < total &&
                greedy < 0 &&
                argmax(logits, vocab) != tokens[position + 1])
                greedy = position + 1 - prompt;
            if (position >= eval_start(prompt)) {
                double kl = 0.0;
                int i, ref_top = 0;
                if (fread(ref32, sizeof(float), (size_t)vocab, dump) !=
                    (size_t)vocab)
                    return 1;
                log_softmax(logits, vocab, cand);
                for (i = 0; i < vocab; ++i) {
                    double p = exp((double)ref32[i]);
                    kl += p * ((double)ref32[i] - cand[i]);
                    if (ref32[i] > ref32[ref_top]) ref_top = i;
                }
                if (kl < 0.0) kl = 0.0;
                conv_kl += kl;
                if (kl > conv_kl_max) conv_kl_max = kl;
                if (argmax(logits, vocab) == ref_top) ++conv_agree;
            }
        }
        if (greedy < 0) greedy = REPLY;
        greedy_sum += greedy;
        if (greedy >= natural) ++natural_kept;
        if (greedy >= REPLY) ++full_kept;
        positions += total - eval_start(prompt);
        agree += conv_agree;
        kl_sum += conv_kl;
        if (conv_kl_max > kl_max) kl_max = conv_kl_max;
        printf("cand %2d: top-1 %2d/%d, KL mean %.2e max %.2e, greedy %2d/%d "
               "(natural %d)\n", conv, conv_agree, total - eval_start(prompt),
               conv_kl / (total - eval_start(prompt)),
               conv_kl_max, greedy, REPLY, natural);
    }
    printf("summary: top-1 %.2f%% (%ld/%ld), KL mean %.3e max %.3e, greedy "
           "identical %.1f/%d tokens on average, natural reply kept %d/%d, "
           "all %d kept %d/%d\n",
           100.0 * (double)agree / (double)positions, agree, positions,
           kl_sum / (double)positions, kl_max,
           (double)greedy_sum / CONVERSATIONS, REPLY, natural_kept,
           CONVERSATIONS, REPLY, full_kept, CONVERSATIONS);
    fclose(dump);
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    return 0;
}

static int run_state(const char *model_path, int reply, int state_mode) {
    FalconModel model;
    FalconRuntime runtime;
    static int tokens[MAX_TOKENS];
    StateStats stats;
    int conv;
    memset(&stats, 0, sizeof(stats));
    if (!open_all(&model, &runtime, model_path, state_mode)) return 1;
    for (conv = 0; conv < CONVERSATIONS; ++conv) {
        int prompt = build_conversation(&model, &conversations[conv], tokens);
        int total = prompt + reply, position;
        if (total > model.config.context) total = model.config.context;
        falcon_runtime_reset(&runtime, &model);
        for (position = 0; position < total; ++position) {
            float *logits = falcon_forward(&model, &runtime, tokens[position],
                                           position);
            if (!logits) { fprintf(stderr, "forward failed\n"); return 1; }
            if (position + 1 >= prompt && position + 1 < total)
                tokens[position + 1] = argmax(logits,
                                              model.config.vocab_size);
            state_scan(&model, &runtime, &stats);
        }
    }
    state_report(&stats, model.config.n_layers);
    falcon_runtime_free(&runtime);
    falcon_model_close(&model);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "ref"))
        return run_reference(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "cand"))
        return run_candidate(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 0);
    if (argc >= 3 && !strcmp(argv[1], "state"))
        return run_state(argv[2], argc > 3 ? atoi(argv[3]) : 400,
                         argc > 4 ? atoi(argv[4]) : 0);
    fprintf(stderr,
            "usage: %s ref MODEL DUMP | cand MODEL DUMP [STATE_MODE] | "
            "state MODEL [REPLY_TOKENS] [STATE_MODE]\n", argv[0]);
    return 2;
}
