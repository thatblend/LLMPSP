/* Host test: the O(vocab + history) repetition-penalty lookup in
 * src/falcon_sampler.c must choose exactly the token the previous
 * O(vocab * history) implementation chose, for greedy and top-k/top-p
 * sampling, on random logits with many ties, random and repeated
 * history ids, history longer than the 64-entry window, and ids outside
 * the vocabulary. The previous implementation is reproduced verbatim
 * below as the reference. Prints "sampler: ok" or the first mismatch. */
#include "falcon_h1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- reference: falcon_sampler.c before the change ---------------- */
typedef struct {
    float logit;
    int id;
} RefCandidate;

static uint32_t ref_random_u32(FalconSampler *sampler) {
    uint64_t x = sampler->rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    sampler->rng = x;
    return (uint32_t)((x * 0x2545f4914f6cdd1dULL) >> 32);
}
static float ref_random_f32(FalconSampler *sampler) {
    return (float)(ref_random_u32(sampler) >> 8) / 16777216.0f;
}
static int ref_in_recent_history(int id, const int *history, int count) {
    int begin = count > 64 ? count - 64 : 0;
    int i;
    for (i = begin; i < count; ++i)
        if (history[i] == id) return 1;
    return 0;
}
static float ref_penalized(const FalconSampler *sampler, float value, int id,
                           const int *history, int history_count) {
    if (ref_in_recent_history(id, history, history_count))
        return value < 0.0f ? value * sampler->repetition_penalty :
                              value / sampler->repetition_penalty;
    return value;
}
static int ref_sample(FalconSampler *sampler, const float *logits,
                      int vocab_size, const int *history, int history_count) {
    RefCandidate top[FALCON_TOP_K_MAX];
    int count = 0, i;
    if (sampler->temperature <= 0.0f) {
        int best = 0;
        float best_value = ref_penalized(sampler, logits[0], 0,
                                         history, history_count);
        for (i = 1; i < vocab_size; ++i) {
            float value = ref_penalized(sampler, logits[i], i,
                                        history, history_count);
            if (value > best_value) {
                best_value = value;
                best = i;
            }
        }
        return best;
    }
    for (i = 0; i < vocab_size; ++i) {
        float value = ref_penalized(sampler, logits[i], i,
                                    history, history_count) /
                      sampler->temperature;
        int position;
        if (count == sampler->top_k && value <= top[count - 1].logit) continue;
        position = count < sampler->top_k ? count++ : count - 1;
        while (position > 0 && top[position - 1].logit < value) {
            top[position] = top[position - 1];
            --position;
        }
        top[position].logit = value;
        top[position].id = i;
    }
    if (count <= 0) return 0;
    {
        float probabilities[FALCON_TOP_K_MAX];
        float maximum = top[0].logit, total = 0.0f, cumulative = 0.0f;
        int keep = count;
        for (i = 0; i < count; ++i) {
            probabilities[i] = expf(top[i].logit - maximum);
            total += probabilities[i];
        }
        if (sampler->top_p < 1.0f) {
            for (i = 0; i < count; ++i) {
                cumulative += probabilities[i] / total;
                if (cumulative >= sampler->top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        total = 0.0f;
        for (i = 0; i < keep; ++i) total += probabilities[i];
        {
            float coin = ref_random_f32(sampler) * total;
            cumulative = 0.0f;
            for (i = 0; i < keep; ++i) {
                cumulative += probabilities[i];
                if (coin < cumulative) return top[i].id;
            }
        }
        return top[keep - 1].id;
    }
}
/* ------------------------------------------------------------------- */

static uint64_t state = 0x853c49e6748fea9bULL;
static uint32_t next_u32(void) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(state >> 33);
}

int main(void) {
    static const int vocabs[] = { 1, 7, 1000, 32768, 40000 };
    static float logits[40000];
    static int history[400];
    int trial, checked = 0, greedy = 0;
    for (trial = 0; trial < 6000; ++trial) {
        int vocab = vocabs[next_u32() % 5u];
        int count = (int)(next_u32() % 300u);
        int mode = (int)(next_u32() % 4u), i, a, b;
        int hot = 1 + (int)(next_u32() % 50u);
        FalconSampler s1, s2;
        float temperature = (trial & 1) ? 0.0f :
                            0.2f + (float)(next_u32() % 100u) / 50.0f;
        float penalty = 1.0f + (float)(next_u32() % 5u) * 0.25f;
        float top_p = (next_u32() % 3u) ? 0.9f : 1.0f;
        int top_k = 1 + (int)(next_u32() % FALCON_TOP_K_MAX);
        uint64_t seed = ((uint64_t)next_u32() << 32) | next_u32();
        for (i = 0; i < vocab; ++i) {
            if (mode == 0)          /* very many exact ties */
                logits[i] = (float)((int)(next_u32() % 9u) - 4);
            else if (mode == 1)     /* ties at the top only */
                logits[i] = (next_u32() % 7u) ? -3.0f : 2.5f;
            else if (mode == 2)     /* everything equal, incl. signed zero */
                logits[i] = (next_u32() & 1u) ? 0.0f : -0.0f;
            else
                logits[i] = (float)((int)(next_u32() % 20001u) - 10000) /
                            1000.0f;
        }
        for (i = 0; i < count; ++i) {
            uint32_t r = next_u32() % 100u;
            if (r < 70)       /* small id range: repeats */
                history[i] = (int)(next_u32() % (uint32_t)hot);
            else if (r < 90)  /* anywhere in the vocabulary */
                history[i] = (int)(next_u32() % (uint32_t)vocab);
            else if (r < 95)  /* out of range */
                history[i] = vocab + (int)(next_u32() % 100u);
            else
                history[i] = -1 - (int)(next_u32() % 5u);
        }
        /* Penalise the current maxima sometimes, to move ties around. */
        for (a = 0, b = 0; a < vocab && b < 3 && b < count; ++a)
            if (logits[a] == 2.5f && (next_u32() & 1u))
                history[count - 1 - b++] = a;
        falcon_sampler_init(&s1, temperature, top_p, top_k, penalty, seed);
        falcon_sampler_init(&s2, temperature, top_p, top_k, penalty, seed);
        {
            int expected = ref_sample(&s1, logits, vocab, history, count);
            int got = falcon_sample(&s2, logits, vocab, history, count);
            /* A second call checks that the bitmap was cleared. */
            int expected2 = ref_sample(&s1, logits, vocab, history,
                                       count / 2);
            int got2 = falcon_sample(&s2, logits, vocab, history, count / 2);
            if (expected != got || expected2 != got2 || s1.rng != s2.rng) {
                printf("sampler: MISMATCH trial %d vocab %d history %d "
                       "temp %.2f: %d/%d vs %d/%d\n", trial, vocab, count,
                       (double)temperature, expected, expected2, got, got2);
                return 1;
            }
        }
        ++checked;
        if (temperature <= 0.0f) ++greedy;
    }
    printf("sampler: ok (%d cases, %d greedy, %d top-k/top-p)\n",
           checked * 2, greedy * 2, (checked - greedy) * 2);
    return 0;
}
