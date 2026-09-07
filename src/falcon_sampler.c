#include "falcon_h1.h"

#include <math.h>

typedef struct {
    float logit;
    int id;
} Candidate;

static uint32_t random_u32(FalconSampler *sampler) {
    uint64_t x = sampler->rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    sampler->rng = x;
    return (uint32_t)((x * 0x2545f4914f6cdd1dULL) >> 32);
}
static float random_f32(FalconSampler *sampler) {
    return (float)(random_u32(sampler) >> 8) / 16777216.0f;
}

void falcon_sampler_init(FalconSampler *sampler, float temperature,
                         float top_p, int top_k,
                         float repetition_penalty, uint64_t seed) {
    sampler->temperature = temperature < 0.0f ? 0.0f : temperature;
    sampler->top_p = top_p <= 0.0f || top_p > 1.0f ? 1.0f : top_p;
    sampler->top_k = top_k < 1 ? 1 :
                     (top_k > FALCON_TOP_K_MAX ? FALCON_TOP_K_MAX : top_k);
    sampler->repetition_penalty = repetition_penalty < 1.0f ?
                                  1.0f : repetition_penalty;
    sampler->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
}

static int in_recent_history(int id, const int *history, int count) {
    int begin = count > 64 ? count - 64 : 0;
    int i;
    for (i = begin; i < count; ++i)
        if (history[i] == id) return 1;
    return 0;
}
static float penalized(const FalconSampler *sampler, float value, int id,
                       const int *history, int history_count) {
    if (in_recent_history(id, history, history_count))
        return value < 0.0f ? value * sampler->repetition_penalty :
                              value / sampler->repetition_penalty;
    return value;
}

int falcon_sample(FalconSampler *sampler, const float *logits, int vocab_size,
                  const int *history, int history_count) {
    Candidate top[FALCON_TOP_K_MAX];
    int count = 0, i;
    if (sampler->temperature <= 0.0f) {
        int best = 0;
        float best_value = penalized(sampler, logits[0], 0,
                                     history, history_count);
        for (i = 1; i < vocab_size; ++i) {
            float value = penalized(sampler, logits[i], i,
                                    history, history_count);
            if (value > best_value) {
                best_value = value;
                best = i;
            }
        }
        return best;
    }
    for (i = 0; i < vocab_size; ++i) {
        float value = penalized(sampler, logits[i], i,
                                history, history_count) / sampler->temperature;
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
            float coin = random_f32(sampler) * total;
            cumulative = 0.0f;
            for (i = 0; i < keep; ++i) {
                cumulative += probabilities[i];
                if (coin < cumulative) return top[i].id;
            }
        }
        return top[keep - 1].id;
    }
}

