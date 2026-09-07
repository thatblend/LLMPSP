#include "falcon_h1.h"

#include <stdlib.h>
#include <string.h>

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

const uint8_t *falcon_token_piece(const FalconTokenizer *tokenizer,
                                  int token, size_t *piece_len) {
    uint32_t begin, end;
    if (piece_len) *piece_len = 0;
    if (!tokenizer || token < 0 || token >= tokenizer->vocab_size) return NULL;
    begin = rd32(tokenizer->offsets + token * 4);
    end = rd32(tokenizer->offsets + (token + 1) * 4);
    if (piece_len) *piece_len = end - begin;
    return tokenizer->pieces + begin;
}

static int piece_compare(const uint8_t *left, size_t left_len,
                         const uint8_t *right, size_t right_len) {
    size_t common = left_len < right_len ? left_len : right_len;
    int result = memcmp(left, right, common);
    if (result) return result;
    return left_len < right_len ? -1 : left_len > right_len;
}

static int find_token(const FalconTokenizer *tokenizer,
                      const uint8_t *piece, size_t length) {
    int low = 0, high = tokenizer->vocab_size - 1;
    while (low <= high) {
        int middle = low + (high - low) / 2;
        int id = rd16(tokenizer->sorted_ids + middle * 2);
        size_t candidate_len;
        const uint8_t *candidate = falcon_token_piece(tokenizer, id, &candidate_len);
        int comparison = piece_compare(piece, length, candidate, candidate_len);
        if (comparison == 0) return id;
        if (comparison < 0) high = middle - 1;
        else low = middle + 1;
    }
    return -1;
}

static int bpe_chunk(const FalconTokenizer *tokenizer,
                     const uint8_t *text, int length,
                     int *output, int capacity) {
    int *work;
    uint8_t combined[512];
    int count = length, i;
    if (length <= 0) return 0;
    if (length > capacity) return -1;
    work = (int *)malloc((size_t)length * sizeof(int));
    if (!work) return -1;
    for (i = 0; i < length; ++i)
        work[i] = rd16(tokenizer->byte_ids + text[i] * 2);
    while (count > 1) {
        uint32_t best_rank = 0xffffffffu;
        int best_index = -1, best_id = -1;
        for (i = 0; i < count - 1; ++i) {
            size_t left_len, right_len;
            const uint8_t *left = falcon_token_piece(tokenizer, work[i], &left_len);
            const uint8_t *right = falcon_token_piece(tokenizer, work[i + 1], &right_len);
            int merged;
            uint32_t rank;
            if (left_len + right_len > (size_t)tokenizer->max_piece_bytes ||
                left_len + right_len > sizeof(combined))
                continue;
            memcpy(combined, left, left_len);
            memcpy(combined + left_len, right, right_len);
            merged = find_token(tokenizer, combined, left_len + right_len);
            if (merged < 0) continue;
            rank = rd32(tokenizer->ranks + merged * 4);
            if (rank < best_rank) {
                best_rank = rank;
                best_index = i;
                best_id = merged;
            }
        }
        if (best_index < 0) break;
        work[best_index] = best_id;
        memmove(work + best_index + 1, work + best_index + 2,
                (size_t)(count - best_index - 2) * sizeof(int));
        --count;
    }
    memcpy(output, work, (size_t)count * sizeof(int));
    free(work);
    return count;
}

static int ascii_letter(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static int ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}
static int horizontal_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\v' || c == '\f';
}
static int newline(uint8_t c) {
    return c == '\r' || c == '\n';
}
static int punctuation(uint8_t c) {
    return c < 128 && !ascii_letter(c) && !ascii_digit(c) &&
           !horizontal_space(c) && !newline(c);
}

static int special_at(const FalconTokenizer *tokenizer, const uint8_t *text,
                      int remaining, int *special_length) {
    static const char *names[] = {
        "<|im_start|>", "<|im_end|>", "<|end_of_text|>",
        "<|begin_of_text|>", "<|eom_id|>", "<|eot_id|>"
    };
    int i;
    for (i = 0; i < (int)(sizeof(names) / sizeof(names[0])); ++i) {
        int length = (int)strlen(names[i]);
        if (remaining >= length && memcmp(text, names[i], (size_t)length) == 0) {
            int id = find_token(tokenizer, (const uint8_t *)names[i], (size_t)length);
            if (id >= 0) {
                *special_length = length;
                return id;
            }
        }
    }
    return -1;
}

int falcon_tokenize(const FalconTokenizer *tokenizer, const char *text,
                    int *tokens, int max_tokens) {
    const uint8_t *bytes = (const uint8_t *)text;
    int length = (int)strlen(text);
    int at = 0, count = 0;
    if (!tokenizer || !text || !tokens || max_tokens <= 0) return -1;
    while (at < length) {
        int begin = at, end, produced, special_length = 0;
        int special = special_at(tokenizer, bytes + at, length - at, &special_length);
        if (special >= 0) {
            if (count >= max_tokens) return -1;
            tokens[count++] = special;
            at += special_length;
            continue;
        }
        if (newline(bytes[at])) {
            while (at < length && newline(bytes[at])) ++at;
            end = at;
        } else if (horizontal_space(bytes[at])) {
            int spaces = at;
            while (at < length && horizontal_space(bytes[at])) ++at;
            if (at < length && newline(bytes[at])) {
                while (at < length && newline(bytes[at])) ++at;
                end = at;
            } else if (at < length &&
                       (ascii_letter(bytes[at]) || bytes[at] >= 128)) {
                if (at - spaces > 1) {
                    end = at - 1;
                    at = end;
                } else {
                    while (at < length &&
                           (ascii_letter(bytes[at]) || bytes[at] >= 128)) ++at;
                    end = at;
                }
            } else if (at < length && punctuation(bytes[at]) && at - spaces == 1) {
                while (at < length && punctuation(bytes[at])) ++at;
                end = at;
            } else {
                end = at;
            }
        } else if (ascii_digit(bytes[at])) {
            end = ++at;
        } else if (ascii_letter(bytes[at]) || bytes[at] >= 128) {
            while (at < length &&
                   (ascii_letter(bytes[at]) || bytes[at] >= 128)) ++at;
            end = at;
        } else {
            while (at < length && punctuation(bytes[at])) ++at;
            end = at;
        }
        produced = bpe_chunk(tokenizer, bytes + begin, end - begin,
                             tokens + count, max_tokens - count);
        if (produced < 0) return -1;
        count += produced;
    }
    return count;
}

static int append_text(const FalconTokenizer *tokenizer, const char *text,
                       int *tokens, int count, int capacity) {
    int made = falcon_tokenize(tokenizer, text, tokens + count, capacity - count);
    return made < 0 ? -1 : count + made;
}

int falcon_build_chat_prompt(const FalconModel *model, const char *prompt,
                             int *tokens, int max_tokens) {
    int count = 0;
    if (!model || !prompt || !tokens || max_tokens < 4) return -1;
    tokens[count++] = model->config.im_start_id;
    count = append_text(&model->tokenizer, "user\n", tokens, count, max_tokens);
    if (count < 0) return -1;
    count = append_text(&model->tokenizer, prompt, tokens, count, max_tokens);
    if (count < 0 || count >= max_tokens) return -1;
    tokens[count++] = model->config.im_end_id;
    count = append_text(&model->tokenizer, "\n", tokens, count, max_tokens);
    if (count < 0 || count >= max_tokens) return -1;
    tokens[count++] = model->config.im_start_id;
    return append_text(&model->tokenizer, "assistant\n",
                       tokens, count, max_tokens);
}

