#ifndef FALCON_H1_H
#define FALCON_H1_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FALCON_MAGIC 0x34514846u
/* Version 3 stores every Q4 matrix in the four-row "tile" layout the PSP
 * VFPU kernel reads, with one 8-bit scale code per 32-weight block instead
 * of an fp16 scale (17 bytes per 32 weights; PROJECT.md sections 2.10 and
 * 8.1). Versions 1 and 2 are rejected and must be re-packed with
 * tools/repack_fhq4.py. */
#define FALCON_VERSION 3u
#define FALCON_HEADER_BYTES 256u
#define FALCON_MAX_LAYERS 24
#define FALCON_QK 32
#define FALCON_TOP_K_MAX 64
/* Rows per tile group; every Q4 matrix row count is a multiple of it. */
#define FALCON_GROUP_ROWS 4
/* Smallest context the runtime may back off to when RAM is short. */
#define FALCON_MIN_CONTEXT 128
/* Largest context the app accepts. The model itself was trained far
 * beyond this; the limit is RAM (about 6.5 KiB of KV cache per token),
 * and anything above ~384 leaves weights streaming from the card. */
#define FALCON_MAX_CONTEXT 1024
/* Q4 matrices: the embedding plus nine per layer. */
#define FALCON_Q4_MATRICES (1 + 9 * FALCON_MAX_LAYERS)

/* Mamba recurrent state storage (FalconModel.state_mode, read by
 * falcon_runtime_init). FALCON_STATE_F16 is the shipped float16 state
 * (section 2.10): round to nearest float16, values below the smallest
 * normal float16 flushed to zero, the PSP's vf2h/vh2f behaviour made
 * explicit. The other float16 modes are host-only experiments kept for
 * tools/quality_eval.c: plain hardware vf2h (truncation toward zero) and
 * truncation after scaling by 1 + 2^-11. */
#define FALCON_HAS_STATE_F16 1
#define FALCON_STATE_F32 0
#define FALCON_STATE_F16 1
#define FALCON_STATE_F16_TRUNC 2
#define FALCON_STATE_F16_TRUNC_CORR 3

typedef struct {
    int dim, hidden_dim, n_layers;
    int n_heads, n_kv_heads, head_dim;
    int vocab_size, model_context, context;
    int pad_id, eos_id, bos_id;
    int im_start_id, im_end_id, eom_id, eot_id;
    int mamba_dim, mamba_heads, mamba_head_dim;
    int mamba_state, mamba_groups, mamba_conv, mamba_projection;
    float rope_theta, rms_eps, embedding_multiplier, lm_head_multiplier;
} FalconConfig;

typedef struct {
    uint32_t input_norm;
    uint32_t wq, wk, wv, wo;
    uint32_t mamba_in;
    uint32_t conv_weight, conv_bias, dt_bias, a_log, d;
    uint32_t mamba_out;
    uint32_t ffn_norm, w_gate, w_up, w_down;
} FalconLayerOffsets;

typedef struct {
    uint8_t *section;
    size_t section_bytes;
    int vocab_size, max_piece_bytes;
    const uint8_t *offsets;
    const uint8_t *pieces;
    const uint8_t *sorted_ids;
    const uint8_t *ranks;
    const uint8_t *byte_ids;
} FalconTokenizer;

/* Small per-layer float tensors preloaded to RAM at model open (about
 * 0.54 MiB total). Keeping them resident removes all small seek-reads
 * from the token loop, which would otherwise thrash the stdio buffer. */
typedef struct {
    const float *input_norm, *ffn_norm;
    const float *conv_weight, *conv_bias;
    const float *dt_bias, *a_log, *d;
} FalconLayerSmall;

/* Decoding parameters of one Q4 matrix's 8-bit block-scale codes (FHQ4
 * version 3): code c (signed byte) means 0 for c == 0, otherwise
 * sign(c) * float_from_bits((|c| + bias) << (23 - mant_bits)). */
typedef struct {
    uint32_t offset;      /* absolute file offset of the matrix */
    uint16_t bias;
    uint8_t mant_bits;    /* 3, 4 or 5 */
    uint8_t reserved;
} FalconQ4Param;

typedef struct {
    FILE *file;
    /* Host builds: the 64 KiB stdio buffer. PSP builds with raw streaming
     * (see stream_*): the raw allocation behind stream_buffer, and stdio
     * runs unbuffered. Either way io_buffer_bytes is its resident size. */
    uint8_t *io_buffer;
    size_t io_buffer_bytes;
    /* PSP only: raw sceIo descriptor (opened with the model) and a 64-byte
     * aligned window [stream_start, stream_end) of the file, refilled with
     * large reads at 512-byte aligned offsets, that serves uncached weight
     * groups. The window is allocated by falcon_model_stream_prepare only
     * when the weight cache left blocks on the card; stream_ready says it
     * exists. */
    int stream_ready;
    int stream_fd;
    uint8_t *stream_buffer;
    size_t stream_capacity;
    uint32_t stream_start, stream_end;
    uint8_t **weight_cache;
    size_t weight_cache_bytes;
    size_t weight_cache_block_bytes;
    int weight_cache_blocks;
    /* Foreign cache backing (for example PSP volatile memory): block
     * pointers inside this range are not heap memory and are not freed. */
    uint8_t *cache_extra_base;
    size_t cache_extra_bytes;
    /* Set when the cache fill stopped because the heap ran out (rather
     * than at its target); only then may the fit policy trade context
     * for cache. */
    int cache_ram_limited;
    uint32_t weights_offset;
    uint32_t embedding;
    FalconLayerOffsets layers[FALCON_MAX_LAYERS];
    uint32_t final_norm;
    uint32_t file_size;
    float *small_block;
    size_t small_bytes;
    FalconLayerSmall small[FALCON_MAX_LAYERS];
    const float *final_norm_values;
    FalconQ4Param q4[FALCON_Q4_MATRICES];
    int q4_count;
    /* FALCON_STATE_* chosen by the frontend before falcon_runtime_init,
     * like config.context. */
    int state_mode;
    FalconConfig config;
    FalconTokenizer tokenizer;
} FalconModel;

typedef void (*FalconProgressCallback)(int position, int layer,
                                       int total_layers, void *user);

/* Optional monotonic microsecond clock supplied by the frontend; when set
 * the runtime accumulates the perf counters below. */
typedef unsigned int (*FalconClockCallback)(void);

#define FALCON_ROPE_HALF 32

typedef struct {
    float *x, *norm, *q, *k, *v, *attention, *attention_out;
    float *projection, *conv_out, *mamba_out;
    float *ff_gate, *ff_up, *temporary, *logits;
    /* Mamba recurrent state: float32 in mamba_state, or float16 bit
     * patterns in mamba_state16 (state_mode != FALCON_STATE_F32); the
     * other pointer is NULL. */
    float *mamba_state, *conv_state;
    uint16_t *mamba_state16;
    int state_mode;
    int8_t *key_cache, *value_cache;
    float *key_scales, *value_scales;
    uint8_t *row_buffer;
    size_t row_buffer_bytes, bytes_allocated;
    /* Token history for the repetition penalty, sized for the whole
     * context at init so generation never allocates. */
    int *history;
    int history_capacity;
    FalconProgressCallback progress;
    void *progress_user;
    int io_error;
    /* RoPE tables: frequencies are position-independent; the cos/sin pair
     * is cached per token position and shared by every layer and head. */
    float rope_freq[FALCON_ROPE_HALF];
    float rope_cos[FALCON_ROPE_HALF];
    float rope_sin[FALCON_ROPE_HALF];
    int rope_position;
    /* Perf counters, accumulated only when clock_us is set. The frontend
     * zeroes them before a turn and reads them afterwards. */
    FalconClockCallback clock_us;
    unsigned int forward_us;   /* total time inside falcon_forward */
    unsigned int matvec_us;    /* time inside q4_matvec (compute + I/O) */
    size_t streamed_bytes;     /* weight bytes read from file, not cache */
    /* PSP alignment guard: tile groups that arrived at an address that
     * is not 16-byte aligned and were copied before the VFPU kernel ran.
     * Must stay 0; anything else means a layout or allocator bug. */
    size_t realigned_groups;
} FalconRuntime;

typedef struct {
    float temperature, top_p, repetition_penalty;
    int top_k;
    uint64_t rng;
} FalconSampler;

typedef int (*FalconTokenCallback)(int token, const uint8_t *piece,
                                   size_t piece_len, void *user);

int falcon_model_open(FalconModel *model, const char *path,
                      char *error, size_t error_size);
size_t falcon_model_cache_weights(FalconModel *model, size_t target_bytes,
                                  char *error, size_t error_size);
/* Like falcon_model_cache_weights, but first fills cache blocks from the
 * caller-provided memory region (for example the PSP's 4 MiB volatile
 * partition), then allocates up to target_bytes more from the heap. The
 * extra region must stay valid until falcon_model_close and is never
 * freed by the runtime. Pass NULL/0 for plain heap behavior.
 *
 * stripe 0 caches a contiguous prefix (embedding, then layers in order).
 * stripe 1 caches the embedding, then spreads the uncached blocks evenly
 * across the layer region (about one 256 KiB hole per layer) so the
 * asynchronous prefetcher can hide each hole behind that layer's
 * compute instead of facing one large burst at the end of the token.
 * Cached bytes and streamed bytes are identical either way. */
size_t falcon_model_cache_weights_extra(FalconModel *model,
                                        size_t target_bytes,
                                        uint8_t *extra, size_t extra_bytes,
                                        int stripe,
                                        char *error, size_t error_size);
/* Allocates and reads every block the cache left on the card, as far as
 * the heap allows (the last block of the region at its real size).
 * Returns the number of blocks still uncached. */
int falcon_model_cache_fill_holes(FalconModel *model,
                                  char *error, size_t error_size);
/* Number of 256 KiB weight-cache blocks not in RAM (all of them when no
 * cache was set up). */
int falcon_model_uncached_blocks(const FalconModel *model);
/* PSP: allocates the raw read window if any weight block is uncached
 * (nothing streams otherwise, so the 128 KiB stay free). If the heap is
 * too full it gives one cached heap block back first. Host builds and a
 * fully cached model: no-op. Returns 1 if the window is ready or not
 * needed; 0 means uncached groups use the (slow, correct) stdio path. */
int falcon_model_stream_prepare(FalconModel *model);
void falcon_model_close(FalconModel *model);
int falcon_runtime_init(FalconRuntime *runtime, const FalconModel *model,
                        char *error, size_t error_size);
/* Exactly the bytes falcon_runtime_init would allocate (as counted in
 * bytes_allocated) for this model at the given context and its
 * state_mode. */
size_t falcon_runtime_plan_bytes(const FalconModel *model, int context);
/* Startup "fit" policy (PROJECT.md section 2.10). Call after the weight
 * cache is filled. If blocks are left uncached and shrinking the context
 * by at most max_reduction tokens (never below minimum_context) frees
 * enough RAM to cache them all, the runtime is re-initialised at the
 * largest such context (1-token granularity) and every block is cached;
 * otherwise the runtime is restored at its current context and the
 * remainder streams as before. Never allocates once it returns. Returns
 * the granted context (model->config.context); 0 only if the runtime could
 * not be restored (fatal, error filled). reason gets a short trace text. */
int falcon_fit_context(FalconModel *model, FalconRuntime *runtime,
                       int max_reduction, int minimum_context,
                       char *reason, size_t reason_size);
/* falcon_runtime_init with graceful back-off: tries model->config.context
 * first, then smaller contexts in steps of `step` tokens down to
 * minimum_context, and keeps the largest that fits. On success it stores
 * the granted context in model->config.context and returns it; it
 * returns 0 (error filled) only if minimum_context does not fit. */
int falcon_runtime_init_fit(FalconRuntime *runtime, FalconModel *model,
                            int minimum_context, int step,
                            char *error, size_t error_size);
/* Test hook: caps the bytes falcon_runtime_init may allocate (0 = no
 * cap), so the context back-off can be exercised on the host. */
void falcon_runtime_set_alloc_limit(size_t bytes);
/* Test hook: when nonzero, runtime arrays, weight-cache blocks and the fit
 * policy's headroom probe all draw from one budget of this many bytes
 * (resets the count; set it before anything it should cover is
 * allocated). 0 = off. */
void falcon_set_heap_budget(size_t bytes);
void falcon_runtime_reset(FalconRuntime *runtime, const FalconModel *model);
void falcon_runtime_set_progress(FalconRuntime *runtime,
                                 FalconProgressCallback callback,
                                 void *user);
void falcon_runtime_free(FalconRuntime *runtime);
float *falcon_forward(FalconModel *model, FalconRuntime *runtime,
                      int token, int position);

int falcon_tokenize(const FalconTokenizer *tokenizer, const char *text,
                    int *tokens, int max_tokens);
int falcon_build_chat_prompt(const FalconModel *model, const char *prompt,
                             int *tokens, int max_tokens);
const uint8_t *falcon_token_piece(const FalconTokenizer *tokenizer,
                                  int token, size_t *piece_len);

void falcon_sampler_init(FalconSampler *sampler, float temperature,
                         float top_p, int top_k,
                         float repetition_penalty, uint64_t seed);
int falcon_sample(FalconSampler *sampler, const float *logits, int vocab_size,
                  const int *history, int history_count);
int falcon_generate(FalconModel *model, FalconRuntime *runtime,
                    FalconSampler *sampler, const int *prompt_tokens,
                    int prompt_count, int max_new_tokens,
                    FalconTokenCallback callback, void *user,
                    int *generated_count);
/* Multi-turn variant. start_position 0 resets the recurrent/KV state and
 * behaves like falcon_generate; a nonzero start_position continues an
 * existing conversation, feeding prompt_tokens at that position without
 * touching accumulated state. On success *next_position is the position
 * for the next turn's tokens and *pending_token is an emitted-but-not-
 * yet-processed token (-1 if none) that the caller must prepend to the
 * next turn's token list. */
int falcon_generate_turn(FalconModel *model, FalconRuntime *runtime,
                         FalconSampler *sampler, const int *prompt_tokens,
                         int prompt_count, int start_position,
                         int max_new_tokens,
                         FalconTokenCallback callback, void *user,
                         int *generated_count, int *next_position,
                         int *pending_token);
size_t falcon_model_resident_bytes(const FalconModel *model);
size_t falcon_runtime_resident_bytes(const FalconRuntime *runtime);

/* Asynchronous weight prefetcher (real implementation on the PSP only;
 * host builds get inert stubs). Open before filling the weight cache so
 * the double buffer can still be allocated, then call configure once the
 * cache is filled: it scans the cache index for uncached extents and
 * streams exactly those. Stats report how much it actually served and
 * how long q4_matvec had to block waiting for reads (the un-overlapped
 * remainder); alive is 0 after an I/O error disabled the module. */
int falcon_prefetch_open(const char *path);
void falcon_prefetch_configure(const FalconModel *model);
void falcon_prefetch_token_begin(void);
void falcon_prefetch_poll(void);
const uint8_t *falcon_prefetch_get(uint32_t offset, uint32_t bytes);
void falcon_prefetch_stats(unsigned int *served_kib, unsigned int *wait_ms,
                           int *alive);
void falcon_prefetch_stats_reset(void);
void falcon_prefetch_close(void);

#ifdef __cplusplus
}
#endif
#endif

