#include "falcon_h1.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* PSP builds stream uncached weights with raw sceIoRead (section 2.9 of
 * PROJECT.md); host builds keep stdio. */
#if defined(_PSP_FW_VERSION) && !defined(FALCON_NO_RAWIO)
#define FALCON_RAWIO 1
#include <pspiofilemgr.h>
#include <unistd.h>
/* Raw read window for streamed weights. 128 KiB: the PSP-2000 measured
 * 16.2 MiB/s for 256 KiB reads and 14.9 for 32 KiB, so 128 KiB keeps
 * almost all of the large-read rate while every KiB not spent here is
 * weight cache (one streamed MiB costs ~100 ms per token). */
#define FALCON_STREAM_BYTES (128u * 1024u)
#endif

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}
static float rdf32(const uint8_t *p) {
    uint32_t bits = rd32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
static void set_error(char *buffer, size_t size, const char *format, ...) {
    va_list args;
    if (!buffer || !size) return;
    va_start(args, format);
    vsnprintf(buffer, size, format, args);
    va_end(args);
}
/* Q4 matrices are stored as tile groups of FALCON_GROUP_ROWS rows (FHQ4
 * version 3). A group of rows r..r+3 with B = cols / 32 blocks is
 *
 *     B * 4 bytes    8-bit scale codes: block 0 rows 0..3, block 1 rows
 *                    0..3, ... (decoded by scale_code_value)
 *     B * 64 bytes   tiles: tile t of block b is 16 bytes; its word r
 *                    belongs to row r, and byte k of that word packs
 *                    weights j = 4t + k (low nibble, w + 8) and j + 16
 *                    (high nibble, w + 8), XOR 0x80
 *
 * 17 bytes per 32 weights. src/falcon_q4_tile_psp.h has the rationale.
 * Hosts built with FALCON_EVAL_V2 (tools/quality_eval.c only) also read
 * version 2, whose groups carry B * 8 bytes of fp16 scales instead. */
#ifdef FALCON_EVAL_V2
#if defined(_PSP_FW_VERSION)
#error FALCON_EVAL_V2 is a host-only evaluation build
#endif
static int eval_v2_model;
#define SCALE_BYTES (eval_v2_model ? 2u : 1u)
#else
#define SCALE_BYTES 1u
#endif
static size_t q4_row_bytes(int cols) {
    return (size_t)(cols / FALCON_QK) * (16u + SCALE_BYTES);
}
static size_t q4_group_bytes(int cols) {
    return (size_t)FALCON_GROUP_ROWS * q4_row_bytes(cols);
}
static size_t q4_bytes(int rows, int cols) {
    return (size_t)rows * q4_row_bytes(cols);
}

/* Test hook (falcon_set_heap_budget): when nonzero, runtime arrays, cache
 * blocks and the fit policy's headroom probe all draw from this many
 * bytes, so host tests can model a console's heap. 0 = off. */
static size_t heap_budget, heap_budget_used;

void falcon_set_heap_budget(size_t bytes) {
    heap_budget = bytes;
    heap_budget_used = 0;
}
static void *budget_malloc(size_t bytes) {
    void *memory;
    if (heap_budget && heap_budget_used + bytes > heap_budget) return NULL;
    memory = malloc(bytes);
    if (memory && heap_budget) heap_budget_used += bytes;
    return memory;
}
static void budget_free(void *memory, size_t bytes) {
    if (!memory) return;
    free(memory);
    if (heap_budget)
        heap_budget_used = heap_budget_used > bytes ?
            heap_budget_used - bytes : 0;
}

/* Runtime arrays and heap cache blocks are 16-byte aligned because the PSP
 * kernels read them with lv.q (newlib's malloc only promises 8 bytes).
 * The malloc pointer and the requested size sit just before the block. */
#define ALIGNED_OVERHEAD (16u + 2u * sizeof(void *))
static void *aligned_block_alloc(size_t bytes) {
    uint8_t *raw, *aligned;
    if (bytes > (size_t)-1 - ALIGNED_OVERHEAD) return NULL;
    raw = (uint8_t *)budget_malloc(bytes + ALIGNED_OVERHEAD);
    if (!raw) return NULL;
    aligned = (uint8_t *)(((uintptr_t)raw + 2u * sizeof(void *) + 15u) &
                          ~(uintptr_t)15u);
    ((void **)aligned)[-1] = raw;
    memcpy(aligned - 2u * sizeof(void *), &bytes, sizeof(bytes));
    return aligned;
}
static void aligned_free(void *memory) {
    size_t bytes;
    if (!memory) return;
    memcpy(&bytes, (uint8_t *)memory - 2u * sizeof(void *), sizeof(bytes));
    budget_free(((void **)memory)[-1], bytes + ALIGNED_OVERHEAD);
}
static uint8_t *cache_block_alloc(size_t bytes) {
    return (uint8_t *)aligned_block_alloc(bytes);
}
static void cache_block_free(uint8_t *block) {
    aligned_free(block);
}
static int seek_read(FILE *file, uint32_t offset, void *data, size_t bytes) {
    return fseek(file, (long)offset, SEEK_SET) == 0 &&
           fread(data, 1, bytes, file) == bytes;
}

/* The weight cache is a sparse index over the whole weight region: one
 * pointer per 256 KiB block, NULL for blocks left on the card. A prefix
 * fill has no interior holes; a striped fill (see
 * falcon_model_cache_weights_extra) leaves evenly spread holes. */
static const uint8_t *cache_pointer(const FalconModel *model,
                                    uint32_t offset, size_t bytes) {
    size_t relative, region, block, inside;
    const uint8_t *base;
    if (!model->weight_cache || offset < model->weights_offset) return NULL;
    relative = (size_t)(offset - model->weights_offset);
    region = (size_t)(model->file_size - model->weights_offset);
    if (relative > region || bytes > region - relative) return NULL;
    block = relative / model->weight_cache_block_bytes;
    inside = relative % model->weight_cache_block_bytes;
    if (inside + bytes > model->weight_cache_block_bytes) return NULL;
    base = model->weight_cache[block];
    return base ? base + inside : NULL;
}

static int cache_copy(const FalconModel *model, uint32_t offset,
                      void *destination, size_t bytes) {
    size_t relative, region, copied = 0;
    uint8_t *out = (uint8_t *)destination;
    if (!model->weight_cache || offset < model->weights_offset) return 0;
    relative = (size_t)(offset - model->weights_offset);
    region = (size_t)(model->file_size - model->weights_offset);
    if (relative > region || bytes > region - relative) return 0;
    while (copied < bytes) {
        size_t position = relative + copied;
        size_t block = position / model->weight_cache_block_bytes;
        size_t inside = position % model->weight_cache_block_bytes;
        size_t available = model->weight_cache_block_bytes - inside;
        size_t amount = bytes - copied < available ? bytes - copied : available;
        const uint8_t *base = model->weight_cache[block];
        if (!base) return 0;
        memcpy(out + copied, base + inside, amount);
        copied += amount;
    }
    return 1;
}

static int model_read(FalconModel *model, uint32_t offset,
                      void *destination, size_t bytes) {
    if (cache_copy(model, offset, destination, bytes)) return 1;
    return seek_read(model->file, offset, destination, bytes);
}

#ifdef FALCON_RAWIO
/* Opens the raw descriptor. Raw sceIo paths must be device-absolute, so a
 * relative path is resolved against the newlib working directory. The
 * read window itself is allocated later by stream_window_alloc, and only
 * if anything streams. Returns 0 (stdio streaming stays in use) on
 * failure. */
static int stream_open(FalconModel *model, const char *path) {
    char absolute[512];
    SceUID fd = sceIoOpen(path, PSP_O_RDONLY, 0777);
    if (fd < 0 && !strchr(path, ':') &&
        strlen(path) + 2 < sizeof(absolute) &&
        getcwd(absolute, sizeof(absolute) - strlen(path) - 2)) {
        strcat(absolute, "/");
        strcat(absolute, path);
        fd = sceIoOpen(absolute, PSP_O_RDONLY, 0777);
    }
    if (fd < 0) return 0;
    model->stream_fd = fd;
    return 1;
}

static int stream_window_alloc(FalconModel *model) {
    uint8_t *raw;
    if (model->stream_ready) return 1;
    if (model->stream_fd < 0) return 0;
    raw = (uint8_t *)malloc(FALCON_STREAM_BYTES + 64u);
    if (!raw) return 0;
    model->io_buffer = raw;
    model->io_buffer_bytes = FALCON_STREAM_BYTES + 64u;
    model->stream_buffer = (uint8_t *)(((uintptr_t)raw + 63u) &
                                       ~(uintptr_t)63u);
    model->stream_capacity = FALCON_STREAM_BYTES;
    model->stream_start = model->stream_end = 0;
    model->stream_ready = 1;
    return 1;
}

/* End of the run of uncached cache blocks containing last_byte; the whole
 * file when nothing is cached. Reads never run into cached blocks. */
static uint32_t uncached_run_end(const FalconModel *model,
                                 uint32_t last_byte) {
    size_t block;
    if (!model->weight_cache || last_byte < model->weights_offset)
        return model->file_size;
    block = (size_t)(last_byte - model->weights_offset) /
            model->weight_cache_block_bytes;
    while (block < (size_t)model->weight_cache_blocks &&
           !model->weight_cache[block])
        ++block;
    if (block >= (size_t)model->weight_cache_blocks) return model->file_size;
    return model->weights_offset +
           (uint32_t)(block * model->weight_cache_block_bytes);
}

/* Returns the uncached bytes [offset, offset + bytes) from the read
 * window, refilling it first if needed: one sceIoRead into the 64-byte
 * aligned buffer at a 512-aligned file offset, up to the window size or
 * the next cached block. buffer[0] always holds a 512-aligned file
 * offset and every group offset is 16-aligned in the file, so the
 * returned pointer is 16-byte aligned. When a group straddles the end of
 * the window (the sequential case), the 512-byte sectors it already has
 * move to the front instead of being read again, so the next read still
 * starts 512-aligned, lands 64-aligned, and no byte is read twice. NULL
 * on an I/O error. */
static const uint8_t *stream_group(FalconModel *model, FalconRuntime *runtime,
                                   uint32_t offset, size_t bytes) {
    uint32_t need = offset + (uint32_t)bytes;
    uint32_t start, end, limit, keep = 0;
    if (offset >= model->stream_start && need <= model->stream_end)
        return model->stream_buffer + (offset - model->stream_start);
    start = offset & ~511u;
    if (offset >= model->stream_start && offset < model->stream_end &&
        !(model->stream_end & 511u)) {
        keep = model->stream_end - start;
        memmove(model->stream_buffer,
                model->stream_buffer + (start - model->stream_start), keep);
    }
    end = uncached_run_end(model, need - 1u);
    if (end < need) end = need;
    end = (end + 511u) & ~511u;
    limit = start + (uint32_t)model->stream_capacity;
    if (end > limit) end = limit;
    if (end > model->file_size) end = model->file_size;
    model->stream_start = model->stream_end = 0;
    if (need > end ||
        sceIoLseek32(model->stream_fd, (int)(start + keep), PSP_SEEK_SET) !=
            (int)(start + keep) ||
        sceIoRead(model->stream_fd, model->stream_buffer + keep,
                  end - start - keep) != (int)(end - start - keep))
        return NULL;
    model->stream_start = start;
    model->stream_end = end;
    runtime->streamed_bytes += end - start - keep;
    return model->stream_buffer + (offset - start);
}
#endif


/* Decoded value of one 8-bit block-scale code (FHQ4 version 3). The bit
 * pattern is built with integer operations only, so every build decodes
 * exactly the same float; the PSP kernel (falcon_q4_tile_psp.h) produces
 * the same bits with vi2f/vabs/vadd/vf2iz/vsgn. The loader guarantees
 * every magnitude is a normal float. */
static float scale_code_value(int8_t code, const FalconQ4Param *param) {
    uint32_t magnitude = (uint32_t)(code < 0 ? -(int)code : (int)code);
    uint32_t bits;
    float value;
    if (!code) return 0.0f;
    bits = (magnitude + param->bias) << (23 - param->mant_bits);
    if (code < 0) bits |= 0x80000000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Whether every code -128..127 decodes to a normal, finite float. */
static int scale_param_valid(unsigned mant_bits, unsigned bias) {
    return mant_bits >= 3 && mant_bits <= 5 &&
           ((1u + bias) >> mant_bits) >= 1 &&
           ((128u + bias) >> mant_bits) <= 254;
}

/* The parameters of the Q4 matrix starting at offset (matrices are stored
 * in increasing offset order). */
static const FalconQ4Param *q4_param(const FalconModel *model,
                                     uint32_t offset) {
    int low = 0, high = model->q4_count - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (model->q4[mid].offset == offset) return &model->q4[mid];
        if (model->q4[mid].offset < offset) low = mid + 1;
        else high = mid - 1;
    }
    return NULL;
}

#ifdef FALCON_EVAL_V2
static float half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    float result;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    memcpy(&result, &bits, sizeof(result));
    return result;
}
#endif

/* Scale of row r, block b of a tile group. */
static float group_scale(const uint8_t *group, int r, int block,
                         const FalconQ4Param *param) {
#ifdef FALCON_EVAL_V2
    if (eval_v2_model) return half_to_float(rd16(group + block * 8 + r * 2));
#endif
    return scale_code_value((int8_t)group[block * 4 + r], param);
}

/* Per-byte nibble lookup tables (2 KiB, resident in the data cache).
 * They replace the unpack + integer-to-float conversion in the hot
 * kernels, which are among the slowest scalar operations on the PSP's
 * Allegrex FPU. Values are exact small integers, so results are
 * bit-identical to the arithmetic form. */
static float q4_lut_low[256], q4_lut_high[256];
static int q4_lut_ready;

static void q4_lut_init(void) {
    int i;
    if (q4_lut_ready) return;
    for (i = 0; i < 256; ++i) {
        q4_lut_low[i] = (float)((i & 15) - 8);
        q4_lut_high[i] = (float)((i >> 4) - 8);
    }
    q4_lut_ready = 1;
}

/* Picks row r (0..3) out of a tile group (embedding lookup). */
static void dequantize_row(float *out, const uint8_t *group, int r, int cols,
                           const FalconQ4Param *param) {
    int blocks = cols / FALCON_QK, block, j;
    const uint8_t *tiles = group + (size_t)blocks * 4u * SCALE_BYTES;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *t = tiles + (size_t)block * 64u + (size_t)r * 4u;
        float d = group_scale(group, r, block, param);
        int base = block * FALCON_QK;
        for (j = 0; j < 16; ++j) {
            uint8_t packed = (uint8_t)(t[(j >> 2) * 16 + (j & 3)] ^ 0x80u);
            out[base + j] = d * q4_lut_low[packed];
            out[base + j + 16] = d * q4_lut_high[packed];
        }
    }
}

/* On the PSP the matvecs run on the VFPU four rows at a time
 * (falcon_q4_tile4_vtfm, src/falcon_q4_psp.c). The kernel needs per-matvec
 * activation tables, built once and amortized over every group: per
 * 32-column block, four tiles of {16 * x_lo quad, x_hi - 16 * x_lo quad},
 * plus the sum of the block's 16 low-nibble columns for the -8 offset. */
#if defined(_PSP_FW_VERSION) && !defined(FALCON_SCALAR_ONLY)
#define FALCON_VFPU 1
/* One kernel per scale-code mantissa width (3, 4, 5 bits): the width is
 * the immediate of the kernel's vi2f. bias_quad holds four copies of
 * bias * 2^(23 - mant_bits) (16-byte aligned). */
typedef void (*FalconQ4Kernel)(const uint8_t *group, const float *xt,
                               const float *block_sums, int blocks,
                               float *out, const float *bias_quad);
extern void falcon_q4_tile4_m3(const uint8_t *group, const float *xt,
                               const float *block_sums, int blocks,
                               float *out, const float *bias_quad);
extern void falcon_q4_tile4_m4(const uint8_t *group, const float *xt,
                               const float *block_sums, int blocks,
                               float *out, const float *bias_quad);
extern void falcon_q4_tile4_m5(const uint8_t *group, const float *xt,
                               const float *block_sums, int blocks,
                               float *out, const float *bias_quad);
extern void falcon_vfpu_attn_score(const float *q, const int8_t *keys,
                                   int stride, int count, float *scores);
extern void falcon_vfpu_attn_accum(float *out, const int8_t *values,
                                   int stride, int count,
                                   const float *weights);
extern void falcon_vfpu_mamba_head(float *state, const float *hidden,
                                   const float *b, const float *cc,
                                   const float *decay_dt, float *acc);
extern void falcon_vfpu_mamba_head_f16(uint16_t *state, const float *hidden,
                                       const float *b, const float *cc,
                                       const float *decay_dt, float *acc);
#endif
#if defined(_PSP_FW_VERSION) && !defined(FALCON_NO_PREFETCH)
#define FALCON_PREFETCH 1
#endif

#define FALCON_MAX_BLOCKS 64

#ifdef FALCON_VFPU
static float q4_xt[FALCON_MAX_BLOCKS * FALCON_QK] __attribute__((aligned(16)));
static float q4_low_sums[FALCON_MAX_BLOCKS] __attribute__((aligned(16)));
static float q4_out4[FALCON_GROUP_ROWS] __attribute__((aligned(16)));
static float q4_bias[4] __attribute__((aligned(16)));
static FalconQ4Kernel q4_kernel;

/* Selects the kernel and its bias quad for one matrix. */
static void q4_prepare_scales(const FalconQ4Param *param) {
    float bias = (float)((uint32_t)param->bias << (23 - param->mant_bits));
    q4_bias[0] = q4_bias[1] = q4_bias[2] = q4_bias[3] = bias;
    q4_kernel = param->mant_bits == 3 ? falcon_q4_tile4_m3 :
                param->mant_bits == 5 ? falcon_q4_tile4_m5 :
                                        falcon_q4_tile4_m4;
}

static void q4_prepare_x(const float *x, int blocks) {
    int block, t, k, j;
    for (block = 0; block < blocks; ++block) {
        const float *v = x + block * FALCON_QK;
        float low = 0.0f;
        for (j = 0; j < 16; ++j) low += v[j];
        q4_low_sums[block] = low;
        for (t = 0; t < 4; ++t) {
            float *q = q4_xt + block * FALCON_QK + t * 8;
            for (k = 0; k < 4; ++k) {
                float lo16 = 16.0f * v[t * 4 + k];
                q[k] = lo16;
                q[4 + k] = v[16 + t * 4 + k] - lo16;
            }
        }
    }
}

/* lv.q/sv.q fault on real hardware (not in PPSSPP) unless the address is
 * 16-byte aligned. Every group pointer is aligned by construction
 * (16-aligned offsets, aligned cache blocks and buffers); this guard turns
 * a future layout mistake into a counted copy instead of a crash. */
static void q4_group(FalconRuntime *runtime, const uint8_t *group,
                     size_t group_bytes, int blocks, float *out) {
    if ((uintptr_t)group & 15u) {
        memmove(runtime->row_buffer, group, group_bytes);
        group = runtime->row_buffer;
        ++runtime->realigned_groups;
    }
    if ((uintptr_t)out & 15u) {
        q4_kernel(group, q4_xt, q4_low_sums, blocks, q4_out4, q4_bias);
        memcpy(out, q4_out4, sizeof(q4_out4));
    } else {
        q4_kernel(group, q4_xt, q4_low_sums, blocks, out, q4_bias);
    }
}
#define DOT_GROUP(bytes, dst) \
    q4_group(runtime, (bytes), group_bytes, blocks, (dst))
#else
/* Host golden reference: per row, exactly the arithmetic and summation
 * order of the version-1 row kernel (with the decoded 8-bit scale codes
 * in place of the fp16 scales), so a version-2 model through
 * FALCON_EVAL_V2 is bit-identical to v1.3. FALCON_HOST_SUM4 (evaluation
 * yardstick only) sums each block in four lanes instead, the way the PSP
 * kernel's quads do. */
static float q4_scale_lut[256];

static float dot_q4_tile_row(const uint8_t *group, int r, const float *x,
                             int blocks, const FalconQ4Param *param) {
    const uint8_t *tiles = group + (size_t)blocks * 4u * SCALE_BYTES;
    int block, j;
    float sum = 0.0f;
    (void)param;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *t = tiles + (size_t)block * 64u + (size_t)r * 4u;
        const float *v = x + block * FALCON_QK;
#ifdef FALCON_EVAL_V2
        float d = eval_v2_model ? group_scale(group, r, block, param)
                                : q4_scale_lut[group[block * 4 + r]];
#else
        float d = q4_scale_lut[group[block * 4 + r]];
#endif
        float block_sum = 0.0f;
#ifdef FALCON_HOST_SUM4
        float lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (j = 0; j < 16; ++j) {
            uint8_t packed = (uint8_t)(t[(j >> 2) * 16 + (j & 3)] ^ 0x80u);
            lane[j & 3] += q4_lut_low[packed] * v[j];
            lane[j & 3] += q4_lut_high[packed] * v[j + 16];
        }
        block_sum = (lane[0] + lane[1]) + (lane[2] + lane[3]);
#else
        for (j = 0; j < 16; ++j) {
            uint8_t packed = (uint8_t)(t[(j >> 2) * 16 + (j & 3)] ^ 0x80u);
            block_sum += q4_lut_low[packed] * v[j];
            block_sum += q4_lut_high[packed] * v[j + 16];
        }
#endif
        sum += d * block_sum;
    }
    return sum;
}
static void dot_q4_group(const uint8_t *group, const float *x, int blocks,
                         float *out, const FalconQ4Param *param) {
    int r;
    for (r = 0; r < FALCON_GROUP_ROWS; ++r)
        out[r] = dot_q4_tile_row(group, r, x, blocks, param);
}
static void q4_prepare_scales(const FalconQ4Param *param) {
    int code;
    for (code = 0; code < 256; ++code)
        q4_scale_lut[code] = scale_code_value((int8_t)(uint8_t)code, param);
}
#define DOT_GROUP(bytes, dst) dot_q4_group((bytes), x, blocks, (dst), param)
#endif

static int q4_matvec(FalconModel *model, FalconRuntime *runtime, float *out,
                     uint32_t offset, int rows, int cols, const float *x) {
    size_t group_bytes = q4_group_bytes(cols);
    int blocks = cols / FALCON_QK;
    int groups = rows / FALCON_GROUP_ROWS, group, file_active = 0;
    uint32_t file_pos = 0;
    unsigned int started = runtime->clock_us ? runtime->clock_us() : 0;
    const FalconQ4Param *param = q4_param(model, offset);
    if (rows % FALCON_GROUP_ROWS || blocks < 4 || blocks % 4 ||
        blocks > FALCON_MAX_BLOCKS || group_bytes > runtime->row_buffer_bytes ||
        !param) {
        runtime->io_error = 1;
        return 0;
    }
    q4_prepare_scales(param);
#ifdef FALCON_VFPU
    q4_prepare_x(x, blocks);
#endif
#ifdef FALCON_PREFETCH
    falcon_prefetch_poll();
#endif
    for (group = 0; group < groups; ++group) {
        uint32_t group_offset =
            offset + (uint32_t)((size_t)group * group_bytes);
        float *dst = out + (size_t)group * FALCON_GROUP_ROWS;
        const uint8_t *cached = cache_pointer(model, group_offset, group_bytes);
        if (cached) {
            DOT_GROUP(cached, dst);
            continue;
        }
        /* Groups straddling two cache blocks go through the aligned row
         * buffer. */
        if (cache_copy(model, group_offset, runtime->row_buffer,
                       group_bytes)) {
            DOT_GROUP(runtime->row_buffer, dst);
            continue;
        }
#ifdef FALCON_PREFETCH
        {
            const uint8_t *ahead =
                falcon_prefetch_get(group_offset, (uint32_t)group_bytes);
            if (ahead) {
                runtime->streamed_bytes += group_bytes;
                DOT_GROUP(ahead, dst);
                continue;
            }
        }
#endif
#ifdef FALCON_RAWIO
        if (model->stream_ready) {
            const uint8_t *streamed =
                stream_group(model, runtime, group_offset, group_bytes);
            if (!streamed) {
                runtime->io_error = 1;
                return 0;
            }
            DOT_GROUP(streamed, dst);
            continue;
        }
#endif
        /* Synchronous stdio fallback (host builds, or a PSP whose raw
         * descriptor could not be opened): read as many consecutive groups
         * as fit in the row buffer with one fread (groups of one matrix
         * are contiguous in the file). file_pos tracks the stdio position
         * so groups served from the cache or prefetcher in between can
         * never desynchronize the sequential stream. */
        {
            int chunk = (int)(runtime->row_buffer_bytes / group_bytes);
            int remaining = groups - group, j;
            size_t chunk_bytes;
            if (chunk > remaining) chunk = remaining;
            chunk_bytes = (size_t)chunk * group_bytes;
            if (!file_active || file_pos != group_offset) {
                if (fseek(model->file, (long)group_offset, SEEK_SET) != 0) {
                    runtime->io_error = 1;
                    return 0;
                }
                file_active = 1;
            }
            if (fread(runtime->row_buffer, 1, chunk_bytes, model->file) !=
                chunk_bytes) {
                runtime->io_error = 1;
                return 0;
            }
            file_pos = group_offset + (uint32_t)chunk_bytes;
            runtime->streamed_bytes += chunk_bytes;
            for (j = 0; j < chunk; ++j)
                DOT_GROUP(runtime->row_buffer + (size_t)j * group_bytes,
                          dst + (size_t)j * FALCON_GROUP_ROWS);
            group += chunk - 1;
        }
    }
#undef DOT_GROUP
    if (runtime->clock_us)
        runtime->matvec_us += runtime->clock_us() - started;
    return 1;
}

static int q4_get_row(FalconModel *model, FalconRuntime *runtime, float *out,
                      uint32_t matrix, int row, int cols) {
    size_t bytes = q4_group_bytes(cols);
    uint32_t offset = matrix + (uint32_t)((size_t)(row / FALCON_GROUP_ROWS) *
                                          bytes);
    const FalconQ4Param *param = q4_param(model, matrix);
    if (bytes > runtime->row_buffer_bytes || !param ||
        !model_read(model, offset, runtime->row_buffer, bytes)) {
        runtime->io_error = 1;
        return 0;
    }
    dequantize_row(out, runtime->row_buffer, row % FALCON_GROUP_ROWS, cols,
                   param);
    return 1;
}

/* Every Q4 matrix must hold whole tile groups and start 16-byte aligned
 * relative to the weight region (true by arithmetic over the tensor sizes;
 * checked because lv.q faults on hardware otherwise). The block count must
 * be a multiple of 4 too: a group's tiles start blocks * 4 bytes in and a
 * group is 68 * blocks bytes, both 16-aligned only then (16 and 24 blocks
 * in this model). The matrix is also registered in model->q4 in file
 * order. */
static uint32_t advance_q4(FalconModel *model, uint32_t *offset, int rows,
                           int cols, int *layout_ok) {
    uint32_t current = *offset;
    if (rows <= 0 || rows % FALCON_GROUP_ROWS || cols <= 0 ||
        cols % (4 * FALCON_QK) ||
        ((current - model->weights_offset) & 15u) ||
        model->q4_count >= FALCON_Q4_MATRICES)
        *layout_ok = 0;
    else
        model->q4[model->q4_count++].offset = current;
    *offset += (uint32_t)q4_bytes(rows, cols);
    return current;
}
static uint32_t advance_f32(uint32_t *offset, int count) {
    uint32_t current = *offset;
    *offset += (uint32_t)((size_t)count * sizeof(float));
    return current;
}

int falcon_model_open(FalconModel *model, const char *path,
                      char *error, size_t error_size) {
    uint8_t header[FALCON_HEADER_BYTES];
    uint8_t tok_header[44];
    uint64_t reported_size, tokenizer_offset, weights_offset;
    uint32_t off, version;
    int i, conv_dim, layout_ok = 1;
    memset(model, 0, sizeof(*model));
    model->stream_fd = -1;
    model->file = fopen(path, "rb");
    if (!model->file) {
        set_error(error, error_size, "cannot open %s: %s", path, strerror(errno));
        return 0;
    }
#ifdef FALCON_RAWIO
    /* Uncached weights stream through the raw window; stdio then only
     * does the header, tokenizer, small tensors, the cache fill (256 KiB
     * reads) and embedding groups, so it runs unbuffered: every fread is
     * one direct read and the 64 KiB stdio buffer is not needed. */
    if (stream_open(model, path))
        setvbuf(model->file, NULL, _IONBF, 0);
#endif
    if (model->stream_fd < 0) {
        model->io_buffer_bytes = 64u * 1024u;
        model->io_buffer = (uint8_t *)malloc(model->io_buffer_bytes);
        if (model->io_buffer)
            setvbuf(model->file, (char *)model->io_buffer, _IOFBF,
                    model->io_buffer_bytes);
        else
            model->io_buffer_bytes = 0;
    }
    if (fread(header, 1, sizeof(header), model->file) != sizeof(header) ||
        rd32(header) != FALCON_MAGIC ||
        rd32(header + 8) != FALCON_HEADER_BYTES) {
        set_error(error, error_size, "not a supported FHQ4 model");
        falcon_model_close(model);
        return 0;
    }
    version = rd32(header + 4);
#ifdef FALCON_EVAL_V2
    eval_v2_model = version == 2;
    if (version != FALCON_VERSION && version != 2) {
#else
    if (version != FALCON_VERSION) {
#endif
        if (version == 1 || version == 2)
            set_error(error, error_size,
                      "model.fhq4 is FHQ4 version %lu; this build needs "
                      "version 3: re-pack it with tools/repack_fhq4.py",
                      (unsigned long)version);
        else
            set_error(error, error_size, "unsupported FHQ4 version %lu",
                      (unsigned long)version);
        falcon_model_close(model);
        return 0;
    }
    reported_size = rd64(header + 16);
    tokenizer_offset = rd64(header + 24);
    weights_offset = rd64(header + 32);
    if (reported_size > 0xffffffffu || tokenizer_offset > 0xffffffffu ||
        weights_offset > 0xffffffffu) {
        set_error(error, error_size, "model offsets exceed PSP file limits");
        falcon_model_close(model);
        return 0;
    }
#define CFG(I) ((int)rd32(header + 40 + (I) * 4))
    model->config.dim = CFG(0);
    model->config.hidden_dim = CFG(1);
    model->config.n_layers = CFG(2);
    model->config.n_heads = CFG(3);
    model->config.n_kv_heads = CFG(4);
    model->config.head_dim = CFG(5);
    model->config.vocab_size = CFG(6);
    model->config.model_context = CFG(7);
    model->config.context = CFG(8);
    model->config.pad_id = CFG(9);
    model->config.eos_id = CFG(10);
    model->config.bos_id = CFG(11);
    model->config.im_start_id = CFG(12);
    model->config.im_end_id = CFG(13);
    model->config.eom_id = CFG(14);
    model->config.eot_id = CFG(15);
    model->config.mamba_dim = CFG(16);
    model->config.mamba_heads = CFG(17);
    model->config.mamba_head_dim = CFG(18);
    model->config.mamba_state = CFG(19);
    model->config.mamba_groups = CFG(20);
    model->config.mamba_conv = CFG(21);
    model->config.mamba_projection = CFG(22);
#undef CFG
    model->config.rope_theta = rdf32(header + 140);
    model->config.rms_eps = rdf32(header + 144);
    model->config.embedding_multiplier = rdf32(header + 148);
    model->config.lm_head_multiplier = rdf32(header + 152);
    if (model->config.dim != 512 || model->config.hidden_dim != 768 ||
        model->config.n_layers != 24 || model->config.n_heads != 8 ||
        model->config.n_kv_heads != 2 || model->config.head_dim != 64 ||
        model->config.vocab_size != 32768 || model->config.mamba_dim != 768 ||
        model->config.mamba_heads != 24 || model->config.mamba_head_dim != 32 ||
        model->config.mamba_state != 64 || model->config.mamba_groups != 1 ||
        model->config.mamba_conv != 4 || rd32(header + 40 + 23 * 4) != FALCON_QK) {
        set_error(error, error_size, "FHQ4 architecture does not match Falcon-H1 Tiny 90M");
        falcon_model_close(model);
        return 0;
    }
    if (!seek_read(model->file, (uint32_t)tokenizer_offset, tok_header,
                   sizeof(tok_header)) || rd32(tok_header) != 0x324b4f54u ||
        rd32(tok_header + 4) != 1 ||
        (int)rd32(tok_header + 8) != model->config.vocab_size) {
        set_error(error, error_size, "invalid embedded tokenizer");
        falcon_model_close(model);
        return 0;
    }
    model->tokenizer.section_bytes = rd32(tok_header + 36);
    model->tokenizer.section = (uint8_t *)malloc(model->tokenizer.section_bytes);
    if (!model->tokenizer.section ||
        !seek_read(model->file, (uint32_t)tokenizer_offset,
                   model->tokenizer.section, model->tokenizer.section_bytes)) {
        set_error(error, error_size, "not enough RAM for tokenizer or short model read");
        falcon_model_close(model);
        return 0;
    }
    model->tokenizer.vocab_size = model->config.vocab_size;
    model->tokenizer.max_piece_bytes = (int)rd32(tok_header + 40);
    model->tokenizer.offsets = model->tokenizer.section + rd32(tok_header + 16);
    model->tokenizer.pieces = model->tokenizer.section + rd32(tok_header + 20);
    model->tokenizer.sorted_ids = model->tokenizer.section + rd32(tok_header + 24);
    model->tokenizer.ranks = model->tokenizer.section + rd32(tok_header + 28);
    model->tokenizer.byte_ids = model->tokenizer.section + rd32(tok_header + 32);

    model->weights_offset = (uint32_t)weights_offset;
    off = (uint32_t)weights_offset;
    /* The PSP kernel's 16-byte alignment rests on group offsets being
     * 16-aligned in the file, i.e. on an aligned weight region. */
    if (off & 15u) layout_ok = 0;
#define ADVANCE_Q4(ROWS, COLS) \
    advance_q4(model, &off, (ROWS), (COLS), &layout_ok)
    model->embedding = ADVANCE_Q4(model->config.vocab_size, model->config.dim);
    conv_dim = model->config.mamba_dim +
               2 * model->config.mamba_groups * model->config.mamba_state;
    for (i = 0; i < model->config.n_layers; ++i) {
        FalconLayerOffsets *layer = &model->layers[i];
        layer->input_norm = advance_f32(&off, model->config.dim);
        layer->wq = ADVANCE_Q4(model->config.n_heads * model->config.head_dim,
                               model->config.dim);
        layer->wk = ADVANCE_Q4(model->config.n_kv_heads * model->config.head_dim,
                               model->config.dim);
        layer->wv = ADVANCE_Q4(model->config.n_kv_heads * model->config.head_dim,
                               model->config.dim);
        layer->wo = ADVANCE_Q4(model->config.dim,
                               model->config.n_heads * model->config.head_dim);
        layer->mamba_in = ADVANCE_Q4(model->config.mamba_projection,
                                     model->config.dim);
        layer->conv_weight = advance_f32(&off, conv_dim * model->config.mamba_conv);
        layer->conv_bias = advance_f32(&off, conv_dim);
        layer->dt_bias = advance_f32(&off, model->config.mamba_heads);
        layer->a_log = advance_f32(&off, model->config.mamba_heads);
        layer->d = advance_f32(&off, model->config.mamba_heads);
        layer->mamba_out = ADVANCE_Q4(model->config.dim,
                                      model->config.mamba_dim);
        layer->ffn_norm = advance_f32(&off, model->config.dim);
        layer->w_gate = ADVANCE_Q4(model->config.hidden_dim, model->config.dim);
        layer->w_up = ADVANCE_Q4(model->config.hidden_dim, model->config.dim);
        layer->w_down = ADVANCE_Q4(model->config.dim, model->config.hidden_dim);
    }
#undef ADVANCE_Q4
    model->final_norm = advance_f32(&off, model->config.dim);
    model->file_size = (uint32_t)reported_size;
    if (!layout_ok) {
        set_error(error, error_size,
                  "FHQ4 Q4 matrices are not 16-byte aligned 4-row groups");
        falcon_model_close(model);
        return 0;
    }
    if (off != model->file_size) {
        set_error(error, error_size, "model layout mismatch (%lu != %lu)",
                  (unsigned long)off, (unsigned long)model->file_size);
        falcon_model_close(model);
        return 0;
    }
#ifdef FALCON_EVAL_V2
    if (eval_v2_model) {
        /* Version 2 has fp16 scales and no code table; the parameters are
         * placeholders that are never used for decoding. */
        for (i = 0; i < model->q4_count; ++i) {
            model->q4[i].mant_bits = 4;
            model->q4[i].bias = 112u << 4;
        }
    } else
#endif
    {
        /* Version 3: one 8-byte entry per Q4 matrix in file order (u32
         * offset relative to the weight region, u8 mantissa bits, u8 0,
         * u16 bias), between the tokenizer and the weight region. Each must
         * name the matrix the walk above found and decode to normal
         * floats only. */
        uint32_t table = rd32(header + 160);
        uint8_t entries[FALCON_Q4_MATRICES * 8];
        int ok = rd32(header + 164) == (uint32_t)model->q4_count &&
                 rd32(header + 168) == 8u &&
                 table >= tokenizer_offset + model->tokenizer.section_bytes &&
                 table + (uint32_t)model->q4_count * 8u <= weights_offset &&
                 seek_read(model->file, table, entries,
                           (size_t)model->q4_count * 8u);
        for (i = 0; ok && i < model->q4_count; ++i) {
            const uint8_t *e = entries + i * 8;
            ok = rd32(e) == model->q4[i].offset - model->weights_offset &&
                 e[5] == 0 && scale_param_valid(e[4], rd16(e + 6));
            model->q4[i].mant_bits = e[4];
            model->q4[i].bias = rd16(e + 6);
        }
        if (!ok) {
            set_error(error, error_size, "invalid FHQ4 scale-code table");
            falcon_model_close(model);
            return 0;
        }
    }
    q4_lut_init();
    /* Preload every small float tensor (about 0.54 MiB). This removes all
     * per-layer small seek-reads from the token loop; only the large Q4
     * matrices are read through the cache or streamed at inference time. */
    {
        const FalconConfig *c = &model->config;
        size_t per_layer = (size_t)c->dim * 2 +
                           (size_t)conv_dim * (c->mamba_conv + 1) +
                           (size_t)c->mamba_heads * 3;
        size_t total_floats = per_layer * c->n_layers + (size_t)c->dim;
        float *p = (float *)malloc(total_floats * sizeof(float));
        int ok = p != NULL;
        model->small_block = p;
        model->small_bytes = total_floats * sizeof(float);
        for (i = 0; ok && i < c->n_layers; ++i) {
            FalconLayerOffsets *lo = &model->layers[i];
            FalconLayerSmall *ls = &model->small[i];
            struct { uint32_t offset; size_t count; const float **target; } part[7];
            int j;
            part[0].offset = lo->input_norm;
            part[0].count = (size_t)c->dim;
            part[0].target = &ls->input_norm;
            part[1].offset = lo->conv_weight;
            part[1].count = (size_t)conv_dim * c->mamba_conv;
            part[1].target = &ls->conv_weight;
            part[2].offset = lo->conv_bias;
            part[2].count = (size_t)conv_dim;
            part[2].target = &ls->conv_bias;
            part[3].offset = lo->dt_bias;
            part[3].count = (size_t)c->mamba_heads;
            part[3].target = &ls->dt_bias;
            part[4].offset = lo->a_log;
            part[4].count = (size_t)c->mamba_heads;
            part[4].target = &ls->a_log;
            part[5].offset = lo->d;
            part[5].count = (size_t)c->mamba_heads;
            part[5].target = &ls->d;
            part[6].offset = lo->ffn_norm;
            part[6].count = (size_t)c->dim;
            part[6].target = &ls->ffn_norm;
            for (j = 0; ok && j < 7; ++j) {
                ok = seek_read(model->file, part[j].offset, p,
                               part[j].count * sizeof(float));
                *part[j].target = p;
                p += part[j].count;
            }
        }
        if (ok) {
            ok = seek_read(model->file, model->final_norm, p,
                           (size_t)c->dim * sizeof(float));
            model->final_norm_values = p;
        }
        if (!ok) {
            set_error(error, error_size,
                      "not enough RAM or short read for layer tables");
            falcon_model_close(model);
            return 0;
        }
    }
    return 1;
}

static int cache_block_is_extra(const FalconModel *model, const uint8_t *p) {
    return model->cache_extra_base && p >= model->cache_extra_base &&
           p < model->cache_extra_base + model->cache_extra_bytes;
}

size_t falcon_model_cache_weights_extra(FalconModel *model,
                                        size_t target_bytes,
                                        uint8_t *extra, size_t extra_bytes,
                                        int stripe,
                                        char *error, size_t error_size) {
    const size_t block_bytes = 256u * 1024u;
    size_t region_bytes, total_blocks, extra_blocks, heap_target;
    size_t pool_count = 0, pool_used = 0, loaded = 0;
    size_t emb_blocks, layer_blocks = 0, layer_holes = 0, i;
    uint8_t **pool;
    int alloc_failed = 0;
    if (error && error_size) error[0] = '\0';
    if (!model || !model->file || model->file_size <= model->weights_offset)
        return 0;
    if (model->weight_cache) return model->weight_cache_bytes;
    region_bytes = (size_t)(model->file_size - model->weights_offset);
    total_blocks = (region_bytes + block_bytes - 1) / block_bytes;
    if (extra && ((uintptr_t)extra & 15u)) {
        /* Cache blocks must be 16-byte aligned for the PSP kernel. */
        size_t skew = 16u - (size_t)((uintptr_t)extra & 15u);
        extra = extra_bytes > skew ? extra + skew : NULL;
        extra_bytes = extra ? extra_bytes - skew : 0;
    }
    extra_blocks = extra ? extra_bytes / block_bytes : 0;
    if (extra_blocks > total_blocks) extra_blocks = total_blocks;
    /* Phase A: gather block memory (foreign region first, then heap). */
    pool = (uint8_t **)calloc(total_blocks, sizeof(uint8_t *));
    model->weight_cache = (uint8_t **)calloc(total_blocks, sizeof(uint8_t *));
    if (!pool || !model->weight_cache) {
        set_error(error, error_size, "no RAM for weight-cache index");
        free(pool);
        free(model->weight_cache);
        model->weight_cache = NULL;
        return 0;
    }
    model->weight_cache_block_bytes = block_bytes;
    model->weight_cache_blocks = (int)total_blocks;
    model->cache_extra_base = extra;
    model->cache_extra_bytes = extra_blocks * block_bytes;
    for (i = 0; i < extra_blocks; ++i)
        pool[pool_count++] = extra + i * block_bytes;
    heap_target = (target_bytes + block_bytes - 1) / block_bytes;
    for (i = 0; i < heap_target && pool_count < total_blocks; ++i) {
        /* The last pool slot exists only if every block gets cached, and
         * then holds the region's last (partial) block: allocate just
         * that. */
        uint8_t *memory = cache_block_alloc(
            pool_count + 1 == total_blocks ?
                region_bytes - (total_blocks - 1) * block_bytes : block_bytes);
        if (!memory) {
            alloc_failed = 1;
            break;
        }
        pool[pool_count++] = memory;
    }
    model->cache_ram_limited = alloc_failed;
    if (alloc_failed && pool_count > extra_blocks) {
        /* Give later prompt/history allocations one block of headroom. */
        cache_block_free(pool[--pool_count]);
    }
    if (!pool_count) {
        free(pool);
        free(model->weight_cache);
        model->weight_cache = NULL;
        model->weight_cache_blocks = 0;
        return 0;
    }
    /* Phase B/C: pick which block indices are cached and read them. A
     * prefix fill caches blocks 0..K-1. A striped fill caches the whole
     * embedding, then spreads the deficit evenly over the layer blocks
     * so every layer keeps roughly the same small streamed share. */
    emb_blocks = ((size_t)(model->layers[0].input_norm -
                           model->weights_offset) +
                  block_bytes - 1) / block_bytes;
    if (stripe && pool_count > emb_blocks && pool_count < total_blocks) {
        layer_blocks = total_blocks - emb_blocks;
        layer_holes = layer_blocks - (pool_count - emb_blocks);
    } else {
        stripe = 0;
    }
    for (i = 0; i < total_blocks && pool_used < pool_count; ++i) {
        size_t amount;
        int cached;
        if (!stripe) {
            cached = pool_used < pool_count; /* prefix */
        } else if (i < emb_blocks) {
            cached = 1;
        } else {
            size_t j = i - emb_blocks;
            cached = (j + 1) * layer_holes / layer_blocks ==
                     j * layer_holes / layer_blocks;
        }
        if (!cached) continue;
        amount = region_bytes - i * block_bytes;
        if (amount > block_bytes) amount = block_bytes;
        if (fseek(model->file,
                  (long)(model->weights_offset + i * block_bytes),
                  SEEK_SET) != 0 ||
            fread(pool[pool_used], 1, amount, model->file) != amount) {
            set_error(error, error_size,
                      "short read while filling weight cache");
            break;
        }
        model->weight_cache[i] = pool[pool_used++];
        loaded += amount;
    }
    /* Return unused pool blocks (read error) to the heap. */
    for (; pool_used < pool_count; ++pool_used)
        if (!cache_block_is_extra(model, pool[pool_used]))
            cache_block_free(pool[pool_used]);
    free(pool);
    model->weight_cache_bytes = loaded;
    {
        size_t wanted = extra_blocks * block_bytes + target_bytes;
        if (wanted > region_bytes) wanted = region_bytes;
        if (loaded + block_bytes < wanted && error && error_size &&
            !error[0])
            set_error(error, error_size, "RAM cache stopped at %.2f MiB",
                      (double)loaded / 1048576.0);
    }
    return loaded;
}

size_t falcon_model_cache_weights(FalconModel *model, size_t target_bytes,
                                  char *error, size_t error_size) {
    return falcon_model_cache_weights_extra(model, target_bytes, NULL, 0, 0,
                                            error, error_size);
}

static size_t cache_block_size(const FalconModel *model, size_t block) {
    size_t region = (size_t)(model->file_size - model->weights_offset);
    size_t start = block * model->weight_cache_block_bytes;
    size_t amount = region - start;
    return amount < model->weight_cache_block_bytes ?
        amount : model->weight_cache_block_bytes;
}

int falcon_model_uncached_blocks(const FalconModel *model) {
    size_t region, block_bytes = 256u * 1024u;
    int holes = 0, i;
    if (!model || model->file_size <= model->weights_offset) return 0;
    if (!model->weight_cache) {
        region = (size_t)(model->file_size - model->weights_offset);
        return (int)((region + block_bytes - 1) / block_bytes);
    }
    for (i = 0; i < model->weight_cache_blocks; ++i)
        if (!model->weight_cache[i]) ++holes;
    return holes;
}

/* Heap bytes the uncached blocks would take, including the per-block
 * alignment slack of cache_block_alloc. */
static size_t uncached_alloc_bytes(const FalconModel *model) {
    size_t total = 0;
    int i;
    for (i = 0; model->weight_cache && i < model->weight_cache_blocks; ++i)
        if (!model->weight_cache[i])
            total += cache_block_size(model, (size_t)i) + ALIGNED_OVERHEAD;
    return total;
}

/* Allocates (without reading) a block for every hole; on the first
 * failure gives back what this call took and returns 0. On success the
 * blocks sit in the index marked by `pending` (read them with
 * cache_read_pending, or drop them with cache_drop_pending). */
static int cache_alloc_pending(FalconModel *model, uint8_t *pending) {
    int i;
    for (i = 0; i < model->weight_cache_blocks; ++i) {
        uint8_t *memory;
        pending[i] = 0;
        if (model->weight_cache[i]) continue;
        memory = cache_block_alloc(cache_block_size(model, (size_t)i));
        if (!memory) {
            while (--i >= 0)
                if (pending[i]) {
                    cache_block_free(model->weight_cache[i]);
                    model->weight_cache[i] = NULL;
                    pending[i] = 0;
                }
            return 0;
        }
        model->weight_cache[i] = memory;
        pending[i] = 1;
    }
    return 1;
}

static void cache_drop_pending(FalconModel *model, const uint8_t *pending) {
    int i;
    for (i = 0; i < model->weight_cache_blocks; ++i)
        if (pending[i]) {
            cache_block_free(model->weight_cache[i]);
            model->weight_cache[i] = NULL;
        }
}

/* Reads the pending blocks; a block whose read fails is freed and stays
 * uncached (it streams like any other). Returns 0 on a read error. */
static int cache_read_pending(FalconModel *model, const uint8_t *pending) {
    int i, ok = 1;
    for (i = 0; i < model->weight_cache_blocks; ++i) {
        size_t amount;
        if (!pending[i]) continue;
        amount = cache_block_size(model, (size_t)i);
        if (!seek_read(model->file, model->weights_offset +
                                        (uint32_t)((size_t)i *
                                                   model->weight_cache_block_bytes),
                       model->weight_cache[i], amount)) {
            cache_block_free(model->weight_cache[i]);
            model->weight_cache[i] = NULL;
            ok = 0;
            continue;
        }
        model->weight_cache_bytes += amount;
    }
    return ok;
}

int falcon_model_cache_fill_holes(FalconModel *model,
                                  char *error, size_t error_size) {
    int i;
    if (error && error_size) error[0] = '\0';
    if (!model || !model->weight_cache) return falcon_model_uncached_blocks(model);
    for (i = 0; i < model->weight_cache_blocks; ++i) {
        size_t amount;
        uint8_t *memory;
        if (model->weight_cache[i]) continue;
        amount = cache_block_size(model, (size_t)i);
        memory = cache_block_alloc(amount);
        if (!memory) continue; /* a smaller tail block may still fit */
        if (!seek_read(model->file, model->weights_offset +
                                        (uint32_t)((size_t)i *
                                                   model->weight_cache_block_bytes),
                       memory, amount)) {
            cache_block_free(memory);
            set_error(error, error_size,
                      "short read while filling weight cache");
            break;
        }
        model->weight_cache[i] = memory;
        model->weight_cache_bytes += amount;
    }
    return falcon_model_uncached_blocks(model);
}

int falcon_model_stream_prepare(FalconModel *model) {
#ifdef FALCON_RAWIO
    int i;
    if (!model || model->stream_ready || model->stream_fd < 0 ||
        !falcon_model_uncached_blocks(model))
        return 1;
    if (stream_window_alloc(model)) return 1;
    /* The heap is full: one cached heap block (the last one, nearest the
     * streamed tail) is worth less than a working window. */
    for (i = model->weight_cache_blocks - 1; i >= 0; --i) {
        uint8_t *block = model->weight_cache[i];
        if (block && !cache_block_is_extra(model, block)) {
            model->weight_cache_bytes -= cache_block_size(model, (size_t)i);
            model->weight_cache[i] = NULL;
            cache_block_free(block);
            break;
        }
    }
    return stream_window_alloc(model);
#else
    (void)model;
    return 1;
#endif
}

void falcon_model_close(FalconModel *model) {
    int block;
    if (!model) return;
    if (model->file) fclose(model->file);
#ifdef FALCON_RAWIO
    if (model->stream_fd >= 0) sceIoClose(model->stream_fd);
#endif
    for (block = 0; block < model->weight_cache_blocks; ++block)
        if (!cache_block_is_extra(model, model->weight_cache[block]))
            cache_block_free(model->weight_cache[block]);
    free(model->weight_cache);
    free(model->io_buffer);
    free(model->small_block);
    free(model->tokenizer.section);
    memset(model, 0, sizeof(*model));
}

/* Test hook (falcon_runtime_set_alloc_limit): 0 means no cap. */
static size_t runtime_alloc_limit;

void falcon_runtime_set_alloc_limit(size_t bytes) {
    runtime_alloc_limit = bytes;
}

/* All runtime arrays are 16-byte aligned so the PSP VFPU kernels can use
 * quad loads and stores on them (the Q4 kernel stores four results at
 * out + row, row a multiple of 4). bytes_allocated counts each array with
 * its alignment overhead (falcon_runtime_plan_bytes uses the same rule). */
static void *alloc_array(FalconRuntime *runtime, size_t count, size_t size) {
    size_t bytes, total;
    uint8_t *aligned;
    if (count && size > ((size_t)-1 - 64) / count) return NULL;
    bytes = count * size;
    total = bytes + ALIGNED_OVERHEAD;
    if (runtime_alloc_limit &&
        runtime->bytes_allocated + total > runtime_alloc_limit)
        return NULL;
    aligned = (uint8_t *)aligned_block_alloc(bytes);
    if (!aligned) return NULL;
    memset(aligned, 0, bytes);
    runtime->bytes_allocated += total;
    return aligned;
}

/* With stdio streaming the row buffer holds dozens of groups: the
 * fallback reads whole group batches per fread, which cuts ~20k stdio
 * calls per token to a few hundred. With the PSP raw descriptor it only
 * holds single groups (cache-block straddles, embedding lookups, the
 * alignment guard, and the stdio path if the read window could not be
 * allocated). Must stay >= the largest group (1632 B). */
static size_t row_buffer_size(const FalconModel *model) {
    return model->stream_fd >= 0 ? 2048u : 32u * 1024u;
}

static int state_mode_supported(int mode) {
#ifdef _PSP_FW_VERSION
    return mode == FALCON_STATE_F32 || mode == FALCON_STATE_F16;
#else
    return mode >= FALCON_STATE_F32 && mode <= FALCON_STATE_F16_TRUNC_CORR;
#endif
}

size_t falcon_runtime_plan_bytes(const FalconModel *model, int context) {
    const FalconConfig *c = &model->config;
    size_t conv_dim = (size_t)c->mamba_dim + 2u * c->mamba_groups * c->mamba_state;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t layers = (size_t)c->n_layers, ctx = (size_t)context;
    size_t mamba_values = layers * c->mamba_dim * c->mamba_state;
    size_t floats[] = {
        (size_t)c->dim, (size_t)c->dim, (size_t)c->n_heads * c->head_dim,
        kv_dim, kv_dim, ctx, (size_t)c->dim, (size_t)c->mamba_projection,
        conv_dim, (size_t)c->mamba_dim, (size_t)c->hidden_dim,
        (size_t)c->hidden_dim, (size_t)c->dim, (size_t)c->vocab_size,
        layers * conv_dim * c->mamba_conv, layers * ctx * c->n_kv_heads,
        layers * ctx * c->n_kv_heads,
    };
    size_t overhead = ALIGNED_OVERHEAD, total = 0, i;
    for (i = 0; i < sizeof(floats) / sizeof(floats[0]); ++i)
        total += floats[i] * sizeof(float) + overhead;
    total += mamba_values * (model->state_mode ? sizeof(uint16_t)
                                               : sizeof(float)) + overhead;
    total += 2u * (layers * ctx * kv_dim + overhead);  /* int8 K and V */
    total += row_buffer_size(model) + overhead;
    total += ctx * sizeof(int) + overhead;              /* history */
    return total;
}

int falcon_runtime_init(FalconRuntime *r, const FalconModel *model,
                        char *error, size_t error_size) {
    const FalconConfig *c = &model->config;
    int conv_dim = c->mamba_dim + 2 * c->mamba_groups * c->mamba_state;
    int kv_dim = c->n_kv_heads * c->head_dim;
    size_t kv_values = (size_t)c->n_layers * c->context * kv_dim;
    size_t kv_scales = (size_t)c->n_layers * c->context * c->n_kv_heads;
    size_t mamba_values = (size_t)c->n_layers * c->mamba_dim * c->mamba_state;
    size_t conv_values = (size_t)c->n_layers * conv_dim * c->mamba_conv;
    memset(r, 0, sizeof(*r));
    if (!state_mode_supported(model->state_mode)) {
        set_error(error, error_size, "unsupported state mode %d",
                  model->state_mode);
        return 0;
    }
    r->state_mode = model->state_mode;
#define AF(NAME, COUNT) r->NAME = (float *)alloc_array(r, (COUNT), sizeof(float))
    AF(x, c->dim); AF(norm, c->dim); AF(q, c->n_heads * c->head_dim);
    AF(k, kv_dim); AF(v, kv_dim); AF(attention, c->context);
    AF(attention_out, c->dim); AF(projection, c->mamba_projection);
    AF(conv_out, conv_dim); AF(mamba_out, c->mamba_dim);
    AF(ff_gate, c->hidden_dim); AF(ff_up, c->hidden_dim);
    AF(temporary, c->dim); AF(logits, c->vocab_size);
    AF(conv_state, conv_values);
    AF(key_scales, kv_scales); AF(value_scales, kv_scales);
    if (r->state_mode == FALCON_STATE_F32)
        AF(mamba_state, mamba_values);
    else
        r->mamba_state16 = (uint16_t *)alloc_array(r, mamba_values,
                                                   sizeof(uint16_t));
#undef AF
    r->key_cache = (int8_t *)alloc_array(r, kv_values, sizeof(int8_t));
    r->value_cache = (int8_t *)alloc_array(r, kv_values, sizeof(int8_t));
    r->row_buffer_bytes = row_buffer_size(model);
    r->row_buffer = (uint8_t *)alloc_array(r, r->row_buffer_bytes, 1);
    /* Generation must never allocate once a conversation runs, so the
     * repetition-penalty history is sized for the whole context here. */
    r->history_capacity = c->context;
    r->history = (int *)alloc_array(r, (size_t)c->context, sizeof(int));
    if (!r->x || !r->norm || !r->q || !r->k || !r->v || !r->attention ||
        !r->attention_out || !r->projection || !r->conv_out || !r->mamba_out ||
        !r->ff_gate || !r->ff_up || !r->temporary || !r->logits ||
        (!r->mamba_state && !r->mamba_state16) || !r->conv_state ||
        !r->key_cache || !r->value_cache ||
        !r->key_scales || !r->value_scales || !r->row_buffer || !r->history) {
        set_error(error, error_size, "not enough RAM (runtime needs about %.2f MiB)",
                  (double)r->bytes_allocated / 1048576.0);
        falcon_runtime_free(r);
        return 0;
    }
    {
        /* RoPE frequencies depend only on the lane index; computing them
         * once here (and cos/sin once per position in falcon_forward)
         * replaces per-head powf/cosf/sinf calls with table lookups while
         * producing bit-identical values. */
        int half = c->head_dim / 2, i;
        if (half > FALCON_ROPE_HALF) {
            set_error(error, error_size, "head_dim exceeds RoPE table");
            falcon_runtime_free(r);
            return 0;
        }
        for (i = 0; i < half; ++i)
            r->rope_freq[i] = powf(c->rope_theta,
                                   -(2.0f * (float)i) / (float)c->head_dim);
        r->rope_position = -1;
    }
    return 1;
}

void falcon_runtime_set_progress(FalconRuntime *runtime,
                                 FalconProgressCallback callback,
                                 void *user) {
    runtime->progress = callback;
    runtime->progress_user = user;
}

void falcon_runtime_reset(FalconRuntime *r, const FalconModel *model) {
    const FalconConfig *c = &model->config;
    int conv_dim = c->mamba_dim + 2 * c->mamba_groups * c->mamba_state;
    int kv_dim = c->n_kv_heads * c->head_dim;
    /* float16 +0 is all zero bits too. */
    if (r->mamba_state)
        memset(r->mamba_state, 0, (size_t)c->n_layers * c->mamba_dim *
               c->mamba_state * sizeof(float));
    else
        memset(r->mamba_state16, 0, (size_t)c->n_layers * c->mamba_dim *
               c->mamba_state * sizeof(uint16_t));
    memset(r->conv_state, 0, (size_t)c->n_layers * conv_dim *
           c->mamba_conv * sizeof(float));
    memset(r->key_cache, 0, (size_t)c->n_layers * c->context * kv_dim);
    memset(r->value_cache, 0, (size_t)c->n_layers * c->context * kv_dim);
    r->io_error = 0;
}

void falcon_runtime_free(FalconRuntime *r) {
    if (!r) return;
    aligned_free(r->x); aligned_free(r->norm); aligned_free(r->q);
    aligned_free(r->k); aligned_free(r->v);
    aligned_free(r->attention); aligned_free(r->attention_out);
    aligned_free(r->projection);
    aligned_free(r->conv_out); aligned_free(r->mamba_out);
    aligned_free(r->ff_gate); aligned_free(r->ff_up);
    aligned_free(r->temporary); aligned_free(r->logits);
    aligned_free(r->mamba_state); aligned_free(r->mamba_state16);
    aligned_free(r->conv_state);
    aligned_free(r->key_cache);
    aligned_free(r->value_cache); aligned_free(r->key_scales);
    aligned_free(r->value_scales);
    aligned_free(r->row_buffer);
    aligned_free(r->history);
    memset(r, 0, sizeof(*r));
}

int falcon_runtime_init_fit(FalconRuntime *r, FalconModel *model,
                            int minimum_context, int step,
                            char *error, size_t error_size) {
    int requested = model->config.context, context = requested;
    if (step < 1) step = 1;
    if (minimum_context < 1) minimum_context = 1;
    if (minimum_context > requested) minimum_context = requested;
    for (;;) {
        model->config.context = context;
        if (falcon_runtime_init(r, model, error, error_size)) return context;
        if (context <= minimum_context) break;
        context -= step;
        if (context < minimum_context) context = minimum_context;
    }
    model->config.context = requested;
    return 0;
}

/* Heap that must stay allocatable after the fit policy has cached every
 * block. Nothing allocates after startup (generation never does, the
 * trace uses sceIo), so this is only a margin for newlib; every 6.5 KiB
 * of it costs one token of context. */
#define FALCON_FIT_HEADROOM (16u * 1024u)
/* Cache blocks the fit policy can track (the region has 187). */
#define FALCON_FIT_MAX_BLOCKS 256

/* Re-initialises the runtime at `context` and tries to allocate a block for
 * every hole with FALCON_FIT_HEADROOM to spare. keep = 1 leaves the
 * blocks allocated (marked in pending) on success. Returns 1 on success;
 * on failure nothing extra stays allocated and the runtime may be freed. */
static int fit_probe(FalconModel *model, FalconRuntime *runtime, int context,
                     uint8_t *pending, int keep) {
    void *probe;
    falcon_runtime_free(runtime);
    model->config.context = context;
    if (!falcon_runtime_init(runtime, model, NULL, 0)) return 0;
    if (!cache_alloc_pending(model, pending)) return 0;
    probe = budget_malloc(FALCON_FIT_HEADROOM);
    if (!probe) {
        cache_drop_pending(model, pending);
        return 0;
    }
    budget_free(probe, FALCON_FIT_HEADROOM);
    if (!keep) cache_drop_pending(model, pending);
    return 1;
}

int falcon_fit_context(FalconModel *model, FalconRuntime *runtime,
                       int max_reduction, int minimum_context,
                       char *reason, size_t reason_size) {
    int context = model->config.context, lowest, bad, good, holes;
    FalconProgressCallback progress = runtime->progress;
    void *progress_user = runtime->progress_user;
    FalconClockCallback clock_us = runtime->clock_us;
    static uint8_t pending[FALCON_FIT_MAX_BLOCKS];
    size_t need;
    holes = falcon_model_uncached_blocks(model);
    if (!holes) {
        set_error(reason, reason_size, "all weight blocks cached");
        return context;
    }
    if (!model->weight_cache) {
        set_error(reason, reason_size, "weight cache off");
        return context;
    }
    if (!model->cache_ram_limited) {
        set_error(reason, reason_size,
                  "%d blocks left out by the cache target", holes);
        return context;
    }
    need = uncached_alloc_bytes(model);
    lowest = context - (max_reduction > 0 ? max_reduction : 0);
    if (lowest < minimum_context) lowest = minimum_context;
    if (lowest >= context) {
        set_error(reason, reason_size,
                  "%d blocks uncached, no context to give up", holes);
        return context;
    }
    if (model->weight_cache_blocks > FALCON_FIT_MAX_BLOCKS) {
        set_error(reason, reason_size, "weight region too large to fit");
        return context;
    }
    /* bad: largest context known not to fit everything; good: smallest
     * context known to fit (0 = none yet). Fitting is monotonic in the
     * context, so a binary search finds the largest fitting context with
     * a handful of runtime re-inits; no block is read until the end. */
    bad = context;
    good = fit_probe(model, runtime, lowest, pending, 0) ? lowest : 0;
    while (good && bad - good > 1) {
        int mid = good + (bad - good) / 2;
        if (fit_probe(model, runtime, mid, pending, 0)) good = mid;
        else bad = mid;
    }
    if (good && fit_probe(model, runtime, good, pending, 1)) {
        int left;
        cache_read_pending(model, pending);
        left = falcon_model_uncached_blocks(model);
        set_error(reason, reason_size,
                  "context %d -> %d caches the last %d blocks (%lu KiB)%s",
                  context, good, holes, (unsigned long)(need / 1024u),
                  left ? "; read error, some still stream" : "");
        context = good;
    } else {
        /* Not reachable within the bound: restore the requested context
         * (it fitted before the probes, and every probe gave its memory
         * back) and stream the remainder as before. */
        falcon_runtime_free(runtime);
        model->config.context = context;
        context = falcon_runtime_init_fit(runtime, model, minimum_context, 16,
                                          NULL, 0);
        if (context)
            set_error(reason, reason_size,
                      "%d blocks (%lu KiB) need more than %d tokens of "
                      "context; streaming them", holes,
                      (unsigned long)(need / 1024u), model->config.context -
                      lowest);
        else
            set_error(reason, reason_size,
                      "runtime could not be restored");
    }
    runtime->progress = progress;
    runtime->progress_user = progress_user;
    runtime->clock_us = clock_us;
    return context;
}

static void rms_norm(float *out, const float *x, const float *weight,
                     int count, float epsilon) {
    int i;
    float mean = 0.0f;
    for (i = 0; i < count; ++i) mean += x[i] * x[i];
    mean = 1.0f / sqrtf(mean / (float)count + epsilon);
    for (i = 0; i < count; ++i) out[i] = x[i] * mean * weight[i];
}
/* exp(a) for a <= 0 only. newlib's expf costs a few hundred cycles and
 * the hot paths (silu, attention softmax) call it ~60k times per token,
 * always with a non-positive argument. Range-reduce to 2^k * 2^f with
 * |f| <= 0.5 and a degree-5 polynomial (max relative error ~1e-7, far
 * below Q4 quantization noise). No FP overflow is possible: the result
 * is in (0, 1], tiny arguments return exactly 0, and 2^k is built from
 * integer exponent bits, which keeps the fpflag probe clean. */
static float fast_exp_neg(float a) {
    float t, f, p, s;
    int k;
    uint32_t bits;
    if (a < -60.0f) return 0.0f;
    t = a * 1.4426950408889634f;      /* log2(e) */
    k = (int)(t - 0.5f);              /* round-to-nearest for t <= 0 */
    f = t - (float)k;                 /* |f| <= 0.5 */
    p = 1.0f + f * (0.69314718056f + f * (0.24022650696f +
        f * (0.05550410866f + f * (0.00961812911f + f * 0.00133335581f))));
    bits = (uint32_t)(127 + k) << 23; /* 2^k, k in [-87, 0] */
    memcpy(&s, &bits, sizeof(s));
    return p * s;
}

static float silu(float x) {
    /* Branch on sign so the exponential never sees a positive argument.
     * On the PSP's Allegrex FPU a float overflow (to infinity) raises an
     * unhandled exception and hard-crashes the console; a PC FPU just
     * returns x/inf = 0. Both branches are the exact silu definition. */
    if (x >= 0.0f) return x / (1.0f + fast_exp_neg(-x));
    {
        float e = fast_exp_neg(x);
        return x * e / (1.0f + e);
    }
}
static float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* Refreshes the per-position cos/sin tables; every layer and head of the
 * same token shares them. Values are bit-identical to computing
 * cosf/sinf((float)position * frequency) in place. */
static void rope_set_position(FalconRuntime *r, int half, int position) {
    int i;
    if (r->rope_position == position) return;
    for (i = 0; i < half; ++i) {
        float angle = (float)position * r->rope_freq[i];
        r->rope_cos[i] = cosf(angle);
        r->rope_sin[i] = sinf(angle);
    }
    r->rope_position = position;
}

static void apply_rope(const FalconRuntime *r, float *values,
                       int heads, int head_dim) {
    int head, i;
    int half = head_dim / 2;
    for (head = 0; head < heads; ++head) {
        float *v = values + head * head_dim;
        for (i = 0; i < half; ++i) {
            float cs = r->rope_cos[i], sn = r->rope_sin[i];
            float first = v[i], second = v[i + half];
            v[i] = first * cs - second * sn;
            v[i + half] = second * cs + first * sn;
        }
    }
}

static void quantize_cache_vector(int8_t *destination, float *scale_out,
                                  const float *source, int count) {
    int i;
    float maximum = 0.0f, scale;
    for (i = 0; i < count; ++i) {
        float a = fabsf(source[i]);
        if (a > maximum) maximum = a;
    }
    scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    *scale_out = scale;
    for (i = 0; i < count; ++i) {
        int value = (int)lrintf(source[i] / scale);
        if (value < -127) value = -127;
        if (value > 127) value = 127;
        destination[i] = (int8_t)value;
    }
}

static int attention_forward(FalconModel *model, FalconRuntime *r,
                             const FalconLayerOffsets *layer, int layer_index,
                             int position) {
    const FalconConfig *c = &model->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int repeats = c->n_heads / c->n_kv_heads;
    int head, kv_head, t;
    size_t current_base = ((size_t)layer_index * c->context + position) * kv_dim;
    size_t current_scale = ((size_t)layer_index * c->context + position) *
                           c->n_kv_heads;
    float inverse_sqrt = 1.0f / sqrtf((float)c->head_dim);
    if (!q4_matvec(model, r, r->q, layer->wq, c->n_heads * c->head_dim,
                   c->dim, r->norm) ||
        !q4_matvec(model, r, r->k, layer->wk, kv_dim, c->dim, r->norm) ||
        !q4_matvec(model, r, r->v, layer->wv, kv_dim, c->dim, r->norm))
        return 0;
    apply_rope(r, r->q, c->n_heads, c->head_dim);
    apply_rope(r, r->k, c->n_kv_heads, c->head_dim);
    for (kv_head = 0; kv_head < c->n_kv_heads; ++kv_head) {
        quantize_cache_vector(r->key_cache + current_base + kv_head * c->head_dim,
                              r->key_scales + current_scale + kv_head,
                              r->k + kv_head * c->head_dim, c->head_dim);
        quantize_cache_vector(r->value_cache + current_base + kv_head * c->head_dim,
                              r->value_scales + current_scale + kv_head,
                              r->v + kv_head * c->head_dim, c->head_dim);
    }
    for (head = 0; head < c->n_heads; ++head) {
        float maximum = -1.0e30f, sum = 0.0f;
        float *out = r->attention_out + head * c->head_dim;
        size_t layer_base = (size_t)layer_index * c->context * kv_dim;
        size_t scale_base = (size_t)layer_index * c->context * c->n_kv_heads;
        kv_head = head / repeats;
#ifdef FALCON_VFPU
        falcon_vfpu_attn_score(r->q + head * c->head_dim,
                               r->key_cache + layer_base +
                                   kv_head * c->head_dim,
                               kv_dim, position + 1, r->attention);
        for (t = 0; t <= position; ++t) {
            float score = r->attention[t] *
                          r->key_scales[scale_base + (size_t)t *
                                        c->n_kv_heads + kv_head] *
                          inverse_sqrt;
            r->attention[t] = score;
            if (score > maximum) maximum = score;
        }
#else
        for (t = 0; t <= position; ++t) {
            const int8_t *key = r->key_cache + layer_base +
                                (size_t)t * kv_dim + kv_head * c->head_dim;
            const float *query = r->q + head * c->head_dim;
            float scale = r->key_scales[scale_base + (size_t)t *
                                        c->n_kv_heads + kv_head];
            float score;
            /* The int8 dot runs with the per-head scale hoisted out of the
             * loop and two partial sums so consecutive FPU madds do not
             * serialize on one accumulator. */
            float even = 0.0f, odd = 0.0f;
            int i;
            for (i = 0; i < c->head_dim; i += 2) {
                even += query[i] * (float)key[i];
                odd += query[i + 1] * (float)key[i + 1];
            }
            score = (even + odd) * scale * inverse_sqrt;
            r->attention[t] = score;
            if (score > maximum) maximum = score;
        }
#endif
        for (t = 0; t <= position; ++t) {
            r->attention[t] = fast_exp_neg(r->attention[t] - maximum);
            sum += r->attention[t];
        }
        /* Turn the softmax numerators into final value weights. */
        for (t = 0; t <= position; ++t)
            r->attention[t] = r->attention[t] / sum *
                              r->value_scales[scale_base + (size_t)t *
                                              c->n_kv_heads + kv_head];
#ifdef FALCON_VFPU
        falcon_vfpu_attn_accum(out,
                               r->value_cache + layer_base +
                                   kv_head * c->head_dim,
                               kv_dim, position + 1, r->attention);
#else
        memset(out, 0, (size_t)c->head_dim * sizeof(float));
        for (t = 0; t <= position; ++t) {
            const int8_t *value = r->value_cache + layer_base +
                                  (size_t)t * kv_dim +
                                  kv_head * c->head_dim;
            float weight = r->attention[t];
            int i;
            for (i = 0; i < c->head_dim; ++i)
                out[i] += weight * (float)value[i];
        }
#endif
    }
    return q4_matvec(model, r, r->temporary, layer->wo, c->dim,
                     c->n_heads * c->head_dim, r->attention_out);
}

/* ---- float16 Mamba state (PROJECT.md section 2.10) ------------------
 * The PSP converts with vh2f/vf2h, which on hardware (measured) truncate
 * toward zero, flush float16 results below 2^-14 to zero and read
 * denormal halves as zero, while PPSSPP rounds and keeps denormals. The
 * shipped mode FALCON_STATE_F16 therefore never lets the hardware decide:
 * the value is first rounded to 11 significant bits with exact float
 * arithmetic (Veltkamp split: c = x * (2^13 + 1), x - ... see below),
 * which makes it exactly representable, and magnitudes below 2^-14 are
 * flushed explicitly, so vf2h only ever sees values it converts exactly
 * on both. These host functions are the bit-exact model of that path.
 * The state magnitude stays far below the float16 maximum (measured and
 * bounded in section 2.10); the store clamps to +-65504 regardless, so no
 * infinity can ever be written. */
#ifndef FALCON_VFPU
#define STATE_F16_MAX 65504.0f
#define STATE_F16_MIN_NORMAL 6.103515625e-05f /* 2^-14 */

static float state_load(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t bits = exponent ? sign | ((exponent + 112u) << 23) |
                                   (((uint32_t)h & 0x3ffu) << 13)
                             : sign; /* vh2f: denormal halves are zero */
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Truncating float -> float16 conversion of a value whose magnitude is
 * in [2^-14, 65504] or below 2^-14 (flushed): the PSP's vf2h. */
static uint16_t state_truncate(float x) {
    uint32_t bits, sign, exponent;
    memcpy(&bits, &x, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exponent = (bits >> 23) & 0xffu;
    if (exponent < 113u) return (uint16_t)sign;        /* < 2^-14: flush */
    return (uint16_t)(sign | ((exponent - 112u) << 10) |
                      ((bits >> 13) & 0x3ffu));
}

static uint16_t state_store(float x, int mode) {
    if (x > STATE_F16_MAX) x = STATE_F16_MAX;
    if (x < -STATE_F16_MAX) x = -STATE_F16_MAX;
    if (mode == FALCON_STATE_F16) {
        /* Round to nearest at 11 significant bits (Veltkamp; exact in
         * IEEE round-to-nearest single precision, no overflow possible
         * at |x| <= 65504), then flush below the smallest normal. */
        volatile float c = x * 8193.0f;
        volatile float t = c - x;
        float hi = c - t;
        if (fabsf(hi) < STATE_F16_MIN_NORMAL) hi = 0.0f;
        return state_truncate(hi);
    }
    if (mode == FALCON_STATE_F16_TRUNC_CORR) {
        volatile float y = x * 1.00048828125f; /* 1 + 2^-11 */
        x = y;
        if (x > STATE_F16_MAX) x = STATE_F16_MAX;
        if (x < -STATE_F16_MAX) x = -STATE_F16_MAX;
    }
    return state_truncate(x);
}
#endif

static int mamba_forward(FalconModel *model, FalconRuntime *r,
                         const FalconLayerOffsets *layer, int layer_index) {
    const FalconConfig *c = &model->config;
    const FalconLayerSmall *small = &model->small[layer_index];
    int conv_dim = c->mamba_dim + 2 * c->mamba_groups * c->mamba_state;
    int h, d, channel, tap;
    const float *dt_bias = small->dt_bias;
    const float *a_log = small->a_log;
    const float *skip = small->d;
    float *gate = r->projection;
    float *conv_input = gate + c->mamba_dim;
    float *dt_input = conv_input + conv_dim;
    float *conv_cache = r->conv_state +
                        (size_t)layer_index * conv_dim * c->mamba_conv;
    size_t state_offset = (size_t)layer_index * c->mamba_dim * c->mamba_state;
    float *state_base = r->mamba_state ? r->mamba_state + state_offset : NULL;
    uint16_t *state16_base = r->mamba_state16 ?
                             r->mamba_state16 + state_offset : NULL;
    if (!q4_matvec(model, r, r->projection, layer->mamba_in,
                   c->mamba_projection, c->dim, r->norm))
        return 0;
    for (channel = 0; channel < conv_dim; ++channel) {
        float *cache = conv_cache + channel * c->mamba_conv;
        float value = small->conv_bias[channel];
        for (tap = 0; tap < c->mamba_conv - 1; ++tap)
            cache[tap] = cache[tap + 1];
        cache[c->mamba_conv - 1] = conv_input[channel];
        for (tap = 0; tap < c->mamba_conv; ++tap)
            value += cache[tap] * small->conv_weight[channel * c->mamba_conv + tap];
        r->conv_out[channel] = silu(value);
    }
    for (h = 0; h < c->mamba_heads; ++h) {
        int group = h / (c->mamba_heads / c->mamba_groups);
        const float *b = r->conv_out + c->mamba_dim + group * c->mamba_state;
        const float *cc = r->conv_out + c->mamba_dim +
                          c->mamba_groups * c->mamba_state +
                          group * c->mamba_state;
        float dt = softplus(dt_input[h] + dt_bias[h]);
        float transition = -expf(a_log[h]);
        float decay = fast_exp_neg(dt * transition);
#ifdef FALCON_VFPU
        {
            /* decay/dt travel through memory because the kernel loads
             * them with lv.s; acc gets one dot per channel back. */
            static float decay_dt[2] __attribute__((aligned(16)));
            static float acc[32] __attribute__((aligned(16)));
            size_t head_offset = (size_t)h * c->mamba_head_dim *
                                 c->mamba_state;
            decay_dt[0] = decay;
            decay_dt[1] = dt;
            if (state16_base)
                falcon_vfpu_mamba_head_f16(state16_base + head_offset,
                                           r->conv_out + h * c->mamba_head_dim,
                                           b, cc, decay_dt, acc);
            else
                falcon_vfpu_mamba_head(state_base + head_offset,
                                       r->conv_out + h * c->mamba_head_dim,
                                       b, cc, decay_dt, acc);
            for (d = 0; d < c->mamba_head_dim; ++d) {
                int index = h * c->mamba_head_dim + d;
                float output = acc[d] + r->conv_out[index] * skip[h];
                r->mamba_out[index] = output * silu(gate[index]);
            }
        }
#else
        for (d = 0; d < c->mamba_head_dim; ++d) {
            int index = h * c->mamba_head_dim + d;
            float hidden = r->conv_out[index];
            float output = 0.0f;
            int s;
            if (state16_base) {
                /* The output uses the updated value before it is rounded
                 * to float16, exactly as the PSP kernel keeps it in
                 * registers. */
                uint16_t *state = state16_base +
                                  (size_t)index * c->mamba_state;
                for (s = 0; s < c->mamba_state; ++s) {
                    float value = state_load(state[s]) * decay +
                                  dt * b[s] * hidden;
                    output += value * cc[s];
                    state[s] = state_store(value, r->state_mode);
                }
            } else {
                float *state = state_base + (size_t)index * c->mamba_state;
                for (s = 0; s < c->mamba_state; ++s) {
                    state[s] = state[s] * decay + dt * b[s] * hidden;
                    output += state[s] * cc[s];
                }
            }
            output += hidden * skip[h];
            r->mamba_out[index] = output * silu(gate[index]);
        }
#endif
    }
    return q4_matvec(model, r, r->attention_out, layer->mamba_out,
                     c->dim, c->mamba_dim, r->mamba_out);
}

/* need_logits 0 skips the final norm and the 32768x512 output-head matvec
 * (about 18% of a token's multiply-accumulates). Prompt tokens other than
 * the last one never sample from their logits, so the head is wasted work
 * for them. Returns r->logits as a non-NULL success marker either way; the
 * buffer only holds valid logits when need_logits was 1. */
static float *forward_internal(FalconModel *model, FalconRuntime *r,
                               int token, int position, int need_logits) {
    const FalconConfig *c = &model->config;
    int layer_index, i;
    unsigned int started;
    if (token < 0 || token >= c->vocab_size ||
        position < 0 || position >= c->context)
        return NULL;
    started = r->clock_us ? r->clock_us() : 0;
    r->io_error = 0;
#ifdef FALCON_PREFETCH
    /* Start reading this token's uncached weight tail from the card now,
     * while the cached-prefix layers compute. */
    falcon_prefetch_token_begin();
#endif
    rope_set_position(r, c->head_dim / 2, position);
    if (!q4_get_row(model, r, r->x, model->embedding, token, c->dim))
        return NULL;
    for (i = 0; i < c->dim; ++i) r->x[i] *= c->embedding_multiplier;
    for (layer_index = 0; layer_index < c->n_layers; ++layer_index) {
        const FalconLayerOffsets *layer = &model->layers[layer_index];
        const FalconLayerSmall *small = &model->small[layer_index];
        if (r->progress)
            r->progress(position, layer_index, c->n_layers, r->progress_user);
        rms_norm(r->norm, r->x, small->input_norm, c->dim, c->rms_eps);
        if (!attention_forward(model, r, layer, layer_index, position) ||
            !mamba_forward(model, r, layer, layer_index))
            return NULL;
        for (i = 0; i < c->dim; ++i)
            r->x[i] += r->temporary[i] + r->attention_out[i];
        rms_norm(r->norm, r->x, small->ffn_norm, c->dim, c->rms_eps);
        if (!q4_matvec(model, r, r->ff_gate, layer->w_gate,
                       c->hidden_dim, c->dim, r->norm) ||
            !q4_matvec(model, r, r->ff_up, layer->w_up,
                       c->hidden_dim, c->dim, r->norm))
            return NULL;
        for (i = 0; i < c->hidden_dim; ++i)
            r->ff_gate[i] = silu(r->ff_gate[i]) * r->ff_up[i];
        if (!q4_matvec(model, r, r->temporary, layer->w_down,
                       c->dim, c->hidden_dim, r->ff_gate))
            return NULL;
        for (i = 0; i < c->dim; ++i) r->x[i] += r->temporary[i];
    }
    if (r->progress)
        r->progress(position, c->n_layers, c->n_layers, r->progress_user);
    if (need_logits) {
        rms_norm(r->norm, r->x, model->final_norm_values, c->dim, c->rms_eps);
        if (!q4_matvec(model, r, r->logits, model->embedding,
                       c->vocab_size, c->dim, r->norm))
            return NULL;
        for (i = 0; i < c->vocab_size; ++i)
            r->logits[i] *= c->lm_head_multiplier;
    }
    if (r->clock_us) r->forward_us += r->clock_us() - started;
    return r->logits;
}

float *falcon_forward(FalconModel *model, FalconRuntime *r,
                      int token, int position) {
    return forward_internal(model, r, token, position, 1);
}

static int falcon_generate_internal(FalconModel *model, FalconRuntime *runtime,
                                    FalconSampler *sampler,
                                    const int *prompt_tokens, int prompt_count,
                                    int start_position, int max_new_tokens,
                                    FalconTokenCallback callback, void *user,
                                    int *generated_count, int *next_position,
                                    int *pending_token) {
    int *history = runtime->history;
    int history_count = 0, i, token, made = 0;
    int position, pending = -1;
    float *logits = NULL;
    const FalconConfig *c = &model->config;
    if (generated_count) *generated_count = 0;
    if (pending_token) *pending_token = -1;
    if (prompt_count <= 0 || max_new_tokens < 0 || start_position < 0 ||
        start_position + prompt_count + max_new_tokens > c->context ||
        !history || prompt_count + max_new_tokens > runtime->history_capacity)
        return 0;
    if (start_position == 0) falcon_runtime_reset(runtime, model);
    position = start_position;
    for (i = 0; i < prompt_count; ++i) {
        history[history_count++] = prompt_tokens[i];
        logits = forward_internal(model, runtime, prompt_tokens[i], position,
                                  i + 1 == prompt_count);
        ++position;
        if (!logits) return 0;
    }
    for (i = 0; i < max_new_tokens; ++i) {
        size_t piece_len = 0;
        const uint8_t *piece;
        int keep_going;
        token = falcon_sample(sampler, logits, c->vocab_size,
                              history, history_count);
        if (token == c->eos_id || token == c->im_end_id ||
            token == c->eom_id || token == c->eot_id)
            break;
        history[history_count++] = token;
        piece = falcon_token_piece(&model->tokenizer, token, &piece_len);
        ++made;
        keep_going = !callback || callback(token, piece, piece_len, user);
        if (!keep_going || i + 1 >= max_new_tokens) {
            /* Token was emitted but never fed through the model; a
             * follow-up turn must prepend it to stay consistent. */
            pending = token;
            break;
        }
        logits = falcon_forward(model, runtime, token, position);
        ++position;
        if (!logits) return 0;
    }
    if (generated_count) *generated_count = made;
    if (next_position) *next_position = position;
    if (pending_token) *pending_token = pending;
    return 1;
}

int falcon_generate(FalconModel *model, FalconRuntime *runtime,
                    FalconSampler *sampler, const int *prompt_tokens,
                    int prompt_count, int max_new_tokens,
                    FalconTokenCallback callback, void *user,
                    int *generated_count) {
    return falcon_generate_internal(model, runtime, sampler, prompt_tokens,
                                    prompt_count, 0, max_new_tokens,
                                    callback, user, generated_count,
                                    NULL, NULL);
}

int falcon_generate_turn(FalconModel *model, FalconRuntime *runtime,
                         FalconSampler *sampler, const int *prompt_tokens,
                         int prompt_count, int start_position,
                         int max_new_tokens,
                         FalconTokenCallback callback, void *user,
                         int *generated_count, int *next_position,
                         int *pending_token) {
    return falcon_generate_internal(model, runtime, sampler, prompt_tokens,
                                    prompt_count, start_position,
                                    max_new_tokens, callback, user,
                                    generated_count, next_position,
                                    pending_token);
}

size_t falcon_model_resident_bytes(const FalconModel *model) {
    return model->io_buffer_bytes + model->tokenizer.section_bytes +
           model->small_bytes + model->weight_cache_bytes;
}
size_t falcon_runtime_resident_bytes(const FalconRuntime *runtime) {
    return runtime->bytes_allocated;
}

