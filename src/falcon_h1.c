#include "falcon_h1.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
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
static size_t q4_row_bytes(int cols) {
    return (size_t)(cols / FALCON_QK) * 18u;
}
static size_t q4_bytes(int rows, int cols) {
    return (size_t)rows * q4_row_bytes(cols);
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

static void dequantize_row(float *out, const uint8_t *row, int cols) {
    int block, j;
    for (block = 0; block < cols / FALCON_QK; ++block) {
        const uint8_t *b = row + block * 18;
        float d = half_to_float(rd16(b));
        int base = block * FALCON_QK;
        for (j = 0; j < 16; ++j) {
            uint8_t packed = b[2 + j];
            out[base + j] = d * q4_lut_low[packed];
            out[base + j + 16] = d * q4_lut_high[packed];
        }
    }
}

/* On the PSP the row dot products run on the VFPU. The kernel needs the
 * per-32-column block sums of x (computed once per matvec, amortized over
 * every row) because it accumulates the raw 0..15 nibbles and corrects
 * with -8 * block_sum at the end. */
#if defined(_PSP_FW_VERSION) && !defined(FALCON_SCALAR_ONLY)
#define FALCON_VFPU 1
extern float falcon_dot_q4_vfpu(const uint8_t *row, const float *x,
                                int blocks, const float *block_sums);
extern void falcon_vfpu_attn_score(const float *q, const int8_t *keys,
                                   int stride, int count, float *scores);
extern void falcon_vfpu_attn_accum(float *out, const int8_t *values,
                                   int stride, int count,
                                   const float *weights);
extern void falcon_vfpu_mamba_head(float *state, const float *hidden,
                                   const float *b, const float *cc,
                                   const float *decay_dt, float *acc);
#endif
#if defined(_PSP_FW_VERSION) && !defined(FALCON_NO_PREFETCH)
#define FALCON_PREFETCH 1
#endif

#ifndef FALCON_VFPU
static float dot_q4(const uint8_t *row, const float *x, int cols) {
    int block, j;
    float sum = 0.0f;
    for (block = 0; block < cols / FALCON_QK; ++block) {
        const uint8_t *b = row + block * 18;
        const float *v = x + block * FALCON_QK;
        float d = half_to_float(rd16(b));
        float block_sum = 0.0f;
        for (j = 0; j < 16; ++j) {
            uint8_t packed = b[2 + j];
            block_sum += q4_lut_low[packed] * v[j];
            block_sum += q4_lut_high[packed] * v[j + 16];
        }
        sum += d * block_sum;
    }
    return sum;
}
#endif /* !FALCON_VFPU */

#define FALCON_MAX_BLOCKS 64

static int q4_matvec(FalconModel *model, FalconRuntime *runtime, float *out,
                     uint32_t offset, int rows, int cols, const float *x) {
    size_t row_bytes = q4_row_bytes(cols);
    int row, file_active = 0;
    uint32_t file_pos = 0;
    unsigned int started = runtime->clock_us ? runtime->clock_us() : 0;
#ifdef FALCON_VFPU
    static float block_sums[FALCON_MAX_BLOCKS] __attribute__((aligned(16)));
    int blocks = cols / FALCON_QK, b, j;
    for (b = 0; b < blocks; ++b) {
        const float *v = x + b * FALCON_QK;
        float sum = 0.0f;
        for (j = 0; j < FALCON_QK; ++j) sum += v[j];
        block_sums[b] = sum;
    }
#define DOT_ROW(bytes) falcon_dot_q4_vfpu((bytes), x, blocks, block_sums)
#else
#define DOT_ROW(bytes) dot_q4((bytes), x, cols)
#endif
    if (row_bytes > runtime->row_buffer_bytes) {
        runtime->io_error = 1;
        return 0;
    }
#ifdef FALCON_PREFETCH
    falcon_prefetch_poll();
#endif
    for (row = 0; row < rows; ++row) {
        uint32_t row_offset = offset + (uint32_t)((size_t)row * row_bytes);
        const uint8_t *cached = cache_pointer(model, row_offset, row_bytes);
        if (cached) {
            out[row] = DOT_ROW(cached);
            continue;
        }
        if (cache_copy(model, row_offset, runtime->row_buffer, row_bytes)) {
            out[row] = DOT_ROW(runtime->row_buffer);
            continue;
        }
#ifdef FALCON_PREFETCH
        {
            const uint8_t *ahead =
                falcon_prefetch_get(row_offset, (uint32_t)row_bytes);
            if (ahead) {
                runtime->streamed_bytes += row_bytes;
                out[row] = DOT_ROW(ahead);
                continue;
            }
        }
#endif
        /* Synchronous fallback: read as many consecutive rows as fit in
         * the buffer with one fread (rows of one matrix are contiguous
         * in the file). file_pos tracks the stdio position so rows
         * served from the cache or prefetcher in between can never
         * desynchronize the sequential stream. */
        {
            int chunk = (int)(runtime->row_buffer_bytes / row_bytes);
            int remaining = rows - row, j;
            size_t chunk_bytes;
            if (chunk > remaining) chunk = remaining;
            chunk_bytes = (size_t)chunk * row_bytes;
            if (!file_active || file_pos != row_offset) {
                if (fseek(model->file, (long)row_offset, SEEK_SET) != 0) {
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
            file_pos = row_offset + (uint32_t)chunk_bytes;
            runtime->streamed_bytes += chunk_bytes;
            for (j = 0; j < chunk; ++j)
                out[row + j] =
                    DOT_ROW(runtime->row_buffer + (size_t)j * row_bytes);
            row += chunk - 1;
        }
    }
#undef DOT_ROW
    if (runtime->clock_us)
        runtime->matvec_us += runtime->clock_us() - started;
    return 1;
}

static int q4_get_row(FalconModel *model, FalconRuntime *runtime, float *out,
                      uint32_t matrix, int row, int cols) {
    size_t bytes = q4_row_bytes(cols);
    uint32_t offset = matrix + (uint32_t)((size_t)row * bytes);
    if (bytes > runtime->row_buffer_bytes ||
        !model_read(model, offset, runtime->row_buffer, bytes)) {
        runtime->io_error = 1;
        return 0;
    }
    dequantize_row(out, runtime->row_buffer, cols);
    return 1;
}

static uint32_t advance_q4(uint32_t *offset, int rows, int cols) {
    uint32_t current = *offset;
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
    uint32_t off;
    int i, conv_dim;
    memset(model, 0, sizeof(*model));
    model->file = fopen(path, "rb");
    if (!model->file) {
        set_error(error, error_size, "cannot open %s: %s", path, strerror(errno));
        return 0;
    }
    model->io_buffer_bytes = 64u * 1024u;
    model->io_buffer = (uint8_t *)malloc(model->io_buffer_bytes);
    if (model->io_buffer)
        setvbuf(model->file, (char *)model->io_buffer, _IOFBF, model->io_buffer_bytes);
    if (fread(header, 1, sizeof(header), model->file) != sizeof(header) ||
        rd32(header) != FALCON_MAGIC || rd32(header + 4) != FALCON_VERSION ||
        rd32(header + 8) != FALCON_HEADER_BYTES) {
        set_error(error, error_size, "not a supported FHQ4 model");
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
    model->embedding = advance_q4(&off, model->config.vocab_size, model->config.dim);
    conv_dim = model->config.mamba_dim +
               2 * model->config.mamba_groups * model->config.mamba_state;
    for (i = 0; i < model->config.n_layers; ++i) {
        FalconLayerOffsets *layer = &model->layers[i];
        layer->input_norm = advance_f32(&off, model->config.dim);
        layer->wq = advance_q4(&off, model->config.n_heads * model->config.head_dim,
                               model->config.dim);
        layer->wk = advance_q4(&off, model->config.n_kv_heads * model->config.head_dim,
                               model->config.dim);
        layer->wv = advance_q4(&off, model->config.n_kv_heads * model->config.head_dim,
                               model->config.dim);
        layer->wo = advance_q4(&off, model->config.dim,
                               model->config.n_heads * model->config.head_dim);
        layer->mamba_in = advance_q4(&off, model->config.mamba_projection,
                                     model->config.dim);
        layer->conv_weight = advance_f32(&off, conv_dim * model->config.mamba_conv);
        layer->conv_bias = advance_f32(&off, conv_dim);
        layer->dt_bias = advance_f32(&off, model->config.mamba_heads);
        layer->a_log = advance_f32(&off, model->config.mamba_heads);
        layer->d = advance_f32(&off, model->config.mamba_heads);
        layer->mamba_out = advance_q4(&off, model->config.dim,
                                      model->config.mamba_dim);
        layer->ffn_norm = advance_f32(&off, model->config.dim);
        layer->w_gate = advance_q4(&off, model->config.hidden_dim, model->config.dim);
        layer->w_up = advance_q4(&off, model->config.hidden_dim, model->config.dim);
        layer->w_down = advance_q4(&off, model->config.dim, model->config.hidden_dim);
    }
    model->final_norm = advance_f32(&off, model->config.dim);
    model->file_size = (uint32_t)reported_size;
    if (off != model->file_size) {
        set_error(error, error_size, "model layout mismatch (%lu != %lu)",
                  (unsigned long)off, (unsigned long)model->file_size);
        falcon_model_close(model);
        return 0;
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
        uint8_t *memory = (uint8_t *)malloc(block_bytes);
        if (!memory) {
            alloc_failed = 1;
            break;
        }
        pool[pool_count++] = memory;
    }
    if (alloc_failed && pool_count > extra_blocks) {
        /* Give later prompt/history allocations one block of headroom. */
        free(pool[--pool_count]);
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
            free(pool[pool_used]);
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

void falcon_model_close(FalconModel *model) {
    int block;
    if (!model) return;
    if (model->file) fclose(model->file);
    for (block = 0; block < model->weight_cache_blocks; ++block)
        if (!cache_block_is_extra(model, model->weight_cache[block]))
            free(model->weight_cache[block]);
    free(model->weight_cache);
    free(model->io_buffer);
    free(model->small_block);
    free(model->tokenizer.section);
    memset(model, 0, sizeof(*model));
}

/* All runtime arrays are 16-byte aligned so the PSP VFPU kernel can use
 * quad loads on the activation vectors. The base pointer returned by
 * malloc is stored immediately before the aligned block. */
static void *alloc_array(FalconRuntime *runtime, size_t count, size_t size) {
    size_t bytes, total;
    void *raw;
    uint8_t *aligned;
    if (count && size > ((size_t)-1 - 32) / count) return NULL;
    bytes = count * size;
    total = bytes + 16 + sizeof(void *);
    raw = malloc(total);
    if (!raw) return NULL;
    aligned = (uint8_t *)(((uintptr_t)raw + sizeof(void *) + 15u) & ~(uintptr_t)15u);
    ((void **)aligned)[-1] = raw;
    memset(aligned, 0, bytes);
    runtime->bytes_allocated += total;
    return aligned;
}

static void aligned_free(void *memory) {
    if (memory) free(((void **)memory)[-1]);
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
#define AF(NAME, COUNT) r->NAME = (float *)alloc_array(r, (COUNT), sizeof(float))
    AF(x, c->dim); AF(norm, c->dim); AF(q, c->n_heads * c->head_dim);
    AF(k, kv_dim); AF(v, kv_dim); AF(attention, c->context);
    AF(attention_out, c->dim); AF(projection, c->mamba_projection);
    AF(conv_out, conv_dim); AF(mamba_out, c->mamba_dim);
    AF(ff_gate, c->hidden_dim); AF(ff_up, c->hidden_dim);
    AF(temporary, c->dim); AF(logits, c->vocab_size);
    AF(mamba_state, mamba_values); AF(conv_state, conv_values);
    AF(key_scales, kv_scales); AF(value_scales, kv_scales);
#undef AF
    r->key_cache = (int8_t *)alloc_array(r, kv_values, sizeof(int8_t));
    r->value_cache = (int8_t *)alloc_array(r, kv_values, sizeof(int8_t));
    /* Sized for dozens of rows, not one: the streaming fallback reads
     * whole row batches per fread, which cuts ~20k stdio calls per token
     * to a few hundred. Must stay >= the largest single row (432 B). */
    r->row_buffer_bytes = 32u * 1024u;
    r->row_buffer = (uint8_t *)alloc_array(r, r->row_buffer_bytes, 1);
    if (!r->x || !r->norm || !r->q || !r->k || !r->v || !r->attention ||
        !r->attention_out || !r->projection || !r->conv_out || !r->mamba_out ||
        !r->ff_gate || !r->ff_up || !r->temporary || !r->logits ||
        !r->mamba_state || !r->conv_state || !r->key_cache || !r->value_cache ||
        !r->key_scales || !r->value_scales || !r->row_buffer) {
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
    memset(r->mamba_state, 0, (size_t)c->n_layers * c->mamba_dim *
           c->mamba_state * sizeof(float));
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
    aligned_free(r->mamba_state); aligned_free(r->conv_state);
    aligned_free(r->key_cache);
    aligned_free(r->value_cache); aligned_free(r->key_scales);
    aligned_free(r->value_scales);
    aligned_free(r->row_buffer);
    memset(r, 0, sizeof(*r));
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
    float *state_base = r->mamba_state +
                        (size_t)layer_index * c->mamba_dim * c->mamba_state;
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
            decay_dt[0] = decay;
            decay_dt[1] = dt;
            falcon_vfpu_mamba_head(state_base +
                                       (size_t)h * c->mamba_head_dim *
                                       c->mamba_state,
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
            float *state = state_base + (size_t)index * c->mamba_state;
            int s;
            for (s = 0; s < c->mamba_state; ++s) {
                state[s] = state[s] * decay + dt * b[s] * hidden;
                output += state[s] * cc[s];
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
    int *history;
    int history_count = 0, i, token, made = 0;
    int position, pending = -1;
    float *logits = NULL;
    const FalconConfig *c = &model->config;
    if (generated_count) *generated_count = 0;
    if (pending_token) *pending_token = -1;
    if (prompt_count <= 0 || max_new_tokens < 0 || start_position < 0 ||
        start_position + prompt_count + max_new_tokens > c->context)
        return 0;
    history = (int *)malloc((size_t)(prompt_count + max_new_tokens) * sizeof(int));
    if (!history) return 0;
    if (start_position == 0) falcon_runtime_reset(runtime, model);
    position = start_position;
    for (i = 0; i < prompt_count; ++i) {
        history[history_count++] = prompt_tokens[i];
        logits = forward_internal(model, runtime, prompt_tokens[i], position,
                                  i + 1 == prompt_count);
        ++position;
        if (!logits) {
            free(history);
            return 0;
        }
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
        if (!logits) {
            free(history);
            return 0;
        }
    }
    free(history);
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

