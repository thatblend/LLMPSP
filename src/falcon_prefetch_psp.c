/* PSP-only asynchronous weight prefetcher.
 *
 * The blocks of the weight region that do not fit in the RAM cache are
 * re-read from the memory stick every token, in ascending file order
 * (matrices are stored in execution order). This module reads exactly
 * those uncached extents with sceIoReadAsync into a small double buffer
 * ahead of consumption, so the card transfers overlap the CPU/VFPU
 * compute instead of serializing after it.
 *
 * Two details matter for real hardware:
 * - the async worker thread must run at a higher priority than the
 *   compute thread (sceIoChangeAsyncPriority), otherwise the "async"
 *   read only progresses while the compute thread sleeps -- which it
 *   never does -- and the transfer silently serializes;
 * - with a striped cache (falcon_model_cache_weights_extra stripe=1)
 *   each uncached extent is about one 256 KiB block per layer, so every
 *   extent has a whole layer of compute to hide behind. A prefix cache
 *   concentrates all streaming at the last layers where almost no
 *   compute is left to overlap.
 *
 * All calls happen on the inference thread. Any error permanently
 * disables the module; q4_matvec then falls back to its synchronous
 * fread path, which stays fully correct. */
#include "falcon_h1.h"

#if defined(_PSP_FW_VERSION) && !defined(FALCON_NO_PREFETCH)

#include <pspiofilemgr.h>
#include <pspkernel.h>
#include <stdlib.h>
#include <string.h>

#define PF_CHUNK (256u * 1024u)
#define PF_BUFFERS 3
#define PF_MAX_ROW 1024
#define PF_MAX_EXTENTS 224
/* Above the main thread (0x20), below the power guard (0x18) and the
 * callback thread (0x11). */
#define PF_ASYNC_PRIORITY 0x19

enum { PF_EMPTY = 0, PF_READING = 1, PF_FULL = 2 };

typedef struct {
    uint8_t *data;
    uint32_t offset;
    uint32_t length;
    int state;
} PfBuffer;

typedef struct {
    uint32_t start, end;
} PfExtent;

static SceUID pf_fd = -1;
static int pf_active;
static PfExtent pf_extent[PF_MAX_EXTENTS];
static int pf_extent_count;
static uint32_t pf_first_layer_pos;  /* first uncached byte in the layers */
static uint32_t pf_last_end;         /* end of the final extent */
static PfBuffer pf_buffer[PF_BUFFERS];
static int pf_reading = -1;          /* buffer index with a read in flight */
static int pf_issue_extent;          /* extent the stream is walking */
static uint32_t pf_issue_pos;        /* next file offset to request */
static uint32_t pf_stream_base;      /* offset the current stream began at */
static uint32_t pf_served_end;       /* end of the last row handed out */
static void *pf_raw[PF_BUFFERS];
static uint8_t pf_assemble[PF_MAX_ROW] __attribute__((aligned(16)));
static unsigned int pf_stat_served;  /* bytes served from the buffers */
static unsigned int pf_stat_wait_us; /* time blocked in WaitAsync */

static void pf_disable(void) {
    if (pf_reading >= 0) {
        SceInt64 result;
        sceIoWaitAsync(pf_fd, &result);
        pf_reading = -1;
    }
    pf_active = 0;
}

/* Locates the extent containing offset, or the next one after it. */
static int pf_find_extent(uint32_t offset) {
    int i;
    for (i = 0; i < pf_extent_count; ++i)
        if (offset < pf_extent[i].end) return i;
    return pf_extent_count;
}

static void pf_issue(void) {
    int i;
    if (pf_reading >= 0 || pf_issue_extent >= pf_extent_count) return;
    for (i = 0; i < PF_BUFFERS; ++i) {
        if (pf_buffer[i].state == PF_EMPTY) {
            uint32_t amount = pf_extent[pf_issue_extent].end - pf_issue_pos;
            if (amount > PF_CHUNK) amount = PF_CHUNK;
            if (sceIoLseek32(pf_fd, (int)pf_issue_pos, PSP_SEEK_SET) !=
                    (int)pf_issue_pos ||
                sceIoReadAsync(pf_fd, pf_buffer[i].data, amount) < 0) {
                pf_disable();
                return;
            }
            pf_buffer[i].offset = pf_issue_pos;
            pf_buffer[i].length = amount;
            pf_buffer[i].state = PF_READING;
            pf_reading = i;
            pf_issue_pos += amount;
            if (pf_issue_pos >= pf_extent[pf_issue_extent].end) {
                ++pf_issue_extent;
                if (pf_issue_extent < pf_extent_count)
                    pf_issue_pos = pf_extent[pf_issue_extent].start;
            }
            return;
        }
    }
}

static void pf_restart(uint32_t offset) {
    int i, extent;
    if (pf_reading >= 0) {
        SceInt64 result;
        sceIoWaitAsync(pf_fd, &result);
        pf_reading = -1;
    }
    for (i = 0; i < PF_BUFFERS; ++i) pf_buffer[i].state = PF_EMPTY;
    extent = pf_find_extent(offset);
    if (extent >= pf_extent_count) {
        pf_issue_extent = pf_extent_count;
        pf_stream_base = pf_last_end;
        return;
    }
    pf_issue_extent = extent;
    pf_issue_pos = offset > pf_extent[extent].start ?
                   offset : pf_extent[extent].start;
    pf_stream_base = pf_issue_pos;
    pf_served_end = 0;
    pf_issue();
}

/* Completes a finished read without blocking, tops the pipeline up, and
 * pre-warms the next token's stream once this token consumed the last
 * extent. Safe to restart here: poll() runs at q4_matvec entry, never
 * between a served row pointer and its dot product. */
void falcon_prefetch_poll(void) {
    if (!pf_active) return;
    if (pf_reading >= 0) {
        SceInt64 result;
        int rc = sceIoPollAsync(pf_fd, &result);
        if (rc == 0) {
            PfBuffer *b = &pf_buffer[pf_reading];
            pf_reading = -1;
            if ((uint32_t)result != b->length) {
                pf_disable();
                return;
            }
            b->state = PF_FULL;
        } else if (rc < 0) {
            pf_disable();
            return;
        }
    }
    if (pf_served_end >= pf_last_end && pf_reading < 0) {
        pf_restart(pf_first_layer_pos);
        return;
    }
    pf_issue();
}

static int pf_wait_reading(void) {
    SceInt64 result;
    PfBuffer *b;
    unsigned int started;
    if (pf_reading < 0) return 0;
    b = &pf_buffer[pf_reading];
    started = sceKernelGetSystemTimeLow();
    if (sceIoWaitAsync(pf_fd, &result) < 0 || (uint32_t)result != b->length) {
        pf_reading = -1;
        pf_disable();
        return -1;
    }
    pf_stat_wait_us += sceKernelGetSystemTimeLow() - started;
    pf_reading = -1;
    b->state = PF_FULL;
    return 0;
}

const uint8_t *falcon_prefetch_get(uint32_t offset, uint32_t bytes) {
    int i, guard, extent;
    if (!pf_active || bytes > PF_MAX_ROW) return NULL;
    /* The row must lie inside one uncached extent; rows that straddle a
     * cached/uncached boundary fall back to the caller's fread path. */
    extent = pf_find_extent(offset);
    if (extent >= pf_extent_count ||
        offset < pf_extent[extent].start ||
        offset + bytes > pf_extent[extent].end)
        return NULL;
    falcon_prefetch_poll();
    if (!pf_active) return NULL;
    /* Rows ascend within a stream; anything wholly behind is consumed. */
    for (i = 0; i < PF_BUFFERS; ++i)
        if (pf_buffer[i].state == PF_FULL &&
            pf_buffer[i].offset + pf_buffer[i].length <= offset)
            pf_buffer[i].state = PF_EMPTY;
    if (offset < pf_stream_base)
        pf_restart(offset);
    for (guard = 0; guard < PF_BUFFERS + 4; ++guard) {
        const PfBuffer *a = NULL, *b = NULL;
        for (i = 0; i < PF_BUFFERS; ++i) {
            const PfBuffer *p = &pf_buffer[i];
            if (p->state != PF_FULL) continue;
            if (offset >= p->offset && offset < p->offset + p->length) a = p;
            if (offset < p->offset && offset + bytes > p->offset) b = p;
        }
        if (a) {
            uint32_t inside = offset - a->offset;
            uint32_t first = a->length - inside;
            if (first >= bytes) {
                pf_served_end = offset + bytes;
                pf_stat_served += bytes;
                return a->data + inside;
            }
            if (b) {
                memcpy(pf_assemble, a->data + inside, first);
                memcpy(pf_assemble + first, b->data, bytes - first);
                pf_served_end = offset + bytes;
                pf_stat_served += bytes;
                return pf_assemble;
            }
        }
        /* Not covered yet: finish the in-flight read or start one. If
         * the stream has already advanced past the request (a skipped
         * range), reposition it. */
        if (pf_issue_extent >= pf_extent_count ||
            (pf_reading < 0 &&
             (pf_issue_extent > extent ||
              (pf_issue_extent == extent && pf_issue_pos > offset)) &&
             !a)) {
            pf_restart(offset);
            continue;
        }
        if (pf_reading >= 0) {
            if (pf_wait_reading() < 0) return NULL;
        } else {
            pf_issue();
            if (pf_reading < 0) return NULL;
            if (pf_wait_reading() < 0) return NULL;
        }
    }
    return NULL;
}

void falcon_prefetch_token_begin(void) {
    if (!pf_active) return;
    /* Skip the restart when poll() already pre-warmed this stream. */
    if (pf_stream_base == pf_first_layer_pos && pf_served_end == 0) {
        falcon_prefetch_poll();
        return;
    }
    pf_restart(pf_first_layer_pos);
}

int falcon_prefetch_open(const char *path) {
    int i;
    falcon_prefetch_close();
    for (i = 0; i < PF_BUFFERS; ++i) {
        pf_raw[i] = malloc(PF_CHUNK + 64);
        if (!pf_raw[i]) {
            falcon_prefetch_close();
            return 0;
        }
        pf_buffer[i].data = (uint8_t *)(((uintptr_t)pf_raw[i] + 63u) &
                                        ~(uintptr_t)63u);
        pf_buffer[i].state = PF_EMPTY;
    }
    pf_fd = sceIoOpen(path, PSP_O_RDONLY, 0777);
    if (pf_fd < 0) {
        falcon_prefetch_close();
        return 0;
    }
    /* Without this the async worker may never preempt the busy compute
     * thread and every read serializes; ignore failure (older kernels)
     * since the module still works, just without overlap. */
    sceIoChangeAsyncPriority(pf_fd, PF_ASYNC_PRIORITY);
    return 1;
}

void falcon_prefetch_configure(const FalconModel *model) {
    int i;
    uint32_t layer_start;
    pf_extent_count = 0;
    pf_active = 0;
    if (pf_fd < 0 || !model || !model->weight_cache) return;
    for (i = 0; i < model->weight_cache_blocks; ++i) {
        uint32_t start = model->weights_offset +
                         (uint32_t)((size_t)i *
                                    model->weight_cache_block_bytes);
        uint32_t end = start + (uint32_t)model->weight_cache_block_bytes;
        if (end > model->file_size) end = model->file_size;
        if (model->weight_cache[i]) continue;
        if (pf_extent_count > 0 &&
            pf_extent[pf_extent_count - 1].end == start) {
            pf_extent[pf_extent_count - 1].end = end;
        } else {
            if (pf_extent_count >= PF_MAX_EXTENTS) return;
            pf_extent[pf_extent_count].start = start;
            pf_extent[pf_extent_count].end = end;
            ++pf_extent_count;
        }
    }
    if (!pf_extent_count) return;   /* fully cached: nothing to stream */
    pf_last_end = pf_extent[pf_extent_count - 1].end;
    layer_start = model->layers[0].input_norm;
    i = pf_find_extent(layer_start);
    if (i >= pf_extent_count) {
        /* Only embedding blocks are uncached; stream from the first. */
        pf_first_layer_pos = pf_extent[0].start;
    } else {
        pf_first_layer_pos = layer_start > pf_extent[i].start ?
                             layer_start : pf_extent[i].start;
    }
    pf_active = 1;
    pf_restart(pf_first_layer_pos);
}

void falcon_prefetch_stats(unsigned int *served_kib, unsigned int *wait_ms,
                           int *alive) {
    if (served_kib) *served_kib = pf_stat_served / 1024u;
    if (wait_ms) *wait_ms = pf_stat_wait_us / 1000u;
    if (alive) *alive = pf_active;
}

void falcon_prefetch_stats_reset(void) {
    pf_stat_served = 0;
    pf_stat_wait_us = 0;
}

void falcon_prefetch_close(void) {
    int i;
    pf_disable();
    if (pf_fd >= 0) {
        sceIoClose(pf_fd);
        pf_fd = -1;
    }
    for (i = 0; i < PF_BUFFERS; ++i) {
        free(pf_raw[i]);
        pf_raw[i] = NULL;
        pf_buffer[i].data = NULL;
        pf_buffer[i].state = PF_EMPTY;
    }
    pf_extent_count = 0;
    pf_active = 0;
}

#else
/* Host builds: the prefetcher exists only on the PSP; these stubs keep
 * the shared runtime linkable if anything references the API. */
int falcon_prefetch_open(const char *path) { (void)path; return 0; }
void falcon_prefetch_configure(const FalconModel *model) { (void)model; }
void falcon_prefetch_token_begin(void) {}
void falcon_prefetch_poll(void) {}
const uint8_t *falcon_prefetch_get(uint32_t o, uint32_t b) {
    (void)o; (void)b; return 0;
}
void falcon_prefetch_stats(unsigned int *served_kib, unsigned int *wait_ms,
                           int *alive) {
    if (served_kib) *served_kib = 0;
    if (wait_ms) *wait_ms = 0;
    if (alive) *alive = 0;
}
void falcon_prefetch_stats_reset(void) {}
void falcon_prefetch_close(void) {}
#endif
