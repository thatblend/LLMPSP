# LLMPSP Project Guide

## 1. What this project is

LLMPSP is an offline text-generation homebrew for the PSP-2000 and
PSP-3000. It runs the 90-million-parameter
tiiuae/Falcon-H1-Tiny-90M-Instruct model locally. It does not call a network
service.

The project has four major parts:

1. A Python converter that turns the official BF16 SafeTensors checkpoint into
   a compact, row-addressable FHQ4 file.
2. A portable C inference runtime implementing Falcon-H1 attention, recurrent
   Mamba state, the feed-forward network, tokenization, and sampling.
3. A PSP frontend with a controller-driven keyboard and a chat display.
4. A PSPDEV build and packaging pipeline that produces a folder ready for
   ms0:/PSP/GAME.

The implementation is intentionally fixed to Falcon-H1-Tiny-90M-Instruct.
Supporting a different Falcon-H1 size requires checking every dimension,
tensor name, and state layout.

The app is named LLMPSP. Public release v1.0 corresponds to the internal
v1.6 build and public v1.1 to internal v1.11; the internal version
numbers in sections 2.1-2.7 are kept because they document the
hardware-only bugs and the performance measurements in order.

## 2. Current status

The following are verified:

- The official checkpoint converts to a 52,244,480-byte FHQ4 file.
- The portable C code compiles with warnings treated as errors.
- Tokenizer and chat-template regression vectors match the official tokenizer.
- End-to-end greedy generation works on the host runtime.
- Cached and uncached host inference produce the same token sequence.
- The host runtime is clean under AddressSanitizer and UBSan for the full
  "hi" chat scenario with cache 32 and cache 0.
- PSPDEV GCC 15.2 builds the EBOOT without compiler warnings, with
  ICON0.PNG embedded as the XMB thumbnail.
- The generated PARAM.SFO requests MEMSIZE=1 (full high-memory unlock).
- The real MIPS EBOOT completes the full "hi" scenario in PPSSPP headless
  (24 MiB user memory), with cache back-off (32 -> 13.25 MiB) and with
  cache 0, producing "Hello! How can I help you today".
- The v1.7 performance build (section 2.3) reproduces the same "hi"
  reply in PPSSPP headless with the VFPU kernel, both with the full
  cache (heap + volatile memory) and with cache 0, and the host build
  passes tokenizer, multiturn, cache-equivalence, and fpflag checks.

### 2.1 Root cause of the v1.1 on-device power-off (fixed in v1.2)

The v1.1 build linked `-lpspkernel` (added to arm the PSPSDK exception
dump handler). That library provides import stubs for kernel-only
libraries, and at link time those stubs silently SHADOWED newlib and the
user syscall libraries. In the shipped v1.1 ELF:

- `memcpy`, `memset`, `memcmp`, `memmove`, `strlen`, `strncpy`,
  `strrchr`, `snprintf` bound to SysclibForKernel import stubs;
- `sceKernelCreateThread/StartThread/DelayThread/CreateCallback/
  SleepThreadCB` bound to ThreadManForKernel;
- `sceKernelRegisterExitCallback`/`sceKernelExitGame` bound to
  LoadExecForKernel, `sceKernelSelfStopUnloadModule` to ModuleMgrForKernel;
- newlib's own `__init_cwd` called the stubbed `strrchr`.

A user-mode PRX cannot legitimately import kernel libraries. Depending on
what the CFW resolves, each such call either works, silently returns
garbage, or faults in kernel context. Because the exception path itself
was broken, a fault produced no register dump: the console simply powered
off. This also explains why editing cache_mb.txt and toggling ARK extra
RAM changed nothing. The stub behavior (`strrchr` returning its argument,
`sceKernelCreateCallback` returning garbage) was reproduced and confirmed
in PPSSPP against the exact shipped ELF.

The v1.2 build removes `-lpspkernel` and the exception-handler call. Both
EBOOTs now import user libraries only (verified with psp-objdump: no
*ForKernel entries, all libc symbols are real newlib code). In addition
v1.2 adds a suspend lock, a power-event callback, and a persistent
on-card trace (section 11).

### 2.2 Root cause of the v1.2/v1.3 deterministic power-off (fixed in v1.4)

With the link bug fixed, on-device runs completed full tokens but the
console froze and then powered itself off at the same place every run
(prompt position 3, layer 5, on AC power, no power events in the trace).
Host instrumentation with fetestexcept() showed the first float OVERFLOW
of the whole run occurs exactly there: silu() computed
`x / (1 + expf(-x))`, and for strongly negative gate activations
(x < -88) `expf(-x)` overflows to infinity. A PC FPU shrugs (x/inf = 0)
and PPSSPP uses the host FPU, but the PSP's Allegrex FPU raises an
exception on the overflow; with no handler the console hangs about ten
seconds and shuts down. Underflow events, by contrast, were measured to
occur earlier (positions 0-2) and are tolerated by the hardware, matching
the psdevwiki description of silent flush-to-zero for denormals.

v1.4 fixes silu() with the numerically stable two-branch sigmoid (never
exponentiates a positive argument; identical math, verified identical
host output and zero overflow events over the full scenario), and the PSP
frontends additionally clear all FPU exception-enable bits and set the
FCR31 FS (flush-to-zero) bit at startup as defense in depth. The
tools/fpflag_probe.c host tool reproduces the measurement. When changing
math code, keep expf/logf arguments bounded and re-run that probe: any
OVERFLOW/INVALID event on the host is a crash on real hardware.

### 2.3 The v1.7 performance pass

Motivation: on-device generation ran at about 0.3 tokens/second. The
changes below target the two dominant costs measured by the new [perf]
instrumentation: scalar Q4 matvec compute (~60-70%) and re-reading the
uncached weight tail from the memory stick every token (~20-30%).

1. VFPU Q4 kernel (src/falcon_q4_psp.c, PSP builds only). Unpacks the
   packed nibbles with integer ops, expands them with vc2i/vi2f, and
   multiply-accumulates 4 lanes at a time. It accumulates raw 0..15
   nibbles and corrects with -8 * (per-32-column block sums of x) that
   q4_matvec computes once per matrix. Expected ~2.5-3x on the matvecs.
   Results are not bit-identical to the scalar kernel (summation order),
   so correctness is verified end to end: the PPSSPP run of the PSP
   binary must reproduce the host reply, and cache 0 must equal cached.
   Host builds keep the scalar kernel unchanged as the golden reference.
2. Volatile-memory cache extension. The 4 MiB volatile partition
   (sceKernelVolatileMemTryLock, a retail user-mode API in
   sceSuspendForUser, available on every CFW) is added to the weight
   cache ahead of the heap blocks via
   falcon_model_cache_weights_extra(). With the default 44 MiB heap
   target this caches the embedding plus 23 of 24 layers; only about
   1.7 MiB streams per token instead of 5.3. If the lock fails the app
   runs exactly as before. cache_mb.txt 0 skips the lock so the
   documented minimum-memory troubleshooting mode stays minimal. The
   cache_mb.txt ceiling is now 46 (default still 44); the allocator
   backs off gracefully as before.
3. Output-head skip during prompt evaluation. The 32768x512 logits
   matvec (~18% of a token's MACs) now runs only for the last prompt
   token and generated tokens; earlier prompt positions never sample.
4. RoPE tables. Frequencies are computed once at runtime init and the
   cos/sin pair once per token position (they do not depend on layer or
   head), replacing 7,680 powf/cosf/sinf calls per token with 32.
   Bit-identical values.
5. Attention int8 dot: per-head scale hoisted out of the inner loop and
   two partial sums so consecutive FPU madds do not serialize (not
   bit-identical; covered by the same end-to-end checks).
6. Frontend overhead: the screen composes/blits 4x per token instead of
   24x, and the synced trace heartbeat runs every 8th token instead of
   every token (a synced card write costs tens of ms on some sticks).
   Per-layer scePowerTick and Circle polling are unchanged.
7. -fno-math-errno in CFLAGS (inlines sqrtf as the sqrt.s instruction;
   never add full -ffast-math, see the Makefile comment and section 2.2).
8. Instrumentation: FalconRuntime gains an optional clock_us hook plus
   forward_us / matvec_us / streamed_bytes counters; both frontends log
   a "[perf] forward X ms, matvec Y ms, streamed Z KiB" line per turn.
   Read it on-device before tuning further.

All host outputs are byte-identical to the pre-change build for the
regression prompts, and the fpflag probe reports zero OVERFLOW/INVALID/
DIVBYZERO events (only the tolerated underflows). tools/host_check.sh
rebuilds and reruns the whole host suite.

First hardware run of v1.7 (PSP-2000, memory stick at ~7 MB/s): reply
speed rose from ~0.3 to ~0.55-0.7 tokens/second. The trace showed the
heap tops out near 40 MiB for the cache on that unit (target 44 and 46
behave identically), so ~5.6 MiB still streamed per token -- roughly
half the remaining token time. That motivated v1.8.

### 2.4 v1.8: asynchronous prefetch and llmpsp.cfg

1. Asynchronous weight prefetch (src/falcon_prefetch_psp.c, PSP only).
   A private sceIoOpen descriptor plus two 512 KiB buffers stream the
   uncached weight tail with sceIoReadAsync while the cached-prefix
   layers compute, so the memory stick works in parallel with the
   CPU/VFPU. falcon_forward kicks the stream at every token;
   q4_matvec polls/serves rows from the buffered window and falls back
   to its synchronous fread path whenever the prefetcher cannot serve
   (any I/O error permanently disables the module, so correctness never
   depends on it). After the tail is consumed the stream restarts early
   so the next token's data loads during the output head. Expected
   effect: token time approaches max(compute, I/O) instead of their
   sum -- around 1.0-1.1 s/token on the measured hardware. Verified in
   PPSSPP with the default cache and with cache 0 (which exercises the
   backward-jump restart when the output head revisits the embedding):
   both produce the reference reply.
2. llmpsp.cfg beside the EBOOT replaces scattered config files: keys
   cache_mb, cpu_mhz, volatile_mem, prefetch, temperature, top_p,
   top_k, repetition_penalty, max_reply_tokens ("key = value", '#'
   comments, everything clamped to safe ranges). Legacy cache_mb.txt /
   cpu_mhz.txt still work; llmpsp.cfg overrides them. temperature 0
   keeps the deterministic greedy default; volatile_mem = 0 and
   prefetch = 0 exist as troubleshooting switches (one on-device quit
   produced a power-off with everything enabled; if that recurs, test
   volatile_mem = 0 first). The active values are traced as a [cfg]
   line at boot.
3. The "Done" status now reports reply-only tokens/second (matching the
   live counter, which starts at the first generated token), and the
   [perf] trace line splits prompt and reply milliseconds.

### 2.5 v1.9: why v1.8 gained nothing, and the striped cache

On-device v1.8 measured no improvement (matvec ~1.6 s/token, ~6.6 MiB
still streamed serially). Two root causes:

1. The sceIoReadAsync worker thread runs at a priority below the main
   thread, so the "asynchronous" read only progressed while the compute
   thread slept -- which it never does. v1.9 calls
   sceIoChangeAsyncPriority(fd, 0x19) (above main 0x20, below the power
   guard 0x18) so the transfer genuinely overlaps compute.
2. The prefix cache concentrates all streamed bytes in the last layers,
   a ~6.6 MiB burst at the end of the token with almost no compute left
   to hide it behind; a ~0.75 MiB buffer can only pre-load a fraction.
   v1.9 stripes the cache instead (falcon_model_cache_weights_extra
   stripe=1): the embedding stays fully cached and the uncached blocks
   are spread evenly across the layer region, about one 256 KiB hole
   per layer, each hidden behind that layer's own compute. Same cached
   and streamed byte counts as the prefix layout.

The prefetcher walks the uncached extents (falcon_prefetch_configure
scans the sparse cache index) with three 256 KiB buffers, and the
[perf] trace now proves what happened on-device:

    [perf] prefetch served X KiB, waited Y ms, alive Z

served ~= streamed means the async path carried the I/O; "waited" is
the un-overlapped remainder (the number that should approach zero);
alive 0 means an I/O error disabled the module and q4_matvec fell back
to synchronous freads (always correct, v1.7 speed). If on-device waits
stay near the full streaming time even in v1.9, the card path is
CPU-bound in the FAT driver and the remaining wins are fewer streamed
bytes (context = 256 in llmpsp.cfg frees ~1.7 MiB of KV RAM for cache)
plus the compute levers below.

llmpsp.cfg additionally gains `context` (128-512, default 512): smaller
contexts shrink the KV cache before the weight cache fills, so the
freed RAM becomes weight cache automatically.

Verified: host suite unchanged and byte-identical, striped-cache
equivalence at two sizes on the host, PPSSPP autotest with striping
(54 holes, 99.8% of streamed bytes served by the prefetcher, reply
identical) and with cache 0 (prefetcher inactive, fallback correct).

### 2.6 v1.9 hardware result: async I/O cannot overlap; defaults reverted

The on-device v1.9 trace was conclusive: the prefetcher served 99.8% of
all streamed bytes with 0 ms of blocking, stayed alive -- and the token
time got WORSE (matvec ~1.6 -> ~2.2 s/token). Both facts together mean
the memory-stick driver performs its transfers on the CPU (FAT layer
plus PIO-style copies), not via background DMA: raising the async
worker's priority made data "ready in time" only by preempting the
compute thread, so I/O and compute still strictly serialize on this
hardware, and the striped layout additionally replaced one fast
sequential scan with ~26 scattered seeks (effective throughput dropped
roughly 8 -> 4.6 MB/s). Asynchronous overlap is a dead end on the stock
memory-stick path; treat card time as CPU time.

Consequences (v1.10):

- prefetch and stripe default to 0; the prefix cache layout and the
  synchronous stream are back (the fastest measured configuration).
  Both remain available in llmpsp.cfg for experiments on other storage
  (an ef0: internal-storage PSP Go may behave differently).
- The synchronous fallback now reads whole row batches per fread
  through a 32 KiB row buffer (a few hundred stdio calls per token
  instead of ~20k), and the stdio buffer shrank 256 -> 64 KiB, freeing
  RAM for the cache. With prefetch off its 768 KiB of buffers are never
  allocated, which is roughly three more cached blocks than v1.9.
- Because streamed bytes now cost CPU-equivalent time at ~8 MB/s, the
  cheapest large speed knob left is streaming fewer bytes:
  context = 256 in llmpsp.cfg frees ~1.7 MiB of KV state that becomes
  weight cache automatically (about -220 ms/token on the measured
  card, at the cost of shorter conversations).

### 2.7 v1.11: compute pass (fast exp, inline scales, VFPU attention/Mamba)

With card I/O proven serial, v1.11 attacks the compute side:

1. fast_exp_neg in falcon_h1.c: silu, the attention softmax, and the
   Mamba decay only ever exponentiate non-positive arguments, so a
   range-reduced degree-5 polynomial (~1e-7 relative error, no FP
   overflow possible, tiny arguments return exactly 0) replaces ~60k
   newlib expf calls per token. Shared by host and PSP builds; the host
   regression outputs stayed token-identical and the fpflag probe is
   clean.
2. The Q4 VFPU kernel converts each block's fp16 scale inline with
   integer ops instead of a branchy per-row C pre-pass that ran ~2.6M
   times per token (worth ~150 ms/token alone). Denormal fp16 scales
   flush to zero: the shipped model has 5 among 2.8M (scanned by
   build/bench/scan_scales.py), all in blocks whose weights are
   numerically irrelevant; zero scales convert exactly.
3. New VFPU kernels for the int8 attention score/accumulate loops
   (whose cost grows with context) and the Mamba state update, in
   src/falcon_q4_psp.c. tools/vfpu_unittest_psp.c +
   tools/Makefile.vfpuunit build a standalone EBOOT that checks every
   kernel against scalar references in PPSSPP (worst relative errors
   1e-5..1e-7, including the zero/denormal scale contract). Because the
   Mamba state is recurrent, the rounding drift can flip an occasional
   greedy token versus the host build (the PPSSPP "hi" reply became
   "Hello! It's nice to hear" -- coherent, kernels unit-verified);
   cached and cache-0 runs still match each other exactly.
4. The on-screen tokens/second (live and final) previously counted the
   first token as if it were free, inflating early readings up to 2x
   (the reported jump from ~1.2 to ~0.5 t/s during a reply was this
   artifact, not real slowdown). Both now divide token intervals by
   elapsed time and agree.
5. llmpsp.cfg is renamed llmpsp_config.cfg, and cache_mb.txt is no
   longer shipped (the code still honors it and cpu_mhz.txt when
   present, and the config file overrides them).

Remaining speed levers: tighter instruction scheduling in the VFPU
kernels and streaming fewer bytes (context = 256). A speculative future
option is storing the Mamba recurrent state as float16 (frees 2.25 MiB
for cache) at some quality risk.

## 3. Quick start

The runnable PSP files are kept at the repository root:

    EBOOT.PBP
    model.fhq4
    llmpsp_config.cfg

Copy those three files to:

    ms0:/PSP/GAME/LLMPSP/

Use a PSP-2000 or PSP-3000 running CFW. A PSP-1000 has only 32 MiB of physical
RAM and is not supported.

Settings live in llmpsp_config.cfg (see sections 2.4-2.7); the defaults
are right for most consoles. If prompt loading powers off or crashes,
set cache_mb = 0 in llmpsp_config.cfg for the minimum-memory mode.
Cache size and prefetch affect performance only. They must not change
generated tokens.

## 4. Repository map

| Path | Purpose |
| --- | --- |
| include/falcon_h1.h | Public model, runtime, tokenizer, sampler, and callback API |
| src/falcon_h1.c | FHQ4 loader, cache, Q4 kernels, attention, Mamba, MLP, generation |
| src/falcon_q4_psp.c | PSP-only VFPU kernel for the Q4 row dot product |
| src/falcon_prefetch_psp.c | PSP-only asynchronous weight-tail prefetcher |
| llmpsp_config.cfg | User configuration (cache, clock, sampling, context) |
| tools/host_check.sh | Rebuilds and reruns the whole host regression suite |
| src/falcon_tokenizer.c | Falcon byte-level BPE and instruct prompt construction |
| src/falcon_sampler.c | Greedy, repetition penalty, top-k, and top-p sampling |
| src/main_psp.c | PSP threads, power guard, trace, input, and generation flow |
| src/falcon_ui.c | Transcript buffer, word wrap, and screen composition |
| include/falcon_ui.h | Screen grid layout constants and UI state |
| tools/convert_falcon_h1.py | Official-checkpoint downloader and FHQ4 converter |
| tools/falcon_host_cli.c | Portable desktop inference test program |
| tools/tokenizer_test.c | Exact tokenizer/chat-template regression vectors |
| tools/fpflag_probe.c | Host FPU-flag probe that predicts on-device FPU crashes |
| tools/multiturn_test.c | Proves incremental multi-turn equals full re-evaluation |
| tools/ui_preview.c | Renders the PSP screen grid on the host for layout work |
| tools/main_psp_autotest.c | No-input on-device self-test frontend |
| tools/Makefile.autotest | Builds the FalconAutoTest self-test EBOOT |
| tools/make_icon.py | Generates ICON0.PNG, the XMB thumbnail |
| Makefile | PSPDEV EBOOT and package build |
| EBOOT.PBP | Current PSP executable |
| model.fhq4 | Current converted PSP model |
| README.md | User-facing build and installation instructions |
| PROJECT.md | Architecture and maintainer handoff |
| MODEL_NOTICE.txt | Falcon model attribution and license notice |

Downloaded BF16 checkpoints, compiler objects, and generated packages are not
kept in the clean tree. The converted PSP model at the root is canonical.

## 5. Hardware and memory constraints

A PSP-2000/3000 has 64 MiB of physical RAM, but an application cannot safely
treat all 64 MiB as heap. Firmware, the user partition, executable, stacks,
libraries, and allocations all consume part of it.

The EBOOT uses:

    PSP_FW_VERSION = 660
    PSP_LARGE_MEMORY = 1
    PSP_HEAP_SIZE_KB(-2048)

PSPSDK writes MEMSIZE=1 into PARAM.SFO, the full high-memory unlock that
ARK-4, PRO, and ME all honor for homebrew.

Measured or deterministic memory costs are approximately:

| Item | Resident memory |
| --- | ---: |
| Recurrent/KV/scratch runtime | 8.19 MiB |
| Tokenizer plus stdio buffer | 0.75 MiB |
| Default cached weights (heap) | up to 44 MiB |
| Volatile-partition cache extension | 4 MiB when granted |
| EBOOT payload | about 0.24 MiB |
| FHQ4 file on microSD | 49.82 MiB |

The cache is allocated only after the required runtime state succeeds. Cache
blocks are 256 KiB. Allocation stops safely if the requested target is not
available and releases one block of heap headroom after a ceiling failure.

The default 44 MiB heap target plus the 4 MiB volatile partition caches
the embedding plus about 23 of 24 layers when the firmware grants full
high memory; the allocator backs off gracefully on smaller layouts.
cache_mb (llmpsp_config.cfg) is clamped to 0 through 46 MiB (heap only;
the volatile 4 MiB is added automatically unless the value is 0).

## 6. Why weights are partly cached

The FHQ4 weight region is about 49.33 MiB. Fully resident weights plus the
8.94 MiB model/runtime tables already exceed a safe PSP application budget
before executable and system overhead.

Weights are stored in execution order:

1. tied token embedding/output matrix;
2. layer 0 tensors;
3. layer 1 tensors;
4. continuing through layer 23;
5. final RMSNorm.

The cache stores a contiguous prefix. This prioritizes the tied embedding,
which is used both to read the input token and to calculate all output logits,
then caches as many early layers as fit.

Every model read checks the cache first. If a matrix crosses the cache
boundary, cached rows are used directly and the remainder is read with one
sequential file seek. With cache 0, the entire model streams from microSD.
With cache 44 plus the volatile partition, only about 1 late layer
streams from the card (about 1.7 MiB per token).

## 7. Falcon-H1 Tiny architecture

The exact configuration accepted by the converter is:

| Parameter | Value |
| --- | ---: |
| Vocabulary | 32,768 |
| Hidden dimension | 512 |
| Layers | 24 |
| Attention query heads | 8 |
| Attention KV heads | 2 |
| Attention head dimension | 64 |
| MLP intermediate dimension | 768 |
| Mamba channels | 768 |
| Mamba heads | 24 |
| Mamba head dimension | 32 |
| Mamba state dimension | 64 |
| Mamba groups | 1 |
| Mamba convolution width | 4 |
| Mamba input projection | 1,688 |
| Model training context | 262,144 |
| PSP runtime context | 512 |
| RoPE theta | 100,000,000,000 |
| RMSNorm epsilon | 0.00001 |

The PSP context is intentionally capped at 512 because attention KV memory
grows with context length. Mamba recurrent memory does not grow with context,
but the parallel attention branch does.

### 7.1 One hybrid layer

For input vector x:

1. n = RMSNorm(x)
2. a = grouped-query causal attention(n)
3. m = recurrent Mamba(n)
4. y = x + a + m
5. f = RMSNorm(y)
6. h = SiLU(gate_projection(f)) multiplied elementwise by up_projection(f)
7. output = y + down_projection(h)

Attention and Mamba run in parallel from the same normalized input. Their
outputs are added to the same residual before the feed-forward block.

### 7.2 Attention branch

The attention branch projects:

- 8 query heads;
- 2 key heads;
- 2 value heads.

Each KV head is shared by four query heads. RoPE uses the rotate-half layout
with 64-dimensional heads and the configured theta.

The current token's keys and values are quantized to signed int8 separately
for each KV head. One float scale per head is stored for keys and values.
Attention scores dequantize cache values during dot products. This saves
roughly half the RAM of a float16 KV cache while retaining a float scale per
head.

### 7.3 Mamba branch

The input projection produces three logical regions:

    gate:        768
    conv/B/C:    896
    dt:           24
    total:      1,688

The 896 convolution channels contain:

    hidden: 768
    B:       64
    C:       64

Each layer keeps:

- a 4-sample depthwise convolution history for 896 channels;
- a recurrent state of 768 by 64 floats.

For each Mamba head:

    delta = softplus(dt_input + dt_bias)
    A = -exp(A_log)
    decay = exp(delta * A)

For each channel and state component:

    state = state * decay + delta * B * hidden
    output = sum(state * C) + D * hidden
    output = output * SiLU(gate)

The 768-channel result is projected back to 512.

## 8. Quantization and FHQ4 format

FHQ4 is a project-specific, little-endian format designed for direct PSP file
access. It is not GGUF.

### 8.1 Q4_0 matrix blocks

Every dense matrix row is divided into groups of 32 values. Each group uses:

- one IEEE float16 scale: 2 bytes;
- 32 signed 4-bit values packed into 16 bytes.

Total:

    18 bytes for 32 weights

Rows remain independently addressable. The runtime can seek directly to one
embedding row or stream a complete matrix without parsing metadata.

Small tensors stay float32:

- RMSNorm weights;
- depthwise convolution weights and bias;
- Mamba dt_bias;
- Mamba A_log;
- Mamba D.

### 8.2 File sections

The first 256 bytes are the FHQ4 header. It contains:

- magic FHQ4 and format version 1;
- file, tokenizer, and weight offsets;
- 25 architecture/token fields;
- RoPE theta, RMS epsilon, embedding multiplier, and LM-head multiplier.

The TOK2 tokenizer section follows. It contains:

- piece offsets and raw byte pieces;
- token IDs sorted by piece for binary lookup;
- merge rank by token ID;
- a byte-to-token table.

Weights begin at the next 64-byte boundary and follow a fixed order known by
both converter and runtime. There is no runtime tensor-name table, which saves
space and parsing code. The loader recomputes all expected offsets and rejects
a file if the final calculated offset does not equal the declared file size.

## 9. Tokenizer and prompt construction

The official tokenizer is a 32,768-token byte-level BPE vocabulary with 32,001
merges and 524 added tokens.

The converter changes tokenizer Unicode display pieces back into their raw
byte representation and embeds merge ranks. The PSP runtime:

1. applies the Falcon ASCII-compatible pre-tokenization rules;
2. maps each byte to a base token;
3. repeatedly merges the adjacent pair with the best official merge rank;
4. returns raw token bytes during decoding.

The instruct prompt is:

    <|im_start|>user
    USER PROMPT<|im_end|>
    <|im_start|>assistant

Important special IDs are:

| Token | ID |
| --- | ---: |
| end_of_text | 11 |
| begin_of_text | 17 |
| im_start | 227 |
| im_end | 228 |
| eom | 15 |
| eot | 16 |

Generation stops on end_of_text, im_end, eom, or eot.

## 10. Runtime execution flow

At application startup:

1. PSP callbacks, power guard, and debug exception handling start.
2. CPU/bus clocks are set to 333/166 MHz.
3. model.fhq4 is opened and its header is validated.
4. The 521 KiB tokenizer section is loaded.
5. Runtime arrays, recurrent state, and the 512-token KV cache are allocated.
6. cache_mb.txt is read and the requested weight prefix is cached.
7. The editor appears.

When Start is pressed:

1. The official user/assistant chat prompt is tokenized.
2. Runtime recurrent and cache state is cleared.
3. Every prompt token runs through all 24 layers.
4. The first screen line shows token and layer progress.
5. A token is sampled from the final logits.
6. Its decoded bytes are appended to the output screen.
7. The sampled token is fed back into the model until a stop token, Circle,
   the 128-new-token limit, or the 512-token context limit is reached.
8. After a reply, the app returns to the keyboard with the conversation
   still active: the next message is appended as a chat turn and only its
   delta tokens are processed (falcon_generate_turn keeps all state).
   SELECT starts a new conversation. When the 512-token context is nearly
   full the app asks for a new conversation.

## 10.1 Screen

The PSPSDK debug font is 7x8 pixels on a 480x272 panel, so its natural
grid is 68x34. src/falcon_ui.c composes the whole screen into a 67x27
character array with no PSP calls and no floating point, and
src/main_psp.c blits only the rows that differ from the previous frame.
Redrawing during generation therefore touches one or two rows, so the
streaming display costs no measurable inference time.

Rows are not on the font's 8-pixel pitch. At that pitch the wrapped chat
has no leading at all and is tiring to read, so falcon_ui_row_y() gives
every grid row an explicit pixel Y and main_psp.c blits it a character
at a time with pspDebugScreenPutChar (pixel coordinates) instead of the
cursor-based pspDebugScreenPrintf. PutChar paints the whole 7x8 cell,
background included, so a redrawn row still needs no separate clear and
the leading between rows keeps whatever pspDebugScreenClear() left.

Layout, top to bottom, with pixel rows: title and context meter (0);
the status line (8); a chat rule carrying the scroll indicator (18);
17 rows of chat on a 10-pixel pitch (28..195); a keyboard rule (198);
a 15x4 keyboard on a 12-pixel pitch (209..252); two rows of controls
(256, 264), which end exactly on line 272. The two extra pixels per
chat line are paid for by spacing the keyboard in pixels rather than
with blank grid rows, and by two chat lines (19 -> 17). While a reply is
generating the status moves to the bottom and replaces the controls,
which are not needed because typing is blocked.

There is no separate message box. The message being typed is kept as a
provisional "You: ..." tail of the same transcript buffer, past
committed_length, and is rewritten on every keystroke. It therefore
wraps, scrolls and renders exactly like committed history, and cannot
appear in two places at once - which was a real bug in the earlier
layout, where the typed text stayed in a message box while also being
shown in the chat. falcon_ui_commit_draft() freezes the tail and opens
the "AI: " line; falcon_ui_append() streams reply bytes into it.

The transcript is a 6 KiB byte buffer, word-wrapped to the grid with a
two-column hanging indent on continuation lines, top-anchored so a
conversation reads from the first line down. The analog stick scrolls
it; new output snaps back to the end. The oldest lines are dropped when
the buffer fills.

tools/ui_preview.c links the same composition code on the host and
prints the grid inside a border, which is how the layout is checked
without hardware (the emulator cannot render the framebuffer headlessly
in this environment). It also prints the pixel Y of every row and fails
if two rows overlap or the last one runs past line 272.

The PSP frontend uses greedy sampling by default for repeatable diagnosis.
The sampler module also supports temperature, repetition penalty, top-k, and
top-p.

## 11. Power guard and crash diagnostics

Prompt evaluation can take long enough to trigger the PSP idle suspend timer
if the application never calls the power API. The v1.2 build prevents
suspend three ways:

- scePowerLock(0) is held for the whole session (suspending with CFW
  extra RAM in use is not survivable and looks like a shutdown);
- a thread calls scePowerTick(PSP_POWER_TICK_ALL) every 500 ms (the same
  thread polls the pad every 20 ms and latches Circle into a sticky
  stop_requested flag, so a short tap cancels the reply even though the
  token callback only runs every few seconds);
- the inference callback calls the same API after every model layer.

The callback also overwrites the first screen row with:

    Falcon-H1: token N, layer L/24

and later:

    Falcon-H1: token N, output head

### 11.1 Persistent trace file

The app appends to llmpsp_trace.txt next to EBOOT.PBP using raw sceIo
writes, closing and reopening the file after every line so data reaches
the card even if power is lost a moment later. It records:

- every boot with power-lock and guard-thread return codes;
- the active CPU/bus clock and free partition memory;
- which layers are served from the RAM cache and which stream from the
  card (a heartbeat at or above the boundary means the card was active);
- battery percent / AC state at boot, per generation, and per token;
- model open, runtime init, and cache stages;
- a heartbeat for every layer of every token (v1.3);
- power events (SUSPENDING / STANDBY / RESUMING / BATTERY_LOW /
  POWER_SWITCH) via a scePower callback;
- generation end status and clean exit.

An optional cpu_mhz.txt beside the EBOOT (100..333, default 333) lowers
the CPU/bus clock for power-delivery experiments. The v1.2/v1.3 on-device
traces produced by this instrumentation localized the deterministic crash
to prompt position 3, layer 5, which led directly to the silu overflow
root cause described in section 2.2. With on-device stability confirmed,
v1.7 relaxed the synced heartbeat to every 8th token (HEARTBEAT_TOKENS
in src/main_psp.c); a crash therefore localizes to an 8-token window
rather than one token. Each turn additionally logs a
"[perf] forward/matvec/streamed" line for performance tuning.

After any unexplained power-off, read the last lines of llmpsp_trace.txt
on a PC. The last heartbeat is the exact token/layer where execution
stopped; a [power] event line immediately before it indicates a suspend
or battery event rather than a crash.

No exception dump handler is installed: the PSPSDK dump handler requires
a kernel-only import that a user-mode module must not link (section 2.1).

### 11.2 Self-test EBOOT

`tools/main_psp_autotest.c` + `tools/Makefile.autotest` build
FalconAutoTest (packaged in build/FalconTest). It runs the "hi" prompt
with no input, mirrors the main app's runtime path exactly, and writes
falcon_autotest_log.txt beside its EBOOT. It searches for model.fhq4
next to itself, then in ms0:/PSP/GAME/LLMPSP/, then ef0:. Use it to
reproduce on-device issues deterministically.

## 12. Building the model

Requirements:

- Python 3.10 or newer;
- NumPy;
- enough desktop RAM and about 250 MiB of disk space.

Install dependencies:

    python -m pip install -r requirements.txt

Download and convert:

    python tools/convert_falcon_h1.py --download

Or convert an existing checkpoint:

    python tools/convert_falcon_h1.py \
      --model-dir models/falcon-h1-tiny-90m \
      --output model.fhq4 \
      --context 512

The converter validates tensor names and shapes before writing. It also writes
model.fhq4.json containing size, architecture, quantization description, and
SHA-256.

## 13. Building the PSP application

Install the current PSPDEV toolchain and run:

    make
    make package

Using the official Docker image from the repository root:

    docker run --rm \
      -v ABSOLUTE_PROJECT_PATH:/work \
      -w /work \
      ghcr.io/pspdev/pspdev:latest \
      sh -c "make clean && make package"

The package target stages EBOOT.PBP, the root model.fhq4, llmpsp_config.cfg,
README_PSP.txt, LICENSE, MODEL_NOTICE.txt, PROJECT.md, and the matching debug
ELF in build/LLMPSP. It does not regenerate model.fhq4; convert the model first.

NEVER add `-lpspkernel` (or any other kernel stub library) to LIBS. It
shadows newlib's memcpy/memset/strlen/strrchr/snprintf and the user
thread/exit syscalls with kernel-only import stubs, which broke the v1.1
build on hardware (section 2.1). After any Makefile change, verify the
import table is user-only:

    psp-objdump -t LLMPSP.elf | grep sceStub
    psp-readelf -p .rodata.sceResident LLMPSP.elf

No *ForKernel library may appear, and no mem*/str*/snprintf symbol may
live in .sceStub.text.

## 14. Host tests

Build the host CLI with any C99 compiler:

    cc -O3 -std=c99 -Iinclude \
      src/falcon_h1.c \
      src/falcon_tokenizer.c \
      src/falcon_sampler.c \
      tools/falcon_host_cli.c \
      -lm -o falcon_host_cli

Run generation:

    ./falcon_host_cli \
      model.fhq4 \
      "What is 2+2?" \
      8 \
      32

Arguments after the model are prompt, maximum new tokens, and cache MiB.

Compile tools/tokenizer_test.c against the same three runtime sources and run
it with model.fhq4. All cases and the chat template must report ok.

Useful equivalence test:

1. Run a prompt with cache 0.
2. Run the same prompt with cache 32.
3. Use greedy sampling and confirm identical bytes.

## 15. On-device troubleshooting procedure

For a shutdown while loading a short prompt:

1. Confirm the folder contains the v1.2 EBOOT (title screen must say
   "Safe v1.2" and the editor status must say "trace ON").
2. Delete any old llmpsp_trace.txt, fully reboot the PSP, and test on AC
   power with unrelated plugins disabled.
3. Launch the app and enter hi.
4. After the failure, read llmpsp_trace.txt (and, when using the
   self-test, falcon_autotest_log.txt) on a PC.

Interpretation:

| Observation | Likely direction |
| --- | --- |
| Trace ends at a [heartbeat] token/layer, no [power] event | Deterministic runtime or file fault at that location; rerun to confirm it repeats |
| [power] SUSPENDING or BATTERY_LOW right before the end | Suspend/battery event, not a code crash; check battery, AC, CFW power settings |
| Battery percent low or ac 0 in the last [battery] line | Power instability under 333 MHz + card I/O load; retest on charger |
| Trace ends during "caching weights" | Memory/card issue while filling the cache; reduce cache_mb.txt in 4 MiB steps |
| Cache 0 works but cache 32 fails | Memory headroom issue; reduce cache in 4 MiB steps |
| Cache value makes no difference and no trace file appears | The EBOOT on the card is not v1.2 |
| Model error appears before editor | Wrong/truncated FHQ4 file |
| Generation returns I/O error | Memory card read failure or filesystem corruption |

Keep LLMPSP.elf from the matching build when diagnosing addresses:

    psp-addr2line -e LLMPSP.elf -f -C 0xEPC_ADDRESS

Do not compare an EPC against an ELF from a different build.

## 16. Safe modification guide

When changing architecture code:

1. Update converter validation first.
2. Keep converter tensor order and runtime offset order identical.
3. Re-run tokenizer tests.
4. Compare cache 0 and cached greedy output.
5. Build with warnings enabled.
6. Test cache 0 on hardware before increasing cache.
7. Preserve per-layer power ticks during any long new kernel.
8. After touching src/falcon_q4_psp.c (or anything feeding it), run the
   autotest EBOOT in PPSSPP twice (default cache and cache 0) and check
   both replies match the host build; the VFPU kernel is only covered by
   this end-to-end path because host builds use the scalar kernel.

When changing memory:

- allocate required state before optional cache;
- use size_t for allocation products;
- keep context-dependent KV calculations explicit;
- leave heap headroom for prompt BPE and generation history;
- never assume ARK storage-backed memory behaves like normal RAM.

When changing quantization:

- preserve independent row addressing;
- update row-byte calculations in converter and runtime together;
- validate embedding-row decode and full matrix dot products;
- change the FHQ4 version if binary compatibility breaks.

## 17. Known limitations

- Only Falcon-H1-Tiny-90M-Instruct is accepted.
- PSP context is 512 tokens.
- The UI generates at most max_reply_tokens (llmpsp_config.cfg,
  default 128) new tokens per request.
- Input keyboard is ASCII-oriented.
- The transcript keeps roughly 6 KiB of scrollback, not the whole session.
- The debug font replaces non-ASCII output bytes visually with question marks.
- The Q4 matvecs run on the VFPU (v1.7) and the streamed weight tail is
  prefetched asynchronously (v1.8); the attention and Mamba loops are
  still scalar (the next speed lever).
- Weight caching is prefix-based, not a layer-frequency cache.
- Conversation history lives in model state only; there is no transcript
  view of earlier turns on screen.
- The v1.7 performance build (VFPU kernel, volatile-memory cache,
  head-skip, RoPE tables, trace/UI throttling) is verified on the host
  and in PPSSPP headless but still needs confirmation and timing on
  physical PSP hardware; pre-v1.7 hardware runs measured about 0.3
  tokens/second.

## 18. Licensing and upstream references

Project C source is MIT licensed under LICENSE.

Falcon-H1-Tiny-90M-Instruct and converted weights use the Falcon LLM License.
See MODEL_NOTICE.txt before distributing model.fhq4.

Primary references:

- Model: https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct
- Falcon terms: https://falconllm.tii.ae/falcon-terms-and-conditions.html
- PSPDEV: https://github.com/pspdev
- PSPSDK build rules:
  https://github.com/pspdev/pspsdk/blob/master/src/base/build.mak
- ARK settings:
  https://github.com/PSP-Archive/ARK-4/wiki/Custom-Firmware-Settings

