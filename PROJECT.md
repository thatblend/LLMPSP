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

Version v1.3 (git tag v1.3, packaged in release/LLMPSP-v1.3) is the build
of section 2.9: FHQ4 version 2, the four-row vtfm4 kernel, raw card
reads and context 384. Measured on a PSP-2000 at 333 MHz: about 850 ms
per prompt token and 935 ms per reply token (~1.07 tokens/second, from
~0.56), with 4.6 MiB still streamed per token. Its weights are bit-for-
bit those of the original conversion, so it is the reference to fall
back to if a later size or speed change costs quality. The build after
it (section 2.10, version string still v1.3) is the first that changes
numerics: FHQ4 version 3 with 8-bit scale codes, a float16 Mamba state
and the context fit policy, to hold the whole model in RAM.

## 2. Current status

The following are verified:

- The shipped model is a 49,402,560-byte FHQ4 file (format version 3
  since section 2.10, re-packed from the 52,244,480-byte version-2
  conversion with tools/repack_fhq4.py; the converter writes version 3
  directly but could not be run end to end without the checkpoint).
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

### 2.8 Post-v1.11 research and the benchmark EBOOT

Nothing in the app changed in this step. A research pass (2026-11)
looked for what is left beyond ~0.55 tokens/second and produced a
benchmark that measures the open questions on real hardware.

Where a reply token goes at context 256 (v1.11, PSP-2000, ~1.78 s):
layer matvecs ~850 ms, card streaming ~600 ms (3.9 MiB), output head
~250 ms, everything else ~82 ms. Two levers follow from that.

1. A faster Q4 kernel. The VFPU issues one instruction per cycle
   (documented timings: vmul/vadd/vi2f/vf2id throughput 1, latency 5;
   vc2i latency 3; vtfm4.q throughput 4, latency 10), so cost is
   instruction count, and the shipped kernel spends about 96
   instructions per 32 weights, a third of them unpacking nibbles with
   integer ops through a scratch buffer. tools/q4_tile_psp.c splits the
   nibbles inside the VFPU (signed byte / 16, then floor, is the high
   nibble) and handles four rows per pass with vtfm4.q: 105 instructions
   per 128 weights. It needs a "tile" layout (four rows interleaved,
   high nibble as two's complement, scales grouped per four rows) that
   is a lossless rearrangement of FHQ4 at the same 18 bytes per 32
   weights; the header comment of that file is the format reference.
   By the timing model that is ~1 cycle per weight against ~3.8 now.
   What the model cannot say is how much of a token is RAM bandwidth:
   every token scans ~49 MiB of weights, far more than the 16 KiB data
   cache.
2. No card streaming. ARK gives the user partition 51 MiB (56 minus the
   4 MiB partition 11 and 1 MiB partition 8), which matches the ~40 MiB
   cache ceiling seen on hardware, so more RAM is not available. The
   model is short by ~5.7 MiB at context 512 and ~3.9 MiB at 256, and
   each streamed MiB costs about 155 ms per token. Candidate savings:
   8-bit log-scale block scales instead of fp16 (2.7 MiB; measured scale
   error ~1% mean, 2.6% worst case with one range per matrix, far below
   the 4-bit weight noise), the Mamba state as float16 through
   vh2f/vf2h (2.25 MiB), and whatever PSP_HEAP_SIZE_KB(-2048) leaves
   unused outside the heap.

Measured and ruled out: dropping non-ASCII vocabulary (only 1,285 of
32,768 tokens; 0.35 MiB) and entropy-coding the nibbles (3.3 to 3.8
bits each, and decoding would cost more than the matvec). Also noted:
the greedy sampler scans 64 history ids for each of the 32,768 logits
(~30 ms per reply token); an overclock beyond 333 MHz exists as a CFW
plugin and in ARK-5 but is experimental and per-unit; the Media Engine
core has no VFPU and needs a kernel module, so it is not worth pursuing
before the items above.

The benchmark (tools/bench_psp.c, EBOOT "LLMPSP Bench") settles the
unknowns. Build it with

    tools/bench_build.sh

inside the PSPDEV environment. It compiles in build/bench (so the app's
EBOOT.PBP in the repository root is never overwritten), stages
build/LLMPSPBench, and runs the EBOOT once in PPSSPP headless when that
is installed. The emulator run only proves correctness: every kernel
must report PASS against the scalar reference. Emulator timings mean
nothing.

On the console: copy build/LLMPSPBench to ms0:/PSP/GAME/, run it with
the normal LLMPSP folder (model.fhq4) in place, wait about 15 seconds,
and read ms0:/PSP/GAME/LLMPSPBench/bench_log.txt on a PC.

| Log line | What it decides |
| --- | --- |
| [kernel] shipped / vtfm4 / vdot | Cycles per weight streaming from RAM and with the data cache warm. vtfm4 at half the shipped figure or better from RAM justifies the tile format; "cached" far below "RAM" means memory bandwidth is the limit |
| [ram] | Cost of one 64-byte line that misses the data cache; times ~800,000 lines per token it is the floor no kernel can beat |
| [project] | Matvec milliseconds per reply token for both kernels if every weight were in RAM |
| [mem] | Free RAM outside the heap (reclaimable by shrinking the heap reserve) and the heap's size in cache blocks |
| [card] | Raw sceIoRead against the app's stdio path, with aligned and unaligned offsets; only matters while anything still streams |
| [probe] | vf2id nibble split must PASS; the vh2f/vf2h lines show how hardware treats denormals and rounding, for the float16 Mamba state |
| [clock] | A pure integer loop: confirms the real clock, and shows the gain when an overclock is active |

### 2.9 Tile-format Q4 kernel, raw card reads, sampler, context back-off

Acting on the section 2.8 benchmark (PSP-2000): the version-1 row kernel
cost 4.83 cycles per weight, the four-row vtfm4 kernel 1.75 (2.76x), all
correctness checks passing on hardware; card reads through the app's
stdio path ran at 10.3 MiB/s against 16.2 MiB/s for raw sceIoRead of
256 KiB into a 64-byte aligned buffer at a 512-aligned file offset (32 KiB
reads 14.9; file offset +100 13.6; buffer +4 14.8); the greedy sampler
spent ~30 ms per reply token.

1. FHQ4 version 2 (section 8.1). Every Q4 matrix is stored as tile
   groups of four rows, a lossless rearrangement at the same 18 bytes
   per 32 weights. tools/convert_falcon_h1.py writes version 2;
   tools/repack_fhq4.py converts a version-1 file without the BF16
   checkpoint and proves the result lossless by decoding both files to
   (scale, nibble) arrays with independent code (217 matrices, 90,996,736
   weights, 2,843,648 scales identical; float tensors, tokenizer and
   header identical apart from the version field). The loader rejects
   version 1 with a message naming the re-pack tool, and rejects any Q4
   matrix whose row count is not a multiple of 4, whose column count is
   not a multiple of 64, or whose offset in the weight region is not a
   multiple of 16. (Section 2.10 replaced version 2 by version 3: the
   loader now rejects versions 1 and 2 and needs columns % 128.)
2. PSP matvecs use falcon_q4_tile4_vtfm (src/falcon_q4_psp.c, macros
   and format reference in src/falcon_q4_tile_psp.h; since 2.10 the
   version-3 kernels falcon_q4_tile4_m3/m4/m5 there, and this version-2
   kernel lives on in tools/q4_tile_v2_psp.h for the benchmark only). q4_matvec builds
   the activation tables (16 x_lo and x_hi - 16 x_lo per tile, plus the
   low-nibble block sums) once per matvec and processes four rows per
   kernel call. Host builds keep a scalar golden kernel that reads the
   tile layout with exactly the version-1 per-row arithmetic and
   summation order, so host outputs are byte-identical to version 1.
   The version-1 row kernel moved to tools/q4_row_psp.c, used only by
   the benchmark.
3. Alignment, the main hardware risk: lv.q/sv.q raise an address error
   on the real PSP for addresses that are not 16-byte aligned, and
   PPSSPP does not emulate that fault. Guarantees: Q4 offsets are
   16-aligned in the weight region (asserted by converter and loader)
   and the region itself is 64-aligned; heap cache blocks are allocated
   16-aligned (newlib's malloc only promises 8); groups that straddle a
   256 KiB cache block go through the 16-aligned row buffer; the raw
   read window is 64-aligned and starts at 512-aligned file offsets;
   runtime arrays (the outputs) are 16-aligned and rows are multiples
   of 4. A runtime guard copies any misaligned group before the kernel
   runs and counts it in runtime.realigned_groups; the autotest logs it
   ("[check] realigned groups 0") and the app's [perf] trace line ends
   with "realigned N". It must stay 0. (The guard cannot see the tiles,
   which start blocks * 8 bytes into a group, blocks * 4 since version 3;
   the block-count check at load, even then and a multiple of 4 now,
   covers them. A unit-test group with 3 blocks hung PPSSPP, so
   the emulator is not entirely blind to misaligned quads, but it cannot
   be relied on.) Denormal fp16 scales: the VFPU docs say vh2f flushes
   them to zero, PPSSPP keeps them; the model has 5 among 2.8 million and
   either result is numerically irrelevant. (Version 3 has no fp16
   scales; its codes decode to normal floats only.)
4. Raw streaming (PSP). Uncached groups are served from a 128 KiB,
   64-byte aligned read window filled with one sceIoRead at a 512-aligned
   offset, up to the window size or the next cached block. A group
   straddling the window end keeps the sectors already read (moved to
   the front), so no byte is read twice and every read stays aligned.
   stdio then runs unbuffered (header, tokenizer, small tensors, the
   256 KiB cache-fill reads and embedding groups are each one direct
   read), so the 64 KiB stdio buffer is gone. If the raw descriptor or
   window cannot be allocated, the app falls back to the version-1 stdio
   path. Any read error still sets io_error and fails the turn cleanly.
   RAM: the window costs 128 KiB + 64 B and the row buffer shrank from
   32 KiB to 2 KiB when the window is active; minus the 64 KiB stdio
   buffer that is a net +34 KiB, about 0.13 of a cache block. Why
   128 KiB: by the measured rates 256 KiB reads save ~2 ms per streamed
   MiB over 128 KiB but cost 128 KiB of cache (~12 ms per token at
   ~100 ms per streamed MiB); 64 KiB saves 64 KiB of cache (~6 ms) but
   reads ~2-3% slower. FALCON_STREAM_BYTES in src/falcon_h1.c is the
   knob. [perf] streamed now counts bytes actually read from the card;
   the window also carries the ~22 KiB of preloaded float tensors per
   streamed layer (about 1% of a layer). Since 2.10 the descriptor is
   opened with the model but the window is allocated only after the
   cache fill, and only if blocks are left uncached
   (falcon_model_stream_prepare; if the heap is full it gives one cached
   block back first). The row buffer is 2 KiB whenever the raw
   descriptor is open.
5. Sampler. The repetition penalty marks the last 64 history ids in a
   bitmap once per call, so the greedy and top-k scans cost
   O(vocab + history) instead of O(vocab x 64). The scan loops are
   unchanged, so the chosen token is identical in every case;
   tools/sampler_test.c checks 12,000 random cases (ties, repeated and
   out-of-range ids, top-k/top-p with identical RNG state) against the
   previous implementation.
6. Context default 384 (was 512; range still 128-512, the FHQ4 header
   keeps 512 as the maximum). Startup backs off instead of refusing to
   start: falcon_runtime_init_fit tries the configured context, then
   16 tokens less at a time down to 128, and keeps the largest that
   fits; the trace logs "[stage] context requested N, granted M". The
   context meter, max_new, the "conversation is full" check and the
   "too long for the N-token space" message all use the granted value.
   Only if 128 does not fit does the fatal memory screen appear.
   Generation never allocates: the repetition-penalty history is
   allocated with the runtime for the whole context, and the tokenizer
   merges chunks in static scratch (chunks over 1024 bytes, which the
   768-byte PSP prompt cannot produce, still use malloc and fail as a
   clean "too long" if it fails). tools/context_backoff_test.c checks
   the grants under capped allocations (test hook
   falcon_runtime_set_alloc_limit) and runs a chat at a backed-off
   context of 224 until it is full; the autotest's
   autotest_free_kb.txt fills the heap to exercise the back-off in
   PPSSPP.

Verified (host): every regression reply byte-identical to version 1 at
cache 0, 32 and striped 20 (four prompts, 40 tokens); host_check.sh
passes (tokenizer, multiturn, cache and striped equivalence, sampler,
back-off, fpflag probe with no OVERFLOW/INVALID/DIVBYZERO). PPSSPP: the
"hi" autotest replies "Hello! How can I help you today" with the default
cache, cache 0, prefetch on, interpreter and JIT (version 1 gave
"Hello! It's nice to hear": the summation order changed, section 2.7),
with 0 realigned groups. Needs hardware: the speed gain (kernel and
card), and the alignment guarantees, which only a real PSP can fault on.

### 2.10 Whole model in RAM: 8-bit scale codes, float16 state, fit policy

Goal: on the reference PSP-2000 the whole weight region sits in RAM at
context 384, so no weight byte is streamed during generation (streaming
cost ~360 ms of a 935 ms reply token in v1.3). Unlike every earlier step
this one changes numerics; v1.3 (release/LLMPSP-v1.3, FHQ4 version 2,
build/model_v2.fhq4) stays the reference.

Budget (the user's v1.3 trace on a PSP-2000): the heap holds 199 blocks of
256 KiB at boot, the volatile partition adds 16, the runtime took
7555 KiB at context 384, and the cache got 179 blocks (16 + 163) of the
198 the version-2 region needs, so 19 blocks (4.57 MiB) streamed per
token. The changes, in blocks of 256 KiB:

| Change | Saves |
| --- | ---: |
| 8-bit block-scale codes (FHQ4 version 3): region 51,723,008 -> 48,879,360 bytes, 198 -> 187 blocks | 11 blocks |
| Mamba recurrent state as float16 (24 x 768 x 64): runtime 7555 -> 5251 KiB | 9 blocks |
| Raw read window (128 KiB) allocated only if anything streams | 0.5 block |
| Last cache block allocated at its real size (117.75 KiB) | 0.5 block |

Resulting budget at context 384 (fit on, float16 state): 199 heap
blocks minus tokenizer and small tensors (1034 KiB, 4.0 blocks) and the
runtime (5251 KiB, 20.5 blocks) leaves ~174.4 blocks for the 170.5 heap
blocks the cache needs (171 allocations, the last one partial) on top of
the 16 volatile ones: about 3.9 blocks (~1 MiB) to spare. PPSSPP with the
autotest's budget ballast (heap capped to exactly 199 blocks at boot,
4 MiB volatile) confirms it: 187 blocks cached (16 + 171), 0 uncached,
"streamed 0 KiB", 3 blocks of heap left. With the float32 state the same
run reproduces the hardware trace to the KiB (runtime 7555 KiB, 45824
KiB in 179 blocks). PSP_HEAP_SIZE_KB(-2048) is unchanged: the budget
does not need the ~744 KiB outside the heap, and taking it would need a
hardware test (thread stacks and kernel objects come out of it).

1. FHQ4 version 3: one signed 8-bit code per 32-weight block instead of
   the fp16 scale (17 bytes per 32 weights). Code c means 0 for c == 0,
   otherwise sign(c) * float_from_bits((|c| + k) << (23 - m)): a
   minifloat with m mantissa bits (3, 4 or 5) placed by a per-matrix
   bias k. The 217 (m, k) pairs live in an 8-byte-per-matrix table
   between the tokenizer and the weight region (section 8.2). Group
   layout in section 8.1; blocks per row must now be a multiple of 4
   (cols % 128 == 0, true for 512 and 768) so the tiles (blocks * 4 bytes
   into the group) and the 68 * blocks byte groups stay 16-byte aligned;
   converter, re-pack tool and loader check it, plus 16-aligned matrix
   offsets as before. The loader rejects versions 1 and 2 with a message
   naming tools/repack_fhq4.py, and rejects (m, k) for which any code
   would decode to a non-normal float.
   Why a minifloat and not a log code: decoding must be bit-identical on
   host and PSP and cheap in the kernel. The VFPU has no table lookup and
   only approximate exp2, but a float's bit pattern is its own
   exponential: vc2i/vi2f/vabs/vadd/vf2iz build (|c| + k) << (23 - m) as
   an integer-valued float, which is exact, and vsgn/vmul apply the sign
   (src/falcon_q4_tile_psp.h). scale_code_value() in falcon_h1.c builds
   the same bits with integer ops; the VFPU unit test checks all 256
   codes for every m bit for bit.
   Choosing codes: per matrix, tools/repack_fhq4.py (and the converter)
   picks (m, k) minimising the added weight-error energy sum(sum(q^2) *
   (d' - d)^2) and stores each scale's nearest code. Measured on the
   shipped model: mean scale error 0.96%, 99th percentile 2.9% (embedding
   and w_up/wv mostly m = 5: 0.6% mean); 199 blocks (of 2.8 million) are
   zeroed and some more clipped at the bottom of their matrix's range
   (186 in the worst matrix, layer7.mamba_out), where the error energy
   is negligible. The added weight-error
   energy is 0.5-2.2% of the Q4 rounding noise per matrix type.
   tools/repack_fhq4.py converts version 2 (or 1, via 2) and always ends
   with a check that decodes both files independently: nibbles and
   float tensors identical, header and tokenizer identical apart from the
   version, size, offset and table fields, decode rules consistent, no
   sign flips; it prints the error statistics per matrix type and checks
   that the converter's quantiser (q4_v3) reproduces the re-pack.
2. Kernel cost: the scale path grew from 3 instructions per block (lv.s
   x 2, vh2f.p) to 8 (lv.s, vc2i, vi2f, vabs, vsgn, vadd, vf2iz, vmul),
   +5; the loop now runs two blocks per iteration, which saves 3 integer
   instructions per block, so the kernel is 107 instructions per 128
   weights instead of 105 (+2, ~+1.2% matvec time by the timing model).
   There are three kernel instances (falcon_q4_tile4_m3/m4/m5; m is the
   vi2f immediate) and q4_matvec picks one per matrix. Host builds keep
   the scalar golden kernel (now with a per-matvec 256-entry decode
   table); with FALCON_EVAL_V2 (tools/quality_eval.c only) it still reads
   version 2 and is then bit-identical to v1.3.
3. Float16 Mamba state (llmpsp_config.cfg state_f16, default 1; host
   tools: STATE_F16=0 for float32). Measured first over long host
   generations (17 conversations, prompt + 400 greedy tokens, every state
   value of every token): max |s| 1679 (float16 max 65504; per layer
   6.9 .. 1679), the bulk at 2^-12 .. 2^1, 0.89% of nonzero values below
   the smallest normal float16 (2^-14) and 0.04% below 2^-24. So plain
   float16 is safe in range; no per-layer scaling is needed.
   Hardware vf2h truncates and flushes results below 2^-14, vh2f flushes
   denormal halves, PPSSPP rounds and keeps denormals (the benchmark's
   [probe] lines). The shipped mode does not let either decide: the new
   value is clamped to +-65504 (so no infinity can ever be stored; the
   measured maximum is 39x below it), rounded to 11 significant bits
   with exact float arithmetic (Veltkamp: c = 8193 x, hi = c - (c - x)),
   flushed to 0 below 2^-14 (vcmp/vcmovt), and only then converted, so
   vf2h sees values it converts exactly and both store the same bits
   (falcon_vfpu_mamba_head_f16; the unit test checks 6,144 values incl.
   clamped and flushed ones bit for bit in PPSSPP). The dot with C uses
   the unrounded value. state_store()/state_load() in falcon_h1.c are the
   host model. Truncation (the raw hardware behaviour) and truncation
   after x (1 + 2^-11) were measured too and are worse (table below);
   they stay as host-only experiment modes. Cost: 68 instead of 32
   instructions per 16 state elements (~2.65M more per token, ~8 ms at
   333 MHz), against half the state traffic (2.25 instead of 4.5 MiB read
   and written per token).
4. Fit policy (llmpsp_config.cfg fit, default 1; falcon_fit_context).
   After the cache fill, if blocks stayed on the card because the heap
   ran out (not because of cache_mb), the runtime is re-initialised at
   the largest context that lets every block be allocated with 16 KiB of
   heap to spare, found by binary search at 1-token granularity (no
   block is read until the final choice), but only within 25% of the
   requested context and never below 128; otherwise the requested
   context is restored and the remainder streams as before. The trace
   logs "[stage] fit: context requested N, granted M (reason)". Nothing
   allocates during generation. PPSSPP with the budget ballast: 199
   blocks -> 384, all cached; 194 -> 383, all cached; 193 -> 325, all
   cached; 192 and 190 -> 384, 5 and 7 blocks stream (they would need
   more than 96 tokens); fit = 0 at 194 -> 384 with 4 blocks streaming.
   tools/context_backoff_test.c checks the same on the host with a
   modelled heap (falcon_set_heap_budget): exactly 379 when the budget
   is five tokens short, streaming when 3 MiB short, identical tokens in
   every case, for both state modes.
5. Quality gate. tools/quality_eval.c teacher-forces the v1.3 reference
   (v1.3 sources + version-2 model) through a candidate: 17
   conversations (short chat, factual, instructions, arithmetic, code, a
   multi-turn one, a 304-token passage), 64 greedy reply tokens each, 1353
   compared positions (the last 16 prompt positions and the reply).
   "greedy" is how many reply tokens free-running greedy decoding keeps
   identical (average of 64), "kept" how many natural replies (up to the
   reference's stop token) stay identical.

| Candidate | top-1 | KL mean | KL max | greedy | kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference against itself (tool noise floor) | 100% | 7.0e-09 | 1.5e-07 | 64.0 | 17/17 |
| Current code, version-2 model, float32 state | 100% | 7.0e-09 | 1.5e-07 | 64.0 | 17/17 |
| Yardstick: v1.3 with four-lane block sums | 99.63% | 1.9e-04 | 1.1e-02 | 55.5 | 13/17 |
| Float16 state only, shipped rounding | 98.23% | 1.2e-03 | 5.5e-02 | 32.1 | 4/17 |
| Float16 state only, truncation (raw vf2h) | 98.00% | 2.2e-03 | 1.4e-01 | 32.8 | 4/17 |
| Float16 state only, truncation of x(1+2^-11) | 97.93% | 1.3e-03 | 2.2e-01 | 30.7 | 4/17 |
| 8-bit scale codes only | 95.49% | 1.8e-02 | 3.0e-01 | 18.1 | 1/17 |
| Both (shipped) | 95.05% | 1.8e-02 | 2.5e-01 | 17.6 | 1/17 |

   Scale codes of one matrix type at a time (version-2 files with the
   decoded codes, build/eval/ablate.py): KL mean embedding 4.3e-3,
   attention 8.7e-3, mamba_in 7.6e-3, mamba_out 4.7e-3, MLP 1.4e-2: the
   cost is spread over every type, the price of ~1% scale error, not one
   bad matrix. Replies stay coherent ("hi" -> "Hello! How can I help you
   today?" on host and in PPSSPP), but they are no longer v1.3's.
   Shipping only one change: the state is a config key (state_f16 = 0);
   the scale format is a file format, so shipping the state alone means
   the version-2 loader and v1.3 kernel again (and ~8 blocks stream).

Verified: host_check.sh (both state modes: cache 0 = cached = striped,
fpflag probe clean on two prompts, multiturn, back-off and fit);
re-pack check; VFPU unit test (interpreter and JIT: all 256 codes exact
for m = 3, 4, 5, kernels within 2.5e-5, float16 state bit-exact); the
autotest in PPSSPP (JIT and interpreter, cache 44 and cache 0, plus the
budget runs above) replies "Hello! How can I help you today" every time
with 0 realigned groups. Needs hardware: the speed gain, the budget
itself (199 blocks, streamed 0 KiB), the fit re-initialisation on
newlib's real heap, the alignment of the new layout, and that the
VFPU's vscl/vsub round to nearest like IEEE (the Veltkamp step relies on
it; PPSSPP uses the host FPU) and vcmp/vcmovt/vsgn/vf2iz behave as in
PPSSPP.

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
| src/falcon_q4_psp.c | PSP-only VFPU kernels: four-row Q4 tile matvec (one per scale-code width), int8 attention, Mamba state (float32 and float16) |
| src/falcon_q4_tile_psp.h | Version-3 tile-format and scale-code reference and the Q4 kernel macros |
| src/falcon_prefetch_psp.c | PSP-only asynchronous weight-tail prefetcher |
| llmpsp_config.cfg | User configuration (cache, clock, sampling, context, fit, state_f16) |
| tools/host_check.sh | Rebuilds and reruns the whole host regression suite |
| src/falcon_tokenizer.c | Falcon byte-level BPE and instruct prompt construction |
| src/falcon_sampler.c | Greedy, repetition penalty, top-k, and top-p sampling |
| src/main_psp.c | PSP threads, power guard, trace, input, and generation flow |
| src/falcon_ui.c | Transcript buffer, word wrap, and screen composition |
| include/falcon_ui.h | Screen grid layout constants and UI state |
| tools/convert_falcon_h1.py | Official-checkpoint downloader and FHQ4 (version 3) converter |
| tools/repack_fhq4.py | FHQ4 re-pack 1 -> 2 (lossless) and 2 -> 3 (8-bit scale codes) with a decode-and-compare check and scale error statistics |
| tools/sampler_test.c | Host test: sampler picks the same token as the previous implementation |
| tools/context_backoff_test.c | Host test: startup context back-off, a chat run to the granted limit, and the fit policy under a modelled heap |
| tools/quality_eval.c | Host quality gate: teacher-forces the v1.3 reference through a candidate (top-1, KL, greedy agreement; section 2.10) |
| tools/falcon_host_cli.c | Portable desktop inference test program |
| tools/tokenizer_test.c | Exact tokenizer/chat-template regression vectors |
| tools/fpflag_probe.c | Host FPU-flag probe that predicts on-device FPU crashes |
| tools/multiturn_test.c | Proves incremental multi-turn equals full re-evaluation |
| tools/ui_preview.c | Renders the PSP screen grid on the host for layout work |
| tools/main_psp_autotest.c | No-input on-device self-test frontend |
| tools/Makefile.autotest | Builds the FalconAutoTest self-test EBOOT |
| tools/bench_psp.c | On-device benchmark: RAM scan, version-1 row kernel ("shipped" in its log) vs tile kernel, card read, heap (section 2.8) |
| tools/q4_tile_psp.c | The version-2 tile kernels (vtfm4 and the vdot cross-check); benchmark only |
| tools/q4_tile_v2_psp.h | The v1.3 (FHQ4 version 2, fp16 scale) kernel macros, kept for the benchmark |
| tools/q4_row_psp.c | The version-1 single-row VFPU kernel; benchmark only, not in the app |
| tools/Makefile.bench | Builds the LLMPSP Bench EBOOT inside build/bench |
| tools/bench_build.sh | Builds the benchmark and runs it once in PPSSPP headless |
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
| Recurrent/KV/scratch runtime | 5.13 MiB at context 384 with the float16 state (7.38 with float32; +6.4 KiB per token) |
| Tokenizer and small float tensors (plus the 128 KiB raw read window only if anything streams) | 1.01 MiB |
| Cached weights (heap), default target 44 MiB | 42.6 MiB for the whole model beyond the volatile part |
| Volatile-partition cache extension | 4 MiB when granted |
| EBOOT payload | about 0.24 MiB |
| FHQ4 file on the memory stick | 47.11 MiB (weight region 46.61 MiB, 187 blocks) |

The cache is allocated only after the required runtime state succeeds. Cache
blocks are 256 KiB. Allocation stops safely if the requested target is not
available and releases one block of heap headroom after a ceiling failure.

The default 44 MiB heap target plus the 4 MiB volatile partition caches
the whole version-3 weight region at context 384 on the reference
PSP-2000 (section 2.10: ~1 MiB to spare by the measured budget); the fit
policy trades up to 25% of the context for it on a console that is
slightly short, and the allocator backs off gracefully on smaller
layouts.
cache_mb (llmpsp_config.cfg) is clamped to 0 through 46 MiB (heap only;
the volatile 4 MiB is added automatically unless the value is 0).

## 6. Why weights were partly cached

Until section 2.10 the FHQ4 weight region was about 49.33 MiB, and with
the 8.94 MiB of model/runtime tables it exceeded the PSP's budget. The
version-3 region is 46.61 MiB and the runtime 5.13 MiB at context 384,
which fits the reference PSP-2000 completely (section 2.10); everything
below still describes what happens when a console, a larger context or
a smaller cache_mb leaves blocks on the card.

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
With cache 44 plus the volatile partition the whole model is cached on
the reference console; each block still missing streams 256 KiB per
token.

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
| PSP runtime context | 512 maximum, 384 default |
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
- a recurrent state of 768 by 64 values (float16 by default since
  section 2.10, float32 with state_f16 = 0).

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

### 8.1 Q4_0 matrix blocks and tile groups

Every dense matrix row is divided into blocks of 32 values. Each block
uses (version 3):

- one signed 8-bit scale code: 1 byte;
- 32 signed 4-bit values packed into 16 bytes.

Total:

    17 bytes for 32 weights (18 with the fp16 scales of versions 1 and 2)

Quantization: weight w in -8..7 is stored as the nibble w + 8, value =
scale * w. The scale of code c is 0 for c == 0, otherwise
sign(c) * float_from_bits((|c| + k) << (23 - m)), with the matrix's
mantissa width m (3..5) and bias k from the scale-code table (section
8.2); the converter picks the scale as in version 1 (the block's signed
maximum / -8), replaces it by its nearest code and rounds the weights
against the decoded scale. In the version-1 row layout a row was cols/32
blocks of {fp16 scale, 16 bytes}, byte j holding weight j (low nibble)
and weight j + 16 (high nibble).

Versions 2 and 3 store four consecutive rows as one tile group of exactly
4 x row bytes (B = cols / 32 blocks):

    B * 4 bytes    scale codes: block 0 rows 0..3, block 1 rows 0..3, ...
                   (version 2: B * 8 bytes of fp16 scales, same order)
    B * 64 bytes   tiles: four 16-byte tiles per block; word r of tile t
                   of block b holds row r's version-1 bytes 4t..4t+3,
                   each XOR 0x80 (high nibble becomes two's complement)

Every Q4 row count is a multiple of 4 (32768, 512, 128, 1688, 768), every
column count a multiple of 128 (512, 768: a multiple of 4 blocks, so the
tiles starting B * 4 bytes into a group and the 68 * B byte groups stay
aligned), and every Q4 offset in the weight region is a multiple of 16,
so every group and tile is 16-byte aligned for the VFPU's lv.q. Converter,
re-pack tool and loader check all three.
Groups remain
independently addressable: the runtime seeks directly to the group that
holds one embedding row, or streams a matrix group by group, without
parsing metadata. src/falcon_q4_tile_psp.h explains why the kernel wants
this layout.

Small tensors stay float32:

- RMSNorm weights;
- depthwise convolution weights and bias;
- Mamba dt_bias;
- Mamba A_log;
- Mamba D.

### 8.2 File sections

The first 256 bytes are the FHQ4 header. It contains:

- magic FHQ4 and format version 3 (versions 1 and 2, with fp16 scales,
  are rejected with a message to run tools/repack_fhq4.py);
- file, tokenizer, and weight offsets;
- 25 architecture/token fields (the context field is the maximum, 512;
  the app's default context is 384, see section 2.9);
- RoPE theta, RMS epsilon, embedding multiplier, and LM-head multiplier;
- at byte 160 (version 3): u32 offset, u32 entry count (217) and u32
  entry size (8) of the scale-code table.

The TOK2 tokenizer section follows. It contains:

- piece offsets and raw byte pieces;
- token IDs sorted by piece for binary lookup;
- merge rank by token ID;
- a byte-to-token table.

Version 3 then has, at the next 16-byte boundary, the scale-code table:
one 8-byte entry per Q4 matrix in file order (u32 offset relative to the
weight region, u8 mantissa bits m, u8 0, u16 bias k). The loader requires
each entry to name the matrix its own offset walk finds and every code to
decode to a normal float.

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
5. Runtime arrays, recurrent state (float16 unless state_f16 = 0), and
   the KV cache for the configured context (default 384) are allocated;
   if they do not fit, the context backs off in 16-token steps down to
   128 (section 2.9).
6. The requested weight prefix is cached (cache_mb, volatile memory
   first).
7. Fit policy (section 2.10): if blocks stayed uncached because the heap
   ran out, the context shrinks by the fewest tokens (at most 25%) that
   let them all be cached, else it stays and they stream.
8. The raw read window is allocated only if anything streams.
9. The editor appears.

When Start is pressed:

1. The official user/assistant chat prompt is tokenized.
2. Runtime recurrent and cache state is cleared.
3. Every prompt token runs through all 24 layers.
4. The first screen line shows token and layer progress.
5. A token is sampled from the final logits.
6. Its decoded bytes are appended to the output screen.
7. The sampled token is fed back into the model until a stop token, Circle,
   the 128-new-token limit, or the context limit is reached.
8. After a reply, the app returns to the keyboard with the conversation
   still active: the next message is appended as a chat turn and only its
   delta tokens are processed (falcon_generate_turn keeps all state).
   SELECT starts a new conversation. When the context (the granted
   value) is nearly full the app asks for a new conversation.

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
with no input, mirrors the main app's runtime path (context 384 with the
same back-off, prefetch off, raw streaming), and writes
falcon_autotest_log.txt beside its EBOOT. It searches for model.fhq4
next to itself, then in ms0:/PSP/GAME/LLMPSP/, then ef0:. Use it to
reproduce on-device issues deterministically. Optional files beside it:
cache_mb.txt (default 32), context.txt, prefetch.txt (1 = prefetch with
a striped cache), autotest_free_kb.txt (fills the heap first, to
exercise the context back-off), autotest_heap_blocks.txt (at boot, caps
the heap so exactly that many 256 KiB blocks fit: 199 models the
reference PSP-2000), fit.txt and state_f16.txt (0 turns the fit policy
or the float16 state off; both default to 1 as in the app). It logs the
heap in blocks at boot, the fit decision and a "[mem]" budget line
(runtime, cached blocks, uncached blocks, heap left). The "[check]
realigned groups" line must read 0.

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

The converter validates tensor names and shapes before writing, writes FHQ4
version 3 (sections 2.10 and 8.1), and checks every Q4 matrix's row count,
column count and 16-byte alignment. It also writes model.fhq4.json containing size,
architecture, quantization description, and SHA-256.

To convert an existing version-1 or version-2 model.fhq4 without the
checkpoint:

    python tools/repack_fhq4.py model_v2.fhq4 model.fhq4

It always finishes by decoding both files and comparing them (version
2 -> 3: identical nibbles and float tensors, and the scale-code error per
matrix type; --to 2 does the lossless 1 -> 2 step only; --check compares
an existing pair). It does not write a manifest.

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

tools/host_check.sh OUTDIR [LABEL] builds everything with -Wall -Wextra
-Werror and runs the whole suite: generation scenarios, cache and striped
cache equivalence, tokenizer, multiturn, sampler equivalence
(tools/sampler_test.c), context back-off and the fit policy
(tools/context_backoff_test.c) and the fpflag probe (must report no
OVERFLOW/INVALID/DIVBYZERO). The generation, equivalence and fpflag runs
happen for both Mamba state modes (the host tools read STATE_F16, default
1 like the app). Set MODEL=path to test a model other than the root
model.fhq4.

tools/quality_eval.c is the numeric quality gate of section 2.10: build
it once against the v1.3 sources (git show v1.3:...) and run "ref" with
the version-2 model to dump the reference, then build it against the
current sources and run "cand MODEL DUMP [STATE_MODE]"; "state" reports
the Mamba state range over long generations. Compile with
-DFALCON_EVAL_V2 to read the version-2 model with the current code (bit-
identical to v1.3), and add -DFALCON_HOST_SUM4 for the summation-order
yardstick. host_check.sh only checks that it builds.

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
   VFPU unit test and the autotest EBOOT in PPSSPP twice (default cache
   and cache 0) and check both replies match each other (and the host
   build, give or take a token, section 2.7); the VFPU kernels are only
   covered by these paths because host builds use the scalar kernel.
   PPSSPP does not fault on misaligned lv.q/sv.q: also check the
   autotest's "realigned groups 0" and keep every new buffer that feeds
   a kernel 16-byte aligned.

When changing memory:

- allocate required state before optional cache;
- use size_t for allocation products;
- keep context-dependent KV calculations explicit;
- leave heap headroom for prompt BPE and generation history;
- never assume ARK storage-backed memory behaves like normal RAM;
- keep falcon_runtime_plan_bytes in step with falcon_runtime_init (the
  back-off test compares them) and re-run the budget with the autotest's
  autotest_heap_blocks.txt = 199 (section 2.10).

When changing quantization or any numerics:

- preserve independent row addressing;
- update row-byte calculations in converter and runtime together;
- validate embedding-row decode and full matrix dot products;
- change the FHQ4 version if binary compatibility breaks;
- run tools/quality_eval.c against the v1.3 reference and report the
  table of section 2.10 next to the yardstick.

## 17. Known limitations

- Only Falcon-H1-Tiny-90M-Instruct is accepted.
- PSP context is 384 tokens by default, at most 512.
- The UI generates at most max_reply_tokens (llmpsp_config.cfg,
  default 128) new tokens per request.
- Input keyboard is ASCII-oriented.
- The transcript keeps roughly 6 KiB of scrollback, not the whole session.
- The debug font replaces non-ASCII output bytes visually with question marks.
- The Q4 matvecs run on the VFPU four rows at a time (section 2.9); the
  attention and Mamba loops have VFPU kernels too (v1.11). Asynchronous
  prefetch exists but is off by default (section 2.6).
- Weight caching is prefix-based, not a layer-frequency cache.
- Since section 2.10 replies differ from v1.3 (8-bit scale codes and
  the float16 state change numerics; quality table there). The fit
  policy can grant a context slightly below the configured one; the
  header meter shows the granted value.
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

