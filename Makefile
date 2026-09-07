TARGET = LLMPSP
OBJS = src/main_psp.o src/falcon_ui.o src/falcon_h1.o src/falcon_q4_psp.o src/falcon_prefetch_psp.o src/falcon_tokenizer.o src/falcon_sampler.o

INCDIR = include
# -fno-math-errno lets GCC inline sqrtf as a bare sqrt.s instruction; it
# never changes results, only removes the errno bookkeeping newlib skips
# anyway. Do NOT add full -ffast-math: reassociation could reintroduce
# overflow paths of the kind that crashed v1.3 on hardware (PROJECT.md 2.2).
CFLAGS = -O3 -G0 -Wall -Wextra -std=c99 -fno-math-errno
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti
ASFLAGS = $(CFLAGS)
# IMPORTANT: never link -lpspkernel (or other kernel stub libs) into this
# user-mode app. libpspkernel.a provides SysclibForKernel/ThreadManForKernel/
# LoadExecForKernel import stubs that silently REPLACE newlib's memcpy,
# memset, strlen, strrchr, snprintf and the user thread/exit syscalls.
# A user-mode PRX cannot import kernel libraries, so those calls become
# garbage no-ops or kernel faults on real hardware (hard power-off).
LIBS = -lpsppower -lm

BUILD_PRX = 1
PSP_FW_VERSION = 660
PSP_LARGE_MEMORY = 1
EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_ICON = ICON0.PNG
PSP_EBOOT_TITLE = LLMPSP

PSPSDK := $(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak

# The autotest Makefile also produces an EBOOT.PBP in the repo root, so
# package always relinks its own EBOOT first instead of trusting a stale
# file (packaging the self-test EBOOT by accident is a real failure mode).
.PHONY: package
package:
	rm -f EBOOT.PBP PARAM.SFO
	$(MAKE) EBOOT.PBP
	mkdir -p build/LLMPSP
	cp EBOOT.PBP build/LLMPSP/EBOOT.PBP
	test -f model.fhq4
	cp model.fhq4 build/LLMPSP/model.fhq4
	mkdir -p build/LLMPSP/debug
	cp README_PSP.txt build/LLMPSP/README.txt
	cp LICENSE build/LLMPSP/LICENSE
	cp MODEL_NOTICE.txt build/LLMPSP/MODEL_NOTICE.txt
	cp PROJECT.md build/LLMPSP/PROJECT.md
	rm -f build/LLMPSP/cache_mb.txt build/LLMPSP/llmpsp.cfg
	cp llmpsp_config.cfg build/LLMPSP/llmpsp_config.cfg
	cp LLMPSP.elf build/LLMPSP/debug/LLMPSP.elf
	cd build/LLMPSP && sha256sum EBOOT.PBP model.fhq4 > CHECKSUMS.sha256
	@echo "Copy build/LLMPSP to ms0:/PSP/GAME/"
