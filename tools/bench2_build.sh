#!/bin/bash -l
# Builds the second benchmark EBOOT (LLMPSP Bench 2) and, when
# PPSSPPHeadless is installed, runs it once in the emulator as a
# correctness check: every kernel variant must report PASS. Emulator
# timings mean nothing. Run from anywhere inside the PSP toolchain
# environment:
#
#     tools/bench2_build.sh          # emulator run with the JIT (-j)
#     tools/bench2_build.sh -i       # emulator run with the interpreter
#
# Output: build/LLMPSPBench2/EBOOT.PBP for the console, and
# build/memstick/PSP/GAME/LLMPSPBench2/bench2_log.txt from the emulator.
# The emulator copy gets a bench2_quick.txt marker that shortens every
# timing loop; the console folder never has it.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p build/bench2
make -C build/bench2 -f ../../tools/Makefile.bench2
make -C build/bench2 -f ../../tools/Makefile.bench2 package

# The import table must stay user-only (PROJECT.md section 2.1).
if psp-objdump -t build/bench2/LLMPSPBench2.elf | grep sceStub |
        grep -E 'mem|str|snprintf'; then
    echo "ERROR: libc symbols are bound to import stubs" >&2
    exit 1
fi
if psp-readelf -p .rodata.sceResident build/bench2/LLMPSPBench2.prx |
        grep -i 'ForKernel'; then
    echo "ERROR: kernel-mode imports" >&2
    exit 1
fi
echo "----- imported libraries -----"
psp-readelf -p .rodata.sceResident build/bench2/LLMPSPBench2.prx || true

HEADLESS="${PPSSPP_HEADLESS:-/opt/ppsspp-src/build-headless/PPSSPPHeadless}"
if [ ! -x "$HEADLESS" ]; then
    echo "PPSSPPHeadless not found: skipping the emulator check"
    exit 0
fi
STICK="$ROOT/build/memstick"
DIR="$STICK/PSP/GAME/LLMPSPBench2"
mkdir -p "$DIR"
cp build/LLMPSPBench2/EBOOT.PBP "$DIR/EBOOT.PBP"
echo "emulator check only" > "$DIR/bench2_quick.txt"
rm -f "$DIR/bench2_log.txt"
"$HEADLESS" "$DIR/EBOOT.PBP" \
    --memstick="$STICK" --timeout=600 "${@:--j}" --graphics=software || true
echo "----- bench2_log.txt (emulator) -----"
cat "$DIR/bench2_log.txt"
