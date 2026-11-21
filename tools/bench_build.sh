#!/bin/bash -l
# Builds the benchmark EBOOT and, when PPSSPPHeadless is installed, runs it
# once in the emulator as a correctness check (emulator timings mean
# nothing; only the PASS/FAIL lines do). Run from the repository root
# inside the PSP toolchain environment:
#
#     tools/bench_build.sh
#
# Output: build/LLMPSPBench/EBOOT.PBP for the console, and
# build/memstick/PSP/GAME/LLMPSPBench/bench_log.txt from the emulator run.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p build/bench
make -C build/bench -f ../../tools/Makefile.bench
make -C build/bench -f ../../tools/Makefile.bench package

# The import table must stay user-only (PROJECT.md section 2.1).
if psp-objdump -t build/bench/LLMPSPBench.elf | grep sceStub |
        grep -E 'mem|str|snprintf'; then
    echo "ERROR: libc symbols are bound to import stubs" >&2
    exit 1
fi

HEADLESS="${PPSSPP_HEADLESS:-/opt/ppsspp-src/build-headless/PPSSPPHeadless}"
if [ ! -x "$HEADLESS" ]; then
    echo "PPSSPPHeadless not found: skipping the emulator check"
    exit 0
fi
STICK="$ROOT/build/memstick"
mkdir -p "$STICK/PSP/GAME/LLMPSPBench" "$STICK/PSP/GAME/LLMPSP"
# The card-read section looks for the model where the app keeps it.
if [ -f model.fhq4 ] && [ ! -f "$STICK/PSP/GAME/LLMPSP/model.fhq4" ]; then
    cp model.fhq4 "$STICK/PSP/GAME/LLMPSP/model.fhq4"
fi
cp build/LLMPSPBench/EBOOT.PBP "$STICK/PSP/GAME/LLMPSPBench/EBOOT.PBP"
rm -f "$STICK/PSP/GAME/LLMPSPBench/bench_log.txt"
"$HEADLESS" "$STICK/PSP/GAME/LLMPSPBench/EBOOT.PBP" \
    --memstick="$STICK" --timeout=180 "${@:--j}" --graphics=software || true
echo "----- bench_log.txt (emulator) -----"
cat "$STICK/PSP/GAME/LLMPSPBench/bench_log.txt"
