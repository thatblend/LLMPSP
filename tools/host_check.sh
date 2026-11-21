#!/bin/sh
# Host regression check: builds the CLI + tests from the current tree and
# runs the standard scenarios. Usage: host_check.sh OUTDIR [LABEL]
# Writes OUTDIR/<LABEL>_*.txt and prints a summary. MODEL=path overrides
# the model (default: model.fhq4 in the repository root, FHQ4 version 3).
# Everything runs with the app's default float16 Mamba state; the cache
# equivalence and the fpflag probe are repeated with the float32 state
# (STATE_F16=0), the other shipped configuration.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/psp_bench}"
LABEL="${2:-current}"
MODEL="${MODEL:-$ROOT/model.fhq4}"
mkdir -p "$OUT"

SRC="$ROOT/src/falcon_h1.c $ROOT/src/falcon_tokenizer.c $ROOT/src/falcon_sampler.c"
CC_FLAGS="-O2 -std=c99 -Wall -Wextra -Werror -I$ROOT/include"
gcc $CC_FLAGS $SRC "$ROOT/tools/falcon_host_cli.c" -lm -o "$OUT/cli_$LABEL"
gcc $CC_FLAGS $SRC "$ROOT/tools/tokenizer_test.c" -lm -o "$OUT/tok_$LABEL"
gcc $CC_FLAGS $SRC "$ROOT/tools/multiturn_test.c" -lm -o "$OUT/multi_$LABEL"
gcc $CC_FLAGS $SRC "$ROOT/tools/fpflag_probe.c" -lm -o "$OUT/fpflag_$LABEL"
gcc $CC_FLAGS $SRC "$ROOT/tools/sampler_test.c" -lm -o "$OUT/sampler_$LABEL"
gcc $CC_FLAGS $SRC "$ROOT/tools/context_backoff_test.c" -lm -o "$OUT/backoff_$LABEL"
gcc $CC_FLAGS $SRC "$ROOT/tools/quality_eval.c" -lm -o "$OUT/quality_$LABEL"
echo "build ok"

for STATE in 1 0; do
  S="${LABEL}_s$STATE"
  export STATE_F16=$STATE
  "$OUT/cli_$LABEL" "$MODEL" "hi" 24 32 > "$OUT/${S}_hi_c32.txt"
  "$OUT/cli_$LABEL" "$MODEL" "hi" 24 0  > "$OUT/${S}_hi_c0.txt"
  "$OUT/cli_$LABEL" "$MODEL" "hi" 24 32 1 > "$OUT/${S}_hi_striped.txt"
  "$OUT/cli_$LABEL" "$MODEL" "hi" 24 20 1 > "$OUT/${S}_hi_striped20.txt"
  "$OUT/cli_$LABEL" "$MODEL" "Tell me about the moon." 48 0 > "$OUT/${S}_moon_c0.txt"
  "$OUT/cli_$LABEL" "$MODEL" "Tell me about the moon." 48 64 > "$OUT/${S}_moon_c64.txt"
  if cmp -s "$OUT/${S}_hi_c32.txt" "$OUT/${S}_hi_c0.txt" &&
     cmp -s "$OUT/${S}_moon_c0.txt" "$OUT/${S}_moon_c64.txt"; then
    echo "cache equivalence (state_f16 $STATE): OK"
  else
    echo "cache equivalence (state_f16 $STATE): FAIL"; exit 1
  fi
  if cmp -s "$OUT/${S}_hi_c32.txt" "$OUT/${S}_hi_striped.txt" &&
     cmp -s "$OUT/${S}_hi_c32.txt" "$OUT/${S}_hi_striped20.txt"; then
    echo "striped cache equivalence (state_f16 $STATE): OK"
  else
    echo "striped cache equivalence (state_f16 $STATE): FAIL"; exit 1
  fi
  # Any OVERFLOW/INVALID/DIVBYZERO on the host is a crash on the real PSP
  # (PROJECT.md section 2.2); underflow is tolerated.
  "$OUT/fpflag_$LABEL" "$MODEL" "hi" > "$OUT/${S}_fpflag.txt" 2>&1
  "$OUT/fpflag_$LABEL" "$MODEL" "Tell me about the moon." >> "$OUT/${S}_fpflag.txt" 2>&1
  if grep -qE "OVERFLOW|INVALID|DIVBYZERO" "$OUT/${S}_fpflag.txt"; then
    echo "fpflag probe (state_f16 $STATE): FAIL"
    grep -E "OVERFLOW|INVALID|DIVBYZERO" "$OUT/${S}_fpflag.txt" | head; exit 1
  else
    echo "fpflag probe (state_f16 $STATE): OK (no OVERFLOW/INVALID/DIVBYZERO;" \
         "$(grep -c UNDERFLOW "$OUT/${S}_fpflag.txt" || true) underflow reports)"
  fi
done
export STATE_F16=1
"$OUT/cli_$LABEL" "$MODEL" "What is 2+2?" 24 32 > "$OUT/${LABEL}_math.txt"
"$OUT/tok_$LABEL" "$MODEL" | tail -n 2
"$OUT/multi_$LABEL" "$MODEL" > "$OUT/${LABEL}_multi.txt"
STATE_F16=0 "$OUT/multi_$LABEL" "$MODEL" >> "$OUT/${LABEL}_multi.txt"
grep "^multiturn" "$OUT/${LABEL}_multi.txt"
test "$(grep -c '^multiturn: ok' "$OUT/${LABEL}_multi.txt")" = 2
"$OUT/sampler_$LABEL"
"$OUT/backoff_$LABEL" "$MODEL" > "$OUT/${LABEL}_backoff.txt"
grep -E "^fit\[|^fit:" "$OUT/${LABEL}_backoff.txt"
tail -n 1 "$OUT/${LABEL}_backoff.txt"
grep -q "^backoff: ok" "$OUT/${LABEL}_backoff.txt"
echo "--- hi ---";   cat "$OUT/${LABEL}_s1_hi_c32.txt"
echo "--- hi (float32 state) ---"; cat "$OUT/${LABEL}_s0_hi_c32.txt"
echo "--- math ---"; cat "$OUT/${LABEL}_math.txt"
echo "--- moon ---"; cat "$OUT/${LABEL}_s1_moon_c0.txt"
