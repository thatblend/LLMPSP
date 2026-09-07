#!/bin/sh
# Host regression check: builds the CLI + tests from the current tree and
# runs the standard scenarios. Usage: host_check.sh OUTDIR [LABEL]
# Writes OUTDIR/<LABEL>_*.txt and prints a summary.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/psp_bench}"
LABEL="${2:-current}"
MODEL="$ROOT/model.fhq4"
mkdir -p "$OUT"

SRC="$ROOT/src/falcon_h1.c $ROOT/src/falcon_tokenizer.c $ROOT/src/falcon_sampler.c"
gcc -O2 -std=c99 -I"$ROOT/include" $SRC "$ROOT/tools/falcon_host_cli.c" -lm -o "$OUT/cli_$LABEL"
gcc -O2 -std=c99 -I"$ROOT/include" $SRC "$ROOT/tools/tokenizer_test.c" -lm -o "$OUT/tok_$LABEL"
gcc -O2 -std=c99 -I"$ROOT/include" $SRC "$ROOT/tools/multiturn_test.c" -lm -o "$OUT/multi_$LABEL"
echo "build ok"

"$OUT/cli_$LABEL" "$MODEL" "hi" 24 32 > "$OUT/${LABEL}_hi_c32.txt"
"$OUT/cli_$LABEL" "$MODEL" "hi" 24 0  > "$OUT/${LABEL}_hi_c0.txt"
"$OUT/cli_$LABEL" "$MODEL" "hi" 24 32 1 > "$OUT/${LABEL}_hi_striped.txt"
"$OUT/cli_$LABEL" "$MODEL" "hi" 24 20 1 > "$OUT/${LABEL}_hi_striped20.txt"
"$OUT/cli_$LABEL" "$MODEL" "What is 2+2?" 24 32 > "$OUT/${LABEL}_math.txt"
"$OUT/cli_$LABEL" "$MODEL" "Tell me about the moon." 48 32 > "$OUT/${LABEL}_moon.txt"

if cmp -s "$OUT/${LABEL}_hi_c32.txt" "$OUT/${LABEL}_hi_c0.txt"; then
  echo "cache equivalence: OK"
else
  echo "cache equivalence: FAIL"; exit 1
fi
if cmp -s "$OUT/${LABEL}_hi_c32.txt" "$OUT/${LABEL}_hi_striped.txt" &&
   cmp -s "$OUT/${LABEL}_hi_c32.txt" "$OUT/${LABEL}_hi_striped20.txt"; then
  echo "striped cache equivalence: OK"
else
  echo "striped cache equivalence: FAIL"; exit 1
fi
"$OUT/tok_$LABEL" "$MODEL" | tail -n 2
"$OUT/multi_$LABEL" "$MODEL" | tail -n 2
echo "--- hi ---";   cat "$OUT/${LABEL}_hi_c32.txt"
echo "--- math ---"; cat "$OUT/${LABEL}_math.txt"
echo "--- moon ---"; cat "$OUT/${LABEL}_moon.txt"
