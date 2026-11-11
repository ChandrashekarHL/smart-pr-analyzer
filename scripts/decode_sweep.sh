#!/usr/bin/env bash
# Choose a decoding strategy on VALIDATION rows only (never test_ood):
# greedy (what every reported number used), beam search, and MBR.
#
#   run-job start prana-decode -- bash scripts/decode_sweep.sh
#
# Writes runs/v1.1/decode/validation-<config>/ with predictions, per-field
# scores and paired comparisons against greedy. The winner is then run once
# on test_ood by scripts/decode_test.sh <config>.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
ADAPTER=runs/v1.1/adapter
OUT=runs/v1.1/decode
LIMIT=400
mkdir -p "$OUT"

gen() {  # name, extra generate args...
  local name=$1; shift
  echo "[$(date -u +%FT%TZ)] $name"
  $PY -m prana.eval.generate --model "$MODEL" --adapter "$ADAPTER" --data data/v1.1 \
      --split validation --limit "$LIMIT" --out "$OUT/validation-$name" "$@"
  $PY -m prana.eval.structured --no-bertscore \
      --predictions "$OUT/validation-$name/validation.predictions.jsonl" \
      > "$OUT/validation-$name/score.txt" 2>&1
}

gen greedy --batch-size 16
gen beam4  --batch-size 8 --num-beams 4
gen mbr8   --batch-size 4 --mbr-samples 8 --mbr-temperature 0.7

for cfg in beam4 mbr8; do
  $PY -m prana.eval.paired --a "$OUT/validation-greedy/validation.predictions.jsonl" \
      --b "$OUT/validation-$cfg/validation.predictions.jsonl" \
      --out "$OUT/paired_greedy_vs_$cfg.json" | tee "$OUT/paired_greedy_vs_$cfg.txt"
done
echo "DONE $(date -u)"
