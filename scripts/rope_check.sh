#!/usr/bin/env bash
# Does the RoPE train/serve mismatch matter?
#
#   run-job start prana-rope -- bash scripts/rope_check.sh
#
# Unsloth trained v1.1 with UNSCALED RoPE; plain transformers (serve, eval)
# applies the config's linear x4 scaling. Re-generate the same 400 validation
# rows as runs/v1.1/decode/validation-greedy (scaled), greedy, with unscaled
# RoPE, and compare paired.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
D=runs/v1.1/decode
echo "[$(date -u +%FT%TZ)] v1.1 greedy, unscaled RoPE"
$PY -m prana.eval.generate --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --adapter runs/v1.1/adapter --data data/v1.1 --split validation --limit 400 \
    --batch-size 16 --rope unscaled --out "$D/validation-greedy-unscaled"
$PY -m prana.eval.structured --no-bertscore \
    --predictions "$D/validation-greedy-unscaled/validation.predictions.jsonl" \
    > "$D/validation-greedy-unscaled/score.txt" 2>&1
$PY -m prana.eval.paired --a "$D/validation-greedy/validation.predictions.jsonl" \
    --b "$D/validation-greedy-unscaled/validation.predictions.jsonl" \
    --out "$D/paired_scaled_vs_unscaled.json" | tee "$D/paired_scaled_vs_unscaled.txt"
echo "DONE $(date -u)"
