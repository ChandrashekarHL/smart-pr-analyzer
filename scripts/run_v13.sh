#!/usr/bin/env bash
# v1.3: train (TRL SFTTrainer, configs/deepseek_v13.yaml), then evaluate on
# test_ood exactly as v1.1's numbers of record (r3) were produced, and compare
# the two on the same rows.
#
#   run-job start prana-v13 -- bash scripts/run_v13.sh
#
# Outputs: runs/v1.3/adapter, runs/v1.3/eval-test_ood/ (predictions, scores),
# runs/v1.3/paired_v11_vs_v13.txt. Status lines in runs/STATUS_v13.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
STATUS=runs/STATUS_v13
mkdir -p runs/v1.3
note() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

note "train start"
$PY -m prana.train --config configs/deepseek_v13.yaml
note "train done"

note "generate test_ood (greedy, as r3)"
$PY -m prana.eval.generate --model "$MODEL" --adapter runs/v1.3/adapter --data data/v1.1 \
    --split test_ood --batch-size 16 --out runs/v1.3/eval-test_ood
$PY -m prana.eval.structured --predictions runs/v1.3/eval-test_ood/test_ood.predictions.jsonl \
    > runs/v1.3/eval-test_ood/score.txt 2>&1
note "scored"

$PY -m prana.eval.paired --a runs/v1.1/eval-test_ood-r3/test_ood.predictions.jsonl \
    --b runs/v1.3/eval-test_ood/test_ood.predictions.jsonl \
    --out runs/v1.3/paired_v11_vs_v13.json | tee runs/v1.3/paired_v11_vs_v13.txt
note "DONE"
