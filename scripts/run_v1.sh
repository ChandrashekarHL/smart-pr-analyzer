#!/usr/bin/env bash
# v1.0 end to end: wait for the teacher labels, finalize the corpus, train,
# generate on test_ood, score per field. Submit on the GPU lane:
#   cd ~/prana && PYTHONPATH=src run-job start prana-v1 -- bash scripts/run_v1.sh
# It holds the GPU while waiting for labelling to finish; that is deliberate -
# nothing else is queued behind it tonight, and a job that starts itself at
# 03:00 beats one that needs a person awake.
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src WANDB_DIR="$HOME/prana/runs"
STATUS=runs/STATUS_v1
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }

say "v1.0 pipeline started (pid $$)"

# ---------------------------------------------------------------- stage 0
# Labels come from the no-GPU job `prana-label`; its STATUS file ends with a
# fixed line when all four splits are written.
WAITED=0
until grep -q "labelling finished" runs/STATUS_label 2>/dev/null; do
  if [ $WAITED -ge 28800 ]; then say "FAIL: labelling not finished after 8 h"; exit 1; fi
  sleep 60; WAITED=$((WAITED + 60))
done
say "stage 0: labels ready (waited ${WAITED}s)"

# ---------------------------------------------------------------- stage 1
if ! $PY -m prana.data.finalize --labelled data/train_v3 --out data/v1 > runs/finalize.log 2>&1; then
  say "FAIL: finalize"; tail -20 runs/finalize.log >> "$STATUS"; exit 1
fi
grep -E "^(train|validation|test|test_ood) " runs/finalize.log | sed 's/^/    /' >> "$STATUS"
N=$(wc -l < data/v1/train.jsonl)
[ "$N" -ge 5000 ] || { say "FAIL: only $N training rows after finalize; not training on that"; exit 1; }
say "stage 1 PASS: data/v1 built, $N train rows"

# ---------------------------------------------------------------- stage 2
$PY -m prana.train --config configs/deepseek_v1.yaml > runs/v1.0.log 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -d runs/v1.0/adapter ]; then
  say "FAIL: train (exit $RC) - checkpoints in runs/v1.0 support --resume"
  grep -vE '_pytree|KernelPreference|ScaleCalculation' runs/v1.0.log | tail -25 >> "$STATUS"; exit 1
fi
grep -oE "'eval_loss': '[0-9.]+'.*'epoch': '[0-9.]+'" runs/v1.0.log | sed 's/^/    /' >> "$STATUS"
say "stage 2 PASS: adapter at runs/v1.0/adapter"

# ---------------------------------------------------------------- stage 3
$PY -m prana.eval.generate --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --adapter runs/v1.0/adapter --data data/v1 --split test_ood --batch-size 16 \
    --out runs/v1.0/eval > runs/v1.0-generate.log 2>&1
P=runs/v1.0/eval/test_ood.predictions.jsonl
say "stage 3: generated $(wc -l < "$P" 2>/dev/null || echo 0) rows"
if $PY -m prana.eval.structured --predictions "$P" > runs/v1.0/eval/structured.txt 2>&1; then
  say "stage 4 PASS: per-field scores in runs/v1.0/eval/structured.txt"
  grep -E "valid|accuracy|ROUGE" runs/v1.0/eval/structured.txt | sed 's/^/    /' >> "$STATUS"
else
  say "stage 4 FAIL: see runs/v1.0/eval/structured.txt"
fi
say "v1.0 pipeline finished"
