#!/usr/bin/env bash
# v0.3.1: retrain with the whitespace-verified tokenizer on corpus v2, then
# evaluate base vs fine-tuned on test_ood. Detached:
#   cd ~/prana && setsid nohup bash scripts/run_v031.sh > runs/v031.log 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src WANDB_DIR="$HOME/prana/runs"
STATUS="$HOME/prana/runs/STATUS_v031"
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }

say "v0.3.1 started (pid $$): tokenizer fix + corpus v2 (22,446 train rows)"
$PY -m prana.train --config configs/deepseek_v031.yaml > runs/v0.3.1.log 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -d runs/v0.3.1/adapter ]; then
  say "TRAIN FAIL (exit $RC) - see runs/v0.3.1.log; checkpoints support --resume"
  grep -vE '_pytree|KernelPreference|ScaleCalculation' runs/v0.3.1.log | tail -25 >> "$STATUS"
  exit 1
fi
say "TRAIN PASS: runs/v0.3.1/adapter"
grep -oE "'eval_loss': '[0-9.]+'.*'epoch': '[0-9.]+'" runs/v0.3.1.log | sed 's/^/    /' >> "$STATUS"

say "handing over to evaluation"
ADAPTER=runs/v0.3.1/adapter DATA=data/train_v2 EV=runs/eval-v0.3.1 bash scripts/run_eval.sh
