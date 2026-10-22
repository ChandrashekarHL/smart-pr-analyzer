#!/usr/bin/env bash
# Base model (no adapter) on test_ood - ablation A. Without this number nothing
# can be said about whether fine-tuning helped.
#
# Submitted through the DGX GPU queue, not launched directly:
#   cd ~/prana && PYTHONPATH=src run-job start prana-eval-base -- bash scripts/eval_base.sh
# run-job copies PYTHONPATH/HF_*/WANDB_* into the job and runs it as a user
# systemd unit, so it survives logout and waits if another GPU job holds the lock.
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src
EV=runs/eval-v0.3.1
STATUS=runs/STATUS_v031
# 16, not 32. At 32 the base model was OOM-killed by the system at the third
# batch - twice, Oct 3 and Oct 6, at exactly 64 rows. The base model generates
# all 256 tokens for every row (the adapter learns to stop early), and on a
# unified-memory GB10 beside ollama's 11 GB that extra KV cache and attention
# scratch is the difference. Generation resumes from what is on disk.
BATCH="${BATCH:-16}"
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }

say "base eval started via run-job (pid $$)"
$PY -m prana.eval.generate --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --data data/train_v2 --split test_ood --batch-size "$BATCH" --out "$EV/base" \
    > "$EV-base.log" 2>&1
rc=$?
n=$(wc -l < "$EV/base/test_ood.predictions.jsonl" 2>/dev/null || echo 0)
say "generate base: exit $rc, $n rows"
if [ "$n" -lt 2199 ]; then
  say "base incomplete - resubmit the same run-job to resume"
  exit 1
fi
if $PY -m prana.eval.score --predictions "$EV/base/test_ood.predictions.jsonl" \
      > "$EV/base/score.txt" 2>&1; then
  say "score base: done -> $EV/base/score.txt"
else
  say "score base FAILED - see $EV/base/score.txt"
fi
say "base eval finished"
