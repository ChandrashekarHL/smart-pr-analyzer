#!/usr/bin/env bash
# Unattended training pipeline for the DGX Spark.
#
# Launch DETACHED - not under tmux:
#   cd ~/prana && setsid nohup bash scripts/run_pipeline.sh > runs/pipeline.log 2>&1 < /dev/null &
#
# setsid puts this in its own session with no controlling terminal. The first
# attempt ran under tmux and died to a stray Ctrl-C in an attached client:
# SIGINT reaches the whole foreground process group, so python and the script
# went together, 16 hours and 389 of 545 steps in. Nothing typed in a terminal
# can reach this process now. Watch it with `tail -f runs/pipeline.log`.
#
# Stage 1 (does 4-bit load a 6.7B on aarch64) already PASSED and is not repeated.

set -uo pipefail
cd "$HOME/prana" || exit 1

PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src
export WANDB_MODE=offline
export WANDB_DIR="$HOME/prana/runs"

STATUS="$HOME/prana/runs/STATUS2"
mkdir -p "$HOME/prana/runs"

# stderr, never stdout. The previous version piped say() through tee to stdout
# while bench() was being captured by $(...), so the timestamp line landed
# inside the speed variable, awk compared garbage, and the chain selected the
# SLOWER loader despite unsloth measuring 2x faster.
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }

say "pipeline v2 started (detached, pid $$)"

# ---------------------------------------------------------------- stage 2
# Measured yesterday at 20 steps; no reason to pay for them again.
SPEED_4BIT_PLAIN=0.183
SPEED_4BIT_UNSLOTH=0.368
say "known: 4bit/plain=${SPEED_4BIT_PLAIN}  4bit/unsloth=${SPEED_4BIT_UNSLOTH} samples/s"

# 6 steps is enough: samples/s is what is being compared and the weights are
# cached. At the slowest observed rate a 20-step probe cost an hour.
bench() {
  local name="$1" cfg="$2"
  say "benchmarking $name"
  timeout 3600 $PY -m prana.train --config "$cfg" --max-steps 6 \
      --out "runs/bench-$name" --no-wandb > "runs/bench-$name.log" 2>&1
  grep -o "'train_samples_per_second': '[0-9.]*'" "runs/bench-$name.log" \
    | tail -1 | grep -oE '[0-9.]+' || echo 0
}

SPEED_BF16_PLAIN=$(bench bf16-plain configs/deepseek_bf16.yaml)
SPEED_BF16_UNSLOTH=$(bench bf16-unsloth configs/deepseek_bf16_unsloth.yaml)
say "measured: bf16/plain=${SPEED_BF16_PLAIN}  bf16/unsloth=${SPEED_BF16_UNSLOTH} samples/s"

pick() {  # name speed cfg
  if awk "BEGIN{exit !($2 > $BEST_SPEED)}" 2>/dev/null; then
    BEST_SPEED=$2; BEST_NAME=$1; BEST_CFG=$3
  fi
}
BEST_SPEED=0; BEST_NAME=none; BEST_CFG=
pick 4bit-plain    "${SPEED_4BIT_PLAIN:-0}"    configs/deepseek_final.yaml
pick 4bit-unsloth  "${SPEED_4BIT_UNSLOTH:-0}"  configs/deepseek_final_unsloth.yaml
pick bf16-plain    "${SPEED_BF16_PLAIN:-0}"    configs/deepseek_bf16.yaml
pick bf16-unsloth  "${SPEED_BF16_UNSLOTH:-0}"  configs/deepseek_bf16_unsloth.yaml

if [ -z "$BEST_CFG" ]; then
  say "FAIL: no configuration produced a throughput number. Chain stopped."
  exit 1
fi

# 8,723 blocks x 2 epochs
ETA=$(awk "BEGIN{printf \"%.1f\", 17446/$BEST_SPEED/3600}")
say "winner: $BEST_NAME at $BEST_SPEED samples/s  -> ~${ETA}h for 2 epochs"

# ---------------------------------------------------------------- stage 3
say "full run started ($BEST_NAME)"
$PY -m prana.train --config "$BEST_CFG" --out runs/v0.3-final > runs/v0.3-final.log 2>&1
RC=$?

if [ $RC -eq 0 ] && [ -d runs/v0.3-final/adapter ]; then
  say "PASS: adapter at runs/v0.3-final/adapter"
  say "next: prana.eval.generate --adapter runs/v0.3-final/adapter --split test_ood"
else
  say "FAIL (exit $RC). Checkpoints in runs/v0.3-final/ support --resume."
  grep -vE '_pytree|KernelPreference|ScaleCalculation' runs/v0.3-final.log | tail -25 >> "$STATUS"
fi
say "pipeline finished"
