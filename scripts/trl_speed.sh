#!/usr/bin/env bash
# Why is the TRL path ~2x slower per epoch than the custom trainer?
#
#   run-job start prana-trl-speed -- bash scripts/trl_speed.sh
#
# Padding-free packing puts a whole micro-batch into ONE row: batch_size 8 x
# max_length 2048 = a 16k-token row (scripts/diagnose_trl.py saw [1, 8143] at
# batch 4). Attention over that row is quadratic in its length even though the
# examples are masked from each other. batch 1 x grad_accum 32 keeps the
# tokens per optimiser step identical (65k) with 2k-token rows.
# Same data and seed, 4 steps each; prints seconds per step.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
OUT=runs/trl-speed
mkdir -p "$OUT"

for shape in "8 4" "1 32"; do
  set -- $shape
  name=b$1x$2
  $PY - "$1" "$2" "$OUT/$name.yaml" "$OUT/$name" <<'EOF'
import sys, yaml
bs, ga, path, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = yaml.safe_load(open("configs/deepseek_v12.yaml"))
cfg["train"].update(batch_size=bs, grad_accum=ga, logging_steps=1,
                    eval_steps=10_000, save_steps=10_000)
cfg["data"]["limit"] = 1600
cfg["wandb"]["enabled"] = False
cfg["output"]["dir"] = out
yaml.safe_dump(cfg, open(path, "w"))
EOF
  echo "[$(date -u +%FT%TZ)] $name"
  $PY -m prana.train --config "$OUT/$name.yaml" --max-steps 4 --no-wandb
done

$PY - <<'EOF' | tee runs/trl-speed/summary.txt
import json
for name in ("b8x4", "b1x32"):
    st = json.load(open(f"runs/trl-speed/{name}/trainer_state.json"))
    logs = [h for h in st["log_history"] if "loss" in h]
    rt = [h for h in st["log_history"] if "train_runtime" in h][0]["train_runtime"]
    print(f"{name:6} {rt / st['global_step']:7.1f} s/step  losses {[round(h['loss'], 3) for h in logs]}")
EOF
echo "DONE $(date -u)"
