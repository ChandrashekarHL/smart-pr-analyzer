#!/usr/bin/env bash
# Before a 12-hour run on a new trainer, check it learns like the old one.
#
#   run-job start prana-parity -- bash scripts/trl_parity.sh
#
# Same data (first 1,600 rows of v1.1 train, v1.2 row selection), same seed,
# same LoRA and optimiser, 15 steps each: custom trainer vs TRL SFTTrainer.
# Prints both loss curves from trainer_state.json. Packing differs between the
# two (greedy vs best-fit-decreasing), so curves agree in level and trend,
# not step for step.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
OUT=runs/parity
mkdir -p "$OUT"

for kind in custom trl; do
  $PY - "$kind" "$OUT/$kind.yaml" <<'EOF'
import sys, yaml
kind, path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/deepseek_v12.yaml"))
cfg["train"]["trainer"] = kind
cfg["train"]["logging_steps"] = 1
cfg["train"]["eval_steps"] = 10_000
cfg["train"]["save_steps"] = 10_000
cfg["data"]["limit"] = 1600
cfg["wandb"]["enabled"] = False
cfg["output"]["dir"] = f"runs/parity/{kind}"
yaml.safe_dump(cfg, open(path, "w"))
EOF
  echo "[$(date -u +%FT%TZ)] $kind"
  $PY -m prana.train --config "$OUT/$kind.yaml" --max-steps 15 --no-wandb
done

$PY - <<'EOF'
import json
for kind in ("custom", "trl"):
    st = json.load(open(f"runs/parity/{kind}/trainer_state.json"))
    losses = [h["loss"] for h in st["log_history"] if "loss" in h]
    print(kind.ljust(7), "steps", len(losses), "| first 3", [round(x, 3) for x in losses[:3]],
          "| last 5 mean", round(sum(losses[-5:]) / 5, 4))
EOF
echo "DONE $(date -u)"
