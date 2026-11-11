#!/usr/bin/env bash
# Does line localisation grow with training? Late checkpoints of v1.1 and v1.2
# on the same 200 validation rows, greedy, no context (as served).
#
#   run-job start prana-probe -- bash scripts/ckpt_probe.sh
#
# If the line-number fill rate rises from checkpoint to checkpoint, v1.2 simply
# stopped early (2 epochs vs 3); if it is flat and low, the cause is in the
# recipe (context dropout 1.0 / row selection), not the amount of training.
# Output: runs/ckpt-probe/<name>/ and runs/ckpt-probe/summary.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
OUT=runs/ckpt-probe
mkdir -p "$OUT"

probe() {   # name checkpoint_dir contract_dir
  local name=$1 ckpt=$2 contract=$3 dir=$OUT/$1/adapter
  mkdir -p "$dir"
  ln -sf "$(realpath "$ckpt")/adapter_config.json" "$dir/"
  ln -sf "$(realpath "$ckpt")/adapter_model.safetensors" "$dir/"
  cp "$contract/prana_contract.json" "$dir/"
  echo "[$(date -u +%FT%TZ)] $name"
  $PY -m prana.eval.generate --model "$MODEL" --adapter "$dir" --data data/v1.1 \
      --split validation --limit 200 --batch-size 16 --out "$OUT/$name"
}

probe v1.1-550 runs/v1.1/checkpoint-550 runs/v1.1/adapter
probe v1.1-600 runs/v1.1/checkpoint-600 runs/v1.1/adapter
probe v1.1-633 runs/v1.1/checkpoint-633 runs/v1.1/adapter
probe v1.2-250 runs/v1.2/checkpoint-250 runs/v1.2/adapter
probe v1.2-300 runs/v1.2/checkpoint-300 runs/v1.2/adapter
probe v1.2-324 runs/v1.2/checkpoint-324 runs/v1.2/adapter

$PY - <<'EOF' | tee runs/ckpt-probe/summary.txt
import json, re
from pathlib import Path
SET = re.compile(r'"line_number"\s*:\s*(\d+)')
truth = {}
for line in open("data/v1.1/validation.jsonl"):
    r = json.loads(line)
    truth[r["id"]] = (r.get("structured_output") or {}).get("line_number")
print("checkpoint   rows  line given  exact (rows with a true line)")
for name in ("v1.1-550", "v1.1-600", "v1.1-633", "v1.2-250", "v1.2-300", "v1.2-324"):
    rows = [json.loads(l) for l in open(f"runs/ckpt-probe/{name}/validation.predictions.jsonl")]
    given = exact = with_truth = 0
    for r in rows:
        m = SET.search(r["prediction"])
        given += m is not None
        t = truth.get(r["id"])
        if t is not None:
            with_truth += 1
            exact += m is not None and int(m.group(1)) == int(t)
    print(f"{name:10} {len(rows):5}  {100 * given / len(rows):6.1f}%   "
          f"{100 * exact / max(with_truth, 1):5.1f}% of {with_truth}")
EOF
echo "DONE $(date -u)"
