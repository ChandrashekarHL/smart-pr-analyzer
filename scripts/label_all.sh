#!/usr/bin/env bash
# Label every split with the Gemini teacher into data/train_v3/. Network-bound,
# no GPU - submit on the queue's no-GPU lane so it survives logout:
#   cd ~/prana && PYTHONPATH=src run-job start prana-label --no-gpu -- bash scripts/label_all.sh
# Resumable: re-running skips rows already labelled. Stop with: run-job stop prana-label
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src
STATUS=runs/STATUS_label
mkdir -p data/train_v3
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }

say "labelling started (pid $$), teacher pinned in prana.data.label.DEFAULT_MODEL"
for split in train validation test test_ood; do
  say "split: $split"
  $PY -m prana.data.label --in "data/train_v2/$split.jsonl" --out "data/train_v3/$split.jsonl" \
      --rows-per-request 8 --workers 6 > "runs/label-$split.log" 2>&1
  rc=$?
  tail -2 "runs/label-$split.log" | head -1 >> "$STATUS"
  say "split $split: exit $rc, $(wc -l < "data/train_v3/$split.jsonl" 2>/dev/null || echo 0) rows written"
done
say "labelling finished"
