#!/usr/bin/env bash
# Evaluate code retrieval (prana.rag.contract) on one split through the live API.
#
#   bash scripts/eval_code_rag.sh test_ood
#
# Needs: data/rag-code/shas.jsonl (python -m prana.rag.shas, run where `gh` is
# logged in), the review API up (run-job prana-api), the key in
# ~/.config/prana/api.key. No GPU of its own: the API holds the model.
#
# Both arms run only on rows where retrieval returned context. On the other
# rows the two arms send the identical request, so they cannot differ; the
# report states coverage beside the paired deltas.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT="${1:?usage: eval_code_rag.sh <split>}"
PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
KEY=~/.config/prana/api.key
DATA="data/v1.1/${SPLIT}.jsonl"
OUT="runs/rag_code/${SPLIT}"
mkdir -p "$OUT" data/rag-code

echo "[$(date -u +%FT%TZ)] checkout"
$PY -m prana.rag.checkout --data "$DATA" --shas data/rag-code/shas.jsonl --root cache/repos

echo "[$(date -u +%FT%TZ)] annotate"
$PY -m prana.rag.code_annotate --data "$DATA" --shas data/rag-code/shas.jsonl \
    --root cache/repos --out data/rag-code

echo "[$(date -u +%FT%TZ)] subset rows with context"
$PY - "$DATA" "data/rag-code/${SPLIT}.jsonl" "$OUT" <<'EOF'
import json, sys
from pathlib import Path
data, side, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ctx = {}
for line in open(side, encoding="utf-8"):
    r = json.loads(line)
    if r["rag_context"]:
        ctx[r["id"]] = r["rag_context"]
n = 0
with open(out / f"{Path(data).stem}.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for line in open(data, encoding="utf-8"):
        if json.loads(line)["id"] in ctx:
            f.write(line if line.endswith("\n") else line + "\n")
            n += 1
print(f"{n} rows with context")
EOF

for arm in none code; do
  echo "[$(date -u +%FT%TZ)] arm ${arm}"
  CTX=()
  [ "$arm" = code ] && CTX=(--context "data/rag-code/${SPLIT}.jsonl")
  $PY -m prana.client eval --key-file "$KEY" --data "$OUT/${SPLIT}.jsonl" "${CTX[@]}" \
      --out "$OUT/$arm"
  $PY -m prana.eval.structured --no-bertscore \
      --predictions "$OUT/$arm/${SPLIT}.predictions.jsonl" > "$OUT/$arm/score.txt" 2>&1
done

echo "[$(date -u +%FT%TZ)] paired"
$PY -m prana.eval.paired --a "$OUT/none/${SPLIT}.predictions.jsonl" \
    --b "$OUT/code/${SPLIT}.predictions.jsonl" --out "$OUT/paired.json" | tee "$OUT/paired.txt"
cp "data/rag-code/${SPLIT}.stats.json" "$OUT/retrieval.stats.json"
echo "DONE $(date -u)"
