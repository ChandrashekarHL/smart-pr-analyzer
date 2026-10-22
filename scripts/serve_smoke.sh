#!/usr/bin/env bash
# One-row smoke of the serving path, queued behind whatever holds the GPU.
#
#   cd ~/prana && PYTHONPATH=src run-job start prana-serve-v1-bf16 -- bash scripts/serve_smoke.sh runs/v1.0/adapter
#   cd ~/prana && PYTHONPATH=src run-job start prana-serve-v1-4bit -- bash scripts/serve_smoke.sh runs/v1.0/adapter 4bit
#
# Arguments, not environment variables: run-job copies only a fixed whitelist
# of variables into the job (PATH, PYTHONPATH, HF_*, WANDB_* ...), so an
# `ADAPTER=...` prefix is silently dropped and the default adapter runs
# instead - which is how the first v1.0 smoke measured v0.3.1.
#
# Proves the adapter loads, the contract check passes, and the backend gets a
# JSON result with text, parsed object, confidence and latency. The latency
# number is what the backend lead plans the 30 s budget around.
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src
ADAPTER="${1:-runs/v0.3.1/adapter}"
MODE="${2:-bf16}"
TAG="$(basename "$(dirname "$ADAPTER")")-$MODE"
OUT="runs/serve-smoke/$TAG"
mkdir -p "$OUT"

$PY - <<'PYEOF'
import json
from prana.data import jsonl
row = jsonl.read("data/train_v2/test_ood.jsonl")[0]
open("runs/serve-smoke/hunk.diff", "w", encoding="utf-8").write(row["diff_hunk"])
json.dump({"id": row["id"], "language": row["language"], "reference": row["review_comment"]},
          open("runs/serve-smoke/row.json", "w", encoding="utf-8"))
PYEOF

LANG_=$($PY -c 'import json; print(json.load(open("runs/serve-smoke/row.json"))["language"])')
EXTRA=""; [ "$MODE" = "bf16" ] && EXTRA="--bf16"   # 4-bit is the serve default
$PY -m prana.serve --adapter "$ADAPTER" --language "$LANG_" $EXTRA \
    --diff-file runs/serve-smoke/hunk.diff > "$OUT/result.json" 2> "$OUT/stderr.log"
RC=$?
echo "serve exit $RC  adapter=$ADAPTER mode=$MODE"
cat "$OUT/result.json"
[ $RC -eq 0 ] && echo "SERVE SMOKE PASS ($TAG)" || { echo "SERVE SMOKE FAIL ($TAG)"; tail -20 "$OUT/stderr.log"; }
