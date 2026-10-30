#!/usr/bin/env bash
# Design doc Module 3 + ablations B and C on one split, through the live API.
#
#   bash scripts/eval_rag_full.sh test_ood
#
# Prerequisites (see prana.rag.*): per-PR base checkouts in cache/repos,
# indexes in cache/index (prana.rag.index_repos), data/rag-code/shas.jsonl,
# data/rag-code/commits_<split>.jsonl, data/rag-code/created_at.jsonl, the API
# up, ~/.config/prana/api.key. Runs CPU-side; the API holds the GPU.
#
# Arms (all on the rows where full retrieval returned context):
#   none            no context                         (baseline)
#   full            all six sources                    ablation B: full vs none
#   no_git_history  all sources except git history     ablation C: full vs this
#   code            enclosing + definitions only       component view
# On rows where an arm's context is empty its request is identical to
# `none`; the determinism check below is what licenses reusing predictions.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT="${1:?usage: eval_rag_full.sh <split>}"
RAGPY=.venv-rag/bin/python          # has chromadb + sentence-transformers
PY=~/ai/envs/unsloth/bin/python
export PYTHONPATH=src
KEY=~/.config/prana/api.key
DATA="data/v1.1/${SPLIT}.jsonl"
OUT="runs/rag_eval/${SPLIT}"
SIDE=data/rag-code
mkdir -p "$OUT"
ts() { date -u +%FT%TZ; }

common=(--data "$DATA" --shas "$SIDE/shas.jsonl" --root cache/repos --out "$SIDE"
        --index-root cache/index --commits "$SIDE/commits_${SPLIT}.jsonl"
        --reviews data/v1.1 --created-at "$SIDE/created_at.jsonl")
echo "[$(ts)] annotate full"
$RAGPY -m prana.rag.code_annotate "${common[@]}" --name full
echo "[$(ts)] annotate no_git_history"
$RAGPY -m prana.rag.code_annotate "${common[@]}" --name no_git_history \
    --sources enclosing,past_reviews,definitions,similar_code,conventions

echo "[$(ts)] rows with full context"
$PY - "$DATA" "$SIDE/${SPLIT}.full.jsonl" "$OUT/${SPLIT}.jsonl" <<'EOF'
import json, sys
data, side, out = sys.argv[1:]
keep = {json.loads(l)["id"] for l in open(side, encoding="utf-8") if json.loads(l)["rag_context"]}
n = 0
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for line in open(data, encoding="utf-8"):
        if json.loads(line)["id"] in keep:
            f.write(line if line.endswith("\n") else line + "\n")
            n += 1
print(f"{n} rows with context")
EOF

# Seed `none` and `code` from the code-retrieval run on the same rows, if any.
for arm in none code; do
  src="runs/rag_code/${SPLIT}/${arm}/${SPLIT}.predictions.jsonl"
  if [ -f "$src" ] && [ ! -f "$OUT/$arm/${SPLIT}.predictions.jsonl" ]; then
    mkdir -p "$OUT/$arm" && cp "$src" "$OUT/$arm/"
  fi
done

echo "[$(ts)] determinism check: 20 rows re-requested with no context"
$PY - "$OUT" "$SPLIT" <<'EOF'
import json, sys
from pathlib import Path
out, split = Path(sys.argv[1]), sys.argv[2]
seed = out / "none" / f"{split}.predictions.jsonl"
if seed.exists():
    rows = [json.loads(l) for l in open(seed, encoding="utf-8")][:20]
    ids = {r["id"] for r in rows}
    data = [json.loads(l) for l in open(out / f"{split}.jsonl", encoding="utf-8")]
    (out / "determinism").mkdir(exist_ok=True)
    with open(out / "determinism" / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in data:
            if r["id"] in ids:
                f.write(json.dumps(r) + "\n")
EOF
if [ -f "$OUT/determinism/${SPLIT}.jsonl" ]; then
  $PY -m prana.client eval --key-file "$KEY" --data "$OUT/determinism/${SPLIT}.jsonl" \
      --out "$OUT/determinism/run"
  $PY - "$OUT/none/${SPLIT}.predictions.jsonl" "$OUT/determinism/run/${SPLIT}.predictions.jsonl" <<'EOF'
import json, sys
a = {json.loads(l)["id"]: json.loads(l)["prediction"] for l in open(sys.argv[1])}
b = {json.loads(l)["id"]: json.loads(l)["prediction"] for l in open(sys.argv[2])}
same = sum(a[i] == b[i] for i in b)
print(f"determinism: {same}/{len(b)} identical")
if same != len(b):
    sys.exit("API is not deterministic; reused predictions are not valid - rerun every arm")
EOF
fi

side_for() {
  case "$1" in
    none) echo "" ;;
    full|no_git_history) echo "$SIDE/${SPLIT}.$1.jsonl" ;;
    code) echo "$SIDE/${SPLIT}.jsonl" ;;
  esac
}

for arm in none full no_git_history code; do
  echo "[$(ts)] arm ${arm}"
  CTX=()
  [ -n "$(side_for "$arm")" ] && CTX=(--context "$(side_for "$arm")")
  # Reuse: a row whose context is byte-identical to one an earlier arm already
  # sent gets that arm's prediction (licensed by the determinism check).
  $PY - "$OUT" "$SPLIT" "$arm" "$(side_for none)" "$(side_for full)" "$(side_for no_git_history)" \
        "$(side_for code)" <<'EOF'
import json, sys
from pathlib import Path
out, split, arm = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
arms = ["none", "full", "no_git_history", "code"]
sides = dict(zip(arms, sys.argv[4:8]))
def ctx(a):
    p = sides[a]
    return {} if not p else {json.loads(l)["id"]: json.loads(l)["rag_context"] or ""
                             for l in open(p, encoding="utf-8")}
mine = ctx(arm)
target = out / arm / f"{split}.predictions.jsonl"
have = {json.loads(l)["id"] for l in open(target, encoding="utf-8")} if target.exists() else set()
copied = 0
for other in arms[:arms.index(arm)]:
    src = out / other / f"{split}.predictions.jsonl"
    if not src.exists():
        continue
    theirs = ctx(other)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8", newline="\n") as f:
        for line in open(src, encoding="utf-8"):
            i = json.loads(line)["id"]
            if i not in have and mine.get(i, "") == theirs.get(i, ""):
                f.write(line if line.endswith("\n") else line + "\n")
                have.add(i)
                copied += 1
print(f"{arm}: reused {copied} predictions with identical context")
EOF
  $PY -m prana.client eval --key-file "$KEY" --data "$OUT/${SPLIT}.jsonl" "${CTX[@]}" \
      --out "$OUT/$arm"
  $PY -m prana.eval.structured --no-bertscore \
      --predictions "$OUT/$arm/${SPLIT}.predictions.jsonl" > "$OUT/$arm/score.txt" 2>&1
done

echo "[$(ts)] paired comparisons"
p() { $PY -m prana.eval.paired --a "$OUT/$1/${SPLIT}.predictions.jsonl" \
        --b "$OUT/$2/${SPLIT}.predictions.jsonl" --out "$OUT/paired_$1_vs_$2.json" \
        | tee "$OUT/paired_$1_vs_$2.txt"; echo; }
p none full                 # ablation B
p no_git_history full       # ablation C (B = with git history)
p none code
cp "$SIDE/${SPLIT}.full.stats.json" "$SIDE/${SPLIT}.no_git_history.stats.json" "$OUT/"
echo "DONE $(date -u)"
