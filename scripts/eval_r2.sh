#!/usr/bin/env bash
# Re-evaluate v1.0 and v1.1 (no context) on test_ood after the hunk-trimming
# fix in prana.prompts. Both adapters had invalid outputs caused by prompt
# truncation, not by the models, so both get the same second pass.
#   cd ~/prana && PYTHONPATH=src run-job start prana-eval-r3 -- bash scripts/eval_r2.sh r3
# The tag names the output directories and status file.
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src
BASE=deepseek-ai/deepseek-coder-6.7b-instruct
TAG="${1:-r2}"
STATUS="runs/STATUS_eval_$TAG"
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }
say "eval $TAG started (pid $$)"
for v in v1.1 v1.0; do
  DIR="runs/$v/eval-test_ood-$TAG"
  DATA=data/$v; [ "$v" = v1.0 ] && DATA=data/v1
  $PY -m prana.eval.generate --model "$BASE" --adapter "runs/$v/adapter" \
      --data "$DATA" --split test_ood --batch-size 16 --out "$DIR" > "runs/$v-generate-$TAG.log" 2>&1
  P="$DIR/test_ood.predictions.jsonl"
  say "$v: generated $(wc -l < "$P" 2>/dev/null || echo 0) rows"
  if $PY -m prana.eval.structured --predictions "$P" > "$DIR/structured.txt" 2>&1; then
    say "$v $TAG PASS"
    grep -E "valid|accuracy|line exact|ROUGE" "$DIR/structured.txt" | sed "s/^/    [$v $TAG] /" >> "$STATUS"
  else
    say "$v $TAG FAIL"
  fi
done
say "eval $TAG finished"
