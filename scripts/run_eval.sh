#!/usr/bin/env bash
# Evaluate the v0.3 adapter against the untuned base model on test_ood.
#
#   cd ~/prana && setsid nohup bash scripts/run_eval.sh > runs/eval.log 2>&1 < /dev/null &
#
# Detached for the same reason as training: nothing typed in a terminal can
# reach it. Generation resumes from what is already on disk, so if this is
# ever stopped, re-launching the same command finishes only what is left.
#
# Order matters: the fine-tuned model goes first so its numbers exist even if
# the base-model pass is cut short. The base pass is ablation A - without it
# there is no evidence fine-tuning changed anything.

set -uo pipefail
cd "$HOME/prana" || exit 1

PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src
BASE=deepseek-ai/deepseek-coder-6.7b-instruct
ADAPTER="${ADAPTER:-runs/v0.3.1/adapter}"
DATA="${DATA:-data/train_v2}"
EV="${EV:-runs/eval-v0.3.1}"
STATUS="$HOME/prana/runs/STATUS_v031"
BATCH=32

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }
say "eval started (detached, pid $$)"

# ---------------------------------------------------------------- stage 0
# A 16-row probe through the exact code path. Catches adapter/base mismatches
# and prompt errors in minutes rather than hours, and measures real speed.
say "smoke: 16 rows with adapter"
if ! $PY -m prana.eval.generate --model "$BASE" --adapter "$ADAPTER" \
      --data "$DATA" --split test_ood --limit 16 --batch-size 16 --out "$EV/smoke" \
      > "$EV-smoke.log" 2>&1; then
  say "smoke FAIL - chain stopped"
  grep -vE '_pytree|KernelPreference|ScaleCalculation' "$EV-smoke.log" | tail -20 >> "$STATUS"
  exit 1
fi
RATE=$(grep -oE '[0-9.]+ rows/s' "$EV-smoke.log" | tail -1 | grep -oE '[0-9.]+')
ETA=$(awk "BEGIN{ if (${RATE:-0}>0) printf \"%.1f\", 2199/${RATE}/3600; else print \"?\" }")
SPACES=$("$PY" -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(sum(x['prediction'].count(' ') for x in r))" "$EV/smoke/test_ood.predictions.jsonl" 2>/dev/null || echo 0)
if [ "${SPACES:-0}" -lt 50 ]; then
  say "smoke FAIL: predictions contain only ${SPACES} spaces - tokenizer decode is broken. Chain stopped."
  exit 1
fi
say "smoke PASS: ${RATE:-?} rows/s -> ~${ETA}h per model on 2,199 rows"

# ---------------------------------------------------------------- stage 1
gen() {  # name, extra args...
  local name="$1"; shift
  say "generate: $name"
  $PY -m prana.eval.generate --model "$BASE" "$@" --data "$DATA" --split test_ood \
      --batch-size $BATCH --out "$EV/$name" > "$EV-$name.log" 2>&1
  local rc=$?
  local n; n=$(wc -l < "$EV/$name/test_ood.predictions.jsonl" 2>/dev/null || echo 0)
  say "generate $name: exit $rc, $n rows written"
  return $rc
}

gen v0.3 --adapter "$ADAPTER" || say "v0.3 generation incomplete - re-run to resume"
gen base                      || say "base generation incomplete - re-run to resume"

# ---------------------------------------------------------------- stage 2
for name in v0.3 base; do
  P="$EV/$name/test_ood.predictions.jsonl"
  [ -s "$P" ] || { say "score $name: no predictions, skipped"; continue; }
  say "score: $name"
  $PY -m prana.eval.score --predictions "$P" > "$EV/$name/score.txt" 2>&1 \
    && say "score $name: done -> $EV/$name/score.txt" \
    || say "score $name: FAILED, see $EV/$name/score.txt"
done

say "eval finished"
