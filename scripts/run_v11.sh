#!/usr/bin/env bash
# v1.1 end to end: finalize (deframed, reasoning-first order) -> build RAG
# sidecars from the training split -> finalize again with context -> train on
# prompt v2 -> evaluate test and test_ood each WITH and WITHOUT context.
# The with/without pairs are ablation B. Submit on the GPU lane:
#   cd ~/prana && PYTHONPATH=src run-job start prana-v11 -- bash scripts/run_v11.sh
set -uo pipefail
cd "$HOME/prana" || exit 1
PY="$HOME/ai/envs/unsloth/bin/python"
export PYTHONPATH=src WANDB_DIR="$HOME/prana/runs"
STATUS=runs/STATUS_v11
BASE=deepseek-ai/deepseek-coder-6.7b-instruct
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$STATUS" >&2; }

say "v1.1 pipeline started (pid $$)"

# ---------------------------------------------------------------- stage 1a
if ! $PY -m prana.data.finalize --labelled data/train_v3 --out data/v1.1 > runs/finalize-v11.log 2>&1; then
  say "FAIL: finalize (pass 1)"; tail -20 runs/finalize-v11.log >> "$STATUS"; exit 1
fi
say "stage 1a PASS: labels finalized (no context yet)"

# ---------------------------------------------------------------- stage 1b
# Retrieval memory comes from the training split only; every split gets a
# sidecar. If the RAG lead's own sidecars exist in data/rag they are kept -
# ours go to data/rag-corpus and are used only when theirs are absent.
if [ -d data/rag ] && ls data/rag/*.jsonl >/dev/null 2>&1; then
  CTX=data/rag; say "stage 1b: using teammate sidecars in data/rag"
else
  CTX=data/rag-corpus
  if ! $PY -m prana.rag.annotate --corpus data/v1.1 --out "$CTX" > runs/rag-annotate.log 2>&1; then
    say "FAIL: rag annotate"; tail -20 runs/rag-annotate.log >> "$STATUS"; exit 1
  fi
  grep -E "^(index|train|validation|test|test_ood) " runs/rag-annotate.log | sed 's/^/    /' >> "$STATUS"
  say "stage 1b PASS: corpus-memory sidecars in $CTX"
fi

# ---------------------------------------------------------------- stage 1c
if ! $PY -m prana.data.finalize --labelled data/train_v3 --out data/v1.1 --context-dir "$CTX" > runs/finalize-v11.log 2>&1; then
  say "FAIL: finalize (pass 2)"; tail -20 runs/finalize-v11.log >> "$STATUS"; exit 1
fi
grep -E "^(train|validation|test|test_ood) " runs/finalize-v11.log | sed 's/^/    /' >> "$STATUS"
N=$(wc -l < data/v1.1/train.jsonl)
[ "$N" -ge 5000 ] || { say "FAIL: only $N training rows; not training on that"; exit 1; }
say "stage 1c PASS: data/v1.1 built with context, $N train rows"

# ---------------------------------------------------------------- stage 2
$PY -m prana.train --config configs/deepseek_v11.yaml > runs/v1.1.log 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -d runs/v1.1/adapter ]; then
  say "FAIL: train (exit $RC) - checkpoints in runs/v1.1 support --resume"
  grep -vE '_pytree|KernelPreference|ScaleCalculation' runs/v1.1.log | tail -25 >> "$STATUS"; exit 1
fi
grep -oE "'eval_loss': '[0-9.]+'.*'epoch': '[0-9.]+'" runs/v1.1.log | sed 's/^/    /' >> "$STATUS"
say "stage 2 PASS: adapter at runs/v1.1/adapter"

# ---------------------------------------------------------------- stage 3
# Ablation B on two splits: `test` shares repositories with training (repo
# memory available), `test_ood` does not (corpus memory only).
for split in test test_ood; do
  for mode in with-context no-context; do
    FLAG=""; [ "$mode" = with-context ] && FLAG="--with-context"
    DIR="runs/v1.1/eval-$split-$mode"
    $PY -m prana.eval.generate --model "$BASE" --adapter runs/v1.1/adapter \
        --data data/v1.1 --split "$split" --batch-size 16 $FLAG \
        --out "$DIR" > "runs/v1.1-generate-$split-$mode.log" 2>&1
    P="$DIR/$split.predictions.jsonl"
    say "stage 3 ($split, $mode): generated $(wc -l < "$P" 2>/dev/null || echo 0) rows"
    if $PY -m prana.eval.structured --predictions "$P" > "$DIR/structured.txt" 2>&1; then
      say "stage 4 PASS ($split, $mode)"
      grep -E "valid|accuracy|line exact|ROUGE" "$DIR/structured.txt" | sed "s/^/    [$split $mode] /" >> "$STATUS"
    else
      say "stage 4 FAIL ($split, $mode)"
    fi
  done
done
say "v1.1 pipeline finished"
