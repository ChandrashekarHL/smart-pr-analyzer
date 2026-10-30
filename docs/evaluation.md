# Evaluation

Generation and scoring are separate commands. Generation is the expensive half
and must survive a dropped SSH session; scoring is cheap and is re-run every
time a metric is added.

```bash
# generate on the held-out repositories (GPU, on the DGX Spark)
python -m prana.eval.generate --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --adapter runs/v1.1/adapter --data data/v1.1 --split test_ood --out runs/v1.1/eval

# score each field with the metric that fits it
python -m prana.eval.structured --predictions runs/v1.1/eval/test_ood.predictions.jsonl

# compare two systems on the same rows (paired bootstrap)
python -m prana.eval.paired --a runs/a/test_ood.predictions.jsonl --b runs/b/test_ood.predictions.jsonl

# or evaluate through the running API, from anywhere on the tailnet
python -m prana.client eval --data data/test_ood.jsonl --out runs/try1 --key-file api.key
```

## Principles

- **Report `test_ood`, not `test`.** Whole repositories are held out and it
  carries 6x less n-gram contamination (0.78% against 4.76%).
- **Never score ROUGE or BERTScore on serialised JSON.** Two unrelated reviews
  in the same schema share enough braces and key names to score 0.39. Each
  field gets its own metric: validity rates; accuracy and macro-F1 **beside the
  majority-class baseline** for severity and issue type; exact match and MAE for
  line numbers; ROUGE-L and BERTScore on the explanation prose only.
- **Every number carries a bootstrap 95% CI and a per-language breakdown.** A
  single aggregate hides whether a gap exceeds the noise and whether one
  language is carrying the average.
- **Comparisons are paired.** Two systems on the same rows, bootstrap over the
  per-row differences (`prana.eval.paired`) — far tighter than comparing two
  independent means, and the only way a component gets switched on.
- **Generation resumes.** Predictions append to disk and a re-run skips ids
  already present.

## Reading the numbers

- **ROUGE-L against free-form review comments lands around 0.10–0.25** in
  published work — two reviewers describing the same bug share almost no
  n-grams. The design document's 0.42 target is not reachable against a single
  human reference and is not treated as a pass/fail gate.
- **Raw BERTScore F1 sits near 0.85 even for unrelated technical English.** Both
  raw and baseline-rescaled figures are reported.
- **Self-consistency** complements the human reference: the teacher re-labels
  the model's *own* explanation, and we measure whether the model's severity and
  issue type agree with its own text (82.4% and 84.7% for v1.1).

## Ablations

| | Question | Result |
|---|---|---|
| A | Does fine-tuning help? (base vs v0.3.1) | Yes, small: ROUGE-L +0.0093 [+0.0072, +0.0115] paired |
| B | Does retrieval help? (BM25 corpus memory) | No: validity −2.5 pts\*, line exact −2.7\* |
| B′ | Does retrieval help? (code from the repository) | No: line exact −8.8 pts\*, ROUGE-L −0.005\* on 822 rows |
| C | Does git history help? | Running (`scripts/eval_rag_full.sh`) |

\* 95% CI excludes zero.

## Where things run

Anything that loads model weights runs on the DGX Spark, through its GPU queue;
everything else runs anywhere. Develop against a tiny model on CPU first — same
code path, same collator, same prompt rendering:

```bash
python -m prana.eval.generate --model sshleifer/tiny-gpt2 --limit 8 --out runs/smoke
```
