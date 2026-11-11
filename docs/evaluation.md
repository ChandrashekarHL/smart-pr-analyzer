# Evaluation

Generation and scoring are separate commands. Generation is the expensive half
and must survive a dropped SSH session; scoring is cheap and is re-run every
time a metric is added.

```bash
# generate on the held-out repositories (GPU, on the DGX Spark)
python -m prana.eval.generate --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --adapter runs/v1.3/adapter --data data/v1.1 --split test_ood --out runs/v1.3/eval

# score each field with the metric that fits it
python -m prana.eval.structured --predictions runs/v1.3/eval/test_ood.predictions.jsonl

# issue detection on real reviews: did the model raise the human's issue? (LLM judge)
python -m prana.eval.judge --predictions runs/v1.3/eval/test_ood.predictions.jsonl \
    --data data/v1.1/test_ood.jsonl --out runs/v1.3/judge-test_ood

# planted bugs (40 buggy/correct pairs) and vulnerabilities (33 pairs), through the API
python -m prana.eval.bench --key-file api.key --out runs/v1.3/bench-v1
PATH=<venv>/bin:$PATH python -m prana_app.security_eval --key-file api.key --out runs/security/v1.3

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
- **Detection rates need a definition.** Three are measured, all judged by
  `gemini-3.8-flash` at temperature 0:

  | Measure | Definition | v1.1 | v1.3 |
  |---|---|---|---|
  | Issue detection, real reviews | judge says the model raised the human's issue (strict / lenient), 1,190 test_ood rows | 4.0% / 12.9% | 3.5% / 12.4% |
  | Issue detection, planted | 40 bugs, each with a correct twin; strict | 25% | 32.5% |
  | False positive rate | a warning or critical finding on the correct twin | 40% | 27.5% |
  | Security detection | 33 vulnerabilities across 12 classes; model alone / with Semgrep | 24% / 58% | 33% / 64% |

  The two benchmarks are small (CIs about ±15 points). Semgrep is silently skipped if
  its binary is not on `PATH`; `--rescan-semgrep` reruns only that part.
- **The ROUGE-L ceiling is measured:** two human reviewers of the same hunk score
  0.177 against each other (771 pairs), the same as the model.

## Ablations

| | Question | Result |
|---|---|---|
| A | Does fine-tuning help? (base vs v0.3.1) | Yes, small: ROUGE-L +0.0093 [+0.0072, +0.0115] paired |
| B | Does retrieval help? (BM25 corpus memory) | No: validity −2.5 pts\*, line exact −2.7\* |
| B′ | Does retrieval help? (code from the repository) | No: line exact −8.8 pts\*, ROUGE-L −0.005\* on 822 rows |
| C | Does git history help? | Running (`scripts/eval_rag_full.sh`) |
| D | Does decoding help ROUGE-L? (400 validation rows) | No: beam 4 −0.004, MBR 8 −0.002; beam raises line exact 11.6% → 27.3%\* |
| E | Training length (v1.2, 2 epochs vs v1.3, 3 epochs) | Line localisation emerges late: line exact 1.7% vs 36.8% on test_ood |

\* 95% CI excludes zero.

## Where things run

Anything that loads model weights runs on the DGX Spark, through its GPU queue;
everything else runs anywhere. Develop against a tiny model on CPU first — same
code path, same collator, same prompt rendering:

```bash
python -m prana.eval.generate --model sshleifer/tiny-gpt2 --limit 8 --out runs/smoke
```
