# Model card — prana v1.3 (DeepSeek-Coder 6.7B, QLoRA)

Structured code-review comments for a single diff hunk. Given one hunk, its language and
(optionally) repository context, the model returns one JSON object:

```json
{"explanation": "...", "suggestion": "...", "severity": "info|warning|critical",
 "issue_type": "BUG|SECURITY|PERF|MAINTAINABILITY|TEST|DOCS|ARCHITECTURE|STYLE",
 "file_path": "...", "line_number": 113}
```

| | |
|---|---|
| Base model | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| Method | QLoRA, 4-bit NF4, r=16 α=32 dropout 0.05, all attention + MLP projections (40.0 M trainable, 0.59% of 6.74 B; PEFT prints 1.13% because it counts 4-bit weights packed two per byte) |
| Training data | 12,483 human review comments from 208 public repositories, teacher-structured (see dataset card); 53 AI-reviewer rows dropped (filters v5) and SECURITY rows repeated ×3 → 13,016 training examples |
| Training | TRL `SFTTrainer` (prompt/completion, completion-only loss, padding-free best-fit packing), 3 epochs, 603 steps, effective batch 32 (1 × 32), LR 1e-4 cosine, seq 2,048; 15.0 h on one GB10 (DGX Spark) with Unsloth |
| Prompt contract | `v2` (fingerprint in `adapter/prana_contract.json`); serving refuses a mismatched build |
| Artifact | `runs/v1.3/adapter` (LoRA weights + contract + tokenizer) |
| Serving | `prana.api` over Tailscale (`http://100.124.121.97:8100`, API key), 4-bit, **median 3.1 s per review** (p90 3.6 s, 146 live requests) on the GB10, no retrieval context |

## Intended use

- **The explanation and suggestion are the product.** They are readable, grounded in the hunk,
  and written in a maintainer's voice.
- **Severity and issue type are hints, not gates.** Do not block merges on them (see results).
- **Serve without retrieval context.** Context made results worse in our ablations (below).
- **Pair it with a rule scanner for security.** The model alone finds a third of planted
  vulnerabilities; with Semgrep beside it (as the GitHub App runs it) it finds 64%.
- One hunk at a time. Python, TypeScript, JavaScript, Go, Rust only.

Not intended for: security auditing on its own, merge gating, languages outside the five, or
whole-PR summaries.

## Evaluation

All numbers on `test_ood`: **1,190 reviews from 234 repositories that never appear in
training**, after n-gram decontamination. 95% bootstrap CIs. Categorical fields are scored on
every row — an unparseable output counts as wrong, not missing.

| Metric | v1.0 | v1.1 | **v1.3** | Baseline |
|---|---|---|---|---|
| Valid JSON / schema | 99.5% | 99.5% | **99.5%** [99.0, 99.8] | — |
| Severity accuracy | 50.6% | 47.0% | 44.8% [41.9, 47.5] | majority class 49.4% |
| Severity macro-F1 | 0.270 | 0.352 | 0.322 | — |
| Predicted severity mix (info/warning/critical) | 93 / 7 / **0** | 70 / 23 / 7 | 71 / 21 / 7 | truth 49 / 37 / 14 |
| Issue-type accuracy | 35.4% | 34.5% | 31.7% [29.2, 34.3] | majority class 28.0% |
| Issue-type macro-F1 | 0.293 | 0.289 | 0.273 | — |
| **Line number exact** (where reference has one, n=544) | 0.7% | 14.7% | **36.8%** [32.5, 40.6] (MAE 4.7) | — |
| Explanation ROUGE-L / BERTScore | 0.178 / 0.854 | 0.169 / 0.856 | 0.169 / 0.856 | human vs human ROUGE-L 0.177 |

**Paired, v1.3 minus v1.1, same 1,190 rows:** line exact **+22.1 pts\*** [+18.4, +25.9],
issue type −2.9\* [−5.0, −0.7], severity −2.2 [−4.6, +0.3], validity and ROUGE-L unchanged.

### Detection benchmarks

| Benchmark | v1.1 | **v1.3** |
|---|---|---|
| Planted bugs, 40 buggy/correct pairs: issue detected (strict / lenient) | 25% / 32.5% | **32.5% / 40%** |
| Planted bugs: false alarm on the correct version | 40% | **27.5%** |
| Planted bugs by category: security / bug / performance | 38% / 22% / 0% | **54% / 26%** / 0% |
| Security suite, 33 pairs, 12 vulnerability classes: model alone (detected / false positive) | 24% / 24% | 33% / 36% |
| Security suite: model + Semgrep rules (the live configuration) | 58% / 27% | **64%** / 36% |
| Real reviews (test_ood): LLM judge says the model raised the human's issue (strict / lenient) | 4.0% / 12.9% | 3.5% / 12.4% |

The benchmarks are small (40 and 33 cases, CIs ±15 pts), so each change there is a direction,
not a significant result on its own. Semgrep alone: 46% detected, 6% false positives.

How to read it:

- **v1.3 is the served model.** It points at the right line 2.5× as often as v1.1, finds more
  planted bugs and raises fewer false alarms on correct code. It pays for that with a small,
  significant drop in issue-type accuracy and more false alarms on the security suite — the
  side effect of the ×3 security weighting.
- v1.0 collapsed to the majority class (`info` 93%, `critical` never). v1.1 broke that
  collapse; v1.3 keeps it.
- The gap to the human reference is mostly *which* finding, not mislabelling: v1.1's
  severity agrees with an independent reading of its own explanation 82% of the time. Two
  human reviewers of the same hunk reach only 0.177 ROUGE-L against each other.

### Ablations

| | Result |
|---|---|
| **A** — fine-tuned vs base (v0.3.1, prose) | ROUGE-L +0.0093 [+0.0072, +0.0115] paired; base rambles (median 162 words vs 51 human, 64 fine-tuned) |
| **B** — retrieval context on vs off (v1.1) | **negative**: valid JSON −2.5 pts*, line exact −2.7*, severity −2.0, issue type −1.6. BM25 corpus memory distracts the model. |
| **B′** — code retrieved from the repository (v1.1) | **negative**: line exact −8.8 pts*, ROUGE-L −0.005* on 822 rows |
| **Training length** — v1.2 (2 epochs) vs v1.3 (3 epochs), same trainer and data changes | line localisation appears late: v1.1 gives a line in 8% of validation rows at epoch 2.6 and 27% at 2.8; v1.2 stopped at 2 epochs and lost it (line exact 1.7%) |

\* CI excludes zero.

## Known limitations and failure modes

- Severity and issue type disagree with the human reference on more than half the rows.
- `MAINTAINABILITY` is over-predicted; `BUG → MAINTAINABILITY` is the top confusion. v1.3 also
  labels more findings `STYLE` than v1.1.
- Line number is emitted on 57% of rows; exact on 37% of rows whose reference has one.
- Performance issues are not detected (0 of the planted cases).
- ~30% of raw outputs narrate in third person ("The reviewer suggests…"); `prana.serve` rewrites
  these before returning.
- Very long hunks are cut in the middle to fit the 2,048-token window; findings in the omitted
  part cannot be raised.
- Labels come from a teacher model (`gemini-3.8-flash`) restructuring human comments, not from
  human annotators; teacher self-agreement is κ 0.92 (severity), 0.82 (issue type).
- The v1.3 run logged `nan` for validation loss (batch size 1 with padding-free packing); training
  loss fell normally (1.83 → 1.06) and all results above come from separate generation on
  `test_ood`.

## Training data in one paragraph

188,052 scraped review comments → remove AI-authored (19%, detected by account, not text),
author replies (43% of what remained, detected by the teacher's `is_review`), AI-tool markup,
short/conversational/prose rows → 12,483 training reviews. Splits are by pull request and by
repository; no PR or repo crosses splits. Details and every count: `data/v1.1/dataset-card.md`.
