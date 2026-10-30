# Model card — prana v1.1 (DeepSeek-Coder 6.7B, QLoRA)

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
| Training data | 12,483 human review comments from 211 public repositories, teacher-structured (see dataset card) |
| Training | 3 epochs, 633 steps, effective batch 32, LR 1e-4 cosine, seq 2,048, packed, completion-only loss; 12.5 h on one GB10 (DGX Spark) |
| Prompt contract | `v2` (fingerprint in `adapter/prana_contract.json`); serving refuses a mismatched build |
| Artifact | `runs/v1.1/adapter` (LoRA weights + contract + tokenizer) |
| Serving | `prana.api` over Tailscale (`http://100.124.121.97:8100`, API key), 4-bit, **~3–5 s per review** on the GB10, no retrieval context |

## Intended use

- **The explanation and suggestion are the product.** They are readable, grounded in the hunk,
  and written in a maintainer's voice.
- **Severity and issue type are hints, not gates.** Do not block merges on them (see results).
- **Serve without retrieval context.** Context made results worse in our ablation (below).
- One hunk at a time. Python, TypeScript, JavaScript, Go, Rust only.

Not intended for: security auditing, merge gating, languages outside the five, or whole-PR
summaries.

## Evaluation

All numbers on `test_ood`: **1,190 reviews from 277 repositories that never appear in
training**, after n-gram decontamination. 95% bootstrap CIs. Categorical fields are scored on
every row — an unparseable output counts as wrong, not missing.

| Metric | v1.0 | **v1.1** | Baseline |
|---|---|---|---|
| Valid JSON / schema | 99.5% [99.1, 99.8] | **99.5%** [99.1, 99.8] | — |
| Severity accuracy | 50.6% [47.7, 53.5] | 47.0% [44.0, 49.9] | majority class 49.4% |
| Severity macro-F1 | 0.270 | **0.352** | — |
| Predicted severity mix (info/warning/critical) | 93 / 7 / **0** | 70 / 23 / 7 | truth 49 / 37 / 14 |
| Issue-type accuracy | 35.4% [32.8, 38.2] | 34.5% [31.9, 37.3] | majority class 28.0% |
| Issue-type macro-F1 | 0.293 | 0.289 | — |
| Line number exact (where reference has one, n=544) | 0.7% | **14.7%** [11.9, 17.6] (MAE 3.2) | — |
| Explanation ROUGE-L / BERTScore | 0.178 / 0.854 | 0.169 / 0.856 | — |
| Self-consistency: severity matches the teacher's reading of the model's *own* text | — | **82.4%** (99.9% within one level) | — |

How to read it:

- v1.0 collapsed to the majority class (`info` 93%, `critical` never). v1.1 broke that collapse:
  macro-F1 up 0.27 → 0.35, and it now says `critical`. Accuracy fell because spreading guesses
  over three classes scores lower against one human's pick than always answering `info`.
- The gap to the human reference is mostly *which* finding, not mislabelling: the model's
  severity agrees with an independent reading of its own explanation 82% of the time. Two
  reviewers of the same hunk often raise different valid points.

### Ablations

| | Result |
|---|---|
| **A** — fine-tuned vs base (v0.3.1, prose) | ROUGE-L +0.0093 [+0.0072, +0.0115] paired; base rambles (median 162 words vs 51 human, 64 fine-tuned) |
| **B** — retrieval context on vs off (v1.1) | **negative**: valid JSON −2.5 pts*, line exact −2.7*, severity −2.0, issue type −1.6. BM25 corpus memory distracts the model. |

\* CI excludes zero.

## Known limitations and failure modes

- Severity and issue type disagree with the human reference on more than half the rows.
- `MAINTAINABILITY` is over-predicted; `BUG → MAINTAINABILITY` is the top confusion.
- ~30% of raw outputs narrate in third person ("The reviewer suggests…"); `prana.serve` rewrites
  these before returning.
- Line number is emitted on ~26% of rows; when present it is usually within 3 lines.
- Very long hunks are cut in the middle to fit the 2,048-token window; findings in the omitted
  part cannot be raised.
- Labels come from a teacher model (`gemini-3.8-flash`) restructuring human comments, not from
  human annotators; teacher self-agreement is κ 0.92 (severity), 0.82 (issue type).
- Retrieval: BM25 over the training corpus only. Embedding retrieval and real repository history
  are untested.

## Training data in one paragraph

188,052 scraped review comments → remove AI-authored (19%, detected by account, not text),
author replies (43% of what remained, detected by the teacher's `is_review`), AI-tool markup,
short/conversational/prose rows → 12,483 training reviews. Splits are by pull request and by
repository; no PR or repo crosses splits. Details and every count: `data/v1.1/dataset-card.md`.
