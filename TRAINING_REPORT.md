# Training report — prana, v0.3 → v1.1

Six training runs over three weeks on one DGX Spark (GB10, 121 GB unified memory). Each run was
evaluated on held-out repositories before the next was designed; each failure below changed
the pipeline, and the change is tested in `tests/`.

## Summary

| Run | Date | Data | Target | Outcome |
|---|---|---|---|---|
| v0.3 | Sep 30 | corpus v1, 24.5k | prose | **Invalid** — tokenizer stripped all whitespace |
| v0.3.1 | Oct 3 | corpus v2, 22.4k | prose | Works; small real gain over base (ablation A) |
| v1.0 | Oct 19 | v1, 12.5k labelled | JSON | 98% valid JSON; severity collapsed to majority class |
| **v1.1** | Oct 22 | v1.1, 12.5k + context | JSON, reasoning first | Collapse broken, line numbers work; context hurts |

## Decisions that shaped the data

1. **AI-authored reviews removed by account, not text** (19% of the scrape). Copilot and Codex
   write prose no regex separates from a human's; in the inherited corpus they were the first
   and third most common "reviewers".
2. **Splits by PR and by repository**, plus 13-gram decontamination. PR splitting alone left
   ~5% of held-out rows sharing long code runs with training (cherry-picks, vendoring).
3. **Teacher labelling** (`gemini-3.8-flash`) to obtain severity, issue type, explanation and
   suggestion, which the source data does not contain. Instruction: restate the reviewer,
   never add a finding. 200-row hand-checked pilot; self-agreement κ 0.92 / 0.82.
4. **Author replies removed** — the teacher flagged 43% of comments as the PR author answering
   a reviewer. A regex had caught a tenth of these. This is the largest single quality fix.

## Run by run

### v0.3 — invalid
transformers 5.5 resolves DeepSeek's `LlamaTokenizerFast` to a rebuilt Metaspace tokenizer that
deletes every space, newline and tab. Loss looked normal for 10.5 h; output read
`Ithinkthisisabitof…`. **Fix:** `prana.compat.load_tokenizer` round-trips a whitespace probe,
rebuilds from the repo's `tokenizer.json` if needed, and refuses to proceed otherwise.

### v0.3.1 — prose baseline
QLoRA 4-bit + Unsloth, chosen by a measured 4-way benchmark (4-bit/Unsloth 0.368 samples/s vs
0.164–0.214 for the others). Ablation A on 2,199 test_ood rows: ROUGE-L 0.123 vs base 0.114,
paired Δ +0.0093 [+0.0072, +0.0115], better on 54.8% of rows; no gain on JavaScript. The main
learned effect is form: 64 words vs the base model's 162.

### v1.0 — structured output
98.2% valid JSON without constrained decoding; ~3 s per review in 4-bit. But severity accuracy
equalled the majority baseline (49.6% vs 49.4%): it predicted `info` 93% of the time and
`critical` never. Diagnosis: labels were reliable (teacher κ 0.92); the model wrote a valid but
*different* finding from the human's and graded it mildly; severity was its ~3rd token, emitted
before any reasoning; line numbers were unanswerable from a bare hunk (null 96%).

### v1.1 — four targeted changes
| Change | Measured effect |
|---|---|
| Field order: explanation, suggestion → severity, issue type | severity macro-F1 0.27 → 0.35; `critical` 0% → 7% of predictions |
| Prompt v2: new-file line numbers on every hunk line | line exact 0.7% → 14.0%, MAE 2.9 |
| Third-person teacher framing removed from labels | framed outputs 78% → 31% (rest rewritten at serve time) |
| BM25 retrieval context, 30% dropout | **negative** in ablation B — see below |

Paired against v1.0 on the same 1,190 rows: line +6.1 pts*, severity accuracy −4.0*, issue
type −2.2, validity −1.8* (\* CI excludes zero). The accuracy drop is the cost of no longer
always answering the majority class; macro-F1 is the fair comparison on a 49/37/14 split.

**Ablation B.** Same adapter, same rows, context on vs off: validity −2.5*, line −2.7*,
severity −2.0, issue type −1.6. Same direction on `test`, where same-repository memory was
available. Conclusion: BM25 "similar reviewed hunk" context is noise to this model; serve
without it. Embedding retrieval and real repository history remain untested.

**Self-consistency.** The teacher re-labelled the model's own explanations as if they were
human comments: 95.2% judged real reviews; severity matched the model's 82.4% (99.9% within
one level), issue type 84.7%. The categorical fields are coherent with what the model writes.

## Evaluation bug found late

Most invalid v1.0/v1.1 outputs (35 of 41 for v1.1) were prompts longer than the 2,048-token
window, truncated from the right so the model never saw "Review this hunk." and continued the
diff. A 6,000-character cap was too loose (code tokenises at ~2 chars/token). The fix renders
with the real tokenizer and shrinks the hunk until the prompt fits 1,792 tokens; it applies to
evaluation and serving. Final validity numbers: _r3 pending_.

## Reproducing

```bash
python -m prana.data.build --out data/train_v2 --allow-any-license
python -m prana.data.decontaminate --data data/train_v2 --apply
python -m prana.data.label --in data/train_v2/train.jsonl --out data/train_v3/train.jsonl   # per split
# on the Spark:
run-job start prana-v11 -- bash scripts/run_v11.sh
```
Every dataset card records the filter version, config fingerprint, teacher model and per-split
SHA-256.
