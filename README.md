# prana — PR Analyzer

Dataset construction and fine-tuning for a repository-aware code-review model.
This repository covers **Module 1 (data)** and **Module 2 (fine-tuning)** of the
Smart GitHub PR Analyzer.

```bash
pip install -e ".[dev]"
pytest

python -m prana.data.build --out data/train --allow-any-license   # local training
python -m prana.data.build --out data/public                      # publishable subset
python -m prana.data.decontaminate --data data/train --apply
```

Both commands are also installed as `prana-build` and `prana-decon`.

---

## What the pipeline does, and why

### Bot reviews are removed by authorship, never by text

19% of the raw scrape was written by code-review bots. The obvious ones —
CodeRabbit, Greptile, Gemini — leave badge images and severity tables in the
comment body, so a regex finds them. **GitHub Copilot and OpenAI Codex do not.**
They write fluent prose that is indistinguishable from a human reviewer's, and
between them they contributed ~15,000 rows that a text filter passes straight
through. In the corpus we inherited they were the **first and third most common
"reviewers"**.

This matters because the project's claim is that the model learns from real
human review. Training on Copilot output is distillation from another model — a
legitimate technique, but a different claim, and one that would also contaminate
any held-out set used to report similarity "against human reviews".

So `filters.is_bot_reviewer` checks the `reviewer` field and runs before any
other filter. The text pattern is only a backstop for accounts we have not
enumerated. Pass `--drop-reviewer` to add one.

The `reviewer audit` printed at the end of every build exists so a new bot gets
noticed: it lists the highest-volume reviewers with the repository count beside
them, and flags anyone leaving thousands of comments across dozens of unrelated
projects. No person reviews that spread.

### Splitting by pull request is not enough

`test_ood` holds out **whole repositories**; `validation` and `test` hold out
whole pull requests from repositories seen in training. `assert_no_leakage`
fails the build if a PR ever lands in two splits.

That still leaves same-*code* leakage — cherry-picks, reverts, vendored
dependencies, generated files, forks. Measured on this corpus:

| split | contaminated |
|---|---|
| validation | 4.90% |
| test | 4.76% |
| **test_ood** | **0.78%** |

Nearly 5% of the held-out rows shared a 13-gram run with training despite a
clean PR split. Those rows are removed by `prana-decon --apply` and kept
alongside in `*.contaminated.jsonl` rather than deleted.

The gap between `test` and `test_ood` is itself the result: holding out whole
repositories is the more honest generalisation measure, and that is the number
to lead with.

### The prompt is a contract, not a string

`prana/prompts.py` is the only definition of the prompt. Training renders it,
serving renders it, and both log `PROMPT_FINGERPRINT` and refuse to run on a
mismatch. Train/serve prompt drift produces no error and no failing test — only
a quietly worse model — so it is made loud on purpose.

Two details the template encodes:

- **The context slot is always rendered**, empty or not. A model trained on bare
  diffs and then served 1,200 tokens of retrieved context meets a distribution it
  has never seen. Training applies context dropout so it handles both, which also
  makes the with/without-RAG ablation measure retrieval rather than formatting shock.
- **`RESPONSE_MARKER` ends the user turn** so loss can be masked to the completion.
  Without masking, most of the gradient teaches the model to generate diffs — which
  it is never asked to do at inference.

### Licences are split from training policy

Training locally on everything and publishing a derived dataset are different
acts. `data/train` keeps all 488 repositories; `data/public` keeps only
permissive ones (MIT, Apache-2.0, BSD, ISC, Unlicense, CC0). GitHub's `other`
label means "unclassified", not "permitted", so it is treated as unknown.

| corpus | rows | policy |
|---|---|---|
| `data/train` | ~30,000 | all licences — **never publish this one** |
| `data/public` | ~21,000 | permissive only — safe for the Hub |

### Credentials are redacted, not dropped

~490 rows carry something credential-shaped. Almost all are test fixtures
(`ghp_1234567890…`, `xoxb-test-token`), but "almost all" is not publishable, and
one live key in a public dataset outlives the project. Values become
`<REDACTED:github_pat>`; the row is kept, because the review comment beside it is
still a valid training target.

Email redaction is off by default: a bare email regex fires on
`@pytest.mark.asyncio` and Go module paths like `toolchain@v0.0.1-go1.26.2`.

### Every build is traceable

`DataConfig` holds every threshold and hashes to a fingerprint. Each run writes
`dataset-card.md` with that fingerprint, the prompt fingerprint, a SHA-256 per
split, full filter accounting, and the known limitations of the data. The same
input and config always produce byte-identical output.

---

## Known limitations of the data

- **Severity, issue type and confidence do not exist in the source.** GitHub
  stores none of them. Any model trained to emit the six-field schema needs a
  separate labelling pass; `suggested_fix` is the one exception, recoverable from
  ` ```suggestion ` blocks in ~10% of comments.
- **Absence of a comment is not a negative label.** A reviewer's silence does not
  mean a hunk is clean, so this corpus cannot directly supervise "no issue here".
- **Language balance reflects what was scraped**, not a designed sample. Go leads.

## Layout

```
src/prana/
  config.py              DataConfig — every threshold, hashed into the card
  prompts.py             the train/serve prompt contract
  data/
    build.py             raw scrape -> splits + dataset card
    filters.py           bot, quality and language predicates
    licenses.py          permissive / copyleft / unknown tiers
    scrub.py             credential redaction
    decontaminate.py     13-gram overlap via bloom filter
    jsonl.py             JSONL that survives U+2028 inside real diffs
tests/                   62 tests, no network, no fixtures on disk
```

---

## Evaluation

Generation and scoring are separate commands. Generation is the expensive half
and must survive a dropped SSH session; scoring is cheap and gets re-run every
time a metric is added.

```bash
# 1. baseline - the BASE model, before any fine-tuning.
#    This is ablation A. Without it there is no evidence fine-tuning helped.
python -m prana.eval.generate \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --split test_ood --out runs/baseline

# 2. score it
python -m prana.eval.score --predictions runs/baseline/test_ood.predictions.jsonl
```

**Generation resumes.** Predictions append to disk and the run skips ids already
present, so an interrupted job costs only the batch in flight. Re-running the
same command finishes what is left.

**Report `test_ood`, not `test`.** Whole repositories are held out and it carries
6x less n-gram contamination (0.78% against 4.76%).

**Every metric comes with a bootstrap 95% CI and a per-language breakdown.** A
single aggregate hides the two things that matter: whether a gap between runs
exceeds the noise, and whether one language is carrying the average.

Two numbers to read carefully:

- **ROUGE-L against free-form review comments lands around 0.10-0.25** in
  published work — two reviewers describing the same bug share almost no
  n-grams. A target of 0.42 is not reachable and should not be treated as a
  pass/fail gate.
- **Raw BERTScore F1 sits near 0.85 even for unrelated technical English.** Both
  raw and baseline-rescaled figures are reported; the rescaled one is the
  meaningful number.

`empty_rate` and the predicted-vs-human length medians are printed alongside,
because a base model that rambles for 400 words or returns nothing scores badly
for reasons ROUGE does not explain.

### Working across two machines

Anything that loads model weights runs on the DGX Spark; everything else runs on
a laptop.

| Work | Where |
|---|---|
| `prana-build`, `prana-decon`, tests, code | laptop |
| baseline eval, training, BERTScore | Spark |

Develop against a tiny model on CPU first — same code path, same collator, same
prompt rendering — then run the real thing on the Spark:

```bash
python -m prana.eval.generate --model sshleifer/tiny-gpt2 --limit 8 --out runs/smoke
```

Tokenizers without a chat template fall back to plain text so tiny test models
work; every real base model ships a template and uses it.
