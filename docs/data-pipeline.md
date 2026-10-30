# Data pipeline: what it does, and why

How 188,052 scraped review comments became 12,483 training examples, and the
reasoning behind each filter. Commands:

```bash
python -m prana.data.build --out data/train --allow-any-license   # local training
python -m prana.data.build --out data/public                      # publishable subset
python -m prana.data.decontaminate --data data/train --apply
python -m prana.data.label --in data/train_v2/train.jsonl --out data/train_v3/train.jsonl
python -m prana.data.finalize ...                                 # schema, deframing, context
```

The two first commands are also installed as `prana-build` and `prana-decon`.

## Bot reviews are removed by authorship, never by text

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

Later filter versions add: AI-tool markup posted under personal accounts
(filters v4) and a machine-translated review wrapper (277 rows, filters v3).

## Author replies are removed with the teacher's help

43% of the remaining comments were the **pull request's author answering a
reviewer** ("done", "fixed in abc123", "good point, will do"). A regex caught a
tenth of them. The teacher model's `is_review` label caught the rest, and it was
the largest single quality fix in the project.

## Splitting by pull request is not enough

`test_ood` holds out **whole repositories**; `validation` and `test` hold out
whole pull requests from repositories seen in training. `assert_no_leakage`
fails the build if a PR ever lands in two splits.

That still leaves same-*code* leakage — cherry-picks, reverts, vendored
dependencies, generated files, forks. Measured on the v1 corpus:

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

## Labels: a teacher restates, it never invents

GitHub stores no severity, issue type or explanation. A teacher model
(`gemini-3.8-flash`, pinned) restructures each **human** comment into the
six-field schema, instructed to restate the reviewer and never add a finding.
A 200-row pilot was checked by hand; re-labelling 400 rows gave agreement of
κ 0.92 (severity) and κ 0.82 (issue type). Third-person framing the teacher
sometimes adds ("The reviewer suggests…") is stripped at finalize
(`prana.data.deframe`).

## The prompt is a contract, not a string

`prana/prompts.py` is the only definition of the prompt, and it is versioned.
Training renders it, serving renders it, and both record the fingerprint;
serving refuses an adapter whose contract it does not know. Train/serve prompt
drift produces no error and no failing test — only a quietly worse model — so
it is made loud on purpose. `v1` is frozen byte-for-byte (a test pins its
fingerprint); `v2` adds new-file line numbers to every hunk line.

Two details the template encodes:

- **The context slot is always rendered**, empty or not, and training applies
  context dropout, so the with/without-retrieval ablation measures retrieval
  rather than formatting shock.
- **The response marker ends the user turn** so loss can be masked to the
  completion. Without masking, most of the gradient teaches the model to
  generate diffs — which it is never asked to do.

## Licences are split from training policy

Training locally on everything and publishing a derived dataset are different
acts. `data/train` keeps all repositories; `data/public` keeps only permissive
ones (MIT, Apache-2.0, BSD, ISC, Unlicense, CC0). GitHub's `other` label means
"unclassified", not "permitted", so it is treated as unknown. **The training
corpus is never published.**

## Credentials are redacted, not dropped

~490 rows carry something credential-shaped. Almost all are test fixtures
(`ghp_1234567890…`, `xoxb-test-token`), but "almost all" is not publishable, and
one live key in a public dataset outlives the project. Values become
`<REDACTED:github_pat>`; the row is kept, because the review comment beside it is
still a valid training target.

Email redaction is off by default: a bare email regex fires on
`@pytest.mark.asyncio` and Go module paths like `toolchain@v0.0.1-go1.26.2`.

## Every build is traceable

`DataConfig` holds every threshold and hashes to a fingerprint. Each run writes
`dataset-card.md` with that fingerprint, the prompt fingerprint, a SHA-256 per
split, full filter accounting, and the known limitations of the data. The same
input and config always produce byte-identical output.

## Known limitations of the data

- **Absence of a comment is not a negative label.** A reviewer's silence does not
  mean a hunk is clean, so this corpus cannot directly supervise "no issue here".
- **Labels come from a teacher model** restructuring human comments, not from
  human annotators.
- **Language balance reflects what was scraped**, not a designed sample.
