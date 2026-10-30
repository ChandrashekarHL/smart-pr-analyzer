# Requirements traceability

Every requirement, milestone and target in the two governing documents —
`PR_Analyzer_Design_Document.docx` (v1.0) and `PR_Analyzer_Project_Document.docx`
(v2.0, which supersedes v1.0 where they differ) — mapped to its status and the
evidence for it. Numbers are measurements unless marked *estimate*. Updated
2026-10-30.

Status key: **Done** · **Partial** (part built or part measured) · **In progress**
(being built or measured now) · **Not started** · **Deviation** (done differently,
with the reason).

## Functional requirements (project doc §8)

| ID | Requirement | Status | Evidence / note |
|---|---|---|---|
| FR-01 | Scrape 50k–80k PR reviews, Python/JS, `{code_diff} → {review_comment}` with severity labels | **Deviation** | 188,052 comments scraped from 656 repos. After removing AI-authored reviews (19%), PR-author replies (43% of the rest) and low-quality rows, **12,483** training reviews remain. The 50k target would have meant training on bots and author replies. Severity comes from a teacher model, not keyword rules (κ 0.92). `data/v1.1/dataset-card.md` |
| FR-02 | QLoRA 4-bit fine-tune of DeepSeek-Coder 6.7B; severity + line numbers | **Done** | v1.1: r16/α32, NF4, 633 steps. Severity macro-F1 0.352; line exact 14.7% where the reference has one. `MODEL_CARD.md` |
| FR-03 | RAG: git history, ChromaDB embeddings, repo conventions in the prompt | **In progress** | Built 2026-10-30 to Module 3: `prana.rag.contract` (six sources). Code sources measured on test_ood (822 rows): line exact −8.8 pts\*, ROUGE-L −0.005\*, others not significant → **off in serving**. Full six-source evaluation (ablations B, C) running |
| FR-04 | FastAPI webhook, async via Redis+Celery, inline comments | **Built, not deployed** | `backend/` (`prana_app`): HMAC-verified webhook, GitHub App JWT → installation tokens, one review per PR with inline comments on validated diff lines, summary comment. 31 tests; Docker smoke test on the Spark passed (signed webhook → queued → worker → GitHub). Needs the App registered and the VM (`backend/DEPLOY.md`) |
| FR-05 | issue_type, line_number, severity, explanation, suggestion, confidence | **Done** | `prana.schema` + `prana.serve`; 99.5% schema-valid on test_ood; confidence = exp(mean token log-prob) |
| FR-06 | Eval suite: ROUGE-L, BERTScore, issue detection, FPR, security detection; W&B | **Partial** | ROUGE-L/BERTScore per field, severity/issue accuracy + macro-F1 with majority baselines, bootstrap CIs, paired comparisons (`prana.eval.*`). Issue detection rate, FPR, security suite not yet. Training runs in W&B; eval metrics not yet |
| FR-07 | Flag SQLi, secrets, deserialization, input validation, path traversal | **Not started** | `SECURITY` is an issue type; the 100-PR synthetic suite is not built |
| FR-08 | Missing tests / untested paths | **Partial** | `TEST` issue type exists; not measured separately |
| FR-09 | Python + JS/TS, extensible | **Done** | Python, TypeScript, JavaScript, Go, Rust |
| FR-10 | Dashboard: history, confidence, feedback, latency, W&B | **Built, not deployed** | Streamlit (`backend/dashboard`) behind Caddy basic auth: reviews, p50/p95 review time per day against the 30 s target, severity, issue type, confidence histogram, 👍/👎 helpful rate (reactions collected every 15 min by a Celery beat task), recent runs and comments, optional W&B report embed. 10 tests incl. the page rendered headlessly; screenshot-checked against seeded data in the Docker stack |
| FR-11 | 200 ms webhook ack, async jobs, 100 concurrent | **Built, not deployed** | Webhook acknowledged in 112 ms (measured, Docker, cold); Celery + Redis (append-only) with late acks, 3 retries with back-off, failed runs kept as dead letters with replay. Throughput is bounded by the one GPU (~3.5 s per hunk); the queue absorbs bursts |
| FR-12 | Duplicate-comment suppression (cosine > 0.9) | **Built** | Exact fingerprint marker in every comment + text similarity ≥ 0.9 within 3 lines on the same file (`prana_app.review.is_duplicate`); stdlib similarity instead of embeddings, so the worker needs no ML libraries |

## Module 2 — Fine-Tuning Engine (project doc §4.2)

| Item | Doc says | We did | Status |
|---|---|---|---|
| 2.1 Base model | deepseek-coder-6.7b-instruct | same | **Done** |
| 2.1 Quantisation | 4-bit NF4, BitsAndBytes | NF4 + double quantisation, BitsAndBytes 0.50 | **Done** |
| 2.2 LoRA | r=16, α=32, dropout 0.05 | same | **Done** |
| 2.2 Target modules | q, k, v, o, gate, up, down | same 7 projections (checked in `runs/v1.1/adapter/adapter_config.json`) | **Done** |
| 2.2 Trainable params | ~40M, <0.6% | 40.0M = 0.59% of 6.74B | **Done** |
| 2.3 Trainer | TRL SFTTrainer with packing | HF `Trainer` + our own packing and completion-only loss (~60 lines, tested) | **Deviation** — SFTTrainer's API changed across the versions on the Spark; loss masking is now ours and tested |
| 2.3 Sequence length | 2,048 | 2,048 | **Done** |
| 2.3 Batch | 4 × accumulation 8 = 32 | 8 × 4 = 32 | **Done** (same effective batch) |
| 2.3 Learning rate | 2e-4, cosine, 3% warmup | **1e-4**, cosine, 3% warmup | **Deviation, untested** — carried over from the first baseline config; no 2e-4 run to compare. Eval loss plateaued by epoch 2.5, so there is no sign of under-training |
| 2.3 Epochs | 3 | 3 (v1.0, v1.1) | **Done** |
| 2.3 Hardware / time | Colab T4, ~8 h | DGX Spark GB10 + Unsloth (fastest of 4 measured setups), 12.5 h | **Deviation** — better hardware available |
| 2.3 W&B every 100 steps | loss, grad norm, eval loss | logged every 10 steps, eval every 50 | **Done** |
| 2.3 Checkpoints to HF Hub every 500 steps | HF Hub | local checkpoints every 50 steps, last 3 kept | **Partial** — not on HF Hub |
| 2.4 Fields | issue_type, severity, line_number, explanation, suggestion, confidence | all six; explanation first so the label follows the reasoning | **Done** |
| 2.4 issue_type values | SECURITY / BUG / STYLE / TEST / PERF | those five + MAINTAINABILITY, DOCS, ARCHITECTURE | **Deviation** — 8 types; maintainability was the most common real category |
| 2.4 confidence | model outputs 0–1 | computed from the model's token probabilities at serve time | **Deviation** — a generated number is not a calibrated confidence |
| 1.2 Severity labelling | keyword rule classifier | teacher model (gemini-3.8-flash) restating each human comment, κ 0.92 | **Deviation** — keyword rules could not separate warning from critical |
| 1.3 Prompt format | `### Instruction / ### Code Diff / ### Repo Context / ### Review` | DeepSeek's chat template, versioned prompt with fingerprint checked at serve time | **Deviation** — the instruct model was trained on its chat template |
| 1.3 Truncation | tiktoken, 2,048 tokens | the model's own tokenizer, prompt fit to 1,792 tokens | **Done** — tiktoken miscounts DeepSeek tokens |
| 1.3 Split | stratified 85/10/5 | by PR and by repository (held-out `test_ood`) | **Deviation** — a random split leaks the same PR into train and test |
| M2 | model on HF Hub; W&B run; eval loss < 1.5; 10 diffs → all fields | W&B run ✓; eval loss 1.157 ✓; 99.5% schema-valid on 1,190 held-out rows ✓; HF Hub ✗ | **Partial** |

## Module 3 — RAG Context Layer (project doc §4.3)

| Item | Status | Implementation |
|---|---|---|
| 3.1 all-MiniLM-L6-v2 embeddings, chunked repo, ChromaDB on disk | **Done** | `prana.rag.embed`: definition-aligned chunks (≤40 lines; MiniLM reads 256 tokens), persistent ChromaDB, cosine. Per-repo cap 2,000 chunks in evaluation (4,000 default), nearest the PR's files first |
| 3.1 Incremental re-embedding of changed files | **Not started** | Full rebuild per base commit for now (~25 s per repo, measured) |
| 3.2 Query = embedding of the new code; top-5 similar chunks | **Done** | `contract._similar`: added lines as the query, same language, own file excluded |
| 3.2 Last 10 commits touching the file | **Done** | `history.commit_items`; data from `GET /commits?path=&sha=<base>` (`prana.rag.commits`) |
| 3.2 Last 3 review comments on the file | **Done** | `history.review_items`: other PRs only, strictly earlier than the PR |
| 3.3 Prompt assembly, token budget ≤1,200 tokens, priority past_reviews > similar_code > git_history | **Done** | `contract.ORDER` keeps that order; 2,000-char budget (under 1,200 tokens); whole items only. Prompt then fit to 1,792 tokens with the real tokenizer (`render_within_budget`). LangChain/tiktoken not used: plain functions and the model's own tokenizer, which tiktoken would miscount |
| 3.4 Convention extraction (≥5 rules) from the repo's code | **Done** | `prana.rag.conventions`: measured rules with evidence, emitted only at ≥80% majority; up to 6 per language. Count per repo reported with the evaluation |
| 3.4 Cross-file dependency awareness | **Done** | `prana.rag.code`: definitions of names the change uses, from file → directory → parent |
| 3.4 Stale embedding detection | **Not started** | `meta.json` records the indexed commit; refresh policy not built |
| Interface for Module 4 | **Done** | One call per hunk, never raises, 2.5 s timeout; `REQUIREMENTS` states what the backend provides |

## Non-functional requirements (project doc §9)

| Category | Target | Status | Evidence / note |
|---|---|---|---|
| Latency | ≤30 s per PR; webhook ≤200 ms | **Built, not deployed** | 3–5 s per hunk (measured); the worker reviews the 8 largest hunks and stops calling the model at a 25 s budget, saying what it skipped. Webhook ack 112 ms (measured) |
| Availability | 99% uptime; error comment on failure | **Partial** | Backend: restart policies, health checks, retries, failure comment on the PR after the last retry (tested). Model API still a GPU-queue job; systemd needs sudo on the Spark |
| Scalability | 100 concurrent reviews | **Partial** | The queue accepts any number and the worker scales with `--concurrency`; generation is serialised on the one GPU, so throughput is ~1 hunk per 3.5 s |
| Reproducibility | Hyperparameters, dataset hash, artifacts in HF + W&B | **Partial** | Configs, dataset SHA-256s, `environment.json`, prompt fingerprints; training runs in W&B. HuggingFace Hub upload not done |
| Security | JWT, no keys in code, HMAC-SHA256, env secrets | **Built** | App JWT (RS256) → 1-hour installation tokens; HMAC-SHA256 on the raw body, constant time; secrets as files (not env values, so they never appear in `docker inspect`); non-root container; dashboard behind a bearer token; HTTPS by Caddy |
| Model quality | Issue detection ≥70%, BERTScore ≥0.75, ROUGE-L ≥0.42, security ≥80% | **Partial** | BERTScore 0.856 ✓. ROUGE-L 0.169 ✗ (see note). Detection/security not measured |
| Portability | Docker Compose; CPU fallback | **Partial** | `backend/docker-compose.yml` (api, worker, Postgres, Redis, Caddy), built and run on aarch64. The model needs the GPU; no CPU fallback |
| Maintainability | Modular, unit + integration tests, ≥80% coverage, CI | **Partial** | 495 model-side + 66 RAG + 31 backend tests pass; ruff clean (backend); GitHub Actions workflow written (`.github/workflows/ci.yml`), not yet running (repo not pushed); coverage not measured |

## Milestones (project doc §7)

| M# | Acceptance criteria | Status |
|---|---|---|
| M1 | ≥50k examples; splits; HF dataset; severity distribution | **Partial** — splits by PR and repo, severity distribution documented; 12,483 rows (deviation above); no HF upload |
| M2 | Model on HF Hub; W&B run; eval loss < 1.5; 10 diffs → all fields | **Partial** — eval loss 1.157 ✓, W&B run ✓, structured output ✓; not on HF Hub |
| M3 | ChromaDB indexes repo; top-5 relevant; prompt < 2048 tokens; ≥5 convention rules | **In progress** — index, top-5 and budget done; rule counts and relevance being measured |
| M4 | Inline comments on 5 test PRs within 30 s | **Blocked on deployment** — code done; needs the App registered and the VM |
| M5 | 10 concurrent jobs, retries, 200 ms ack | **Partial** — retries and 112 ms ack verified; 10-concurrent load test after deployment |
| M6 | ROUGE-L ≥0.42, BERTScore ≥0.75, detection ≥70%, FPR ≤15%, security ≥80% | **Partial** — see model quality |
| M7 | Ablations A, B, C in W&B | **In progress** — A done (+0.0093 ROUGE-L, CI excludes 0); B (BM25 corpus memory) negative; B and C for Module 3 running now; not yet in W&B |
| M8 | CI badge, live demo, video, blog, resume bullets | **Not started** |

## Notes on targets

- **ROUGE-L ≥ 0.42** compares one generated comment with one human's comment.
  Two humans reviewing the same hunk usually raise different points, so even a
  perfect reviewer scores far below 0.42 against a single reference. We
  report it honestly (0.169) beside per-field accuracy, macro-F1 and a human
  evaluation, which measure what the target was meant to.
- **Issue detection rate** needs a definition before it can be measured: the
  plan is the doc's own — a labelled set where each PR has known issues, and
  the synthetic security suite for FR-07.
- **The resume bullets** in both documents quote numbers (72% detection, F1
  0.78, 65k reviews) that were targets. Only measured numbers go on a resume.
