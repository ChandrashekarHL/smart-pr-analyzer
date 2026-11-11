"""Issue detection rate (design doc §11: "% of real bugs/issues in the test set
correctly flagged"), measured with an LLM judge.

    python -m prana.eval.judge --predictions runs/v1.1/eval-test_ood-r3/test_ood.predictions.jsonl \\
        --data data/v1.1/test_ood.jsonl --out runs/v1.1/judge-test_ood

Every held-out row has a human review comment: the issue that was really
there. For each row the judge reads the hunk, the human comment and the
model's review, and decides whether the model raised **the same issue**
(`same`), touched it without its main point (`partial`) or raised something
else (`different`). It also says whether the model's own finding is a real
problem in the hunk (`model_valid`), because two reviewers of the same code
often raise different, equally valid points.

Issue detection rate = same / rows (strict) and (same + partial) / rows
(lenient); an output that does not parse counts as a miss, never skipped.

The judge is the pinned teacher model at temperature 0. It compares against
the raw human comment, not the teacher's own restatement of it, and a sample
of verdicts is exported for a human to check (`--audit`). Appends and resumes.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from prana.data import jsonl
from prana.eval.score import bootstrap_ci
from prana.schema import parse_structured

HUNK_MAX_CHARS = 2500
COMMENT_MAX_CHARS = 1200
BATCH = 8

SYSTEM = """You are an expert software engineer acting as an impartial evaluator of \
automated code reviews.

Each item gives a diff hunk, the comment a human reviewer actually left on it \
(ground truth: the issue that was really there), and an automated review of \
the same hunk. For every item decide:

- verdict:
  "same"      - the automated review raises the same underlying issue as the \
human, even if worded differently or proposing a different fix;
  "partial"   - it concerns the same code and problem area but misses the \
human's main point, or is too vague to act on;
  "different" - it raises a different issue, or no concrete issue.
- model_valid: is the automated review's own issue a real, defensible problem \
in this hunk? "yes", "no" or "unsure".
- reason: one short sentence.

Judge substance, not wording, tone or length. Do not reward generic advice \
that would fit any code. Use only the information given."""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "verdict": {"type": "STRING", "enum": ["same", "partial", "different"]},
            "model_valid": {"type": "STRING", "enum": ["yes", "no", "unsure"]},
            "reason": {"type": "STRING"},
        },
        "required": ["id", "verdict", "model_valid", "reason"],
    },
}


def _cut(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n... [cut] ...\n" + text[-half:]


def model_review_text(prediction: str) -> str | None:
    """The model's review as prose, or None when the output does not parse."""
    p = parse_structured(prediction or "")
    if not p.schema_valid or not p.data:
        return None
    d = p.data
    out = [d.get("explanation") or ""]
    if d.get("suggestion"):
        out.append(f"Suggestion: {d['suggestion']}")
    out.append(f"(severity {d.get('severity')}, type {d.get('issue_type')}, "
               f"line {d.get('line_number')})")
    return "\n".join(out).strip()


def build_prompt(items: list[dict]) -> str:
    parts = ["Evaluate each item. Answer with a JSON array, one object per item, same ids.\n"]
    for it in items:
        parts.append(
            f"=== item {it['id']} ({it['language']}) ===\n"
            f"--- diff hunk ---\n{_cut(it['hunk'], HUNK_MAX_CHARS)}\n"
            f"--- human reviewer's comment (ground truth) ---\n"
            f"{_cut(it['human'], COMMENT_MAX_CHARS)}\n"
            f"--- automated review ---\n{it['model']}\n")
    return "\n".join(parts)


def parse_verdicts(text: str, ids: list[str]) -> dict[str, dict]:
    data = json.loads(text)
    out = {}
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and d.get("id") in ids and d.get("verdict") in (
                "same", "partial", "different"):
            out[d["id"]] = {"verdict": d["verdict"],
                            "model_valid": d.get("model_valid", "unsure"),
                            "reason": (d.get("reason") or "")[:300]}
    return out


def judge_rows(client, items: list[dict], batch: int = BATCH, sink=None,
               log=print) -> list[dict]:
    """items: {id, language, hunk, human, model}; `model` None = unparseable
    output, recorded as a miss without a judge call."""
    results = []
    todo = []
    for it in items:
        if it["model"] is None:
            r = {"id": it["id"], "language": it["language"], "verdict": "different",
                 "model_valid": "no", "reason": "model output did not parse",
                 "judged": False}
            results.append(r)
            if sink:
                sink(r)
        else:
            todo.append(it)
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        ids = [c["id"] for c in chunk]
        verdicts: dict[str, dict] = {}
        for _attempt in range(3):
            try:
                text = client.generate(build_prompt(chunk), system=SYSTEM, schema=SCHEMA)
                verdicts.update(parse_verdicts(text, [i for i in ids if i not in verdicts]))
            except (ValueError, json.JSONDecodeError) as e:
                log(f"  batch at {start}: {e}")
            if len(verdicts) == len(ids):
                break
        for c in chunk:
            v = verdicts.get(c["id"])
            if v is None:            # judge failed: say so, never guess
                v = {"verdict": "different", "model_valid": "unsure",
                     "reason": "judge gave no verdict", "judged": False}
            r = {"id": c["id"], "language": c["language"], **v, "judged": v.get("judged", True)}
            results.append(r)
            if sink:
                sink(r)
        if (start // batch) % 10 == 0:
            log(f"  judged {min(start + batch, len(todo))}/{len(todo)}")
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    same = [1.0 if r["verdict"] == "same" else 0.0 for r in results]
    lenient = [1.0 if r["verdict"] in ("same", "partial") else 0.0 for r in results]
    valid = [1.0 if r["model_valid"] == "yes" else 0.0 for r in results]

    def ci(xs):
        mean, lo, hi = bootstrap_ci(xs)
        return {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    by_lang = collections.defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
    return {
        "rows": n,
        "verdicts": dict(collections.Counter(r["verdict"] for r in results)),
        "not_judged": sum(1 for r in results if not r.get("judged", True)),
        "issue_detection_strict": ci(same),
        "issue_detection_lenient": ci(lenient),
        "model_finding_valid": ci(valid),
        "by_language": {lang: {"rows": len(rs),
                               "strict": round(sum(r["verdict"] == "same" for r in rs) / len(rs), 4),
                               "lenient": round(sum(r["verdict"] in ("same", "partial")
                                                    for r in rs) / len(rs), 4)}
                        for lang, rs in sorted(by_lang.items())},
    }


def load_items(predictions: Path, data: Path) -> list[dict]:
    rows = {r["id"]: r for r in jsonl.iter_read(data)}
    items = []
    for p in jsonl.iter_read(predictions):
        row = rows.get(p["id"])
        if row is None:
            continue
        items.append({"id": p["id"], "language": p.get("language") or row["language"],
                      "hunk": row["diff_hunk"], "human": row.get("review_comment") or "",
                      "model": model_review_text(p.get("prediction") or "")})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data", required=True, help="the split the predictions are for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--audit", type=int, default=40, help="rows exported for a human check")
    ap.add_argument("--key-file", default=None)
    args = ap.parse_args()

    from prana.data.label import GeminiClient, load_key
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "judgments.jsonl"
    done = {r["id"]: r for r in jsonl.iter_read(path)} if path.exists() else {}
    items = load_items(Path(args.predictions), Path(args.data))[: args.limit]
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} rows, {len(done)} judged, {len(todo)} to go", flush=True)
    client = GeminiClient(load_key(args.key_file))
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        def sink(r):
            f.write(json.dumps(r) + "\n")
            f.flush()
        judge_rows(client, todo, args.batch, sink, log=lambda *a: print(*a, flush=True))
    by_id = {r["id"]: r for r in jsonl.iter_read(path)}
    results = [by_id[it["id"]] for it in items if it["id"] in by_id]
    summary = {**summarize(results), "judge_model": client.model,
               "judge_version": client.model_version, "predictions": args.predictions}
    (out / "summary.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode())
    rnd = random.Random(13)
    sample = rnd.sample([it for it in items if it["model"] is not None],
                        min(args.audit, len(items)))
    with open(out / "audit.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in sample:
            f.write(json.dumps({**it, "judge": by_id.get(it["id"])}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
