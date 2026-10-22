"""Human usefulness evaluation: export a rating sheet, score it when it comes back.

    python -m prana.eval.human export \\
        --predictions runs/v1.1/eval-test_ood-r2/test_ood.predictions.jsonl \\
        --data data/v1.1 --split test_ood --n 100 --raters 4 --out runs/v1.1/human
    python -m prana.eval.human score --sheet runs/v1.1/human/ratings.csv

Every automatic metric here scores the model against the finding one human
happened to make. A developer reading the comment on a pull request asks a
different question: is this useful, and is the severity reasonable for what it
says? Four raters, 25 rows each, answer exactly that. The reference comment is
written to a separate answer key so raters judge the review on its own terms.

Rows are sampled stratified by language with a fixed seed, so the sheet is
reproducible and no language dominates it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import statistics
from pathlib import Path

from prana.data import jsonl
from prana.eval.score import bootstrap_ci
from prana.schema import parse_structured

RATING_COLUMNS = (
    "helpful_1to5",
    "severity_reasonable_yn",
    "issue_type_reasonable_yn",
    "grounded_in_diff_yn",
    "notes",
)


def stratified_sample(rows: list[dict], n: int, seed: int = 13) -> list[dict]:
    """Round-robin across languages so a 100-row sheet is not 40% Python."""
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_lang[r.get("language") or "unknown"].append(r)
    for v in by_lang.values():
        rng.shuffle(v)
    out: list[dict] = []
    langs = sorted(by_lang)
    while len(out) < n and any(by_lang.values()):
        for lang in langs:
            if by_lang[lang] and len(out) < n:
                out.append(by_lang[lang].pop())
    return out


def assign_raters(rows: list[dict], raters: int, seed: int = 13) -> list[int]:
    """Rater number per row, balanced, shuffled so no rater gets one language."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rater_of = [0] * len(rows)
    for k, idx in enumerate(order):
        rater_of[idx] = k % raters + 1
    return rater_of


def export(
    predictions: Path,
    data_dir: Path,
    split: str,
    n: int,
    raters: int,
    out: Path,
    seed: int = 13,
) -> dict:
    corpus = {r["id"]: r for r in jsonl.read(data_dir / f"{split}.jsonl")}
    valid = []
    for r in jsonl.read(predictions):
        p = parse_structured(r["prediction"])
        if p.ok and r["id"] in corpus:
            valid.append({**r, "parsed": p.data, "row": corpus[r["id"]]})
    sample = stratified_sample(valid, n, seed)
    rater_of = assign_raters(sample, raters, seed)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "ratings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rater",
                "id",
                "language",
                "file_path",
                "line_number",
                "diff_hunk",
                "model_explanation",
                "model_suggestion",
                "model_severity",
                "model_issue_type",
                *RATING_COLUMNS,
            ]
        )
        for r, rater in zip(sample, rater_of, strict=True):
            p = r["parsed"]
            w.writerow(
                [
                    rater,
                    r["id"],
                    r["language"],
                    p.get("file_path") or "",
                    p.get("line_number") if p.get("line_number") is not None else "",
                    r["row"]["diff_hunk"],
                    p["explanation"],
                    p["suggestion"],
                    p["severity"],
                    p["issue_type"],
                    *([""] * len(RATING_COLUMNS)),
                ]
            )

    # The human's original comment, kept apart so it does not anchor the rater.
    with open(out / "answer_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "human_comment", "ref_severity", "ref_issue_type"])
        for r in sample:
            ref = r.get("reference_structured") or {}
            w.writerow(
                [
                    r["id"],
                    r["row"].get("review_comment", ""),
                    ref.get("severity", ""),
                    ref.get("issue_type", ""),
                ]
            )

    (out / "README.md").write_text(
        "# Human rating sheet\n\n"
        f"{len(sample)} model reviews, {raters} raters (column `rater`), "
        f"~{len(sample) // raters} each.\n\n"
        "Fill the five rating columns in `ratings.csv`:\n\n"
        "| column | meaning |\n|---|---|\n"
        "| `helpful_1to5` | As the PR author, would this comment help you? "
        "1 = noise, 3 = fair point, 5 = I would act on it |\n"
        "| `severity_reasonable_yn` | y if the severity fits what the comment says |\n"
        "| `issue_type_reasonable_yn` | y if the category fits |\n"
        "| `grounded_in_diff_yn` | y if everything it claims is actually in the diff shown |\n"
        "| `notes` | anything, optional |\n\n"
        "Do not open `answer_key.csv` until you have finished your rows.\n"
        "Then: `python -m prana.eval.human score --sheet ratings.csv`\n",
        encoding="utf-8",
    )
    langs = collections.Counter(r["language"] for r in sample)
    return {"rows": len(sample), "raters": raters, "languages": dict(langs)}


def _yn(v: str) -> float | None:
    s = (v or "").strip().lower()
    if s in ("y", "yes", "1", "true"):
        return 1.0
    if s in ("n", "no", "0", "false"):
        return 0.0
    return None


def score(sheet: Path) -> dict:
    with open(sheet, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    helpful = [float(r["helpful_1to5"]) for r in rows if (r.get("helpful_1to5") or "").strip()]
    out: dict = {"rows": len(rows), "rated": len(helpful)}
    if helpful:
        mean, lo, hi = bootstrap_ci(helpful)
        out["helpful"] = {
            "mean": round(mean, 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "median": statistics.median(helpful),
            "share_4_or_5": round(sum(1 for h in helpful if h >= 4) / len(helpful), 3),
        }
    for col in ("severity_reasonable_yn", "issue_type_reasonable_yn", "grounded_in_diff_yn"):
        vals = [v for v in (_yn(r.get(col, "")) for r in rows) if v is not None]
        if vals:
            mean, lo, hi = bootstrap_ci(vals)
            out[col] = {
                "rate": round(mean, 3),
                "ci95": [round(lo, 3), round(hi, 3)],
                "n": len(vals),
            }
    by_rater: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if (r.get("helpful_1to5") or "").strip():
            by_rater[r["rater"]].append(float(r["helpful_1to5"]))
    out["by_rater_mean_helpful"] = {
        k: round(statistics.fmean(v), 2) for k, v in sorted(by_rater.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--predictions", required=True)
    e.add_argument("--data", default="data/v1.1")
    e.add_argument("--split", default="test_ood")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--raters", type=int, default=4)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--sheet", required=True)
    args = ap.parse_args()

    if args.cmd == "export":
        info = export(
            Path(args.predictions),
            Path(args.data),
            args.split,
            args.n,
            args.raters,
            Path(args.out),
        )
        print(json.dumps(info))
        print(f"wrote {Path(args.out) / 'ratings.csv'} (+ answer_key.csv, README.md)")
    else:
        print(json.dumps(score(Path(args.sheet)), indent=2))


if __name__ == "__main__":
    main()
