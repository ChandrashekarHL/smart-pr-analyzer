"""Paired comparison of two prediction files on the same rows.

    python -m prana.eval.paired --a runs/x/none/test_ood.predictions.jsonl \\
                                --b runs/x/code/test_ood.predictions.jsonl

Reports B minus A per field with a paired bootstrap 95% CI over rows (same
row, two systems - far tighter than comparing two independent means). A CI
that excludes zero is marked `*`. Rows present in only one file are dropped
and counted. Every decision to switch a component on goes through this.

Per-row scores: schema validity; severity and issue type correct (an invalid
output counts as wrong); line number exact where the reference has one;
explanation ROUGE-L (optional, needs rouge-score).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prana.data import jsonl
from prana.schema import parse_structured

FIELDS = ("valid", "severity", "issue_type", "line_exact", "rougeL")


def row_scores(pred: dict) -> dict:
    p = parse_structured(pred.get("prediction") or "")
    data = p.data if p.schema_valid and p.data else {}
    ref = pred.get("reference_structured") or {}
    has_line = ref.get("line_number") is not None
    return {
        "valid": float(p.schema_valid),
        "severity": float(data.get("severity") is not None
                          and data.get("severity") == ref.get("severity")),
        "issue_type": float(data.get("issue_type") is not None
                            and data.get("issue_type") == ref.get("issue_type")),
        "line_exact": (float(data.get("line_number") == ref.get("line_number"))
                       if has_line else None),
        "_explanation": data.get("explanation") or "",
        "_reference": ref.get("explanation") or "",
    }


def score_file(path: Path, rouge: bool) -> dict[str, dict]:
    rows = {r["id"]: row_scores(r) for r in jsonl.iter_read(path)}
    if rouge and rows:
        from prana.eval.score import rouge_l_scores
        ids = sorted(rows)
        vals = rouge_l_scores([rows[i]["_explanation"] for i in ids],
                              [rows[i]["_reference"] for i in ids])
        for i, v in zip(ids, vals, strict=True):
            rows[i]["rougeL"] = v
    return rows


def paired_delta(a: dict[str, dict], b: dict[str, dict], field: str,
                 n_boot: int = 5000, seed: int = 0) -> dict | None:
    ids = sorted(i for i in a.keys() & b.keys()
                 if a[i].get(field) is not None and b[i].get(field) is not None)
    if not ids:
        return None
    d = [b[i][field] - a[i][field] for i in ids]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {"n": n, "a": sum(a[i][field] for i in ids) / n,
            "b": sum(b[i][field] for i in ids) / n, "delta": sum(d) / n,
            "ci95": [lo, hi], "significant": lo > 0 or hi < 0}


def compare(a_path: Path, b_path: Path, rouge: bool = True, n_boot: int = 5000) -> dict:
    a, b = score_file(a_path, rouge), score_file(b_path, rouge)
    fields = [f for f in FIELDS if f != "rougeL" or rouge]
    return {"a": str(a_path), "b": str(b_path), "rows_paired": len(a.keys() & b.keys()),
            "only_a": len(a.keys() - b.keys()), "only_b": len(b.keys() - a.keys()),
            "fields": {f: paired_delta(a, b, f, n_boot) for f in fields}}


def format_report(res: dict) -> str:
    out = [f"B minus A on {res['rows_paired']} paired rows "
           f"(dropped: {res['only_a']} only in A, {res['only_b']} only in B)",
           f"  A = {res['a']}", f"  B = {res['b']}", ""]
    for f, r in res["fields"].items():
        if r is None:
            continue
        star = "*" if r["significant"] else " "
        out.append(f"  {f:11s} n={r['n']:5d}  A {r['a']:.3f}  B {r['b']:.3f}  "
                   f"delta {r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]{star}")
    out.append("\n  * 95% CI excludes zero")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline predictions")
    ap.add_argument("--b", required=True, help="candidate predictions")
    ap.add_argument("--no-rouge", action="store_true")
    ap.add_argument("--out", help="write the result as JSON here")
    args = ap.parse_args()
    res = compare(Path(args.a), Path(args.b), rouge=not args.no_rouge)
    print(format_report(res))
    if args.out:
        Path(args.out).write_bytes((json.dumps(res, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
