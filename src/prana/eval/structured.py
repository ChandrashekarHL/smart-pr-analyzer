"""Per-field evaluation of structured review output.

    python -m prana.eval.structured --predictions runs/v1.0/eval/test_ood.predictions.jsonl

Never score ROUGE or BERTScore on the serialised JSON: two unrelated reviews
in the same schema share enough braces and key names to score 0.39, and an
empty-but-valid object scores 0.45 - above the v0.2.4 model it was meant to
measure. Each field gets the metric that fits it:

- JSON validity and schema validity: rates, reported separately.
- severity, issue_type: accuracy and macro-F1 **with the majority-class
  baseline beside them**, plus the top confusions for error analysis.
- line_number: exact-match rate and MAE where the reference has one.
- explanation (+ suggestion): ROUGE-L and BERTScore on the prose alone.

Every rate carries a bootstrap 95% CI and a per-language breakdown.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from prana.data import jsonl
from prana.eval.score import bertscore_f1, bootstrap_ci, rouge_l_scores
from prana.schema import ISSUE_TYPES, SEVERITIES, parse_structured


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def majority_baseline(y_true: list[str]) -> float:
    if not y_true:
        return 0.0
    return collections.Counter(y_true).most_common(1)[0][1] / len(y_true)


def confusions(y_true: list[str], y_pred: list[str], top: int = 5) -> list[dict]:
    c = collections.Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    return [{"true": t, "pred": p, "n": n} for (t, p), n in c.most_common(top)]


def _ci(values: list[float]) -> dict:
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def evaluate(rows: list[dict], with_bertscore: bool = True) -> dict:
    """`rows` carry `prediction` (model text) and `reference_structured` (dict)."""
    parsed = [parse_structured(r["prediction"]) for r in rows]
    refs = [r["reference_structured"] for r in rows]
    n = len(rows)

    out: dict = {"n": n, "overall": {}, "by_language": {}, "errors": {}}
    o = out["overall"]
    o["json_valid"] = _ci([1.0 if p.json_valid else 0.0 for p in parsed])
    o["schema_valid"] = _ci([1.0 if p.schema_valid else 0.0 for p in parsed])

    # Categorical fields are scored on every row: an unparseable prediction is
    # a wrong prediction, not a missing one. Scoring only the valid subset
    # would reward a model for refusing to answer the hard cases.
    sev_true = [r["severity"] for r in refs]
    sev_pred = [p.data["severity"] if p.ok else "__invalid__" for p in parsed]
    typ_true = [r["issue_type"] for r in refs]
    typ_pred = [p.data["issue_type"] if p.ok else "__invalid__" for p in parsed]
    for name, yt, yp, labels in (
        ("severity", sev_true, sev_pred, SEVERITIES),
        ("issue_type", typ_true, typ_pred, ISSUE_TYPES),
    ):
        hits = [1.0 if t == p else 0.0 for t, p in zip(yt, yp, strict=True)]
        o[name] = {
            "accuracy": _ci(hits),
            "majority_baseline": round(majority_baseline(yt), 4),
            "macro_f1": round(macro_f1(yt, yp, labels), 4),
        }
        out["errors"][f"{name}_confusions"] = confusions(yt, yp)

    # line_number: only where the reference has one; the model is told
    # nothing about line numbers for the other rows, so null there is correct.
    pairs = [
        (r["line_number"], p.data["line_number"] if p.ok else None)
        for r, p in zip(refs, parsed, strict=True)
        if r["line_number"] is not None
    ]
    if pairs:
        exact = [1.0 if t == p else 0.0 for t, p in pairs]
        mae = [abs(t - p) for t, p in pairs if p is not None]
        o["line_number"] = {
            "n": len(pairs),
            "exact": _ci(exact),
            "mae_when_present": round(sum(mae) / len(mae), 2) if mae else None,
            "missing": sum(1 for _, p in pairs if p is None),
        }

    # Prose quality on the explanation text alone, valid predictions only -
    # the validity rate above already charges for the rest.
    valid_idx = [i for i, p in enumerate(parsed) if p.ok]
    if valid_idx:
        preds = [
            parsed[i].data["explanation"] + " " + parsed[i].data["suggestion"]
            for i in valid_idx
        ]
        golds = [refs[i]["explanation"] + " " + refs[i]["suggestion"] for i in valid_idx]
        rouge = rouge_l_scores(preds, golds)
        o["explanation_rougeL"] = {**_ci(rouge), "n": len(valid_idx)}
        if with_bertscore:
            o["explanation_bertscore_raw"] = _ci(bertscore_f1(preds, golds, rescale=False))

    by_lang: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_lang[r.get("language") or "unknown"].append(i)
    for lang, idx in sorted(by_lang.items()):
        out["by_language"][lang] = {
            "n": len(idx),
            "schema_valid": round(sum(1 for i in idx if parsed[i].ok) / len(idx), 4),
            "severity_acc": round(
                sum(1 for i in idx if sev_true[i] == sev_pred[i]) / len(idx), 4
            ),
            "issue_type_acc": round(
                sum(1 for i in idx if typ_true[i] == typ_pred[i]) / len(idx), 4
            ),
        }

    # Error analysis: keep concrete examples, not just counts.
    invalid = [
        {"id": r["id"], "error": p.error, "prediction": r["prediction"][:300]}
        for r, p in zip(rows, parsed, strict=True)
        if not p.ok
    ][:25]
    out["errors"]["invalid_examples"] = invalid
    return out


def render(summary: dict, title: str) -> str:
    o = summary["overall"]
    lines = [f"\n{title}  (n={summary['n']})", "=" * 70]

    def rate(name, d):
        lo, hi = (100 * x for x in d["ci95"])
        return f"{name:<24}{100 * d['mean']:6.1f}%  [{lo:.1f}, {hi:.1f}]"

    def score(name, d):
        return f"{name:<24}{d['mean']:.4f}  [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"

    lines.append(rate("JSON valid", o["json_valid"]))
    lines.append(rate("schema valid", o["schema_valid"]))
    for name in ("severity", "issue_type"):
        d = o[name]
        lines.append(
            rate(f"{name} accuracy", d["accuracy"])
            + f"   majority {100 * d['majority_baseline']:.1f}%   macro-F1 {d['macro_f1']:.3f}"
        )
    if "line_number" in o:
        d = o["line_number"]
        lines.append(
            rate(f"line exact (n={d['n']})", d["exact"])
            + f"   MAE {d['mae_when_present']}   missing {d['missing']}"
        )
    if "explanation_rougeL" in o:
        d = o["explanation_rougeL"]
        lines.append(score("explanation ROUGE-L", d) + f"   (valid rows only, n={d['n']})")
    if "explanation_bertscore_raw" in o:
        lines.append(score("explanation BERTScore", o["explanation_bertscore_raw"]))
    lines.append("-" * 70)
    lines.append(f"{'language':<12}{'n':>6}{'valid':>8}{'sev acc':>9}{'type acc':>10}")
    for lang, d in summary["by_language"].items():
        lines.append(
            f"{lang:<12}{d['n']:>6}{100 * d['schema_valid']:>7.1f}%"
            f"{100 * d['severity_acc']:>8.1f}%{100 * d['issue_type_acc']:>9.1f}%"
        )
    lines.append("-" * 70)
    for name in ("severity_confusions", "issue_type_confusions"):
        conf = summary["errors"].get(name) or []
        if conf:
            lines.append(
                f"{name}: " + ", ".join(f"{c['true']}->{c['pred']} x{c['n']}" for c in conf)
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()

    path = Path(args.predictions)
    rows = [
        r
        for r in jsonl.read(path)
        if r.get("prediction") is not None and r.get("reference_structured")
    ]
    if not rows:
        raise SystemExit(f"no rows with both a prediction and reference_structured in {path}")
    summary = evaluate(rows, with_bertscore=not args.no_bertscore)
    run_meta = path.with_suffix("").with_suffix(".run.json")
    if run_meta.exists():
        summary["run"] = json.loads(run_meta.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else path.with_suffix(".structured.metrics.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(render(summary, path.stem))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
