"""Score generated reviews against the human references.

Every number is reported with a bootstrap confidence interval and a per-language
breakdown. A single aggregate figure hides the two things that actually matter
here: whether a difference between runs is larger than the noise, and whether
the model is carrying one language and failing the rest.

    python -m prana.eval.score --predictions runs/baseline/test_ood.predictions.jsonl

BERTScore needs roberta-large, so on a laptop pass --no-bertscore and run the
full scoring on the Spark.

A caution on targets: ROUGE-L against free-form human review comments lands
around 0.10-0.25 in published work, because two reviewers describing the same
bug share almost no n-grams. Raw BERTScore F1 sits near 0.85 even for unrelated
technical English, which is why the rescaled figure is reported alongside it -
rescaling against the baseline is what makes the number mean something.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from prana.data import jsonl


def bootstrap_ci(
    values: list[float], confidence: float = 0.95, rounds: int = 1000, seed: int = 13
) -> tuple[float, float, float]:
    """Return (mean, low, high). Percentile bootstrap over per-example scores."""
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * rounds)]
    hi = means[int((1 + confidence) / 2 * rounds) - 1]
    return statistics.fmean(values), lo, hi


def rouge_l_scores(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs, strict=True)]


def bertscore_f1(preds: list[str], refs: list[str], rescale: bool) -> list[float]:
    from bert_score import score as bscore
    _, _, f1 = bscore(preds, refs, lang="en", rescale_with_baseline=rescale, verbose=False)
    return [float(x) for x in f1]


def empty_rate(preds: list[str]) -> float:
    """A base model with no fine-tuning often emits nothing usable. Reporting
    only ROUGE would hide that; this makes it explicit."""
    return sum(1 for p in preds if len(p.strip()) < 10) / max(len(preds), 1)


def summarise(rows: list[dict], metrics: dict[str, list[float]]) -> dict:
    out: dict = {"n": len(rows), "overall": {}, "by_language": {}}

    for name, values in metrics.items():
        mean, lo, hi = bootstrap_ci(values)
        out["overall"][name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    out["overall"]["empty_rate"] = round(empty_rate([r["prediction"] for r in rows]), 4)
    out["overall"]["pred_words_median"] = statistics.median(
        [len(r["prediction"].split()) for r in rows]
    )
    out["overall"]["ref_words_median"] = statistics.median(
        [len(r["reference"].split()) for r in rows]
    )

    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_lang[row.get("language") or "unknown"].append(i)

    for lang, idx in sorted(by_lang.items()):
        entry: dict = {"n": len(idx)}
        for name, values in metrics.items():
            mean, lo, hi = bootstrap_ci([values[i] for i in idx])
            entry[name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        out["by_language"][lang] = entry

    return out


def render(summary: dict, title: str) -> str:
    lines = [f"\n{title}  (n={summary['n']})", "=" * 72]
    names = [k for k in summary["overall"] if isinstance(summary["overall"][k], dict)]

    head = f"{'':<14}" + "".join(f"{n:>26}" for n in names)
    lines.append(head)
    row = f"{'overall':<14}"
    for n in names:
        m = summary["overall"][n]
        row += f"{m['mean']:>12.4f} [{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}]"
    lines.append(row)
    lines.append("-" * 72)

    for lang, entry in summary["by_language"].items():
        row = f"{lang:<10}{entry['n']:>4}"
        for n in names:
            m = entry[n]
            row += f"{m['mean']:>12.4f} [{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}]"
        lines.append(row)

    o = summary["overall"]
    lines += [
        "-" * 72,
        f"empty predictions   {100 * o['empty_rate']:.1f}%",
        f"median length       {o['pred_words_median']} words predicted "
        f"vs {o['ref_words_median']} words human",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None, help="defaults to <predictions>.metrics.json")
    ap.add_argument("--no-bertscore", action="store_true",
                    help="skip BERTScore (needs roberta-large; use on a laptop)")
    ap.add_argument("--no-rescale", action="store_true",
                    help="report raw BERTScore only, without baseline rescaling")
    ap.add_argument("--bootstrap-rounds", type=int, default=1000)
    args = ap.parse_args()

    path = Path(args.predictions)
    rows = [r for r in jsonl.read(path) if r.get("prediction") is not None]
    if not rows:
        raise SystemExit(f"no predictions in {path}")

    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]

    metrics: dict[str, list[float]] = {"rougeL": rouge_l_scores(preds, refs)}
    if not args.no_bertscore:
        metrics["bertscore_raw"] = bertscore_f1(preds, refs, rescale=False)
        if not args.no_rescale:
            metrics["bertscore_rescaled"] = bertscore_f1(preds, refs, rescale=True)

    summary = summarise(rows, metrics)

    run_meta_path = path.with_suffix("").with_suffix(".run.json")
    if run_meta_path.exists():
        summary["run"] = json.loads(run_meta_path.read_text(encoding="utf-8"))

    out_path = Path(args.out) if args.out else path.with_suffix(".metrics.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(render(summary, path.stem))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
