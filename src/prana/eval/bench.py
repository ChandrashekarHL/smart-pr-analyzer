"""Planted-defect benchmark: issue detection and false positive rate
(design doc §11), on `bench_cases.CASES`.

    python -m prana.eval.bench --key-file ~/.config/prana/api.key --out runs/v1.1/bench-v1

1. Every case is reviewed twice through the model API: the version with the
   planted defect (`bad`) and the corrected version (`good`).
2. The judge (`prana.eval.judge`) decides whether the review of `bad` names
   the planted issue (same / partial / different), and whether each review's
   own claim is a real problem (`model_valid`).
3. Metrics:
   - issue detection rate = reviews of `bad` that name the planted issue
     (strict: same; lenient: same or partial);
   - false positive rate = reviews of `good` that raise a warning or critical
     finding; also reported as the stricter "alarm the judge does not accept
     as a real problem". The model always writes some comment (it was trained
     on commented hunks only), so info-level remarks are reported separately
     rather than counted as alarms.
Predictions and verdicts are written as JSONL; reruns resume.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from prana.data import jsonl
from prana.eval.bench_cases import CASES, as_hunk
from prana.eval.judge import judge_rows, model_review_text
from prana.eval.score import bootstrap_ci
from prana.schema import parse_structured

ALARM = ("warning", "critical")


def review_all(client, out: Path, log=print) -> dict[tuple[str, str], dict]:
    path = out / "predictions.jsonl"
    done = {(r["id"], r["variant"]): r for r in jsonl.iter_read(path)} if path.exists() else {}
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for c in CASES:
            for variant in ("bad", "good"):
                if (c["id"], variant) in done:
                    continue
                res = client.review(as_hunk(c[variant]), c["language"], file_path=c["path"])
                rec = {"id": c["id"], "variant": variant, "language": c["language"],
                       "category": c["category"], "prediction": res.get("text") or "",
                       "confidence": res.get("confidence")}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(c["id"], variant)] = rec
        log(f"reviews: {len(done)}")
    return done


def judge_all(gemini, preds: dict, out: Path, log=print) -> dict[str, dict]:
    path = out / "verdicts.jsonl"
    done = {r["id"]: r for r in jsonl.iter_read(path)} if path.exists() else {}
    items = []
    for c in CASES:
        for variant in ("bad", "good"):
            key = f"{c['id']}:{variant}"
            if key in done:
                continue
            human = (f"Known issue in this code: {c['issue']}" if variant == "bad" else
                     "This version is correct: it is the fixed version of code that had "
                     f"this issue: {c['issue']} A careful reviewer would raise no defect here.")
            items.append({"id": key, "language": c["language"], "hunk": as_hunk(c[variant]),
                          "human": human,
                          "model": model_review_text(preds[(c["id"], variant)]["prediction"])})
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        def sink(r):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["id"]] = r
        judge_rows(gemini, items, sink=sink, log=log)
    return done


def _rate(xs: list[float]) -> dict:
    mean, lo, hi = bootstrap_ci(xs)
    return {"rate": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(xs)}


def metrics(preds: dict, verdicts: dict) -> dict:
    det_strict, det_lenient, fp_alarm, fp_judged, good_info = [], [], [], [], []
    by_cat: dict[str, list[float]] = collections.defaultdict(list)
    by_lang: dict[str, list[float]] = collections.defaultdict(list)
    sev_bad, sev_good = collections.Counter(), collections.Counter()
    for c in CASES:
        vb = verdicts.get(f"{c['id']}:bad", {})
        same = 1.0 if vb.get("verdict") == "same" else 0.0
        det_strict.append(same)
        det_lenient.append(1.0 if vb.get("verdict") in ("same", "partial") else 0.0)
        by_cat[c["category"]].append(same)
        by_lang[c["language"]].append(same)
        pb = parse_structured(preds[(c["id"], "bad")]["prediction"])
        sev_bad[(pb.data or {}).get("severity") if pb.schema_valid else "invalid"] += 1

        pg = parse_structured(preds[(c["id"], "good")]["prediction"])
        sev = (pg.data or {}).get("severity") if pg.schema_valid else "invalid"
        sev_good[sev] += 1
        alarm = sev in ALARM
        fp_alarm.append(1.0 if alarm else 0.0)
        vg = verdicts.get(f"{c['id']}:good", {})
        fp_judged.append(1.0 if alarm and vg.get("model_valid") != "yes" else 0.0)
        good_info.append(1.0 if sev == "info" else 0.0)
    return {
        "cases": len(CASES),
        "issue_detection_strict": _rate(det_strict),
        "issue_detection_lenient": _rate(det_lenient),
        "false_positive_rate_alarm": _rate(fp_alarm),
        "false_positive_rate_judged": _rate(fp_judged),
        "correct_code_info_comment_rate": _rate(good_info),
        "detection_by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
        "detection_by_language": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_lang.items())},
        "severity_on_buggy": dict(sev_bad),
        "severity_on_correct": dict(sev_good),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://100.124.121.97:8100")
    ap.add_argument("--key-file", required=True, help="model API key file")
    ap.add_argument("--gemini-key-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from prana.client import Client
    from prana.data.label import GeminiClient, load_key
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = Client(args.url, Path(args.key_file).read_text(encoding="utf-8").strip())
    preds = review_all(api, out, log=lambda *a: print(*a, flush=True))
    gemini = GeminiClient(load_key(args.gemini_key_file))
    verdicts = judge_all(gemini, preds, out, log=lambda *a: print(*a, flush=True))
    m = {**metrics(preds, verdicts), "judge_model": gemini.model}
    (out / "metrics.json").write_bytes((json.dumps(m, indent=2) + "\n").encode())
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
