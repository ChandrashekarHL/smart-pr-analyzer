"""Measure security detection and false positives (design doc FR-07, target
>= 80% detection) for each security system and their combinations, on the
33 vulnerable/fixed pairs of `prana.eval.bench_security`.

    python -m prana_app.security_eval --key-file ~/.config/prana/api.key \\
        --out ../runs/security/v1   # run with the backend environment

Systems:
  reviewer    the fine-tuned model, as it reviews today (/review)
  llm_pass    the LangChain security pass on the base model (/generate, adapter off)
  semgrep     Semgrep rules (p/security-audit, p/secrets)
Combinations count a vulnerability as detected if any member detects it and a
false alarm if any member raises one, so combining can never hide its cost.

Detection and validity are decided by the evaluation judge
(`prana.eval.judge`, temperature 0): a finding counts only if it names the
planted vulnerability. A reviewer comment on the fixed code counts as a false
alarm when it is warning/critical and the judge does not accept it as real;
for the security systems, any finding on fixed code the judge does not accept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prana.eval.bench_cases import as_hunk
from prana.eval.bench_security import CASES
from prana.eval.judge import judge_rows, model_review_text
from prana.eval.score import bootstrap_ci
from prana.schema import parse_structured

SYSTEMS = ("reviewer", "llm_pass", "semgrep")
COMBOS = {"reviewer+semgrep": ("reviewer", "semgrep"),
          "llm_pass+semgrep": ("llm_pass", "semgrep"),
          "all_three": ("reviewer", "llm_pass", "semgrep")}


def findings_text(findings: list[dict]) -> str | None:
    if not findings:
        return None
    return "\n".join(f"- {f.get('vulnerability') or f.get('rule')}: "
                     f"{f.get('explanation') or f.get('message')}" for f in findings)


def collect(api, chain, out: Path, log=print) -> dict:
    """Outputs of every system on every case and variant (resumable)."""
    from prana_app.security_pass import review_security
    from prana_app.semgrep_scan import scan
    path = out / "outputs.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for c in CASES:
        for variant in ("bad", "good"):
            key = f"{c['id']}:{variant}"
            if key in data:
                continue
            hunk = as_hunk(c[variant])
            rev = api.review(hunk, c["language"], file_path=c["path"])
            sec = [f.model_dump() for f in review_security(chain, hunk, c["language"], c["path"])]
            sg = [vars(f) for f in scan({c["path"]: c[variant]}).get(c["path"], [])]
            data[key] = {"reviewer": rev.get("text") or "", "llm_pass": sec, "semgrep": sg}
            path.write_text(json.dumps(data, indent=1))
        log(f"  collected {c['id']}")
    return data


def rescan_semgrep(data: dict, out: Path, log=print) -> dict:
    """Re-run Semgrep on each snippet *as a file* (snippet + the imports it
    uses, `snippet_files.as_file`), replace its outputs, and drop its old
    verdicts so they are judged again. The other systems are untouched."""
    from prana_app.semgrep_scan import scan
    from prana_app.snippet_files import as_file
    for c in CASES:
        for variant in ("bad", "good"):
            found = scan({c["path"]: as_file(c[variant], c["language"])}).get(c["path"], [])
            data[f"{c['id']}:{variant}"]["semgrep"] = [vars(f) for f in found]
        log(f"  rescanned {c['id']}")
    (out / "outputs.json").write_text(json.dumps(data, indent=1))
    path = out / "verdicts.jsonl"
    if path.exists():
        keep = [ln for ln in path.read_text().splitlines()
                if not json.loads(ln)["id"].endswith(":semgrep")]
        path.write_text("".join(ln + "\n" for ln in keep))
    return data


def judge(gemini, data: dict, out: Path, log=print) -> dict:
    path = out / "verdicts.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    items = []
    for c in CASES:
        for variant in ("bad", "good"):
            outs = data[f"{c['id']}:{variant}"]
            human = (f"Known vulnerability in this code: {c['issue']}" if variant == "bad" else
                     "This version is correct: it is the fixed version of code that had this "
                     f"vulnerability: {c['issue']} A careful reviewer would raise no defect here.")
            texts = {"reviewer": model_review_text(outs["reviewer"]),
                     "llm_pass": findings_text(outs["llm_pass"]),
                     "semgrep": findings_text(outs["semgrep"])}
            for system, text in texts.items():
                key = f"{c['id']}:{variant}:{system}"
                if key in done or text is None:
                    continue
                items.append({"id": key, "language": c["language"], "hunk": as_hunk(c[variant]),
                              "human": human, "model": text})
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        def sink(r):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["id"]] = r
        judge_rows(gemini, items, sink=sink, log=log)
    return done


def outcomes(data: dict, verdicts: dict) -> dict[str, dict[str, list[float]]]:
    """Per system: detection (strict) on bad, false alarm on good, per case."""
    res = {s: {"detect": [], "lenient": [], "fp": []} for s in (*SYSTEMS, *COMBOS)}
    for c in CASES:
        per = {}
        for s in SYSTEMS:
            vb = verdicts.get(f"{c['id']}:bad:{s}", {})
            vg = verdicts.get(f"{c['id']}:good:{s}", {})
            good_out = data[f"{c['id']}:good"][s]
            if s == "reviewer":
                p = parse_structured(good_out)
                severity = (p.data or {}).get("severity")
                raised = p.schema_valid and severity in ("warning", "critical")
            else:
                raised = bool(good_out)
            per[s] = (vb.get("verdict") == "same", vb.get("verdict") in ("same", "partial"),
                      raised and vg.get("model_valid") != "yes")
        for s in SYSTEMS:
            res[s]["detect"].append(float(per[s][0]))
            res[s]["lenient"].append(float(per[s][1]))
            res[s]["fp"].append(float(per[s][2]))
        for name, members in COMBOS.items():
            res[name]["detect"].append(float(any(per[m][0] for m in members)))
            res[name]["lenient"].append(float(any(per[m][1] for m in members)))
            res[name]["fp"].append(float(any(per[m][2] for m in members)))
    return res


def summarize(res: dict) -> dict:
    def r(xs):
        mean, lo, hi = bootstrap_ci(xs)
        return {"rate": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    return {s: {"detection": r(v["detect"]), "detection_lenient": r(v["lenient"]),
                "false_positive": r(v["fp"]), "cases": len(v["detect"])}
            for s, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://100.124.121.97:8100")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescan-semgrep", action="store_true",
                    help="re-run only Semgrep, on snippets as files, and re-judge it")
    args = ap.parse_args()

    from prana.client import Client
    from prana.data.label import GeminiClient, load_key

    from prana_app.security_pass import PranaChatModel, build_chain
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = Client(args.url, Path(args.key_file).read_text(encoding="utf-8").strip())
    chain = build_chain(PranaChatModel(client=api))
    if args.rescan_semgrep:
        data = rescan_semgrep(json.loads((out / "outputs.json").read_text()), out,
                              log=lambda *a: print(*a, flush=True))
    else:
        data = collect(api, chain, out, log=lambda *a: print(*a, flush=True))
    verdicts = judge(GeminiClient(load_key(None)), data, out, log=lambda *a: print(*a, flush=True))
    summary = summarize(outcomes(data, verdicts))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for s, v in summary.items():
        print(f"{s:18s} detection {v['detection']['rate']:.0%} {v['detection']['ci95']}  "
              f"lenient {v['detection_lenient']['rate']:.0%}  "
              f"false positives {v['false_positive']['rate']:.0%} {v['false_positive']['ci95']}")


if __name__ == "__main__":
    main()
