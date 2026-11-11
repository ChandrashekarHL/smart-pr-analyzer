"""The issue-detection judge: prompt, parsing, misses and aggregation, offline."""

from __future__ import annotations

import json

from prana.eval.judge import (build_prompt, judge_rows, model_review_text, parse_verdicts,
                              summarize)

VALID = json.dumps({"explanation": "SQL built from input.", "suggestion": "Use params.",
                    "severity": "critical", "issue_type": "SECURITY", "file_path": None,
                    "line_number": 4})


def _item(i, model="x", lang="python"):
    return {"id": f"r{i}", "language": lang, "hunk": "@@ -1 +1 @@\n+q = f'{x}'",
            "human": "This is SQL injection.", "model": model}


class FakeJudge:
    def __init__(self, verdicts, fail_first=False):
        self.verdicts = verdicts           # id -> verdict
        self.calls = 0
        self.fail_first = fail_first

    def generate(self, prompt, system=None, schema=None):
        self.calls += 1
        assert system and schema                    # the judge's own, never the labeller's
        if self.fail_first and self.calls == 1:
            raise ValueError("HTTP 500")
        ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("=== item ")]
        return json.dumps([{"id": i, "verdict": self.verdicts.get(i, "different"),
                            "model_valid": "yes", "reason": "ok"} for i in ids])


def test_model_review_text_and_unparseable_output():
    text = model_review_text(VALID)
    assert "SQL built from input." in text and "Suggestion: Use params." in text
    assert "severity critical, type SECURITY, line 4" in text
    assert model_review_text("not json") is None


def test_prompt_contains_all_three_parts_and_cuts_long_hunks():
    item = _item(1)
    item["hunk"] = "a" * 10_000
    p = build_prompt([item])
    assert "=== item r1 (python) ===" in p and "This is SQL injection." in p
    assert "[cut]" in p and len(p) < 4_000


def test_parse_ignores_unknown_ids_and_bad_verdicts():
    text = json.dumps([{"id": "r1", "verdict": "same", "model_valid": "yes", "reason": "a"},
                       {"id": "zz", "verdict": "same"}, {"id": "r2", "verdict": "maybe"}])
    assert list(parse_verdicts(text, ["r1", "r2"])) == ["r1"]


def test_unparseable_model_output_is_a_miss_without_a_judge_call():
    judge = FakeJudge({})
    out = judge_rows(judge, [_item(1, model=None)], log=lambda *a: None)
    assert out[0]["verdict"] == "different" and out[0]["judged"] is False and judge.calls == 0


def test_batches_retry_and_never_guess():
    judge = FakeJudge({"r0": "same", "r1": "partial"}, fail_first=True)
    out = judge_rows(judge, [_item(i) for i in range(3)], batch=8, log=lambda *a: None)
    assert [r["verdict"] for r in out] == ["same", "partial", "different"]
    assert judge.calls == 2                       # one failure, one retry


def test_summary_rates_and_languages():
    rows = [{"id": "a", "language": "go", "verdict": "same", "model_valid": "yes"},
            {"id": "b", "language": "go", "verdict": "partial", "model_valid": "yes"},
            {"id": "c", "language": "rust", "verdict": "different", "model_valid": "no"},
            {"id": "d", "language": "rust", "verdict": "different", "model_valid": "unsure",
             "judged": False}]
    s = summarize(rows)
    assert s["issue_detection_strict"]["mean"] == 0.25
    assert s["issue_detection_lenient"]["mean"] == 0.5
    assert s["model_finding_valid"]["mean"] == 0.5 and s["not_judged"] == 1
    assert s["by_language"]["go"] == {"rows": 2, "strict": 0.5, "lenient": 1.0}


# ------------------------------------------------------------ planted benchmark

def test_bench_cases_are_well_formed():
    from prana.eval.bench_cases import CASES, as_hunk
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) == 40
    langs = {c["language"] for c in CASES}
    assert langs == {"python", "typescript", "javascript", "go", "rust"}
    for c in CASES:
        assert c["bad"].strip() != c["good"].strip() and c["issue"].endswith(".")
        assert c["category"] in {"SECURITY", "BUG", "PERF"}
        h = as_hunk(c["bad"])
        assert h.startswith("@@ -0,0 +1,") and all(ln.startswith("+") for ln in h.splitlines()[1:])


def test_bench_metrics_count_alarms_and_detection():
    from prana.eval.bench import metrics
    from prana.eval.bench_cases import CASES
    alarm = json.dumps({"explanation": "x", "suggestion": "", "severity": "warning",
                        "issue_type": "BUG", "file_path": None, "line_number": None})
    info = alarm.replace("warning", "info")
    preds, verdicts = {}, {}
    for i, c in enumerate(CASES):
        preds[(c["id"], "bad")] = {"prediction": alarm}
        preds[(c["id"], "good")] = {"prediction": alarm if i < 10 else info}
        verdicts[f"{c['id']}:bad"] = {"verdict": "same" if i < 20 else "different"}
        verdicts[f"{c['id']}:good"] = {"model_valid": "yes" if i < 5 else "no"}
    m = metrics(preds, verdicts)
    assert m["issue_detection_strict"]["rate"] == 0.5
    assert m["false_positive_rate_alarm"]["rate"] == 0.25           # 10 of 40 alarms
    assert m["false_positive_rate_judged"]["rate"] == 0.125         # 5 of those not accepted
    assert m["correct_code_info_comment_rate"]["rate"] == 0.75
