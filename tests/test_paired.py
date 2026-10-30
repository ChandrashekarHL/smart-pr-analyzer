"""Paired comparison: per-row scoring and the bootstrap."""

from __future__ import annotations

import json

from prana.data import jsonl
from prana.eval.paired import compare, paired_delta, row_scores

REF = {"explanation": "Guard nil.", "suggestion": "", "severity": "warning",
       "issue_type": "BUG", "file_path": None, "line_number": 4}


def _pred(sev="warning", issue="BUG", line=4, valid=True):
    if not valid:
        return {"prediction": "not json", "reference_structured": REF}
    obj = {"explanation": "x", "suggestion": "", "severity": sev, "issue_type": issue,
           "file_path": None, "line_number": line}
    return {"prediction": json.dumps(obj), "reference_structured": REF}


def test_row_scores_count_invalid_as_wrong():
    s = row_scores(_pred(valid=False))
    assert s["valid"] == 0 and s["severity"] == 0 and s["issue_type"] == 0
    assert s["line_exact"] == 0.0


def test_row_scores_line_is_none_when_reference_has_none():
    p = _pred()
    p["reference_structured"] = {**REF, "line_number": None}
    assert row_scores(p)["line_exact"] is None


def test_identical_systems_have_zero_delta_and_no_significance():
    a = {str(i): {"severity": float(i % 2)} for i in range(100)}
    r = paired_delta(a, a, "severity")
    assert r["delta"] == 0 and r["ci95"] == [0, 0] and not r["significant"]


def test_consistent_improvement_is_significant():
    a = {str(i): {"severity": 0.0} for i in range(200)}
    b = {str(i): {"severity": 1.0 if i % 4 else 0.0} for i in range(200)}
    r = paired_delta(a, b, "severity")
    assert abs(r["delta"] - 0.75) < 1e-9 and r["significant"] and r["ci95"][0] > 0.6


def test_compare_pairs_by_id_and_counts_unpaired(tmp_path):
    a = [{"id": "1", **_pred()}, {"id": "2", **_pred(sev="info")}, {"id": "3", **_pred()}]
    b = [{"id": "1", **_pred()}, {"id": "2", **_pred()}, {"id": "4", **_pred()}]
    jsonl.write(tmp_path / "a.jsonl", a)
    jsonl.write(tmp_path / "b.jsonl", b)
    res = compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl", rouge=False, n_boot=200)
    assert res["rows_paired"] == 2 and res["only_a"] == 1 and res["only_b"] == 1
    assert res["fields"]["severity"]["delta"] == 0.5
    assert "rougeL" not in res["fields"]
