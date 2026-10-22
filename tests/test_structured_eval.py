"""Tests for the per-field evaluator. No models: predictions are strings and
references are dicts, so every number can be checked by hand."""

from __future__ import annotations

import json

import pytest

from prana.eval.structured import confusions, evaluate, macro_f1, majority_baseline
from prana.schema import render_structured


def _ref(sev="warning", typ="BUG", line=10):
    return {
        "file_path": "a.go",
        "line_number": line,
        "severity": sev,
        "issue_type": typ,
        "explanation": "Nil is unguarded here.",
        "suggestion": "Guard it.",
    }


def _row(i, pred, ref, lang="go"):
    return {"id": str(i), "language": lang, "prediction": pred, "reference_structured": ref}


def test_perfect_predictions_score_perfectly():
    rows = [_row(i, render_structured(_ref(line=i)), _ref(line=i)) for i in range(6)]
    s = evaluate(rows, with_bertscore=False)["overall"]
    assert s["json_valid"]["mean"] == 1.0 and s["schema_valid"]["mean"] == 1.0
    assert s["severity"]["accuracy"]["mean"] == 1.0
    assert (
        s["line_number"]["exact"]["mean"] == 1.0 and s["line_number"]["mae_when_present"] == 0
    )
    assert s["explanation_rougeL"]["mean"] == pytest.approx(1.0)


def test_invalid_prediction_counts_as_wrong_not_missing():
    """Scoring only the parseable subset would reward refusing to answer."""
    rows = [
        _row(0, render_structured(_ref()), _ref()),
        _row(1, "I think this is fine.", _ref()),
    ]
    s = evaluate(rows, with_bertscore=False)["overall"]
    assert s["schema_valid"]["mean"] == 0.5
    assert s["severity"]["accuracy"]["mean"] == 0.5
    assert s["explanation_rougeL"]["n"] == 1


def test_majority_baseline_is_reported_beside_accuracy():
    refs = [_ref("info")] * 7 + [_ref("critical")] * 3
    rows = [_row(i, render_structured(_ref("info")), r) for i, r in enumerate(refs)]
    s = evaluate(rows, with_bertscore=False)["overall"]["severity"]
    # Always guessing the majority class scores exactly the baseline.
    assert s["accuracy"]["mean"] == pytest.approx(0.7)
    assert s["majority_baseline"] == pytest.approx(0.7)


def test_confusions_list_top_mistakes():
    assert confusions(["a", "a", "b"], ["b", "b", "b"]) == [{"true": "a", "pred": "b", "n": 2}]


def test_macro_f1_treats_classes_equally():
    # Perfect on a frequent class, wrong on a rare one: macro-F1 must drop well
    # below accuracy, which is the whole reason to report it.
    yt = ["BUG"] * 9 + ["SECURITY"]
    yp = ["BUG"] * 10
    assert majority_baseline(yt) == 0.9
    assert macro_f1(yt, yp, ("BUG", "SECURITY")) < 0.6


def test_line_number_scored_only_where_reference_has_one():
    rows = [
        _row(0, render_structured(_ref(line=None)), _ref(line=None)),
        _row(1, render_structured(_ref(line=12)), _ref(line=10)),
    ]
    s = evaluate(rows, with_bertscore=False)["overall"]["line_number"]
    assert s["n"] == 1 and s["exact"]["mean"] == 0.0 and s["mae_when_present"] == 2


def test_per_language_breakdown_and_error_examples():
    rows = [
        _row(0, render_structured(_ref()), _ref(), "go"),
        _row(1, json.dumps({"severity": "high"}), _ref(), "rust"),
    ]
    s = evaluate(rows, with_bertscore=False)
    assert s["by_language"]["go"]["schema_valid"] == 1.0
    assert s["by_language"]["rust"]["schema_valid"] == 0.0
    assert s["errors"]["invalid_examples"][0]["id"] == "1"
