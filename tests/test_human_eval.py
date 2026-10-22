"""Tests for the human rating sheet exporter and scorer."""

from __future__ import annotations

import csv

from prana.data import jsonl
from prana.eval.human import assign_raters, export, score, stratified_sample
from prana.schema import render_structured


def _pred(i, lang):
    so = {
        "explanation": f"Finding {i}.",
        "suggestion": "Fix it.",
        "severity": "warning",
        "issue_type": "BUG",
        "file_path": "a.go",
        "line_number": i,
    }
    return {
        "id": f"r/p_{i}-{i}",
        "language": lang,
        "repo_name": "r/p",
        "reference": f"human {i}",
        "reference_structured": so,
        "prediction": render_structured(so),
    }


def test_stratified_sample_balances_languages():
    rows = [_pred(i, "python") for i in range(80)] + [_pred(100 + i, "rust") for i in range(20)]
    s = stratified_sample(rows, 40)
    langs = [r["language"] for r in s]
    assert langs.count("rust") == 20 and langs.count("python") == 20


def test_rater_assignment_is_balanced_and_stable():
    rows = [_pred(i, "go") for i in range(100)]
    a = assign_raters(rows, 4)
    assert sorted(set(a)) == [1, 2, 3, 4]
    assert all(a.count(k) == 25 for k in (1, 2, 3, 4))
    assert a == assign_raters(rows, 4)


def test_export_writes_sheet_without_reference_and_key_with_it(tmp_path):
    preds = tmp_path / "p.jsonl"
    rows = [_pred(i, "go" if i % 2 else "python") for i in range(12)]
    jsonl.write(preds, rows)
    data = tmp_path / "data"
    data.mkdir()
    jsonl.write(
        data / "test_ood.jsonl",
        [
            {"id": r["id"], "diff_hunk": "@@ -1 +1 @@\n+x", "review_comment": f"human {i}"}
            for i, r in enumerate(rows)
        ],
    )
    info = export(preds, data, "test_ood", n=8, raters=2, out=tmp_path / "h")
    assert info["rows"] == 8 and info["raters"] == 2
    with open(tmp_path / "h" / "ratings.csv", encoding="utf-8", newline="") as fh:
        sheet = list(csv.DictReader(fh))
    assert len(sheet) == 8
    assert "human_comment" not in sheet[0]  # reference kept out of the rater's view
    assert sheet[0]["helpful_1to5"] == ""  # blank, to be filled
    with open(tmp_path / "h" / "answer_key.csv", encoding="utf-8", newline="") as fh:
        key = list(csv.DictReader(fh))
    assert {k["id"] for k in key} == {s["id"] for s in sheet}


def test_score_reads_back_ratings(tmp_path):
    sheet = tmp_path / "ratings.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rater",
                "id",
                "helpful_1to5",
                "severity_reasonable_yn",
                "issue_type_reasonable_yn",
                "grounded_in_diff_yn",
                "notes",
            ]
        )
        w.writerow([1, "a", 5, "y", "y", "y", ""])
        w.writerow([1, "b", 3, "n", "y", "Y", ""])
        w.writerow([2, "c", 4, "yes", "no", "y", ""])
        w.writerow([2, "d", "", "", "", "", ""])  # unrated row is ignored
    s = score(sheet)
    assert s["rated"] == 3
    assert s["helpful"]["mean"] == 4.0 and s["helpful"]["share_4_or_5"] == round(2 / 3, 3)
    assert s["severity_reasonable_yn"]["rate"] == round(2 / 3, 3)
    assert s["grounded_in_diff_yn"]["rate"] == 1.0
    assert s["by_rater_mean_helpful"] == {"1": 4.0, "2": 4.0}
