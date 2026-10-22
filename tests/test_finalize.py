"""Tests for the v1.0 corpus finalizer."""

from __future__ import annotations

import json

from prana.data.finalize import distribution, finalize_rows
from prana.schema import FIELDS


def _row(i, *, is_review=True, label=True, comment="Guard nil before use."):
    so = {
        "file_path": f"f{i}.go",
        "line_number": i,
        "severity": "warning",
        "issue_type": "BUG",
        "explanation": "Nil unguarded.",
        "suggestion": "Guard it.",
    }
    r = {
        "id": f"r/p_{i}-{i}",
        "language": "go",
        "review_comment": comment,
        "diff_hunk": "@@ -1 +1 @@\n+x",
        "label_model": "gemini-3.8-flash",
    }
    if label:
        r["label"] = {
            "is_review": is_review,
            "severity": "warning",
            "issue_type": "BUG",
            "explanation": "Nil unguarded.",
            "suggestion": "Guard it.",
        }
        r["structured_output"] = so
    else:
        r["label"] = None
        r["label_error"] = "HTTP 500"
    return r


def test_keeps_only_labelled_reviews():
    rows = [_row(1), _row(2, is_review=False), _row(3, label=False)]
    kept, stats = finalize_rows(rows)
    assert [r["id"] for r in kept] == ["r/p_1-1"]
    assert stats["drop_not_review"] == 1 and stats["drop_unlabelled"] == 1


def test_ai_tool_markup_is_dropped_even_when_teacher_says_review():
    """The badge rows are findings in form; the teacher keeps them. Only the
    markup filter knows they were not written by a person."""
    rows = [_row(1, comment='<!-- metadata:{"confidence":9} --> P2: Leaks a handle.')]
    kept, stats = finalize_rows(rows)
    assert not kept and stats["drop_ai_markup"] == 1


def test_target_text_is_canonical_serialisation():
    kept, _ = finalize_rows([_row(7)])
    r = kept[0]
    assert list(json.loads(r["target_text"])) == list(FIELDS)
    assert json.loads(r["target_text"])["line_number"] == 7
    assert list(r["structured_output"]) == list(FIELDS)


def test_distribution_counts_train_labels():
    kept, _ = finalize_rows([_row(1), _row(2)])
    assert distribution(kept, "severity") == {"warning": 2}
