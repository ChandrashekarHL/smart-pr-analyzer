"""Tests for the teacher labeller. No network: a fake client stands in for
Gemini, which lets the batch/fallback/resume logic be tested exactly - the
parts that would otherwise mislabel rows silently."""

from __future__ import annotations

import json

import pytest

from prana.data import jsonl
from prana.data.label import (
    ISSUE_TYPES,
    SEVERITIES,
    LabelError,
    build_prompt,
    coerce_line_number,
    label_batch,
    label_file,
    parse_response,
    structured_output,
)


def _row(i, **kw):
    return {"id": f"org/repo_{i}-{i}", "language": "go", "file_path": f"pkg/f{i}.go",
            "line_number": str(10 + i), "diff_hunk": "@@ -1 +1 @@\n+x := y",
            "review_comment": f"Guard nil here, number {i}.", **kw}


def _label(rid, **kw):
    base = {"id": rid, "is_review": True, "severity": "warning", "issue_type": "BUG",
            "explanation": "Guard nil before use.", "suggestion": "Add an if x == nil check."}
    return {**base, **kw}


# ------------------------------------------------------------- prompt

def test_prompt_carries_every_id_and_comment():
    rows = [_row(1), _row(2)]
    p = build_prompt(rows)
    for r in rows:
        assert r["id"] in p and r["review_comment"] in p
    assert "2 item(s)" in p


def test_long_diff_is_truncated_but_comment_is_not():
    r = _row(1, diff_hunk="@@ -1 +1 @@\n" + "+a\n" * 5000)
    p = build_prompt([r])
    assert "[diff truncated]" in p
    assert r["review_comment"] in p


# ------------------------------------------------------------- parse

def test_valid_reply_parses():
    out = parse_response(json.dumps([_label("a"), _label("b")]), ["a", "b"])
    assert set(out) == {"a", "b"}
    assert out["a"]["severity"] == "warning"


@pytest.mark.parametrize("bad,why", [
    ("not json", "not JSON"),
    (json.dumps({"id": "a"}), "not a list"),
    (json.dumps([_label("a", severity="high")]), "bad severity"),
    (json.dumps([_label("a", issue_type="bug")]), "bad issue_type"),
    (json.dumps([_label("a", is_review="yes")]), "not boolean"),
    (json.dumps([_label("a", explanation="  ")]), "empty explanation"),
    (json.dumps([_label("zzz")]), "unexpected id"),
    (json.dumps([_label("a"), _label("a")]), "duplicate id"),
    (json.dumps([]), "missing ids"),
])
def test_off_schema_replies_are_rejected(bad, why):
    with pytest.raises(LabelError, match=why):
        parse_response(bad, ["a"])


def test_reply_for_subset_of_ids_is_rejected():
    """Fewer answers than questions would silently drop rows."""
    with pytest.raises(LabelError, match="missing ids"):
        parse_response(json.dumps([_label("a")]), ["a", "b"])


# ------------------------------------------------------------- structured output

@pytest.mark.parametrize("raw,expected", [("142", 142), ("None", None), (None, None),
                                          (" 7 ", 7), ("abc", None), (12, 12)])
def test_line_number_coercion(raw, expected):
    assert coerce_line_number(raw) == expected


def test_structured_output_has_schema_order_and_row_fields():
    row = _row(3)
    lab = parse_response(json.dumps([_label(row["id"])]), [row["id"]])[row["id"]]
    so = structured_output(row, lab)
    assert list(so) == ["file_path", "line_number", "severity", "issue_type",
                        "explanation", "suggestion"]
    assert so["file_path"] == "pkg/f3.go" and so["line_number"] == 13


def test_enums_match_the_handoff_taxonomy():
    assert set(SEVERITIES) == {"info", "warning", "critical"}
    assert set(ISSUE_TYPES) == {"BUG", "SECURITY", "PERF", "MAINTAINABILITY",
                                "TEST", "DOCS", "ARCHITECTURE", "STYLE"}


# ------------------------------------------------------------- fake client

class FakeClient:
    """Answers from a script: each entry is the text for one generate() call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.requests = 0
        self.retries = 0

    def generate(self, user_text):
        self.prompts.append(user_text)
        self.requests += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_batch_happy_path_is_one_request():
    rows = [_row(1), _row(2)]
    c = FakeClient([json.dumps([_label(r["id"]) for r in rows])])
    out = label_batch(c, rows)
    assert set(out) == {r["id"] for r in rows} and c.requests == 1


def test_bad_batch_falls_back_to_single_rows():
    """One malformed item must not sink its neighbours."""
    rows = [_row(1), _row(2)]
    c = FakeClient([
        "garbage",                                    # batch reply fails
        json.dumps([_label(rows[0]["id"])]),          # row 1 alone: fine
        json.dumps([_label("wrong-id")]),             # row 2 alone: bad
    ])
    out = label_batch(c, rows)
    assert set(out) == {rows[0]["id"]}
    assert c.requests == 3


def test_single_row_failure_raises():
    with pytest.raises(LabelError):
        label_batch(FakeClient(["garbage"]), [_row(1)])


# ------------------------------------------------------------- file driver

def test_label_file_writes_labels_and_resumes(tmp_path):
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    rows = [_row(i) for i in range(5)]
    jsonl.write(src, rows)

    c = FakeClient([json.dumps([_label(r["id"]) for r in rows[:3]])])
    stats = label_file(src, out, c, rows_per_request=3, workers=1, limit=3, log=lambda *_: None)
    assert stats["labelled"] == 3
    got = jsonl.read(out)
    assert all("structured_output" in r for r in got)

    # Second run: the three are skipped, only the remaining two are requested.
    c2 = FakeClient([json.dumps([_label(r["id"]) for r in rows[3:]])])
    stats2 = label_file(src, out, c2, rows_per_request=3, workers=1, log=lambda *_: None)
    assert stats2["labelled"] == 2 and c2.requests == 1
    assert len(jsonl.read(out)) == 5


def test_failed_rows_are_recorded_not_dropped(tmp_path):
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    jsonl.write(src, [_row(1)])
    stats = label_file(src, out, FakeClient(["garbage"]), rows_per_request=1, workers=1,
                       log=lambda *_: None)
    assert stats["failed"] == 1
    r = jsonl.read(out)[0]
    assert r["label"] is None and "label_error" in r


def test_not_review_rows_are_counted(tmp_path):
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    row = _row(1)
    jsonl.write(src, [row])
    c = FakeClient([json.dumps([_label(row["id"], is_review=False)])])
    stats = label_file(src, out, c, rows_per_request=1, workers=1, log=lambda *_: None)
    assert stats["not_review"] == 1
