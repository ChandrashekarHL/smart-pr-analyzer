"""Tests for the shared review schema and its parser."""

from __future__ import annotations

import json

import pytest

from prana.schema import FIELDS, ISSUE_TYPES, SEVERITIES, parse_structured, render_structured

GOOD = {
    "file_path": "src/a.go",
    "line_number": 42,
    "severity": "warning",
    "issue_type": "BUG",
    "explanation": "Nil is not guarded.",
    "suggestion": "Check it.",
}


def test_render_is_one_line_in_schema_order():
    s = render_structured({**GOOD, "zzz": 1})
    assert "\n" not in s
    assert list(json.loads(s)) == list(FIELDS)
    assert "zzz" not in s


def test_render_then_parse_roundtrips():
    p = parse_structured(render_structured(GOOD))
    assert p.ok and p.data == GOOD


def test_fenced_json_is_tolerated():
    p = parse_structured("```json\n" + json.dumps(GOOD) + "\n```")
    assert p.ok


@pytest.mark.parametrize(
    "text,json_valid,why",
    [
        ("not json at all", False, "json"),
        ("[1, 2]", True, "not an object"),
        (json.dumps({k: v for k, v in GOOD.items() if k != "suggestion"}), True, "missing"),
        (json.dumps({**GOOD, "confidence": 0.9}), True, "extra"),
        (json.dumps({**GOOD, "severity": "high"}), True, "severity"),
        (json.dumps({**GOOD, "issue_type": "bug"}), True, "issue_type"),
        (json.dumps({**GOOD, "line_number": "42"}), True, "line_number"),
        (json.dumps({**GOOD, "line_number": True}), True, "line_number"),
        (json.dumps({**GOOD, "explanation": " "}), True, "explanation"),
        ("Sure! " + json.dumps(GOOD), False, "json"),
    ],
)
def test_invalid_outputs_are_classified(text, json_valid, why):
    p = parse_structured(text)
    assert not p.ok
    assert p.json_valid is json_valid
    assert why in (p.error or "")


def test_null_line_number_and_file_path_are_valid():
    assert parse_structured(json.dumps({**GOOD, "line_number": None, "file_path": None})).ok


def test_enums_are_the_handoff_taxonomy():
    assert set(SEVERITIES) == {"info", "warning", "critical"}
    assert len(ISSUE_TYPES) == 8 and "SECURITY" in ISSUE_TYPES
