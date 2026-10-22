"""The structured review schema - one definition shared by labelling, training,
evaluation and serving.

Every place that produces or consumes a review object imports from here. The
alternative - each module keeping its own copy of the field list and enums -
is how a training target and a parser silently disagree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SEVERITIES: tuple[str, ...] = ("info", "warning", "critical")
ISSUE_TYPES: tuple[str, ...] = (
    "BUG",
    "SECURITY",
    "PERF",
    "MAINTAINABILITY",
    "TEST",
    "DOCS",
    "ARCHITECTURE",
    "STYLE",
)
# Order matters: this is the order the model *generates* the fields, and an
# autoregressive model commits to each token with only the ones before it as
# evidence. v1.0 emitted severity as roughly its third token - before it had
# written a word about the code - and collapsed to the majority class (93%
# `info`, 0% `critical`). Explanation and suggestion now come first, so the
# classification is conditioned on the reasoning. Parsing accepts any order;
# only rendering and training use this one.
FIELDS: tuple[str, ...] = (
    "explanation",
    "suggestion",
    "severity",
    "issue_type",
    "file_path",
    "line_number",
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.S)


def render_structured(obj: dict) -> str:
    """The exact serialisation the model is trained to emit: one line, key
    order as in FIELDS, non-ASCII kept. Training and evaluation both call this,
    so a prediction can be compared byte-for-byte when needed."""
    return json.dumps({k: obj.get(k) for k in FIELDS}, ensure_ascii=False)


@dataclass(frozen=True)
class Parsed:
    json_valid: bool
    schema_valid: bool
    data: dict | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.schema_valid


def parse_structured(text: str) -> Parsed:
    """Parse model output into a review object, reporting *why* it failed.

    JSON validity and schema validity are tracked separately: a model that
    emits well-formed JSON with `severity: "high"` has a different problem
    from one that emits prose, and the evaluation should say which.

    A surrounding ```json fence is tolerated - it is a formatting habit, not a
    content error - but nothing else is: stray text before or after the object
    counts as invalid, because a backend cannot post it.
    """
    if text is None:
        return Parsed(False, False, None, "empty")
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        return Parsed(False, False, None, f"json: {e.msg} at {e.pos}")
    if not isinstance(data, dict):
        return Parsed(True, False, None, "not an object")

    missing = [k for k in FIELDS if k not in data]
    if missing:
        return Parsed(True, False, data, f"missing {missing}")
    extra = [k for k in data if k not in FIELDS]
    if extra:
        return Parsed(True, False, data, f"extra {extra}")
    if data["severity"] not in SEVERITIES:
        return Parsed(True, False, data, f"severity {data['severity']!r}")
    if data["issue_type"] not in ISSUE_TYPES:
        return Parsed(True, False, data, f"issue_type {data['issue_type']!r}")
    ln = data["line_number"]
    if ln is not None and not (isinstance(ln, int) and not isinstance(ln, bool)):
        return Parsed(True, False, data, f"line_number {ln!r}")
    if not isinstance(data["explanation"], str) or not data["explanation"].strip():
        return Parsed(True, False, data, "explanation empty")
    if not isinstance(data["suggestion"], str):
        return Parsed(True, False, data, "suggestion not a string")
    if data["file_path"] is not None and not isinstance(data["file_path"], str):
        return Parsed(True, False, data, "file_path not a string")
    return Parsed(True, True, data)
