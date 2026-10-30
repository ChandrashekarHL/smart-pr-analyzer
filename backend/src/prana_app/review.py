"""Turn model outputs into GitHub review comments: anchor, filter, render,
de-duplicate (FR-05, FR-12, and the design doc's line-number validator)."""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass

from prana_app.diff import Hunk

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
MARKER = "<!-- prana:{fp} -->"
_MARKER_RE = re.compile(r"<!-- prana:([0-9a-f]{16}) -->")
DUPLICATE_SIMILARITY = 0.9
DUPLICATE_LINE_WINDOW = 3


@dataclass
class Finding:
    path: str
    line: int
    line_exact: bool             # the model's line was inside the hunk
    severity: str
    issue_type: str
    explanation: str
    suggestion: str
    confidence: float
    model: str = ""
    adapter: str = ""

    @property
    def fingerprint(self) -> str:
        key = f"{self.path}\n{self.line}\n{_norm(self.explanation)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def body(self) -> str:
        icon = SEVERITY_ICON.get(self.severity, "")
        out = [f"{icon} **{self.severity} · {self.issue_type}** — {self.explanation.strip()}"]
        if self.suggestion.strip():
            out.append(f"\n**Suggestion:** {self.suggestion.strip()}")
        note = "" if self.line_exact else " · line approximate"
        out.append(f"\n<sub>prana · confidence {self.confidence:.2f}{note}</sub>")
        out.append(MARKER.format(fp=self.fingerprint))
        return "\n".join(out)

    def as_github_comment(self) -> dict:
        return {"path": self.path, "line": self.line, "side": "RIGHT", "body": self.body()}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def finding_from(hunk: Hunk, result: dict) -> Finding | None:
    """A Finding from one model API response, or None when the output did not
    parse (counted by the caller, never posted)."""
    parsed = result.get("parsed")
    if not parsed or not (parsed.get("explanation") or "").strip():
        return None
    line, exact = hunk.anchor(parsed.get("line_number"))
    return Finding(
        path=hunk.path, line=line, line_exact=exact,
        severity=parsed.get("severity") or "info",
        issue_type=parsed.get("issue_type") or "MAINTAINABILITY",
        explanation=parsed["explanation"], suggestion=parsed.get("suggestion") or "",
        confidence=float(result.get("confidence") or 0.0),
        model=result.get("model") or "", adapter=result.get("adapter") or "")


def is_duplicate(f: Finding, existing: list[dict]) -> bool:
    """`existing`: the PR's review comments ({path, line, body}). Same marker,
    or same file, nearby line and near-identical text."""
    for c in existing:
        m = _MARKER_RE.search(c.get("body") or "")
        if m and m.group(1) == f.fingerprint:
            return True
        if c.get("path") != f.path:
            continue
        line = c.get("line") or c.get("original_line")
        if isinstance(line, int) and abs(line - f.line) > DUPLICATE_LINE_WINDOW:
            continue
        body = _MARKER_RE.sub("", c.get("body") or "")
        ratio = difflib.SequenceMatcher(None, _norm(body), _norm(f.explanation)).ratio()
        if ratio >= DUPLICATE_SIMILARITY:
            return True
    return False


def summary(posted: list[Finding], low_confidence: list[Finding], reviewed: int,
            total: int, failed: int, skipped_for_time: int, duplicates: int,
            held_back_title: str = "Low-confidence suggestions") -> str:
    counts = {s: sum(f.severity == s for f in posted) for s in SEVERITY_ORDER}
    lines = ["### prana review",
             f"Reviewed **{reviewed}** of {total} changed hunks in supported languages."]
    if posted:
        lines.append(" · ".join(f"{SEVERITY_ICON[s]} {n} {s}" for s, n in counts.items() if n))
    else:
        lines.append("No inline comments.")
    notes = []
    if total > reviewed + skipped_for_time:
        notes.append(f"{total - reviewed - skipped_for_time} smaller hunks were not reviewed "
                     "(per-PR hunk limit)")
    if skipped_for_time:
        notes.append(f"{skipped_for_time} hunks skipped to stay within the time budget")
    if failed:
        notes.append(f"{failed} hunks had no usable model output")
    if duplicates:
        notes.append(f"{duplicates} comments suppressed as duplicates of earlier ones")
    if notes:
        lines.append("\n" + "\n".join(f"- {n}" for n in notes))
    if low_confidence:
        lines.append(f"\n<details><summary>{held_back_title} "
                     f"({len(low_confidence)})</summary>\n")
        for f in low_confidence:
            lines.append(f"- `{f.path}:{f.line}` {f.severity} · {f.issue_type} — "
                         f"{f.explanation.strip()}")
        lines.append("\n</details>")
    lines.append("\n<sub>Severity and issue type are hints, not merge gates.</sub>")
    return "\n".join(lines)
