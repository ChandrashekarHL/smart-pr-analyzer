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
_RULE_RE = re.compile(r"Semgrep rule <code>([^<]+)</code>")
_HEADER_RE = re.compile(r"^\*\*\S+ (?:Critical|Warning|Info)\*\* · ")
DUPLICATE_SIMILARITY = 0.9
DUPLICATE_LINE_WINDOW = 3


def category(severity: str, issue_type: str) -> str:
    """How a reader should treat the comment (the label reviewers skim for)."""
    if issue_type == "TEST":
        return "🧪 Testing"
    if issue_type in ("BUG", "SECURITY", "PERF") and severity != "info":
        return "⚠️ Potential issue"
    if severity == "info" and issue_type in ("STYLE", "DOCS"):
        return "🧹 Nitpick"
    return "🛠️ Refactor suggestion"


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
    rule: str = ""              # set for Semgrep findings: the rule id
    cwe: tuple[str, ...] = ()   # Semgrep findings: rule metadata
    owasp: tuple[str, ...] = ()
    url: str = ""

    @property
    def fingerprint(self) -> str:
        key = f"{self.path}\n{self.line}\n{_norm(self.explanation)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _why(self) -> list[str]:
        if self.rule:
            # A rule match, not a model guess: say which rule, not a confidence.
            link = f" ([rule]({self.url}))" if self.url else ""
            return ([f"Semgrep rule <code>{self.rule}</code>{link}: a rule match, "
                     "not a model guess"] + list(self.cwe) + [f"OWASP {o}" for o in self.owasp])
        why = [f"prana review model, confidence {self.confidence:.2f}"]
        if not self.line_exact:
            why.append("line approximate: the model pointed outside the change, so the comment "
                       "sits on the nearest changed line")
        return why

    def body(self) -> str:
        icon = SEVERITY_ICON.get(self.severity, "")
        out = [f"**{icon} {self.severity.capitalize()}** · "
               f"{category(self.severity, self.issue_type)} · `{self.issue_type}`",
               "", self.explanation.strip()]
        if self.suggestion.strip():
            out += ["", f"**💡 Suggestion:** {self.suggestion.strip()}"]
        out += ["", "<details>", "<summary>Why this was flagged</summary>", ""]
        out += [f"- {w}" for w in self._why()]
        out += ["", "</details>", "", MARKER.format(fp=self.fingerprint)]
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


def _explanation_of(body: str) -> str:
    """The explanation inside a comment `Finding.body` rendered: the paragraph
    after the header line. Comments in the pre-2026-11-11 format had it after
    the badge on the first line (`🔴 **critical · BUG** — ...`). Any other
    comment (a human's) is returned whole."""
    text = _MARKER_RE.sub("", body or "").strip()
    if _HEADER_RE.match(text):
        parts = text.split("\n\n")
        return parts[1] if len(parts) > 1 else ""
    text = re.split(r"\n\*\*Suggestion:\*\*|\n<sub>", text, maxsplit=1)[0]
    _, sep, rest = text.partition(" — ")
    return rest if sep else text


def is_duplicate(f: Finding, existing: list[dict]) -> bool:
    """`existing`: the PR's review comments ({path, line, body}). Same marker;
    or same file and nearby line with the same Semgrep rule or near-identical
    explanation. The fingerprint includes the line, so a later commit that
    shifts the code (an added import) needs the nearby-line checks."""
    for c in existing:
        body = c.get("body") or ""
        m = _MARKER_RE.search(body)
        if m and m.group(1) == f.fingerprint:
            return True
        if c.get("path") != f.path:
            continue
        line = c.get("line") or c.get("original_line")
        if isinstance(line, int) and abs(line - f.line) > DUPLICATE_LINE_WINDOW:
            continue
        if f.rule:
            r = _RULE_RE.search(body)
            if r and r.group(1) == f.rule:
                return True
        ratio = difflib.SequenceMatcher(None, _norm(_explanation_of(body)),
                                        _norm(f.explanation)).ratio()
        if ratio >= DUPLICATE_SIMILARITY:
            return True
    return False


def _file_table(posted: list[Finding]) -> list[str]:
    per_file: dict[str, dict[str, int]] = {}
    for f in posted:
        counts = per_file.setdefault(f.path, dict.fromkeys(SEVERITY_ORDER, 0))
        counts[f.severity] = counts.get(f.severity, 0) + 1
    rows = ["", "| File | " + " | ".join(SEVERITY_ICON[s] for s in SEVERITY_ORDER) + " |",
            "|---|" + "---|" * len(SEVERITY_ORDER)]
    for path, n in sorted(per_file.items()):
        rows.append(f"| `{path}` | " + " | ".join(str(n[s] or "") for s in SEVERITY_ORDER) + " |")
    return rows


def summary(posted: list[Finding], low_confidence: list[Finding], reviewed: int,
            total: int, failed: int, skipped_for_time: int, duplicates: int,
            held_back_title: str = "Low-confidence suggestions", head_sha: str = "",
            adapter: str = "") -> str:
    counts = {s: sum(f.severity == s for f in posted) for s in SEVERITY_ORDER}
    lines = ["### prana review", "",
             f"**Comments posted: {len(posted)}**" if posted else "**No inline comments.**",
             f"Reviewed **{reviewed}** of {total} changed hunks in supported languages."]
    if posted:
        lines.append(" · ".join(f"{SEVERITY_ICON[s]} {n} {s}" for s, n in counts.items() if n))
        n_rules = sum(1 for f in posted if f.rule)
        if n_rules:
            lines.append(f"{n_rules} of these come from Semgrep security rules.")
        lines += _file_table(posted)
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
    details = []
    if head_sha:
        details.append(f"Commit `{head_sha[:7]}`")
    if adapter:
        details.append(f"Review model: DeepSeek-Coder 6.7B + LoRA `{adapter}`")
    if any(f.rule for f in posted + low_confidence):
        details.append("Security rules: Semgrep")
    if details:
        lines += ["\n<details><summary>Review details</summary>\n"] + \
                 [f"- {d}" for d in details] + ["\n</details>"]
    lines.append("\n<sub>Severity and issue type are hints, not merge gates.</sub>")
    return "\n".join(lines)
