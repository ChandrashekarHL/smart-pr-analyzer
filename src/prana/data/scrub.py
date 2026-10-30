"""Credential and PII redaction.

A scan of the filtered corpus finds roughly 490 rows carrying something that
looks like a credential. Nearly all are test fixtures - `ghp_1234567890...`,
`xoxb-test-token`, `my-auth-token-here` - but "nearly all" is not a standard
you can publish against, and a single live key in a public dataset is an
incident that outlives the project.

We redact rather than drop. The review comment is still a valid training
target when the secret beside it becomes `<REDACTED:github_pat>`; throwing the
row away would cost signal for no additional safety.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered: the specific high-confidence patterns run before the generic ones,
# so a GitHub token is labelled as such rather than caught by `generic_secret`.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        r"[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
    )),
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{25,}={0,2}")),
    ("generic_secret", re.compile(
        r"""(?ix)
        \b(api[_-]?key|secret|passwd|password|access[_-]?token|auth[_-]?token)
        \s*[:=]\s*
        (['"])(?P<val>[A-Za-z0-9_\-/+.]{16,})\2
        """
    )),
]

# A bare email regex fires on `@pytest.mark.asyncio` and Go module paths like
# `toolchain@v0.0.1-go1.26.2.darwin`, which are not addresses. Requiring a
# plausible TLD and rejecting a leading `@`-decorator cuts the false positives
# that made an earlier pass report 379 "emails", nearly all spurious.
EMAIL = re.compile(
    r"(?<![\w.@+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+"
    r"(?:com|org|net|io|dev|edu|gov|co|uk|de|fr|in|cn|jp|ru|br|au|nl|se|es|it)\b"
)

# Committers' addresses appear legitimately in diffs of AUTHORS, mailmap and
# changelog files. Those are public by construction, so we keep them unless
# the caller asks otherwise.
NOREPLY = re.compile(r"(?i)@(?:users\.noreply\.github\.com|example\.(?:com|org)|localhost)")


@dataclass
class ScrubResult:
    text: str
    hits: dict[str, int]

    @property
    def changed(self) -> bool:
        return bool(self.hits)


def scrub(text: str, redact_emails: bool = False) -> ScrubResult:
    """Replace credential-shaped substrings with typed placeholders."""
    hits: dict[str, int] = {}
    out = text

    for name, pattern in PATTERNS:
        if name == "generic_secret":
            def _repl(m: re.Match[str], _n: str = name) -> str:
                hits[_n] = hits.get(_n, 0) + 1
                return f"{m.group(1)}={m.group(2)}<REDACTED:{_n}>{m.group(2)}"
            out, n = pattern.subn(_repl, out)
        else:
            out, n = pattern.subn(f"<REDACTED:{name}>", out)
            if n:
                hits[name] = hits.get(name, 0) + n

    if redact_emails:
        def _email(m: re.Match[str]) -> str:
            if NOREPLY.search(m.group(0)):
                return m.group(0)
            hits["email"] = hits.get("email", 0) + 1
            return "<REDACTED:email>"
        out = EMAIL.sub(_email, out)

    return ScrubResult(out, hits)


def scan(text: str) -> dict[str, int]:
    """Count credential hits without modifying the text."""
    found: dict[str, int] = {}
    for name, pattern in PATTERNS:
        n = len(pattern.findall(text))
        if n:
            found[name] = n
    return found
