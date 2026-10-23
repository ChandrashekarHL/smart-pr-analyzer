"""The prompt contract shared by training, evaluation and serving.

This module is the single place prompts are defined. Training renders a
version, writes its fingerprint into the adapter, and serving renders the
version whose fingerprint the adapter carries. Drift between the two produces
no error and no failing test - only a quietly worse model - so the check is
loud: `assert_contract` raises if no known version matches.

Versions are kept, never edited. v1 is what v0.3.1 and v1.0 were trained on;
its text must stay byte-identical or those adapters refuse to load.

    v1  bare diff hunk, context slot, prose or JSON response (whatever the
        training target was).
    v2  the hunk with new-file line numbers in front of each line, so
        `line_number` is answerable; the response is stated to be one JSON
        object. v1.0 predicted null for line_number on 96% of rows because a
        bare hunk gives it nothing to point at.

Do not inline a copy of a template anywhere. Import it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

NO_CONTEXT = "(none)"

# Marks where the assistant turn begins; the user message ends with it.
# Training masks the loss to the response - without that, most of the
# gradient teaches the model to generate diffs, which it is never asked to do.
RESPONSE_MARKER = "Review this hunk."


@dataclass(frozen=True)
class PromptVersion:
    name: str
    system: str
    user_template: str
    numbered_lines: bool

    def fingerprint(self) -> str:
        blob = "\x1f".join([self.name, self.system, self.user_template, NO_CONTEXT])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------- v1 (frozen)

_V1 = PromptVersion(
    name="v1",
    system=(
        "You are a senior software engineer reviewing a pull request. "
        "You are given one hunk of a unified diff, the language it is written in, "
        "and optionally some context retrieved from the repository. "
        "Report only issues you can justify from the code shown. "
        "If the hunk is correct, say so briefly rather than inventing a concern."
    ),
    user_template=(
        "Language: {language}\n\n"
        "Repository context:\n{context}\n\n"
        "Diff hunk:\n```{language}\n{diff}\n```\n\n"
        "Review this hunk."
    ),
    numbered_lines=False,
)

# ----------------------------------------------------------------- v2

_V2 = PromptVersion(
    name="v2",
    system=(
        "You are a senior software engineer reviewing a pull request. "
        "You are given one hunk of a unified diff, the language it is written in, "
        "and optionally some context retrieved from the repository. "
        "Each line of the hunk is prefixed with its line number in the new version "
        "of the file; removed lines have no number. "
        "Report only issues you can justify from the code shown, and respond with "
        "exactly one JSON object: explanation, suggestion, severity, issue_type, "
        "file_path, line_number. Set line_number to the numbered line your finding "
        "is about, or null if it concerns the hunk as a whole. "
        "If the hunk is correct, say so briefly rather than inventing a concern."
    ),
    user_template=(
        "Language: {language}\n\n"
        "Repository context:\n{context}\n\n"
        "Diff hunk (new-file line numbers on the left):\n```{language}\n{diff}\n```\n\n"
        "Review this hunk."
    ),
    numbered_lines=True,
)

VERSIONS: dict[str, PromptVersion] = {_V1.name: _V1, _V2.name: _V2}
CURRENT: PromptVersion = _V2

# Names kept from the v1-only module so existing imports keep working.
PROMPT_VERSION = CURRENT.name
SYSTEM = CURRENT.system
PROMPT_FINGERPRINT = CURRENT.fingerprint()

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# A hunk over this is cut in the MIDDLE so the template's tail ("Review this
# hunk.") always survives: with right-side tokenizer truncation it did not,
# and the model - never shown the end of the prompt - simply continued the
# diff. That was 35 of v1.1's 41 invalid outputs, measured with the real
# tokenizer. Code tokenises at roughly 2 characters per token (a 6,016-char hex
# table came to 3,036 tokens), so the first cap of 6,000 still let prompts
# through at 2,300-3,000 tokens. 3,600 chars is ~1,800 tokens at that rate;
# callers that have the tokenizer also enforce an exact budget
# (`prana.eval.generate.render_within_budget`).
MAX_HUNK_CHARS = 3600
_OMITTED = "... [{n} lines omitted] ..."


def fit_hunk(diff: str, max_chars: int = MAX_HUNK_CHARS) -> str:
    """Keep the head and the tail of an oversized hunk, drop the middle."""
    if len(diff) <= max_chars:
        return diff
    lines = diff.splitlines()
    budget = max_chars - len(_OMITTED.format(n=len(lines))) - 2
    head: list[str] = []
    tail: list[str] = []
    size = 0
    i, j = 0, len(lines) - 1
    while i <= j:
        take_head = len(head) <= len(tail)
        line = lines[i] if take_head else lines[j]
        if size + len(line) + 1 > budget:
            break
        size += len(line) + 1
        if take_head:
            head.append(line)
            i += 1
        else:
            tail.insert(0, line)
            j -= 1
    omitted = j - i + 1
    if omitted <= 0:
        return diff
    return chr(10).join([*head, _OMITTED.format(n=omitted), *tail])


def number_hunk(diff: str) -> str:
    """Prefix each hunk line with its new-file line number.

    Context and added lines advance the counter; removed lines (and the
    `\\ No newline at end of file` marker) get a blank prefix because they do
    not exist in the new file. Text before the first `@@` header is left as
    is. A diff with no header is returned unchanged.
    """
    out: list[str] = []
    new_line: int | None = None
    for raw in diff.splitlines():
        header = _HUNK_HEADER.match(raw)
        if header:
            new_line = int(header.group(3))
            out.append(raw)
            continue
        if new_line is None:
            out.append(raw)
            continue
        if raw.startswith("-") or raw.startswith("\\"):
            out.append(f"{'':>5} {raw}")
            continue
        out.append(f"{new_line:>5} {raw}")
        new_line += 1
    return "\n".join(out)


def _resolve(version: str | None) -> PromptVersion:
    if version is None:
        return CURRENT
    try:
        return VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"unknown prompt version {version!r}; known: {sorted(VERSIONS)}"
        ) from None


def build_user_message(diff: str, language: str, context: str | None = None,
                       version: str | None = None,
                       max_hunk_chars: int = MAX_HUNK_CHARS) -> str:
    pv = _resolve(version)
    body = diff.rstrip()
    if pv.numbered_lines:
        body = number_hunk(body)
    # Cut AFTER numbering so the cap covers the prefixes too: a hunk of many
    # short lines gains ~6 chars per line from the numbers.
    body = fit_hunk(body, max_hunk_chars)
    return pv.user_template.format(
        language=language,
        context=(context.strip() if context and context.strip() else NO_CONTEXT),
        diff=body,
    )


def build_messages(
    diff: str,
    language: str,
    context: str | None = None,
    review: str | None = None,
    version: str | None = None,
    max_hunk_chars: int = MAX_HUNK_CHARS,
) -> list[dict[str, str]]:
    """Chat messages for `tokenizer.apply_chat_template`.

    The tokenizer owns its chat format; hard-coding special-token markers here
    breaks the moment the base model changes. Pass `review` to build a training
    example, omit it to build a prompt for generation.
    """
    pv = _resolve(version)
    messages = [
        {"role": "system", "content": pv.system},
        {"role": "user", "content": build_user_message(diff, language, context, pv.name,
                                                       max_hunk_chars)},
    ]
    if review is not None:
        messages.append({"role": "assistant", "content": review.strip()})
    return messages


def fingerprint(version: str | None = None) -> str:
    return _resolve(version).fingerprint()


def resolve_version(expected_fingerprint: str) -> str:
    """The version name whose rendered text has this fingerprint."""
    for pv in VERSIONS.values():
        if pv.fingerprint() == expected_fingerprint:
            return pv.name
    raise RuntimeError(
        f"prompt contract mismatch: no prompt version in this build renders "
        f"{expected_fingerprint}; known are "
        + ", ".join(f"{pv.name}={pv.fingerprint()}" for pv in VERSIONS.values())
        + ". Re-export the adapter or pin prana to the matching revision before serving."
    )


def assert_contract(expected_fingerprint: str) -> str:
    """Fail loudly when nothing in this build matches the adapter's prompt.
    Returns the matching version name so the caller renders with it."""
    return resolve_version(expected_fingerprint)
