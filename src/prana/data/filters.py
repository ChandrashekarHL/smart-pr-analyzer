"""Row-level filter predicates.

Separated from the pipeline so each rule can be tested on its own and so the
reason a row was dropped is always a named constant rather than an inline
regex buried in a loop.
"""

from __future__ import annotations

import re

from prana.config import EXT_LANG

# Bump whenever a predicate's behaviour changes. DataConfig.fingerprint() only
# covers thresholds, so without this two different corpora - before and after
# the author-reply filter, say - would carry the same identifier.
#   1  bot authorship, markup, conversational, suggestion split, language
#   2  author-reply removal
#   3  machine-translated review wrapper (cherry-studio template)
#   4  AI-tool markup under personal accounts (priority badges, metadata headers)
FILTERS_VERSION = 4

# --------------------------------------------------------------- authorship

# 19% of the raw scrape was written by code-review bots. This is the single
# most consequential filter in the project: the stated novelty is training on
# real human review, and Copilot and chatgpt-codex-connector between them
# contribute ~15,000 rows of fluent prose that no text heuristic separates
# from a human's. Authorship is the only reliable signal, so it is checked
# first and checked on the `reviewer` field, never on the comment body.
BOT_REVIEWERS: frozenset[str] = frozenset({
    "copilot",
    "coderabbitai[bot]",
    "chatgpt-codex-connector[bot]",
    "gemini-code-assist[bot]",
    "greptile-apps[bot]",
    "sourcery-ai[bot]",
    "deepsource-autofix[bot]",
    "sonarcloud[bot]",
    "codecov[bot]",
    "dependabot[bot]",
    "renovate[bot]",
    "cursor[bot]",
    "github-actions[bot]",
    # Anonymised in the scrape, but 3,771 comments across 82 unrelated
    # repositories - django, celery, react, vscode, ComfyUI, Rust crates.
    # No person reviews that spread. Uniform phrasing, exact file:line
    # citations, 53-word mean against 81 for named humans.
    "49f5b0e4",
})


def is_bot_reviewer(name: str | None, extra: frozenset[str] = frozenset()) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    return (
        n.endswith("[bot]")
        or n.endswith("-bot")
        or n.endswith("_bot")
        or n in BOT_REVIEWERS
        or n in extra
    )


# Structural markers only - badge images, HTML metadata, CodeRabbit's severity
# table. Bare mentions of "copilot" or "sonarcloud" are deliberately absent: a
# human arguing with a scanner's finding is a review worth keeping, and an
# earlier pass that matched those names discarded ~4,000 of them.
BOT_MARKUP = re.compile(
    r"(coderabbit|greptile-static-assets|codereviewagent|cubic:v=|"
    r"<!--\s*DESCRIPTION START\s*-->|gemini-code-assist\[bot\]|"
    # Machine-translated review wrapper. 277 rows of corpus v2, all from
    # CherryHQ/cherry-studio: a human wrote Chinese, a bot posted the English
    # with the original folded under <details>. The v0.3.1 model learned to
    # emit the wrapper verbatim ("> This review comment was translated
    # automatically"). The English is machine output, so it goes.
    r"This review comment was translated automatically|"
    r"<summary>\s*Original Content\s*</summary>|"
    # AI review tools posting under personal accounts. Found 2026-10-19 while
    # hand-checking teacher labels: `magyargergo` (230 rows) opens every
    # comment with a `**[P2 · test]**` priority badge, `b95ddda8` (215 rows)
    # with cubic's `<!-- metadata:{"confidence":8,...} -->` header. Both are
    # findings, so the teacher's is_review keeps them; only the markup tells.
    r"<!--\s*metadata:|"
    r"^\s*\*\*\[P[0-4]\b)",
    re.I,
)

# --------------------------------------------------------------- content

# Thread replies, not reviews: "Done.", "Updated. Thank you!", "Same as above".
# 9% of the corpus. Training on these teaches the model to answer a review
# rather than write one.
CONVERSATIONAL = re.compile(
    r"^\s*(done|thanks|thank you|fixed|good catch|same as above|as above|agreed|"
    r"ok|okay|sure|\+1|lgtm|looks good|nice|yes|no|oops|ack|sgtm|will do|"
    r"makes sense|removed|added|updated|reverted|resolved|nit)\b[\s.!,:;)-]*",
    re.I,
)

# The PR author answering a review, rather than a reviewer reviewing:
# "Addressed in 5fa47ff: setup now computes...", "Real find - fixed in
# f5de37aef7", "You're right that a single boolean can't...". About 9.6% of
# the training targets after every other filter. Left in, they teach the model
# to reply to review instead of performing it.
#
# The scrape carries no pr_author field, so authorship cannot be compared
# directly; these patterns are the next best signal. The first-person rule is
# deliberately limited to reply verbs: a bare "\w+ed" also matches
# "I tried this locally and it crashes", which is a genuine review.
AUTHOR_REPLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(addressed|fixed|done|resolved|updated|changed|implemented|reverted|"
        r"removed|added|applied|switched|moved|renamed|good point|good catch|"
        r"thanks|thank you|agreed|you'?re right|you are right|real find)\b",
        re.I,
    ),
    re.compile(r"\b(?:in|commit|via|see|at)\s+`?[0-9a-f]{7,40}`?(?![0-9a-z])", re.I),
    re.compile(
        r"^\s*(?:i|we)(?:\s+have|'ve)?\s+(?:added|fixed|changed|updated|removed|"
        r"pushed|reverted|addressed|implemented|moved|renamed|switched|rebased)\b",
        re.I,
    ),
)


def is_author_reply(body: str) -> bool:
    return any(p.search(body) for p in AUTHOR_REPLY_PATTERNS)


SUGGESTION_BLOCK = re.compile(r"```suggestion\s*\n(.*?)```", re.S)


def split_suggestion(comment: str) -> tuple[str, str | None]:
    """Separate prose from GitHub's ```suggestion block.

    10.6% of comments carry one. It is literal human-written ground truth for
    the `suggested_fix` field and beats anything a teacher model will invent
    for those rows, so it is extracted rather than discarded with the markup.
    """
    match = SUGGESTION_BLOCK.search(comment)
    if not match:
        return comment.strip(), None
    return SUGGESTION_BLOCK.sub("", comment).strip(), match.group(1).rstrip("\n")


def detect_language(file_path: str | None, declared: str | None) -> str | None:
    """The scraper left `programming_language` empty on 100,196 of 188,052
    rows but kept `file_path` on all of them, so the extension is the more
    reliable source and the declared value is only a fallback shortcut."""
    if declared:
        return declared.strip().lower()
    if not file_path:
        return None
    tail = file_path.rsplit("/", 1)[-1]
    if "." not in tail:
        return None
    return EXT_LANG.get(tail.rsplit(".", 1)[-1].lower())


def normalise_for_dedup(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())[:200]


def is_bare_question(body: str, max_words: int) -> bool:
    """A short question cannot be answered from a diff alone, so it is not a
    learnable review target - it teaches the model to ask instead of find."""
    return body.rstrip().endswith("?") and len(body.split()) < max_words
