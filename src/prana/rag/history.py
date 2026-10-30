"""Repository history for one file: past review comments and recent commits
(design doc, Module 3.2: "last 3 review comments on this file from the
repository's PR history" and "last 10 git commits touching this file").

`retrieve` stays a pure function of what it is given, so the data comes in as
a `FileHistory` the caller fetched:

- **Backend:** commits from `GET /repos/{o}/{r}/commits?path=<file>&sha=<base_sha>`;
  past reviews from its own store of review comments (webhook
  `pull_request_review_comment`, backfilled once from
  `GET /repos/{o}/{r}/pulls/comments`), with bot accounts removed exactly as
  the training data was.
- **Evaluation:** commits from the same API through `gh` (`prana.rag.shas`),
  past reviews from the scrape, only from other PRs and strictly earlier
  than the PR under review (`ReviewStore`).

Leakage rules, enforced here as well as by the caller: a review from the PR
under review is never used, and nothing dated at or after `before` is used.
"""

from __future__ import annotations

import bisect
import collections
from dataclasses import dataclass, field

MAX_REVIEWS = 3
MAX_COMMITS = 10
REVIEW_MAX_CHARS = 280
COMMIT_MAX_CHARS = 100


@dataclass(frozen=True)
class PastReview:
    pr_number: str
    created_at: str          # ISO 8601, UTC
    body: str


@dataclass(frozen=True)
class Commit:
    sha: str
    date: str                # ISO 8601
    message: str


@dataclass
class FileHistory:
    reviews: list[PastReview] = field(default_factory=list)
    commits: list[Commit] = field(default_factory=list)


def _one_line(text: str, limit: int) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def review_items(history: FileHistory | None, pr_number, before: str | None = None,
                 limit: int = MAX_REVIEWS) -> list[str]:
    """Most recent past reviews on the file, newest first, never from
    `pr_number` and never at or after `before`."""
    if not history or not history.reviews:
        return []
    pr = str(pr_number) if pr_number is not None else None
    keep = [r for r in history.reviews
            if (pr is None or str(r.pr_number) != pr)
            and (before is None or (r.created_at or "") < before)
            and r.body and r.body.strip()]
    keep.sort(key=lambda r: r.created_at or "", reverse=True)
    return [f"- PR #{r.pr_number} ({(r.created_at or '')[:10]}): "
            f"{_one_line(r.body, REVIEW_MAX_CHARS)}" for r in keep[:limit]]


def commit_items(history: FileHistory | None, limit: int = MAX_COMMITS) -> list[str]:
    """Recent commits touching the file, newest first, first line of each message."""
    if not history or not history.commits:
        return []
    keep = sorted(history.commits, key=lambda c: c.date or "", reverse=True)[:limit]
    return [f"- {(c.date or '')[:10]} {_one_line((c.message or '').splitlines()[0] if c.message else '', COMMIT_MAX_CHARS)}"
            for c in keep if c.message and c.message.strip()]


class ReviewStore:
    """Past review comments per (repo, file), for evaluation from the scrape.
    `history_for` returns only reviews from other PRs that predate the PR."""

    def __init__(self) -> None:
        self._by_file: dict[tuple[str, str], list[tuple[str, PastReview]]] = \
            collections.defaultdict(list)
        self._pr_start: dict[tuple[str, str], str] = {}

    def add(self, repo: str, file_path: str, pr_number, created_at: str, body: str) -> None:
        if not (repo and file_path and created_at and body):
            return
        pr = str(pr_number)
        self._by_file[(repo, file_path)].append(
            (created_at, PastReview(pr, created_at, body)))
        key = (repo, pr)
        if key not in self._pr_start or created_at < self._pr_start[key]:
            self._pr_start[key] = created_at

    def finalize(self) -> ReviewStore:
        for v in self._by_file.values():
            v.sort(key=lambda t: t[0])
        return self

    def pr_start(self, repo: str, pr_number) -> str | None:
        """Earliest review comment we have on the PR: the cut-off for 'past'."""
        return self._pr_start.get((repo, str(pr_number)))

    def reviews_for(self, repo: str, file_path: str, pr_number) -> list[PastReview]:
        cut = self.pr_start(repo, pr_number)
        rows = self._by_file.get((repo, file_path), [])
        if cut is None:
            return []
        hi = bisect.bisect_left([t[0] for t in rows], cut)
        pr = str(pr_number)
        return [r for _, r in rows[:hi] if r.pr_number != pr]
