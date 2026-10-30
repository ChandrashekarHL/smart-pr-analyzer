"""Review one pull request at one head commit (design doc 3.2, steps 2-7).

    outcome = review_pull_request(gh, model, policy, repo, number, head_sha)

1. Fetch the PR; stop if its head has moved on (a newer event will review it).
2. Pick hunks (`diff.reviewable_hunks`): supported languages, largest first,
   at most `policy.max_hunks`.
3. Per hunk, optionally retrieve context, call the model API, keep the parsed
   finding. Stop calling the model when the time budget would be exceeded.
4. Anchor each finding's line inside its hunk, drop duplicates of comments
   already on the PR, hold back low-confidence ones for the summary.
5. Post one review (one notification) with the inline comments and a summary.
   If GitHub rejects an inline comment, post the summary with every finding
   in it rather than lose the review.

Everything external is injected, so tests run with fakes and no network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from prana_app.diff import Hunk, reviewable_hunks
from prana_app.github import GitHubError
from prana_app.review import Finding, finding_from, is_duplicate, summary

log = logging.getLogger("prana_app.pipeline")

ModelFn = Callable[[str, str, str | None, str | None], dict]   # (hunk, lang, ctx, path) -> result
RetrieveFn = Callable[[Hunk, str, int, str], str]              # (hunk, repo, pr, base_sha) -> ctx


class ModelUnavailable(RuntimeError):
    """Every model call failed: worth retrying the whole job later."""


@dataclass(frozen=True)
class Policy:
    max_hunks: int = 8
    time_budget_s: float = 25.0
    min_confidence: float = 0.0
    bot_login: str = "prana-reviewer[bot]"


@dataclass
class Outcome:
    status: str                       # posted | no_hunks | superseded | posted_summary_only
    hunks_total: int = 0
    hunks_reviewed: int = 0
    hunks_failed: int = 0
    hunks_skipped_time: int = 0
    duplicates: int = 0
    posted: list[Finding] = field(default_factory=list)
    low_confidence: list[Finding] = field(default_factory=list)
    review_id: int | None = None
    seconds: float = 0.0


def review_pull_request(gh, model: ModelFn, policy: Policy, repo: str, number: int,
                        head_sha: str, retrieve: RetrieveFn | None = None,
                        clock: Callable[[], float] = time.monotonic) -> Outcome:
    started = clock()
    pr = gh.pull_request(repo, number)
    if pr.get("head", {}).get("sha") != head_sha:
        return Outcome("superseded", seconds=clock() - started)
    base_sha = pr.get("base", {}).get("sha", "")

    hunks, total = reviewable_hunks(gh.pull_files(repo, number), policy.max_hunks)
    out = Outcome("no_hunks", hunks_total=total)
    if not hunks:
        out.seconds = clock() - started
        return out

    findings: list[Finding] = []
    per_hunk = 0.0
    for i, h in enumerate(hunks):
        elapsed = clock() - started
        if i and elapsed + per_hunk > policy.time_budget_s:
            out.hunks_skipped_time = len(hunks) - i
            break
        t = clock()
        ctx = None
        if retrieve is not None:
            try:
                ctx = retrieve(h, repo, number, base_sha) or None
            except Exception as e:  # noqa: BLE001 - retrieval never blocks a review
                log.warning("retrieve failed for %s: %s", h.path, e)
        try:
            result = model(h.text, h.language, ctx, h.path)
        except Exception as e:  # noqa: BLE001 - one bad hunk must not sink the PR
            log.warning("model failed for %s: %s", h.path, e)
            out.hunks_failed += 1
            continue
        finally:
            per_hunk = max(per_hunk, clock() - t)
        out.hunks_reviewed += 1
        f = finding_from(h, result)
        if f is None:
            out.hunks_failed += 1
        else:
            findings.append(f)

    if out.hunks_reviewed == 0 and out.hunks_failed:
        raise ModelUnavailable(f"all {out.hunks_failed} model calls failed")

    existing = [c for c in gh.review_comments(repo, number)]
    for f in findings:
        if is_duplicate(f, existing):
            out.duplicates += 1
        elif f.confidence < policy.min_confidence:
            out.low_confidence.append(f)
        else:
            out.posted.append(f)
    out.posted.sort(key=lambda f: (f.path, f.line))

    body = summary(out.posted, out.low_confidence, out.hunks_reviewed, total,
                   out.hunks_failed, out.hunks_skipped_time, out.duplicates)
    try:
        res = gh.create_review(repo, number, head_sha, body,
                               [f.as_github_comment() for f in out.posted])
        out.status = "posted"
    except GitHubError as e:
        if e.status != 422:
            raise
        # An inline target GitHub refuses (e.g. the head moved under us): keep
        # the review, in the summary.
        log.warning("inline comments rejected (%s); posting summary only", e)
        everything = summary([], out.posted + out.low_confidence, out.hunks_reviewed, total,
                             out.hunks_failed, out.hunks_skipped_time, out.duplicates,
                             held_back_title="Findings that could not be placed inline")
        res = gh.create_review(repo, number, head_sha, everything, [])
        out.status = "posted_summary_only"
    out.review_id = res.get("id")
    out.seconds = clock() - started
    return out
