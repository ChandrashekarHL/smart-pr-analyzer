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
6. Optionally write or update the PR walkthrough comment (`walkthrough.py`);
   it runs after the review is on GitHub and never fails the job.

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
from prana_app.walkthrough import GenerateFn, post_walkthrough

log = logging.getLogger("prana_app.pipeline")

ModelFn = Callable[[str, str, str | None, str | None], dict]   # (hunk, lang, ctx, path) -> result
RetrieveFn = Callable[[Hunk, str, int, str], str]              # (hunk, repo, pr, base_sha) -> ctx
ScanFn = Callable[[dict[str, str]], dict[str, list]]           # {path: text} -> {path: findings}

MAX_SCAN_FILES = 10
SCAN_GRACE_S = 15.0       # extra wait for Semgrep after the model calls, if still running


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
    status: str     # posted | posted_summary_only | no_hunks | superseded | already_posted
    hunks_total: int = 0
    hunks_reviewed: int = 0
    hunks_failed: int = 0
    hunks_skipped_time: int = 0
    duplicates: int = 0
    rule_findings: int = 0          # Semgrep findings on added lines (before dedup)
    posted: list[Finding] = field(default_factory=list)
    low_confidence: list[Finding] = field(default_factory=list)
    review_id: int | None = None
    seconds: float = 0.0
    walkthrough: str = ""           # created | updated | unusable | failed | "" (off)


def _start_scan(gh, scan: ScanFn, repo: str, head_sha: str, files: list[dict]):
    """Fetch the changed source files at the head commit and start the rule
    scan in a background thread. Returns (thread, result holder)."""
    import threading

    from prana_app.diff import language_for
    paths = [f["filename"] for f in files
             if f.get("status") != "removed" and language_for(f.get("filename", ""))]
    holder: dict = {}

    def run():
        try:
            texts = {}
            for p in paths[:MAX_SCAN_FILES]:
                text = gh.file_content(repo, p, head_sha)
                if text is not None:
                    texts[p] = text
            holder["result"] = scan(texts) if texts else {}
        except Exception as e:  # noqa: BLE001 - rules are an extra; never fail the review
            log.warning("rule scan failed for %s: %s", repo, e)
            holder["result"] = {}
    t = threading.Thread(target=run, name="prana-scan", daemon=True)
    t.start()
    return t, holder


def _rule_findings(scanned: dict[str, list], files: list[dict]) -> list[Finding]:
    """Rule hits on lines this PR added - pre-existing code is not this PR's to fix."""
    from prana_app.diff import language_for, parse_patch
    added: dict[str, set[int]] = {}
    for f in files:
        lang = language_for(f.get("filename", ""))
        if lang and f.get("patch"):
            added[f["filename"]] = {n for h in parse_patch(f["filename"], f["patch"], lang)
                                    for n in h.added}
    out = []
    for path, hits in scanned.items():
        for h in hits:
            if h.line not in added.get(path, set()):
                continue
            out.append(Finding(
                path=path, line=h.line, line_exact=True,
                severity="critical" if h.severity in ("error", "critical") else "warning",
                issue_type="SECURITY", explanation=h.message, suggestion="",
                confidence=1.0, rule=h.rule, cwe=getattr(h, "cwe", ()),
                owasp=getattr(h, "owasp", ()), url=getattr(h, "url", "")))
    return out


def review_pull_request(gh, model: ModelFn, policy: Policy, repo: str, number: int,
                        head_sha: str, retrieve: RetrieveFn | None = None,
                        clock: Callable[[], float] = time.monotonic,
                        scan: ScanFn | None = None,
                        walkthrough: GenerateFn | None = None) -> Outcome:
    started = clock()
    pr = gh.pull_request(repo, number)
    if pr.get("head", {}).get("sha") != head_sha:
        return Outcome("superseded", seconds=clock() - started)
    # Idempotent at the source of truth: if our bot already reviewed this exact
    # commit (a crash after posting, a redelivered job), never post it again.
    for r in gh.reviews(repo, number):
        if r.get("commit_id") == head_sha and (r.get("user") or {}).get("login") == \
                policy.bot_login:
            return Outcome("already_posted", review_id=r.get("id"), seconds=clock() - started)
    base_sha = pr.get("base", {}).get("sha", "")

    files = gh.pull_files(repo, number)
    hunks, total = reviewable_hunks(files, policy.max_hunks)
    out = Outcome("no_hunks", hunks_total=total)
    if not hunks:
        out.seconds = clock() - started
        return out
    scan_thread = _start_scan(gh, scan, repo, head_sha, files) if scan else None

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

    if scan_thread is not None:
        thread, holder = scan_thread
        thread.join(timeout=SCAN_GRACE_S)
        rules = _rule_findings(holder.get("result") or {}, files)
        out.rule_findings = len(rules)
        findings.extend(rules)

    existing = [c for c in gh.review_comments(repo, number)]
    for f in findings:
        if is_duplicate(f, existing):
            out.duplicates += 1
        elif f.confidence < policy.min_confidence:
            out.low_confidence.append(f)
        else:
            out.posted.append(f)
    out.posted.sort(key=lambda f: (f.path, f.line))

    adapter = next((f.adapter for f in findings if f.adapter), "")
    body = summary(out.posted, out.low_confidence, out.hunks_reviewed, total,
                   out.hunks_failed, out.hunks_skipped_time, out.duplicates,
                   head_sha=head_sha, adapter=adapter)
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
                             held_back_title="Findings that could not be placed inline",
                             head_sha=head_sha, adapter=adapter)
        res = gh.create_review(repo, number, head_sha, everything, [])
        out.status = "posted_summary_only"
    out.review_id = res.get("id")
    if walkthrough is not None:
        out.walkthrough = post_walkthrough(gh, walkthrough, repo, number, head_sha,
                                           pr.get("title") or "", files, policy.bot_login)
    out.seconds = clock() - started
    return out
