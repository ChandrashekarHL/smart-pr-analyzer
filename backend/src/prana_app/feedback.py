"""User feedback (FR-10): 👍/👎 reactions on the comments we posted.

GitHub sends no webhook for reactions, so a periodic worker task reads the
review comments of PRs we reviewed in the last week (one paginated call per
PR; each comment carries a `reactions` summary), finds ours by the hidden
fingerprint marker, and stores the counts.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("prana_app.feedback")

_MARKER = re.compile(r"<!-- prana:([0-9a-f]{16}) -->")


def reactions_from(comments: list[dict]) -> list[tuple[str, int, int, int]]:
    """(fingerprint, comment id, 👍, 👎) for each of our comments."""
    out = []
    for c in comments:
        m = _MARKER.search(c.get("body") or "")
        if not m or not c.get("id"):
            continue
        r = c.get("reactions") or {}
        out.append((m.group(1), int(c["id"]), int(r.get("+1") or 0), int(r.get("-1") or 0)))
    return out


def refresh(store, client_for, days: int = 7) -> dict:
    """client_for(installation_id) -> GitHub client. Failures on one PR are
    logged and skipped; the next run picks it up again."""
    prs = updated = errors = 0
    for inst, repo, pr in store.feedback_targets(days):
        prs += 1
        try:
            items = reactions_from(client_for(inst).review_comments(repo, pr))
            updated += store.update_feedback(repo, pr, items)
        except Exception as e:  # noqa: BLE001 - one PR must not stop the sweep
            errors += 1
            log.warning("feedback for %s#%s failed: %s", repo, pr, e)
    return {"prs": prs, "comments_updated": updated, "errors": errors}
