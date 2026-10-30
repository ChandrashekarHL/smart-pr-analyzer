"""Periodic sweep: review open pull requests whose head commit has no run.

Webhooks are the fast path, but they can be lost: GitHub gives each delivery
10 seconds and does not retry, the public ingress can be slow, and nothing is
delivered while the server is down. This sweep makes reviews independent of
that. Every few minutes it asks GitHub (outbound calls only) for the open,
non-draft pull requests in every repository the App is installed on, and
queues a run for any head commit not seen before.

It never reopens a failed run (that is `prana_app.admin replay`), skips
owners outside the allowlist, and queues at most `max_new` runs per sweep so a
first install on a busy repository cannot flood the shared GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger("prana_app.reconcile")

MAX_NEW_PER_SWEEP = 10


def sweep(app, store, enqueue: Callable[[int], None],
          allowed_owners: frozenset[str] = frozenset(),
          max_new: int = MAX_NEW_PER_SWEEP) -> dict:
    stats = {"installations": 0, "repos": 0, "open_prs": 0, "queued": 0,
             "skipped_owner": 0, "errors": 0}
    for inst in app.installations():
        login = ((inst.get("account") or {}).get("login") or "").lower()
        if allowed_owners and login not in allowed_owners:
            stats["skipped_owner"] += 1
            continue
        stats["installations"] += 1
        try:
            gh = app.client(int(inst["id"]))
            repos = gh.installation_repos()
        except Exception as e:  # noqa: BLE001 - one installation must not stop the sweep
            stats["errors"] += 1
            log.warning("installation %s: %s", inst.get("id"), e)
            continue
        for repo in repos:
            name = repo.get("full_name") or ""
            if allowed_owners and name.split("/", 1)[0].lower() not in allowed_owners:
                continue
            stats["repos"] += 1
            try:
                pulls = gh.open_pulls(name)
            except Exception as e:  # noqa: BLE001
                stats["errors"] += 1
                log.warning("%s: %s", name, e)
                continue
            for pr in pulls:
                if pr.get("draft"):
                    continue
                stats["open_prs"] += 1
                if stats["queued"] >= max_new:
                    continue
                sha = (pr.get("head") or {}).get("sha")
                if not sha:
                    continue
                run_id = store.open_run(int(inst["id"]), name, int(pr["number"]), sha,
                                        reopen_failed=False)
                if run_id is not None:
                    enqueue(run_id)
                    stats["queued"] += 1
                    log.info("queued %s#%s @ %s (missed by webhook)", name, pr["number"],
                             sha[:12])
    return stats
