"""Celery worker: one task per (repo, PR, head commit) review run (FR-11).

- `acks_late` + `reject_on_worker_lost`: a worker killed mid-review does not
  lose the job; it is redelivered.
- Up to 3 retries with exponential back-off for transient failures (model API
  down, GitHub 5xx / rate limit).
- After the last retry the run is marked `failed` with its error (the
  dead-letter record: `/api/runs?status=failed`, replay with
  `python -m prana_app.admin replay <run_id>`), and the PR gets one comment
  saying the review could not run, instead of silence.
"""

from __future__ import annotations

import logging
import os

from celery import Celery

from prana_app.github import GitHubError
from prana_app.pipeline import ModelUnavailable, review_pull_request

log = logging.getLogger("prana_app.tasks")

celery = Celery("prana_app", broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
celery.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=240,
    task_time_limit=300,
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
)

MAX_RETRIES = 3
FAILURE_COMMENT = ("⚠️ prana could not review this pull request ({reason}). "
                   "It will try again on the next push.")


def _transient(e: Exception) -> bool:
    return isinstance(e, ModelUnavailable) or (isinstance(e, GitHubError) and e.retryable)


def run_review(svc, run_id: int, attempt: int) -> str:
    """The task body, separated from Celery so it can be tested directly.
    Returns the final status, or raises for a transient error to retry."""
    run = svc.store.get_run(run_id)
    if run is None:
        log.error("run %s not found", run_id)
        return "missing"
    svc.store.mark(run_id, "processing")
    gh = svc.app.client(run.installation_id)
    try:
        outcome = review_pull_request(gh, svc.model, svc.policy, run.repo, run.pr_number,
                                      run.head_sha, svc.retrieve)
    except Exception as e:  # noqa: BLE001 - classified below
        if _transient(e) and attempt < MAX_RETRIES:
            svc.store.mark(run_id, "pending", error=f"attempt {attempt + 1}: {e}")
            raise
        svc.store.mark(run_id, "failed", error=f"{type(e).__name__}: {e}")
        reason = "the review model is unavailable" if isinstance(e, ModelUnavailable) \
            else "an internal error"
        try:
            gh.issue_comment(run.repo, run.pr_number, FAILURE_COMMENT.format(reason=reason))
        except Exception as e2:  # noqa: BLE001 - best effort; the run is already failed
            log.error("could not post failure comment on %s#%s: %s", run.repo,
                      run.pr_number, e2)
        return "failed"
    svc.store.finish(run_id, outcome)
    log.info("run %s %s#%s %s: %d comments, %d/%d hunks, %.1f s", run_id, run.repo,
             run.pr_number, outcome.status, len(outcome.posted), outcome.hunks_reviewed,
             outcome.hunks_total, outcome.seconds)
    return outcome.status


@celery.task(bind=True, name="prana_app.review_pull_request", max_retries=MAX_RETRIES)
def review_task(self, run_id: int) -> str:
    from prana_app.services import services
    try:
        return run_review(services(), run_id, self.request.retries)
    except Exception as e:  # noqa: BLE001 - run_review only re-raises transient errors
        raise self.retry(exc=e, countdown=30 * 2 ** self.request.retries) from e


def enqueue(run_id: int) -> None:
    review_task.delay(run_id)


FEEDBACK_EVERY_S = 15 * 60
celery.conf.beat_schedule = {
    "refresh-feedback": {"task": "prana_app.refresh_feedback", "schedule": FEEDBACK_EVERY_S},
}


@celery.task(name="prana_app.refresh_feedback")
def feedback_task() -> dict:
    from prana_app.feedback import refresh
    from prana_app.services import services
    svc = services()
    result = refresh(svc.store, svc.app.client)
    log.info("feedback refresh: %s", result)
    return result
