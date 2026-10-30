"""GitHub App backend for the PR Analyzer (design doc, Module 4).

Flow: GitHub webhook -> `app` (verify HMAC, record delivery, enqueue, 202 in
milliseconds) -> Celery `tasks.review_pull_request` -> `pipeline` (fetch the
diff, pick hunks, optional retrieval, model API per hunk, validate line
numbers, de-duplicate) -> one GitHub review with inline comments -> `store`
records every run and comment for the dashboard.
"""

__version__ = "0.1.0"
