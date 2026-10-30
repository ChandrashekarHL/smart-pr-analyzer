"""FastAPI app: the GitHub webhook, health, and the dashboard API.

    uvicorn prana_app.app:app --host 0.0.0.0 --port 8000

`POST /webhook` does only cheap work before answering (FR-11: <= 200 ms):
verify the HMAC signature on the raw body, record the delivery id (GitHub
redelivers; a repeat is acknowledged and ignored), open a run for
(repo, PR, head commit) and enqueue it. The review happens in the worker.
"""

# No `from __future__ import annotations`: FastAPI resolves route annotations
# at definition time (see prana.api for the bug this avoids).
import hmac
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from prana_app.security import verify_signature
from prana_app.store import Store

log = logging.getLogger("prana_app.app")

PR_ACTIONS = frozenset({"opened", "synchronize", "reopened", "ready_for_review"})
MAX_BODY_BYTES = 25 * 1024 * 1024        # GitHub caps payloads at 25 MB


def create_app(webhook_secret: str, store: Store, enqueue: Callable[[int], None],
               dashboard_token: str = "",
               allowed_owners: frozenset[str] = frozenset()) -> FastAPI:
    """`allowed_owners`: lower-case GitHub logins whose repositories are
    reviewed; empty means every installation is served."""
    app = FastAPI(title="prana reviewer", version="0.1.0", docs_url=None, redoc_url=None)

    def check_dashboard(authorization: str | None) -> None:
        if not dashboard_token:
            raise HTTPException(403, "dashboard API disabled (no DASHBOARD_TOKEN_FILE)")
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, dashboard_token):
            raise HTTPException(401, "missing or wrong bearer token")

    @app.post("/webhook")
    async def webhook(request: Request,
                      x_github_event: str = Header(default=""),
                      x_github_delivery: str = Header(default=""),
                      x_hub_signature_256: str = Header(default="")):
        started = time.perf_counter()
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(413, "payload too large")
        if not verify_signature(webhook_secret, body, x_hub_signature_256):
            raise HTTPException(401, "bad signature")
        if not x_github_delivery:
            raise HTTPException(400, "missing X-GitHub-Delivery")
        try:
            payload = json.loads(body)
        except ValueError as e:
            raise HTTPException(400, "body is not JSON") from e

        result = await _handle(x_github_event, x_github_delivery, payload)
        result["ms"] = round((time.perf_counter() - started) * 1e3, 1)
        log.info(json.dumps({"delivery": x_github_delivery, "event": x_github_event, **result}))
        return JSONResponse(result, status_code=202 if result.get("run_id") else 200)

    async def _handle(event: str, delivery: str, payload: dict) -> dict:
        from starlette.concurrency import run_in_threadpool
        if not await run_in_threadpool(store.record_delivery, delivery, event):
            return {"result": "duplicate delivery"}
        if event == "ping":
            return {"result": "pong"}
        if event != "pull_request":
            return {"result": f"ignored event {event}"}
        action = payload.get("action")
        pr = payload.get("pull_request") or {}
        if action not in PR_ACTIONS:
            return {"result": f"ignored action {action}"}
        if pr.get("draft"):
            return {"result": "draft pull request"}
        inst = (payload.get("installation") or {}).get("id")
        repo = (payload.get("repository") or {}).get("full_name")
        sha = (pr.get("head") or {}).get("sha")
        number = pr.get("number")
        if not (inst and repo and sha and number):
            return {"result": "incomplete payload"}
        owner = repo.split("/", 1)[0].lower()
        if allowed_owners and owner not in allowed_owners:
            # Installed by someone outside the allowlist: acknowledge, never
            # spend the shared GPU on it.
            return {"result": "owner not allowed"}
        run_id = await run_in_threadpool(store.open_run, int(inst), repo, int(number), sha)
        if run_id is None:
            return {"result": "already reviewed or queued"}
        await run_in_threadpool(enqueue, run_id)
        return {"result": "queued", "run_id": run_id}

    @app.get("/health")
    def health():
        try:
            with store.engine.connect() as c:
                c.execute(text("select 1"))
            return {"ok": True}
        except Exception as e:  # noqa: BLE001 - health reports, never raises
            return JSONResponse({"ok": False, "error": type(e).__name__}, status_code=503)

    @app.get("/api/runs")
    def runs(limit: int = 50, status: str | None = None,
             authorization: str | None = Header(default=None)):
        check_dashboard(authorization)
        rows = store.recent_runs(min(max(limit, 1), 500))
        return [r for r in rows if status is None or r["status"] == status]

    @app.get("/api/stats")
    def stats(authorization: str | None = Header(default=None)):
        check_dashboard(authorization)
        return store.stats()

    @app.get("/api/dashboard")
    def dashboard(days: int = 30, authorization: str | None = Header(default=None)):
        check_dashboard(authorization)
        return store.dashboard(days=min(max(days, 1), 365))

    return app


def _from_env() -> FastAPI:
    from prana_app.services import services
    from prana_app.tasks import enqueue
    svc = services()
    token_file = os.environ.get("DASHBOARD_TOKEN_FILE")
    token = Path(token_file).read_text(encoding="utf-8").strip() if token_file else ""
    return create_app(svc.settings.webhook_secret, svc.store, enqueue, token,
                      svc.settings.allowed_owners)


def __getattr__(name: str):
    # `uvicorn prana_app.app:app` builds the app on first access, so importing
    # this module (tests, tooling) never needs the production environment.
    if name == "app":
        return _from_env()
    raise AttributeError(name)
