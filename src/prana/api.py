"""HTTP API for the review model. This is how teammates and the GitHub App
reach the model; nobody needs a shell on the Spark.

    # on the Spark, through the GPU queue (holds the GPU while it runs):
    run-job start prana-api -- bash scripts/serve_api.sh runs/v1.1/adapter

    curl -s http://100.124.121.97:8100/health
    curl -s -X POST http://100.124.121.97:8100/review \\
         -H "X-API-Key: $KEY" -H "Content-Type: application/json" \\
         -d '{"diff_hunk": "@@ -1 +1 @@\\n+x = 1", "language": "python"}'

Design:
- Bound to the Tailscale address only, so the campus LAN cannot reach it.
- API key in the `X-API-Key` header, compared in constant time. The key is
  read from a 600-mode file, never from argv or the environment of the job.
- One generation at a time behind a lock: there is one GPU and one model in
  memory, and concurrent `generate` calls on the same model are unsafe.
  Requests queue in FastAPI's thread pool; `/health` reports the queue depth.
- Inputs are validated and bounded before they reach the GPU.
- Each request is logged as one JSON line with latency and sizes, not content.
"""

# No `from __future__ import annotations` here: FastAPI resolves parameter
# annotations at route-definition time, and string annotations naming the
# request model (defined inside create_app) cannot be resolved from module
# globals - FastAPI then silently treats the body as missing query parameters
# and every request fails with 422.
import argparse
import hmac
import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from prana.serve import LANGUAGES

MAX_DIFF_CHARS = 20_000
MAX_CONTEXT_CHARS = 4_000          # about 800-1,000 tokens of English and code
DEFAULT_KEY_FILE = Path.home() / ".config" / "prana" / "api.key"

log = logging.getLogger("prana.api")


def load_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if len(key) < 24:
        raise SystemExit(f"API key in {path} is too short; generate a new one")
    return key


def create_app(reviewer, api_key: str):
    """`reviewer` is anything with `.review(diff, language, context)` returning
    a ReviewResult-like dataclass, plus `.adapter`, `.model_name`,
    `.prompt_version`. Tests pass a fake."""
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from pydantic import BaseModel, Field

    class ReviewRequest(BaseModel):
        diff_hunk: str = Field(min_length=1, max_length=MAX_DIFF_CHARS)
        language: str
        context: str | None = Field(default=None, max_length=MAX_CONTEXT_CHARS)
        file_path: str | None = Field(default=None, max_length=500)

    app = FastAPI(title="prana reviewer", version="1.1")
    gpu = threading.Lock()
    state = {"waiting": 0, "served": 0, "started": time.time()}
    state_lock = threading.Lock()

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "adapter": getattr(reviewer, "adapter", None),
            "model": getattr(reviewer, "model_name", None),
            "prompt_version": getattr(reviewer, "prompt_version", None),
            "languages": list(LANGUAGES),
            "waiting": state["waiting"],
            "served": state["served"],
            "uptime_s": round(time.time() - state["started"]),
        }

    @app.post("/review", dependencies=[Depends(check_key)])
    def review(req: ReviewRequest, request: Request):
        lang = req.language.strip().lower()
        if lang not in LANGUAGES:
            raise HTTPException(status_code=422,
                                detail=f"language must be one of {list(LANGUAGES)}")
        rid = uuid.uuid4().hex[:12]
        with state_lock:
            state["waiting"] += 1
        queued = time.perf_counter()
        try:
            with gpu:
                with state_lock:
                    state["waiting"] -= 1
                result = reviewer.review(req.diff_hunk, lang, req.context)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        total = time.perf_counter() - queued
        with state_lock:
            state["served"] += 1
        body = asdict(result)
        if req.file_path and body.get("parsed"):
            # The model sees the hunk, not the path; the caller knows it exactly.
            body["parsed"]["file_path"] = req.file_path
        body["request_id"] = rid
        body["total_s"] = round(total, 3)
        log.info(json.dumps({
            "id": rid, "client": request.client.host if request.client else None,
            "language": lang, "diff_chars": len(req.diff_hunk),
            "context_chars": len(req.context or ""), "valid": body.get("parsed") is not None,
            "gen_s": body.get("latency_s"), "total_s": body["total_s"],
        }))
        return body

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--host", default="100.124.121.97",
                    help="bind address; the Tailscale IP keeps it off the campus LAN")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    api_key = load_api_key(Path(args.key_file))

    import uvicorn

    from prana.serve import Reviewer
    log.info("loading %s", args.adapter)
    reviewer = Reviewer.load(args.adapter)
    log.info("model ready; serving on http://%s:%d", args.host, args.port)
    uvicorn.run(create_app(reviewer, api_key), host=args.host, port=args.port,
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()
