"""Tests for the HTTP API and its client. A fake reviewer stands in for the
model, so these run anywhere FastAPI is installed (the Spark) in seconds."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prana.api import MAX_CONTEXT_CHARS, create_app  # noqa: E402

KEY = "k" * 32


@dataclass
class _Result:
    text: str
    parsed: dict | None
    parse_error: str | None
    latency_s: float
    confidence: float = 0.7


class FakeReviewer:
    adapter = "runs/v1.1/adapter"
    model_name = "fake"
    prompt_version = "v2"

    def __init__(self, delay: float = 0.0):
        self.calls = []
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def review(self, diff, language, context):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        self.calls.append((diff, language, context))
        with self._lock:
            self.active -= 1
        parsed = {"explanation": "Guard nil.", "suggestion": "", "severity": "warning",
                  "issue_type": "BUG", "file_path": None, "line_number": 2}
        return _Result(text='{"explanation": "Guard nil."}', parsed=parsed,
                       parse_error=None, latency_s=0.01)


def _client(reviewer=None):
    reviewer = reviewer or FakeReviewer()
    return TestClient(create_app(reviewer, KEY)), reviewer


def _post(c, body, key=KEY):
    return c.post("/review", json=body, headers={"X-API-Key": key} if key else {})


GOOD = {"diff_hunk": "@@ -1 +1 @@\n+x = y", "language": "python"}


def test_health_needs_no_key_and_reports_the_adapter():
    c, _ = _client()
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["adapter"] == "runs/v1.1/adapter"
    assert r.json()["prompt_version"] == "v2"


@pytest.mark.parametrize("key", [None, "wrong", KEY[:-1]])
def test_review_rejects_missing_or_wrong_key(key):
    c, rev = _client()
    assert _post(c, GOOD, key).status_code == 401
    assert rev.calls == []          # nothing reached the GPU


def test_review_happy_path_and_context_passthrough():
    c, rev = _client()
    r = _post(c, {**GOOD, "context": "Errors wrap with %w.", "language": "Python"})
    assert r.status_code == 200
    body = r.json()
    assert body["parsed"]["severity"] == "warning" and "request_id" in body
    assert rev.calls == [("@@ -1 +1 @@\n+x = y", "python", "Errors wrap with %w.")]


def test_file_path_from_caller_overrides_the_model():
    c, _ = _client()
    r = _post(c, {**GOOD, "file_path": "pkg/db.py"})
    assert r.json()["parsed"]["file_path"] == "pkg/db.py"


@pytest.mark.parametrize("bad", [
    {"diff_hunk": "", "language": "python"},
    {"diff_hunk": "x", "language": "java"},
    {"diff_hunk": "x", "language": "go", "context": "c" * (MAX_CONTEXT_CHARS + 1)},
    {"language": "go"},
])
def test_invalid_input_never_reaches_the_model(bad):
    c, rev = _client()
    assert _post(c, bad).status_code == 422
    assert rev.calls == []


def test_generation_is_serialised():
    """One GPU, one model in memory: two concurrent generate calls on the
    same model are unsafe, so the API runs them one at a time."""
    rev = FakeReviewer(delay=0.05)
    c, _ = _client(rev)
    threads = [threading.Thread(target=_post, args=(c, GOOD)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(rev.calls) == 4 and rev.max_active == 1


# ------------------------------------------------------------- client eval

def test_client_eval_writes_scorable_predictions_and_resumes(tmp_path, monkeypatch):
    from prana.client import Client, eval_split
    from prana.data import jsonl

    rows = [{"id": f"r/p_{i}-{i}", "language": "go", "diff_hunk": f"@@ -1 +1 @@\n+v{i}",
             "review_comment": "c", "structured_output": {"severity": "info"}}
            for i in range(3)]
    data = tmp_path / "test_ood.jsonl"
    jsonl.write(data, rows)
    side = tmp_path / "ctx.jsonl"
    jsonl.write(side, [{"id": "r/p_1-1", "rag_context": "ctx one"}])

    seen = []

    def fake_review(self, diff, language, context=None, file_path=None, retries=3):
        seen.append(context)
        return {"text": '{"x": 1}', "parsed": None}

    monkeypatch.setattr(Client, "review", fake_review)
    c = Client("http://x", KEY)
    out = tmp_path / "out"
    stats = eval_split(c, data, out, side, limit=2, log=lambda *a: None)
    assert stats["generated"] == 2 and seen == [None, "ctx one"]
    stats = eval_split(c, data, out, side, log=lambda *a: None)
    assert stats["generated"] == 1                     # resumed
    preds = jsonl.read(out / "test_ood.predictions.jsonl")
    assert len(preds) == 3
    assert {"prediction", "reference_structured", "language"} <= set(preds[0])


def test_client_does_not_retry_client_errors(monkeypatch):
    from prana import client as mod
    calls = []

    def boom(self, method, path, body=None):
        calls.append(path)
        raise mod.APIError(401, "HTTP 401 on /review: nope")

    monkeypatch.setattr(mod.Client, "_request", boom)
    with pytest.raises(mod.APIError):
        mod.Client("http://x", KEY).review("d", "go")
    assert len(calls) == 1


def test_client_retries_server_errors(monkeypatch):
    from prana import client as mod
    calls = []

    def flaky(self, method, path, body=None):
        calls.append(path)
        if len(calls) < 3:
            raise mod.APIError(503, "HTTP 503")
        return {"text": "{}"}

    monkeypatch.setattr(mod.Client, "_request", flaky)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    assert mod.Client("http://x", KEY).review("d", "go") == {"text": "{}"}
    assert len(calls) == 3
