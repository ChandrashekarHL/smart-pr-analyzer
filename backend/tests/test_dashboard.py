"""Dashboard (FR-10): store aggregates, feedback collection, the API endpoint,
and the Streamlit page itself rendered headlessly against a live HTTP API."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import FakeGitHub, make_store

from prana_app.feedback import reactions_from, refresh
from prana_app.pipeline import Outcome
from prana_app.review import Finding
from prana_app.store import percentile

DASH = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASH))
import data  # noqa: E402

TOKEN = "dash-token-for-tests"


def _finding(sev, issue, conf, line=10, text="Check the return value."):
    return Finding(path="app/x.py", line=line, line_exact=True, severity=sev, issue_type=issue,
                   explanation=f"{text} ({sev} {line})", suggestion="", confidence=conf)


@pytest.fixture
def seeded(tmp_path):
    store = make_store(tmp_path, 'db.sqlite')
    r1 = store.open_run(1, "o/r", 1, "a1")
    store.finish(r1, Outcome("posted", hunks_total=3, hunks_reviewed=3, seconds=8.0,
                             posted=[_finding("critical", "SECURITY", 0.91, 10),
                                     _finding("warning", "BUG", 0.55, 20)],
                             low_confidence=[_finding("info", "STYLE", 0.10, 30)]))
    r2 = store.open_run(1, "o/r", 2, "b1")
    store.finish(r2, Outcome("posted", hunks_total=9, hunks_reviewed=8, seconds=24.0,
                             posted=[_finding("warning", "BUG", 0.72, 5)]))
    r3 = store.open_run(1, "o/q", 7, "c1")
    store.mark(r3, "failed", error="ModelUnavailable: all 4 model calls failed")
    return store


# ------------------------------------------------------------------- store

def test_percentile_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_dashboard_aggregates(seeded):
    d = seeded.dashboard(days=30)
    assert d["totals"] == {"runs": 3, "reviews_posted": 2, "failed": 1, "comments_inline": 3,
                           "comments_held_back": 1, "hunks_reviewed": 11}
    assert d["comments_by_severity"] == {"warning": 2, "critical": 1}
    assert d["comments_by_issue_type"] == {"BUG": 2, "SECURITY": 1}
    assert d["latency"] == {"p50_s": 8.0, "p95_s": 24.0, "timed_runs": 2}
    assert sum(b["comments"] for b in d["confidence_histogram"]) == 3
    assert d["confidence_histogram"][9]["comments"] == 1          # 0.91
    assert d["feedback"] == {"thumbs_up": 0, "thumbs_down": 0, "rated_comments": 0,
                             "helpful_rate": None}
    assert d["recent_runs"][0]["status"] == "failed" and "ModelUnavailable" in \
        d["recent_runs"][0]["error"]
    assert len(d["daily"]) == 1 and d["daily"][0]["runs"] == 3
    json.dumps(d)                                                  # serialisable as is


def test_feedback_is_collected_from_reactions(seeded):
    body = _finding("critical", "SECURITY", 0.91, 10).body()
    gh = FakeGitHub(existing=[
        {"id": 501, "body": body, "reactions": {"+1": 3, "-1": 1}},
        {"id": 502, "body": "a human comment", "reactions": {"+1": 9}},
    ])
    assert reactions_from(gh.existing) == [(_finding("critical", "SECURITY", 0.91, 10)
                                            .fingerprint, 501, 3, 1)]
    result = refresh(seeded, lambda inst: gh)
    assert result == {"prs": 2, "comments_updated": 1, "errors": 0}
    fb = seeded.dashboard()["feedback"]
    assert fb == {"thumbs_up": 3, "thumbs_down": 1, "rated_comments": 1, "helpful_rate": 0.75}


def test_feedback_sweep_survives_a_failing_pr(seeded):
    class Boom(FakeGitHub):
        def review_comments(self, repo, number):
            raise RuntimeError("404")
    assert refresh(seeded, lambda inst: Boom()) == {"prs": 2, "comments_updated": 0,
                                                    "errors": 2}


def test_dashboard_endpoint_requires_token(seeded):
    from fastapi.testclient import TestClient

    from prana_app.app import create_app
    client = TestClient(create_app("s", seeded, lambda r: None, dashboard_token=TOKEN))
    assert client.get("/api/dashboard").status_code == 401
    r = client.get("/api/dashboard?days=7", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200 and r.json()["window_days"] == 7


# ------------------------------------------------------------ data module

def test_data_helpers():
    d = {"comments_by_severity": {"warning": 2}, "latency": {"p95_s": 24.0},
         "daily": [{"day": "2026-10-30", "p50_s": 8.0, "p95_s": None}]}
    assert data.severity_rows(d) == [{"severity": "critical", "comments": 0},
                                     {"severity": "warning", "comments": 2},
                                     {"severity": "info", "comments": 0}]
    assert data.latency_rows(d) == [{"day": "2026-10-30", "measure": "p50", "seconds": 8.0}]
    assert data.p95_delta(d) == "-6.0 s vs 30 s target"
    assert data.p95_delta({"latency": {"p95_s": None}}) is None


class _Api:
    """A real HTTP server answering /api/dashboard like the backend does."""

    def __init__(self, payload: dict | None):
        payload_bytes = json.dumps(payload, default=str).encode() if payload else b""

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload_bytes)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def test_fetch_auth_and_unreachable(seeded):
    api = _Api(seeded.dashboard())
    try:
        assert data.fetch(api.url, TOKEN, 30)["totals"]["runs"] == 3
        with pytest.raises(data.DashboardError, match="HTTP 401.*DASHBOARD_TOKEN_FILE"):
            data.fetch(api.url, "wrong", 30)
    finally:
        api.close()
    with pytest.raises(data.DashboardError, match="unreachable"):
        data.fetch("http://127.0.0.1:9", TOKEN, 30, timeout=2)
    with pytest.raises(data.DashboardError, match="must be http"):
        data.fetch("file:///etc/passwd", TOKEN, 30)


# ------------------------------------------------------------ the page

def _run_page(monkeypatch, tmp_path, api_url):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()                 # each test is a fresh dashboard session
    tok = tmp_path / "token"
    tok.write_text(TOKEN)
    monkeypatch.setenv("DASHBOARD_API_URL", api_url)
    monkeypatch.setenv("DASHBOARD_TOKEN_FILE", str(tok))
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=30)
    return at.run()


def test_page_renders_kpis_charts_and_tables(seeded, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    api = _Api(seeded.dashboard())
    try:
        at = _run_page(monkeypatch, tmp_path, api.url)
    finally:
        api.close()
    assert not at.exception, at.exception
    assert at.title[0].value == "PR review dashboard"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Reviews posted": "2", "Inline comments": "3",
                       "p95 review time": "24.0 s", "Helpful (👍 share)": "–",
                       "Failed runs": "1"}
    assert [s.value for s in at.subheader] == [
        "Review time per day", "Comments by severity", "Comments by issue type",
        "Model confidence", "Recent reviews", "Recent comments", "Training and evaluation"]
    assert len(at.dataframe) == 2 and len(at.dataframe[0].value) == 3
    assert not at.error


def test_page_shows_an_error_when_the_backend_is_down(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    at = _run_page(monkeypatch, tmp_path, "http://127.0.0.1:9")
    assert not at.exception
    assert "Cannot load dashboard data" in at.error[0].value and not at.metric


def test_page_empty_state(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    empty = make_store(tmp_path, 'e.sqlite')
    api = _Api(empty.dashboard())
    try:
        at = _run_page(monkeypatch, tmp_path, api.url)
    finally:
        api.close()
    assert not at.exception and "No reviews in this window yet" in at.info[0].value
