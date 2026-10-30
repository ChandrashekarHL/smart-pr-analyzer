from __future__ import annotations

import json
import time

import httpx
import jwt
import pytest
from conftest import make_store
from fastapi.testclient import TestClient

from prana_app.app import create_app
from prana_app.github import GitHubApp, GitHubClient, GitHubError, app_jwt
from prana_app.security import sign

SECRET = "whsec_test"


# ------------------------------------------------------------- GitHub auth

def test_app_jwt_claims_verify_with_public_key(rsa_key):
    pem, pub = rsa_key
    claims = jwt.decode(app_jwt("123", pem, now=1_000_000), pub, algorithms=["RS256"],
                        options={"verify_exp": False, "verify_iat": False})
    assert claims == {"iat": 999_940, "exp": 1_000_540, "iss": "123"}


def test_installation_token_is_cached_and_refreshed_before_expiry(rsa_key):
    pem, pub = rsa_key
    calls = []
    clock = {"t": 1_000_000.0}

    def handler(req: httpx.Request):
        calls.append(req)
        assert req.url.path == "/app/installations/42/access_tokens"
        token = req.headers["Authorization"].removeprefix("Bearer ")
        assert jwt.decode(token, pub, algorithms=["RS256"],
                          options={"verify_exp": False, "verify_iat": False})["iss"] == "7"
        exp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock["t"] + 3600))
        return httpx.Response(201, json={"token": f"tok{len(calls)}", "expires_at": exp})

    app = GitHubApp("7", pem, transport=httpx.MockTransport(handler), clock=lambda: clock["t"])
    assert app.installation_token(42) == "tok1"
    clock["t"] += 3000                                   # 10 min before expiry: still cached
    assert app.installation_token(42) == "tok1" and len(calls) == 1
    clock["t"] += 400                                    # inside the 5-min margin: refresh
    assert app.installation_token(42) == "tok2" and len(calls) == 2


def _client(handler):
    return GitHubClient(lambda: "t", transport=httpx.MockTransport(handler),
                        sleep=lambda s: None)


def test_pagination_collects_all_pages():
    def handler(req):
        page = int(req.url.params["page"])
        return httpx.Response(200, json=[{"n": i} for i in range(100 if page < 3 else 7)])
    assert len(_client(handler).pull_files("o/r", 1)) == 207


def test_retries_5xx_and_rate_limit_but_not_4xx():
    seq = iter([httpx.Response(502), httpx.Response(429, headers={"retry-after": "1"}),
                httpx.Response(200, json={"head": {"sha": "x"}})])
    assert _client(lambda req: next(seq)).pull_request("o/r", 1)["head"]["sha"] == "x"

    calls = []

    def not_found(req):
        calls.append(1)
        return httpx.Response(404, json={"message": "Not Found"})
    with pytest.raises(GitHubError) as e:
        _client(not_found).pull_request("o/r", 1)
    assert e.value.status == 404 and not e.value.retryable and len(calls) == 1


def test_create_review_sends_one_review_with_inline_comments():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        assert req.url.path == "/repos/o/r/pulls/5/reviews"
        assert req.headers["X-GitHub-Api-Version"] == "2022-11-28"
        return httpx.Response(200, json={"id": 1})
    _client(handler).create_review("o/r", 5, "h1", "summary",
                                   [{"path": "a.py", "line": 3, "side": "RIGHT", "body": "x"}])
    assert seen["event"] == "COMMENT" and seen["commit_id"] == "h1" and len(seen["comments"]) == 1


# ------------------------------------------------------------- webhook app

@pytest.fixture
def web(tmp_path):
    store = make_store(tmp_path, 'db.sqlite')
    queued = []
    app = create_app(SECRET, store, queued.append, dashboard_token="dash-token")
    return TestClient(app), queued, store


def _pr_payload(action="opened", sha="abc", draft=False, number=5):
    return {"action": action, "installation": {"id": 42},
            "repository": {"full_name": "o/r"},
            "pull_request": {"number": number, "draft": draft, "head": {"sha": sha}}}


def _post(client, payload, event="pull_request", delivery="d1", secret=SECRET):
    body = json.dumps(payload).encode()
    return client.post("/webhook", content=body, headers={
        "X-GitHub-Event": event, "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": sign(secret, body), "Content-Type": "application/json"})


def test_bad_signature_is_rejected_before_anything_else(web):
    client, queued, _ = web
    assert _post(client, _pr_payload(), secret="wrong").status_code == 401
    assert queued == []


def test_pr_opened_is_queued_fast_and_redelivery_ignored(web):
    client, queued, store = web
    r = _post(client, _pr_payload())
    assert r.status_code == 202 and r.json()["result"] == "queued"
    assert r.json()["ms"] < 200                         # NFR: acknowledge within 200 ms
    assert queued == [r.json()["run_id"]]
    assert _post(client, _pr_payload()).json()["result"] == "duplicate delivery"
    assert _post(client, _pr_payload(), delivery="d2").json()["result"] == \
        "already reviewed or queued"                     # same head, new delivery id
    assert _post(client, _pr_payload(action="synchronize", sha="def"),
                 delivery="d3").status_code == 202       # new head: new run
    assert len(queued) == 2


@pytest.mark.parametrize("payload,event,expected", [
    (_pr_payload(draft=True), "pull_request", "draft pull request"),
    (_pr_payload(action="closed"), "pull_request", "ignored action closed"),
    ({"zen": "hi"}, "ping", "pong"),
    ({"action": "created"}, "installation", "ignored event installation"),
])
def test_events_that_do_not_review(web, payload, event, expected):
    client, queued, _ = web
    r = _post(client, payload, event=event)
    assert r.status_code == 200 and r.json()["result"] == expected and queued == []


def test_health_and_dashboard_auth(web):
    client, _, _ = web
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/api/stats").status_code == 401
    r = client.get("/api/stats", headers={"Authorization": "Bearer dash-token"})
    assert r.status_code == 200 and "runs_by_status" in r.json()


def test_allowlist_ignores_other_owners_case_insensitively(tmp_path):
    from prana_app.config import parse_owners
    assert parse_owners(" ChandrashekarHL, demo-org ,,") == {"chandrashekarhl", "demo-org"}
    assert parse_owners("") == frozenset()

    store = make_store(tmp_path, 'db.sqlite')
    queued = []
    client = TestClient(create_app(SECRET, store, queued.append,
                                   allowed_owners=parse_owners("o")))
    stranger = _pr_payload()
    stranger["repository"]["full_name"] = "stranger/repo"
    r = _post(client, stranger, delivery="x1")
    assert r.status_code == 200 and r.json()["result"] == "owner not allowed" and queued == []
    shouty = _pr_payload()
    shouty["repository"]["full_name"] = "O/r"                       # same owner, any case
    assert _post(client, shouty, delivery="x2").status_code == 202 and len(queued) == 1


def test_empty_allowlist_serves_everyone(web):
    client, queued, _ = web                                         # created without a list
    p = _pr_payload()
    p["repository"]["full_name"] = "anyone/repo"
    assert _post(client, p, delivery="y1").status_code == 202 and len(queued) == 1
