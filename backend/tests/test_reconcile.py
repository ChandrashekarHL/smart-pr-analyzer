"""The periodic sweep that catches pull requests a webhook missed."""

from __future__ import annotations

import httpx
import jwt
import pytest
from conftest import make_store

from prana_app.github import GitHubApp
from prana_app.reconcile import sweep


def _pr(n, sha, draft=False):
    return {"number": n, "draft": draft, "head": {"sha": sha}}


class FakeGH:
    def __init__(self, repos, pulls, fail=False):
        self.repos, self.pulls, self.fail = repos, pulls, fail

    def installation_repos(self):
        if self.fail:
            raise RuntimeError("token refused")
        return [{"full_name": r} for r in self.repos]

    def open_pulls(self, repo):
        return self.pulls.get(repo, [])


class FakeApp:
    def __init__(self, installs):
        self._installs = installs          # [(id, login, FakeGH)]

    def installations(self):
        return [{"id": i, "account": {"login": login}} for i, login, _ in self._installs]

    def client(self, inst_id):
        return next(gh for i, _, gh in self._installs if i == inst_id)


@pytest.fixture
def store(tmp_path):
    s = make_store(tmp_path, 'db.sqlite')
    return s


def test_queues_only_unseen_heads_and_skips_drafts(store):
    gh = FakeGH(["Me/demo"], {"Me/demo": [_pr(1, "a1"), _pr(2, "b1", draft=True), _pr(3, "c1")]})
    queued = []
    store.open_run(7, "Me/demo", 3, "c1")                     # webhook already got this one
    stats = sweep(FakeApp([(7, "Me", gh)]), store, queued.append)
    assert stats["queued"] == 1 and stats["open_prs"] == 2 and len(queued) == 1
    assert store.get_run(queued[0]).pr_number == 1
    assert sweep(FakeApp([(7, "Me", gh)]), store, queued.append)["queued"] == 0   # idempotent


def test_never_reopens_a_failed_run(store):
    gh = FakeGH(["Me/demo"], {"Me/demo": [_pr(1, "a1")]})
    rid = store.open_run(7, "Me/demo", 1, "a1")
    store.mark(rid, "failed", error="x")
    queued = []
    assert sweep(FakeApp([(7, "Me", gh)]), store, queued.append)["queued"] == 0
    assert store.get_run(rid).status == "failed" and queued == []


def test_allowlist_and_cap(store):
    mine = FakeGH(["me/a"], {"me/a": [_pr(i, f"s{i}") for i in range(1, 6)]})
    theirs = FakeGH(["stranger/x"], {"stranger/x": [_pr(1, "z")]})
    queued = []
    stats = sweep(FakeApp([(1, "Me", mine), (2, "stranger", theirs)]), store, queued.append,
                  allowed_owners=frozenset({"me"}), max_new=3)
    assert stats["skipped_owner"] == 1 and stats["queued"] == 3 and stats["open_prs"] == 5
    assert {store.get_run(r).repo for r in queued} == {"me/a"}


def test_a_failing_installation_does_not_stop_the_sweep(store):
    ok = FakeGH(["me/a"], {"me/a": [_pr(1, "s1")]})
    queued = []
    stats = sweep(FakeApp([(1, "me", FakeGH([], {}, fail=True)), (2, "me", ok)]), store,
                  queued.append)
    assert stats["errors"] == 1 and stats["queued"] == 1


def test_github_calls_are_authenticated_and_paginated(rsa_key):
    pem, pub = rsa_key
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.url.path, dict(req.url.params)))
        auth = req.headers["Authorization"].removeprefix("Bearer ")
        if req.url.path == "/app/installations":
            assert jwt.decode(auth, pub, algorithms=["RS256"],
                              options={"verify_exp": False, "verify_iat": False})["iss"] == "9"
            return httpx.Response(200, json=[{"id": 5, "account": {"login": "Me"}}])
        if req.url.path == "/app/installations/5/access_tokens":
            return httpx.Response(201, json={"token": "inst-tok"})
        assert auth == "inst-tok"
        if req.url.path == "/installation/repositories":
            page = int(req.url.params["page"])
            n = 100 if page == 1 else 3
            return httpx.Response(200, json={"repositories": [{"full_name": f"me/r{page}-{i}"}
                                                              for i in range(n)]})
        if req.url.path == "/repos/me/demo/pulls":
            assert req.url.params["state"] == "open"
            return httpx.Response(200, json=[_pr(1, "a")])
        return httpx.Response(404)

    app = GitHubApp("9", pem, transport=httpx.MockTransport(handler))
    assert app.installations()[0]["id"] == 5
    gh = app.client(5)
    assert len(gh.installation_repos()) == 103
    assert gh.open_pulls("me/demo")[0]["number"] == 1
