from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

PATCH = """@@ -10,4 +10,6 @@ def handler(req):
     user = load(req)
-    q = "SELECT * FROM t WHERE id=" + req.id
+    q = f"SELECT * FROM t WHERE id={req.id}"
+    rows = db.execute(q)
     return user
@@ -40,2 +42,3 @@ def other():
     a = 1
+    b = 2
     return a"""


@pytest.fixture(scope="session")
def rsa_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return pem, pub


def make_store(tmp_path, name: str = "db.sqlite"):
    """A fresh, empty store. SQLite by default; set PRANA_TEST_DATABASE_URL to a
    throwaway Postgres to run every store-backed test against the production
    engine (SQLite has no 32-bit INTEGER limit, so it hid a real overflow)."""
    import os

    from prana_app.store import Base, Store
    url = os.environ.get("PRANA_TEST_DATABASE_URL")
    if url:
        s = Store(url)
        Base.metadata.drop_all(s.engine)
    else:
        s = Store(f"sqlite:///{tmp_path / name}")
    s.create_all()
    return s


class FakeGitHub:
    def __init__(self, head="h1", files=None, existing=None, reject_inline=False):
        self.head = head
        self.files = files if files is not None else [
            {"filename": "app/api.py", "status": "modified", "patch": PATCH},
            {"filename": "README.md", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        ]
        self.existing = existing or []
        self.reject_inline = reject_inline
        self.posted = []          # reviews created through create_review
        self.issue_comments = []

    def pull_request(self, repo, number):
        return {"head": {"sha": self.head}, "base": {"sha": "b1"}}

    def pull_files(self, repo, number):
        return self.files

    def review_comments(self, repo, number):
        return self.existing

    def create_review(self, repo, number, commit_id, body, comments):
        from prana_app.github import GitHubError
        if self.reject_inline and comments:
            raise GitHubError(422, "Unprocessable: line not part of the diff")
        self.posted.append({"id": 99 + len(self.posted), "commit_id": commit_id, "body": body,
                            "comments": comments, "user": {"login": "prana-reviewer[bot]"}})
        return {"id": self.posted[-1]["id"]}

    def reviews(self, repo, number):
        return self.posted

    def issue_comment(self, repo, number, body):
        self.issue_comments.append(body)
        return {"id": 7}


def model_result(line=12, severity="critical", issue="SECURITY", confidence=0.8,
                 explanation="Query built from user input is open to SQL injection."):
    return {"text": json.dumps({"explanation": explanation}), "confidence": confidence,
            "parsed": {"explanation": explanation, "suggestion": "Use a parameterised query.",
                       "severity": severity, "issue_type": issue, "file_path": None,
                       "line_number": line},
            "model": "deepseek", "adapter": "runs/v1.1/adapter"}


class FakeModel:
    def __init__(self, results=None, fail=False, delay=0.0, clock=None):
        self.results = results or {}
        self.fail = fail
        self.calls = []
        self.delay = delay
        self.clock = clock

    def __call__(self, hunk, language, context, path):
        self.calls.append((hunk.splitlines()[0], language, context, path))
        if self.clock is not None:
            self.clock.t += self.delay
        if self.fail:
            raise RuntimeError("model API down")
        return self.results.get(hunk.splitlines()[0], model_result())


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def gh():
    return FakeGitHub()
