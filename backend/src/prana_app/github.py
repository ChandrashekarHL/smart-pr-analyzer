"""GitHub App authentication and the few REST calls the reviewer needs.

Auth (design doc 4.2): the App's RSA private key signs a 10-minute JWT; the
JWT buys an installation access token (1 hour); tokens are cached per
installation and refreshed 5 minutes before they expire. No personal tokens.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import jwt

API_VERSION = "2022-11-28"
REFRESH_MARGIN_S = 300
MAX_RETRIES = 3


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in (0, 429, 500, 502, 503, 504)


def app_jwt(app_id: str, private_key: str, now: float | None = None) -> str:
    now = int(now if now is not None else time.time())
    # iat backdated 60 s against clock drift, as GitHub recommends.
    return jwt.encode({"iat": now - 60, "exp": now + 540, "iss": app_id}, private_key,
                      algorithm="RS256")


@dataclass
class _Token:
    value: str
    expires_at: float


class GitHubApp:
    def __init__(self, app_id: str, private_key: str, api: str = "https://api.github.com",
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.time):
        self.app_id = app_id
        self._key = private_key
        self.api = api
        self._transport = transport
        self._clock = clock
        self._tokens: dict[int, _Token] = {}
        self._lock = threading.Lock()

    def installation_token(self, installation_id: int) -> str:
        with self._lock:
            tok = self._tokens.get(installation_id)
            if tok and tok.expires_at - REFRESH_MARGIN_S > self._clock():
                return tok.value
            token = app_jwt(self.app_id, self._key, self._clock())
            with httpx.Client(base_url=self.api, transport=self._transport, timeout=30) as c:
                r = c.post(f"/app/installations/{installation_id}/access_tokens",
                           headers=_headers(f"Bearer {token}"))
            if r.status_code != 201:
                raise GitHubError(r.status_code, f"installation token: {r.text[:200]}")
            body = r.json()
            expires = _parse_time(body.get("expires_at")) or self._clock() + 3600
            self._tokens[installation_id] = _Token(body["token"], expires)
            return body["token"]

    def client(self, installation_id: int) -> GitHubClient:
        return GitHubClient(lambda: self.installation_token(installation_id), self.api,
                            self._transport)

    def installations(self) -> list[dict]:
        """Every installation of this App (authenticated as the App itself)."""
        app_client = GitHubClient(lambda: app_jwt(self.app_id, self._key, self._clock()),
                                  self.api, self._transport)
        return app_client._paginate("/app/installations")


def _parse_time(s: str | None) -> float | None:
    if not s:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _headers(auth: str) -> dict:
    return {"Authorization": auth, "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION, "User-Agent": "prana-reviewer"}


class GitHubClient:
    """REST calls with a token supplier (so a refreshed token is picked up),
    pagination, and retries on 5xx / secondary rate limits."""

    def __init__(self, token: Callable[[], str], api: str = "https://api.github.com",
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._token = token
        self.api = api
        self._transport = transport
        self._sleep = sleep

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with httpx.Client(base_url=self.api, transport=self._transport, timeout=30) as c:
                    r = c.request(method, path, headers=_headers(f"Bearer {self._token()}"), **kw)
            except httpx.TransportError as e:
                if attempt == MAX_RETRIES:
                    raise GitHubError(0, f"{method} {path}: {e}") from e
                self._sleep(2 ** attempt)
                continue
            limited = r.status_code in (403, 429) and (
                r.headers.get("retry-after") or r.headers.get("x-ratelimit-remaining") == "0")
            if r.status_code >= 500 or limited:
                if attempt == MAX_RETRIES:
                    raise GitHubError(429 if limited else r.status_code,
                                      f"{method} {path}: HTTP {r.status_code}")
                wait = float(r.headers.get("retry-after") or 2 ** attempt)
                self._sleep(min(wait, 60))
                continue
            if r.status_code >= 400:
                raise GitHubError(r.status_code, f"{method} {path}: HTTP {r.status_code} "
                                                 f"{r.text[:300]}")
            return r
        raise GitHubError(0, "unreachable")

    def _paginate(self, path: str, limit: int = 3000, params: dict | None = None) -> list[dict]:
        out: list[dict] = []
        page = 1
        while len(out) < limit:
            batch = self._request("GET", path,
                                  params={**(params or {}), "per_page": 100, "page": page}).json()
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    def pull_request(self, repo: str, number: int) -> dict:
        return self._request("GET", f"/repos/{repo}/pulls/{number}").json()

    def pull_files(self, repo: str, number: int) -> list[dict]:
        return self._paginate(f"/repos/{repo}/pulls/{number}/files")

    def review_comments(self, repo: str, number: int) -> list[dict]:
        return self._paginate(f"/repos/{repo}/pulls/{number}/comments")

    def reviews(self, repo: str, number: int) -> list[dict]:
        return self._paginate(f"/repos/{repo}/pulls/{number}/reviews")

    def installation_repos(self, limit: int = 1000) -> list[dict]:
        """Repositories this installation can see (the response wraps the list)."""
        out: list[dict] = []
        page = 1
        while len(out) < limit:
            body = self._request("GET", "/installation/repositories",
                                 params={"per_page": 100, "page": page}).json()
            batch = body.get("repositories") or []
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    def open_pulls(self, repo: str) -> list[dict]:
        return self._paginate(f"/repos/{repo}/pulls", params={"state": "open"})

    def create_review(self, repo: str, number: int, commit_id: str, body: str,
                      comments: list[dict]) -> dict:
        return self._request("POST", f"/repos/{repo}/pulls/{number}/reviews", json={
            "commit_id": commit_id, "event": "COMMENT", "body": body,
            "comments": comments}).json()

    def issue_comment(self, repo: str, number: int, body: str) -> dict:
        return self._request("POST", f"/repos/{repo}/issues/{number}/comments",
                             json={"body": body}).json()
