"""Process-wide services built once from Settings: GitHub App auth, the model
client, the store, the review policy and (optionally) retrieval."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

from prana_app.config import Settings
from prana_app.diff import Hunk
from prana_app.github import GitHubApp
from prana_app.pipeline import Policy, RetrieveFn
from prana_app.store import Store

log = logging.getLogger("prana_app")
REPO_CACHE = Path("/var/cache/prana/repos")


@dataclass
class Services:
    settings: Settings
    app: GitHubApp
    store: Store
    policy: Policy
    model: object
    retrieve: RetrieveFn | None


def model_fn(settings: Settings):
    from prana.client import Client  # stdlib-only client shipped with the model
    client = Client(settings.model_api_url, settings.model_api_key, timeout=120)

    def call(hunk: str, language: str, context: str | None, path: str | None) -> dict:
        return client.review(hunk, language, context=context, file_path=path, retries=2)
    return call


def retrieve_fn(settings: Settings) -> RetrieveFn | None:
    """Code retrieval through the contract (design doc Module 3). Off unless
    RAG_ENABLED=1: on test_ood the v1.1 model did not benefit from it."""
    if not settings.rag_enabled:
        return None
    from prana.rag.checkout import checkout, dest_for
    from prana.rag.contract import retrieve

    def run(h: Hunk, repo: str, pr: int, base_sha: str) -> str:
        dest = dest_for(REPO_CACHE, repo, base_sha)
        checkout(repo, base_sha, [h.path], dest)
        return retrieve(repo, base_sha, h.path, h.text, h.language, pr, str(dest),
                        sources=frozenset({"enclosing", "definitions"}))
    return run


@functools.lru_cache(maxsize=1)
def services() -> Services:
    s = Settings.from_env()
    store = Store(s.database_url)
    store.create_all()
    return Services(
        settings=s,
        app=GitHubApp(s.github_app_id, s.github_private_key, s.github_api),
        store=store,
        policy=Policy(s.max_hunks, s.time_budget_s, s.min_confidence, s.bot_login),
        model=model_fn(s),
        retrieve=retrieve_fn(s),
    )
