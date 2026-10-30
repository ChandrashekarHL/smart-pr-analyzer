"""Settings from the environment. Secrets come from files (`*_FILE`), never
from the environment's values or the command line, so they do not show up
in `ps`, `docker inspect` or crash reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(RuntimeError):
    pass


def _secret(name: str, required: bool = True) -> str:
    path = os.environ.get(f"{name}_FILE")
    if not path:
        if required:
            raise ConfigError(f"{name}_FILE is not set")
        return ""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ConfigError(f"{name}_FILE={path} cannot be read: {e}") from e
    if required and not value:
        raise ConfigError(f"{name}_FILE={path} is empty")
    return value


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer") from e


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as e:
        raise ConfigError(f"{name} must be a number") from e


@dataclass(frozen=True)
class Settings:
    github_app_id: str
    github_private_key: str = field(repr=False)
    webhook_secret: str = field(repr=False)
    model_api_url: str
    model_api_key: str = field(repr=False)
    database_url: str
    redis_url: str
    # Review policy
    max_hunks: int = 8              # ~3.5 s per hunk keeps a PR under the 30 s budget
    time_budget_s: float = 25.0     # stop calling the model after this; say so in the summary
    min_confidence: float = 0.0     # below: listed in the summary, not posted inline
    rag_enabled: bool = False       # switched on only when the test_ood result wins
    bot_login: str = "prana-reviewer[bot]"
    github_api: str = "https://api.github.com"

    @classmethod
    def from_env(cls) -> Settings:
        app_id = os.environ.get("GITHUB_APP_ID", "").strip()
        if not app_id.isdigit():
            raise ConfigError("GITHUB_APP_ID must be the numeric App ID")
        return cls(
            github_app_id=app_id,
            github_private_key=_secret("GITHUB_PRIVATE_KEY"),
            webhook_secret=_secret("GITHUB_WEBHOOK_SECRET"),
            model_api_url=os.environ.get("MODEL_API_URL", "http://100.124.121.97:8100"),
            model_api_key=_secret("MODEL_API_KEY"),
            # The URL carries the database password, so production passes it as a file.
            database_url=(_secret("DATABASE_URL", required=False)
                          or os.environ.get("DATABASE_URL", "sqlite:///prana_app.db")),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            max_hunks=_int("MAX_HUNKS", 8),
            time_budget_s=_float("TIME_BUDGET_S", 25.0),
            min_confidence=_float("MIN_CONFIDENCE", 0.0),
            rag_enabled=os.environ.get("RAG_ENABLED", "0") == "1",
            bot_login=os.environ.get("BOT_LOGIN", "prana-reviewer[bot]"),
            github_api=os.environ.get("GITHUB_API", "https://api.github.com").rstrip("/"),
        )
