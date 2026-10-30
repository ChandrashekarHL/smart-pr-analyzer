"""Data access for the dashboard: one authenticated call to the backend's
`/api/dashboard`. Standard library only, so it is trivially testable."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

SEVERITY_ORDER = ["critical", "warning", "info"]
TARGET_P95_S = 30.0          # design doc NFR: <= 30 s per PR


class DashboardError(RuntimeError):
    pass


def settings() -> tuple[str, str, str]:
    """(api url, token, W&B report url) from the environment."""
    url = os.environ.get("DASHBOARD_API_URL", "http://api:8000").rstrip("/")
    token_file = os.environ.get("DASHBOARD_TOKEN_FILE", "")
    token = Path(token_file).read_text(encoding="utf-8").strip() if token_file else ""
    return url, token, os.environ.get("WANDB_REPORT_URL", "")


def fetch(api_url: str, token: str, days: int, timeout: float = 10.0) -> dict:
    if not api_url.startswith(("http://", "https://")):
        raise DashboardError(f"DASHBOARD_API_URL must be http(s), got {api_url!r}")
    url = f"{api_url}/api/dashboard?days={int(days)}"
    auth = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, headers=auth)  # noqa: S310 - scheme checked above
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:     # noqa: S310 - scheme checked above
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        hint = " (check DASHBOARD_TOKEN_FILE)" if e.code in (401, 403) else ""
        raise DashboardError(f"backend answered HTTP {e.code}{hint}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DashboardError(f"backend unreachable at {api_url}: {e}") from e
    except ValueError as e:
        raise DashboardError("backend returned something that is not JSON") from e


def severity_rows(d: dict) -> list[dict]:
    counts = d.get("comments_by_severity") or {}
    return [{"severity": s, "comments": counts.get(s, 0)} for s in SEVERITY_ORDER]


def latency_rows(d: dict) -> list[dict]:
    """Long form for the chart: one row per (day, measure)."""
    rows = []
    for x in d.get("daily") or []:
        for key, label in (("p50_s", "p50"), ("p95_s", "p95")):
            if x.get(key) is not None:
                rows.append({"day": x["day"], "measure": label, "seconds": x[key]})
    return rows


def p95_delta(d: dict) -> str | None:
    p95 = (d.get("latency") or {}).get("p95_s")
    if p95 is None:
        return None
    return f"{p95 - TARGET_P95_S:+.1f} s vs 30 s target"
