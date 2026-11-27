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

# Offline evaluation of the model versions, shown when no W&B report is
# embedded. Measured numbers, copied from MODEL_CARD.md: test_ood = 1,190
# reviews from 234 repositories never used in training; benchmarks are
# planted bug/correct pairs judged by an LLM at temperature 0.
MODEL_RESULTS = [
    {"metric": "Valid JSON output (%)", "v1.0": 99.5, "v1.1": 99.5, "v1.3": 99.5},
    {"metric": "Exact line (%)", "v1.0": 0.7, "v1.1": 14.7, "v1.3": 36.8},
    {"metric": "Severity accuracy (%)", "v1.0": 50.6, "v1.1": 47.0, "v1.3": 44.8},
    {"metric": "Issue-type accuracy (%)", "v1.0": 35.4, "v1.1": 34.5, "v1.3": 31.7},
    {"metric": "Explanation BERTScore", "v1.0": 0.854, "v1.1": 0.856, "v1.3": 0.856},
    {"metric": "Explanation ROUGE-L", "v1.0": 0.178, "v1.1": 0.169, "v1.3": 0.169},
]
DETECTION_RESULTS = [
    {"benchmark": "Planted bugs found", "v1.1": 25.0, "v1.3": 32.5},
    {"benchmark": "False alarms on correct code", "v1.1": 40.0, "v1.3": 27.5},
    {"benchmark": "Security bugs: model", "v1.1": 24.0, "v1.3": 33.0},
    {"benchmark": "Security bugs: model + Semgrep", "v1.1": 58.0, "v1.3": 64.0},
]
DEPLOYED_MODEL = "v1.3 (DeepSeek-Coder 6.7B + QLoRA, TRL SFTTrainer, 3 epochs)"


def detection_rows() -> list[dict]:
    """DETECTION_RESULTS in long form for a grouped bar chart."""
    return [{"benchmark": r["benchmark"], "version": v, "percent": r[v]}
            for r in DETECTION_RESULTS for v in ("v1.1", "v1.3")]


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
