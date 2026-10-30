"""Licence policy for the published corpus.

The scrape covers 656 repositories under mixed terms. 28.5% of the filtered
rows come from copyleft, unclassified or unlicensed repositories. Training
privately on all of it is one question; publishing a derived dataset to the
Hugging Face Hub is a different one, and the safe answer there is permissive
only - the policy The Stack and similar public code corpora settled on.

`other` is GitHub's label for a licence its classifier did not recognise. It is
not a permission grant, so it is treated as unknown rather than as permitted.
"""

from __future__ import annotations

import json
from pathlib import Path

# SPDX-ish keys as returned by the GitHub licence API.
PERMISSIVE: frozenset[str] = frozenset({
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "isc", "unlicense", "cc0-1.0", "0bsd", "zlib", "postgresql", "ncsa",
})

# File-level copyleft. Including a review *comment* about such a file is a far
# weaker act than redistributing the file, but we keep them separable so the
# decision is explicit rather than accidental.
WEAK_COPYLEFT: frozenset[str] = frozenset({"mpl-2.0", "epl-2.0", "lgpl-3.0", "lgpl-2.1"})

STRONG_COPYLEFT: frozenset[str] = frozenset({
    "gpl-2.0", "gpl-3.0", "agpl-3.0", "cc-by-sa-4.0",
})

ATTRIBUTION_REQUIRED: frozenset[str] = frozenset({"cc-by-4.0"})


def tier(key: str | None) -> str:
    """One of: permissive, weak-copyleft, strong-copyleft, attribution, unknown."""
    if not key:
        return "unknown"
    k = key.strip().lower()
    if k in PERMISSIVE:
        return "permissive"
    if k in WEAK_COPYLEFT:
        return "weak-copyleft"
    if k in STRONG_COPYLEFT:
        return "strong-copyleft"
    if k in ATTRIBUTION_REQUIRED:
        return "attribution"
    return "unknown"


def publishable(key: str | None) -> bool:
    return tier(key) == "permissive"


class LicenseIndex:
    """repo_name -> licence key, from the scraper's cache."""

    def __init__(self, mapping: dict[str, str | None]):
        self._map = mapping

    @classmethod
    def load(cls, path: str | Path) -> LicenseIndex:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping: dict[str, str | None] = {}
        for repo, value in raw.items():
            if isinstance(value, str):
                mapping[repo] = value
            elif isinstance(value, dict):
                mapping[repo] = value.get("key") or value.get("spdx_id")
            else:
                mapping[repo] = None
        return cls(mapping)

    def key(self, repo: str) -> str | None:
        return self._map.get(repo)

    def tier(self, repo: str) -> str:
        return tier(self.key(repo))

    def publishable(self, repo: str) -> bool:
        return publishable(self.key(repo))

    def __contains__(self, repo: str) -> bool:
        return repo in self._map
