"""The retrieval contract the backend calls once per hunk - the design
document's Module 3 (RAG Context Layer) behind one function.

    from prana.rag.contract import retrieve
    context = retrieve(repo, base_sha, file_path, diff_hunk, language,
                       pr_number, repo_dir,
                       history=file_history,       # past reviews + commits (optional)
                       index="cache/index/o__r")   # ChromaDB index (optional)

Sources, in the order they claim the budget (the design doc ranks
past_reviews > similar_code > git_history; the two code sources and the
conventions slot in around them):

| source        | design doc            | needs                         |
|---------------|-----------------------|-------------------------------|
| enclosing     | (repo context)        | `repo_dir` at `base_sha`      |
| past_reviews  | 3.2 last 3 reviews    | `history.reviews`             |
| definitions   | 3.4 cross-file deps   | `repo_dir` at `base_sha`      |
| similar_code  | 3.1-3.2 ChromaDB top-5| `index` (prana.rag.embed)     |
| conventions   | 3.4 convention rules  | `index` (built with it)       |
| git_history   | 3.2 last 10 commits   | `history.commits`             |

Guarantees (tests/test_rag_contract.py):

- **Never raises.** Any error, timeout, missing input or unsupported language
  gives "" (or the sources that did succeed) and one JSON log line on the
  `prana.rag` logger.
- **Bounded time** (`timeout_s`, default 2.5 s) and **size** (`budget_chars`,
  default 2,000 characters, under the doc's 1,200-token context cap), whole
  sections and items only, plain text.
- **No diff hunk, nothing from this PR.** Code is read at `base_sha`, which
  never contains the PR's change; reviews from `pr_number` are dropped here
  even if the caller passes them.
- `sources` switches sources off one by one - that is how ablations B and C
  are run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from prana.rag.code import EXTENSIONS, Section, assemble, code_sections, parse_hunk
from prana.rag.history import FileHistory, commit_items, review_items

DEFAULT_TIMEOUT_S = 2.5
DEFAULT_BUDGET_CHARS = 2000
CONTRACT_VERSION = "2"
SIMILAR_K = 5
SIMILAR_SNIPPET_LINES = 8
SIMILAR_SNIPPET_CHARS = 360

ORDER = ("enclosing", "past_reviews", "definitions", "similar_code", "conventions",
         "git_history")
ALL_SOURCES = frozenset(ORDER)

REQUIREMENTS = {
    "contract_version": CONTRACT_VERSION,
    "checkout": "repo_dir: the repository at base_sha (git fetch --depth 1 origin "
                "<base_sha>; a sparse checkout of the hunk file's directory and its "
                "parent is enough for 'enclosing' and 'definitions')",
    "history": "optional FileHistory: commits from GET /repos/{o}/{r}/commits"
               "?path=<file>&sha=<base_sha>&per_page=10; past review comments on the "
               "file from earlier PRs (bots removed)",
    "index": "optional, per repository: python -m prana.rag.embed build --repo-dir "
             "<full checkout> --out <dir>; ChromaDB + all-MiniLM-L6-v2 on CPU, about "
             "65 chunks/s on 12 threads, capped at 4,000 chunks (~1 min); rebuild when "
             "the default branch moves; call prana.rag.embed.warm() at worker start",
    "runtime": "CPU only; standard library without an index, requirements-rag.txt with one",
    "latency": f"bounded by timeout_s (default {DEFAULT_TIMEOUT_S} s)",
    "output": f"plain text, at most {DEFAULT_BUDGET_CHARS} characters, or ''",
    "languages": sorted(EXTENSIONS),
}

log = logging.getLogger("prana.rag")
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prana-rag")
_INDEXES: dict[str, object] = {}
_INDEX_LOCK = threading.Lock()


def _log(event: str, **fields) -> None:
    log.warning(json.dumps({"event": event, **fields}, default=str))


def _open_index(index):
    """A CodeIndex for a path (cached per process), or the object itself."""
    if index is None or not isinstance(index, (str, Path)):
        return index
    key = str(Path(index).resolve())
    with _INDEX_LOCK:
        if key not in _INDEXES:
            if not (Path(key) / "chroma").is_dir():
                _INDEXES[key] = None
            else:
                from prana.rag.embed import CodeIndex
                _INDEXES[key] = CodeIndex(key)
        return _INDEXES[key]


def _similar(index, diff_hunk: str, language: str, file_path: str) -> Section | None:
    hunk = parse_hunk(diff_hunk)
    if hunk is None:
        return None
    query = "\n".join(hunk.added or hunk.old_lines).strip()
    if not query:
        return None
    items = []
    for h in index.query(query, language, k=SIMILAR_K, exclude_path=file_path):
        snippet = "\n".join("    " + ln.rstrip()
                            for ln in h.text.splitlines()[:SIMILAR_SNIPPET_LINES])
        items.append(f"- {h.path}, lines {h.start}-{h.end}:\n{snippet[:SIMILAR_SNIPPET_CHARS]}")
    return ("## Similar code elsewhere in this repository", items) if items else None


def build_sections(repo: str, base_sha: str, file_path: str, diff_hunk: str,
                   language: str, pr_number, repo_dir, history: FileHistory | None,
                   index, sources: frozenset[str], before: str | None) -> dict[str, Section]:
    """Every requested source that produced something. May raise; the caller
    (`retrieve`) contains it."""
    language = (language or "").strip().lower()
    if language not in EXTENSIONS:
        return {}
    out: dict[str, Section] = {}

    def code() -> None:
        for k, v in code_sections(repo_dir, file_path, diff_hunk, language).items():
            if k in sources:
                out[k] = v

    def hist() -> None:
        if "past_reviews" in sources:
            items = review_items(history, pr_number, before)
            if items:
                out["past_reviews"] = (f"## Past reviews on {file_path}", items)
        if "git_history" in sources:
            items = commit_items(history)
            if items:
                out["git_history"] = (f"## Recent commits to {file_path}", items)

    def indexed() -> None:
        idx = _open_index(index)
        if idx is None:
            return
        if "similar_code" in sources:
            sec = _similar(idx, diff_hunk, language, file_path)
            if sec:
                out["similar_code"] = sec
        rules = (idx.rules.get(language) or []) if "conventions" in sources else []
        if rules:
            out["conventions"] = ("## Conventions in this repository (measured)",
                                  [f"- {r}" for r in rules])

    # One failing source (say, the vector store) must not take the others down.
    for name, run, wanted in (
            ("code", code, repo_dir and sources & {"enclosing", "definitions"}),
            ("history", hist, history is not None),
            ("index", indexed, index is not None and sources & {"similar_code", "conventions"})):
        if not wanted:
            continue
        try:
            run()
        except Exception as e:  # noqa: BLE001 - logged, the other sources continue
            _log("source_error", source=name, error=f"{type(e).__name__}: {e}"[:300],
                 repo=repo, pr=pr_number, file=file_path)
    return out


def retrieve(repo: str, base_sha: str, file_path: str, diff_hunk: str,
             language: str, pr_number: int | None, repo_dir: str | None,
             *, history: FileHistory | None = None, index=None,
             sources: frozenset[str] = ALL_SOURCES, before: str | None = None,
             timeout_s: float = DEFAULT_TIMEOUT_S,
             budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """Context for one hunk, as plain text. "" when there is nothing useful.

    `before` (ISO time) drops past reviews at or after it; the backend passes
    the PR's creation time."""
    where = {"repo": repo, "pr": pr_number, "sha": (base_sha or "")[:12], "file": file_path}
    if not isinstance(diff_hunk, str) or not isinstance(file_path, str) or not diff_hunk:
        if repo_dir or history or index:
            _log("bad_input", **where)
        return ""
    if not (repo_dir or history or index):
        return ""                           # nothing to retrieve from: normal, silent
    started = time.perf_counter()
    try:
        fut = _POOL.submit(build_sections, repo, base_sha, file_path, diff_hunk, language,
                           pr_number, repo_dir, history, index, frozenset(sources), before)
        secs = fut.result(timeout=timeout_s)
    except FutureTimeout:
        _log("timeout", timeout_s=timeout_s, **where)
        return ""
    except Exception as e:  # noqa: BLE001 - the contract is "never raises"
        _log("error", error=f"{type(e).__name__}: {e}"[:300], **where)
        return ""
    out = assemble([secs[k] for k in ORDER if k in secs], budget_chars)
    if len(out) > budget_chars:             # defence in depth; assemble() already caps
        out = out[:budget_chars].rsplit("\n", 1)[0]
    elapsed = time.perf_counter() - started
    if elapsed > timeout_s * 0.8:
        _log("slow", seconds=round(elapsed, 3), **where)
    return out


if __name__ == "__main__":
    print(json.dumps(REQUIREMENTS, indent=2))
