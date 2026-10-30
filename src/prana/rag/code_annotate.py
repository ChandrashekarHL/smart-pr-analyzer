"""Write retrieval sidecars for a split through the retrieval contract.

    python -m prana.rag.code_annotate --data data/v1.1/test_ood.jsonl \\
        --shas data/rag-code/shas.jsonl --root cache/repos --out data/rag-code \\
        [--index-root cache/index] [--commits data/rag-code/commits_test_ood.jsonl] \\
        [--reviews data/v1.1 --created-at data/rag-code/created_at.jsonl] \\
        [--sources enclosing,definitions] [--name code]

Calls `prana.rag.contract.retrieve` exactly as the backend will, one row at a
time, and writes `<out>/<split>[.<name>].jsonl` with {id, rag_context} - the
shape `prana.client eval --context` reads - plus `.stats.json` with coverage
per source and latency.

Past reviews for evaluation come from the corpus (`--reviews`, every split,
the human comment text) dated with the scrape's `created_at`: only other PRs,
only before the reviewed PR's first comment (`history.ReviewStore`).
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from prana.data import jsonl
from prana.rag.checkout import MARKER, dest_for, load_shas
from prana.rag.contract import ALL_SOURCES, retrieve
from prana.rag.history import Commit, FileHistory, ReviewStore

HEADINGS = {"enclosing": "## Enclosing code", "past_reviews": "## Past reviews",
            "definitions": "## Definitions used", "similar_code": "## Similar code",
            "conventions": "## Conventions", "git_history": "## Recent commits"}


def load_review_store(corpus_dir: Path, created_at: Path) -> ReviewStore:
    when = {r["id"]: r.get("created_at") for r in jsonl.iter_read(created_at)}
    store = ReviewStore()
    for split in sorted(corpus_dir.glob("*.jsonl")):
        for r in jsonl.iter_read(split):
            store.add(r["repo_name"], r.get("file_path") or "", r.get("pr_number"),
                      when.get(r["id"]) or "", r.get("review_comment") or "")
    return store.finalize()


def load_commits(path: Path) -> dict[tuple[str, str, str], list[Commit]]:
    out = {}
    for r in jsonl.iter_read(path):
        if "commits" in r:
            out[(r["repo"], r["sha"], r["path"])] = [
                Commit(c.get("sha", ""), c.get("date", ""), c.get("message", ""))
                for c in r["commits"]]
    return out


def annotate(rows: list[dict], shas: dict[tuple[str, str], str], root: Path | None,
             index_root: Path | None = None, store: ReviewStore | None = None,
             commits: dict | None = None, sources: frozenset[str] = ALL_SOURCES,
             ) -> tuple[list[dict], dict]:
    out, lat = [], []
    why: collections.Counter = collections.Counter()
    per_source: collections.Counter = collections.Counter()
    for r in rows:
        repo, pr, fp = r["repo_name"], r.get("pr_number"), r.get("file_path") or ""
        sha = shas.get((repo, str(pr)))
        repo_dir = None
        if sha and root is not None:
            d = dest_for(root, repo, sha)
            repo_dir = str(d) if (d / MARKER).exists() else None
        hist = None
        if store is not None or commits is not None:
            hist = FileHistory(
                reviews=store.reviews_for(repo, fp, pr) if store is not None else [],
                commits=(commits or {}).get((repo, sha, fp), []) if sha else [])
        index = None
        if index_root is not None:
            d = index_root / repo.replace("/", "__")
            index = str(d) if (d / "meta.json").exists() else None
        before = store.pr_start(repo, pr) if store is not None else None
        t = time.perf_counter()
        ctx = retrieve(repo, sha or "", fp, r["diff_hunk"], r["language"], pr, repo_dir,
                       history=hist, index=index, sources=sources, before=before)
        lat.append(time.perf_counter() - t)
        if not sha:
            why["no_base_sha"] += 1
        elif not ctx:
            why["nothing_found"] += 1
        else:
            why["context"] += 1
            for k, h in HEADINGS.items():
                per_source[k] += h in ctx
        out.append({"id": r["id"], "rag_context": ctx})
    lat.sort()
    n = len(rows)
    sizes = sorted(len(o["rag_context"]) for o in out if o["rag_context"])
    stats = {
        "rows": n, "sources_requested": sorted(sources), "outcome": dict(why),
        "coverage": round(why["context"] / max(n, 1), 4),
        "rows_per_source": dict(per_source),
        "chars_median": sizes[len(sizes) // 2] if sizes else 0,
        "chars_max": sizes[-1] if sizes else 0,
        "latency_ms_p50": round(lat[n // 2] * 1e3, 2) if lat else None,
        "latency_ms_p99": round(lat[min(n - 1, int(n * .99))] * 1e3, 2) if lat else None,
        "latency_ms_max": round(lat[-1] * 1e3, 2) if lat else None,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shas", required=True)
    ap.add_argument("--root", help="per-PR base checkouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-root")
    ap.add_argument("--commits")
    ap.add_argument("--reviews", help="corpus directory of split jsonl files")
    ap.add_argument("--created-at")
    ap.add_argument("--sources", default=",".join(sorted(ALL_SOURCES)))
    ap.add_argument("--name", default="", help="sidecar suffix, e.g. full")
    args = ap.parse_args()

    sources = frozenset(s.strip() for s in args.sources.split(",") if s.strip())
    unknown = sources - ALL_SOURCES
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}")
    if bool(args.reviews) != bool(args.created_at):
        raise SystemExit("--reviews and --created-at go together")
    store = load_review_store(Path(args.reviews), Path(args.created_at)) if args.reviews else None
    commits = load_commits(Path(args.commits)) if args.commits else None

    data = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = annotate(list(jsonl.iter_read(data)), load_shas(Path(args.shas)),
                           Path(args.root) if args.root else None,
                           Path(args.index_root) if args.index_root else None,
                           store, commits, sources)
    stem = data.stem + (f".{args.name}" if args.name else "")
    jsonl.write(out_dir / f"{stem}.jsonl", rows)
    (out_dir / f"{stem}.stats.json").write_bytes((json.dumps(stats, indent=2) + "\n").encode())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
