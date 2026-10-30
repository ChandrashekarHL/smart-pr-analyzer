"""Build one embedding index per repository of a split, for evaluation.

    python -m prana.rag.index_repos --data data/v1.1/test_ood.jsonl \\
        --shas data/rag-code/shas.jsonl --checkouts cache/repos \\
        --full cache/full --out cache/index

Which commit: for each repository, the base commit with the **earliest
commit time** among its PRs in the split. Every PR's own base is at or after
it, and a PR's change lands only after its base, so the index cannot contain
the change under review or a later fix of it. The price is staleness for
later PRs, which makes the evaluation conservative.

Needs requirements-rag.txt (run it with the RAG environment's Python).
Idempotent per repository (`meta.json` present = done); writes
`<out>/index_repos.jsonl` with one record per repository.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from prana.data import jsonl
from prana.rag.checkout import (MARKER, checkout_source, commit_time, dest_for, load_shas)
from prana.rag.code import EXTENSIONS


def index_dir(root: Path, repo: str) -> Path:
    return root / repo.replace("/", "__")


def earliest_base(repo: str, shas: set[str], checkouts: Path) -> str | None:
    timed = []
    for sha in shas:
        d = dest_for(checkouts, repo, sha)
        t = commit_time(d) if (d / MARKER).exists() else None
        if t is not None:
            timed.append((t, sha))
    return min(timed)[1] if timed else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shas", required=True)
    ap.add_argument("--checkouts", required=True, help="per-PR base checkouts (prana.rag.checkout)")
    ap.add_argument("--full", required=True, help="where source checkouts go")
    ap.add_argument("--out", required=True, help="where indexes go")
    ap.add_argument("--max-chunks", type=int, default=2000)
    args = ap.parse_args()

    from prana.rag import embed            # optional dependencies

    shas = load_shas(Path(args.shas))
    by_repo: dict[str, dict] = collections.defaultdict(lambda: {"shas": set(), "files": [],
                                                                  "langs": set()})
    for r in jsonl.iter_read(Path(args.data)):
        sha = shas.get((r["repo_name"], str(r["pr_number"])))
        if sha:
            g = by_repo[r["repo_name"]]
            g["shas"].add(sha)
            g["files"].append(r.get("file_path") or "")
            g["langs"].add(r["language"])
    out, full, checkouts = Path(args.out), Path(args.full), Path(args.checkouts)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "index_repos.jsonl"
    embed.warm()
    print(f"{len(by_repo)} repositories", flush=True)
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for n, (repo, g) in enumerate(sorted(by_repo.items()), 1):
            idx = index_dir(out, repo)
            if (idx / "meta.json").exists():
                continue
            rec: dict = {"repo": repo}
            try:
                sha = earliest_base(repo, g["shas"], checkouts)
                if sha is None:
                    raise RuntimeError("no base checkout with a commit time")
                exts = [e for lang in sorted(g["langs"]) for e in EXTENSIONS[lang]]
                src = dest_for(full, repo, sha)
                rec["checkout"] = checkout_source(repo, sha, exts, src)
                rec["index"] = embed.build(src, idx, repo=repo, sha=sha,
                                           focus=sorted(set(g["files"])),
                                           max_chunks=args.max_chunks)
                rec["ok"] = True
            except Exception as e:  # noqa: BLE001 - recorded; the next repo continues
                rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            meta = rec.get("index") or {}
            print(f"  {n}/{len(by_repo)} {repo}: "
                  + (f"{meta.get('chunks')} chunks in {meta.get('seconds')} s" if rec["ok"]
                     else rec["error"]), flush=True)


if __name__ == "__main__":
    main()
