"""Check out exactly what `prana.rag.contract.retrieve` reads: the hunk files'
directories and their parents, at one commit. Used by the evaluation here and
by the backend before it calls `retrieve`.

    python -m prana.rag.checkout --data data/v1.1/test_ood.jsonl \\
        --shas data/rag-code/shas.jsonl --root cache/repos

One checkout per (repository, base commit):

    git init; git sparse-checkout set --no-cone <dirs>
    git fetch --depth 1 --filter=blob:none origin <sha>     # trees only
    git checkout FETCH_HEAD                                 # blobs for <dirs> only

Shallow, blob-less and sparse, so even a very large repository costs one
commit's trees plus the few files read. A finished checkout is marked with
`.prana_ok` and never fetched again.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from prana.data import jsonl

GIT_TIMEOUT_S = 300
MARKER = ".prana_ok"


def sparse_patterns(file_paths: list[str]) -> list[str]:
    """Non-cone patterns selecting the direct files (not subdirectories) of
    each file's directory and of its parent."""
    dirs: set[str] = set()
    for fp in file_paths:
        d = PurePosixPath(fp).parent
        for x in (d, d.parent):
            dirs.add("" if str(x) == "." else str(x))
    out = []
    for d in sorted(dirs):
        prefix = f"/{d}" if d else ""
        out += [f"{prefix}/*", f"!{prefix}/*/"]
    return out


def dest_for(root: Path, repo: str, sha: str) -> Path:
    return root / repo.replace("/", "__") / sha[:12]


def _git(args: list[str], cwd: Path | None = None) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                       timeout=GIT_TIMEOUT_S, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"git {args[0]}: {(p.stderr or p.stdout).strip()[:300]}")


def checkout(repo: str, sha: str, file_paths: list[str], dest: Path,
             url: str | None = None) -> dict:
    """Sparse, shallow, blob-less checkout of `repo` at `sha` into `dest`.
    Idempotent: returns the recorded result if `dest` is already complete."""
    marker = dest / MARKER
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if dest.exists():                      # a previous attempt died half-way
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    started = time.time()
    url = url or f"https://github.com/{repo}.git"
    _git(["init", "-q"], dest)
    _git(["remote", "add", "origin", url], dest)
    _git(["config", "remote.origin.promisor", "true"], dest)
    _git(["config", "remote.origin.partialclonefilter", "blob:none"], dest)
    _git(["config", "core.sparseCheckout", "true"], dest)
    (dest / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (dest / ".git" / "info" / "sparse-checkout").write_bytes(
        ("\n".join(sparse_patterns(file_paths)) + "\n").encode())
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha], dest)
    _git(["checkout", "-q", "FETCH_HEAD"], dest)
    files = [p for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts]
    result = {"repo": repo, "sha": sha, "files": len(files),
              "bytes": sum(p.stat().st_size for p in files),
              "seconds": round(time.time() - started, 1)}
    marker.write_bytes(json.dumps(result).encode())
    return result


VENDORED = ("node_modules", "vendor", "third_party", "dist", "build", "target",
            "coverage", ".venv", "venv", "bower_components")


def source_patterns(extensions: list[str]) -> list[str]:
    """Non-cone patterns selecting source files of the given extensions
    anywhere in the tree, minus vendored and build directories."""
    pats = [f"*{e}" for e in sorted(set(extensions))]
    # A directory negation ("!node_modules/") does not override a file
    # pattern in sparse checkout; the paths themselves must be negated.
    pats += [f"!**/{d}/**" for d in VENDORED] + ["!*.min.js"]
    return pats


def checkout_source(repo: str, sha: str, extensions: list[str], dest: Path,
                    url: str | None = None) -> dict:
    """Shallow, blob-less checkout of every source file of the given
    extensions at `sha` (for the embedding index). Idempotent like `checkout`."""
    marker = dest / MARKER
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    started = time.time()
    _git(["init", "-q"], dest)
    _git(["remote", "add", "origin", url or f"https://github.com/{repo}.git"], dest)
    _git(["config", "remote.origin.promisor", "true"], dest)
    _git(["config", "remote.origin.partialclonefilter", "blob:none"], dest)
    _git(["config", "core.sparseCheckout", "true"], dest)
    (dest / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (dest / ".git" / "info" / "sparse-checkout").write_bytes(
        ("\n".join(source_patterns(extensions)) + "\n").encode())
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha], dest)
    _git(["checkout", "-q", "FETCH_HEAD"], dest)
    files = [p for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts]
    result = {"repo": repo, "sha": sha, "files": len(files),
              "bytes": sum(p.stat().st_size for p in files),
              "seconds": round(time.time() - started, 1)}
    marker.write_bytes(json.dumps(result).encode())
    return result


def commit_time(checkout_dir: Path) -> int | None:
    """Committer time of HEAD in a checkout, in Unix seconds (timezone-safe
    to compare), or None."""
    try:
        p = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=checkout_dir,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0 or not p.stdout.strip().isdigit():
        return None
    return int(p.stdout.strip())


def plan(rows: list[dict], shas: dict[tuple[str, str], str]) -> dict[tuple[str, str], list[str]]:
    """(repo, base_sha) -> file paths needed, for rows whose PR has a base sha."""
    need: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for r in rows:
        sha = shas.get((r["repo_name"], str(r.get("pr_number"))))
        if sha and r.get("file_path"):
            need[(r["repo_name"], sha)].append(r["file_path"])
    return dict(need)


def load_shas(path: Path) -> dict[tuple[str, str], str]:
    return {(r["repo"], str(r["pr_number"])): r["base_sha"]
            for r in jsonl.iter_read(path) if r.get("base_sha")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shas", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    need = plan(list(jsonl.iter_read(Path(args.data))), load_shas(Path(args.shas)))
    print(f"{len(need)} checkouts to ensure", flush=True)
    ok = failed = 0
    log_path = root / "checkouts.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool, \
            open(log_path, "a", encoding="utf-8", newline="\n") as log:
        futs = {pool.submit(checkout, repo, sha, files, dest_for(root, repo, sha)): (repo, sha)
                for (repo, sha), files in need.items()}
        for n, f in enumerate(as_completed(futs), 1):
            repo, sha = futs[f]
            try:
                rec = {**f.result(), "ok": True}
                ok += 1
            except Exception as e:  # noqa: BLE001 - recorded, the run continues
                rec = {"repo": repo, "sha": sha, "ok": False, "error": str(e)[:300]}
                failed += 1
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if n % 25 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)}  ok {ok}  failed {failed}", flush=True)
    print(json.dumps({"checkouts": len(need), "ok": ok, "failed": failed}))


if __name__ == "__main__":
    main()
