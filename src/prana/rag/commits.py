"""Last commits touching each reviewed file, as of the PR's base commit.

    python -m prana.rag.commits --rows rows.jsonl --shas shas.jsonl --out commits.jsonl

`rows.jsonl` needs repo_name, pr_number, file_path per row. One call per
(repo, base_sha, file):

    GET /repos/{o}/{r}/commits?path=<file>&sha=<base_sha>&per_page=10

through the `gh` CLI (its own login supplies the token). Listing from
`base_sha` means only commits that were already in the base branch when the
PR was reviewed - never the PR's own commits or anything later. This is the
same call the backend makes with the installation token. Appends and resumes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prana.data import jsonl
from prana.rag.checkout import load_shas

JQ = "[.[] | {sha: .sha, date: .commit.committer.date, message: .commit.message}]"


def file_commits(repo: str, sha: str, path: str, n: int = 10, gh: str = "gh") -> dict:
    q = urllib.parse.urlencode({"path": path, "sha": sha, "per_page": n})
    err = ""
    for attempt in range(3):
        try:
            p = subprocess.run([gh, "api", f"repos/{repo}/commits?{q}", "--jq", JQ],
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            err = "gh timed out after 60 s"
            time.sleep(2 * (attempt + 1))
            continue
        if p.returncode == 0:
            commits = [{**c, "message": (c.get("message") or "")[:500]}
                       for c in json.loads(p.stdout or "[]")]
            return {"repo": repo, "sha": sha, "path": path, "commits": commits}
        err = (p.stderr or "").strip()[:200]
        if "404" in err or "422" in err:
            break
        time.sleep(60 if "rate limit" in err.lower() else 2 * (attempt + 1))
    return {"repo": repo, "sha": sha, "path": path, "error": err}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--shas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    shas = load_shas(Path(args.shas))
    want = sorted({(r["repo_name"], shas[(r["repo_name"], str(r["pr_number"]))], r["file_path"])
                   for r in jsonl.iter_read(Path(args.rows))
                   if (r["repo_name"], str(r["pr_number"])) in shas and r.get("file_path")})
    out = Path(args.out)
    done = {(r["repo"], r["sha"], r["path"]) for r in jsonl.iter_read(out)} \
        if out.exists() else set()
    todo = [k for k in want if k not in done]
    print(f"{len(want)} files, {len(done)} done, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(args.workers) as pool, \
            open(out, "a", encoding="utf-8", newline="\n") as sink:
        for n, rec in enumerate(pool.map(lambda k: file_commits(*k), todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    rows = list(jsonl.iter_read(out))
    print(json.dumps({"files": len(rows), "errors": sum("error" in r for r in rows),
                      "with_commits": sum(bool(r.get("commits")) for r in rows)}))


if __name__ == "__main__":
    main()
