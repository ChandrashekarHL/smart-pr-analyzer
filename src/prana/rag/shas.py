"""Look up each pull request's base and head commit on GitHub.

    python -m prana.rag.shas --data data/v1.1/test_ood.jsonl --out data/rag-code/shas.jsonl

The scrape kept repo, PR number and comment, but not the commit, and code
retrieval needs the code as it was before the PR. Uses the `gh` CLI, so its
own login supplies the token (5,000 requests/hour); this module never sees
the token. Appends as it goes and resumes from what is already on disk.

`base_sha` is the base branch commit GitHub records for the PR (for merged
PRs, the base at the last update). It can be newer than the base a reviewer
saw; `prana.rag.code.locate` therefore finds the hunk by content, not only
by line number.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from prana.data import jsonl

JQ = ('{base_sha: .base.sha, head_sha: .head.sha, base_ref: .base.ref, '
      'state: .state, merged_at: .merged_at}')


def lookup(repo: str, pr: int | str, gh: str = "gh", retries: int = 3) -> dict:
    """{repo, pr_number, base_sha, ...} or {repo, pr_number, error}."""
    err = ""
    for attempt in range(retries):
        try:
            p = subprocess.run([gh, "api", f"repos/{repo}/pulls/{pr}", "--jq", JQ],
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            err = "gh timed out after 60 s"
            time.sleep(2 * (attempt + 1))
            continue
        if p.returncode == 0:
            return {"repo": repo, "pr_number": str(pr), **json.loads(p.stdout)}
        err = (p.stderr or "").strip()[:200]
        if "404" in err or "Not Found" in err:
            break
        if "rate limit" in err.lower():
            time.sleep(60)
        else:
            time.sleep(2 * (attempt + 1))
    return {"repo": repo, "pr_number": str(pr), "error": err}


def prs_in(paths: list[Path]) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for p in paths:
        for r in jsonl.iter_read(p):
            seen[(r["repo_name"], str(r.get("pr_number")))] = None
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="split jsonl files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["repo"], r["pr_number"]) for r in jsonl.iter_read(out)} if out.exists() else set()
    todo = [k for k in prs_in([Path(p) for p in args.data]) if k not in done]
    print(f"{len(done)} done, {len(todo)} to look up", flush=True)
    with open(out, "a", encoding="utf-8", newline="\n") as sink:
        for n, (repo, pr) in enumerate(todo, 1):
            sink.write(json.dumps(lookup(repo, pr)) + "\n")
            sink.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    rows = list(jsonl.iter_read(out))
    print(json.dumps({"prs": len(rows), "with_base_sha": sum(bool(r.get("base_sha")) for r in rows)}))


if __name__ == "__main__":
    main()
