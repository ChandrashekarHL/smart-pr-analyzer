"""Client for the prana review API, standard library only.

    from prana.client import Client
    c = Client("http://100.124.121.97:8100", api_key="...")
    c.review(diff_hunk, "python", context="...")["parsed"]

Evaluate a retrieval method end to end against the API:

    python -m prana.client eval --url http://100.124.121.97:8100 --key-file api.key \\
        --data data/test_ood.jsonl --context rag/test_ood.jsonl --out runs/try1-ctx
    python -m prana.client eval ... --out runs/try1-none            # no --context
    python -m prana.eval.structured --no-bertscore \\
        --predictions runs/try1-ctx/test_ood.predictions.jsonl

The eval writes predictions in exactly the format `prana.eval.structured`
scores, appends as it goes, and resumes from what is already on disk.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from prana.data import jsonl


class APIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Client:
    def __init__(self, url: str, api_key: str, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self._key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.url + path, data=data, method=method,
            headers={"Content-Type": "application/json", "X-API-Key": self._key},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise APIError(e.code, f"HTTP {e.code} on {path}: {detail}") from e

    def health(self) -> dict:
        return self._request("GET", "/health")

    def review(self, diff_hunk: str, language: str, context: str | None = None,
               file_path: str | None = None, retries: int = 3) -> dict:
        body = {"diff_hunk": diff_hunk, "language": language, "context": context,
                "file_path": file_path}
        delay = 2.0
        for attempt in range(1, retries + 1):
            try:
                return self._request("POST", "/review", body)
            except (urllib.error.URLError, TimeoutError, APIError) as e:
                # 4xx is the caller's fault; retrying will not help.
                if isinstance(e, APIError) and 400 <= e.status < 500:
                    raise
                if attempt == retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise APIError(0, "unreachable")


def load_context(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {r["id"]: r.get("rag_context") or "" for r in jsonl.iter_read(path) if r.get("id")}


def eval_split(client: Client, data: Path, out: Path, context: Path | None,
               limit: int | None = None, log=print) -> dict:
    rows = jsonl.read(data)[:limit] if limit else jsonl.read(data)
    ctx = load_context(context)
    out.mkdir(parents=True, exist_ok=True)
    preds = out / f"{data.stem}.predictions.jsonl"
    done = {r["id"] for r in jsonl.iter_read(preds)} if preds.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    log(f"{data.name}: {len(rows)} rows, {len(done)} done, {len(todo)} to go, "
        f"context: {'yes (' + str(len(ctx)) + ' ids)' if ctx else 'none'}")
    started = time.time()
    with open(preds, "a", encoding="utf-8", newline="\n") as sink:
        for n, r in enumerate(todo, 1):
            res = client.review(r["diff_hunk"], r["language"], ctx.get(r["id"]) or None,
                                r.get("file_path"))
            sink.write(jsonl.dumps({
                "id": r["id"], "language": r["language"], "repo_name": r.get("repo_name"),
                "reference": r.get("review_comment"),
                "reference_structured": r.get("structured_output"),
                "prediction": res["text"],
            }) + "\n")
            sink.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / max(time.time() - started, 1e-6)
                log(f"  {n}/{len(todo)}  {rate:.2f} rows/s  "
                    f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min")
    (out / "run.json").write_text(json.dumps({
        "api": client.url, "data": str(data), "context": str(context) if context else None,
    }, indent=2), encoding="utf-8")
    return {"rows": len(rows), "generated": len(todo), "predictions": str(preds)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("health", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://100.124.121.97:8100")
        p.add_argument("--key-file", default="api.key")
    e = sub.choices["eval"]
    e.add_argument("--data", required=True, help="a split, e.g. data/test_ood.jsonl")
    e.add_argument("--context", default=None, help="sidecar jsonl with id + rag_context")
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    key = Path(args.key_file).read_text(encoding="utf-8").strip()
    client = Client(args.url, key)
    if args.cmd == "health":
        print(json.dumps(client.health(), indent=2))
        return
    stats = eval_split(client, Path(args.data), Path(args.out),
                       Path(args.context) if args.context else None, args.limit,
                       log=lambda *a: print(*a, flush=True))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
