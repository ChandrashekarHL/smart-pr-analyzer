"""Teacher labelling: turn free-text human reviews into the 7-field schema.

    python -m prana.data.label --in data/train_v2/train.jsonl \\
        --out data/train_v3/train.jsonl --limit 200        # pilot, then hand-check

The corpus has no severity, issue_type, explanation or suggestion fields -
GitHub stores none of them. A model trained to emit them needs labels, and
22,417 rows is beyond hand-labelling, so a larger model writes them under one
rule that matters more than any other: **restructure what the reviewer said,
never add a finding the reviewer did not raise.** The teacher is a formatter,
not a second reviewer. Every label keeps the human's judgement; the teacher
only names its severity and category and splits it into explanation and fix.

Design points:
- Plain HTTPS via urllib. No SDK, so it runs in both environments as-is.
- Gemini's `responseSchema` enforces the enums and field set server-side; a
  parse step validates again client-side and rejects anything off-schema.
- Several rows per request, each tagged with its id, and the reply must echo
  exactly those ids. A batch that fails validation is retried row by row.
- Resumable: output is appended per row and existing ids are skipped.
- The key never touches argv, logs or stdout. It comes from a 600-mode file
  or the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prana.data import jsonl
from prana.schema import ISSUE_TYPES, SEVERITIES

# Pinned, not the "-latest" alias: an alias can change mid-run and the
# dataset card needs to name the exact teacher. gemini-2.5-flash was
# retired for new keys on 2026-10-19; this is what the server pointed at.
DEFAULT_MODEL = "gemini-3.8-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_KEY_FILES = (
    Path.home() / ".config" / "prana" / "gemini.key",
    Path(".secrets") / "gemini.key",
)
MAX_DIFF_CHARS = 2500  # the comment is the signal; the diff is context

SYSTEM_INSTRUCTION = f"""You convert code-review comments written by human engineers into a
fixed JSON schema.

You are a formatter, not a reviewer. For each item you receive a diff hunk and the
human reviewer's comment on it. Produce one JSON object per item:

- "id": copy the item's id exactly.
- "is_review": true if the comment raises a finding, request or concern about the
  code; false if it is only a question, an acknowledgement, a discussion reply or
  praise with no actionable content.
- "severity": one of {list(SEVERITIES)}.
    critical = must be fixed before merge: a bug, security hole, data loss, crash,
               wrong result, broken contract.
    warning  = should be fixed: correctness risk, missing test, performance problem,
               misleading behaviour, error handling gap.
    info     = optional: style, naming, documentation, readability, a suggestion
               the reviewer frames as a preference or a question.
- "issue_type": one of {list(ISSUE_TYPES)}. Pick the single best fit for what the
  reviewer actually raised.
- "explanation": the reviewer's finding restated in one to three plain sentences.
  Keep their technical content and their claims. Do NOT add issues, causes or
  consequences the reviewer did not state. Do NOT soften or strengthen them.
- "suggestion": the fix the reviewer proposed, in one or two sentences, or an empty
  string if they proposed none. Never invent a fix.

Rules: use only what is in the comment; the diff is context for understanding
references, not a source of new findings. Keep identifiers in backticks as the
reviewer wrote them. Write in English. Return a JSON array with exactly one object
per item, in the order given."""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "is_review": {"type": "BOOLEAN"},
            "severity": {"type": "STRING", "enum": list(SEVERITIES)},
            "issue_type": {"type": "STRING", "enum": list(ISSUE_TYPES)},
            "explanation": {"type": "STRING"},
            "suggestion": {"type": "STRING"},
        },
        "required": ["id", "is_review", "severity", "issue_type", "explanation", "suggestion"],
        "propertyOrdering": ["id", "is_review", "severity", "issue_type",
                             "explanation", "suggestion"],
    },
}


# ------------------------------------------------------------------ prompt

def build_prompt(rows: list[dict]) -> str:
    parts = [f"{len(rows)} item(s). Return a JSON array with one object per item.\n"]
    for i, r in enumerate(rows, 1):
        diff = r["diff_hunk"]
        if len(diff) > MAX_DIFF_CHARS:
            diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated]"
        parts.append(
            f"### Item {i}\n"
            f"id: {r['id']}\n"
            f"language: {r.get('language', 'unknown')}\n"
            f"file: {r.get('file_path') or 'unknown'}\n"
            f"diff:\n```\n{diff}\n```\n"
            f"reviewer comment:\n\"\"\"\n{r['review_comment'].strip()}\n\"\"\"\n"
        )
    return "\n".join(parts)


# ------------------------------------------------------------------ parse

class LabelError(ValueError):
    pass


def parse_response(text: str, expected_ids: list[str]) -> dict[str, dict]:
    """Validate the teacher's reply against the schema and the request.

    Server-side schema enforcement is not trusted on its own: a model can still
    return fewer items, duplicate an id, or answer for an id it was not asked
    about, and any of those silently mislabels a row.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LabelError(f"not JSON: {e}") from e
    if not isinstance(data, list):
        raise LabelError("top level is not a list")

    out: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            raise LabelError("item is not an object")
        rid = item.get("id")
        if rid not in expected_ids:
            raise LabelError(f"unexpected id {rid!r}")
        if rid in out:
            raise LabelError(f"duplicate id {rid!r}")
        sev, typ = item.get("severity"), item.get("issue_type")
        if sev not in SEVERITIES:
            raise LabelError(f"bad severity {sev!r} for {rid}")
        if typ not in ISSUE_TYPES:
            raise LabelError(f"bad issue_type {typ!r} for {rid}")
        if not isinstance(item.get("is_review"), bool):
            raise LabelError(f"is_review not boolean for {rid}")
        expl = (item.get("explanation") or "").strip()
        if not expl:
            raise LabelError(f"empty explanation for {rid}")
        out[rid] = {
            "is_review": item["is_review"],
            "severity": sev,
            "issue_type": typ,
            "explanation": expl,
            "suggestion": (item.get("suggestion") or "").strip(),
        }
    missing = [i for i in expected_ids if i not in out]
    if missing:
        raise LabelError(f"missing ids {missing}")
    return out


def coerce_line_number(value) -> int | None:
    """The scrape stored line numbers as strings, and "None" for absent."""
    if value is None:
        return None
    s = str(value).strip()
    return int(s) if s.isdigit() else None


def structured_output(row: dict, label: dict) -> dict:
    """The exact object the model is trained to emit, in schema order."""
    return {
        "file_path": row.get("file_path"),
        "line_number": coerce_line_number(row.get("line_number")),
        "severity": label["severity"],
        "issue_type": label["issue_type"],
        "explanation": label["explanation"],
        "suggestion": label["suggestion"],
    }


# ------------------------------------------------------------------ client

def load_key(key_file: str | None = None) -> str:
    """Key from an explicit file, then the environment, then default files.
    Never from argv - it would show in `ps` and in run-job logs."""
    candidates = [Path(key_file)] if key_file else []
    env = os.environ.get("GEMINI_API_KEY")
    if env:
        return env.strip()
    candidates += list(DEFAULT_KEY_FILES)
    for p in candidates:
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise SystemExit(
        "no Gemini key: set GEMINI_API_KEY or put the key in "
        + " or ".join(str(p) for p in DEFAULT_KEY_FILES)
    )


class GeminiClient:
    def __init__(self, key: str, model: str = DEFAULT_MODEL, timeout: float = 120.0):
        self._key = key
        self.model = model
        self.timeout = timeout
        self.requests = 0
        self.retries = 0
        self.model_version: str | None = None  # as reported by the server

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> list[str]:
        req = urllib.request.Request(f"{API_ROOT}/models",
                                     headers={"x-goog-api-key": self._key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return [m["name"].split("/", 1)[-1] for m in payload.get("models", [])]

    def generate(self, user_text: str, max_attempts: int = 6) -> str:
        """One structured-output call with backoff on rate limits and 5xx."""
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                self.requests += 1
                payload = self._post(url, body)
                self.model_version = payload.get("modelVersion") or self.model_version
                cands = payload.get("candidates") or []
                if not cands:
                    feedback = payload.get("promptFeedback", {})
                    reason = feedback.get("blockReason", "no candidates")
                    raise LabelError(f"empty response: {reason}")
                parts = cands[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
                if not text.strip():
                    raise LabelError("empty text in candidate")
                return text
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 404:
                    # A 404 here is not always "no such model": the first pilot
                    # got one for a model that the list endpoint reported as
                    # present. Show the server's own message before guessing.
                    listed = ", ".join(self.list_models()[:24])
                    raise SystemExit(
                        f"HTTP 404 from {self.model!r}: {detail}\n"
                        f"models this key can list: {listed}"
                    ) from e
                if e.code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                    self.retries += 1
                    time.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, 60)
                    continue
                raise LabelError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < max_attempts:
                    self.retries += 1
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise LabelError(f"network: {e}") from e
        raise LabelError("exhausted attempts")


# ------------------------------------------------------------------ driver

def label_batch(client: GeminiClient, rows: list[dict]) -> dict[str, dict]:
    """Label a batch; on a bad reply, retry one row at a time so a single
    malformed item cannot sink its neighbours."""
    ids = [r["id"] for r in rows]
    try:
        return parse_response(client.generate(build_prompt(rows)), ids)
    except LabelError as first:
        if len(rows) == 1:
            raise
        out: dict[str, dict] = {}
        errors = [str(first)]
        for r in rows:
            try:
                out.update(parse_response(client.generate(build_prompt([r])), [r["id"]]))
            except LabelError as e:
                errors.append(f"{r['id']}: {e}")
        if not out:
            raise LabelError("; ".join(errors)) from first
        return out


def chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def label_file(in_path: Path, out_path: Path, client: GeminiClient, *,
               rows_per_request: int = 8, workers: int = 4,
               limit: int | None = None, log=print) -> dict:
    rows = jsonl.read(in_path)
    if limit:
        rows = rows[:limit]
    done = {r["id"] for r in jsonl.iter_read(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    log(f"{in_path.name}: {len(rows)} rows, {len(done)} already labelled, {len(todo)} to do")
    if not todo:
        return {"labelled": 0, "failed": 0, "requests": 0}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stats = {"labelled": 0, "failed": 0, "not_review": 0}
    started = time.time()

    def work(batch: list[dict]) -> list[dict]:
        try:
            labels = label_batch(client, batch)
        except LabelError as e:
            return [{**r, "label": None, "label_error": str(e)[:200]} for r in batch]
        result = []
        for r in batch:
            lab = labels.get(r["id"])
            if lab is None:
                result.append({**r, "label": None, "label_error": "missing from reply"})
            else:
                result.append({**r, "label": lab,
                               "label_model": getattr(client, "model_version", None)
                               or getattr(client, "model", "unknown"),
                               "structured_output": structured_output(r, lab)})
        return result

    with open(out_path, "a", encoding="utf-8", newline="\n") as sink, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, b) for b in chunks(todo, rows_per_request)]
        for n, fut in enumerate(as_completed(futures), 1):
            labelled_rows = fut.result()
            with lock:
                for r in labelled_rows:
                    sink.write(jsonl.dumps(r) + "\n")
                    if r.get("label") is None:
                        stats["failed"] += 1
                    else:
                        stats["labelled"] += 1
                        if not r["label"]["is_review"]:
                            stats["not_review"] += 1
                sink.flush()
            if n % 10 == 0 or n == len(futures):
                done_rows = stats["labelled"] + stats["failed"]
                rate = done_rows / max(time.time() - started, 1e-6)
                eta = (len(todo) - done_rows) / max(rate, 1e-6) / 60
                log(f"  {done_rows}/{len(todo)}  {rate:.2f} rows/s  eta {eta:.0f} min  "
                    f"failed {stats['failed']}  not_review {stats['not_review']}  "
                    f"retries {client.retries}")
    stats["requests"] = client.requests
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rows-per-request", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel requests; lower it if you see many retries")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N rows")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt for the first batch and exit; no network")
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    if args.dry_run:
        rows = jsonl.read(inp)[: args.rows_per_request]
        print(SYSTEM_INSTRUCTION, "\n\n=== USER ===\n", build_prompt(rows))
        return

    client = GeminiClient(load_key(args.key_file), model=args.model)

    # Under run-job stdout is a file, so plain print() is block-buffered and
    # the progress lines only appear when the process exits - useless for a
    # three-hour job. Flush each one.
    def log(*parts):
        print(*parts, flush=True)

    stats = label_file(inp, out, client, rows_per_request=args.rows_per_request,
                       workers=args.workers, limit=args.limit, log=log)
    print(json.dumps(stats))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
