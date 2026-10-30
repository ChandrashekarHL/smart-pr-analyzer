"""JSONL read/write that survives real source code.

`str.splitlines()` breaks on U+2028, U+2029, U+000B, U+000C and U+0085. Those
characters occur in real diffs - minified JavaScript and Unicode test fixtures
both carry them - and `json.dumps(..., ensure_ascii=False)` writes them out
raw. A reader built on `splitlines()` then tears a record in half and fails
with an unterminated-string error that points at a column, not a cause.

So: writers escape the two separators JSON itself permits raw, and readers
split on "\\n" and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

# Valid inside a JSON string but treated as line breaks by str.splitlines()
# and by several JS parsers. Escaping them keeps the file one-record-per-line
# for every consumer.
_LINE_SEPARATORS = {" ": "\\u2028", " ": "\\u2029"}


def dumps(record: dict, *, sort_keys: bool = True) -> str:
    line = json.dumps(record, ensure_ascii=False, sort_keys=sort_keys)
    for raw, escaped in _LINE_SEPARATORS.items():
        if raw in line:
            line = line.replace(raw, escaped)
    return line


def write(path: str | Path, records: Iterable[dict]) -> str:
    payload = "".join(dumps(r) + "\n" for r in records)
    Path(path).write_text(payload, encoding="utf-8", newline="\n")
    return payload


def iter_read(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8", newline="\n") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def read(path: str | Path) -> list[dict]:
    return list(iter_read(path))
