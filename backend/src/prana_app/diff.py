"""Split a pull request's file patches into reviewable hunks.

GitHub's "list pull request files" returns each file's `patch`: unified diff
hunks without the `---`/`+++` header. An inline review comment must name a
line that appears in the diff on the side it comments on; we comment on the
new file (`side: RIGHT`), so the valid targets are the hunk's added and
context lines, numbered in the new file. `Hunk.right_lines` is that set and
`anchor` maps whatever line the model gave onto it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

LANGUAGE_BY_EXT = {
    ".py": "python", ".go": "go", ".rs": "rust",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
}
SKIP_PARTS = frozenset({"node_modules", "vendor", "third_party", "dist", "build", "__generated__"})
SKIP_SUFFIXES = (".min.js", ".d.ts", ".pb.go", "_pb2.py", ".generated.ts")


@dataclass
class Hunk:
    path: str
    language: str
    header: str
    text: str                                   # the hunk as the model sees it, header included
    added: list[int] = field(default_factory=list)       # new-file line numbers of + lines
    context: list[int] = field(default_factory=list)     # new-file line numbers of context lines

    @property
    def right_lines(self) -> set[int]:
        return set(self.added) | set(self.context)

    def anchor(self, line: int | None) -> tuple[int, bool]:
        """(line to comment on, whether the model's line was usable as is).
        A line outside the hunk is moved to the nearest added line."""
        if isinstance(line, int) and line in self.right_lines:
            return line, True
        targets = self.added or sorted(self.right_lines)
        if isinstance(line, int):
            return min(targets, key=lambda t: (abs(t - line), t)), False
        return targets[-1], False


def language_for(path: str) -> str | None:
    p = PurePosixPath(path)
    if any(part in SKIP_PARTS for part in p.parts) or path.endswith(SKIP_SUFFIXES):
        return None
    return LANGUAGE_BY_EXT.get(p.suffix)


def parse_patch(path: str, patch: str, language: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    new_line = 0
    lines: list[str] = []
    for raw in (patch or "").splitlines():
        m = _HEADER.match(raw)
        if m:
            if cur is not None:
                cur.text = "\n".join(lines)
                hunks.append(cur)
            cur = Hunk(path, language, raw, "")
            lines = [raw]
            new_line = int(m.group(3))
            continue
        if cur is None:
            continue
        lines.append(raw)
        if raw.startswith("\\"):                # "\ No newline at end of file"
            continue
        tag = raw[:1]
        if tag == "+":
            cur.added.append(new_line)
            new_line += 1
        elif tag == "-":
            pass
        else:                                   # context (" " or an empty line)
            cur.context.append(new_line)
            new_line += 1
    if cur is not None:
        cur.text = "\n".join(lines)
        hunks.append(cur)
    return hunks


def reviewable_hunks(files: list[dict], max_hunks: int) -> tuple[list[Hunk], int]:
    """(hunks to review, total reviewable). Only supported languages, not
    removed or generated files, only hunks that add lines; the largest
    changes first, ties in file order, so the budget goes where most changed."""
    pool: list[tuple[int, int, Hunk]] = []
    order = 0
    for f in files:
        if f.get("status") == "removed" or not f.get("patch"):
            continue
        lang = language_for(f.get("filename", ""))
        if lang is None:
            continue
        for h in parse_patch(f["filename"], f["patch"], lang):
            if h.added:
                pool.append((-len(h.added), order, h))
                order += 1
    pool.sort(key=lambda t: (t[0], t[1]))
    return [h for _, _, h in pool[:max_hunks]], len(pool)
