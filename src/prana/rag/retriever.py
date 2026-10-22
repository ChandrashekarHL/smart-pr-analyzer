"""Retrieve repository context for a diff hunk from the review corpus.

Three sources, in the priority the design document gave them:

1. Past reviews on the same file in the same repository - "we have said this
   before" is how senior reviewers actually work.
2. Similar hunks reviewed elsewhere in the same repository, with what the
   reviewer said about them.
3. Similar hunks reviewed in *other* repositories - the only memory available
   to a fresh install, and the only source that fires on held-out repos.

Leakage rules, enforced here and tested:
- the index is built from the training split only;
- a row's own pull request is never retrieved, so a training row cannot see
  its own answer and a test row cannot see a sibling comment from its PR.

Output is plain text with short headers, under a character budget that keeps
it near 800 tokens, exactly as the RAG brief specifies. `""` is a valid answer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from prana.rag.bm25 import BM25

# ~800 tokens of English-plus-code at ~3.5 chars per token.
DEFAULT_BUDGET_CHARS = 2800
MAX_PER_SECTION = 3
SNIPPET_LINES = 6


@dataclass
class Doc:
    id: str
    repo: str
    pr: str
    file_path: str
    hunk: str
    review: str          # the reviewer's explanation (deframed) or raw comment
    suggestion: str = ""


@dataclass
class Retrieved:
    context: str
    sources: list[dict] = field(default_factory=list)


def _snippet(hunk: str, lines: int = SNIPPET_LINES) -> str:
    body = [ln for ln in hunk.splitlines() if not ln.startswith("@@")]
    changed = [ln for ln in body if ln.startswith("+") or ln.startswith("-")] or body
    return "\n".join(changed[:lines])


class Retriever:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.by_id = {d.id: d for d in docs}
        self.by_file: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_repo: dict[str, list[int]] = defaultdict(list)
        self.index = BM25()
        for i, d in enumerate(docs):
            self.by_file[(d.repo, d.file_path)].append(i)
            self.by_repo[d.repo].append(i)
            self.index.add(d.id, d.hunk)

    # ------------------------------------------------------------ build

    @classmethod
    def from_rows(cls, rows: list[dict]) -> Retriever:
        """Rows from a finalized split (have `structured_output`) or a raw one
        (have `review_comment`). Only ever pass the training split."""
        docs = []
        for r in rows:
            so = r.get("structured_output") or {}
            review = so.get("explanation") or r.get("review_comment") or ""
            if not review.strip():
                continue
            docs.append(Doc(
                id=r["id"], repo=r["repo_name"], pr=str(r.get("pr_number", "")),
                file_path=r.get("file_path") or "", hunk=r["diff_hunk"],
                review=review.strip(), suggestion=(so.get("suggestion") or "").strip(),
            ))
        return cls(docs)

    def save(self, path: str | Path) -> None:
        payload = [d.__dict__ for d in self.docs]
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Retriever:
        return cls([Doc(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])

    # ------------------------------------------------------------ query

    def retrieve(self, repo: str, file_path: str, hunk: str, *,
                 exclude_pr: str | None = None,
                 budget_chars: int = DEFAULT_BUDGET_CHARS) -> Retrieved:
        exclude_pr = str(exclude_pr) if exclude_pr is not None else None
        used: set[str] = set()
        sections: list[tuple[str, list[str]]] = []
        sources: list[dict] = []

        def ok(d: Doc) -> bool:
            return d.id not in used and not (exclude_pr is not None
                                            and d.repo == repo and d.pr == exclude_pr)

        # 1. past reviews on this file
        lines = []
        for i in self.by_file.get((repo, file_path), []):
            d = self.docs[i]
            if not ok(d):
                continue
            lines.append(f"- {d.review}" + (f" Fix: {d.suggestion}" if d.suggestion else ""))
            used.add(d.id)
            sources.append({"kind": "same_file", "id": d.id})
            if len(lines) >= MAX_PER_SECTION:
                break
        if lines:
            sections.append(("Past reviews on this file", lines))

        # 2. similar hunks in this repo
        same_repo = [i for i in self.by_repo.get(repo, []) if ok(self.docs[i])]
        lines = []
        for doc_id, _ in self.index.query(hunk, top_k=MAX_PER_SECTION * 2, allow=same_repo):
            d = self.by_id[doc_id]
            if not ok(d):
                continue
            lines.append(f"- {d.file_path}: reviewed as \"{d.review}\"")
            used.add(d.id)
            sources.append({"kind": "same_repo", "id": d.id})
            if len(lines) >= MAX_PER_SECTION:
                break
        if lines:
            sections.append(("Similar code reviewed elsewhere in this repository", lines))

        # 3. similar hunks in other repos - the fresh-install case
        if len(used) < MAX_PER_SECTION:
            lines = []
            for doc_id, _ in self.index.query(hunk, top_k=MAX_PER_SECTION * 4):
                d = self.by_id[doc_id]
                if d.repo == repo or not ok(d):
                    continue
                lines.append(f"- {d.repo} {d.file_path}:\n  "
                             + _snippet(d.hunk).replace("\n", "\n  ")
                             + f"\n  reviewed as \"{d.review}\"")
                used.add(d.id)
                sources.append({"kind": "other_repo", "id": d.id})
                if len(lines) >= MAX_PER_SECTION:
                    break
            if lines:
                sections.append(("Similar code reviewed in other repositories", lines))

        # Assemble under budget, whole items only, in priority order.
        out: list[str] = []
        size = 0
        kept_sources = 0
        for title, items in sections:
            block = [f"## {title}"]
            for item in items:
                if size + len(item) + 2 > budget_chars:
                    break
                block.append(item)
                size += len(item) + 2
                kept_sources += 1
            if len(block) > 1:
                out.append("\n".join(block))
        return Retrieved("\n\n".join(out), sources[:kept_sources])
