"""A small BM25 index over code hunks. Pure Python, no dependencies.

Chosen over embeddings for v1.1 deliberately: the design document's risk
register lists BM25 as the fallback if embedding quality or infrastructure is
a problem, and on an aarch64 box with a GPU that is busy training, a
dependency-free retriever that indexes 12k hunks in a second is the one that
ships this week. The interface is the same one an embedding retriever would
expose, so swapping it later is a one-module change.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_STOP = frozenset({
    "the", "and", "for", "int", "str", "var", "let", "const", "return", "self", "this",
    "nil", "null", "none", "true", "false", "func", "def", "fn", "pub", "use", "import",
    "from", "new", "err", "error", "ctx", "val", "value", "data", "get", "set",
})


def tokenize(text: str) -> list[str]:
    """Identifier tokens, camelCase and snake_case split, lowercased, short and
    stop tokens dropped. Diff markers and punctuation vanish, so a reformat of
    the same code tokenises the same way."""
    out: list[str] = []
    for ident in _IDENT.findall(text):
        for part in _CAMEL.sub("_", ident).split("_"):
            p = part.lower()
            if len(p) >= 3 and p not in _STOP:
                out.append(p)
    return out


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(toks))
        c = Counter(toks)
        self.tf.append(c)
        self.df.update(c.keys())
        self.avgdl = sum(self.doc_len) / len(self.doc_len)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, text: str, top_k: int = 5,
              allow: Iterable[int] | None = None) -> list[tuple[str, float]]:
        """Top-k (doc_id, score). `allow` restricts scoring to those document
        positions - the caller uses it for same-repo retrieval and for leaving
        a row's own pull request out."""
        q = Counter(tokenize(text))
        if not q or not self.doc_ids:
            return []
        candidates = range(len(self.doc_ids)) if allow is None else allow
        scores: dict[int, float] = defaultdict(float)
        for term in q:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i in candidates:
                f = self.tf[i].get(term)
                if not f:
                    continue
                dl = self.doc_len[i]
                scores[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                )
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.doc_ids[i], s) for i, s in ranked]
