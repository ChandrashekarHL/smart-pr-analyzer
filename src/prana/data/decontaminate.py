"""N-gram decontamination between the training set and the held-out sets.

Splitting by pull request stops the same PR appearing on both sides, but it
does not stop the same *code* appearing on both sides. Cherry-picks, reverts,
vendored dependencies, generated files and forks all put near-identical hunks
in unrelated pull requests. When that happens the held-out score measures
recall of memorised text and reads several points too high.

The standard remedy is n-gram overlap detection, and 13-gram is the size the
GPT-3 paper and most work since has used. Train n-grams go into a bloom filter
rather than a set: 20M 13-grams in a Python set costs most of a gigabyte,
while the same population at a 1% false-positive rate fits in 24MB. A false
positive here only means we discard one clean evaluation row, which is the
harmless direction to be wrong in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from prana.data import jsonl

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Identifier-level tokens, lowercased. Punctuation and indentation are
    dropped so that a reformat does not defeat the match."""
    return _TOKEN.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Iterator[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


class BloomFilter:
    """Fixed-memory approximate set. No false negatives."""

    __slots__ = ("bits", "size", "hashes", "count")

    def __init__(self, expected: int, error_rate: float = 0.01):
        expected = max(expected, 1)
        size = int(-expected * math.log(error_rate) / (math.log(2) ** 2))
        self.size = max(size, 1024)
        self.hashes = max(1, round((self.size / expected) * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)
        self.count = 0

    def _positions(self, item: str) -> Iterator[int]:
        digest = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1
        for i in range(self.hashes):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        for pos in self._positions(item):
            self.bits[pos >> 3] |= 1 << (pos & 7)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[pos >> 3] & (1 << (pos & 7)) for pos in self._positions(item))


def build_index(rows: Iterable[dict], n: int, field: str = "diff_hunk") -> BloomFilter:
    rows = list(rows)
    est = max(sum(max(len(tokenize(r[field])) - n + 1, 0) for r in rows), 1)
    bloom = BloomFilter(est)
    for row in rows:
        for gram in ngrams(tokenize(row[field]), n):
            bloom.add(gram)
    return bloom


def contamination(row: dict, bloom: BloomFilter, n: int, field: str = "diff_hunk") -> float:
    """Fraction of this row's n-grams already present in the training index."""
    grams = list(ngrams(tokenize(row[field]), n))
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in bloom) / len(grams)


def clean_split(
    rows: list[dict], bloom: BloomFilter, n: int, threshold: float
) -> tuple[list[dict], list[dict]]:
    """Partition a held-out split into (clean, contaminated)."""
    clean, dirty = [], []
    for row in rows:
        score = contamination(row, bloom, n)
        if score > threshold:
            dirty.append({**row, "contamination": round(score, 4)})
        else:
            clean.append(row)
    return clean, dirty


def _read(path: Path) -> list[dict]:
    return jsonl.read(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_v3", help="directory holding the splits")
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="drop a held-out row when this fraction of its n-grams "
                         "already appears in training")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the held-out splits with contaminated rows removed")
    args = ap.parse_args()

    root = Path(args.data)
    train = _read(root / "train.jsonl")
    bloom = build_index(train, args.ngram)
    print(f"indexed {bloom.count:,} {args.ngram}-grams from {len(train):,} train rows "
          f"({len(bloom.bits) / 1e6:.1f} MB)")

    summary = {}
    for name in ("validation", "test", "test_ood"):
        path = root / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = _read(path)
        clean, dirty = clean_split(rows, bloom, args.ngram, args.threshold)
        pct = 100 * len(dirty) / max(len(rows), 1)
        flag = "  <-- investigate" if pct > 5 else ""
        print(f"  {name:<12}{len(rows):>7} rows  "
              f"contaminated {len(dirty):>5} ({pct:5.2f}%){flag}")
        summary[name] = {"rows": len(rows), "contaminated": len(dirty), "pct": round(pct, 3)}

        if args.apply and dirty:
            jsonl.write(path, clean)
            jsonl.write(root / f"{name}.contaminated.jsonl", dirty)

    (root / "decontamination.json").write_text(
        json.dumps({"ngram": args.ngram, "threshold": args.threshold, "splits": summary},
                   indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {root / 'decontamination.json'}"
          + ("  (splits rewritten)" if args.apply else "  (dry run; pass --apply to remove)"))


if __name__ == "__main__":
    main()
