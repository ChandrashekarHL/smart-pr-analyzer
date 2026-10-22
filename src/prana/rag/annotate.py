"""Write `rag_context` sidecars for every split.

    python -m prana.rag.annotate --corpus data/v1.1 --out data/rag

Reads the finalized splits, builds the retriever from **train only**, and
writes `<out>/<split>.jsonl` with {id, rag_context, sources} - the sidecar
shape `prana.data.finalize --context-dir` consumes. The retriever index is
saved beside them so `prana.serve` can load the same memory at inference.

Coverage is printed per split. test_ood repositories are held out of training,
so their context comes only from the cross-repository source; that is the
fresh-install condition and it is reported as such, not hidden.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from prana.data import jsonl
from prana.rag.retriever import DEFAULT_BUDGET_CHARS, Retriever

SPLITS = ("train", "validation", "test", "test_ood")


def annotate_split(
    retriever: Retriever, rows: list[dict], budget: int
) -> tuple[list[dict], dict]:
    out = []
    kinds: collections.Counter = collections.Counter()
    with_ctx = 0
    for r in rows:
        got = retriever.retrieve(r["repo_name"], r.get("file_path") or "", r["diff_hunk"],
                                 exclude_pr=r.get("pr_number"), budget_chars=budget)
        if got.context:
            with_ctx += 1
            for s in got.sources:
                kinds[s["kind"]] += 1
        out.append({"id": r["id"], "rag_context": got.context, "sources": got.sources})
    return out, {"rows": len(rows), "with_context": with_ctx, "sources": dict(kinds)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/v1.1", help="finalized splits to read")
    ap.add_argument("--out", default="data/rag")
    ap.add_argument("--budget-chars", type=int, default=DEFAULT_BUDGET_CHARS)
    args = ap.parse_args()

    corpus, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = jsonl.read(corpus / "train.jsonl")
    t0 = time.time()
    retriever = Retriever.from_rows(train)
    retriever.save(out / "index.json")
    print(f"index: {len(retriever.docs):,} training hunks from "
          f"{len(retriever.by_repo):,} repos in {time.time() - t0:.1f}s")

    report = {}
    for split in SPLITS:
        path = corpus / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = jsonl.read(path)
        t0 = time.time()
        side, stats = annotate_split(retriever, rows, args.budget_chars)
        jsonl.write(out / f"{split}.jsonl", side)
        stats["seconds"] = round(time.time() - t0, 1)
        report[split] = stats
        pct = 100 * stats["with_context"] / max(stats["rows"], 1)
        print(f"{split:<12}{stats['rows']:>7} rows  {stats['with_context']:>7} with context "
              f"({pct:5.1f}%)  sources {stats['sources']}  {stats['seconds']}s")
    (out / "annotate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
