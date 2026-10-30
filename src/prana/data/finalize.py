"""Turn teacher-labelled splits into the v1.0 training corpus.

    python -m prana.data.finalize --labelled data/train_v3 --out data/v1

Three things leave the corpus here, each counted in the dataset card:
- rows the teacher could not label (`label` is null);
- rows the teacher marked `is_review=false` - the pilot showed these are PR
  authors replying to reviewers, 43% of corpus v2, and a model trained on
  them learns to answer reviews instead of writing them;
- rows carrying AI-tool markup under a personal account (filters v4), which
  are findings in form but not human review.

The same filter is applied to every split, so test sets measure the model on
real reviews only. Every row keeps its original fields; `structured_output`
is the training target and is re-rendered through `prana.schema` so the
serialisation the model learns is the one the evaluator parses.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from prana.data import filters, jsonl
from prana.data.deframe import deframe, is_framed
from prana.schema import FIELDS, render_structured

SPLITS = ("train", "validation", "test", "test_ood")


def load_context(context_dir: Path | None, split: str) -> dict[str, str]:
    """`rag_context` by row id from a sidecar file the RAG lead produces:
    `<context_dir>/<split>.jsonl` with at least {"id", "rag_context"}."""
    if context_dir is None:
        return {}
    path = context_dir / f"{split}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for r in jsonl.iter_read(path):
        ctx = (r.get("rag_context") or "").strip()
        if r.get("id") and ctx:
            out[r["id"]] = ctx
    return out


def finalize_rows(
    rows: list[dict], *, deframe_labels: bool = True, context: dict[str, str] | None = None
) -> tuple[list[dict], collections.Counter]:
    stats: collections.Counter = collections.Counter()
    kept: list[dict] = []
    context = context or {}
    for r in rows:
        stats["read"] += 1
        label = r.get("label")
        if not label:
            stats["drop_unlabelled"] += 1
            continue
        if not label.get("is_review"):
            stats["drop_not_review"] += 1
            continue
        if filters.BOT_MARKUP.search(r.get("review_comment", "")):
            stats["drop_ai_markup"] += 1
            continue
        so = r.get("structured_output")
        if not so or any(k not in so for k in FIELDS):
            stats["drop_bad_structured"] += 1
            continue
        so = {k: so.get(k) for k in FIELDS}
        if deframe_labels:
            for field in ("explanation", "suggestion"):
                if is_framed(so[field] or ""):
                    stats[f"deframed_{field}"] += 1
                    so[field] = deframe(so[field])
        row = {**r, "structured_output": so, "target_text": render_structured(so)}
        if r["id"] in context:
            row["rag_context"] = context[r["id"]]
            stats["with_context"] += 1
        kept.append(row)
        stats["kept"] += 1
    return kept, stats


def distribution(rows: list[dict], key: str) -> collections.Counter:
    return collections.Counter(r["structured_output"][key] for r in rows)


def write_card(
    outdir: Path,
    per_split: dict[str, dict],
    teachers: collections.Counter,
    train_rows: list[dict],
) -> None:
    lines = [
        "# Dataset card - structured review corpus",
        "",
        f"- filters version: `{filters.FILTERS_VERSION}`",
        "- teacher model(s): "
        + ", ".join(f"`{m}` ({n:,} rows)" for m, n in teachers.most_common()),
        "- training target: `target_text` = "
        "`prana.schema.render_structured(structured_output)`",
        "",
        "## Provenance",
        "",
        "Human review comments from public GitHub pull requests (corpus v2), restructured",
        "into the fixed schema by the teacher model under the instruction to restate the",
        "reviewer's finding and never add one. `severity` and `issue_type` are therefore",
        "teacher judgements about human text, not human labels; a 200-row pilot was",
        "hand-checked before the full run.",
        "",
        "## Splits",
        "",
        "| split | read | kept | yield | not_review | unlabelled | ai_markup | sha256 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, info in per_split.items():
        s = info["stats"]
        pct = 100 * s["kept"] / max(s["read"], 1)
        lines.append(
            f"| {name} | {s['read']:,} | {s['kept']:,} | {pct:.1f}% "
            f"| {s['drop_not_review']:,} | {s['drop_unlabelled']:,} | {s['drop_ai_markup']:,} "
            f"| `{info['sha']}` |"
        )
    for key in ("severity", "issue_type"):
        dist = distribution(train_rows, key)
        total = sum(dist.values()) or 1
        lines += [
            "",
            f"## Train {key} distribution",
            "",
            f"| {key} | rows | share |",
            "|---|---|---|",
        ]
        for k, v in dist.most_common():
            lines.append(f"| {k} | {v:,} | {100 * v / total:.1f}% |")
        top = dist.most_common(1)[0][1] / total if dist else 0
        lines.append(
            f"\nMajority-class baseline for {key} accuracy: **{100 * top:.1f}%**. "
            "Any reported accuracy must be read against this."
        )
    with_line = sum(1 for r in train_rows if r["structured_output"]["line_number"] is not None)
    with_ctx = sum(1 for r in train_rows if r.get("rag_context"))
    deframed = sum(v["stats"].get("deframed_explanation", 0) for v in per_split.values())
    lines += [
        "",
        "## Processing",
        "",
        f"- field order in `target_text`: {', '.join(FIELDS)} "
        "(reasoning before classification)",
        f"- third-person teacher framing removed from {deframed:,} explanations "
        "(`prana.data.deframe`)",
        f"- rows with `rag_context`: {with_ctx:,} of {len(train_rows):,} train rows",
        "",
        "## Known limitations",
        "",
        f"- `line_number` is present on {100 * with_line / max(len(train_rows), 1):.0f}% "
        "of train rows (the scrape lost it on the rest); the model learns to emit "
        "`null` for the remainder.",
        "- Rows without `rag_context` render the context slot as `(none)`; training "
        "applies context dropout to the rest so the model handles both.",
        "- Labels inherit the teacher's biases; disagreement with a human on severity "
        "is expected and should be measured on the human-annotated subset.",
        "",
    ]
    (outdir / "dataset-card.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labelled", default="data/train_v3")
    ap.add_argument("--out", default="data/v1")
    ap.add_argument("--context-dir", default=None,
                    help="directory of <split>.jsonl sidecars with id + rag_context")
    ap.add_argument("--keep-framing", action="store_true",
                    help="leave the teacher's third-person openings in place (v1.0 behaviour)")
    args = ap.parse_args()
    context_dir = Path(args.context_dir) if args.context_dir else None

    src, outdir = Path(args.labelled), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, dict] = {}
    teachers: collections.Counter = collections.Counter()
    train_rows: list[dict] = []

    for split in SPLITS:
        path = src / f"{split}.jsonl"
        if not path.exists():
            print(f"{split}: missing, skipped")
            continue
        rows = jsonl.read(path)
        kept, stats = finalize_rows(rows, deframe_labels=not args.keep_framing,
                                    context=load_context(context_dir, split))
        teachers.update(r.get("label_model", "unknown") for r in kept)
        payload = jsonl.write(outdir / f"{split}.jsonl", kept)
        sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        per_split[split] = {"stats": stats, "sha": sha}
        if split == "train":
            train_rows = kept
        print(
            f"{split:<12}{stats['read']:>7} read  {stats['kept']:>7} kept "
            f"({100 * stats['kept'] / max(stats['read'], 1):.1f}%)  "
            f"not_review {stats['drop_not_review']}  unlabelled {stats['drop_unlabelled']}  "
            f"ai_markup {stats['drop_ai_markup']}  deframed {stats['deframed_explanation']}  "
            f"with_context {stats['with_context']}  sha {sha}"
        )

    write_card(outdir, per_split, teachers, train_rows)
    (outdir / "finalize.json").write_text(
        json.dumps(
            {k: {"stats": dict(v["stats"]), "sha": v["sha"]} for k, v in per_split.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {outdir.resolve()}  (see dataset-card.md)")


if __name__ == "__main__":
    main()
