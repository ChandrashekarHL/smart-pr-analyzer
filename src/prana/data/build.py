"""Build the training corpus from the raw scrape.

    prana-build --raw v2_strict/master_dataset_v1.jsonl --out data_v3

Deterministic: the same raw file and the same DataConfig always produce
byte-identical splits. Every run writes a dataset card recording the config
fingerprint, the content hash of each split, and the counts behind every
filter decision, so a result can be traced to the corpus that produced it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from prana.config import PROSE_LANGUAGES, DataConfig
from prana.data import filters, jsonl
from prana.data.licenses import LicenseIndex
from prana.data.scrub import scrub
from prana.prompts import PROMPT_FINGERPRINT, PROMPT_VERSION


def _iter_raw(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def build_rows(raw_path: Path, cfg: DataConfig, licenses: LicenseIndex | None):
    stats: collections.Counter[str] = collections.Counter()
    langs: collections.Counter[str] = collections.Counter()
    reviewers: collections.Counter[str] = collections.Counter()
    reviewer_repos: dict[str, set[str]] = collections.defaultdict(set)
    license_tiers: collections.Counter[str] = collections.Counter()
    redactions: collections.Counter[str] = collections.Counter()
    seen_pair: set[bytes] = set()
    seen_norm: set[str] = set()
    rows: list[dict] = []
    extra_bots = frozenset(b.lower() for b in cfg.extra_bot_reviewers)

    for line in _iter_raw(raw_path):
        stats["read"] += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["drop_unparseable"] += 1
            continue

        repo = (rec.get("repository") or {}).get("repo_name")
        pr = (rec.get("pull_request") or {}).get("pr_number")
        fl = rec.get("file") or {}
        rv = rec.get("review") or {}
        diff = (rv.get("code_diff") or "").strip()
        comment = (rv.get("review_comment") or "").strip()

        if not (repo and pr and diff and comment):
            stats["drop_incomplete"] += 1
            continue

        # Authorship first: it is the most consequential filter and the
        # cheapest, so nothing else is computed for a row a bot wrote.
        reviewer = rv.get("reviewer")
        if filters.is_bot_reviewer(reviewer, extra_bots):
            stats["drop_bot_author"] += 1
            continue
        if filters.BOT_MARKUP.search(comment):
            stats["drop_bot_markup"] += 1
            continue

        lang = filters.detect_language(fl.get("file_path"), fl.get("programming_language"))
        if lang is None:
            stats["drop_no_language"] += 1
            continue
        if cfg.drop_prose and lang in PROSE_LANGUAGES:
            stats["drop_prose"] += 1
            continue
        if cfg.languages and lang not in cfg.languages:
            stats["drop_out_of_scope"] += 1
            continue

        if (licenses is not None and cfg.require_permissive_license
                and not licenses.publishable(repo)):
            stats["drop_license"] += 1
            continue

        if not diff.startswith("@@"):
            stats["drop_not_hunk"] += 1
            continue
        if len(diff) > cfg.max_diff_chars:
            stats["drop_diff_too_long"] += 1
            continue

        if filters.CONVERSATIONAL.match(comment):
            stats["drop_conversational"] += 1
            continue

        body, fix = filters.split_suggestion(comment)
        if filters.is_author_reply(body):
            stats["drop_author_reply"] += 1
            continue
        if len(body.split()) < cfg.min_comment_words:
            stats["drop_too_short"] += 1
            continue
        if cfg.drop_bare_questions and filters.is_bare_question(
            body, cfg.bare_question_max_words
        ):
            stats["drop_bare_question"] += 1
            continue

        pair = hashlib.sha1(f"{diff}\x1f{comment}".encode()).digest()
        if pair in seen_pair:
            stats["drop_dup_exact"] += 1
            continue
        seen_pair.add(pair)

        key = filters.normalise_for_dedup(body)
        if key in seen_norm:
            stats["drop_dup_near"] += 1
            continue
        seen_norm.add(key)

        if cfg.redact_secrets:
            d, b = scrub(diff), scrub(body)
            if d.changed or b.changed:
                for name, count in {**d.hits, **b.hits}.items():
                    redactions[name] += count
                stats["redacted"] += 1
            diff, body = d.text, b.text

        line_no = rv.get("line_number")
        if line_no in ("None", "", None):
            line_no = None

        rows.append({
            "id": f"{repo}_{pr}-{rv.get('review_id')}",
            "repo_name": repo,
            "pr_number": str(pr),
            "language": lang,
            "file_path": fl.get("file_path"),
            "line_number": line_no,
            "reviewer": reviewer,
            "license": licenses.key(repo) if licenses else None,
            "diff_hunk": diff,
            "review_comment": body,
            "suggested_fix": fix,
            "has_suggestion": fix is not None,
        })
        stats["kept"] += 1
        langs[lang] += 1
        if reviewer:
            reviewers[reviewer] += 1
            reviewer_repos[reviewer].add(repo)
        if licenses:
            license_tiers[licenses.tier(repo)] += 1

    return rows, {
        "stats": stats, "languages": langs, "reviewers": reviewers,
        "reviewer_repos": reviewer_repos, "license_tiers": license_tiers,
        "redactions": redactions,
    }


def split_rows(rows: list[dict], cfg: DataConfig) -> dict[str, list[dict]]:
    """Group-aware split.

    test_ood holds out whole repositories, so it measures transfer to an
    unseen codebase. validation and test hold out whole pull requests from
    repositories the model has seen. A PR never spans two splits.
    """
    rng = random.Random(cfg.seed)
    by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_repo[row["repo_name"]].append(row)

    # Prefer small repositories for the OOD set so hitting the target
    # fraction does not remove a large share of any one language.
    ordered = sorted(by_repo, key=lambda k: (len(by_repo[k]), k))
    target = int(len(rows) * cfg.ood_fraction)
    ood_repos: set[str] = set()
    acc = 0
    for repo in ordered:
        if acc >= target:
            break
        ood_repos.add(repo)
        acc += len(by_repo[repo])

    test_ood = [r for r in rows if r["repo_name"] in ood_repos]
    rest = [r for r in rows if r["repo_name"] not in ood_repos]

    by_pr: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rest:
        by_pr[(row["repo_name"], row["pr_number"])].append(row)
    keys = sorted(by_pr)
    rng.shuffle(keys)

    n = len(rest)
    val: list[dict] = []
    test: list[dict] = []
    train: list[dict] = []
    for key in keys:
        group = by_pr[key]
        if len(val) < n * cfg.val_fraction:
            val.extend(group)
        elif len(test) < n * cfg.test_fraction:
            test.extend(group)
        else:
            train.extend(group)

    return {"train": train, "validation": val, "test": test, "test_ood": test_ood}


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    """A pull request in two splits invalidates every metric downstream."""
    seen: dict[tuple[str, str], str] = {}
    for name, rows in splits.items():
        for row in rows:
            key = (row["repo_name"], row["pr_number"])
            if seen.get(key, name) != name:
                raise SystemExit(f"LEAK: PR {key} in both {seen[key]} and {name}")
            seen[key] = name


def write_splits(splits: dict[str, list[dict]], outdir: Path) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, rows in splits.items():
        payload = jsonl.write(outdir / f"{name}.jsonl", rows)
        digests[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return digests


def write_card(outdir: Path, cfg: DataConfig, meta: dict, splits: dict, digests: dict) -> None:
    stats = meta["stats"]
    total = stats["read"]
    kept = stats["kept"]
    lines = [
        "# Dataset card - PR Analyzer review corpus",
        "",
        f"- config fingerprint: `{cfg.fingerprint()}`",
        f"- filters version: `{filters.FILTERS_VERSION}`",
        f"- prompt contract: `{PROMPT_VERSION}` / `{PROMPT_FINGERPRINT}`",
        f"- rows: **{kept:,}** kept from {total:,} raw ({100 * kept / total:.1f}%)",
        "",
        "## Provenance",
        "",
        "Pull-request review comments scraped from public GitHub repositories via",
        "the REST API. Each row is one human review comment paired with the unified",
        "diff hunk it was attached to.",
        "",
        "## Known limitations",
        "",
        "- Severity, issue type and confidence are **not** present in the source data.",
        "  Any model trained to emit them needs a separate labelling pass.",
        "- Review comments are noisy supervision: a reviewer's silence does not mean",
        "  a hunk is clean, so absence of a comment is not a negative label.",
        "- Repository distribution is long-tailed; the largest repository contributes",
        "  well under 2% of rows, but language balance still reflects what the",
        "  scraped population happened to contain.",
        "",
        "## Splits",
        "",
        "| split | rows | PRs | repos | sha256 |",
        "|---|---|---|---|---|",
    ]
    for name, rows in splits.items():
        prs = len({(r["repo_name"], r["pr_number"]) for r in rows})
        repos = len({r["repo_name"] for r in rows})
        lines.append(f"| {name} | {len(rows):,} | {prs:,} | {repos} | `{digests[name]}` |")

    lines += ["", "## Filter accounting", "", "| rule | rows | share |", "|---|---|---|"]
    for key, val in stats.most_common():
        if key in ("read", "kept"):
            continue
        lines.append(f"| {key} | {val:,} | {100 * val / total:.1f}% |")

    lines += ["", "## Language", "", "| language | rows | share |", "|---|---|---|"]
    for key, val in meta["languages"].most_common():
        lines.append(f"| {key} | {val:,} | {100 * val / kept:.1f}% |")

    if meta["license_tiers"]:
        lines += ["", "## Licence", "", "| tier | rows |", "|---|---|"]
        for key, val in meta["license_tiers"].most_common():
            lines.append(f"| {key} | {val:,} |")

    if meta["redactions"]:
        lines += ["", "## Redactions", "", "| kind | occurrences |", "|---|---|"]
        for key, val in meta["redactions"].most_common():
            lines.append(f"| {key} | {val:,} |")

    lines += ["", "## Configuration", "", "```json",
              json.dumps(cfg.to_dict(), indent=2, default=list), "```", ""]
    (outdir / "dataset-card.md").write_text("\n".join(lines), encoding="utf-8")


def print_reviewer_audit(meta: dict, limit: int = 10) -> None:
    print("\nreviewer audit - a person does not review dozens of unrelated repos")
    for name, n in meta["reviewers"].most_common(limit):
        repos = len(meta["reviewer_repos"][name])
        flag = "  <-- check" if n > 1500 and repos > 20 else ""
        print(f"  {name:<34}{n:>7}{repos:>7}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="v2_strict/master_dataset_v1.jsonl")
    ap.add_argument("--out", default="data_v3")
    ap.add_argument("--licenses", default="v2_strict/data/processed/license_cache.json")
    ap.add_argument("--languages", default=None, help="comma-separated override; 'all' for any")
    ap.add_argument("--min-words", type=int, default=None)
    ap.add_argument("--allow-any-license", action="store_true",
                    help="keep copyleft and unclassified repositories (local training only)")
    ap.add_argument("--no-redact", action="store_true")
    args = ap.parse_args()

    overrides: dict = {}
    if args.languages:
        overrides["languages"] = (
            () if args.languages == "all"
            else tuple(s.strip().lower() for s in args.languages.split(",") if s.strip())
        )
    if args.min_words is not None:
        overrides["min_comment_words"] = args.min_words
    if args.allow_any_license:
        overrides["require_permissive_license"] = False
    if args.no_redact:
        overrides["redact_secrets"] = False
    cfg = DataConfig(**overrides)

    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"not found: {raw}")

    lic_path = Path(args.licenses)
    licenses = LicenseIndex.load(lic_path) if lic_path.exists() else None
    if licenses is None:
        print(f"warning: no licence cache at {lic_path}; licence filter disabled")

    rows, meta = build_rows(raw, cfg, licenses)
    if not rows:
        raise SystemExit("no rows survived the filter")

    splits = split_rows(rows, cfg)
    assert_no_leakage(splits)

    outdir = Path(args.out)
    digests = write_splits(splits, outdir)
    write_card(outdir, cfg, meta, splits, digests)

    stats = meta["stats"]
    print(f"config {cfg.fingerprint()}  filters v{filters.FILTERS_VERSION}  "
          f"prompt {PROMPT_FINGERPRINT}")
    print(f"kept {stats['kept']:,} of {stats['read']:,} "
          f"({100 * stats['kept'] / stats['read']:.1f}%)")
    for name, rows_ in splits.items():
        print(f"  {name:<12}{len(rows_):>7} rows  sha {digests[name]}")
    print_reviewer_audit(meta)
    print(f"\nwrote {outdir.resolve()}  (see dataset-card.md)")
    print("next: prana-decon --data", args.out)


if __name__ == "__main__":
    main()
