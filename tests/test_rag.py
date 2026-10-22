"""Tests for the retriever. The leakage rules are the part that matters: a
training row that can see its own review, or a test row that can see a
sibling comment from its own PR, inflates every with-context number."""

from __future__ import annotations

import pytest

from prana.rag.bm25 import BM25, tokenize
from prana.rag.retriever import Retriever


def _row(i, repo="acme/svc", pr="1", file="pkg/db.go", hunk=None, review=None):
    return {
        "id": f"{repo}_{pr}-{i}", "repo_name": repo, "pr_number": pr, "file_path": file,
        "diff_hunk": hunk or (f"@@ -1 +1 @@\n+rows, err := db.QueryContext(ctx, q{i})\n"
                              "+defer rows.Close()"),
        "structured_output": {"explanation": review or f"Close rows on the error path {i}.",
                              "suggestion": "Use defer after the error check."},
    }


# ------------------------------------------------------------- tokeniser / bm25

def test_tokenizer_splits_identifiers_and_drops_noise():
    toks = tokenize("+\tdefer rowsIter.Close() // parse_config\n@@ -1 +1 @@")
    assert "rows" in toks and "iter" in toks and "close" in toks
    assert "parse" in toks and "config" in toks
    assert "defer" in toks and "@@" not in toks


def test_bm25_ranks_the_overlapping_document_first():
    idx = BM25()
    idx.add("a", "rows err db QueryContext ctx defer rows Close")
    idx.add("b", "render template html escape user input")
    assert idx.query("db QueryContext rows Close")[0][0] == "a"


def test_bm25_allow_restricts_candidates():
    idx = BM25()
    idx.add("a", "db QueryContext rows")
    idx.add("b", "db QueryContext rows Close")
    assert [d for d, _ in idx.query("db rows", allow=[0])] == ["a"]


# ------------------------------------------------------------- retrieval sources

def test_same_file_reviews_come_first():
    r = Retriever.from_rows([_row(1, pr="10"), _row(2, pr="11", file="pkg/other.go")])
    got = r.retrieve("acme/svc", "pkg/db.go", "@@ -1 +1 @@\n+x := y", exclude_pr="99")
    assert got.context.startswith("## Past reviews on this file")
    assert "Close rows on the error path 1." in got.context
    assert got.sources[0]["kind"] == "same_file"


def test_other_repo_fallback_fires_for_unseen_repo():
    """test_ood repositories have no history; the cross-repo source is all
    a fresh install gets, and it must still produce something relevant."""
    r = Retriever.from_rows([_row(1, repo="seen/repo", pr="3")])
    hunk = "@@ -1 +1 @@\n+rows, err := db.QueryContext(ctx, q)"
    got = r.retrieve("unseen/repo", "x.go", hunk)
    assert "Similar code reviewed in other repositories" in got.context
    assert got.sources == [{"kind": "other_repo", "id": "seen/repo_3-1"}]


def test_empty_when_nothing_relevant():
    r = Retriever.from_rows([_row(1, repo="seen/repo", pr="3")])
    got = r.retrieve("unseen/repo", "x.py", "@@ -1 +1 @@\n+zzz = qqq")
    assert got.context == "" and got.sources == []


# ------------------------------------------------------------- leakage

def test_own_pull_request_is_never_retrieved():
    """A training row must not see its own review, nor a sibling comment on
    the same PR - either one teaches the model to copy the answer."""
    rows = [_row(1, pr="7"), _row(2, pr="7", file="pkg/db.go"), _row(3, pr="8")]
    r = Retriever.from_rows(rows)
    got = r.retrieve("acme/svc", "pkg/db.go", rows[0]["diff_hunk"], exclude_pr="7")
    ids = {s["id"] for s in got.sources}
    assert "acme/svc_7-1" not in ids and "acme/svc_7-2" not in ids
    assert "acme/svc_8-3" in ids


def test_exclusion_is_scoped_to_the_repo():
    """PR numbers collide across repositories; excluding '7' must not drop
    another repo's PR 7."""
    rows = [_row(1, repo="acme/svc", pr="7"), _row(2, repo="other/lib", pr="7")]
    r = Retriever.from_rows(rows)
    got = r.retrieve("acme/svc", "pkg/db.go", rows[0]["diff_hunk"], exclude_pr="7")
    assert {s["id"] for s in got.sources} == {"other/lib_7-2"}


# ------------------------------------------------------------- budget / shape

def test_budget_is_respected_with_whole_items():
    rows = [_row(i, pr=str(100 + i), review="x" * 900) for i in range(5)]
    r = Retriever.from_rows(rows)
    got = r.retrieve("acme/svc", "pkg/db.go", rows[0]["diff_hunk"], exclude_pr="999",
                     budget_chars=2000)
    assert len(got.context) <= 2000 + len("## Past reviews on this file\n")
    assert all(line.startswith(("##", "- ")) for line in got.context.splitlines() if line)


def test_round_trip_through_save_and_load(tmp_path):
    r = Retriever.from_rows([_row(1, pr="2")])
    r.save(tmp_path / "index.json")
    r2 = Retriever.load(tmp_path / "index.json")
    assert len(r2.docs) == 1 and r2.docs[0].id == "acme/svc_2-1"


@pytest.mark.parametrize("field", ["structured_output", "review_comment"])
def test_builds_from_finalized_or_raw_rows(field):
    row = _row(1, pr="2")
    if field == "review_comment":
        row.pop("structured_output")
        row["review_comment"] = "Raw human comment."
    r = Retriever.from_rows([row])
    assert r.docs[0].review in ("Close rows on the error path 1.", "Raw human comment.")
