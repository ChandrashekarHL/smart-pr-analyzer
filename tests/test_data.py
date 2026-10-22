"""Tests for the filters that decide what the model learns from.

These are the rules that are expensive to get wrong: a bot slipping through
invalidates the project's central claim, a leaked PR invalidates every metric,
and a missed credential is unrecoverable once the dataset is published.
"""

from __future__ import annotations

import pytest

from prana.config import DataConfig
from prana.data import filters
from prana.data.build import assert_no_leakage, split_rows
from prana.data.decontaminate import BloomFilter, build_index, contamination, ngrams, tokenize
from prana.data.licenses import LicenseIndex, tier
from prana.data.scrub import scan, scrub

# ------------------------------------------------------------- authorship

@pytest.mark.parametrize("name", [
    "Copilot", "copilot", "coderabbitai[bot]", "chatgpt-codex-connector[bot]",
    "gemini-code-assist[bot]", "greptile-apps[bot]", "49f5b0e4",
    "some-new-thing[bot]", "ingress-bot", "helper_bot",
])
def test_bot_reviewers_rejected(name):
    assert filters.is_bot_reviewer(name)


@pytest.mark.parametrize("name", ["krobelus", "zkochan", "961b8ddc", "robotnik", "abbot"])
def test_humans_kept(name):
    assert not filters.is_bot_reviewer(name)


def test_bot_detection_does_not_rely_on_comment_text():
    """Copilot's prose is indistinguishable from a human's - which is exactly
    why 9,751 of its comments survived a text-based filter. Authorship is the
    only signal that works."""
    human_looking = (
        "The new branch is not used by the provider's background refresh: "
        "`copilotCloudSessionsProvider.ts:447` still calls fetchSessionList."
    )
    assert not filters.BOT_MARKUP.search(human_looking)
    assert filters.is_bot_reviewer("Copilot")


def test_human_arguing_with_a_scanner_is_kept():
    """An earlier pass matched bare tool names and discarded ~4,000 of these."""
    comment = ("False positive. `source_to_sanitize` is always a valid "
               "null-terminated string, so the SonarCloud warning is wrong here.")
    assert not filters.BOT_MARKUP.search(comment)


# ------------------------------------------------------------- content

@pytest.mark.parametrize("text", [
    "Done.", "Updated. Thank you!", "Same as above", "LGTM", "+1", "nit: spacing",
])
def test_conversational_replies_rejected(text):
    assert filters.CONVERSATIONAL.match(text)


def test_real_review_not_treated_as_conversational():
    assert not filters.CONVERSATIONAL.match(
        "Guard against circular cause chains here; render recurses on err.cause "
        "without a depth limit and will stack-overflow on a cycle."
    )


def test_suggestion_block_extracted_as_structured_fix():
    body, fix = filters.split_suggestion(
        "Use the constant instead.\n\n```suggestion\nmaxInt64 = math.MaxInt64\n```"
    )
    assert body == "Use the constant instead."
    assert fix == "maxInt64 = math.MaxInt64"


def test_comment_without_suggestion_is_unchanged():
    body, fix = filters.split_suggestion("Please add a regression test.")
    assert body == "Please add a regression test."
    assert fix is None


def test_language_recovered_from_path_when_scraper_left_it_blank():
    assert filters.detect_language("cmd/server/main.go", None) == "go"
    assert filters.detect_language("src/app.tsx", "") == "typescript"
    assert filters.detect_language("Makefile", None) is None


def test_declared_language_wins_when_present():
    assert filters.detect_language("x.go", "python") == "python"


def test_bare_question_rejected_but_long_question_kept():
    assert filters.is_bare_question("Should it be >= instead?", 40)
    long_q = " ".join(["why"] * 45) + "?"
    assert not filters.is_bare_question(long_q, 40)


# ------------------------------------------------------------- secrets

@pytest.mark.parametrize("secret,kind", [
    ("AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
    ("ghp_1234567890abcdefghijklmnopqrstuvwxyz", "github_pat"),
    ("AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8", "google_api_key"),
    ("xoxb-test-token-abcdefghij", "slack_token"),
])
def test_credentials_redacted(secret, kind):
    result = scrub(f"const key = '{secret}';")
    assert secret not in result.text
    assert kind in result.hits


def test_redaction_preserves_surrounding_code():
    result = scrub("headers['Authorization'] = 'Bearer abcdefghijklmnopqrstuvwxyz012345'")
    assert "headers['Authorization']" in result.text
    assert "REDACTED" in result.text


def test_clean_code_untouched():
    code = "func Add(a, b int) int { return a + b }"
    assert scrub(code).text == code
    assert not scan(code)


def test_email_false_positives_not_redacted():
    """A bare email regex fires on these; reporting them as PII is noise."""
    for text in ["@pytest.mark.asyncio", "toolchain@v0.0.1-go1.26.2.darwin"]:
        assert scrub(text, redact_emails=True).text == text


def test_real_email_redacted_when_requested():
    out = scrub("contact alice@example.io for details", redact_emails=True)
    assert "alice@example.io" not in out.text


# ------------------------------------------------------------- licences

def test_license_tiers():
    assert tier("mit") == "permissive"
    assert tier("apache-2.0") == "permissive"
    assert tier("agpl-3.0") == "strong-copyleft"
    assert tier("mpl-2.0") == "weak-copyleft"
    # GitHub's label for "we could not classify this", not a permission grant.
    assert tier("other") == "unknown"
    assert tier(None) == "unknown"


def test_license_index_handles_scraper_cache_shape(tmp_path):
    p = tmp_path / "lic.json"
    p.write_text('{"a/b": {"key": "mit"}, "c/d": "agpl-3.0", "e/f": null}', encoding="utf-8")
    idx = LicenseIndex.load(p)
    assert idx.publishable("a/b")
    assert not idx.publishable("c/d")
    assert idx.tier("e/f") == "unknown"


# ------------------------------------------------------------- splitting

def _row(repo, pr, i):
    return {"repo_name": repo, "pr_number": str(pr), "id": f"{repo}-{pr}-{i}",
            "diff_hunk": f"@@ -1 +1 @@\n+x{i}", "review_comment": "c"}


def test_split_never_puts_one_pr_in_two_splits():
    rows = [_row(f"org/r{r}", pr, i)
            for r in range(20) for pr in range(10) for i in range(3)]
    splits = split_rows(rows, DataConfig())
    assert_no_leakage(splits)          # raises on failure
    assert sum(len(v) for v in splits.values()) == len(rows)


def test_ood_repos_are_absent_from_training():
    rows = [_row(f"org/r{r}", pr, i)
            for r in range(20) for pr in range(10) for i in range(3)]
    splits = split_rows(rows, DataConfig())
    train_repos = {r["repo_name"] for r in splits["train"]}
    ood_repos = {r["repo_name"] for r in splits["test_ood"]}
    assert ood_repos and not (train_repos & ood_repos)


def test_leakage_assertion_actually_fires():
    bad = {"train": [_row("a/b", 1, 0)], "test": [_row("a/b", 1, 1)]}
    with pytest.raises(SystemExit, match="LEAK"):
        assert_no_leakage(bad)


def test_split_is_deterministic():
    rows = [_row(f"org/r{r}", pr, 0) for r in range(12) for pr in range(8)]
    a = split_rows(rows, DataConfig())
    b = split_rows(rows, DataConfig())
    assert [r["id"] for r in a["train"]] == [r["id"] for r in b["train"]]


# ------------------------------------------------------------- decontamination

def test_bloom_filter_has_no_false_negatives():
    bloom = BloomFilter(1000)
    items = [f"gram number {i}" for i in range(500)]
    for item in items:
        bloom.add(item)
    assert all(item in bloom for item in items)


def test_bloom_false_positive_rate_is_near_target():
    bloom = BloomFilter(5000, error_rate=0.01)
    for i in range(5000):
        bloom.add(f"present {i}")
    fp = sum(1 for i in range(5000) if f"absent {i}" in bloom)
    assert fp / 5000 < 0.05


def test_ngram_tokenisation_ignores_formatting():
    assert tokenize("foo_bar(baz)") == tokenize("foo_bar ( baz )")


def test_identical_hunk_in_another_pr_is_detected():
    """PR-level splitting does not catch a cherry-pick; n-grams do."""
    shared = "@@ -1,4 +1,4 @@\n" + " ".join(f"token{i}" for i in range(60))
    train = [{"diff_hunk": shared}]
    bloom = build_index(train, n=13)
    assert contamination({"diff_hunk": shared}, bloom, n=13) > 0.9


def test_unrelated_hunk_is_clean():
    train = [{"diff_hunk": " ".join(f"alpha{i}" for i in range(60))}]
    bloom = build_index(train, n=13)
    other = {"diff_hunk": " ".join(f"omega{i}" for i in range(60))}
    assert contamination(other, bloom, n=13) == 0.0


def test_short_text_yields_no_ngrams():
    assert list(ngrams(tokenize("too short"), 13)) == []


# ------------------------------------------------------------- jsonl io

def test_jsonl_survives_unicode_line_separators(tmp_path):
    """U+2028 and U+2029 are legal inside a JSON string but str.splitlines()
    breaks on them. Real minified JS in the corpus carries them, and a reader
    built on splitlines() tears the record in half."""
    from prana.data import jsonl
    row = {"diff_hunk": "@@ -1 +1 @@\n+const a = 'x\u2028y\u2029z';", "id": "r/p-1"}
    path = tmp_path / "x.jsonl"
    jsonl.write(path, [row, row])
    assert jsonl.read(path) == [row, row]


def test_jsonl_roundtrips_crlf_and_tabs(tmp_path):
    from prana.data import jsonl
    row = {"diff_hunk": "@@ -1 +1 @@\r\n+\tif x:\r\n", "id": "a/b-2"}
    path = tmp_path / "y.jsonl"
    jsonl.write(path, [row])
    assert jsonl.read(path)[0]["diff_hunk"] == row["diff_hunk"]


# ------------------------------------------------------------- author replies

@pytest.mark.parametrize("text", [
    "Addressed in 5fa47ff: setup now computes allow_browser_cookies through the same policy.",
    "Real find — fixed in f5de37aef7: parse_import_method now accepts clone-or-copy.",
    "You're right that a single boolean can't distinguish backup-restore from reset.",
    "I've added a regression test for the dual-URL case.",
    "We pushed a fix that guards the empty path.",
    "Thanks, updated the docstring.",
])
def test_author_replies_rejected(text):
    """The PR author answering a review is not a review."""
    assert filters.is_author_reply(text)


@pytest.mark.parametrize("text", [
    "I tried this locally and it crashes when the list is empty; guard len(xs) first.",
    "I need to understand why this is synchronous - it blocks the event loop.",
    "This leaks the file handle on the error path; use a context manager.",
    "Consider extracting this into a helper; the same logic appears three times.",
])
def test_reviews_mentioning_first_person_are_kept(text):
    """A bare any-word-ending-in-ed rule matched 'I need' and 'I tried',
    which are reviews."""
    assert not filters.is_author_reply(text)


def test_machine_translated_review_wrapper_rejected():
    """277 rows of corpus v2 were a translation bot's English with the human's
    Chinese folded under <details>; v0.3.1 learned to emit the wrapper."""
    text = ("> This review comment was translated automatically.\n\n**[A1]** This range "
            "is not in the CJK block.\n\n---\n<details>\n<summary>Original Content</summary>")
    assert filters.BOT_MARKUP.search(text)


def test_details_block_alone_is_not_bot_markup():
    """<details> is ordinary GitHub markdown; 963 human rows use it."""
    assert not filters.BOT_MARKUP.search(
        "<details><summary>Benchmark output</summary>\n\n```\n1.2ms\n```\n</details>"
    )


@pytest.mark.parametrize("text", [
    '<!-- metadata:{"confidence":8,"steps":[]} --> P3: The added comment describes '
    'an invalid collection.',
    "**[P2 \u00b7 test]** Coverage + assertion gaps in this new test: missing the scoped case.",
    "**[P1]** This leaks the handle on the error path.",
])
def test_ai_tool_markup_under_personal_accounts_rejected(text):
    """Two human accounts post AI-review output verbatim; the badge or the
    metadata header is the only signal, since the author field is a person."""
    assert filters.BOT_MARKUP.search(text)


def test_priority_word_in_prose_is_not_markup():
    assert not filters.BOT_MARKUP.search(
        "I'd treat this as P2 - it only affects the debug build, but worth a fix."
    )
