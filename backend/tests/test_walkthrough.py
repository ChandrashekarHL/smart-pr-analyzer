from __future__ import annotations

from conftest import FakeGitHub, FakeModel

from prana_app.pipeline import Policy, review_pull_request
from prana_app.walkthrough import MARKER, build_messages, parse, post_walkthrough, render

# The base model's answer for the demo repo's Go PR (2026-11-11), verbatim
# apart from the trimmed third bullet.
MODEL_TEXT = """## Walkthrough
This pull request modifies the `stock.go` file to include a new method `Move`. The `Move`
method allows for the transfer of a specified quantity of a specific SKU from one store to
another.

## Changes
| File | Change summary |
| --- | --- |
| services/inventory/stock.go | Added the `Move` method to the `Store` struct |
| services/inventory/ledger.go | Added an audit ledger |

## Risk Areas
- The `Move` method does not check if the source store has enough stock.
- The `Move` method does not handle errors that may occur during the transfer process.
"""

FILES = [
    {"filename": "services/inventory/stock.go", "status": "modified", "additions": 12,
     "deletions": 1,
     "patch": "@@ -40,3 +40,14 @@\n func (s *Store) Add() {}\n+func (s *Store) Move() {}"},
    {"filename": "docs/inventory.md", "status": "added", "additions": 4, "deletions": 0,
     "patch": "@@ -0,0 +1,4 @@\n+# Inventory | stock"},
]
PATHS = [f["filename"] for f in FILES]


def test_parse_keeps_real_files_only_and_the_risks():
    w = parse(MODEL_TEXT, PATHS)
    assert w.summary.startswith("This pull request modifies") and "\n" not in w.summary
    # ledger.go is not in the PR: a model row for it is dropped, never rendered
    assert w.per_file == {"services/inventory/stock.go":
                          "Added the `Move` method to the `Store` struct"}
    assert len(w.risks) == 2 and "handle errors" in w.risks[1]
    assert w.usable


def test_parse_matches_a_row_by_unique_basename():
    w = parse("## Changes\n| File | Summary |\n|---|---|\n| `stock.go` | New Move method |\n",
              PATHS)
    assert w.per_file == {"services/inventory/stock.go": "New Move method"}
    w = parse("## Changes\n| a.go | x |\n", ["one/a.go", "two/a.go"])      # ambiguous
    assert w.per_file == {} and not w.usable


def test_render_takes_files_and_line_counts_from_github():
    body = render(parse(MODEL_TEXT, PATHS), FILES, "abcdef1234567")
    assert body.startswith("## Walkthrough") and body.endswith(MARKER)
    assert "| `services/inventory/stock.go` | Added the `Move` method" in body
    assert "| +12 / −1 |" in body
    assert "| `docs/inventory.md` | — | +4 / −0 |" in body      # no model row: still listed
    assert "ledger.go" not in body
    assert "## Things to check" in body and "`abcdef1`" in body


def test_prompt_is_the_diff_with_a_note_for_files_that_did_not_fit():
    big = [{"filename": f"f{i}.py", "patch": "+x = 1\n" * 400} for i in range(5)]
    messages = build_messages("Speed up imports", big)
    user = messages[1]["content"]
    assert messages[0]["role"] == "system" and "never invent" in messages[0]["content"]
    assert "Pull request: Speed up imports" in user and "diff --git a/f0.py" in user
    assert "also changed, diff not shown: f2.py, f3.py, f4.py" in user


def test_one_comment_per_pr_edited_on_each_commit():
    gh = FakeGitHub()
    gh.conversation.append({"id": 1, "body": f"I copied your {MARKER}", "user": {"login": "dev"}})
    assert post_walkthrough(gh, lambda m: MODEL_TEXT, "o/r", 5, "h1", "Move", FILES) == "created"
    assert post_walkthrough(gh, lambda m: MODEL_TEXT, "o/r", 5, "h2", "Move", FILES) == "updated"
    ours = [c for c in gh.conversation if c["user"]["login"] == "prana-reviewer[bot]"]
    assert len(ours) == 1 and "`h2`" in ours[0]["body"]
    assert gh.conversation[0]["body"] == f"I copied your {MARKER}"      # a human's, untouched


def test_unusable_or_failing_generation_leaves_a_short_note_not_a_walkthrough():
    gh = FakeGitHub()
    assert post_walkthrough(gh, lambda m: "I cannot help with that.", "o/r", 5, "h1", "t",
                            FILES) == "unusable"

    def boom(messages):
        raise TimeoutError("model API timed out")
    assert post_walkthrough(gh, boom, "o/r", 5, "h1", "t", FILES) == "failed"
    # one comment, edited: the "review posted" note replaces any "reviewing" status
    assert len(gh.conversation) == 1
    body = gh.conversation[0]["body"]
    assert "prana review posted" in body and "## Walkthrough" not in body
    assert "timed out" not in body                        # internal errors are not shown


def test_pipeline_posts_the_walkthrough_after_the_review(gh):
    seen = []

    def generate(messages):
        seen.append(len(gh.posted))          # the review is already on GitHub
        return MODEL_TEXT.replace("services/inventory/stock.go", "app/api.py")
    out = review_pull_request(gh, FakeModel(), Policy(), "o/r", 5, "h1", walkthrough=generate)
    # "updated": the progress comment posted at the start becomes the walkthrough
    assert out.status == "posted" and out.walkthrough == "updated" and seen == [1]
    assert len(gh.conversation) == 1 and MARKER in gh.conversation[0]["body"]


def test_progress_comment_appears_before_the_model_runs_then_becomes_the_walkthrough(gh):
    seen = []

    def model(hunk, language, context, path):
        seen.append([c["body"] for c in gh.conversation])     # what the PR shows meanwhile
        return FakeModel()(hunk, language, context, path)

    out = review_pull_request(gh, model, Policy(), "o/r", 5, "h1", walkthrough=lambda m: MODEL_TEXT)
    assert out.status == "posted" and seen
    assert "prana is reviewing this pull request" in seen[0][0] and "`h1`" in seen[0][0]
    assert len(gh.conversation) == 1                           # the same comment, edited
    assert gh.conversation[0]["body"].startswith("## Walkthrough")


def test_progress_comment_never_stays_at_reviewing(gh):
    from prana_app.pipeline import ModelUnavailable
    try:
        review_pull_request(gh, FakeModel(fail=True), Policy(), "o/r", 5, "h1",
                            walkthrough=lambda m: MODEL_TEXT)
    except ModelUnavailable:
        pass
    else:
        raise AssertionError("expected ModelUnavailable")
    assert "could not finish this review" in gh.conversation[0]["body"]
    assert "review model is unavailable" in gh.conversation[0]["body"]

    gh2 = FakeGitHub()
    review_pull_request(gh2, FakeModel(), Policy(), "o/r", 5, "h1",
                        walkthrough=lambda m: "no sections here")
    assert len(gh2.conversation) == 1 and "prana review posted" in gh2.conversation[0]["body"]


def test_no_progress_comment_when_the_walkthrough_is_off(gh):
    review_pull_request(gh, FakeModel(), Policy(), "o/r", 5, "h1")
    assert gh.conversation == []


def test_a_failing_walkthrough_never_breaks_the_review(gh):
    def generate(messages):
        raise RuntimeError("GPU busy")
    out = review_pull_request(gh, FakeModel(), Policy(), "o/r", 5, "h1", walkthrough=generate)
    assert out.status == "posted" and out.walkthrough == "failed" and len(gh.posted) == 1
