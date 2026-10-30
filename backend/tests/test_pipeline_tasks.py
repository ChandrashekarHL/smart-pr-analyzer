from __future__ import annotations

import pytest
from conftest import Clock, FakeGitHub, FakeModel, model_result

from prana_app.pipeline import ModelUnavailable, Policy, review_pull_request
from prana_app.store import Store
from prana_app.tasks import FAILURE_COMMENT, run_review

H1 = "@@ -10,4 +10,6 @@ def handler(req):"
H2 = "@@ -40,2 +42,3 @@ def other():"


def test_happy_path_posts_one_review_with_inline_comments(gh):
    model = FakeModel({H2: model_result(line=43, severity="info", issue="STYLE",
                                        explanation="Name b says nothing.")})
    out = review_pull_request(gh, model, Policy(), "o/r", 5, "h1")
    assert out.status == "posted" and out.hunks_total == 2 and out.hunks_reviewed == 2
    assert [c[1] for c in model.calls] == ["python", "python"]        # README skipped
    assert model.calls[0][0] == H1                                    # largest hunk first
    (review,) = gh.reviews
    assert review["commit_id"] == "h1" and len(review["comments"]) == 2
    assert {(c["path"], c["line"]) for c in review["comments"]} == {("app/api.py", 12),
                                                                   ("app/api.py", 43)}
    assert "Reviewed **2** of 2" in review["body"]


def test_moved_head_is_superseded_and_posts_nothing():
    gh = FakeGitHub(head="h2")
    out = review_pull_request(gh, FakeModel(), Policy(), "o/r", 5, "h1")
    assert out.status == "superseded" and gh.reviews == []


def test_no_supported_hunks():
    gh = FakeGitHub(files=[{"filename": "README.md", "status": "modified",
                            "patch": "@@ -1 +1 @@\n-a\n+b"}])
    out = review_pull_request(gh, FakeModel(), Policy(), "o/r", 5, "h1")
    assert out.status == "no_hunks" and gh.reviews == []


def test_time_budget_stops_model_calls_and_says_so(gh):
    clock = Clock()
    model = FakeModel(delay=20.0, clock=clock)
    out = review_pull_request(gh, model, Policy(time_budget_s=25.0), "o/r", 5, "h1",
                              clock=clock)
    assert len(model.calls) == 1 and out.hunks_skipped_time == 1
    assert "1 hunks skipped to stay within the time budget" in gh.reviews[0]["body"]


def test_duplicates_and_low_confidence_are_not_posted_inline():
    from conftest import PATCH

    from prana_app.diff import parse_patch
    from prana_app.review import finding_from
    h1 = parse_patch("app/api.py", PATCH, "python")[0]
    earlier = finding_from(h1, model_result())
    gh = FakeGitHub(existing=[{"path": "app/api.py", "line": 12, "body": earlier.body()}])
    model = FakeModel({H2: model_result(line=43, confidence=0.1, explanation="Maybe rename b.")})
    out = review_pull_request(gh, model, Policy(min_confidence=0.5), "o/r", 5, "h1")
    assert out.duplicates == 1 and len(out.low_confidence) == 1 and out.posted == []
    assert gh.reviews[0]["comments"] == []
    assert "Low-confidence suggestions (1)" in gh.reviews[0]["body"]


def test_rejected_inline_comments_fall_back_to_summary():
    gh = FakeGitHub(reject_inline=True)
    out = review_pull_request(gh, FakeModel(), Policy(), "o/r", 5, "h1")
    assert out.status == "posted_summary_only"
    assert gh.reviews[0]["comments"] == []
    assert "Findings that could not be placed inline (2)" in gh.reviews[0]["body"]


def test_one_failed_hunk_is_counted_all_failed_raises(gh):
    class Flaky(FakeModel):
        def __call__(self, hunk, language, context, path):
            if hunk.startswith(H2):
                raise RuntimeError("timeout")
            return super().__call__(hunk, language, context, path)
    out = review_pull_request(gh, Flaky(), Policy(), "o/r", 5, "h1")
    assert out.hunks_failed == 1 and out.hunks_reviewed == 1 and len(out.posted) == 1
    with pytest.raises(ModelUnavailable):
        review_pull_request(FakeGitHub(), FakeModel(fail=True), Policy(), "o/r", 5, "h1")


def test_retrieval_context_reaches_the_model_and_its_failure_does_not(gh):
    model = FakeModel()
    review_pull_request(gh, model, Policy(), "o/r", 5, "h1",
                        retrieve=lambda h, repo, pr, base: f"ctx for {h.path}@{base}")
    assert model.calls[0][2] == "ctx for app/api.py@b1"

    def broken(*a):
        raise OSError("clone failed")
    model2 = FakeModel()
    out = review_pull_request(FakeGitHub(), model2, Policy(), "o/r", 5, "h1", retrieve=broken)
    assert out.status == "posted" and model2.calls[0][2] is None


# ------------------------------------------------------------------ worker

class Svc:
    def __init__(self, gh, model, store):
        self.store, self.model, self.retrieve = store, model, None
        self.policy = Policy()
        self.app = type("A", (), {"client": lambda _self, inst: gh})()


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path / 'db.sqlite'}")
    s.create_all()
    return s


def test_worker_success_records_run_and_comments(store, gh):
    rid = store.open_run(1, "o/r", 5, "h1")
    assert run_review(Svc(gh, FakeModel(), store), rid, attempt=0) == "posted"
    run = store.get_run(rid)
    assert run.status == "posted" and run.attempts == 1 and run.review_id == 99
    assert store.stats()["comments_by_severity"] == {"critical": 2}


def test_worker_transient_failure_reraises_until_last_attempt(store):
    gh = FakeGitHub()
    rid = store.open_run(1, "o/r", 5, "h1")
    with pytest.raises(ModelUnavailable):
        run_review(Svc(gh, FakeModel(fail=True), store), rid, attempt=0)
    assert store.get_run(rid).status == "pending" and gh.issue_comments == []
    assert run_review(Svc(gh, FakeModel(fail=True), store), rid, attempt=3) == "failed"
    run = store.get_run(rid)
    assert run.status == "failed" and "ModelUnavailable" in run.error
    assert gh.issue_comments == [FAILURE_COMMENT.format(reason="the review model is unavailable")]


def test_failed_run_can_be_reopened_but_done_runs_cannot(store):
    rid = store.open_run(1, "o/r", 5, "h1")
    assert store.open_run(1, "o/r", 5, "h1") is None            # queued: no duplicate
    store.mark(rid, "failed", error="x")
    assert store.open_run(1, "o/r", 5, "h1") == rid             # replay reopens it
    assert store.open_run(1, "o/r", 5, "h2") != rid             # new head, new run
