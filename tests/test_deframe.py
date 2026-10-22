"""Tests for third-person frame removal on teacher labels."""

from __future__ import annotations

import pytest

from prana.data.deframe import deframe, is_framed


@pytest.mark.parametrize("framed,expected", [
    ("The reviewer suggests that the import be hoisted to module level.",
     "The import be hoisted to module level."),
    ("The reviewer notes `acmeMu` already guards this.", "`acmeMu` already guards this."),
    ("The author points out that the path is validated upstream.",
     "The path is validated upstream."),
    ("The reviewer also recommends adding a regression test.",
     "Adding a regression test."),
    ("the commenter prefers a single helper.", "A single helper."),
])
def test_statement_frames_are_dropped(framed, expected):
    assert deframe(framed) == expected


@pytest.mark.parametrize("framed,expected", [
    ("The reviewer questions whether `createRpc` should be exported.",
     "It is unclear whether `createRpc` should be exported."),
    ("The reviewer asks why the lock is global.", "It is unclear why the lock is global."),
    ("The reviewer wonders if this handles the empty case.",
     "It is unclear whether this handles the empty case."),
])
def test_question_frames_keep_the_question(framed, expected):
    """Dropping 'questions whether' would assert what was only asked."""
    assert deframe(framed) == expected


@pytest.mark.parametrize("text", [
    "Guard against circular `cause` chains; `render` recurses without a depth limit.",
    "The function returns the wrong type on the error path.",   # 'The function', not a frame
    "Reviewer-facing docs are out of date here.",
    "",
])
def test_unframed_text_is_untouched(text):
    assert deframe(text) == text
    assert not is_framed(text)


def test_is_framed_matches_what_deframe_changes():
    s = "The reviewer suggests that we add a test."
    assert is_framed(s) and deframe(s) != s


@pytest.mark.parametrize("framed,expected", [
    ("The reviewer questions the need for a separate helper.",
     "There is a question about the need for a separate helper."),
    ("The reviewer raises a concern that the lock is global.", "The lock is global."),
    ("The reviewer raises the question of whether this is reachable.",
     "It is unclear whether this is reachable."),
    ("The reviewer agrees that the fallback is correct.", "The fallback is correct."),
])
def test_residual_frames_from_v11_output(framed, expected):
    assert deframe(framed) == expected
