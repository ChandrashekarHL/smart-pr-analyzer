"""Tests for the evaluation harness.

Deliberately model-free: these cover the plumbing that silently corrupts a
result - resume bookkeeping, batching, and the statistics - so they run on a
laptop in under a second without torch installed.
"""

from __future__ import annotations

import pytest

from prana.data import jsonl
from prana.eval.generate import batches, completed_ids, pending
from prana.eval.score import bootstrap_ci, empty_rate, summarise

# ------------------------------------------------------------- resume

def _pred(i, lang="go", pred="looks fine"):
    return {"id": f"r/p-{i}", "language": lang, "repo_name": "r/p",
            "reference": "human text here", "prediction": pred}


def test_resume_skips_completed_rows(tmp_path):
    path = tmp_path / "p.jsonl"
    jsonl.write(path, [_pred(0), _pred(1)])
    rows = [{"id": f"r/p-{i}"} for i in range(5)]
    assert [r["id"] for r in pending(rows, completed_ids(path))] == [
        "r/p-2", "r/p-3", "r/p-4"
    ]


def test_resume_on_missing_file_generates_everything(tmp_path):
    rows = [{"id": "a"}, {"id": "b"}]
    assert pending(rows, completed_ids(tmp_path / "nope.jsonl")) == rows


def test_partial_row_is_regenerated(tmp_path):
    """A row written without a prediction - e.g. a crash mid-batch - must not
    count as done, or the gap is silently carried into the metrics."""
    path = tmp_path / "p.jsonl"
    jsonl.write(path, [{"id": "r/p-0", "prediction": None}])
    assert pending([{"id": "r/p-0"}], completed_ids(path)) == [{"id": "r/p-0"}]


def test_pending_preserves_order():
    rows = [{"id": str(i)} for i in range(10)]
    got = pending(rows, {"3", "7"})
    assert [r["id"] for r in got] == ["0", "1", "2", "4", "5", "6", "8", "9"]


# ------------------------------------------------------------- batching

def test_batches_cover_every_row_exactly_once():
    rows = list(range(23))
    out = [r for b in batches(rows, 8) for r in b]
    assert out == rows


def test_final_batch_is_short_not_padded():
    assert [len(b) for b in batches(list(range(23)), 8)] == [8, 8, 7]


def test_batches_of_empty_list():
    assert list(batches([], 4)) == []


# ------------------------------------------------------------- statistics

def test_bootstrap_ci_brackets_the_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    mean, lo, hi = bootstrap_ci(values)
    assert lo < mean < hi
    assert abs(mean - 0.3) < 1e-9


def test_bootstrap_ci_is_deterministic():
    values = [0.1, 0.9] * 50
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_bootstrap_ci_narrows_with_more_data():
    few = bootstrap_ci([0.2, 0.8] * 5)
    many = bootstrap_ci([0.2, 0.8] * 500)
    assert (many[2] - many[1]) < (few[2] - few[1])


@pytest.mark.parametrize("values", [[], [0.42]])
def test_bootstrap_ci_degenerate_inputs(values):
    mean, lo, hi = bootstrap_ci(values)
    assert lo == hi == mean


def test_empty_rate_counts_unusable_predictions():
    """A base model frequently emits nothing. ROUGE alone would hide that."""
    assert empty_rate(["", "   ", "ok", "a real review of some length"]) == 0.75


# ------------------------------------------------------------- summary

def test_summary_breaks_down_by_language():
    rows = [_pred(0, "go"), _pred(1, "go"), _pred(2, "rust")]
    summary = summarise(rows, {"rougeL": [0.5, 0.3, 0.9]})
    assert summary["n"] == 3
    assert summary["by_language"]["go"]["n"] == 2
    assert summary["by_language"]["rust"]["rougeL"]["mean"] == pytest.approx(0.9)


def test_summary_reports_confidence_intervals():
    rows = [_pred(i) for i in range(20)]
    summary = summarise(rows, {"rougeL": [i / 20 for i in range(20)]})
    lo, hi = summary["overall"]["rougeL"]["ci95"]
    assert lo < summary["overall"]["rougeL"]["mean"] < hi


def test_summary_records_length_gap():
    """A base model that rambles for 400 words scores badly for reasons ROUGE
    does not explain; the length medians make that visible."""
    rows = [{**_pred(0), "prediction": "word " * 100, "reference": "short human note"}]
    summary = summarise(rows, {"rougeL": [0.1]})
    assert summary["overall"]["pred_words_median"] == 100
    assert summary["overall"]["ref_words_median"] == 3


def test_rows_with_missing_language_are_bucketed_not_dropped():
    rows = [{**_pred(0), "language": None}, _pred(1, "go")]
    summary = summarise(rows, {"rougeL": [0.4, 0.6]})
    assert summary["by_language"]["unknown"]["n"] == 1
    assert sum(v["n"] for v in summary["by_language"].values()) == 2


# ------------------------------------------------------------- decode artefacts

@pytest.mark.parametrize("raw,clean", [
    ("Use errors.Is here. <|EOT|>", "Use errors.Is here."),
    ("<|begin▁of▁sentence|>Looks fine.<|end▁of▁sentence|>", "Looks fine."),
    ("a <| b", "a <| b"),                     # not a marker: no closing |>
    ("x < y || y > z", "x < y || y > z"),      # ordinary code stays intact
])
def test_special_token_markers_are_stripped(raw, clean):
    """<|EOT|> leaked into every v0.3.1 prediction because the rebuilt
    tokenizer did not know it was special. Scoring must never see it."""
    from prana.eval.generate import clean_completion
    assert clean_completion(raw) == clean
