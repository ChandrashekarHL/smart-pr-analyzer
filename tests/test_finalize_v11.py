"""v1.1 additions to the finalizer: deframing and RAG-context merge."""

from __future__ import annotations

import json

from prana.data import jsonl
from prana.data.finalize import finalize_rows, load_context
from prana.schema import FIELDS


def _row(i, explanation="The reviewer suggests that nil be guarded."):
    so = {"file_path": f"f{i}.go", "line_number": i, "severity": "warning",
          "issue_type": "BUG", "explanation": explanation, "suggestion": "Guard it."}
    return {"id": f"r/p_{i}-{i}", "language": "go", "review_comment": "Guard nil.",
            "diff_hunk": "@@ -1 +1 @@\n+x", "label_model": "gemini-3.8-flash",
            "label": {"is_review": True, **{k: so[k] for k in ("severity", "issue_type",
                                                              "explanation", "suggestion")}},
            "structured_output": so}


def test_framing_is_removed_by_default_and_counted():
    kept, stats = finalize_rows([_row(1)])
    assert kept[0]["structured_output"]["explanation"] == "Nil be guarded."
    assert stats["deframed_explanation"] == 1
    assert json.loads(kept[0]["target_text"])["explanation"] == "Nil be guarded."


def test_framing_can_be_kept_for_reproducing_v10():
    kept, stats = finalize_rows([_row(1)], deframe_labels=False)
    assert kept[0]["structured_output"]["explanation"].startswith("The reviewer suggests")
    assert stats["deframed_explanation"] == 0


def test_target_text_uses_reasoning_first_order():
    kept, _ = finalize_rows([_row(2)])
    assert list(json.loads(kept[0]["target_text"])) == list(FIELDS)
    assert FIELDS[0] == "explanation" and FIELDS[2] == "severity"


def test_context_sidecar_is_merged_by_id(tmp_path):
    side = tmp_path / "train.jsonl"
    jsonl.write(side, [{"id": "r/p_1-1", "rag_context": "Errors are wrapped with %w here."},
                       {"id": "r/p_9-9", "rag_context": "unused"},
                       {"id": "r/p_2-2", "rag_context": "   "}])      # blank: ignored
    ctx = load_context(tmp_path, "train")
    assert ctx == {"r/p_1-1": "Errors are wrapped with %w here.", "r/p_9-9": "unused"}
    kept, stats = finalize_rows([_row(1), _row(2)], context=ctx)
    assert kept[0]["rag_context"].startswith("Errors are wrapped")
    assert "rag_context" not in kept[1]
    assert stats["with_context"] == 1


def test_missing_sidecar_means_no_context(tmp_path):
    assert load_context(tmp_path, "validation") == {}
    assert load_context(None, "train") == {}
