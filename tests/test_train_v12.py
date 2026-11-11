"""v1.2 training additions: TRL prompt/completion rendering and row selection.
No torch needed."""

from __future__ import annotations

import json
import random

from prana.train import Config, prompt_completion, select_rows


class TemplateTokenizer:
    """Mimics a chat template the way DeepSeek's behaves for this purpose:
    the generation prompt is a prefix of the full conversation."""
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return out + ("<assistant>" if add_generation_prompt else "<|EOT|>")


class BrokenTemplate(TemplateTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "X" + super().apply_chat_template(messages, tokenize, add_generation_prompt) \
            if add_generation_prompt else super().apply_chat_template(messages, tokenize, False)


def _cfg(**data):
    return Config({"data": {"target_field": "structured_output", "context_dropout": 0.0, **data}})


def _row(i, issue="BUG", reviewer="alice", comment="Looks off."):
    return {"id": f"r{i}", "diff_hunk": "@@ -1 +1 @@\n+x = 1", "language": "python",
            "reviewer": reviewer, "review_comment": comment,
            "structured_output": {"explanation": f"e{i}", "suggestion": "", "severity": "info",
                                  "issue_type": issue, "file_path": None, "line_number": None}}


def test_prompt_plus_completion_is_exactly_the_training_string():
    from prana.prompts import build_messages
    tok = TemplateTokenizer()
    row = _row(1)
    pc = prompt_completion(tok, row, _cfg(), random.Random(0))
    target = json.dumps(row["structured_output"], ensure_ascii=False)
    expected = tok.apply_chat_template(
        [*build_messages(row["diff_hunk"], row["language"], None),
         {"role": "assistant", "content": target}], add_generation_prompt=False)
    assert pc["prompt"] + pc["completion"] == expected       # what the custom path trains on
    assert pc["prompt"].endswith("<assistant>") and pc["completion"].endswith("<|EOT|>")


def test_unsplittable_template_is_skipped_not_mangled():
    assert prompt_completion(BrokenTemplate(), _row(1), _cfg(), random.Random(0)) is None


def test_select_rows_drops_bots_and_oversamples_deterministically():
    rows = [_row(1, "SECURITY"), _row(2), _row(3, reviewer="2dc80ef9"),
            _row(4, comment='![P2 Badge](https://img.shields.io/badge/P2-yellow) x')]
    cfg = Config({"data": {"drop_bot_rows": True, "oversample": {"SECURITY": 3}}})
    out, stats = select_rows(rows, cfg, seed=13)
    assert stats == {"input": 4, "dropped_bot_rows": 2, "oversampled_extra_rows": 2, "output": 4}
    assert sorted(r["id"] for r in out) == ["r1", "r1", "r1", "r2"]
    again, _ = select_rows(rows, cfg, seed=13)
    assert [r["id"] for r in again] == [r["id"] for r in out]


def test_select_rows_is_a_no_op_by_default():
    rows = [_row(1), _row(2, reviewer="2dc80ef9")]
    out, stats = select_rows(rows, Config({}), seed=1)
    assert out == rows and stats == {"input": 2, "output": 2}


def test_target_is_the_structured_json():
    pc = prompt_completion(TemplateTokenizer(), _row(7), _cfg(), random.Random(0))
    body = pc["completion"].removesuffix("</assistant><|EOT|>")
    assert json.loads(body)["explanation"] == "e7"
