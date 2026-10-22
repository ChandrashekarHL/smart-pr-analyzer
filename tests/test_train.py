"""Tests for the training plumbing.

Torch-free where possible. The LoRA-target tests use tiny fake modules rather
than a real model, because the bug they guard against - detection returning an
empty set and LoRA attaching to nothing - is about type matching, not weights.
"""

from __future__ import annotations

import pytest

from prana.train import (
    EXCLUDED_TARGETS,
    PREFERRED_TARGETS,
    Config,
    collate,
    pack,
    supported_kwargs,
)

torch = pytest.importorskip("torch")


# ------------------------------------------------------------- LoRA targets

class _FakeConv1D(torch.nn.Module):
    """Stands in for transformers' Conv1D, which GPT-2 uses for projections."""


def _model(**children) -> torch.nn.Module:
    m = torch.nn.Module()
    for name, mod in children.items():
        m.add_module(name, mod)
    return m


def test_detects_llama_style_projections():
    from prana.train import detect_target_modules
    model = _model(**{n: torch.nn.Linear(4, 4) for n in PREFERRED_TARGETS})
    assert set(detect_target_modules(model)) == set(PREFERRED_TARGETS)


def test_preferred_order_is_used_when_available():
    from prana.train import detect_target_modules
    model = _model(q_proj=torch.nn.Linear(4, 4), v_proj=torch.nn.Linear(4, 4),
                   weird=torch.nn.Linear(4, 4))
    assert detect_target_modules(model) == ["q_proj", "v_proj"]


def test_falls_back_to_whatever_is_there():
    from prana.train import detect_target_modules
    model = _model(c_attn=torch.nn.Linear(4, 4), c_proj=torch.nn.Linear(4, 4))
    assert detect_target_modules(model) == ["c_attn", "c_proj"]


def test_output_heads_excluded():
    from prana.train import linear_module_names
    model = _model(q_proj=torch.nn.Linear(4, 4), lm_head=torch.nn.Linear(4, 4))
    names = linear_module_names(model)
    assert "q_proj" in names
    assert not (names & EXCLUDED_TARGETS)


def test_empty_detection_raises_instead_of_returning_nothing():
    """PEFT accepts an empty target list and trains nothing. This is the bug
    the first Spark smoke test hit, via GPT-2's Conv1D projections."""
    from prana.train import detect_target_modules
    with pytest.raises(SystemExit, match="no adaptable linear modules"):
        detect_target_modules(_model(norm=torch.nn.LayerNorm(4)))


# ------------------------------------------------------------- packing

def _ex(n_prompt, n_answer, start=0):
    ids = list(range(start, start + n_prompt + n_answer))
    return {"input_ids": ids, "labels": [-100] * n_prompt + ids[n_prompt:]}


def test_packing_preserves_every_token_and_label():
    examples = [_ex(3, 2, s) for s in (0, 100, 200)]
    blocks = pack(examples, max_len=64)
    assert [t for b in blocks for t in b["input_ids"]] == [
        t for e in examples for t in e["input_ids"]
    ]
    assert [t for b in blocks for t in b["labels"]] == [
        t for e in examples for t in e["labels"]
    ]


def test_packing_respects_max_len():
    blocks = pack([_ex(2, 2, s * 10) for s in range(10)], max_len=8)
    assert all(len(b["input_ids"]) <= 8 for b in blocks)


def test_packing_keeps_prompt_masked_across_the_join():
    """Concatenation must not turn a masked prompt into a training target."""
    blocks = pack([_ex(3, 2), _ex(3, 2, 50)], max_len=64)
    labels = [t for b in blocks for t in b["labels"]]
    assert labels.count(-100) == 6


def test_packing_of_empty_input():
    assert pack([], max_len=16) == []


# ------------------------------------------------------------- collate

def test_collate_pads_right_and_masks_padding():
    batch = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
             {"input_ids": [4, 5], "labels": [-100, 5]}]
    out = collate(batch, pad_token_id=0)
    assert out["input_ids"].shape == (2, 3)
    # Padding must never be a loss target.
    assert out["labels"][1].tolist() == [-100, 5, -100]
    assert out["attention_mask"][1].tolist() == [1, 1, 0]


# ------------------------------------------------------------- version compat

def test_supported_kwargs_drops_retired_arguments():
    class Old:
        def __init__(self, keep=None, also=None):
            pass
    got = supported_kwargs(Old, {"keep": 1, "also": 2, "evaluation_strategy": "steps"})
    assert got == {"keep": 1, "also": 2}


def test_supported_kwargs_keeps_everything_supported():
    class New:
        def __init__(self, a=None, b=None):
            pass
    assert supported_kwargs(New, {"a": 1, "b": 2}) == {"a": 1, "b": 2}


# ------------------------------------------------------------- config

def test_config_dotted_lookup_and_defaults(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"model": {"name": "x"}, "train": {"lr": 0.001}}', encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.get("model.name") == "x"
    assert cfg.get("train.lr") == 0.001
    assert cfg.get("train.missing", 7) == 7
    assert cfg.get("nothing.at.all") is None


# ------------------------------------------------------------- tokenizer contract

class _StripWhitespace:
    """Behaves like transformers 5's Metaspace-rebuilt DeepSeek tokenizer:
    every space, newline and tab vanishes between encode and decode."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list("".join(text.split()))}
    def decode(self, ids):
        return "".join(ids)


class _Faithful:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}
    def decode(self, ids):
        return "".join(ids)


def test_whitespace_stripping_tokenizer_is_detected():
    """The v0.3 run trained 10.5 hours on whitespace-stripped text because
    nothing checked for this. It must never pass again."""
    from prana.compat import tokenizer_roundtrips
    assert not tokenizer_roundtrips(_StripWhitespace())


def test_faithful_tokenizer_passes():
    from prana.compat import tokenizer_roundtrips
    assert tokenizer_roundtrips(_Faithful())


def test_probe_covers_every_whitespace_kind_code_depends_on():
    from prana.compat import WHITESPACE_PROBE
    assert " " in WHITESPACE_PROBE
    assert "  " in WHITESPACE_PROBE
    assert "\n" in WHITESPACE_PROBE
    assert "\t" in WHITESPACE_PROBE


def _cached(repo):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, "tokenizer.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _cached("deepseek-ai/deepseek-coder-6.7b-instruct"),
                    reason="DeepSeek tokenizer not in the local HF cache")
def test_deepseek_tokenizer_preserves_whitespace_after_load():
    """Integration test against the real model: runs on the Spark, where the
    weights are cached. This is the exact case that broke."""
    from prana.compat import WHITESPACE_PROBE, load_tokenizer
    tok = load_tokenizer("deepseek-ai/deepseek-coder-6.7b-instruct")
    ids = tok(WHITESPACE_PROBE, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == WHITESPACE_PROBE
    assert tok.chat_template, "repair must carry the chat template across"
    assert tok.eos_token, "repair must carry special tokens across"


@pytest.mark.skipif(not _cached("deepseek-ai/deepseek-coder-6.7b-instruct"),
                    reason="DeepSeek tokenizer not in the local HF cache")
def test_deepseek_eot_never_reaches_a_prediction():
    """On transformers 5.5.0 skip_special_tokens leaves <|EOT|> in the text
    (measured; re-registering it on the backend changed nothing). What we
    guarantee is the pipeline's output, not the decoder's behaviour."""
    from prana.compat import load_tokenizer
    from prana.eval.generate import clean_completion
    tok = load_tokenizer("deepseek-ai/deepseek-coder-6.7b-instruct")
    assert tok.eos_token == "<|EOT|>"
    assert "<|EOT|>" in set(tok.all_special_tokens)
    ids = tok("fine. <|EOT|>", add_special_tokens=False)["input_ids"]
    assert "<|EOT|>" not in clean_completion(tok.decode(ids, skip_special_tokens=True))
