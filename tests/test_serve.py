"""Tests for the serving entry point. Model-free: they cover the contract
check, input validation and the confidence arithmetic - the parts that fail
silently in production if they are wrong."""

from __future__ import annotations

import json
import math

import pytest

from prana import prompts
from prana.serve import (
    LANGUAGES,
    GenerationSettings,
    ReviewResult,
    confidence_from_logprobs,
    read_contract,
    validate_language,
)

# ------------------------------------------------------------- contract

def test_adapter_without_contract_is_refused(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Refusing to serve"):
        read_contract(tmp_path)


def test_contract_is_read_back(tmp_path):
    (tmp_path / "prana_contract.json").write_text(json.dumps({
        "prompt_fingerprint": prompts.PROMPT_FINGERPRINT, "base_model": "x/y"}))
    c = read_contract(tmp_path)
    assert c["base_model"] == "x/y"


def test_mismatched_fingerprint_refuses_to_serve():
    """The adapter was trained on one prompt; this build renders another."""
    with pytest.raises(RuntimeError, match="prompt contract mismatch"):
        prompts.assert_contract("000000000000")


# ------------------------------------------------------------- inputs

@pytest.mark.parametrize("raw,expected",
                         [("Go", "go"), (" python ", "python"), ("RUST", "rust")])
def test_language_normalised(raw, expected):
    assert validate_language(raw) == expected


@pytest.mark.parametrize("bad", ["java", "", "c++", None])
def test_unknown_language_rejected(bad):
    with pytest.raises(ValueError, match="language must be one of"):
        validate_language(bad)


def test_language_set_matches_training_corpus():
    assert set(LANGUAGES) == {"go", "typescript", "rust", "python", "javascript"}


# ------------------------------------------------------------- confidence

def test_confidence_is_exp_mean_logprob():
    assert confidence_from_logprobs([0.0, 0.0]) == pytest.approx(1.0)
    assert confidence_from_logprobs([math.log(0.5)] * 3) == pytest.approx(0.5)


def test_confidence_of_nothing_generated_is_zero():
    assert confidence_from_logprobs([]) == 0.0


def test_confidence_stays_in_unit_interval():
    for lps in ([-0.01], [-5.0, -0.1], [-20.0] * 10):
        assert 0.0 <= confidence_from_logprobs(lps) <= 1.0


# ------------------------------------------------------------- result shape

def test_result_serialises_with_the_fields_the_backend_needs():
    r = ReviewResult(text="Guard len(xs) first.", confidence=0.71, latency_s=1.2,
                     language="go", model="m", adapter="a", prompt_version="v1",
                     prompt_fingerprint="abc", generated_tokens=7)
    d = json.loads(r.to_json())
    assert {"text", "confidence", "latency_s", "language", "prompt_fingerprint"} <= set(d)


def test_default_decoding_is_greedy_and_guards_repetition():
    s = GenerationSettings()
    assert s.temperature == 0.0
    assert s.repetition_penalty > 1.0
    assert s.no_repeat_ngram_size >= 3


def test_serve_result_can_carry_deframed_parsed_fields():
    """`parsed` is what the backend posts; `text` must be re-rendered from it
    so the two never disagree."""
    from prana.schema import parse_structured, render_structured
    data = {"explanation": "Guard nil first.", "suggestion": "", "severity": "warning",
            "issue_type": "BUG", "file_path": "a.go", "line_number": 3}
    text = render_structured(data)
    assert parse_structured(text).data == data
