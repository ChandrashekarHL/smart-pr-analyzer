"""Tests for the prompt contract shared by training and serving.

The failure this guards against is silent: if serving renders the prompt even
slightly differently from what the adapter was trained on, nothing errors and
quality just quietly drops. The fingerprint turns that into a loud failure,
and the version table lets a v1.0 adapter keep serving after v2 ships.
"""

from __future__ import annotations

import pytest

from prana import prompts

DIFF = "@@ -140,3 +140,4 @@ func resolve()\n context\n-old\n+new one\n+new two\n context2"


# ------------------------------------------------------------- v1 is frozen

def test_v1_fingerprint_is_the_one_v030_and_v10_were_trained_on():
    """Change this constant only by retraining every adapter that carries it."""
    assert prompts.fingerprint("v1") == "9c1ea91e177d"


def test_v1_does_not_number_lines():
    msg = prompts.build_user_message(DIFF, "go", version="v1")
    assert "-old" in msg and "  141 " not in msg


# ------------------------------------------------------------- v2 numbering

def test_v2_is_current_and_differs_from_v1():
    assert prompts.PROMPT_VERSION == "v2"
    assert prompts.fingerprint("v2") != prompts.fingerprint("v1")
    assert prompts.fingerprint("v2") == prompts.PROMPT_FINGERPRINT


def test_number_hunk_follows_new_file_line_numbers():
    out = prompts.number_hunk(DIFF).splitlines()
    assert out[0].startswith("@@")                 # header untouched
    assert out[1] == "  140  context"              # context line gets the number
    assert out[2] == "      -old"                  # removed line: no number
    assert out[3] == "  141 +new one"
    assert out[4] == "  142 +new two"
    assert out[5] == "  143  context2"


def test_number_hunk_leaves_headerless_text_alone():
    assert prompts.number_hunk("just some text\n+x") == "just some text\n+x"


def test_number_hunk_handles_no_newline_marker():
    out = prompts.number_hunk("@@ -1 +1 @@\n+x\n\\ No newline at end of file").splitlines()
    assert out[-1].startswith("      \\ No newline")


def test_v2_user_message_carries_numbers_and_marker():
    msg = prompts.build_user_message(DIFF, "go")
    assert "  141 +new one" in msg
    assert msg.rstrip().endswith(prompts.RESPONSE_MARKER)
    assert msg.count(prompts.RESPONSE_MARKER) == 1


# ------------------------------------------------------------- shared behaviour

@pytest.mark.parametrize("version", ["v1", "v2"])
def test_context_slot_always_rendered(version):
    msg = prompts.build_user_message(DIFF, "go", context=None, version=version)
    assert "Repository context:" in msg and prompts.NO_CONTEXT in msg
    msg2 = prompts.build_user_message(DIFF, "go", context="uses errors.Is", version=version)
    assert "uses errors.Is" in msg2 and prompts.NO_CONTEXT not in msg2


def test_training_example_has_assistant_turn():
    msgs = prompts.build_messages(DIFF, "go", review="Use errors.Is here.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"] == "Use errors.Is here."


def test_generation_prompt_stops_before_assistant():
    assert [m["role"] for m in prompts.build_messages(DIFF, "go")] == ["system", "user"]


def test_unknown_version_rejected():
    with pytest.raises(ValueError, match="unknown prompt version"):
        prompts.build_messages(DIFF, "go", version="v9")


# ------------------------------------------------------------- contract

def test_contract_resolves_each_known_version():
    assert prompts.assert_contract(prompts.fingerprint("v1")) == "v1"
    assert prompts.assert_contract(prompts.fingerprint("v2")) == "v2"


def test_contract_assertion_fails_on_drift():
    with pytest.raises(RuntimeError, match="prompt contract mismatch"):
        prompts.assert_contract("deadbeef1234")


def test_fingerprint_tracks_template_text(monkeypatch):
    altered = prompts.PromptVersion("v2", prompts.CURRENT.system + " Be terse.",
                                    prompts.CURRENT.user_template, True)
    assert altered.fingerprint() != prompts.fingerprint("v2")


# ------------------------------------------------------------- long hunks

def test_long_hunk_is_cut_in_the_middle_and_keeps_the_header():
    """Right-side truncation cut off 'Review this hunk.' and the model continued
    the diff instead of answering - 34 of v1.1's 42 invalid outputs."""
    big = "@@ -1,400 +1,400 @@\n" + "\n".join(f"+line {i} {'x' * 40}" for i in range(400))
    out = prompts.fit_hunk(big, max_chars=2000)
    assert len(out) <= 2000
    assert out.startswith("@@ -1,400")
    assert "lines omitted" in out
    assert out.rstrip().endswith("+line 399 " + "x" * 40)


def test_short_hunk_is_untouched():
    assert prompts.fit_hunk(DIFF) == DIFF


def test_user_message_tail_survives_a_huge_hunk():
    big = "@@ -1,9000 +1,9000 @@\n" + "\n".join(f"+l{i}" for i in range(9000))
    msg = prompts.build_user_message(big, "go")
    assert msg.rstrip().endswith(prompts.RESPONSE_MARKER)
    assert len(msg) < prompts.MAX_HUNK_CHARS + 2000
