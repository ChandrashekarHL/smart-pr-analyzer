"""Version shims for the transformers API.

The laptop and the DGX Spark are three major versions apart (4.35 against
5.5.0), and this project has already lost time to that class of breakage. Each
shim here names the version that moved and why the naive call fails silently
rather than loudly, which is what makes these worth centralising.
"""

from __future__ import annotations

import sys
from typing import Any


def transformers_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


def dtype_kwarg(dtype: Any) -> dict[str, Any]:
    """`from_pretrained` renamed `torch_dtype` to `dtype` in transformers 5.

    `from_pretrained` accepts **kwargs, so the retired spelling raises nothing -
    it is simply ignored, and the model loads in fp32. On a 6.7B that is 27 GB
    instead of 13 GB and roughly half the training throughput, with no error to
    explain either.
    """
    return {"dtype": dtype} if transformers_major() >= 5 else {"torch_dtype": dtype}


# Spaces, a double space, newlines and tabs: everything code review depends on.
WHITESPACE_PROBE = "I think this is fine.\n\tif x:\n\t\treturn y  # two  spaces"


def tokenizer_roundtrips(tokenizer: Any) -> bool:
    """True when encode -> decode preserves whitespace exactly."""
    ids = tokenizer(WHITESPACE_PROBE, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids) == WHITESPACE_PROBE


def _tokenizer_json(name: str):
    from pathlib import Path
    local = Path(name) / "tokenizer.json"
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(name, "tokenizer.json"))
    except Exception:
        return None


def load_tokenizer(name: str) -> Any:
    """Load a tokenizer and prove it preserves whitespace before anything uses it.

    transformers 5 resolves DeepSeek-Coder's declared `LlamaTokenizerFast` to a
    rebuilt `LlamaTokenizer` with a Metaspace pre-tokenizer, ignoring the
    repo's byte-level `tokenizer.json`. Metaspace deletes every space, newline
    and tab. Nothing errors: the v0.3 run trained for 10.5 hours on
    whitespace-stripped diffs and reviews, reached a believable eval loss, and
    the damage only showed when generated text came out as
    "Ithinkthisisabitof...".

    So: load normally, round-trip a whitespace probe, rebuild from the repo's
    own tokenizer.json if the probe fails, and refuse to continue if the rebuilt
    tokenizer fails too. Training on a broken tokenizer must be impossible,
    not merely unlikely.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer_roundtrips(tok):
        return tok

    path = _tokenizer_json(name)
    if path is None:
        raise SystemExit(
            f"tokenizer for {name} drops whitespace and no tokenizer.json is "
            "available to rebuild it from; refusing to continue"
        )

    fixed = PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        bos_token=tok.bos_token,
        eos_token=tok.eos_token,
        unk_token=tok.unk_token,
        pad_token=tok.pad_token,
    )
    fixed.chat_template = tok.chat_template
    fixed.model_max_length = tok.model_max_length

    # The rebuilt tokenizer knows the vocabulary but not which entries are
    # special, and `skip_special_tokens` strips only what it has been told
    # about. DeepSeek's instruct turns end with <|EOT|>; left unregistered it
    # leaked into every decoded review of the v0.3.1 eval as literal text.
    specials = set(getattr(tok, "additional_special_tokens", None) or [])
    try:
        import json
        for entry in json.loads(path.read_text(encoding="utf-8")).get("added_tokens", []):
            if entry.get("special") and entry.get("content"):
                specials.add(entry["content"])
    except (OSError, ValueError):
        pass
    core = {fixed.bos_token, fixed.eos_token, fixed.unk_token, fixed.pad_token}
    extra = sorted(s for s in specials if s and s not in core)
    if extra:
        fixed.add_special_tokens({"additional_special_tokens": extra})

    # Measured on transformers 5.5.0: <|EOT|> is in `all_special_tokens`, is
    # flagged special on the backend AddedToken as well, and
    # `decode(skip_special_tokens=True)` still prints it. Re-registering it on
    # the backend was tried and changed nothing, so it is not attempted here.
    # Generation strips markers post-decode (prana.eval.generate.clean_completion);
    # this probe only records the behaviour in the log, so a future transformers
    # that fixes it gets noticed.
    if fixed.eos_token:
        probe = fixed("x " + fixed.eos_token, add_special_tokens=False)["input_ids"]
        if fixed.eos_token in fixed.decode(probe, skip_special_tokens=True):
            print(f"note: {fixed.eos_token} survives skip_special_tokens on this "
                  "transformers; generation strips it post-decode", file=sys.stderr, flush=True)

    if not tokenizer_roundtrips(fixed):
        raise SystemExit(
            f"tokenizer for {name} drops whitespace even when rebuilt from "
            f"{path}; refusing to continue"
        )
    print(f"note: {type(tok).__name__} from {name} dropped whitespace; "
          f"rebuilt from {path.name}", file=sys.stderr, flush=True)
    return fixed


def tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    """`Trainer(tokenizer=...)` became `processing_class=` in 4.46 and the old
    name was removed in 5. Unlike the dtype case this one does raise."""
    return (
        {"processing_class": tokenizer}
        if transformers_major() >= 5
        else {"tokenizer": tokenizer}
    )
