"""Inference entry point for the backend and the GitHub App.

    from prana.serve import Reviewer
    reviewer = Reviewer.load(adapter="runs/v0.3.1/adapter")
    result = reviewer.review(diff_hunk, language="go", context=rag_context)
    result.text, result.confidence, result.latency_s

Or from a shell:

    python -m prana.serve --adapter runs/v0.3.1/adapter --language go \\
        --diff-file hunk.diff [--context-file ctx.txt] > review.json

This is the only place the model is called for serving. It enforces the
prompt contract: the adapter directory carries the fingerprint of the prompt
it was trained against, and loading refuses to proceed if this build renders
a different one. Train/serve prompt drift produces no error and no failing
test, only a quietly worse model, which is why the check is loud.

Confidence is derived, not predicted. The data carries no confidence labels
(a reviewer never writes "I am 0.8 sure"), so training one would be
fiction. The mean token log-probability of the generated text is an honest,
cheap proxy: it is what the model actually believed while writing. Member 4
should treat it as a ranking signal, not a calibrated probability, until the
evaluation lead has measured it against human judgements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from prana.compat import dtype_kwarg, load_tokenizer
from prana.data.deframe import deframe
from prana.eval.generate import clean_completion
from prana.prompts import PROMPT_VERSION, assert_contract, build_messages, fingerprint
from prana.schema import parse_structured, render_structured

BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
LANGUAGES = ("go", "typescript", "rust", "python", "javascript")


@dataclass(frozen=True)
class GenerationSettings:
    """Decoding knobs. Greedy by default so a result is reproducible.

    `repetition_penalty` and `no_repeat_ngram_size` exist because v0.3.1 was
    seen looping a sentence under plain greedy decoding. These defaults are a
    reasonable starting point and are NOT yet tuned on the validation split;
    tuning them on test_ood would leak the test set into a design decision.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4


@dataclass(frozen=True)
class ReviewResult:
    text: str
    confidence: float
    latency_s: float
    language: str
    model: str
    adapter: str | None
    prompt_version: str
    prompt_fingerprint: str
    generated_tokens: int
    # v1.0+ adapters emit the structured schema; `parsed` is the validated
    # object or None, and `parse_error` says why. A prose adapter (v0.3.x)
    # simply yields parsed=None, so the backend handles one shape for both.
    parsed: dict | None = None
    parse_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def read_contract(adapter: str | Path) -> dict:
    path = Path(adapter) / "prana_contract.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: this adapter was not produced by prana.train, "
            "so its prompt format is unknown. Refusing to serve it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_from_logprobs(token_logprobs: list[float]) -> float:
    """exp(mean log-prob) of the generated tokens: 1.0 = every token was the
    model's certain choice, 0.0 = it was guessing throughout."""
    if not token_logprobs:
        return 0.0
    return math.exp(sum(token_logprobs) / len(token_logprobs))


def validate_language(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
    return lang


class Reviewer:
    def __init__(self, model, tokenizer, model_name: str, adapter: str | None,
                 settings: GenerationSettings, prompt_version: str = PROMPT_VERSION):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.adapter = adapter
        self.settings = settings
        # The prompt the adapter was trained on, resolved from its contract.
        # A v1.0 adapter keeps rendering v1 after v2 ships; nothing to redeploy.
        self.prompt_version = prompt_version

    @classmethod
    def load(cls, adapter: str | Path | None, base: str = BASE_MODEL,
             settings: GenerationSettings | None = None,
             load_in_4bit: bool = True) -> Reviewer:
        """4-bit by default. Measured on the GB10 (2026-10-21, same hunk, same
        adapter): bf16 7.24 s, 4-bit 3.34 s, byte-identical parsed output. The
        chip is bandwidth-bound, so a quarter of the weight bytes is a 2.2x
        decode speedup. bf16 remains available for A/B checks."""
        import torch
        from transformers import AutoModelForCausalLM

        prompt_version = PROMPT_VERSION
        if adapter is not None:
            contract = read_contract(adapter)
            prompt_version = assert_contract(contract["prompt_fingerprint"])
            base = contract.get("base_model", base)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tokenizer = load_tokenizer(base)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {**dtype_kwarg(dtype), "trust_remote_code": True}
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        if not load_in_4bit:
            model = model.to(device)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        return cls(model, tokenizer, base, str(adapter) if adapter else None,
                   settings or GenerationSettings(), prompt_version)

    def review(self, diff_hunk: str, language: str, context: str | None = None) -> ReviewResult:
        import torch

        lang = validate_language(language)
        if not diff_hunk or not diff_hunk.strip():
            raise ValueError("diff_hunk is empty")

        messages = build_messages(diff_hunk, lang, context, version=self.prompt_version)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=2048).to(self.model.device)

        s = self.settings
        started = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=s.max_new_tokens,
                do_sample=s.temperature > 0,
                temperature=s.temperature if s.temperature > 0 else None,
                repetition_penalty=s.repetition_penalty,
                no_repeat_ngram_size=s.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - started

        prompt_len = enc["input_ids"].shape[1]
        gen_ids = out.sequences[0, prompt_len:]
        logprobs = []
        for step, tok_id in enumerate(gen_ids.tolist()):
            if tok_id == self.tokenizer.eos_token_id:
                break
            lp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[tok_id].item()
            logprobs.append(lp)

        text = clean_completion(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        parsed = parse_structured(text)
        data = None
        if parsed.ok:
            # The model still narrates in third person on ~30% of outputs
            # ("The reviewer suggests ..."). A posted comment must speak as
            # the reviewer, so the same transform used on training labels is
            # applied here, and `text` is re-rendered to match.
            data = dict(parsed.data)
            data["explanation"] = deframe(data["explanation"])
            data["suggestion"] = deframe(data["suggestion"])
            text = render_structured(data)
        else:
            text = deframe(text)
        return ReviewResult(
            parsed=data,
            parse_error=None if parsed.ok else parsed.error,
            text=text,
            confidence=round(confidence_from_logprobs(logprobs), 4),
            latency_s=round(latency, 3),
            language=lang,
            model=self.model_name,
            adapter=self.adapter,
            prompt_version=self.prompt_version,
            prompt_fingerprint=fingerprint(self.prompt_version),
            generated_tokens=len(logprobs),
        )


def create_app(reviewer: Reviewer):
    """Minimal HTTP wrapper for the backend lead. FastAPI is optional: the
    core package stays dependency-free and this import only runs here."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class ReviewRequest(BaseModel):
        diff_hunk: str
        language: str
        context: str | None = None

    app = FastAPI(title="prana reviewer")

    @app.get("/health")
    def health():
        return {"ok": True, "model": reviewer.model_name, "adapter": reviewer.adapter,
                "prompt_version": reviewer.prompt_version,
                "prompt_fingerprint": fingerprint(reviewer.prompt_version)}

    @app.post("/review")
    def review(req: ReviewRequest):
        try:
            return asdict(reviewer.review(req.diff_hunk, req.language, req.context))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="omit to run the untuned base model")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--language", required=True, choices=LANGUAGES)
    ap.add_argument("--diff-file", required=True)
    ap.add_argument("--context-file", default=None)
    ap.add_argument("--bf16", dest="bf16", action="store_true",
                    help="load full-precision weights (default is 4-bit, 2.2x faster here)")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="kept for compatibility; 4-bit is already the default")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    diff = Path(args.diff_file).read_text(encoding="utf-8")
    context = Path(args.context_file).read_text(encoding="utf-8") if args.context_file else None
    reviewer = Reviewer.load(args.adapter, base=args.base, load_in_4bit=not args.bf16,
                             settings=GenerationSettings(max_new_tokens=args.max_new_tokens))
    result = reviewer.review(diff, args.language, context)
    sys.stdout.write(result.to_json() + "\n")


if __name__ == "__main__":
    main()
