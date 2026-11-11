"""Generate reviews for a held-out split.

Separated from scoring on purpose: generation is the expensive half and must
survive a dropped SSH session, while scoring is cheap and gets re-run many
times as metrics are added. Predictions are appended to disk as they are
produced and the run resumes by skipping ids already present, so an
interrupted job costs only the batch in flight.

    # laptop smoke test - catches every bug in two minutes, on CPU
    python -m prana.eval.generate --model HuggingFaceTB/SmolLM2-135M --limit 8

    # Spark, under tmux
    python -m prana.eval.generate \
        --model deepseek-ai/deepseek-coder-6.7b-instruct \
        --split test_ood --out runs/baseline

Run this against the *base* model before any fine-tuning. It is ablation A,
and without it there is no evidence that fine-tuning changed anything.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from prana.compat import dtype_kwarg, load_tokenizer
from prana.data import jsonl
from prana.prompts import (
    MAX_HUNK_CHARS,
    PROMPT_VERSION,
    assert_contract,
    build_messages,
    fingerprint,
)


def pending(rows: list[dict], done_ids: set[str]) -> list[dict]:
    """Rows still needing generation, in stable order."""
    return [r for r in rows if r["id"] not in done_ids]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for rec in jsonl.iter_read(path):
        if rec.get("prediction") is not None:
            ids.add(rec["id"])
    return ids


# Belt and braces behind skip_special_tokens: any <|...|> marker that still
# reaches the text is a decode artefact, never something a reviewer wrote.
_SPECIAL_MARKER = re.compile(r"<\|[^|<>]{1,40}\|>")


def clean_completion(text: str) -> str:
    return _SPECIAL_MARKER.sub("", text).strip()


def unscaled_rope_config(model_name: str):
    """The base config with linear RoPE scaling removed (rope_type 'default',
    same theta): what Unsloth actually trained with."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    params = dict(getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {})
    theta = params.get("rope_theta") or getattr(cfg, "rope_theta", None) or 10000.0
    unscaled = {"rope_type": "default", "rope_theta": theta}
    if hasattr(cfg, "rope_parameters"):
        cfg.rope_parameters = unscaled
    cfg.rope_scaling = None if not hasattr(cfg, "rope_parameters") else unscaled
    return cfg


def mbr_select(candidates: list[str]) -> str:
    """Minimum-Bayes-risk choice among sampled reviews: the candidate whose
    prose agrees most (mean ROUGE-L) with the other candidates. Valid
    structured outputs are preferred; ties go to the earliest candidate, so
    the choice is deterministic for a given set of samples. It never looks at
    the reference."""
    from prana.eval.score import rouge_l_scores
    from prana.schema import parse_structured

    parsed = [parse_structured(c) for c in candidates]
    pool = [i for i, p in enumerate(parsed) if p.schema_valid] or list(range(len(candidates)))
    if len(pool) == 1:
        return candidates[pool[0]]

    def prose(i: int) -> str:
        p = parsed[i]
        if p.schema_valid and p.data:
            return f"{p.data.get('explanation') or ''} {p.data.get('suggestion') or ''}"
        return candidates[i]
    texts = [prose(i) for i in pool]
    pairs = [(a, b) for a in range(len(pool)) for b in range(len(pool)) if a != b]
    scores = rouge_l_scores([texts[a] for a, _ in pairs], [texts[b] for _, b in pairs])
    totals = [0.0] * len(pool)
    for (a, _), s in zip(pairs, scores, strict=True):
        totals[a] += s
    best = max(range(len(pool)), key=lambda i: (totals[i], -i))
    return candidates[pool[best]]


def batches(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _plain_text(messages: list[dict[str, str]]) -> str:
    """Fallback for tokenizers with no chat template.

    Only used by the tiny models the laptop smoke test runs on. Every real
    base model we target ships a template, and using it matters - so this
    stays a fallback rather than becoming the default path.
    """
    parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    return "\n\n".join(parts) + "\n\nASSISTANT:"


def adapter_prompt_version(adapter: str | None) -> str:
    """The prompt version an adapter was trained on, from its contract file.
    No adapter (base model) means the current version."""
    if adapter is None:
        return PROMPT_VERSION
    contract = Path(adapter) / "prana_contract.json"
    if not contract.exists():
        raise SystemExit(f"{contract} missing: cannot tell which prompt this adapter expects")
    expected = json.loads(contract.read_text(encoding="utf-8"))["prompt_fingerprint"]
    return assert_contract(expected)


# Input budget: the model was trained at 2,048 tokens for prompt AND response.
# Leaving 256 for the answer keeps the prompt inside the distribution it saw.
PROMPT_TOKEN_BUDGET = 2048 - 256
_HUNK_CAPS = (MAX_HUNK_CHARS, 2400, 1600, 1000, 600)


def render_within_budget(tokenizer, diff: str, language: str, context: str | None,
                         version: str | None = None,
                         budget: int = PROMPT_TOKEN_BUDGET) -> str:
    """Render the prompt and shrink the hunk until it fits the token budget.

    The character cap in `prana.prompts` is a guess at tokens; this is the
    measurement. A prompt that is still too long after the smallest cap is
    returned as is - truncation then cuts the hunk, never the template tail,
    because the hunk is already tiny.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    text = ""
    for cap in _HUNK_CAPS:
        messages = build_messages(diff=diff, language=language, context=context,
                                  version=version, max_hunk_chars=cap)
        text = (tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
                if has_template else _plain_text(messages))
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= budget:
            break
    return text


def render_prompts(tokenizer, rows: list[dict], with_context: bool,
                   version: str | None = None) -> list[str]:
    """Render through the shared contract, then through the model's own chat
    template. We never hand-write the special tokens: the tokenizer owns its
    chat format, and hard-coding it silently breaks when the base model changes.
    """
    return [
        render_within_budget(
            tokenizer, row["diff_hunk"], row["language"],
            row.get("rag_context") if with_context else None, version,
        )
        for row in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter to load on top")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--split", default="test_ood",
                    help="test_ood is the honest generalisation set: whole repos "
                         "held out, and 6x less n-gram contamination than test")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 means greedy; keep it there so a baseline "
                         "is reproducible")
    ap.add_argument("--num-beams", type=int, default=1,
                    help="beam search width; 1 = greedy")
    ap.add_argument("--mbr-samples", type=int, default=0,
                    help="minimum-Bayes-risk decoding: sample this many reviews per "
                         "row and keep the consensus one (0 = off)")
    ap.add_argument("--mbr-temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rope", choices=["config", "unscaled"], default="config",
                    help="'unscaled' drops the base config's linear RoPE scaling. Unsloth "
                         "trained every adapter so far with unscaled RoPE (it logs a "
                         "fallback), while plain transformers applies the scaling: this "
                         "flag measures that train/serve mismatch")
    ap.add_argument("--with-context", action="store_true",
                    help="fill the RAG slot from each row's rag_context field")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    args = ap.parse_args()

    # Imported here so --help and the unit tests work without torch installed.
    import torch
    from transformers import AutoModelForCausalLM

    rows = jsonl.read(Path(args.data) / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    preds_path = outdir / f"{args.split}.predictions.jsonl"

    done = completed_ids(preds_path)
    todo = pending(rows, done)
    print(f"{args.split}: {len(rows)} rows, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        print("nothing to do")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "auto": torch.bfloat16 if device == "cuda" else torch.float32,
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }[args.dtype]
    version = adapter_prompt_version(args.adapter)
    print(f"device={device} dtype={dtype} prompt={version}/{fingerprint(version)}")

    tokenizer = load_tokenizer(args.model)
    # Decoder-only generation requires left padding. With right padding the
    # model continues from pad tokens and emits confident nonsense - a bug that
    # produces plausible output and no error, so it is set explicitly.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra = {}
    if args.rope == "unscaled":
        extra["config"] = unscaled_rope_config(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwarg(dtype),
                                                 **extra).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    meta = {
        "model": args.model, "adapter": args.adapter, "split": args.split,
        "prompt_version": version, "prompt_fingerprint": fingerprint(version),
        "with_context": args.with_context, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "device": device, "dtype": str(dtype),
        "num_beams": args.num_beams, "mbr_samples": args.mbr_samples, "rope": args.rope,
        "mbr_temperature": args.mbr_temperature if args.mbr_samples else None,
        "seed": args.seed,
    }
    torch.manual_seed(args.seed)
    (outdir / f"{args.split}.run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    started = time.time()
    generated = 0
    with open(preds_path, "a", encoding="utf-8", newline="\n") as sink:
        for batch in batches(todo, args.batch_size):
            prompts = render_prompts(tokenizer, batch, args.with_context, version)
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(device)

            k = args.mbr_samples
            with torch.no_grad():
                if k:
                    out = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=args.mbr_temperature, num_return_sequences=k,
                        pad_token_id=tokenizer.pad_token_id)
                else:
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        num_beams=args.num_beams,
                        pad_token_id=tokenizer.pad_token_id,
                    )

            # Slice off the prompt: with left padding every sequence in the
            # batch shares the same input width, so one offset is correct.
            completions = tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            if k:
                # num_return_sequences keeps each row's k samples together.
                completions = [mbr_select([clean_completion(c) for c in completions[i:i + k]])
                               for i in range(0, len(completions), k)]

            for row, text in zip(batch, completions, strict=True):
                sink.write(jsonl.dumps({
                    "id": row["id"],
                    "language": row["language"],
                    "repo_name": row["repo_name"],
                    "reference": row["review_comment"],
                    # Present on finalized (v1+) corpora; the structured
                    # evaluator needs the object, not the prose.
                    "reference_structured": row.get("structured_output"),
                    "prediction": clean_completion(text),
                }) + "\n")
            sink.flush()

            generated += len(batch)
            rate = generated / max(time.time() - started, 1e-6)
            remaining = (len(todo) - generated) / max(rate, 1e-6)
            print(f"  {generated}/{len(todo)}  {rate:.2f} rows/s  eta {remaining / 60:.1f} min",
                  flush=True)

    print(f"\nwrote {preds_path}")
    print(f"next: python -m prana.eval.score --predictions {preds_path}")


if __name__ == "__main__":
    main()
