"""QLoRA / LoRA fine-tuning, driven entirely by a YAML config.

    # laptop - catches every bug in two minutes, on CPU, no GPU needed
    python -m prana.train --config configs/smoke.yaml

    # DGX Spark, under tmux
    python -m prana.train --config configs/deepseek_final.yaml

Design notes, each one a scar from the v0.2.x notebooks:

* **No TRL SFTTrainer.** Its API moved repeatedly across versions and took the
  notebooks with it. Packing and completion-only masking are ~60 lines we own
  outright and can unit-test, so we do, and depend only on `Trainer`.
* **TrainingArguments is built by signature inspection.** `evaluation_strategy`
  became `eval_strategy`, `tokenizer=` became `processing_class=`. Passing an
  unknown kwarg is a hard crash 40 minutes into a run, so unsupported keys are
  dropped with a warning instead.
* **The prompt comes from `prana.prompts`.** Never inlined here. The fingerprint
  is written into the adapter directory so serving can refuse a mismatch.
* **Nothing about the model is hard-coded.** Change `model.name` and the same
  config trains a different architecture; LoRA target modules are detected from
  the loaded model when not given.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prana.compat import dtype_kwarg, load_tokenizer, tokenizer_kwarg
from prana.data import jsonl
from prana.prompts import PROMPT_FINGERPRINT, PROMPT_VERSION, build_messages

# --------------------------------------------------------------------- config


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, path: str, default=None):
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @classmethod
    def load(cls, path: str | Path) -> Config:
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml
            return cls(yaml.safe_load(text) or {})
        except ImportError:
            # PyYAML is not in the core dependency set; JSON configs work too.
            return cls(json.loads(text))


def environment() -> dict[str, str]:
    """Recorded with every run. A result you cannot reproduce is not a result."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": PROMPT_FINGERPRINT,
    }
    for mod in ("torch", "transformers", "peft", "datasets", "accelerate",
                "bitsandbytes", "trl", "wandb"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "absent"
    try:
        import torch
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda or "?"
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:
        pass
    return env


# ----------------------------------------------------------------- encoding


def target_text(row: dict, cfg: Config) -> str | None:
    """What the model is trained to produce.

    Defaults to the human review comment. Point `data.target_field` at a
    structured column once a labelling pass has produced one - the rest of the
    pipeline does not change.
    """
    field_name = cfg.get("data.target_field", "review_comment")
    value = row.get(field_name)
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def encode_example(tokenizer, row: dict, cfg: Config, rng: random.Random) -> dict | None:
    """Tokenise one row into input_ids plus completion-only labels.

    The prompt/response boundary is found by tokenising the prompt alone and
    masking that many leading positions - exact, and immune to the marker
    string appearing inside a diff.

    Context dropout: a fraction of examples render with an empty RAG slot so
    the model handles both. Without it, a model trained on bare diffs meets an
    unfamiliar distribution the first time Module 3 sends real context.
    """
    answer = target_text(row, cfg)
    if not answer or not answer.strip():
        return None

    context = row.get("rag_context")
    if context and rng.random() < float(cfg.get("data.context_dropout", 0.3)):
        context = None

    prompt_messages = build_messages(row["diff_hunk"], row["language"], context)
    full_messages = [*prompt_messages, {"role": "assistant", "content": answer.strip()}]

    def render(messages, generation_prompt):
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation_prompt
            )
        body = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return body + ("\n\nASSISTANT:" if generation_prompt else "")

    prompt_ids = tokenizer(render(prompt_messages, True), add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(render(full_messages, False), add_special_tokens=False)["input_ids"]

    if len(full_ids) <= len(prompt_ids):
        return None

    max_len = int(cfg.get("data.max_seq_length", 2048))
    full_ids = full_ids[:max_len]
    if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids[: max_len - 1] + [tokenizer.eos_token_id]

    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    if all(x == -100 for x in labels):
        return None  # truncation ate the whole answer
    return {"input_ids": full_ids, "labels": labels}


def pack(examples: list[dict], max_len: int) -> list[dict]:
    """Concatenate short examples into full-length blocks.

    Median example here is ~400 tokens against a 2048 window, so padding every
    sequence to the window would waste roughly 80% of the compute. Labels are
    carried through the concatenation, so completion-only masking survives it.
    """
    blocks: list[dict] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    for ex in examples:
        if cur_ids and len(cur_ids) + len(ex["input_ids"]) > max_len:
            blocks.append({"input_ids": cur_ids, "labels": cur_labels})
            cur_ids, cur_labels = [], []
        cur_ids = cur_ids + ex["input_ids"]
        cur_labels = cur_labels + ex["labels"]
    if cur_ids:
        blocks.append({"input_ids": cur_ids, "labels": cur_labels})
    return blocks


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad to the longest block in the batch. Training pads right; only
    generation needs left padding."""
    import torch
    width = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        gap = width - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_token_id] * gap)
        labels.append(b["labels"] + [-100] * gap)
        mask.append([1] * len(b["input_ids"]) + [0] * gap)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def supported_kwargs(cls, candidate: dict) -> dict:
    """Keep only the kwargs this installed version accepts.

    `evaluation_strategy` -> `eval_strategy`, `tokenizer` -> `processing_class`,
    `save_safetensors` appearing and moving. An unknown kwarg is a TypeError at
    construction, which in practice means a crash after the dataset has already
    been tokenised.
    """
    allowed = set(inspect.signature(cls.__init__).parameters)
    kept, dropped = {}, []
    for key, value in candidate.items():
        if key in allowed:
            kept[key] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"note: {cls.__name__} in this version ignores {dropped}", file=sys.stderr)
    return kept


# --------------------------------------------------------------------- main


# Llama-family naming, which covers DeepSeek-Coder, Qwen and CodeLlama. Used
# as a preference order when present; never assumed.
PREFERRED_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")

# Output heads are excluded: adapting them changes the vocabulary projection
# for no benefit on an instruction-tuning run.
EXCLUDED_TARGETS = frozenset({"lm_head", "score", "classifier", "embed_out"})


def linear_module_names(model) -> set[str]:
    """Leaf names of every adaptable projection in the model.

    Matched by type, not by class name. GPT-2 implements its projections as
    `Conv1D` rather than `Linear`, so a name-based test finds nothing and LoRA
    silently attaches to zero modules - which is how the first smoke test
    failed. bitsandbytes' `Linear4bit` subclasses `nn.Linear`, so isinstance
    covers the quantised path too.
    """
    import torch.nn as nn
    types: list[type] = [nn.Linear]
    try:
        from transformers.pytorch_utils import Conv1D
        types.append(Conv1D)
    except ImportError:
        pass

    return {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, tuple(types)) and name.split(".")[-1] not in EXCLUDED_TARGETS
    }


def detect_target_modules(model) -> list[str]:
    """Choose LoRA target modules for an unknown architecture.

    Raises rather than returning an empty list: PEFT's own error for empty
    targets names neither the model nor the cause, and an adapter attached to
    nothing trains happily while learning nothing.
    """
    found = linear_module_names(model)
    if not found:
        raise SystemExit(
            f"no adaptable linear modules found in {model.__class__.__name__}; "
            "set lora.target_modules explicitly in the config"
        )
    preferred = [t for t in PREFERRED_TARGETS if t in found]
    return preferred or sorted(found)


def build_dataset(tokenizer, rows: list[dict], cfg: Config, seed: int) -> list[dict]:
    rng = random.Random(seed)
    encoded = [e for e in (encode_example(tokenizer, r, cfg, rng) for r in rows) if e]
    if not cfg.get("data.packing", True):
        return encoded
    return pack(encoded, int(cfg.get("data.max_seq_length", 2048)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="override model.name")
    ap.add_argument("--max-steps", type=int, default=None, help="override; use for smoke tests")
    ap.add_argument("--out", default=None, help="override output.dir")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from last checkpoint")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    model_name = args.model or cfg.get("model.name")
    if not model_name:
        raise SystemExit("model.name missing from config")
    outdir = Path(args.out or cfg.get("output.dir", "runs/train"))
    outdir.mkdir(parents=True, exist_ok=True)

    use_unsloth = bool(cfg.get("model.use_unsloth", False))
    if use_unsloth:
        # Must precede transformers. Unsloth patches attention, RoPE and the
        # cross-entropy kernel at import time; imported afterwards it patches
        # already-bound references and silently does nothing.
        import unsloth  # noqa: F401

    import torch
    from transformers import (
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
    )

    env = environment()
    print(json.dumps(env, indent=2))
    (outdir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    seed = int(cfg.get("train.seed", 13))
    random.seed(seed)
    torch.manual_seed(seed)

    use_wandb = bool(cfg.get("wandb.enabled", False)) and not args.no_wandb
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT",
                              cfg.get("wandb.project", "smart-github-pr-analyzer"))

    # ---- data
    data_dir = Path(cfg.get("data.dir", "data/train"))
    train_rows = jsonl.read(data_dir / "train.jsonl")
    eval_rows = jsonl.read(data_dir / "validation.jsonl")
    if cfg.get("data.limit"):
        train_rows = train_rows[: int(cfg.get("data.limit"))]
        eval_rows = eval_rows[: max(int(cfg.get("data.limit")) // 10, 2)]

    # ---- model + tokenizer
    # Loaded before the dataset because Unsloth hands back both together; the
    # weights simply idle for the minute tokenisation takes.
    dtype = getattr(torch, str(cfg.get("model.dtype", "bfloat16")))
    max_seq = int(cfg.get("data.max_seq_length", 2048))
    load_in_4bit = bool(cfg.get("model.load_in_4bit", False))

    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        # Unsloth's tokenizer comes from the same transformers resolution and
        # inherits the same whitespace defect. Replace it with a verified one.
        tokenizer = load_tokenizer(model_name)
    else:
        load_kwargs: dict[str, Any] = {**dtype_kwarg(dtype), "trust_remote_code": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.get("model.bnb_quant_type", "nf4"),
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=bool(cfg.get("model.bnb_double_quant", True)),
            )
        if cfg.get("model.attn_implementation"):
            load_kwargs["attn_implementation"] = cfg.get("model.attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = load_tokenizer(model_name)

    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(tokenizer, train_rows, cfg, seed)
    eval_ds = build_dataset(tokenizer, eval_rows, cfg, seed + 1)
    tokens = sum(len(b["input_ids"]) for b in train_ds)
    print(f"train {len(train_rows)} rows -> {len(train_ds)} blocks, {tokens / 1e6:.1f}M tokens")
    print(f"eval  {len(eval_rows)} rows -> {len(eval_ds)} blocks")

    # ---- LoRA
    targets = cfg.get("lora.target_modules") or detect_target_modules(model)
    print(f"LoRA target modules: {sorted(targets)}")

    lora_r = int(cfg.get("lora.r", 16))
    lora_alpha = int(cfg.get("lora.alpha", 32))
    lora_dropout = float(cfg.get("lora.dropout", 0.05))
    grad_ckpt = bool(cfg.get("train.gradient_checkpointing", True))

    if use_unsloth:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(targets),
            bias="none",
            # Unsloth's own checkpointing is what buys most of its memory win.
            use_gradient_checkpointing="unsloth" if grad_ckpt else False,
            max_seq_length=max_seq,
            random_state=seed,
        )
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=grad_ckpt
            )
        model = get_peft_model(model, LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(targets),
            bias="none",
            task_type="CAUSAL_LM",
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={lora_r} alpha={lora_alpha} "
          f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # ---- trainer
    ta = supported_kwargs(TrainingArguments, {
        "output_dir": str(outdir),
        "per_device_train_batch_size": int(cfg.get("train.batch_size", 4)),
        "per_device_eval_batch_size": int(cfg.get("train.eval_batch_size", 4)),
        "gradient_accumulation_steps": int(cfg.get("train.grad_accum", 8)),
        "num_train_epochs": float(cfg.get("train.epochs", 2)),
        "max_steps": int(args.max_steps if args.max_steps is not None
                         else cfg.get("train.max_steps", -1)),
        "learning_rate": float(cfg.get("train.lr", 1e-4)),
        "lr_scheduler_type": cfg.get("train.scheduler", "cosine"),
        "warmup_ratio": float(cfg.get("train.warmup_ratio", 0.03)),
        "weight_decay": float(cfg.get("train.weight_decay", 0.0)),
        "max_grad_norm": float(cfg.get("train.max_grad_norm", 1.0)),
        "logging_steps": int(cfg.get("train.logging_steps", 10)),
        "save_steps": int(cfg.get("train.save_steps", 200)),
        "save_total_limit": int(cfg.get("train.save_total_limit", 3)),
        "eval_steps": int(cfg.get("train.eval_steps", 200)),
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "gradient_checkpointing": bool(cfg.get("train.gradient_checkpointing", True)),
        "bf16": dtype is torch.bfloat16 and torch.cuda.is_available(),
        "fp16": dtype is torch.float16 and torch.cuda.is_available(),
        "optim": cfg.get("train.optim", "adamw_torch"),
        "seed": seed,
        "report_to": ["wandb"] if use_wandb else [],
        "run_name": cfg.get("wandb.run_name", outdir.name),
        "save_safetensors": True,
        "load_best_model_at_end": False,
    })

    trainer_kwargs = supported_kwargs(Trainer, {
        "model": model,
        "args": TrainingArguments(**ta),
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": lambda b: collate(b, tokenizer.pad_token_id),
        **tokenizer_kwarg(tokenizer),
    })

    trainer = Trainer(**trainer_kwargs)
    result = trainer.train(resume_from_checkpoint=args.resume or None)

    adapter_dir = outdir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Serving reads this and refuses to run against a different prompt build.
    (adapter_dir / "prana_contract.json").write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": PROMPT_FINGERPRINT,
        "base_model": model_name,
        "target_field": cfg.get("data.target_field", "review_comment"),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "pre_tokenizer": type(tokenizer.backend_tokenizer.pre_tokenizer).__name__
            if getattr(tokenizer, "backend_tokenizer", None) is not None else None,
            "whitespace_roundtrip": True,  # load_tokenizer refuses to return otherwise
        },
        "data_dir": str(data_dir),
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
                 "target_modules": list(targets), "unsloth": use_unsloth},
        "trainable_params": trainable,
        "total_params": total,
        "train_metrics": result.metrics,
        "environment": env,
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nadapter -> {adapter_dir}")
    print(f"next: python -m prana.eval.generate --model {model_name} "
          f"--adapter {adapter_dir} --split test_ood --out {outdir}/eval")


if __name__ == "__main__":
    main()
