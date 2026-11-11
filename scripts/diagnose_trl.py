"""What did the v1.2 TRL/Unsloth path actually train on?

    run-job start prana-diag -- bash -c 'cd ~/prana && PYTHONPATH=src \
        ~/ai/envs/unsloth/bin/python scripts/diagnose_trl.py'

v1.2 lost line localisation (line exact 14.7% -> 1.7%) and its eval loss
plateaued higher than v1.1's. This builds the trainer exactly as
`prana.train` does for `train.trainer: trl` on a sample of rows and checks:

1. supervision - which tokens carry labels: the whole completion and nothing
   else? Is `"line_number": N` inside the supervised span, or cut off?
2. isolation - in a padding-free packed row, does a later example attend to
   the one before it? The logits of a packed segment are compared with the
   same segment run alone. Equal => isolated. Different => cross-example
   attention: position ids restart at 0 but attention is not masked, so every
   numbered line in the hunk has a twin from the neighbouring example.

Writes runs/v1.2/diagnose_trl.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

LINE_SET = re.compile(r'"line_number"\s*:\s*\d')


def segments(position_ids: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each packed example: a new one starts where positions restart."""
    starts = [i for i, p in enumerate(position_ids) if p == 0]
    return [(s, e) for s, e in zip(starts, starts[1:] + [len(position_ids)], strict=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deepseek_v12.yaml")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--out", default="runs/v1.2/diagnose_trl.json")
    args = ap.parse_args()

    # Not UNSLOTH_RETURN_LOGITS: it makes Unsloth silently turn packing off,
    # which is the thing under test. Isolation is measured on the loss instead.
    os.environ["WANDB_DISABLED"] = "true"
    import torch
    import unsloth  # noqa: F401  (must precede transformers, as in prana.train)
    from transformers import TrainingArguments
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from prana.compat import load_tokenizer
    from prana.data import jsonl
    from prana.train import Config, prompt_completion, select_rows, supported_kwargs

    cfg = Config.load(args.config)
    seed = int(cfg.get("train.seed", 13))
    max_seq = int(cfg.get("data.max_seq_length", 2048))
    rows = jsonl.read(Path(cfg.get("data.dir")) / "train.jsonl")[: args.rows]
    rows, _ = select_rows(rows, cfg, seed)

    model, _ = FastLanguageModel.from_pretrained(
        model_name=cfg.get("model.name"), max_seq_length=max_seq,
        dtype=torch.bfloat16, load_in_4bit=bool(cfg.get("model.load_in_4bit", True)))
    tokenizer = load_tokenizer(cfg.get("model.name"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=int(cfg.get("lora.r", 16)), lora_alpha=int(cfg.get("lora.alpha", 32)),
        lora_dropout=float(cfg.get("lora.dropout", 0.05)),
        target_modules=cfg.get("lora.target_modules") or
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", max_seq_length=max_seq,
        random_state=seed)

    rng = random.Random(seed)
    pcs = [p for p in (prompt_completion(tokenizer, r, cfg, rng) for r in rows) if p]
    from datasets import Dataset
    ta = supported_kwargs(TrainingArguments, {
        "output_dir": "runs/diag-trl", "per_device_train_batch_size": 4,
        "max_steps": 1, "report_to": [], "bf16": True, "seed": seed})
    sft = supported_kwargs(SFTConfig, {
        **ta, "max_length": max_seq, "completion_only_loss": True,
        "packing": bool(cfg.get("data.packing", True)),
        "packing_strategy": cfg.get("data.packing_strategy", "bfd"),
        "dataset_num_proc": 1})
    trainer = SFTTrainer(model=model, args=SFTConfig(**sft),
                         train_dataset=Dataset.from_list(pcs), processing_class=tokenizer)

    report: dict = {"rows": len(pcs),
                    "train_dataset_columns": trainer.train_dataset.column_names,
                    "attn_implementation": getattr(model.config, "_attn_implementation", None)}

    # 1. supervision, per packed example
    want_line = sum(bool(LINE_SET.search(p["completion"])) for p in pcs)
    batch = next(iter(trainer.get_train_dataloader()))
    report["batch_keys"] = {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
    sup_line = n_ex = only_completion = 0
    examples = []
    for b in range(batch["input_ids"].shape[0]):
        ids = batch["input_ids"][b].tolist()
        labels = batch["labels"][b].tolist()
        pos = (batch["position_ids"][b].tolist() if "position_ids" in batch
               else list(range(len(ids))))
        for s, e in segments(pos):
            seg_ids, seg_lab = ids[s:e], labels[s:e]
            if all(t == tokenizer.pad_token_id for t in seg_ids):
                continue
            n_ex += 1
            sup = tokenizer.decode([t for t in seg_lab if t != -100])
            text = tokenizer.decode(seg_ids)
            sup_line += bool(LINE_SET.search(sup))
            only_completion += sup.lstrip().startswith("{")
            if len(examples) < 4:
                examples.append({"tokens": e - s, "supervised_tail": sup[-160:],
                                 "text_tail": text[-160:]})
    report["supervision"] = {
        "sample_completions_with_line": f"{want_line}/{len(pcs)}",
        "batch_examples": n_ex,
        "batch_examples_with_line_supervised": sup_line,
        "supervised_span_starts_at_json": f"{only_completion}/{n_ex}",
        "examples": examples}

    # 2. isolation: second packed segment, packed vs alone
    model.eval()
    iso = []
    with torch.no_grad():
        for b in range(batch["input_ids"].shape[0]):
            pos = batch["position_ids"][b].tolist() if "position_ids" in batch else None
            if pos is None:
                break
            segs = [(s, e) for s, e in segments(pos) if e - s > 8]
            if len(segs) < 2:
                continue
            s, e = segs[1]
            # Loss on segment 2's completion only, with segment 1 packed in
            # front of it, versus segment 2 run on its own.
            lab_packed = torch.full_like(batch["labels"][b:b + 1], -100)
            lab_packed[0, s:e] = batch["labels"][b, s:e]
            # Everything the trainer would pass, including packed_seq_lengths
            # (1-D), which is how Unsloth learns where each example ends.
            row = {k: (v[b:b + 1] if v.dim() == 2 else v).cuda() for k, v in batch.items()
                   if hasattr(v, "shape") and k != "labels"}
            packed = float(model(**row, labels=lab_packed.cuda()).loss)
            alone = float(model(input_ids=batch["input_ids"][b:b + 1, s:e].cuda(),
                                position_ids=torch.arange(e - s).unsqueeze(0).cuda(),
                                labels=batch["labels"][b:b + 1, s:e].cuda()).loss)
            iso.append({"row": b, "segments_in_row": len(segs), "segment_tokens": e - s,
                        "first_segment_tokens": segs[0][1] - segs[0][0],
                        "nll_packed": packed, "nll_alone": alone,
                        "diff": packed - alone})
            if len(iso) >= 4:
                break
    report["isolation"] = iso
    report["verdict"] = {
        "line_supervised": sup_line > 0,
        "isolated": bool(iso) and all(abs(x["diff"]) < 0.02 for x in iso)}

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
