"""Which RoPE did the v1.1 adapter actually train with? Completion-only loss of
the adapter on validation rows, loaded with the config's linear x4 scaling and
with unscaled RoPE. The lower loss is the training condition.

    run-job start prana-rope-loss -- ~/ai/envs/unsloth/bin/python scripts/rope_loss_check.py
"""
import json
import random
import sys

import torch

sys.path.insert(0, "src")
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from prana.compat import load_tokenizer  # noqa: E402
from prana.data import jsonl  # noqa: E402
from prana.eval.generate import unscaled_rope_config  # noqa: E402
from prana.train import Config, encode_example  # noqa: E402

MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
ADAPTER = "runs/v1.1/adapter"
N = 200

tok = load_tokenizer(MODEL)
cfg = Config.load("configs/deepseek_v11.yaml")
rows = jsonl.read("data/v1.1/validation.jsonl")[:N]
rng = random.Random(0)
cfg.raw["data"]["context_dropout"] = 1.0           # served without context
examples = [e for e in (encode_example(tok, r, cfg, rng) for r in rows) if e]

out = {}
for mode in ("scaled", "unscaled"):
    kw = {"config": unscaled_rope_config(MODEL)} if mode == "unscaled" else {}
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, **kw).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for e in examples:
            ids = torch.tensor([e["input_ids"]], device="cuda")
            labels = torch.tensor([e["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=labels).loss
            n = int((labels[0, 1:] != -100).sum())
            total += float(loss) * n
            count += n
    out[mode] = round(total / count, 4)
    print(mode, "completion loss", out[mode], "over", count, "tokens", flush=True)
    del model, base
    torch.cuda.empty_cache()
print(json.dumps(out))
