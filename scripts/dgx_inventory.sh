#!/usr/bin/env bash
# Read-only inventory of the DGX Spark. Creates nothing, changes nothing.
# Run first, before any project directory is made or any package installed.
#
#   ssh dgxspark1 'bash -s' < scripts/dgx_inventory.sh | tee dgx_inventory.txt

set -uo pipefail
line() { printf '\n=== %s ===\n' "$1"; }

line "HOST"
hostname; uname -srm; . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
echo "uptime:$(uptime -p 2>/dev/null)"

line "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null \
  || echo "nvidia-smi unavailable"

line "CPU / MEMORY"
nproc; free -h 2>/dev/null | head -3

line "DISK"
# Training artifacts are large; confirm headroom before pulling a 6.7B model.
df -h / /home 2>/dev/null | sort -u

line "CUDA"
(nvcc --version 2>/dev/null | tail -2) || echo "nvcc not on PATH"
ls -d /usr/local/cuda* 2>/dev/null

line "PYTHON ENVIRONMENTS"
# The handoff names ~/ai/envs/pytorch and ~/ai/envs/unsloth. Leave both alone.
ls -d "$HOME"/ai/envs/* 2>/dev/null || echo "no ~/ai/envs"
for env in "$HOME"/ai/envs/*; do
  [ -x "$env/bin/python" ] || continue
  printf '\n-- %s\n' "$env"
  "$env/bin/python" -c "
import importlib.metadata as md, platform
print('python', platform.python_version())
for p in ('torch','transformers','peft','trl','datasets','accelerate',
          'bitsandbytes','unsloth','wandb','pyyaml','rouge-score','bert-score'):
    try: print(f'  {p:16s}{md.version(p)}')
    except Exception: print(f'  {p:16s}-')
try:
    import torch
    print('  cuda available :', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('  device         :', torch.cuda.get_device_name(0))
        print('  capability     :', '.'.join(map(str, torch.cuda.get_device_capability(0))))
        print('  torch cuda     :', torch.version.cuda)
except Exception as e:
    print('  torch import failed:', e)
" 2>&1 | sed 's/^/  /'
done

line "EXISTING PR-ANALYZER WORK"
# Do not delete or overwrite any of this.
find "$HOME" -maxdepth 3 \
     \( -iname '*pr_analy*' -o -iname '*pr-analy*' -o -iname '*cleaned_reviews*' \
        -o -iname '*prana*' -o -iname '*qwen*' -o -iname '*adapter*' \) \
     -not -path '*/.cache/*' -not -path '*/site-packages/*' 2>/dev/null | head -40

line "DATASETS ON DISK"
find "$HOME" -maxdepth 4 \( -name '*.jsonl' -o -name '*cleaned*.csv' \) \
     -not -path '*/.cache/*' -size +100k 2>/dev/null \
  | head -20 | while read -r f; do printf '%10s  %s\n' "$(du -h "$f" | cut -f1)" "$f"; done

line "HUGGINGFACE CACHE"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || echo "none"
ls "$HOME/.cache/huggingface/hub" 2>/dev/null | head -10

line "DIRECTORIES TO LEAVE UNTOUCHED"
for d in jupyterlab nvidia-workbench unsloth_compiled_cache; do
  [ -e "$HOME/$d" ] && echo "  present: ~/$d"
done

line "GPU IN USE RIGHT NOW"
# Training is a temporary batch workload; check nothing long-lived owns the GPU.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
  || echo "none reported"

line "TOOLING"
for t in git tmux rsync python3 pip3 uv; do
  printf '  %-8s %s\n' "$t" "$(command -v $t || echo '-')"
done

line "DONE"
echo "nothing was created or modified"
