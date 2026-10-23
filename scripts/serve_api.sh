#!/usr/bin/env bash
# Serve the review API through the GPU queue. It holds the GPU while running,
# so stop it before any training or evaluation job:
#   cd ~/prana && PYTHONPATH=src run-job start prana-api -- bash scripts/serve_api.sh runs/v1.1/adapter
#   run-job stop prana-api
# Logs (one JSON line per request): run-job logs prana-api
set -uo pipefail
cd "$HOME/prana" || exit 1
export PYTHONPATH=src
ADAPTER="${1:-runs/v1.1/adapter}"
PORT="${2:-8100}"
exec "$HOME/ai/envs/unsloth/bin/python" -m prana.api --adapter "$ADAPTER" --port "$PORT"
