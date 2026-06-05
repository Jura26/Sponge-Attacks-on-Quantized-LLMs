#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_DEST="${SPONGE_HF_DIR:-$ROOT_DIR/backend/hf}"

python3 "$ROOT_DIR/scripts/download_hf_models.py" --dest "$HF_DEST"

if [[ -f "$ROOT_DIR/backend/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/backend/.venv/bin/activate"
fi

python3 "$ROOT_DIR/backend/main.py" &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID"' EXIT

npm --prefix "$ROOT_DIR/frontend" start
