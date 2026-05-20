#!/usr/bin/env bash
# Install / verify GPTQModel on AMD ROCm (e.g. RX 9070 XT, gfx1201).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/backend/.venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

if [[ ! -x "$PY" ]]; then
  echo "Create backend venv first: cd backend && python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi

ARCH="${PYTORCH_ROCM_ARCH:-gfx1201}"
echo "ROCm arch: ${ARCH} (override with PYTORCH_ROCM_ARCH)"

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Installing PyTorch ROCm 6.3 wheels..."
  "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
fi

"$PIP" install -r "${ROOT}/backend/requirements.txt"
"$PIP" install -r "${ROOT}/backend/requirements-rocm.txt"

# Source build only if import fails (pip wheel is often CUDA-only).
if ! "$PY" -c "import gptqmodel" 2>/dev/null; then
  echo "Building GPTQModel from source for ROCm..."
  TMP="${TMPDIR:-/tmp}/gptqmodel-rocm-build"
  rm -rf "$TMP"
  git clone --depth 1 https://github.com/ModelCloud/GPTQModel.git "$TMP"
  PYTORCH_ROCM_ARCH="$ARCH" "$PIP" install --no-build-isolation "$TMP"
fi

echo "Verifying GPTQ on GPU (Qwen 0.5B GPTQ smoke test)..."
# Invalid HF_TOKEN breaks public downloads; unset for verify.
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$PY" <<'PY'
from gptqmodel import GPTQModel
import torch

assert torch.cuda.is_available(), "ROCm GPU not visible to PyTorch"
model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", device="cuda:0")
out = model.generate("Hi:", max_new_tokens=4)
print("GPTQ ROCm OK, sample:", out)
PY

echo "Done. Set SPONGE_GPTQ_MODEL_ID in backend/.env for quant_mode=gptq runs."
