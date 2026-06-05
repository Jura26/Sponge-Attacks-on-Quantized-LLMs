# Sponge Attacks on Quantized LLMs

A research tool for running sponge attacks against large language models — both at full precision and GPTQ 4-bit quantized — to measure the energy/compute impact of adversarial inputs.

---

## Requirements

| Component | Minimum |
|-----------|---------|
| OS | Ubuntu 22.04 / 24.04 |
| Python | 3.10 – 3.12 |
| Node.js | 18+ |
| RAM | 16 GB (32 GB recommended for 7B models) |
| CPU-only VRAM | — |
| AMD GPU VRAM | 16 GB+ (RX 9070 XT or Instinct) |

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd Sponge-Attacks-on-Quantized-LLMs
```

---

### 2. Backend — CPU (no GPU)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch (smaller download, no GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Rest of dependencies
pip install -r requirements.txt
```

---

### 2. Backend — AMD GPU (ROCm)

#### Step 1: Install ROCm

**Ubuntu 22.04 (Jammy):**
```bash
wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo apt install ./amdgpu-install_6.3.60300-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $USER
reboot
```

**Ubuntu 24.04 (Noble):**
```bash
wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/noble/amdgpu-install_6.3.60300-1_all.deb
sudo apt install ./amdgpu-install_6.3.60300-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $USER
reboot
```

> **RX 9070 XT users:** This card is RDNA 4 (gfx1201). ROCm 6.3 is the first version with official support. Make sure you're on ROCm 6.3+.
> **WARNING:** The pre-built PyTorch ROCm wheels do NOT include gfx1201 kernels yet.
> The backend automatically detects this at startup (GPU kernel smoke test) and falls back to CPU.
> The `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround commonly suggested online **crashes the GPU and reboots your system** on RDNA 4 — do NOT use it.
> To use the GPU you must build PyTorch from source with `PYTORCH_ROCM_ARCH=gfx1201`.

#### Step 2: Verify ROCm sees your GPU

```bash
rocminfo | grep -E "Name|gfx"
```

You should see your GPU name and something like `gfx1201` (RX 9070 XT).

#### Step 3: Create venv and install ROCm PyTorch

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate

# PyTorch built for ROCm 6.3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Rest of dependencies
pip install -r requirements.txt
```

#### Step 4: Install GPTQModel from source (for GPTQ quantization on GPU)

AutoGPTQ has been abandoned. Its official successor is **GPTQModel**, which has ROCm support and is actively maintained. The pip wheel is CUDA-only, so for ROCm you must build from source:

```bash
# Make sure the venv is active
git clone https://github.com/ModelCloud/GPTQModel.git /tmp/gptqmodel
cd /tmp/gptqmodel

# --no-build-isolation lets pip see the already-installed torch in your venv
# Set your GPU arch: RX 9070 XT = gfx1201, RX 7900 XTX = gfx1100, MI300X = gfx942
PYTORCH_ROCM_ARCH=gfx1201 pip install --no-build-isolation .
cd -
```

#### Step 4b: Optional - Build ROCm bitsandbytes for NF4 4-bit (recommended for compare mode)

AMD's ROCm quantization guide recommends building ROCm bitsandbytes from source and
setting the target GPU arch explicitly.

```bash
# Make sure the backend venv is active
git clone --recurse https://github.com/ROCm/bitsandbytes.git /tmp/bitsandbytes
cd /tmp/bitsandbytes
git checkout rocm_enabled_multi_backend

pip install -r requirements-dev.txt

# RX 9070 XT = gfx1201
cmake -DBNB_ROCM_ARCH="gfx1201" -DCOMPUTE_BACKEND=hip -S .
make -j"$(nproc)"
python setup.py install
cd -
```

Then enable ROCm bitsandbytes in this project:

```bash
export SPONGE_ENABLE_BNB_ROCM=1
```

Without this env var, the backend intentionally falls back to CPU int8 during
the quantized phase to avoid `invalid device function` crashes.

#### Step 5: Verify GPU is visible to PyTorch

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
AMD Radeon RX 9070 XT
```

---

### 3. Add swap space (strongly recommended for 7B+ models)

Loading large models causes RAM spikes. Swap prevents the Linux OOM killer from terminating other processes (like VS Code):

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### 4. Download a model

Models are downloaded automatically on first run, or you can pre-download them:

```bash
# Requires HuggingFace account — set your token first
export HF_TOKEN=hf_your_token_here

pip install huggingface_hub
hf download mistralai/Mistral-7B-Instruct-v0.3
```

Models are cached at `~/.cache/huggingface/hub/`. Mistral 7B is ~14 GB.

For smaller models that don't require authentication, GPT-2 variants work out of the box:
```bash
hf download gpt2
hf download gpt2-xl
```

### 4b. Download HF FP16 models locally (new HF backend)

For the HF FP16 backend, download the three requested models into a local
folder (default: `backend/hf`) so the backend can load them without network:

```bash
export HF_TOKEN=hf_your_token_here   # required for Llama
python scripts/download_hf_models.py
```

Or download + start backend/frontend in one step:

```bash
bash scripts/start_with_hf_models.sh
```

---

### 5. Frontend

```bash
cd frontend
npm install
npm start
```

The frontend runs at `http://localhost:3000`.

---

## Running

**Important:** Run the backend in a standalone terminal (not VS Code's integrated terminal) to avoid VS Code being killed by the OOM killer during model loading.

Open a terminal (Ctrl+Alt+T):

```bash
cd /path/to/Sponge-Attacks-on-Quantized-LLMs/backend
source .venv/bin/activate
python3 main.py
```

The backend API runs at `http://localhost:8000`.

In a second terminal:

```bash
cd /path/to/Sponge-Attacks-on-Quantized-LLMs/frontend
npm start
```

Then open `http://localhost:3000` in your browser.

---

## Attack types

| Attack | Description |
|--------|-------------|
| **Evolutionary Sponge** | Genetic algorithm that evolves prompts to maximise inference time / CPU-GPU load |
| **Context Exhaustion** | Floods the model's context window to force maximum KV-cache usage |
| **AutoDoS (Tree-based)** | Tree-search that expands adversarial prompts depth-first to find high-cost inputs |

## Quantization modes

| Mode | How to trigger | Notes |
|------|---------------|-------|
| `fp16` | Default on GPU | Full precision, ~14 GB VRAM for 7B |
| `bnb-nf4` | `quant_mode=bnb-nf4` | 4-bit NF4 bitsandbytes |
| `bnb-fp4` | `quant_mode=bnb-fp4` | 4-bit FP4 bitsandbytes |
| `bnb-int8` | `quant_mode=bnb-int8` | 8-bit bitsandbytes |
| `gguf-llamacpp` | `quant_mode=gguf-llamacpp` | Local GGUF model via llama.cpp backend (third runtime option) |
| `gptq-int4` | `quant_mode=gptq` with GPTQ model IDs | Only for pre-quantized GPTQ model repos (ID typically contains `gptq`) |
| `hf-fp16` | `quant_mode=hf-fp16` | Local HF FP16 models via transformers (requires pre-download) |
| `int8-cpu` | `quant_mode=int8-cpu` | PyTorch dynamic int8 quantization (CPU, may spike RAM on large models) |
| `int1-sim` | `quant_mode=int1-sim` | Experimental 1-bit simulation (binarized linear weights, not native 1-bit kernels) |
| `fp32` | CPU fallback | Used automatically when no GPU or GPU kernels non-functional |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace API token — required for gated models (Mistral, LLaMA) and faster downloads |
| `HF_HOME` | Override the model cache directory (default: `~/.cache/huggingface`) |
| `SPONGE_ENABLE_BNB_ROCM` | Set to `1` to enable ROCm bitsandbytes NF4 path after source-building bitsandbytes for your GPU arch |
| `SPONGE_GGUF_PATH` | Absolute path to local `.gguf` model file for `gguf-llamacpp` mode |
| `SPONGE_GGUF_CTX` | Context size for llama.cpp backend (default: `4096`) |
| `SPONGE_GGUF_GPU_LAYERS` | Number of offloaded layers in llama.cpp (default: `-1`, auto/max) |
| `SPONGE_HF_DIR` | Local folder containing HF FP16 snapshots (default: `backend/hf`) |
| `SPONGE_HF_MODEL_LLAMA3` | Override HF repo for Llama3 (default: `meta-llama/Llama-3.2-3B`) |
| `SPONGE_HF_MODEL_QWEN` | Override HF repo for Qwen (default: `Qwen/Qwen2.5-3B-Instruct`) |
| `SPONGE_HF_MODEL_HUNYUAN` | Override HF repo for Hunyuan (default: `tencent/Hunyuan-4B-Instruct`) |
| `SPONGE_HF_TRUST_REMOTE_CODE` | Set to `1` only if required by a model (default: `0`) |
| `SPONGE_ALLOW_CPU_INT8_FALLBACK` | Set to `1` to allow CPU int8 fallback on large models (disabled by default to avoid RAM spikes) |

```bash
export HF_TOKEN=hf_your_token_here
export HF_HOME=/data/models   # optional, to store models on a larger drive
export SPONGE_ENABLE_BNB_ROCM=1   # optional, only after building ROCm bitsandbytes
export SPONGE_GGUF_PATH=/absolute/path/to/model.gguf   # required for gguf-llamacpp mode
export SPONGE_GGUF_CTX=4096   # optional llama.cpp context
export SPONGE_GGUF_GPU_LAYERS=-1   # optional llama.cpp offload setting
export SPONGE_HF_DIR=/absolute/path/to/hf_models   # local HF snapshots
export SPONGE_ALLOW_CPU_INT8_FALLBACK=1   # optional, enables CPU int8 fallback
```

---

## GPU architecture reference

| GPU | Architecture | ROCm arch flag |
|-----|-------------|----------------|
| RX 9070 XT | RDNA 4 | `gfx1201` |
| RX 7900 XTX / 7900 XT | RDNA 3 | `gfx1100` |
| RX 7800 XT / 7700 XT | RDNA 3 | `gfx1101` |
| MI300X | CDNA 3 | `gfx942` |
| MI250X | CDNA 2 | `gfx90a` |
