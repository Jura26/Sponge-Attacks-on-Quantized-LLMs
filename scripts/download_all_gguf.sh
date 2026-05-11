#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Download all available .gguf files for selected model families from Hugging Face.
# Defaults to: gpt2-small, opt-6.7b, mistral7b
#
# For OPT-6.7B, if no working public GGUF repo is found, this script can
# optionally build GGUF locally from facebook/opt-6.7b using llama.cpp tools.
#
# Usage:
#   ./scripts/download_all_gguf.sh
#   ./scripts/download_all_gguf.sh --dest /home/models
#   ./scripts/download_all_gguf.sh --models gpt2-small,mistral7b
#   ./scripts/download_all_gguf.sh --dry-run
#   ./scripts/download_all_gguf.sh --variant-set best
#   ./scripts/download_all_gguf.sh --variant-set all
#   ./scripts/download_all_gguf.sh --opt-local-build never
#   ./scripts/download_all_gguf.sh --gpt2-local-build auto
#   ./scripts/download_all_gguf.sh --mistral-local-build auto
#   HF_TOKEN=... ./scripts/download_all_gguf.sh
#
# Optional override for OPT repo (if you know a working one):
#   OPT67_GGUF_REPO=<owner/repo> ./scripts/download_all_gguf.sh

DEST_DIR="${DEST_DIR:-/home/jura/models}"
MODELS="${MODELS:-gpt2-small,opt-6.7b,mistral7b}"
DRY_RUN="${DRY_RUN:-0}"
OPT_LOCAL_BUILD="${OPT_LOCAL_BUILD:-auto}"  # auto | always | never
GPT2_LOCAL_BUILD="${GPT2_LOCAL_BUILD:-auto}"  # auto | always | never
MISTRAL_LOCAL_BUILD="${MISTRAL_LOCAL_BUILD:-auto}"  # auto | always | never
VARIANT_SET="${VARIANT_SET:-best}"      # best | all
OPT_HF_MODEL_ID="${OPT_HF_MODEL_ID:-facebook/opt-6.7b}"
GPT2_HF_MODEL_ID="${GPT2_HF_MODEL_ID:-openai-community/gpt2}"
MISTRAL_HF_MODEL_ID="${MISTRAL_HF_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
OPT_WORK_DIR="${OPT_WORK_DIR:-}"
GPT2_WORK_DIR="${GPT2_WORK_DIR:-}"
MISTRAL_WORK_DIR="${MISTRAL_WORK_DIR:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/jura/source/Sponge-Attacks-on-Quantized-LLMs/backend/lib/llama.cpp}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST_DIR="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --variant-set)
      VARIANT_SET="$2"
      shift 2
      ;;
    --opt-local-build)
      OPT_LOCAL_BUILD="$2"
      shift 2
      ;;
    --gpt2-local-build)
      GPT2_LOCAL_BUILD="$2"
      shift 2
      ;;
    --mistral-local-build)
      MISTRAL_LOCAL_BUILD="$2"
      shift 2
      ;;
    --opt-workdir)
      OPT_WORK_DIR="$2"
      shift 2
      ;;
    --gpt2-workdir)
      GPT2_WORK_DIR="$2"
      shift 2
      ;;
    --mistral-workdir)
      MISTRAL_WORK_DIR="$2"
      shift 2
      ;;
    --llama-cpp-dir)
      LLAMA_CPP_DIR="$2"
      shift 2
      ;;
    --opt-hf-model)
      OPT_HF_MODEL_ID="$2"
      shift 2
      ;;
    --gpt2-hf-model)
      GPT2_HF_MODEL_ID="$2"
      shift 2
      ;;
    --mistral-hf-model)
      MISTRAL_HF_MODEL_ID="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$DEST_DIR"
if [[ -z "$OPT_WORK_DIR" ]]; then
  OPT_WORK_DIR="$DEST_DIR/opt-6.7b-hf"
fi
if [[ -z "$GPT2_WORK_DIR" ]]; then
  GPT2_WORK_DIR="$DEST_DIR/gpt2-hf"
fi
if [[ -z "$MISTRAL_WORK_DIR" ]]; then
  MISTRAL_WORK_DIR="$DEST_DIR/mistral-7b-hf"
fi

if [[ "$OPT_LOCAL_BUILD" != "auto" && "$OPT_LOCAL_BUILD" != "always" && "$OPT_LOCAL_BUILD" != "never" ]]; then
  echo "--opt-local-build must be one of: auto, always, never" >&2
  exit 1
fi
if [[ "$GPT2_LOCAL_BUILD" != "auto" && "$GPT2_LOCAL_BUILD" != "always" && "$GPT2_LOCAL_BUILD" != "never" ]]; then
  echo "--gpt2-local-build must be one of: auto, always, never" >&2
  exit 1
fi
if [[ "$MISTRAL_LOCAL_BUILD" != "auto" && "$MISTRAL_LOCAL_BUILD" != "always" && "$MISTRAL_LOCAL_BUILD" != "never" ]]; then
  echo "--mistral-local-build must be one of: auto, always, never" >&2
  exit 1
fi

if [[ "$VARIANT_SET" != "best" && "$VARIANT_SET" != "all" ]]; then
  echo "--variant-set must be one of: best, all" >&2
  exit 1
fi

download_file() {
  local repo="$1"
  local file="$2"
  local out="$DEST_DIR/$file"
  local url="https://huggingface.co/${repo}/resolve/main/${file}?download=true"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] $url -> $out"
    return 0
  fi

  echo "Downloading: $file"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 -C - -o "$out" "$url"
}

ensure_opt_dependencies() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Would ensure dependencies: git cmake python3 pip huggingface_hub"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "Missing dependency: git" >&2
    exit 1
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    echo "Missing dependency: cmake" >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Missing dependency: python3" >&2
    exit 1
  fi

  python3 - <<'PY'
try:
    import huggingface_hub  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
  if [[ $? -ne 0 ]]; then
    echo "Installing huggingface_hub (user site)..."
    python3 -m pip install --user --upgrade huggingface_hub
  fi
}

build_opt67b_locally() {
  echo "Using local OPT GGUF build path from $OPT_HF_MODEL_ID"

  local out_dir="$DEST_DIR"
  local work_dir="$OPT_WORK_DIR"
  local llama_cpp_dir="$LLAMA_CPP_DIR"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Would clone/update llama.cpp at $llama_cpp_dir"
    echo "[dry-run] Would build llama.cpp tools with cmake"
    echo "[dry-run] Would download HF model: $OPT_HF_MODEL_ID -> $work_dir"
    echo "[dry-run] Would produce:"
    if [[ "$VARIANT_SET" == "all" ]]; then
      echo "[dry-run]   $out_dir/OPT-6.7B-f16.gguf"
    fi
    echo "[dry-run]   $out_dir/OPT-6.7B-Q8_0.gguf"
    echo "[dry-run]   $out_dir/OPT-6.7B-Q6_K.gguf"
    echo "[dry-run]   $out_dir/OPT-6.7B-Q5_K_M.gguf"
    echo "[dry-run]   $out_dir/OPT-6.7B-Q4_K_M.gguf"
    echo "[dry-run]   $out_dir/OPT-6.7B-Q3_K_M.gguf"
    echo "[dry-run]   $out_dir/OPT-6.7B-Q2_K.gguf"
    return 0
  fi

  ensure_opt_dependencies
  mkdir -p "$out_dir"
  mkdir -p "$work_dir"
  mkdir -p "$(dirname "$llama_cpp_dir")"

  if [[ ! -d "$llama_cpp_dir/.git" ]]; then
    echo "Cloning llama.cpp into $llama_cpp_dir"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$llama_cpp_dir"
  else
    echo "Updating llama.cpp in $llama_cpp_dir"
    git -C "$llama_cpp_dir" pull --ff-only
  fi

  echo "Building llama.cpp tools"
  cmake -S "$llama_cpp_dir" -B "$llama_cpp_dir/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$llama_cpp_dir/build" -j

  echo "Downloading HF model: $OPT_HF_MODEL_ID"
  OPT_HF_MODEL_ID="$OPT_HF_MODEL_ID" OPT_WORK_DIR="$work_dir" HF_TOKEN="${HF_TOKEN:-}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ.get("OPT_HF_MODEL_ID", "facebook/opt-6.7b"),
    local_dir=os.environ.get("OPT_WORK_DIR", "/home/jura/models/opt-6.7b-hf"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None,
)
PY

  local converter="$llama_cpp_dir/convert_hf_to_gguf.py"
  if [[ ! -f "$converter" ]]; then
    echo "convert_hf_to_gguf.py not found at $converter" >&2
    return 1
  fi

  local f16_out="$out_dir/OPT-6.7B-f16.gguf"
  echo "Converting HF -> F16 GGUF"
  python3 "$converter" "$work_dir" --outfile "$f16_out" --outtype f16

  local quant_bin="$llama_cpp_dir/build/bin/llama-quantize"
  if [[ ! -x "$quant_bin" ]]; then
    quant_bin="$llama_cpp_dir/build/bin/quantize"
  fi
  if [[ ! -x "$quant_bin" ]]; then
    echo "Quantize binary not found in llama.cpp build/bin" >&2
    return 1
  fi

  quantize_one() {
    local qtype="$1"
    local out="$2"
    echo "Quantizing -> $qtype : $out"
    "$quant_bin" "$f16_out" "$out" "$qtype"
  }

  quantize_one Q8_0 "$out_dir/OPT-6.7B-Q8_0.gguf"
  quantize_one Q6_K "$out_dir/OPT-6.7B-Q6_K.gguf"
  quantize_one Q5_K_M "$out_dir/OPT-6.7B-Q5_K_M.gguf"
  quantize_one Q4_K_M "$out_dir/OPT-6.7B-Q4_K_M.gguf"
  quantize_one Q3_K_M "$out_dir/OPT-6.7B-Q3_K_M.gguf"
  quantize_one Q2_K "$out_dir/OPT-6.7B-Q2_K.gguf"

  # In storage-saving mode, remove intermediate f16 file after quantization.
  if [[ "$VARIANT_SET" == "best" ]]; then
    rm -f "$f16_out"
  fi
}

build_hf_f16_locally() {
  local model_key="$1"
  local hf_id="$2"
  local work_dir="$3"
  local out_name="$4"
  local llama_cpp_dir="$LLAMA_CPP_DIR"
  local out_dir="$DEST_DIR"
  local out_path="$out_dir/$out_name"

  if [[ -f "$out_path" ]]; then
    echo "Base GGUF already exists: $out_path"
    return 0
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Would clone/update llama.cpp at $llama_cpp_dir"
    echo "[dry-run] Would build llama.cpp tools with cmake"
    echo "[dry-run] Would download HF model: $hf_id -> $work_dir"
    echo "[dry-run] Would produce: $out_path"
    return 0
  fi

  echo "Using local $model_key GGUF build path from $hf_id"
  ensure_opt_dependencies
  mkdir -p "$out_dir"
  mkdir -p "$work_dir"
  mkdir -p "$(dirname "$llama_cpp_dir")"

  if [[ ! -d "$llama_cpp_dir/.git" ]]; then
    echo "Cloning llama.cpp into $llama_cpp_dir"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$llama_cpp_dir"
  else
    echo "Updating llama.cpp in $llama_cpp_dir"
    git -C "$llama_cpp_dir" pull --ff-only
  fi

  echo "Building llama.cpp tools"
  cmake -S "$llama_cpp_dir" -B "$llama_cpp_dir/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$llama_cpp_dir/build" -j

  echo "Downloading HF model: $hf_id"
  HF_MODEL_ID="$hf_id" HF_WORK_DIR="$work_dir" HF_TOKEN="${HF_TOKEN:-}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ.get("HF_MODEL_ID"),
    local_dir=os.environ.get("HF_WORK_DIR"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None,
)
PY

  local converter="$llama_cpp_dir/convert_hf_to_gguf.py"
  if [[ ! -f "$converter" ]]; then
    echo "convert_hf_to_gguf.py not found at $converter" >&2
    return 1
  fi

  echo "Converting HF -> F16 GGUF: $out_path"
  python3 "$converter" "$work_dir" --outfile "$out_path" --outtype f16
}

list_gguf_files() {
  local repo="$1"
  python - "$repo" <<'PY'
import json
import sys
import urllib.request

repo = sys.argv[1]
url = f"https://huggingface.co/api/models/{repo}"
try:
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
except Exception:
    sys.exit(0)

files = [
    s.get("rfilename", "")
    for s in data.get("siblings", [])
    if s.get("rfilename", "").endswith(".gguf")
]
for name in sorted(files):
    print(name)
PY
}

filter_variants() {
  local model="$1"
  local files_input="$2"

  if [[ "$VARIANT_SET" == "all" ]]; then
    printf "%s\n" "$files_input"
    return 0
  fi

  # Storage-saving preset: one file per target quantization level.
  case "$model" in
    gpt2-small)
      printf "%s\n" "$files_input" | grep -E 'gpt2\.(Q8_0|Q6_K|Q5_K_M|Q4_K_M|Q3_K_M|Q2_K|f16|f32)\.gguf$' || true
      ;;
    mistral7b)
      printf "%s\n" "$files_input" | grep -E 'Mistral-7B-Instruct-v0\.3-(Q8_0|Q6_K|Q5_K_M|Q4_K_M|Q3_K_M|Q2_K|f16)\.gguf$' || true
      ;;
    opt-6.7b)
      printf "%s\n" "$files_input" | grep -E '(Q8_0|Q6_K|Q5_K_M|Q4_K_M|Q3_K_M|Q2_K|f16|f32)\.gguf$' || true
      ;;
    *)
      printf "%s\n" "$files_input"
      ;;
  esac
}

pick_repo_with_gguf() {
  local candidates_csv="$1"
  IFS=',' read -r -a candidates <<< "$candidates_csv"
  for repo in "${candidates[@]}"; do
    repo="${repo// /}"
    [[ -z "$repo" ]] && continue
    local files
    files="$(list_gguf_files "$repo" || true)"
    if [[ -n "$files" ]]; then
      echo "$repo"
      return 0
    fi
  done
  return 1
}

# Known-good repositories (as of 2026-04-10):
# - gpt2-small: QuantFactory/gpt2-GGUF
#   Note: this repo does not currently include f16/f32; set GPT2_GGUF_REPO if needed.
# - mistral7b: bartowski/Mistral-7B-Instruct-v0.3-GGUF
# - opt-6.7b: no consistently discoverable public GGUF repo found automatically;
#             use OPT67_GGUF_REPO override if you have one.
model_repo_candidates() {
  local model="$1"
  case "$model" in
    gpt2-small)
      if [[ -n "${GPT2_GGUF_REPO:-}" ]]; then
        echo "$GPT2_GGUF_REPO"
      else
        echo "QuantFactory/gpt2-GGUF"
      fi
      ;;
    mistral7b)
      echo "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
      ;;
    opt-6.7b)
      if [[ -n "${OPT67_GGUF_REPO:-}" ]]; then
        echo "$OPT67_GGUF_REPO"
      else
        # Keep likely historical candidates for convenience.
        echo "TheBloke/OPT-6.7B-GGUF,RichardErkhov/facebook_-_opt-6.7b-gguf,QuantFactory/opt-6.7b-GGUF"
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

IFS=',' read -r -a selected_models <<< "$MODELS"

echo "Destination: $DEST_DIR"
echo "Selected models: $MODELS"
[[ $DRY_RUN -eq 1 ]] && echo "Mode: dry-run"
echo "OPT local build mode: $OPT_LOCAL_BUILD"
echo "GPT-2 local build mode: $GPT2_LOCAL_BUILD"
echo "Mistral local build mode: $MISTRAL_LOCAL_BUILD"
echo "Variant set: $VARIANT_SET"

declare -A summary

for raw_model in "${selected_models[@]}"; do
  model="${raw_model// /}"
  [[ -z "$model" ]] && continue

  echo
  echo "=== $model ==="

  candidates="$(model_repo_candidates "$model" || true)"
  if [[ -z "$candidates" ]]; then
    echo "Skipping unknown model key: $model"
    summary["$model"]="unknown model key"
    continue
  fi

  if ! repo="$(pick_repo_with_gguf "$candidates")"; then
    echo "No working GGUF repo found for $model."
    if [[ "$model" == "opt-6.7b" ]]; then
      if [[ "$OPT_LOCAL_BUILD" == "always" || "$OPT_LOCAL_BUILD" == "auto" ]]; then
        if build_opt67b_locally; then
          summary["$model"]="local build from $OPT_HF_MODEL_ID"
        else
          summary["$model"]="local build failed"
        fi
      else
        echo "Tip: set --opt-local-build auto|always, or export OPT67_GGUF_REPO=<owner/repo>."
        summary["$model"]="no repo found"
      fi
    else
      summary["$model"]="no repo found"
    fi
    continue
  fi

  if [[ "$model" == "opt-6.7b" && "$OPT_LOCAL_BUILD" == "always" ]]; then
    if build_opt67b_locally; then
      summary["$model"]="local build from $OPT_HF_MODEL_ID"
    else
      summary["$model"]="local build failed"
    fi
    continue
  fi

  echo "Using repo: $repo"
  files="$(list_gguf_files "$repo")"
  files="$(filter_variants "$model" "$files")"
  if [[ -z "$files" ]]; then
    echo "No matching GGUF files after applying variant filter for $model."
    summary["$model"]="no files after filter"
    continue
  fi
  count=0
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    download_file "$repo" "$f"
    count=$((count + 1))
  done <<< "$files"

  summary["$model"]="$count files from $repo"

  if [[ "$model" == "gpt2-small" && "$GPT2_LOCAL_BUILD" != "never" ]]; then
    if build_hf_f16_locally "gpt2" "$GPT2_HF_MODEL_ID" "$GPT2_WORK_DIR" "gpt2-f16.gguf"; then
      summary["$model"]="${summary["$model"]} + base f16"
    else
      summary["$model"]="${summary["$model"]} + base f16 failed"
    fi
  fi

  if [[ "$model" == "mistral7b" && "$MISTRAL_LOCAL_BUILD" != "never" ]]; then
    if build_hf_f16_locally "mistral" "$MISTRAL_HF_MODEL_ID" "$MISTRAL_WORK_DIR" "Mistral-7B-Instruct-v0.2-f16.gguf"; then
      summary["$model"]="${summary["$model"]} + base f16"
    else
      summary["$model"]="${summary["$model"]} + base f16 failed"
    fi
  fi
done

echo
echo "=== Summary ==="
for k in "${!summary[@]}"; do
  echo "$k: ${summary[$k]}"
done
