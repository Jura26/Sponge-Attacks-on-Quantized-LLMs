"""Runtime capability checks for quantized backends."""

from __future__ import annotations

import os

from model import gptq_available


def build_quantization_capabilities() -> dict:
    """Return quantization capability metadata for the frontend."""
    import torch

    modes = {
        "gguf-f16": {"supported": True, "reason": "GGUF F16 baseline"},
        "gguf-q8": {"supported": True, "reason": "GGUF Q8_0"},
        "gguf-q6": {"supported": True, "reason": "GGUF Q6_K"},
        "gguf-q5": {"supported": True, "reason": "GGUF Q5_*"},
        "gguf-q4": {"supported": True, "reason": "GGUF Q4_*"},
        "gguf-q3": {"supported": True, "reason": "GGUF Q3_*"},
        "gguf-q2": {"supported": True, "reason": "GGUF Q2_*"},
    }

    gpu_available = torch.cuda.is_available()
    gpu_name = None
    rocm_arch = None
    hip_version = getattr(torch.version, "hip", None)

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            rocm_arch = str(getattr(props, "gcnArchName", "") or "")
        except Exception:
            rocm_arch = ""

    try:
        import llama_cpp  # noqa: F401
    except Exception:
        for key in modes:
            modes[key]["supported"] = False
            modes[key]["reason"] = "Install llama-cpp-python in backend venv"

    if all(m["supported"] for m in modes.values()):
        gguf_path = os.environ.get("SPONGE_GGUF_PATH", "").strip()
        gguf_dir = os.environ.get("SPONGE_GGUF_DIR", "").strip()
        has_path = bool(gguf_path and os.path.isfile(gguf_path))
        has_dir = bool(
            gguf_dir
            and os.path.isdir(gguf_dir)
            and any(name.lower().endswith(".gguf") for name in os.listdir(gguf_dir))
        )
        if not (has_path or has_dir):
            for key in modes:
                modes[key]["supported"] = False
                modes[key]["reason"] = "Set valid SPONGE_GGUF_PATH or SPONGE_GGUF_DIR with .gguf files"

    gguf_gpu_offload = None
    try:
        from llama_cpp import llama_supports_gpu_offload

        gguf_gpu_offload = bool(llama_supports_gpu_offload())
        if all(m["supported"] for m in modes.values()) and not gguf_gpu_offload:
            for key in modes:
                modes[key]["reason"] = "CPU-only llama-cpp-python build (no GPU offload support)"
    except Exception:
        pass

    gptq_id = os.environ.get("SPONGE_GPTQ_MODEL_ID", "").strip()
    modes["gptq"] = {
        "supported": bool(gptq_available() and gpu_available and gptq_id),
        "reason": "GPTQ via GPTQModel on GPU (set SPONGE_GPTQ_MODEL_ID)",
    }
    if gptq_available() and gpu_available and not gptq_id:
        modes["gptq"]["reason"] = "Set SPONGE_GPTQ_MODEL_ID to a Hugging Face GPTQ repo"
    elif not gptq_available():
        modes["gptq"]["reason"] = "Install GPTQModel: bash scripts/install_gptq_rocm.sh"
    elif not gpu_available:
        modes["gptq"]["reason"] = "GPTQ GPU path needs ROCm/CUDA visible to PyTorch"

    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "rocm_arch": rocm_arch,
        "hip_version": hip_version,
        "gguf_gpu_offload": gguf_gpu_offload,
        "gptq_available": gptq_available(),
        "gptq_model_id": gptq_id or None,
        "modes": modes,
    }
