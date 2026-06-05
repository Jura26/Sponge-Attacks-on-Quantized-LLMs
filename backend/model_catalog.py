"""Model families, GGUF variants, and attack target resolution."""

from __future__ import annotations

import glob
import os
import re
from typing import Any

# UI variant id -> internal gguf quant mode
GGUF_VARIANT_TO_MODE = {
    "f16": "gguf-f16",
    "q8_0": "gguf-q8",
    "q6_k": "gguf-q6",
    "q5_k_m": "gguf-q5",
    "q4_k_m": "gguf-q4",
    "q3_k_m": "gguf-q3",
    "q2_k": "gguf-q2",
}

GGUF_VARIANT_LABELS = {
    "f16": "F16 (puna preciznost)",
    "q8_0": "Q8_0 (8-bit)",
    "q6_k": "Q6_K (6-bit)",
    "q5_k_m": "Q5_K_M (5-bit)",
    "q4_k_m": "Q4_K_M (4-bit)",
    "q3_k_m": "Q3_K_M (3-bit)",
    "q2_k": "Q2_K (2-bit)",
}

# Substrings used to match files in SPONGE_GGUF_DIR
MODEL_FAMILIES: dict[str, dict[str, Any]] = {
    "Llama3": {
        "label": "Llama3.2-3B",
        "gguf_basename": "llama",
        "gptq_env": "SPONGE_GPTQ_MODEL_MISTRAL7B",
        "default_gptq": "ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3",
        "hf_env": "SPONGE_HF_MODEL_LLAMA3",
        "default_hf": "meta-llama/Llama-3.2-3B",
        "gguf_variants": ["f16", "q8_0", "q4_k_m", "q2_k"],
    },
    "Qwen": {
        "label": "Qwen2.5-3B",
        "gguf_basename": "qwen",
        "gptq_env": "SPONGE_GPTQ_MODEL_GPT2",
        "default_gptq": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        "hf_env": "SPONGE_HF_MODEL_QWEN",
        "default_hf": "Qwen/Qwen2.5-3B-Instruct",
        "gguf_variants": ["f16", "q8_0", "q4_k_m", "q2_k"],
    },
    "Hunyuan": {
        "label": "Hunyuan-4B",
        "gguf_basename": "hunyuan",
        "gptq_env": "SPONGE_GPTQ_MODEL_OPT_6_7B",
        "default_gptq": "tencent/Hunyuan-4B-Instruct-GPTQ-Int4",
        "hf_env": "SPONGE_HF_MODEL_HUNYUAN",
        "default_hf": "tencent/Hunyuan-4B-Instruct",
        "gguf_variants": ["f16", "q8_0", "q4_k_m", "q2_k"],
    },
}

# Filename tokens per UI variant (first match wins)
GGUF_VARIANT_FILE_TOKENS = {
    "f16": ["f16", "fp16"],
    "q8_0": ["q8_0", "q8"],
    "q6_k": ["q6_k"],
    "q5_k_m": ["q5_k_m", "q5_k_s"],
    "q4_k_m": ["q4_k_m", "q4_k_s", "q4_0"],
    "q3_k_m": ["q3_k_m", "q3_k_s"],
    "q2_k": ["q2_k", "q2_0"],
}


def _gguf_dir() -> str:
    gguf_dir = os.environ.get("SPONGE_GGUF_DIR", "").strip()
    if gguf_dir and os.path.isdir(gguf_dir):
        return gguf_dir
    gguf_path = os.environ.get("SPONGE_GGUF_PATH", "").strip()
    if gguf_path and os.path.isfile(gguf_path):
        return os.path.dirname(gguf_path)
    return ""


def _hf_dir() -> str:
    hf_dir = os.environ.get("SPONGE_HF_DIR", "").strip()
    if hf_dir:
        return hf_dir
    return os.path.join(os.path.dirname(__file__), "hf")


def list_gguf_files_for_family(family_id: str) -> list[str]:
    family = MODEL_FAMILIES.get(family_id)
    if not family:
        return []
    gguf_dir = _gguf_dir()
    if not gguf_dir:
        return []
    hint = family["gguf_basename"].lower()
    paths = []
    for path in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
        if hint in os.path.basename(path).lower():
            paths.append(path)
    return paths


def family_has_gguf_variant(family_id: str, variant: str) -> bool:
    tokens = GGUF_VARIANT_FILE_TOKENS.get(variant, [])
    name_l = lambda p: os.path.basename(p).lower()
    return any(any(tok in name_l(p) for tok in tokens) for p in list_gguf_files_for_family(family_id))


def resolve_gguf_path_for_variant(family_id: str, variant: str) -> str | None:
    tokens = GGUF_VARIANT_FILE_TOKENS.get(variant, [])
    if not tokens:
        return None
    for path in list_gguf_files_for_family(family_id):
        base = os.path.basename(path).lower()
        for tok in tokens:
            if tok in base:
                return path
    return None


def resolve_gptq_repo(family_id: str) -> str:
    family = MODEL_FAMILIES.get(family_id)
    if not family:
        raise ValueError(f"Unknown model family: {family_id}")

    env_key = family.get("gptq_env", "")
    repo = os.environ.get(env_key, "").strip() if env_key else ""
    if not repo:
        repo = os.environ.get("SPONGE_GPTQ_MODEL_ID", "").strip()
    if not repo:
        repo = (family.get("default_gptq") or "").strip()
    if not repo:
        raise RuntimeError(
            f"No GPTQ checkpoint configured for {family['label']}. "
            f"Set {env_key} or SPONGE_GPTQ_MODEL_ID in backend/.env"
        )
    return repo


def resolve_hf_repo(family_id: str) -> str:
    family = MODEL_FAMILIES.get(family_id)
    if not family:
        raise ValueError(f"Unknown model family: {family_id}")

    env_key = family.get("hf_env", "")
    repo = os.environ.get(env_key, "").strip() if env_key else ""
    if not repo:
        repo = os.environ.get("SPONGE_HF_MODEL_ID", "").strip()
    if not repo:
        repo = (family.get("default_hf") or "").strip()
    if not repo:
        raise RuntimeError(
            f"No HF checkpoint configured for {family['label']}. "
            f"Set {env_key} or SPONGE_HF_MODEL_ID in backend/.env"
        )
    return repo


def resolve_hf_local_path(family_id: str) -> str | None:
    repo = resolve_hf_repo(family_id)
    hf_dir = _hf_dir()
    if not hf_dir:
        return None
    local_path = os.path.join(hf_dir, repo)
    if os.path.isfile(os.path.join(local_path, "config.json")):
        return local_path
    return None


def resolve_attack_target(
    family_id: str,
    backend: str,
    gguf_variant: str | None = None,
) -> dict[str, str]:
    """Resolve UI selections to model_id + quant_mode for load_model_and_tokenizer."""
    family = MODEL_FAMILIES.get(family_id)
    if not family:
        raise ValueError(f"Unknown model family: {family_id}")

    backend = (backend or "gguf").strip().lower()
    label = family["label"]

    if backend == "gptq":
        repo = resolve_gptq_repo(family_id)
        return {
            "family_id": family_id,
            "model_id": family_id,
            "quant_mode": "gptq",
            "backend": "gptq",
            "gguf_variant": "",
            "gptq_repo": repo,
            "display": f"{label} · GPTQ 4-bit ({repo})",
        }

    if backend == "hf":
        repo = resolve_hf_repo(family_id)
        path = resolve_hf_local_path(family_id)
        if not path:
            raise RuntimeError(
                f"HF model for {label} not found in SPONGE_HF_DIR. "
                "Run scripts/download_hf_models.py to download it first."
            )
        return {
            "family_id": family_id,
            "model_id": family_id,
            "quant_mode": "hf-fp16",
            "backend": "hf",
            "gguf_variant": "",
            "hf_repo": repo,
            "hf_path": path,
            "display": f"{label} · HF FP16 ({repo})",
        }

    variant = (gguf_variant or "q4_k_m").strip().lower()
    quant_mode = GGUF_VARIANT_TO_MODE.get(variant)
    if not quant_mode:
        raise ValueError(f"Unknown GGUF variant: {variant}")

    path = resolve_gguf_path_for_variant(family_id, variant)
    if not path:
        raise RuntimeError(
            f"No GGUF file for {label} variant {GGUF_VARIANT_LABELS.get(variant, variant)} "
            f"in SPONGE_GGUF_DIR (hint: {family['gguf_basename']})"
        )

    return {
        "family_id": family_id,
        "model_id": family_id,
        "quant_mode": quant_mode,
        "backend": "gguf",
        "gguf_variant": variant,
        "gguf_path": path,
        "display": f"{label} · GGUF {GGUF_VARIANT_LABELS.get(variant, variant)}",
    }


def build_model_catalog() -> dict:
    """Catalog for the frontend: families, variants on disk, GPTQ availability."""
    import torch

    from model import gptq_available, hf_available

    gguf_dir = _gguf_dir()
    hf_dir = _hf_dir()
    families = []

    for family_id, meta in MODEL_FAMILIES.items():
        variants = []
        for vid in meta.get("gguf_variants", []):
            path = resolve_gguf_path_for_variant(family_id, vid)
            variants.append({
                "id": vid,
                "label": GGUF_VARIANT_LABELS.get(vid, vid),
                "available": path is not None,
                "path": path,
            })

        gptq_repo = None
        gptq_error = None
        try:
            gptq_repo = resolve_gptq_repo(family_id)
            gptq_ok = bool(gptq_available() and torch.cuda.is_available())
        except Exception as exc:
            gptq_ok = False
            gptq_error = str(exc)

        hf_repo = None
        hf_path = None
        hf_error = None
        try:
            hf_repo = resolve_hf_repo(family_id)
            hf_path = resolve_hf_local_path(family_id)
            hf_ok = bool(hf_available() and hf_path)
        except Exception as exc:
            hf_ok = False
            hf_error = str(exc)
        if not hf_error:
            if not hf_available():
                hf_error = "Install transformers in the backend venv"
            elif not hf_path:
                hf_error = "Run scripts/download_hf_models.py to pre-download"

        families.append({
            "id": family_id,
            "label": meta["label"],
            "gguf_basename": meta["gguf_basename"],
            "gguf_variants": variants,
            "gptq": {
                "available": gptq_ok,
                "repo": gptq_repo,
                "label": "GPTQ 4-bit (GPTQModel)",
                "error": gptq_error,
            },
            "hf": {
                "available": hf_ok,
                "repo": hf_repo,
                "label": "HF FP16 (transformers)",
                "path": hf_path,
                "error": hf_error,
            },
        })

    return {
        "gguf_dir": gguf_dir or None,
        "hf_dir": hf_dir or None,
        "families": families,
    }
