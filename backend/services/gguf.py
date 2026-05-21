"""GGUF path helpers used by API endpoints."""

from __future__ import annotations

import os

from model import resolve_gguf_variant_path


def resolve_gguf_paths(
    model_id: str,
    regular_quant_mode: str,
    quant_mode: str,
) -> dict:
    regular_path = resolve_gguf_variant_path(model_id, regular_quant_mode)
    quant_path = resolve_gguf_variant_path(model_id, quant_mode)
    return {
        "model_id": model_id,
        "regular_quant_mode": regular_quant_mode,
        "quant_mode": quant_mode,
        "regular": {
            "path": regular_path,
            "exists": bool(regular_path and os.path.isfile(regular_path)),
        },
        "quantized": {
            "path": quant_path,
            "exists": bool(quant_path and os.path.isfile(quant_path)),
        },
    }


def list_gguf_files() -> dict:
    gguf_dir = os.environ.get("SPONGE_GGUF_DIR", "").strip()
    gguf_path = os.environ.get("SPONGE_GGUF_PATH", "").strip()

    files = []
    seen = set()

    def _add_file(path: str) -> None:
        if path in seen:
            return
        seen.add(path)
        size_bytes = None
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = None
        size_gb = None
        if isinstance(size_bytes, (int, float)):
            size_gb = round(size_bytes / (1024 ** 3), 2)
        files.append({
            "path": path,
            "name": os.path.basename(path),
            "size_bytes": size_bytes,
            "size_gb": size_gb,
        })

    if gguf_path and os.path.isfile(gguf_path):
        _add_file(gguf_path)

    if gguf_dir and os.path.isdir(gguf_dir):
        for name in sorted(os.listdir(gguf_dir)):
            if name.lower().endswith(".gguf"):
                _add_file(os.path.join(gguf_dir, name))

    return {"files": files}
