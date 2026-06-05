#!/usr/bin/env python3
"""Download Hugging Face models for the HF FP16 backend.

Defaults to the three requested repos and stores them under SPONGE_HF_DIR
(or ./backend/hf if unset).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_REPOS = [
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "tencent/Hunyuan-4B-Instruct",
]


def _default_dest() -> Path:
    return Path(__file__).resolve().parents[1] / "backend" / "hf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF models for local FP16 runs")
    parser.add_argument("--dest", default=str(_default_dest()), help="Destination root (SPONGE_HF_DIR)")
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS, help="HF repo ids to download")
    parser.add_argument("--revision", default="", help="Optional HF revision/tag")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit("huggingface_hub is required. Install it in the backend venv.") from exc

    token = os.environ.get("HF_TOKEN", "").strip() or None
    if not token:
        print("⚠️  HF_TOKEN not set. Gated models (Llama) will fail to download.")

    revision = args.revision.strip() or None

    for repo_id in args.repos:
        repo_id = repo_id.strip()
        if not repo_id:
            continue
        local_dir = dest / repo_id
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            revision=revision,
            token=token,
        )

    print("✅ Done. Set SPONGE_HF_DIR to use these local models.")


if __name__ == "__main__":
    main()
