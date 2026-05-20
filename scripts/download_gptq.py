#!/usr/bin/env python3
"""Download GPTQ checkpoints from Hugging Face repos.

Defaults to GPTQ repos referenced by backend/model_catalog.py for known model
families. You can override with --repos or --families.

Usage:
  python scripts/download_gptq.py
  python scripts/download_gptq.py --dest /home/jura/models/gptq
  python scripts/download_gptq.py --families Llama3,Qwen,Hunyuan
  python scripts/download_gptq.py --repos tencent/Hunyuan-4B-Instruct-GPTQ-Int4

Set HF_TOKEN (or HUGGINGFACE_TOKEN) in env for faster downloads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download

# Add backend directory to path for model_catalog imports
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

try:
    import model_catalog
except Exception:  # pragma: no cover - fallback if import fails
    model_catalog = None


def normalize_repo_arg(arg: str) -> str:
    if arg.startswith("http://") or arg.startswith("https://"):
        parsed = urlparse(arg)
        path = parsed.path.lstrip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return path
    return arg


def resolve_default_repos(families: list[str] | None) -> list[str]:
    if model_catalog is None:
        raise RuntimeError("backend/model_catalog.py is not available")

    repos: list[str] = []
    if families:
        for fam in families:
            fam_id = fam.strip()
            if not fam_id:
                continue
            repos.append(model_catalog.resolve_gptq_repo(fam_id))
        return repos

    for fam_id in model_catalog.MODEL_FAMILIES.keys():
        try:
            repos.append(model_catalog.resolve_gptq_repo(fam_id))
        except Exception:
            continue
    return repos


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GPTQ checkpoints from HF")
    parser.add_argument("--dest", default="", help="Destination directory")
    parser.add_argument("--families", default="", help="Comma-separated family ids")
    parser.add_argument("--repos", nargs="+", help="HF repo ids or URLs")
    parser.add_argument("--dry-run", action="store_true", help="Show actions only")
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Optional allow patterns for snapshot_download",
    )
    args = parser.parse_args()

    dest = args.dest
    if not dest:
        dest = str(BACKEND_DIR / "gptq")

    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    families = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else None
    repos: list[str]
    if args.repos:
        repos = [normalize_repo_arg(r) for r in args.repos]
    else:
        repos = resolve_default_repos(families)

    if not repos:
        raise SystemExit("No GPTQ repos resolved. Use --repos or --families.")

    for repo_id in repos:
        repo_dir = dest_dir / repo_id
        if args.dry_run:
            print(f"[dry-run] {repo_id} -> {repo_dir}")
            continue

        print(f"Downloading {repo_id} -> {repo_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(repo_dir),
            local_dir_use_symlinks=False,
            allow_patterns=args.allow_patterns,
            token=token,
        )

    print("Done.")


if __name__ == "__main__":
    main()
