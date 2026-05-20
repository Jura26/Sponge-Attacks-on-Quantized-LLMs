#!/usr/bin/env python3
"""Download GGUF files from given Hugging Face model repos and copy .gguf files
into SPONGE_GGUF_DIR (or ./backend/ggufs by default).

Usage:
  python scripts/download_ggufs.py \
      bartowski/tencent_Hunyuan-4B-Instruct-GGUF \
      bartowski/Llama-3.2-3B-Instruct-GGUF \
      Qwen/Qwen2.5-3B-Instruct-GGUF

Set `HF_TOKEN` in environment if private or rate-limited.
"""
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from huggingface_hub import snapshot_download, HfApi, hf_hub_download

# Add backend directory to path for imports
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    import model_catalog
except Exception:
    model_catalog = None


def normalize_repo_arg(arg: str) -> str:
    # accept full HF URLs or namespace/repo
    if arg.startswith("http://") or arg.startswith("https://"):
        parsed = urlparse(arg)
        # path looks like '/namespace/repo' or '/namespace/repo/-/tree/...'
        path = parsed.path.lstrip("/")
        # take first two path components as namespace/repo
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return path
    return arg


def main(repos, dest_dir: str | Path, variants: list[str] | None = None):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    for repo in repos:
        repo_id = normalize_repo_arg(repo)
        print(f"Checking {repo} -> {repo_id} for .gguf files...")
        api = HfApi()
        try:
            repo_files = api.list_repo_files(repo_id)
        except Exception as e:
            print(f"  Failed to list files for {repo_id}: {e}")
            continue

        gguf_files = [f for f in repo_files if f.lower().endswith('.gguf')]
        if not gguf_files:
            print(f"  No .gguf files found in {repo_id}.")
            continue

        # determine which variants to download: prefer command-line arg, then model_catalog
        desired_variants = variants
        if not desired_variants and model_catalog:
            for fam_id, meta in model_catalog.MODEL_FAMILIES.items():
                basename = (meta.get('gguf_basename') or '').lower()
                if basename and basename in repo_id.lower():
                    desired_variants = meta.get('gguf_variants', [])
                    break
        if not desired_variants:
            # fallback to common set
            desired_variants = ['q8_0', 'q4_k_m', 'q2_k']

        # collect tokens for desired variants
        tokens = []
        if model_catalog:
            for v in desired_variants:
                toks = model_catalog.GGUF_VARIANT_FILE_TOKENS.get(v, [])
                tokens.extend(toks)
        tokens = [t.lower() for t in tokens]


        # Filter by desired variants (command-line or from model_catalog)
        to_download = []
        for f in gguf_files:
            fl = f.lower()
            if any(tok in fl for tok in tokens):
                to_download.append(f)

        if not to_download:
            print(f"  No GGUF files matching variants {desired_variants} found in {repo_id}.")
            continue

        for fname in to_download:
            print(f"  Downloading file {fname}...")
            try:
                if token:
                    cached = hf_hub_download(repo_id=repo_id, filename=fname, repo_type='model', token=token)
                else:
                    cached = hf_hub_download(repo_id=repo_id, filename=fname, repo_type='model')
            except Exception as e:
                print(f"    Failed to download {fname}: {e}")
                continue
            target = dest / Path(cached).name
            shutil.copy2(cached, target)
            print(f"    Saved {target}")

    print("Done. Set SPONGE_GGUF_DIR to the destination folder to make models available to the backend.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download GGUF files from HF repos")
    parser.add_argument("repos", nargs="+", help="Hugging Face repo URLs or ids")
    parser.add_argument("--variants", nargs="+", help="GGUF variants to download (e.g., q2_k q8_0 q4_k_m)")
    args = parser.parse_args()
    repos = args.repos
    dest = os.environ.get("SPONGE_GGUF_DIR") or Path(__file__).resolve().parents[1] / "backend" / "ggufs"
    main(repos, dest, variants=args.variants)
