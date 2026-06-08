#!/usr/bin/env python3
import os
import sys
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

# Dynamically calculate paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # path/to/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)               # path/to/project_root

# Synced directly with your backend catalog mappings
MODEL_REGISTRY = {
    "Llama3": {
        "hf_repo": "meta-llama/Llama-3.2-3B",
        "gptq_repo": "ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3",
        "gguf_repo": "tensorblock/Llama-3.2-3B-Instruct-GGUF",
        "basename": "llama",
        "variants": {
            "f16": "Llama-3.2-3B-Instruct-F16.gguf",
            "q8_0": "Llama-3.2-3B-Instruct-Q8_0.gguf",
            "q4_k_m": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "q2_k": "Llama-3.2-3B-Instruct-Q2_K.gguf"
        }
    },
    "Qwen": {
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "gptq_repo": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        "gguf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "basename": "qwen",
        "variants": {
            "f16": "qwen2.5-3b-instruct-f16.gguf",
            "q8_0": "qwen2.5-3b-instruct-q8_0.gguf",
            "q4_k_m": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "q2_k": "qwen2.5-3b-instruct-q2_k.gguf"
        }
    },
    "Hunyuan": {
        "hf_repo": "tencent/Hunyuan-4B-Instruct",
        "gptq_repo": "tencent/Hunyuan-4B-Instruct-GPTQ-Int4",
        "gguf_repo": "gabriellarson/Hunyuan-4B-Instruct-GGUF",
        "basename": "hunyuan",
        "variants": {
            "f16": "hunyuan-4b-instruct-f16.gguf",
            "q8_0": "hunyuan-4b-instruct-q8_0.gguf",
            "q4_k_m": "hunyuan-4b-instruct-q4_k_m.gguf",
            "q2_k": "hunyuan-4b-instruct-q2_k.gguf"
        }
    }
}

def download_family(family_id: str, hf_dir: str, gptq_dir: str, gguf_dir: str, token: str | None, dry_run: bool):
    meta = MODEL_REGISTRY[family_id]
    
    print(f"\n======== Processing Family Target: {family_id} ========")

    # 1. HF Native Storage (Nested under backend/hf/org/repo)
    target_hf_path = os.path.join(hf_dir, meta["hf_repo"])
    print(f"[-->] HF Native Target Location: {target_hf_path}")
    if dry_run:
        print(f"      [DRY RUN] Would snapshot pull from: {meta['hf_repo']}")
    else:
        try:
            snapshot_download(
                repo_id=meta["hf_repo"],
                local_dir=target_hf_path,
                local_dir_use_symlinks=False,
                token=token,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.git*"]
            )
            print("[✓] HF Native loaded.")
        except KeyboardInterrupt:
            print("\n[!] Skipping current step via user interrupt...")
        except Exception as e:
            print(f"[X] Error downloading HF Native: {e}", file=sys.stderr)

    # 2. GPTQ Storage (Nested mirror match under backend/gptq/org/repo)
    target_gptq_path = os.path.join(gptq_dir, meta["gptq_repo"])
    print(f"[-->] GPTQ Target Location: {target_gptq_path}")
    if dry_run:
        print(f"      [DRY RUN] Would snapshot pull from: {meta['gptq_repo']}")
    else:
        try:
            snapshot_download(
                repo_id=meta["gptq_repo"],
                local_dir=target_gptq_path,
                local_dir_use_symlinks=False,
                token=token,
                ignore_patterns=["*.git*"]
            )
            print("[✓] GPTQ loaded.")
        except KeyboardInterrupt:
            print("\n[!] Skipping current step via user interrupt...")
        except Exception as e:
            print(f"[X] Error downloading GPTQ: {e}", file=sys.stderr)

    # 3. GGUF Flat Storage with Verification Remapping (Directly inside backend/gguf/)
    print(f"[-->] GGUF Flat Directory Location: {gguf_dir}")
    for variant_id, remote_filename in meta["variants"].items():
        local_filename = f"{meta['basename']}_{variant_id}.gguf"
        target_gguf_file = os.path.join(gguf_dir, local_filename)

        if dry_run:
            print(f"      [DRY RUN] Would download {remote_filename} from {meta['gguf_repo']} -> Save as: {local_filename}")
        else:
            print(f"   -> Downloading GGUF variant [{variant_id}]...")
            try:
                hf_hub_download(
                    repo_id=meta["gguf_repo"],
                    filename=remote_filename,
                    local_dir=gguf_dir,
                    local_dir_use_symlinks=False,
                    token=token
                )
                # Ensure filename exactly matches the expected tokens
                downloaded_path = os.path.join(gguf_dir, remote_filename)
                if os.path.exists(downloaded_path) and downloaded_path != target_gguf_file:
                    os.rename(downloaded_path, target_gguf_file)
                print(f"      [✓] Saved flat as: {local_filename}")
            except KeyboardInterrupt:
                print("\n[!] Skipping current GGUF variant via user interrupt...")
            except Exception as e:
                print(f"      [X] Failed tracking GGUF configuration: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalog-Compliant Downloader for Sponge Attack Architecture.")
    parser.add_argument(
        "--family", 
        choices=list(MODEL_REGISTRY.keys()) + ["all"], 
        default="all",
        help="Target model configuration family to pull down."
    )
    
    # Automatically resolve paths to point into the sibling backend/ directory
    parser.add_argument(
        "--hf_dir", 
        default=os.environ.get("SPONGE_HF_DIR", os.path.join(PROJECT_ROOT, "backend", "hf")), 
        help="Root path for Hugging Face architectures."
    )
    parser.add_argument(
        "--gguf_dir", 
        default=os.environ.get("SPONGE_GGUF_DIR", os.path.join(PROJECT_ROOT, "backend", "gguf")), 
        help="Flat destination folder where all GGUF matrices are stored."
    )
    parser.add_argument(
        "--gptq_dir", 
        default=os.environ.get("SPONGE_GPTQ_DIR", os.path.join(PROJECT_ROOT, "backend", "gptq")), 
        help="Root path for GPTQ storage structures."
    )
    parser.add_argument(
        "--token", 
        default=os.environ.get("HF_TOKEN"), 
        help="Hugging Face User Access Token for gated models."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Simulate layout placement mapping paths without downloading."
    )

    args = parser.parse_args()

    # Create target deployment workspaces inside backend/
    os.makedirs(args.hf_dir, exist_ok=True)
    os.makedirs(args.gguf_dir, exist_ok=True)
    os.makedirs(args.gptq_dir, exist_ok=True)

    if (args.family == "all" or args.family == "Llama3") and not args.token:
        print("[!] Warning: Access token required for Meta-Llama verification clearances.")

    target_families = MODEL_REGISTRY.keys() if args.family == "all" else [args.family]
    
    for family in target_families:
        download_family(family, args.hf_dir, args.gptq_dir, args.gguf_dir, args.token, args.dry_run)

    print("\n[✓] Runtime execution synchronization task completed.")