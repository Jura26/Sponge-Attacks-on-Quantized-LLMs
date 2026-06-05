#!/usr/bin/env python3

import argparse
import csv
import os
import random
from collections import defaultdict


NIGHTMARE_FRAGMENTS = [
    "👩‍👩‍👦‍👦", "\u200D", "_\u200B", "a\u0328", "\U000e0061\U000e0062",
    "أ", "﷽", "𒈙", "𒐫", " ﷺ", ":_:", "><", "```", "'''", "[[[", "]]]",
    "🧑🏽‍🚀", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🏴‍☠️", "🤦🏿‍♂️", "🧠", "🧬", "🪐",
    "🇺🇳", "🇺🇸", "🇯🇵", "🇫🇷", "🇩🇪", "🇬🇧", "🇺🇦",
    "e\u0301", "o\u0308", "n\u0303", "a\u030a", "u\u0308", "i\u0307",
    "\u200C", "\u200E", "\u200F", "\u2060", "\u2061", "\u2062", "\u2063",
    "\uFE0E", "\uFE0F", "\u034F", "\u00AD", "\u00A0",
    "अ", "आ", "इ", "उ", "ए", "क", "ष",
    "語", "漢", "語", "字", "かな", "カナ",
    "😊", "😂", "🤖", "💥", "✨", "🔥", "⚡",
    "=!=", "====", "::::", "////", "\\\\", "----", "____",
    "\u202E", "\u2066", "\u2067", "\u2068", "\u2069", "\u180E",
    "\U0001F469\u200D\U0001F52C", "\U0001F469\u200D\U0001F9AF", "\U0001F9D1\u200D\U0001F692",
    "\U0001F9D1\u200D\U0001F52C", "\U0001F9D1\u200D\U0001F393", "\U0001F469\u200D\U0001F9D1\u200D\U0001F466",
]


def _repo_root() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, ".."))


def _default_models(repo_root: str) -> dict[str, str]:
    return {
        "llama3.2-3b": os.path.join(repo_root, "backend", "hf", "meta-llama", "Llama-3.2-3B"),
        "qwen2.5-3b": os.path.join(repo_root, "backend", "hf", "Qwen", "Qwen2.5-3B-Instruct"),
        "hunyuan-4b": os.path.join(repo_root, "backend", "hf", "tencent", "Hunyuan-4B-Instruct"),
    }


def _load_tokenizer(model_ref: str, allow_remote: bool):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required. Install with: pip install transformers") from exc

    try:
        return AutoTokenizer.from_pretrained(
            model_ref,
            use_fast=True,
            local_files_only=not allow_remote,
            trust_remote_code=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_ref,
            use_fast=False,
            local_files_only=not allow_remote,
            trust_remote_code=True,
        )


def _build_prompt(target_chars: int, rng: random.Random) -> str:
    parts = []
    total = 0
    while total < target_chars:
        frag = rng.choice(NIGHTMARE_FRAGMENTS)
        parts.append(frag)
        total += len(frag)
    text = "".join(parts)
    if len(text) > target_chars:
        text = text[:target_chars]
    return text


def _count_tokens(tokenizer, text: str) -> int:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    return len(tokens)


def _parse_model_overrides(values: list[str]) -> dict[str, str]:
    models = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid model override: {item!r}. Use name=path_or_id")
        name, ref = item.split("=", 1)
        name = name.strip()
        ref = ref.strip()
        if not name or not ref:
            raise ValueError(f"Invalid model override: {item!r}")
        models[name] = ref
    return models


def _char_range(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("char-step must be > 0")
    if end < start:
        raise ValueError("char-end must be >= char-start")
    return list(range(start, end + 1, step))


def _write_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["model", "chars", "avg_tokens", "avg_tokens_per_char"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows_by_model: dict[str, list[dict]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    for model, rows in rows_by_model.items():
        xs = [row["chars"] for row in rows]
        ys = [row["avg_tokens"] for row in rows]
        plt.plot(xs, ys, marker="o", label=model)

    plt.xlabel("Number of characters")
    plt.ylabel("Average tokens")
    plt.title("Token explosion vs. input size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure token explosion for nightmare Unicode prompts.")
    parser.add_argument("--char-start", type=int, default=200)
    parser.add_argument("--char-end", type=int, default=2000)
    parser.add_argument("--char-step", type=int, default=200)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Override models: name=path_or_id",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare_tokens_vs_chars.csv"),
    )
    parser.add_argument(
        "--plot",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare_tokens_vs_chars.png"),
    )

    args = parser.parse_args()

    repo_root = _repo_root()
    models = _default_models(repo_root)
    if args.models:
        models = _parse_model_overrides(args.models)

    rng = random.Random(args.seed)
    char_targets = _char_range(args.char_start, args.char_end, args.char_step)

    rows = []
    rows_by_model = defaultdict(list)

    for model_name, model_ref in models.items():
        tokenizer = _load_tokenizer(model_ref, allow_remote=args.allow_remote)
        for target_chars in char_targets:
            sample_tokens = []
            for _ in range(args.samples):
                prompt = _build_prompt(target_chars, rng)
                token_count = _count_tokens(tokenizer, prompt)
                sample_tokens.append(token_count)

            avg_tokens = sum(sample_tokens) / len(sample_tokens)
            row = {
                "model": model_name,
                "chars": target_chars,
                "avg_tokens": round(avg_tokens, 4),
                "avg_tokens_per_char": round(avg_tokens / target_chars, 6),
            }
            rows.append(row)
            rows_by_model[model_name].append(row)

    _write_csv(rows, args.csv)
    _plot(rows_by_model, args.plot)

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
