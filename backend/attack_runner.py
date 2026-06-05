"""Attack orchestration for background tasks."""

from __future__ import annotations

import gc
import random
from typing import Callable

from autodos_attack import run_autodos_attack
from context_exhaustion import run_context_exhaustion
from evolutionary_sponge import run_sponge_attack
from lingoloop_attack import run_lingoloop_attack
from model import get_last_gguf_selection, resolve_gguf_variant_path
from model_catalog import resolve_attack_target
from runtime_state import attack_state, comparison_state
from services.supabase import insert_payload
from state_entrapment_attack import run_state_entrapment_attack
from token_busting_attack import run_token_busting_attack


def _dispatch_attack(
    attack_type: str,
    model_id: str,
    quant_mode: str,
    *,
    gens: int,
    pop: int,
    num_requests: int,
    autodos_iterations: int,
    tree_depth: int,
    tree_breadth: int,
    context_mode: str,
    force_decode_tokens: int,
    disable_eos_stop: bool,
    progress_callback: Callable[[dict], None],
) -> None:
    if attack_type == "evolutionary":
        run_sponge_attack(
            model_id,
            gens=gens,
            pop=pop,
            quantize=quant_mode,
            progress_callback=progress_callback,
        )
    elif attack_type == "context_exhaustion":
        run_context_exhaustion(
            model_id,
            num_requests=num_requests,
            is_quantized=quant_mode,
            progress_callback=progress_callback,
            context_mode=context_mode,
            force_decode_tokens=force_decode_tokens,
            disable_eos_stop=disable_eos_stop,
        )
    elif attack_type == "autodos":
        run_autodos_attack(
            model_id,
            num_iterations=autodos_iterations,
            depth=tree_depth,
            breadth=tree_breadth,
            is_quantized=quant_mode,
            progress_callback=progress_callback,
        )
    elif attack_type == "token_busting":
        run_token_busting_attack(
            model_id,
            num_requests=num_requests,
            is_quantized=quant_mode,
            progress_callback=progress_callback,
        )
    elif attack_type == "lingoloop":
        run_lingoloop_attack(
            model_id,
            num_requests=num_requests,
            is_quantized=quant_mode,
            progress_callback=progress_callback,
        )
    elif attack_type == "state_entrapment":
        run_state_entrapment_attack(
            model_id,
            num_requests=num_requests,
            is_quantized=quant_mode,
            progress_callback=progress_callback,
        )


def _resolve_logged_quant_mode(requested_mode: str | None, result: dict | None = None) -> str | None:
    if isinstance(result, dict):
        label = result.get("quant_label") or result.get("quant_mode")
        if label:
            return label
    if requested_mode and str(requested_mode).startswith("gguf"):
        last = get_last_gguf_selection().get("quant_mode")
        return last or requested_mode
    return requested_mode


def sponge_attack_worker(
    model_id: str,
    gens: int,
    pop: int,
    attack_type: str = "evolutionary",
    num_requests: int = 10,
    autodos_iterations: int = 3,
    tree_depth: int = 3,
    tree_breadth: int = 4,
    quant_mode: str = "none",
    context_mode: str = "combined",
    force_decode_tokens: int = 64,
    disable_eos_stop: bool = False,
) -> None:
    """Background task wrapper for attack runs."""

    def callback(data: dict) -> None:
        if data.get("status") == "eval":
            msg = data.get("message", "")
            if msg:
                attack_state["logs"].append(msg)
        elif data.get("status") == "progress":
            attack_state["current_generation"] = data.get("generation")
            attack_state["best_result"] = {
                "score": data.get("best_score"),
                "temp": data.get("best_temp"),
                "prompt": data.get("best_prompt"),
                "output": data.get("best_output"),
                "avg_gpu": data.get("best_avg_gpu", 0),
                "duration": data.get("best_duration", 0),
                "input_tokens": data.get("best_input_tokens", 0),
                "output_tokens": data.get("best_output_tokens", 0),
                "energy_joules": data.get("best_energy", 0),
            }
            gen_log = (
                f"Gen {data.get('generation')}: Best Score {data.get('best_score'):.2f} "
                f"(Temp: {data.get('best_temp')}C)"
            )
            attack_state["logs"].append(gen_log)
        elif data.get("status") == "complete":
            attack_state["status"] = "complete"
            attack_state["is_running"] = False
            attack_state["best_result"] = data.get("result")
            attack_state["logs"].append("Attack Complete!")
            actual_quant_mode = _resolve_logged_quant_mode(quant_mode, data.get("result"))
            payload = {
                "attack_type": attack_type,
                "model_id": model_id,
                "quant_mode": actual_quant_mode,
                "params": {
                    "gens": gens,
                    "pop": pop,
                    "num_requests": num_requests,
                    "autodos_iterations": autodos_iterations,
                    "tree_depth": tree_depth,
                    "tree_breadth": tree_breadth,
                    "context_mode": context_mode,
                    "force_decode_tokens": force_decode_tokens,
                    "disable_eos_stop": disable_eos_stop,
                },
                "result": data.get("result"),
                "logs": attack_state["logs"][-200:],
            }
            insert_payload(payload)
        else:
            msg = data.get("message", "")
            if msg:
                attack_state["logs"].append(msg)
            if data.get("status"):
                attack_state["status"] = data.get("status")

    try:
        attack_state["is_running"] = True
        attack_state["status"] = "starting"
        attack_state["logs"] = [f"Starting {attack_type} attack process..."]
        attack_state["total_generations"] = gens if attack_type == "evolutionary" else 0

        _dispatch_attack(
            attack_type,
            model_id,
            quant_mode,
            gens=gens,
            pop=pop,
            num_requests=num_requests,
            autodos_iterations=autodos_iterations,
            tree_depth=tree_depth,
            tree_breadth=tree_breadth,
            context_mode=context_mode,
            force_decode_tokens=force_decode_tokens,
            disable_eos_stop=disable_eos_stop,
            progress_callback=callback,
        )
    except Exception as exc:
        attack_state["status"] = "error"
        attack_state["is_running"] = False
        attack_state["logs"].append(f"Error: {str(exc)}")


def _make_comparison_callback(target_logs_key: str, meta: dict | None = None):
    """Return a progress callback that writes into comparison_state."""

    def callback(data: dict) -> None:
        if data.get("status") == "eval":
            msg = data.get("message", "")
            if msg:
                comparison_state[target_logs_key].append(msg)
        elif data.get("status") == "progress":
            comparison_state["current_generation"] = data.get("generation")
            result = {
                "score": data.get("best_score"),
                "temp": data.get("best_temp"),
                "prompt": data.get("best_prompt"),
                "output": data.get("best_output"),
                "avg_gpu": data.get("best_avg_gpu", 0),
                "duration": data.get("best_duration", 0),
                "input_tokens": data.get("best_input_tokens", 0),
                "output_tokens": data.get("best_output_tokens", 0),
            }
            key = "regular_result" if target_logs_key == "regular_logs" else "quantized_result"
            comparison_state[key] = result
            gen_log = f"Gen {data.get('generation')}: Best Score {data.get('best_score'):.2f}"
            comparison_state[target_logs_key].append(gen_log)
        elif data.get("status") == "complete":
            key = "regular_result" if target_logs_key == "regular_logs" else "quantized_result"
            comparison_state[key] = data.get("result")
            comparison_state[target_logs_key].append("Phase complete!")
            actual_quant_mode = _resolve_logged_quant_mode(
                meta.get("quant_mode") if meta else None,
                data.get("result"),
            )
            payload = {
                "attack_type": meta.get("attack_type") if meta else None,
                "model_id": meta.get("model_id") if meta else None,
                "quant_mode": actual_quant_mode,
                "params": meta.get("params") if meta else None,
                "result": data.get("result"),
                "logs": comparison_state[target_logs_key][-200:],
            }
            insert_payload(payload)
        else:
            msg = data.get("message", "")
            if msg:
                comparison_state[target_logs_key].append(msg)

    return callback


def comparison_worker(
    model_id_a: str,
    model_id_b: str,
    gens: int,
    pop: int,
    seed: int,
    attack_type: str = "evolutionary",
    num_requests: int = 10,
    autodos_iterations: int = 3,
    tree_depth: int = 3,
    tree_breadth: int = 4,
    regular_quant_mode: str = "gguf-f16",
    quant_mode: str = "gguf-q4",
    phase_a_display: str | None = None,
    phase_b_display: str | None = None,
    context_mode: str = "combined",
    force_decode_tokens: int = 64,
    disable_eos_stop: bool = False,
) -> None:
    """Run the sponge attack twice: phase A config, then phase B config."""
    import torch

    def _maybe_gguf_path(model_id: str, mode: str) -> str | None:
        if str(mode or "").startswith("gguf"):
            try:
                return resolve_gguf_variant_path(model_id, mode)
            except Exception:
                return None
        return None

    label_a = phase_a_display or f"{model_id_a} ({regular_quant_mode})"
    label_b = phase_b_display or f"{model_id_b} ({quant_mode})"

    try:
        print("[main.py] Verifying VRAM is clear before taking baseline...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[main.py] VRAM before baseline: {allocated:.2f} GB")

        comparison_state["phase"] = "regular"
        comparison_state["regular_gguf_path"] = _maybe_gguf_path(model_id_a, regular_quant_mode)
        comparison_state["regular_logs"].append(f"=== Phase 1/2: {label_a} ===")
        random.seed(seed)

        regular_meta = {
            "attack_type": attack_type,
            "model_id": model_id_a,
            "quant_mode": regular_quant_mode,
            "phase": "regular",
            "params": {
                "gens": gens,
                "pop": pop,
                "num_requests": num_requests,
                "autodos_iterations": autodos_iterations,
                "tree_depth": tree_depth,
                "tree_breadth": tree_breadth,
                "context_mode": context_mode,
                "force_decode_tokens": force_decode_tokens,
                "disable_eos_stop": disable_eos_stop,
                "seed": seed,
            },
        }

        _dispatch_attack(
            attack_type,
            model_id_a,
            regular_quant_mode,
            gens=gens,
            pop=pop,
            num_requests=num_requests,
            autodos_iterations=autodos_iterations,
            tree_depth=tree_depth,
            tree_breadth=tree_breadth,
            context_mode=context_mode,
            force_decode_tokens=force_decode_tokens,
            disable_eos_stop=disable_eos_stop,
            progress_callback=_make_comparison_callback("regular_logs", regular_meta),
        )
        if str(regular_quant_mode).startswith("gguf"):
            comparison_state["regular_gguf_path"] = get_last_gguf_selection().get("path")
        else:
            comparison_state["regular_gguf_path"] = None

        print("[main.py] Verifying VRAM is clear between phases...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[main.py] VRAM after inter-phase cleanup: {allocated:.2f} GB")

        comparison_state["phase"] = "quantized"
        comparison_state["current_generation"] = 0
        comparison_state["quantized_gguf_path"] = _maybe_gguf_path(model_id_b, quant_mode)
        comparison_state["quantized_logs"].append(f"=== Phase 2/2: {label_b} ===")
        random.seed(seed)

        quant_meta = {
            "attack_type": attack_type,
            "model_id": model_id_b,
            "quant_mode": quant_mode,
            "phase": "quantized",
            "params": {
                "gens": gens,
                "pop": pop,
                "num_requests": num_requests,
                "autodos_iterations": autodos_iterations,
                "tree_depth": tree_depth,
                "tree_breadth": tree_breadth,
                "context_mode": context_mode,
                "force_decode_tokens": force_decode_tokens,
                "disable_eos_stop": disable_eos_stop,
                "seed": seed,
            },
        }

        _dispatch_attack(
            attack_type,
            model_id_b,
            quant_mode,
            gens=gens,
            pop=pop,
            num_requests=num_requests,
            autodos_iterations=autodos_iterations,
            tree_depth=tree_depth,
            tree_breadth=tree_breadth,
            context_mode=context_mode,
            force_decode_tokens=force_decode_tokens,
            disable_eos_stop=disable_eos_stop,
            progress_callback=_make_comparison_callback("quantized_logs", quant_meta),
        )
        if str(quant_mode).startswith("gguf"):
            comparison_state["quantized_gguf_path"] = get_last_gguf_selection().get("path")
        else:
            comparison_state["quantized_gguf_path"] = None

        comparison_state["phase"] = "complete"
        comparison_state["is_running"] = False

    except Exception as exc:
        comparison_state["phase"] = "error"
        comparison_state["is_running"] = False
        target = "quantized_logs" if comparison_state.get("regular_result") else "regular_logs"
        comparison_state[target].append(f"Error: {str(exc)}")


def resolve_compare_phase(
    family: str | None,
    backend: str | None,
    gguf_variant: str | None,
    fallback_model_id: str,
    fallback_quant_mode: str,
):
    if family:
        target = resolve_attack_target(family, backend or "gguf", gguf_variant)
        return (
            target["model_id"],
            target["quant_mode"],
            target["display"],
            target.get("gguf_path") or target.get("hf_path"),
        )
    return fallback_model_id, fallback_quant_mode, f"{fallback_model_id} ({fallback_quant_mode})", None
