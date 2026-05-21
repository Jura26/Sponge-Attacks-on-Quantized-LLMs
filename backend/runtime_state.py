"""Mutable runtime state for long-running attacks."""

from __future__ import annotations


def _base_attack_state() -> dict:
    return {
        "is_running": False,
        "status": "idle",
        "logs": [],
        "current_generation": 0,
        "total_generations": 0,
        "best_result": None,
    }


def _base_comparison_state() -> dict:
    return {
        "is_running": False,
        "phase": "idle",
        "regular_result": None,
        "quantized_result": None,
        "regular_logs": [],
        "quantized_logs": [],
        "regular_model_id": None,
        "quantized_model_id": None,
        "regular_gguf_path": None,
        "quantized_gguf_path": None,
        "phase_a_display": None,
        "phase_b_display": None,
        "current_generation": 0,
        "total_generations": 0,
    }


attack_state = _base_attack_state()
comparison_state = _base_comparison_state()


def reset_attack_state(*, is_running: bool, status: str, total_generations: int) -> dict:
    attack_state.clear()
    attack_state.update({
        "is_running": is_running,
        "status": status,
        "logs": [],
        "current_generation": 0,
        "total_generations": total_generations,
        "best_result": None,
    })
    return attack_state


def reset_comparison_state(
    *,
    is_running: bool,
    phase: str,
    total_generations: int,
    regular_model_id: str,
    quantized_model_id: str,
    phase_a_display: str,
    phase_b_display: str,
    regular_gguf_path: str | None,
    quantized_gguf_path: str | None,
) -> dict:
    comparison_state.clear()
    comparison_state.update({
        "is_running": is_running,
        "phase": phase,
        "regular_result": None,
        "quantized_result": None,
        "regular_logs": [],
        "quantized_logs": [],
        "regular_model_id": regular_model_id,
        "quantized_model_id": quantized_model_id,
        "phase_a_display": phase_a_display,
        "phase_b_display": phase_b_display,
        "regular_gguf_path": regular_gguf_path,
        "quantized_gguf_path": quantized_gguf_path,
        "current_generation": 0,
        "total_generations": total_generations,
    })
    return comparison_state
