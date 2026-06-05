from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import uvicorn

from attack_runner import comparison_worker, resolve_compare_phase, sponge_attack_worker
from model import resolve_gguf_variant_path
from model_catalog import build_model_catalog
from runtime_state import attack_state, comparison_state, reset_attack_state, reset_comparison_state
from services.capabilities import build_quantization_capabilities
from services.gguf import list_gguf_files, resolve_gguf_paths
from services.stats import collect_system_stats

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except Exception:
    # Backend still works if python-dotenv is not installed.
    pass

try:
    with open("/proc/self/oom_score_adj", "w") as _f:
        _f.write("500")
except OSError:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/attack/start")
def start_attack(
    background_tasks: BackgroundTasks,
    model_id: str = "gpt2",
    gens: int = 5,
    pop: int = 10,
    attack_type: str = "evolutionary",
    num_requests: int = 10,
    autodos_iterations: int = 3,
    tree_depth: int = 3,
    tree_breadth: int = 4,
    quant_mode: str = "gguf-f16",
    context_mode: str = "combined",
    force_decode_tokens: int = 64,
    disable_eos_stop: bool = False,
):
    if attack_state["is_running"]:
        return {"error": "Attack already running"}

    reset_attack_state(
        is_running=True,
        status="queued",
        total_generations=gens if attack_type == "evolutionary" else 0,
    )

    background_tasks.add_task(
        sponge_attack_worker,
        model_id,
        gens,
        pop,
        attack_type,
        num_requests,
        autodos_iterations,
        tree_depth,
        tree_breadth,
        quant_mode,
        context_mode,
        force_decode_tokens,
        disable_eos_stop,
    )
    return {"message": "Attack started"}


@app.get("/api/attack/status")
def get_attack_status():
    return attack_state


@app.post("/api/attack/compare")
def start_comparison(
    background_tasks: BackgroundTasks,
    model_id: str = "mistral7b",
    model_id_a: str | None = None,
    model_id_b: str | None = None,
    model_family_a: str | None = None,
    model_family_b: str | None = None,
    phase_a_backend: str = "gguf",
    phase_a_gguf_variant: str = "f16",
    phase_b_backend: str = "gguf",
    phase_b_gguf_variant: str = "q4_k_m",
    gens: int = 5,
    pop: int = 10,
    attack_type: str = "evolutionary",
    num_requests: int = 10,
    autodos_iterations: int = 3,
    tree_depth: int = 3,
    tree_breadth: int = 4,
    regular_quant_mode: str | None = None,
    quant_mode: str | None = None,
    context_mode: str = "combined",
    force_decode_tokens: int = 64,
    disable_eos_stop: bool = False,
):
    if comparison_state["is_running"]:
        return {"error": "Comparison already running"}

    def _gguf_path_if_needed(model_id: str, mode: str) -> str | None:
        if str(mode or "").startswith("gguf"):
            return resolve_gguf_variant_path(model_id, mode)
        return None

    seed = random.randint(0, 2**31)

    legacy_a = model_id_a or model_id
    legacy_b = model_id_b or model_id

    try:
        if model_family_a:
            resolved_a, mode_a, display_a, path_a = resolve_compare_phase(
                model_family_a, phase_a_backend, phase_a_gguf_variant, legacy_a, "gguf-f16"
            )
        else:
            mode_a = regular_quant_mode or "gguf-f16"
            resolved_a = legacy_a
            display_a = f"{legacy_a} ({mode_a})"
            path_a = _gguf_path_if_needed(resolved_a, mode_a)

        if model_family_b:
            resolved_b, mode_b, display_b, path_b = resolve_compare_phase(
                model_family_b, phase_b_backend, phase_b_gguf_variant, legacy_b, "gguf-q4"
            )
        else:
            mode_b = quant_mode or "gguf-q4"
            resolved_b = legacy_b
            display_b = f"{legacy_b} ({mode_b})"
            path_b = _gguf_path_if_needed(resolved_b, mode_b)
    except Exception as exc:
        return {"error": str(exc)}

    reset_comparison_state(
        is_running=True,
        phase="queued",
        total_generations=gens if attack_type == "evolutionary" else 0,
        regular_model_id=resolved_a,
        quantized_model_id=resolved_b,
        phase_a_display=display_a,
        phase_b_display=display_b,
        regular_gguf_path=path_a or _gguf_path_if_needed(resolved_a, mode_a),
        quantized_gguf_path=path_b or _gguf_path_if_needed(resolved_b, mode_b),
    )

    background_tasks.add_task(
        comparison_worker,
        resolved_a,
        resolved_b,
        gens,
        pop,
        seed,
        attack_type,
        num_requests,
        autodos_iterations,
        tree_depth,
        tree_breadth,
        mode_a,
        mode_b,
        display_a,
        display_b,
        context_mode,
        force_decode_tokens,
        disable_eos_stop,
    )
    return {"message": "Comparison started", "phase_a": display_a, "phase_b": display_b}


@app.get("/api/attack/compare/status")
def get_comparison_status():
    return comparison_state


@app.get("/api/stats")
def get_system_stats():
    return collect_system_stats()


@app.get("/api/gguf/resolve")
def resolve_gguf_paths_endpoint(
    model_id: str = "gpt2",
    regular_quant_mode: str = "gguf-f16",
    quant_mode: str = "gguf-q4",
):
    return resolve_gguf_paths(model_id, regular_quant_mode, quant_mode)


@app.get("/api/gguf/list")
def list_gguf_files_endpoint():
    return list_gguf_files()


@app.get("/api/models/catalog")
def get_models_catalog():
    return build_model_catalog()


@app.get("/api/capabilities")
def get_capabilities():
    return {"quantization": build_quantization_capabilities()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
