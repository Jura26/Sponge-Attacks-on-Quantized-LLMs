# ...existing imports...
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import psutil
import uvicorn
import platform
import asyncio
import sys
import os
import random
import gc

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except Exception:
    # Backend still works if python-dotenv is not installed.
    pass

# Tell the Linux OOM killer to target this process before other processes
# (e.g. VS Code) when RAM runs out during large model loading.
# Score 500 means "kill me first" — range is -1000 (never kill) to 1000 (always kill).
try:
    with open("/proc/self/oom_score_adj", "w") as _f:
        _f.write("500")
except OSError:
    pass

# Add current directory to path for local module imports
sys.path.append(os.path.dirname(__file__))
from evolutionary_sponge import run_sponge_attack
from context_exhaustion import run_context_exhaustion
from autodos_attack import run_autodos_attack
from model import cleanup_model, resolve_gguf_variant_path, get_last_gguf_selection

app = FastAPI()


def _quant_mode_capabilities():
    """Return runtime quantization capabilities for the current machine.

    Used by the frontend to disable unsupported modes before a run starts.
    """
    import torch

    modes = {
        "gguf-f16": {"supported": True, "reason": "GGUF F16 baseline"},
        "gguf-q8": {"supported": True, "reason": "GGUF Q8_0"},
        "gguf-q6": {"supported": True, "reason": "GGUF Q6_K"},
        "gguf-q5": {"supported": True, "reason": "GGUF Q5_*"},
        "gguf-q4": {"supported": True, "reason": "GGUF Q4_*"},
        "gguf-q3": {"supported": True, "reason": "GGUF Q3_*"},
        "gguf-q2": {"supported": True, "reason": "GGUF Q2_*"},
    }

    gpu_available = torch.cuda.is_available()
    gpu_name = None
    rocm_arch = None
    hip_version = getattr(torch.version, "hip", None)

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            rocm_arch = str(getattr(props, "gcnArchName", "") or "")
        except Exception:
            rocm_arch = ""

    # GGUF backend availability checks
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        for key in modes:
            modes[key]["supported"] = False
            modes[key]["reason"] = "Install llama-cpp-python in backend venv"

    if all(m["supported"] for m in modes.values()):
        gguf_path = os.environ.get("SPONGE_GGUF_PATH", "").strip()
        gguf_dir = os.environ.get("SPONGE_GGUF_DIR", "").strip()
        has_path = bool(gguf_path and os.path.isfile(gguf_path))
        has_dir = bool(
            gguf_dir and os.path.isdir(gguf_dir) and any(
                name.lower().endswith(".gguf") for name in os.listdir(gguf_dir)
            )
        )
        if not (has_path or has_dir):
            for key in modes:
                modes[key]["supported"] = False
                modes[key]["reason"] = "Set valid SPONGE_GGUF_PATH or SPONGE_GGUF_DIR with .gguf files"

    # Inform UI whether GGUF backend is CPU-only on this installation.
    gguf_gpu_offload = None
    try:
        from llama_cpp import llama_supports_gpu_offload
        gguf_gpu_offload = bool(llama_supports_gpu_offload())
        if all(m["supported"] for m in modes.values()) and not gguf_gpu_offload:
            for key in modes:
                modes[key]["reason"] = "CPU-only llama-cpp-python build (no GPU offload support)"
    except Exception:
        pass

    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "rocm_arch": rocm_arch,
        "hip_version": hip_version,
        "gguf_gpu_offload": gguf_gpu_offload,
        "modes": modes,
    }

# Store attack status in memory 
# TODO: In production, consider Redis or a persistent database.
attack_state = {
    "is_running": False,
    "status": "idle",
    "logs": [],
    "current_generation": 0,
    "total_generations": 0,
    "best_result": None
}

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
):
    """Background task wrapper for the attack script."""
    global attack_state
    
    def callback(data):
        """Update global state with progress from the script."""
        global attack_state
        if data.get("status") == "eval":
            # Per-prompt evaluation updates
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
                # CPU load removed as per user request
                "avg_gpu": data.get("best_avg_gpu", 0),
                "duration": data.get("best_duration", 0),
                "input_tokens": data.get("best_input_tokens", 0),
                "output_tokens": data.get("best_output_tokens", 0),
                "energy_joules": data.get("best_energy", 0)
            }
            # Log best of gen
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
        else:
            # Generic status update
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
        
        if attack_type == "evolutionary":
            run_sponge_attack(model_id, gens=gens, pop=pop, quantize=quant_mode, progress_callback=callback)
        elif attack_type == "context_exhaustion":
            run_context_exhaustion(
                model_id,
                num_requests=num_requests,
                is_quantized=quant_mode,
                progress_callback=callback,
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
                progress_callback=callback,
            )
        
    except Exception as e:
        attack_state["status"] = "error"
        attack_state["is_running"] = False
        attack_state["logs"].append(f"Error: {str(e)}")

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
    global attack_state
    if attack_state["is_running"]:
        return {"error": "Attack already running"}
    
    # Reset state
    attack_state = {
        "is_running": True,
        "status": "queued",
        "logs": [],
        "current_generation": 0,
        "total_generations": gens if attack_type == "evolutionary" else 0,
        "best_result": None
    }
    
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
    global attack_state
    return attack_state

# ── A/B Comparison: Regular vs Quantized ─────────────────────

comparison_state = {
    "is_running": False,
    "phase": "idle",            # idle | regular | quantized | complete | error
    "regular_result": None,
    "quantized_result": None,
    "regular_logs": [],
    "quantized_logs": [],
    "regular_model_id": None,
    "quantized_model_id": None,
    "regular_gguf_path": None,
    "quantized_gguf_path": None,
    "current_generation": 0,
    "total_generations": 0,
}

def _make_comparison_callback(target_logs_key: str):
    """Return a progress callback that writes into comparison_state."""
    def callback(data):
        global comparison_state
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
                # CPU load removed as per user request
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
    context_mode: str = "combined",
    force_decode_tokens: int = 64,
    disable_eos_stop: bool = False,
):
    """Run the sponge attack twice: GGUF F16 baseline, then GGUF quantized variant."""
    global comparison_state
    import torch

    try:
        # Free memory before first run
        print("🧹 [main.py] Verifying VRAM is clear before taking baseline...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"🧹 [main.py] VRAM before baseline: {allocated:.2f} GB")

        # ── Phase 1: Regular (fp16) ──
        comparison_state["phase"] = "regular"
        comparison_state["regular_gguf_path"] = resolve_gguf_variant_path(model_id_a, regular_quant_mode)
        comparison_state["regular_logs"].append(f"═══ Phase 1/2: Model A ({model_id_a}) ═══")
        random.seed(seed)
        
        if attack_type == "evolutionary":
            run_sponge_attack(
                model_id_a, gens=gens, pop=pop, quantize=regular_quant_mode,
                progress_callback=_make_comparison_callback("regular_logs"),
            )
        elif attack_type == "context_exhaustion":
            run_context_exhaustion(
                model_id_a,
                num_requests=num_requests,
                is_quantized=regular_quant_mode,
                progress_callback=_make_comparison_callback("regular_logs"),
                context_mode=context_mode,
                force_decode_tokens=force_decode_tokens,
                disable_eos_stop=disable_eos_stop,
            )
        elif attack_type == "autodos":
            run_autodos_attack(
                model_id_a, num_iterations=autodos_iterations,
                depth=tree_depth, breadth=tree_breadth, is_quantized=regular_quant_mode,
                progress_callback=_make_comparison_callback("regular_logs"),
            )
        comparison_state["regular_gguf_path"] = get_last_gguf_selection().get("path")

        # Free memory between runs
        print("🧹 [main.py] Verifying VRAM is clear between phases...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"🧹 [main.py] VRAM after inter-phase cleanup: {allocated:.2f} GB")

        # ── Phase 2: Quantized GGUF variant ──
        comparison_state["phase"] = "quantized"
        comparison_state["current_generation"] = 0
        comparison_state["quantized_gguf_path"] = resolve_gguf_variant_path(model_id_b, quant_mode)
        comparison_state["quantized_logs"].append(
            f"═══ Phase 2/2: Model B ({model_id_b}) ═══"
        )
        random.seed(seed)
        
        if attack_type == "evolutionary":
            run_sponge_attack(
                model_id_b, gens=gens, pop=pop, quantize=quant_mode,
                progress_callback=_make_comparison_callback("quantized_logs"),
            )
        elif attack_type == "context_exhaustion":
            run_context_exhaustion(
                model_id_b,
                num_requests=num_requests,
                is_quantized=quant_mode,
                progress_callback=_make_comparison_callback("quantized_logs"),
                context_mode=context_mode,
                force_decode_tokens=force_decode_tokens,
                disable_eos_stop=disable_eos_stop,
            )
        elif attack_type == "autodos":
            run_autodos_attack(
                model_id_b, num_iterations=autodos_iterations,
                depth=tree_depth, breadth=tree_breadth, is_quantized=quant_mode,
                progress_callback=_make_comparison_callback("quantized_logs"),
            )
        comparison_state["quantized_gguf_path"] = get_last_gguf_selection().get("path")

        comparison_state["phase"] = "complete"
        comparison_state["is_running"] = False

    except Exception as e:
        comparison_state["phase"] = "error"
        comparison_state["is_running"] = False
        target = "quantized_logs" if comparison_state.get("regular_result") else "regular_logs"
        comparison_state[target].append(f"Error: {str(e)}")


@app.post("/api/attack/compare")
def start_comparison(
    background_tasks: BackgroundTasks,
    model_id: str = "facebook/opt-2.7b",
    model_id_a: str | None = None,
    model_id_b: str | None = None,
    gens: int = 5,
    pop: int = 10,
    attack_type: str = "evolutionary",
    num_requests: int = 10,
    autodos_iterations: int = 3,
    tree_depth: int = 3,
    tree_breadth: int = 4,
    regular_quant_mode: str = "gguf-f16",
    quant_mode: str = "gguf-q4",
    context_mode: str = "combined",
    force_decode_tokens: int = 64,
    disable_eos_stop: bool = False,
):
    global comparison_state
    if comparison_state["is_running"]:
        return {"error": "Comparison already running"}

    seed = random.randint(0, 2**31)

    resolved_a = model_id_a or model_id
    resolved_b = model_id_b or model_id

    comparison_state = {
        "is_running": True,
        "phase": "queued",
        "regular_result": None,
        "quantized_result": None,
        "regular_logs": [],
        "quantized_logs": [],
        "regular_model_id": resolved_a,
        "quantized_model_id": resolved_b,
        "regular_gguf_path": resolve_gguf_variant_path(resolved_a, regular_quant_mode),
        "quantized_gguf_path": resolve_gguf_variant_path(resolved_b, quant_mode),
        "current_generation": 0,
        "total_generations": gens if attack_type == "evolutionary" else 0,
    }

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
        regular_quant_mode,
        quant_mode,
        context_mode,
        force_decode_tokens,
        disable_eos_stop,
    )
    return {"message": "Comparison started"}


@app.get("/api/attack/compare/status")
def get_comparison_status():
    global comparison_state
    return comparison_state

@app.get("/api/stats")
def get_system_stats():
    # CPU
    cpu_total = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    
    # Memory
    mem = psutil.virtual_memory()
    
    # Disk
    disk = psutil.disk_usage('/')
    
    # Adjust disk usage path for Windows
    if platform.system() == "Windows":
        disk = psutil.disk_usage('C:\\')
    else:
        disk = psutil.disk_usage('/')

    # Battery
    battery = psutil.sensors_battery()
    battery_info = {
        "percent": battery.percent if battery else None,
        "power_plugged": battery.power_plugged if battery else None,
        "secsleft": battery.secsleft if battery else None
    }

    stats = {
        "cpu_percent": cpu_total,
        "cpu_per_core": cpu_per_core,
        "memory_percent": mem.percent,
        "memory_total": mem.total,
        "memory_used": mem.used,
        "disk_percent": disk.percent,
        "disk_free": disk.free,
        "battery": battery_info,
        "temperatures": {}
    }

    # Collect temperatures from all available sources
    try:
        if platform.system() == "Linux":
            temps = psutil.sensors_temperatures()
            if not temps:
                stats["temperatures"]["error"] = "No sensors found"
            else:
                for name, entries in temps.items():
                    stats["temperatures"][name] = []
                    for entry in entries:
                        stats["temperatures"][name].append({
                            "label": entry.label or name,
                            "current": entry.current,
                            "high": entry.high,
                            "critical": entry.critical
                        })
        elif platform.system() == "Windows":
            found_any = False

            # --- Primary: LibreHardwareMonitorLib via .NET ---
            try:
                from hardware_monitor import get_all_sensors
                sensor_data = get_all_sensors()
                
                for group_name, readings in sensor_data.items():
                    if readings:
                        stats["temperatures"][group_name] = readings
                        found_any = True
            except Exception as lhm_err:
                stats["temperatures"]["_lhm_error"] = str(lhm_err)

            # --- Fallback: ACPI Thermal Zones (no admin needed) ---
            if not found_any:
                try:
                    import wmi
                    import pythoncom
                    pythoncom.CoInitialize()
                    w = wmi.WMI(namespace="root\\cimv2")
                    zones = w.Win32_PerfFormattedData_Counters_ThermalZoneInformation()
                    if zones:
                        stats["temperatures"]["acpi_thermal_zones"] = []
                        for zone in zones:
                            celsius = float(zone.Temperature) - 273.15
                            stats["temperatures"]["acpi_thermal_zones"].append({
                                "label": zone.Name,
                                "current": round(celsius, 2),
                                "high": None,
                                "critical": None,
                                "source": "ACPI"
                            })
                            found_any = True
                except Exception:
                    pass

            if not found_any:
                stats["temperatures"]["error"] = (
                    "No sensors found. Make sure LibreHardwareMonitorLib.dll "
                    "is in the backend/lib/ folder and pythonnet is installed."
                )

    except Exception as e:
        stats["temperatures"]["error"] = str(e)

    return stats


@app.get("/api/gguf/resolve")
def resolve_gguf_paths(
    model_id: str = "gpt2",
    regular_quant_mode: str = "gguf-f16",
    quant_mode: str = "gguf-q4",
):
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


@app.get("/api/gguf/list")
def list_gguf_files():
    gguf_dir = os.environ.get("SPONGE_GGUF_DIR", "").strip()
    gguf_path = os.environ.get("SPONGE_GGUF_PATH", "").strip()

    files = []
    seen = set()

    def _add_file(path: str):
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


@app.get("/api/capabilities")
def get_capabilities():
    return {"quantization": _quant_mode_capabilities()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
