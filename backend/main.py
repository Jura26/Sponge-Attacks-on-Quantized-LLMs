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

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(__file__))
from sponge_attack import run_sponge_attack
from model import cleanup_model

app = FastAPI()

# Store attack status in memory (simple solution for demo)
# In production, use Redis or a database.
attack_state = {
    "is_running": False,
    "status": "idle", # idle, starting, loading, running, complete, error
    "logs": [],       # List of progress messages
    "current_generation": 0,
    "total_generations": 0,
    "best_result": None
}

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sponge_attack_worker(model_id: str, gens: int, pop: int):
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
                "avg_cpu": data.get("best_avg_cpu", 0),
                "avg_gpu": data.get("best_avg_gpu", 0),
                "duration": data.get("best_duration", 0),
                "input_tokens": data.get("best_input_tokens", 0),
                "output_tokens": data.get("best_output_tokens", 0)
            }
            # Log best of gen
            gen_log = f"Gen {data.get('generation')}: Best Score {data.get('best_score'):.2f} (Temp: {data.get('best_temp')}C)"
            if data.get("best_avg_gpu", 0) > 0:
                gen_log += f" [GPU: {data.get('best_avg_gpu'):.1f}%]"
            else:
                gen_log += f" [CPU: {data.get('best_avg_cpu', 0):.1f}%]"
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
        attack_state["logs"] = ["Starting attack process..."]
        attack_state["total_generations"] = gens
        
        # Run the actual attack synchronously in this thread (it's already in a background task)
        run_sponge_attack(model_id, gens=gens, pop=pop, progress_callback=callback)
        
    except Exception as e:
        attack_state["status"] = "error"
        attack_state["is_running"] = False
        attack_state["logs"].append(f"Error: {str(e)}")

@app.post("/api/attack/start")
def start_attack(background_tasks: BackgroundTasks, model_id: str = "gpt2", gens: int = 5, pop: int = 10):
    global attack_state
    if attack_state["is_running"]:
        return {"error": "Attack already running"}
    
    # Reset state
    attack_state = {
        "is_running": True,
        "status": "queued",
        "logs": [],
        "current_generation": 0,
        "total_generations": gens,
        "best_result": None
    }
    
    background_tasks.add_task(sponge_attack_worker, model_id, gens, pop)
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
                "avg_cpu": data.get("best_avg_cpu", 0),
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


def comparison_worker(model_id: str, gens: int, pop: int, seed: int):
    """Run the sponge attack twice: regular (fp16), then quantized (bnb 4-bit)."""
    global comparison_state
    import torch

    try:
        # ── Phase 1: Regular (fp16) ──
        comparison_state["phase"] = "regular"
        comparison_state["regular_logs"].append(f"═══ Phase 1/2: Regular model ({model_id}) ═══")
        random.seed(seed)
        run_sponge_attack(
            model_id, gens=gens, pop=pop, quantize=False,
            progress_callback=_make_comparison_callback("regular_logs"),
        )

        # Free memory between runs
        print("🧹 [main.py] Verifying VRAM is clear between phases...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"🧹 [main.py] VRAM after inter-phase cleanup: {allocated:.2f} GB")

        # ── Phase 2: Quantized (bitsandbytes NF4 4-bit) ──
        comparison_state["phase"] = "quantized"
        comparison_state["current_generation"] = 0
        comparison_state["quantized_logs"].append(
            f"═══ Phase 2/2: Quantized model ({model_id} — 4-bit) ═══"
        )
        random.seed(seed)
        run_sponge_attack(
            model_id, gens=gens, pop=pop, quantize=True,
            progress_callback=_make_comparison_callback("quantized_logs"),
        )

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
    gens: int = 5,
    pop: int = 10,
):
    global comparison_state
    if comparison_state["is_running"]:
        return {"error": "Comparison already running"}

    seed = random.randint(0, 2**31)

    comparison_state = {
        "is_running": True,
        "phase": "queued",
        "regular_result": None,
        "quantized_result": None,
        "regular_logs": [],
        "quantized_logs": [],
        "regular_model_id": model_id,
        "quantized_model_id": model_id,
        "current_generation": 0,
        "total_generations": gens,
    }

    background_tasks.add_task(comparison_worker, model_id, gens, pop, seed)
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
