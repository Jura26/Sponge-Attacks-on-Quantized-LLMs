"""Linux-only system monitoring helpers for attack runs."""

from __future__ import annotations

import os
import statistics
import threading
import time
from typing import Dict, Optional, Tuple

import psutil


class SystemMonitor:
    """Collect CPU/GPU stats while a workload runs.

    GPU metrics are read from AMD sysfs nodes under /sys/class/drm.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.running = False
        self.stats: Dict[str, list] = {
            "temps": [],
            "cpu": [],
            "power": [],
            "gpu_load": [],
            "gpu_temp": [],
        }
        self._amd_paths: Optional[Dict[str, Optional[str]]] = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.token_count = 0
        self.thread: Optional[threading.Thread] = None

    def _find_amd_paths(self) -> Dict[str, Optional[str]]:
        if self._amd_paths is not None:
            return self._amd_paths
        base = "/sys/class/drm"
        power_path = None
        temp_path = None
        load_path = None
        try:
            for entry in os.listdir(base):
                if not entry.startswith("card"):
                    continue
                hwmon_base = os.path.join(base, entry, "device", "hwmon")
                if not os.path.isdir(hwmon_base):
                    continue
                for hwmon in os.listdir(hwmon_base):
                    hwmon_dir = os.path.join(hwmon_base, hwmon)
                    for fname in ("power1_average", "power1_input"):
                        candidate = os.path.join(hwmon_dir, fname)
                        if os.path.isfile(candidate):
                            power_path = candidate
                            break
                    temp_candidate = os.path.join(hwmon_dir, "temp1_input")
                    if os.path.isfile(temp_candidate):
                        temp_path = temp_candidate
                    if power_path and temp_path:
                        break
                load_candidate = os.path.join(base, entry, "device", "gpu_busy_percent")
                if os.path.isfile(load_candidate):
                    load_path = load_candidate
                if power_path or temp_path or load_path:
                    break
        except Exception:
            pass
        self._amd_paths = {
            "power": power_path,
            "temp": temp_path,
            "load": load_path,
        }
        return self._amd_paths

    def _read_amd_stats(self) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        paths = self._find_amd_paths()
        if not paths:
            return None
        power_w = None
        temp_c = None
        load_pct = None
        try:
            if paths.get("power"):
                with open(paths["power"], "r", encoding="utf-8") as f:
                    micro_watts = int(f.read().strip())
                    power_w = micro_watts / 1_000_000.0
        except Exception:
            pass
        try:
            if paths.get("temp"):
                with open(paths["temp"], "r", encoding="utf-8") as f:
                    milli_c = int(f.read().strip())
                    temp_c = milli_c / 1000.0
        except Exception:
            pass
        try:
            if paths.get("load"):
                with open(paths["load"], "r", encoding="utf-8") as f:
                    load_pct = float(f.read().strip())
        except Exception:
            pass
        if power_w is None and temp_c is None and load_pct is None:
            return None
        return power_w, load_pct, temp_c

    def _get_temp(self) -> float:
        max_temp = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return 0.0
            for entries in temps.values():
                for entry in entries:
                    current = getattr(entry, "current", 0.0)
                    if current > max_temp:
                        max_temp = current
        except Exception:
            return 0.0
        return max_temp

    def _monitor_loop(self) -> None:
        while self.running:
            self.stats["temps"].append(self._get_temp())
            self.stats["cpu"].append(psutil.cpu_percent(interval=None))

            if self.device == "cuda":
                amd_stats = self._read_amd_stats()
                if amd_stats:
                    power_w, load_pct, temp_c = amd_stats
                    if power_w is not None and power_w > 0:
                        self.stats["power"].append(power_w)
                    if load_pct is not None:
                        self.stats["gpu_load"].append(min(load_pct, 100.0))
                    if temp_c is not None:
                        self.stats["gpu_temp"].append(temp_c)

            time.sleep(0.1)

    def start(self) -> None:
        self.running = True
        self.stats = {"temps": [], "cpu": [], "power": [], "gpu_load": [], "gpu_temp": []}
        self.start_time = time.time()
        self.token_count = 0
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self, token_count: int = 0) -> None:
        self.running = False
        self.end_time = time.time()
        self.token_count = token_count
        if self.thread:
            self.thread.join()

    def get_score(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Return (score, max_temp, tps, avg_cpu, avg_gpu, duration, avg_power, energy)."""
        if not self.stats["temps"] and not self.stats["cpu"]:
            return 0, 0, 0, 0, 0, 0, 0, 0

        avg_temp = statistics.mean(self.stats["temps"]) if self.stats["temps"] else 0
        max_temp = max(self.stats["temps"]) if self.stats["temps"] else 0
        avg_cpu = statistics.mean(self.stats["cpu"]) if self.stats["cpu"] else 0

        avg_gpu_load = min(statistics.mean(self.stats["gpu_load"]), 100.0) if self.stats.get("gpu_load") else 0
        avg_power = statistics.mean(self.stats["power"]) if self.stats.get("power") else 0

        duration = self.end_time - self.start_time
        if duration <= 0:
            duration = 0.001

        tps = self.token_count / duration
        energy_joules = avg_power * duration
        score = energy_joules if energy_joules > 0 else duration

        return score, max_temp, tps, avg_cpu, avg_gpu_load, duration, avg_power, energy_joules
