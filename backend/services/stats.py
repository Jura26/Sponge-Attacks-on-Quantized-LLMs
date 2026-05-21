"""System stats collection for the API."""

from __future__ import annotations

import psutil


def collect_system_stats() -> dict:
    cpu_total = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    battery = psutil.sensors_battery()
    battery_info = {
        "percent": battery.percent if battery else None,
        "power_plugged": battery.power_plugged if battery else None,
        "secsleft": battery.secsleft if battery else None,
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
        "temperatures": {},
    }

    try:
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
                        "critical": entry.critical,
                    })
    except Exception as exc:
        stats["temperatures"]["error"] = str(exc)

    return stats
