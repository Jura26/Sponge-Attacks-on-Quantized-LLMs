"""
Hardware monitoring via LibreHardwareMonitorLib (.NET DLL).
Provides per-core CPU temps, GPU temps, SSD temps, fan speeds, power draw, etc.

Setup:
  1. pip install pythonnet
  2. Download LibreHardwareMonitorLib.dll from:
     https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases
     (extract from the zip → net472/LibreHardwareMonitorLib.dll)
  3. Place the DLL in  backend/lib/LibreHardwareMonitorLib.dll
"""

import os
import sys
import threading

# --- Singleton Computer handle (expensive to create, reuse it) ---
_computer = None
_lock = threading.Lock()
_LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")
_DLL_NAME = "LibreHardwareMonitorLib.dll"


def _init_computer():
    """
    Initialise the LibreHardwareMonitor Computer object once.
    Must be called from a thread that has COM initialised on Windows.
    """
    global _computer

    if _computer is not None:
        return _computer

    with _lock:
        if _computer is not None:
            return _computer

        dll_path = os.path.join(_LIB_DIR, _DLL_NAME)
        if not os.path.isfile(dll_path):
            raise FileNotFoundError(
                f"LibreHardwareMonitorLib.dll not found at {dll_path}. "
                f"Download it from https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases "
                f"and place it in backend/lib/"
            )

        # pythonnet (.NET interop)
        import clr  # noqa: E402  (from pythonnet)
        clr.AddReference(dll_path)

        # Now we can import .NET types
        from LibreHardwareMonitor.Hardware import Computer  # type: ignore

        c = Computer()
        c.IsCpuEnabled = True
        c.IsGpuEnabled = True
        c.IsMemoryEnabled = True
        c.IsMotherboardEnabled = True
        c.IsStorageEnabled = True
        c.IsNetworkEnabled = False
        c.IsControllerEnabled = True
        c.Open()

        _computer = c
        return _computer


def _update_hardware(computer):
    """Call Update() on every hardware node and sub-hardware."""
    for hw in computer.Hardware:
        hw.Update()
        for sub in hw.SubHardware:
            sub.Update()


# Mapping from LHM SensorType enum int → friendly string (v0.9.4+)
_SENSOR_TYPE_MAP = {
    0: "Voltage",
    1: "Current",
    2: "Power",
    3: "Clock",
    4: "Temperature",
    5: "Load",
    6: "Frequency",
    7: "Fan",
    8: "Flow",
    9: "Control",
    10: "Level",
    11: "Factor",
    12: "Data",
    13: "SmallData",
    14: "Throughput",
    15: "TimeSpan",
    16: "Energy",
    17: "Noise",
}

_HARDWARE_TYPE_MAP = {
    0: "Motherboard",
    1: "SuperIO",
    2: "CPU",
    3: "RAM",
    4: "GPU (Nvidia)",
    5: "GPU (AMD)",
    6: "GPU (Intel)",
    7: "Storage",
    8: "Network",
    9: "Cooler",
    10: "EmbeddedController",
    11: "PSU",
    12: "Battery",
}


def get_all_sensors():
    """
    Returns a dict of sensor groups, each containing a list of readings.
    Groups: cpu_temps, gpu_temps, storage_temps, fans, power, motherboard_temps, etc.
    Each reading: {label, current, high, critical, source, unit}
    """
    import math

    computer = _init_computer()
    _update_hardware(computer)

    result = {}
    has_cpu_temp = False

    for hw in computer.Hardware:
        hw_type_int = int(hw.HardwareType)
        hw_type_name = _HARDWARE_TYPE_MAP.get(hw_type_int, f"Unknown({hw_type_int})")
        hw_name = str(hw.Name)

        # Collect sensors from hardware and sub-hardware
        all_sensors = list(hw.Sensors)
        for sub in hw.SubHardware:
            sub.Update()
            all_sensors.extend(list(sub.Sensors))

        for sensor in all_sensors:
            s_type_int = int(sensor.SensorType)
            s_type = _SENSOR_TYPE_MAP.get(s_type_int, "Other")
            value = sensor.Value
            if value is None:
                continue
            val = float(value)
            # Skip NaN values
            if math.isnan(val):
                continue
            val = round(val, 1)
            max_val = round(float(sensor.Max), 1) if sensor.Max and not math.isnan(float(sensor.Max)) else None
            s_name = str(sensor.Name)

            # Determine which group this sensor goes into
            group = None
            unit = ""
            source = hw_type_name

            if s_type == "Temperature":
                # Skip 0.0°C readings (usually means no permission / unavailable)
                if val == 0.0:
                    if hw_type_int == 2:  # CPU
                        # Mark that CPU temp failed (needs admin)
                        pass
                    continue
                unit = "°C"
                if hw_type_int == 2:   # CPU
                    group = "cpu_temperatures"
                    has_cpu_temp = True
                elif hw_type_int in (4, 5, 6):  # GPU
                    group = "gpu_temperatures"
                elif hw_type_int == 7:  # Storage
                    group = "storage_temperatures"
                elif hw_type_int in (0, 1):  # Motherboard / SuperIO
                    group = "motherboard_temperatures"
                else:
                    group = "other_temperatures"

            elif s_type == "Fan":
                group = "fans"
                unit = "RPM"

            elif s_type == "Power":
                # Skip 0W readings
                if val == 0.0:
                    continue
                group = "power"
                unit = "W"

            elif s_type == "Voltage":
                group = "voltages"
                unit = "V"

            elif s_type == "Load":
                # Only include summary loads (skip D3D engine detail noise)
                if "D3D" in s_name:
                    continue
                if hw_type_int in (2, 4, 5, 6):
                    group = "load"
                    unit = "%"

            elif s_type == "Clock":
                if hw_type_int in (2, 4, 5, 6):
                    # Skip 0 MHz clocks
                    if val == 0.0:
                        continue
                    group = "clocks"
                    unit = "MHz"

            elif s_type == "SmallData":
                # GPU memory info
                if "Memory" in s_name and hw_type_int in (4, 5, 6):
                    group = "gpu_memory"
                    unit = "MB"
                else:
                    continue

            if group is None:
                continue

            if group not in result:
                result[group] = []

            result[group].append({
                "label": f"{hw_name} — {s_name}",
                "current": val,
                "high": max_val,
                "critical": None,
                "source": source,
                "unit": unit,
            })

    # Add hint if CPU temps are missing (needs admin)
    if not has_cpu_temp:
        if "_info" not in result:
            result["_info"] = []
        result["_info"].append({
            "label": "CPU temperatures require running as Administrator",
            "current": "N/A",
            "high": None,
            "critical": None,
            "source": "Info",
            "unit": "",
        })

    return result


def get_max_temperature():
    """Return the highest temperature reading across all hardware (for sponge_attack scoring)."""
    import math

    computer = _init_computer()
    _update_hardware(computer)

    max_temp = 0.0
    for hw in computer.Hardware:
        for sensor in hw.Sensors:
            if int(sensor.SensorType) == 4 and sensor.Value is not None:  # Temperature
                t = float(sensor.Value)
                if not math.isnan(t) and t > max_temp:
                    max_temp = t
        for sub in hw.SubHardware:
            for sensor in sub.Sensors:
                if int(sensor.SensorType) == 4 and sensor.Value is not None:
                    t = float(sensor.Value)
                    if not math.isnan(t) and t > max_temp:
                        max_temp = t
    return max_temp


def get_gpu_stats():
    """Return current max GPU temperature and max GPU core load."""
    import math

    computer = _init_computer()
    _update_hardware(computer)

    max_temp = 0.0
    max_load = 0.0
    
    for hw in computer.Hardware:
        # Check if it is a GPU (Nvidia=4, AMD=5, Intel=6)
        # Note: HardwareType enum: 4=GpuNvidia, 5=GpuAmd, 6=GpuIntel
        if int(hw.HardwareType) in (4, 5, 6):
            for sensor in hw.Sensors:
                val = float(sensor.Value) if sensor.Value is not None else 0.0
                if math.isnan(val): continue
                
                # Temp (SensorType=4)
                if int(sensor.SensorType) == 4:
                    if val > max_temp:
                        max_temp = val
                
                # Load (SensorType=5)
                elif int(sensor.SensorType) == 5:
                    s_name = str(sensor.Name)
                    # "GPU Core" usually represents main load
                    if "Core" in s_name:
                        if val > max_load:
                            max_load = val
                    elif max_load == 0 and val > max_load: # Fallback
                        max_load = val
                        
    return max_temp, max_load


if __name__ == "__main__":
    # Quick test
    try:
        data = get_all_sensors()
        for group, readings in data.items():
            print(f"\n=== {group.upper()} ===")
            for r in readings:
                print(f"  {r['label']}: {r['current']}{r.get('unit','')}  (max: {r['high']})")
        print(f"\nMax temp: {get_max_temperature()}°C")
    except Exception as e:
        print(f"Error: {e}")

def get_gpu_stats():
    """Returns (max_gpu_temp, max_gpu_load) across all GPUs."""
    import math

    computer = _init_computer()
    _update_hardware(computer)

    max_temp = 0.0
    max_load = 0.0
    
    # GPU Hardware Types: 4 (Nvidia), 5 (AMD), 6 (Intel)
    gpu_types = (4, 5, 6)

    for hw in computer.Hardware:
        if int(hw.HardwareType) in gpu_types:
            for sensor in hw.Sensors:
                st = int(sensor.SensorType)
                if sensor.Value is None:
                    continue
                val = float(sensor.Value)
                if math.isnan(val):
                    continue
                
                # Temperature (SensorType 4)
                if st == 4:
                    if val > max_temp:
                        max_temp = val
                
                # Load (SensorType 5)
                # We want Core Load usually, but maximize across partial loads works too
                if st == 5:
                    if val > max_load:
                        max_load = val
        # Check sub-hardware too just in case
        for sub in hw.SubHardware:
             for sensor in sub.Sensors:
                st = int(sensor.SensorType)
                if sensor.Value is None: continue
                val = float(sensor.Value)
                if math.isnan(val): continue
                if st == 4 and val > max_temp: max_temp = val
                if st == 5 and val > max_load: max_load = val

    return max_temp, max_load
