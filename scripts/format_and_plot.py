#!/usr/bin/env python3

import csv
import json
import os
import statistics
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

TARGET_NUM_REQUESTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "llama.txt")

OUT_CSV = os.path.join(
    BASE_DIR,
    "formatted_data_interpolated.csv",
)

PNG_ENERGY = os.path.join(
    BASE_DIR,
    "energy_vs_num_requests.png",
)

PNG_POWER = os.path.join(
    BASE_DIR,
    "avg_power_vs_num_requests.png",
)

PNG_TOKENS = os.path.join(
    BASE_DIR,
    "output_tokens_vs_num_requests.png",
)

PNG_LATENCY = os.path.join(
    BASE_DIR,
    "latency_vs_num_requests.png",
)

PNG_COMBINED = os.path.join(
    BASE_DIR,
    "energy_avg_power_latency_vs_num_requests.png",
)

# ============================================================
# HELPERS
# ============================================================

def extract_json_arrays(text: str) -> list:
    """
    Extract all JSON arrays from mixed log text.
    """

    decoder = json.JSONDecoder()

    arrays = []
    idx = 0

    while True:

        start = text.find("[", idx)

        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue

        if (
            isinstance(obj, list)
            and len(obj) > 0
            and isinstance(obj[0], dict)
            and "quant_mode" in obj[0]
        ):
            arrays.append(obj)

        idx = start + end

    if not arrays:
        raise ValueError("No JSON arrays found in data.txt")

    merged = []

    for arr in arrays:
        merged.extend(arr)

    return merged


def to_float(value):

    if isinstance(value, (int, float)):
        return float(value)

    return None


def parse_quant_mode(line: str):

    marker = "Quant:"

    if marker not in line:
        return None

    start = line.find(marker)

    if start == -1:
        return None

    start += len(marker)

    end = line.find(")", start)

    if end == -1:
        end = len(line)

    quant_mode = line[start:end].strip()

    return quant_mode or None


def parse_request_total(line: str):

    if not line.startswith("--- Request"):
        return None

    slash = line.find("/")

    if slash == -1:
        return None

    digits = []

    for ch in line[slash + 1:]:
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    if not digits:
        return None

    return int("".join(digits))


def parse_request_metrics(line: str):

    if "output tokens=" not in line or "latency=" not in line:
        return None

    token_start = line.find("output tokens=") + len("output tokens=")
    latency_start = line.find("latency=") + len("latency=")

    token_digits = []
    for ch in line[token_start:]:
        if ch.isdigit():
            token_digits.append(ch)
        else:
            break

    latency_digits = []
    for ch in line[latency_start:]:
        if ch.isdigit() or ch == ".":
            latency_digits.append(ch)
        else:
            break

    if not token_digits or not latency_digits:
        return None

    return int("".join(token_digits)), float("".join(latency_digits))


def parse_request_logs(text: str):

    phases = []
    current = None

    for line in text.splitlines():

        if "Starting LingoLoop Attack" in line and "Quant:" in line:

            quant_mode = parse_quant_mode(line)

            if quant_mode:
                if current is not None:
                    phases.append(current)

                current = {
                    "quant_mode": quant_mode,
                    "total_requests": None,
                    "latencies": [],
                    "output_tokens": [],
                }

            continue

        if current is None:
            continue

        total_requests = parse_request_total(line)

        if total_requests is not None:
            current["total_requests"] = total_requests
            continue

        metrics = parse_request_metrics(line)

        if metrics is not None:
            output_tokens, latency = metrics
            current["output_tokens"].append(output_tokens)
            current["latencies"].append(latency)

    if current is not None:
        phases.append(current)

    return phases


# ============================================================
# DATA PROCESSING
# ============================================================

def build_summary(entries: list, text: str, targets: list[int]) -> list[dict]:

    phases = parse_request_logs(text)

    avg_power_by_quant = defaultdict(list)

    for entry in entries:

        quant_mode = entry.get("quant_mode")
        result = entry.get("result") or {}
        avg_power = to_float(result.get("avg_power"))

        if quant_mode is None or avg_power is None:
            continue

        avg_power_by_quant[quant_mode].append(avg_power)

    rows = []

    for phase in phases:

        quant_mode = phase["quant_mode"]
        total_requests = phase["total_requests"] or len(phase["latencies"])
        latencies = phase["latencies"]
        output_tokens = phase["output_tokens"]

        avg_power_values = avg_power_by_quant.get(quant_mode, [])

        if not avg_power_values:
            continue

        avg_power = statistics.mean(avg_power_values)

        latency_per_request = (
            statistics.mean(latencies) if latencies else None
        )
        tokens_per_request = (
            statistics.mean(output_tokens) if output_tokens else None
        )

        observed_total_latency = sum(latencies) if latencies else None
        observed_total_tokens = sum(output_tokens) if output_tokens else None

        print()
        print(f"[INFO] {quant_mode}")
        print(f"       source num_requests: {total_requests}")
        print(f"       requests observed: {len(latencies)}")

        for target in targets:

            if latencies:
                if target <= len(latencies):
                    target_latency = sum(latencies[:target])
                    target_tokens = sum(output_tokens[:target])
                else:
                    target_latency = (
                        latency_per_request * target
                        if latency_per_request is not None
                        else None
                    )
                    target_tokens = (
                        tokens_per_request * target
                        if tokens_per_request is not None
                        else None
                    )
            else:
                target_latency = None
                target_tokens = None

            rows.append({
                "quant_mode": quant_mode,
                "num_requests": target,
                "score": target_latency * avg_power if target_latency is not None else None,
                "avg_power": avg_power,
                "output_tokens": target_tokens,
                "latency": target_latency,
                "count": len(latencies),
            })

        if observed_total_latency is not None:
            print(f"       observed total latency: {observed_total_latency:.2f}s")
        if observed_total_tokens is not None:
            print(f"       observed total output tokens: {observed_total_tokens}")

    json_grouped = defaultdict(list)

    for entry in entries:

        quant_mode = entry.get("quant_mode")
        params = entry.get("params") or {}
        result = entry.get("result") or {}

        num_requests = params.get("num_requests")
        avg_power = to_float(result.get("avg_power"))
        energy = to_float(result.get("energy_joules"))
        duration = to_float(result.get("duration"))
        output_tokens = to_float(result.get("output_tokens"))

        if quant_mode is None or num_requests is None:
            continue

        try:
            num_requests = int(num_requests)
        except Exception:
            continue

        if avg_power is None or duration is None or output_tokens is None:
            continue

        json_grouped[quant_mode].append({
            "num_requests": num_requests,
            "avg_power": avg_power,
            "latency_per_request": duration / num_requests if num_requests else None,
            "tokens_per_request": output_tokens / num_requests if num_requests else None,
            "energy_per_request": energy / num_requests if energy is not None and num_requests else None,
        })

    phase_quants = {phase["quant_mode"] for phase in phases}

    for quant_mode, samples in json_grouped.items():

        if quant_mode in phase_quants or not samples:
            continue

        avg_power = statistics.mean([sample["avg_power"] for sample in samples])
        latency_per_request = statistics.mean(
            [sample["latency_per_request"] for sample in samples if sample["latency_per_request"] is not None]
        ) if any(sample["latency_per_request"] is not None for sample in samples) else None
        tokens_per_request = statistics.mean(
            [sample["tokens_per_request"] for sample in samples if sample["tokens_per_request"] is not None]
        ) if any(sample["tokens_per_request"] is not None for sample in samples) else None

        print()
        print(f"[INFO] {quant_mode}")
        print(f"       source num_requests: {[sample['num_requests'] for sample in samples]}")
        print(f"       requests observed: 0 (JSON fallback)")

        for target in targets:
            target_latency = latency_per_request * target if latency_per_request is not None else None
            target_tokens = tokens_per_request * target if tokens_per_request is not None else None

            rows.append({
                "quant_mode": quant_mode,
                "num_requests": target,
                "score": target_latency * avg_power if target_latency is not None else None,
                "avg_power": avg_power,
                "output_tokens": target_tokens,
                "latency": target_latency,
                "count": 0,
            })

    rows.sort(
        key=lambda r: (
            r["quant_mode"],
            r["num_requests"],
        )
    )

    return rows


# ============================================================
# CSV
# ============================================================

def write_csv(rows: list, path: str):

    fieldnames = [
        "quant_mode",
        "num_requests",
        "score",
        "avg_power",
        "output_tokens",
        "latency",
        "count",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# PLOTTING
# ============================================================

def build_series(summary: list, metric: str):

    series = defaultdict(list)

    for row in summary:

        quant_mode = row.get("quant_mode")

        if quant_mode is None:
            continue

        value = row.get(metric)

        if value is None:
            continue

        series[quant_mode].append((
            row["num_requests"],
            value,
        ))

    return series


def plot_metric_png(
    summary: list,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    x_ticks: list[int],
):

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib required.\n"
            "Install with:\n"
            "pip install matplotlib"
        ) from exc

    series = build_series(summary, metric)

    if not series:
        print(f"[WARN] No data for metric: {metric}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_anything = False

    for quant_mode in sorted(series.keys()):

        points = series[quant_mode]

        if not points:
            continue

        points.sort(key=lambda p: p[0])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        print(
            f"[PLOT] metric={metric} "
            f"quant={quant_mode} "
            f"points={len(xs)}"
        )

        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=quant_mode,
        )

        plotted_anything = True

    if not plotted_anything:
        print(f"[WARN] Nothing plotted for {metric}")
        return

    ax.set_title(title)
    ax.set_xlabel("num_requests")
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(title="quant_mode")

    ax.set_xticks(x_ticks)

    ax.set_xlim(
        min(x_ticks),
        max(x_ticks),
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[OK] Saved: {filename}")


def plot_combined_metrics_png(summary: list, x_ticks: list[int], filename: str):

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib required.\n"
            "Install with:\n"
            "pip install matplotlib"
        ) from exc

    panels = [
        ("score", "Energy vs Num Requests", "Energy (Joules)"),
        ("avg_power", "Average Power vs Num Requests", "Average Power (W)"),
        ("latency", "Latency vs Num Requests", "Latency (seconds)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 14), sharex=True)

    if len(panels) == 1:
        axes = [axes]

    for ax, (metric, title, ylabel) in zip(axes, panels):
        series = build_series(summary, metric)

        for quant_mode in sorted(series.keys()):
            points = sorted(series[quant_mode], key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", linewidth=2, label=quant_mode)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("num_requests")
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xlim(min(x_ticks), max(x_ticks))

    axes[0].legend(title="quant_mode", loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("======================================")
    print("Loading data")
    print("======================================")

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as handle:

        text = handle.read()

    data = extract_json_arrays(text)

    print(f"Found {len(data)} total benchmark entries")

    # show quant modes discovered
    quant_modes = sorted(set([
        d.get("quant_mode")
        for d in data
        if d.get("quant_mode")
    ]))

    print()
    print("Quant modes found:")

    for q in quant_modes:
        print(" -", q)

    print()
    print("======================================")
    print("Building summary")
    print("======================================")

    summary = build_summary(
        data,
        text,
        TARGET_NUM_REQUESTS,
    )

    print()
    print(f"Generated {len(summary)} summary rows")

    print()
    print("======================================")
    print("Writing CSV")
    print("======================================")

    write_csv(summary, OUT_CSV)

    print(f"[OK] Saved: {OUT_CSV}")

    print()
    print("======================================")
    print("Generating graphs")
    print("======================================")

    plot_metric_png(
        summary=summary,
        metric="score",
        title="Energy vs Num Requests",
        ylabel="Energy (Joules)",
        filename=PNG_ENERGY,
        x_ticks=TARGET_NUM_REQUESTS,
    )

    plot_metric_png(
        summary=summary,
        metric="avg_power",
        title="Average Power vs Num Requests",
        ylabel="Average Power (W)",
        filename=PNG_POWER,
        x_ticks=TARGET_NUM_REQUESTS,
    )

    plot_metric_png(
        summary=summary,
        metric="output_tokens",
        title="Output Tokens vs Num Requests",
        ylabel="Output Tokens",
        filename=PNG_TOKENS,
        x_ticks=TARGET_NUM_REQUESTS,
    )

    plot_metric_png(
        summary=summary,
        metric="latency",
        title="Latency vs Num Requests",
        ylabel="Latency (seconds)",
        filename=PNG_LATENCY,
        x_ticks=TARGET_NUM_REQUESTS,
    )

    plot_combined_metrics_png(
        summary=summary,
        x_ticks=TARGET_NUM_REQUESTS,
        filename=PNG_COMBINED,
    )

    print()
    print("======================================")
    print("DONE")
    print("======================================")
    print()
    print("Generated files:")
    print(" - formatted_data_interpolated.csv")
    print(" - energy_vs_num_requests.png")
    print(" - avg_power_vs_num_requests.png")
    print(" - output_tokens_vs_num_requests.png")
    print(" - latency_vs_num_requests.png")
    print(" - energy_avg_power_latency_vs_num_requests.png")
    print()


if __name__ == "__main__":
    main()