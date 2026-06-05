#!/usr/bin/env python3
"""Convert run JSON into Excel-friendly TSV/CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
import os
from typing import Any, Iterable, Optional


def load_items(path: str | None) -> list[dict[str, Any]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if not text.strip():
        raise ValueError("No JSON input provided.")

    data = json.loads(text)

    if isinstance(data, dict):
        for key in ("data", "results", "items"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of objects.")

    items: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            items.append(item)
    return items


def coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def extract_latency_seconds(result: dict[str, Any]) -> Optional[float]:
    for key in ("duration", "latency", "latency_seconds", "latency_s", "total_latency"):
        if key in result:
            try:
                value = float(result[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    for key in ("duration_ms", "latency_ms"):
        if key in result:
            try:
                value_ms = float(result[key])
            except (TypeError, ValueError):
                continue
            if value_ms > 0:
                return value_ms / 1000.0

    latency_per_token = result.get("avg_latency_per_input_token_ms") or result.get(
        "latency_per_input_token_ms"
    )
    input_tokens = result.get("input_tokens")
    if latency_per_token is not None and input_tokens is not None:
        try:
            per_token_ms = float(latency_per_token)
            tokens = float(input_tokens)
        except (TypeError, ValueError):
            return None
        if per_token_ms > 0 and tokens > 0:
            return (per_token_ms * tokens) / 1000.0

    return None


def extract_prefill_seconds(result: dict[str, Any]) -> Optional[float]:
    for key in ("prefill_duration", "prefill_seconds", "prefill_s", "avg_prefill_duration"):
        if key in result:
            try:
                value = float(result[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    for key in ("prefill_ms", "avg_prefill_duration_ms"):
        if key in result:
            try:
                value_ms = float(result[key])
            except (TypeError, ValueError):
                continue
            if value_ms > 0:
                return value_ms / 1000.0

    return None


def extract_rows(items: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0

    for item in items:
        result = coerce_mapping(item.get("result") or {})
        params = coerce_mapping(item.get("params") or {})
        quant_mode = item.get("quant_mode") or result.get("quant_mode")

        num_requests = params.get("num_requests")
        energy_joules = result.get("energy_joules", item.get("energy_joules"))
        total_latency_seconds = extract_latency_seconds(result)
        prefill_seconds = extract_prefill_seconds(result)

        if (
            quant_mode is None
            or num_requests is None
            or energy_joules is None
            or total_latency_seconds is None
        ):
            skipped += 1
            continue

        try:
            num_requests_int = int(num_requests)
            energy_float = float(energy_joules)
            total_latency_float = float(total_latency_seconds)
        except (TypeError, ValueError):
            skipped += 1
            continue

        if total_latency_float <= 0 or num_requests_int <= 0:
            skipped += 1
            continue

        latency_float = total_latency_float / num_requests_int
        power_watts = energy_float / total_latency_float

        rows.append(
            {
                "quant_mode": str(quant_mode),
                "num_requests": num_requests_int,
                "energy_joules": energy_float,
                "latency_seconds": latency_float,
                "prefill_seconds": prefill_seconds,
                "power_watts": power_watts,
            }
        )

    return rows, skipped


def aggregate_values(values: list[float], mode: str) -> float:
    if mode == "avg":
        return sum(values) / len(values)
    if mode == "sum":
        return sum(values)
    if mode == "min":
        return min(values)
    if mode == "max":
        return max(values)
    if mode == "median":
        return statistics.median(values)
    raise ValueError(f"Unsupported aggregate mode: {mode}")


def aggregate_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "none":
        return rows

    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        key = (row["quant_mode"], row["num_requests"])
        bucket = grouped.setdefault(key, {"energy": [], "latency": [], "prefill": []})
        bucket["energy"].append(row["energy_joules"])
        bucket["latency"].append(row["latency_seconds"])
        prefill_value = row.get("prefill_seconds")
        if prefill_value is not None:
            bucket["prefill"].append(prefill_value)

    aggregated: list[dict[str, Any]] = []
    for (quant_mode, num_requests), values in grouped.items():
        energy_value = aggregate_values(values["energy"], mode)
        latency_value = aggregate_values(values["latency"], mode)
        denom = latency_value * num_requests if latency_value and num_requests else 0
        power_value = energy_value / denom if denom > 0 else None
        prefill_values = values.get("prefill") or []
        prefill_value = (
            aggregate_values(prefill_values, mode) if prefill_values else None
        )
        aggregated.append(
            {
                "quant_mode": quant_mode,
                "num_requests": num_requests,
                "energy_joules": energy_value,
                "latency_seconds": latency_value,
                "prefill_seconds": prefill_value,
                "power_watts": power_value,
            }
        )

    return aggregated


def write_long(rows: list[dict[str, Any]], writer: csv.writer) -> None:
    writer.writerow(
        [
            "quant_mode",
            "num_requests",
            "energy_joules",
            "latency_seconds",
            "prefill_seconds",
            "power_watts",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["quant_mode"], r["num_requests"])):
        prefill_value = row.get("prefill_seconds")
        prefill_cell = "" if prefill_value is None else prefill_value
        writer.writerow(
            [
                row["quant_mode"],
                row["num_requests"],
                row["energy_joules"],
                row["latency_seconds"],
                prefill_cell,
                row["power_watts"],
            ]
        )


def write_wide(rows: list[dict[str, Any]], writer: csv.writer, value_key: str) -> None:
    quant_modes = sorted({row["quant_mode"] for row in rows})
    num_requests = sorted({row["num_requests"] for row in rows})
    value_map = {
        (row["quant_mode"], row["num_requests"]): row[value_key]
        for row in rows
    }

    writer.writerow(["num_requests", *quant_modes])
    for num in num_requests:
        row = [num]
        for quant_mode in quant_modes:
            value = value_map.get((quant_mode, num))
            row.append("" if value is None else value)
        writer.writerow(row)


def build_series(rows: list[dict[str, Any]], value_key: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        series.setdefault(row["quant_mode"], []).append((row["num_requests"], value))
    return series


def plot_rows(rows: list[dict[str, Any]], plot_file: Optional[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(
            "Skipping plot: matplotlib is required for plotting. Install it (pip install matplotlib).",
            file=sys.stderr,
        )
        return

    metrics = [
        ("energy_joules", "Energy (J)", ""),
        ("latency_seconds", "Latency per request (s)", ""),
        ("prefill_seconds", "Prefill (s)", ""),
        ("power_watts", "Power (W)", ""),
    ]

    cols = 2
    num_rows = max(1, math.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(num_rows, cols, figsize=(8 * cols, 4 * num_rows), sharex=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (key, y_label, title) in zip(axes_list, metrics):
        series = build_series(rows, key)
        for quant_mode, points in sorted(series.items()):
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=str(quant_mode))

        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="quant_mode")

    for ax in axes_list[len(metrics) :]:
        ax.set_visible(False)

    for ax in axes_list[: len(metrics)]:
        ax.set_xlabel("num_requests")
    fig.tight_layout()

    backend = plt.get_backend().lower()
    display_env = bool(os.environ.get("DISPLAY"))
    non_interactive_backends = {"agg", "template", "pdf", "ps", "svg"}

    if plot_file:
        fig.savefig(plot_file)
        plt.close(fig)
        print(f"Saved plot to {plot_file}", file=sys.stderr)
        return

    # If backend is non-interactive or no DISPLAY, save a fallback file
    if backend in non_interactive_backends or not display_env:
        fallback = "plot.png"
        fig.savefig(fallback)
        plt.close(fig)
        print(f"No interactive display available — saved plot to {fallback}", file=sys.stderr)
        return

    # Otherwise try to show; if that fails, save fallback
    try:
        plt.show()
    except Exception:
        fallback = "plot.png"
        fig.savefig(fallback)
        plt.close(fig)
        print(f"Could not display plot — saved to {fallback}", file=sys.stderr)


def open_output(path: str | None):
    if path:
        return open(path, "w", encoding="utf-8", newline="")
    return sys.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert JSON runs to TSV/CSV for Excel charting."
    )
    parser.add_argument("-i", "--input", help="Path to JSON file. Reads stdin if omitted.")
    parser.add_argument("-o", "--output", help="Output file. Writes stdout if omitted.")
    parser.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format (default: tsv).",
    )
    parser.add_argument(
        "--aggregate",
        choices=["avg", "sum", "min", "max", "median", "none"],
        default="avg",
        help="Aggregate duplicate (quant_mode, num_requests) pairs.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Output pivoted columns (num_requests x quant_mode).",
    )
    parser.add_argument(
        "--wide-metric",
        choices=["energy", "latency", "power"],
        default="energy",
        help="Metric to use for wide output (default: energy).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots of energy, latency, prefill, and power vs num_requests (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-file",
        help="Save plots to this filename instead of showing them (e.g. out.png).",
    )

    args = parser.parse_args()

    try:
        items = load_items(args.input)
        rows, skipped = extract_rows(items)
        if not rows:
            print("No valid rows found.", file=sys.stderr)
            return 2

        if args.wide and args.aggregate == "none":
            print(
                "Warning: --wide needs unique pairs; using avg aggregation.",
                file=sys.stderr,
            )
            rows = aggregate_rows(rows, "avg")
        else:
            rows = aggregate_rows(rows, args.aggregate)

        delimiter = "\t" if args.format == "tsv" else ","
        out = open_output(args.output)
        writer = csv.writer(out, delimiter=delimiter)

        if args.wide:
            value_key = {
                "energy": "energy_joules",
                "latency": "latency_seconds",
                "power": "power_watts",
            }[args.wide_metric]
            write_wide(rows, writer, value_key)
        else:
            write_long(rows, writer)

        # Optionally plot
        if args.plot or args.plot_file:
            try:
                plot_rows(rows, args.plot_file)
            except Exception as exc:
                print(f"Plotting error: {exc}", file=sys.stderr)
                return 1

        if out is not sys.stdout:
            out.close()

        print(f"Rows: {len(rows)} (skipped {skipped})", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
