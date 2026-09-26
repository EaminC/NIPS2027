"""CSV, JSON, and optional matplotlib output."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ab_aa_lab.experiment import RunResult


def write_results(results: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [to_builtin(result.summary) for result in results]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for result in results:
        write_csv(out_dir / f"{result.name}.csv", result.rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fields})


def plot_results(results: list[RunResult], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting needs matplotlib; pip install matplotlib") from exc

    if not results:
        raise ValueError("nothing to plot")
    feasible = [result for result in results if result.strategy != "oracle"]
    oracles = [result for result in results if result.strategy == "oracle"]
    panels: list[tuple[str, list[RunResult]]] = []
    if feasible:
        panels.append(("Feasible AB", feasible))
    if oracles:
        panels.append(("Oracle AB", oracles))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.8 * (len(panels) + 1), 4.3), sharex=True)
    if len(panels) + 1 == 1:
        axes = [axes]
    for axis, (title, group) in zip(axes, panels):
        _draw_ab(axis, group)
        axis.set_title(title)
        axis.set_ylabel("solid claim, dashed population")
    _draw_aa(axes[-1], results)
    axes[-1].set_title("AA latent")
    axes[-1].set_ylabel("population mean change")
    _mark_phases(axes, results[0].rows)
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xlabel("round")
        axis.grid(True, linewidth=0.3, alpha=0.6)
        axis.legend(
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _draw_ab(axis, results: list[RunResult]) -> None:
    for result in results:
        rounds = [row["round"] for row in result.rows]
        claim = axis.plot(
            rounds,
            [row["ab_claim"] for row in result.rows],
            label=result.name,
            linewidth=1.6,
        )[0]
        axis.plot(
            rounds,
            [row["ab_population"] for row in result.rows],
            color=claim.get_color(),
            linestyle="--",
            linewidth=1.1,
        )


def _draw_aa(axis, results: list[RunResult]) -> None:
    series = [[row["aa_latent"] for row in result.rows] for result in results]
    shared = all(item == series[0] for item in series[1:])
    if shared:
        axis.plot(
            [row["round"] for row in results[0].rows],
            series[0],
            color="black",
            linewidth=1.6,
            label="shared latent path",
        )
        return
    for result, values in zip(results, series):
        axis.plot(
            [row["round"] for row in result.rows],
            values,
            linewidth=1.6,
            label=result.name,
        )


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def format_summary(results: list[RunResult]) -> str:
    headers = (
        "name",
        "system",
        "strategy",
        "ab_claim_search",
        "ab_pop_search",
        "ab_claim_final",
        "aa_latent",
        "oracle_upper",
    )
    lines = ["  ".join(f"{header:>18}" for header in headers)]
    for result in results:
        summary = result.summary
        cells = [
            result.name,
            result.system,
            result.strategy,
            _num(summary.get("ab_claim_search_end")),
            _num(summary.get("ab_population_search_end")),
            _num(summary.get("ab_claim_final")),
            _num(summary.get("aa_latent_final")),
            _num(summary.get("oracle_upper_final")),
        ]
        lines.append("  ".join(f"{cell:>18}" for cell in cells))
    return "\n".join(lines)


def _mark_phases(axes, rows: list[dict]) -> None:
    if not rows:
        return
    phase = rows[0]["phase"]
    for row in rows[1:]:
        if row["phase"] != phase:
            for axis in axes:
                axis.axvline(row["round"] - 0.5, color="gray", linewidth=0.7, linestyle=":")
            phase = row["phase"]


def _csv_value(value):
    value = to_builtin(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return value


def _num(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"
