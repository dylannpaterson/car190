"""Plot spline and non-spline ice accumulation-rate trajectories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CURRENT_COLOR = "#0072B2"
SPLINE_COLOR = "#D55E00"
RESET_COLOR = "#555555"


def _label_for_path(path):
    tile = path.parent.name.replace("tile_", "")
    detector = path.parents[2].name.upper()
    return f"{detector} tile {tile}"


def load_comparison(path):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")
    required = {
        "days",
        "current_rate_nm_day",
        "spline_rate_nm_day",
        "delta_rate_nm_day",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing trajectory fields in {path}: {sorted(missing)}")
    comparison_path = path.with_name("comparison.json")
    with comparison_path.open() as handle:
        metadata = json.load(handle)
    return {
        "path": path,
        "label": _label_for_path(path),
        "days": np.asarray([float(row["days"]) for row in rows]),
        "current_rate": np.asarray(
            [float(row["current_rate_nm_day"]) for row in rows]
        ),
        "spline_rate": np.asarray(
            [float(row["spline_rate_nm_day"]) for row in rows]
        ),
        "delta_rate": np.asarray(
            [float(row["delta_rate_nm_day"]) for row in rows]
        ),
        "reset_day": float(metadata["reset_day"]),
    }


def plot_comparisons(comparison_paths, output_path):
    comparisons = [load_comparison(path) for path in comparison_paths]
    row_count = len(comparisons)
    figure, axes = plt.subplots(
        row_count,
        2,
        figsize=(12.0, max(7.0, 2.6 * row_count)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, comparison in enumerate(comparisons):
        rate_axis, delta_axis = axes[row_index]
        days = comparison["days"]
        rate_axis.plot(
            days,
            comparison["current_rate"],
            color=CURRENT_COLOR,
            linewidth=1.2,
            label="Current non-spline NP",
        )
        rate_axis.plot(
            days,
            comparison["spline_rate"],
            color=SPLINE_COLOR,
            linewidth=1.2,
            label="72-knot natural cubic spline",
        )
        delta_axis.plot(
            days,
            comparison["delta_rate"],
            color=SPLINE_COLOR,
            linewidth=1.1,
        )
        for axis in (rate_axis, delta_axis):
            axis.axvline(
                comparison["reset_day"],
                color=RESET_COLOR,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )
            axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
            axis.set_axisbelow(True)
        delta_axis.axhline(0.0, color="#333333", linewidth=0.8)
        rate_axis.set_title(comparison["label"], loc="left", fontsize=10)
        rate_axis.set_ylabel("Rate (nm/day)")
        delta_axis.set_ylabel("Spline - NP\n(nm/day)")
        delta_axis.text(
            0.99,
            0.90,
            f"max |delta| = {np.max(np.abs(comparison['delta_rate'])):.3g}",
            transform=delta_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#444444",
        )
        if row_index == row_count - 1:
            rate_axis.set_xlabel("Days since first observation")
            delta_axis.set_xlabel("Days since first observation")
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[0, 0].set_title(
        f"{comparisons[0]['label']}\nInstantaneous accumulation rate",
        loc="left",
        fontsize=10,
    )
    axes[0, 1].set_title("Spline-rate difference", loc="left", fontsize=10)
    figure.suptitle(
        "20261016 ice-trajectory rate comparison\n"
        "Dashed line: physical Decon2/reset boundary",
        fontsize=13,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        action="append",
        required=True,
        help="trajectory_comparison.csv; repeat for each tile",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = plot_comparisons(args.comparison, args.output)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
