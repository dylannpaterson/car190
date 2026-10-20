"""Plot tile-136 grouped data versus model and residuals for MaxEnt cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DETECTORS = ("wfi01", "wfi10")
CASES = (
    ("maxent", "MaxEnt, weight=0.1"),
    ("no_entropy", "No entropy regularization"),
)
LED_COLORS = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#f0e442",
    "#56b4e9",
)
RESET_DAY = 12.907397928240743


def load_predictions(root, detector, case):
    path = (
        Path(root)
        / detector
        / "diagnostics"
        / "tile_136_production_arhgcdte_maxent_compare"
        / case
        / "tile_group_predictions.csv"
    )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    values = {
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "observed": np.asarray(
            [float(row["observed_log_change"]) for row in rows]
        ),
        "model": np.asarray([float(row["model_log_change"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }
    values["rms"] = float(np.sqrt(np.mean(values["residual"] ** 2)))
    return values


def common_format(axis):
    axis.axvline(RESET_DAY, color="0.4", linestyle="--", linewidth=0.8)
    axis.grid(alpha=0.2)


def plot_data_vs_model(series, output):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, detector in enumerate(DETECTORS):
        for column, (case, title) in enumerate(CASES):
            axis = axes[row, column]
            values = series[detector][case]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                axis.scatter(
                    values["days"][mask],
                    1000.0 * values["observed"][mask],
                    color=color,
                    s=8,
                    alpha=0.45,
                    edgecolors="none",
                )
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["model"][mask][order],
                    color=color,
                    linewidth=1.0,
                )
            common_format(axis)
            axis.set_title(f"{detector.upper()} | {title}\nRMS={values['rms']:.6f}")
            if row == 1:
                axis.set_xlabel("Days since first exposure")
            if column == 0:
                axis.set_ylabel(f"{detector.upper()}\nlog change [10^-3]")
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="data"),
        Line2D([], [], color="black", linewidth=1.2, label="model"),
        Line2D([], [], color="0.4", linestyle="--", linewidth=0.8, label="ice reset"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    figure.suptitle("WFI tile 136: corrected grouped data versus MAP model", fontsize=14)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_residuals(series, output):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, detector in enumerate(DETECTORS):
        for column, (case, title) in enumerate(CASES):
            axis = axes[row, column]
            values = series[detector][case]
            for led, color in enumerate(LED_COLORS, start=1):
                mask = values["led"] == led
                order = np.argsort(values["days"][mask])
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["residual"][mask][order],
                    color=color,
                    marker=".",
                    markersize=2.5,
                    linewidth=0.7,
                )
            common_format(axis)
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_title(f"{detector.upper()} | {title}\nRMS={values['rms']:.6f}")
            if row == 1:
                axis.set_xlabel("Days since first exposure")
            if column == 0:
                axis.set_ylabel(f"{detector.upper()}\nobserved - model [10^-3]")
    handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"LED {led}")
        for led, color in enumerate(LED_COLORS, start=1)
    ]
    axes[0, 0].legend(handles=handles, ncol=3, frameon=False, fontsize=8, loc="best")
    figure.suptitle("WFI tile 136: MAP residuals versus time", fontsize=14)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(root, output_dir):
    series = {
        detector: {
            case: load_predictions(root, detector, case)
            for case, _ in CASES
        }
        for detector in DETECTORS
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data_vs_model(series, output_dir / "tile136_data_vs_model_maxent_comparison.png")
    plot_residuals(series, output_dir / "tile136_residuals_maxent_comparison.png")
    print(f"Wrote {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.output_dir)
