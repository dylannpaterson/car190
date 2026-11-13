"""Plot WFI tile 136 rate trajectories across the regularization sweep."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production


ROOT = Path("20261012/corrected_statistics_program1063_4h/production/tile_size_256")
RESET_DAY = 12.907397928240743
DETECTORS = ("wfi01", "wfi10")
SWEEP = (
    ("Constant", "diagnostics/tile_136_led_offset_constant_rate/constant_rate", "#111111"),
    ("sigma=0.10", "diagnostics/trajectory_rate_prior_sweep/rate_sigma_0p10/fits/tile_136", "#0072b2"),
    ("sigma=0.25", "diagnostics/trajectory_rate_prior_sweep/rate_sigma_0p25/fits/tile_136", "#009e73"),
    ("sigma=0.50", "diagnostics/trajectory_rate_prior_sweep/rate_sigma_0p50/fits/tile_136", "#e69f00"),
    ("sigma=1.00", "diagnostics/trajectory_rate_prior_sweep/rate_sigma_1p00/fits/tile_136", "#d55e00"),
    ("sigma=1.50", "diagnostics/trajectory_rate_prior_sweep/rate_sigma_1p50/fits/tile_136", "#cc79a7"),
)


def load_rates(detector, fit_dir):
    statistics_path = (
        ROOT
        / detector
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )
    node_days = np.asarray(
        production.load_tile_statistics(statistics_path)["node_days"],
        dtype=float,
    )
    archive_path = ROOT / detector / fit_dir / "posterior_samples.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        rates = np.asarray(archive["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    if rates.size != node_days.size:
        raise ValueError(f"Rate/node mismatch for {detector}: {rates.size} != {node_days.size}")
    return node_days, rates


def load_residuals(detector, fit_dir):
    predictions_path = ROOT / detector / fit_dir / "tile_group_predictions.csv"
    with predictions_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    return {
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "repeat": np.asarray([int(row["repeat"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }


def plot_rate_sweep(output):
    figure, (axis, histogram_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 8),
        sharey=True,
        gridspec_kw={"width_ratios": (4.5, 1.2), "wspace": 0.04},
        constrained_layout=True,
    )
    rate_samples = {label: [] for label, _, _ in SWEEP}
    for detector in DETECTORS:
        detector_style = "-" if detector == "wfi01" else "--"
        for label, fit_dir, color in SWEEP:
            days, rates = load_rates(detector, fit_dir)
            rate_samples[label].append(rates)
            axis.plot(
                days,
                rates,
                color=color,
                linestyle=detector_style,
                linewidth=1.8 if label == "Constant" else 1.5,
            )
    axis.axvline(RESET_DAY, color="0.4", linestyle=":", linewidth=1.0)
    axis.grid(alpha=0.2)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("ice accumulation rate [nm/day]")
    for label, _, color in SWEEP:
        histogram_axis.hist(
            np.concatenate(rate_samples[label]),
            bins=28,
            orientation="horizontal",
            histtype="step",
            color=color,
            linewidth=1.5,
        )
    histogram_axis.set_xlabel("count")
    histogram_axis.tick_params(axis="y", labelleft=False)
    histogram_axis.grid(axis="x", alpha=0.2)
    histogram_axis.set_title("Rate\ndistribution", fontsize=10)
    rate_handles = [
        Line2D([], [], color=color, linewidth=1.8, label=label)
        for label, _, color in SWEEP
    ]
    detector_handles = [
        Line2D([], [], color="0.2", linestyle="-", label="WFI01"),
        Line2D([], [], color="0.2", linestyle="--", label="WFI10"),
    ]
    rate_legend = axis.legend(
        handles=rate_handles,
        title="Rate prior",
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    axis.add_artist(rate_legend)
    axis.legend(
        handles=detector_handles,
        title="Detector",
        frameon=False,
        loc="lower left",
    )
    figure.suptitle("WFI tile 136: accumulation-rate regularization sweep")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def plot_residual_sweep(output):
    figure, axis = plt.subplots(
        figsize=(12, 4.5),
        constrained_layout=True,
    )
    for detector in DETECTORS:
        detector_style = "-" if detector == "wfi01" else "--"
        for _, fit_dir, color in SWEEP:
            residuals = load_residuals(detector, fit_dir)
            nonreference = residuals["repeat"] != 0
            for led in range(1, 7):
                mask = nonreference & (residuals["led"] == led)
                order = np.argsort(residuals["days"][mask])
                axis.plot(
                    residuals["days"][mask][order],
                    1000.0 * residuals["residual"][mask][order],
                    color=color,
                    linestyle=detector_style,
                    linewidth=0.65,
                    marker=".",
                    markersize=2.0,
                    alpha=0.38,
                )
    axis.axvline(RESET_DAY, color="0.4", linestyle=":", linewidth=1.0)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.grid(alpha=0.2)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("model residual [10^-3]")
    axis.set_title("WFI tile 136: model residuals by rate prior")
    rate_handles = [
        Line2D([], [], color=color, linewidth=1.5, label=label)
        for label, _, color in SWEEP
    ]
    detector_handles = [
        Line2D([], [], color="0.2", linestyle="-", label="WFI01"),
        Line2D([], [], color="0.2", linestyle="--", label="WFI10"),
    ]
    rate_legend = axis.legend(
        handles=rate_handles,
        title="Rate prior",
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    axis.add_artist(rate_legend)
    axis.legend(
        handles=detector_handles,
        title="Detector",
        frameon=False,
        loc="lower left",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def main(output):
    output = Path(output)
    residual_output = output.with_name("tile136_rate_regularization_residuals.png")
    plot_rate_sweep(output)
    plot_residual_sweep(residual_output)


if __name__ == "__main__":
    output = ROOT / "wfi01" / "diagnostics" / "trajectory_rate_prior_sweep" / "tile136_rate_regularization_sweep.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    main(output)
