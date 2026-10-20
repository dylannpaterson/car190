"""Plot MaxEnt-on versus entropy-off tile-136 trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production


DETECTORS = ("wfi01", "wfi10")
CASES = (
    ("maxent", "MaxEnt, weight=0.1", "#0072b2"),
    ("no_entropy", "No entropy regularization", "#d55e00"),
)
RESET_DAY = 12.907397928240743


def load_case(root, detector, case):
    detector_root = Path(root) / detector
    statistics_path = (
        detector_root
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )
    posterior_path = (
        detector_root
        / "diagnostics"
        / "tile_136_production_arhgcdte_maxent_compare"
        / case
        / "posterior_samples.npz"
    )
    data = production.load_tile_statistics(statistics_path)
    with np.load(posterior_path, allow_pickle=False) as archive:
        thickness = np.asarray(archive["node_ice_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        mean_rate = float(np.asarray(archive["tile_mean_rate_nm_day"]).reshape(-1)[0])
        rms = float(np.asarray(archive["map_group_rms_log"]).reshape(-1)[0])
    days = np.asarray(data["node_days"], dtype=float)
    if days.shape != thickness.shape or days.shape != rates.shape:
        raise ValueError(
            f"Trajectory shape mismatch for {detector} {case}: "
            f"days={days.shape}, thickness={thickness.shape}, rates={rates.shape}"
        )
    return days, thickness, rates, mean_rate, rms


def make_plot(root, output):
    series = {
        detector: {case: load_case(root, detector, case) for case, _, _ in CASES}
        for detector in DETECTORS
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex="col",
        constrained_layout=True,
    )
    for row, detector in enumerate(DETECTORS):
        thickness_axis, rate_axis = axes[row]
        for case, label, color in CASES:
            days, thickness, rates, mean_rate, rms = series[detector][case]
            thickness_axis.plot(
                days,
                thickness,
                color=color,
                linewidth=1.8,
                label=f"{label} (RMS={rms:.6f})",
            )
            rate_axis.plot(
                days,
                rates,
                color=color,
                linewidth=1.8,
                label=f"{label} ({mean_rate:.2f} nm/day mean)",
            )
        for axis in (thickness_axis, rate_axis):
            axis.axvline(
                RESET_DAY,
                color="0.35",
                linestyle="--",
                linewidth=0.9,
                label="Reset day" if axis is thickness_axis else None,
            )
            axis.grid(alpha=0.2)
        thickness_axis.set_ylabel(f"{detector.upper()}\nice thickness [nm]")
        rate_axis.set_ylabel(f"{detector.upper()}\nrate [nm/day]")
        thickness_axis.legend(frameon=False, fontsize=8, loc="best")
        rate_axis.legend(frameon=False, fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("Days since first exposure")
    axes[-1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].set_title("Ice thickness trajectory")
    axes[0, 1].set_title("Instantaneous accumulation rate")
    figure.suptitle(
        "WFI tile 136: AR/HgCdTe production MAP with and without MaxEnt",
        fontsize=14,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plot(args.root, args.output)
