"""Compare the matched rate-node and direct-thickness fits for tile 136."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production


RATE_ROOT = Path(
    "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
)
DIRECT_ROOT = Path("20261014/direct_thickness_fresh_emcee")
OUTPUT_ROOT = DIRECT_ROOT / "comparison"
RESET_DAY = 12.907397928240743
DETECTORS = ("wfi01", "wfi10")
LED_COLORS = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#f0e442",
    "#56b4e9",
)


def statistics_path(detector):
    return (
        RATE_ROOT
        / detector
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )


def fit_path(detector, parameterization):
    if parameterization == "rate":
        return (
            RATE_ROOT
            / detector
            / "diagnostics"
            / "production_arhgcdte_v2_led_offsets_hgcdte_nmodprior005"
            / "fits"
            / "tile_136"
        )
    return DIRECT_ROOT / detector / "direct_fit"


def load_archive(detector, parameterization):
    path = fit_path(detector, parameterization) / "posterior_samples.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            "node_ice": np.asarray(archive["node_ice_thickness"], dtype=float).reshape(-1),
            "node_rate": np.asarray(
                archive["node_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1),
        }


def load_predictions(detector, parameterization):
    path = fit_path(detector, parameterization) / "tile_group_predictions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    return {
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "repeat": np.asarray([int(row["repeat"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }


def plot_trajectories(output):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex="col",
        constrained_layout=True,
    )
    for column, detector in enumerate(DETECTORS):
        data = production.load_tile_statistics(statistics_path(detector))
        node_days = np.asarray(data["node_days"], dtype=float)
        rate = load_archive(detector, "rate")
        direct = load_archive(detector, "direct")
        thickness_axis = axes[0, column]
        rate_axis = axes[1, column]
        thickness_axis.plot(
            node_days,
            rate["node_ice"],
            color="#222222",
            linewidth=1.7,
            label="Rate-node fit",
        )
        thickness_axis.plot(
            node_days,
            direct["node_ice"],
            color="#0072b2",
            linewidth=1.4,
            label="Direct thickness fit",
        )
        rate_axis.plot(
            node_days,
            rate["node_rate"],
            color="#222222",
            linewidth=1.7,
        )
        rate_axis.plot(
            node_days,
            direct["node_rate"],
            color="#0072b2",
            linewidth=1.4,
        )
        for axis in (thickness_axis, rate_axis):
            axis.axvline(RESET_DAY, color="0.45", linestyle=":", linewidth=0.9)
            axis.grid(alpha=0.2)
        thickness_axis.set_title(detector.upper())
        thickness_axis.set_ylabel("ice thickness [nm]")
        rate_axis.set_ylabel("derived rate [nm/day]")
        rate_axis.set_xlabel("Days since first exposure")
        rate_axis.axhline(0.0, color="black", linewidth=0.7)
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.suptitle("WFI tile 136: rate-node versus direct-thickness fit")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def plot_residuals(output):
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, detector in zip(axes, DETECTORS):
        for parameterization, linestyle in (("rate", "-"), ("direct", "--")):
            values = load_predictions(detector, parameterization)
            nonreference = values["repeat"] != 0
            for led, color in enumerate(LED_COLORS, start=1):
                mask = nonreference & (values["led"] == led)
                order = np.argsort(values["days"][mask])
                axis.plot(
                    values["days"][mask][order],
                    1000.0 * values["residual"][mask][order],
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.7,
                    marker=".",
                    markersize=2.0,
                    alpha=0.55,
                )
        axis.axvline(RESET_DAY, color="0.45", linestyle=":", linewidth=0.9)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.set_ylabel(f"{detector.upper()}\nresidual [10^-3]")
    axes[-1].set_xlabel("Days since first exposure")
    fit_handles = [
        Line2D([], [], color="0.15", linestyle="-", label="Rate-node fit"),
        Line2D([], [], color="0.15", linestyle="--", label="Direct thickness fit"),
    ]
    led_handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"LED {led}")
        for led, color in enumerate(LED_COLORS, start=1)
    ]
    legend = axes[0].legend(
        handles=fit_handles + led_handles,
        ncol=4,
        frameon=False,
        loc="upper right",
    )
    axes[0].add_artist(legend)
    figure.suptitle("WFI tile 136: residual comparison")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plot_trajectories(OUTPUT_ROOT / "tile136_rate_vs_direct_thickness_trajectories.png")
    plot_residuals(OUTPUT_ROOT / "tile136_rate_vs_direct_thickness_residuals.png")


if __name__ == "__main__":
    main()
