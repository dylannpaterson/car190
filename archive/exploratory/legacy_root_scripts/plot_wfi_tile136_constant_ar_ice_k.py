"""Plot tile-136 accumulation and final AR/ice-k model residuals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production


DETECTORS = ("wfi01", "wfi10")
LED_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442", "#56b4e9")
RESET_DAY = 12.907397928240743


def statistics_path(root, detector):
    root = Path(root)
    direct = root / f"{detector}_tile_log_statistics_256_tile_136_count_corrected.csv"
    if direct.is_file():
        return direct
    return (
        root
        / detector
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )


def load_fit(fit_root, statistics_root, detector):
    fit_dir = Path(fit_root) / f"{detector}_tile136"
    data = production.load_tile_statistics(statistics_path(statistics_root, detector))
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as archive:
        thickness = np.asarray(archive["node_ice_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        ice_k = float(np.asarray(archive["ice_k_constant"], dtype=float).reshape(-1)[0])
        ar_k = np.asarray(archive["ar_k_constant"], dtype=float).reshape(-1)
    with (fit_dir / "tile_group_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    predictions = {
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "repeat": np.asarray([int(row["repeat"]) for row in rows]),
        "observed": np.asarray(
            [float(row["observed_log_change"]) for row in rows]
        ),
        "model": np.asarray([float(row["model_log_change"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }
    return {
        "days": np.asarray(data["node_days"], dtype=float),
        "thickness": thickness,
        "rates": rates,
        "ice_k": ice_k,
        "ar_k": ar_k,
        "predictions": predictions,
    }


def plot_accumulation(fits, output):
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )
    for detector, color in zip(DETECTORS, ("#0072b2", "#d55e00")):
        fit = fits[detector]
        label = (
            f"{detector.upper()} | ice k={fit['ice_k']:.3g}; "
            f"AR k=[{', '.join(f'{value:.3g}' for value in fit['ar_k'])}]"
        )
        axes[0].plot(fit["days"], fit["thickness"], color=color, linewidth=1.5, label=label)
        axes[1].plot(fit["days"], fit["rates"], color=color, linewidth=1.5)
    for axis in axes:
        axis.axvline(RESET_DAY, color="0.4", linestyle="--", linewidth=0.9)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Ice thickness [nm]")
    axes[1].set_ylabel("Instantaneous rate [nm/day]")
    axes[1].set_xlabel("Days since first exposure")
    axes[0].set_title("WFI tile 136: fitted ice accumulation")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_data_model_residuals(fits, output):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex="col",
        constrained_layout=True,
    )
    for row, detector in enumerate(DETECTORS):
        values = fits[detector]["predictions"]
        for led, color in enumerate(LED_COLORS, start=1):
            mask = values["led"] == led
            order = np.argsort(values["days"][mask])
            axes[row, 0].scatter(
                values["days"][mask],
                1000.0 * values["observed"][mask],
                color=color,
                s=8,
                alpha=0.45,
                edgecolors="none",
            )
            axes[row, 0].plot(
                values["days"][mask][order],
                1000.0 * values["model"][mask][order],
                color=color,
                linewidth=0.9,
            )
            axes[row, 1].plot(
                values["days"][mask][order],
                1000.0 * values["residual"][mask][order],
                color=color,
                marker=".",
                markersize=2.0,
                linewidth=0.65,
            )
        for axis in axes[row]:
            axis.axvline(RESET_DAY, color="0.4", linestyle="--", linewidth=0.9)
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.grid(alpha=0.2)
        rms = float(np.sqrt(np.mean(values["residual"] ** 2)))
        axes[row, 0].set_title(f"{detector.upper()} data and model")
        axes[row, 1].set_title(f"{detector.upper()} residuals, RMS={rms:.6f}")
        axes[row, 0].set_ylabel("log change [10^-3]")
        axes[row, 1].set_ylabel("observed - model [10^-3]")
    axes[1, 0].set_xlabel("Days since first exposure")
    axes[1, 1].set_xlabel("Days since first exposure")
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="data"),
        Line2D([], [], color="black", linewidth=1.2, label="model"),
        Line2D([], [], color="0.4", linestyle="--", linewidth=0.9, label="ice reset"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    figure.suptitle("WFI tile 136: final AR-k plus ice-k model", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=Path("20261014/constant_ar_ice_k_fresh_map"),
    )
    parser.add_argument(
        "--statistics-root",
        type=Path,
        default=Path("20261012/corrected_statistics_program1063_4h/production/tile_size_256"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits = {
        detector: load_fit(args.fit_root, args.statistics_root, detector)
        for detector in DETECTORS
    }
    plot_accumulation(fits, args.output_dir / "tile136_ice_accumulation.png")
    plot_data_model_residuals(
        fits,
        args.output_dir / "tile136_model_data_residuals.png",
    )
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
