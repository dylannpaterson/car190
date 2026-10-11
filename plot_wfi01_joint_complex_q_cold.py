"""Plot and summarize the cold single-tile WFI01 complex-q MAP fit."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_FIT = ROOT / "wfi01_joint_complex_q_cold_map10000"
DEFAULT_ARCHIVE = (
    ROOT / "wfi_production_20261005_shared_hgcdte_prior10_smooth001_WFI01_map10000_emcee1000"
)
RESET_DAY = 15.064405023148149


def load_predictions(path):
    by_led = {led: [] for led in range(1, 7)}
    required = {
        "days",
        "led",
        "observed_log_change",
        "model_log_change",
        "residual_log",
    }
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
        for row in reader:
            led = int(row["led"])
            by_led[led].append(
                {
                    key: float(row[key])
                    for key in (
                        "days",
                        "observed_log_change",
                        "model_log_change",
                        "residual_log",
                    )
                }
            )
    if any(not rows for rows in by_led.values()):
        raise ValueError("Prediction table must contain all six LEDs")
    result = {}
    for led, rows in by_led.items():
        rows.sort(key=lambda row: row["days"])
        result[led] = {
            key: np.asarray([row[key] for row in rows], dtype=float)
            for key in rows[0]
        }
    return result


def load_csv_series(path, fields):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in fields
    }


def load_trajectory(output_dir, tile_index=0):
    output_dir = Path(output_dir)
    with np.load(output_dir / "posterior_samples.npz", allow_pickle=False) as archive:
        observation_days = np.asarray(archive["observation_days"], dtype=float)
        thickness = np.median(np.asarray(archive["ice_thickness"], dtype=float), axis=0)
        node_thickness = np.median(
            np.asarray(archive["tile_node_thickness"], dtype=float), axis=0
        )
    if thickness.ndim == 2:
        thickness = thickness[tile_index]
    if node_thickness.ndim == 2:
        node_thickness = node_thickness[tile_index]
    node_series = load_csv_series(
        output_dir / "np_node_posterior.csv",
        ("days", "instantaneous_rate_nm_day_median"),
    )
    if node_series["days"].size != node_thickness.size:
        raise ValueError("Node posterior and node-thickness sizes do not match")
    return {
        "observation_days": observation_days,
        "thickness": thickness,
        "node_days": node_series["days"],
        "node_thickness": node_thickness,
        "node_rate": node_series["instantaneous_rate_nm_day_median"],
    }


def load_q_parameters(output_dir):
    with np.load(
        Path(output_dir) / "posterior_samples.npz", allow_pickle=False
    ) as archive:
        names = np.asarray(archive["complex_q_parameter_names"]).astype(str)
        lower = np.asarray(archive["complex_q_lower"], dtype=float)
        upper = np.asarray(archive["complex_q_upper"], dtype=float)
    with (Path(output_dir) / "complex_q_posterior.csv").open(newline="") as handle:
        values = {row["parameter"]: float(row["median"]) for row in csv.DictReader(handle)}
    median = np.asarray([values[name] for name in names], dtype=float)
    return names, median, lower, upper


def add_reset_line(axis):
    axis.axvline(
        RESET_DAY,
        color="0.35",
        linestyle=":",
        linewidth=0.9,
        alpha=0.9,
    )
    axis.grid(alpha=0.2)


def plot_data_model(output, predictions, rms):
    figure, axes = plt.subplots(
        3, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        values = predictions[led]
        axis.scatter(
            values["days"],
            100.0 * np.expm1(values["observed_log_change"]),
            color="black",
            s=13,
            alpha=0.75,
            label="corrected data",
            zorder=3,
        )
        axis.plot(
            values["days"],
            100.0 * np.expm1(values["model_log_change"]),
            color="tab:blue",
            linewidth=1.5,
            label="joint complex-q model",
        )
        axis.set_title(f"LED {led}")
        add_reset_line(axis)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.supxlabel("Days since first exposure")
    figure.supylabel("Reference-normalized change (%)")
    figure.suptitle(f"WFI01 tile 0 cold joint fit: data versus model (RMS={rms:.6f} log)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_residuals(output, predictions, rms):
    figure, axes = plt.subplots(
        3, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        values = predictions[led]
        axis.plot(
            values["days"],
            1000.0 * values["residual_log"],
            color="tab:blue",
            marker=".",
            markersize=3,
            linewidth=0.8,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"LED {led}")
        add_reset_line(axis)
    figure.supxlabel("Days since first exposure")
    figure.supylabel("Observed - model (10^-3 log)")
    figure.suptitle(f"WFI01 tile 0 cold joint fit residuals (RMS={rms:.6f} log)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_trajectory(output, fit, archive):
    archive_at_fit_days = np.interp(
        fit["observation_days"], archive["observation_days"], archive["thickness"]
    )
    observation_difference = fit["thickness"] - archive_at_fit_days
    archive_at_fit_nodes = np.interp(
        fit["node_days"], archive["node_days"], archive["node_thickness"]
    )
    node_difference = fit["node_thickness"] - archive_at_fit_nodes

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(
        fit["observation_days"],
        fit["thickness"],
        color="tab:blue",
        linewidth=2,
        label="cold joint complex-q, tile 0",
    )
    axes[0, 0].plot(
        archive["observation_days"],
        archive["thickness"],
        color="tab:orange",
        linewidth=2,
        label="archived production, tile 0",
    )
    axes[0, 0].set_title("Fitted thickness at observation epochs")
    axes[0, 0].set_ylabel("Ice thickness (nm)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(
        fit["observation_days"],
        observation_difference,
        color="black",
        linewidth=1.6,
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Joint minus archived at observations")
    axes[0, 1].set_ylabel("Thickness difference (nm)")

    axes[1, 0].plot(
        fit["node_days"],
        fit["node_thickness"],
        "o-",
        color="tab:blue",
        markersize=3,
        linewidth=1.5,
        label="cold joint complex-q",
    )
    axes[1, 0].plot(
        archive["node_days"],
        archive["node_thickness"],
        "o-",
        color="tab:orange",
        markersize=3,
        linewidth=1.5,
        label="archived production",
    )
    axes[1, 0].set_title("Non-parametric node thickness")
    axes[1, 0].set_xlabel("Days since first exposure")
    axes[1, 0].set_ylabel("Node thickness (nm)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(
        fit["node_days"],
        node_difference,
        color="black",
        linewidth=1.6,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Joint minus archived at nodes")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[1, 1].set_ylabel("Thickness difference (nm)")
    for axis in axes.flat:
        add_reset_line(axis)
    figure.suptitle("WFI01 tile 0 ice trajectory comparison")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return observation_difference, node_difference


def plot_q_bounds(output, names, median, lower, upper):
    figure, axes = plt.subplots(5, 1, figsize=(10, 8), constrained_layout=True)
    for axis, name, value, low, high in zip(axes, names, median, lower, upper):
        axis.hlines(0.0, low, high, color="0.65", linewidth=4)
        axis.plot(value, 0.0, "o", color="tab:red", markersize=7)
        axis.set_xlim(low - 0.04 * (high - low), high + 0.04 * (high - low))
        axis.set_yticks([])
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
        axis.grid(axis="x", alpha=0.2)
        axis.text(
            1.0,
            0.6,
            f"{value:.6g}  [{low:.6g}, {high:.6g}]",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
    figure.suptitle("Cold joint complex-q MAP values and hard bounds")
    figure.supxlabel("Parameter value")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def print_metrics(predictions, fit, archive, q_values):
    residual = np.concatenate([predictions[led]["residual_log"] for led in range(1, 7)])
    days = np.concatenate([predictions[led]["days"] for led in range(1, 7)])
    print(f"group_rms_log={np.sqrt(np.mean(residual * residual)):.12g}")
    print(
        f"pre_reset_rms_log={np.sqrt(np.mean(residual[days < RESET_DAY] ** 2)):.12g}"
    )
    print(
        f"post_reset_rms_log={np.sqrt(np.mean(residual[days >= RESET_DAY] ** 2)):.12g}"
    )
    for led in range(1, 7):
        led_residual = predictions[led]["residual_log"]
        print(f"led_{led}_rms_log={np.sqrt(np.mean(led_residual ** 2)):.12g}")
    archive_at_fit_days = np.interp(
        fit["observation_days"], archive["observation_days"], archive["thickness"]
    )
    difference = fit["thickness"] - archive_at_fit_days
    print(f"joint_initial_thickness_nm={fit['thickness'][0]:.12g}")
    print(f"joint_final_thickness_nm={fit['thickness'][-1]:.12g}")
    print(f"archive_tile0_initial_thickness_nm={archive['thickness'][0]:.12g}")
    print(f"archive_tile0_final_thickness_nm={archive['thickness'][-1]:.12g}")
    print(f"joint_minus_archive_final_nm={difference[-1]:.12g}")
    print(f"joint_minus_archive_difference_rms_nm={np.sqrt(np.mean(difference ** 2)):.12g}")
    for name, value in zip(q_values[0], q_values[1]):
        print(f"{name}={value:.12g}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args.fit_dir / "tile_group_predictions.csv")
    residual = np.concatenate(
        [predictions[led]["residual_log"] for led in range(1, 7)]
    )
    rms = float(np.sqrt(np.mean(residual * residual)))
    fit = load_trajectory(args.fit_dir)
    archive = load_trajectory(args.archive_dir)
    q_values = load_q_parameters(args.fit_dir)

    plot_data_model(args.output_dir / "joint_complex_q_data_model.png", predictions, rms)
    plot_residuals(args.output_dir / "joint_complex_q_residuals.png", predictions, rms)
    plot_trajectory(
        args.output_dir / "joint_complex_q_trajectory_comparison.png", fit, archive
    )
    plot_q_bounds(args.output_dir / "joint_complex_q_parameter_bounds.png", *q_values)
    print_metrics(predictions, fit, archive, q_values)
    for name in (
        "joint_complex_q_data_model.png",
        "joint_complex_q_residuals.png",
        "joint_complex_q_trajectory_comparison.png",
        "joint_complex_q_parameter_bounds.png",
    ):
        print(f"Wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
