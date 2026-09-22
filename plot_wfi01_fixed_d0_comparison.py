"""Plot matched WFI01 tile-0 fully-NP fits with fixed or free initial ice."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_wfi_tmm_numpyro import load_model_prediction_quantiles


ROOT = Path(__file__).resolve().parent
DEFAULT_FIXED = ROOT / "wfi01_tile0_fully_np_fixed_d0_5000"
DEFAULT_FREE = ROOT / "wfi01_tile0_fully_np_free_d0_5000"
DEFAULT_STATISTICS = ROOT / "wfi_tile_log_statistics_512" / "wfi01_tile_log_statistics_512.csv"


def read_node_table(output_dir):
    with (output_dir / "np_node_posterior.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in (
            "days",
            "instantaneous_rate_nm_day_median",
        )
    }


def read_fit(output_dir, statistics_path):
    with np.load(output_dir / "posterior_samples.npz") as samples:
        initial_thickness = float(
            np.asarray(samples["initial_thickness_nm"], dtype=float).reshape(-1)[0]
        )
        node_thickness = np.asarray(samples["np_node_thickness"], dtype=float)
        node_thickness = node_thickness.reshape(node_thickness.shape[0], -1)[0]
    data, prediction_quantiles = load_model_prediction_quantiles(
        output_dir,
        statistics_path,
        Path("unused-model-path"),
        "core",
        1,
        5.0,
        max_tiles=1,
    )
    return {
        "nodes": read_node_table(output_dir),
        "initial_thickness": initial_thickness,
        "node_thickness": node_thickness,
        "data": data,
        "predictions": prediction_quantiles,
    }


def led_series(fit, led):
    data = fit["data"]
    groups = np.flatnonzero(data["led"] == led)
    order = groups[np.argsort(data["days"][groups], kind="stable")]
    observed = data["sums"][0, order] / data["counts"][0, order]
    predicted = fit["predictions"][1, 0, order]
    return data["days"][order], observed, predicted


def plot_summary(path, fixed, free):
    colors = {"fixed": "#D55E00", "free": "#0072B2"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for fit, label, color in (
        (fixed, "fixed d0=0", colors["fixed"]),
        (free, "free d0", colors["free"]),
    ):
        nodes = fit["nodes"]
        axes[0, 0].plot(
            nodes["days"],
            nodes["instantaneous_rate_nm_day_median"],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            nodes["days"],
            fit["node_thickness"],
            marker="o",
            markersize=3,
            color=color,
            label=f"{label}; d0={fit['initial_thickness']:.2f} nm",
        )

    axes[0, 0].set_title("Fully-NP instantaneous rate")
    axes[0, 0].set_xlabel("Days since first exposure")
    axes[0, 0].set_ylabel("Rate (nm/day)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].set_title("Integrated ice thickness")
    axes[0, 1].set_xlabel("Days since first exposure")
    axes[0, 1].set_ylabel("Thickness (nm)")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    led_rms = {"fixed": [], "free": []}
    for led in range(1, 7):
        x, observed, fixed_predicted = led_series(fixed, led)
        _, _, free_predicted = led_series(free, led)
        fixed_residual = observed - fixed_predicted
        free_residual = observed - free_predicted
        led_rms["fixed"].append(np.sqrt(np.mean(fixed_residual**2)) * 1000.0)
        led_rms["free"].append(np.sqrt(np.mean(free_residual**2)) * 1000.0)

    positions = np.arange(1, 7)
    width = 0.36
    axes[1, 0].bar(
        positions - width / 2,
        led_rms["fixed"],
        width,
        color=colors["fixed"],
        label="fixed d0=0",
    )
    axes[1, 0].bar(
        positions + width / 2,
        led_rms["free"],
        width,
        color=colors["free"],
        label="free d0",
    )
    axes[1, 0].set_title("Per-LED RMS residual")
    axes[1, 0].set_xlabel("LED")
    axes[1, 0].set_ylabel("RMS residual (millilog)")
    axes[1, 0].set_xticks(positions)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.25)

    labels = ("fixed d0=0", "free d0")
    group_rms = []
    all_pixel_rms = []
    for fit in (fixed, free):
        residuals = []
        for led in range(1, 7):
            _, observed, predicted = led_series(fit, led)
            residuals.extend(observed - predicted)
        residuals = np.asarray(residuals)
        group_rms.append(np.sqrt(np.mean(residuals**2)) * 1000.0)
        all_pixel_rms.append(
            float("nan")
        )
    axes[1, 1].bar(
        np.arange(2),
        group_rms,
        color=(colors["fixed"], colors["free"]),
        width=0.6,
    )
    axes[1, 1].set_title("Overall group-mean RMS")
    axes[1, 1].set_ylabel("RMS residual (millilog)")
    axes[1, 1].set_xticks(np.arange(2), labels)
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("WFI01 tile 0: fixed initial ice versus free initial ice")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_led_fits(path, fixed, free):
    led_colors = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    )
    figure, axis = plt.subplots(figsize=(13, 8), constrained_layout=True)
    for led, color in enumerate(led_colors, start=1):
        x, observed, fixed_predicted = led_series(fixed, led)
        _, _, free_predicted = led_series(free, led)
        axis.scatter(
            x,
            100.0 * np.expm1(observed),
            color=color,
            s=18,
            alpha=0.55,
            label=f"LED {led} data",
        )
        axis.plot(
            x,
            100.0 * np.expm1(fixed_predicted),
            color=color,
            linewidth=2.4,
            linestyle="-",
            label=f"LED {led} fixed",
        )
        axis.plot(
            x,
            100.0 * np.expm1(free_predicted),
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"LED {led} free",
        )
        fixed_rms = np.sqrt(np.mean((observed - fixed_predicted) ** 2)) * 1000.0
        free_rms = np.sqrt(np.mean((observed - free_predicted) ** 2)) * 1000.0
        axis.text(
            0.99,
            0.97 - 0.045 * (led - 1),
            f"LED {led}: fixed {fixed_rms:.2f}, free {free_rms:.2f} mlog",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=8,
        )
    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([0], [0], color="black", linewidth=2.4, linestyle="-", label="fixed d0=0"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label="free d0"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", alpha=0.55, label="observed"),
    ]
    led_handles = [
        Line2D([0], [0], color=color, linewidth=2.0, label=f"LED {led}")
        for led, color in enumerate(led_colors, start=1)
    ]
    axis.set_title("WFI01 tile 0: all LEDs, fully-NP model comparison")
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Relative change (%)")
    color_legend = axis.legend(
        handles=led_handles,
        frameon=False,
        loc="upper left",
        ncol=2,
        title="LED color",
    )
    axis.add_artist(color_legend)
    axis.legend(handles=style_handles, frameon=False, loc="lower left", title="Line style")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--free-dir", type=Path, default=DEFAULT_FREE)
    parser.add_argument("--tile-statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "wfi01_tile0_fixed_d0_comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fixed = read_fit(args.fixed_dir, args.tile_statistics)
    free = read_fit(args.free_dir, args.tile_statistics)
    summary_path = args.output_prefix.with_suffix(".png")
    led_path = args.output_prefix.with_name(args.output_prefix.name + "_led_fits.png")
    plot_summary(summary_path, fixed, free)
    plot_led_fits(led_path, fixed, free)
    print(f"Wrote {summary_path}")
    print(f"Wrote {led_path}")


if __name__ == "__main__":
    main()
