"""Plot WFI tile statistics before and after fixed gain/ACM correction."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fit_wfi_tmm_numpyro import apply_fixed_corrections, load_tile_statistics


def plot_corrected_data(output, data, corrected):
    raw_mean = data["sums"] / data["counts"]
    corrected_mean = corrected["sums"] / corrected["counts"]
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    detector = str(np.asarray(data.get("sca_names", ["WFI"])).reshape(-1)[0])

    figure, axes = plt.subplots(
        2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        indices = np.flatnonzero(leds == led)
        x = days[indices]
        raw = raw_mean[:, indices]
        corrected_values = corrected_mean[:, indices]
        raw_median = np.median(raw, axis=0)
        corrected_quantiles = np.percentile(corrected_values, [16, 84], axis=0)
        for tile_index in range(raw.shape[0]):
            axis.plot(x, raw[tile_index], color="0.65", alpha=0.16, linewidth=0.45)
            axis.plot(
                x,
                corrected_values[tile_index],
                color="tab:blue",
                alpha=0.16,
                linewidth=0.45,
            )
        axis.fill_between(
            x,
            corrected_quantiles[0],
            corrected_quantiles[1],
            color="tab:blue",
            alpha=0.20,
            label="corrected 16th-84th percentile",
        )
        axis.plot(x, raw_median, color="0.25", linewidth=1.4, label="raw median")
        axis.plot(
            x,
            np.median(corrected_values, axis=0),
            color="tab:blue",
            linewidth=1.8,
            label="corrected median",
        )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("Tile mean log change")
    figure.suptitle(
        f"{detector} tile statistics: raw versus {detector} fixed gain/ACM corrected data"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_correction_curves(output, data, corrected):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    detector = str(np.asarray(data.get("sca_names", ["WFI"])).reshape(-1)[0])
    log_gain = np.asarray(corrected["fixed_log_gain_correction"], dtype=float)
    total = np.asarray(corrected["fixed_total_log_correction"], dtype=float)
    if log_gain.ndim == 2:
        log_gain = np.median(log_gain, axis=0)
        total = np.median(total, axis=0)
    acm_component = total - log_gain

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for led in range(1, 7):
        indices = np.flatnonzero(leds == led)
        axes[0].plot(
            days[indices],
            log_gain[indices],
            color=colors(led - 1),
            label=f"LED {led}",
        )
        axes[1].plot(days[indices], total[indices], color=colors(led - 1), label=f"LED {led}")
    axes[0].set_ylabel("Log gain correction")
    axes[1].set_ylabel("Total applied correction")
    axes[1].set_xlabel("Days since first exposure")
    axes[0].set_title(f"{detector} fixed correction components")
    axes[1].set_title(
        "Total correction applied to tile statistics "
        f"(three known ACM steps; component range {acm_component.min():.4g} "
        f"to {acm_component.max():.4g})"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=None)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=None)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--correction-output", type=Path, required=True)
    args = parser.parse_args()

    data = load_tile_statistics(
        args.statistics,
        additional_acm_step_repeat=args.additional_acm_step_repeat,
        pre_reset_acm_step_repeat=args.pre_reset_acm_step_repeat,
    )
    corrected = apply_fixed_corrections(data, args.corrections)
    plot_corrected_data(args.data_output, data, corrected)
    plot_correction_curves(args.correction_output, data, corrected)
    log_gain = np.asarray(corrected["fixed_log_gain_correction"], dtype=float)
    total = np.asarray(corrected["fixed_total_log_correction"], dtype=float)
    print(f"Wrote {args.data_output}")
    print(f"Wrote {args.correction_output}")
    print(
        f"tiles={data['n_tiles']}; groups={len(data['groups'])}; "
        f"log_gain_range=[{log_gain.min():.6g}, {log_gain.max():.6g}]; "
        f"total_correction_range=[{total.min():.6g}, {total.max():.6g}]"
    )


if __name__ == "__main__":
    main()
