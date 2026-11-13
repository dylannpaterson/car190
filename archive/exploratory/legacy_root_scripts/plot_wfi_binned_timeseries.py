import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent

AXES = {
    "x": {
        "input_path": ROOT / "wfi01_x_binned_candlesticks.csv",
        "plot_path": ROOT / "wfi01_x_binned_timeseries.png",
        "label": "X pixel",
    },
    "y": {
        "input_path": ROOT / "wfi01_y_binned_candlesticks.csv",
        "plot_path": ROOT / "wfi01_y_binned_timeseries.png",
        "label": "Y pixel",
    },
}


def read_binned_rows(path):
    bins = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            bin_index = int(row["bin_index"])
            led = int(row["led"])
            bin_data = bins.setdefault(
                bin_index,
                {
                    "start": int(row["pixel_start"]),
                    "stop": int(row["pixel_end_exclusive"]),
                    "leds": {},
                },
            )
            bin_data["leds"].setdefault(led, []).append(
                {
                    "repeat": int(row["repeat"]),
                    "raw": np.asarray(
                        [
                            float(row["raw_p05"]),
                            float(row["raw_p25"]),
                            float(row["raw_median"]),
                            float(row["raw_p75"]),
                            float(row["raw_p95"]),
                        ]
                    ),
                }
            )
    for bin_data in bins.values():
        for records in bin_data["leds"].values():
            records.sort(key=lambda record: record["repeat"])
    return bins


colors = {led: plt.get_cmap("tab10")(led - 1) for led in range(1, 7)}


def get_limits(bins, key):
    values = np.concatenate(
        [
            np.concatenate([record[key] for record in records])
            for bin_data in bins.values()
            for records in bin_data["leds"].values()
        ]
    )
    values = values[np.isfinite(values)]
    lower, upper = np.percentile(values, [0.5, 99.5])
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    padding = max((upper - lower) * 0.08, 0.1)
    return lower - padding, upper + padding


def draw_candlesticks(axis, bin_data, key, limits, show_ylabel, show_xlabel):
    for led in range(1, 7):
        records = bin_data["leds"].get(led, [])
        if not records:
            continue
        repeats = np.asarray([record["repeat"] for record in records])
        quantiles = np.asarray([record[key] for record in records])
        offset = (led - 3.5) * 0.11
        x_values = repeats + offset
        color = colors[led]

        # The connected median line keeps the bin evolution visible between candles.
        axis.plot(x_values, quantiles[:, 2], color=color, alpha=0.55, linewidth=0.7)
        for x, values in zip(x_values, quantiles):
            q05, q25, median, q75, q95 = values
            axis.vlines(x, q05, q95, color=color, linewidth=0.55)
            axis.vlines(x, q25, q75, color=color, linewidth=3.0, alpha=0.9)
            axis.plot(
                x,
                median,
                marker="_",
                markersize=4,
                color="black",
                markeredgewidth=0.8,
            )

    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    axis.set_xlim(-0.75, 12.75)
    axis.set_ylim(limits)
    axis.set_xticks([0, 6, 12])
    axis.tick_params(labelsize=7)
    axis.grid(color="0.9", linewidth=0.4)
    if show_ylabel:
        axis.set_ylabel("Pixel change (%)", fontsize=8)
    else:
        axis.set_yticklabels([])
    if show_xlabel:
        axis.set_xlabel("Repeat", fontsize=8)
    else:
        axis.set_xticklabels([])


def plot_axis(axis_name, details):
    bins = read_binned_rows(details["input_path"])
    bin_indices = sorted(bins)
    raw_limits = get_limits(bins, "raw")

    bins_per_row = 6
    rows = int(np.ceil(len(bin_indices) / bins_per_row))
    columns = bins_per_row
    fig, axes = plt.subplots(rows, columns, figsize=(18, 2.45 * rows), squeeze=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.035, right=0.995, wspace=0.12, hspace=0.55)
    fig.suptitle(
        f"WFI01 RCS-flat candidate: raw 100-pixel {details['label']} bin time series\n"
        "Candlesticks show the pixel distribution within each bin at every repeat; lines connect medians",
        fontsize=16,
        y=0.99,
    )
    legend_handles = [
        Line2D([0], [0], color=colors[led], linewidth=2.0, label=f"LED {led}")
        for led in range(1, 7)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=6,
        frameon=False,
    )

    for index, bin_index in enumerate(bin_indices):
        row = index // bins_per_row
        column = index % bins_per_row
        raw_axis = axes[row, column]
        bin_data = bins[bin_index]
        raw_axis.set_title(
            f"{details['label']} {bin_data['start']}:{bin_data['stop']}",
            fontsize=8,
        )
        draw_candlesticks(
            raw_axis,
            bin_data,
            "raw",
            raw_limits,
            show_ylabel=column == 0,
            show_xlabel=row == rows - 1,
        )

    for index in range(len(bin_indices), bins_per_row * rows):
        row = index // bins_per_row
        column = index % bins_per_row
        axes[row, column].set_visible(False)

    fig.text(
        0.5,
        0.025,
        "Whisker=5th-95th percentile, body=25th-75th percentile, black tick=median. "
        "Each bin contains all DQ-clean pixels with a positive first-frame baseline.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(details["plot_path"], dpi=180)
    plt.close(fig)
    print(f"Wrote {details['plot_path']}")


for axis_name, details in AXES.items():
    plot_axis(axis_name, details)
