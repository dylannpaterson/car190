"""Plot corrected tile statistics for one WFI tile and its neighbors."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from wfi_production_layout import DEFAULT_TILE_SIZE, positive_integer
from wfi_production_plot_annotations import (
    AOM_TEMPERATURE_BUMP_LABEL,
    latest_observation_label,
    load_time_annotations,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = "WFI06"
DEFAULT_TILE_ID = 98


def tile_statistics_path(root, detector, tile_size, tile_id):
    return (
        Path(root)
        / "production"
        / detector.lower()
        / "relative_response"
        / "tile_statistics"
        / f"{detector.lower()}_tile_log_statistics_{tile_size}_tile_{tile_id:03d}.csv"
    )


def load_corrected_tile_series(path):
    by_led = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            valid = float(row["n_pair_valid"])
            if valid <= 0.0:
                continue
            led = int(row["led"])
            days, values = by_led.setdefault(led, ([], []))
            days.append(float(row["days"]))
            values.append(1000.0 * float(row["sum_log_change"]) / valid)
    series = {}
    for led, (days, values) in by_led.items():
        days = np.asarray(days, dtype=float)
        values = np.asarray(values, dtype=float)
        order = np.argsort(days)
        series[led] = (days[order], values[order])
    return series


def neighborhood_tile_ids(tile_id, tile_size):
    grid_size = (4096 + tile_size - 1) // tile_size
    row, column = divmod(int(tile_id), grid_size)
    if not (0 < row < grid_size - 1 and 0 < column < grid_size - 1):
        raise ValueError("The center tile must have a complete 3x3 neighborhood")
    return tuple(
        (neighbor_row * grid_size + neighbor_column)
        for neighbor_row in range(row - 1, row + 2)
        for neighbor_column in range(column - 1, column + 2)
    )


def plot_neighborhood(root, detector, tile_id, tile_size, output):
    center_path = tile_statistics_path(root, detector, tile_size, tile_id)
    if not center_path.is_file():
        raise FileNotFoundError(center_path)
    annotations = load_time_annotations(center_path)
    tile_ids = neighborhood_tile_ids(tile_id, tile_size)
    aom_day = next(
        event.day
        for event in annotations.events
        if event.label == AOM_TEMPERATURE_BUMP_LABEL
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(tile_ids)))
    tile_series = {}
    for color, current_tile_id in zip(colors, tile_ids):
        path = tile_statistics_path(root, detector, tile_size, current_tile_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        tile_series[current_tile_id] = load_corrected_tile_series(path)

    handles = []
    for color, current_tile_id in zip(colors, tile_ids):
        is_center = current_tile_id == tile_id
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markersize=5 if is_center else 4,
                color="#d62728" if is_center else color,
                label=(
                    f"tile {current_tile_id} (center)"
                    if is_center
                    else f"tile {current_tile_id}"
                ),
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color="#009E73",
            linestyle="-.",
            linewidth=1.0,
            label=AOM_TEMPERATURE_BUMP_LABEL,
        )
    )
    for led, axis in zip(range(1, 7), axes):
        for color, current_tile_id in zip(colors, tile_ids):
            days, values = tile_series[current_tile_id].get(
                led, (np.asarray([]), np.asarray([]))
            )
            if days.size == 0:
                continue
            is_center = current_tile_id == tile_id
            axis.scatter(
                days,
                values,
                color="#d62728" if is_center else color,
                s=16 if is_center else 9,
                alpha=0.85 if is_center else 0.65,
                edgecolors="none",
                zorder=3 if is_center else 2,
            )
        axis.axvline(
            aom_day,
            color="#009E73",
            linestyle="-.",
            linewidth=0.9,
            alpha=0.9,
        )
        axis.set_title(f"LED {led}", fontsize=9)
        axis.grid(alpha=0.2)
        if led > 3:
            axis.set_xlabel("Days since first exposure")
        if led in (1, 4):
            axis.set_ylabel("Corrected mean log change (millilog)")

    figure.suptitle(
        f"{detector} corrected tile data: tile {tile_id} and 3x3 neighborhood\n"
        f"{latest_observation_label(center_path, tile_size)}",
        fontsize=11,
    )
    figure.legend(
        handles=handles,
        ncol=5,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--tile-id", type=int, default=DEFAULT_TILE_ID)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = str(args.detector).upper()
    output = args.output
    if output is None:
        output = (
            args.root
            / "production"
            / "relative_response_plots"
            / f"{detector.lower()}_tile{args.tile_id:03d}_neighborhood_corrected_data.png"
        )
    plot_neighborhood(
        args.root,
        detector,
        args.tile_id,
        args.tile_size,
        output,
    )


if __name__ == "__main__":
    main()
