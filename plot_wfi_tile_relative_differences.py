"""Plot per-tile relative changes from the first exposure for two reductions."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def read_tile_series(path):
    series = defaultdict(list)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            count = int(row["n_pair_valid"])
            if count <= 0:
                continue
            key = (int(row["led"]), int(row["tile_id"]))
            mean_log_change = float(row["sum_log_change"]) / count
            series[key].append(
                (
                    float(row["days"]),
                    int(row["repeat"]),
                    int(row["group"]),
                    mean_log_change,
                )
            )

    normalized = {}
    for key, points in series.items():
        points.sort(key=lambda point: (point[0], point[1], point[2]))
        baseline = points[0][3]
        normalized[key] = (
            np.asarray([point[0] for point in points]),
            np.asarray([point[3] - baseline for point in points]),
        )
    return normalized


def plot_reductions(existing_path, rrn_path, output_path):
    reductions = [
        ("Existing", read_tile_series(existing_path)),
        ("RRN", read_tile_series(rrn_path)),
    ]
    all_values = np.concatenate(
        [values for _, reduction in reductions for _, (_, values) in reduction.items()]
    )
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    margin = max((y_max - y_min) * 0.05, 0.001)

    tile_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 64))
    figure, axes = plt.subplots(
        6,
        2,
        figsize=(15, 20),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    figure.suptitle(
        "WFI01 tile relative differences from the first exposure\n"
        "Per-tile mean log change, existing versus RRN reduction",
        fontsize=16,
    )

    for column, (label, reduction) in enumerate(reductions):
        for led in range(1, 7):
            axis = axes[led - 1, column]
            by_time = defaultdict(list)
            for tile_id in range(64):
                points = reduction.get((led, tile_id))
                if points is None:
                    continue
                days, values = points
                axis.plot(
                    days,
                    values,
                    color=tile_colors[tile_id],
                    linewidth=0.55,
                    alpha=0.38,
                )
                for day, value in zip(days, values):
                    by_time[day].append(value)

            if by_time:
                days = np.asarray(sorted(by_time))
                quantiles = np.asarray(
                    [np.percentile(by_time[day], [16, 50, 84]) for day in days]
                )
                axis.fill_between(
                    days,
                    quantiles[:, 0],
                    quantiles[:, 2],
                    color="black",
                    alpha=0.08,
                    linewidth=0,
                )
                axis.plot(
                    days,
                    quantiles[:, 1],
                    color="black",
                    linewidth=1.4,
                    label="tile median",
                )

            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
            axis.set_ylim(y_min - margin, y_max + margin)
            axis.set_title(f"{label} | LED {led}")
            axis.grid(axis="y", alpha=0.18, linewidth=0.5)
            if column == 0:
                axis.set_ylabel("mean log change")
            if led == 6:
                axis.set_xlabel("days since first exposure")

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 63)),
        ax=axes,
        pad=0.01,
        fraction=0.02,
    )
    colorbar.set_label("tile ID")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--rrn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_reductions(args.existing, args.rrn, args.output)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
