"""Plot old versus refreshed WFI tile sufficient statistics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def row_mean(row):
    count = float(row["n_pair_valid"])
    return np.nan if count <= 0.0 else float(row["sum_log_change"]) / count


def grouped_values(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["led"]), row["file_set_name"])].append(row)
    values = {}
    for key, group in grouped.items():
        means = np.asarray([row_mean(row) for row in group], dtype=float)
        values[key] = {
            "days": float(group[0]["days"]),
            "repeat": int(group[0]["repeat"]),
            "median": float(np.nanmedian(means)),
            "p16": float(np.nanpercentile(means, 16)),
            "p84": float(np.nanpercentile(means, 84)),
        }
    return values


def comparison_rows(old_rows, new_rows):
    old_by_key = {
        (row["file_set_name"], row["tile_id"]): row for row in old_rows
    }
    new_by_key = {
        (row["file_set_name"], row["tile_id"]): row for row in new_rows
    }
    common = sorted(set(old_by_key).intersection(new_by_key))
    return [
        (old_by_key[key], new_by_key[key])
        for key in common
    ]


def plot_time_series(output, old_rows, new_rows):
    old = grouped_values(old_rows)
    new = grouped_values(new_rows)
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        old_values = sorted(
            (value for (row_led, _), value in old.items() if row_led == led),
            key=lambda value: (value["days"], value["repeat"]),
        )
        new_values = sorted(
            (value for (row_led, _), value in new.items() if row_led == led),
            key=lambda value: (value["days"], value["repeat"]),
        )
        if old_values:
            x = np.asarray([value["days"] for value in old_values])
            axis.fill_between(
                x,
                [value["p16"] for value in old_values],
                [value["p84"] for value in old_values],
                color="0.65",
                alpha=0.25,
            )
            axis.plot(
                x,
                [value["median"] for value in old_values],
                color="0.35",
                linewidth=1.5,
                label="old median",
            )
        if new_values:
            x = np.asarray([value["days"] for value in new_values])
            axis.fill_between(
                x,
                [value["p16"] for value in new_values],
                [value["p84"] for value in new_values],
                color="tab:blue",
                alpha=0.18,
            )
            axis.plot(
                x,
                [value["median"] for value in new_values],
                color="tab:blue",
                linewidth=1.6,
                label="refreshed median",
            )
        axis.axhline(0.0, color="0.2", linewidth=0.7, alpha=0.5)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("Tile mean log change")
    figure.suptitle("WFI01 old versus refreshed tile statistics")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_deltas(output, old_rows, new_rows):
    grouped = defaultdict(list)
    for old_row, new_row in comparison_rows(old_rows, new_rows):
        grouped[(int(new_row["led"]), new_row["file_set_name"])].append(
            row_mean(new_row) - row_mean(old_row)
        )
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        points = []
        for (row_led, file_set_name), deltas in grouped.items():
            if row_led != led:
                continue
            points.append(
                (
                    file_set_name,
                    float(next(
                        row["days"]
                        for row in new_rows
                        if row["led"] == str(led)
                        and row["file_set_name"] == file_set_name
                    )),
                    np.asarray(deltas, dtype=float),
                )
            )
        points.sort(key=lambda item: item[1])
        if points:
            x = np.asarray([item[1] for item in points])
            median = np.asarray([np.median(item[2]) for item in points])
            p16 = np.asarray([np.percentile(item[2], 16) for item in points])
            p84 = np.asarray([np.percentile(item[2], 84) for item in points])
            axis.fill_between(x, p16, p84, color="tab:orange", alpha=0.25)
            axis.plot(x, median, color="tab:orange", linewidth=1.5)
        axis.axhline(0.0, color="0.2", linewidth=0.8)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Days since first exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("Refreshed - old mean log change")
    figure.suptitle("WFI01 refreshed minus old tile-statistics difference")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_scatter(output, old_rows, new_rows):
    pairs = comparison_rows(old_rows, new_rows)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for led in range(1, 7):
        selected = [
            (old_row, new_row)
            for old_row, new_row in pairs
            if int(new_row["led"]) == led
        ]
        if not selected:
            continue
        x = np.asarray([row_mean(old_row) for old_row, _ in selected])
        y = np.asarray([row_mean(new_row) for _, new_row in selected])
        axis.scatter(x, y, s=5, alpha=0.25, color=colors(led - 1), label=f"LED {led}")
    limits = axis.get_xlim() + axis.get_ylim()
    lower = min(limits)
    upper = max(limits)
    axis.plot([lower, upper], [lower, upper], color="0.2", linewidth=1.0)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Old tile mean log change")
    axis.set_ylabel("Refreshed tile mean log change")
    axis.set_title("WFI01 common tile rows: old versus refreshed")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_rows = read_rows(args.old)
    new_rows = read_rows(args.new)
    plot_time_series(
        args.output_dir / "wfi01_old_vs_refreshed_tile_means.png",
        old_rows,
        new_rows,
    )
    plot_deltas(
        args.output_dir / "wfi01_refreshed_minus_old_tile_means.png",
        old_rows,
        new_rows,
    )
    plot_scatter(
        args.output_dir / "wfi01_old_vs_refreshed_tile_scatter.png",
        old_rows,
        new_rows,
    )
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
