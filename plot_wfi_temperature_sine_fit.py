"""Plot the existing pre-1063 sinusoid fits and late 1063 holdout points."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from analyze_wfi_post_reset_led_sinusoid import (
    DEFAULT_RESET_DAY,
    evaluate_model,
    fit_sinusoid,
    latest_block_mask,
    read_rows,
    select_rows,
)


DEFAULT_HOLDOUT_PROGRAM = 1063
DEFAULT_LED = 3


def fit_tile(train_rows, test_rows, tile, led, reset_day):
    train = sorted(
        [row for row in train_rows if row["tile"] == tile and row["led"] == led],
        key=lambda row: row["days"],
    )
    test = sorted(
        [row for row in test_rows if row["tile"] == tile and row["led"] == led],
        key=lambda row: row["days"],
    )
    if len(train) < 6 or len(test) < 3:
        raise ValueError(f"Insufficient samples for tile {tile}, LED {led}")

    train_days = np.asarray([row["days"] for row in train], dtype=float)
    train_values = np.asarray([row["value"] for row in train], dtype=float)
    test_days = np.asarray([row["days"] for row in test], dtype=float)
    fit = fit_sinusoid(train_days - reset_day, train_values)
    late_mask, late_start_day = latest_block_mask(test_days)
    return {
        "train": train,
        "fit": fit,
        "late_days": test_days[late_mask],
        "late_values": np.asarray(
            [row["value"] for row, is_late in zip(test, late_mask) if is_late],
            dtype=float,
        ),
        "late_repeats": np.asarray(
            [row["repeat"] for row, is_late in zip(test, late_mask) if is_late],
            dtype=int,
        ),
        "late_start_day": late_start_day,
    }


def plot_fits(output_path, train_rows, test_rows, led, reset_day):
    tiles = sorted({row["tile"] for row in train_rows})
    results = {
        tile: fit_tile(train_rows, test_rows, tile, led, reset_day)
        for tile in tiles
    }
    fit_start_day = min(
        float(row["days"]) for result in results.values() for row in result["train"]
    )
    fit_end_day = max(
        float(row["days"]) for result in results.values() for row in result["train"]
    )
    late_start_day = min(result["late_start_day"] for result in results.values())
    late_start_repeat = min(
        int(result["late_repeats"][0]) for result in results.values()
    )

    figure, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    tile_norm = Normalize(vmin=min(tiles), vmax=max(tiles))
    tile_cmap = plt.get_cmap("viridis")
    curve_days = np.linspace(fit_start_day, late_start_day, 800)
    for tile in tiles:
        result = results[tile]
        color = tile_cmap(tile_norm(tile))
        train_days = np.asarray(
            [row["days"] for row in result["train"]], dtype=float
        )
        train_values = np.asarray(
            [row["value"] for row in result["train"]], dtype=float
        )
        axis.scatter(
            train_days,
            1000.0 * train_values,
            color=color,
            s=8,
            alpha=0.35,
            label="non-1063 fit data" if tile == tiles[0] else None,
        )
        axis.plot(
            curve_days,
            1000.0 * evaluate_model(curve_days, result["fit"], reset_day),
            color=color,
            linewidth=0.9,
            alpha=0.65,
            label="fitted sine" if tile == tiles[0] else None,
        )
        axis.scatter(
            result["late_days"],
            1000.0 * result["late_values"],
            color=color,
            edgecolor="black",
            linewidth=0.25,
            s=22,
            alpha=0.95,
            label="late 1063 only" if tile == tiles[0] else None,
        )

    axis.axvspan(
        fit_end_day,
        late_start_day,
        color="tab:orange",
        alpha=0.12,
        label="sine extrapolation interval",
    )
    axis.axvline(
        fit_end_day,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="last non-1063 fit point",
    )
    axis.axvline(
        late_start_day,
        color="tab:red",
        linestyle="-.",
        linewidth=1.3,
        label=f"late 1063 start: repeat {late_start_repeat}, day {late_start_day:.3f}",
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Mean log change (millilog)")
    axis.set_title(
        f"WFI01 LED {led}: pre-1063 sine fits and late 1063 data\n"
        "Early 1063 data are excluded from both the fit and this plot"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="upper left")
    colorbar = figure.colorbar(
        ScalarMappable(norm=tile_norm, cmap=tile_cmap),
        ax=axis,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("tile ID")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return results, fit_start_day, fit_end_day, late_start_day, late_start_repeat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--led", type=int, default=DEFAULT_LED)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument(
        "--holdout-program", type=int, default=DEFAULT_HOLDOUT_PROGRAM
    )
    return parser.parse_args()


def main():
    args = parse_args()
    statistics_path = args.statistics_dir / (
        f"{args.detector.lower()}_tile_log_statistics_512_count_corrected.csv"
    )
    all_rows = read_rows(statistics_path)
    train_rows = select_rows(
        all_rows,
        args.reset_day,
        program=args.holdout_program,
        include_program=False,
    )
    test_rows = select_rows(
        all_rows,
        args.reset_day,
        program=args.holdout_program,
        include_program=True,
    )
    results, fit_start, fit_end, late_start, late_repeat = plot_fits(
        args.output,
        train_rows,
        test_rows,
        args.led,
        args.reset_day,
    )
    print(f"Loaded {len(all_rows)} valid rows for {args.detector}")
    print(f"Non-1063 fit rows: {len(train_rows)}")
    print(f"Program {args.holdout_program} rows held out: {len(test_rows)}")
    print(f"Plotted {len(results)} tiles, LED {args.led}")
    print(f"Fit span: day {fit_start:.9f} to {fit_end:.9f}")
    print(f"Late 1063 starts at repeat {late_repeat}, day {late_start:.9f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
