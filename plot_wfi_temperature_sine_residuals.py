"""Compare pre-1063 sinusoid residuals with late Program 1063 extrapolation."""

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


def tile_residuals(train_rows, test_rows, tile, led, reset_day):
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
    test_values = np.asarray([row["value"] for row in test], dtype=float)
    fit = fit_sinusoid(train_days - reset_day, train_values)
    train_residual = 1000.0 * (
        train_values - evaluate_model(train_days, fit, reset_day)
    )
    test_residual = 1000.0 * (
        test_values - evaluate_model(test_days, fit, reset_day)
    )
    late_mask, late_start_day = latest_block_mask(test_days)
    return {
        "fit": fit,
        "train": train,
        "train_residual": train_residual,
        "late_days": test_days[late_mask],
        "late_residual": test_residual[late_mask],
        "late_repeats": np.asarray(
            [row["repeat"] for row, is_late in zip(test, late_mask) if is_late],
            dtype=int,
        ),
        "late_start_day": late_start_day,
    }


def plot_residuals(
    output_path,
    train_rows,
    test_rows,
    led,
    reset_day,
):
    tiles = sorted({row["tile"] for row in train_rows})
    results = {
        tile: tile_residuals(train_rows, test_rows, tile, led, reset_day)
        for tile in tiles
    }
    train_end_day = max(
        float(row["days"])
        for result in results.values()
        for row in result["train"]
    )
    late_start_day = min(result["late_start_day"] for result in results.values())
    late_start_repeat = min(
        int(repeats[0]) for repeats in (result["late_repeats"] for result in results.values())
    )

    figure, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    tile_norm = Normalize(vmin=min(tiles), vmax=max(tiles))
    tile_cmap = plt.get_cmap("viridis")
    for tile in tiles:
        result = results[tile]
        color = tile_cmap(tile_norm(tile))
        train_days = [row["days"] for row in result["train"]]
        axis.scatter(
            train_days,
            result["train_residual"],
            color=color,
            s=8,
            alpha=0.45,
            label="all pre-1063 fit points" if tile == tiles[0] else None,
        )
        axis.scatter(
            result["late_days"],
            result["late_residual"],
            color=color,
            s=22,
            alpha=0.95,
            label="late 1063 extrapolation" if tile == tiles[0] else None,
        )

    axis.axvspan(
        train_end_day,
        late_start_day,
        color="tab:orange",
        alpha=0.12,
        label="extrapolation interval",
    )
    axis.axvline(
        train_end_day,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="last pre-1063 fit point",
    )
    axis.axvline(
        late_start_day,
        color="tab:red",
        linestyle="-.",
        linewidth=1.3,
        label=f"late 1063 start: repeat {late_start_repeat}, day {late_start_day:.3f}",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Sinusoid residual (millilog)")
    axis.set_title(
        f"WFI01 LED {led}: all pre-1063 fit residuals and late-1063 extrapolation\n"
        "Fit uses all non-1063 data before the late 1063 block"
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
    return results, late_start_day, late_start_repeat


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
    results, late_start_day, late_start_repeat = plot_residuals(
        args.output,
        train_rows,
        test_rows,
        args.led,
        args.reset_day,
    )
    train_rms = np.asarray(
        [
            np.sqrt(np.mean(result["train_residual"] ** 2))
            for result in results.values()
        ]
    )
    late_rms = np.asarray(
        [
            np.sqrt(np.mean(result["late_residual"] ** 2))
            for result in results.values()
        ]
    )
    train_values = np.concatenate(
        [result["train_residual"] for result in results.values()]
    )
    late_values = np.concatenate(
        [result["late_residual"] for result in results.values()]
    )
    print(f"Loaded {len(all_rows)} valid rows for {args.detector}")
    print(f"Fit rows: {len(train_rows)}; held-out 1063 rows: {len(test_rows)}")
    print(f"Plotted {len(results)} tiles, LED {args.led}")
    print(
        f"Late 1063 starts at repeat {late_start_repeat}, "
        f"day {late_start_day:.9f}"
    )
    print(
        "Pre-1063 fit residual RMS millilog: "
        f"median_per_tile={np.median(train_rms):.6g}, "
        f"pooled={np.sqrt(np.mean(train_values**2)):.6g}, "
        f"max_abs={np.max(np.abs(train_values)):.6g}"
    )
    print(
        "Late 1063 extrapolation residual RMS millilog: "
        f"median_per_tile={np.median(late_rms):.6g}, "
        f"pooled={np.sqrt(np.mean(late_values**2)):.6g}, "
        f"max_abs={np.max(np.abs(late_values)):.6g}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
