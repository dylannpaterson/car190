"""Measure and plot correlations between production rates and residuals."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


DETECTORS = ("wfi01", "wfi02")
MAX_LAG_REPEATS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def lagged_correlations(rate, residual):
    rate = np.asarray(rate, dtype=float)
    residual = np.asarray(residual, dtype=float)
    values = []
    for lag in range(-MAX_LAG_REPEATS, MAX_LAG_REPEATS + 1):
        if lag > 0:
            rate_slice = rate[:-lag]
            residual_slice = residual[lag:]
        elif lag < 0:
            rate_slice = rate[-lag:]
            residual_slice = residual[:lag]
        else:
            rate_slice = rate
            residual_slice = residual
        value = correlation(rate_slice, residual_slice)
        values.append((lag, value))
    return values


def load_tile(tile_dir):
    with np.load(tile_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        rate = np.asarray(samples["node_rates"], dtype=float).reshape(-1)
    rows = []
    with (tile_dir / "tile_group_predictions.csv").open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
    repeats = np.asarray(sorted({int(row["repeat"]) for row in rows}), dtype=int)
    days = np.asarray(
        [
            np.mean([float(row["days"]) for row in rows if int(row["repeat"]) == repeat])
            for repeat in repeats
        ]
    )
    common_residual = np.asarray(
        [
            np.mean(
                [
                    float(row["residual_log"])
                    for row in rows
                    if int(row["repeat"]) == repeat
                ]
            )
            for repeat in repeats
        ]
    )
    if rate.size != repeats.size:
        raise ValueError(
            f"{tile_dir}: {rate.size} rates for {repeats.size} repeat nodes"
        )
    # Repeat zero is the reference normalization and has exactly zero residual.
    valid = repeats != repeats.min()
    rate_valid = rate[valid]
    residual_valid = common_residual[valid]
    days_valid = days[valid]
    rate_detrended = signal.detrend(rate_valid)
    residual_detrended = signal.detrend(residual_valid)
    lagged = lagged_correlations(rate_detrended, residual_detrended)
    best_lag, best_corr = max(lagged, key=lambda item: abs(item[1]))
    return {
        "tile_id": int(tile_dir.name.split("_", 1)[1]),
        "days": days,
        "rate": rate,
        "common_residual": common_residual,
        "valid_days": days_valid,
        "valid_rate": rate_valid,
        "valid_residual": residual_valid,
        "rate_detrended": rate_detrended,
        "residual_detrended": residual_detrended,
        "lagged": lagged,
        "same_correlation": correlation(rate_detrended, residual_detrended),
        "difference_correlation": correlation(
            np.diff(rate_valid), np.diff(residual_valid)
        ),
        "best_lag": int(best_lag),
        "best_correlation": float(best_corr),
        "rate_std": float(np.std(rate_detrended)),
        "residual_std": float(np.std(residual_detrended)),
    }


def load_detector(production_dir, detector):
    fits_dir = production_dir / detector / "fits"
    tiles = []
    for tile_dir in sorted(fits_dir.glob("tile_*")):
        if not (tile_dir / "posterior_samples.npz").is_file():
            continue
        if not (tile_dir / "tile_group_predictions.csv").is_file():
            continue
        tiles.append(load_tile(tile_dir))
    if not tiles:
        raise FileNotFoundError(f"No production tile fits found under {fits_dir}")
    return tiles


def write_summary(rows, output_path):
    with output_path.open("w", newline="") as handle:
        fieldnames = (
            "detector",
            "tile_id",
            "same_correlation",
            "difference_correlation",
            "best_lag_repeats",
            "best_correlation",
            "rate_detrended_std_nm_day",
            "residual_detrended_std_log",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_representative(loaded, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for row_index, detector in enumerate(DETECTORS):
        tiles = loaded[detector]
        tile = next(tile for tile in tiles if tile["tile_id"] == 0)
        days = tile["valid_days"]
        rate_z = tile["rate_detrended"] / np.std(tile["rate_detrended"])
        residual_z = tile["residual_detrended"] / np.std(tile["residual_detrended"])
        axis = axes[row_index, 0]
        axis.plot(days, rate_z, color="#0072b2", label="rate residual")
        axis.plot(days, residual_z, color="#d55e00", label="common residual")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{detector.upper()} tile 000: aligned series")
        axis.set_ylabel("detrended z-score")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)

        axis = axes[row_index, 1]
        lag, values = zip(*tile["lagged"])
        axis.plot(lag, values, marker="o", color="#009e73")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(
            f"Cross-correlation; best lag={tile['best_lag']} repeats"
        )
        axis.set_xlabel("residual lag [repeat nodes]")
        axis.set_ylabel("corr(rate[t], residual[t + lag])")
        axis.grid(alpha=0.2)

        axis = axes[row_index, 2]
        axis.scatter(rate_z, residual_z, s=15, color="#7f3c8d", alpha=0.75)
        fit = np.polyfit(rate_z, residual_z, 1)
        grid = np.linspace(np.min(rate_z), np.max(rate_z), 100)
        axis.plot(grid, fit[0] * grid + fit[1], color="black", linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"Same-node r={tile['same_correlation']:.3f}")
        axis.set_xlabel("detrended rate z-score")
        axis.set_ylabel("detrended residual z-score")
        axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_tile_heatmaps(loaded, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for row_index, detector in enumerate(DETECTORS):
        tiles = sorted(loaded[detector], key=lambda tile: tile["tile_id"])
        rate_matrix = np.asarray(
            [tile["rate_detrended"] / np.std(tile["rate_detrended"]) for tile in tiles]
        )
        residual_matrix = np.asarray(
            [
                tile["residual_detrended"] / np.std(tile["residual_detrended"])
                for tile in tiles
            ]
        )
        for axis, matrix, title in (
            (
                axes[row_index, 0],
                rate_matrix,
                f"{detector.upper()} rate trajectories",
            ),
            (
                axes[row_index, 1],
                residual_matrix,
                f"{detector.upper()} common residuals",
            ),
        ):
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-2.5,
                vmax=2.5,
            )
            axis.set_title(title)
            axis.set_xlabel("repeat node")
            axis.set_ylabel("tile sorted by tile id")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_correlation_distributions(loaded, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for detector, color in (("wfi01", "#0072b2"), ("wfi02", "#d55e00")):
        tiles = loaded[detector]
        same = [tile["same_correlation"] for tile in tiles]
        best = [tile["best_correlation"] for tile in tiles]
        axes[0].hist(same, bins=14, alpha=0.5, color=color, label=detector.upper())
        axes[1].hist(best, bins=14, alpha=0.5, color=color, label=detector.upper())
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Same-node detrended correlation")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Best lagged detrended correlation")
    for axis in axes:
        axis.set_xlabel("correlation")
        axis.set_ylabel("tile count")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {detector: load_detector(args.production_dir, detector) for detector in DETECTORS}
    rows = []
    for detector, tiles in loaded.items():
        for tile in tiles:
            rows.append(
                {
                    "detector": detector,
                    "tile_id": tile["tile_id"],
                    "same_correlation": tile["same_correlation"],
                    "difference_correlation": tile["difference_correlation"],
                    "best_lag_repeats": tile["best_lag"],
                    "best_correlation": tile["best_correlation"],
                    "rate_detrended_std_nm_day": tile["rate_std"],
                    "residual_detrended_std_log": tile["residual_std"],
                }
            )
    write_summary(rows, args.output_dir / "rate_residual_correlations.csv")
    plot_representative(loaded, args.output_dir / "representative_rate_residuals.png")
    plot_tile_heatmaps(loaded, args.output_dir / "rate_residual_heatmaps.png")
    plot_correlation_distributions(
        loaded,
        args.output_dir / "rate_residual_correlation_distributions.png",
    )
    print(f"Wrote rate/residual diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
