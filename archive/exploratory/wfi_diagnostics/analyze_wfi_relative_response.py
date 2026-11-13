"""Plot and benchmark portable representations of WFI response caches."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.interpolate import CubicSpline


REPRESENTATIVE_TILES = (0, 44, 123, 239, 255)
LINEAR_SPACINGS_NM = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CUBIC_SPACINGS_NM = LINEAR_SPACINGS_NM
CHEBYSHEV_DEGREES = (8, 12, 16, 24, 32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_caches(cache_root):
    paths = sorted(Path(cache_root).glob("tile_*/relative_response.npz"))
    if not paths:
        raise FileNotFoundError(f"No response caches found under {cache_root}")
    with np.load(paths[0], allow_pickle=False) as source:
        grid = np.asarray(source["grid_nm"], dtype=float)
        first = np.asarray(source["relative_log_response"], dtype=float)
    curves = np.empty((len(paths),) + first.shape, dtype=float)
    tile_ids = []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as source:
            current_grid = np.asarray(source["grid_nm"], dtype=float)
            current_curves = np.asarray(source["relative_log_response"], dtype=float)
        if not np.array_equal(grid, current_grid) or current_curves.shape != first.shape:
            raise ValueError(f"Cache grid or shape differs in {path}")
        curves[index] = current_curves
        tile_ids.append(int(path.parent.name.split("_")[-1]))
    return np.asarray(tile_ids, dtype=int), grid, curves


def knot_indices(grid, spacing_nm):
    stride = int(round(float(spacing_nm) / np.median(np.diff(grid))))
    if stride < 1:
        raise ValueError("Requested spacing is smaller than the native grid")
    indices = np.arange(0, grid.size, stride, dtype=int)
    if indices[-1] != grid.size - 1:
        indices = np.append(indices, grid.size - 1)
    return indices


def approximate_grid(curves, grid, spacing_nm, method):
    indices = knot_indices(grid, spacing_nm)
    approximation = np.empty_like(curves)
    for tile in range(curves.shape[0]):
        for led in range(curves.shape[1]):
            x = grid[indices]
            y = curves[tile, led, indices]
            if method == "linear":
                approximation[tile, led] = np.interp(grid, x, y)
            elif method == "cubic":
                approximation[tile, led] = CubicSpline(x, y)(grid)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
    return approximation, int(indices.size)


def approximate_chebyshev(curves, grid, degree):
    scaled = 2.0 * (grid - grid[0]) / (grid[-1] - grid[0]) - 1.0
    approximation = np.empty_like(curves)
    coefficients = np.empty(curves.shape[:2] + (degree + 1,), dtype=np.float32)
    for tile in range(curves.shape[0]):
        for led in range(curves.shape[1]):
            coefficients[tile, led] = chebfit(
                scaled, curves[tile, led], degree
            ).astype(np.float32)
            approximation[tile, led] = chebval(
                scaled, coefficients[tile, led].astype(float)
            )
    return approximation, coefficients


def error_summary(approximation, curves):
    error = np.abs(approximation - curves)
    return {
        "rms_log": float(np.sqrt(np.mean(error**2))),
        "p99_abs_log": float(np.quantile(error, 0.99)),
        "max_abs_log": float(np.max(error)),
    }


def compression_benchmarks(grid, curves):
    rows = []
    for method, spacings in (
        ("linear", LINEAR_SPACINGS_NM),
        ("cubic", CUBIC_SPACINGS_NM),
    ):
        for spacing in spacings:
            approximation, knots = approximate_grid(curves, grid, spacing, method)
            row = {
                "method": method,
                "spacing_nm": spacing,
                "stored_values_per_tile": int(curves.shape[1] * knots),
                "stored_float32_bytes_per_tile": int(curves.shape[1] * knots * 4),
                **error_summary(approximation, curves),
            }
            rows.append(row)
    for degree in CHEBYSHEV_DEGREES:
        approximation, coefficients = approximate_chebyshev(curves, grid, degree)
        rows.append(
            {
                "method": "chebyshev",
                "degree": degree,
                "stored_values_per_tile": int(coefficients.shape[1] * coefficients.shape[2]),
                "stored_float32_bytes_per_tile": int(coefficients.shape[1] * coefficients.shape[2] * 4),
                **error_summary(approximation, curves),
            }
        )
    return rows


def plot_examples(output, tile_ids, grid, curves):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for led, axis in enumerate(axes.flat):
        for tile_id in REPRESENTATIVE_TILES:
            index = int(np.flatnonzero(tile_ids == tile_id)[0])
            axis.plot(grid, curves[index, led], linewidth=1.0, label=f"tile {tile_id:03d}")
        axis.axhline(0.0, color="0.3", linewidth=0.6)
        axis.set_title(f"LED {led + 1}")
        axis.set_ylabel("log response change")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Ice thickness (nm)")
    axes[-1, 1].set_xlabel("Ice thickness (nm)")
    axes[-1, 2].set_xlabel("Ice thickness (nm)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Per-tile cached relative response functions")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_tile_bands(output, grid, curves):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for led, axis in enumerate(axes.flat):
        low, median, high = np.percentile(curves[:, led], [16, 50, 84], axis=0)
        minimum = np.min(curves[:, led], axis=0)
        maximum = np.max(curves[:, led], axis=0)
        axis.fill_between(grid, minimum, maximum, color="#d9d9d9", label="min-max")
        axis.fill_between(grid, low, high, color="#56B4E9", alpha=0.45, label="16-84%")
        axis.plot(grid, median, color="#0072B2", linewidth=1.8, label="median")
        axis.axhline(0.0, color="0.3", linewidth=0.6)
        axis.set_title(f"LED {led + 1}")
        axis.set_ylabel("log response change")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Ice thickness (nm)")
    axes[-1, 1].set_xlabel("Ice thickness (nm)")
    axes[-1, 2].set_xlabel("Ice thickness (nm)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Tile-to-tile spread of cached relative response functions")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_compression_examples(output, grid, curves):
    tile_index = 44
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, led in zip(axes, (0, 1, 4)):
        exact = curves[tile_index, led]
        linear, _ = approximate_grid(curves[tile_index : tile_index + 1, led : led + 1], grid, 2.0, "linear")
        cubic, _ = approximate_grid(curves[tile_index : tile_index + 1, led : led + 1], grid, 8.0, "cubic")
        chebyshev, _ = approximate_chebyshev(curves[tile_index : tile_index + 1, led : led + 1], grid, 12)
        axis.plot(grid, exact, color="black", linewidth=1.4, label="native 0.25 nm")
        axis.plot(grid, linear[0, 0], color="#0072B2", linestyle="--", label="linear 2 nm")
        axis.plot(grid, cubic[0, 0], color="#D55E00", linestyle=":", label="cubic 8 nm")
        axis.plot(grid, chebyshev[0, 0], color="#009E73", linestyle="-.", label="Chebyshev degree 12")
        axis.set_title(f"Tile 044, LED {led + 1}")
        axis.set_xlabel("Ice thickness (nm)")
        axis.set_ylabel("log response change")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Portable response approximations")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_compression_errors(output, rows):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    for metric, axis in zip(("rms_log", "max_abs_log"), axes):
        for method, marker in (("linear", "o"), ("cubic", "s"), ("chebyshev", "^") ):
            selected = [row for row in rows if row["method"] == method]
            selected.sort(key=lambda row: row["stored_values_per_tile"])
            axis.loglog(
                [row["stored_values_per_tile"] for row in selected],
                [row[metric] for row in selected],
                marker=marker,
                linewidth=1.2,
                label=method,
            )
        axis.axhline(0.005, color="0.3", linestyle="--", linewidth=0.9, label="noise sigma")
        axis.set_xlabel("Stored float values per tile")
        axis.set_ylabel(f"{metric} (log response)")
        axis.grid(alpha=0.2, which="both")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Response compression error across all 256 tiles and six LEDs")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    tile_ids, grid, curves = load_caches(args.cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = compression_benchmarks(grid, curves)
    summary = {
        "cache_root": str(Path(args.cache_root).resolve()),
        "tile_count": int(curves.shape[0]),
        "led_count": int(curves.shape[1]),
        "grid_count": int(grid.size),
        "grid_min_nm": float(grid[0]),
        "grid_max_nm": float(grid[-1]),
        "grid_step_nm": float(np.median(np.diff(grid))),
        "native_response_min": float(np.min(curves)),
        "native_response_max": float(np.max(curves)),
        "compression_benchmarks": rows,
    }
    (args.output_dir / "relative_response_compression_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot_examples(args.output_dir / "relative_response_examples.png", tile_ids, grid, curves)
    plot_tile_bands(args.output_dir / "relative_response_tile_bands.png", grid, curves)
    plot_compression_examples(
        args.output_dir / "relative_response_compression_examples.png", grid, curves
    )
    plot_compression_errors(
        args.output_dir / "relative_response_compression_error.png", rows
    )
    print(f"Analyzed {curves.shape[0]} tile caches")
    print(f"Wrote diagnostics under {args.output_dir}")


if __name__ == "__main__":
    main()
