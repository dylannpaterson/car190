"""Compare pre/post-reset residuals at matching fitted ice thickness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fit_wfi_tmm_production import load_tile_statistics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fit",
        action="append",
        nargs=3,
        metavar=("DATASET", "STATISTICS", "MAP"),
        required=True,
        help="Add one dataset label, statistics CSV, and MAP directory per detector.",
    )
    parser.add_argument("--grid-size", type=int, default=50)
    return parser.parse_args()


def unique_curve(thickness, residual):
    order = np.argsort(thickness)
    thickness = np.asarray(thickness, dtype=float)[order]
    residual = np.asarray(residual, dtype=float)[order]
    unique, inverse = np.unique(thickness, return_inverse=True)
    sums = np.bincount(inverse, weights=residual)
    counts = np.bincount(inverse)
    return unique, sums / counts


def compare_dataset(label, statistics_path, map_path, grid_size, output_dir):
    data = load_tile_statistics(statistics_path)
    with np.load(Path(map_path) / "posterior_samples.npz", allow_pickle=False) as samples:
        thickness = np.asarray(samples["ice_thickness"], dtype=float).squeeze()
        prediction = np.asarray(
            samples["_map_least_squares_prediction"], dtype=float
        ).squeeze()
        post_offset = np.asarray(samples["post_reset_tile_log_offset"], dtype=float).squeeze()
    observed = data["sums"][0] / data["counts"][0]
    post_mask = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    observed = observed + post_offset * post_mask
    residual = observed - prediction
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = str(data["sca_names"][0]).lower()
    curve_rows = []
    summary_rows = []
    for led in range(1, 7):
        curves = {}
        for side, side_mask in (("pre", ~post_mask), ("post", post_mask)):
            selected = (np.asarray(data["led"]) == led) & side_mask
            if np.count_nonzero(selected) < 3:
                continue
            curves[side] = unique_curve(thickness[selected], residual[selected])
        if set(curves) != {"pre", "post"}:
            continue
        pre_x, pre_y = curves["pre"]
        post_x, post_y = curves["post"]
        lower = max(pre_x[0], post_x[0])
        upper = min(pre_x[-1], post_x[-1])
        if upper <= lower:
            continue
        grid = np.linspace(lower, upper, grid_size)
        pre_values = np.interp(grid, pre_x, pre_y)
        post_values = np.interp(grid, post_x, post_y)
        difference = post_values - pre_values
        constant_translation = float(np.mean(difference))
        translated_difference = difference - constant_translation
        denominator = np.sqrt(
            np.sum((pre_values - np.mean(pre_values)) ** 2)
            * np.sum((post_values - np.mean(post_values)) ** 2)
        )
        correlation = (
            float(
                np.sum(
                    (pre_values - np.mean(pre_values))
                    * (post_values - np.mean(post_values))
                )
                / denominator
            )
            if denominator
            else 0.0
        )
        summary_rows.append(
            {
                "dataset": label,
                "detector": detector,
                "led": led,
                "common_points": grid_size,
                "thickness_min_nm": lower,
                "thickness_max_nm": upper,
                "pre_rms_log": float(np.sqrt(np.mean(pre_values**2))),
                "post_rms_log": float(np.sqrt(np.mean(post_values**2))),
                "pre_post_difference_rms_log": float(np.sqrt(np.mean(difference**2))),
                "best_constant_translation_log": constant_translation,
                "translated_difference_rms_log": float(
                    np.sqrt(np.mean(translated_difference**2))
                ),
                "translation_rms_reduction": float(
                    1.0
                    - np.sqrt(np.mean(translated_difference**2))
                    / max(np.sqrt(np.mean(difference**2)), np.finfo(float).eps)
                ),
                "pre_post_correlation": correlation,
            }
        )
        for grid_index, value in enumerate(grid):
            curve_rows.append(
                {
                    "dataset": label,
                    "detector": detector,
                    "led": led,
                    "grid_index": grid_index,
                    "thickness_nm": value,
                    "pre_residual_log": pre_values[grid_index],
                    "post_residual_log": post_values[grid_index],
                    "post_minus_pre_log": difference[grid_index],
                }
            )

    curve_path = output_dir / f"{label}_{detector}_matching_thickness.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)
    return summary_rows


def main():
    args = parse_args()
    all_rows = []
    for label, statistics_path, map_path in args.fit:
        all_rows.extend(
            compare_dataset(
                label,
                Path(statistics_path),
                Path(map_path),
                args.grid_size,
                args.output_dir,
            )
        )
    summary_path = args.output_dir / "matching_thickness_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    for row in all_rows:
        print(
            f"{row['dataset']} {row['detector']} LED{row['led']}: "
            f"corr={row['pre_post_correlation']:.6g}, "
            f"difference_rms={row['pre_post_difference_rms_log']:.6g}"
        )


if __name__ == "__main__":
    main()
