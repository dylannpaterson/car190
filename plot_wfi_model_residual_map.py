"""Plot per-tile production model residual maps for one WFI detector."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import run_wfi_production as production_runner
from fit_wfi_tmm_production import load_tile_statistics
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    partial_tile_grid_size,
    positive_integer,
    tile_statistics_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    return parser.parse_args()


def residual_summary(rows, reset_day):
    residual = np.asarray([float(row["residual_log"]) for row in rows], dtype=float)
    days = np.asarray([float(row["days"]) for row in rows], dtype=float)
    repeats = np.asarray([int(row["repeat"]) for row in rows], dtype=int)
    valid = (repeats > np.min(repeats)) & np.isfinite(residual)
    if not np.any(valid):
        raise ValueError("No non-reference residual groups found")

    def metrics(mask):
        values = residual[valid & mask]
        if not values.size:
            return np.nan, np.nan
        return float(np.mean(values)), float(np.sqrt(np.mean(values**2)))

    mean, rms = metrics(np.ones(residual.shape, dtype=bool))
    pre_mean, pre_rms = metrics(days < reset_day)
    post_mean, post_rms = metrics(days >= reset_day)
    return {
        "n_groups": int(np.count_nonzero(valid)),
        "mean_residual_log": mean,
        "rms_residual_log": rms,
        "pre_mean_residual_log": pre_mean,
        "pre_rms_residual_log": pre_rms,
        "post_mean_residual_log": post_mean,
        "post_rms_residual_log": post_rms,
    }


def load_residual_rows(detector_root, detector, tile_size=DEFAULT_TILE_SIZE):
    detector_prefix = detector.lower()
    rows = []
    for fit_dir in sorted((detector_root / "fits").glob("tile_*")):
        tile_id = int(fit_dir.name.split("_")[1])
        statistics_path = tile_statistics_path(
            detector_root / "tile_statistics", detector, tile_size, tile_id
        )
        if not production_runner.archive_matches_statistics(fit_dir, statistics_path):
            continue
        data = load_tile_statistics(statistics_path)
        with (fit_dir / "tile_group_predictions.csv").open(newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        summary = residual_summary(prediction_rows, float(data["reset_day"]))
        metadata = data["metadata"][int(tile_id)]
        rows.append(
            {
                "detector": detector,
                "tile_id": int(tile_id),
                "tile_row": int(metadata["tile_row"]),
                "tile_col": int(metadata["tile_col"]),
                **summary,
            }
        )
    if not rows:
        raise ValueError(f"No matching production residual files found under {detector_root}")
    return sorted(rows, key=lambda row: row["tile_id"])


def map_values(rows, field, grid_size):
    values = np.full((grid_size, grid_size), np.nan)
    for row in rows:
        values[row["tile_row"], row["tile_col"]] = row[field]
    return values


def draw_map(axis, values, title, signed):
    finite = values[np.isfinite(values)] * 1000.0
    if not finite.size:
        raise ValueError(f"No finite values available for {title}")
    if signed:
        limit = max(float(np.max(np.abs(finite))), 1.0e-12)
        image = axis.imshow(
            values * 1000.0,
            origin="lower",
            interpolation="none",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
    else:
        image = axis.imshow(
            values * 1000.0,
            origin="lower",
            interpolation="none",
            cmap="viridis",
            vmin=0.0,
            vmax=max(float(np.max(finite)), 1.0e-12),
        )
    axis.set_title(title)
    axis.set_xlabel("Tile column")
    axis.set_ylabel("Tile row")
    axis.set_xticks(range(values.shape[1]))
    axis.set_yticks(range(values.shape[0]))
    axis.grid(color="white", alpha=0.35, linewidth=0.5)
    return image


def write_outputs(output_dir, detector, rows, grid_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / f"{detector.lower()}_model_residual_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    maps = (
        ("mean_residual_log", "Mean observed - model residual", True),
        ("rms_residual_log", "RMS observed - model residual", False),
        ("pre_mean_residual_log", "Pre-Decon2 mean observed - model residual", True),
        ("post_mean_residual_log", "Post-Decon2 mean observed - model residual", True),
    )
    for axis, (field, title, signed) in zip(axes.flat, maps):
        image = draw_map(axis, map_values(rows, field, grid_size), title, signed)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="10^-3 log")
    figure.suptitle(
        f"{detector} production model residuals; reference groups excluded\n"
        f"{len(rows)} of {grid_size * grid_size} tiles"
    )
    figure.savefig(output_dir / f"{detector.lower()}_model_residual_maps.png", dpi=180)
    plt.close(figure)

    signed_fields = (
        "mean_residual_log",
        "pre_mean_residual_log",
        "post_mean_residual_log",
    )
    lines = [
        f"# {detector} model residual maps",
        "",
        f"- Matching tiles: `{len(rows)}`.",
        "- Residual convention: `observed - model`, in log response units.",
        "- The first reference group for each LED is excluded because its residual is zero by construction.",
        "- Map color units are `10^-3 log` for readability.",
        "",
    ]
    for field in signed_fields + ("rms_residual_log",):
        values = np.asarray([row[field] for row in rows], dtype=float)
        lines.append(
            f"- `{field}`: min `{np.min(values):.6g}`, median `{np.median(values):.6g}`, "
            f"max `{np.max(values):.6g}`, RMS across tiles `{np.sqrt(np.mean(values**2)):.6g}`."
        )
    (output_dir / f"{detector.lower()}_model_residual_summary.md").write_text(
        "\n".join(lines) + "\n"
    )


def main():
    args = parse_args()
    rows = load_residual_rows(args.detector_root, args.detector, args.tile_size)
    grid_size = partial_tile_grid_size(
        [row["tile_id"] for row in rows]
    )
    write_outputs(args.output_dir, args.detector, rows, grid_size)
    print(f"Matching tiles: {len(rows)}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
