"""Plot reset-aware accumulation acceleration in the Roman WFI mosaic."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from plot_wfi_full_focal_plane_rates import (
    SCA_SIZE,
    SCAS,
    sca_position,
)


TILE_GRID_SIZE = 8
METRICS = (
    (
        "Post-reset strong log-rate spline",
        "post_natural_strong_median_acceleration_nm_day2",
    ),
    (
        "Post-reset native piecewise log-rate",
        "post_native_median_acceleration_nm_day2",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceleration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary_path(root, detector):
    detector_lower = detector.lower()
    return root / f"acceleration_{detector_lower}" / (
        f"{detector_lower}_tile_acceleration_summary.csv"
    )


def load_acceleration_map(root, detector, field):
    path = summary_path(root, detector)
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row.get("detector") for row in rows} != {detector}:
        raise ValueError(f"Unexpected detector rows in {path}")

    values = np.full((TILE_GRID_SIZE, TILE_GRID_SIZE), np.nan)
    for row in rows:
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        if not 0 <= tile_row < TILE_GRID_SIZE or not 0 <= tile_col < TILE_GRID_SIZE:
            raise ValueError(f"Tile coordinates out of range in {path}: {row}")
        if np.isfinite(values[tile_row, tile_col]):
            raise ValueError(f"Duplicate tile coordinate in {path}: {row}")
        values[tile_row, tile_col] = float(row[field])
    return values


def load_maps(root, field):
    return {
        detector: load_acceleration_map(root, detector, field)
        for detector in SCAS
    }


def color_limit(maps):
    finite = np.concatenate(
        [values[np.isfinite(values)] for values in maps.values() if values is not None]
    )
    if finite.size == 0:
        raise ValueError("No acceleration summaries were found")
    return max(float(np.max(np.abs(finite))), 1.0e-12)


def draw_panel(axis, maps, title, norm):
    image = None
    label_effect = [
        patheffects.withStroke(linewidth=2.0, foreground="black", alpha=0.7)
    ]
    for detector in SCAS:
        x, y = sca_position(detector)
        values = maps[detector]
        if values is None or not np.isfinite(values).any():
            axis.add_patch(
                Rectangle(
                    (x, y),
                    SCA_SIZE,
                    SCA_SIZE,
                    facecolor="#e6e6e6",
                    edgecolor="#333333",
                    hatch="///",
                    linewidth=1.0,
                )
            )
            axis.text(
                x + SCA_SIZE / 2.0,
                y + SCA_SIZE / 2.0,
                f"{detector}\npending",
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
            )
            continue

        image = axis.imshow(
            values,
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
            norm=norm,
            extent=(x, x + SCA_SIZE, y, y + SCA_SIZE),
        )
        axis.add_patch(
            Rectangle(
                (x, y),
                SCA_SIZE,
                SCA_SIZE,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )
        median = float(np.nanmedian(values))
        axis.text(
            x + SCA_SIZE / 2.0,
            y + SCA_SIZE - 0.55,
            detector,
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            path_effects=label_effect,
        )
        axis.text(
            x + SCA_SIZE / 2.0,
            y + 0.4,
            f"{median:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="white",
            path_effects=label_effect,
        )
        for tile in range(1, TILE_GRID_SIZE):
            axis.plot(
                (x + tile, x + tile),
                (y, y + SCA_SIZE),
                color="white",
                alpha=0.35,
                linewidth=0.35,
            )
            axis.plot(
                (x, x + SCA_SIZE),
                (y + tile, y + tile),
                color="white",
                alpha=0.35,
                linewidth=0.35,
            )

    x_positions = [sca_position(detector)[0] for detector in SCAS]
    y_positions = [sca_position(detector)[1] for detector in SCAS]
    axis.set_xlim(min(x_positions) - 1.0, max(x_positions) + SCA_SIZE + 1.0)
    axis.set_ylim(min(y_positions) - 1.0, max(y_positions) + SCA_SIZE + 1.0)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("Focal-plane X detector position")
    axis.set_ylabel("Focal-plane Y detector position")
    axis.grid(alpha=0.15)
    return image


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metric_maps = {
        title: load_maps(args.acceleration_root, field)
        for title, field in METRICS
    }
    available = {
        detector
        for maps in metric_maps.values()
        for detector, values in maps.items()
        if values is not None and np.isfinite(values).any()
    }
    if not available:
        raise ValueError("No detector acceleration summaries were found")

    all_values = [
        values
        for maps in metric_maps.values()
        for values in maps.values()
        if values is not None
    ]
    limit = color_limit({str(index): values for index, values in enumerate(all_values)})
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    figure, axes = plt.subplots(1, 2, figsize=(19, 9), constrained_layout=True)
    image = None
    for axis, (title, _) in zip(axes, metric_maps.items()):
        image = draw_panel(axis, metric_maps[title], title, norm)
    if image is None:
        raise ValueError("No finite acceleration map values were found")
    figure.colorbar(
        image,
        ax=axes,
        shrink=0.8,
        pad=0.02,
        label="Acceleration (nm/day$^2$)",
    )
    figure.suptitle(
        "Roman WFI post-reset accumulation acceleration\n"
        f"{len(available)} of {len(SCAS)} detectors available; pending detectors are hatched"
    )
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Available detectors: {', '.join(sorted(available))}")
    print(f"Color scale: +/- {limit:.6g} nm/day^2")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
