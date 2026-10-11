"""Plot post-reset grey log-offset values from the current production layout."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from plot_wfi_full_focal_plane_rates import (
    SCAS,
    SCA_PITCH,
    SCA_SIZE,
    COLUMN_Y_OFFSET,
    sca_position,
)


TILE_GRID_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_detector_map(production_root, detector):
    detector_root = Path(production_root) / "production" / detector.lower()
    rate_path = detector_root / "tile_rate_posterior.csv"
    if not rate_path.is_file():
        return None
    with rate_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = np.full((TILE_GRID_SIZE, TILE_GRID_SIZE), np.nan)
    for row in rows:
        tile_id = int(row["tile_id"])
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        if tile_id != tile_row * TILE_GRID_SIZE + tile_col:
            raise ValueError(f"Invalid tile coordinates in {rate_path}: {row}")
        posterior_path = (
            detector_root / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz"
        )
        if not posterior_path.is_file():
            raise FileNotFoundError(posterior_path)
        with np.load(posterior_path, allow_pickle=False) as samples:
            if "post_reset_tile_log_offset" not in samples:
                raise ValueError(f"Missing post-reset grey offset in {posterior_path}")
            offset = float(
                np.median(np.asarray(samples["post_reset_tile_log_offset"], dtype=float))
            )
        values[tile_row, tile_col] = 100.0 * np.expm1(offset)
    if not np.isfinite(values).all():
        raise ValueError(f"Expected 64 grey offsets in {detector_root}")
    return values


def load_maps(production_root):
    return {
        detector: load_detector_map(production_root, detector) for detector in SCAS
    }


def draw_mosaic(path, maps):
    available = [
        detector
        for detector in SCAS
        if maps[detector] is not None and np.isfinite(maps[detector]).any()
    ]
    if not available:
        raise ValueError("No current production grey-offset maps were found")
    all_values = np.concatenate(
        [maps[detector][np.isfinite(maps[detector])] for detector in available]
    )
    limit = max(float(np.max(np.abs(all_values))), 1.0e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    figure, axis = plt.subplots(figsize=(13, 8.5), constrained_layout=True)
    image = None
    for detector in SCAS:
        x, y = sca_position(detector)
        values = maps[detector]
        if values is None:
            axis.add_patch(
                Rectangle(
                    (x, y),
                    SCA_SIZE,
                    SCA_SIZE,
                    facecolor="#e6e6e6",
                    edgecolor="#333333",
                    hatch="///",
                    linewidth=1.2,
                )
            )
            axis.text(
                x + SCA_SIZE / 2,
                y + SCA_SIZE / 2,
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
        median = float(np.median(values))
        axis.text(
            x + SCA_SIZE / 2,
            y + SCA_SIZE / 2 + 0.7,
            detector,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
        axis.text(
            x + SCA_SIZE / 2,
            y + 0.35,
            f"median {median:+.3f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="white",
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

    axis.set_aspect("equal")
    axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
    axis.set_ylim(
        min(COLUMN_Y_OFFSET) - 1.0,
        SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Roman WFI focal-plane mosaic")
    if image is not None:
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="Grey offset (%)")
    figure.suptitle(
        "Roman WFI fitted post-reset grey log-offset\n"
        f"{len(available)}/18 detectors; median per tile"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return all_values


def main():
    args = parse_args()
    maps = load_maps(args.production_root)
    values = draw_mosaic(args.output, maps)
    print(f"Available detectors: {sum(value is not None for value in maps.values())}")
    print(f"Grey offset range: {np.min(values):+.6g}% to {np.max(values):+.6g}%")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
