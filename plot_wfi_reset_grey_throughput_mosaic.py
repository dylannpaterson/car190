"""Plot production per-tile reset grey-throughput changes in the WFI mosaic."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from plot_wfi_full_focal_plane_rates import (
    COLUMN_Y_OFFSET,
    SCAS,
    SCA_PITCH,
    SCA_SIZE,
    sca_position,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PRODUCTION_PREFIX = (
    "wfi_production_20261005_shared_hgcdte_prior10_smooth001_"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rrn_reduction_20261005"
    / "wfi_all_detectors_count_space_common_delta_free_scale_mcmc"
    / "wfi_reset_grey_throughput_change_focal_plane.png"
)
DEFAULT_CONTRAST_OUTPUT = DEFAULT_OUTPUT.with_name(
    "wfi_reset_grey_throughput_change_focal_plane_tile_contrast.png"
)
POSTERIOR_NAME = "post_reset_tile_log_offset_posterior.csv"
FIT_SUFFIX = "_map10000_emcee1000"
TILE_GRID_SIZE = 8


def production_posterior_paths(root, fit_prefix):
    root = Path(root)
    paths = {}
    for detector in SCAS:
        path = (
            root
            / f"{fit_prefix}{detector}{FIT_SUFFIX}"
            / POSTERIOR_NAME
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing production grey-offset posterior: {path}")
        paths[detector] = path
    return paths


def load_production_maps(root, fit_prefix=DEFAULT_PRODUCTION_PREFIX):
    maps = {}
    for detector, path in production_posterior_paths(root, fit_prefix).items():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
            "post_reset_log_offset_p16",
            "post_reset_log_offset_median",
            "post_reset_log_offset_p84",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Production grey-offset posterior is incomplete: {path}")

        change = np.full((TILE_GRID_SIZE, TILE_GRID_SIZE), np.nan)
        interval_width = np.full_like(change, np.nan)
        seen = set()
        for row in rows:
            if str(row["detector"]).strip().upper() != detector:
                raise ValueError(f"Detector label mismatch in {path}")
            tile_id = int(row["tile_id"])
            tile_row = int(row["tile_row"])
            tile_col = int(row["tile_col"])
            if not (
                0 <= tile_id < TILE_GRID_SIZE**2
                and 0 <= tile_row < TILE_GRID_SIZE
                and 0 <= tile_col < TILE_GRID_SIZE
                and tile_id == tile_row * TILE_GRID_SIZE + tile_col
            ):
                raise ValueError(f"Invalid tile coordinates in {path}: {row}")
            if tile_id in seen:
                raise ValueError(f"Duplicate tile {tile_id} in {path}")
            seen.add(tile_id)

            p16 = 100.0 * np.expm1(float(row["post_reset_log_offset_p16"]))
            median = 100.0 * np.expm1(float(row["post_reset_log_offset_median"]))
            p84 = 100.0 * np.expm1(float(row["post_reset_log_offset_p84"]))
            if not np.isfinite((p16, median, p84)).all() or not p16 <= median <= p84:
                raise ValueError(f"Invalid grey-throughput posterior in {path}: {row}")
            change[tile_row, tile_col] = median
            interval_width[tile_row, tile_col] = p84 - p16

        if seen != set(range(TILE_GRID_SIZE**2)):
            raise ValueError(f"Expected 64 complete tiles in {path}")
        maps[detector] = {
            "change": change,
            "interval_width": interval_width,
        }
    return maps


def _mosaic_limits(axis):
    axis.set_aspect("equal")
    axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
    axis.set_ylim(
        min(COLUMN_Y_OFFSET) - 1.0,
        SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Roman WFI focal-plane mosaic")
    center_x = SCA_PITCH * 3 - 0.5
    center_y = SCA_PITCH * 2 + COLUMN_Y_OFFSET[2] + 1.2
    axis.plot(
        center_x,
        center_y,
        marker="x",
        color="#d62728",
        markersize=7,
        markeredgewidth=1.5,
    )
    axis.text(
        center_x + 0.3,
        center_y + 0.5,
        "WFI_CEN",
        color="#d62728",
        fontsize=7,
        va="bottom",
    )


def draw_scale_mosaic(path, maps, local_contrast=False):
    raw_values = {detector: maps[detector]["change"] for detector in SCAS}
    if local_contrast:
        values = {}
        for detector, detector_values in raw_values.items():
            detector_min = float(np.min(detector_values))
            detector_max = float(np.max(detector_values))
            values[detector] = (
                np.full_like(detector_values, 0.5)
                if detector_max <= detector_min
                else (detector_values - detector_min) / (detector_max - detector_min)
            )
        color_min, color_max = 0.0, 1.0
        colorbar_label = "Within-detector relative change (min to max)"
        title = "Roman WFI fitted post-reset grey-throughput change (within-detector contrast)"
    else:
        values = raw_values
        all_values = np.concatenate(
            [detector_values[np.isfinite(detector_values)] for detector_values in values.values()]
        )
        color_min = float(np.min(all_values))
        color_max = float(np.max(all_values))
        colorbar_label = "Fitted grey-throughput change (%)"
        title = "Roman WFI fitted post-reset grey-throughput change"
    if color_max <= color_min:
        color_max = color_min + 1.0e-6

    figure, axis = plt.subplots(figsize=(13, 8.5), constrained_layout=True)
    image = None
    for detector in SCAS:
        x, y = sca_position(detector)
        detector_values = values[detector]
        image = axis.imshow(
            detector_values,
            origin="lower",
            interpolation="none",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
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
        median = float(np.nanmedian(raw_values[detector]))
        axis.text(
            x + SCA_SIZE / 2,
            y + SCA_SIZE / 2 + 0.7,
            detector,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
        value_label = f"median {median:+.2f}%"
        axis.text(
            x + SCA_SIZE / 2,
            y + 0.35,
            value_label,
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

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.035,
        pad=0.02,
        label=colorbar_label,
    )
    _mosaic_limits(axis)
    figure.suptitle(
        f"{title}\n"
        "Production semi-infinite HgCdTe fits; one post-reset grey log correction per tile",
        fontsize=14,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, default=ROOT)
    parser.add_argument("--fit-prefix", default=DEFAULT_PRODUCTION_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contrast-output", type=Path, default=DEFAULT_CONTRAST_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    maps = load_production_maps(args.production_root, args.fit_prefix)
    draw_scale_mosaic(args.output, maps)
    draw_scale_mosaic(args.contrast_output, maps, local_contrast=True)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.contrast_output}")


if __name__ == "__main__":
    main()
