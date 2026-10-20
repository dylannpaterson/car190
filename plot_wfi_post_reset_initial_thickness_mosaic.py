"""Plot post-reset initial ice thickness in the Roman WFI focal-plane mosaic."""

import argparse
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
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    partial_tile_grid_size,
    positive_integer,
    production_output_root,
    tile_directory_entries,
    tile_grid_size_for_tile_size,
)


ROOT = Path(__file__).resolve().parent
LEGACY_MODEL_NAME = (
    "absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
)
RESET_DAY = 15.064405023148149


def detector_fit_dirs(root, detector, tile_size=DEFAULT_TILE_SIZE):
    """Return a complete current layout, then a complete legacy layout."""
    candidates = (
        production_output_root(root, tile_size) / detector.lower() / "fits",
        Path(root) / f"{detector.lower()}_{LEGACY_MODEL_NAME}" / "fits",
    )
    for fit_root in candidates:
        found = {}
        for tile_id, path in tile_directory_entries(fit_root):
            posterior = path / "posterior_samples.npz"
            if posterior.is_file():
                found[tile_id] = path
        try:
            complete_tile_ids(found, label=f"tile IDs in {fit_root}")
        except ValueError:
            continue
        if found:
            return sorted(found.items())
    return []


def _scalar(samples, name):
    if name not in samples:
        return None
    values = np.asarray(samples[name], dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        return None
    return float(values[0])


def load_post_reset_initial_thickness_map(
    root,
    detector,
    tile_size=DEFAULT_TILE_SIZE,
):
    """Load the physical post-reset initial thickness for one detector."""
    found = detector_fit_dirs(root, detector, tile_size)
    if not found:
        return None
    tile_grid_size = tile_grid_size_for_tile_size(tile_size)
    if max(tile_id for tile_id, _path in found) >= tile_grid_size**2:
        tile_grid_size = partial_tile_grid_size(tile_id for tile_id, _path in found)
    values = np.full((tile_grid_size, tile_grid_size), np.nan)
    for tile_id, fit_dir in found:
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            thickness = _scalar(samples, "reset_initial_thickness_nm")
            if thickness is None:
                log_thickness = _scalar(samples, "reset_initial_thickness_log_nm")
                if log_thickness is not None:
                    thickness = float(np.exp(log_thickness))
        if thickness is not None:
            values[tile_id // tile_grid_size, tile_id % tile_grid_size] = thickness
    return values if np.isfinite(values).any() else None


def load_all_post_reset_initial_thickness_maps(root, tile_size=DEFAULT_TILE_SIZE):
    return {
        detector: load_post_reset_initial_thickness_map(root, detector, tile_size)
        for detector in SCAS
    }


def draw_post_reset_initial_thickness_mosaic(path, maps, reset_day=RESET_DAY):
    available = [
        detector
        for detector in SCAS
        if maps[detector] is not None and np.isfinite(maps[detector]).any()
    ]
    if not available:
        raise ValueError("No post-reset initial-thickness maps were found")

    all_values = np.concatenate(
        [maps[detector][np.isfinite(maps[detector])] for detector in available]
    )
    color_min = float(np.min(all_values))
    color_max = float(np.max(all_values))
    if color_max <= color_min:
        color_max = color_min + 1.0e-6

    figure, axis = plt.subplots(figsize=(13, 8.5), constrained_layout=True)
    image = None
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
        axis.text(
            x + SCA_SIZE / 2,
            y + SCA_SIZE / 2,
            detector,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )
        tile_grid_size = values.shape[0]
        for tile in range(1, tile_grid_size):
            position = SCA_SIZE * tile / tile_grid_size
            axis.plot(
                (x + position, x + position),
                (y, y + SCA_SIZE),
                color="white",
                alpha=0.35,
                linewidth=0.35,
            )
            axis.plot(
                (x, x + SCA_SIZE),
                (y + position, y + position),
                color="white",
                alpha=0.35,
                linewidth=0.35,
            )

    if image is not None:
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="nm")
    axis.set_aspect("equal")
    axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
    axis.set_ylim(
        min(COLUMN_Y_OFFSET) - 1.0,
        SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Roman WFI focal-plane mosaic")
    axis.set_title("Post-Decon2 initial ice thickness (nm)")
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
    axis.text(
        0.99,
        0.02,
        f"reset day = {reset_day:.9f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    figure.suptitle(
        "Roman WFI post-reset initial ice-thickness mosaic\n"
        f"independent per-tile MAP fits; {len(available)}/18 detector maps available; "
        f"scale {color_min:.3g}--{color_max:.3g} nm",
        fontsize=14,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Parent folder containing production or legacy detector fit outputs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile edge length used by the production reduction; default: {DEFAULT_TILE_SIZE}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or (
        production_output_root(args.root, args.tile_size)
        / "wfi_post_reset_initial_thickness_mosaic.png"
    )
    maps = load_all_post_reset_initial_thickness_maps(args.root, args.tile_size)
    draw_post_reset_initial_thickness_mosaic(output, maps, args.reset_day)
    available = [
        detector
        for detector in SCAS
        if maps[detector] is not None and np.isfinite(maps[detector]).any()
    ]
    pending = [detector for detector in SCAS if detector not in available]
    print(f"Wrote {output}")
    print(f"Available detectors: {', '.join(available)}")
    if pending:
        print(f"Pending or invalid detectors: {', '.join(pending)}")


if __name__ == "__main__":
    main()
