"""Plot per-tile ice rates in the Roman WFI focal-plane mosaic."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
METRICS = (
    ("Initial rate", "initial_rate_nm_day_median"),
    ("Final rate", "final_rate_nm_day_median"),
    ("Mean rate", "mean_rate_nm_day_median"),
)

# The Roman WFI layout is a staggered six-column by three-row mosaic.  The
# offsets reproduce the smile-shaped focal plane shown in the Roman reference
# frame figure; y increases toward the top of the plot.
COLUMN_SCA = (
    ("WFI16", "WFI17", "WFI18"),
    ("WFI13", "WFI14", "WFI15"),
    ("WFI10", "WFI11", "WFI12"),
    ("WFI01", "WFI02", "WFI03"),
    ("WFI04", "WFI05", "WFI06"),
    ("WFI07", "WFI08", "WFI09"),
)
COLUMN_Y_OFFSET = (0.0, -4.0, -6.0, -6.0, -4.0, 0.0)
SCA_SIZE = 8.0
SCA_PITCH = 9.0


def sca_position(sca):
    for column, scas in enumerate(COLUMN_SCA):
        if sca in scas:
            # COLUMN_SCA is listed top-to-bottom, while plot y increases up.
            row = len(scas) - 1 - scas.index(sca)
            return SCA_PITCH * column, SCA_PITCH * row + COLUMN_Y_OFFSET[column]
    raise ValueError(f"Unknown SCA: {sca}")


def load_rate_map(path, expected_sca):
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row.get("detector") for row in rows} != {expected_sca}:
        return None

    maps = {}
    for _, field in METRICS:
        values = np.full((8, 8), np.nan)
        for row in rows:
            values[int(row["tile_row"]), int(row["tile_col"])] = float(row[field])
        # MAST L1/L2 arrays are already in the common WFI science frame.
        # Tile row 0 is the science-frame bottom, so no physical-detector flip
        # belongs here.
        maps[field] = values
    return maps


def load_all_maps(root, fit_suffix):
    maps = {}
    for sca in SCAS:
        output_dirs = (
            root / f"{sca.lower()}_{fit_suffix}_prod",
            root / f"{sca.lower()}_{fit_suffix}",
        )
        maps[sca] = next(
            (
                rate_map
                for output_dir in output_dirs
                if (rate_map := load_rate_map(output_dir / "tile_rate_posterior.csv", sca))
                is not None
            ),
            None,
        )
    return maps


def latest_pass_label(statistics_path):
    if statistics_path.is_file():
        paths = [statistics_path]
    elif statistics_path.is_dir():
        paths = sorted(statistics_path.glob("wfi*_tile_log_statistics_*.csv"))
    else:
        return "Pass unavailable"
    latest_repeat = -1
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                latest_repeat = max(latest_repeat, int(row["repeat"]))
    if latest_repeat < 0:
        return "Pass unavailable"
    return f"Pass {latest_repeat + 1} (repeat index {latest_repeat})"


def draw_rate_mosaic(path, maps, model_label, pass_label, local_contrast=False):
    available = [sca for sca in SCAS if maps[sca] is not None]
    figure, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    if not local_contrast:
        all_values = np.concatenate(
            [
                maps[sca][field][np.isfinite(maps[sca][field])]
                for sca in available
                for _, field in METRICS
            ]
        )
        global_min = float(np.min(all_values))
        global_max = float(np.max(all_values))
        if global_min == global_max:
            global_max = global_min + 1.0

    for axis, (title, field) in zip(axes, METRICS):
        plotted = {}
        if local_contrast:
            for sca in available:
                values = maps[sca][field]
                finite = values[np.isfinite(values)]
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                plotted[sca] = np.zeros_like(values) + 0.5 if vmin == vmax else (values - vmin) / (vmax - vmin)
            color_min, color_max = 0.0, 1.0
        else:
            color_min = global_min
            color_max = global_max
            plotted = {sca: maps[sca][field] for sca in available}
        image = None
        for sca in SCAS:
            x, y = sca_position(sca)
            if maps[sca] is None:
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
                    f"{sca}\npending",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#333333",
                )
                continue
            image = axis.imshow(
                plotted[sca],
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
                sca,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                path_effects=None,
            )
            for tile in range(1, 8):
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
        if image is not None:
            colorbar_label = (
                "Within-SCA relative rate (min to max)"
                if local_contrast
                else "nm/day"
            )
            figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label=colorbar_label)
        axis.set_aspect("equal")
        axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
        min_y = min(COLUMN_Y_OFFSET) - 1.0
        max_y = SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0
        axis.set_ylim(min_y, max_y)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{title} (within-SCA contrast)"
            if local_contrast
            else f"{title} (nm/day)"
        )
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

    figure.suptitle(
        f"Roman WFI full focal-plane ice accumulation rates\n"
        f"{model_label}; {pass_label}; {len(available)}/18 SCA rate fields available"
        + ("; each SCA independently scaled" if local_contrast else ""),
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--fit-suffix",
        default="rcs1039_full64_rate_5nm_np_ledacm_5000",
        help="Suffix after the lower-case SCA name in each fit directory.",
    )
    parser.add_argument(
        "--tile-statistics",
        type=Path,
        default=None,
        help="Statistics CSV or directory used to derive the latest pass label.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.root / "wfi_rcs1039_full_focal_plane_rate_maps.png"
    maps = load_all_maps(args.root, args.fit_suffix)
    statistics_path = args.tile_statistics or args.root / "wfi_tile_log_statistics_512"
    if not statistics_path.is_absolute():
        statistics_path = args.root / statistics_path
    pass_label = latest_pass_label(statistics_path)
    if "production_sharedAR_tileRate_twoCommonACM" in args.fit_suffix:
        model_label = (
            "constant-rate modulation; tile-specific rate scales; "
            "shared AR and common ACM steps"
        )
    elif "np_ledacm" in args.fit_suffix:
        model_label = "square-root + NP modulation"
    else:
        model_label = "fully nonparametric trajectory"
    draw_rate_mosaic(output, maps, model_label, pass_label)
    contrast_output = output.with_name(output.stem + "_tile_contrast.png")
    draw_rate_mosaic(
        contrast_output,
        maps,
        model_label,
        pass_label,
        local_contrast=True,
    )
    available = [sca for sca in SCAS if maps[sca] is not None]
    pending = [sca for sca in SCAS if maps[sca] is None]
    print(f"Wrote {output}")
    print(f"Wrote {contrast_output}")
    print(f"Available SCAs: {', '.join(available)}")
    if pending:
        print(f"Pending or invalid SCAs: {', '.join(pending)}")


if __name__ == "__main__":
    main()
