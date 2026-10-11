"""Measure late-1063 deviations from per-tile temperature sine fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from diagnose_wfi_temperature_correction import fit_window, load_rows
from plot_wfi_full_focal_plane_rates import (
    COLUMN_Y_OFFSET,
    SCAS,
    SCA_PITCH,
    SCA_SIZE,
    sca_position,
)
from run_wfi_reduction_pipeline import _evaluate_count_sinusoid


REFERENCE_LED = 3
REFERENCE_PROGRAM = "1039"
TEMPERATURE_PROGRAM = "1063"
DEFAULT_FIT_START_REPEAT = 669
DEFAULT_FIT_END_REPEAT = 682
DEFAULT_LATE_START_REPEAT = 682
DEFAULT_END_POINTS = 5


def log_residual_millilog(values, expected):
    values = np.asarray(values, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if np.any(values <= 0.0) or np.any(expected <= 0.0):
        raise ValueError("Residual inputs must be positive")
    return 1000.0 * np.log(values / expected)


def summarize_tile(
    rows,
    detector,
    tile,
    fit_start_repeat,
    fit_end_repeat,
    late_start_repeat,
    end_points,
):
    fit_rows, fit = fit_window(
        rows,
        tile,
        REFERENCE_LED,
        fit_start_repeat,
        fit_end_repeat,
    )
    fit_expected = _evaluate_count_sinusoid(
        fit, np.asarray([row["days"] for row in fit_rows], dtype=float)
    )
    fit_residual = log_residual_millilog(
        [row["acm_mean"] for row in fit_rows], fit_expected
    )

    late_rows = sorted(
        [
            row
            for row in rows
            if row["tile"] == tile
            and row["led"] == REFERENCE_LED
            and row["program"] == TEMPERATURE_PROGRAM
            and row["repeat"] >= late_start_repeat
        ],
        key=lambda row: row["days"],
    )
    if not late_rows:
        raise ValueError(
            f"No late {TEMPERATURE_PROGRAM} LED {REFERENCE_LED} rows "
            f"for {detector} tile {tile}"
        )
    end_rows = late_rows[-end_points:]
    late_expected = _evaluate_count_sinusoid(
        fit, np.asarray([row["days"] for row in late_rows], dtype=float)
    )
    end_expected = _evaluate_count_sinusoid(
        fit, np.asarray([row["days"] for row in end_rows], dtype=float)
    )
    late_residual = log_residual_millilog(
        [row["final_mean"] for row in late_rows], late_expected
    )
    end_residual = log_residual_millilog(
        [row["final_mean"] for row in end_rows], end_expected
    )

    return {
        "detector": detector,
        "tile": tile,
        "tile_row": tile // 8,
        "tile_col": tile % 8,
        "fit_start_repeat": fit_start_repeat,
        "fit_end_repeat": fit_end_repeat,
        "fit_points": len(fit_rows),
        "fit_period_days": fit["period_days"],
        "fit_rmse_counts": fit["rmse"],
        "fit_r2": fit["r2"],
        "fit_residual_rms_mlog": float(np.sqrt(np.mean(fit_residual**2))),
        "late_start_repeat": int(late_rows[0]["repeat"]),
        "late_end_repeat": int(late_rows[-1]["repeat"]),
        "late_points": len(late_rows),
        "late_residual_median_mlog": float(np.median(late_residual)),
        "late_residual_rms_mlog": float(np.sqrt(np.mean(late_residual**2))),
        "late_residual_min_mlog": float(np.min(late_residual)),
        "late_residual_max_mlog": float(np.max(late_residual)),
        "end_points": len(end_rows),
        "end_start_repeat": int(end_rows[0]["repeat"]),
        "end_end_repeat": int(end_rows[-1]["repeat"]),
        "end_residual_mean_mlog": float(np.mean(end_residual)),
        "end_residual_median_mlog": float(np.median(end_residual)),
        "end_residual_rms_mlog": float(np.sqrt(np.mean(end_residual**2))),
        "end_residual_min_mlog": float(np.min(end_residual)),
        "end_residual_max_mlog": float(np.max(end_residual)),
        "last_residual_mlog": float(end_residual[-1]),
    }


def write_summary(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def detector_map(summary, detector, field):
    values = np.full((8, 8), np.nan, dtype=float)
    for row in summary:
        if row["detector"] == detector:
            values[int(row["tile_row"]), int(row["tile_col"])] = float(row[field])
    return values if np.isfinite(values).any() else None


def plot_focal_plane(path, summary):
    panels = (
        ("fit_r2", "Fit R2", "viridis", False),
        ("fit_residual_rms_mlog", "Fit residual RMS (millilog)", "magma", False),
        ("fit_period_days", "Fitted period (days)", "cividis", False),
        ("end_residual_median_mlog", "End-5 median residual (millilog)", "coolwarm", True),
        ("end_residual_rms_mlog", "End-5 residual RMS (millilog)", "magma", False),
        ("last_residual_mlog", "Last 1063 residual (millilog)", "coolwarm", True),
    )
    available = [
        detector
        for detector in SCAS
        if any(row["detector"] == detector for row in summary)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for axis, (field, label, cmap, signed) in zip(axes.flat, panels):
        maps = {
            detector: detector_map(summary, detector, field)
            for detector in available
        }
        finite_values = np.concatenate(
            [values[np.isfinite(values)] for values in maps.values() if values is not None]
        )
        if signed:
            limit = max(
                abs(float(np.min(finite_values))),
                abs(float(np.max(finite_values))),
                1.0e-12,
            )
            image_kwargs = {
                "norm": TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            }
        else:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            if vmin == vmax:
                vmax = vmin + 1.0
            image_kwargs = {"vmin": vmin, "vmax": vmax}

        image = None
        for detector in SCAS:
            x, y = sca_position(detector)
            values = maps.get(detector)
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
                cmap=cmap,
                extent=(x, x + SCA_SIZE, y, y + SCA_SIZE),
                **image_kwargs,
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
                x + SCA_SIZE / 2,
                y + SCA_SIZE / 2 + 0.65,
                detector,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
            axis.text(
                x + SCA_SIZE / 2,
                y + 0.25,
                f"med {median:+.3g}" if signed else f"med {median:.3g}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="white",
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
            figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
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
        axis.set_title(label)
    figure.suptitle(
        "Roman WFI LED 3 temperature sine fit and late Program 1063 residuals\n"
        "Per-detector/tile fits: 1039 repeats [669, 682); "
        "end residuals: final 5 1063 observations"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--detector",
        default=None,
        choices=SCAS,
        help="Analyze one detector; default analyzes the full WFI focal plane.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-start-repeat", type=int, default=DEFAULT_FIT_START_REPEAT)
    parser.add_argument("--fit-end-repeat", type=int, default=DEFAULT_FIT_END_REPEAT)
    parser.add_argument("--late-start-repeat", type=int, default=DEFAULT_LATE_START_REPEAT)
    parser.add_argument("--end-points", type=int, default=DEFAULT_END_POINTS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end_points < 1:
        raise ValueError("--end-points must be positive")
    detectors = (args.detector,) if args.detector is not None else SCAS
    summary = []
    row_count = 0
    for detector in detectors:
        rows = load_rows(args.run_dir, detector)
        row_count += len(rows)
        tiles = sorted({row["tile"] for row in rows})
        summary.extend(
            summarize_tile(
                rows,
                detector,
                tile,
                args.fit_start_repeat,
                args.fit_end_repeat,
                args.late_start_repeat,
                args.end_points,
            )
            for tile in tiles
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.detector.lower() if args.detector is not None else "wfi_all_detectors"
    summary_path = args.output_dir / (
        f"{prefix}_temperature_sine_focal_plane_summary.csv"
    )
    plot_path = args.output_dir / f"{prefix}_temperature_sine_focal_plane.png"
    write_summary(summary_path, summary)
    plot_focal_plane(plot_path, summary)
    print(f"Loaded {row_count} valid scaled rows for {len(detectors)} detector(s)")
    print(f"Wrote {len(summary)} per-tile metrics to {summary_path}")
    print(f"Wrote focal-plane maps to {plot_path}")


if __name__ == "__main__":
    main()
