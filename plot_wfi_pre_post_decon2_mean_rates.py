"""Compare per-tile mean ice rates before and after decon2."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_wfi_pre_post_ice_ratio import (
    DEFAULT_FIT_SUFFIX,
    ROOT,
    SCAS,
    TILE_COUNT,
    _one_tile_thickness,
    detector_output_dir,
)


def endpoint_mean_rates(days, thickness, decon2_day):
    days = np.asarray(days, dtype=float).reshape(-1)
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    if days.size != thickness.size or days.size < 4:
        raise ValueError("Days and thickness must have at least four matching values")
    order = np.argsort(days, kind="stable")
    days = days[order]
    thickness = thickness[order]
    pre = days < decon2_day
    post = days >= decon2_day
    if not np.any(pre) or not np.any(post):
        raise ValueError("Decon2 day does not split the trajectory")
    pre_days = days[pre]
    post_days = days[post]
    pre_duration = pre_days[-1] - pre_days[0]
    post_duration = post_days[-1] - post_days[0]
    if pre_duration <= 0.0 or post_duration <= 0.0:
        raise ValueError("A decon2 segment has no elapsed-time span")
    pre_rate = (thickness[pre][-1] - thickness[pre][0]) / pre_duration
    post_rate = (thickness[post][-1] - thickness[post][0]) / post_duration
    return pre_rate, post_rate, pre_duration, post_duration


def load_detector_rates(output_dir, detector):
    fit_dirs = sorted((Path(output_dir) / "fits").glob("tile_*"))
    if len(fit_dirs) != TILE_COUNT:
        raise ValueError(
            f"Expected {TILE_COUNT} independent tile fits in {output_dir / 'fits'}, "
            f"found {len(fit_dirs)}"
        )

    rows = []
    decon2_days = []
    for tile_id, fit_dir in enumerate(fit_dirs):
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            archived_tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
            if archived_tile_ids.size != 1 or archived_tile_ids[0] != tile_id:
                raise ValueError(f"Unexpected tile IDs in {fit_dir}")
            days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            thickness = _one_tile_thickness(samples, days)
            decon2_values = np.asarray(samples["ice_reset_day"], dtype=float).reshape(-1)
            if decon2_values.size != 1 or not np.isfinite(decon2_values[0]):
                raise ValueError(f"Missing decon2 day in {fit_dir}")
            decon2_day = float(decon2_values[0])
        pre_rate, post_rate, pre_duration, post_duration = endpoint_mean_rates(
            days, thickness, decon2_day
        )
        decon2_days.append(decon2_day)
        rows.append(
            {
                "detector": detector,
                "tile_id": tile_id,
                "tile_row": tile_id // 8,
                "tile_col": tile_id % 8,
                "pre_decon2_start_day": float(np.min(days[days < decon2_day])),
                "pre_decon2_end_day": float(np.max(days[days < decon2_day])),
                "post_decon2_start_day": float(np.min(days[days >= decon2_day])),
                "post_decon2_end_day": float(np.max(days[days >= decon2_day])),
                "pre_decon2_duration_day": pre_duration,
                "post_decon2_duration_day": post_duration,
                "pre_decon2_mean_rate_nm_day": pre_rate,
                "post_decon2_mean_rate_nm_day": post_rate,
            }
        )
    if not np.allclose(decon2_days, decon2_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Inconsistent decon2 days in {output_dir}: {decon2_days}")
    return rows


def write_rows(path, rows):
    fields = tuple(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rates(path, rows_by_detector):
    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20", len(rows_by_detector))
    legend_handles = []
    all_values = []
    for index, detector in enumerate(rows_by_detector):
        rows = rows_by_detector[detector]
        color = colors(index)
        pre = np.asarray([row["pre_decon2_mean_rate_nm_day"] for row in rows])
        post = np.asarray([row["post_decon2_mean_rate_nm_day"] for row in rows])
        axis.scatter(pre, post, color=color, alpha=0.35, s=20, linewidths=0)
        legend_handles.append(Line2D([], [], color=color, linewidth=2.0, label=detector))
        all_values.extend((pre, post))

    all_values = np.concatenate(all_values)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    padding = 0.04 * max(upper - lower, 1.0)
    limits = (lower - padding, upper + padding)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Pre-decon2 mean ice rate (nm/day)")
    axis.set_ylabel("Post-decon2 mean ice rate (nm/day)")
    axis.set_title("Post-decon2 versus pre-decon2 mean ice rate per tile")
    axis.grid(alpha=0.2)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(
        "WFI01-WFI18; each point is one independent tile; dashed line is equality",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fit-suffix", default=DEFAULT_FIT_SUFFIX)
    parser.add_argument("--detector", action="append", choices=SCAS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = args.detector or SCAS
    rows_by_detector = {
        detector: load_detector_rates(
            detector_output_dir(args.root, detector, args.fit_suffix), detector
        )
        for detector in detectors
    }
    output = args.output or (
        args.root
        / DEFAULT_FIT_SUFFIX
        / "wfi_pre_post_decon2_mean_rate_comparison_all_tiles.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_name(output.stem + ".csv")
    plot_rates(output, rows_by_detector)
    write_rows(csv_path, [row for rows in rows_by_detector.values() for row in rows])
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
