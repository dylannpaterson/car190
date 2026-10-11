"""Compare pre-decon2 and post-decon2 ice trajectories at matched epoch age."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_FIT_SUFFIX = "production"
TILE_COUNT = 64


def detector_output_dir(root, detector, fit_suffix):
    root = Path(root)
    if fit_suffix == DEFAULT_FIT_SUFFIX:
        return root / DEFAULT_FIT_SUFFIX / detector.lower()
    return root / f"{detector.lower()}_{fit_suffix}"


def matched_epoch_ratio(days, thickness, decon2_day, sample_count=200):
    """Return pre/post-decon2 thickness and fractional difference on a shared age grid."""
    days = np.asarray(days, dtype=float).reshape(-1)
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    if days.size != thickness.size or days.size < 4:
        raise ValueError("Days and thickness must have at least four matching values")
    order = np.argsort(days, kind="stable")
    days = days[order]
    thickness = thickness[order]
    pre = days < decon2_day
    decon2 = days >= decon2_day
    if not np.any(pre) or not np.any(decon2):
        raise ValueError("Decon2 day does not split the trajectory")
    pre_days = days[pre] - days[pre][0]
    decon2_days = days[decon2] - days[decon2][0]
    common_duration = min(pre_days[-1], decon2_days[-1])
    if common_duration <= 0.0:
        raise ValueError("Pre/post-decon2 trajectories have no common elapsed-time range")
    elapsed_days = np.linspace(0.0, common_duration, sample_count)
    pre_ice = np.interp(elapsed_days, pre_days, thickness[pre])
    decon2_ice = np.interp(elapsed_days, decon2_days, thickness[decon2])
    if np.any(pre_ice == 0.0):
        raise ValueError("Pre-decon2 thickness contains zero on the comparison grid")
    fractional_difference = (decon2_ice - pre_ice) / pre_ice
    return elapsed_days, pre_ice, decon2_ice, fractional_difference


def _one_tile_thickness(samples, days):
    values = np.asarray(samples["tile_ice_thickness"], dtype=float)
    if values.size % days.size != 0:
        raise ValueError("Tile thickness does not match observation days")
    values = values.reshape(-1, days.size)
    return np.median(values, axis=0)


def load_detector_tile_trajectories(output_dir, sample_count=200):
    fit_dirs = sorted((Path(output_dir) / "fits").glob("tile_*"))
    if len(fit_dirs) != TILE_COUNT:
        raise ValueError(
            f"Expected {TILE_COUNT} independent tile fits in {output_dir / 'fits'}, "
            f"found {len(fit_dirs)}"
        )

    elapsed_days = None
    decon2_days = []
    pre_trajectories = []
    decon2_trajectories = []
    ratios = []
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
            current_elapsed, pre_ice, decon2_ice, ratio = matched_epoch_ratio(
                days, thickness, float(decon2_values[0]), sample_count=sample_count
            )
        if elapsed_days is None:
            elapsed_days = current_elapsed
        elif not np.allclose(elapsed_days, current_elapsed):
            raise ValueError(f"Common comparison grid differs in {fit_dir}")
        decon2_days.append(float(decon2_values[0]))
        pre_trajectories.append(pre_ice)
        decon2_trajectories.append(decon2_ice)
        ratios.append(ratio)

    if not np.allclose(decon2_days, decon2_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Inconsistent decon2 days in {output_dir}: {decon2_days}")
    return (
        elapsed_days,
        np.asarray(pre_trajectories),
        np.asarray(decon2_trajectories),
        np.asarray(ratios),
        decon2_days[0],
    )


def summarize_tile_trajectories(tile_trajectories):
    return {
        detector: (
            elapsed,
            np.percentile(pre, [16, 50, 84], axis=0),
            np.percentile(decon2, [16, 50, 84], axis=0),
            np.percentile(ratio, [16, 50, 84], axis=0),
            decon2_day,
        )
        for detector, (elapsed, pre, decon2, ratio, decon2_day)
        in tile_trajectories.items()
    }


def write_quantiles(path, trajectories):
    fields = (
        "detector",
        "elapsed_days",
        "pre_ice_p16_nm",
        "pre_ice_median_nm",
        "pre_ice_p84_nm",
        "post_decon2_ice_p16_nm",
        "post_decon2_ice_median_nm",
        "post_decon2_ice_p84_nm",
        "fractional_difference_p16",
        "fractional_difference_median",
        "fractional_difference_p84",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, (elapsed, pre, decon2, ratio, _) in trajectories.items():
            for index, day in enumerate(elapsed):
                writer.writerow(
                    {
                        "detector": detector,
                        "elapsed_days": day,
                        "pre_ice_p16_nm": pre[0, index],
                        "pre_ice_median_nm": pre[1, index],
                        "pre_ice_p84_nm": pre[2, index],
                        "post_decon2_ice_p16_nm": decon2[0, index],
                        "post_decon2_ice_median_nm": decon2[1, index],
                        "post_decon2_ice_p84_nm": decon2[2, index],
                        "fractional_difference_p16": ratio[0, index],
                        "fractional_difference_median": ratio[1, index],
                        "fractional_difference_p84": ratio[2, index],
                    }
                )


def plot_ratio(path, trajectories):
    figure, axis = plt.subplots(figsize=(13, 7))
    colors = plt.get_cmap("tab20", len(trajectories))
    legend_handles = []
    for index, detector in enumerate(trajectories):
        elapsed, _, _, ratio, _ = trajectories[detector]
        color = "#D55E00" if detector == "WFI13" else colors(index)
        linewidth = 2.8 if detector == "WFI13" else 1.2
        alpha = 0.16 if detector != "WFI13" else 0.22
        axis.fill_between(elapsed, ratio[0], ratio[2], color=color, alpha=alpha)
        axis.plot(elapsed, ratio[1], color=color, linewidth=linewidth)
        legend_handles.append(Line2D([], [], color=color, linewidth=2.0, label=detector))

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Elapsed days within epoch (pre-decon2: first exposure; post-decon2: boundary)")
    axis.set_ylabel("(post-decon2 ice - pre-decon2 ice) / pre-decon2 ice")
    axis.set_title("Post-decon2 versus pre-decon2 ice at matched epoch age")
    axis.grid(alpha=0.2)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(
        "WFI01-WFI18 tile trajectory comparison; bands are tile 16th-84th percentiles",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_thickness_comparison(path, trajectories):
    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20", len(trajectories))
    legend_handles = []
    all_values = []
    for index, detector in enumerate(trajectories):
        _, pre, decon2, _, _ = trajectories[detector]
        color = "#D55E00" if detector == "WFI13" else colors(index)
        linewidth = 2.8 if detector == "WFI13" else 1.2
        axis.plot(pre[1], decon2[1], color=color, linewidth=linewidth)
        legend_handles.append(Line2D([], [], color=color, linewidth=2.0, label=detector))
        all_values.extend((pre[1], decon2[1]))

    all_values = np.concatenate(all_values)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    padding = 0.03 * max(upper - lower, 1.0)
    limits = (lower - padding, upper + padding)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Pre-decon2 ice thickness (nm)")
    axis.set_ylabel("Post-decon2 ice thickness (nm)")
    axis.set_title("Post-decon2 versus pre-decon2 ice at matched epoch age")
    axis.grid(alpha=0.2)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(
        "WFI01-WFI18 median tile trajectories; dashed line is equality",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_tile_thickness_comparison(path, trajectories):
    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20", len(trajectories))
    legend_handles = []
    all_values = []
    for index, detector in enumerate(trajectories):
        _, pre, decon2, _, _ = trajectories[detector]
        color = colors(index)
        for pre_tile, decon2_tile in zip(pre, decon2):
            axis.plot(
                pre_tile,
                decon2_tile,
                color=color,
                linewidth=0.75,
                alpha=0.24,
            )
        legend_handles.append(Line2D([], [], color=color, linewidth=2.0, label=detector))
        all_values.extend((pre.ravel(), decon2.ravel()))

    all_values = np.concatenate(all_values)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    padding = 0.03 * max(upper - lower, 1.0)
    limits = (lower - padding, upper + padding)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Pre-decon2 ice thickness (nm)")
    axis.set_ylabel("Post-decon2 ice thickness (nm)")
    axis.set_title("Post-decon2 versus pre-decon2 ice at matched epoch age")
    axis.grid(alpha=0.2)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(
        "WFI01-WFI18 independent tile trajectories; each detector has one color",
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
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = args.detector or SCAS
    trajectories = {}
    for detector in detectors:
        output_dir = detector_output_dir(args.root, detector, args.fit_suffix)
        trajectories[detector] = load_detector_tile_trajectories(
            output_dir, sample_count=args.sample_count
        )
    output = args.output or (
        args.root
        / DEFAULT_FIT_SUFFIX
        / "wfi_pre_post_decon2_ice_thickness_comparison_all_tiles.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    quantile_path = output.with_name(output.stem + "_quantiles.csv")
    plot_tile_thickness_comparison(output, trajectories)
    write_quantiles(quantile_path, summarize_tile_trajectories(trajectories))
    print(f"Wrote {output}")
    print(f"Wrote {quantile_path}")


if __name__ == "__main__":
    main()
