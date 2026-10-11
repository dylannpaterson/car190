"""Plot tile-mean ice/rate trajectories for the 18 production WFI fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from wfi_production_plot_annotations import (
    annotate_time_axis,
    event_legend_handles,
    load_time_annotations,
    split_time_series_at_reset,
)


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
TILES_PER_DETECTOR = 64
DEFAULT_FIT_SUFFIX = "production"
DEFAULT_RESET_DAY = 12.907397928240743


def _one_tile_series(samples, key):
    values = np.asarray(samples[key], dtype=float)
    if values.ndim == 3:
        values = values[0, 0]
    elif values.ndim == 2:
        values = values[0]
    return values.reshape(-1)


def detector_output_dir(root, detector, fit_suffix):
    root = Path(root)
    if fit_suffix == DEFAULT_FIT_SUFFIX:
        return root / DEFAULT_FIT_SUFFIX / detector.lower()
    return root / f"{detector.lower()}_{fit_suffix}"


def load_independent_trajectory(output_dir):
    fit_dirs = sorted(
        (Path(output_dir) / "fits").glob("tile_*")
    )
    if not fit_dirs:
        raise FileNotFoundError(f"No independent tile fits found in {output_dir}")
    thickness = []
    rates = []
    days = None
    for fit_dir in fit_dirs:
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            current_days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            if days is None:
                days = current_days
            elif not np.array_equal(days, current_days):
                raise ValueError(f"Observation days differ in {fit_dir}")
            thickness.append(_one_tile_series(samples, "tile_ice_thickness"))
            rates.append(_one_tile_series(samples, "tile_instantaneous_rate_nm_day"))
    thickness = np.asarray(thickness, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if thickness.shape[1] != days.size or rates.shape != thickness.shape:
        raise ValueError(f"Unexpected independent trajectory shape in {output_dir}")
    order = np.argsort(days, kind="stable")
    return (
        days[order],
        np.percentile(thickness[:, order], [16, 50, 84], axis=0),
        np.percentile(rates[:, order], [16, 50, 84], axis=0),
    )


def load_trajectory(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir / "posterior_samples.npz").is_file():
        return load_independent_trajectory(output_dir)
    with np.load(output_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float)
        ice = np.asarray(samples["ice_thickness"], dtype=float)
        if ice.ndim == 3:
            ice = np.mean(ice, axis=1)
        if ice.ndim != 2 or ice.shape[1] != len(days):
            raise ValueError(f"Unexpected ice-thickness shape in {output_dir}")

        rates = np.asarray(samples["instantaneous_rate_nm_day"], dtype=float)
        if rates.ndim == 3:
            rates = np.mean(rates, axis=1)
        if rates.ndim != 2 or rates.shape != ice.shape:
            raise ValueError(f"Unexpected rate shape in {output_dir}")

    order = np.argsort(days, kind="stable")
    days = days[order]
    ice_quantiles = np.percentile(ice[:, order], [16, 50, 84], axis=0)
    rate_quantiles = np.percentile(rates[:, order], [16, 50, 84], axis=0)
    return days, ice_quantiles, rate_quantiles


def load_independent_tile_thickness(output_dir):
    fit_dirs = sorted((Path(output_dir) / "fits").glob("tile_*"))
    if len(fit_dirs) != TILES_PER_DETECTOR:
        raise ValueError(
            f"Expected {TILES_PER_DETECTOR} tile fits in {output_dir}, "
            f"found {len(fit_dirs)}"
        )
    thickness = []
    days = None
    for fit_dir in fit_dirs:
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            current_days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            if days is None:
                days = current_days
            elif not np.array_equal(days, current_days):
                raise ValueError(f"Observation days differ in {fit_dir}")
            thickness.append(_one_tile_series(samples, "tile_ice_thickness"))
    thickness = np.asarray(thickness, dtype=float)
    order = np.argsort(days, kind="stable")
    return days[order], thickness[:, order]


def detrend_tile_trajectories(tile_trajectories, reset_day):
    detrended = {}
    for detector in SCAS:
        days, thickness = tile_trajectories[detector]
        pre_mask = days < reset_day
        post_mask = ~pre_mask
        if not np.any(pre_mask) or not np.any(post_mask):
            raise ValueError("Reset day must split every tile trajectory")
        residual = np.empty_like(thickness)
        for tile_id, values in enumerate(thickness):
            pre_trend = np.polyfit(days[pre_mask], values[pre_mask], 1)
            post_trend = np.polyfit(days[post_mask], values[post_mask], 1)
            residual[tile_id, pre_mask] = values[pre_mask] - np.polyval(
                pre_trend, days[pre_mask]
            )
            residual[tile_id, post_mask] = values[post_mask] - np.polyval(
                post_trend, days[post_mask]
            )
        detrended[detector] = (days, residual)
    return detrended


def fractional_tile_trajectories(detrended, tile_trajectories):
    fractional = {}
    for detector in SCAS:
        days, residual = detrended[detector]
        current_days, thickness = tile_trajectories[detector]
        if not np.array_equal(days, current_days):
            raise ValueError(f"Observation days differ for {detector}")
        if np.any(thickness <= 0.0):
            raise ValueError(f"Non-positive current ice thickness for {detector}")
        fractional[detector] = (days, residual / thickness)
    return fractional


def write_quantiles(path, trajectories):
    fields = (
        "detector",
        "days",
        "ice_thickness_p16_nm",
        "ice_thickness_median_nm",
        "ice_thickness_p84_nm",
        "mean_tile_rate_p16_nm_day",
        "mean_tile_rate_median_nm_day",
        "mean_tile_rate_p84_nm_day",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, (days, ice, rates) in trajectories.items():
            for index, day in enumerate(days):
                writer.writerow(
                    {
                        "detector": detector,
                        "days": day,
                        "ice_thickness_p16_nm": ice[0, index],
                        "ice_thickness_median_nm": ice[1, index],
                        "ice_thickness_p84_nm": ice[2, index],
                        "mean_tile_rate_p16_nm_day": rates[0, index],
                        "mean_tile_rate_median_nm_day": rates[1, index],
                        "mean_tile_rate_p84_nm_day": rates[2, index],
                    }
                )


def detrend_ice_trajectories(trajectories, reset_day):
    days = next(iter(trajectories.values()))[0]
    median_ice = []
    for detector in SCAS:
        detector_days, ice, _ = trajectories[detector]
        if not np.array_equal(days, detector_days):
            raise ValueError(f"Observation days differ for {detector}")
        median_ice.append(ice[1])

    average_ice = np.mean(median_ice, axis=0)
    pre_mask = days < reset_day
    post_mask = ~pre_mask
    if not np.any(pre_mask) or not np.any(post_mask):
        raise ValueError("Reset day must split the trajectory into two phases")

    pre_trend = np.polyfit(days[pre_mask], average_ice[pre_mask], 1)
    post_trend = np.polyfit(days[post_mask], average_ice[post_mask], 1)
    trend = np.empty_like(days)
    trend[pre_mask] = np.polyval(pre_trend, days[pre_mask])
    trend[post_mask] = np.polyval(post_trend, days[post_mask])

    detrended = {}
    for detector in SCAS:
        detector_days, ice, _ = trajectories[detector]
        detrended[detector] = (detector_days, ice - trend)
    return detrended, float(pre_trend[0]), float(post_trend[0])


def fractional_detrended_ice_trajectories(detrended, trajectories):
    fractional = {}
    for detector in SCAS:
        days, residual = detrended[detector]
        current_days, current_ice, _ = trajectories[detector]
        if not np.array_equal(days, current_days):
            raise ValueError(f"Observation days differ for {detector}")
        if np.any(current_ice <= 0.0):
            raise ValueError(f"Non-positive current ice thickness for {detector}")
        fractional[detector] = (days, residual / current_ice)
    return fractional


def plot_detrended_ice_thickness(
    path,
    trajectories,
    pre_rate,
    post_rate,
    reset_day=None,
    time_annotations=None,
    fractional=False,
):
    figure, axis = plt.subplots(figsize=(15, 6.5))
    colors = plt.get_cmap("tab20", len(SCAS))
    plot_reset_day = reset_day
    if plot_reset_day is None and time_annotations is not None:
        plot_reset_day = time_annotations.physical_reset_day
    legend_handles = []
    for index, detector in enumerate(SCAS):
        days, ice = trajectories[detector]
        days, ice_low, ice_median, ice_high = split_time_series_at_reset(
            days,
            ice[0],
            ice[1],
            ice[2],
            reset_day=plot_reset_day,
        )
        color = colors(index)
        axis.fill_between(days, ice_low, ice_high, color=color, alpha=0.08)
        axis.plot(days, ice_median, color=color, linewidth=1.4)
        legend_handles.append(
            Line2D([], [], color=color, linewidth=2.0, label=detector)
        )

    if time_annotations is not None:
        annotate_time_axis(axis, time_annotations)
    if plot_reset_day is not None:
        axis.axvline(
            plot_reset_day,
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
            label="Ice reset",
        )
    axis.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.6)
    axis.set_ylabel(
        "Detrended ice thickness / current ice thickness"
        if fractional
        else "Detrended tile-mean ice thickness (nm)"
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_title(
        "WFI01–WFI18 detrended ice thickness"
        + (" as a fraction of current thickness " if fractional else " ")
        + f"(linear trends removed; {pre_rate:.3f} pre / {post_rate:.3f} post nm/day)"
    )
    axis.set_xlim(left=0.0)
    axis.grid(alpha=0.2)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        frameon=False,
    )
    if time_annotations is not None:
        figure.legend(
            handles=event_legend_handles(time_annotations),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=8,
            frameon=False,
            fontsize=7,
        )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.90))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_all_tile_axis(
    axis,
    trajectories,
    reset_day=None,
    time_annotations=None,
    fractional=False,
):
    colors = plt.get_cmap("tab20", len(SCAS))
    plot_reset_day = reset_day
    if plot_reset_day is None and time_annotations is not None:
        plot_reset_day = time_annotations.physical_reset_day
    legend_handles = []
    for index, detector in enumerate(SCAS):
        days, values = trajectories[detector]
        plot_days, plot_values = split_time_series_at_reset(
            days,
            values,
            reset_day=plot_reset_day,
        )
        color = colors(index)
        for tile_values in plot_values:
            axis.plot(
                plot_days,
                tile_values,
                color=color,
                alpha=0.16,
                linewidth=0.55,
            )
        legend_handles.append(
            Line2D([], [], color=color, linewidth=2.0, label=detector)
        )

    if time_annotations is not None:
        annotate_time_axis(axis, time_annotations)
    if plot_reset_day is not None:
        axis.axvline(
            plot_reset_day,
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
            label="Ice reset",
        )
    axis.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.6)
    axis.set_ylabel(
        "Per-tile detrended ice thickness / current ice thickness"
        if fractional
        else "Per-tile detrended ice thickness (nm)"
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_title(
        "WFI01–WFI18 all-tile detrended ice thickness"
        + (" as a fraction of current thickness" if fractional else "")
        + " (separate linear trend for each tile before and after reset)"
    )
    axis.set_xlim(left=0.0)
    axis.grid(alpha=0.2)
    return legend_handles


def _add_all_tile_legends(figure, legend_handles, time_annotations):
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        frameon=False,
    )
    if time_annotations is not None:
        figure.legend(
            handles=event_legend_handles(time_annotations),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=8,
            frameon=False,
            fontsize=7,
        )


def plot_all_tile_detrended_ice_thickness(
    path,
    trajectories,
    reset_day=None,
    time_annotations=None,
    fractional=False,
):
    figure, axis = plt.subplots(figsize=(15, 6.5))
    legend_handles = _plot_all_tile_axis(
        axis,
        trajectories,
        reset_day=reset_day,
        time_annotations=time_annotations,
        fractional=fractional,
    )
    _add_all_tile_legends(figure, legend_handles, time_annotations)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.90))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_all_tile_detrended_ice_thickness_stacked(
    path,
    detrended,
    fractional,
    reset_day=None,
    time_annotations=None,
    alias_path=None,
):
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    legend_handles = _plot_all_tile_axis(
        axes[0],
        detrended,
        reset_day=reset_day,
        time_annotations=time_annotations,
        fractional=False,
    )
    _plot_all_tile_axis(
        axes[1],
        fractional,
        reset_day=reset_day,
        time_annotations=time_annotations,
        fractional=True,
    )
    axes[0].set_xlabel("")
    _add_all_tile_legends(figure, legend_handles, time_annotations)
    figure.tight_layout(rect=(0.0, 0.11, 1.0, 0.90))
    output_paths = (path,) if alias_path is None else (path, alias_path)
    for output_path in output_paths:
        figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_detector_trajectories(
    path,
    trajectories,
    reset_day=None,
    independent_fit=False,
    time_annotations=None,
):
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    colors = plt.get_cmap("tab20", len(SCAS))
    plot_reset_day = reset_day
    if plot_reset_day is None and time_annotations is not None:
        plot_reset_day = time_annotations.physical_reset_day
    legend_handles = []
    for index, detector in enumerate(SCAS):
        days, ice, rates = trajectories[detector]
        (
            days,
            ice_low,
            ice_median,
            ice_high,
            rate_low,
            rate_median,
            rate_high,
        ) = split_time_series_at_reset(
            days,
            ice[0],
            ice[1],
            ice[2],
            rates[0],
            rates[1],
            rates[2],
            reset_day=plot_reset_day,
        )
        color = colors(index)
        axes[0].fill_between(days, ice_low, ice_high, color=color, alpha=0.08)
        axes[0].plot(days, ice_median, color=color, linewidth=1.4)
        axes[1].fill_between(days, rate_low, rate_high, color=color, alpha=0.08)
        axes[1].plot(days, rate_median, color=color, linewidth=1.4)
        legend_handles.append(
            Line2D([], [], color=color, linewidth=2.0, label=detector)
        )

    for axis in axes:
        if time_annotations is not None:
            annotate_time_axis(axis, time_annotations)
        if plot_reset_day is not None:
            axis.axvline(
                plot_reset_day,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
                label="Ice reset",
            )

    axes[0].set_ylabel("Tile-mean ice thickness (nm)")
    axes[0].set_title("Ice trajectory")
    axes[1].set_ylabel("Mean tile accumulation rate (nm/day)")
    axes[1].set_title("Accumulation rate")
    for index, axis in enumerate(axes):
        axis.set_xlabel("Days since first exposure" if index == len(axes) - 1 else "")
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.2)
    title = "WFI01–WFI18 median trajectories with 16th–84th percentile bands"
    if independent_fit:
        title = "WFI01–WFI18 independent-fit trajectories; bands span tiles"
    figure.suptitle(title, fontsize=14, y=0.985)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        frameon=False,
    )
    if time_annotations is not None:
        figure.legend(
            handles=event_legend_handles(time_annotations),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=8,
            frameon=False,
            fontsize=7,
        )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.90))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fit-suffix", default=DEFAULT_FIT_SUFFIX)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--reset-day",
        type=float,
        default=None,
        help="Mark the ice reset day on both panels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trajectories = {}
    independent_fit = args.fit_suffix == DEFAULT_FIT_SUFFIX
    tile_trajectories = {}
    for detector in SCAS:
        output_dir = detector_output_dir(args.root, detector, args.fit_suffix)
        posterior_path = output_dir / "posterior_samples.npz"
        independent_paths = (output_dir / "fits").glob("tile_*/posterior_samples.npz")
        if not posterior_path.is_file() and not any(independent_paths):
            raise FileNotFoundError(f"Missing production posterior: {posterior_path}")
        trajectories[detector] = load_trajectory(output_dir)
        if independent_fit:
            tile_trajectories[detector] = load_independent_tile_thickness(output_dir)
    statistics_path = args.root / "wfi01_tile_log_statistics_512_count_corrected.csv"
    time_annotations = load_time_annotations(statistics_path)
    output = args.output or (
        args.root / DEFAULT_FIT_SUFFIX / "wfi_all_detector_ice_rate_trajectories.png"
    )
    quantile_path = output.with_name(output.stem + "_quantiles.csv")
    plot_all_detector_trajectories(
        output,
        trajectories,
        reset_day=args.reset_day,
        independent_fit=independent_fit,
        time_annotations=time_annotations,
    )
    detrended, pre_rate, post_rate = detrend_ice_trajectories(
        trajectories,
        reset_day=args.reset_day if args.reset_day is not None else DEFAULT_RESET_DAY,
    )
    detrended_output = output.with_name("wfi_all_detector_detrended_ice_thickness.png")
    plot_detrended_ice_thickness(
        detrended_output,
        detrended,
        pre_rate,
        post_rate,
        reset_day=args.reset_day if args.reset_day is not None else DEFAULT_RESET_DAY,
        time_annotations=time_annotations,
    )
    fractional_output = output.with_name(
        "wfi_all_detector_detrended_ice_thickness_fraction.png"
    )
    plot_detrended_ice_thickness(
        fractional_output,
        fractional_detrended_ice_trajectories(detrended, trajectories),
        pre_rate,
        post_rate,
        reset_day=args.reset_day if args.reset_day is not None else DEFAULT_RESET_DAY,
        time_annotations=time_annotations,
        fractional=True,
    )
    if tile_trajectories:
        reset_day = (
            args.reset_day
            if args.reset_day is not None
            else DEFAULT_RESET_DAY
        )
        tile_detrended = detrend_tile_trajectories(tile_trajectories, reset_day)
        tile_output = output.with_name("wfi_all_tile_detrended_ice_thickness.png")
        tile_fraction_output = output.with_name(
            "wfi_all_tile_detrended_ice_thickness_fraction.png"
        )
        plot_all_tile_detrended_ice_thickness_stacked(
            tile_output,
            tile_detrended,
            fractional_tile_trajectories(tile_detrended, tile_trajectories),
            reset_day=reset_day,
            time_annotations=time_annotations,
            alias_path=tile_fraction_output,
        )
    write_quantiles(quantile_path, trajectories)
    print(f"Wrote {output}")
    print(f"Wrote {detrended_output}")
    print(f"Wrote {fractional_output}")
    if tile_trajectories:
        print(f"Wrote {tile_output}")
        print(f"Wrote {tile_fraction_output}")
    print(f"Wrote {quantile_path}")


if __name__ == "__main__":
    main()
