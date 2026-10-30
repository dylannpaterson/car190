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
    latest_observation_label,
    load_time_annotations,
    split_time_series_at_resets,
    split_time_series_at_reset,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    corrected_statistics_path,
    detector_output_root as production_detector_output_root,
    positive_integer,
    production_output_root,
    tile_directory_entries,
)


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_FIT_SUFFIX = "production"
DEFAULT_RESET_DAY = 12.907397928240743


def available_detector_label(trajectories):
    names = tuple(trajectories)
    return f"{len(names)} detectors"


def _one_tile_series(samples, key):
    values = np.asarray(samples[key], dtype=float)
    if values.ndim == 3:
        values = values[0, 0]
    elif values.ndim == 2:
        values = values[0]
    return values.reshape(-1)


def detector_output_dir(
    root,
    detector,
    fit_suffix,
    tile_size=DEFAULT_TILE_SIZE,
    fit_subdir=None,
):
    root = Path(root)
    if fit_suffix == DEFAULT_FIT_SUFFIX:
        output_dir = production_detector_output_root(root, detector, tile_size)
    else:
        output_dir = root / f"{detector.lower()}_{fit_suffix}"
    return output_dir if fit_subdir is None else output_dir / fit_subdir


def independent_fit_dirs(output_dir):
    entries = tile_directory_entries(Path(output_dir) / "fits")
    if not entries:
        raise FileNotFoundError(f"No independent tile fits found in {output_dir}")
    complete_tile_ids(
        (tile_id for tile_id, _path in entries),
        label=f"independent tile fit IDs in {output_dir}",
    )
    return [path for _tile_id, path in entries]


def load_independent_trajectory(output_dir):
    fit_dirs = independent_fit_dirs(output_dir)
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
    fit_dirs = independent_fit_dirs(output_dir)
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
    for detector in tile_trajectories:
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
    for detector in detrended:
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
    for detector in trajectories:
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
    for detector in trajectories:
        detector_days, ice, _ = trajectories[detector]
        detrended[detector] = (detector_days, ice - trend)
    return detrended, float(pre_trend[0]), float(post_trend[0])


def fractional_detrended_ice_trajectories(detrended, trajectories):
    fractional = {}
    for detector in detrended:
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
    observation_label="unknown",
):
    figure, axis = plt.subplots(figsize=(15, 6.5))
    colors = plt.get_cmap("tab20", len(trajectories))
    plot_reset_day = reset_day
    if plot_reset_day is None and time_annotations is not None:
        plot_reset_day = time_annotations.physical_reset_day
    legend_handles = []
    for index, detector in enumerate(trajectories):
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
            label="Decon2",
        )
    axis.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.6)
    axis.set_ylabel(
        "Detrended ice thickness / current ice thickness"
        if fractional
        else "Detrended tile-mean ice thickness (nm)"
    )
    axis.set_xlabel("Days since first exposure")
    title = "Detrended ice thickness fraction" if fractional else "Detrended ice thickness"
    if observation_label != "unknown":
        title = f"{title}\n{observation_label}"
    axis.set_title(title)
    axis.set_xlim(left=0.0)
    axis.grid(alpha=0.2)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
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
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.90))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_all_tile_axis(
    axis,
    trajectories,
    reset_day=None,
    time_annotations=None,
    fractional=False,
    observation_label="unknown",
):
    colors = plt.get_cmap("tab20", len(trajectories))
    plot_reset_day = reset_day
    if plot_reset_day is None and time_annotations is not None:
        plot_reset_day = time_annotations.physical_reset_day
    legend_handles = []
    for index, detector in enumerate(trajectories):
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
            label="Decon2",
        )
    axis.axhline(0.0, color="0.35", linewidth=0.7, alpha=0.6)
    axis.set_ylabel(
        "Per-tile detrended ice thickness / current ice thickness"
        if fractional
        else "Per-tile detrended ice thickness (nm)"
    )
    axis.set_xlabel("Days since first exposure")
    title = (
        "Per-tile detrended ice thickness fraction"
        if fractional
        else "Per-tile detrended ice thickness"
    )
    if observation_label != "unknown":
        title = f"{title}\n{observation_label}"
    axis.set_title(title)
    axis.set_xlim(left=0.0)
    axis.grid(alpha=0.2)
    return legend_handles


def _add_all_tile_legends(figure, legend_handles, time_annotations):
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
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
    observation_label="unknown",
):
    figure, axis = plt.subplots(figsize=(15, 6.5))
    legend_handles = _plot_all_tile_axis(
        axis,
        trajectories,
        reset_day=reset_day,
        time_annotations=time_annotations,
        fractional=fractional,
        observation_label=observation_label,
    )
    _add_all_tile_legends(figure, legend_handles, time_annotations)
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.90))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_all_tile_detrended_ice_thickness_stacked(
    path,
    detrended,
    fractional,
    reset_day=None,
    time_annotations=None,
    alias_path=None,
    observation_label="unknown",
):
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    legend_handles = _plot_all_tile_axis(
        axes[0],
        detrended,
        reset_day=reset_day,
        time_annotations=time_annotations,
        fractional=False,
        observation_label=observation_label,
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
    figure.tight_layout(rect=(0.0, 0.15, 1.0, 0.90))
    output_paths = (path,) if alias_path is None else (path, alias_path)
    for output_path in output_paths:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_detector_trajectories(
    path,
    trajectories,
    reset_day=None,
    break_days=None,
    independent_fit=False,
    time_annotations=None,
    observation_label="unknown",
    xlim_days=None,
):
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    colors = plt.get_cmap("tab20", len(trajectories))
    plot_reset_day = reset_day
    if plot_reset_day is None and time_annotations is not None:
        plot_reset_day = time_annotations.physical_reset_day
    legend_handles = []
    visible_ice = []
    visible_rates = []
    split_days = [] if plot_reset_day is None else [plot_reset_day]
    split_days.extend(break_days or ())
    for index, detector in enumerate(trajectories):
        days, ice, rates = trajectories[detector]
        (
            days,
            ice_low,
            ice_median,
            ice_high,
            rate_low,
            rate_median,
            rate_high,
        ) = split_time_series_at_resets(
            days,
            ice[0],
            ice[1],
            ice[2],
            rates[0],
            rates[1],
            rates[2],
            reset_days=split_days or None,
        )
        color = colors(index)
        axes[0].fill_between(days, ice_low, ice_high, color=color, alpha=0.08)
        axes[0].plot(days, ice_median, color=color, linewidth=1.4)
        axes[1].fill_between(days, rate_low, rate_high, color=color, alpha=0.08)
        axes[1].plot(days, rate_median, color=color, linewidth=1.4)
        if xlim_days is not None:
            visible = (
                np.isfinite(days)
                & (days >= float(xlim_days[0]))
                & (days <= float(xlim_days[1]))
            )
            visible_ice.extend((ice_low[visible], ice_high[visible]))
            visible_rates.extend((rate_low[visible], rate_high[visible]))
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
                label="Decon2",
            )

    axes[0].set_ylabel("Tile-mean ice thickness (nm)")
    axes[0].set_title("Ice trajectory")
    axes[1].set_ylabel("Mean tile accumulation rate (nm/day)")
    axes[1].set_title("Accumulation rate")
    for index, axis in enumerate(axes):
        axis.set_xlabel("Days since first exposure" if index == len(axes) - 1 else "")
        if xlim_days is None:
            axis.set_xlim(left=0.0)
        else:
            axis.set_xlim(*xlim_days)
        axis.grid(alpha=0.2)
    if xlim_days is not None:
        for axis, values in ((axes[0], visible_ice), (axes[1], visible_rates)):
            finite = np.concatenate(values)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lower = float(np.min(finite))
                upper = float(np.max(finite))
                padding = max((upper - lower) * 0.06, 0.1)
                axis.set_ylim(lower - padding, upper + padding)
    if xlim_days is None:
        axes[1].set_ylim(top=30.0)
    title = "Ice trajectories and accumulation rates"
    if observation_label != "unknown":
        title = f"{title}\n{observation_label}"
    figure.suptitle(title, fontsize=13, y=0.985)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
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
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.89))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        default=None,
        help="Restrict the plot to the selected detector(s); may be repeated.",
    )
    parser.add_argument(
        "--detector-output-root",
        type=Path,
        default=None,
        help="Direct directory containing detector subdirectories (wfi01, wfi02, ...).",
    )
    parser.add_argument("--fit-suffix", default=DEFAULT_FIT_SUFFIX)
    parser.add_argument(
        "--fit-subdir",
        type=Path,
        default=None,
        help="Read fits from a subdirectory below each detector output root.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile edge length used by the production reduction; default: {DEFAULT_TILE_SIZE}.",
    )
    parser.add_argument(
        "--reset-day",
        type=float,
        default=None,
        help="Mark the ice reset day on both panels.",
    )
    parser.add_argument(
        "--break-day",
        dest="break_days",
        action="append",
        type=float,
        default=None,
        help="Break trajectory lines at this day; may be repeated.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Plot completed detectors while the remaining production fits are running.",
    )
    parser.add_argument(
        "--last-days",
        type=float,
        default=None,
        help="Restrict the main trajectory plot to this many days at the end of the series.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = tuple(detector.upper() for detector in (args.detectors or SCAS))
    unknown = set(detectors).difference(SCAS)
    if unknown:
        raise ValueError(f"Unknown detectors: {sorted(unknown)}")
    trajectories = {}
    independent_fit = args.fit_suffix == DEFAULT_FIT_SUFFIX
    tile_trajectories = {}
    for detector in detectors:
        if args.detector_output_root is not None:
            output_dir = Path(args.detector_output_root) / detector.lower()
        else:
            output_dir = detector_output_dir(
                args.root,
                detector,
                args.fit_suffix,
                args.tile_size,
                args.fit_subdir,
            )
        posterior_path = output_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            try:
                fit_dirs = independent_fit_dirs(output_dir)
            except (FileNotFoundError, ValueError):
                if args.allow_partial:
                    continue
                raise FileNotFoundError(f"Missing production posterior: {posterior_path}")
            if not all((fit_dir / "posterior_samples.npz").is_file() for fit_dir in fit_dirs):
                if args.allow_partial:
                    continue
                raise FileNotFoundError(f"Incomplete production posterior set: {output_dir}")
        trajectories[detector] = load_trajectory(output_dir)
        if independent_fit:
            tile_trajectories[detector] = load_independent_tile_thickness(output_dir)
    if not trajectories:
        raise FileNotFoundError("No complete production detector trajectories are available")
    if args.last_days is not None and args.last_days <= 0.0:
        raise ValueError("last-days must be positive")
    statistics_path = corrected_statistics_path(args.root, detectors[0], args.tile_size)
    time_annotations = load_time_annotations(statistics_path)
    observation_label = latest_observation_label(args.root, args.tile_size)
    output = args.output or (
        production_output_root(args.root, args.tile_size)
        / (args.fit_subdir if args.fit_subdir is not None else Path())
        / "wfi_all_detector_ice_rate_trajectories.png"
    )
    quantile_path = output.with_name(output.stem + "_quantiles.csv")
    xlim_days = None
    if args.last_days is not None:
        latest_day = max(float(trajectory[0][-1]) for trajectory in trajectories.values())
        xlim_days = (latest_day - float(args.last_days), latest_day)
    plot_all_detector_trajectories(
        output,
        trajectories,
        reset_day=args.reset_day,
        break_days=args.break_days,
        independent_fit=independent_fit,
        time_annotations=time_annotations,
        observation_label=observation_label,
        xlim_days=xlim_days,
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
        observation_label=observation_label,
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
        observation_label=observation_label,
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
            observation_label=observation_label,
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
