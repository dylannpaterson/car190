"""Compare trajectories from two embedded production MAP bundles."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
SUMMARY_FIELDS = (
    "tile_id",
    "tile_row",
    "tile_col",
    "old_node_count",
    "current_node_count",
    "overlap_days",
    "rate_correlation",
    "rate_rmse_nm_day",
    "rate_bias_current_minus_old_nm_day",
    "old_initial_rate_nm_day",
    "current_initial_rate_nm_day",
    "old_overlap_end_rate_nm_day",
    "current_overlap_end_rate_nm_day",
    "old_initial_ice_nm",
    "current_initial_ice_nm",
    "old_overlap_end_ice_nm",
    "current_overlap_end_ice_nm",
    "ice_bias_current_minus_old_nm",
    "ice_rmse_nm",
)
HISTOGRAM_QUANTILES = (0.5, 1.0, 5.0, 16.0, 50.0, 84.0, 95.0, 99.0, 99.5)
HISTOGRAM_FIELDS = (
    "model",
    "phase",
    "quantity",
    "count",
    *(f"q{str(quantile).replace('.', 'p')}" for quantile in HISTOGRAM_QUANTILES),
)
SIGMA_CLIP_FIELDS = (
    "model",
    "phase",
    "sigma",
    "count_raw",
    "count_kept",
    "count_rejected",
    "rejected_fraction",
    "median_raw",
    "mad_std_raw",
    "raw_min",
    "raw_max",
    "kept_min",
    "kept_max",
)
EXCLUSION_FIELDS = (
    "model",
    "detector",
    "tile_id",
    "phase",
    "node_index",
    "day_since_first_exposure",
    "rate_nm_day",
    "reason",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--current-statistics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", choices=DETECTORS, default="WFI01")
    parser.add_argument(
        "--all-detectors",
        action="store_true",
        help="Plot one rate-envelope subplot for every detector.",
    )
    parser.add_argument(
        "--rate-sigma",
        type=float,
        default=3.0,
        help="Robust sigma threshold for the all-detector rate diagnostic.",
    )
    return parser.parse_args()


def text_value(value):
    return str(value).strip()


def is_maxent_archive(samples):
    try:
        return (
            text_value(np.asarray(samples["trajectory_regularization"]).item())
            == "maxent"
            and np.isclose(
                float(np.asarray(samples["trajectory_entropy_weight"]).item()),
                0.1,
                rtol=0.0,
                atol=1.0e-12,
            )
            and int(np.asarray(samples["constant_rate_map_warmup"]).item()) == 1
        )
    except (KeyError, OSError, ValueError):
        return False


def maxent_node_days(samples):
    days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
    repeats = np.asarray(samples["observation_repeat"], dtype=int).reshape(-1)
    return np.asarray(
        [np.mean(days[repeats == repeat]) for repeat in sorted(set(repeats.tolist()))],
        dtype=float,
    )


def load_maxent_detector(statistics_dir, detector):
    fit_root = Path(statistics_dir) / "production" / detector.lower() / "fits"
    tiles = {}
    run_dates = set()
    reset_days = []
    for posterior_path in sorted(fit_root.glob("tile_*/posterior_samples.npz")):
        with np.load(posterior_path, allow_pickle=False) as samples:
            if not is_maxent_archive(samples):
                continue
            tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
            days = maxent_node_days(samples)
            rates = np.asarray(
                samples["node_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
            ice = np.asarray(samples["np_node_thickness"], dtype=float).reshape(-1)
            if tile_ids.size != 1 or rates.size != days.size or ice.size != days.size:
                raise ValueError(f"Unexpected MaxEnt trajectory shape in {posterior_path}")
            reset_days.append(float(np.asarray(samples["ice_reset_day"]).item()))
            run_dates.add(
                datetime.fromtimestamp(
                    posterior_path.stat().st_mtime, timezone.utc
                ).date().isoformat()
            )
            tiles[int(tile_ids[0])] = {
                "tile_row": int(np.asarray(samples["tile_row"]).item())
                if "tile_row" in samples
                else 0,
                "tile_col": int(np.asarray(samples["tile_col"]).item())
                if "tile_col" in samples
                else 0,
                "days": days,
                "ice": ice,
                "rate": rates,
            }
    if not tiles:
        raise ValueError(f"No completed MaxEnt archives found below {fit_root}")
    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-12):
        raise ValueError(f"MaxEnt archives use different reset days for {detector}")
    if len(run_dates) != 1:
        raise ValueError(f"MaxEnt archives use multiple run dates for {detector}: {run_dates}")
    return {
        "path": str(fit_root),
        "created": next(iter(run_dates)),
        "git_revision": "",
        "model": {
            "trajectory_regularization": "maxent",
            "trajectory_entropy_weight": "0.1",
            "ice_reset_day": str(reset_days[0]),
            "fit_layouts": "MaxEnt per-tile posterior archives",
        },
        "tiles": tiles,
    }


def load_completed_maxent_bundles(statistics_dir):
    bundles = {}
    for detector in DETECTORS:
        fit_root = Path(statistics_dir) / "production" / detector.lower() / "fits"
        if not any(fit_root.glob("tile_*/posterior_samples.npz")):
            continue
        try:
            bundles[detector] = load_maxent_detector(statistics_dir, detector)
        except ValueError as error:
            if "No completed MaxEnt archives" not in str(error):
                raise
    if not bundles:
        raise ValueError(f"No completed MaxEnt detector archives found below {statistics_dir}")
    return bundles


def load_bundle(path, detector):
    with fits.open(path, memmap=True) as hdul:
        model = {
            text_value(row["key"]): text_value(row["value"])
            for row in hdul["MODEL"].data
        }
        primary = hdul[0].header
        nodes = hdul["NODE_TRAJECTORIES"].data
        nodes = nodes[nodes["detector"] == detector]
        tiles = {}
        for tile_id in sorted(set(int(value) for value in nodes["tile_id"])):
            tile_nodes = nodes[nodes["tile_id"] == tile_id]
            order = np.argsort(tile_nodes["node_day_since_first_exposure"])
            tile_nodes = tile_nodes[order]
            tiles[tile_id] = {
                "tile_row": int(tile_nodes["tile_row"][0]),
                "tile_col": int(tile_nodes["tile_col"][0]),
                "days": np.asarray(tile_nodes["node_day_since_first_exposure"], float),
                "ice": np.asarray(tile_nodes["ice_thickness_nm"], float),
                "rate": np.asarray(tile_nodes["instantaneous_rate_nm_day"], float),
            }
        return {
            "path": str(path),
            "created": text_value(primary.get("CREATED", "")),
            "git_revision": text_value(primary.get("GITREV", "")),
            "model": model,
            "tiles": tiles,
        }


def correlation(left, right):
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compare_tiles(old, current):
    node_rows = []
    summary_rows = []
    rate_difference_matrix = []
    ice_difference_matrix = []
    old_days_reference = None
    for tile_id in sorted(old["tiles"]):
        old_tile = old["tiles"][tile_id]
        current_tile = current["tiles"][tile_id]
        old_days = old_tile["days"]
        if old_days_reference is None:
            old_days_reference = old_days
        current_days = current_tile["days"]
        if old_days[0] < current_days[0] or old_days[-1] > current_days[-1]:
            raise ValueError(f"Trajectory overlap is incomplete for tile {tile_id}")
        current_rate = np.interp(old_days, current_days, current_tile["rate"])
        current_ice = np.interp(old_days, current_days, current_tile["ice"])
        rate_difference = current_rate - old_tile["rate"]
        ice_difference = current_ice - old_tile["ice"]
        rate_difference_matrix.append(rate_difference)
        ice_difference_matrix.append(ice_difference)
        for node_index, values in enumerate(
            zip(old_days, old_tile["ice"], current_ice, old_tile["rate"], current_rate)
        ):
            day, old_ice, current_ice_value, old_rate, current_rate_value = values
            node_rows.append(
                {
                    "tile_id": tile_id,
                    "node_index": node_index,
                    "day_since_first_exposure": day,
                    "old_ice_nm": old_ice,
                    "current_ice_nm_interpolated": current_ice_value,
                    "ice_difference_current_minus_old_nm": current_ice_value - old_ice,
                    "old_rate_nm_day": old_rate,
                    "current_rate_nm_day_interpolated": current_rate_value,
                    "rate_difference_current_minus_old_nm_day": current_rate_value - old_rate,
                }
            )
        summary_rows.append(
            {
                "tile_id": tile_id,
                "tile_row": old_tile["tile_row"],
                "tile_col": old_tile["tile_col"],
                "old_node_count": len(old_days),
                "current_node_count": len(current_days),
                "overlap_days": old_days[-1] - old_days[0],
                "rate_correlation": correlation(old_tile["rate"], current_rate),
                "rate_rmse_nm_day": np.sqrt(np.mean(rate_difference**2)),
                "rate_bias_current_minus_old_nm_day": np.mean(rate_difference),
                "old_initial_rate_nm_day": old_tile["rate"][0],
                "current_initial_rate_nm_day": current_rate[0],
                "old_overlap_end_rate_nm_day": old_tile["rate"][-1],
                "current_overlap_end_rate_nm_day": current_rate[-1],
                "old_initial_ice_nm": old_tile["ice"][0],
                "current_initial_ice_nm": current_ice[0],
                "old_overlap_end_ice_nm": old_tile["ice"][-1],
                "current_overlap_end_ice_nm": current_ice[-1],
                "ice_bias_current_minus_old_nm": np.mean(ice_difference),
                "ice_rmse_nm": np.sqrt(np.mean(ice_difference**2)),
            }
        )
    return (
        summary_rows,
        node_rows,
        np.asarray(old_days_reference),
        np.asarray(rate_difference_matrix),
        np.asarray(ice_difference_matrix),
    )


def write_csv(rows, path, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_mean_trajectories(old, current, detector, output_path):
    old_label = old["created"][:10]
    current_label = current["created"][:10]
    old_days = old["tiles"][0]["days"]
    current_days = current["tiles"][0]["days"]
    old_rate = np.mean([tile["rate"] for tile in old["tiles"].values()], axis=0)
    current_rate = np.mean(
        [tile["rate"] for tile in current["tiles"].values()], axis=0
    )
    old_ice = np.mean([tile["ice"] for tile in old["tiles"].values()], axis=0)
    current_ice = np.mean(
        [tile["ice"] for tile in current["tiles"].values()], axis=0
    )
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False, constrained_layout=True)
    axes[0].plot(old_days, old_rate, color="#0072b2", label=old_label)
    axes[0].plot(current_days, current_rate, color="#d55e00", label=current_label)
    axes[0].axvline(float(old["model"]["ice_reset_day"]), color="#0072b2", linestyle="--", alpha=0.5)
    axes[0].axvline(float(current["model"]["ice_reset_day"]), color="#d55e00", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("mean instantaneous rate [nm/day]")
    axes[0].set_title(f"{detector} mean ice-rate trajectory")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(old_days, old_ice, color="#0072b2", label=old_label)
    axes[1].plot(current_days, current_ice, color="#d55e00", label=current_label)
    axes[1].set_xlabel("days since first exposure")
    axes[1].set_ylabel("mean ice thickness [nm]")
    axes[1].set_title(f"{detector} mean accumulated ice trajectory")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_tile_trajectories(old, current, detector, output_path):
    old_label = old["created"][:10]
    current_label = current["created"][:10]
    old_tiles = list(old["tiles"].values())
    current_tiles = list(current["tiles"].values())
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for tile in old_tiles:
        axes[0].plot(
            tile["days"],
            tile["rate"],
            color="#0072b2",
            alpha=0.16,
            linewidth=0.8,
        )
        axes[1].plot(
            tile["days"],
            tile["ice"],
            color="#0072b2",
            alpha=0.16,
            linewidth=0.8,
        )
    for tile in current_tiles:
        axes[0].plot(
            tile["days"],
            tile["rate"],
            color="#d55e00",
            alpha=0.16,
            linewidth=0.8,
        )
        axes[1].plot(
            tile["days"],
            tile["ice"],
            color="#d55e00",
            alpha=0.16,
            linewidth=0.8,
        )
    old_days = old_tiles[0]["days"]
    current_days = current_tiles[0]["days"]
    axes[0].plot(
        old_days,
        np.mean([tile["rate"] for tile in old_tiles], axis=0),
        color="#0072b2",
        linewidth=2.2,
        label=old_label,
    )
    axes[0].plot(
        current_days,
        np.mean([tile["rate"] for tile in current_tiles], axis=0),
        color="#d55e00",
        linewidth=2.2,
        label=current_label,
    )
    axes[1].plot(
        old_days,
        np.mean([tile["ice"] for tile in old_tiles], axis=0),
        color="#0072b2",
        linewidth=2.2,
        label=old_label,
    )
    axes[1].plot(
        current_days,
        np.mean([tile["ice"] for tile in current_tiles], axis=0),
        color="#d55e00",
        linewidth=2.2,
        label=current_label,
    )
    axes[0].set_title(f"{detector} all tile ice-rate trajectories")
    axes[0].set_ylabel("instantaneous rate [nm/day]")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(title="model run date", frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].set_title(f"{detector} all tile accumulated ice trajectories")
    axes[1].set_xlabel("days since first exposure")
    axes[1].set_ylabel("ice thickness [nm]")
    axes[1].legend(title="model run date", frameon=False)
    axes[1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_all_detector_envelopes(
    old_bundles,
    current_bundles,
    output_path,
    field,
    y_label,
    quantity_title,
):
    """Plot old/new detector medians with 16th/84th tile envelopes."""
    detectors = list(old_bundles)
    old_label = old_bundles[detectors[0]]["created"][:10]
    current_label = current_bundles[detectors[0]]["created"][:10]
    ncols = 6 if len(detectors) > 15 else 5
    nrows = (len(detectors) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.3 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for index, detector in enumerate(detectors):
        axis = axes[index]
        old_tiles = list(old_bundles[detector]["tiles"].values())
        current_tiles = list(current_bundles[detector]["tiles"].values())
        old_days = old_tiles[0]["days"]
        current_days = current_tiles[0]["days"]
        old_values = np.asarray([tile[field] for tile in old_tiles])
        current_values = np.asarray([tile[field] for tile in current_tiles])
        old_lower, old_median, old_upper = np.percentile(
            old_values, [16.0, 50.0, 84.0], axis=0
        )
        current_lower, current_median, current_upper = np.percentile(
            current_values, [16.0, 50.0, 84.0], axis=0
        )
        axis.fill_between(
            old_days,
            old_lower,
            old_upper,
            color="#0072b2",
            alpha=0.18,
        )
        axis.plot(
            old_days,
            old_median,
            color="#0072b2",
            linewidth=1.6,
            label=old_label,
        )
        axis.fill_between(
            current_days,
            current_lower,
            current_upper,
            color="#d55e00",
            alpha=0.18,
        )
        axis.plot(
            current_days,
            current_median,
            color="#d55e00",
            linewidth=1.6,
            label=current_label,
        )
        axis.axvline(
            float(old_bundles[detector]["model"]["ice_reset_day"]),
            color="#0072b2",
            linestyle=":",
            linewidth=0.7,
            alpha=0.65,
        )
        axis.axvline(
            float(current_bundles[detector]["model"]["ice_reset_day"]),
            color="#d55e00",
            linestyle=":",
            linewidth=0.7,
            alpha=0.65,
        )
        axis.set_title(f"{detector} (n={len(old_tiles)})", fontsize=10)
        if index % ncols == 0:
            axis.set_ylabel(y_label, fontsize=9)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=8)
        if index // ncols == nrows - 1:
            axis.set_xlabel("days", fontsize=9)
        if index == 0:
            axis.legend(title="model run date", frameon=False, fontsize=8)
    for axis in axes[len(detectors) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Old versus new WFI detector {quantity_title}\n"
        "Median with 16th/84th percentile tile envelopes; "
        f"model run dates: {old_label} and {current_label}"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_detector_rate_envelopes(old_bundles, current_bundles, output_path):
    _plot_all_detector_envelopes(
        old_bundles,
        current_bundles,
        output_path,
        field="rate",
        y_label="nm/day",
        quantity_title="rate trajectories",
    )


def plot_all_detector_thickness_envelopes(old_bundles, current_bundles, output_path):
    _plot_all_detector_envelopes(
        old_bundles,
        current_bundles,
        output_path,
        field="ice",
        y_label="nm",
        quantity_title="ice-thickness trajectories",
    )


def collect_trajectory_histogram_values(bundles):
    values = {
        phase: {quantity: [] for quantity in ("rate", "ice")}
        for phase in ("pre_reset", "post_reset")
    }
    for bundle in bundles.values():
        reset_day = float(bundle["model"]["ice_reset_day"])
        for tile in bundle["tiles"].values():
            days = tile["days"]
            phase_masks = {
                "pre_reset": days <= reset_day,
                "post_reset": days > reset_day,
            }
            for phase, mask in phase_masks.items():
                for quantity in ("rate", "ice"):
                    values[phase][quantity].extend(tile[quantity][mask].tolist())
    return {
        phase: {
            quantity: np.asarray(quantity_values, dtype=float)
            for quantity, quantity_values in phase_values.items()
        }
        for phase, phase_values in values.items()
    }


def trajectory_histogram_rows(model, values):
    rows = []
    for phase, phase_values in values.items():
        for quantity, quantity_values in phase_values.items():
            finite_values = quantity_values[np.isfinite(quantity_values)]
            quantiles = np.percentile(finite_values, HISTOGRAM_QUANTILES)
            row = {
                "model": model,
                "phase": phase,
                "quantity": quantity,
                "count": len(finite_values),
            }
            row.update(
                {
                    f"q{str(quantile).replace('.', 'p')}": value
                    for quantile, value in zip(HISTOGRAM_QUANTILES, quantiles)
                }
            )
            rows.append(row)
    return rows


def plot_all_detector_trajectory_histograms(
    old_bundles,
    current_bundles,
    output_path,
):
    """Plot pooled pre/post-reset old/new rate and thickness distributions."""
    old_values = collect_trajectory_histogram_values(old_bundles)
    current_values = collect_trajectory_histogram_values(current_bundles)
    old_label = old_bundles["WFI01"]["created"][:10]
    current_label = current_bundles["WFI01"]["created"][:10]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row_index, quantity in enumerate(("rate", "ice")):
        combined = np.concatenate(
            [
                old_values[phase][quantity]
                for phase in ("pre_reset", "post_reset")
            ]
            + [
                current_values[phase][quantity]
                for phase in ("pre_reset", "post_reset")
            ]
        )
        combined = combined[np.isfinite(combined)]
        bins = np.histogram_bin_edges(combined, bins=50)
        for column_index, phase in enumerate(("pre_reset", "post_reset")):
            axis = axes[row_index, column_index]
            axis.hist(
                old_values[phase][quantity],
                bins=bins,
                density=True,
                color="#0072b2",
                alpha=0.28,
                label=old_label,
            )
            axis.hist(
                current_values[phase][quantity],
                bins=bins,
                density=True,
                color="#d55e00",
                alpha=0.28,
                label=current_label,
            )
            axis.hist(
                old_values[phase][quantity],
                bins=bins,
                density=True,
                histtype="step",
                color="#0072b2",
                linewidth=1.5,
            )
            axis.hist(
                current_values[phase][quantity],
                bins=bins,
                density=True,
                histtype="step",
                color="#d55e00",
                linewidth=1.5,
            )
            axis.set_title(phase.replace("_", "-"))
            axis.set_xlim(left=0.0)
            axis.set_ylabel("density")
            axis.set_xlabel(
                "instantaneous rate (nm/day)"
                if quantity == "rate"
                else "ice thickness (nm)"
            )
            axis.grid(alpha=0.2)
            if row_index == 0 and column_index == 0:
                axis.legend(title="model run date", frameon=False)
    reset_days = sorted(
        {
            round(float(bundle["model"]["ice_reset_day"]), 6)
            for bundles in (old_bundles, current_bundles)
            for bundle in bundles.values()
        }
    )
    figure.suptitle(
        "Old versus new pooled ice-trajectory distributions\n"
        "Pre/post reset split at each model's reset day; "
        f"reset days: {', '.join(f'{day:g}' for day in reset_days)}"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return trajectory_histogram_rows(old_label, old_values) + trajectory_histogram_rows(
        current_label, current_values
    )


def sigma_clip_mask(values, sigma):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    clipped = sigma_clip(
        finite_values,
        sigma=sigma,
        cenfunc="median",
        stdfunc="mad_std",
        maxiters=None,
    )
    return finite_values, np.ma.getmaskarray(clipped)


def sigma_clip_rate_values(values, sigma):
    finite_values, rejected = sigma_clip_mask(values, sigma)
    keep = ~rejected
    return finite_values[keep], {
        "count_raw": len(finite_values),
        "count_kept": int(keep.sum()),
        "count_rejected": int((~keep).sum()),
        "rejected_fraction": float((~keep).mean()),
        "median_raw": float(np.median(finite_values)),
        "mad_std_raw": float(mad_std(finite_values)),
        "raw_min": float(np.min(finite_values)),
        "raw_max": float(np.max(finite_values)),
        "kept_min": float(np.min(finite_values[keep])),
        "kept_max": float(np.max(finite_values[keep])),
    }


def rate_sigma_clip_rows(model, values, sigma):
    rows = []
    for phase, phase_values in values.items():
        clipped_values, summary = sigma_clip_rate_values(
            phase_values["rate"], sigma
        )
        rows.append(
            {
                "model": model,
                "phase": phase,
                "sigma": sigma,
                **summary,
            }
        )
    return rows


def rate_sigma_clip_exclusion_masks(bundles, model, sigma):
    masks = {
        detector: {
            tile_id: np.zeros(len(tile["rate"]), dtype=bool)
            for tile_id, tile in bundle["tiles"].items()
        }
        for detector, bundle in bundles.items()
    }
    phase_values = {"pre_reset": [], "post_reset": []}
    phase_references = {"pre_reset": [], "post_reset": []}
    for detector, bundle in bundles.items():
        reset_day = float(bundle["model"]["ice_reset_day"])
        for tile_id, tile in bundle["tiles"].items():
            days = tile["days"]
            for phase, phase_mask in (
                ("pre_reset", days <= reset_day),
                ("post_reset", days > reset_day),
            ):
                for node_index in np.flatnonzero(phase_mask):
                    phase_values[phase].append(float(tile["rate"][node_index]))
                    phase_references[phase].append(
                        (
                            detector,
                            int(tile_id),
                            int(node_index),
                            float(days[node_index]),
                            float(tile["rate"][node_index]),
                        )
                    )
    rows = []
    for phase in ("pre_reset", "post_reset"):
        raw_values = np.asarray(phase_values[phase], dtype=float)
        finite_values, rejected = sigma_clip_mask(raw_values, sigma)
        kept_values = finite_values[~rejected]
        kept_min = float(np.min(kept_values))
        kept_max = float(np.max(kept_values))
        for reference, rate, is_rejected in zip(
            phase_references[phase], finite_values, rejected
        ):
            if not is_rejected:
                continue
            detector, tile_id, node_index, day, rate = reference
            if rate < kept_min:
                reason = "lower_tail"
            elif rate > kept_max:
                reason = "upper_tail"
            else:
                reason = "iterative_outlier"
            masks[detector][tile_id][node_index] = True
            rows.append(
                {
                    "model": model,
                    "detector": detector,
                    "tile_id": tile_id,
                    "phase": phase,
                    "node_index": node_index,
                    "day_since_first_exposure": day,
                    "rate_nm_day": rate,
                    "reason": reason,
                }
            )
    return masks, rows


def rate_sigma_clip_exclusion_masks_per_tile(bundles, model, sigma):
    masks = {
        detector: {
            tile_id: np.zeros(len(tile["rate"]), dtype=bool)
            for tile_id, tile in bundle["tiles"].items()
        }
        for detector, bundle in bundles.items()
    }
    rows = []
    for detector, bundle in bundles.items():
        reset_day = float(bundle["model"]["ice_reset_day"])
        for tile_id, tile in bundle["tiles"].items():
            days = tile["days"]
            for phase, phase_mask in (
                ("pre_reset", days <= reset_day),
                ("post_reset", days > reset_day),
            ):
                node_indices = np.flatnonzero(phase_mask)
                finite = np.isfinite(tile["rate"][node_indices])
                node_indices = node_indices[finite]
                raw_values, rejected = sigma_clip_mask(
                    tile["rate"][node_indices], sigma
                )
                if not len(raw_values):
                    continue
                kept_values = raw_values[~rejected]
                kept_min = float(np.min(kept_values))
                kept_max = float(np.max(kept_values))
                for node_index, rate, is_rejected in zip(
                    node_indices, raw_values, rejected
                ):
                    if not is_rejected:
                        continue
                    if rate < kept_min:
                        reason = "lower_tail"
                    elif rate > kept_max:
                        reason = "upper_tail"
                    else:
                        reason = "iterative_outlier"
                    masks[detector][tile_id][node_index] = True
                    rows.append(
                        {
                            "model": model,
                            "detector": detector,
                            "tile_id": int(tile_id),
                            "phase": phase,
                            "node_index": int(node_index),
                            "day_since_first_exposure": float(days[node_index]),
                            "rate_nm_day": float(rate),
                            "reason": reason,
                        }
                    )
    return masks, rows


def plot_sigma_clipped_rate_histograms(
    old_bundles,
    current_bundles,
    output_path,
    sigma,
):
    """Plot raw and robust-sigma-clipped pooled rate distributions."""
    old_values = collect_trajectory_histogram_values(old_bundles)
    current_values = collect_trajectory_histogram_values(current_bundles)
    old_label = old_bundles["WFI01"]["created"][:10]
    current_label = current_bundles["WFI01"]["created"][:10]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    rows = []
    for axis, phase in zip(axes, ("pre_reset", "post_reset")):
        old_clipped, old_summary = sigma_clip_rate_values(
            old_values[phase]["rate"], sigma
        )
        current_clipped, current_summary = sigma_clip_rate_values(
            current_values[phase]["rate"], sigma
        )
        rows.extend(
            [
                {"model": old_label, "phase": phase, "sigma": sigma, **old_summary},
                {
                    "model": current_label,
                    "phase": phase,
                    "sigma": sigma,
                    **current_summary,
                },
            ]
        )
        raw_combined = np.concatenate(
            [old_values[phase]["rate"], current_values[phase]["rate"]]
        )
        bins = np.histogram_bin_edges(raw_combined, bins=50)
        for label, color, raw, clipped in (
            (
                old_label,
                "#0072b2",
                old_values[phase]["rate"],
                old_clipped,
            ),
            (
                current_label,
                "#d55e00",
                current_values[phase]["rate"],
                current_clipped,
            ),
        ):
            axis.hist(
                raw,
                bins=bins,
                density=True,
                histtype="step",
                linestyle="--",
                color=color,
                alpha=0.55,
                label=f"{label} raw",
            )
            axis.hist(
                clipped,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=0.25,
                label=f"{label} clipped",
            )
            axis.hist(
                clipped,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.6,
            )
        axis.set_title(phase.replace("_", "-"))
        axis.set_xlabel("instantaneous accumulation rate (nm/day)")
        axis.set_ylabel("density")
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        axis.text(
            0.98,
            0.96,
            f"{old_label}: rejected {old_summary['count_rejected']:,} "
            f"({100.0 * old_summary['rejected_fraction']:.2f}%)\n"
            f"{current_label}: rejected {current_summary['count_rejected']:,} "
            f"({100.0 * current_summary['rejected_fraction']:.2f}%)",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    figure.suptitle(
        f"Sigma-clipped accumulation-rate distributions ({sigma:g} sigma)\n"
        "Dashed outlines are raw; filled distributions are retained after "
        "iterative median/MAD clipping"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return rows


def plot_sigma_clipped_rate_trajectories(
    old_bundles,
    current_bundles,
    output_path,
    sigma,
    per_tile=False,
):
    """Plot raw tile rate trajectories and mark points rejected by clipping."""
    old_label = old_bundles["WFI01"]["created"][:10]
    current_label = current_bundles["WFI01"]["created"][:10]
    exclusion_function = (
        rate_sigma_clip_exclusion_masks_per_tile
        if per_tile
        else rate_sigma_clip_exclusion_masks
    )
    old_masks, old_rows = exclusion_function(
        old_bundles, old_label, sigma
    )
    current_masks, current_rows = exclusion_function(
        current_bundles, current_label, sigma
    )
    detectors = list(old_bundles)
    ncols = 6 if len(detectors) > 15 else 5
    nrows = (len(detectors) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.3 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for index, detector in enumerate(detectors):
        axis = axes[index]
        old_detector_tiles = old_bundles[detector]["tiles"]
        current_detector_tiles = current_bundles[detector]["tiles"]
        for tile_id, tile in sorted(old_detector_tiles.items()):
            axis.plot(
                tile["days"],
                tile["rate"],
                color="#0072b2",
                alpha=0.13,
                linewidth=0.45,
            )
            rejected = old_masks[detector][tile_id]
            if np.any(rejected):
                axis.scatter(
                    tile["days"][rejected],
                    tile["rate"][rejected],
                    color="#0072b2",
                    marker="x",
                    s=13,
                    linewidths=0.7,
                )
        for tile_id, tile in sorted(current_detector_tiles.items()):
            axis.plot(
                tile["days"],
                tile["rate"],
                color="#d55e00",
                alpha=0.13,
                linewidth=0.45,
            )
            rejected = current_masks[detector][tile_id]
            if np.any(rejected):
                axis.scatter(
                    tile["days"][rejected],
                    tile["rate"][rejected],
                    color="#d55e00",
                    marker="x",
                    s=13,
                    linewidths=0.7,
                )
        axis.axvline(
            float(old_bundles[detector]["model"]["ice_reset_day"]),
            color="#0072b2",
            linestyle=":",
            linewidth=0.7,
            alpha=0.65,
        )
        axis.axvline(
            float(current_bundles[detector]["model"]["ice_reset_day"]),
            color="#d55e00",
            linestyle=":",
            linewidth=0.7,
            alpha=0.65,
        )
        old_count = sum(
            int(np.sum(mask)) for mask in old_masks[detector].values()
        )
        current_count = sum(
            int(np.sum(mask)) for mask in current_masks[detector].values()
        )
        axis.set_title(f"{detector} (n={len(old_detector_tiles)})", fontsize=10)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=8)
        axis.text(
            0.98,
            0.03,
            f"x old {old_count}; new {current_count}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
        if index // ncols == nrows - 1:
            axis.set_xlabel("days", fontsize=9)
        if index % ncols == 0:
            axis.set_ylabel("nm/day", fontsize=9)
        if index == 0:
            axis.plot([], [], color="#0072b2", label=f"{old_label} trajectories")
            axis.plot([], [], color="#d55e00", label=f"{current_label} trajectories")
            axis.plot(
                [],
                [],
                color="#0072b2",
                marker="x",
                linestyle="None",
                label=f"{old_label} excluded",
            )
            axis.plot(
                [],
                [],
                color="#d55e00",
                marker="x",
                linestyle="None",
                label=f"{current_label} excluded",
            )
            axis.legend(frameon=False, fontsize=8)
    for axis in axes[len(detectors) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Old/new accumulation-rate trajectories with {sigma:g}-sigma exclusions\n"
        f"Clipping scope: {'each detector/tile and phase' if per_tile else 'all tiles per model and phase'}; "
        "Thin lines are raw tile trajectories; x markers are rejected nodes"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return old_rows + current_rows


def plot_representative_tiles(old, current, detector, output_path):
    tile_ids = (0, 7, 32, 63)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, tile_id in zip(axes.flat, tile_ids):
        old_tile = old["tiles"][tile_id]
        current_tile = current["tiles"][tile_id]
        axis.plot(
            old_tile["days"],
            old_tile["rate"],
            color="#0072b2",
            label="prior rate",
        )
        axis.plot(
            current_tile["days"],
            current_tile["rate"],
            color="#d55e00",
            label="current rate",
        )
        axis.set_title(f"{detector} tile {tile_id:03d}: rate")
        axis.set_xlabel("days")
        axis.set_ylabel("nm/day")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_difference_heatmaps(days, rate_difference, ice_difference, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for axis, matrix, title, label in (
        (
            axes[0],
            rate_difference,
            "Current minus prior rate trajectory",
            "rate difference [nm/day]",
        ),
        (
            axes[1],
            ice_difference,
            "Current minus prior ice trajectory",
            "ice difference [nm]",
        ),
    ):
        limit = np.nanpercentile(np.abs(matrix), 98)
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            extent=(days[0], days[-1], 63.5, -0.5),
        )
        axis.set_ylabel("tile id")
        axis.set_title(title)
        axis.figure.colorbar(image, ax=axis, label=label)
    axes[1].set_xlabel("days since first exposure")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_statistics_dir = args.current_statistics_dir
    if args.all_detectors:
        current_bundles = load_completed_maxent_bundles(current_statistics_dir)
        old_bundles = {}
        for detector, current in current_bundles.items():
            old = load_bundle(args.old_bundle, detector)
            common_tiles = set(old["tiles"]) & set(current["tiles"])
            if not common_tiles:
                raise ValueError(f"Old bundle has no matching tiles for {detector}")
            old["tiles"] = {
                tile_id: old["tiles"][tile_id] for tile_id in sorted(common_tiles)
            }
            current["tiles"] = {
                tile_id: current["tiles"][tile_id]
                for tile_id in sorted(common_tiles)
            }
            old_bundles[detector] = old
        output_path = args.output_dir / "all_detector_rate_envelope_comparison.png"
        plot_all_detector_rate_envelopes(old_bundles, current_bundles, output_path)
        thickness_output_path = (
            args.output_dir / "all_detector_thickness_envelope_comparison.png"
        )
        plot_all_detector_thickness_envelopes(
            old_bundles,
            current_bundles,
            thickness_output_path,
        )
        histogram_output_path = (
            args.output_dir / "all_detector_trajectory_histograms.png"
        )
        histogram_rows = plot_all_detector_trajectory_histograms(
            old_bundles,
            current_bundles,
            histogram_output_path,
        )
        write_csv(
            histogram_rows,
            args.output_dir / "all_detector_trajectory_histogram_quantiles.csv",
            HISTOGRAM_FIELDS,
        )
        sigma_clip_output_path = (
            args.output_dir / "all_detector_rate_histograms_sigma_clipped.png"
        )
        sigma_clip_rows = plot_sigma_clipped_rate_histograms(
            old_bundles,
            current_bundles,
            sigma_clip_output_path,
            args.rate_sigma,
        )
        write_csv(
            sigma_clip_rows,
            args.output_dir / "all_detector_rate_sigma_clip_summary.csv",
            SIGMA_CLIP_FIELDS,
        )
        trajectory_sigma_clip_output_path = (
            args.output_dir / "all_detector_rate_trajectories_sigma_clipped.png"
        )
        exclusion_rows = plot_sigma_clipped_rate_trajectories(
            old_bundles,
            current_bundles,
            trajectory_sigma_clip_output_path,
            args.rate_sigma,
        )
        write_csv(
            exclusion_rows,
            args.output_dir / "all_detector_rate_excluded_nodes.csv",
            EXCLUSION_FIELDS,
        )
        per_tile_trajectory_output_path = (
            args.output_dir / "all_detector_rate_trajectories_per_tile_sigma_clipped.png"
        )
        per_tile_exclusion_rows = plot_sigma_clipped_rate_trajectories(
            old_bundles,
            current_bundles,
            per_tile_trajectory_output_path,
            args.rate_sigma,
            per_tile=True,
        )
        write_csv(
            per_tile_exclusion_rows,
            args.output_dir / "all_detector_rate_per_tile_excluded_nodes.csv",
            EXCLUSION_FIELDS,
        )
        metadata = {
            "detectors": list(current_bundles),
            "maxent_tile_count": sum(
                len(bundle["tiles"]) for bundle in current_bundles.values()
            ),
            "old_bundle": str(args.old_bundle),
            "old_bundle_created": old_bundles[next(iter(old_bundles))]["created"],
            "current_source": str(current_statistics_dir / "production"),
            "current_source_type": "completed per-tile MaxEnt posterior_samples.npz archives",
            "current_created": current_bundles[next(iter(current_bundles))]["created"],
            "comparison": "Old FITS bundle versus completed MaxEnt per-tile trajectories.",
            "rate_output": str(output_path),
            "thickness_output": str(thickness_output_path),
            "histogram_output": str(histogram_output_path),
            "histogram_quantiles": str(
                args.output_dir / "all_detector_trajectory_histogram_quantiles.csv"
            ),
            "rate_sigma": args.rate_sigma,
            "sigma_clip_output": str(sigma_clip_output_path),
            "sigma_clip_summary": str(
                args.output_dir / "all_detector_rate_sigma_clip_summary.csv"
            ),
            "sigma_clip_trajectory_output": str(trajectory_sigma_clip_output_path),
            "excluded_nodes": str(
                args.output_dir / "all_detector_rate_excluded_nodes.csv"
            ),
            "per_tile_sigma_clip_trajectory_output": str(
                per_tile_trajectory_output_path
            ),
            "per_tile_excluded_nodes": str(
                args.output_dir / "all_detector_rate_per_tile_excluded_nodes.csv"
            ),
        }
        (args.output_dir / "all_detector_rate_envelope_comparison_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(f"Wrote all-detector trajectory comparison to {args.output_dir}")
        return
    old = load_bundle(args.old_bundle, args.detector)
    current = load_maxent_detector(current_statistics_dir, args.detector)
    common_tiles = set(old["tiles"]) & set(current["tiles"])
    if not common_tiles:
        raise ValueError(f"Old bundle has no matching MaxEnt tiles for {args.detector}")
    old["tiles"] = {
        tile_id: old["tiles"][tile_id] for tile_id in sorted(common_tiles)
    }
    current["tiles"] = {
        tile_id: current["tiles"][tile_id] for tile_id in sorted(common_tiles)
    }
    summary_rows, node_rows, days, rate_difference, ice_difference = compare_tiles(
        old, current
    )
    write_csv(summary_rows, args.output_dir / "tile_trajectory_comparison.csv", SUMMARY_FIELDS)
    write_csv(
        node_rows,
        args.output_dir / "node_trajectory_comparison.csv",
        tuple(node_rows[0].keys()),
    )
    metadata = {
        "detector": args.detector,
        "old_bundle": old["path"],
        "old_bundle_created": old["created"],
        "old_bundle_git_revision": old["git_revision"],
        "old_bundle_model": old["model"],
        "current_bundle": current["path"],
        "current_bundle_created": current["created"],
        "current_bundle_git_revision": current["git_revision"],
        "current_bundle_model": current["model"],
        "comparison": "Current values linearly interpolated at each prior node day.",
    }
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    plot_mean_trajectories(
        old,
        current,
        args.detector,
        args.output_dir / "mean_trajectory_comparison.png",
    )
    plot_all_tile_trajectories(
        old,
        current,
        args.detector,
        args.output_dir / "all_tile_trajectory_comparison.png",
    )
    plot_representative_tiles(
        old,
        current,
        args.detector,
        args.output_dir / "representative_rate_trajectories.png",
    )
    plot_difference_heatmaps(
        days,
        rate_difference,
        ice_difference,
        args.output_dir / "trajectory_difference_heatmaps.png",
    )
    print(f"Wrote {args.detector} trajectory comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
