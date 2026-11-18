"""Compare uncompressed tile histories with H5-compatible rank-1/rank-2 fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from export_joint_ice_history_reference import deposition_factorizations_on_blocks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = (
    ROOT
    / "20261113/production/pipeline_runs/relative_response_24h_update/production/"
    "wavelength_response_bundle/intermediates/h5/joint_ice_history_reference.h5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "20261113/production/pipeline_runs/relative_response_24h_update/production/"
    "plots/relative_response/ice_history_rank_comparison"
)


def _error_metrics(reference, fitted):
    absolute = np.abs(np.asarray(fitted, dtype=float) - np.asarray(reference, dtype=float))
    return {
        "pointwise_absolute_error_nm_p50_p95_p99_max": np.percentile(
            absolute, [50, 95, 99, 100]
        ).tolist(),
        "per_tile_max_absolute_error_nm_p50_p95_p99_max": np.percentile(
            np.max(absolute, axis=1), [50, 95, 99, 100]
        ).tolist(),
        "per_epoch_absolute_error_nm_p50_p95_p99": np.percentile(
            absolute, [50, 95, 99], axis=0
        ).tolist(),
    }


def _plot_history_overview(path, days, episodes, truth, rank1, rank2, reset_days):
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {"truth": "black", "rank 1": "tab:orange", "rank 2": "tab:blue"}
    for label, history in (("truth", truth), ("rank 1", rank1), ("rank 2", rank2)):
        median = np.median(history, axis=0)
        low, high = np.percentile(history, [16, 84], axis=0)
        axes[0].plot(days, median, color=colors[label], linewidth=1.7, label=label)
        axes[0].fill_between(days, low, high, color=colors[label], alpha=0.08)
    for axis in axes:
        for reset_day in reset_days:
            axis.axvline(reset_day, color="0.5", linestyle=":", linewidth=0.9)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].set_title("Uncompressed histories and H5-compatible low-rank reconstructions")
    axes[0].legend(frameon=False)

    for label, history in (("rank 1", rank1), ("rank 2", rank2)):
        p50, p95 = np.percentile(np.abs(history - truth), [50, 95], axis=0)
        axes[1].plot(days, p50, color=colors[label], linewidth=1.5, label=f"{label} p50")
        axes[1].plot(
            days, p95, color=colors[label], linewidth=1.2, linestyle="--",
            label=f"{label} p95",
        )
    axes[1].set_xlabel("Days from history reference")
    axes[1].set_ylabel("Absolute thickness error (nm)")
    axes[1].set_title("Per-epoch reconstruction error across tiles")
    axes[1].legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_error_cdf(path, rank1_error, rank2_error):
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for label, errors, color in (
        ("rank 1", rank1_error, "tab:orange"),
        ("rank 2", rank2_error, "tab:blue"),
    ):
        values = np.sort(np.max(errors, axis=1))
        probability = np.arange(1, values.size + 1) / values.size
        axis.plot(values, probability, color=color, label=label)
    axis.set_xlabel("Per-tile maximum absolute thickness error (nm)")
    axis.set_ylabel("Fraction of tiles")
    axis.set_title("Distribution of worst-case tile-history errors")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_example_tiles(path, days, truth, rank1, rank2, detectors, tile_ids, errors, reset_days):
    worst_index = int(np.argmax(np.max(errors["rank1"], axis=1)))
    rank2_worst_index = int(np.argmax(np.max(errors["rank2"], axis=1)))
    selected = []
    for index in (worst_index, rank2_worst_index):
        if index not in selected:
            selected.append(index)
    figure, axes = plt.subplots(len(selected), 1, figsize=(11, 4.5 * len(selected)), squeeze=False)
    for axis, index in zip(axes[:, 0], selected, strict=True):
        axis.plot(days, truth[index], color="black", label="uncompressed", linewidth=1.5)
        axis.plot(days, rank1[index], color="tab:orange", label="rank 1", linewidth=1.2)
        axis.plot(days, rank2[index], color="tab:blue", label="rank 2", linewidth=1.2)
        for reset_day in reset_days:
            axis.axvline(reset_day, color="0.5", linestyle=":", linewidth=0.9)
        axis.set_title(
            f"{detectors[index]} tile {int(tile_ids[index])}: "
            f"rank-1 max error {np.max(errors['rank1'][index]):.2f} nm; "
            f"rank-2 max error {np.max(errors['rank2'][index]):.2f} nm"
        )
        axis.set_ylabel("Ice thickness (nm)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[-1, 0].set_xlabel("Days from history reference")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_latest_focal_plane(
    path, x_mm, y_mm, full_history, h5_rank1_history, day, timestamp
):
    full_latest = np.asarray(full_history[:, -1], dtype=float)
    rank1_latest = np.asarray(h5_rank1_history[:, -1], dtype=float)
    difference = rank1_latest - full_latest
    thickness_min = float(min(full_latest.min(), rank1_latest.min()))
    thickness_max = float(max(full_latest.max(), rank1_latest.max()))
    difference_limit = float(np.max(np.abs(difference)))
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    panels = (
        (full_latest, "Uncompressed full histories", "viridis", thickness_min, thickness_max),
        (rank1_latest, "H5 rank-1 deposition projection", "viridis", thickness_min, thickness_max),
        (difference, "H5 rank-1 minus full", "coolwarm", -difference_limit, difference_limit),
    )
    for axis, (values, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        image = axis.scatter(
            x_mm, y_mm, c=values, s=11, cmap=cmap, vmin=vmin, vmax=vmax,
            linewidths=0,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("MPA x (mm)")
        axis.set_ylabel("MPA y (mm)")
        axis.set_title(title)
        axis.grid(alpha=0.15)
        label = "Thickness (nm)" if title != "H5 rank-1 minus full" else "Difference (nm)"
        figure.colorbar(image, ax=axis, shrink=0.78, label=label)
    figure.suptitle(
        f"Saved H5 focal-plane ice pattern versus full histories at day {day:.3f}"
        + (f" ({timestamp})" if timestamp else "")
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(history_path, output_dir, spacing_days=2.0):
    history_path = Path(history_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    with h5py.File(history_path, "r") as history_file:
        observation = history_file["bundle_observation_axis"]
        days = np.asarray(observation["time_days_from_reference"], dtype=float)
        episodes = np.asarray(observation["episode"], dtype=np.int8)
        truth = np.asarray(observation["tile_thickness_nm"], dtype=float)
        stored_rates = np.asarray(observation["tile_rate_nm_per_day"], dtype=float)
        detectors = np.asarray(history_file["superpixels/detector"]).astype(str)
        tile_ids = np.asarray(history_file["superpixels/tile_id"], dtype=int)
        reset_days = np.asarray(
            history_file.attrs["reset_days_from_reference"], dtype=float
        ).reshape(-1)
        source_bundle_path = Path(str(history_file.attrs["source_bundle"]))
        timestamps = np.asarray(
            observation["timestamp_utc"].asstr()[()]
        ).astype(str)
        saved_rank1_tile_error = np.asarray(
            history_file.attrs[
                "rank1_projection_per_tile_max_error_nm_p50_p95_max"
            ],
            dtype=float,
        )
    if not source_bundle_path.is_file():
        raise FileNotFoundError(
            "History reference's source bundle is unavailable for focal-plane geometry: "
            f"{source_bundle_path}"
        )
    with np.load(source_bundle_path, allow_pickle=False) as source_bundle:
        source_detectors = np.asarray(source_bundle["detectors"]).astype(str)
        source_tile_ids = np.asarray(source_bundle["tile_ids"], dtype=int)
        x_mm = np.asarray(source_bundle["tile_center_x_mm"], dtype=float)
        y_mm = np.asarray(source_bundle["tile_center_y_mm"], dtype=float)
    if not (
        np.array_equal(source_detectors, detectors)
        and np.array_equal(source_tile_ids, tile_ids)
        and x_mm.shape == y_mm.shape == tile_ids.shape
    ):
        raise ValueError("Source-bundle tile geometry does not match full histories")
    if truth.shape != (len(detectors), days.size):
        raise ValueError("Full-history H5 arrays have inconsistent tile/time axes")

    factorization = deposition_factorizations_on_blocks(
        days,
        episodes,
        truth,
        spacing_days=float(spacing_days),
        multi_rank=2,
    )
    rank1 = np.asarray(factorization["rank1"]["fitted_thickness_nm"], dtype=float)
    rank2 = np.asarray(factorization["rankk"]["fitted_thickness_nm"], dtype=float)

    rank1_error = np.abs(rank1 - truth)
    rank2_error = np.abs(rank2 - truth)
    rank1_latest_error = np.abs(rank1[:, -1] - truth[:, -1])
    rank1_metrics = _error_metrics(truth, rank1)
    rank2_metrics = _error_metrics(truth, rank2)
    recomputed_rank1_tile_error = np.asarray(
        rank1_metrics["per_tile_max_absolute_error_nm_p50_p95_p99_max"],
        dtype=float,
    )[[0, 1, 3]]
    rank1_matches_saved_h5 = bool(
        np.allclose(
            recomputed_rank1_tile_error,
            saved_rank1_tile_error,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
    )
    improvement = rank1_error - rank2_error
    rank2_rates = (
        factorization["rankk"]["spatial_rate_coefficients"]
        @ factorization["rankk"]["temporal_rate_functions_nm_per_hour"]
    ) * 24.0
    negative_rank2_fraction = float(np.mean(rank2_rates < -1.0e-8))
    rank2_metrics["negative_block_rate_fraction"] = negative_rank2_fraction
    metrics = {
        "source_history": str(history_path),
        "tile_count": int(truth.shape[0]),
        "epoch_count": int(truth.shape[1]),
        "episode_count": int(np.unique(episodes).size),
        "reset_days_from_reference": reset_days.tolist(),
        "deposition_block_spacing_days": float(spacing_days),
        "uncompressed_history": "reference (zero reconstruction error)",
        "rank1_h5_projection_metrics": rank1_metrics,
        "rank2_rate_svd_metrics": {
            **rank2_metrics,
            "weighted_rate_variance_fraction": float(
                factorization["rankk"]["metrics"]["weighted_rate_variance_fraction"]
            ),
        },
        "rate_factorization_error_p50_p95_max": {
            "rank1_relative_error": factorization["rank1"]["metrics"][
                "rate_weighted_relative_error_p50_p95_max"
            ],
            "rank2_relative_error": factorization["rankk"]["metrics"][
                "rate_weighted_relative_error_p50_p95_max"
            ],
            "rank2_weighted_rate_variance_fraction": float(
                factorization["rankk"]["metrics"]["weighted_rate_variance_fraction"]
            ),
        },
        "rank2_vs_rank1_error_reduction_nm_p50_p95_p99": np.percentile(
            improvement, [50, 95, 99]
        ).tolist(),
        "rank2_block_rate_negative_fraction": negative_rank2_fraction,
        "latest_epoch_day": float(days[-1]),
        "latest_epoch_timestamp_utc": str(timestamps[-1]),
        "latest_epoch_rank1_abs_thickness_error_nm_p50_p95_p99_max": np.percentile(
            rank1_latest_error, [50, 95, 99, 100]
        ).tolist(),
        "saved_h5_rank1_per_tile_max_error_nm_p50_p95_max": saved_rank1_tile_error.tolist(),
        "recomputed_rank1_matches_saved_h5_metrics": rank1_matches_saved_h5,
        "source_rate_shape": list(stored_rates.shape),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rank1_rank2_per_tile_errors.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "detector", "tile_id",
                "rank1_abs_error_p50_nm", "rank1_abs_error_p95_nm", "rank1_abs_error_max_nm",
                "rank2_abs_error_p50_nm", "rank2_abs_error_p95_nm", "rank2_abs_error_max_nm",
                "max_error_reduction_nm",
            ),
        )
        writer.writeheader()
        for index, (detector, tile_id) in enumerate(zip(detectors, tile_ids, strict=True)):
            writer.writerow(
                {
                    "detector": detector,
                    "tile_id": int(tile_id),
                    "rank1_abs_error_p50_nm": float(np.percentile(rank1_error[index], 50)),
                    "rank1_abs_error_p95_nm": float(np.percentile(rank1_error[index], 95)),
                    "rank1_abs_error_max_nm": float(np.max(rank1_error[index])),
                    "rank2_abs_error_p50_nm": float(np.percentile(rank2_error[index], 50)),
                    "rank2_abs_error_p95_nm": float(np.percentile(rank2_error[index], 95)),
                    "rank2_abs_error_max_nm": float(np.max(rank2_error[index])),
                    "max_error_reduction_nm": float(
                        np.max(rank1_error[index]) - np.max(rank2_error[index])
                    ),
                }
            )
    focal_fields = (
        "detector", "tile_id", "full_thickness_nm", "h5_rank1_thickness_nm",
        "h5_rank1_minus_full_nm",
    )
    focal_csv = output_dir / "h5_rank1_vs_full_latest_epoch.csv"
    with focal_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=focal_fields)
        writer.writeheader()
        for index, (detector, tile_id) in enumerate(zip(detectors, tile_ids, strict=True)):
            writer.writerow(
                {
                    "detector": detector,
                    "tile_id": int(tile_id),
                    "full_thickness_nm": float(truth[index, -1]),
                    "h5_rank1_thickness_nm": float(rank1[index, -1]),
                    "h5_rank1_minus_full_nm": float(rank1[index, -1] - truth[index, -1]),
                }
            )
    _plot_history_overview(
        output_dir / "rank1_rank2_history_and_epoch_error.png",
        days, episodes, truth, rank1, rank2, reset_days,
    )
    _plot_error_cdf(
        output_dir / "rank1_rank2_tile_max_error_cdf.png", rank1_error, rank2_error
    )
    _plot_example_tiles(
        output_dir / "rank1_rank2_example_tiles.png",
        days, truth, rank1, rank2, detectors, tile_ids,
        {"rank1": rank1_error, "rank2": rank2_error}, reset_days,
    )
    _plot_latest_focal_plane(
        output_dir / "h5_rank1_vs_full_focal_plane_latest_epoch.png",
        x_mm,
        y_mm,
        truth,
        rank1,
        float(days[-1]),
        str(timestamps[-1]),
    )
    metrics["latest_epoch_focal_plane_csv"] = str(focal_csv)
    metrics["latest_epoch_focal_plane_plot"] = str(
        output_dir / "h5_rank1_vs_full_focal_plane_latest_epoch.png"
    )
    (output_dir / "rank1_rank2_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spacing-days", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = analyze(args.history, args.output_dir, spacing_days=args.spacing_days)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
