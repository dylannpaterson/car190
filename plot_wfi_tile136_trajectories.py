"""Plot tile-136 fitted ice trajectories and instantaneous rates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production


RESET_DAY = 12.907397928240743
DETECTORS = ("wfi01", "wfi10")
VARIANTS = (
    ("baseline", "Production baseline", "#4d4d4d"),
    ("mod002", "AR/HgCdTe, sigma=0.02", "#0072b2"),
    ("mod010", "AR/HgCdTe, sigma=0.10", "#d55e00"),
    ("all025", "All optical, sigma=0.25", "#009e73"),
    ("all050", "All optical, sigma=0.50", "#cc79a7"),
    ("ice001", "Ice n=0.01; AR/HgCdTe=0.25", "#e69f00"),
    (
        "ice001_hn5",
        "Ice n=0.01; half-normal d0 sigma=5 nm",
        "#56b4e9",
    ),
    (
        "iceconstant_hn5",
        "Constant ice n; angle fixed; half-normal d0 sigma=5 nm",
        "#332288",
    ),
    (
        "iceconstant_hn5_freeangle",
        "Constant ice n; angle fitted; half-normal d0 sigma=5 nm",
        "#117733",
    ),
    (
        "iceconstant005_fixedangle",
        "Constant ice n; prior sigma=0.05; angle fixed",
        "#aa4499",
    ),
)


def variant_directory(root, detector, variant):
    if variant == "baseline":
        return root / detector / "n_prior_sigma_0p05" / "fits" / "tile_136"
    paths = {
        "mod002": "tile_136_hgcdte_modulation_matched_np_nprior005",
        "mod010": "tile_136_hgcdte_modulation_matched_np_nprior005_modsigma010",
        "all025": "tile_136_all_optical_modulation_prior025",
        "all050": "tile_136_all_optical_modulation_prior050",
        "ice001": "tile_136_all_optical_iceprior001_arhgcdteprior025",
        "ice001_hn5": (
            "tile_136_all_optical_iceprior001_arhgcdteprior025_initialhalfnormal5_correct"
        ),
        "iceconstant005_fixedangle": (
            "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_sig005_fixedangle"
        ),
    }
    if variant == "iceconstant_hn5":
        paths[variant] = (
            "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fixedangle"
        )
    if variant == "iceconstant_hn5_freeangle":
        paths[variant] = {
            "wfi01": "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fromconstant",
            "wfi10": "tile_136_all_optical_iceconstant_arhgcdteprior025_initialhalfnormal5_fromemcee",
        }[detector]
    return root / detector / "diagnostics" / paths[variant]


def statistics_path(root, detector):
    return (
        root
        / detector
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )


def load_series(root, detector, variant):
    directory = variant_directory(root, detector, variant)
    data = production.load_tile_statistics(statistics_path(root, detector))
    with np.load(directory / "posterior_samples.npz", allow_pickle=False) as archive:
        thickness_key = (
            "node_ice_thickness" if "node_ice_thickness" in archive else "node_ice"
        )
        rate_key = (
            "node_instantaneous_rate_nm_day"
            if "node_instantaneous_rate_nm_day" in archive
            else "node_rates"
        )
        thickness = np.asarray(archive[thickness_key], dtype=float).reshape(-1)
        rates = np.asarray(archive[rate_key], dtype=float).reshape(-1)
        mean_rate = float(np.asarray(archive["tile_mean_rate_nm_day"]).reshape(-1)[0])
    days = np.asarray(data["node_days"], dtype=float)
    if thickness.shape != days.shape or rates.shape != days.shape:
        raise ValueError(
            f"Trajectory shape mismatch for {detector} {variant}: "
            f"days={days.shape}, thickness={thickness.shape}, rates={rates.shape}"
        )
    return days, thickness, rates, mean_rate


def write_csv(series, path):
    fields = [
        "detector",
        "variant",
        "node",
        "days",
        "ice_thickness_nm",
        "instantaneous_rate_nm_day",
        "mean_rate_nm_day",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector in DETECTORS:
            for variant, _, _ in VARIANTS:
                days, thickness, rates, mean_rate = series[detector][variant]
                for node, (day, ice, rate) in enumerate(zip(days, thickness, rates)):
                    writer.writerow(
                        {
                            "detector": detector.upper(),
                            "variant": variant,
                            "node": node,
                            "days": float(day),
                            "ice_thickness_nm": float(ice),
                            "instantaneous_rate_nm_day": float(rate),
                            "mean_rate_nm_day": mean_rate,
                        }
                    )


def plot_series(series, output):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17, 10),
        sharex="col",
        constrained_layout=True,
    )
    for detector_index, detector in enumerate(DETECTORS):
        thickness_axis = axes[detector_index, 0]
        rate_axis = axes[detector_index, 1]
        for variant, label, color in VARIANTS:
            days, thickness, rates, _ = series[detector][variant]
            thickness_axis.plot(
                days,
                thickness,
                color=color,
                linewidth=1.25,
                label=label,
            )
            rate_axis.plot(days, rates, color=color, linewidth=1.25)
        for axis in (thickness_axis, rate_axis):
            axis.axvline(RESET_DAY, color="0.45", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.2)
        thickness_axis.set_ylabel(f"{detector.upper()}\nice thickness [nm]")
        rate_axis.set_ylabel(f"{detector.upper()}\nrate [nm/day]")
        if detector_index == len(DETECTORS) - 1:
            thickness_axis.set_xlabel("Days since first exposure")
            rate_axis.set_xlabel("Days since first exposure")
    axes[0, 0].set_title("Fitted ice trajectory")
    axes[0, 1].set_title("Instantaneous accumulation rate")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle("WFI tile 136: fitted trajectories and accumulation rates", y=1.02)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = {
        detector: {
            variant: load_series(args.root, detector, variant)
            for variant, _, _ in VARIANTS
        }
        for detector in DETECTORS
    }
    write_csv(series, args.output_dir / "tile136_trajectory_rate_summary.csv")
    plot_series(series, args.output_dir / "tile136_trajectories_and_rates.png")
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
