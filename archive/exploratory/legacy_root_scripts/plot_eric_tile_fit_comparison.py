#!/usr/bin/env python3
"""Plot MAP ice trajectories and accumulation rates for local versus Eric fits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DETECTORS = ("wfi01", "wfi10")
BRANCHES = ("local", "eric_calibrated")
COLORS = {"local": "#d95f02", "eric_calibrated": "#1b9e77"}
LABELS = {"local": "Local corrected", "eric_calibrated": "Eric calibrated"}


def scalar(samples, name):
    return float(np.asarray(samples[name], dtype=float).reshape(-1)[0])


def input_path(input_root, detector, branch):
    return (
        Path(input_root)
        / detector
        / branch
        / f"{detector}_tile_log_statistics_256_count_corrected.csv"
    )


def fit_path(input_root, detector, branch, fit_subdir):
    return (
        Path(input_root)
        / detector
        / branch
        / fit_subdir
        / detector
        / "fits"
        / "tile_000"
    )


def load_days(path):
    by_repeat = {}
    means = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["repeat"]), int(row["led"]))
            n = float(row["n_pair_valid"])
            means[key] = float(row["sum_log_change"]) / n
            by_repeat.setdefault(int(row["repeat"]), []).append(float(row["days"]))
    repeats = sorted(by_repeat)
    return np.asarray([np.mean(by_repeat[repeat]) for repeat in repeats]), means


def load_fit(input_root, detector, branch, fit_subdir):
    fit_dir = fit_path(input_root, detector, branch, fit_subdir)
    input_file = input_path(input_root, detector, branch)
    days, observed = load_days(input_file)
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        ice = np.asarray(samples["node_ice"], dtype=float).reshape(-1)
        rates = np.asarray(samples["node_rates"], dtype=float).reshape(-1)
        residuals = np.asarray(
            [
                float(row["residual_log"])
                for row in csv.DictReader(
                    (fit_dir / "tile_group_predictions.csv").open(newline="")
                )
            ],
            dtype=float,
        )
        values = {
            "days": days,
            "ice": ice,
            "rates": rates,
            "initial_rate": scalar(samples, "tile_initial_rate_nm_day"),
            "final_rate": scalar(samples, "tile_final_rate_nm_day"),
            "mean_rate": scalar(samples, "tile_mean_rate_nm_day"),
            "initial_ice": float(ice[0]),
            "final_ice": float(ice[-1]),
            "residual_rms_log": float(np.sqrt(np.mean(residuals * residuals))),
            "residual_max_log": float(np.max(np.abs(residuals))),
            "reset_day": scalar(samples, "ice_reset_day"),
            "observed": observed,
        }
    if values["days"].size != values["ice"].size:
        raise ValueError(f"Node-day/ice mismatch in {fit_dir}")
    if values["days"].size != values["rates"].size:
        raise ValueError(f"Node-day/rate mismatch in {fit_dir}")
    return values


def comparison_rows(input_roots, fit_subdir):
    rows = []
    nodes = []
    for tile_id, input_root in input_roots:
        for detector in DETECTORS:
            fits = {
                branch: load_fit(input_root, detector, branch, fit_subdir)
                for branch in BRANCHES
            }
            local = fits["local"]
            eric = fits["eric_calibrated"]
            ice_delta = eric["ice"] - local["ice"]
            rate_delta = eric["rates"] - local["rates"]
            observed_delta = np.asarray(
                [
                    eric["observed"][key] - local["observed"][key]
                    for key in sorted(local["observed"])
                ],
                dtype=float,
            )
            rows.append(
                {
                    "detector": detector.upper(),
                    "source_tile_id": tile_id,
                    "node_count": len(local["days"]),
                    "local_initial_ice_nm": local["initial_ice"],
                    "eric_initial_ice_nm": eric["initial_ice"],
                    "local_final_ice_nm": local["final_ice"],
                    "eric_final_ice_nm": eric["final_ice"],
                    "ice_delta_endpoint_nm": ice_delta[-1],
                    "ice_delta_mae_nm": float(np.mean(np.abs(ice_delta))),
                    "ice_delta_rmse_nm": float(np.sqrt(np.mean(ice_delta * ice_delta))),
                    "ice_delta_max_nm": float(np.max(np.abs(ice_delta))),
                    "local_initial_rate_nm_day": local["initial_rate"],
                    "eric_initial_rate_nm_day": eric["initial_rate"],
                    "local_final_rate_nm_day": local["final_rate"],
                    "eric_final_rate_nm_day": eric["final_rate"],
                    "local_mean_rate_nm_day": local["mean_rate"],
                    "eric_mean_rate_nm_day": eric["mean_rate"],
                    "rate_delta_mae_nm_day": float(np.mean(np.abs(rate_delta))),
                    "rate_delta_rmse_nm_day": float(np.sqrt(np.mean(rate_delta * rate_delta))),
                    "rate_delta_max_nm_day": float(np.max(np.abs(rate_delta))),
                    "observed_log_delta_mae_percent": 100.0 * float(np.mean(np.abs(observed_delta))),
                    "observed_log_delta_max_percent": 100.0 * float(np.max(np.abs(observed_delta))),
                    "local_fit_residual_rms_percent": 100.0 * local["residual_rms_log"],
                    "eric_fit_residual_rms_percent": 100.0 * eric["residual_rms_log"],
                }
            )
            for index, day in enumerate(local["days"]):
                nodes.append(
                    {
                        "detector": detector.upper(),
                        "source_tile_id": tile_id,
                        "node": index,
                        "days": day,
                        "local_ice_nm": local["ice"][index],
                        "eric_ice_nm": eric["ice"][index],
                        "ice_delta_nm": ice_delta[index],
                        "local_rate_nm_day": local["rates"][index],
                        "eric_rate_nm_day": eric["rates"][index],
                        "rate_delta_nm_day": rate_delta[index],
                    }
                )
    return rows, nodes


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = tuple(rows[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def panel_title(row):
    return (
        f"{row['detector']} tile {row['source_tile_id']}  "
        f"RMSE={row['ice_delta_rmse_nm']:.3f} nm, "
        f"max={row['ice_delta_max_nm']:.3f} nm"
    )


def plot_quantity(path, node_rows, quantity, ylabel, delta_ylabel):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for axis, row_key in zip(axes.flat, sorted({(r["detector"], r["source_tile_id"]) for r in node_rows})):
        detector, tile_id = row_key
        current = [
            row
            for row in node_rows
            if row["detector"] == detector and row["source_tile_id"] == tile_id
        ]
        current.sort(key=lambda row: row["days"])
        days = np.asarray([row["days"] for row in current], dtype=float)
        local = np.asarray([row[f"local_{quantity}"] for row in current], dtype=float)
        eric = np.asarray([row[f"eric_{quantity}"] for row in current], dtype=float)
        delta = eric - local
        axis.plot(days, local, color=COLORS["local"], linewidth=1.4, label=LABELS["local"])
        axis.plot(days, eric, color=COLORS["eric_calibrated"], linewidth=1.4, label=LABELS["eric_calibrated"])
        reset_day = next(
            row["days"]
            for row in current
            if row["days"] >= 12.9
        )
        axis.axvline(reset_day, color="0.45", linestyle=":", linewidth=0.9)
        axis.set_title(
            f"{detector} tile {tile_id} | "
            f"RMSE {np.sqrt(np.mean(delta * delta)):.3g} {delta_ylabel.split('(')[-1].rstrip(')')}"
        )
        axis.set_xlabel("Days since first product")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_difference(path, node_rows, quantity, ylabel):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for axis, row_key in zip(axes.flat, sorted({(r["detector"], r["source_tile_id"]) for r in node_rows})):
        detector, tile_id = row_key
        current = [
            row
            for row in node_rows
            if row["detector"] == detector and row["source_tile_id"] == tile_id
        ]
        current.sort(key=lambda row: row["days"])
        days = [row["days"] for row in current]
        values = [row[f"{quantity}_delta_{'nm_day' if quantity == 'rate' else 'nm'}"] for row in current]
        axis.plot(days, values, color="#6a3d9a", linewidth=1.2)
        axis.axhline(0.0, color="0.2", linewidth=0.8)
        axis.set_title(f"{detector} tile {tile_id}")
        axis.set_xlabel("Days since first product")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-root-136", type=Path, required=True)
    parser.add_argument("--input-root-137", type=Path, required=True)
    parser.add_argument(
        "--fit-subdir",
        default="production/tile_size_256",
        help="Fit directory below each detector/branch (default: production/tile_size_256).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, nodes = comparison_rows(
        ((136, args.input_root_136), (137, args.input_root_137)),
        args.fit_subdir,
    )
    write_csv(args.output_dir / "fit_trajectory_metrics.csv", rows)
    write_csv(args.output_dir / "fit_trajectory_nodes.csv", nodes)
    plot_quantity(
        args.output_dir / "ice_trajectory_comparison.png",
        nodes,
        "ice_nm",
        "Ice thickness (nm)",
        "nm",
    )
    plot_quantity(
        args.output_dir / "accumulation_rate_comparison.png",
        nodes,
        "rate_nm_day",
        "Accumulation rate (nm/day)",
        "nm/day",
    )
    plot_difference(
        args.output_dir / "ice_trajectory_difference.png",
        nodes,
        "ice",
        "Eric - local ice (nm)",
    )
    plot_difference(
        args.output_dir / "accumulation_rate_difference.png",
        nodes,
        "rate",
        "Eric - local rate (nm/day)",
    )
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
