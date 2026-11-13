#!/usr/bin/env python3
"""Calculate constant-rate CIs and 0%-versus-200% count differences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


Z_95 = 1.959963984540054


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def scenario_path(output_dir, point_count, jump_percent):
    return (
        output_dir
        / f"points_{point_count}"
        / f"jump_{float(jump_percent):g}pct"
    )


def load_post_decon_variance(summary):
    statistics = Path(summary["statistics"])
    data = production.load_tile_statistics(statistics)
    baseline_path = (
        Path(summary["baseline_fit"]) / "posterior_samples.npz"
    )
    with np.load(baseline_path, allow_pickle=False) as archive:
        node_days = np.asarray(data["node_days"], dtype=float)
        rates = np.asarray(
            archive["node_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        reset_day = float(np.asarray(archive["ice_reset_day"]).reshape(-1)[0])
    post_rates = rates[node_days >= reset_day]
    if post_rates.size < 2:
        raise ValueError("Need at least two post-decon rate nodes")
    return {
        "reset_day": reset_day,
        "node_count": int(post_rates.size),
        "mean_nm_day": float(np.mean(post_rates)),
        "variance_nm2_day2": float(np.var(post_rates, ddof=1)),
        "sigma_nm_day": float(np.std(post_rates, ddof=1)),
    }


def calculate_ci(output_dir, summary, rate_variance):
    rows = []
    sigma = rate_variance["sigma_nm_day"]
    hours_per_point = float(summary["hours_between_synthetic_cycles"])
    for result in summary["results"]:
        point_count = int(result["post_jump_points"])
        fit_dir = scenario_path(
            output_dir, point_count, result["jump_percent"]
        ) / "fits" / "tile_136"
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as archive:
            node_rates = np.asarray(archive["node_rates"], dtype=float).reshape(-1)
        pre_rate = float(node_rates[-(point_count + 2)])
        post_rates = node_rates[-point_count:]
        post_mean = float(np.mean(post_rates))
        post_se = sigma / np.sqrt(point_count)
        difference = post_mean - pre_rate
        lower = difference - Z_95 * post_se
        upper = difference + Z_95 * post_se
        rows.append(
            {
                "post_jump_points": point_count,
                "hours_post_temperature_change": point_count * hours_per_point,
                "jump_percent": float(result["jump_percent"]),
                "pre_rate_nm_day": pre_rate,
                "post_mean_rate_nm_day": post_mean,
                "post_mean_se_nm_day": post_se,
                "difference_nm_day": difference,
                "z_from_constant_rate": difference / post_se,
                "difference_ci95_low_nm_day": lower,
                "difference_ci95_high_nm_day": upper,
                "difference_ci95_low_percent_of_pre": 100.0 * lower / pre_rate,
                "difference_ci95_high_percent_of_pre": 100.0 * upper / pre_rate,
                "consistent_with_constant_rate": bool(lower <= 0.0 <= upper),
            }
        )
    return rows


def calculate_count_differences(output_dir, summary):
    rows = []
    hours_per_point = float(summary["hours_between_synthetic_cycles"])
    for point_count in sorted({int(r["post_jump_points"]) for r in summary["results"]}):
        zero_path = scenario_path(output_dir, point_count, 0.0)
        two_path = scenario_path(output_dir, point_count, 200.0)
        zero_rows = {
            row["file_set_name"]: row
            for row in read_rows(
                zero_path / "wfi10_tile_log_statistics_256_tile_136_count_corrected.csv"
            )
            if row["file_set_name"].startswith("synthetic_rate_jump_")
        }
        two_rows = {
            row["file_set_name"]: row
            for row in read_rows(
                two_path / "wfi10_tile_log_statistics_256_tile_136_count_corrected.csv"
            )
            if row["file_set_name"].startswith("synthetic_rate_jump_")
        }
        for cycle in range(1, point_count + 1):
            for led in range(1, 7):
                name = f"synthetic_rate_jump_{cycle}_{led}"
                zero = zero_rows[name]
                two = two_rows[name]
                count = float(zero["n_pair_valid"])
                reference_mean = float(zero["sum_reference_counts"]) / count
                zero_log_change = float(zero["sum_log_change"]) / count
                two_log_change = float(two["sum_log_change"]) / count
                zero_current_mean = reference_mean * np.exp(zero_log_change)
                two_current_mean = reference_mean * np.exp(two_log_change)
                delta_mean = two_current_mean - zero_current_mean
                rows.append(
                    {
                        "post_jump_points": point_count,
                        "post_jump_cycle": cycle,
                        "hours_post_temperature_change": cycle * hours_per_point,
                        "led": led,
                        "reference_mean_count": reference_mean,
                        "zero_percent_predicted_current_mean_count": zero_current_mean,
                        "two_hundred_percent_predicted_current_mean_count": two_current_mean,
                        "delta_mean_count_200_minus_0": delta_mean,
                        "percent_difference_200_minus_0": (
                            100.0 * delta_mean / zero_current_mean
                        ),
                        "delta_sum_count_200_minus_0": count * delta_mean,
                    }
                )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output_dir / "rate_jump_summary.json"
    summary = json.loads(summary_path.read_text())
    rate_variance = load_post_decon_variance(summary)
    ci_rows = calculate_ci(args.output_dir, summary, rate_variance)
    count_rows = calculate_count_differences(args.output_dir, summary)
    analysis = {
        "post_decon_rate_variance": rate_variance,
        "ci_method": "post-jump constant-rate mean; pre-jump rate treated as fixed reference",
        "ci_z_95": Z_95,
        "ci_results": ci_rows,
    }
    (args.output_dir / "constant_rate_ci_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    write_csv(args.output_dir / "count_difference_200pct_minus_0pct.csv", count_rows)
    print(json.dumps(analysis, indent=2))
    print("count_difference_200pct_minus_0pct.csv")


if __name__ == "__main__":
    main()
