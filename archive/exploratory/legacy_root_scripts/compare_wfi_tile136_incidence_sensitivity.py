"""Measure fixed-MAP incidence-angle sensitivity for tile 136."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fit_wfi_tmm_production import (
    LOG_NOISE_SIGMA,
    build_prediction_function,
    load_response_context,
    load_tile_statistics,
)
from compare_wfi_tile136_spectral_weighting import FIT_FLAGS, load_map_values


ANGLES_DEG = (0.0, 5.0, 10.0, 15.0, 20.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", default=("wfi01", "wfi10"))
    return parser.parse_args()


def weighted_correlation(left, right, weights):
    left_centered = left - np.average(left, weights=weights)
    right_centered = right - np.average(right, weights=weights)
    denominator = np.sqrt(
        np.sum(weights * left_centered**2) * np.sum(weights * right_centered**2)
    )
    return float(np.sum(weights * left_centered * right_centered) / denominator) if denominator else 0.0


def evaluate_detector(statistics_path, map_path, output_dir):
    data = load_tile_statistics(statistics_path)
    context = load_response_context(data["sca_names"])
    values = load_map_values(Path(map_path) / "posterior_samples.npz")
    observed = data["sums"][0] / data["counts"][0]
    observed += values["post_reset"] * (
        np.asarray(data["days"]) >= float(data.get("reset_day"))
    )
    normal = values["saved_prediction"]
    residual = observed - normal
    weights = np.asarray(data["counts"][0], dtype=float) / LOG_NOISE_SIGMA**2
    detector = str(data["sca_names"][0]).lower()
    summary_rows = []
    group_rows = []
    prediction_modes = {
        "fixed_angle": lambda angle: build_prediction_function(
            data,
            context,
            incidence_pupil_half_angle_deg=0.0,
            **FIT_FLAGS,
        )(values["ice"], values["ar"], values["hgcdte"], angle, **_prediction_kwargs(values)),
        "pupil_cone": lambda angle: build_prediction_function(
            data,
            context,
            incidence_pupil_half_angle_deg=angle,
            **FIT_FLAGS,
        )(values["ice"], values["ar"], values["hgcdte"], 0.0, **_prediction_kwargs(values)),
    }
    for mode, evaluator in prediction_modes.items():
        for angle in ANGLES_DEG:
            prediction = np.asarray(evaluator(angle)).reshape(-1)
            delta = prediction - normal
            chi2_normal = float(np.sum(weights * residual**2))
            chi2_angle = float(np.sum(weights * (residual - delta) ** 2))
            summary_rows.append(
                {
                    "detector": detector,
                    "mode": mode,
                    "angle_deg": angle,
                    "weighted_residual_rms": float(np.sqrt(np.average(residual**2, weights=weights))),
                    "prediction_delta_weighted_rms": float(np.sqrt(np.average(delta**2, weights=weights))),
                    "prediction_delta_max": float(np.max(np.abs(delta))),
                    "weighted_correlation_with_residual": weighted_correlation(residual, delta, weights),
                    "chi2_normal": chi2_normal,
                    "chi2_angle_at_fixed_map": chi2_angle,
                    "delta_chi2_at_fixed_map": chi2_angle - chi2_normal,
                }
            )
            for group_index, day in enumerate(data["days"]):
                group_rows.append(
                    {
                        "detector": detector,
                        "mode": mode,
                        "angle_deg": angle,
                        "group": int(data["groups"][group_index]),
                        "days": float(day),
                        "led": int(data["led"][group_index]),
                        "delta_log_prediction": float(delta[group_index]),
                        "residual_log": float(residual[group_index]),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{detector}_tile136_incidence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / f"{detector}_tile136_incidence_groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_rows[0].keys())
        writer.writeheader()
        writer.writerows(group_rows)
    return summary_rows


def _prediction_kwargs(values):
    return {
        "ice_n_log_modulation": values["ice_n"],
        "ice_k_constant": values["ice_k"],
        "ar_n_log_modulation": values["ar_n"],
        "hgcdte_n_log_modulation": values["hgcdte_n"],
        "hgcdte_k_log_modulation": values["hgcdte_k"],
        "led_log_offsets": values["led"],
    }


def write_plot(summary_rows, output_dir):
    detectors = sorted({row["detector"] for row in summary_rows})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for detector in detectors:
        for mode, style in (("fixed_angle", "-"), ("pupil_cone", "--")):
            rows = [
                row
                for row in summary_rows
                if row["detector"] == detector and row["mode"] == mode
            ]
            rows.sort(key=lambda row: row["angle_deg"])
            angles = [row["angle_deg"] for row in rows]
            delta_rms = [row["prediction_delta_weighted_rms"] for row in rows]
            delta_chi2 = [row["delta_chi2_at_fixed_map"] for row in rows]
            label = f"{detector.upper()} {mode.replace('_', ' ')}"
            axes[0].plot(angles, delta_rms, marker="o", linestyle=style, label=label)
            axes[1].plot(angles, delta_chi2, marker="o", linestyle=style, label=label)
    axes[0].axhline(1.0e-3, color="black", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Angle or pupil half-angle (deg)")
    axes[0].set_ylabel("Prediction change RMS (log)")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Angle or pupil half-angle (deg)")
    axes[1].set_ylabel("Fixed-MAP delta chi-squared")
    axes[1].legend(fontsize=8)
    figure.savefig(output_dir / "tile136_incidence_sensitivity.png", dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    all_rows = []
    for detector in args.detectors:
        detector = detector.lower()
        statistics_path = args.statistics_dir / (
            f"{detector}_tile_log_statistics_256_tile_136_count_corrected.csv"
        )
        map_path = args.map_dir / f"{detector}_tile136"
        all_rows.extend(evaluate_detector(statistics_path, map_path, args.output_dir))
    with (args.output_dir / "tile136_incidence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    write_plot(all_rows, args.output_dir)
    for row in all_rows:
        print(
            f"{row['detector']} {row['mode']} {row['angle_deg']:.0f} deg: "
            f"delta_rms={row['prediction_delta_weighted_rms']:.7g}, "
            f"delta_chi2={row['delta_chi2_at_fixed_map']:.7g}"
        )


if __name__ == "__main__":
    main()
