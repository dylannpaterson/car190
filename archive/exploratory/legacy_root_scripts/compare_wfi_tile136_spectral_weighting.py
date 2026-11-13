"""Compare energy, photon, and clean-QE-relative spectral weighting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fit_wfi_tmm_production import (
    LOG_NOISE_SIGMA,
    build_prediction_function,
    load_response_context,
    load_tile_statistics,
)


FIT_FLAGS = {
    "fit_ice_n_modulation": True,
    "fit_ice_k_constant": True,
    "fit_ar_hgcdte_modulation": True,
    "fit_led_log_offsets": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", default=("wfi01", "wfi10"))
    return parser.parse_args()


def squeeze_sample(samples, name):
    return np.asarray(samples[name], dtype=float).squeeze()


def load_map_values(path):
    with np.load(path, allow_pickle=False) as samples:
        values = {
            "ice": squeeze_sample(samples, "ice_thickness"),
            "ar": squeeze_sample(samples, "ar_parameters"),
            "hgcdte": np.asarray(
                [
                    squeeze_sample(samples, "absolute_hgcdte_n_log_scale"),
                    squeeze_sample(samples, "absolute_hgcdte_k_log_scale"),
                ]
            ),
            "ice_n": squeeze_sample(samples, "ice_n_log_modulation"),
            "ice_k": squeeze_sample(samples, "ice_k_constant"),
            "ar_n": squeeze_sample(samples, "ar_n_log_modulation")[None, ...],
            "hgcdte_n": squeeze_sample(samples, "hgcdte_n_log_modulation")[None, ...],
            "hgcdte_k": squeeze_sample(samples, "hgcdte_k_log_modulation")[None, ...],
            "ar_k": squeeze_sample(samples, "ar_k_constant"),
            "led": squeeze_sample(samples, "led_log_offsets")[None, ...],
            "saved_prediction": squeeze_sample(samples, "_map_least_squares_prediction"),
            "post_reset": squeeze_sample(samples, "post_reset_tile_log_offset"),
        }
    return values


def make_contexts(data, context):
    photon = load_response_context(data["sca_names"], sed_weighting="photon")
    clean_qe = load_response_context(
        data["sca_names"],
        sed_weighting="photon",
        qe_response_mode="clean_relative",
    )
    return {
        "energy": (context, False),
        "photon": (photon, False),
        "photon_clean_qe_relative": (clean_qe, True),
    }


def weighted_correlation(left, right, weights):
    left_mean = np.average(left, weights=weights)
    right_mean = np.average(right, weights=weights)
    left_centered = left - left_mean
    right_centered = right - right_mean
    denominator = np.sqrt(
        np.sum(weights * left_centered**2) * np.sum(weights * right_centered**2)
    )
    return float(np.sum(weights * left_centered * right_centered) / denominator) if denominator else 0.0


def evaluate_detector(statistics_path, map_path, output_dir):
    data = load_tile_statistics(statistics_path)
    context = load_response_context(data["sca_names"])
    values = load_map_values(map_path)
    contexts = make_contexts(data, context)
    predictors = {
        name: build_prediction_function(
            data,
            variant_context,
            normalize_clean_response=normalize_clean,
            **FIT_FLAGS,
        )
        for name, (variant_context, normalize_clean) in contexts.items()
    }
    kwargs = {
        "ice_n_log_modulation": values["ice_n"],
        "ice_k_constant": values["ice_k"],
        "ar_n_log_modulation": values["ar_n"],
        "hgcdte_n_log_modulation": values["hgcdte_n"],
        "hgcdte_k_log_modulation": values["hgcdte_k"],
        "led_log_offsets": values["led"],
    }
    predictions = {
        name: np.asarray(
            predictor(values["ice"], values["ar"], values["hgcdte"], 0.0, **kwargs)
        ).reshape(-1)
        for name, predictor in predictors.items()
    }
    predictions["energy"] = values["saved_prediction"]
    observed = data["sums"][0] / data["counts"][0]
    observed += values["post_reset"] * (
        np.asarray(data["days"]) >= float(data.get("reset_day"))
    )
    residual = observed - predictions["energy"]
    weights = np.asarray(data["counts"][0], dtype=float) / LOG_NOISE_SIGMA**2
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = str(data["sca_names"][0]).lower()
    rows = []
    summary = []
    for group_index, day in enumerate(data["days"]):
        row = {
            "detector": detector,
            "tile_id": int(data["tile_ids"][0]),
            "group": int(data["groups"][group_index]),
            "days": float(day),
            "led": int(data["led"][group_index]),
            "count": float(data["counts"][0, group_index]),
            "residual_energy": float(residual[group_index]),
        }
        for name, prediction in predictions.items():
            row[f"prediction_{name}"] = float(prediction[group_index])
            row[f"delta_{name}"] = float(prediction[group_index] - predictions["energy"][group_index])
        rows.append(row)

    with (output_dir / f"{detector}_tile136_spectral_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for name in ("photon", "photon_clean_qe_relative"):
        delta = predictions[name] - predictions["energy"]
        chi2_energy = float(np.sum(weights * residual**2))
        chi2_variant = float(np.sum(weights * (residual - delta) ** 2))
        summary.append(
            {
                "detector": detector,
                "variant": name,
                "weighted_delta_rms": float(np.sqrt(np.average(delta**2, weights=weights))),
                "weighted_delta_max": float(np.max(np.abs(delta))),
                "weighted_residual_rms": float(np.sqrt(np.average(residual**2, weights=weights))),
                "weighted_correlation": weighted_correlation(residual, delta, weights),
                "chi2_energy": chi2_energy,
                "chi2_variant_at_fixed_map": chi2_variant,
                "delta_chi2_at_fixed_map": chi2_variant - chi2_energy,
            }
        )

    summary_path = output_dir / f"{detector}_tile136_spectral_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main():
    args = parse_args()
    all_summary = []
    for detector in args.detectors:
        detector = detector.lower()
        statistics_path = args.statistics_dir / (
            f"{detector}_tile_log_statistics_256_tile_136_count_corrected.csv"
        )
        map_path = args.map_dir / f"{detector}_tile136/posterior_samples.npz"
        all_summary.extend(evaluate_detector(statistics_path, map_path, args.output_dir))
    with (args.output_dir / "tile136_spectral_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_summary[0].keys())
        writer.writeheader()
        writer.writerows(all_summary)
    for row in all_summary:
        print(
            f"{row['detector']} {row['variant']}: "
            f"corr={row['weighted_correlation']:.6g}, "
            f"delta_chi2={row['delta_chi2_at_fixed_map']:.6g}"
        )


if __name__ == "__main__":
    main()
