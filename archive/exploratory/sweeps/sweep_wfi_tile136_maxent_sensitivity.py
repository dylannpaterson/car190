"""Sweep MaxEnt weight while reoptimizing the existing tile MAP model.

The constant-rate warmup is recomputed once per dataset/detector and its
constant rate is held as the entropy and node-rate reference for every weight.
All optical flags match the existing AR/ice-k diagnostic MAP archives.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fit",
        action="append",
        nargs=4,
        metavar=("DATASET", "DETECTOR", "STATISTICS", "MAP"),
        required=True,
    )
    parser.add_argument(
        "--entropy-weights",
        default="0,0.01,0.1,1",
        help="Comma-separated nonnegative MaxEnt weights.",
    )
    return parser.parse_args()


def _production_flags():
    return {
        "smoothness_weight": 0.0,
        "trajectory_parameterization": "rate",
        "initial_thickness_prior": production.INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        "initial_thickness_prior_sigma_nm": production.INITIAL_THICKNESS_HALF_NORMAL_SIGMA_NM,
        "trajectory_rate_prior_sigma_log": production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        "fit_incidence_angle": False,
        "fit_ice_n_modulation": True,
        "fit_ice_n_constant": False,
        "ice_n_modulation_prior_sigma_log": 0.01,
        "fit_ice_k_constant": True,
        "ice_k_prior_sigma": production.ICE_K_PRIOR_SIGMA,
        "fit_ar_k_constant": True,
        "ar_k_prior_sigma": production.AR_K_PRIOR_SIGMA,
        "fit_ar_hgcdte_modulation": True,
        "ar_hgcdte_modulation_prior_sigma_log": 0.10,
        "hgcdte_n_modulation_prior_sigma_log": 0.05,
        "fit_led_log_offsets": True,
        "led_log_offset_prior_sigma_log": production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
        "sed_weighting": "energy",
        "qe_response_mode": "none",
        "absolute_hgcdte_n_scale_prior_sigma_log": (
            production.ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG
        ),
    }


def _config(statistics_path, output_dir, **kwargs):
    values = _production_flags()
    values.update(kwargs)
    return production.ProductionConfig(
        tile_statistics=Path(statistics_path),
        output_dir=Path(output_dir),
        **values,
    )


def _fit_map_state_silently(config):
    with contextlib.redirect_stdout(io.StringIO()):
        return production.fit_map_state(config)


def _epoch_metrics(data, evaluated):
    observed = np.asarray(data["sums"][0] / data["counts"][0], dtype=float)
    post_mask = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    post_offset = float(np.asarray(evaluated["post_reset_tile_log_offset"]).reshape(-1)[0])
    observed = observed + post_offset * post_mask
    prediction = np.asarray(
        evaluated.get("prediction", evaluated["_map_least_squares_prediction"]),
        dtype=float,
    ).reshape(-1)
    residual = observed - prediction
    rates = np.asarray(evaluated["rates"], dtype=float).reshape(-1)
    metrics = {}
    for side, mask in (("pre", ~post_mask), ("post", post_mask)):
        metrics[f"{side}_rate_nm_day"] = float(np.mean(rates[mask]))
        metrics[f"{side}_group_rms_log"] = float(np.sqrt(np.mean(residual[mask] ** 2)))
        metrics[f"{side}_count_weighted_rms_log"] = float(
            np.sqrt(
                np.sum(data["counts"][0, mask] * residual[mask] ** 2)
                / np.sum(data["counts"][0, mask])
            )
        )
    metrics["rate_difference_pre_minus_post_nm_day"] = (
        metrics["pre_rate_nm_day"] - metrics["post_rate_nm_day"]
    )
    return metrics


def sweep_one(dataset, detector, statistics_path, map_path, weights, output_dir):
    statistics_path = Path(statistics_path)
    map_path = Path(map_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_dir = output_dir / f"{dataset}_{detector}_constant_rate_warmup"
    warmup_config = _config(
        statistics_path,
        warmup_dir,
        map_init_from=map_path,
        constant_rate=True,
        entropy_weight=0.0,
    )
    data, _, warm_values, warm_diagnostics, _ = _fit_map_state_silently(warmup_config)
    rate_reference = np.broadcast_to(
        np.asarray(warm_values["log_rate_scale"], dtype=float)[..., None],
        (
            int(data["n_scas"]),
            int(data["n_tiles"]),
            len(data["node_days"]),
        ),
    ).copy()
    initial_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in warm_values.items()
        if name != "log_rate_scale"
    }
    initial_values["np_log_rate_nodes"] = rate_reference.copy()
    rows = []
    for weight in weights:
        label = f"weight_{weight:g}".replace(".", "p")
        fit_dir = output_dir / f"{dataset}_{detector}_{label}"
        config = _config(
            statistics_path,
            fit_dir,
            map_initial_values=initial_values,
            entropy_reference_log_rates=rate_reference if weight else None,
            entropy_weight=float(weight),
            rate_prior_log_mean=rate_reference,
            constant_rate=False,
            model_metadata={
                "trajectory_regularization": "maxent" if weight else "none",
                "trajectory_entropy_weight": float(weight),
                "trajectory_entropy_reference": "constant_rate_sensitivity_warmup",
                "constant_rate_map_warmup": 1,
                "constant_rate_map_rate_nm_day": float(
                    np.exp(np.asarray(warm_values["log_rate_scale"]).reshape(-1)[0])
                ),
            },
        )
        fit_data, _, map_values, diagnostics, evaluated = _fit_map_state_silently(config)
        metrics = _epoch_metrics(fit_data, evaluated)
        np.savez_compressed(
            fit_dir.with_suffix(".npz"),
            **{name: np.asarray(value) for name, value in {**map_values, **evaluated}.items()},
            objective=np.asarray(diagnostics["objective"]),
            entropy_weight=np.asarray(weight),
        )
        rows.append(
            {
                "dataset": dataset,
                "detector": detector,
                "entropy_weight": float(weight),
                "warmup_rate_nm_day": float(
                    np.exp(np.asarray(warm_values["log_rate_scale"]).reshape(-1)[0])
                ),
                "objective": float(diagnostics["objective"]),
                "group_rms_log": float(np.sqrt(np.mean((
                    np.asarray(fit_data["sums"][0] / fit_data["counts"][0])
                    + np.asarray(evaluated["post_reset_tile_log_offset"]).reshape(-1)[0]
                    * (np.asarray(fit_data["days"]) >= float(fit_data["reset_day"]))
                    - np.asarray(
                        evaluated.get(
                            "prediction", evaluated["_map_least_squares_prediction"]
                        )
                    ).reshape(-1)
                ) ** 2))),
                "map_initial_thickness_nm": float(
                    np.exp(np.asarray(map_values["initial_thickness_log_nm"]).reshape(-1)[0])
                ),
                "map_reset_initial_thickness_nm": float(
                    np.exp(
                        np.asarray(map_values["reset_initial_thickness_log_nm"]).reshape(-1)[0]
                    )
                ),
                **metrics,
            }
        )
    return rows


def main():
    args = parse_args()
    weights = [float(value) for value in args.entropy_weights.split(",")]
    if any(value < 0.0 or not np.isfinite(value) for value in weights):
        raise ValueError("Entropy weights must be finite and nonnegative")
    rows = []
    for dataset, detector, statistics_path, map_path in args.fit:
        rows.extend(
            sweep_one(
                dataset,
                detector,
                statistics_path,
                map_path,
                weights,
                args.output_dir,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "maxent_sensitivity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['dataset']} {row['detector']} weight={row['entropy_weight']:g}: "
            f"pre={row['pre_rate_nm_day']:.8g}, "
            f"post={row['post_rate_nm_day']:.8g}, "
            f"delta={row['rate_difference_pre_minus_post_nm_day']:.8g}"
        )


if __name__ == "__main__":
    main()
