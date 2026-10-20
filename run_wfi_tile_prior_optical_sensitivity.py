"""Probe rate-prior, incidence-angle, and spectral-optics sensitivity for one tile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
from benchmark_wfi_fixed_optics import _load_reference_values
from plot_wfi_tile_rate_counterfactual import load_source_median_node_rates


DEFAULT_DETECTOR = "WFI01"
DEFAULT_SOURCE_DETECTOR = "WFI10"
DEFAULT_TILE_ID = 136
DEFAULT_TILE_SIZE = 256
DEFAULT_RESET_DAY = 12.907397928240743


def _scalar(values, name):
    return float(np.asarray(values[name], dtype=float).reshape(-1)[0])


def _load_context(path):
    data = production.load_tile_statistics(path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    return data, context


def _metrics(data, state, values, groups=None):
    observed = (
        np.asarray(data["sums"], dtype=float)
        / np.asarray(data["counts"], dtype=float)
    ).reshape(-1)
    prediction = np.asarray(state["prediction"], dtype=float).reshape(-1).copy()
    prediction -= _scalar(values, "post_reset_tile_log_offset") * (
        np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    )
    residual = observed - prediction
    valid = np.ones(residual.size, dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    if groups is not None:
        valid &= np.isin(
            np.asarray(data["groups"], dtype=int),
            np.asarray(groups, dtype=int),
        )
    weights = np.asarray(data["counts"], dtype=float).reshape(-1)[valid]
    residual = residual[valid]
    percent = 100.0 * np.expm1(residual)
    return {
        "n_non_reference_groups": int(residual.size),
        "weighted_rms_percent": float(
            np.sqrt(np.sum(weights * percent**2) / np.sum(weights))
        ),
        "max_abs_percent": float(np.max(np.abs(percent))),
        "data_objective": float(
            0.5
            * production.default_likelihood_weight(data["counts"])
            * np.sum(weights * residual**2)
            / production.LOG_NOISE_SIGMA**2
        ),
    }


def _fit_variant(
    data,
    context,
    full_data,
    full_context,
    base_values,
    rate_reference,
    rate_sigma,
    fit_incidence_angle,
    fit_spectral_optics,
    name,
    jax,
):
    initial = production.initial_values_for_data(
        data,
        fit_incidence_angle=fit_incidence_angle,
        fit_ar_hgcdte_modulation=fit_spectral_optics,
    )
    for parameter in (
        "ar_tile_raw",
        "initial_thickness_log_nm",
        "reset_initial_thickness_log_nm",
        "post_reset_tile_log_offset",
        "absolute_hgcdte_n_log_scale",
        "absolute_hgcdte_k_log_scale",
        "np_log_rate_nodes",
    ):
        initial[parameter] = np.asarray(base_values[parameter], dtype=float).copy()

    model_kwargs = {
        "smoothness_weight": 0.0,
        "entropy_reference_log_rates": rate_reference,
        "entropy_weight": 0.1,
        "rate_prior_log_mean": rate_reference,
        "trajectory_rate_prior_sigma_log": float(rate_sigma),
        "constant_rate": False,
        "fit_incidence_angle": fit_incidence_angle,
        "fit_ar_hgcdte_modulation": fit_spectral_optics,
    }
    model = production.build_model(data, context, **model_kwargs)
    values, diagnostics = production.run_least_squares_map_optimization(
        model,
        initial,
        jax,
        trajectory_rate_prior_sigma_log=float(rate_sigma),
    )
    state = model.production_state(values)
    metrics = _metrics(data, state, values)
    node_rates = np.asarray(state["node_rates"], dtype=float).reshape(-1)
    rate_reference_values = np.exp(np.asarray(rate_reference, dtype=float).reshape(-1))
    rate_log_residual = np.log(node_rates) - np.log(rate_reference_values)
    prior_residual = rate_log_residual / float(rate_sigma)
    result = {
        "name": name,
        "fit_incidence_angle": bool(fit_incidence_angle),
        "fit_spectral_optics": bool(fit_spectral_optics),
        "rate_prior_sigma_log": float(rate_sigma),
        "rate_prior_reference_nm_day": {
            "minimum": float(np.min(rate_reference_values)),
            "median": float(np.median(rate_reference_values)),
            "maximum": float(np.max(rate_reference_values)),
        },
        "rate_nodes_nm_day": {
            "minimum": float(np.min(node_rates)),
            "median": float(np.median(node_rates)),
            "maximum": float(np.max(node_rates)),
        },
        "rate_prior_residual_rms": float(np.sqrt(np.mean(prior_residual**2))),
        "rate_prior_residual_max_abs": float(np.max(np.abs(prior_residual))),
        "metrics": metrics,
        "map_objective": float(diagnostics["objective"]),
        "least_squares_residual_cost": float(
            0.5
            * np.sum(
                np.asarray(model.production_residuals(values), dtype=float) ** 2
            )
        ),
        "function_evaluations": int(diagnostics["function_evaluations"]),
        "converged": bool(diagnostics["converged"]),
        "gradient_converged": bool(diagnostics["gradient_converged"]),
        "scaled_gradient_norm": float(diagnostics["scaled_gradient_norm"]),
        "message": diagnostics["message"],
        "parameters": {
            "incidence_angle_deg": (
                _scalar(values, "incidence_angle_deg")
                if fit_incidence_angle
                else 0.0
            ),
            "initial_thickness_nm": float(
                np.exp(_scalar(values, "initial_thickness_log_nm"))
            ),
            "reset_initial_thickness_nm": float(
                np.exp(_scalar(values, "reset_initial_thickness_log_nm"))
            ),
            "post_reset_tile_log_offset": _scalar(
                values, "post_reset_tile_log_offset"
            ),
            "absolute_hgcdte_n_log_scale": _scalar(
                values, "absolute_hgcdte_n_log_scale"
            ),
            "absolute_hgcdte_k_log_scale": _scalar(
                values, "absolute_hgcdte_k_log_scale"
            ),
        }
    }
    full_rate_nodes = np.interp(
        np.asarray(full_data["node_days"], dtype=float),
        np.asarray(data["node_days"], dtype=float),
        node_rates,
    )
    full_rate_reference = np.interp(
        np.asarray(full_data["node_days"], dtype=float),
        np.asarray(data["node_days"], dtype=float),
        rate_reference_values,
    )
    full_values = {
        parameter: np.asarray(value, dtype=float).copy()
        for parameter, value in values.items()
    }
    full_values["np_log_rate_nodes"] = np.log(full_rate_nodes).reshape(
        np.asarray(full_values["np_log_rate_nodes"]).shape[:-1]
        + (full_data["node_days"].size,)
    )
    full_rate_prior = np.log(full_rate_reference).reshape(
        np.asarray(full_values["np_log_rate_nodes"]).shape
    )
    full_model = production.build_model(
        full_data,
        full_context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=full_rate_prior,
        entropy_weight=0.1,
        rate_prior_log_mean=full_rate_prior,
        trajectory_rate_prior_sigma_log=float(rate_sigma),
        constant_rate=False,
        fit_incidence_angle=fit_incidence_angle,
        fit_ar_hgcdte_modulation=fit_spectral_optics,
    )
    full_state = full_model.production_state(full_values)
    retained_groups = np.asarray(data["groups"], dtype=int)
    excluded_groups = np.setdiff1d(
        np.asarray(full_data["groups"], dtype=int),
        retained_groups,
    )
    result["full_grid_metrics"] = {
        "all_groups": _metrics(full_data, full_state, full_values),
        "retained_groups": _metrics(
            full_data,
            full_state,
            full_values,
            groups=retained_groups,
        ),
        "excluded_groups": _metrics(
            full_data,
            full_state,
            full_values,
            groups=excluded_groups,
        ),
    }
    if fit_spectral_optics:
        result["spectral_optics"] = {
            "ar_n_modulation_max_abs": float(
                np.max(np.abs(np.asarray(values["ar_n_log_modulation"])))
            ),
            "hgcdte_n_modulation_max_abs": float(
                np.max(np.abs(np.asarray(values["hgcdte_n_log_modulation"])))
            ),
            "hgcdte_k_modulation_max_abs": float(
                np.max(np.abs(np.asarray(values["hgcdte_k_log_modulation"])))
            ),
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--source-detector", default=DEFAULT_SOURCE_DETECTOR)
    parser.add_argument("--tile-id", type=int, default=DEFAULT_TILE_ID)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--filtered-statistics", type=Path, default=None)
    parser.add_argument("--free-archive", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = args.detector.lower()
    source_detector = args.source_detector.lower()
    detector_root = args.root / "production" / f"tile_size_{args.tile_size}" / detector
    tile_dir = detector_root / "fits" / f"tile_{args.tile_id:03d}"
    source_root = (
        args.root / "production" / f"tile_size_{args.tile_size}" / source_detector
    )
    source_statistics = detector_root / "tile_statistics" / (
        f"{detector}_tile_log_statistics_{args.tile_size}_tile_{args.tile_id:03d}.csv"
    )
    filtered_statistics = args.filtered_statistics or (
        tile_dir / "no_temperature_gain" / (
            f"{detector}_tile_log_statistics_{args.tile_size}_tile_{args.tile_id:03d}.csv"
        )
    )
    free_archive = args.free_archive or (
        tile_dir / "no_temperature_gain" / "posterior_samples.npz"
    )
    output = args.output or (
        detector_root / "diagnostics" / f"tile_{args.tile_id:03d}_prior_optical_sensitivity"
    )
    output.mkdir(parents=True, exist_ok=True)

    full_data, full_context = _load_context(source_statistics)
    data, context = _load_context(filtered_statistics)
    data["reset_day"] = float(full_data["reset_day"])
    full_data["reset_day"] = float(data["reset_day"])
    source_rates = load_source_median_node_rates(
        source_root,
        len(full_data["node_days"]),
    )
    source_rates = np.interp(
        np.asarray(data["node_days"], dtype=float),
        np.asarray(full_data["node_days"], dtype=float),
        source_rates,
    )
    with np.load(free_archive, allow_pickle=False) as archive:
        warmup_rate = float(np.asarray(archive["constant_rate_map_rate_nm_day"]).reshape(-1)[0])
    base_values = _load_reference_values(data, free_archive, filtered_statistics)
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)

    references = {
        "production_warmup": np.full_like(
            np.asarray(base_values["np_log_rate_nodes"], dtype=float),
            np.log(warmup_rate),
        ),
        "wfi10_median_trajectory": np.log(source_rates).reshape(
            np.asarray(base_values["np_log_rate_nodes"]).shape
        ),
    }
    variants = (
        ("production_prior_angle_free", "production_warmup", 1.5, True, False),
        ("production_prior_angle_zero", "production_warmup", 1.5, False, False),
        ("wfi10_prior_sigma_0p10_angle_zero", "wfi10_median_trajectory", 0.10, False, False),
        ("wfi10_prior_sigma_0p05_angle_zero", "wfi10_median_trajectory", 0.05, False, False),
        ("wfi10_prior_sigma_0p10_spectral_angle_zero", "wfi10_median_trajectory", 0.10, False, True),
        ("wfi10_prior_sigma_0p10_spectral_angle_free", "wfi10_median_trajectory", 0.10, True, True),
    )
    results = []
    for name, reference, sigma, fit_angle, spectral in variants:
        print(f"Running {name}", flush=True)
        results.append(
            _fit_variant(
                data,
                context,
                full_data,
                full_context,
                base_values,
                references[reference],
                sigma,
                fit_angle,
                spectral,
                name,
                jax,
            )
        )
    payload = {
        "detector": detector.upper(),
        "source_detector": source_detector.upper(),
        "tile_id": int(args.tile_id),
        "source_statistics": str(source_statistics.resolve()),
        "filtered_statistics": str(filtered_statistics.resolve()),
        "free_archive": str(free_archive.resolve()),
        "reset_day": float(data["reset_day"]),
        "production_warmup_rate_nm_day": warmup_rate,
        "wfi10_reference_rate_summary_nm_day": {
            "minimum": float(np.min(source_rates)),
            "median": float(np.median(source_rates)),
            "maximum": float(np.max(source_rates)),
        },
        "variants": results,
    }
    output_path = output / "sensitivity_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
