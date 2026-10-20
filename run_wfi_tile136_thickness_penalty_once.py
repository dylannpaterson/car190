"""Run one baseline-versus-thickness-penalty tile-136 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fit_wfi_tmm_production import (  # noqa: E402
    AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    AR_K_PRIOR_SIGMA,
    AR_MEAN_PRIOR_SIGMA,
    ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
    ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
    ICE_INDEX_DEFAULT,
    ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    INITIAL_THICKNESS_PRIOR_LOGNORMAL,
    LOG_NOISE_SIGMA,
    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    _evaluate_model,
    ar_active_parameter_indices,
    ar_prior_arrays,
    ar_reference_raw,
    build_model,
    default_likelihood_weight,
    initial_values_for_data,
    load_map_warm_start,
    load_response_context,
    load_tile_statistics,
    run_least_squares_map_optimization,
)
from fit_wfi_tile136_thickness_penalty import (  # noqa: E402
    _evaluate_projection,
    _penalty_plan,
    attach_penalty,
)

PRODUCTION_AR_MODULATION_PRIOR_SIGMA_LOG = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--start-values", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-shared", type=float, default=1.0)
    parser.add_argument("--smooth-sigma-nm", type=float, default=1.0)
    parser.add_argument("--rate-prior-mean-nm-day", type=float, default=None)
    return parser.parse_args()


def _rate_reference(map_dir, shape):
    with np.load(map_dir / "constant_rate_map.npz", allow_pickle=False) as archive:
        value = np.asarray(archive["log_rate_scale"], dtype=float)
    value = np.broadcast_to(value.reshape(-1, 1, 1), shape).copy()
    return value


def _model(data, context, rate_reference, rate_prior_mean_nm_day=None):
    rate_prior_log_mean = rate_reference
    if rate_prior_mean_nm_day is not None:
        if not np.isfinite(rate_prior_mean_nm_day) or rate_prior_mean_nm_day <= 0.0:
            raise ValueError("rate-prior-mean-nm-day must be positive and finite")
        rate_prior_log_mean = np.full_like(
            rate_reference,
            np.log(rate_prior_mean_nm_day),
        )
    return build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=0.1,
        rate_prior_log_mean=rate_prior_log_mean,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        trajectory_parameterization="rate",
        initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        absolute_hgcdte_n_scale_prior_sigma_log=0.05,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
        ice_n_constant_reference=ICE_INDEX_DEFAULT,
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=PRODUCTION_AR_MODULATION_PRIOR_SIGMA_LOG,
    )


def _load_start_values(path, template):
    with np.load(path, allow_pickle=False) as archive:
        values = {}
        for name, target in template.items():
            if name not in archive.files:
                raise ValueError(f"{path} is missing model value {name}")
            value = np.asarray(archive[name], dtype=float)
            while value.ndim > np.asarray(target).ndim and value.shape[0] == 1:
                value = value[0]
            if value.shape != np.asarray(target).shape:
                raise ValueError(
                    f"{path} value {name} has shape {value.shape}; "
                    f"expected {np.asarray(target).shape}"
                )
            values[name] = value.copy()
    return values


def _metrics(model, values, data, projection=None, augmented_objective=None):
    state = model.production_state(values)
    evaluated = _evaluate_model(model, values)
    original_residual = np.asarray(model.original_production_residuals(values), dtype=float)
    prediction = np.asarray(evaluated["_map_least_squares_prediction"], dtype=float).reshape(-1)
    counts = np.asarray(data["counts"], dtype=float).reshape(-1)
    observed = np.asarray(data["sums"], dtype=float).reshape(-1) / np.maximum(counts, 1.0)
    days = np.asarray(data["days"], dtype=float)
    post = days >= float(data["reset_day"])
    observed = observed + float(np.asarray(values["post_reset_tile_log_offset"]).reshape(-1)[0]) * post
    residual = prediction - observed
    within = np.asarray(data["sum_squared"], dtype=float).reshape(-1) - (
        np.asarray(data["sums"], dtype=float).reshape(-1) ** 2 / np.maximum(counts, 1.0)
    )
    likelihood_weight = default_likelihood_weight(data["counts"])
    data_nll = 0.5 * likelihood_weight / int(data["n_tiles"]) * np.sum(
        within + counts * residual**2
    ) / LOG_NOISE_SIGMA**2
    rates = np.asarray(state["rates"], dtype=float).reshape(-1)
    ar = np.asarray(state["ar_parameters"], dtype=float).reshape(-1, 9)[0]
    reference, lower, upper = ar_prior_arrays()
    reference_raw = ar_reference_raw(
        reference,
        lower,
        upper,
        ar_active_parameter_indices(),
    )
    rate_prior_reference = np.asarray(model._thickness_penalty_rate_prior_log_mean)
    entropy_reference = np.asarray(model._thickness_penalty_rate_reference)
    log_rates = np.asarray(values["np_log_rate_nodes"], dtype=float)
    rate_prior = 0.5 * np.sum(
        ((log_rates - rate_prior_reference) / TRAJECTORY_RATE_PRIOR_SIGMA_LOG) ** 2
    )
    log_ratio = log_rates - entropy_reference
    ratio = np.exp(log_ratio)
    entropy = np.where(
        np.abs(log_ratio) < 1.0e-4,
        0.5 * log_ratio**2,
        ratio * log_ratio - np.expm1(log_ratio),
    )
    likelihood_scale = (
        0.5
        * likelihood_weight
        / int(data["n_tiles"])
        * float(np.sum(data["sum_squared"]))
        / LOG_NOISE_SIGMA**2
    )
    ar_prior = 0.5 * np.sum(
        ((np.asarray(values["ar_tile_raw"]) - reference_raw) / AR_MEAN_PRIOR_SIGMA) ** 2
    )
    initial_prior = 0.5 * np.sum(
        ((np.asarray(values["initial_thickness_log_nm"]) - np.log(10.0)) / 1.5) ** 2
        + ((np.asarray(values["reset_initial_thickness_log_nm"]) - np.log(10.0)) / 1.5) ** 2
    )
    nuisance_prior = (
        0.5
        * np.sum(
            (np.asarray(values["absolute_hgcdte_n_log_scale"]) / ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG) ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(values["absolute_hgcdte_k_log_scale"]) / ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG) ** 2
        )
        + 0.5
        * np.sum(
            (
                (
                    np.asarray(values["ice_n_log_constant"])
                    - np.log(ICE_INDEX_DEFAULT / ICE_INDEX_DEFAULT)
                )
                / ICE_N_CONSTANT_PRIOR_SIGMA_LOG
            )
            ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(values["ar_n_log_modulation"]) / PRODUCTION_AR_MODULATION_PRIOR_SIGMA_LOG) ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(values["hgcdte_n_log_modulation"]) / PRODUCTION_AR_MODULATION_PRIOR_SIGMA_LOG) ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(values["hgcdte_k_log_modulation"]) / PRODUCTION_AR_MODULATION_PRIOR_SIGMA_LOG) ** 2
        )
    )
    ice_n = ICE_INDEX_DEFAULT * np.exp(
        float(np.asarray(values["ice_n_log_constant"]).reshape(-1)[0])
    )
    result = {
        "original_objective": float(0.5 * np.sum(original_residual**2)),
        "data_nll": float(data_nll),
        "data_rms_log": float(np.sqrt(np.mean(residual**2))),
        "data_count_weighted_rms_log": float(
            np.sqrt(np.sum(counts * residual**2) / np.sum(counts))
        ),
        "pre_mean_rate_nm_day": float(np.mean(rates[~post])),
        "post_mean_rate_nm_day": float(np.mean(rates[post])),
        "pre_minus_post_rate_nm_day": float(np.mean(rates[~post]) - np.mean(rates[post])),
        "ice_n_fitted": float(ice_n),
        "rate_prior_penalty": float(rate_prior),
        "entropy_penalty": float(likelihood_scale * 0.1 * np.mean(entropy)),
        "ar_prior_penalty": float(ar_prior),
        "initial_thickness_prior_penalty": float(initial_prior),
        "post_offset_prior_penalty": float(
            0.5
            * np.sum(
                (np.asarray(values["post_reset_tile_log_offset"]) / 0.1) ** 2
            )
        ),
        "other_nuisance_prior_penalty": float(nuisance_prior),
        "ar1_thickness_nm": float(ar[0] * 100.0),
        "ar1_n": float(ar[1]),
        "ar1_k": float(ar[2]),
        "ar2_thickness_nm": float(ar[3] * 100.0),
        "ar2_n": float(ar[4]),
        "ar2_k": float(ar[5]),
        "ar3_thickness_nm": float(ar[6] * 100.0),
        "ar3_n": float(ar[7]),
        "ar3_k": float(ar[8]),
    }
    if projection is not None:
        result.update(projection)
    if augmented_objective is None:
        augmented_objective = 0.5 * np.sum(original_residual**2)
    result["augmented_objective"] = float(augmented_objective)
    return result


def run(args):
    data = load_tile_statistics(args.statistics)
    context = load_response_context(
        data["sca_names"], sed_weighting="energy", qe_response_mode="none"
    )
    template = initial_values_for_data(
        data,
        constant_rate=False,
        trajectory_parameterization="rate",
        initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        fit_ar_hgcdte_modulation=True,
    )
    rate_reference = _rate_reference(args.map, np.asarray(template["np_log_rate_nodes"]).shape)
    values = (
        _load_start_values(args.start_values, template)
        if args.start_values is not None
        else load_map_warm_start(args.map, template, data)
    )
    model = _model(data, context, rate_reference, args.rate_prior_mean_nm_day)
    model._thickness_penalty_rate_reference = rate_reference
    model._thickness_penalty_rate_prior_log_mean = (
        rate_reference
        if args.rate_prior_mean_nm_day is None
        else np.full_like(rate_reference, np.log(args.rate_prior_mean_nm_day))
    )
    baseline_state = model.production_state(values)
    solution = {"ice": np.asarray(baseline_state["ice"], dtype=float)[0]}
    plan = _penalty_plan(
        data,
        [solution, solution, solution],
        5.0,
        120.0,
        64,
        args.smooth_sigma_nm,
    )
    lambda_disagreement = args.lambda_shared / 4.0
    plan["lambda_shared"] = args.lambda_shared
    plan["lambda_disagreement"] = lambda_disagreement
    penalized_model = attach_penalty(
        model,
        data,
        plan,
        args.lambda_shared,
        lambda_disagreement,
    )
    baseline_projection = _evaluate_projection(penalized_model, values, data, plan)
    baseline_augmented = 0.5 * np.sum(
        np.asarray(penalized_model.production_residuals(values), dtype=float) ** 2
    )
    baseline = _metrics(
        penalized_model,
        values,
        data,
        baseline_projection,
        baseline_augmented,
    )
    refit_values, diagnostics = run_least_squares_map_optimization(
        penalized_model,
        values,
        jax,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        initial_thickness_prior_sigma_nm=10.0,
        absolute_hgcdte_n_scale_prior_sigma_log=0.05,
        ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
        ice_k_prior_sigma=0.02,
        ar_k_prior_sigma=0.02,
        ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
        ar_hgcdte_modulation_prior_sigma_log=PRODUCTION_AR_MODULATION_PRIOR_SIGMA_LOG,
    )
    refit_projection = _evaluate_projection(penalized_model, refit_values, data, plan)
    refit_augmented = 0.5 * np.sum(
        np.asarray(penalized_model.production_residuals(refit_values), dtype=float) ** 2
    )
    refit = _metrics(
        penalized_model,
        refit_values,
        data,
        refit_projection,
        refit_augmented,
    )
    refit.update(
        {
            "optimizer_success": bool(diagnostics["converged"]),
            "gradient_converged": bool(diagnostics["gradient_converged"]),
            "function_evaluations": int(diagnostics["function_evaluations"]),
            "scaled_gradient_norm": float(diagnostics["scaled_gradient_norm"]),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "penalized_map_values.npz",
        **{name: np.asarray(value) for name, value in refit_values.items()},
    )
    with (args.output_dir / "comparison.json").open("w") as handle:
        json.dump(
            {
                "lambda_shared": args.lambda_shared,
                "lambda_disagreement": lambda_disagreement,
                "smooth_sigma_nm": args.smooth_sigma_nm,
                "rate_prior_mean_nm_day": args.rate_prior_mean_nm_day,
                "fixed_grid_lower_nm": float(plan["grid"][0]),
                "fixed_grid_upper_nm": float(plan["grid"][-1]),
                "baseline": baseline,
                "penalized_refit": refit,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    run(parse_args())
