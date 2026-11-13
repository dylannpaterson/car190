"""Profile fixed AR base indices with the current full MaxEnt MAP model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fit_wfi_tmm_production import (  # noqa: E402
    ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
    AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    AR_K_PRIOR_SIGMA,
    AR_MEAN_PRIOR_SIGMA,
    ICE_K_PRIOR_SIGMA,
    ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    INITIAL_THICKNESS_LOG_SIGMA,
    INITIAL_THICKNESS_PRIOR_LOGNORMAL,
    LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    LOG_NOISE_SIGMA,
    POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    ar_active_parameter_indices,
    ar_prior_arrays,
    ar_reference_raw,
    _evaluate_model,
    build_model,
    default_likelihood_weight,
    initial_values_for_data,
    load_map_warm_start,
    load_response_context,
    load_tile_statistics,
    run_least_squares_map_optimization,
)

DEFAULT_LOW_N = np.asarray([1.459396724, 1.580425586, 2.254509268])
DEFAULT_HIGH_N = np.asarray([1.535, 1.954, 3.171])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--entropy-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-count", type=int, default=11)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--low-n", default=','.join(str(value) for value in DEFAULT_LOW_N))
    parser.add_argument("--high-n", default=','.join(str(value) for value in DEFAULT_HIGH_N))
    return parser.parse_args()


def _squeeze_rate_reference(path, target_shape):
    with np.load(Path(path) / "posterior_samples.npz", allow_pickle=False) as samples:
        value = np.asarray(samples["np_log_rate_nodes"], dtype=float)
    while value.ndim > len(target_shape) and value.shape[0] == 1:
        value = value[0]
    if value.shape != target_shape:
        raise ValueError(f"Rate reference shape {value.shape} does not match {target_shape}")
    return value


def _map_options():
    return {
        "constant_rate": False,
        "trajectory_parameterization": "rate",
        "initial_thickness_prior": INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        "fit_incidence_angle": False,
        "fit_ice_n_modulation": True,
        "fit_ice_k_constant": True,
        "fit_ar_k_constant": True,
        "fit_ar_hgcdte_modulation": True,
        "fit_led_log_offsets": True,
    }


def _profile_model(
    data,
    context,
    initial_values,
    n_values,
    rate_reference,
    entropy_weight,
):
    return build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=entropy_weight,
        rate_prior_log_mean=rate_reference,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        trajectory_parameterization="rate",
        initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        absolute_hgcdte_n_scale_prior_sigma_log=0.05,
        fit_incidence_angle=False,
        fit_ice_n_modulation=True,
        ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
        fit_ice_k_constant=True,
        ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
        fit_ar_k_constant=True,
        ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
        ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
        fixed_ar_base_n=n_values,
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        fit_led_log_offsets=True,
    )


def _row(
    alpha,
    n_values,
    data,
    map_values,
    diagnostics,
    evaluated,
    rate_reference,
    entropy_weight,
):
    days = np.asarray(data["days"], dtype=float)
    reset_day = float(data["reset_day"])
    rates = np.asarray(evaluated["instantaneous_rate_nm_day"], dtype=float).reshape(-1)
    pre = days < reset_day
    post = ~pre
    prediction = np.asarray(evaluated["_map_least_squares_prediction"], dtype=float).reshape(-1)
    observed = np.asarray(data["sums"][0] / data["counts"][0], dtype=float)
    observed = observed + float(np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1)[0]) * post
    residual = prediction - observed
    counts = np.asarray(data["counts"][0], dtype=float)
    ar = np.asarray(evaluated["ar_parameters"], dtype=float).reshape(-1, 9)[0]
    likelihood_weight = default_likelihood_weight(data["counts"])
    within_group = np.asarray(data["sum_squared"][0]) - (
        np.asarray(data["sums"][0]) ** 2 / counts
    )
    data_nll = 0.5 * likelihood_weight / int(data["n_tiles"]) * np.sum(
        within_group + counts * residual**2
    ) / LOG_NOISE_SIGMA**2
    log_rates = np.asarray(map_values["np_log_rate_nodes"], dtype=float)
    rate_prior = 0.5 * np.sum(
        ((log_rates - rate_reference) / TRAJECTORY_RATE_PRIOR_SIGMA_LOG) ** 2
    )
    log_ratio = log_rates - rate_reference
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
    entropy_penalty = likelihood_scale * entropy_weight * float(np.mean(entropy))
    reference, lower, upper = ar_prior_arrays()
    reference_raw = ar_reference_raw(
        reference, lower, upper, ar_active_parameter_indices()
    )
    ar_raw = np.asarray(map_values["ar_tile_raw"], dtype=float)
    ar_prior = 0.5 * np.sum(((ar_raw - reference_raw) / AR_MEAN_PRIOR_SIGMA) ** 2)
    initial_prior = 0.5 * np.sum(
        (
            (np.asarray(map_values["initial_thickness_log_nm"]) - np.log(10.0))
            / INITIAL_THICKNESS_LOG_SIGMA
        )
        ** 2
        + (
            (np.asarray(map_values["reset_initial_thickness_log_nm"]) - np.log(10.0))
            / INITIAL_THICKNESS_LOG_SIGMA
        )
        ** 2
    )
    offset_prior = 0.5 * np.sum(
        (np.asarray(map_values["post_reset_tile_log_offset"]) / POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)
        ** 2
    )
    nuisance_prior = (
        0.5 * np.sum((np.asarray(map_values["absolute_hgcdte_n_log_scale"]) / 0.05) ** 2)
        + 0.5
        * np.sum(
            (
                np.asarray(map_values["absolute_hgcdte_k_log_scale"])
                / ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG
            )
            ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(map_values["ice_n_log_modulation"]) / ICE_N_MODULATION_PRIOR_SIGMA_LOG)
            ** 2
        )
        + 0.5 * np.sum((np.asarray(map_values["ice_k_constant"]) / ICE_K_PRIOR_SIGMA) ** 2)
        + 0.5 * np.sum((np.asarray(map_values["ar_k_constant"]) / AR_K_PRIOR_SIGMA) ** 2)
        + 0.5
        * np.sum(
            (np.asarray(map_values["ar_n_log_modulation"]) / AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG)
            ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(map_values["hgcdte_n_log_modulation"]) / 0.05) ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(map_values["hgcdte_k_log_modulation"]) / AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG)
            ** 2
        )
        + 0.5
        * np.sum(
            (np.asarray(map_values["led_log_offsets"]) / LED_LOG_OFFSET_PRIOR_SIGMA_LOG)
            ** 2
        )
    )
    known_penalty = (
        data_nll
        + rate_prior
        + entropy_penalty
        + ar_prior
        + initial_prior
        + offset_prior
        + nuisance_prior
    )
    return {
        "alpha": float(alpha),
        "n1": float(n_values[0]),
        "n2": float(n_values[1]),
        "n3": float(n_values[2]),
        "pre_mean_rate_nm_day": float(np.mean(rates[pre])),
        "post_mean_rate_nm_day": float(np.mean(rates[post])),
        "delta_pre_minus_post_nm_day": float(np.mean(rates[pre]) - np.mean(rates[post])),
        "pre_initial_rate_nm_day": float(rates[np.flatnonzero(pre)[0]]),
        "post_initial_rate_nm_day": float(rates[np.flatnonzero(post)[0]]),
        # The serialized AR thickness coordinate is 1/100 of a nm.
        "ar1_thickness_nm": float(ar[0] * 100.0),
        "ar2_thickness_nm": float(ar[3] * 100.0),
        "ar3_thickness_nm": float(ar[6] * 100.0),
        "objective": float(diagnostics["objective"]),
        "data_nll": float(data_nll),
        "rate_prior_penalty": float(rate_prior),
        "entropy_penalty": float(entropy_penalty),
        "ar_prior_penalty": float(ar_prior),
        "initial_thickness_prior_penalty": float(initial_prior),
        "post_offset_prior_penalty": float(offset_prior),
        "other_nuisance_prior_penalty": float(nuisance_prior),
        "objective_minus_known_terms": float(diagnostics["objective"] - known_penalty),
        "data_rms_log": float(np.sqrt(np.mean(residual**2))),
        "data_count_weighted_rms_log": float(
            np.sqrt(np.sum(counts * residual**2) / np.sum(counts))
        ),
        "optimizer_success": bool(diagnostics["converged"]),
        "gradient_converged": bool(diagnostics["gradient_converged"]),
        "function_evaluations": int(diagnostics["function_evaluations"]),
        "scaled_gradient_norm": float(diagnostics["scaled_gradient_norm"]),
        "optimizer_message": diagnostics["message"],
    }


def fit_profile(statistics_path, map_path, reference_path, output_dir, alpha_count, low_n, high_n, entropy_weight):
    data = load_tile_statistics(statistics_path)
    context = load_response_context(
        data["sca_names"], sed_weighting="energy", qe_response_mode="none"
    )
    options = _map_options()
    initial_values = initial_values_for_data(data, **options)
    initial_values = load_map_warm_start(map_path, initial_values, data)
    rate_reference = _squeeze_rate_reference(
        reference_path,
        np.asarray(initial_values["np_log_rate_nodes"]).shape,
    )
    rows = []
    node_rates = []
    node_ice = []
    current = initial_values
    for alpha in np.linspace(0.0, 1.0, alpha_count):
        n_values = (1.0 - alpha) * low_n + alpha * high_n
        model = _profile_model(
            data, context, current, n_values, rate_reference, entropy_weight
        )
        map_values, diagnostics = run_least_squares_map_optimization(
            model,
            current,
            jax,
            trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            initial_thickness_prior_sigma_nm=10.0,
            absolute_hgcdte_n_scale_prior_sigma_log=0.05,
            ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
            ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
            ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
            ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
            ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        )
        evaluated = _evaluate_model(model, map_values)
        row = _row(
            alpha,
            n_values,
            data,
            map_values,
            diagnostics,
            evaluated,
            rate_reference,
            entropy_weight,
        )
        rows.append(row)
        node_rates.append(
            np.asarray(evaluated["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        )
        node_ice.append(
            np.asarray(evaluated["node_ice_thickness"], dtype=float).reshape(-1)
        )
        current = map_values
        print(
            f"alpha={alpha:.4f}: objective={row['objective']:.8g}, "
            f"pre={row['pre_mean_rate_nm_day']:.8g}, "
            f"post={row['post_mean_rate_nm_day']:.8g}, "
            f"delta={row['delta_pre_minus_post_nm_day']:.8g}"
        )
    minimum = min(row["objective"] for row in rows)
    for row in rows:
        row["delta_objective"] = row["objective"] - minimum
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ar_index_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        output_dir / "ar_index_profile_trajectory.npz",
        alpha=np.asarray([row["alpha"] for row in rows]),
        n=np.asarray([[row["n1"], row["n2"], row["n3"]] for row in rows]),
        node_rates=np.asarray(node_rates),
        node_ice=np.asarray(node_ice),
    )


def main():
    args = parse_args()
    if args.alpha_count < 2:
        raise ValueError("alpha-count must be at least two")
    low_n = np.asarray([float(value) for value in args.low_n.split(",")])
    high_n = np.asarray([float(value) for value in args.high_n.split(",")])
    if low_n.shape != (3,) or high_n.shape != (3,):
        raise ValueError("Both low-n and high-n require three comma-separated values")
    fit_profile(
        args.statistics,
        args.map,
        args.entropy_reference,
        args.output_dir,
        args.alpha_count,
        low_n,
        high_n,
        args.entropy_weight,
    )


if __name__ == "__main__":
    main()
