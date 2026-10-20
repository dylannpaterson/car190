"""Test a fixed-grid thickness-correlated residual penalty on AR seed states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from diagnose_wfi_tile136_shared_residual import (  # noqa: E402
    _curves,
    _project_shared,
)
from fit_wfi_tmm_production import (  # noqa: E402
    AR_LAYERS,
    AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    AR_K_PRIOR_SIGMA,
    AR_MEAN_PRIOR_SIGMA,
    ICE_K_PRIOR_SIGMA,
    ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    LOG_NOISE_SIGMA,
    POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    _evaluate_model,
    initial_values_for_data,
    load_response_context,
    load_tile_statistics,
    run_least_squares_map_optimization,
)
from profile_wfi_tile136_ar_indices_maxent import (  # noqa: E402
    _map_options,
    _profile_model,
    _row,
    _squeeze_rate_reference,
)


SEED_ALPHAS = (0.0, 0.45, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--seeds-dir", type=Path, required=True)
    parser.add_argument("--entropy-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thickness-min", type=float, default=5.0)
    parser.add_argument("--thickness-max", type=float, default=120.0)
    parser.add_argument("--grid-count", type=int, default=64)
    parser.add_argument("--smooth-sigma-nm", type=float, default=1.0)
    parser.add_argument("--lambda-shared", type=float, default=1.0)
    parser.add_argument("--lambda-disagreement", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    return parser.parse_args()


def _load_seed(path, template):
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


def _seed_paths(seeds_dir):
    return {
        alpha: seeds_dir / f"released_seed_alpha_{alpha:.2f}.npz"
        for alpha in SEED_ALPHAS
    }


def _fixed_grid(
    data,
    solutions,
    thickness_min,
    thickness_max,
    grid_count,
    lock_grid=False,
):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    reset_day = float(data["reset_day"])
    ranges = []
    curve_indices = {}
    for side, side_mask in (("pre", days < reset_day), ("post", days >= reset_day)):
        for led in sorted(np.unique(leds)):
            indices = np.flatnonzero(side_mask & (leds == led))
            if indices.size < 2:
                raise ValueError(f"LED {led} has fewer than two {side} observations")
            curve_indices[(side, int(led))] = indices
            for solution in solutions:
                thickness = np.asarray(solution["ice"], dtype=float)[indices]
                ranges.append((float(np.min(thickness)), float(np.max(thickness))))
    if lock_grid:
        lower = float(thickness_min)
        upper = float(thickness_max)
        # jax.numpy.interp uses endpoint values outside the observed range.
        # Keep the production grid global even when a tile starts slightly
        # above the lower bound or does not reach the upper bound.
    else:
        lower = max(float(thickness_min), max(value[0] for value in ranges))
        upper = min(float(thickness_max), min(value[1] for value in ranges))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"No fixed overlap range: {lower} to {upper} nm")
    return np.linspace(lower, upper, int(grid_count)), curve_indices


def _smooth_matrix(grid, sigma):
    grid = np.asarray(grid, dtype=float)
    if sigma <= 0.0:
        return np.eye(grid.size)
    distances = grid[:, None] - grid[None, :]
    matrix = np.exp(-0.5 * (distances / float(sigma)) ** 2)
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _penalty_plan(
    data,
    solutions,
    thickness_min,
    thickness_max,
    grid_count,
    smooth_sigma,
    lock_grid=False,
):
    grid, curve_indices = _fixed_grid(
        data,
        solutions,
        thickness_min,
        thickness_max,
        grid_count,
        lock_grid=lock_grid,
    )
    counts = np.asarray(data["counts"], dtype=float)[0]
    weights = {}
    reference = solutions[1]
    for key, indices in curve_indices.items():
        order = np.argsort(np.asarray(reference["ice"])[indices])
        indices = indices[order]
        curve_indices[key] = indices
        reference_thickness = np.asarray(reference["ice"])[indices]
        counts_grid = np.interp(grid, reference_thickness, counts[indices])
        weights[key] = counts_grid / LOG_NOISE_SIGMA**2
    smooth = _smooth_matrix(grid, smooth_sigma)
    return {
        "grid": np.asarray(grid, dtype=float),
        "curve_indices": curve_indices,
        "weights": weights,
        "smooth": smooth,
    }


def _prediction(model, values, state, ice, data):
    n_rows = int(data["n_scas"]) * int(data["n_tiles"])
    predictor = model.production_predictor
    kwargs = {
        "post_reset_tile_log_offsets": values["post_reset_tile_log_offset"].reshape(-1),
        "ice_n_log_modulation": values.get("ice_n_log_modulation"),
        "ice_n_log_constant": values.get("ice_n_log_constant"),
        "ice_k_constant": values.get("ice_k_constant"),
        "ar_n_log_modulation": (
            values["ar_n_log_modulation"].reshape(
                (n_rows, AR_LAYERS, -1)
            )
            if "ar_n_log_modulation" in values
            else None
        ),
        "hgcdte_n_log_modulation": (
            values["hgcdte_n_log_modulation"].reshape((n_rows, -1))
            if "hgcdte_n_log_modulation" in values
            else None
        ),
        "hgcdte_k_log_modulation": (
            values["hgcdte_k_log_modulation"].reshape((n_rows, -1))
            if "hgcdte_k_log_modulation" in values
            else None
        ),
        "led_log_offsets": values.get("led_log_offsets"),
        "return_prediction": True,
    }
    _, prediction = predictor.residual_sum_squares(
        ice,
        state["ar_parameters"].reshape(n_rows, -1),
        state["hgcdte"].reshape(n_rows, 2),
        values.get("incidence_angle_deg", 0.0),
        **kwargs,
    )
    return prediction.reshape((n_rows, -1))


def _jax_penalty_parts(model, values, data, plan):
    state = model.production_state(values)
    ice = state["ice"]
    prediction = _prediction(model, values, state, ice, data)
    counts = jax.numpy.asarray(data["counts"])
    observed = jax.numpy.asarray(data["sums"]) / jax.numpy.maximum(counts, 1.0)
    post_mask = jax.numpy.asarray(
        np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    )
    observed = observed + (
        values["post_reset_tile_log_offset"].reshape(-1, 1) * post_mask[None, :]
    )
    residual = prediction - observed
    tangent_mask = np.ones((int(data["n_scas"]) * int(data["n_tiles"]), residual.shape[-1]))
    tangent_mask[:, np.asarray(data["first_by_led"], dtype=int)] = 0.0
    tangent_step = 1.0e-3
    tangent = (
        _prediction(model, values, state, ice + tangent_step * tangent_mask, data)
        - _prediction(model, values, state, ice - tangent_step * tangent_mask, data)
    ) / (2.0 * tangent_step)
    shared_weights = []
    perpendicular = {"pre": [], "post": []}
    for side in ("pre", "post"):
        side_residual = []
        side_tangent = []
        side_weights = []
        for led in sorted(np.unique(np.asarray(data["led"], dtype=int))):
            indices = plan["curve_indices"][(side, int(led))]
            grid = jax.numpy.asarray(plan["grid"])
            current_ice = ice[0, indices]
            current_residual = residual[0, indices]
            current_tangent = tangent[0, indices]
            side_residual.append(jax.numpy.interp(grid, current_ice, current_residual))
            side_tangent.append(jax.numpy.interp(grid, current_ice, current_tangent))
            side_weights.append(jax.numpy.asarray(plan["weights"][(side, int(led))]))
        side_residual = jax.numpy.stack(side_residual)
        side_tangent = jax.numpy.stack(side_tangent)
        side_weights = jax.numpy.stack(side_weights)
        center = jax.numpy.sum(side_weights * side_residual, axis=1) / jax.numpy.sum(
            side_weights, axis=1
        )
        centered = side_residual - center[:, None]
        coefficient = jax.numpy.sum(side_weights * side_tangent * centered, axis=0) / jax.numpy.maximum(
            jax.numpy.sum(side_weights * side_tangent**2, axis=0), 1.0e-18
        )
        perpendicular[side] = jax.numpy.asarray(plan["smooth"]) @ (
            centered - side_tangent * coefficient[None, :]
        ).T
        perpendicular[side] = perpendicular[side].T
        shared_weights.append(side_weights)
    pre = perpendicular["pre"]
    post = perpendicular["post"]
    shared = 0.5 * (pre + post)
    disagreement = 0.5 * (pre - post)
    shared_weights = 0.5 * (shared_weights[0] + shared_weights[1])
    return shared, disagreement, shared_weights


def attach_penalty(model, data, plan, lambda_shared, lambda_disagreement):
    original_residuals = model.production_residuals
    lambda_shared = float(lambda_shared)
    lambda_disagreement = float(lambda_disagreement)
    if lambda_shared < 0.0 or lambda_disagreement < 0.0:
        raise ValueError("Penalty weights must be non-negative")

    def production_residuals(values):
        original = original_residuals(values)
        shared, disagreement, weights = _jax_penalty_parts(model, values, data, plan)
        denominator = jax.numpy.sum(weights)
        shared_residual = jax.numpy.sqrt(
            lambda_shared * weights / denominator
        ) * shared
        disagreement_residual = jax.numpy.sqrt(
            lambda_disagreement * weights / denominator
        ) * disagreement
        return jax.numpy.concatenate(
            (original, shared_residual.reshape(-1), disagreement_residual.reshape(-1))
        )

    model.production_residuals = production_residuals
    model.original_production_residuals = original_residuals
    return model


def _evaluate_projection(model, values, data, plan):
    shared, disagreement, weights = _jax_penalty_parts(model, values, data, plan)
    shared = np.asarray(shared, dtype=float)
    disagreement = np.asarray(disagreement, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return {
        "shared_perpendicular_rms_log": float(
            np.sqrt(np.sum(weights * shared**2) / np.sum(weights))
        ),
        "disagreement_rms_log": float(
            np.sqrt(np.sum(weights * disagreement**2) / np.sum(weights))
        ),
        "shared_penalty_objective": float(
            0.5 * np.sum(float(plan["lambda_shared"]) * weights * shared**2) / np.sum(weights)
        ),
        "disagreement_penalty_objective": float(
            0.5
            * np.sum(float(plan["lambda_disagreement"]) * weights * disagreement**2)
            / np.sum(weights)
        ),
        "fixed_grid_lower_nm": float(plan["grid"][0]),
        "fixed_grid_upper_nm": float(plan["grid"][-1]),
        "fixed_grid_count": int(plan["grid"].size),
    }


def _original_metrics(model, values, data, alpha, stage, n_values, rate_reference, entropy_weight):
    evaluated = _evaluate_model(model, values)
    original_residual = np.asarray(model.original_production_residuals(values), dtype=float)
    ar = np.asarray(evaluated["ar_parameters"], dtype=float).reshape(-1, 3 * AR_LAYERS)[0]
    if n_values is None:
        n_values = ar[[1, 4, 7]]
    diagnostics = {
        "objective": float(0.5 * np.sum(original_residual**2)),
        "converged": True,
        "gradient_converged": False,
        "function_evaluations": 0,
        "scaled_gradient_norm": float("nan"),
        "message": "diagnostic evaluation",
    }
    row = _row(
        alpha,
        n_values,
        data,
        values,
        diagnostics,
        evaluated,
        rate_reference,
        entropy_weight,
    )
    row["stage"] = stage
    row["original_objective"] = diagnostics["objective"]
    for layer in range(AR_LAYERS):
        row[f"ar{layer + 1}_thickness_nm"] = float(ar[3 * layer] * 100.0)
        row[f"ar{layer + 1}_n"] = float(ar[3 * layer + 1])
        row[f"ar{layer + 1}_k"] = float(ar[3 * layer + 2])
    return row, evaluated


def _fit(model, initial_values):
    return run_least_squares_map_optimization(
        model,
        initial_values,
        jax,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        initial_thickness_prior_sigma_nm=10.0,
        absolute_hgcdte_n_scale_prior_sigma_log=0.05,
        ice_n_modulation_prior_sigma_log=ICE_N_MODULATION_PRIOR_SIGMA_LOG,
        ice_k_prior_sigma=ICE_K_PRIOR_SIGMA,
        ar_k_prior_sigma=AR_K_PRIOR_SIGMA,
        ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
        ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        led_log_offset_prior_sigma_log=LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    )


def run(args):
    if args.grid_count < 2:
        raise ValueError("grid-count must be at least two")
    lambda_disagreement = (
        args.lambda_shared / 4.0
        if args.lambda_disagreement is None
        else args.lambda_disagreement
    )
    data = load_tile_statistics(args.statistics)
    context = load_response_context(
        data["sca_names"], sed_weighting="energy", qe_response_mode="none"
    )
    template = initial_values_for_data(data, **_map_options())
    seeds = {
        alpha: _load_seed(path, template)
        for alpha, path in _seed_paths(args.seeds_dir).items()
    }
    rate_reference = _squeeze_rate_reference(
        args.entropy_reference,
        np.asarray(template["np_log_rate_nodes"]).shape,
    )
    seed_solutions = [
        {"ice": np.asarray(_profile_model(
            data, context, seeds[alpha], None, rate_reference, args.entropy_weight
        ).production_state(seeds[alpha])["ice"])[0]}
        for alpha in SEED_ALPHAS
    ]
    plan = _penalty_plan(
        data,
        seed_solutions,
        args.thickness_min,
        args.thickness_max,
        args.grid_count,
        args.smooth_sigma_nm,
    )
    plan["lambda_shared"] = float(args.lambda_shared)
    plan["lambda_disagreement"] = float(lambda_disagreement)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in SEED_ALPHAS:
        values = seeds[alpha]
        base_model = _profile_model(
            data, context, values, None, rate_reference, args.entropy_weight
        )
        frozen_model = attach_penalty(
            base_model,
            data,
            plan,
            args.lambda_shared,
            lambda_disagreement,
        )
        row, evaluated = _original_metrics(
            frozen_model,
            values,
            data,
            alpha,
            "frozen",
            None,
            rate_reference,
            args.entropy_weight,
        )
        row.update(_evaluate_projection(frozen_model, values, data, plan))
        row["augmented_objective"] = float(
            0.5 * np.sum(np.asarray(frozen_model.production_residuals(values), dtype=float) ** 2)
        )
        row["lambda_shared"] = float(args.lambda_shared)
        row["lambda_disagreement"] = float(lambda_disagreement)
        rows.append(row)
        refit_model = _profile_model(
            data, context, values, None, rate_reference, args.entropy_weight
        )
        refit_model = attach_penalty(
            refit_model,
            data,
            plan,
            args.lambda_shared,
            lambda_disagreement,
        )
        refit_values, diagnostics = _fit(refit_model, values)
        row, evaluated = _original_metrics(
            refit_model,
            refit_values,
            data,
            alpha,
            "refit",
            None,
            rate_reference,
            args.entropy_weight,
        )
        row.update(_evaluate_projection(refit_model, refit_values, data, plan))
        row["augmented_objective"] = float(
            0.5 * np.sum(np.asarray(refit_model.production_residuals(refit_values), dtype=float) ** 2)
        )
        row["lambda_shared"] = float(args.lambda_shared)
        row["lambda_disagreement"] = float(lambda_disagreement)
        row["optimizer_success"] = bool(diagnostics["converged"])
        row["gradient_converged"] = bool(diagnostics["gradient_converged"])
        row["function_evaluations"] = int(diagnostics["function_evaluations"])
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"refit_seed_alpha_{alpha:.2f}.npz",
            **{name: np.asarray(value) for name, value in refit_values.items()},
            **{name: np.asarray(value) for name, value in evaluated.items()},
        )
    with (args.output_dir / "thickness_penalty_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "penalty_configuration.json").open("w") as handle:
        json.dump(
            {
                "lambda_shared": float(args.lambda_shared),
                "lambda_disagreement": float(lambda_disagreement),
                "smooth_sigma_nm": float(args.smooth_sigma_nm),
                "grid_lower_nm": float(plan["grid"][0]),
                "grid_upper_nm": float(plan["grid"][-1]),
                "grid_count": int(plan["grid"].size),
                "overlap_locked_across_seeds": True,
            },
            handle,
            indent=2,
        )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
