"""Compare thickness-shared perpendicular residuals for fixed AR solutions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fit_wfi_tmm_production import (  # noqa: E402
    AR_LAYERS,
    LOG_NOISE_SIGMA,
    _evaluate_model,
    initial_values_for_data,
    load_response_context,
    load_tile_statistics,
)
from profile_wfi_tile136_ar_indices_maxent import (  # noqa: E402
    _map_options,
    _profile_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--solutions-dir", type=Path, required=True)
    parser.add_argument("--entropy-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thickness-min", type=float, default=5.0)
    parser.add_argument("--thickness-max", type=float, default=120.0)
    parser.add_argument("--grid-count", type=int, default=64)
    parser.add_argument("--tangent-step-nm", type=float, default=1.0e-3)
    parser.add_argument(
        "--trajectory-mode",
        choices=("own", "common"),
        default="common",
        help="Use each fitted trajectory or the alpha=0.45 trajectory for every AR state.",
    )
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    return parser.parse_args()


def _load_solution(path, template):
    with np.load(path, allow_pickle=False) as archive:
        values = {
            name: np.asarray(archive[name], dtype=float).copy()
            for name in template
            if name in archive.files
        }
        if set(values) != set(template):
            missing = sorted(set(template) - set(values))
            raise ValueError(f"{path} is missing model values: {missing}")
        base_n = np.asarray(archive["ar_base_n"], dtype=float)
    if base_n.shape != (AR_LAYERS,):
        raise ValueError(f"{path} has invalid AR base-index shape {base_n.shape}")
    return values, base_n


def _predictor_arguments(model, values, state):
    n_rows = int(np.asarray(state["hgcdte"]).reshape(-1, 2).shape[0])
    arguments = {
        "ice_thickness": None,
        "ar_parameters": np.asarray(state["ar_parameters"]).reshape(n_rows, -1),
        "hgcdte_parameters": np.asarray(state["hgcdte"]).reshape(n_rows, 2),
        "incidence_angle_deg": 0.0,
        "ice_n_log_modulation": np.asarray(values["ice_n_log_modulation"]).reshape(
            n_rows, -1
        ),
        "ice_k_constant": np.asarray(values["ice_k_constant"]).reshape(n_rows),
        "ar_n_log_modulation": np.asarray(values["ar_n_log_modulation"]).reshape(
            n_rows, AR_LAYERS, -1
        ),
        "hgcdte_n_log_modulation": np.asarray(
            values["hgcdte_n_log_modulation"]
        ).reshape(n_rows, -1),
        "hgcdte_k_log_modulation": np.asarray(
            values["hgcdte_k_log_modulation"]
        ).reshape(n_rows, -1),
        "led_log_offsets": np.asarray(values["led_log_offsets"]),
    }
    return arguments


def _evaluate_fixed_solution(model, values, data, tangent_step_nm):
    state = model.production_state(values)
    predictor = model.production_predictor
    arguments = _predictor_arguments(model, values, state)
    ice = np.asarray(state["ice"], dtype=float)
    arguments["ice_thickness"] = ice
    prediction = np.asarray(predictor(**arguments), dtype=float)[0]
    baseline_mask = np.ones(ice.shape, dtype=float)
    baseline_mask[:, np.asarray(data["first_by_led"], dtype=int)] = 0.0
    arguments["ice_thickness"] = ice + tangent_step_nm * baseline_mask
    plus = np.asarray(predictor(**arguments), dtype=float)[0]
    arguments["ice_thickness"] = ice - tangent_step_nm * baseline_mask
    minus = np.asarray(predictor(**arguments), dtype=float)[0]
    tangent = (plus - minus) / (2.0 * tangent_step_nm)
    counts = np.asarray(data["counts"], dtype=float)[0]
    observed = np.asarray(data["sums"], dtype=float)[0] / np.maximum(counts, 1.0)
    observed = observed + float(np.asarray(values["post_reset_tile_log_offset"]).reshape(-1)[0]) * (
        np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    )
    return {
        "ice": ice[0],
        "prediction": prediction,
        "residual": prediction - observed,
        "tangent": tangent,
        "counts": counts,
    }


def _unique_curve(thickness, values, weights):
    order = np.argsort(thickness)
    thickness = np.asarray(thickness, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    unique, inverse = np.unique(thickness, return_inverse=True)
    result = np.zeros(unique.size, dtype=float)
    result_weights = np.zeros(unique.size, dtype=float)
    for index in range(unique.size):
        mask = inverse == index
        result_weights[index] = np.sum(weights[mask])
        result[index] = np.average(values[mask], weights=weights[mask])
    return unique, result, result_weights


def _curves(solution, data, grid):
    days = np.asarray(data["days"], dtype=float)
    led_values = np.asarray(data["led"], dtype=int)
    reset_day = float(data["reset_day"])
    curves = {}
    for side, side_mask in (
        ("pre", days < reset_day),
        ("post", days >= reset_day),
    ):
        for led in sorted(np.unique(led_values)):
            mask = side_mask & (led_values == led)
            if np.count_nonzero(mask) < 2:
                continue
            thickness, residual, counts = _unique_curve(
                solution["ice"][mask],
                solution["residual"][mask],
                solution["counts"][mask],
            )
            _, tangent, _ = _unique_curve(
                solution["ice"][mask],
                solution["tangent"][mask],
                solution["counts"][mask],
            )
            curves[(side, led)] = {
                "thickness": thickness,
                "residual": np.interp(grid, thickness, residual),
                "tangent": np.interp(grid, thickness, tangent),
                "counts": np.maximum(np.interp(grid, thickness, counts), 1.0),
            }
    return curves


def _common_grid(solution, data, thickness_min, thickness_max, grid_count):
    days = np.asarray(data["days"], dtype=float)
    leds = sorted(np.unique(np.asarray(data["led"], dtype=int)))
    reset_day = float(data["reset_day"])
    ranges = []
    for side, side_mask in (("pre", days < reset_day), ("post", days >= reset_day)):
        for led in leds:
            mask = side_mask & (np.asarray(data["led"], dtype=int) == led)
            if np.count_nonzero(mask) < 2:
                continue
            values = np.unique(np.asarray(solution["ice"])[mask])
            ranges.append((float(values.min()), float(values.max())))
    if len(ranges) != 2 * len(leds):
        raise ValueError("Every LED must have at least two pre and post observations")
    lower = max(thickness_min, max(value[0] for value in ranges))
    upper = min(thickness_max, min(value[1] for value in ranges))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"No common thickness interval: {lower} to {upper} nm")
    return np.linspace(lower, upper, grid_count)


def _project_shared(curves, leds, grid):
    side_residuals = {}
    side_tangents = {}
    side_weights = {}
    for side in ("pre", "post"):
        residual = np.stack([curves[(side, led)]["residual"] for led in leds])
        tangent = np.stack([curves[(side, led)]["tangent"] for led in leds])
        weights = np.stack([curves[(side, led)]["counts"] for led in leds]) / LOG_NOISE_SIGMA**2
        center = np.sum(weights * residual, axis=1) / np.sum(weights, axis=1)
        side_residuals[side] = residual - center[:, None]
        side_tangents[side] = tangent
        side_weights[side] = weights
    perpendicular = {}
    for side in ("pre", "post"):
        residual = side_residuals[side]
        tangent = side_tangents[side]
        weights = side_weights[side]
        denominator = np.sum(weights * tangent**2, axis=0)
        numerator = np.sum(weights * tangent * residual, axis=0)
        coefficient = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1.0e-18)
        perpendicular[side] = residual - tangent * coefficient[None, :]
    shared = 0.5 * (perpendicular["pre"] + perpendicular["post"])
    shared_vertical = 0.5 * (side_residuals["pre"] + side_residuals["post"])
    return {
        "residuals": side_residuals,
        "tangents": side_tangents,
        "weights": side_weights,
        "perpendicular": perpendicular,
        "shared": shared,
        "shared_vertical": shared_vertical,
    }


def _weighted_rms(values, weights):
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def _weighted_correlation(first, second, weights):
    first_mean = np.sum(weights * first) / np.sum(weights)
    second_mean = np.sum(weights * second) / np.sum(weights)
    first_centered = first - first_mean
    second_centered = second - second_mean
    covariance = np.sum(weights * first_centered * second_centered)
    variance = np.sum(weights * first_centered**2) * np.sum(weights * second_centered**2)
    return float(covariance / np.sqrt(variance)) if variance > 0.0 else float("nan")


def _diagnostic_summary(projected, alpha, base_n, grid, leds):
    shared_weights = 0.5 * (projected["weights"]["pre"] + projected["weights"]["post"])
    pre = projected["perpendicular"]["pre"]
    post = projected["perpendicular"]["post"]
    shared = projected["shared"]
    disagreement = 0.5 * (pre - post)
    raw_pre = projected["residuals"]["pre"]
    raw_post = projected["residuals"]["post"]
    total_raw = np.concatenate((raw_pre.ravel(), raw_post.ravel()))
    total_perp = np.concatenate((pre.ravel(), post.ravel()))
    total_weights = np.concatenate(
        (projected["weights"]["pre"].ravel(), projected["weights"]["post"].ravel())
    )
    return {
        "alpha": float(alpha),
        "n1": float(base_n[0]),
        "n2": float(base_n[1]),
        "n3": float(base_n[2]),
        "grid_lower_nm": float(grid[0]),
        "grid_upper_nm": float(grid[-1]),
        "grid_count": int(grid.size),
        "vertical_shared_rms_log": _weighted_rms(projected["shared_vertical"], shared_weights),
        "perpendicular_shared_rms_log": _weighted_rms(shared, shared_weights),
        "pre_perpendicular_rms_log": _weighted_rms(pre, projected["weights"]["pre"]),
        "post_perpendicular_rms_log": _weighted_rms(post, projected["weights"]["post"]),
        "epoch_difference_component_rms_log": _weighted_rms(
            disagreement, shared_weights
        ),
        "pre_post_perpendicular_correlation": _weighted_correlation(
            pre, post, shared_weights
        ),
        "raw_vertical_rms_log": _weighted_rms(total_raw, total_weights),
        "raw_perpendicular_rms_log": _weighted_rms(total_perp, total_weights),
        "tangent_explained_fraction": 1.0
        - float(np.sum(total_weights * total_perp**2))
        / max(float(np.sum(total_weights * total_raw**2)), 1.0e-30),
        "shared_perpendicular_max_abs_log": float(np.max(np.abs(shared))),
        "led_count": len(leds),
    }


def run_diagnostic(args):
    data = load_tile_statistics(args.statistics)
    context = load_response_context(
        data["sca_names"], sed_weighting="energy", qe_response_mode="none"
    )
    template = initial_values_for_data(data, **_map_options())
    paths = {
        0.0: args.solutions_dir / "fixed_seed_alpha_0.00.npz",
        0.45: args.solutions_dir / "fixed_seed_alpha_0.45.npz",
        1.0: args.solutions_dir / "fixed_seed_alpha_1.00.npz",
    }
    loaded = {alpha: _load_solution(path, template) for alpha, path in paths.items()}
    reference_values = loaded[0.45][0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    curve_rows = []
    for alpha, (values, base_n) in loaded.items():
        values = {name: np.asarray(value).copy() for name, value in values.items()}
        if args.trajectory_mode == "common":
            for name in (
                "np_log_rate_nodes",
                "initial_thickness_log_nm",
                "reset_initial_thickness_log_nm",
            ):
                values[name] = np.asarray(reference_values[name]).copy()
        model = _profile_model(
            data,
            context,
            values,
            base_n,
            np.asarray(reference_values["np_log_rate_nodes"]),
            args.entropy_weight,
        )
        solution = _evaluate_fixed_solution(model, values, data, args.tangent_step_nm)
        grid = _common_grid(
            solution,
            data,
            args.thickness_min,
            args.thickness_max,
            args.grid_count,
        )
        curves = _curves(solution, data, grid)
        leds = sorted(np.unique(np.asarray(data["led"], dtype=int)))
        projected = _project_shared(curves, leds, grid)
        row = _diagnostic_summary(projected, alpha, base_n, grid, leds)
        row["trajectory_mode"] = args.trajectory_mode
        summary_rows.append(row)
        for index, thickness in enumerate(grid):
            for led_index, led in enumerate(leds):
                curve_rows.append(
                    {
                        "alpha": float(alpha),
                        "n1": float(base_n[0]),
                        "n2": float(base_n[1]),
                        "n3": float(base_n[2]),
                        "trajectory_mode": args.trajectory_mode,
                        "thickness_nm": float(thickness),
                        "led": int(led),
                        "pre_residual_centered_log": float(projected["residuals"]["pre"][led_index, index]),
                        "post_residual_centered_log": float(projected["residuals"]["post"][led_index, index]),
                        "pre_tangent_log_per_nm": float(projected["tangents"]["pre"][led_index, index]),
                        "post_tangent_log_per_nm": float(projected["tangents"]["post"][led_index, index]),
                        "pre_perpendicular_log": float(projected["perpendicular"]["pre"][led_index, index]),
                        "post_perpendicular_log": float(projected["perpendicular"]["post"][led_index, index]),
                        "shared_vertical_log": float(projected["shared_vertical"][led_index, index]),
                        "shared_perpendicular_log": float(projected["shared"][led_index, index]),
                        "pre_count_weight": float(projected["weights"]["pre"][led_index, index]),
                        "post_count_weight": float(projected["weights"]["post"][led_index, index]),
                    }
                )
        print(
            f"alpha={alpha:g}: shared_perp_rms={row['perpendicular_shared_rms_log']:.8g}, "
            f"raw_perp_rms={row['raw_perpendicular_rms_log']:.8g}, "
            f"tangent_fraction={row['tangent_explained_fraction']:.8g}"
        )
    with (args.output_dir / "shared_residual_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "shared_residual_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)


def main():
    args = parse_args()
    if args.grid_count < 2:
        raise ValueError("Grid count must be at least two")
    if args.tangent_step_nm <= 0.0:
        raise ValueError("Tangent step must be positive")
    run_diagnostic(args)


if __name__ == "__main__":
    main()
