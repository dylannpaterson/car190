#!/usr/bin/env python3
"""Test production recovery of a short-timescale ice-rate jump.

The canonical WFI statistics are never modified.  A real WFI10 tile-136
production fit supplies the exact pre-jump rate and optical state.  Each
synthetic case appends one pre-jump six-LED cycle followed by a configurable
number of post-jump cycles at four-hour intervals.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import run_wfi_production as production_runner
from fit_wfi_tmm_production import (
    ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    INITIAL_THICKNESS_PRIOR_LOGNORMAL,
    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    _evaluate_model,
    build_model,
    initial_values_for_data,
    load_map_warm_start,
    load_response_context,
    load_tile_statistics,
)


DEFAULT_JUMPS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.50, 1.0)
PRE_JUMP_CYCLE_COUNT = 1
HOURS_TO_FIRST_SYNTHETIC_CYCLE = 4.0
HOURS_BETWEEN_SYNTHETIC_CYCLES = 4.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--emcee", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=production_runner.PRODUCTION_MAXENT_WEIGHT,
        help="Trajectory MaxEnt weight; use 0 to disable trend regularization.",
    )
    parser.add_argument(
        "--disable-thickness-penalty",
        action="store_true",
        help="Disable the thickness-correlated residual penalty for baseline and fits.",
    )
    parser.add_argument(
        "--jump-percent",
        type=float,
        action="append",
        dest="jump_percent",
        help="Jump percentages; may be repeated. Defaults to 0, 1, 2, 5, 10, 15, 50, 100.",
    )
    parser.add_argument(
        "--post-jump-points",
        type=int,
        action="append",
        dest="post_jump_points",
        help="Number of post-jump cycles; may be repeated. Defaults to 1.",
    )
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, list(rows[0].keys())


def _write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_time(value):
    # Python accepts the seven-digit fractional seconds used by the WFI files.
    return datetime.fromisoformat(value)


def _format_time(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f") + "0"


def _latest_cycle(rows):
    latest_repeat = max(int(row["repeat"]) for row in rows)
    cycle = [row for row in rows if int(row["repeat"]) == latest_repeat]
    cycle.sort(key=lambda row: int(row["led"]))
    if [int(row["led"]) for row in cycle] != list(range(1, 7)):
        raise ValueError("The latest WFI10 cycle is not a complete six-LED cycle")
    return cycle


def _synthetic_structure(rows, post_jump_points):
    """Return one pre-jump and N post-jump future six-LED cycles."""
    cycle = _latest_cycle(rows)
    max_group = max(int(row["group"]) for row in rows)
    max_repeat = max(int(row["repeat"]) for row in rows)
    cycle_mean_day = np.mean([float(row["days"]) for row in cycle])
    first_offset_days = HOURS_TO_FIRST_SYNTHETIC_CYCLE / 24.0
    synthetic = []
    cycle_count = PRE_JUMP_CYCLE_COUNT + post_jump_points
    for cycle_index in range(cycle_count):
        offset_days = (
            HOURS_TO_FIRST_SYNTHETIC_CYCLE
            + cycle_index * HOURS_BETWEEN_SYNTHETIC_CYCLES
        ) / 24.0
        for led_index, source in enumerate(cycle):
            row = dict(source)
            row["group"] = str(max_group + cycle_index * 6 + led_index + 1)
            row["repeat"] = str(max_repeat + cycle_index + 1)
            row["days"] = str(float(source["days"]) - cycle_mean_day + cycle_mean_day + offset_days)
            row["exposure_start_time"] = _format_time(
                _parse_time(source["exposure_start_time"])
                + timedelta(days=offset_days)
            )
            row["file_set_name"] = (
                f"synthetic_rate_jump_{cycle_index}_{int(source['led'])}"
            )
            row["baseline_file_set_name"] = row["file_set_name"]
            synthetic.append(row)
    return synthetic


def _model_for_data(data):
    context = load_response_context(
        data["sca_names"],
        sed_weighting="energy",
        qe_response_mode="none",
    )
    n_nodes = len(data["node_days"])
    rate_prior = np.full(
        (int(data["n_scas"]), int(data["n_tiles"]), n_nodes),
        np.log(production_runner.PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )
    return build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_weight=0.0,
        rate_prior_log_mean=rate_prior,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        trajectory_parameterization="rate",
        initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
        ice_n_constant_reference=production_runner.production.ICE_INDEX_DEFAULT,
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=(
            production_runner.PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG
        ),
    )


def _map_rate_nodes(map_path):
    samples_path = Path(map_path) / "posterior_samples.npz"
    with np.load(samples_path, allow_pickle=False) as samples:
        log_rates = np.asarray(samples["np_log_rate_nodes"], dtype=float)
    return np.squeeze(log_rates).reshape(-1)


def _extend_map_values(map_path, data, log_rates):
    template = initial_values_for_data(
        data,
        constant_rate=False,
        trajectory_parameterization="rate",
        initial_thickness_prior=INITIAL_THICKNESS_PRIOR_LOGNORMAL,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        fit_ar_hgcdte_modulation=True,
    )
    target = np.asarray(template.pop("np_log_rate_nodes"))
    values = load_map_warm_start(map_path, template, data)
    values["np_log_rate_nodes"] = np.asarray(log_rates, dtype=float).reshape(
        target.shape
    )
    return values


def _prediction_for_truth(data, map_path, log_rates):
    model = _model_for_data(data)
    values = _extend_map_values(map_path, data, log_rates)
    evaluated = _evaluate_model(model, values)
    prediction = np.asarray(
        evaluated["_map_least_squares_prediction"], dtype=float
    ).reshape(-1)
    return prediction


def _noise_scales(rows):
    cycle = _latest_cycle(rows)
    result = {}
    for row in cycle:
        led = int(row["led"])
        count = float(row["n_pair_valid"])
        total = float(row["sum_log_change"])
        total_squared = float(row["sum_squared_log_change"])
        within = max(total_squared - total * total / count, 0.0)
        result[led] = (count, within, np.sqrt(within) / count)
    return result


def _apply_observations(rows, synthetic_rows, predicted, noise_scales, noise_draws):
    result = [dict(row) for row in rows]
    for index, row in enumerate(synthetic_rows):
        led = int(row["led"])
        count, within, _ = noise_scales[led]
        mean = float(predicted[index] + noise_draws[index])
        row = dict(row)
        row["n_pair_valid"] = str(int(round(count)))
        row["sum_log_change"] = str(count * mean)
        row["sum_squared_log_change"] = str(within + count * mean * mean)
        within_sd = np.sqrt(within / max(count, 1.0))
        row["min_log_change"] = str(mean - 3.0 * within_sd)
        row["max_log_change"] = str(mean + 3.0 * within_sd)
        result.append(row)
    return result


def _fit_tile(
    statistics,
    output_dir,
    emcee,
    workers,
    force,
    entropy_weight,
    thickness_penalty_enabled,
):
    fit_root = output_dir / "fits"
    return production_runner.fit_tile(
        136,
        statistics,
        fit_root,
        emcee,
        force,
        entropy_weight,
        thickness_penalty_enabled=thickness_penalty_enabled,
    )


def _fit_summary(fit_dir, jump_fraction, true_rate, post_jump_points):
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        node_rates = np.asarray(samples["node_rates"], dtype=float).reshape(-1)
        initial_rate = float(node_rates[-(post_jump_points + 2)])
        recovered_rate = float(node_rates[-1])
        objective = float(np.asarray(samples["map_objective_history"])[-1])
        gradient_converged = bool(
            int(np.asarray(samples["map_gradient_converged"]).item())
        )
    recovered_jump = recovered_rate / initial_rate - 1.0
    return {
        "jump_percent": 100.0 * jump_fraction,
        "true_pre_rate_nm_day": true_rate,
        "true_post_rate_nm_day": true_rate * (1.0 + jump_fraction),
        "recovered_pre_rate_nm_day": initial_rate,
        "recovered_post_rate_nm_day": recovered_rate,
        "true_jump_fraction": jump_fraction,
        "recovered_jump_fraction": recovered_jump,
        "recovered_jump_percent": 100.0 * recovered_jump,
        "absolute_jump_error_percent": 100.0 * (recovered_jump - jump_fraction),
        "map_objective": objective,
        "map_gradient_converged": gradient_converged,
    }


def run(args):
    jumps = (
        DEFAULT_JUMPS
        if args.jump_percent is None
        else tuple(float(value) / 100.0 for value in args.jump_percent)
    )
    post_jump_points = (
        (1,)
        if args.post_jump_points is None
        else tuple(int(value) for value in args.post_jump_points)
    )
    if any(value < 0.0 or not np.isfinite(value) for value in jumps):
        raise ValueError("Jump percentages must be finite and non-negative")
    if any(value < 1 for value in post_jump_points):
        raise ValueError("Post-jump point counts must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.entropy_weight < 0.0 or not np.isfinite(args.entropy_weight):
        raise ValueError("Entropy weight must be finite and non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, fieldnames = _read_rows(args.statistics)
    noise_scales = _noise_scales(rows)
    rng = np.random.default_rng(args.seed)
    maximum_structure = _synthetic_structure(rows, max(post_jump_points))
    maximum_noise_draws = rng.normal(
        0.0,
        np.asarray(
            [noise_scales[int(row["led"])][2] for row in maximum_structure]
        ),
    )

    baseline_dir = args.output_dir / "baseline"
    baseline_fit = _fit_tile(
        args.statistics,
        baseline_dir,
        args.emcee,
        args.workers,
        False,
        args.entropy_weight,
        not args.disable_thickness_penalty,
    )
    base_log_rates = _map_rate_nodes(baseline_fit)
    base_rate = float(np.exp(base_log_rates[-1]))

    summaries = []
    for point_count in post_jump_points:
        synthetic_structure = _synthetic_structure(rows, point_count)
        noise_draws = maximum_noise_draws[: len(synthetic_structure)]
        points_dir = args.output_dir / f"points_{point_count}"
        points_dir.mkdir(parents=True, exist_ok=True)
        provisional = [dict(row) for row in rows] + [
            dict(row) for row in synthetic_structure
        ]
        provisional_path = points_dir / (
            "wfi10_tile_log_statistics_256_tile_136_structure.csv"
        )
        _write_rows(provisional_path, fieldnames, provisional)
        provisional_data = load_tile_statistics(provisional_path)

        for jump_fraction in jumps:
            scenario_name = f"jump_{100.0 * jump_fraction:g}pct"
            scenario_dir = points_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            truth_log_rates = np.concatenate(
                (
                    base_log_rates,
                    [base_log_rates[-1]]
                    + [base_log_rates[-1] + np.log1p(jump_fraction)] * point_count,
                )
            )
            prediction = _prediction_for_truth(
                provisional_data,
                baseline_fit,
                truth_log_rates,
            )
            synthetic_prediction = prediction[-len(synthetic_structure) :]
            scenario_rows = _apply_observations(
                rows,
                synthetic_structure,
                synthetic_prediction,
                noise_scales,
                noise_draws,
            )
            scenario_statistics = scenario_dir / (
                "wfi10_tile_log_statistics_256_tile_136_count_corrected.csv"
            )
            _write_rows(scenario_statistics, fieldnames, scenario_rows)
            fit_dir = _fit_tile(
                scenario_statistics,
                scenario_dir,
                args.emcee,
                args.workers,
                True,
                args.entropy_weight,
                not args.disable_thickness_penalty,
            )
            result = _fit_summary(
                fit_dir,
                jump_fraction,
                base_rate,
                point_count,
            )
            result["post_jump_points"] = point_count
            summaries.append(result)

    summary_path = args.output_dir / "rate_jump_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "statistics": str(args.statistics.resolve()),
                "baseline_fit": str(baseline_fit.resolve()),
                "seed": args.seed,
                "hours_to_first_synthetic_cycle": HOURS_TO_FIRST_SYNTHETIC_CYCLE,
                "hours_between_synthetic_cycles": HOURS_BETWEEN_SYNTHETIC_CYCLES,
                "pre_jump_cycle_count": PRE_JUMP_CYCLE_COUNT,
                "post_jump_points": list(post_jump_points),
                "baseline_final_rate_nm_day": base_rate,
                "noise_model": "empirical per-LED aggregate-mean standard error",
                "trajectory_entropy_weight": args.entropy_weight,
                "thickness_penalty_enabled": not args.disable_thickness_penalty,
                "trajectory_rate_prior_mean_nm_day": (
                    production_runner.PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY
                ),
                "results": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"baseline_rate_nm_day": base_rate, "results": summaries}, indent=2))


if __name__ == "__main__":
    run(parse_args())
