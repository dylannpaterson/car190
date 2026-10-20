"""Release AR base indices from three fixed-index MaxEnt seed solutions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fit_wfi_tmm_production import (  # noqa: E402
    AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    AR_K_PRIOR_SIGMA,
    AR_MEAN_PRIOR_SIGMA,
    ICE_K_PRIOR_SIGMA,
    ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    _evaluate_model,
    initial_values_for_data,
    load_map_warm_start,
    load_response_context,
    load_tile_statistics,
    run_least_squares_map_optimization,
)
from fit_wfi_tile136_ar_restrictions import _raw_from_physical  # noqa: E402
from profile_wfi_tile136_ar_indices_maxent import (  # noqa: E402
    DEFAULT_HIGH_N,
    DEFAULT_LOW_N,
    _map_options,
    _profile_model,
    _row,
    _squeeze_rate_reference,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--entropy-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--low-n", default=','.join(str(value) for value in DEFAULT_LOW_N))
    parser.add_argument("--high-n", default=','.join(str(value) for value in DEFAULT_HIGH_N))
    return parser.parse_args()


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
    )


def _release_ar_start(map_values, evaluated):
    released = {name: np.asarray(value, dtype=float).copy() for name, value in map_values.items()}
    physical_ar = np.asarray(evaluated["ar_parameters"], dtype=float).reshape(-1, 9)[0]
    raw_ar = _raw_from_physical(physical_ar.reshape(1, 9), "A")
    released["ar_tile_raw"] = raw_ar.reshape(np.asarray(released["ar_tile_raw"]).shape)
    return released


def fit_seeds(statistics_path, map_path, reference_path, output_dir, low_n, high_n, entropy_weight):
    data = load_tile_statistics(statistics_path)
    context = load_response_context(
        data["sca_names"], sed_weighting="energy", qe_response_mode="none"
    )
    initial_values = load_map_warm_start(
        map_path,
        initial_values_for_data(data, **_map_options()),
        data,
    )
    rate_reference = _squeeze_rate_reference(
        reference_path,
        np.asarray(initial_values["np_log_rate_nodes"]).shape,
    )
    seed_alphas = (0.0, 0.45, 1.0)
    fixed_results = {}
    current = initial_values
    for alpha in seed_alphas:
        n_values = (1.0 - alpha) * low_n + alpha * high_n
        model = _profile_model(
            data, context, current, n_values, rate_reference, entropy_weight
        )
        map_values, diagnostics = _fit(model, current)
        evaluated = _evaluate_model(model, map_values)
        fixed_results[alpha] = (map_values, evaluated, diagnostics, n_values)
        current = map_values
        print(f"fixed alpha={alpha:.2f}: objective={diagnostics['objective']:.8g}")

    rows = []
    released_states = {}
    for alpha in seed_alphas:
        map_values, evaluated, diagnostics, n_values = fixed_results[alpha]
        fixed_row = _row(
            alpha, n_values, data, map_values, diagnostics, evaluated, rate_reference, entropy_weight
        )
        fixed_row.update({"seed_alpha": alpha, "stage": "fixed_index"})
        rows.append(fixed_row)

        released_start = _release_ar_start(map_values, evaluated)
        released_model = _profile_model(
            data, context, released_start, None, rate_reference, entropy_weight
        )
        released_values, released_diagnostics = _fit(released_model, released_start)
        released_evaluated = _evaluate_model(released_model, released_values)
        released_ar = np.asarray(released_evaluated["ar_parameters"], dtype=float).reshape(-1, 9)[0]
        released_n = released_ar[[1, 4, 7]]
        released_row = _row(
            alpha,
            released_n,
            data,
            released_values,
            released_diagnostics,
            released_evaluated,
            rate_reference,
            entropy_weight,
        )
        released_row.update({"seed_alpha": alpha, "stage": "released_index"})
        rows.append(released_row)
        released_states[alpha] = (released_values, released_evaluated)
        print(
            f"released seed alpha={alpha:.2f}: objective={released_diagnostics['objective']:.8g}, "
            f"n={released_n.tolist()}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "released_ar_seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for alpha, (values, evaluated) in released_states.items():
        np.savez_compressed(
            output_dir / f"released_seed_alpha_{alpha:.2f}.npz",
            **{name: np.asarray(value) for name, value in values.items()},
            ar_parameters=np.asarray(evaluated["ar_parameters"]),
            ice_thickness=np.asarray(evaluated["ice_thickness"]),
            rates=np.asarray(evaluated["instantaneous_rate_nm_day"]),
            node_ice=np.asarray(evaluated["node_ice_thickness"]),
            node_rates=np.asarray(evaluated["node_instantaneous_rate_nm_day"]),
        )
    for alpha, (map_values, evaluated, _, n_values) in fixed_results.items():
        np.savez_compressed(
            output_dir / f"fixed_seed_alpha_{alpha:.2f}.npz",
            **{name: np.asarray(value) for name, value in map_values.items()},
            ar_base_n=np.asarray(n_values),
            ar_parameters=np.asarray(evaluated["ar_parameters"]),
            ice_thickness=np.asarray(evaluated["ice_thickness"]),
            rates=np.asarray(evaluated["instantaneous_rate_nm_day"]),
            node_ice=np.asarray(evaluated["node_ice_thickness"]),
            node_rates=np.asarray(evaluated["node_instantaneous_rate_nm_day"]),
        )


def main():
    args = parse_args()
    low_n = np.asarray([float(value) for value in args.low_n.split(",")])
    high_n = np.asarray([float(value) for value in args.high_n.split(",")])
    if low_n.shape != (3,) or high_n.shape != (3,):
        raise ValueError("Both low-n and high-n require three comma-separated values")
    fit_seeds(
        args.statistics,
        args.map,
        args.entropy_reference,
        args.output_dir,
        low_n,
        high_n,
        args.entropy_weight,
    )


if __name__ == "__main__":
    main()
