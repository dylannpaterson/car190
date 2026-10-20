"""Profile between two AR-index solutions for WFI01 tile 136.

This first-pass profile uses independent constant pre/post rates.  At each
fixed interpolation value alpha it reoptimizes the two initial thicknesses,
two positive rates, three ordered AR thicknesses, and the post-reset tile
offset.  Other spectral nuisance components remain at the current low-index
MAP, so this is a reduced basin diagnostic rather than a full production fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)

from fit_wfi_tmm_production import (  # noqa: E402
    AR_MEAN_PRIOR_SIGMA,
    INITIAL_THICKNESS_LOG_MEDIAN_NM,
    INITIAL_THICKNESS_LOG_SIGMA,
    LOG_NOISE_SIGMA,
    TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
    ar_active_parameter_indices,
    ar_prior_arrays,
    ar_reference_raw,
    build_prediction_function,
    default_likelihood_weight,
    load_response_context,
    load_tile_statistics,
)
from fit_wfi_tile136_ar_restrictions import (  # noqa: E402
    _epoch_setup,
    _initial_rate_parameters,
    _load_fixed_state,
    _physical_ar_from_raw,
    _prediction_kwargs,
    _raw_from_physical,
)


DEFAULT_LOW_N = np.asarray([1.459396724, 1.580425586, 2.254509268])
DEFAULT_HIGH_N = np.asarray([1.535, 1.954, 3.171])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-count", type=int, default=21)
    parser.add_argument("--thickness-min", type=float, default=0.0)
    parser.add_argument("--thickness-max", type=float, default=250.0)
    parser.add_argument("--low-n", default=','.join(str(value) for value in DEFAULT_LOW_N))
    parser.add_argument("--high-n", default=','.join(str(value) for value in DEFAULT_HIGH_N))
    return parser.parse_args()


def _prediction_for_setup(parameters, setup, ar, state, post_offset):
    initial_index, rate_index = ((0, 1) if setup["side"] == "pre" else (2, 3))
    thickness = np.exp(parameters[initial_index]) + np.exp(
        parameters[rate_index]
    ) * (setup["days"] - setup["epoch_start"])
    prediction = np.asarray(
        setup["predictor"](
            thickness,
            ar,
            state["hgcdte"],
            state["incidence_angle_deg"],
            **_prediction_kwargs(state),
        )
    ).reshape(-1)
    observed = np.asarray(setup["epoch"]["sums"][0] / setup["epoch"]["counts"][0])
    if setup["side"] == "post":
        observed = observed + post_offset
    return prediction, observed


def _residual(parameters, setups, state, fixed_ar, reference_thickness_raw, likelihood_weight):
    ar = _physical_ar_from_raw(parameters[5:], fixed_ar, "B")
    post_offset = float(parameters[4])
    residuals = []
    for setup in setups:
        prediction, observed = _prediction_for_setup(
            parameters, setup, ar, state, post_offset
        )
        selected = setup["fit_mask"]
        residuals.append(
            np.sqrt(likelihood_weight)
            * np.sqrt(setup["counts"][selected])
            * (prediction[selected] - observed[selected])
            / LOG_NOISE_SIGMA
        )
    residuals.extend(
        (
            (parameters[0] - np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM))
            / INITIAL_THICKNESS_LOG_SIGMA,
            (parameters[1] - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY))
            / TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            (parameters[2] - np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM))
            / INITIAL_THICKNESS_LOG_SIGMA,
            (parameters[3] - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY))
            / TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            (parameters[5:] - reference_thickness_raw) / AR_MEAN_PRIOR_SIGMA,
        )
    )
    return np.concatenate([np.asarray(value).reshape(-1) for value in residuals])


def _initial_parameters(data, map_thickness, fixed_ar, state):
    rate_values = _initial_rate_parameters(data, map_thickness)
    reference, lower, upper = ar_prior_arrays()
    active = ar_active_parameter_indices()
    reference_raw = ar_reference_raw(reference, lower, upper, active)
    reference_thickness_raw = reference_raw[[0, 2, 4]]
    thickness_raw = _raw_from_physical(fixed_ar, "B")
    post_offset = float(state["post_reset_tile_log_offset"])
    return (
        np.concatenate((rate_values[:4], [post_offset], thickness_raw)),
        reference_thickness_raw,
    )


def fit_profile(statistics_path, map_path, output_dir, alpha_count, lower, upper, low_n, high_n):
    data = load_tile_statistics(statistics_path)
    state = _load_fixed_state(map_path)
    with np.load(Path(map_path) / "posterior_samples.npz", allow_pickle=False) as samples:
        map_thickness = np.asarray(samples["ice_thickness"], dtype=float).squeeze()
    context = load_response_context(
        data["sca_names"],
        sed_weighting=state["sed_weighting"],
        qe_response_mode="none",
    )
    setups = [
        _epoch_setup(data, map_thickness, context, state, side, lower, upper)
        for side in ("pre", "post")
    ]
    fixed_ar = np.asarray(state["ar_parameters"], dtype=float).copy()
    initial, reference_thickness_raw = _initial_parameters(
        data, map_thickness, fixed_ar, state
    )
    likelihood_weight = default_likelihood_weight(data["counts"])
    lower_bounds = np.asarray([-20.0, -20.0, -20.0, -20.0, -0.1, -20.0, -20.0, -20.0])
    upper_bounds = np.asarray([np.log(500.0), np.log(100.0), np.log(500.0), np.log(100.0), 0.1, 20.0, 20.0, 20.0])
    rows = []
    current = initial
    for alpha in np.linspace(0.0, 1.0, alpha_count):
        n_values = (1.0 - alpha) * low_n + alpha * high_n
        alpha_ar = fixed_ar.reshape(1, 3, 3).copy()
        alpha_ar[0, :, 1] = n_values
        alpha_ar = alpha_ar.reshape(1, 9)
        result = least_squares(
            lambda parameters: _residual(
                parameters,
                setups,
                state,
                alpha_ar,
                reference_thickness_raw,
                likelihood_weight,
            ),
            x0=current,
            bounds=(lower_bounds, upper_bounds),
            diff_step=1.0e-4,
            x_scale="jac",
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
            max_nfev=2000,
        )
        current = result.x
        final_ar = _physical_ar_from_raw(current[5:], alpha_ar, "B").reshape(3, 3)
        rates = np.exp(current[[1, 3]])
        row = {
            "alpha": float(alpha),
            "n1": float(n_values[0]),
            "n2": float(n_values[1]),
            "n3": float(n_values[2]),
            "pre_rate_nm_day": float(rates[0]),
            "post_rate_nm_day": float(rates[1]),
            "delta_pre_minus_post_nm_day": float(rates[0] - rates[1]),
            "pre_initial_thickness_nm": float(np.exp(current[0])),
            "post_initial_thickness_nm": float(np.exp(current[2])),
            "post_reset_tile_log_offset": float(current[4]),
            # The serialized AR thickness coordinate is 1/100 of a nm.
            "ar1_thickness_nm": float(final_ar[0, 0] * 100.0),
            "ar2_thickness_nm": float(final_ar[1, 0] * 100.0),
            "ar3_thickness_nm": float(final_ar[2, 0] * 100.0),
            "objective": float(2.0 * result.cost),
            "optimizer_success": bool(result.success),
            "optimizer_message": result.message,
        }
        data_residuals = []
        for setup in setups:
            prediction, observed = _prediction_for_setup(
                current, setup, _physical_ar_from_raw(current[5:], alpha_ar, "B"), state, current[4]
            )
            selected = setup["fit_mask"]
            data_residuals.append(prediction[selected] - observed[selected])
        data_residuals = np.concatenate(data_residuals)
        row["data_rms_log"] = float(np.sqrt(np.mean(data_residuals**2)))
        rows.append(row)
        print(
            f"alpha={alpha:.4f}: objective={row['objective']:.8g}, "
            f"pre={row['pre_rate_nm_day']:.8g}, post={row['post_rate_nm_day']:.8g}, "
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
    return rows


def main():
    args = parse_args()
    if args.alpha_count < 2:
        raise ValueError("alpha-count must be at least two")
    if args.thickness_min < 0.0 or args.thickness_max <= args.thickness_min:
        raise ValueError("Thickness bounds must be ordered")
    low_n = np.asarray([float(value) for value in args.low_n.split(",")])
    high_n = np.asarray([float(value) for value in args.high_n.split(",")])
    if low_n.shape != (3,) or high_n.shape != (3,):
        raise ValueError("Both low-n and high-n require three comma-separated values")
    fit_profile(
        args.statistics,
        args.map,
        args.output_dir,
        args.alpha_count,
        args.thickness_min,
        args.thickness_max,
        low_n,
        high_n,
    )


if __name__ == "__main__":
    main()
