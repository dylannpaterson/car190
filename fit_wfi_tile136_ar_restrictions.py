"""Compare reduced AR-stack freedoms while fitting independent epoch rates.

The three modes are intentionally explicit controls:

``A`` frees the three AR base indices and three thicknesses.
``B`` fixes the base indices to the saved MAP and frees thicknesses.
``C`` fixes the complete saved MAP AR stack.

These modes test the index/thickness degeneracy.  Mode B is not a material
model and mode C is not a preflight constraint; physical B/C variants require
an external coating-index and fabrication-tolerance table.
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
    INITIAL_THICKNESS_LOG_SIGMA,
    INITIAL_THICKNESS_LOG_MEDIAN_NM,
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
from fit_wfi_tile136_common_range_rates import (  # noqa: E402
    _epoch_data,
    _load_fixed_state,
)


MODE_NAMES = {
    "A": "free_map_ar_n_and_thickness",
    "B": "fixed_map_ar_n_free_thickness",
    "C": "fixed_map_ar_stack",
}
_, AR_LOWER_ALL, AR_UPPER_ALL = ar_prior_arrays()


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
    parser.add_argument("--thickness-min", type=float, default=5.0)
    parser.add_argument("--thickness-max", type=float, default=120.0)
    parser.add_argument("--modes", default="ABC")
    return parser.parse_args()


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700.0, 700.0)))


def _physical_ar_from_raw(raw, fixed_ar, mode):
    physical = np.asarray(fixed_ar, dtype=float).reshape(1, 3, 3).copy()
    active = ar_active_parameter_indices()
    if mode == "A":
        mode_raw = np.asarray(raw, dtype=float)
        raw_by_index = dict(zip(active.tolist(), mode_raw.tolist()))
        thickness_indices = [0, 3, 6]
        index_indices = [1, 4, 7]
        for layer in range(3):
            thickness_index = thickness_indices[layer]
            index_index = index_indices[layer]
            physical[0, layer, 0] = AR_LOWER_ALL[thickness_index] + (
                AR_UPPER_ALL[thickness_index] - AR_LOWER_ALL[thickness_index]
            ) * _sigmoid(raw_by_index[thickness_index])
            lower = (
                AR_LOWER_ALL[index_index]
                if layer == 0
                else physical[0, layer - 1, 1]
            )
            physical[0, layer, 1] = lower + (
                AR_UPPER_ALL[index_index] - lower
            ) * _sigmoid(raw_by_index[index_index])
    elif mode == "B":
        for layer, raw_value in enumerate(np.asarray(raw, dtype=float)):
            index = 3 * layer
            physical[0, layer, 0] = AR_LOWER_ALL[index] + (
                AR_UPPER_ALL[index] - AR_LOWER_ALL[index]
            ) * _sigmoid(raw_value)
    return physical.reshape(1, 9)


def _raw_from_physical(ar, mode):
    ar = np.asarray(ar, dtype=float).reshape(3, 3)
    values = []
    if mode == "A":
        active = ar_active_parameter_indices()
        for index in active:
            layer = index // 3
            component = index % 3
            value = ar[layer, component]
            lower = (
                ar[layer - 1, component]
                if component == 1 and layer > 0
                else AR_LOWER_ALL[index]
            )
            upper = AR_UPPER_ALL[index]
            probability = np.clip((value - lower) / (upper - lower), 1.0e-8, 1.0 - 1.0e-8)
            values.append(np.log(probability / (1.0 - probability)))
    else:
        for layer in range(3):
            index = 3 * layer
            value = ar[layer, 0]
            probability = np.clip(
                (value - AR_LOWER_ALL[index])
                / (AR_UPPER_ALL[index] - AR_LOWER_ALL[index]),
                1.0e-8,
                1.0 - 1.0e-8,
            )
            values.append(np.log(probability / (1.0 - probability)))
    return np.asarray(values, dtype=float)


def _epoch_setup(data, map_thickness, context, state, side, lower, upper):
    reset_day = float(data["reset_day"])
    side_mask = (
        np.asarray(data["days"], dtype=float) < reset_day
        if side == "pre"
        else np.asarray(data["days"], dtype=float) >= reset_day
    )
    epoch = _epoch_data(data, side_mask)
    epoch_thickness = np.asarray(map_thickness, dtype=float)[side_mask]
    fit_mask = (epoch_thickness >= lower) & (epoch_thickness <= upper)
    predictor = build_prediction_function(
        epoch,
        context,
        fit_ice_n_modulation=state["fit_ice_n_modulation"],
        fit_ice_k_constant=True,
        fit_ar_hgcdte_modulation=state["fit_ar_hgcdte_modulation"],
        fit_led_log_offsets=True,
    )
    return {
        "side": side,
        "epoch": epoch,
        "predictor": predictor,
        "fit_mask": fit_mask,
        "days": np.asarray(epoch["days"], dtype=float),
        "counts": np.asarray(epoch["counts"][0], dtype=float),
        "epoch_start": float(np.min(epoch["days"])),
        "observed": np.asarray(epoch["sums"][0] / epoch["counts"][0], dtype=float)
        + (
            float(state["post_reset_tile_log_offset"])
            if side == "post"
            else 0.0
        ),
    }


def _initial_rate_parameters(data, map_thickness):
    values = []
    reset_day = float(data["reset_day"])
    for side in ("pre", "post"):
        side_mask = (
            np.asarray(data["days"], dtype=float) < reset_day
            if side == "pre"
            else np.asarray(data["days"], dtype=float) >= reset_day
        )
        days = np.asarray(data["days"], dtype=float)[side_mask]
        thickness = np.asarray(map_thickness, dtype=float)[side_mask]
        rate = max(float(np.polyfit(days, thickness, 1)[0]), 1.0e-6)
        values.extend((np.log(max(float(thickness[0]), 1.0e-6)), np.log(rate)))
    return np.asarray(values, dtype=float)


def _parameter_initial_values(state, data, map_thickness, mode):
    values = list(_initial_rate_parameters(data, map_thickness))
    if mode in ("A", "B"):
        values.extend(_raw_from_physical(state["ar_parameters"], mode))
    return np.asarray(values, dtype=float)


def _parameter_bounds(mode):
    lower = [-20.0, -20.0, -20.0, -20.0]
    upper = [np.log(500.0), np.log(100.0), np.log(500.0), np.log(100.0)]
    if mode == "A":
        lower.extend([-20.0] * 6)
        upper.extend([20.0] * 6)
    elif mode == "B":
        lower.extend([-20.0] * 3)
        upper.extend([20.0] * 3)
    return np.asarray(lower), np.asarray(upper)


def _prediction_kwargs(state):
    return {
        "ice_n_log_modulation": state.get("ice_n_log_modulation"),
        "ice_k_constant": state["ice_k_constant"],
        "ar_n_log_modulation": state.get("ar_n_log_modulation"),
        "hgcdte_n_log_modulation": state.get("hgcdte_n_log_modulation"),
        "hgcdte_k_log_modulation": state.get("hgcdte_k_log_modulation"),
        "led_log_offsets": state["led_log_offsets"],
    }


def _residual(parameters, setups, state, mode, reference_raw, likelihood_weight):
    if mode == "A":
        ar = _physical_ar_from_raw(parameters[4:], state["ar_parameters"], mode)
    elif mode == "B":
        ar = _physical_ar_from_raw(parameters[4:], state["ar_parameters"], mode)
    else:
        ar = np.asarray(state["ar_parameters"], dtype=float)
    result = []
    for setup, initial_index, rate_index in zip(setups, (0, 2), (1, 3)):
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
        selected = setup["fit_mask"]
        result.append(
            np.sqrt(likelihood_weight)
            * np.sqrt(setup["counts"][selected])
            * (prediction[selected] - setup["observed"][selected])
            / LOG_NOISE_SIGMA
        )
    # Match the production initial-thickness and positive-rate priors.
    result.extend(
        [
            (parameters[0] - np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM))
            / INITIAL_THICKNESS_LOG_SIGMA,
            (parameters[1] - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY))
            / TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            (parameters[2] - np.log(INITIAL_THICKNESS_LOG_MEDIAN_NM))
            / INITIAL_THICKNESS_LOG_SIGMA,
            (parameters[3] - np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY))
            / TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        ]
    )
    if mode == "A":
        result.append((parameters[4:] - reference_raw) / AR_MEAN_PRIOR_SIGMA)
    return np.concatenate([np.asarray(value).reshape(-1) for value in result])


def fit_mode(dataset, detector, statistics_path, map_path, lower, upper, mode):
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
    if any(np.count_nonzero(setup["fit_mask"]) < 3 for setup in setups):
        raise ValueError(f"{dataset} {detector} lacks common-range data")
    reference_raw = ar_reference_raw(*ar_prior_arrays(), ar_active_parameter_indices())
    initial = _parameter_initial_values(state, data, map_thickness, mode)
    bounds = _parameter_bounds(mode)
    likelihood_weight = default_likelihood_weight(data["counts"])
    result = least_squares(
        lambda parameters: _residual(
            parameters, setups, state, mode, reference_raw, likelihood_weight
        ),
        x0=initial,
        bounds=bounds,
        diff_step=1.0e-4,
        x_scale="jac",
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=2000,
    )
    parameters = result.x
    rates = np.exp(parameters[[1, 3]])
    initial_thickness = np.exp(parameters[[0, 2]])
    if mode in ("A", "B"):
        fitted_ar = _physical_ar_from_raw(parameters[4:], state["ar_parameters"], mode)
    else:
        fitted_ar = np.asarray(state["ar_parameters"], dtype=float)
    fitted_ar = fitted_ar.reshape(3, 3)
    residual = _residual(
        parameters, setups, state, mode, reference_raw, likelihood_weight
    )
    data_residual = residual[: sum(np.count_nonzero(setup["fit_mask"]) for setup in setups)]
    rows = []
    delta = float(rates[0] - rates[1])
    data_offset = 0
    for index, (side, setup) in enumerate(zip(("pre", "post"), setups)):
        group_count = int(np.count_nonzero(setup["fit_mask"]))
        data_segment = data_residual[data_offset : data_offset + group_count]
        data_offset += group_count
        data_weighted_rms = np.sqrt(
            np.sum(data_segment**2)
            * LOG_NOISE_SIGMA**2
            / likelihood_weight
            / np.sum(setup["counts"][setup["fit_mask"]])
        )
        rows.append(
            {
                "dataset": dataset,
                "detector": detector,
                "mode": mode,
                "mode_description": MODE_NAMES[mode],
                "side": side,
                "groups_fit": int(np.count_nonzero(setup["fit_mask"])),
                "rate_nm_day": float(rates[index]),
                "initial_thickness_nm": float(initial_thickness[index]),
                "delta_pre_minus_post_nm_day": delta,
                "data_weighted_rms_log": float(data_weighted_rms),
                "objective_cost": float(2.0 * result.cost),
                "optimizer_success": bool(result.success),
                **{
                    f"ar_layer{layer + 1}_{component}": float(fitted_ar[layer, component_index])
                    for layer in range(3)
                    for component_index, component in enumerate(("thickness_scale", "n", "k"))
                },
                "statistics": str(statistics_path),
                "map": str(map_path),
            }
        )
    return rows


def main():
    args = parse_args()
    if args.thickness_min < 0.0 or args.thickness_max <= args.thickness_min:
        raise ValueError("Thickness bounds must be ordered")
    modes = [mode for mode in args.modes if mode in MODE_NAMES]
    if not modes:
        raise ValueError("--modes must contain at least one of A, B, or C")
    rows = []
    for dataset, detector, statistics_path, map_path in args.fit:
        for mode in modes:
            rows.extend(
                fit_mode(
                    dataset,
                    detector,
                    Path(statistics_path),
                    Path(map_path),
                    args.thickness_min,
                    args.thickness_max,
                    mode,
                )
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "ar_restriction_rate_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["side"] == "pre":
            print(
                f"{row['dataset']} {row['detector']} {row['mode']}: "
                f"pre={row['rate_nm_day']:.8g} nm/day, "
                f"delta={row['delta_pre_minus_post_nm_day']:.8g}"
            )


if __name__ == "__main__":
    main()
