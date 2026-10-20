"""Measure local AR/rate sensitivity correlations for WFI tile 136.

The calculation is a fixed-MAP structural diagnostic.  It differentiates the
two-epoch relative-response prediction with respect to independent pre/post
initial thicknesses and rates, plus the six AR thickness/index coordinates.
It does not refit the production model or claim that the local linearization
is a posterior uncertainty estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from fit_wfi_tmm_production import (  # noqa: E402
    LOG_NOISE_SIGMA,
    build_prediction_function,
    load_response_context,
    load_tile_statistics,
)
from fit_wfi_tile136_common_range_rates import (  # noqa: E402
    _epoch_data,
    _load_fixed_state,
)


PARAMETER_NAMES = (
    "pre_log_initial_thickness",
    "pre_log_rate",
    "post_log_initial_thickness",
    "post_log_rate",
    "ar1_log_thickness_nm",
    "ar2_log_thickness_nm",
    "ar3_log_thickness_nm",
    "ar1_log_n",
    "ar2_log_n",
    "ar3_log_n",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fit",
        action="append",
        nargs=3,
        metavar=("DETECTOR", "STATISTICS", "MAP"),
        required=True,
    )
    parser.add_argument("--thickness-min", type=float, default=5.0)
    parser.add_argument("--thickness-max", type=float, default=120.0)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-4)
    return parser.parse_args()


def _fixed_prediction_kwargs(state):
    return {
        "ice_n_log_modulation": state.get("ice_n_log_modulation"),
        "ice_k_constant": state["ice_k_constant"],
        "ar_n_log_modulation": state.get("ar_n_log_modulation"),
        "hgcdte_n_log_modulation": state.get("hgcdte_n_log_modulation"),
        "hgcdte_k_log_modulation": state.get("hgcdte_k_log_modulation"),
        "led_log_offsets": state["led_log_offsets"],
    }


def _epoch_setup(data, map_thickness, context, state, side, lower, upper):
    reset_day = float(data["reset_day"])
    side_mask = (
        np.asarray(data["days"], dtype=float) < reset_day
        if side == "pre"
        else np.asarray(data["days"], dtype=float) >= reset_day
    )
    epoch = _epoch_data(data, side_mask)
    thickness = np.asarray(map_thickness, dtype=float)[side_mask]
    fit_mask = (thickness >= lower) & (thickness <= upper)
    predictor = build_prediction_function(
        epoch,
        context,
        fit_ice_n_modulation=state["fit_ice_n_modulation"],
        fit_ice_k_constant=True,
        fit_ar_hgcdte_modulation=state["fit_ar_hgcdte_modulation"],
        fit_led_log_offsets=True,
    )
    counts = np.asarray(epoch["counts"][0], dtype=float)
    days = np.asarray(epoch["days"], dtype=float)
    return {
        "side": side,
        "epoch": epoch,
        "predictor": predictor,
        "fit_mask": fit_mask,
        "counts": counts,
        "days": days,
        "epoch_start": float(np.min(days)),
        "map_thickness": thickness[fit_mask],
    }


def _initial_parameter_vector(map_path, data, map_thickness):
    values = []
    with np.load(Path(map_path) / "posterior_samples.npz", allow_pickle=False) as samples:
        for side in ("pre", "post"):
            reset_day = float(data["reset_day"])
            side_mask = (
                np.asarray(data["days"], dtype=float) < reset_day
                if side == "pre"
                else np.asarray(data["days"], dtype=float) >= reset_day
            )
            side_days = np.asarray(data["days"], dtype=float)[side_mask]
            side_thickness = np.asarray(map_thickness, dtype=float)[side_mask]
            rate = np.polyfit(side_days, side_thickness, 1)[0]
            initial = float(side_thickness[0])
            values.extend((np.log(max(initial, 1.0e-6)), np.log(max(rate, 1.0e-6))))
        ar = np.asarray(samples["ar_parameters"], dtype=float)[0, 0, 0].reshape(3, 3)
    values.extend(np.log(np.maximum(ar[:, 0] * 100.0, 1.0e-6)))
    values.extend(np.log(np.maximum(ar[:, 1], 1.0e-6)))
    return np.asarray(values, dtype=float)


def _weighted_prediction(parameters, setups, state):
    ar = np.asarray(state["ar_parameters"], dtype=float).reshape(1, 3, 3).copy()
    ar[0, :, 0] = np.exp(parameters[4:7]) / 100.0
    ar[0, :, 1] = np.exp(parameters[7:10])
    ar = ar.reshape(1, 9)
    predictions = []
    for setup, initial_index, rate_index in zip(
        setups, (0, 2), (1, 3)
    ):
        thickness = np.exp(parameters[initial_index]) + np.exp(
            parameters[rate_index]
        ) * (setup["days"] - setup["epoch_start"])
        prediction = setup["predictor"](
            thickness,
            ar,
            state["hgcdte"],
            state["incidence_angle_deg"],
            **_fixed_prediction_kwargs(state),
        )
        predictions.append(
            np.asarray(prediction).reshape(-1)[setup["fit_mask"]]
            * np.sqrt(setup["counts"][setup["fit_mask"]])
            / LOG_NOISE_SIGMA
        )
    return np.concatenate(predictions)


def analyze_detector(detector, statistics_path, map_path, lower, upper, step):
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
        raise ValueError(f"{detector} does not have enough common-range observations")
    parameters = _initial_parameter_vector(map_path, data, map_thickness)
    base = _weighted_prediction(parameters, setups, state)
    jacobian = np.empty((base.size, parameters.size), dtype=float)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            _weighted_prediction(plus, setups, state)
            - _weighted_prediction(minus, setups, state)
        ) / (2.0 * step)
    column_norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / np.maximum(column_norms[None, :], np.finfo(float).eps)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    correlations = np.corrcoef(jacobian, rowvar=False)
    summary = {
        "detector": detector,
        "groups_pre": int(np.count_nonzero(setups[0]["fit_mask"])),
        "groups_post": int(np.count_nonzero(setups[1]["fit_mask"])),
        "weighted_rows": int(base.size),
        "condition_number_normalized": float(singular_values[0] / singular_values[-1]),
        "singular_values_normalized": ";".join(f"{value:.9g}" for value in singular_values),
        "statistics": str(statistics_path),
        "map": str(map_path),
    }
    correlation_rows = []
    for row_index, row_name in enumerate(PARAMETER_NAMES):
        for column_index in range(row_index + 1, len(PARAMETER_NAMES)):
            correlation_rows.append(
                {
                    "detector": detector,
                    "parameter_a": row_name,
                    "parameter_b": PARAMETER_NAMES[column_index],
                    "correlation": float(correlations[row_index, column_index]),
                    "absolute_correlation": float(
                        abs(correlations[row_index, column_index])
                    ),
                    "column_norm_a": float(column_norms[row_index]),
                    "column_norm_b": float(column_norms[column_index]),
                }
            )
    return summary, correlation_rows


def main():
    args = parse_args()
    if args.thickness_min < 0.0 or args.thickness_max <= args.thickness_min:
        raise ValueError("Thickness bounds must be ordered")
    if args.finite_difference_step <= 0.0:
        raise ValueError("Finite-difference step must be positive")
    summaries = []
    correlation_rows = []
    for detector, statistics_path, map_path in args.fit:
        summary, rows = analyze_detector(
            detector,
            Path(statistics_path),
            Path(map_path),
            args.thickness_min,
            args.thickness_max,
            args.finite_difference_step,
        )
        summaries.append(summary)
        correlation_rows.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ar_identifiability_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_dir / "ar_identifiability_correlations.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=correlation_rows[0].keys())
        writer.writeheader()
        writer.writerows(correlation_rows)
    for summary in summaries:
        strongest = sorted(
            (
                row
                for row in correlation_rows
                if row["detector"] == summary["detector"]
            ),
            key=lambda row: row["absolute_correlation"],
            reverse=True,
        )[:5]
        print(
            f"{summary['detector']}: "
            f"condition={summary['condition_number_normalized']:.6g}; "
            + ", ".join(
                f"{row['parameter_a']}~{row['parameter_b']}="
                f"{row['correlation']:.4f}"
                for row in strongest
            )
        )


if __name__ == "__main__":
    main()
