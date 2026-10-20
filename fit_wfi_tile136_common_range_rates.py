"""Fit independent constant pre/post rates over a common MAP thickness range.

This is a diagnostic only.  The saved production optical state is held fixed
for both reset epochs, while each epoch gets its own positive rate and initial
thickness.  The thickness-window selection is made from the saved MAP state so
that the comparison tests unequal coverage without adding optical nuisance
parameters.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_wfi_tmm_production import (
    LOG_NOISE_SIGMA,
    build_prediction_function,
    load_response_context,
    load_tile_statistics,
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
    return parser.parse_args()


def _epoch_data(data, mask):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Epoch contains no observation groups")
    result = dict(data)
    for key in ("counts", "sums", "sum_squared"):
        result[key] = np.asarray(data[key])[:, indices]
    for key in ("groups", "led", "repeat", "program", "days", "exposure_times"):
        result[key] = np.asarray(data[key])[indices]
    result["first_by_led"] = np.asarray(
        [np.flatnonzero(result["led"] == led)[0] for led in range(1, 7)],
        dtype=int,
    )
    result["reset_day"] = float(np.max(result["days"]) + 1.0)
    return result


def _load_fixed_state(map_path):
    with np.load(Path(map_path) / "posterior_samples.npz", allow_pickle=False) as samples:
        sed_weighting = str(np.asarray(samples["sed_weighting"]).item())
        fit_ice_n_modulation = "ice_n_log_modulation" in samples
        fit_ar_hgcdte_modulation = bool(
            np.asarray(samples["fit_ar_hgcdte_modulation"]).item()
        )
        state = {
            "sed_weighting": sed_weighting,
            "ar_parameters": np.asarray(samples["ar_parameters"], dtype=float)[0, 0],
            "hgcdte": np.asarray(samples["hgcdte"], dtype=float)[0, 0],
            "incidence_angle_deg": 0.0,
            "ice_k_constant": np.asarray(samples["ice_k_constant"], dtype=float)[0, 0],
            "led_log_offsets": np.asarray(samples["led_log_offsets"], dtype=float)[0],
            "post_reset_tile_log_offset": np.asarray(
                samples["post_reset_tile_log_offset"], dtype=float
            )[0, 0].reshape(-1)[0],
            "fit_ice_n_modulation": fit_ice_n_modulation,
            "fit_ar_hgcdte_modulation": fit_ar_hgcdte_modulation,
        }
        if fit_ice_n_modulation:
            state["ice_n_log_modulation"] = np.asarray(
                samples["ice_n_log_modulation"], dtype=float
            )[0, 0]
        if fit_ar_hgcdte_modulation:
            state["ar_n_log_modulation"] = np.asarray(
                samples["ar_n_log_modulation"], dtype=float
            )[0, 0]
            state["hgcdte_n_log_modulation"] = np.asarray(
                samples["hgcdte_n_log_modulation"], dtype=float
            )[0, 0]
            state["hgcdte_k_log_modulation"] = np.asarray(
                samples["hgcdte_k_log_modulation"], dtype=float
            )[0, 0]
        return state


def fit_epoch(data, map_thickness, state, context, side, lower, upper):
    side_mask = (
        np.asarray(data["days"], dtype=float) < float(data["reset_day"])
        if side == "pre"
        else np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    )
    epoch = _epoch_data(data, side_mask)
    epoch_thickness = np.asarray(map_thickness, dtype=float)[side_mask]
    fit_mask = (epoch_thickness >= lower) & (epoch_thickness <= upper)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError(f"{side} epoch has fewer than three common-range groups")

    predictor = build_prediction_function(
        epoch,
        context,
        fit_ice_n_modulation=state["fit_ice_n_modulation"],
        fit_ice_k_constant=True,
        fit_ar_hgcdte_modulation=state["fit_ar_hgcdte_modulation"],
        fit_led_log_offsets=True,
    )
    days = np.asarray(epoch["days"], dtype=float)
    observed = epoch["sums"][0] / epoch["counts"][0]
    if side == "post":
        observed = observed + float(state["post_reset_tile_log_offset"])
    epoch_start = float(np.min(days))
    ar_parameters = state["ar_parameters"]
    hgcdte = state["hgcdte"]
    fixed_kwargs = {
        "ice_n_log_modulation": state.get("ice_n_log_modulation"),
        "ice_k_constant": state["ice_k_constant"],
        "ar_n_log_modulation": state.get("ar_n_log_modulation"),
        "hgcdte_n_log_modulation": state.get("hgcdte_n_log_modulation"),
        "hgcdte_k_log_modulation": state.get("hgcdte_k_log_modulation"),
        "led_log_offsets": state["led_log_offsets"],
    }

    def predict_from_log_parameters(log_parameters):
        initial = np.exp(float(log_parameters[0]))
        rate = np.exp(float(log_parameters[1]))
        thickness = initial + rate * (days - epoch_start)
        prediction = predictor(
            thickness,
            ar_parameters,
            hgcdte,
            state["incidence_angle_deg"],
            **fixed_kwargs,
        )
        return np.asarray(prediction).reshape(-1), initial, rate

    initial_name = "initial_thickness_log_nm" if side == "pre" else "reset_initial_thickness_log_nm"
    with np.load(state["map_path"], allow_pickle=False) as samples:
        initial_log = float(np.asarray(samples[initial_name]).reshape(-1)[0])
        map_rate = float(np.asarray(samples["mean_rate_nm_day"]).reshape(-1)[0])
    result = least_squares(
        lambda parameters: (
            predict_from_log_parameters(parameters)[0][fit_mask]
            - observed[fit_mask]
        )
        * np.sqrt(epoch["counts"][0][fit_mask])
        / LOG_NOISE_SIGMA,
        x0=np.asarray([initial_log, np.log(max(map_rate, 1.0e-6))]),
        bounds=(np.log([1.0e-6, 1.0e-6]), np.log([500.0, 100.0])),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        diff_step=1.0e-4,
        max_nfev=1000,
    )
    prediction, initial, rate = predict_from_log_parameters(result.x)
    residual = observed[fit_mask] - prediction[fit_mask]
    weighted_rms = np.sqrt(
        np.sum(epoch["counts"][0][fit_mask] * residual**2)
        / np.sum(epoch["counts"][0][fit_mask])
    )
    map_slope = np.polyfit(days[fit_mask], epoch_thickness[fit_mask], 1)[0]
    return {
        "side": side,
        "groups_total": int(days.size),
        "groups_fit": int(np.count_nonzero(fit_mask)),
        "thickness_min_nm": float(np.min(epoch_thickness[fit_mask])),
        "thickness_max_nm": float(np.max(epoch_thickness[fit_mask])),
        "initial_thickness_nm": float(initial),
        "constant_rate_nm_day": float(rate),
        "saved_map_linear_rate_nm_day": float(map_slope),
        "fit_count_weighted_rms_log": float(weighted_rms),
        "optimizer_cost": float(2.0 * result.cost),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
    }


def fit_detector(detector, statistics_path, map_path, lower, upper):
    data = load_tile_statistics(statistics_path)
    state = _load_fixed_state(map_path)
    state["map_path"] = str(Path(map_path) / "posterior_samples.npz")
    with np.load(state["map_path"], allow_pickle=False) as samples:
        map_thickness = np.asarray(samples["ice_thickness"], dtype=float).squeeze()
    context = load_response_context(
        data["sca_names"],
        sed_weighting=state["sed_weighting"],
        qe_response_mode="none",
    )
    rows = []
    for side in ("pre", "post"):
        row = fit_epoch(data, map_thickness, state, context, side, lower, upper)
        row["detector"] = detector
        row["statistics"] = str(statistics_path)
        row["map"] = str(map_path)
        rows.append(row)
    return rows


def main():
    args = parse_args()
    if args.thickness_min < 0.0 or args.thickness_max <= args.thickness_min:
        raise ValueError("Thickness bounds must be positive and ordered")
    rows = []
    for detector, statistics_path, map_path in args.fit:
        rows.extend(
            fit_detector(
                detector,
                Path(statistics_path),
                Path(map_path),
                args.thickness_min,
                args.thickness_max,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "common_range_rate_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['detector']} {row['side']}: "
            f"rate={row['constant_rate_nm_day']:.8g} nm/day, "
            f"initial={row['initial_thickness_nm']:.8g} nm, "
            f"groups={row['groups_fit']}, "
            f"rms={row['fit_count_weighted_rms_log']:.8g}"
        )


if __name__ == "__main__":
    main()
