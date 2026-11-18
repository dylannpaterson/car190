"""Compute per-detector CAR201 and CAR190 comparisons from saved joint fits.

Run from any directory with ``python docs/analyze_wfi_all_detector_metrics.py``.
To analyze a pipeline run, pass ``--run-root`` with its output directory (or
the parent date directory). The script reads saved fits only; it does not rerun
any optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.time import Time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS = ROOT / "20261111/corrected_statistics_4h_thinned"
PINNED_LED_QUADRATURE = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle/"
    "eric_v3_reference/ice_response_model_v3.h5"
)
STATS = DEFAULT_STATS
PRODUCTION_ROOT = STATS / "production"
FIT_NAME = "car190_amber_joint_2day_reader_compatible_current_gray_v2"
EFFECTIVE_AREA_ROOT = ROOT / "external_reference_data/roman_effarea_v8_20240301"
LED_H5 = PINNED_LED_QUADRATURE
LED_FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184")

sys.path.insert(0, str(ROOT))
from docs.paper_run_paths import resolve_pipeline_run

sys.path[:0] = [
    str(ROOT / "wfi_pipeline"),
    str(ROOT / "archive/exploratory/legacy_root_scripts"),
]

import plot_joint_amber_car190_detailed as car190_plotter
import run_amber_student_t_detectors as run
import run_joint_car190_amber_inverse_wavelength_maxent as joint_fit
from experiment_joint_wfi01_fixed_ice import LedModel
from fit_amber_wfi01_student_t import NU, profile_student_locations
from reader_compatible_joint_forward import ReaderCompatiblePhotometryForward
from support_aligned_modulation import effective_area_support


def nmad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def residual_summary_millilog(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(1000.0 * np.sqrt(np.mean(values**2))), float(1000.0 * nmad(values))


def lag_one_correlation(values) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 3:
        return float("nan")
    earlier = values[:-1]
    later = values[1:]
    if np.std(earlier) == 0.0 or np.std(later) == 0.0:
        return float("nan")
    return float(np.corrcoef(earlier, later)[0, 1])


def profile_group_locations(data, prediction, scales, valid):
    pass_count = valid.shape[-1]
    values = (data - prediction).reshape((-1, pass_count))
    group_scales = scales.reshape((-1, pass_count))
    group_valid = valid.reshape((-1, pass_count))
    active = group_valid.sum(axis=1) >= 2
    location = np.zeros(len(values), dtype=float)
    location[active] = profile_student_locations(
        values[active], group_scales[active], group_valid[active], nu=NU
    )[0]
    return location.reshape(data.shape[:2])


def student_nll_per_row(residual, scales, valid) -> float:
    values = np.asarray(residual, dtype=float)[valid]
    sigma = np.asarray(scales, dtype=float)[valid]
    nll = np.log(sigma) + 0.5 * (NU + 1.0) * np.log1p(
        values**2 / (NU * sigma**2)
    )
    return float(nll.mean())


def detector_effective_area(detector: str) -> Path:
    detector_number = int(detector[3:])
    return EFFECTIVE_AREA_ROOT / f"Roman_effarea_v8_SCA{detector_number:02d}_20240301.ecsv"


def detector_fit_root(detector: str) -> Path:
    return PRODUCTION_ROOT / f"{detector.lower()}/relative_response/experiments/{FIT_NAME}"


def compute_detector(detector: str, led_wavelength_nm, led_weights):
    fit_root = detector_fit_root(detector)
    if not fit_root.is_dir():
        raise FileNotFoundError(f"Missing saved joint fit for {detector}: {fit_root}")
    relative_root = PRODUCTION_ROOT / f"{detector.lower()}/relative_response"
    summary = json.loads((fit_root / "summary.json").read_text())
    area_path = detector_effective_area(detector)

    source_ids, seds, raw_magnitude, raw_error, raw_valid, common, frames = (
        joint_fit.raw_common_training_sample(detector)
    )
    with np.load(fit_root / "training_inputs.npz", allow_pickle=False) as archive:
        magnitude = np.asarray(archive["magnitude"], dtype=float)
        error = np.asarray(archive["error"], dtype=float)
        valid = np.asarray(archive["valid"], dtype=bool)
    with np.load(fit_root / "joint_fit.npz", allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=float)
        fit_source_ids = np.asarray(archive["source_ids"]).astype(str)
        fit_valid = np.asarray(archive["stellar_valid"], dtype=bool)
        prediction_with_roll = np.asarray(archive["stellar_prediction_mag"], dtype=float) * 1000.0
    with np.load(fit_root / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
        history_times = np.asarray(archive["times"], dtype=float)
        baseline_history = np.asarray(archive["baseline_thickness_nm"], dtype=float)
        fitted_history = np.asarray(archive["fitted_thickness_nm"], dtype=float)
        shared_parameters = np.asarray(archive["shared_parameters"], dtype=float)
        saved_rsc_chi2 = float(np.asarray(archive["car190_chi2"]).item())

    np.testing.assert_array_equal(source_ids, fit_source_ids)
    np.testing.assert_array_equal(raw_valid, fit_valid)
    np.testing.assert_array_equal(valid, fit_valid)
    np.testing.assert_allclose(magnitude, raw_magnitude, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(error, raw_error, rtol=0, atol=0, equal_nan=True)

    history_root = Path(summary["ice_history_fit_root"])
    optical_cache_root = Path(summary["optical_cache_root"])
    model = joint_fit.model_for_raw_sample(
        detector,
        source_ids,
        seds,
        valid,
        frames,
        area_path,
        history_root,
        optical_cache_root,
    )
    with np.load(history_root / "tile_000/posterior_samples.npz", allow_pickle=False) as archive:
        reset_days = np.asarray(archive["ice_reset_days"], dtype=float)
    reference_mjd = Time(
        run.workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc"
    ).mjd
    pass_days = np.asarray([
        [joint_fit.common_pass_mjd(frames, band, name) - reference_mjd for name in run.PASSES]
        for band in run.FILTERS
    ])
    pass_indices = np.searchsorted(history_times, pass_days)
    np.testing.assert_allclose(history_times[pass_indices], pass_days, rtol=0, atol=1e-10)
    # The RCS-only CAR201 prediction uses the RCS-derived baseline histories;
    # the saved joint prediction below uses the separately refitted histories.
    model.tile_thickness_by_id = baseline_history[:, pass_indices]
    gray_by_tile = np.asarray(summary["reader_gray_log_offsets_by_tile"], dtype=float)
    model.reader_gray_log_by_tile = gray_by_tile[:, np.searchsorted(reset_days, pass_days, side="right")]

    filter_count = len(run.FILTERS)
    pass_count = len(run.PASSES)
    source_count = len(source_ids)
    mode = model.mode_count
    zero_points = parameters[2 * mode :].reshape(filter_count, model.zero_point_group_count)
    roll_magnitude = np.zeros((source_count, filter_count, pass_count), dtype=float)
    for pass_index, group in enumerate(model.pass_zero_point_group_ids):
        if group:
            roll_magnitude[:, :, pass_index] = (
                -zero_points[:, group - 1] / run.workflow.LOG_FLUX_PER_MAG
            )[None, :]

    reference = np.zeros(magnitude.shape[:2], dtype=float)
    for source in range(source_count):
        for filter_index in range(filter_count):
            values = magnitude[source, filter_index, valid[source, filter_index]]
            if values.size:
                reference[source, filter_index] = np.median(values)
    observed = np.where(valid, (magnitude - reference[:, :, None]) * 1000.0, 0.0)
    roll_adjusted = np.where(valid, observed - roll_magnitude * 1000.0, 0.0)
    joint_no_roll = prediction_with_roll - roll_magnitude * 1000.0
    extra_scale = float(summary["extra_student_t_scale_mmag"])
    scales = np.where(valid, np.hypot(error * 1000.0, extra_scale), 1.0)

    forward = ReaderCompatiblePhotometryForward(model)
    baseline_log, _ = forward.evaluate(np.zeros(2 * mode, dtype=float))
    rcs_only_prediction = -1000.0 * baseline_log / run.workflow.LOG_FLUX_PER_MAG
    no_ice_prediction = np.zeros_like(roll_adjusted)

    model_predictions = {
        "no_ice": no_ice_prediction,
        "rcs_only_tmm": rcs_only_prediction,
        "joint": joint_no_roll,
    }
    residuals = {}
    for name, prediction in model_predictions.items():
        location = profile_group_locations(roll_adjusted, prediction, scales, valid)
        residuals[name] = np.where(valid, roll_adjusted - prediction - location[:, :, None], 0.0)

    photometry_rows = []
    for filter_index, band in enumerate(run.FILTERS):
        use = valid[:, filter_index, :]
        photometry_rows.append({
            "detector": detector,
            "filter": band,
            "n_stars": int(np.count_nonzero(use.any(axis=1))),
            "n_measurements": int(use.sum()),
            "nmad_no_ice_mmag": nmad(residuals["no_ice"][:, filter_index, :][use]),
            "nmad_rcs_only_tmm_mmag": nmad(residuals["rcs_only_tmm"][:, filter_index, :][use]),
            "nmad_joint_mmag": nmad(residuals["joint"][:, filter_index, :][use]),
            "mean_student_nll_no_ice": student_nll_per_row(
                residuals["no_ice"][:, filter_index, :], scales[:, filter_index, :], use
            ),
            "mean_student_nll_rcs_only": student_nll_per_row(
                residuals["rcs_only_tmm"][:, filter_index, :], scales[:, filter_index, :], use
            ),
            "mean_student_nll_joint": student_nll_per_row(
                residuals["joint"][:, filter_index, :], scales[:, filter_index, :], use
            ),
        })
    for row in photometry_rows:
        row["delta_mean_log_likelihood_joint_minus_rcs_only_per_measurement"] = (
            row["mean_student_nll_rcs_only"] - row["mean_student_nll_joint"]
        )

    led_model = LedModel(
        effective_area_support(area_path),
        detector=detector,
        relative_response_root=relative_root,
        optical_cache_root=optical_cache_root,
        led_wavelength_nm=led_wavelength_nm,
        led_weights=led_weights,
    )
    joint_rows, standardized_joint_residual = car190_plotter.predict_car190(
        led_model, shared_parameters, history_times, fitted_history
    )
    np.testing.assert_allclose(
        np.sum(standardized_joint_residual**2), saved_rsc_chi2, rtol=2e-7, atol=2e-4
    )
    rcs_rows = []
    all_rcs_only_residuals = []
    all_joint_residuals = []
    detector_rcs_only_lag_one = []
    detector_joint_lag_one = []
    for led_number, tile_items in enumerate(led_model.bands, start=1):
        rcs_only_chi2 = 0.0
        joint_chi2 = 0.0
        group_count = 0
        rcs_only_residuals = []
        joint_residuals = []
        rcs_only_lag_one = []
        joint_lag_one = []
        nonreference_group_count = 0
        for tile, item in enumerate(tile_items):
            rcs_only_prediction = np.asarray(item[6], dtype=float)
            observed_groups = np.asarray(item[7], dtype=float)
            sigma = np.asarray(item[8], dtype=float)
            days = np.asarray(item[9], dtype=float)
            rows = joint_rows.loc[
                (joint_rows["led"] == led_number) & (joint_rows["tile"] == tile)
            ].sort_values("day")
            np.testing.assert_allclose(rows["day"].to_numpy(float), days, rtol=0, atol=1e-10)
            joint_prediction = rows["predicted"].to_numpy(float)
            np.testing.assert_allclose(rows["observed"].to_numpy(float), observed_groups, rtol=0, atol=1e-10)
            rcs_only_chi2 += float(np.sum(((rcs_only_prediction - observed_groups) / sigma) ** 2))
            joint_chi2 += float(np.sum(((joint_prediction - observed_groups) / sigma) ** 2))
            nonreference = days > days[0] + 1.0e-12
            rcs_residual = rcs_only_prediction[nonreference] - observed_groups[nonreference]
            joint_residual = joint_prediction[nonreference] - observed_groups[nonreference]
            rcs_only_residuals.append(rcs_residual)
            joint_residuals.append(joint_residual)
            rcs_only_lag_one.append(lag_one_correlation(rcs_residual))
            joint_lag_one.append(lag_one_correlation(joint_residual))
            nonreference_group_count += int(nonreference.sum())
            group_count += len(days)
        rcs_only_rmse, rcs_only_nmad = residual_summary_millilog(
            np.concatenate(rcs_only_residuals)
        )
        joint_rmse, joint_nmad = residual_summary_millilog(
            np.concatenate(joint_residuals)
        )
        all_rcs_only_residuals.append(np.concatenate(rcs_only_residuals))
        all_joint_residuals.append(np.concatenate(joint_residuals))
        detector_rcs_only_lag_one.extend(rcs_only_lag_one)
        detector_joint_lag_one.extend(joint_lag_one)
        rcs_rows.append({
            "detector": detector,
            "led": led_number,
            "filter": LED_FILTERS[led_number - 1],
            "n_tile_groups": group_count,
            "n_nonreference_tile_groups": int(sum(values.size for values in rcs_only_residuals)),
            "rcs_only_weighted_chi2_per_group": rcs_only_chi2 / nonreference_group_count,
            "joint_weighted_chi2_per_group": joint_chi2 / nonreference_group_count,
            "rcs_only_group_mean_rmse_millilog": rcs_only_rmse,
            "rcs_only_group_mean_nmad_millilog": rcs_only_nmad,
            "joint_group_mean_rmse_millilog": joint_rmse,
            "joint_group_mean_nmad_millilog": joint_nmad,
            "rcs_only_median_tile_lag1_residual_correlation": float(
                np.nanmedian(rcs_only_lag_one)
            ),
            "joint_median_tile_lag1_residual_correlation": float(
                np.nanmedian(joint_lag_one)
            ),
            "delta_mean_log_likelihood_joint_minus_rcs_only_per_group": (
                0.5 * (rcs_only_chi2 - joint_chi2) / nonreference_group_count
            ),
        })
    detector_rcs_summary = {}
    for model_name, residual_parts in (
        ("rcs_only", all_rcs_only_residuals),
        ("joint", all_joint_residuals),
    ):
        rmse, robust_sigma = residual_summary_millilog(np.concatenate(residual_parts))
        detector_rcs_summary[f"{model_name}_all_led_group_mean_rmse_millilog"] = rmse
        detector_rcs_summary[f"{model_name}_all_led_group_mean_nmad_millilog"] = robust_sigma
    detector_rcs_summary["rcs_only_median_tile_lag1_residual_correlation"] = float(
        np.nanmedian(detector_rcs_only_lag_one)
    )
    detector_rcs_summary["joint_median_tile_lag1_residual_correlation"] = float(
        np.nanmedian(detector_joint_lag_one)
    )
    return photometry_rows, rcs_rows, detector_rcs_summary


def main() -> None:
    global STATS, PRODUCTION_ROOT, LED_H5

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="pipeline output root or date root; defaults to the validated 20261111 run",
    )
    parser.add_argument(
        "--detector",
        action="append",
        choices=run.workflow.DETECTORS,
        help="analyze this detector (repeatable); default analyzes all saved detectors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the result CSVs (defaults to docs/ or <run-root>/analysis)",
    )
    args = parser.parse_args()

    run_paths = None
    if args.run_root is not None:
        run_paths = resolve_pipeline_run(args.run_root)
        STATS = run_paths.statistics_dir
        PRODUCTION_ROOT = run_paths.production_dir
        run_specific_led_h5 = (
            PRODUCTION_ROOT
            / "wavelength_response_bundle/eric_v3_reference/ice_response_model_v3.h5"
        )
        LED_H5 = run_specific_led_h5 if run_specific_led_h5.is_file() else PINNED_LED_QUADRATURE
        print("Using pipeline run:", run_paths.run_root)

    detectors = args.detector or list(run.workflow.DETECTORS)
    output_dir = args.output_dir
    if output_dir is None:
        if run_paths is not None:
            output_dir = run_paths.run_root / "analysis"
        elif args.detector:
            output_dir = ROOT / "docs/detector_analysis"
        else:
            output_dir = ROOT / "docs"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LED_H5.is_file():
        raise FileNotFoundError(f"Missing LED quadrature reference: {LED_H5}")
    with h5py.File(LED_H5, "r") as archive:
        led_wavelength_nm = np.asarray(archive["led_bands/wavelength_nm"][()], dtype=float)
        led_weights = np.asarray(archive["led_bands/weight"][()], dtype=float)

    all_photometry_rows = []
    all_rcs_rows = []
    detector_summary = []
    for detector in detectors:
        photometry_rows, rcs_rows, detector_rcs_summary = compute_detector(
            detector, led_wavelength_nm, led_weights
        )
        all_photometry_rows.extend(photometry_rows)
        all_rcs_rows.extend(rcs_rows)
        measurements = sum(row["n_measurements"] for row in photometry_rows)
        loglike_gain = sum(
            row["delta_mean_log_likelihood_joint_minus_rcs_only_per_measurement"]
            * row["n_measurements"]
            for row in photometry_rows
        )
        rcs_groups = sum(row["n_nonreference_tile_groups"] for row in rcs_rows)
        rcs_loglike_gain = sum(
            row["delta_mean_log_likelihood_joint_minus_rcs_only_per_group"]
            * row["n_nonreference_tile_groups"]
            for row in rcs_rows
        )
        nmad_rcs = np.asarray([row["nmad_rcs_only_tmm_mmag"] for row in photometry_rows])
        nmad_joint = np.asarray([row["nmad_joint_mmag"] for row in photometry_rows])
        detector_summary.append({
            "detector": detector,
            **detector_rcs_summary,
            "car201_measurements": measurements,
            "car201_filters_joint_nmad_improved": int(np.count_nonzero(nmad_joint < nmad_rcs)),
            "car201_median_filter_nmad_rcs_only_mmag": float(np.median(nmad_rcs)),
            "car201_median_filter_nmad_joint_mmag": float(np.median(nmad_joint)),
            "car201_delta_mean_log_likelihood_joint_minus_rcs_only_per_measurement": (
                loglike_gain / measurements
            ),
            "rcs_tile_groups": rcs_groups,
            "rcs_weighted_chi2_per_group_rcs_only": float(np.average(
                [row["rcs_only_weighted_chi2_per_group"] for row in rcs_rows],
                weights=[row["n_nonreference_tile_groups"] for row in rcs_rows],
            )),
            "rcs_weighted_chi2_per_group_joint": float(np.average(
                [row["joint_weighted_chi2_per_group"] for row in rcs_rows],
                weights=[row["n_nonreference_tile_groups"] for row in rcs_rows],
            )),
            "rcs_delta_mean_log_likelihood_joint_minus_rcs_only_per_group": (
                rcs_loglike_gain / rcs_groups
            ),
        })
        print(f"Finished {detector}: {measurements} CAR201 measurements, {rcs_groups} RCS groups")

    summary = pd.DataFrame(detector_summary).sort_values("detector")
    photometry = pd.DataFrame(all_photometry_rows).sort_values(["detector", "filter"])
    rcs = pd.DataFrame(all_rcs_rows).sort_values(["detector", "led"])
    summary_path = output_dir / "wfi_all_detector_model_comparison.csv"
    photometry_path = output_dir / "wfi_all_detector_car201_by_filter.csv"
    rcs_path = output_dir / "wfi_all_detector_rcs_by_led.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8g")
    photometry.to_csv(photometry_path, index=False, float_format="%.8g")
    rcs.to_csv(rcs_path, index=False, float_format="%.8g")
    print("\nPer-detector summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print("Wrote", summary_path)
    print("Wrote", photometry_path)
    print("Wrote", rcs_path)


if __name__ == "__main__":
    main()
