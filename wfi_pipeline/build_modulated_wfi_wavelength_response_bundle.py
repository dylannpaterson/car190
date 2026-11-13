"""Build a portable modulation-aware WFI wavelength-response bundle.

The existing bundle is required as the source of its shared grids and tile
geometry. Relative response surfaces are recomputed from the current per-tile
TMM fits, with each detector's fitted complex-reflection modulation applied
before the ice layer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Keep the heavy per-tile TMM/SVD build from spawning a full machine's worth of
# BLAS workers. The build is intentionally sequential over explicit detectors.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np
import pandas as pd

import mock_fit_wfi_star_reflectance_modulation as mock_model
import wfi_relative_response as relative_response
from build_wfi_wavelength_response_bundle import (
    fit_root_for_detector,
    load_fit_trajectory,
    string_array,
)
from fit_car201_zane_star_reflectance_modulation import (
    DEFAULT_PRODUCTION_ROOT,
)
from wfi_wavelength_response_bundle import (
    MODULATED_BUNDLE_VERSION,
    bundle_summary,
    load_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_BUNDLE = (
    ROOT
    / "20261102/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/wfi_all_2nm_rank16.npz"
)
DEFAULT_FIT_ROOT = DEFAULT_PRODUCTION_ROOT
DEFAULT_OUTPUT = (
    ROOT
    / "20261105/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/"
    "wfi_all_2nm_rank16_car201_ep3ep4_modulated_v1.npz"
)
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
JOINT_MODULATION_FIT = "joint-car190-amber-2day"
DEFAULT_JOINT_FIT_NAME = "car190_amber_joint_per_tile_2day_rate_test"
DEFAULT_JOINT_ICE_FIT_NAME = "car190_amber_joint_fixed_ice_full_reflection_test"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joint_experiment_dir(root, detector, name, fit_subdir=None):
    detector_root = Path(root) / detector.lower()
    if fit_subdir is not None:
        candidate = detector_root / fit_subdir / "experiments" / name
        return candidate
    candidates = (
        detector_root / "relative_response/experiments" / name,
        detector_root / "spline_relative_response_normal/experiments" / name,
    )
    return next((path for path in candidates if path.exists()), candidates[-1])


def _load_joint_detector_modulation(
    modulation_root,
    detector,
    wavelengths_nm,
    joint_fit_name=DEFAULT_JOINT_FIT_NAME,
    joint_fit_subdir=None,
):
    """Load joint CAR190/Amber spectral coefficients on a requested wavelength grid."""
    import run_amber_student_t_detectors as amber
    from experiment_joint_wfi01_fixed_ice import full_basis
    from support_aligned_modulation import effective_area_support, gap_basis

    joint_dir = _joint_experiment_dir(
        modulation_root, detector, joint_fit_name, fit_subdir=joint_fit_subdir
    )
    fixed_dir = _joint_experiment_dir(
        modulation_root, detector, DEFAULT_JOINT_ICE_FIT_NAME,
        fit_subdir=joint_fit_subdir,
    )
    joint_summary_path = joint_dir / "summary.json"
    joint_archive_path = joint_dir / "joint_weekly_ice_fit.npz"
    fixed_summary_path = fixed_dir / "joint_fit_summary.json"
    joint_summary = json.loads(joint_summary_path.read_text())
    rate_spacing = joint_summary.get(
        "rate_control_spacing_days", joint_summary.get("ice_rate_control_spacing_days")
    )
    if (joint_summary.get("detector") != detector
            or not joint_summary.get("optimizer_success")
            or rate_spacing != 2.0
            or not joint_summary.get("ice_histories_refitted", True)):
        raise ValueError(f"Missing converged detector-specific 2-day fit: {joint_summary_path}")
    # Current reader-compatible runs store the fitted spectral coefficients in
    # their own joint_fit.npz and record the exact optical cache in summary.json.
    # Historical experiments split the fixed-ice spectral fit from the joint
    # ice-history fit; preserve that route only for those older products.
    reader_compatible_fit = joint_summary.get("response_convention") == "eric-reader"
    fixed_archive_path = fixed_dir / "joint_fit.npz"
    has_fixed_fit = (
        not reader_compatible_fit
        and fixed_summary_path.is_file()
        and fixed_archive_path.is_file()
    )
    parameter_archive_path = joint_archive_path if has_fixed_fit else joint_dir / "joint_fit.npz"
    if has_fixed_fit:
        fixed_summary = json.loads(fixed_summary_path.read_text())
        if fixed_summary.get("detector") != detector or not fixed_summary.get("optimizer_success"):
            raise ValueError(f"Missing converged detector-specific fixed-ice fit: {fixed_summary_path}")
        with np.load(fixed_archive_path, allow_pickle=False) as archive:
            wavelength_key = "wavelength_nm" if "wavelength_nm" in archive else "wavelengths_nm"
            fit_wavelengths = np.asarray(archive[wavelength_key], dtype=float)
            fit_basis = np.asarray(archive["basis"], dtype=float)
        with np.load(joint_archive_path, allow_pickle=False) as archive:
            theta = np.asarray(archive["shared_parameters"], dtype=float)
        car190_observations = int(fixed_summary["LED_observations"])
        optical_root = None
    else:
        with np.load(parameter_archive_path, allow_pickle=False) as archive:
            wavelength_key = "wavelength_nm" if "wavelength_nm" in archive else "wavelengths_nm"
            fit_wavelengths = np.asarray(archive[wavelength_key], dtype=float)
            fit_basis = np.asarray(archive["basis"], dtype=float)
            theta = np.asarray(archive["parameters"], dtype=float)
        car190_observations = int(joint_summary["CAR190_observations"])
        optical_root = joint_summary.get("optical_cache_root")

    area, _ = amber.canonical_configuration(detector)
    bounds = effective_area_support(area)
    fit_basis_expected = full_basis(fit_wavelengths, bounds)[0]
    np.testing.assert_allclose(fit_basis, fit_basis_expected, rtol=0, atol=2e-12)
    basis = full_basis(np.asarray(wavelengths_nm, dtype=float), bounds)[0]
    mode_count = basis.shape[0]
    if theta.size < 2 * mode_count:
        raise ValueError(f"Joint fit has too few optical parameters for {detector}")
    log_amplitudes = theta[:mode_count]
    phases = theta[mode_count:2 * mode_count]
    ratio = np.exp(log_amplitudes @ basis + 1j * (phases @ basis))

    gap_basis_at_fit, knots = gap_basis(fit_wavelengths, bounds)
    del gap_basis_at_fit
    centers = np.concatenate([
        (knots[:-1] + knots[1:]) / 2.0,
        np.asarray([bounds[0], bounds[1]]),
        knots[1:-1],
    ])
    names = ([f"joint_gap_{index + 1}" for index in range(len(knots) - 1)]
             + ["free_blue_endpoint", "free_red_endpoint"]
             + [f"free_LED_{index + 1}" for index in range(6)])
    if len(names) != mode_count or centers.size != mode_count:
        raise ValueError(f"Unexpected joint spectral basis dimensions for {detector}")

    led_basis = full_basis(mock_model.LED_CENTERS_NM, bounds)[0]
    led_ratio = np.exp(log_amplitudes @ led_basis + 1j * (phases @ led_basis))
    led_anchor_deviation = float(np.max(np.abs(led_ratio - 1.0)))
    if optical_root is None:
        old_dir, _ = amber.paths(detector)
        old_summary = json.loads((old_dir / "car201_actual_star_modulation_fit_summary.json").read_text())
        optical_root = old_summary["ice_history"]["optical_state_source"]
        maxent_regularization = old_summary["modulation"]["maxent_regularization"]
        phase_regularization = old_summary["modulation"]["phase_smoothness_regularization"]
        reciprocal_maxent_weight = float(maxent_regularization["weight"])
        curvature_weight = None
        curvature_grid_points = None
    else:
        maxent_regularization = {
            "functional": joint_summary["spectral_maxent"],
            "weight": float(joint_summary["reciprocal_maxent_weight"]),
        }
        phase_regularization = {"weight": float(joint_summary["phase_smoothness_weight"])}
        reciprocal_maxent_weight = float(joint_summary["reciprocal_maxent_weight"])
        curvature_weight = float(joint_summary["curvature_weight"])
        curvature_grid_points = int(joint_summary["curvature_grid_points"])
    modulation_summary = {
        "detector": detector,
        "pass_set": "ep3v2_ep4",
        "optimizer_success": True,
        "ice_history": {
            "per_measurement_tile_position_available": True,
            "optical_state_source": optical_root,
        },
        "modulation": {
            "basis": "joint detector-specific support-aligned gaps, endpoints, and six LED values",
            "maxent_regularization": maxent_regularization,
            "curvature_regularization": (
                {"weight": curvature_weight, "grid_points": curvature_grid_points}
                if curvature_weight is not None else None
            ),
            "phase_smoothness_regularization": phase_regularization,
        },
    }
    provenance = {
        "photometry": "Amber CAR201 + CAR190",
        "amber_photometry_dataset": "CAR201",
        "amber_input": str(amber.AMBER.resolve()),
        "likelihood": "joint: Amber Student-t nu=4 plus CAR190 production Gaussian",
        "nu": 4.0,
        "fit_kind": JOINT_MODULATION_FIT,
        "joint_fit_directory": str(joint_dir.resolve()),
        "joint_fit_summary_sha256": _sha256(joint_summary_path),
        "joint_fit_archive": str(parameter_archive_path.resolve()),
        "joint_fit_archive_sha256": _sha256(parameter_archive_path),
        "joint_ice_history_archive": str(joint_archive_path.resolve()),
        "joint_ice_history_archive_sha256": _sha256(joint_archive_path),
        "rate_control_spacing_days": 2.0,
        "tile_specific_ice_history_refitted": True,
        "ice_parameters_per_tile": int(
            joint_summary.get("ice_parameters_per_tile", joint_summary["total_ice_parameters"] // 256)
        ),
        "total_ice_parameters": joint_summary["total_ice_parameters"],
        "amber_stars": int(joint_summary.get("stellar_stars", joint_summary.get("training_stars", 0))),
        "amber_measurements": int(
            joint_summary.get("stellar_measurements", joint_summary.get("training_measurements", 0))
        ),
        "car190_observations": car190_observations,
        "effective_area_file": str(area.resolve()),
        "effective_area_sha256": _sha256(area),
        "effective_area_provenance": json.loads(area.with_suffix(".provenance.json").read_text()),
        "modulation_zero_at_led_centers": False,
        "led_center_max_abs_complex_deviation": led_anchor_deviation,
        "reciprocal_maxent_weight": reciprocal_maxent_weight,
        "curvature_weight": curvature_weight,
        "curvature_grid_points": curvature_grid_points,
        "roll_zeropoints_embedded": False,
        "scatter_scale_applied_to_optical_response": False,
    }
    return {
        "output_dir": joint_dir,
        "summary_path": joint_summary_path,
        "parameters_path": parameter_archive_path,
        "summary": modulation_summary,
        "basis_name": modulation_summary["modulation"]["basis"],
        "basis_names": names,
        "basis_centers_nm": centers,
        "log_amplitudes": log_amplitudes,
        "phases": phases,
        "ratio": ratio,
        "anchor_error": led_anchor_deviation,
        "fit_provenance": provenance,
        "summary_sha256": _sha256(joint_summary_path),
        "parameters_sha256": _sha256(joint_archive_path),
        "joint_ice_archive_path": joint_archive_path,
        "joint_ice_summary": joint_summary,
    }


def _load_joint_ice_history(modulation_root, detector, modulation_item=None):
    """Load 2-day joint thicknesses and finite-difference rates at bundle times."""
    from experiment_joint_wfi01_weekly_ice import RESETS
    from per_tile_rate_modulation import PerTileRateModulation

    item = modulation_item or _load_joint_detector_modulation(
        modulation_root, detector, np.asarray([500.0])
    )
    joint_path = item["joint_ice_archive_path"]
    with np.load(joint_path, allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=float)
        baseline = np.asarray(archive["baseline_thickness_nm"], dtype=float)
        thickness = np.asarray(archive["fitted_thickness_nm"], dtype=float)
        parameters = np.asarray(archive["ice_parameters"], dtype=float)
    if baseline.shape != thickness.shape or baseline.shape != (256, times.size):
        raise ValueError(f"Invalid joint ice trajectory dimensions in {joint_path}")
    trajectory = PerTileRateModulation(times, baseline, RESETS, spacing_days=2.0)
    if parameters.shape != (256, trajectory.parameter_count):
        raise ValueError(f"Invalid joint ice parameter dimensions in {joint_path}")
    np.testing.assert_allclose(trajectory.evaluate(parameters), thickness, rtol=0, atol=2e-8)
    rates = np.zeros_like(thickness)
    for spec in trajectory.specs:
        indices = spec["indices"]
        slopes = np.diff(thickness[:, indices], axis=1) / np.diff(times[indices])[None, :]
        if np.any(slopes < -1e-7):
            raise ValueError(f"Joint fitted ice trajectory decreases in a growth stage: {joint_path}")
        rates[:, indices] = np.concatenate([slopes, slopes[:, -1:]], axis=1)
    summary = dict(item["joint_ice_summary"])
    summary.setdefault("reset_days", RESETS.tolist())
    summary.setdefault("ice_parameters_per_tile", int(parameters.shape[1]))
    summary.setdefault("total_ice_parameters", int(parameters.size))
    summary.setdefault("rate_control_spacing_days", 2.0)
    return {
        "times": times,
        "thickness": thickness,
        "rates": rates,
        "archive_path": joint_path,
        "summary": summary,
    }


def _sample_joint_tile_ice(joint_history, tile_id, source_days):
    """Sample the joint fit onto the bundle's existing CAR190 observation axis."""
    times = joint_history["times"]
    indices = np.searchsorted(times, np.asarray(source_days, dtype=float))
    if (np.any(indices >= times.size)
            or not np.allclose(times[indices], source_days, rtol=0.0, atol=1.0e-10)):
        raise ValueError(f"Joint ice history lacks source observation times in {joint_history['archive_path']}")
    if not 0 <= int(tile_id) < joint_history["thickness"].shape[0]:
        raise ValueError(f"Tile {tile_id} is outside {joint_history['archive_path']}")
    return (joint_history["thickness"][int(tile_id), indices],
            joint_history["rates"][int(tile_id), indices])


def _modulation_dir(modulation_root, detector):
    return (
        Path(modulation_root)
        / detector.lower()
        / "spline_relative_response_normal/experiments"
        / f"car201_zane_{detector.lower()}_ep3v2_ep4_faint_v2_tile_matched"
    )


def _load_detector_modulation(
    modulation_root,
    detector,
    wavelengths_nm,
    fit_kind="zane",
    joint_fit_name=DEFAULT_JOINT_FIT_NAME,
    joint_fit_subdir=None,
):
    if fit_kind == JOINT_MODULATION_FIT:
        return _load_joint_detector_modulation(
            modulation_root, detector, wavelengths_nm,
            joint_fit_name=joint_fit_name,
            joint_fit_subdir=joint_fit_subdir,
        )
    output_dir = _modulation_dir(modulation_root, detector)
    masked_fit = fit_kind == "amber-student-t-strong-red-led-mask"
    support_fit = fit_kind == "amber-student-t-support-aligned" or masked_fit
    amber_fit = fit_kind in ("amber-student-t", "amber-student-t-detector-effective-area") or support_fit
    if amber_fit:
        output_dir = output_dir.with_name(
            f"car201_amber_{detector.lower()}_ep3v2_ep4_common_zane_quality_student_t_nu4"
        )
        if fit_kind == "amber-student-t-detector-effective-area" or support_fit:
            output_dir = output_dir.with_name(output_dir.name + f"_sca{int(detector[3:]):02d}_effective_area")
        if support_fit:
            output_dir = output_dir.with_name(output_dir.name + "_support_aligned_free_endpoints_bound0p60")
        if masked_fit:
            output_dir = output_dir.with_name(output_dir.name + "_soft_led_sed_red_power_0p05")
        summary_path = output_dir / ("production_fit_summary.json" if masked_fit else "fit_summary.json")
        parameters_path = output_dir / ("modulation_parameters.csv" if support_fit else "student_t_modulation_parameters.csv")
    else:
        summary_path = output_dir / "car201_actual_star_modulation_fit_summary.json"
        parameters_path = output_dir / "car201_actual_star_modulation_parameters.csv"
    with summary_path.open() as handle:
        source_summary = json.load(handle)
    fit_provenance = {"photometry": "Zane", "likelihood": "Gaussian", "fit_kind": fit_kind}
    if amber_fit:
        if (source_summary.get("detector") != detector
            or source_summary.get("photometry") != "Amber"
            or source_summary.get("likelihood") != "Student-t"
            or source_summary.get("nu") != 4.0
            or not source_summary.get("optimizer_success", False)):
            raise ValueError(f"Unexpected or unsuccessful Amber Student-t fit in {summary_path}")
        # WFI01 was the initial verified experiment and names its fixed optical
        # inputs through the associated Gaussian-comparison summary. The newer
        # detector summaries carry these paths directly.
        optical_root = source_summary.get("optical_state_source")
        photometry_path = source_summary.get("amber_input")
        if support_fit:
            if masked_fit:
                prior_support_dir = output_dir.with_name(output_dir.name.removesuffix("_soft_led_sed_red_power_0p05"))
                previous_dir = prior_support_dir.with_name(prior_support_dir.name.removesuffix("_support_aligned_free_endpoints_bound0p60"))
            else:
                previous_dir = output_dir.with_name(output_dir.name.removesuffix("_support_aligned_free_endpoints_bound0p60"))
            previous = json.loads((previous_dir / "fit_summary.json").read_text())
            optical_root = previous["optical_state_source"]
            photometry_path = previous["amber_input"]
            if (not source_summary.get("endpoint_values_free") or source_summary.get("LED_values_free")
                or source_summary.get("endpoint_amplitude_bound") != .6):
                raise ValueError(f"Incorrect support-aligned endpoint configuration: {summary_path}")
        if optical_root is None:
            comparison_dir = Path(source_summary["gaussian_comparison_directory"])
            comparison = json.loads((comparison_dir / "comparison_summary.json").read_text())
            source_fit_dir = Path(comparison["frozen_model_source"])
            reference_summary = json.loads(
                (source_fit_dir / "car201_actual_star_modulation_fit_summary.json").read_text()
            )
            optical_root = reference_summary["ice_history"]["optical_state_source"]
            photometry_path = comparison["amber_input"]
        summary = {
            "detector": detector,
            "pass_set": "ep3v2_ep4",
            "ice_history": {"per_measurement_tile_position_available": True,
                            "optical_state_source": optical_root},
            "modulation": {
                "basis": "per-gap",
                "maxent_regularization": {"weight": source_summary["maxent_weight"]},
                "phase_smoothness_regularization": {"weight": source_summary["phase_weight"]},
            },
        }
        fit_provenance = {
            "photometry": "Amber", "likelihood": "Student-t", "nu": source_summary["nu"],
            "photometry_path": photometry_path,
            "fit_kind": fit_kind,
            "extra_student_t_scale_mmag": source_summary["extra_student_t_scale_mmag"],
            "used_stars": source_summary["used_stars"],
            "used_observations": source_summary["used_observations"],
            "selection": source_summary.get("selection", "Common Amber/Zane measurements passing Zane saved likelihood mask"),
            "scatter_scale_applied_to_optical_response": False,
        }
        if fit_kind == "amber-student-t-detector-effective-area" or support_fit:
            area = Path(source_summary["effective_area_file"])
            expected_name = f"Roman_effarea_v8_SCA{int(detector[3:]):02d}_20240301.ecsv"
            if area.name != expected_name or _sha256(area) != source_summary["effective_area_sha256"]:
                raise ValueError(f"Incorrect or changed detector-specific effective-area input for {detector}")
            provenance = source_summary.get("effective_area_provenance")
            if provenance is None:
                provenance = json.loads(area.with_suffix(".provenance.json").read_text())
            fit_provenance.update({"effective_area_file": str(area.resolve()),
                                   "effective_area_sha256": source_summary["effective_area_sha256"],
                                   "effective_area_provenance": provenance,
                                   "effective_area_applied_to_exported_response": False})
            if support_fit:
                from support_aligned_modulation import effective_area_support
                bounds = effective_area_support(area)
                if not np.allclose(bounds, source_summary["modulation_boundaries_nm"], rtol=0, atol=1e-9):
                    raise ValueError(f"Effective-area support differs from fitted endpoints for {detector}")
                fit_provenance.update({"modulation_boundaries_nm": list(bounds),
                                       "endpoint_log_amplitude_bound": .6,
                                       "parameters_at_bounds": source_summary["parameters_at_bounds"]})
                if masked_fit:
                    if source_summary.get("LED_sed_red_exponent") != .05:
                        raise ValueError("Incorrect strong-red LED SED exponent")
                    led_sed=Path(source_summary["led_sed_file"])
                    if _sha256(led_sed)!=source_summary["led_sed_sha256"]:
                        raise ValueError("LED SED file changed since the fit")
                    fit_provenance.update({"LED_sed_red_exponent": .05,
                        "LED_sed_mask": source_summary["LED_sed_mask"],
                        "led_sed_file":str(led_sed),"led_sed_sha256":source_summary["led_sed_sha256"],
                        "LED_integrated_response_guaranteed_unchanged":False})
    else:
        summary = source_summary
    if summary.get("detector") != detector or summary.get("pass_set") != "ep3v2_ep4":
        raise ValueError(f"Unexpected modulation-fit identity in {summary_path}")
    if not summary.get("ice_history", {}).get(
        "per_measurement_tile_position_available", False
    ):
        raise ValueError(f"{detector} modulation fit is not tile-position mapped")

    basis_name = summary["modulation"]["basis"]
    if support_fit:
        from support_aligned_modulation import gap_basis
        from experiment_free_led_modulation import free_value_basis
        gap, knots = gap_basis(wavelengths_nm, bounds)
        extra, *_ = free_value_basis(wavelengths_nm, boundaries=bounds)
        basis = np.concatenate([gap, extra[[0, -1]]])
        actual_names = [f"gap_{left:.6f}_{right:.6f}nm" for left, right in zip(knots[:-1], knots[1:])]
        actual_names += [f"free_value_{bounds[0]:.6f}nm", f"free_value_{bounds[1]:.6f}nm"]
        basis_names = [f"gap_{i+1}" for i in range(7)] + ["free_blue_endpoint", "free_red_endpoint"]
        basis_centers_nm = np.concatenate([(knots[:-1]+knots[1:])/2, bounds])
        parameters = pd.read_csv(parameters_path).set_index("mode").loc[actual_names]
        led_gap, _ = gap_basis(mock_model.LED_CENTERS_NM, bounds)
        led_extra, *_ = free_value_basis(mock_model.LED_CENTERS_NM, boundaries=bounds)
        led_basis = np.concatenate([led_gap, led_extra[[0, -1]]])
        if masked_fit:
            from experiment_led_sed_soft_mask import sed_mask
            basis = basis * sed_mask(wavelengths_nm, red_exponent=.05)[None, :]
            led_basis = led_basis * sed_mask(mock_model.LED_CENTERS_NM, red_exponent=.05)[None, :]
    else:
        basis, basis_names, basis_centers_nm, *_ = mock_model.modulation_basis(
            wavelengths_nm, grouping=basis_name
        )
        parameters = pd.read_csv(parameters_path).set_index("mode").loc[basis_names]
        led_basis, *_ = mock_model.modulation_basis(mock_model.LED_CENTERS_NM, grouping=basis_name)
    log_amplitudes = parameters["fit_log_amplitude"].to_numpy(dtype=float)
    phases = parameters["fit_phase_rad"].to_numpy(dtype=float)
    log_amplitude_curve = log_amplitudes @ basis
    phase_curve = phases @ basis
    modulation_ratio = np.exp(log_amplitude_curve + 1j * phase_curve)

    led_ratio = np.exp(
        log_amplitudes @ led_basis + 1j * (phases @ led_basis)
    )
    anchor_error = float(np.max(np.abs(led_ratio - 1.0)))
    if anchor_error > 1.0e-12:
        raise ValueError(
            f"{detector} modulation is not zero at its LED anchors: {anchor_error:g}"
        )
    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "parameters_path": parameters_path,
        "summary": summary,
        "basis_name": basis_name,
        "basis_names": basis_names,
        "basis_centers_nm": basis_centers_nm,
        "log_amplitudes": log_amplitudes,
        "phases": phases,
        "ratio": modulation_ratio,
        "anchor_error": anchor_error,
        "fit_provenance": fit_provenance,
        "summary_sha256": _sha256(summary_path),
        "parameters_sha256": _sha256(parameters_path),
    }


def _compress_response(relative_response, rank):
    left, singular_values, right = np.linalg.svd(relative_response, full_matrices=False)
    left = left[:, :rank] * singular_values[:rank]
    right = right[:rank]
    approximation = left @ right
    return left, right, float(np.max(np.abs(relative_response - approximation)))


def _load_latest_tmm_optical_values(fit_path):
    """Load the optical sites used by the wavelength/thickness TMM builder."""
    required = (
        "ar_tile_raw",
        "absolute_hgcdte_n_log_scale",
        "absolute_hgcdte_k_log_scale",
    )
    with np.load(Path(fit_path), allow_pickle=False) as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"Latest TMM fit is missing {missing}: {fit_path}")
        values = {
            name: np.asarray(source[name], dtype=float).copy() for name in required
        }
        values["incidence_angle_deg"] = np.asarray(
            source["incidence_angle_deg"]
            if "incidence_angle_deg" in source
            else 0.0,
            dtype=float,
        )
    return values


def _same_values(reference, current, label, path):
    if not np.array_equal(np.asarray(reference), np.asarray(current)):
        raise ValueError(f"Latest fit {label} differs in {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT_ROOT)
    parser.add_argument(
        "--fit-subdir",
        help="Per-detector fit directory below --fit-root/<detector> (for example, spline_relative_response_normal).",
    )
    parser.add_argument("--modulation-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument(
        "--modulation-fit", choices=("zane", "amber-student-t", "amber-student-t-detector-effective-area", "amber-student-t-support-aligned", "amber-student-t-strong-red-led-mask", JOINT_MODULATION_FIT), default="zane",
        help="Source modulation fit family; Amber uses the converged fixed-nu=4 Student-t experiments.",
    )
    parser.add_argument(
        "--joint-fit-name",
        default=DEFAULT_JOINT_FIT_NAME,
        help="Experiment directory name under the detector's selected fit route.",
    )
    parser.add_argument(
        "--joint-fit-subdir",
        choices=("relative_response", "full_tmm"),
        default=None,
        help="Route-specific detector fit root containing the joint experiment.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detector",
        action="append",
        required=True,
        choices=DETECTORS,
        help="Detector to include; repeat explicitly for multiple detectors.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Response SVD rank (default: retain the base bundle rank).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume tile-response work from checkpoint blocks")
    parser.add_argument("--checkpoint-interval", type=int, default=128,
                        help="Number of tiles per durable response checkpoint block")
    return parser.parse_args(argv)


def build_bundle(args):
    with np.load(args.base_bundle, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    base_version = int(payload["bundle_version"].reshape(-1)[0])
    if base_version not in (1, 2, MODULATED_BUNDLE_VERSION):
        raise ValueError(f"Unsupported base bundle version {base_version}")
    base = payload
    base_detectors = np.asarray(base["detectors"]).astype(str)
    requested_detectors = [str(value).upper() for value in args.detector]
    if len(set(requested_detectors)) != len(requested_detectors):
        raise ValueError("Detector selections must be unique")
    if set(requested_detectors) != set(base_detectors.tolist()):
        missing = sorted(set(base_detectors.tolist()) - set(requested_detectors))
        extra = sorted(set(requested_detectors) - set(base_detectors.tolist()))
        raise ValueError(f"Detector selection must match base bundle; missing={missing}, extra={extra}")

    wavelengths_nm = np.asarray(base["wavelengths_nm"], dtype=float)
    thickness_grid_nm = np.asarray(base["ice_thickness_nm"], dtype=float)
    if thickness_grid_nm[0] != 0.0:
        raise ValueError("Base response bundle must include zero ice thickness")
    rank = (
        int(np.asarray(base["response_rank"]).reshape(-1)[0])
        if args.rank is None
        else int(args.rank)
    )
    if rank < 1 or rank > min(thickness_grid_nm.size, wavelengths_nm.size):
        raise ValueError("Response rank is outside the base grid dimensions")
    checkpoint_interval = int(getattr(args, "checkpoint_interval", 128))
    if checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive")
    checkpoint_dir = args.output.with_name(args.output.name + ".tile_checkpoints")
    checkpoint_manifest_path = checkpoint_dir / "build_manifest.json"
    checkpoint_manifest = {
        "base_bundle_sha256": _sha256(args.base_bundle),
        "detectors": requested_detectors,
        "fit_root": str(Path(args.fit_root).resolve()),
        "fit_subdir": getattr(args, "fit_subdir", None),
        "modulation_root": str(Path(args.modulation_root).resolve()),
        "modulation_fit": getattr(args, "modulation_fit", "zane"),
        "joint_fit_subdir": getattr(args, "joint_fit_subdir", None),
        "rank": rank,
        "tile_count": int(base["tile_ids"].size),
        "output": str(args.output.resolve()),
    }
    if getattr(args, "resume", False):
        if not checkpoint_manifest_path.is_file():
            raise FileNotFoundError(f"No compatible build checkpoint at {checkpoint_dir}")
        previous_manifest = json.loads(checkpoint_manifest_path.read_text())
        if previous_manifest != checkpoint_manifest:
            raise ValueError("Build checkpoint configuration differs from this invocation")
    else:
        if checkpoint_dir.exists():
            raise FileExistsError(
                f"Build checkpoints already exist at {checkpoint_dir}; use --resume"
            )
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing bundle {args.output}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest, indent=2) + "\n")

    modulation = {
        detector: _load_detector_modulation(
            args.modulation_root, detector, wavelengths_nm,
            fit_kind=getattr(args, "modulation_fit", "zane"),
            joint_fit_name=getattr(args, "joint_fit_name", DEFAULT_JOINT_FIT_NAME),
            joint_fit_subdir=getattr(args, "joint_fit_subdir", None),
        )
        for detector in requested_detectors
    }
    joint_fit_kind = getattr(args, "modulation_fit", "zane") == JOINT_MODULATION_FIT
    joint_ice_histories = (
        {
            detector: _load_joint_ice_history(args.modulation_root, detector, modulation[detector])
            for detector in requested_detectors
        }
        if joint_fit_kind else {}
    )
    reference_names = modulation[requested_detectors[0]]["basis_names"]
    reference_centers = modulation[requested_detectors[0]]["basis_centers_nm"]
    for detector, item in modulation.items():
        if tuple(item["basis_names"]) != tuple(reference_names):
            raise ValueError(f"Modulation basis differs for {detector}")
        if (not joint_fit_kind
                and getattr(args, "modulation_fit", "zane") not in ("amber-student-t-support-aligned", "amber-student-t-strong-red-led-mask")
                and not np.array_equal(item["basis_centers_nm"], reference_centers)):
            raise ValueError(f"Modulation basis centers differ for {detector}")

    tile_count = int(base["tile_ids"].size)
    clean_power = np.empty((tile_count, wavelengths_nm.size), dtype=np.float32)
    response_left = np.empty(
        (tile_count, thickness_grid_nm.size, rank), dtype=np.float32
    )
    response_right = np.empty(
        (tile_count, rank, wavelengths_nm.size), dtype=np.float32
    )
    svd_error_by_tile = np.zeros(tile_count, dtype=np.float32)
    zero_error_by_tile = np.zeros(tile_count, dtype=np.float32)
    completed_rows = 0
    if getattr(args, "resume", False):
        blocks = sorted(checkpoint_dir.glob("response_block_*.npz"))
        for block_path in blocks:
            with np.load(block_path, allow_pickle=False) as block:
                start = int(np.asarray(block["start_index"]).reshape(-1)[0])
                stop = int(np.asarray(block["stop_index"]).reshape(-1)[0])
                if start != completed_rows or stop <= start or stop > tile_count:
                    raise ValueError(f"Noncontiguous or invalid response checkpoint: {block_path}")
                clean_power[start:stop] = block["clean_power"]
                response_left[start:stop] = block["response_left"]
                response_right[start:stop] = block["response_right"]
                svd_error_by_tile[start:stop] = block["svd_error"]
                zero_error_by_tile[start:stop] = block["zero_thickness_error"]
                completed_rows = stop
        print(f"Resuming from {completed_rows}/{tile_count} checkpointed tile responses", flush=True)
    checkpoint_start = completed_rows

    tile_ice_thickness = None
    tile_rate = None
    post_reset_offsets = np.empty(tile_count, dtype=np.float64)
    fit_paths = []
    optical_cache_paths = []
    optical_fit_paths = []
    statistics_paths = []
    fit_statistics_fingerprints = []
    observation_days = None
    observation_start_time = None
    observation_fields = None
    latest_fit_settings = None
    ice_reset_day = None
    third_decon_day = None
    third_decon_included = None
    post_third_decon_offsets = np.empty(tile_count, dtype=np.float64)
    third_decon_offsets_present = False

    fit_root = Path(args.fit_root)
    detector_fit_roots = {
        detector: fit_root_for_detector(
            fit_root, detector, fit_subdir=getattr(args, "fit_subdir", None)
        )
        for detector in requested_detectors
    }
    for tile_index, (detector_value, tile_id_value) in enumerate(
        zip(base_detectors, base["tile_ids"])
    ):
        detector = str(detector_value)
        tile_id = int(tile_id_value)
        detector_fit_root = detector_fit_roots[detector]
        fit_path = detector_fit_root / f"tile_{tile_id:03d}/posterior_samples.npz"
        if not fit_path.is_file():
            raise FileNotFoundError(f"Latest per-tile fit is missing: {fit_path}")
        trajectory = load_fit_trajectory(fit_path)
        fit = trajectory["fit"]
        fit_detector = str(np.asarray(fit["sca_names"]).reshape(-1)[0]).upper()
        fit_tile_id = int(np.asarray(fit["tile_ids"]).reshape(-1)[0])
        if (fit_detector, fit_tile_id) != (detector, tile_id):
            raise ValueError(f"Latest fit identity differs in {fit_path}")

        days = np.asarray(trajectory["days"], dtype=float)
        start_times = np.asarray(trajectory["start_times"]).astype(str)
        reset_day = float(trajectory["reset_day"])
        local_third_day = trajectory["third_day"]
        joint_history = joint_ice_histories.get(detector) if joint_fit_kind else None
        joint_tile_ice = joint_tile_rates = None
        if joint_fit_kind:
            joint_tile_ice, joint_tile_rates = _sample_joint_tile_ice(joint_history, tile_id, days)
        bundle_fields = {
            name: np.asarray(fit[name])
            for name in ("observation_led", "observation_repeat", "observation_program", "groups")
        }
        if observation_days is None:
            observation_days = days
            observation_start_time = start_times
            tile_ice_thickness = np.empty(
                (tile_count, days.size), dtype=np.float32
            )
            tile_rate = np.empty_like(tile_ice_thickness)
            ice_reset_day = reset_day
            third_decon_day = local_third_day
            third_decon_included = local_third_day is not None
            latest_fit_settings = {
                name: np.asarray(fit[name]).reshape(-1)[0].item()
                for name in (
                    "fit_mode",
                    "ice_reset_day",
                    "trajectory_rate_prior_mean_nm_day",
                    "trajectory_rate_prior_sigma_log",
                    "trajectory_entropy_weight",
                    "trajectory_regularization",
                    "map_optimizer",
                    "map_steps",
                    "map_parameter_scaling",
                )
                if name in fit
            }
            observation_fields = bundle_fields
        else:
            _same_values(observation_days, days, "observation days", fit_path)
            _same_values(observation_start_time, start_times, "observation times", fit_path)
            if not np.isclose(reset_day, ice_reset_day, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"Ice reset day differs in {fit_path}")
            if (local_third_day is None) != (third_decon_day is None):
                raise ValueError(f"Third-decon metadata differs in {fit_path}")
            if local_third_day is not None and not np.isclose(
                float(local_third_day), float(third_decon_day), rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"Third-decon day differs in {fit_path}")
            for name, reference in observation_fields.items():
                _same_values(reference, fit[name], name, fit_path)

        optical_cache_path = (
            Path(modulation[detector]["summary"]["ice_history"]["optical_state_source"])
            / f"tile_{tile_id:03d}/relative_response.npz"
        )
        optical_fit_path = mock_model.source_fit_from_cache(
            optical_cache_path, detector, tile_id
        )
        if tile_index >= completed_rows:
            optical_values = _load_latest_tmm_optical_values(optical_fit_path)
            spectral = relative_response.build_spectral_response_lookup(
                optical_values,
                wavelengths_nm,
                thickness_grid_nm,
                complex_reflection_modulation_ratio=modulation[detector]["ratio"],
            )
            zero_error = float(np.max(np.abs(spectral["relative_log_response"][0])))
            if zero_error > 1.0e-10:
                raise ValueError(f"Modulated zero-thickness response is not unity in {fit_path}")
            left, right, svd_error = _compress_response(
                np.asarray(spectral["relative_log_response"], dtype=float), rank
            )
            clean_power[tile_index] = np.asarray(spectral["clean_power"], dtype=np.float32)
            response_left[tile_index] = left.astype(np.float32)
            response_right[tile_index] = right.astype(np.float32)
            svd_error_by_tile[tile_index] = svd_error
            zero_error_by_tile[tile_index] = zero_error

        tile_thickness = np.asarray(trajectory["thickness"], dtype=float)
        tile_rates = np.asarray(trajectory["rates"], dtype=float)
        if joint_fit_kind:
            expected_resets = np.asarray(joint_history["summary"]["reset_days"], dtype=float)
            if (expected_resets.size != 2
                    or not np.isclose(reset_day, expected_resets[0], rtol=0.0, atol=1.0e-10)
                    or local_third_day is None
                    or not np.isclose(float(local_third_day), expected_resets[1], rtol=0.0, atol=1.0e-10)):
                raise ValueError(f"Joint ice reset boundaries differ from source fit {fit_path}")
            tile_thickness, tile_rates = joint_tile_ice, joint_tile_rates
        if not np.isfinite(tile_thickness).all() or not np.isfinite(tile_rates).all():
            raise ValueError(f"Nonfinite ice trajectory in {fit_path}")
        tile_ice_thickness[tile_index] = np.asarray(tile_thickness, dtype=np.float32)
        tile_rate[tile_index] = np.asarray(tile_rates, dtype=np.float32)
        post_reset_offsets[tile_index] = float(trajectory["grey_offset"])
        if third_decon_included:
            if trajectory["third_offset"] is None:
                raise ValueError(f"Third-decon offset is missing in {fit_path}")
            post_third_decon_offsets[tile_index] = float(trajectory["third_offset"])
            third_decon_offsets_present = True
        fit_paths.append(str(fit_path.resolve()))
        optical_cache_paths.append(str(optical_cache_path.resolve()))
        optical_fit_paths.append(str(optical_fit_path.resolve()))
        statistics_paths.append(
            str(Path(str(np.asarray(fit["statistics_path"]).reshape(-1)[0])).resolve())
        )
        fit_statistics_fingerprints.append(
            str(np.asarray(fit["statistics_fingerprint"]).reshape(-1)[0])
        )
        if tile_index >= completed_rows and (
            tile_index + 1 - checkpoint_start >= checkpoint_interval or tile_index + 1 == tile_count
        ):
            stop = tile_index + 1
            block_path = checkpoint_dir / f"response_block_{checkpoint_start:05d}_{stop:05d}.npz"
            temporary_path = checkpoint_dir / f".response_block_{checkpoint_start:05d}_{stop:05d}.tmp.npz"
            np.savez_compressed(
                temporary_path,
                start_index=np.asarray(checkpoint_start, dtype=np.int32),
                stop_index=np.asarray(stop, dtype=np.int32),
                clean_power=clean_power[checkpoint_start:stop],
                response_left=response_left[checkpoint_start:stop],
                response_right=response_right[checkpoint_start:stop],
                svd_error=svd_error_by_tile[checkpoint_start:stop],
                zero_thickness_error=zero_error_by_tile[checkpoint_start:stop],
            )
            temporary_path.replace(block_path)
            checkpoint_start = stop
            print(f"Checkpointed tile responses through {stop}/{tile_count}", flush=True)
        if tile_index >= completed_rows and (tile_index + 1) % 128 == 0:
            print(f"Recomputed modulated response for {tile_index + 1}/{tile_count} tiles", flush=True)

    if checkpoint_start < tile_count:
        raise RuntimeError(f"Only {checkpoint_start}/{tile_count} tile responses were checkpointed")
    max_svd_error_by_detector = {
        detector: float(np.max(svd_error_by_tile[base_detectors == detector]))
        for detector in requested_detectors
    }
    max_zero_thickness_error = float(np.max(zero_error_by_tile))

    # Replace the bundle's older trajectory and response arrays, preserving its
    # tile geometry and shared spectral grids.
    payload = {
        name: value
        for name, value in payload.items()
        if name not in (
            "metadata_json",
            "modulation_detectors",
            "modulation_wavelengths_nm",
            "modulation_ratio_real",
            "modulation_ratio_imag",
            "modulation_basis_names",
            "modulation_basis_centers_nm",
            "modulation_log_amplitude_coefficients",
            "modulation_phase_coefficients",
        )
    }
    payload.update(
        {
            "bundle_version": np.asarray(MODULATED_BUNDLE_VERSION, dtype=np.int16),
            "response_rank": np.asarray(rank, dtype=np.int16),
            "clean_power": clean_power,
            "response_left": response_left,
            "response_right": response_right,
            "observation_days": np.asarray(observation_days, dtype=np.float64),
            "observation_start_time": string_array(observation_start_time),
            "observation_led": np.asarray(observation_fields["observation_led"], dtype=np.int16),
            "observation_repeat": np.asarray(observation_fields["observation_repeat"], dtype=np.int16),
            "observation_program": np.asarray(observation_fields["observation_program"], dtype=np.int16),
            "groups": np.asarray(observation_fields["groups"], dtype=np.int32),
            "tile_ice_thickness": tile_ice_thickness,
            "tile_instantaneous_rate_nm_day": tile_rate,
            "post_reset_tile_log_offset": post_reset_offsets,
            "post_reset_tile_throughput_factor": np.exp(post_reset_offsets).astype(np.float32),
            "ice_reset_day": np.asarray(ice_reset_day, dtype=np.float64),
            "fit_paths": string_array(fit_paths),
            "optical_cache_paths": string_array(optical_cache_paths),
            "optical_fit_paths": string_array(optical_fit_paths),
            "statistics_paths": string_array(statistics_paths),
            "fit_statistics_fingerprints": string_array(fit_statistics_fingerprints),
        }
    )
    if third_decon_included and third_decon_offsets_present:
        payload["post_third_decon_tile_log_offset"] = post_third_decon_offsets
        payload["post_third_decon_tile_throughput_factor"] = np.exp(
            post_third_decon_offsets
        ).astype(np.float32)
        payload["third_decon_day"] = np.asarray(third_decon_day, dtype=np.float64)
    else:
        for key in (
            "post_third_decon_tile_log_offset",
            "post_third_decon_tile_throughput_factor",
            "third_decon_day",
        ):
            payload.pop(key, None)

    modulation_detectors = list(requested_detectors)
    names = modulation[modulation_detectors[0]]["basis_names"]
    centers = modulation[modulation_detectors[0]]["basis_centers_nm"]
    payload.update(
        {
            "modulation_detectors": string_array(modulation_detectors),
            "modulation_wavelengths_nm": wavelengths_nm.astype(np.float32),
            "modulation_ratio_real": np.asarray(
                [modulation[detector]["ratio"].real for detector in modulation_detectors],
                dtype=np.float32,
            ),
            "modulation_ratio_imag": np.asarray(
                [modulation[detector]["ratio"].imag for detector in modulation_detectors],
                dtype=np.float32,
            ),
            "modulation_basis_names": string_array(names),
            "modulation_basis_centers_nm": np.asarray(centers, dtype=np.float32),
            "modulation_basis_centers_nm_by_detector": np.asarray(
                [modulation[detector]["basis_centers_nm"] for detector in modulation_detectors], dtype=np.float32
            ),
            "modulation_log_amplitude_coefficients": np.asarray(
                [modulation[detector]["log_amplitudes"] for detector in modulation_detectors],
                dtype=np.float32,
            ),
            "modulation_phase_coefficients": np.asarray(
                [modulation[detector]["phases"] for detector in modulation_detectors],
                dtype=np.float32,
            ),
        }
    )

    metadata = json.loads(str(np.asarray(base["metadata_json"]).reshape(-1)[0]))
    base_cache_roots = metadata.pop("source_cache_roots", {})
    metadata.update(
        {
            "bundle_version": int(MODULATED_BUNDLE_VERSION),
            "base_bundle_path": str(Path(args.base_bundle).resolve()),
            "base_bundle_sha256": _sha256(args.base_bundle),
            "latest_tmm_fit_root": str(Path(args.fit_root).resolve()),
            "tile_optical_cache_roots": {
                detector: str(
                    Path(modulation[detector]["summary"]["ice_history"]["optical_state_source"]).resolve()
                )
                for detector in modulation_detectors
            },
            "response_definition": (
                "log(coated_modulated_power / clean_modulated_power) before "
                "filter/SED/QE weighting"
            ),
            "response_representation": (
                "modulation-aware rank-reduced TMM grid; interpolation of response factors"
            ),
            "clean_reflectance_modulation_included": True,
            "modulation_fit_family": getattr(args, "modulation_fit", "zane"),
            "joint_fit_subdir": getattr(args, "joint_fit_subdir", None),
            "modulation_scope": (
                "detector-specific complex-reflection coefficients jointly fitted to CAR190 and Amber, "
                "with an independent per-tile 2-day ice-rate trajectory"
                if joint_fit_kind else
                "one detector-wide multiplier inferred from tile-matched stellar measurements"
            ),
            "clean_reflectance_modulation_form": (
                "r_mod = r_clean_tile * exp(sum(a_j B_j) + i sum(phi_j B_j))"
            ),
            "modulation_application": (
                "same detector-level complex multiplier applied to each tile's clean "
                "s- and p-polarized reflection before propagating the ice layer"
            ),
            "tile_specific_modulation_fit": not joint_fit_kind,
            "modulation_zero_at_led_centers": not joint_fit_kind,
            "led_center_max_abs_complex_deviation_by_detector": {
                detector: float(modulation[detector]["anchor_error"])
                for detector in modulation_detectors
            },
            "roll_zeropoints_embedded": False,
            "modulation_fit_sources": {
                detector: {
                    "summary": str(modulation[detector]["summary_path"].resolve()),
                    "parameters": str(modulation[detector]["parameters_path"].resolve()),
                    "summary_sha256": modulation[detector]["summary_sha256"],
                    "parameters_sha256": modulation[detector]["parameters_sha256"],
                    **modulation[detector]["fit_provenance"],
                    "source_optical_state": modulation[detector]["summary"]["ice_history"]["optical_state_source"],
                    "maxent_weight": float(
                        modulation[detector]["summary"]["modulation"]["maxent_regularization"]["weight"]
                    ),
                    "phase_smoothness_weight": float(
                        modulation[detector]["summary"]["modulation"]["phase_smoothness_regularization"]["weight"]
                    ),
                }
                for detector in modulation_detectors
            },
            "latest_tmm_fit_roots": {
                detector: str(detector_fit_roots[detector].resolve())
                for detector in modulation_detectors
            },
            **({
                "joint_ice_history_fit_roots": {
                    detector: str(joint_ice_histories[detector]["archive_path"].parent.resolve())
                    for detector in modulation_detectors
                },
                "joint_ice_history_fit_summaries": {
                    detector: {
                        "summary": str((joint_ice_histories[detector]["archive_path"].parent / "summary.json").resolve()),
                        "summary_sha256": _sha256(joint_ice_histories[detector]["archive_path"].parent / "summary.json"),
                        "archive": str(joint_ice_histories[detector]["archive_path"].resolve()),
                        "archive_sha256": _sha256(joint_ice_histories[detector]["archive_path"]),
                        "rate_control_spacing_days": 2.0,
                        "ice_parameters_per_tile": joint_ice_histories[detector]["summary"]["ice_parameters_per_tile"],
                        "total_ice_parameters": joint_ice_histories[detector]["summary"]["total_ice_parameters"],
                    }
                    for detector in modulation_detectors
                },
            } if joint_fit_kind else {}),
            "base_bundle_source_cache_roots": base_cache_roots,
            "fit_settings": latest_fit_settings,
            "trajectory_definition": (
                "joint CAR190 + Amber MAP thickness from independent positive 2-day per-tile rate controls"
                if joint_fit_kind else
                "MAP tile_ice_thickness from the current 20261106 per-detector spline fits"
            ),
            "trajectory_rate_definition": (
                "piecewise interval slopes of the joint fitted thickness trajectory, sampled at CAR190 observation days"
                if joint_fit_kind else "MAP tile_instantaneous_rate_nm_day"
            ),
            "trajectory_rate_control_spacing_days": 2.0 if joint_fit_kind else None,
            "trajectory_uncertainties": "not included; deterministic MAP export",
            "response_svd_max_abs_error_by_detector": max_svd_error_by_detector,
            "zero_thickness_max_abs_log_response": max_zero_thickness_error,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    metadata["trajectory_fit_roots"] = (
        metadata["joint_ice_history_fit_roots"] if joint_fit_kind else metadata["latest_tmm_fit_roots"]
    )
    metadata["trajectory_included"] = True
    metadata["grey_throughput_included"] = True
    metadata["third_decon_included"] = bool(third_decon_included)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    del base, payload, clean_power, response_left, response_right
    del tile_ice_thickness, tile_rate, post_reset_offsets, post_third_decon_offsets
    gc.collect()
    loaded = load_bundle(args.output)
    summary = bundle_summary(loaded)
    (checkpoint_dir / "build_complete.json").write_text(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "tile_count": tile_count,
        "modulation_fit": getattr(args, "modulation_fit", "zane"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"Wrote modulation-aware bundle {args.output}")
    print(
        f"Tiles={summary['tile_count']}, detectors={summary['detector_count']}, "
        f"rank={rank}, max grid SVD error="
        f"{max(max_svd_error_by_detector.values()):.6g}, "
        f"zero-thickness max log response={max_zero_thickness_error:.3g}"
    )
    return 0


def main(argv=None):
    return build_bundle(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
