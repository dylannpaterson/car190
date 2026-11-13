"""Evaluate joint CAR190/Amber corrections on stars excluded from the fit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import heatmap_car201_allstar_model_correction as workflow
import run_amber_student_t_detectors as amber
from car201_photometry_forward_jacobian import CachedPhotometryForward
from experiment_joint_wfi01_fixed_ice import add_full_basis
from experiment_joint_wfi01_weekly_ice import RESETS, cadence_output
from plot_car201_matched_visit_changes import quality_filter
from plot_joint_amber_car190_detailed import amber_variant_cells, plot_amber_variants
from support_aligned_modulation import effective_area_support


ROOT = Path(__file__).resolve().parent
OUTPUT = workflow.PRODUCTION_ROOT / (
    "wfi01/spline_relative_response_normal/experiments/"
    "car190_amber_joint_2day_all_detectors_overview"
)
SED_SOURCES = ((workflow.FAINT_SED_PATH, "faint_v2"),
               (workflow.ORIGINAL_SED_PATH, "original_successful"))


def all_star_rows(detectors=None, use_saved_quality=True):
    reference_path = (
        workflow.PRODUCTION_ROOT
        / "wfi01/spline_relative_response_normal/experiments/"
        "car201_zane_wfi01_ep3v2_ep4_faint_v2_tile_matched/"
        "car201_actual_star_modulation_fit_summary.json"
    )
    if reference_path.is_file() and use_saved_quality:
        summary = json.loads(reference_path.read_text())
        start_utc, end_utc = summary["date_window_utc"]
    else:
        # Newer model runs retain the CAR201 pass window in source config; they
        # need not depend on a dated historical Zane fit-summary artifact.
        config = workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]
        start_utc, end_utc = config["start"], config["end"]
    start_mjd = float(Time(start_utc, format="isot", scale="utc").mjd)
    end_mjd = float(Time(end_utc, format="isot", scale="utc").mjd)
    rows = []
    selected_detectors = list(detectors) if detectors is not None else list(workflow.DETECTORS)
    for detector in selected_detectors:
        for pass_name, frame in zip(workflow.PASSES,
                                    workflow._load_detector_frames(detector, start_mjd, end_mjd),
                                    strict=True):
            rows.append(frame.assign(detector=detector, pass_name=pass_name)
                        .rename(columns={"pass_name": "pass"}))
    result = pd.concat(rows, ignore_index=True)
    if reference_path.is_file() and use_saved_quality:
        result = quality_filter(result)
    else:
        # For the current production model, derive a fresh held-out cohort from
        # the catalog and DQ flags. The historical manual/repeat-outlier table
        # is not part of the newer model inputs.
        dq = pd.to_numeric(result["dq_flag"], errors="coerce").fillna(-1).to_numpy(np.int64)
        saturated = (dq >= 0) & ((dq & 2) != 0)
        result["saturated"] = saturated
        source_saturated = result.groupby(["detector", "source_id"])["saturated"].transform("any")
        result = result.loc[~source_saturated & ~((dq >= 0) & ((dq & 1) != 0))].copy()
        magnitude = pd.to_numeric(result["mag_obs"], errors="coerce")
        error = pd.to_numeric(result["mag_err"], errors="coerce")
        result = result.loc[np.isfinite(magnitude) & np.isfinite(error) & (error > 0.0)].copy()
    result["source_id"] = result["source_id"].astype(str)
    magnitude = pd.to_numeric(result["mag_obs"], errors="coerce")
    error = pd.to_numeric(result["mag_err"], errors="coerce")
    result = result.loc[np.isfinite(magnitude) & np.isfinite(error) & (error > 0.0)].copy()
    return result


def oos_ids_for_detector(rows, detector, fixed_ice_fit_dir=None, joint_ice_fit_dir=None,
                         joint_fit_root=None, joint_fit_name=None):
    if joint_ice_fit_dir is not None:
        fit_path = Path(joint_ice_fit_dir) / "joint_weekly_ice_fit.npz"
    elif joint_fit_root is not None:
        if joint_fit_name is None:
            raise ValueError("A joint fit name is required with --joint-fit-root")
        fit_path = Path(joint_fit_root) / detector.lower() / joint_fit_name / "joint_weekly_ice_fit.npz"
    elif fixed_ice_fit_dir is None:
        fit_path = cadence_output(detector, 2.0) / "joint_weekly_ice_fit.npz"
    else:
        fit_path = Path(fixed_ice_fit_dir) / "joint_fit.npz"
    with np.load(fit_path, allow_pickle=False) as archive:
        trained = set(np.asarray(archive["source_ids"]).astype(str))
    selected = rows.loc[rows["detector"] == detector].copy()
    counts = selected.groupby(["source_id", "filter"]).size()
    repeated = set(counts[counts >= 2].index)
    repeated_ids = {source_id for source_id, _band in repeated}
    available = set(selected["source_id"].astype(str))
    return sorted((available & repeated_ids) - trained)


def visits_and_frames(data, source_ids):
    source_rows = data.loc[data["source_id"].isin(source_ids)].copy()
    frames = []
    target_mjd = np.empty((len(amber.FILTERS), len(amber.PASSES)), dtype=float)
    for pass_index, pass_name in enumerate(amber.PASSES):
        frame = source_rows.loc[source_rows["pass"] == pass_name,
                                ["source_id", "filter", "mjd", "x_pixel", "y_pixel"]].copy()
        for filter_index, band in enumerate(amber.FILTERS):
            values = frame.loc[frame["filter"] == band, "mjd"].to_numpy(float)
            if values.size == 0:
                # The optical model needs a date/tile for every filter/pass
                # cell. Add a non-photometric placeholder only for empty cells;
                # its value is never included in the held-out plot or metrics.
                available = data.loc[(data["detector"] == data["detector"].iloc[0])
                                     & (data["filter"] == band)
                                     & (data["pass"] == pass_name)]
                if available.empty:
                    raise ValueError(f"No CAR201 visit date for {band}/{pass_name}")
                exemplar = source_rows.iloc[0]
                placeholder = {"source_id": source_ids[0], "filter": band,
                               "mjd": float(available["mjd"].median()),
                               "x_pixel": float(exemplar["x_pixel"]),
                               "y_pixel": float(exemplar["y_pixel"])}
                frame = pd.concat([frame, pd.DataFrame([placeholder])], ignore_index=True)
                values = np.asarray([placeholder["mjd"]])
            target_mjd[filter_index, pass_index] = float(np.median(values))
        frames.append(frame)
    return source_rows, frames, target_mjd


def thickness_at_days(joint_times, fitted_thickness, target_days):
    joint_times = np.asarray(joint_times, dtype=float)
    fitted_thickness = np.asarray(fitted_thickness, dtype=float)
    target_days = np.asarray(target_days, dtype=float)
    result = np.empty((fitted_thickness.shape[0], *target_days.shape), dtype=float)
    joint_stage = np.searchsorted(RESETS, joint_times, side="right")
    target_stage = np.searchsorted(RESETS, target_days, side="right")
    for filter_index in range(target_days.shape[0]):
        for pass_index in range(target_days.shape[1]):
            day = target_days[filter_index, pass_index]
            stage = target_stage[filter_index, pass_index]
            mask = joint_stage == stage
            if day < joint_times[mask][0] or day > joint_times[mask][-1]:
                raise ValueError(f"Held-out Amber epoch day {day:.8f} is outside the joint ice segment")
            for tile in range(fitted_thickness.shape[0]):
                result[tile, filter_index, pass_index] = np.interp(
                    day, joint_times[mask], fitted_thickness[tile, mask])
    return result


def arrays_for_stars(rows, source_ids):
    source_lookup = {source: i for i, source in enumerate(source_ids)}
    filter_lookup = {name: i for i, name in enumerate(amber.FILTERS)}
    pass_lookup = {name: i for i, name in enumerate(amber.PASSES)}
    shape = (len(source_ids), len(amber.FILTERS), len(amber.PASSES))
    magnitude = np.full(shape, np.nan)
    error = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    indices = (
        rows["source_id"].map(source_lookup).to_numpy(int),
        rows["filter"].map(filter_lookup).to_numpy(int),
        rows["pass"].map(pass_lookup).to_numpy(int),
    )
    if len(set(zip(*indices))) != len(rows):
        raise ValueError("Held-out sample has duplicate source/filter/visit measurements")
    magnitude[indices] = rows["mag_obs"].to_numpy(float)
    error[indices] = rows["mag_err"].to_numpy(float)
    valid[indices] = True
    return magnitude, error, valid


def model_from_fixed_joint_fit(detector, frames, source_ids, sed_assignments,
                               target_mjd, reference_mjd, effective_area_path,
                               fit_dir, ice_history_fit_root, optical_cache_root):
    with np.load(Path(fit_dir) / "joint_fit.npz", allow_pickle=False) as archive:
        wavelength_key = "wavelength_nm" if "wavelength_nm" in archive else "wavelengths_nm"
        wavelengths = np.asarray(archive[wavelength_key], dtype=float)
        saved_basis = np.asarray(archive["basis"], dtype=float)
    expected_wavelengths, effective_area = workflow._make_wavelength_grid(
        5.0,
        effective_area_path=effective_area_path,
        full_effective_area_support=True,
    )
    np.testing.assert_array_equal(expected_wavelengths, wavelengths)
    target_days = np.asarray(target_mjd, dtype=float) - reference_mjd
    tile_thickness = workflow._tile_histories(
        detector, target_days, history_root=ice_history_fit_root
    )
    tile_ids = workflow._source_tile_ids(source_ids, frames)
    star_photon_shapes, _, fallback_count = workflow.fit_model.reconstruct_sed_photon_shapes(
        source_ids, sed_assignments, workflow.SED_GRID_PATH, wavelengths
    )
    if fallback_count:
        raise ValueError(f"Unexpected SED fallback for held-out {detector} stars")
    optical_arrays = workflow._optical_tile_arrays(
        detector, wavelengths, optical_cache_root
    )
    pass_groups = workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]["zero_point_group_ids"]
    model = workflow.fit_model.RealPhotometryFit(
        wavelengths,
        effective_area,
        star_photon_shapes,
        tile_thickness,
        tile_ids,
        *optical_arrays,
        "per-gap",
        entropy_weight=1000.0,
        phase_smoothness_weight=100.0,
        pass_zero_point_group_ids=pass_groups,
    )
    add_full_basis(model, effective_area_support(effective_area_path))
    np.testing.assert_allclose(model.basis, saved_basis, rtol=0.0, atol=0.0)
    model.detector = detector
    return model


def model_predictions(model, theta, thickness):
    model.tile_thickness_by_id = thickness
    forward = CachedPhotometryForward(model)
    response, _ = forward.evaluate(theta[:2 * model.mode_count])
    zpts = theta[2 * model.mode_count:].reshape(len(amber.FILTERS), model.zero_point_group_count)
    roll = np.zeros_like(response)
    for pass_index, group in enumerate(model.pass_zero_point_group_ids):
        if group:
            response[:, :, pass_index] += zpts[:, group - 1][None, :]
            roll[:, :, pass_index] = -zpts[:, group - 1][None, :] / workflow.LOG_FLUX_PER_MAG
    return -response / workflow.LOG_FLUX_PER_MAG, roll


def summarize_oos_detector(detector, source_rows, source_ids, magnitude, valid, variants):
    use = valid & valid[:, :, :1]
    use[:, :, 0] = False
    records = []
    for name, corrected in variants.items():
        delta = corrected - corrected[:, :, :1]
        flux = 100 * np.expm1(-workflow.LOG_FLUX_PER_MAG * delta)
        values = flux[use]
        records.append({
            "detector": detector,
            "variant": name,
            "held_out_stars_with_direct_sed": len(source_ids),
            "matched_source_filter_visit_values": int(use.sum()),
            "fraction_within_0p5_percent": float(np.mean(np.abs(values) <= 0.5)) if values.size else np.nan,
            "relative_flux_change_rms_percent": float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan,
            "relative_flux_change_median_percent": float(np.median(values)) if values.size else np.nan,
        })
    return records


def summarize_oos_filter_counts(detector, valid, variants, cohort="direct_sed"):
    comparisons = valid & valid[:, :, :1]
    comparisons[:, :, 0] = False
    records = []
    for variant, corrected in variants.items():
        delta = corrected - corrected[:, :, :1]
        flux = 100 * np.expm1(-workflow.LOG_FLUX_PER_MAG * delta)
        for filter_index, band in enumerate(amber.FILTERS):
            selected = comparisons[:, filter_index, :]
            values = flux[:, filter_index, :][selected]
            finite = np.isfinite(values)
            count = int(np.count_nonzero(finite & (np.abs(values) <= 0.5)))
            total = int(np.count_nonzero(finite))
            records.append({
                "detector": detector,
                "filter": band,
                "variant": variant,
                "cohort": cohort,
                "within_0p5_percent_count": count,
                "comparison_count": total,
                "fraction_within_0p5_percent": count / total if total else np.nan,
            })
    return records


def extreme_oos_rows(detector, source_rows, source_ids, magnitude, error, valid, variants,
                     limit_per_variant=20):
    """Return the largest held-out visit changes with their raw measurements."""
    comparisons = valid & valid[:, :, :1]
    comparisons[:, :, 0] = False
    rows_by_cell = source_rows.set_index(["source_id", "filter", "pass"])
    if not rows_by_cell.index.is_unique:
        raise ValueError("Held-out raw rows contain duplicate source/filter/visit cells")
    records = []
    for variant, corrected in variants.items():
        delta = corrected - corrected[:, :, :1]
        with np.errstate(over="ignore", invalid="ignore"):
            flux = 100.0 * np.expm1(-workflow.LOG_FLUX_PER_MAG * delta)
        selected = np.argwhere(comparisons & np.isfinite(flux))
        if not len(selected):
            continue
        selected_values = np.abs(flux[tuple(selected.T)])
        order = np.argsort(selected_values)[-limit_per_variant:][::-1]
        for selected_index in order:
            star_index, filter_index, pass_index = selected[selected_index]
            source_id = source_ids[star_index]
            band = amber.FILTERS[filter_index]
            pass_name = amber.PASSES[pass_index]
            current = rows_by_cell.loc[(source_id, band, pass_name)]
            reference = rows_by_cell.loc[(source_id, band, amber.PASSES[0])]
            records.append({
                "detector": detector,
                "variant": variant,
                "source_id": source_id,
                "filter": band,
                "pass": pass_name,
                "mjd": float(current["mjd"]),
                "mag_obs": float(magnitude[star_index, filter_index, pass_index]),
                "reference_mag_obs": float(magnitude[star_index, filter_index, 0]),
                "mag_err": float(error[star_index, filter_index, pass_index]),
                "reference_mag_err": float(error[star_index, filter_index, 0]),
                "dq_flag": int(current["dq_flag"]) if pd.notna(current["dq_flag"]) else -1,
                "reference_dq_flag": int(reference["dq_flag"])
                if pd.notna(reference["dq_flag"]) else -1,
                "corrected_delta_mag": float(delta[star_index, filter_index, pass_index]),
                "relative_flux_change_percent": float(flux[star_index, filter_index, pass_index]),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True, choices=workflow.DETECTORS)
    parser.add_argument("--fixed-ice-fit-dir", type=Path,
                        help="Evaluate a fixed-ice joint fit directory instead of the 2-day fit")
    parser.add_argument("--joint-ice-fit-dir", type=Path,
                        help="Evaluate a fresh 2-day joint fit directory containing joint_weekly_ice_fit.npz")
    parser.add_argument("--joint-fit-root", type=Path,
                        help="Production root containing per-detector 2-day joint fits")
    parser.add_argument("--joint-fit-name",
                        help="Per-detector relative path below --joint-fit-root/<detector>")
    parser.add_argument("--production-root", type=Path,
                        help="Current production root; derive each detector's relative-response histories/cache")
    parser.add_argument("--ice-history-fit-root", type=Path,
                        help="Use this production per-tile history directory for held-out predictions")
    parser.add_argument("--optical-cache-root", type=Path,
                        help="Use this pinned optical-cache directory for held-out predictions")
    parser.add_argument("--output-dir", type=Path,
                        help="Write this evaluation to a separate directory")
    args = parser.parse_args()
    detectors = list(dict.fromkeys(args.detector))
    if (args.fixed_ice_fit_dir is not None or args.joint_ice_fit_dir is not None) and len(detectors) != 1:
        parser.error("A detector-specific fit directory requires exactly one explicit detector")
    if (args.joint_fit_root is None) != (args.joint_fit_name is None):
        parser.error("--joint-fit-root and --joint-fit-name must be supplied together")
    if args.joint_ice_fit_dir is not None and (
        args.ice_history_fit_root is None or args.optical_cache_root is None
    ):
        parser.error("--joint-ice-fit-dir requires current --ice-history-fit-root and --optical-cache-root")
    if args.joint_fit_root is not None and args.production_root is None:
        parser.error("--joint-fit-root requires --production-root for matching optical/history inputs")
    output_dir = args.output_dir or OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = all_star_rows(
        detectors,
        use_saved_quality=(args.ice_history_fit_root is None and args.production_root is None),
    )
    candidate_ids = {
        detector: oos_ids_for_detector(
            all_rows,
            detector,
            args.fixed_ice_fit_dir,
            args.joint_ice_fit_dir,
            args.joint_fit_root,
            args.joint_fit_name,
        )
        for detector in detectors
    }
    all_candidate_ids = sorted(set().union(*(set(values) for values in candidate_ids.values())))
    direct_seds = workflow._load_successful_seds(SED_SOURCES, all_candidate_ids)
    output_cells = []
    output_metrics = []
    output_counts = []
    output_filter_counts = []
    output_extreme_rows = []

    for detector_index, detector in enumerate(detectors, start=1):
        print(f"=== Held-out validation {detector} ({detector_index}/{len(detectors)}) ===", flush=True)
        detector_rows = all_rows.loc[all_rows["detector"] == detector].copy()
        source_ids = sorted(set(candidate_ids[detector]) & set(direct_seds))
        if not source_ids:
            raise ValueError(f"No out-of-sample stars with direct successful SEDs for {detector}")
        # Retain only source/filter groups with at least two measured visits.
        source_rows = detector_rows.loc[detector_rows["source_id"].isin(source_ids)].copy()
        counts = source_rows.groupby(["source_id", "filter"]).size()
        repeated_keys = set(counts[counts >= 2].index)
        row_keys = pd.MultiIndex.from_frame(source_rows[["source_id", "filter"]])
        source_rows = source_rows.loc[row_keys.isin(repeated_keys)].copy()
        source_ids = sorted(source_rows["source_id"].unique())
        if not source_ids:
            raise ValueError(f"No repeated held-out source/filter groups remain for {detector}")
        # Drop placeholders for sources removed by the singleton filter; rebuild
        # frame arrays from only the retained out-of-sample rows.
        source_rows, frames, target_mjd = visits_and_frames(detector_rows, source_ids)
        magnitude, error, valid = arrays_for_stars(source_rows, source_ids)

        area, _ = amber.canonical_configuration(detector)
        reference_mjd = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
        if args.joint_ice_fit_dir is not None:
            detector_joint_fit_dir = Path(args.joint_ice_fit_dir)
        elif args.joint_fit_root is not None:
            detector_joint_fit_dir = (
                Path(args.joint_fit_root) / detector.lower() / args.joint_fit_name
            )
        else:
            detector_joint_fit_dir = None
        detector_history_root = args.ice_history_fit_root
        detector_cache_root = args.optical_cache_root
        if args.production_root is not None:
            model_root = Path(args.production_root) / detector.lower() / "relative_response"
            detector_history_root = detector_history_root or (model_root / "fits")
            if detector_cache_root is None:
                model_mode = json.loads((model_root / "relative_response_mode.json").read_text())
                detector_cache_root = Path(model_mode["response_cache_root"])
        if (detector_joint_fit_dir is not None or args.fixed_ice_fit_dir is not None) \
                and detector_history_root is not None and detector_cache_root is not None:
            selected_fit_dir = detector_joint_fit_dir or args.fixed_ice_fit_dir
            model = model_from_fixed_joint_fit(
                detector,
                frames,
                source_ids,
                {source: direct_seds[source] for source in source_ids},
                target_mjd,
                reference_mjd,
                area,
                selected_fit_dir,
                detector_history_root,
                detector_cache_root,
            )
        else:
            old_fit, _ = amber.paths(detector)
            waves, areas = workflow._make_wavelength_grid(
                float(json.loads((old_fit / "car201_actual_star_modulation_fit_summary.json").read_text())
                      ["modulation"]["wavelength_stride_nm"]),
                effective_area_path=area,
                full_effective_area_support=True,
            )
            _, model = workflow._predict_detector(
                detector, frames, source_ids,
                {source: direct_seds[source] for source in source_ids},
                workflow.SED_GRID_PATH, old_fit, waves, areas, reference_mjd,
                unmodulated=True, return_model=True, effective_area_path=area,
                full_effective_area_support=True,
                ice_history_fit_root=detector_history_root,
                optical_cache_root=detector_cache_root,
            )
            add_full_basis(model, effective_area_support(area))

        if detector_joint_fit_dir is not None:
            joint_fit_dir = detector_joint_fit_dir
            with np.load(joint_fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as joint_archive:
                joint_times = np.asarray(joint_archive["times"], dtype=float)
                baseline_history = np.asarray(joint_archive["baseline_thickness_nm"], dtype=float)
                fitted_history = np.asarray(joint_archive["fitted_thickness_nm"], dtype=float)
                joint_theta = np.asarray(joint_archive["shared_parameters"], dtype=float)
                trained_ids = set(np.asarray(joint_archive["source_ids"]).astype(str))
            relative_days = target_mjd - reference_mjd
            baseline_thickness = thickness_at_days(joint_times, baseline_history, relative_days)
            joint_thickness = thickness_at_days(joint_times, fitted_history, relative_days)
        elif detector_history_root is not None:
            if args.fixed_ice_fit_dir is None:
                raise ValueError("A fixed-ice fit directory is required with an alternate ice history")
            fixed_fit_dir = Path(args.fixed_ice_fit_dir)
            fixed_summary = json.loads((fixed_fit_dir / "joint_fit_summary.json").read_text())
            if fixed_summary.get("ice_histories_refitted"):
                raise ValueError("The fixed-ice out-of-sample path requires an unchanged ice history")
            with np.load(fixed_fit_dir / "joint_fit.npz", allow_pickle=False) as fixed_archive:
                joint_theta = np.asarray(fixed_archive["parameters"], dtype=float)
                trained_ids = set(np.asarray(fixed_archive["source_ids"]).astype(str))
            baseline_thickness = np.asarray(model.tile_thickness_by_id, dtype=float).copy()
            joint_thickness = baseline_thickness.copy()
        else:
            fit_dir = cadence_output(detector, 2.0)
            with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as history_archive:
                joint_times = np.asarray(history_archive["times"], dtype=float)
                baseline_history = np.asarray(history_archive["baseline_thickness_nm"], dtype=float)
                if args.fixed_ice_fit_dir is None:
                    fitted_history = np.asarray(history_archive["fitted_thickness_nm"], dtype=float)
                    joint_theta = np.asarray(history_archive["shared_parameters"], dtype=float)
                    trained_ids = set(np.asarray(history_archive["source_ids"]).astype(str))
                else:
                    fitted_history = baseline_history.copy()
                    fixed_fit_dir = Path(args.fixed_ice_fit_dir)
                    fixed_summary = json.loads((fixed_fit_dir / "joint_fit_summary.json").read_text())
                    if fixed_summary.get("ice_histories_refitted"):
                        raise ValueError("The fixed-ice out-of-sample path requires an unchanged ice history")
                    with np.load(fixed_fit_dir / "joint_fit.npz", allow_pickle=False) as fixed_archive:
                        joint_theta = np.asarray(fixed_archive["parameters"], dtype=float)
                        trained_ids = set(np.asarray(fixed_archive["source_ids"]).astype(str))
        overlap = trained_ids & set(source_ids)
        if overlap:
            raise ValueError(f"OOS source selection leaked {len(overlap)} joint-fit stars for {detector}")
        if args.ice_history_fit_root is None:
            relative_days = target_mjd - reference_mjd
            joint_thickness = thickness_at_days(joint_times, fitted_history, relative_days)
            baseline_thickness = thickness_at_days(joint_times, baseline_history, relative_days)
        if joint_theta.size != 2 * model.mode_count + model.zero_point_count:
            raise ValueError(f"Joint shared parameter vector does not match OOS model dimensions for {detector}")

        theta_unmodulated = joint_theta.copy()
        theta_unmodulated[:2 * model.mode_count] = 0.0
        unmodulated_prediction, roll = model_predictions(model, theta_unmodulated, baseline_thickness)
        modulated_prediction, _ = model_predictions(model, joint_theta, joint_thickness)
        variants = {
            "no_ice_correction": magnitude - roll,
            "unmodulated": magnitude - unmodulated_prediction,
            "joint_modulated": magnitude - modulated_prediction,
        }
        detector_cells = []
        for name, corrected in variants.items():
            summary_rows = amber_variant_cells(
                detector, source_rows.rename(columns={"mjd": "mjd_amber"}), magnitude, valid,
                {name: corrected},
            )
            detector_cells.extend(summary_rows)
        output_cells.extend(detector_cells)
        output_metrics.extend(summarize_oos_detector(
            detector, source_rows, source_ids, magnitude, valid, variants))
        output_filter_counts.extend(summarize_oos_filter_counts(detector, valid, variants))
        output_extreme_rows.extend(extreme_oos_rows(
            detector, source_rows, source_ids, magnitude, error, valid, variants
        ))
        output_counts.append({
            "detector": detector,
            "training_stars_excluded": len(trained_ids),
            "held_out_stars_with_direct_successful_sed": len(source_ids),
            "held_out_measurements": int(valid.sum()),
            "SED_sources": json.dumps(sorted({direct_seds[source]["sed_source"] for source in source_ids})),
            "nearest_neighbor_seds_used": False,
        })
        print(f"{detector}: held out {len(source_ids)} stars, {int(valid.sum())} measurements", flush=True)

    cells = pd.DataFrame(output_cells)
    counts = pd.DataFrame(output_counts)
    metrics = pd.DataFrame(output_metrics)
    cells.to_csv(output_dir / "amber_out_of_sample_epoch_percentiles.csv", index=False)
    counts.to_csv(output_dir / "amber_out_of_sample_sample_counts.csv", index=False)
    metrics.to_csv(output_dir / "amber_out_of_sample_correction_metrics.csv", index=False)
    pd.DataFrame(output_filter_counts).to_csv(
        output_dir / "amber_out_of_sample_filter_within_0p5.csv", index=False)
    pd.DataFrame(output_extreme_rows).to_csv(
        output_dir / "amber_out_of_sample_largest_visit_changes.csv", index=False)
    pdf_path = output_dir / "amber_out_of_sample_corrections_all_detectors.pdf"
    joint_refit_mode = args.joint_ice_fit_dir is not None or args.joint_fit_root is not None
    with PdfPages(pdf_path) as pdf:
        plot_amber_variants(
            cells,
            output_dir,
            pdf,
            out_of_sample=True,
            output_suffix=("_joint_2day_inverse_wavelength_maxent" if joint_refit_mode else
                           "_fixed_ice_curvature" if args.fixed_ice_fit_dir is not None else None),
            cohort_note=(
                (
                    "Inverse-wavelength MaxEnt and curvature modulation; refitted 2-day tile ice histories "
                    f"from the selected production model under {args.production_root}. "
                    if joint_refit_mode and args.joint_fit_root is not None else
                    "Inverse-wavelength MaxEnt and curvature modulation; 2-day tile ice fit "
                    f"centered on production histories from {args.ice_history_fit_root}. "
                    if joint_refit_mode else
                    "Optical modulation from the reciprocal-wavelength curvature fit; "
                    f"production ice histories fixed from {args.ice_history_fit_root}. "
                ) if args.fixed_ice_fit_dir is not None or joint_refit_mode else None
            ),
            variant_labels=(
                ({"joint_modulated": "Joint modulation; refitted 2-day ice"}
                 if joint_refit_mode else
                 {"joint_modulated": "Joint modulation; fixed production ice"})
                if args.fixed_ice_fit_dir is not None or joint_refit_mode else None
            ),
        )
    print(f"Wrote held-out correction plots and metrics under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
