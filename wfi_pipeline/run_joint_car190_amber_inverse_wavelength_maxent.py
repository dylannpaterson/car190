"""Fit one detector from raw CAR201 catalogs using 1/lambda MaxEnt and curvature."""

import argparse
import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import h5py
from scipy.optimize import minimize
from astropy.time import Time

import run_amber_student_t_detectors as run
from car201_photometry_forward_jacobian import CachedPhotometryForward
from reader_compatible_joint_forward import NonPassiveReflectionError
from experiment_joint_wfi01_fixed_ice import LedModel, TILE_COUNT, add_full_basis
from experiment_joint_wfi01_inverse_wavelength_curvature import (
    InverseWavelengthCurvature,
)
from experiment_joint_wfi01_inverse_wavelength_maxent import (
    ReciprocalWavelengthMaxEnt,
)
from experiment_joint_wfi01_weekly_ice import (
    RESETS,
    led_loss,
    stellar_ice_gradient,
)
from fit_amber_wfi01_student_t import StudentObjective
from per_tile_rate_modulation import PerTileRateModulation
from support_aligned_modulation import effective_area_support


ROOT = Path(__file__).resolve().parents[1]
MAXENT_U_WEIGHT = 1890.5989417500793
CURVATURE_WEIGHT = 0.01
PHASE_SMOOTHNESS_WEIGHT = 100.0


def raw_common_training_sample(detector):
    """Rebuild Zane/Amber common repeated measurements from raw catalogs."""
    workflow = run.workflow
    config = workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]
    (
        zane_ids,
        zane_seds,
        _zane_magnitude,
        _zane_error,
        _zane_raw_valid,
        zane_valid,
        _saturated,
        _do_not_use,
        _source_has_saturation,
        _dq_flags,
        _zane_mjd,
        _zane_x,
        _zane_y,
        _zane_tile_ids,
    ) = workflow.fit_model.load_photometry_and_seds(
        workflow.PHOTOMETRY_DIR,
        workflow.FAINT_SED_PATH,
        detector=detector,
        pass_names=run.PASSES,
        window_start_utc=config["start"],
        window_end_utc=config["end"],
        require_common_source=False,
        exclude_dq_flagged=True,
        exclude_any_saturated_source=True,
    )
    zane_source_lookup = {source_id: index for index, source_id in enumerate(zane_ids)}
    filter_lookup = {name: index for index, name in enumerate(run.FILTERS)}
    pass_lookup = {name: index for index, name in enumerate(run.PASSES)}
    zane_indices = np.argwhere(zane_valid)
    zane_keys = pd.DataFrame({
        "source_id": [zane_ids[index] for index in zane_indices[:, 0]],
        "filter": [run.FILTERS[index] for index in zane_indices[:, 1]],
        "pass": [run.PASSES[index] for index in zane_indices[:, 2]],
    })

    amber = pd.read_parquet(run.AMBER)
    amber = amber.loc[amber["sca"] == detector].copy()
    amber["source_id"] = amber["source_id"].astype(str)
    epoch_map = {f"r010430300{i}00": f"ep3_pass{i}" for i in range(1, 7)}
    epoch_map.update({f"r010430400{i}00": f"ep4_pass{i}" for i in range(1, 6)})
    amber["pass"] = amber["epoch"].map(epoch_map)
    amber = amber.loc[amber["pass"].notna() & amber["source_id"].isin(zane_source_lookup)].copy()
    amber["mag_obs"] = pd.to_numeric(amber["mag_obs"], errors="coerce")
    amber["mag_err"] = pd.to_numeric(amber["mag_err"], errors="coerce")
    amber["mjd"] = pd.to_numeric(amber["mjd"], errors="coerce")
    amber["x_pixel"] = pd.to_numeric(amber["x_pixel"], errors="coerce")
    amber["y_pixel"] = pd.to_numeric(amber["y_pixel"], errors="coerce")
    amber = amber.loc[
        np.isfinite(amber["mag_obs"])
        & np.isfinite(amber["mag_err"])
        & (amber["mag_err"] > 0.0)
        & np.isfinite(amber["mjd"])
        & np.isfinite(amber["x_pixel"])
        & np.isfinite(amber["y_pixel"])
    ].copy()
    common = zane_keys.merge(
        amber,
        on=["source_id", "filter", "pass"],
        how="inner",
        validate="one_to_one",
    )
    repeated = common.groupby(["source_id", "filter"]).size()
    repeated_keys = set(repeated[repeated >= 2].index)
    key_index = pd.MultiIndex.from_frame(common[["source_id", "filter"]])
    common = common.loc[key_index.isin(repeated_keys)].copy()
    if common.empty:
        raise ValueError(f"No raw common repeated Zane/Amber measurements for {detector}")

    source_ids = sorted(common["source_id"].unique())
    source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    shape = (len(source_ids), len(run.FILTERS), len(run.PASSES))
    magnitude = np.full(shape, np.nan)
    error = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    indices = (
        common["source_id"].map(source_lookup).to_numpy(int),
        common["filter"].map(filter_lookup).to_numpy(int),
        common["pass"].map(pass_lookup).to_numpy(int),
    )
    if len(set(zip(*indices))) != len(common):
        raise ValueError(f"Duplicate raw Amber cells for {detector}")
    magnitude[indices] = common["mag_obs"].to_numpy(float)
    error[indices] = common["mag_err"].to_numpy(float)
    valid[indices] = True
    seds = {source_id: zane_seds[source_id] for source_id in source_ids}
    common["mjd_amber"] = common["mjd"]
    frames = []
    for pass_name in run.PASSES:
        frame_for_pass = common.loc[
            common["pass"] == pass_name,
            ["source_id", "filter", "mjd", "x_pixel", "y_pixel"],
        ].copy()
        frame_for_pass["pass"] = pass_name
        frames.append(frame_for_pass)
    return source_ids, seds, magnitude, error, valid, common, frames


def model_for_raw_sample(detector, source_ids, seds, valid, frames,
                         area_path, history_root, cache_root):
    workflow = run.workflow
    wavelengths, effective_area = workflow._make_wavelength_grid(
        5.0, effective_area_path=area_path, full_effective_area_support=True
    )
    photon_shapes, _, fallback_count = workflow.fit_model.reconstruct_sed_photon_shapes(
        source_ids, seds, workflow.SED_GRID_PATH, wavelengths
    )
    if fallback_count:
        raise ValueError(f"Unexpected SED fallback for {detector}: {fallback_count}")
    tile_thickness = np.asarray(
        workflow._tile_histories(
            detector,
            np.array([
                [common_pass_mjd(frames, band, pass_name) for pass_name in run.PASSES]
                for band in run.FILTERS
            ]) - Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd,
            history_root=history_root,
        ),
        dtype=float,
    )
    source_tile_ids = workflow._source_tile_ids(source_ids, frames)
    optical_arrays = workflow._optical_tile_arrays(detector, wavelengths, cache_root)
    pass_groups = workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]["zero_point_group_ids"]
    model = workflow.fit_model.RealPhotometryFit(
        wavelengths,
        effective_area,
        photon_shapes,
        tile_thickness,
        source_tile_ids,
        *optical_arrays,
        "per-gap",
        entropy_weight=0.0,
        phase_smoothness_weight=PHASE_SMOOTHNESS_WEIGHT,
        pass_zero_point_group_ids=pass_groups,
    )
    add_full_basis(model, effective_area_support(area_path))
    return model


def common_pass_mjd(frames, band, pass_name):
    frame = next(frame for frame in frames if len(frame) and frame["pass"].iloc[0] == pass_name)
    values = frame.loc[frame["filter"] == band, "mjd"].to_numpy(float)
    if not values.size:
        raise ValueError(f"No matched Amber epoch for {band}/{pass_name}")
    return float(np.median(values))


def tile_baseline_at_times(detector, history_root, times):
    baseline = np.empty((TILE_COUNT, len(times)), dtype=float)
    target_segment = np.searchsorted(RESETS, times, side="right")
    reset_reference = None
    for tile in range(TILE_COUNT):
        path = Path(history_root) / f"tile_{tile:03d}/posterior_samples.npz"
        with np.load(path, allow_pickle=False) as archive:
            days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
            thickness = np.asarray(archive["tile_ice_thickness"], dtype=float).reshape(-1, days.size)[0]
            resets = np.sort(np.asarray(
                archive.get("ice_reset_days", archive.get("ice_reset_day", [])), dtype=float
            ).reshape(-1))
        if reset_reference is None:
            reset_reference = resets
        elif not np.allclose(resets, reset_reference, rtol=0.0, atol=1e-8):
            raise ValueError(f"{detector} tile histories disagree on ice reset days")
        source_segment = np.searchsorted(resets, days, side="right")
        for segment in range(len(RESETS) + 1):
            use_target = target_segment == segment
            use_source = source_segment == segment
            if not np.any(use_target):
                continue
            if not np.any(use_source):
                raise ValueError(f"No baseline history in segment {segment} for tile {tile}")
            if (times[use_target].min() < days[use_source].min() - 1e-9
                    or times[use_target].max() > days[use_source].max() + 1e-9):
                raise ValueError(f"{detector} model time lies outside tile {tile} history")
            baseline[tile, use_target] = np.interp(
                times[use_target], days[use_source], thickness[use_source]
            )
    if reset_reference is None or not np.allclose(reset_reference, RESETS, rtol=0.0, atol=1e-8):
        raise ValueError(f"{detector} production reset days do not match the fixed boundaries")
    return baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, choices=run.workflow.DETECTORS)
    parser.add_argument("--statistics-dir", type=Path,
                        default=ROOT / "20261111/corrected_statistics_4h_thinned")
    parser.add_argument(
        "--production-root",
        type=Path,
        help="Run output root containing detector fit products; defaults to statistics-dir/production.",
    )
    parser.add_argument(
        "--joint-input-root",
        type=Path,
        help=(
            "Detector fit root containing fits/ and tile_statistics/. Defaults to "
            "the detector's relative_response/ root under production-root."
        ),
    )
    parser.add_argument(
        "--optical-cache-root",
        type=Path,
        help="Six-LED optical cache root; defaults to relative_response_mode.json under joint-input-root.",
    )
    parser.add_argument("--maxiter", type=int, default=700,
                        help="Maximum L-BFGS-B iterations in this optimizer segment")
    parser.add_argument("--ftol", type=float, default=1e-7,
                        help="Relative objective-reduction stopping tolerance")
    parser.add_argument("--gtol", type=float, default=1e-4,
                        help="Projected-gradient stopping tolerance")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same joint optimization from its saved checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reciprocal-maxent-weight", type=float, default=MAXENT_U_WEIGHT)
    parser.add_argument(
        "--response-convention", choices=("original", "eric-reader"), default="original",
        help="Separate experiment: fit OUR TMM modulation using the unchanged reader's band normalization.",
    )
    parser.add_argument("--initial-fit-dir", type=Path,
                        help="Initialize from an existing full-history joint fit, without modifying it")
    args = parser.parse_args()
    if (args.maxiter <= 0 or not np.isfinite(args.ftol) or args.ftol <= 0.0
            or not np.isfinite(args.gtol) or args.gtol <= 0.0):
        parser.error("--maxiter, --ftol, and --gtol must be finite and positive")
    detector = args.detector
    production_root = (
        Path(args.production_root)
        if args.production_root is not None
        else args.statistics_dir / "production"
    )
    default_relative_root = production_root / detector.lower() / "relative_response"
    relative_root = (
        Path(args.joint_input_root)
        if args.joint_input_root is not None
        else default_relative_root
    )
    if args.optical_cache_root is None:
        mode_path = relative_root / "relative_response_mode.json"
        mode_config = json.loads(mode_path.read_text())
        from wfi_relative_response import resolve_portable_path

        optical_cache_root = resolve_portable_path(mode_config["response_cache_root"])
    else:
        mode_path = None
        optical_cache_root = Path(args.optical_cache_root)
    history_root = relative_root / "fits"
    cache_root = optical_cache_root
    detector_number = int(detector[3:])
    area_path = (
        ROOT / "external_reference_data/roman_effarea_v8_20240301"
        / f"Roman_effarea_v8_SCA{detector_number:02d}_20240301.ecsv"
    ).resolve()
    if not area_path.is_file():
        raise FileNotFoundError(f"Missing detector-specific effective area: {area_path}")
    experiment_name = (
        "car190_amber_joint_2day_reader_compatible"
        if args.response_convention == "eric-reader" else
        "car190_amber_joint_2day_reciprocal_maxent"
    )
    output = args.output_dir or relative_root / "experiments" / experiment_name
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing fit output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_ids, seds, magnitude, error, valid, common, frames = raw_common_training_sample(detector)
    print(f"{detector}: raw common sample has {len(source_ids)} stars and {int(valid.sum())} Amber measurements", flush=True)
    model = model_for_raw_sample(
        detector, source_ids, seds, valid, frames, area_path, history_root, cache_root
    )
    objective = StudentObjective(model, magnitude, error, valid)
    forward_class = CachedPhotometryForward
    stellar_gradient_function = stellar_ice_gradient
    mode = model.mode_count
    led_kwargs = {}
    if args.response_convention == "eric-reader":
        h5_path = ROOT / "20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle/eric_v3_reference/ice_response_model_v3.h5"
        with h5py.File(h5_path, "r") as h5:
            led_kwargs = {
                "led_wavelength_nm": h5["led_bands/wavelength_nm"][()],
                "led_weights": h5["led_bands/weight"][()],
            }
    led = LedModel(
        effective_area_support(area_path), detector=detector,
        relative_response_root=relative_root,
        optical_cache_root=cache_root,
        **led_kwargs,
    )
    if led.basis.shape[0] != mode:
        raise ValueError(f"{detector}: CAR190/Amber modulation basis mismatch")

    reference_mjd = Time(run.workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    star_days = np.array([
        [common_pass_mjd(frames, band, pass_name) - reference_mjd for pass_name in run.PASSES]
        for band in run.FILTERS
    ])
    led_days = np.unique(np.concatenate([item[9] for tiles in led.bands for item in tiles]))
    times = np.unique(np.r_[led_days, star_days.ravel()])
    baseline = tile_baseline_at_times(detector, history_root, times)
    trajectory = PerTileRateModulation(times, baseline, RESETS, spacing_days=2.0)
    star_indices = np.searchsorted(times, star_days)
    np.testing.assert_allclose(times[star_indices], star_days, rtol=0.0, atol=1e-10)
    led_indices = [[np.searchsorted(times, item[9]) for item in tiles] for tiles in led.bands]

    if args.response_convention == "eric-reader":
        from reader_compatible_joint_forward import (
            ReaderCompatiblePhotometryForward,
            stellar_ice_gradient as reader_stellar_ice_gradient,
        )
        forward_class = ReaderCompatiblePhotometryForward
        stellar_gradient_function = reader_stellar_ice_gradient
        # Preserve the existing source model's fixed decon throughput terms,
        # not Eric's unrelated fitted gray levels. Roll/L-flat terms remain free
        # nuisance parameters in the stellar objective, as before.
        grey_offsets = np.zeros((TILE_COUNT, 3), dtype=float)
        for tile in range(TILE_COUNT):
            gray_source = history_root / f"tile_{tile:03d}/posterior_samples.npz"
            with np.load(gray_source, allow_pickle=False) as source:
                np.testing.assert_allclose(source["ice_reset_days"], RESETS, atol=1e-8, rtol=0)
                grey_offsets[tile, 1] = float(np.asarray(source["post_reset_tile_log_offset"]).reshape(-1)[0])
                grey_offsets[tile, 2] = float(np.asarray(source["post_third_decon_tile_log_offset"]).reshape(-1)[0])

        def gray_at_days(days):
            return grey_offsets[:, np.searchsorted(RESETS, np.asarray(days), side="right")]

        model.reader_gray_log_by_tile = gray_at_days(star_days)

    curvature = InverseWavelengthCurvature(
        effective_area_support(area_path), mode, grid_points=513,
        amplitude_scale=0.30, phase_scale=1.0,
    )
    maxent = ReciprocalWavelengthMaxEnt(curvature)
    nopt = len(objective.coordinate_scale) + 1
    ice_parameters = np.zeros((TILE_COUNT, trajectory.parameter_count), dtype=float)
    initial_shared = np.zeros(objective.coordinate_scale.size, dtype=float)
    initial = np.r_[initial_shared / objective.coordinate_scale, np.log(6.0), ice_parameters.ravel()]
    if args.initial_fit_dir is not None:
        with np.load(args.initial_fit_dir / "joint_fit.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["source_ids"].astype(str), source_ids)
            np.testing.assert_array_equal(previous["stellar_valid"], valid)
            previous_shared = np.asarray(previous["parameters"], float)
        with np.load(args.initial_fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as previous:
            np.testing.assert_allclose(previous["times"], times, rtol=0, atol=1e-10)
            np.testing.assert_allclose(previous["baseline_thickness_nm"], baseline, rtol=0, atol=1e-10)
            previous_ice = np.asarray(previous["ice_parameters"], float)
            previous_extra = float(previous["extra_student_t_scale_mmag"])
        initial = np.r_[previous_shared / objective.coordinate_scale,
                        np.log(previous_extra), previous_ice.ravel()]
    if args.resume:
        checkpoint_path = output / "optimizer_checkpoint.npz"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No joint-fit checkpoint to resume: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            initial = np.asarray(checkpoint["coordinates"], dtype=float)
        if initial.shape != (nopt + TILE_COUNT * trajectory.parameter_count,):
            raise ValueError("Saved optimizer checkpoint dimensions do not match this fit")

    sigma = np.full(trajectory.parameter_count, 1.5)
    sigma[trajectory.specs[2]["initial_index"]] = 0.1
    shared_bounds = [(-3.0, 3.0)] * mode + [(-10.0, 10.0)] * mode
    shared_bounds += [(-25.0, 25.0)] * model.zero_point_count
    shared_bounds += [(np.log(1e-4), np.log(100.0))]
    shared_bounds[7] = shared_bounds[8] = (-6.0, 6.0)
    bounds = shared_bounds + [(-np.log(4.0), np.log(4.0))] * (
        TILE_COUNT * trajectory.parameter_count
    )
    calls = 0

    def total(values):
        nonlocal calls
        ice_parameters_local = values[nopt:].reshape(TILE_COUNT, trajectory.parameter_count)
        thickness, jacobian = trajectory.evaluate(ice_parameters_local, with_jacobian=True)
        model.tile_thickness_by_id = thickness[:, star_indices]
        objective.forward = forward_class(model)
        try:
            stellar, stellar_gradient = objective(values[:nopt])
        except NonPassiveReflectionError:
            # Keep line-search probes inside the passive optical domain instead
            # of aborting a long multistart/continuation run.
            optical = values[:2 * mode]
            barrier = 1.0e12 + 1.0e8 * float(optical @ optical)
            barrier_gradient = np.zeros_like(values)
            barrier_gradient[:2 * mode] = 2.0e8 * optical
            return barrier, barrier_gradient
        physical = values[:2 * mode] * objective.coordinate_scale[:2 * mode]
        maxent_cost, maxent_gradient = maxent.evaluate(physical, args.reciprocal_maxent_weight)
        curvature_cost, curvature_gradient, _, _ = curvature.evaluate(physical, CURVATURE_WEIGHT)
        led_cost, led_optical, led_ice = led_loss(
            led, physical, thickness, jacobian, led_indices
        )
        stellar_ice = stellar_gradient_function(objective, values[:nopt], jacobian, star_indices)
        ice_prior = 0.5 * np.sum((ice_parameters_local / sigma[None, :]) ** 2)
        gradient = np.empty_like(values)
        gradient[:nopt] = stellar_gradient
        gradient[:2 * mode] += (
            maxent_gradient + curvature_gradient + led_optical
        ) * objective.coordinate_scale[:2 * mode]
        gradient[nopt:] = (
            led_ice + stellar_ice + ice_parameters_local / sigma[None, :] ** 2
        ).ravel()
        calls += 1
        if calls % 10 == 0:
            print(
                f"{detector}: eval {calls}; joint={stellar + led_cost + ice_prior + maxent_cost + curvature_cost:.5f}; "
                f"CAR190 chi2={2*led_cost:.3f}; reciprocal MaxEnt={maxent_cost:.3f}; curvature={curvature_cost:.3f}",
                flush=True,
            )
        return stellar + led_cost + ice_prior + maxent_cost + curvature_cost, gradient

    initial_value, initial_gradient = total(initial)
    check_indices = (0, mode, nopt, len(initial) - 1)
    gradient_errors = []
    for index in check_indices:
        step = 1e-4
        upper, lower = initial.copy(), initial.copy()
        upper[index] += step
        lower[index] -= step
        numerical = (total(upper)[0] - total(lower)[0]) / (2.0 * step)
        gradient_errors.append(abs(numerical - initial_gradient[index]) / (1.0 + abs(numerical)))
    if max(gradient_errors) > 2e-3:
        raise RuntimeError(f"{detector}: joint reciprocal-MaxEnt gradient check failed: {gradient_errors}")

    def checkpoint(values):
        np.savez_compressed(output / "optimizer_checkpoint.npz", coordinates=values)

    started = time.monotonic()
    result = minimize(
        total, initial, jac=True, bounds=bounds, method="L-BFGS-B", callback=checkpoint,
        options={"maxiter": args.maxiter, "ftol": args.ftol, "gtol": args.gtol, "maxls": 30},
    )
    final_value, _ = total(result.x)
    fitted_parameters = result.x[nopt:].reshape(TILE_COUNT, trajectory.parameter_count)
    fitted_thickness, fitted_ice_jacobian = trajectory.evaluate(
        fitted_parameters, with_jacobian=True
    )
    model.tile_thickness_by_id = fitted_thickness[:, star_indices]
    objective.forward = forward_class(model)
    stellar_value, stellar_prediction, _, _, _, _ = objective.state(result.x[:nopt])
    shared_parameters = result.x[:nopt - 1] * objective.coordinate_scale
    maxent_cost, _ = maxent.evaluate(shared_parameters[:2 * mode], args.reciprocal_maxent_weight)
    curvature_cost, _, amp_cost, phase_cost = curvature.evaluate(
        shared_parameters[:2 * mode], CURVATURE_WEIGHT
    )
    fitted_led_cost, _, _ = led_loss(
        led,
        shared_parameters[:2 * mode],
        fitted_thickness,
        fitted_ice_jacobian,
        led_indices,
    )
    led_observation_count = sum(
        np.asarray(item[6]).size for band_tiles in led.bands for item in band_tiles
    )
    report = {
        "detector": detector,
        "run_mode": "raw_common_zane_amber_student_t_joint_fit",
        "response_convention": args.response_convention,
        "full_independent_tile_histories_in_fit": True,
        "initial_fit_dir": str(args.initial_fit_dir) if args.initial_fit_dir is not None else None,
        "training_selection": (
        "raw ep3v2/ep4 Zane rows with current DQ/saturation cuts, union of pass source coverage, "
            "successful faint-v2 SEDs, joined to finite Amber rows; source/filter groups require >=2 visits; "
            "no prior fit tables, manual exclusions, or repeat-outlier output tables used"
        ),
        "training_stars": len(source_ids),
        "training_measurements": int(valid.sum()),
        "CAR190_observations": int(led_observation_count),
        "initialization": (
            "existing saved joint modulation, roll offsets, extra scale, and full tile rate controls"
            if args.initial_fit_dir is not None else
            "zero spectral modulation, zero roll offsets, and zero rate-control multipliers; no warm-up fit"
        ),
        "ice_history_fit_root": str(history_root),
        "optical_cache_root": str(cache_root),
        "relative_response_root": str(relative_root),
        "joint_input_root": str(relative_root),
        "spectral_maxent": "relative entropy of reconstructed log-amplitude integrated over normalized reciprocal wavelength",
        "reciprocal_maxent_weight": float(args.reciprocal_maxent_weight),
        "curvature_weight": CURVATURE_WEIGHT,
        "optimizer_ftol": float(args.ftol),
        "optimizer_gtol": float(args.gtol),
        "curvature_grid_points": 513,
        "phase_smoothness_weight": PHASE_SMOOTHNESS_WEIGHT,
        "ice_rate_control_spacing_days": 2.0,
        "total_ice_parameters": int(fitted_parameters.size),
        "ice_histories_refitted": True,
        "TMM_parameters_refitted": False,
        "reader_gray_log_offsets_by_tile": grey_offsets.tolist() if args.response_convention == "eric-reader" else None,
        "reader_gray_source": str(history_root / "tile_###/posterior_samples.npz") if args.response_convention == "eric-reader" else None,
        "maxent_penalty_final": float(maxent_cost),
        "curvature_penalty_final": float(curvature_cost),
        "curvature_log_amplitude_penalty_final": float(amp_cost),
        "curvature_phase_penalty_final": float(phase_cost),
        "car190_chi2": float(2.0 * fitted_led_cost),
        "stellar_objective_including_phase_regularization": float(stellar_value),
        "joint_objective_initial": float(initial_value),
        "joint_objective_final": float(final_value),
        "extra_student_t_scale_mmag": float(np.exp(result.x[nopt - 1])),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "seconds": float(time.monotonic() - started),
        "gradient_check_max_relative_discrepancy": float(max(gradient_errors)),
        "ice_change_p16_median_p84_nm": np.percentile(fitted_thickness - baseline, [16, 50, 84], axis=0).tolist(),
        "max_abs_ice_change_nm": float(np.max(np.abs(fitted_thickness - baseline))),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "joint_weekly_ice_fit.npz",
        times=times,
        baseline_thickness_nm=baseline,
        fitted_thickness_nm=fitted_thickness,
        ice_parameters=fitted_parameters,
        shared_parameters=shared_parameters,
        extra_student_t_scale_mmag=np.exp(result.x[nopt - 1]),
        source_ids=np.asarray(source_ids),
        amber_prediction_mag=stellar_prediction / 1000.0,
        car190_chi2=2.0 * fitted_led_cost,
    )
    np.savez_compressed(
        output / "joint_fit.npz",
        parameters=shared_parameters,
        source_ids=np.asarray(source_ids),
        stellar_valid=valid,
        stellar_prediction_mag=stellar_prediction / 1000.0,
        wavelengths_nm=model.wavelengths_nm,
        basis=model.basis,
        car190_chi2=2.0 * fitted_led_cost,
        ice_history_fit_root=str(history_root),
        optical_cache_root=str(cache_root),
    )
    np.savez_compressed(output / "training_inputs.npz", magnitude=magnitude, error=error, valid=valid)
    console_report = {
        key: value for key, value in report.items()
        if key not in ("ice_change_p16_median_p84_nm", "reader_gray_log_offsets_by_tile")
    }
    print(json.dumps(console_report, indent=2), flush=True)
    if not result.success:
        raise RuntimeError(f"{detector}: joint fit failed: {result.message}")


if __name__ == "__main__":
    main()
