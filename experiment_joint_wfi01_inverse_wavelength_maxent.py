"""WFI01 sensitivity fit swapping coefficient MaxEnt for a 1/lambda entropy prior."""

import argparse
import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import mock_fit_wfi_star_reflectance_modulation as mock
import run_amber_student_t_detectors as run
from experiment_joint_wfi01_fixed_ice import LedModel, TILE_COUNT, add_full_basis
from experiment_joint_wfi01_inverse_wavelength_curvature import InverseWavelengthCurvature
from fit_amber_wfi01_student_t import StudentObjective
from support_aligned_modulation import effective_area_support


ENTROPY_WEIGHT_NM_BASIS = 1000.0
PHASE_SMOOTHNESS_WEIGHT = 100.0
CURVATURE_WEIGHT = 0.01
AMPLITUDE_SCALE = 0.30
PHASE_SCALE_RAD = 1.0


def load_training_data(source_ids, valid):
    """Recreate fitted Amber cells directly from the current Amber catalog."""
    frame = pd.read_parquet(run.AMBER)
    frame = frame.loc[frame["sca"] == "WFI01"].copy()
    epoch_map = {f"r010430300{i}00": f"ep3_pass{i}" for i in range(1, 7)}
    epoch_map.update({f"r010430400{i}00": f"ep4_pass{i}" for i in range(1, 6)})
    frame["pass"] = frame["epoch"].map(epoch_map)
    frame = frame.loc[frame["pass"].notna()].copy()
    frame["source_id"] = frame["source_id"].astype(str)

    source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    filter_lookup = {name: index for index, name in enumerate(run.FILTERS)}
    pass_lookup = {name: index for index, name in enumerate(run.PASSES)}
    valid_key = (
        frame["source_id"].map(source_lookup),
        frame["filter"].map(filter_lookup),
        frame["pass"].map(pass_lookup),
    )
    present = np.logical_and.reduce([index.notna().to_numpy() for index in valid_key])
    frame = frame.loc[present].copy()
    indices = (
        frame["source_id"].map(source_lookup).to_numpy(int),
        frame["filter"].map(filter_lookup).to_numpy(int),
        frame["pass"].map(pass_lookup).to_numpy(int),
    )
    frame = frame.loc[valid[indices]].copy()
    if frame.duplicated(["source_id", "filter", "pass"]).any():
        raise ValueError("Amber training cells contain duplicate source/filter/pass measurements")

    shape = valid.shape
    magnitude = np.full(shape, np.nan)
    error = np.full(shape, np.nan)
    indices = (
        frame["source_id"].map(source_lookup).to_numpy(int),
        frame["filter"].map(filter_lookup).to_numpy(int),
        frame["pass"].map(pass_lookup).to_numpy(int),
    )
    magnitude[indices] = pd.to_numeric(frame["mag_obs"], errors="raise").to_numpy(float)
    error[indices] = pd.to_numeric(frame["mag_err"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(magnitude[valid])) or not np.all(np.isfinite(error[valid])):
        raise ValueError("A fitted Amber measurement or uncertainty is non-finite")
    if np.any(error[valid] <= 0.0):
        raise ValueError("A fitted Amber uncertainty is not positive")

    common = frame.copy()
    common["mjd_amber"] = pd.to_numeric(common["mjd"], errors="raise")
    frames = [
        common.loc[common["pass"] == pass_name,
                   ["source_id", "filter", "mjd", "x_pixel", "y_pixel"]].copy()
        for pass_name in run.PASSES
    ]
    return magnitude, error, common, frames


class ReciprocalWavelengthMaxEnt:
    """Relative entropy of the reconstructed log-amplitude curve in u=1/lambda."""

    def __init__(self, grid):
        self.u = np.asarray(grid.u, dtype=float)
        self.basis = np.asarray(grid.basis, dtype=float)
        if self.u.ndim != 1 or self.basis.ndim != 2 or self.basis.shape[1] != self.u.size:
            raise ValueError("Reciprocal-wavelength entropy grid and basis are misaligned")
        step = np.diff(self.u)
        if np.any(step <= 0.0) or not np.allclose(step, step[0], rtol=1e-12, atol=0.0):
            raise ValueError("The MaxEnt integration grid must be uniform in reciprocal wavelength")
        self.integration_weights = np.full(self.u.size, step[0], dtype=float)
        self.integration_weights[[0, -1]] *= 0.5
        self.mode_count = self.basis.shape[0]

    @staticmethod
    def density(log_amplitude):
        value = np.exp(log_amplitude) * log_amplitude - np.expm1(log_amplitude)
        small = np.abs(log_amplitude) < 1e-4
        value = np.where(small, 0.5 * log_amplitude**2, value)
        derivative = np.exp(log_amplitude) * log_amplitude
        derivative = np.where(small, log_amplitude, derivative)
        return value, derivative

    def evaluate(self, parameters, weight):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (2 * self.mode_count,):
            raise ValueError("MaxEnt requires amplitude and phase parameter blocks")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("The reciprocal-wavelength MaxEnt weight must be positive")
        amplitude = parameters[:self.mode_count] @ self.basis
        density, derivative = self.density(amplitude)
        weighted_density = self.integration_weights * density
        cost = float(weight) * float(np.sum(weighted_density))
        gradient = np.zeros_like(parameters)
        gradient[:self.mode_count] = (
            float(weight) * self.basis @ (self.integration_weights * derivative)
        )
        return cost, gradient


def load_model(area_path, wavelengths, source_ids, magnitude, error, frames,
               valid, ice_history_root, optical_cache_root):
    workflow = run.workflow
    effective_wavelengths, effective_area = workflow._make_wavelength_grid(
        5.0,
        effective_area_path=area_path,
        full_effective_area_support=True,
    )
    np.testing.assert_array_equal(effective_wavelengths, wavelengths)
    target_mjd = np.empty((len(run.FILTERS), len(run.PASSES)), dtype=float)
    for pass_index, frame in enumerate(frames):
        for filter_index, band in enumerate(run.FILTERS):
            values = frame.loc[frame["filter"] == band, "mjd"].to_numpy(float)
            if not values.size:
                raise ValueError(f"No training Amber epoch for {band}/{run.PASSES[pass_index]}")
            target_mjd[filter_index, pass_index] = float(np.median(values))
    from astropy.time import Time
    reference_mjd = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    target_days = target_mjd - reference_mjd
    tile_thickness = workflow._tile_histories(
        "WFI01", target_days, history_root=ice_history_root
    )
    source_tiles = workflow._source_tile_ids(source_ids, frames)
    sed_assignments = workflow._load_successful_seds(
        ((workflow.FAINT_SED_PATH, "faint_v2"),), source_ids
    )
    if len(sed_assignments) != len(source_ids):
        raise ValueError("A fitted WFI01 source has no successful faint-v2 SED")
    photon_shapes, _, fallback_count = workflow.fit_model.reconstruct_sed_photon_shapes(
        source_ids, sed_assignments, workflow.SED_GRID_PATH, wavelengths
    )
    if fallback_count:
        raise ValueError(f"Unexpected SED fallback for {fallback_count} fitted training stars")
    optical_arrays = workflow._optical_tile_arrays(
        "WFI01", wavelengths, optical_cache_root
    )
    pass_groups = workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]["zero_point_group_ids"]
    model = workflow.fit_model.RealPhotometryFit(
        wavelengths,
        effective_area,
        photon_shapes,
        tile_thickness,
        source_tiles,
        *optical_arrays,
        "per-gap",
        entropy_weight=0.0,
        phase_smoothness_weight=PHASE_SMOOTHNESS_WEIGHT,
        pass_zero_point_group_ids=pass_groups,
    )
    add_full_basis(model, effective_area_support(area_path))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--ice-history-fit-root", type=Path, required=True)
    parser.add_argument("--optical-cache-root", type=Path, required=True)
    parser.add_argument("--relative-response-root", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.maxiter <= 0:
        parser.error("--maxiter must be positive")

    detector = "WFI01"
    source_fit = args.sweep_root / "weight_0p01"
    output = args.output_dir or args.sweep_root / "weight_0p01_maxent_inverse_wavelength"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing fit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    area, _ = run.canonical_configuration(detector)
    with np.load(source_fit / "joint_fit.npz", allow_pickle=False) as archive:
        initial_parameters = np.asarray(archive["parameters"], dtype=float)
        source_ids = np.asarray(archive["source_ids"]).astype(str)
        valid = np.asarray(archive["stellar_valid"], dtype=bool)
        wavelengths = np.asarray(archive["wavelength_nm"], dtype=float)
        saved_basis = np.asarray(archive["basis"], dtype=float)
        saved_prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float)
    source_summary = json.loads((source_fit / "joint_fit_summary.json").read_text())
    magnitude, error, common, frames = load_training_data(source_ids, valid)
    model = load_model(
        area,
        wavelengths,
        source_ids,
        magnitude,
        error,
        frames,
        valid,
        args.ice_history_fit_root,
        args.optical_cache_root,
    )
    np.testing.assert_allclose(model.basis, saved_basis, rtol=0.0, atol=0.0)
    objective = StudentObjective(model, magnitude, error, valid)
    mode = model.mode_count
    if initial_parameters.shape != objective.coordinate_scale.shape:
        raise ValueError("Saved 0.01 fit parameters do not match reconstructed WFI01 model")
    led = LedModel(
        effective_area_support(area),
        detector=detector,
        relative_response_root=args.relative_response_root,
    )
    if led.basis.shape[0] != mode:
        raise ValueError("CAR190 and Amber modulation bases do not match")

    # Reconstruct the ordinary coefficient-wise MaxEnt cost at the warm start.
    model.entropy_weight = ENTROPY_WEIGHT_NM_BASIS
    old_maxent_residual = model.maxent_regularization_residuals(
        initial_parameters[:2 * mode]
    )
    old_maxent_cost = 0.5 * float(old_maxent_residual @ old_maxent_residual)
    model.entropy_weight = 0.0

    reciprocal_grid = InverseWavelengthCurvature(
        effective_area_support(area), mode, grid_points=513,
        amplitude_scale=AMPLITUDE_SCALE, phase_scale=PHASE_SCALE_RAD,
    )
    maxent_u = ReciprocalWavelengthMaxEnt(reciprocal_grid)
    unit_functional_cost, _ = maxent_u.evaluate(initial_parameters[:2 * mode], 1.0)
    if not np.isfinite(unit_functional_cost) or unit_functional_cost <= 0.0:
        raise ValueError("Cannot match reciprocal-wavelength MaxEnt to the starting prior cost")
    maxent_u_weight = old_maxent_cost / unit_functional_cost

    coordinates = np.concatenate([
        initial_parameters / objective.coordinate_scale,
        [np.log(float(source_summary["extra_student_t_scale_mmag"]))],
    ])
    reconstructed_prediction = objective.prediction(coordinates[:-1]) / 1000.0
    np.testing.assert_allclose(reconstructed_prediction, saved_prediction, rtol=0.0, atol=2e-9)

    curvature = InverseWavelengthCurvature(
        effective_area_support(area), mode, grid_points=513,
        amplitude_scale=AMPLITUDE_SCALE, phase_scale=PHASE_SCALE_RAD,
    )

    def total(values):
        stellar_value, stellar_gradient = objective(values)
        physical = values[:2 * mode] * objective.coordinate_scale[:2 * mode]
        maxent_value, maxent_gradient = maxent_u.evaluate(physical, maxent_u_weight)
        curvature_value, curvature_gradient, _, _ = curvature.evaluate(
            physical, CURVATURE_WEIGHT
        )
        led_residual, led_jacobian, _ = led.evaluate(physical)
        led_value = 0.5 * float(led_residual @ led_residual)
        value = stellar_value + maxent_value + curvature_value + led_value
        gradient = stellar_gradient.copy()
        parameter_scale = objective.coordinate_scale[:2 * mode]
        gradient[:2 * mode] += (
            maxent_gradient + curvature_gradient
        ) * parameter_scale
        gradient[:2 * mode] += (led_jacobian.T @ led_residual) * parameter_scale
        return value, gradient

    initial_value, initial_gradient = total(coordinates)
    gradient_checks = []
    for index in (0, mode, 2 * mode, len(coordinates) - 1):
        step = 1e-4
        upper, lower = coordinates.copy(), coordinates.copy()
        upper[index] += step
        lower[index] -= step
        numerical = (total(upper)[0] - total(lower)[0]) / (2.0 * step)
        gradient_checks.append(
            abs(numerical - initial_gradient[index]) / (1.0 + abs(numerical))
        )
    if max(gradient_checks) > 1e-2:
        raise RuntimeError(f"Inverse-wavelength MaxEnt gradient check failed: {gradient_checks}")

    fit_bounds = [(-3.0, 3.0)] * mode + [(-10.0, 10.0)] * mode
    fit_bounds += [(-25.0, 25.0)] * model.zero_point_count
    fit_bounds += [(np.log(1e-4), np.log(100.0))]
    fit_bounds[7] = fit_bounds[8] = (-6.0, 6.0)
    started = time.monotonic()
    result = minimize(
        total,
        coordinates,
        method="L-BFGS-B",
        jac=True,
        bounds=fit_bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-9, "gtol": 1e-4, "maxls": 30},
    )
    final_value, _ = total(result.x)
    stellar_value, prediction, _, _, _, _ = objective.state(result.x)
    parameters = result.x[:-1] * objective.coordinate_scale
    physical_modulation = parameters[:2 * mode]
    led_residual, _, _ = led.evaluate(physical_modulation)
    maxent_value, _ = maxent_u.evaluate(physical_modulation, maxent_u_weight)
    curvature_value, _, amp_cost, phase_cost = curvature.evaluate(
        physical_modulation, CURVATURE_WEIGHT
    )
    amplitude = physical_modulation[:mode] @ maxent_u.basis
    density, _ = maxent_u.density(amplitude)
    initial_curvature = curvature.evaluate(initial_parameters[:2 * mode], CURVATURE_WEIGHT)[0]
    summary = {
        "detector": detector,
        "source_fit_directory": str(source_fit),
        "ice_history_fit_root": str(args.ice_history_fit_root),
        "optical_cache_root": str(args.optical_cache_root),
        "relative_response_root": str(args.relative_response_root),
        "ice_histories_refitted": False,
        "TMM_parameters_refitted": False,
        "stellar_sources": len(source_ids),
        "stellar_measurements": int(valid.sum()),
        "CAR190_observations": len(led_residual),
        "curvature_weight": CURVATURE_WEIGHT,
        "maxent_form": "integral of relative entropy of reconstructed log-amplitude over uniform normalized reciprocal wavelength",
        "maxent_reciprocal_wavelength_grid_points": int(len(maxent_u.u)),
        "maxent_u_weight_matched_to_existing_cost_at_initial_state": float(maxent_u_weight),
        "existing_coefficient_maxent_cost_at_initial_state": float(old_maxent_cost),
        "functional_maxent_cost_at_initial_state": float(
            maxent_u.evaluate(initial_parameters[:2 * mode], maxent_u_weight)[0]
        ),
        "functional_maxent_cost_final": float(maxent_value),
        "functional_maxent_integrated_relative_entropy_final": float(
            np.sum(maxent_u.integration_weights * density)
        ),
        "curvature_penalty_final": float(curvature_value),
        "curvature_log_amplitude_component_final": float(amp_cost),
        "curvature_phase_component_final": float(phase_cost),
        "baseline_curvature_penalty_at_weight_0p01": float(initial_curvature),
        "CAR190_chi2": float(led_residual @ led_residual),
        "stellar_objective_including_phase_regularization": float(stellar_value),
        "joint_objective_final": float(final_value),
        "extra_student_t_scale_mmag": float(np.exp(result.x[-1])),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "seconds": float(time.monotonic() - started),
        "max_gradient_check_relative_difference": float(max(gradient_checks)),
    }
    (output / "joint_fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output / "joint_fit.npz",
        parameters=parameters,
        stellar_prediction_mag=prediction / 1000.0,
        led_normalized_residual=led_residual,
        stellar_valid=valid,
        source_ids=source_ids,
        wavelength_nm=wavelengths,
        basis=model.basis,
    )
    pd.DataFrame({
        "mode": model.basis_names,
        "log_amplitude": parameters[:mode],
        "phase_rad": parameters[mode:2 * mode],
    }).to_csv(output / "joint_modulation_parameters.csv", index=False)
    pd.DataFrame({
        "wavelength_nm": reciprocal_grid.wavelengths_nm,
        "normalized_inverse_wavelength": reciprocal_grid.u,
        "log_amplitude": amplitude,
        "relative_entropy_density": density,
    }).to_csv(output / "reciprocal_wavelength_maxent_density.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    if not result.success:
        raise RuntimeError(f"Inverse-wavelength MaxEnt fit failed: {result.message}")


if __name__ == "__main__":
    main()
