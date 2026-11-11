"""Fixed-nu Student-t experiment on the exact matched Amber WFI01 sample.

The Student-t scale is sqrt(catalog_error**2 + extra_scale**2). It is a scale,
not an RMS. Each star/filter constant is re-estimated under this likelihood.
"""

import json
import os
from pathlib import Path
import time
import warnings

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import check_amber_wfi01_photometry as comparison
import heatmap_car201_allstar_model_correction as workflow
from car201_photometry_forward_jacobian import CachedPhotometryForward

NU = 4.
OUTPUT = comparison.OUTPUT.with_name(comparison.OUTPUT.name.replace("fitted_floor", "student_t_nu4"))


def profile_student_locations(values, scales, valid, nu=NU, tolerance=1e-8):
    """Find robust locations from three independent starts per group.

    Starts are the median, Gaussian weighted mean and best observed-value
    candidate. Keep the converged location with the smallest Student-t cost.
    Missing observations contribute neither cost nor weight.
    """
    values = np.asarray(values, float)
    scales = np.asarray(scales, float)
    valid = np.asarray(valid, bool)
    if np.any(valid.sum(axis=1) < 2):
        raise ValueError("Robust locations require repeated groups")
    safe_values = np.where(valid, values, 0.)
    safe_scales = np.where(valid, scales, 1.)
    median = np.nanmedian(np.where(valid, values, np.nan), axis=1)
    weights = np.where(valid, 1. / safe_scales**2, 0.)
    mean = (weights * safe_values).sum(axis=1) / weights.sum(axis=1)
    # At most 11 observations per group: screening all observed-value starts
    # avoids selecting a distant outlier mode just from the initial mean.
    candidate_residual = safe_values[:, :, None] - safe_values[:, None, :]
    candidate_cost = np.where(
        valid[:, :, None],
        .5 * (nu + 1.) * np.log1p(candidate_residual**2 / (nu * safe_scales[:, :, None]**2)),
        0.,
    ).sum(axis=1)
    candidate_cost = np.where(valid, candidate_cost, np.inf)
    best_candidate = safe_values[np.arange(len(values)), np.argmin(candidate_cost, axis=1)]
    locations = np.stack([median, mean, best_candidate], axis=1)
    active = np.ones(locations.shape, dtype=bool)
    for _ in range(500):
        gi, si = np.where(active)
        if not len(gi):
            break
        residual = safe_values[gi] - locations[gi, si, None]
        weight = np.where(valid[gi], 1. / (nu*safe_scales[gi]**2 + residual**2), 0.)
        new = (weight*safe_values[gi]).sum(axis=1) / weight.sum(axis=1)
        difference = np.abs(new-locations[gi,si])
        locations[gi,si] = new
        active[gi[difference<tolerance],si[difference<tolerance]] = False
    residual = safe_values[:, :, None] - locations[:, None, :]
    costs = np.where(valid[:, :, None],
                     .5 * (nu + 1.) * np.log1p(residual**2 / (nu * safe_scales[:, :, None]**2)),
                     0.).sum(axis=1)
    best = np.argmin(costs, axis=1)
    location = locations[np.arange(len(values)), best]
    residual = np.where(valid, safe_values - location[:, None], 0.)
    score = np.where(valid, (nu + 1.) * residual / (nu * safe_scales**2 + residual**2), 0.)
    return location, residual, score


def build_model():
    archive = np.load(comparison.OUTPUT / "predictions.npz", allow_pickle=False)
    common = pd.read_csv(comparison.OUTPUT / "common_measurements.csv", dtype={"source_id": str})
    ids = archive["source_ids"].astype(str).tolist()
    frames = []
    for epoch in workflow.PASSES:
        rows = common.loc[common["pass"] == epoch]
        frames.append(rows[["source_id", "filter", "mjd_amber", "x_pixel_amber", "y_pixel_amber"]].rename(
            columns={"mjd_amber": "mjd", "x_pixel_amber": "x_pixel", "y_pixel_amber": "y_pixel"}))
    sed = workflow._load_successful_seds(((workflow.FAINT_SED_PATH, "faint_v2"),), ids)
    wave, area = workflow._make_wavelength_grid(5.)
    from astropy.time import Time
    reference = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    _, model = workflow._predict_detector("WFI01", frames, ids, sed, workflow.SED_GRID_PATH,
                                          comparison.BASE, wave, area, reference, return_model=True)
    model.entropy_weight = 1000.
    model.phase_smoothness_weight = 100.
    return archive, common, model


class StudentObjective:
    def __init__(self, model, magnitudes, errors, valid):
        self.model = model
        self.forward = CachedPhotometryForward(model)
        self.valid_full = valid
        self.active_groups = valid.reshape((-1, valid.shape[-1])).sum(axis=1) >= 2
        self.valid = valid.reshape((-1, valid.shape[-1]))[self.active_groups]
        mag = magnitudes.reshape((-1, valid.shape[-1]))[self.active_groups] * 1000.
        self.reference = np.nanmedian(np.where(self.valid, mag, np.nan), axis=1)
        self.observed = np.where(self.valid, mag-self.reference[:, None], 0.)
        self.errors = errors.reshape((-1, valid.shape[-1]))[self.active_groups] * 1000.
        self.errors = np.where(self.valid, self.errors, 1.)
        self.coordinate_scale = np.concatenate([
            np.full(2*model.mode_count, .1), np.full(model.zero_point_count, .01)])
        self.calls = 0

    def prediction(self, coordinates, with_jacobian=False):
        parameters = coordinates * self.coordinate_scale
        log_response, modulation_jacobian = self.forward.evaluate(parameters[:2*self.model.mode_count])
        zp = parameters[2*self.model.mode_count:].reshape((len(workflow.FILTERS), self.model.zero_point_group_count))
        for pi, group in enumerate(self.model.pass_zero_point_group_ids):
            if group:
                log_response[:, :, pi] += zp[:, group-1][None, :]
        prediction = -1000. * log_response / comparison.fitter.LOG_FLUX_PER_MAG
        if with_jacobian:
            jac = np.zeros(prediction.shape + (len(parameters),))
            jac[:, :, :, :2*self.model.mode_count] = modulation_jacobian
            for pi, group in enumerate(self.model.pass_zero_point_group_ids):
                if group:
                    for fi in range(self.model.filter_count):
                        index = 2*self.model.mode_count + fi*self.model.zero_point_group_count + group-1
                        jac[:, fi, pi, index] = 1.
            jac *= -1000. / comparison.fitter.LOG_FLUX_PER_MAG
            jac *= self.coordinate_scale[None, None, None, :]
            return prediction, jac
        return prediction

    def penalty(self, coordinates):
        theta = coordinates * self.coordinate_scale
        residual = self.model.regularization_residuals(theta[:2*self.model.mode_count])
        return .5 * float(residual @ residual)

    def state(self, coordinates, with_jacobian=False):
        predicted = self.prediction(coordinates[:-1], with_jacobian=with_jacobian)
        prediction_full = predicted[0] if with_jacobian else predicted
        prediction = prediction_full.reshape((-1, prediction_full.shape[-1]))[self.active_groups]
        extra = np.exp(coordinates[-1])
        scales = np.hypot(self.errors, extra)
        location, residual, score = profile_student_locations(self.observed-prediction, scales, self.valid)
        nll = np.sum(np.where(self.valid,
                             np.log(scales)+.5*(NU+1)*np.log1p(residual**2/(NU*scales**2)), 0.))
        penalty = self.penalty(coordinates[:-1])
        result = (float(nll+penalty), prediction_full, location, residual, score, scales)
        return result + (predicted[1],) if with_jacobian else result

    def __call__(self, coordinates):
        value, prediction_full, location, residual, score, scales, full_jacobian = self.state(coordinates, with_jacobian=True)
        gradient = np.empty(len(coordinates))
        h = 1e-5
        base_penalty = self.penalty(coordinates[:-1])
        jacobian = full_jacobian.reshape((-1, prediction_full.shape[-1], len(coordinates)-1))[self.active_groups]
        for index in range(len(coordinates)-1):
            shifted = coordinates[:-1].copy()
            shifted[index] += h
            gradient[index] = -float(np.sum(score*jacobian[:, :, index])) + (self.penalty(shifted)-base_penalty)/h
        extra = np.exp(coordinates[-1])
        gradient[-1] = np.sum(np.where(self.valid,
            (1.-(NU+1)*residual**2/(NU*scales**2+residual**2))*extra**2/scales**2, 0.))
        self.calls += 1
        if self.calls % 10 == 0:
            print(f"evaluation {self.calls}: objective={value:.6f}; extra t-scale={extra:.4f} mmag; "
                  f"max gradient={np.max(np.abs(gradient)):.3g}", flush=True)
        return value, gradient


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive, common, model = build_model()
    valid = archive["valid"]
    objective = StudentObjective(model, archive["amber_magnitude"], archive["amber_error"], valid)
    initial = np.concatenate([archive["fit_parameters"] / objective.coordinate_scale, [np.log(6.)]])
    base_value, base_gradient = objective(initial)
    # Check the envelope-theorem gradient against independent central objective
    # differences, including the fitted floor and re-estimated stellar levels.
    errors = []
    for index in (0, model.mode_count, len(initial)-2, len(initial)-1):
        h = 1e-4
        upper, lower = initial.copy(), initial.copy()
        upper[index] += h
        lower[index] -= h
        numerical = (objective.state(upper)[0]-objective.state(lower)[0])/(2*h)
        print(f"gradient coordinate {index}: analytic={base_gradient[index]:.8g}; numerical={numerical:.8g}", flush=True)
        errors.append(abs(base_gradient[index]-numerical)/(1.+abs(numerical)))
    max_error = float(max(errors))
    print(f"Gradient check maximum relative discrepancy: {max_error:.3g}", flush=True)
    if max_error > 2e-3:
        _, _, _, _, scores, _ = objective.state(initial)
        print("Maximum robust-location score:", np.max(np.abs(scores.sum(axis=1))), flush=True)
        raise RuntimeError("Student-t gradient check failed")
    parameter_bounds = [( -.3/.1, .3/.1)]*model.mode_count + [(-1./.1, 1./.1)]*model.mode_count
    parameter_bounds += [(-.25/.01, .25/.01)]*model.zero_point_count
    bounds = parameter_bounds + [(np.log(1e-4), np.log(100.))]
    started = time.monotonic()
    iteration = 0
    def checkpoint(coordinates):
        nonlocal iteration
        iteration += 1
        np.savez(OUTPUT / "optimizer_checkpoint.npz", coordinates=coordinates, iterations=iteration)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                      callback=checkpoint,
                      options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-5, "maxls": 30})
    value, prediction_mmag, location, residual, score, scales = objective.state(result.x)
    fitted_scale = float(np.exp(result.x[-1]))
    parameters = result.x[:-1] * objective.coordinate_scale
    prediction_mag = prediction_mmag / 1000.
    summary = {
        "detector": "WFI01", "photometry": "Amber", "likelihood": "Student-t", "nu": NU,
        "extra_student_t_scale_mmag": fitted_scale,
        "scale_definition": "sqrt(catalog_magnitude_error**2 + extra_scale**2); Student-t scale, not RMS",
        "gaussian_comparison_directory": str(comparison.OUTPUT),
        "used_observations": int(valid.sum()), "used_stars": len(archive["source_ids"]),
        "star_filter_brightness_groups": int(objective.active_groups.sum()),
        "star_filter_locations_reestimated_under_student_t": True,
        "objective": value, "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "objective_gradient_evaluations": objective.calls,
        "elapsed_seconds": time.monotonic()-started,
        "gradient_check_max_relative_discrepancy": max_error,
        "max_abs_profile_location_score_per_mmag": float(np.max(np.abs(score.sum(axis=1)))),
        "amplitude_bound": .3, "phase_bound_rad": 1.,
        "maxent_weight": model.entropy_weight, "phase_weight": model.phase_smoothness_weight,
        "profile_search": "Student-t location IRLS from median, Gaussian weighted mean, and best observed-value candidate; lowest cost retained",
    }
    (OUTPUT / "fit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(OUTPUT / "predictions.npz", source_ids=archive["source_ids"], valid=valid,
                        student_t_prediction_mag=prediction_mag, student_t_fit_parameters=parameters,
                        robust_locations_mmag=location, robust_residuals_mmag=residual,
                        robust_scales_mmag=scales, active_groups=objective.active_groups)
    pd.DataFrame({"mode": model.basis_names, "fit_log_amplitude": parameters[:model.mode_count],
                  "fit_phase_rad": parameters[model.mode_count:2*model.mode_count]}).to_csv(
                      OUTPUT / "student_t_modulation_parameters.csv", index=False)
    zpts = parameters[2*model.mode_count:].reshape((len(workflow.FILTERS),model.zero_point_group_count))
    pd.DataFrame({"filter": workflow.FILTERS, "fit_log_flux_offset": zpts[:,0],
                  "equivalent_magnitude_offset_mmag": -1000*zpts[:,0]/comparison.fitter.LOG_FLUX_PER_MAG}).to_csv(
                      OUTPUT / "student_t_roll_offsets.csv", index=False)
    if not result.success:
        raise RuntimeError(f"Student-t fit did not converge: {result.message}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
