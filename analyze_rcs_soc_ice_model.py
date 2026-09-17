"""Fit the SOC empirical ice-transmission model to RCS metadata.

The default fit treats the measured light change as ice-driven and compares
the SCA1 model directly with WFI01.  ``--fit-mode projected`` retains the old,
conservative all-detector check that removes a free source level for every
LED/repeat.  That projected mode tests only tiny detector-to-detector spectral
differences; it cannot measure the dominant band-common ice signal.

The direct analysis compares linear and square-root trajectories with a
maximum-entropy extension that places one smooth positive multiplier on the
square-root model at each repeat.  The results remain model-dependent: the
WFI-to-SCU correspondence and independent source calibration still need to be
established.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from assess_rcs_spectral_identifiability import (
    BAND_FILTERS,
    MAPPING_URL,
    QE_URL,
    SED_URL,
    load_mapping,
    load_qe,
    load_sed,
)
from analyze_rcs_metadata_spectral_fit import (
    METADATA_PATH,
    load_metadata,
    nuisance_basis,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    Path.home()
    / "Documents"
    / "rgespit-comissioning-ice-simulations"
    / "data"
    / "ice_optical_constants"
    / "spectral_response_model_SCA1.ecsv"
)
MODES = ("signed", "clipped", "core")
TRAJECTORIES = ("soc_sqrt", "linear")
FIT_MODES = ("direct", "projected")
ENTROPY_TRAJECTORY = "entropy_sqrt"
MODEL_THICKNESSES_NM = np.arange(0.0, 301.0, 50.0)
MODE_COLORS = {"signed": "#2f5597", "clipped": "#c55a11", "core": "#548235"}
TRAJECTORY_COLORS = {"soc_sqrt": "#7030a0", "linear": "#00a6a6"}


def read_model_table(path):
    """Read the numerical portion of the SOC ECSV without astropy."""
    if not path.exists():
        raise FileNotFoundError(f"SOC transmission model not found: {path}")

    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError(f"SOC transmission model has no data rows: {path}")

    columns = lines[0].split()
    expected = [
        "wavelength",
        "T_ratio_d0nm",
        "T_ratio_d50nm",
        "T_ratio_d100nm",
        "T_ratio_d150nm",
        "T_ratio_d200nm",
        "T_ratio_d250nm",
        "T_ratio_d300nm",
    ]
    if columns != expected:
        raise ValueError(f"Unexpected SOC model columns: {columns}")

    values = np.asarray([[float(value) for value in line.split()] for line in lines[1:]])
    wavelengths = values[:, 0]
    ratios = values[:, 1:].T
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("SOC model wavelengths must be strictly increasing")
    if not np.allclose(ratios[0], 1.0):
        raise ValueError("SOC model d=0 column is not normalized to one")

    return wavelengths, MODEL_THICKNESSES_NM.copy(), ratios


def fit_periodic_model(wavelengths, thicknesses, ratios):
    """Reproduce the continuous periodic fit used by ``ice_optics.py``."""
    coefficients = np.zeros((len(wavelengths), 3), dtype=float)
    omegas = np.zeros(len(wavelengths), dtype=float)
    sample_index = thicknesses / 50.0
    for index, wavelength in enumerate(wavelengths):
        omega = 1.0 / (wavelength / (4.0 * np.pi * 1.25) / 50.0)
        design = np.column_stack(
            (
                np.ones_like(sample_index),
                np.cos(sample_index * omega),
                np.sin(sample_index * omega),
            )
        )
        coefficients[index] = np.linalg.lstsq(
            design, ratios[:, index], rcond=None
        )[0]
        omegas[index] = omega
    return coefficients, omegas


def transmission_factor(model, wave_nm, d_nm):
    """Evaluate the SOC periodic model using a dependency-light interpolation."""
    model_wavelengths, _, coefficients, omegas = model
    wave_nm = np.asarray(wave_nm, dtype=float)
    d_array = np.atleast_1d(np.asarray(d_nm, dtype=float))
    if np.any(d_array < 0.0):
        raise ValueError("Ice thickness must be non-negative")
    if wave_nm.min() < model_wavelengths[0] or wave_nm.max() > model_wavelengths[-1]:
        raise ValueError(
            f"Requested wavelength range {wave_nm.min():.1f}-{wave_nm.max():.1f} nm "
            f"is outside the SOC model range {model_wavelengths[0]:.1f}-"
            f"{model_wavelengths[-1]:.1f} nm"
        )

    x = d_array[:, None] / 50.0
    a = coefficients[:, 0][None, :]
    b = coefficients[:, 1][None, :]
    c = coefficients[:, 2][None, :]
    fitted = (a + b * np.cos(omegas[None, :] * x) + c * np.sin(omegas[None, :] * x))
    fitted /= (a + b)

    result = np.empty((len(d_array), len(wave_nm)), dtype=float)
    for index, values in enumerate(fitted):
        # scipy is intentionally not required by this analysis workspace;
        # the SOC implementation uses cubic interpolation when scipy exists.
        result[index] = np.interp(wave_nm, model_wavelengths, values)
    result = np.maximum(result, 0.0)
    if np.ndim(d_nm) == 0:
        return result[0]
    return result


def sed_output(spectrum, mode, threshold_percent=5.0):
    raw_output = spectrum["raw_output"]
    if mode == "signed":
        return raw_output
    if mode == "clipped":
        return np.maximum(raw_output, 0.0)
    if mode == "core":
        threshold = np.max(raw_output) * threshold_percent / 100.0
        return np.where(raw_output >= threshold, raw_output, 0.0)
    raise ValueError(f"Unknown SED mode: {mode}")


def broadband_response_curves(model, spectra, qe_wavelengths, qe_values, d_grid):
    """Return normalized RCS response curves for every mode, SCU, and LED."""
    responses = {
        mode: np.zeros((18, 6, len(d_grid)), dtype=float) for mode in MODES
    }
    model_wavelengths = model[0]
    for band in range(1, 7):
        spectrum = spectra[band]
        wavelengths = spectrum["wavelength"]
        supported = (wavelengths >= model_wavelengths[0]) & (
            wavelengths <= model_wavelengths[-1]
        )
        # The empirical table has no support below 480 nm.  Do not extrapolate
        # it; assume no modeled ice change outside the supported interval.
        factors = np.ones((len(d_grid), len(wavelengths)), dtype=float)
        if np.any(supported):
            factors[:, supported] = transmission_factor(
                model, wavelengths[supported], d_grid
            )
        photon_weight = wavelengths
        qe = np.asarray(
            [
                np.interp(
                    wavelengths,
                    qe_wavelengths,
                    qe_values[:, scu],
                    left=0.0,
                    right=0.0,
                )
                for scu in range(18)
            ]
        ).T

        for mode in MODES:
            weighted_output = (
                sed_output(spectrum, mode)[:, None] * photon_weight[:, None] * qe
            )
            clean_response = np.asarray(
                    [np.trapezoid(weighted_output[:, scu], wavelengths) for scu in range(18)]
            )
            if np.any(clean_response <= 0.0):
                raise ValueError(f"Nonpositive clean response in {mode}, band {band}")

            for scu in range(18):
                numerator = np.trapezoid(
                    factors * weighted_output[:, scu][None, :],
                    wavelengths,
                    axis=1,
                )
                response = numerator / clean_response[scu]
                if np.any(response <= 0.0):
                    raise ValueError(
                        f"Nonpositive SOC response in {mode}, SCU {scu + 1}, "
                        f"band {band}"
                    )
                responses[mode][scu, band - 1] = response

    if model_wavelengths[0] > 480.0 or model_wavelengths[-1] < 2300.0:
        print(
            "Warning: SOC model does not cover the full nominal WFI range; "
            "unsupported RCS SED samples use unity ice transmission."
        )
    return responses


def soc_sqrt_thickness(elapsed_days, reference_thickness_nm, reference_time_days, reset_day):
    absolute_days = reset_day + np.asarray(elapsed_days, dtype=float)
    if np.any(absolute_days < 0.0):
        raise ValueError("SOC square-root trajectory requires non-negative absolute time")
    return reference_thickness_nm * (
        np.sqrt(absolute_days / reference_time_days)
        - np.sqrt(reset_day / reference_time_days)
    )


def trajectory_thickness(
    elapsed_days,
    trajectory,
    parameter,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
):
    if trajectory == "soc_sqrt":
        return parameter * soc_sqrt_thickness(
            elapsed_days,
            reference_thickness_nm,
            reference_time_days,
            reset_day,
        )
    if trajectory == "linear":
        return parameter * np.asarray(elapsed_days, dtype=float)
    raise ValueError(f"Unknown trajectory: {trajectory}")


def make_nuisance_projector(nuisance):
    """Build the projection that removes the free source and detector terms."""
    u, singular_values, _ = np.linalg.svd(nuisance, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((nuisance.shape[0], 0))
    tolerance = max(nuisance.shape) * singular_values[0] * np.finfo(float).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    return u[:, :rank]


def project_out(values, projection_basis):
    if projection_basis.shape[1] == 0:
        return np.asarray(values)
    values = np.asarray(values)
    return values - projection_basis @ (projection_basis.T @ values)


def prepare_fit_observations(observations, fit_mode, detector):
    """Select the fit scope and build only the requested nuisance projection."""
    if fit_mode == "direct":
        selected = [row for row in observations if row["detector"] == detector]
        if not selected:
            raise ValueError(f"No observations found for WFI{detector:02d}")
        return selected, np.zeros((len(selected), 0), dtype=float)
    if fit_mode == "projected":
        nuisance = nuisance_basis(observations)
        return observations, make_nuisance_projector(nuisance)
    raise ValueError(f"Unknown fit mode: {fit_mode}")


def model_prediction(
    observations,
    response_curves,
    d_grid,
    trajectory,
    parameter,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
):
    days = np.asarray([row["days"] for row in observations])
    thickness = trajectory_thickness(
        days,
        trajectory,
        parameter,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    prediction = response_prediction(
        observations, response_curves, d_grid, thickness
    )
    return prediction, thickness


def response_prediction(observations, response_curves, d_grid, thickness):
    """Evaluate relative broadband responses for supplied absolute thicknesses."""
    thickness = np.asarray(thickness, dtype=float)
    if thickness.shape != (len(observations),):
        raise ValueError("Thickness must contain one value per observation")
    if not np.all(np.isfinite(thickness)):
        raise ValueError("Thickness values must be finite")
    tolerance = 8.0 * np.finfo(float).eps * max(
        1.0, abs(float(d_grid[0])), abs(float(d_grid[-1]))
    )
    if np.any(thickness < d_grid[0] - tolerance) or np.any(
        thickness > d_grid[-1] + tolerance
    ):
        raise ValueError(
            f"Thickness range {thickness.min():.6g}-{thickness.max():.6g} nm "
            f"is outside the response grid {d_grid[0]:.6g}-{d_grid[-1]:.6g} nm"
        )
    thickness = np.clip(thickness, d_grid[0], d_grid[-1])

    first_indices = {
        led: min(
            (index for index, row in enumerate(observations) if row["led"] == led),
            key=lambda index: observations[index]["days"],
        )
        for led in sorted({row["led"] for row in observations})
    }

    prediction = np.zeros(len(observations), dtype=float)
    for index, row in enumerate(observations):
        curve = response_curves[row["detector"] - 1, row["led"] - 1]
        current = np.interp(thickness[index], d_grid, curve)
        baseline = np.interp(
            thickness[first_indices[row["led"]]], d_grid, curve
        )
        prediction[index] = np.log(current / baseline)
    return prediction


def mean_endpoint_rate(observations, thickness):
    days = np.asarray([row["days"] for row in observations], dtype=float)
    span = float(np.max(days) - np.min(days))
    if span <= 0.0:
        return 0.0
    first = int(np.argmin(days))
    last = int(np.argmax(days))
    return float((thickness[last] - thickness[first]) / span)


def generalized_relative_entropy(values):
    """Return mean generalized KL divergence from positive values to unity."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Entropy node multipliers must be finite and positive")
    return float(np.mean(values * np.log(values) - values + 1.0))


def mean_squared_log_node_curvature(log_nodes, node_days):
    """Measure log-node curvature while accounting for irregular repeat times."""
    log_nodes = np.asarray(log_nodes, dtype=float)
    node_days = np.asarray(node_days, dtype=float)
    intervals = np.diff(node_days)
    if len(log_nodes) != len(node_days) or len(log_nodes) < 3:
        raise ValueError("Node curvature requires matching arrays of length three or more")
    if not np.all(np.isfinite(log_nodes)) or np.any(intervals <= 0.0):
        raise ValueError("Node values must be finite and times strictly increasing")

    scaled_time = (node_days - node_days[0]) / np.median(intervals)
    slopes = np.diff(log_nodes) / np.diff(scaled_time)
    curvature = 2.0 * np.diff(slopes) / (scaled_time[2:] - scaled_time[:-2])
    return float(np.mean(curvature**2))


def bounded_pattern_search(
    objective,
    initial,
    lower,
    upper,
    initial_step=0.15,
    tolerance=1.0e-4,
    max_sweeps=300,
):
    """Dependency-light coordinate pattern search for the small node problem."""
    parameters = np.clip(
        np.asarray(initial, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    steps = np.full_like(parameters, float(initial_step))
    score = float(objective(parameters))
    if not np.isfinite(score):
        raise ValueError("Pattern-search initial objective must be finite")
    evaluations = 1

    for sweep in range(1, max_sweeps + 1):
        sweep_start_score = score
        improved = False
        for index in range(len(parameters)):
            best_score = score
            best_candidate = None
            for direction in (-1.0, 1.0):
                candidate = parameters.copy()
                candidate[index] = np.clip(
                    candidate[index] + direction * steps[index],
                    lower[index],
                    upper[index],
                )
                candidate_score = float(objective(candidate))
                evaluations += 1
                if not np.isfinite(candidate_score):
                    continue
                if candidate_score < best_score - 1.0e-12:
                    best_score = candidate_score
                    best_candidate = candidate
            if best_candidate is not None:
                parameters = best_candidate
                score = best_score
                improved = True

        progress = sweep_start_score - score
        if not improved or progress <= 1.0e-10 * max(1.0, abs(sweep_start_score)):
            steps *= 0.5
        if float(np.max(steps)) < tolerance:
            return parameters, score, sweep, evaluations, True

    return parameters, score, max_sweeps, evaluations, False


def fit_entropy_sqrt_trajectory(
    observations,
    y_fit,
    projection_basis,
    response_curves,
    d_grid,
    initial_scale,
    max_scale,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
    entropy_weight,
    smoothness_weight,
    max_node_multiplier,
):
    """Fit smooth positive node multipliers on the square-root trajectory."""
    repeats = sorted({row["repeat"] for row in observations})
    if len(repeats) < 3:
        raise ValueError("The entropy trajectory requires at least three repeats")
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeats)}
    observation_nodes = np.asarray(
        [repeat_to_index[row["repeat"]] for row in observations], dtype=int
    )
    days = np.asarray([row["days"] for row in observations], dtype=float)
    base_thickness = soc_sqrt_thickness(
        days,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    node_days = np.asarray(
        [
            np.mean(
                [row["days"] for row in observations if row["repeat"] == repeat]
            )
            for repeat in repeats
        ],
        dtype=float,
    )
    base_node_thickness = soc_sqrt_thickness(
        node_days,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    if np.any(np.diff(node_days) <= 0.0):
        raise ValueError("Entropy-node repeat times must be strictly increasing")
    null_mse = max(float(np.mean(np.asarray(y_fit) ** 2)), 1.0e-12)

    def evaluate_state(scale, log_nodes, details=False):
        node_multipliers = np.exp(log_nodes)
        node_thickness = scale * base_node_thickness * node_multipliers
        if np.any(np.diff(node_thickness) < -1.0e-12):
            return np.inf
        thickness = scale * base_thickness * node_multipliers[observation_nodes]
        prediction_raw = response_prediction(
            observations, response_curves, d_grid, thickness
        )
        prediction_fit = project_out(prediction_raw, projection_basis)
        residual = np.asarray(y_fit) - prediction_fit
        data_term = float(np.mean(residual**2) / null_mse)
        entropy = generalized_relative_entropy(node_multipliers)
        smoothness = mean_squared_log_node_curvature(log_nodes, node_days)
        objective = (
            data_term
            + entropy_weight * entropy
            + smoothness_weight * smoothness
        )
        if not np.isfinite(objective):
            return np.inf
        if details:
            return {
                "scale": scale,
                "log_nodes": log_nodes,
                "node_multipliers": node_multipliers,
                "thickness_nm": thickness,
                "prediction_raw": prediction_raw,
                "prediction_fit": prediction_fit,
                "residual": residual,
                "data_term": data_term,
                "entropy_penalty": entropy,
                "smoothness_penalty": smoothness,
                "objective": objective,
            }
        return objective

    def evaluate(parameters, details=False):
        log_nodes = np.zeros(len(repeats), dtype=float)
        log_nodes[1:] = parameters[1:]
        return evaluate_state(float(np.exp(parameters[0])), log_nodes, details)

    log_node_bound = float(np.log(max_node_multiplier))
    initial = np.concatenate(
        ([np.log(max(float(initial_scale), 1.0e-6))], np.zeros(len(repeats) - 1))
    )
    lower = np.concatenate(
        ([np.log(1.0e-6)], np.full(len(repeats) - 1, -log_node_bound))
    )
    upper = np.concatenate(
        ([np.log(max_scale)], np.full(len(repeats) - 1, log_node_bound))
    )
    parameters, objective, sweeps, evaluations, converged = bounded_pattern_search(
        evaluate, initial, lower, upper
    )
    if not converged:
        raise RuntimeError(
            f"Entropy-node optimizer did not converge after {sweeps} sweeps"
        )
    details = evaluate(parameters, details=True)
    zero_details = evaluate_state(0.0, np.zeros(len(repeats)), details=True)
    if zero_details["objective"] < objective:
        details = zero_details
        objective = zero_details["objective"]

    node_prior_thickness = details["scale"] * base_node_thickness
    node_thickness = node_prior_thickness * details["node_multipliers"]
    raw_signal_rms = float(np.sqrt(np.mean(details["prediction_raw"] ** 2)))
    fit_signal_rms = float(np.sqrt(np.mean(details["prediction_fit"] ** 2)))
    retained_fraction = (
        fit_signal_rms / raw_signal_rms if raw_signal_rms > 0.0 else 1.0
    )
    return {
        "trajectory": ENTROPY_TRAJECTORY,
        "parameter": details["scale"],
        "model_rms": float(np.sqrt(np.mean(details["residual"] ** 2))),
        "prediction_raw": details["prediction_raw"],
        "prediction_residualized": details["prediction_fit"],
        "raw_signal_rms": raw_signal_rms,
        "fit_signal_rms": fit_signal_rms,
        "signal_retained_fraction": retained_fraction,
        "thickness_nm": details["thickness_nm"],
        "mean_rate_nm_day": mean_endpoint_rate(
            observations, details["thickness_nm"]
        ),
        "node_repeats": np.asarray(repeats, dtype=int),
        "node_days": node_days,
        "node_prior_thickness_nm": node_prior_thickness,
        "node_multipliers": details["node_multipliers"],
        "node_thickness_nm": node_thickness,
        "entropy_penalty": details["entropy_penalty"],
        "smoothness_penalty": details["smoothness_penalty"],
        "penalized_objective": objective,
        "normalized_data_term": details["data_term"],
        "entropy_weight": entropy_weight,
        "smoothness_weight": smoothness_weight,
        "optimizer_sweeps": sweeps,
        "optimizer_evaluations": evaluations,
        "optimizer_converged": converged,
    }


def fit_trajectory(
    observations,
    y_residual,
    projection_basis,
    response_curves,
    d_grid,
    trajectory,
    parameter_grid,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
):
    scores = []
    raw_predictions = []
    predictions = []
    thicknesses = []
    for parameter in parameter_grid:
        prediction, thickness = model_prediction(
            observations,
            response_curves,
            d_grid,
            trajectory,
            parameter,
            reference_thickness_nm,
            reference_time_days,
            reset_day,
        )
        prediction_residual = project_out(prediction, projection_basis)
        residual = y_residual - prediction_residual
        scores.append(float(np.sqrt(np.mean(residual**2))))
        raw_predictions.append(prediction)
        predictions.append(prediction_residual)
        thicknesses.append(thickness)

    best_index = int(np.argmin(scores))
    raw_signal_rms = float(np.sqrt(np.mean(raw_predictions[best_index] ** 2)))
    fit_signal_rms = float(np.sqrt(np.mean(predictions[best_index] ** 2)))
    retained_fraction = (
        fit_signal_rms / raw_signal_rms if raw_signal_rms > 0.0 else 1.0
    )
    return {
        "trajectory": trajectory,
        "parameter": float(parameter_grid[best_index]),
        "parameter_grid": parameter_grid,
        "scores": np.asarray(scores),
        "model_rms": scores[best_index],
        "prediction_raw": raw_predictions[best_index],
        "prediction_residualized": predictions[best_index],
        "raw_signal_rms": raw_signal_rms,
        "fit_signal_rms": fit_signal_rms,
        "signal_retained_fraction": retained_fraction,
        "thickness_nm": thicknesses[best_index],
        "mean_rate_nm_day": mean_endpoint_rate(
            observations, thicknesses[best_index]
        ),
        "entropy_penalty": 0.0,
        "smoothness_penalty": 0.0,
        "penalized_objective": (
            scores[best_index] ** 2
            / max(float(np.mean(np.asarray(y_residual) ** 2)), 1.0e-12)
        ),
    }


def make_fit_row(fit_mode, fit_detector, mode, result, null_rms):
    units = "nm/day" if result["trajectory"] == "linear" else "scale"
    n_parameters = (
        len(result["node_multipliers"])
        if "node_multipliers" in result
        else 1
    )
    return {
        "fit_mode": fit_mode,
        "fit_detector": fit_detector,
        "sed_mode": mode,
        "trajectory": result["trajectory"],
        "parameter": f"{result['parameter']:.12g}",
        "parameter_units": units,
        "mean_rate_nm_day": f"{result['mean_rate_nm_day']:.12g}",
        "max_thickness_nm": f"{np.max(result['thickness_nm']):.12g}",
        "null_fit_rms_log": f"{null_rms:.12g}",
        "model_fit_rms_log": f"{result['model_rms']:.12g}",
        "rms_improvement_fraction": f"{1.0 - result['model_rms'] / null_rms:.12g}",
        "model_signal_rms_log": f"{result['fit_signal_rms']:.12g}",
        "model_signal_retained_fraction": (
            f"{result['signal_retained_fraction']:.12g}"
        ),
        "penalized_objective": f"{result['penalized_objective']:.12g}",
        "entropy_relative_to_unity": f"{result['entropy_penalty']:.12g}",
        "log_node_curvature": f"{result['smoothness_penalty']:.12g}",
        "entropy_weight": result.get("entropy_weight", ""),
        "smoothness_weight": result.get("smoothness_weight", ""),
        "optimizer_converged": result.get("optimizer_converged", ""),
        "optimizer_evaluations": result.get("optimizer_evaluations", ""),
        "n_trajectory_parameters": n_parameters,
    }


def write_fit_csv(path, fit_rows):
    fields = [
        "fit_mode",
        "fit_detector",
        "sed_mode",
        "trajectory",
        "parameter",
        "parameter_units",
        "mean_rate_nm_day",
        "max_thickness_nm",
        "null_fit_rms_log",
        "model_fit_rms_log",
        "rms_improvement_fraction",
        "model_signal_rms_log",
        "model_signal_retained_fraction",
        "penalized_objective",
        "entropy_relative_to_unity",
        "log_node_curvature",
        "entropy_weight",
        "smoothness_weight",
        "optimizer_converged",
        "optimizer_evaluations",
        "n_trajectory_parameters",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fit_rows)


def write_response_csv(path, responses, d_grid, mapping):
    fields = [
        "sed_mode",
        "scu",
        "sca",
        "led",
        "filter",
        "thickness_nm",
        "normalized_broadband_response",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            for scu in range(1, 19):
                for band in range(1, 7):
                    curve = responses[mode][scu - 1, band - 1]
                    values = np.interp(MODEL_THICKNESSES_NM, d_grid, curve)
                    for thickness, value in zip(MODEL_THICKNESSES_NM, values):
                        writer.writerow(
                            {
                                "sed_mode": mode,
                                "scu": scu,
                                "sca": mapping[scu],
                                "led": band,
                                "filter": BAND_FILTERS[band],
                                "thickness_nm": f"{thickness:.6g}",
                                "normalized_broadband_response": f"{value:.12g}",
                            }
                        )


def write_prediction_csv(
    path, observations, fit_results, y_residual, fit_mode, fit_detector
):
    fields = [
        "fit_mode",
        "fit_detector",
        "sed_mode",
        "trajectory",
        "parameter",
        "mean_rate_nm_day",
        "group",
        "repeat",
        "days",
        "detector",
        "led",
        "thickness_nm",
        "observed_fit_log_change",
        "model_fit_log_change",
        "remaining_residual_log_change",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            result = fit_results[mode]
            remaining = y_residual - result["prediction_residualized"]
            for index, row in enumerate(observations):
                writer.writerow(
                    {
                        "fit_mode": fit_mode,
                        "fit_detector": fit_detector,
                        "sed_mode": mode,
                        "trajectory": result["trajectory"],
                        "parameter": f"{result['parameter']:.12g}",
                        "mean_rate_nm_day": f"{result['mean_rate_nm_day']:.12g}",
                        "group": row["group"],
                        "repeat": row["repeat"],
                        "days": f"{row['days']:.12g}",
                        "detector": row["detector"],
                        "led": row["led"],
                        "thickness_nm": f"{result['thickness_nm'][index]:.12g}",
                        "observed_fit_log_change": f"{y_residual[index]:.12g}",
                        "model_fit_log_change": f"{result['prediction_residualized'][index]:.12g}",
                        "remaining_residual_log_change": f"{remaining[index]:.12g}",
                    }
                )


def write_entropy_nodes_csv(path, fit_results):
    fields = [
        "sed_mode",
        "repeat",
        "days",
        "hours",
        "sqrt_prior_thickness_nm",
        "node_multiplier",
        "fitted_thickness_nm",
        "interval_rate_nm_day",
        "relative_entropy_contribution",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            result = fit_results[mode]
            if "node_multipliers" not in result:
                continue
            previous_day = None
            previous_thickness = None
            for repeat, days, prior, multiplier, thickness in zip(
                result["node_repeats"],
                result["node_days"],
                result["node_prior_thickness_nm"],
                result["node_multipliers"],
                result["node_thickness_nm"],
            ):
                if previous_day is None or days == previous_day:
                    interval_rate = ""
                else:
                    interval_rate = (
                        f"{(thickness - previous_thickness) / (days - previous_day):.12g}"
                    )
                writer.writerow(
                    {
                        "sed_mode": mode,
                        "repeat": int(repeat),
                        "days": f"{days:.12g}",
                        "hours": f"{24.0 * days:.12g}",
                        "sqrt_prior_thickness_nm": f"{prior:.12g}",
                        "node_multiplier": f"{multiplier:.12g}",
                        "fitted_thickness_nm": f"{thickness:.12g}",
                        "interval_rate_nm_day": interval_rate,
                        "relative_entropy_contribution": (
                            f"{multiplier * np.log(multiplier) - multiplier + 1.0:.12g}"
                        ),
                    }
                )
                previous_day = days
                previous_thickness = thickness


def write_report(
    path,
    model_path,
    metadata_path,
    all_fits,
    fits_by_mode,
    null_rms,
    mapping,
    global_start,
    interpolation,
    fit_mode,
    fit_detector,
    n_observations,
):
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if fit_mode == "direct":
        scope = fit_detector
        nuisance_description = (
            "none; the per-detector/per-LED first exposure already defines zero "
            "relative change"
        )
        source_description = (
            "The fit assumes the measured relative-light evolution is ice-driven; "
            "it does not fit a free source level at every repeat."
        )
        model_application = (
            f"SCA1 empirical transmission model, applied only to {fit_detector} "
            "under the WFI-to-SCU ordering assumption."
        )
    else:
        scope = "WFI01-WFI18"
        nuisance_description = (
            "free LED/repeat source term and detector offset, projected out before scoring"
        )
        source_description = (
            "This conservative mode discards band-common changes and tests only "
            "detector-by-band interactions."
        )
        model_application = (
            "SCA1 empirical transmission model, applied to all SCUs as an explicit approximation."
        )
    entropy_description = []
    if fit_mode == "direct":
        entropy_description = [
            "- Entropy trajectory: `d_i = scale * m_i * d_sqrt(t_i)`, with one node shared by all six LEDs at each repeat.",
            "- Node constraints: `m_i > 0`, first node fixed to one, and non-decreasing fitted thickness.",
            "- Node regularization: generalized relative entropy `mean(m log(m) - m + 1)` plus time-aware curvature of `log(m)`.",
        ]
    lines = [
        "# RCS Fit Using the SOC Empirical Ice Model",
        "",
        source_description,
        "",
        f"- Metadata: `{metadata_path}`",
        f"- Sequence start: `{global_start.isoformat()}`",
        f"- Fit mode: `{fit_mode}`",
        f"- Fit scope: `{scope}` ({n_observations} observations)",
        f"- SOC model: `{model_path}`",
        f"- SOC model SHA-256: `{model_hash}`",
        f"- SOC model interpolation: `{interpolation}`",
        f"- SOC file: {model_application}",
        "- Detector mapping assumption: `WFI01` through `WFI18` correspond to SCU 1 through 18.",
        f"- Public SCU-to-SCA mapping contains {len(mapping)} entries, but does not verify that WFI ordering assumption.",
        "- Archive metadata selects `optical_element=DARK`; the use of its LED labels as RCS spectral bands must be verified.",
        "- RCS SED samples below 480 nm use unity ice transmission because the SOC table has no support there.",
        "- Response integral: published relative LED output × wavelength photon weighting × public SCU QE × SOC ice factor.",
        f"- Nuisance treatment: {nuisance_description}.",
        "- SED modes: `signed`, `clipped`, and 5%-`core`.",
        *entropy_description,
        "",
        "## Trajectory Fits",
        "",
        "| SED mode | Trajectory | Parameter | Units | Mean rate | Max thickness | Model RMS | RMS improvement | Signal retained | Selected |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode in MODES:
        for fit in all_fits[mode].values():
            improvement = 1.0 - fit["model_rms"] / null_rms
            units = "nm/day" if fit["trajectory"] == "linear" else "scale"
            selected = "yes" if fit is fits_by_mode[mode] else ""
            lines.append(
                f"| {mode} | {fit['trajectory']} | {fit['parameter']:.6g} | "
                f"{units} | {fit['mean_rate_nm_day']:.6g} nm/day | "
                f"{np.max(fit['thickness_nm']):.6g} nm | {fit['model_rms']:.6g} | "
                f"{improvement:.2%} | {fit['signal_retained_fraction']:.3%} | "
                f"{selected} |"
            )
    lines.extend(["", "## Interpretation", ""])
    if fit_mode == "direct":
        linear_rates = [
            all_fits[mode]["linear"]["parameter"] for mode in MODES
        ]
        entropy_fits = [all_fits[mode][ENTROPY_TRAJECTORY] for mode in MODES]
        entropy_rates = [fit["mean_rate_nm_day"] for fit in entropy_fits]
        node_min = min(float(np.min(fit["node_multipliers"])) for fit in entropy_fits)
        node_max = max(float(np.max(fit["node_multipliers"])) for fit in entropy_fits)
        entropy_weight = entropy_fits[0]["entropy_weight"]
        smoothness_weight = entropy_fits[0]["smoothness_weight"]
        lines.extend(
            [
                f"The linear rates span {min(linear_rates):.3g}-"
                f"{max(linear_rates):.3g} nm/day. The entropy-regularized "
                f"trajectories span {min(entropy_rates):.3g}-"
                f"{max(entropy_rates):.3g} nm/day in endpoint-average rate.",
                "",
                f"The fitted node multipliers span {node_min:.6g}-{node_max:.6g}.",
                f"The normalized objective uses entropy weight {entropy_weight:.6g} "
                f"and time-aware log-node curvature weight {smoothness_weight:.6g}.",
                "",
            ]
        )
        lines.extend(
            [
                "The direct model is evaluated relative to the first observation of",
                "each LED, matching the metadata normalization without removing the",
                "band-common ice response. This is the appropriate fit space when the",
                "light evolution is independently known to be ice-driven. Remaining",
                "residuals measure limitations of the single-SCA optical model, source",
                "stability, and the provisional detector/LED mapping rather than an",
                "absence of ice.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The free LED/repeat terms absorb almost all of a common ice response.",
                "Consequently this projected result is an interaction diagnostic, not",
                "an ice-rate fit; a small improvement cannot be interpreted as evidence",
                "against independently established ice accumulation.",
            ]
        )
    lines.extend(
        [
            "The empirical file is only tabulated through 300 nm, so larger",
            "thicknesses rely on model extrapolation and should not be treated as",
            "measured support.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_results(
    output_dir,
    responses,
    d_grid,
    fit_results,
    observations,
    y_residual,
    null_rms,
    fit_mode,
    fit_detector,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    response_path = output_dir / "rcs_soc_ice_response_curves.png"
    fit_path = output_dir / "rcs_soc_ice_model_fits.png"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for column, band in enumerate(range(1, 7)):
        axis = axes.ravel()[column]
        for mode in MODES:
            curve = np.median(responses[mode][:, band - 1], axis=0)
            axis.plot(
                d_grid,
                curve,
                color=MODE_COLORS[mode],
                linewidth=1.8,
                label=mode,
            )
            axis.plot(
                d_grid,
                responses[mode][:, band - 1].T,
                color=MODE_COLORS[mode],
                alpha=0.08,
                linewidth=0.5,
            )
        axis.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
        axis.set_title(f"LED {band} / {BAND_FILTERS[band]}")
        axis.set_xlabel("Ice thickness (nm)")
        axis.set_ylabel("RCS response / clean response")
        axis.grid(color="0.9", linewidth=0.5)
        if column == 0:
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "SOC SCA1 ice model folded through RCS LED SEDs and SCU QE\n"
        "Faint lines are individual SCUs; thick lines are SCU medians",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(response_path, dpi=180)
    plt.close(fig)

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 9),
        sharey="row",
        gridspec_kw={"height_ratios": (1.1, 1.0)},
    )
    grouped = {}
    for index, row in enumerate(observations):
        grouped.setdefault((row["led"], row["group"]), []).append(index)
    led_indices = {
        led: np.asarray(
            [index for index, row in enumerate(observations) if row["led"] == led]
        )
        for led in range(1, 7)
    }
    fit_space_label = "direct" if fit_mode == "direct" else "nuisance-projected"
    for column, mode in enumerate(MODES):
        result = fit_results[mode]
        observed_all = y_residual
        predicted_all = result["prediction_residualized"]
        scatter_axis = axes[0, column]
        limit = max(
            float(np.max(np.abs(observed_all))),
            float(np.max(np.abs(predicted_all))),
        ) * 1.1
        if limit == 0.0:
            limit = 1.0
        for led in range(1, 7):
            indices = led_indices[led]
            scatter_axis.scatter(
                predicted_all[indices],
                observed_all[indices],
                s=7,
                alpha=0.28,
                color=f"C{led - 1}",
                edgecolors="none",
            )
        scatter_axis.plot(
            [-limit, limit],
            [-limit, limit],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        scatter_axis.axvline(0.0, color="0.65", linewidth=0.6)
        scatter_axis.axhline(0.0, color="0.65", linewidth=0.6)
        r_squared = 1.0 - np.sum((observed_all - predicted_all) ** 2) / np.sum(
            observed_all**2
        )
        improvement = 1.0 - result["model_rms"] / null_rms
        units = "nm/day" if result["trajectory"] == "linear" else "scale"
        if result["trajectory"] == ENTROPY_TRAJECTORY:
            parameter_title = (
                f"{mode}: entropy sqrt; mean rate = "
                f"{result['mean_rate_nm_day']:.4g} nm/day"
            )
        else:
            parameter_title = (
                f"{mode}: {result['trajectory']} = "
                f"{result['parameter']:.4g} {units}"
            )
        scatter_axis.set_xlim(-limit, limit)
        scatter_axis.set_ylim(-limit, limit)
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.set_title(
            f"{parameter_title}\n"
            f"RMS improvement {improvement:.2%}; $R^2$ = {r_squared:.3f}\n"
            f"max |model| = {np.max(np.abs(predicted_all)):.2g}"
        )
        scatter_axis.set_xlabel(f"SOC model {fit_space_label} log change")
        if column == 0:
            scatter_axis.set_ylabel(f"Observed {fit_space_label} log change")
        scatter_axis.grid(color="0.9", linewidth=0.5)

        axis = axes[1, column]
        for led in range(1, 7):
            points = [
                (group, indices)
                for (group_led, group), indices in grouped.items()
                if group_led == led
            ]
            points.sort()
            x = [
                24.0 * np.mean([observations[index]["days"] for index in indices])
                for _, indices in points
            ]
            observed = [np.median(y_residual[indices]) for _, indices in points]
            predicted = [
                np.median(result["prediction_residualized"][indices])
                for _, indices in points
            ]
            color = f"C{led - 1}"
            axis.plot(
                x,
                observed,
                marker="o",
                markersize=3,
                linestyle="none",
                alpha=0.75,
                color=color,
            )
            axis.plot(
                x,
                predicted,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                color=color,
            )
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_title("LED/repeat values in the fitted data space")
        axis.set_xlabel("Elapsed time (hours)")
        if column == 0:
            axis.set_ylabel(f"{fit_space_label.title()} log change")
        axis.grid(color="0.9", linewidth=0.5)

    led_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=f"C{led - 1}",
            markersize=5,
            label=f"LED {led}",
        )
        for led in range(1, 7)
    ]
    style_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="black", label="observed"),
        Line2D([0], [0], linestyle="--", color="black", label="SOC model"),
    ]
    axes[1, 0].legend(
        handles=led_handles + style_handles,
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper left",
    )
    scope = fit_detector if fit_mode == "direct" else "WFI01-WFI18"
    fig.suptitle(
        f"SOC model versus {scope} RCS metadata ({len(observations):,} medians)\n"
        f"Top: every fitted point; bottom: LED/repeat values. "
        f"Both rows use the {fit_space_label} fit space.",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(fit_path, dpi=180)
    plt.close(fig)

    output_paths = [response_path, fit_path]
    entropy_results = {
        mode: result
        for mode, result in fit_results.items()
        if "node_multipliers" in result
    }
    if entropy_results:
        node_path = output_dir / "rcs_soc_ice_entropy_nodes.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for mode, result in entropy_results.items():
            hours = 24.0 * result["node_days"]
            color = MODE_COLORS[mode]
            axes[0].plot(
                hours,
                result["node_multipliers"],
                marker="o",
                color=color,
                label=mode,
            )
            axes[1].plot(
                hours,
                result["node_prior_thickness_nm"],
                linestyle="--",
                color=color,
                alpha=0.45,
            )
            axes[1].plot(
                hours,
                result["node_thickness_nm"],
                marker="o",
                color=color,
                label=mode,
            )
        first_result = next(iter(entropy_results.values()))
        axes[0].axhline(1.0, color="black", linewidth=0.7, linestyle="--")
        axes[0].set_ylabel("Multiplicative node")
        axes[0].set_title(
            "Entropy-regularized modulation of the square-root trajectory\n"
            f"KL weight={first_result['entropy_weight']:.4g}; "
            f"time-curvature weight={first_result['smoothness_weight']:.4g}"
        )
        axes[0].legend(frameon=False)
        axes[1].set_xlabel("Elapsed time (hours)")
        axes[1].set_ylabel("Ice thickness (nm)")
        axes[1].set_title(
            "Fitted thickness (solid) and scaled square-root prior (dashed)"
        )
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.grid(color="0.9", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(node_path, dpi=180)
        plt.close(fig)
        output_paths.append(node_path)

    return tuple(output_paths)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--fit-mode",
        choices=FIT_MODES,
        default="direct",
        help=(
            "direct fits one detector without source projection; projected "
            "reproduces the conservative all-detector interaction test"
        ),
    )
    parser.add_argument(
        "--detector",
        type=int,
        choices=range(1, 19),
        default=1,
        help="WFI detector number for --fit-mode direct (default: 1)",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.01,
        help="generalized-KL weight in the normalized node objective",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.1,
        help="time-aware curvature weight on log node multipliers",
    )
    parser.add_argument(
        "--max-node-multiplier",
        type=float,
        default=2.0,
        help="symmetric node bound: 1/value through value",
    )
    parser.add_argument("--max-rate", type=float, default=100.0)
    parser.add_argument("--rate-steps", type=int, default=501)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--scale-steps", type=int, default=401)
    parser.add_argument("--thickness-grid-steps", type=int, default=1201)
    parser.add_argument("--sqrt-ref-thickness-nm", type=float, default=122.7)
    parser.add_argument("--sqrt-ref-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.thickness_grid_steps < 2:
        raise ValueError("--thickness-grid-steps must be at least two")
    if args.rate_steps < 2 or args.scale_steps < 2:
        raise ValueError("Parameter grids must contain at least two points")
    if args.max_rate <= 0.0 or args.max_scale <= 0.0:
        raise ValueError("Parameter-grid maxima must be positive")
    if args.entropy_weight < 0.0 or args.smoothness_weight < 0.0:
        raise ValueError("Entropy and smoothness weights must be non-negative")
    if args.max_node_multiplier <= 1.0:
        raise ValueError("--max-node-multiplier must be greater than one")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_wavelengths, thicknesses, ratios = read_model_table(args.model_path)
    coefficients, omegas = fit_periodic_model(model_wavelengths, thicknesses, ratios)
    model = (model_wavelengths, thicknesses, coefficients, omegas)

    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    mapping = load_mapping(MAPPING_URL)
    all_observations, detectors, global_start = load_metadata(args.metadata)
    observations, projection_basis = prepare_fit_observations(
        all_observations, args.fit_mode, args.detector
    )
    y = np.asarray([row["log_change"] for row in observations])
    y_residual = project_out(y, projection_basis)
    null_rms = float(np.sqrt(np.mean(y_residual**2)))

    elapsed_max = max(row["days"] for row in observations)
    soc_max = float(
        args.max_scale
        * soc_sqrt_thickness(
            elapsed_max,
            args.sqrt_ref_thickness_nm,
            args.sqrt_ref_time_days,
            args.reset_day,
        )
    )
    linear_max = args.max_rate * elapsed_max
    entropy_max = (
        soc_max * args.max_node_multiplier
        if args.fit_mode == "direct"
        else soc_max
    )
    d_max = max(300.0, entropy_max, linear_max)
    d_grid = np.linspace(0.0, d_max, args.thickness_grid_steps)
    responses = broadband_response_curves(
        model,
        spectra,
        qe_wavelengths,
        qe_values,
        d_grid,
    )

    parameter_grids = {
        "soc_sqrt": np.linspace(0.0, args.max_scale, args.scale_steps),
        "linear": np.linspace(0.0, args.max_rate, args.rate_steps),
    }
    all_fits = {}
    best_fits = {}
    fit_rows = []
    fit_detector = (
        f"WFI{args.detector:02d}" if args.fit_mode == "direct" else "all"
    )
    for mode in MODES:
        all_fits[mode] = {}
        for trajectory in TRAJECTORIES:
            result = fit_trajectory(
                observations,
                y_residual,
                projection_basis,
                responses[mode],
                d_grid,
                trajectory,
                parameter_grids[trajectory],
                args.sqrt_ref_thickness_nm,
                args.sqrt_ref_time_days,
                args.reset_day,
            )
            all_fits[mode][trajectory] = result
            fit_rows.append(
                make_fit_row(
                    args.fit_mode, fit_detector, mode, result, null_rms
                )
            )
        if args.fit_mode == "direct":
            entropy_result = fit_entropy_sqrt_trajectory(
                observations,
                y_residual,
                projection_basis,
                responses[mode],
                d_grid,
                all_fits[mode]["soc_sqrt"]["parameter"],
                args.max_scale,
                args.sqrt_ref_thickness_nm,
                args.sqrt_ref_time_days,
                args.reset_day,
                args.entropy_weight,
                args.smoothness_weight,
                args.max_node_multiplier,
            )
            all_fits[mode][ENTROPY_TRAJECTORY] = entropy_result
            fit_rows.append(
                make_fit_row(
                    args.fit_mode,
                    fit_detector,
                    mode,
                    entropy_result,
                    null_rms,
                )
            )
        best_fits[mode] = min(
            all_fits[mode].values(), key=lambda result: result["model_rms"]
        )

    fit_path = args.output_dir / "rcs_soc_ice_model_fit.csv"
    response_path = args.output_dir / "rcs_soc_ice_response_summary.csv"
    prediction_path = args.output_dir / "rcs_soc_ice_model_predictions.csv"
    node_path = args.output_dir / "rcs_soc_ice_entropy_nodes.csv"
    report_path = args.output_dir / "rcs_soc_ice_model_fit.md"
    write_fit_csv(fit_path, fit_rows)
    write_response_csv(response_path, responses, d_grid, mapping)
    if args.fit_mode == "direct":
        write_entropy_nodes_csv(
            node_path,
            {
                mode: all_fits[mode][ENTROPY_TRAJECTORY]
                for mode in MODES
            },
        )
    write_prediction_csv(
        prediction_path,
        observations,
        best_fits,
        y_residual,
        args.fit_mode,
        fit_detector,
    )
    write_report(
        report_path,
        args.model_path,
        args.metadata,
        all_fits,
        best_fits,
        null_rms,
        mapping,
        global_start,
        "linear wavelength interpolation (scipy unavailable in this workspace)",
        args.fit_mode,
        fit_detector,
        len(observations),
    )
    plot_paths = plot_results(
        args.output_dir,
        responses,
        d_grid,
        best_fits,
        observations,
        y_residual,
        null_rms,
        args.fit_mode,
        fit_detector,
    )

    print(
        f"Fit observations: {len(observations)}; mode={args.fit_mode}; "
        f"scope={fit_detector} (archive contains {len(all_observations)} observations "
        f"from {len(detectors)} detectors)"
    )
    print(f"SOC model wavelengths: {model_wavelengths[0]:.1f}-{model_wavelengths[-1]:.1f} nm")
    print(f"RCS sequence span: {elapsed_max * 24.0:.3f} hours")
    print(f"Null {args.fit_mode} RMS: {null_rms:.9g}")
    for mode in MODES:
        result = best_fits[mode]
        units = "nm/day" if result["trajectory"] == "linear" else "scale"
        print(
            f"{mode}: {result['trajectory']}={result['parameter']:.6g} {units}; "
            f"mean rate={result['mean_rate_nm_day']:.6g} nm/day; "
            f"max thickness={np.max(result['thickness_nm']):.6g} nm; "
            f"model RMS={result['model_rms']:.9g}"
        )
    print(f"Wrote {fit_path}")
    print(f"Wrote {response_path}")
    print(f"Wrote {prediction_path}")
    if args.fit_mode == "direct":
        print(f"Wrote {node_path}")
    print(f"Wrote {report_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
