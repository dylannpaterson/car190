"""Local multiplicative side-swap calibration in log space."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SideSwapFit:
    """Result from fitting one local image or photodiode time series."""

    event_time: float
    selected_indices: np.ndarray
    selected_times: np.ndarray
    knot_times: np.ndarray
    step_log: float
    correction_factor: float
    rmse_log: float
    r2: float
    fitted_log_values: np.ndarray
    residual_log_values: np.ndarray


def _natural_spline_basis(knot_times, sample_times):
    try:
        from scipy.interpolate import CubicSpline
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Side-swap calibration requires scipy for its natural cubic spline"
        ) from error

    basis = np.empty((sample_times.size, knot_times.size), dtype=float)
    for column in range(knot_times.size):
        knot_values = np.zeros(knot_times.size, dtype=float)
        knot_values[column] = 1.0
        spline = CubicSpline(knot_times, knot_values, bc_type="natural")
        basis[:, column] = spline(sample_times)
    return basis


def _weighted_lstsq(design, values, weights):
    square_root_weights = np.sqrt(weights)
    weighted_design = design * square_root_weights[:, None]
    weighted_values = values * square_root_weights
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design,
        weighted_values,
        rcond=None,
    )
    return coefficients


def fit_log_step_spline(
    times,
    values,
    event_time,
    *,
    pre_points=5,
    post_points=5,
    weights=None,
    robust_iterations=3,
):
    """Fit a continuous local spline plus one multiplicative step.

    The fitted model is ``log(value) = spline(time) + step * post``.  The
    spline has two fixed knots on each side of the event and is continuous
    across the event; the step coefficient is therefore the only allowed
    discontinuity.  The returned correction factor is ``exp(-step)`` for
    samples on the right-hand side of the event.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or times.shape != values.shape:
        raise ValueError("times and values must be one-dimensional arrays of equal length")
    if times.size == 0 or np.any(~np.isfinite(times)):
        raise ValueError("times must contain finite values")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("values must be finite and positive")
    if not np.isfinite(event_time):
        raise ValueError("event_time must be finite")
    if pre_points < 2 or post_points < 2:
        raise ValueError("At least two points are required on each side")

    before = np.flatnonzero(times < event_time)
    after = np.flatnonzero(times >= event_time)
    if before.size < pre_points or after.size < post_points:
        raise ValueError(
            f"Need {pre_points} pre-event and {post_points} post-event points; "
            f"found {before.size} and {after.size}"
        )
    before = before[np.argsort(times[before])][-pre_points:]
    after = after[np.argsort(times[after])][:post_points]
    selected_indices = np.concatenate((before, after))
    selected_indices = selected_indices[np.argsort(times[selected_indices])]
    selected_times = times[selected_indices]
    if np.any(np.diff(selected_times) <= 0.0):
        raise ValueError("Selected side-swap samples must have unique increasing times")

    knot_indices = np.asarray(
        (
            before[0],
            before[-1],
            after[0],
            after[-1],
        ),
        dtype=int,
    )
    knot_times = times[knot_indices]
    if np.any(np.diff(knot_times) <= 0.0):
        raise ValueError("Side-swap spline knots must have unique increasing times")

    selected_log_values = np.log(values[selected_indices])
    post_mask = (selected_times >= event_time).astype(float)
    spline_basis = _natural_spline_basis(knot_times, selected_times)
    design = np.column_stack((spline_basis, post_mask))

    if weights is None:
        base_weights = np.ones(selected_indices.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != times.shape:
            raise ValueError("weights must have the same shape as times")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and positive")
        base_weights = weights[selected_indices]

    fit_weights = base_weights.copy()
    coefficients = _weighted_lstsq(design, selected_log_values, fit_weights)
    for _ in range(int(robust_iterations)):
        residual = selected_log_values - design @ coefficients
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            break
        huber_limit = 1.345 * scale
        robust_weights = np.ones_like(residual)
        large = np.abs(residual) > huber_limit
        robust_weights[large] = huber_limit / np.abs(residual[large])
        fit_weights = base_weights * robust_weights
        coefficients = _weighted_lstsq(design, selected_log_values, fit_weights)

    fitted_log_values = design @ coefficients
    residual_log_values = selected_log_values - fitted_log_values
    residual_sum = float(np.sum(residual_log_values * residual_log_values))
    total_sum = float(
        np.sum((selected_log_values - np.mean(selected_log_values)) ** 2)
    )
    return SideSwapFit(
        event_time=float(event_time),
        selected_indices=selected_indices,
        selected_times=selected_times,
        knot_times=knot_times,
        step_log=float(coefficients[-1]),
        correction_factor=float(np.exp(-coefficients[-1])),
        rmse_log=float(np.sqrt(np.mean(residual_log_values ** 2))),
        r2=float(1.0 - residual_sum / total_sum)
        if total_sum > 0.0
        else np.nan,
        fitted_log_values=fitted_log_values,
        residual_log_values=residual_log_values,
    )


def apply_side_swap_correction(times, values, event_time, correction_factor):
    """Apply a fitted right-hand multiplicative correction to a series."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or times.shape != values.shape:
        raise ValueError("times and values must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(times)):
        raise ValueError("times must contain finite values")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("values must be finite and positive")
    if not np.isfinite(correction_factor) or correction_factor <= 0.0:
        raise ValueError("correction_factor must be finite and positive")
    corrected = values.copy()
    corrected[times >= float(event_time)] *= float(correction_factor)
    return corrected
