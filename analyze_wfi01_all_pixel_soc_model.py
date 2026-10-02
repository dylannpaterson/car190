"""Fit one global NP SOC trajectory to every valid WFI01 pixel.

For each LED and observation epoch, the observation for pixel ``p`` is

    log(current[p] / first[p]).

The model predicts one common log throughput ratio for all pixels in that
exposure. The primary trajectory jointly fits the physical thickness at the
first exposure and one positive epoch multiplier on a square-root accumulation
prior; a one-parameter square-root fit is retained as a baseline.
The exact all-pixel squared-error loss is retained using count, sum, and
sum-of-squares sufficient statistics, so fitting does not require keeping
roughly one billion pixel ratios in memory.
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_rcs_soc_ice_model import (
    DEFAULT_MODEL_PATH,
    MODES,
    broadband_response_curves,
    fit_periodic_model,
    generalized_relative_entropy,
    mean_endpoint_rate,
    mean_squared_log_node_curvature,
    read_model_table,
    response_prediction,
    soc_sqrt_thickness,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "wfi01_rcs_flat_index.csv"
DATA_DIR = ROOT / "MAST" / "Roman"
STATISTICS_PATH = ROOT / "wfi01_all_pixel_log_statistics.csv"
FIT_PATH = ROOT / "wfi01_all_pixel_soc_global_fit.csv"
PREDICTION_PATH = ROOT / "wfi01_all_pixel_soc_global_predictions.csv"
NODE_PATH = ROOT / "wfi01_all_pixel_soc_global_nodes.csv"
REPORT_PATH = ROOT / "wfi01_all_pixel_soc_global_fit.md"
PLOT_PATH = ROOT / "wfi01_all_pixel_soc_global_fit.png"
GLOBAL_TRAJECTORIES = ("soc_sqrt", "entropy_sqrt")
DEFAULT_STEP_TIME = "2026-09-14T23:38:00Z"


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def read_wfi01_index(path, data_dir):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No WFI01 index rows found")
    if {row["detector"] for row in rows} != {"WFI01"}:
        raise ValueError("All index rows must be from WFI01")

    global_start = min(parse_mast_time(row["exposure_start_time"]) for row in rows)
    observations = []
    led_counts = []
    n_repeats = None
    for led in range(1, 7):
        led_rows = sorted(
            [row for row in rows if int(row["led"]) == led],
            key=lambda row: parse_mast_time(row["exposure_start_time"]),
        )
        led_counts.append(len(led_rows))
        if len(led_rows) < 3:
            raise ValueError(f"LED {led} has too few epochs: {len(led_rows)}")
        if n_repeats is None:
            n_repeats = len(led_rows)
        for repeat, row in enumerate(led_rows):
            image_path = data_dir / f'{row["fileSetName"]}_cal.asdf'
            if not image_path.exists():
                raise FileNotFoundError(f"Calibrated image not found: {image_path}")
            observations.append(
                {
                    "led": led,
                    "repeat": repeat,
                    "group": (led - 1) * n_repeats + repeat,
                    "detector": 1,
                    "file_set_name": row["fileSetName"],
                    "exposure_start_time": row["exposure_start_time"],
                    "days": (
                        parse_mast_time(row["exposure_start_time"]) - global_start
                    ).total_seconds()
                    / 86400.0,
                    "image_path": image_path,
                }
            )
    if len(set(led_counts)) != 1:
        raise ValueError(f"LEDs have unequal epoch counts: {led_counts}")
    return observations, global_start


def load_asdf_image(path):
    try:
        import asdf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Building all-pixel statistics requires the 'asdf' package"
        ) from error

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
            model = af.tree["roman"]
            data = np.asarray(model["data"])
            dq = np.asarray(model["dq"])
    if data.shape != dq.shape:
        raise ValueError(f"Data/DQ shape mismatch in {path}")
    return data, dq


def pixel_log_statistics(baseline_data, baseline_dq, current_data, current_dq, chunk_rows=256):
    """Return exact sufficient statistics for valid per-pixel log ratios."""
    if baseline_data.shape != current_data.shape or baseline_dq.shape != current_dq.shape:
        raise ValueError("Baseline and current image shapes must match")
    if baseline_data.shape != baseline_dq.shape:
        raise ValueError("Image and DQ shapes must match")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    count = 0
    total = 0.0
    total_squared = 0.0
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, baseline_data.shape[0], chunk_rows):
        stop = min(start + chunk_rows, baseline_data.shape[0])
        baseline = np.asarray(baseline_data[start:stop])
        current = np.asarray(current_data[start:stop])
        valid = (
            np.isfinite(baseline)
            & np.isfinite(current)
            & (np.asarray(baseline_dq[start:stop]) == 0)
            & (np.asarray(current_dq[start:stop]) == 0)
            & (baseline > 0.0)
            & (current > 0.0)
        )
        if not np.any(valid):
            continue
        values = np.log(current[valid].astype(np.float64)) - np.log(
            baseline[valid].astype(np.float64)
        )
        count += int(values.size)
        total += float(np.sum(values, dtype=np.float64))
        total_squared += float(np.sum(values * values, dtype=np.float64))
        minimum = min(minimum, float(np.min(values)))
        maximum = max(maximum, float(np.max(values)))

    if count == 0:
        raise ValueError("No valid positive pixels remain for an exposure pair")
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    return {
        "pixel_count": count,
        "sum_log_change": total,
        "sum_squared_log_change": total_squared,
        "mean_log_change": mean,
        "pixel_rms_about_mean_log": float(np.sqrt(variance)),
        "naive_standard_error_log": float(np.sqrt(variance / count)),
        "min_log_change": minimum,
        "max_log_change": maximum,
    }


def write_statistics(path, rows):
    fields = [
        "led",
        "repeat",
        "group",
        "file_set_name",
        "baseline_file_set_name",
        "exposure_start_time",
        "days",
        "pixel_count",
        "sum_log_change",
        "sum_squared_log_change",
        "mean_log_change",
        "pixel_rms_about_mean_log",
        "naive_standard_error_log",
        "min_log_change",
        "max_log_change",
    ]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    temporary_path.replace(path)


def build_all_pixel_statistics(index_path, data_dir, output_path, chunk_rows=256):
    observations, global_start = read_wfi01_index(index_path, data_dir)
    result_rows = []
    for led in range(1, 7):
        led_rows = [row for row in observations if row["led"] == led]
        baseline_row = led_rows[0]
        baseline_data, baseline_dq = load_asdf_image(baseline_row["image_path"])
        for row in led_rows:
            if row["repeat"] == 0:
                current_data, current_dq = baseline_data, baseline_dq
            else:
                current_data, current_dq = load_asdf_image(row["image_path"])
            statistics = pixel_log_statistics(
                baseline_data,
                baseline_dq,
                current_data,
                current_dq,
                chunk_rows=chunk_rows,
            )
            output_row = {
                **row,
                **statistics,
                "baseline_file_set_name": baseline_row["file_set_name"],
            }
            result_rows.append(output_row)
            print(
                f"LED {led}, epoch {row['repeat']:02d}: "
                f"n={statistics['pixel_count']:,}, "
                f"mean={statistics['mean_log_change']:.9g}, "
                f"pixel RMS={statistics['pixel_rms_about_mean_log']:.9g}"
            )
            if row["repeat"] != 0:
                del current_data, current_dq
        del baseline_data, baseline_dq
    write_statistics(output_path, result_rows)
    return result_rows, global_start


def load_all_pixel_statistics(path):
    with path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("No all-pixel statistic rows found")

    rows = []
    for source in source_rows:
        count = int(source["pixel_count"])
        total = float(source["sum_log_change"])
        total_squared = float(source["sum_squared_log_change"])
        if count <= 0 or not np.isfinite([total, total_squared]).all():
            raise ValueError("Invalid all-pixel sufficient statistics")
        mean = total / count
        rows.append(
            {
                "led": int(source["led"]),
                "repeat": int(source["repeat"]),
                "group": int(source["group"]),
                "detector": 1,
                "file_set_name": source["file_set_name"],
                "exposure_start_time": source["exposure_start_time"],
                "days": float(source["days"]),
                "pixel_count": count,
                "sum_log_change": total,
                "sum_squared_log_change": total_squared,
                "log_change": mean,
                "within_sum_squares": max(total_squared - count * mean * mean, 0.0),
            }
        )
    rows.sort(key=lambda row: row["group"])
    expected_groups = list(range(len(rows)))
    if [row["group"] for row in rows] != expected_groups:
        raise ValueError(
            "All-pixel statistic groups must be unique and span "
            f"0 through {len(rows) - 1}"
        )
    return rows


def all_pixel_loss_components(observations, prediction):
    """Decompose exact pixel SSE into within- and between-exposure terms."""
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(observations),) or not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction must be finite and match the observations")
    counts = np.asarray([row["pixel_count"] for row in observations], dtype=float)
    means = np.asarray([row["log_change"] for row in observations], dtype=float)
    within = float(np.sum([row["within_sum_squares"] for row in observations]))
    between = float(np.sum(counts * (means - prediction) ** 2))
    return {
        "within_sse": within,
        "between_sse": between,
        "total_sse": within + between,
        "pixel_count": int(np.sum(counts)),
    }


def reference_observation_mask(observations):
    """Identify the first, identity-normalized observation for every LED."""
    mask = np.zeros(len(observations), dtype=bool)
    for led in sorted({row["led"] for row in observations}):
        index = min(
            (index for index, row in enumerate(observations) if row["led"] == led),
            key=lambda index: observations[index]["days"],
        )
        mask[index] = True
    return mask


def event_step_design(observations, step_mask=None):
    """Center an event indicator on each LED's normalization exposure."""
    if step_mask is None:
        return None
    supplied = np.asarray(step_mask)
    if supplied.shape != (len(observations),):
        raise ValueError("Step mask must contain one value per observation")
    if supplied.dtype != bool:
        design = np.asarray(supplied, dtype=float)
        if not np.all(np.isfinite(design)):
            raise ValueError("Step design values must be finite")
        return design
    first_indices = {
        led: min(
            (index for index, row in enumerate(observations) if row["led"] == led),
            key=lambda index: observations[index]["days"],
        )
        for led in sorted({row["led"] for row in observations})
    }
    return np.asarray(
        [
            float(supplied[index]) - float(supplied[first_indices[row["led"]]])
            for index, row in enumerate(observations)
        ]
    )


def profile_step_offset(observations, prediction, step_mask=None):
    """Profile one common reference-centered log-throughput offset exactly."""
    design = event_step_design(observations, step_mask)
    if design is None or not np.any(design):
        return 0.0
    counts = np.asarray([row["pixel_count"] for row in observations], dtype=float)
    means = np.asarray([row["log_change"] for row in observations], dtype=float)
    denominator = float(np.sum(counts * design**2))
    if denominator <= 0.0:
        raise ValueError("The event step must include positive-pixel observations")
    return float(
        np.sum(counts * design * (means - prediction))
        / denominator
    )


def apply_step_offset(prediction, step_mask, step_log_offset, observations=None):
    """Add a common log-throughput offset to observations after an event."""
    if step_mask is None or step_log_offset == 0.0:
        return np.asarray(prediction, dtype=float)
    design = (
        np.asarray(step_mask, dtype=float)
        if observations is None
        else event_step_design(observations, step_mask)
    )
    prediction = np.asarray(prediction, dtype=float).copy()
    if design.shape != prediction.shape:
        raise ValueError("Step design must match the prediction shape")
    prediction += step_log_offset * design
    return prediction


def fit_global_soc_sqrt(
    observations,
    response_curves,
    d_grid,
    max_scale,
    scale_steps,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
    refinement_steps=2,
    step_mask=None,
    initial_thickness_nm=0.0,
    fit_initial_thickness=False,
    initial_thickness_starts=None,
    max_initial_thickness_nm=None,
):
    """Fit the square-root scale and optionally the initial physical thickness."""
    if max_scale <= 0.0 or scale_steps < 3:
        raise ValueError("Scale search requires a positive maximum and at least 3 steps")
    if max_initial_thickness_nm is None:
        max_initial_thickness_nm = float(d_grid[-1])
    if not d_grid[0] <= initial_thickness_nm <= max_initial_thickness_nm:
        raise ValueError("Initial thickness must lie inside its fitted bounds")

    days = np.asarray([row["days"] for row in observations], dtype=float)
    base_thickness = soc_sqrt_thickness(
        days,
        reference_thickness_nm,
        reference_time_days,
        reset_day,
    )
    step_design = event_step_design(observations, step_mask)

    evaluations = 0

    def evaluate(scale, fitted_initial_thickness=initial_thickness_nm):
        incremental_thickness = scale * base_thickness
        thickness = fitted_initial_thickness + incremental_thickness
        if np.min(thickness) < d_grid[0] or np.max(thickness) > d_grid[-1]:
            range_violation = max(
                float(d_grid[0] - np.min(thickness)),
                float(np.max(thickness) - d_grid[-1]),
                0.0,
            )
            return (
                1.0e12 + 1.0e8 * range_violation**2,
                None,
                thickness,
                incremental_thickness,
                0.0,
            )
        prediction = response_prediction(
            observations, response_curves, d_grid, thickness
        )
        step_log_offset = profile_step_offset(observations, prediction, step_design)
        prediction = apply_step_offset(prediction, step_design, step_log_offset)
        loss = all_pixel_loss_components(observations, prediction)
        return (
            loss["between_sse"],
            prediction,
            thickness,
            incremental_thickness,
            step_log_offset,
        )

    coarse_scales = np.linspace(0.0, max_scale, scale_steps)
    coarse_scores = np.empty_like(coarse_scales)
    for index, scale in enumerate(coarse_scales):
        coarse_scores[index] = evaluate(float(scale))[0]
        evaluations += 1
    best_coarse_index = int(np.argmin(coarse_scores))
    local_minima = [
        index
        for index in range(1, len(coarse_scores) - 1)
        if np.isfinite(coarse_scores[index])
        and coarse_scores[index] <= coarse_scores[index - 1]
        and coarse_scores[index] <= coarse_scores[index + 1]
    ]
    if np.isfinite(coarse_scores[0]) and coarse_scores[0] <= coarse_scores[1]:
        local_minima.insert(0, 0)
    if np.isfinite(coarse_scores[-1]) and coarse_scores[-1] <= coarse_scores[-2]:
        local_minima.append(len(coarse_scores) - 1)

    scales = coarse_scales
    scores = coarse_scores
    best_index = best_coarse_index
    for _ in range(refinement_steps):
        if best_index == 0 or best_index == len(scales) - 1:
            break
        scales = np.linspace(scales[best_index - 1], scales[best_index + 1], 101)
        scores = np.empty_like(scales)
        for index, scale in enumerate(scales):
            scores[index] = evaluate(float(scale))[0]
            evaluations += 1
        best_index = int(np.argmin(scores))

    scale = float(scales[best_index])
    fitted_initial_thickness = float(initial_thickness_nm)
    optimizer_method = "grid refinement"
    optimizer_converged = best_coarse_index not in (0, len(coarse_scales) - 1)
    optimizer_starts = 1
    optimizer_converged_starts = int(optimizer_converged)
    if fit_initial_thickness:
        if initial_thickness_starts is None:
            initial_thickness_starts = [initial_thickness_nm]
        thickness_starts = sorted(
            {
                min(
                    max(float(value), float(d_grid[0])),
                    float(max_initial_thickness_nm),
                )
                for value in initial_thickness_starts
                if np.isfinite(value)
            }
        )
        if not thickness_starts:
            raise ValueError("At least one finite initial-thickness start is required")
        scale_starts = sorted(
            {
                min(max(value, 1.0e-6), max_scale)
                for value in (
                    scale,
                    *[coarse_scales[index] for index in local_minima],
                )
                if np.isfinite(value) and value > 0.0
            }
        )
        starts = [
            np.asarray([d0, candidate_scale], dtype=float)
            for d0 in thickness_starts
            for candidate_scale in scale_starts
        ]
        joint_results = []
        try:
            from scipy.optimize import minimize

            for start in starts:
                result = minimize(
                    lambda parameters: evaluate(parameters[1], parameters[0])[0],
                    start,
                    method="SLSQP",
                    bounds=(
                        (float(d_grid[0]), float(max_initial_thickness_nm)),
                        (1.0e-6, max_scale),
                    ),
                    constraints={
                        "type": "ineq",
                        "fun": lambda parameters: d_grid[-1]
                        - np.max(parameters[0] + parameters[1] * base_thickness),
                    },
                    options={"maxiter": 1000, "ftol": 1.0e-12, "disp": False},
                )
                evaluations += int(result.nfev)
                if result.success and np.isfinite(result.fun):
                    joint_results.append((float(result.fun), result.x))
            optimizer_method = "scipy.optimize SLSQP multistart"
        except ModuleNotFoundError:
            from analyze_rcs_soc_ice_model import bounded_pattern_search

            for start in starts:
                parameters, objective, _, count, converged = bounded_pattern_search(
                    lambda values: evaluate(values[1], values[0])[0],
                    start,
                    np.asarray([d_grid[0], 1.0e-6]),
                    np.asarray([max_initial_thickness_nm, max_scale]),
                    initial_step=0.1,
                    tolerance=1.0e-6,
                    max_sweeps=1000,
                )
                evaluations += count
                if converged and np.isfinite(objective):
                    joint_results.append((float(objective), parameters))
            optimizer_method = "bounded pattern-search multistart fallback"
        if not joint_results:
            raise RuntimeError(
                f"No square-root initial-thickness fit converged out of {len(starts)} starts"
            )
        joint_results.sort(key=lambda item: item[0])
        fitted_initial_thickness, scale = map(float, joint_results[0][1])
        optimizer_converged = True
        optimizer_starts = len(starts)
        optimizer_converged_starts = len(joint_results)

    (
        _,
        prediction,
        thickness,
        incremental_thickness,
        step_log_offset,
    ) = evaluate(scale, fitted_initial_thickness)
    evaluations += 1
    loss = all_pixel_loss_components(observations, prediction)
    null_loss = all_pixel_loss_components(observations, np.zeros(len(observations)))
    nonreference_mask = ~reference_observation_mask(observations)
    nonreference_observations = [
        row for row, selected in zip(observations, nonreference_mask) if selected
    ]
    nonreference_loss = all_pixel_loss_components(
        nonreference_observations, prediction[nonreference_mask]
    )
    nonreference_null_loss = all_pixel_loss_components(
        nonreference_observations,
        np.zeros(np.count_nonzero(nonreference_mask)),
    )
    pixel_count = loss["pixel_count"]
    second_basin_delta = np.nan
    ordered_local_scores = sorted(coarse_scores[index] for index in local_minima)
    if len(ordered_local_scores) > 1:
        second_basin_delta = float(ordered_local_scores[1] - ordered_local_scores[0])
    return {
        "trajectory": "soc_sqrt",
        "parameter": scale,
        "initial_thickness_nm": fitted_initial_thickness,
        "fit_initial_thickness": bool(fit_initial_thickness),
        "prediction": prediction,
        "thickness_nm": thickness,
        "incremental_thickness_nm": incremental_thickness,
        "step_log_offset": step_log_offset,
        "step_percent": 100.0 * np.expm1(step_log_offset),
        "step_observations": (
            int(np.count_nonzero(step_design)) if step_design is not None else 0
        ),
        "mean_rate_nm_day": mean_endpoint_rate(observations, thickness),
        "pixel_count": pixel_count,
        "within_sse": loss["within_sse"],
        "between_sse": loss["between_sse"],
        "total_sse": loss["total_sse"],
        "null_between_sse": null_loss["between_sse"],
        "null_total_sse": null_loss["total_sse"],
        "pixel_rms_log": float(np.sqrt(loss["total_sse"] / pixel_count)),
        "null_pixel_rms_log": float(np.sqrt(null_loss["total_sse"] / pixel_count)),
        "nonreference_pixel_count": nonreference_loss["pixel_count"],
        "nonreference_total_sse": nonreference_loss["total_sse"],
        "nonreference_null_total_sse": nonreference_null_loss["total_sse"],
        "nonreference_pixel_rms_log": float(
            np.sqrt(
                nonreference_loss["total_sse"]
                / nonreference_loss["pixel_count"]
            )
        ),
        "nonreference_null_pixel_rms_log": float(
            np.sqrt(
                nonreference_null_loss["total_sse"]
                / nonreference_null_loss["pixel_count"]
            )
        ),
        "between_rms_log": float(np.sqrt(loss["between_sse"] / pixel_count)),
        "null_between_rms_log": float(
            np.sqrt(null_loss["between_sse"] / pixel_count)
        ),
        "within_rms_log": float(np.sqrt(loss["within_sse"] / pixel_count)),
        "coarse_grid_spacing": float(coarse_scales[1] - coarse_scales[0]),
        "coarse_local_minima": len(local_minima),
        "coarse_local_minimum_scales": [
            float(coarse_scales[index]) for index in local_minima
        ],
        "second_basin_delta_sse": second_basin_delta,
        "optimizer_method": optimizer_method,
        "optimizer_starts": optimizer_starts,
        "optimizer_converged_starts": optimizer_converged_starts,
        "optimizer_evaluations": evaluations,
        "optimizer_converged": optimizer_converged,
    }


def fit_global_entropy_sqrt(
    observations,
    response_curves,
    d_grid,
    initial_scale,
    initial_scales,
    max_scale,
    reference_thickness_nm,
    reference_time_days,
    reset_day,
    entropy_weight,
    smoothness_weight,
    max_node_multiplier,
    step_mask=None,
    initial_thickness_nm=0.0,
    fit_initial_thickness=False,
    initial_thickness_starts=None,
    max_initial_thickness_nm=None,
    response_factory=None,
    optical_initial_parameters=None,
    optical_parameter_starts=None,
    optical_parameter_bounds=None,
    optical_penalty=None,
    fit_mask=None,
    initial_node_multipliers=None,
    node_start_perturbation=0.03,
    optimizer_maxiter=2000,
    optimizer_ftol=1.0e-12,
):
    """Fit one shared entropy-regularized trajectory and optional response nuisance."""
    try:
        from scipy.optimize import minimize

        scipy_available = True
    except ModuleNotFoundError:
        from analyze_rcs_soc_ice_model import bounded_pattern_search

        minimize = None
        scipy_available = False

    repeats = sorted({row["repeat"] for row in observations})
    if len(repeats) < 3:
        raise ValueError("The NP trajectory requires at least three epochs")
    if optimizer_maxiter < 1 or optimizer_ftol <= 0.0:
        raise ValueError("Optimizer iteration and tolerance controls must be positive")
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeats)}
    observation_nodes = np.asarray(
        [repeat_to_index[row["repeat"]] for row in observations], dtype=int
    )
    days = np.asarray([row["days"] for row in observations], dtype=float)
    chronological_order = np.argsort(days, kind="stable")
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
    if fit_mask is None:
        fit_mask = np.ones(len(observations), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != (len(observations),) or not np.any(fit_mask):
        raise ValueError("Fit mask must select at least one supplied observation")
    fit_observations = [
        row for row, selected in zip(observations, fit_mask) if selected
    ]
    step_design = event_step_design(observations, step_mask)
    null_between_sse = all_pixel_loss_components(
        fit_observations, np.zeros(np.count_nonzero(fit_mask))
    )["between_sse"]
    if null_between_sse <= 0.0:
        raise ValueError("The all-pixel NP fit requires nonzero exposure means")

    if max_initial_thickness_nm is None:
        max_initial_thickness_nm = float(d_grid[-1])
    if not d_grid[0] <= initial_thickness_nm <= max_initial_thickness_nm:
        raise ValueError("Initial thickness must lie inside its fitted bounds")

    if response_factory is None:
        optical_initial_parameters = np.empty(0, dtype=float)
        optical_parameter_starts = [optical_initial_parameters]
        optical_parameter_bounds = []
    else:
        if optical_initial_parameters is None or optical_parameter_bounds is None:
            raise ValueError(
                "A response factory requires optical initial parameters and bounds"
            )
        optical_initial_parameters = np.asarray(
            optical_initial_parameters, dtype=float
        )
        if optical_initial_parameters.ndim != 1 or not np.all(
            np.isfinite(optical_initial_parameters)
        ):
            raise ValueError("Optical initial parameters must be a finite vector")
        optical_parameter_bounds = list(optical_parameter_bounds)
        if len(optical_parameter_bounds) != len(optical_initial_parameters):
            raise ValueError("Each optical parameter requires one bound")
        if optical_parameter_starts is None:
            optical_parameter_starts = [optical_initial_parameters]
        optical_parameter_starts = [
            np.asarray(start, dtype=float) for start in optical_parameter_starts
        ]
        if any(
            start.shape != optical_initial_parameters.shape
            or not np.all(np.isfinite(start))
            for start in optical_parameter_starts
        ):
            raise ValueError("Optical starts must match the initial-parameter vector")

    def unpack(parameters):
        if fit_initial_thickness:
            fitted_initial_thickness = float(parameters[0])
            scale_index = 1
        else:
            fitted_initial_thickness = float(initial_thickness_nm)
            scale_index = 0
        log_nodes = np.zeros(len(repeats), dtype=float)
        node_start = scale_index + 1
        node_stop = node_start + len(repeats) - 1
        log_nodes[1:] = parameters[node_start:node_stop]
        scale = float(np.exp(parameters[scale_index]))
        node_multipliers = np.exp(log_nodes)
        optical_parameters = np.asarray(parameters[node_stop:], dtype=float)
        incremental_thickness = (
            scale * base_thickness * node_multipliers[observation_nodes]
        )
        node_incremental_thickness = scale * base_node_thickness * node_multipliers
        thickness = fitted_initial_thickness + incremental_thickness
        node_thickness = fitted_initial_thickness + node_incremental_thickness
        return (
            fitted_initial_thickness,
            scale,
            log_nodes,
            node_multipliers,
            incremental_thickness,
            node_incremental_thickness,
            thickness,
            node_thickness,
            optical_parameters,
        )

    constraint_tolerance = 1.0e-7

    def evaluate(parameters, details=False):
        (
            fitted_initial_thickness,
            scale,
            log_nodes,
            node_multipliers,
            incremental_thickness,
            node_incremental_thickness,
            thickness,
            node_thickness,
            optical_parameters,
        ) = unpack(parameters)
        monotonic_violation = np.maximum(-np.diff(node_thickness), 0.0)
        range_violation = np.asarray(
            [
                max(float(d_grid[0] - np.min(thickness)), 0.0),
                max(float(np.max(thickness) - d_grid[-1]), 0.0),
            ]
        )
        chronological_violation = np.maximum(
            -np.diff(thickness[chronological_order]),
            0.0,
        )
        if (
            np.any(monotonic_violation > constraint_tolerance)
            or np.any(chronological_violation > constraint_tolerance)
            or np.any(range_violation > constraint_tolerance)
        ):
            return 1.0e6 + 1.0e3 * float(
                np.sum(monotonic_violation**2)
                + np.sum(chronological_violation**2)
                + np.sum(range_violation**2)
            )
        fitted_response_curves = (
            response_curves
            if response_factory is None
            else response_factory(optical_parameters)
        )
        if fitted_response_curves is None:
            return 1.0e6
        prediction = response_prediction(
            observations,
            fitted_response_curves,
            d_grid,
            np.clip(thickness, d_grid[0], d_grid[-1]),
        )
        fitted_step_design = (
            None if step_design is None else step_design[fit_mask]
        )
        step_log_offset = profile_step_offset(
            fit_observations,
            prediction[fit_mask],
            fitted_step_design,
        )
        prediction = apply_step_offset(prediction, step_design, step_log_offset)
        loss = all_pixel_loss_components(fit_observations, prediction[fit_mask])
        data_term = loss["between_sse"] / null_between_sse
        entropy = generalized_relative_entropy(node_multipliers)
        smoothness = mean_squared_log_node_curvature(log_nodes, node_days)
        optical_penalty_value = (
            0.0
            if optical_penalty is None
            else float(optical_penalty(optical_parameters))
        )
        if not np.isfinite(optical_penalty_value):
            return 1.0e6
        objective = (
            data_term
            + entropy_weight * entropy
            + smoothness_weight * smoothness
            + optical_penalty_value
        )
        if details:
            return {
                "scale": scale,
                "initial_thickness_nm": fitted_initial_thickness,
                "log_nodes": log_nodes,
                "node_multipliers": node_multipliers,
                "incremental_thickness_nm": incremental_thickness,
                "node_incremental_thickness_nm": node_incremental_thickness,
                "thickness_nm": thickness,
                "node_thickness_nm": node_thickness,
                "optical_parameters": optical_parameters,
                "prediction": prediction,
                "step_log_offset": step_log_offset,
                "loss": loss,
                "data_term": data_term,
                "entropy": entropy,
                "smoothness": smoothness,
                "optical_penalty": optical_penalty_value,
                "objective": objective,
            }
        return objective

    def constraints(parameters):
        thickness = unpack(parameters)[6]
        node_thickness = unpack(parameters)[7]
        return np.concatenate(
            (
                np.diff(node_thickness),
                np.diff(thickness[chronological_order]),
                [np.min(thickness) - d_grid[0], d_grid[-1] - np.max(thickness)],
            )
        )

    log_node_bound = float(np.log(max_node_multiplier))
    if node_start_perturbation < 0.0:
        raise ValueError("Node-start perturbation must be non-negative")
    bounds = [(np.log(1.0e-6), np.log(max_scale))] + [
        (-log_node_bound, log_node_bound)
    ] * (len(repeats) - 1)
    if fit_initial_thickness:
        bounds.insert(
            0,
            (float(d_grid[0]), float(max_initial_thickness_nm)),
        )
    bounds.extend(optical_parameter_bounds)
    candidate_scales = list(initial_scales) + [initial_scale]
    if not fit_initial_thickness:
        candidate_scales.extend(
            initial_scale * factor for factor in (0.75, 1.0, 1.25)
        )
    candidate_scales = sorted(
        {
            min(max(float(scale), 1.0e-6), max_scale)
            for scale in candidate_scales
            if np.isfinite(scale) and scale > 0.0
        }
    )
    phase = np.linspace(0.0, 2.0 * np.pi, len(repeats) - 1)
    if initial_node_multipliers is None:
        initial_log_nodes = np.zeros(len(repeats) - 1)
    else:
        initial_node_multipliers = np.asarray(
            initial_node_multipliers, dtype=float
        )
        if (
            initial_node_multipliers.shape != (len(repeats),)
            or not np.all(np.isfinite(initial_node_multipliers))
            or np.any(initial_node_multipliers <= 0.0)
        ):
            raise ValueError(
                "Initial node multipliers must be one finite positive value per epoch"
            )
        normalized_nodes = initial_node_multipliers / initial_node_multipliers[0]
        initial_log_nodes = np.clip(
            np.log(normalized_nodes[1:]),
            -log_node_bound,
            log_node_bound,
        )
    node_starts = [initial_log_nodes]
    if node_start_perturbation > 0.0:
        node_starts.extend(
            [
                np.clip(
                    initial_log_nodes + node_start_perturbation * np.sin(phase),
                    -log_node_bound,
                    log_node_bound,
                ),
                np.clip(
                    initial_log_nodes - node_start_perturbation * np.sin(phase),
                    -log_node_bound,
                    log_node_bound,
                ),
            ]
        )
    if fit_initial_thickness:
        if initial_thickness_starts is None:
            initial_thickness_starts = [initial_thickness_nm]
        thickness_starts = sorted(
            {
                min(
                    max(float(value), float(d_grid[0])),
                    float(max_initial_thickness_nm),
                )
                for value in initial_thickness_starts
                if np.isfinite(value)
            }
        )
        if not thickness_starts:
            raise ValueError("At least one finite initial-thickness start is required")
        trajectory_starts = [
            np.concatenate(([d0, np.log(scale)], node_start))
            for d0 in thickness_starts
            for scale in candidate_scales
            for node_start in node_starts
        ]
    else:
        trajectory_starts = [
            np.concatenate(([np.log(scale)], node_start))
            for scale in candidate_scales
            for node_start in node_starts
        ]
    starts = [
        np.concatenate((trajectory_start, optical_start))
        for trajectory_start in trajectory_starts
        for optical_start in optical_parameter_starts
    ]

    if not scipy_available:
        # Keep the analysis runnable in the dependency-light workspace.  The
        # SciPy path below is preferred because it handles the inequality
        # constraints directly and supports deliberately infeasible starts.
        trajectory_starts = [
            np.concatenate(
                (
                    [d0, np.log(scale)] if fit_initial_thickness else [np.log(scale)],
                    initial_log_nodes,
                )
            )
            for d0 in (
                thickness_starts if fit_initial_thickness else [initial_thickness_nm]
            )
            for scale in candidate_scales
        ]
        starts = [
            np.concatenate((trajectory_start, optical_start))
            for trajectory_start in trajectory_starts
            for optical_start in optical_parameter_starts
        ]

    results = []
    total_evaluations = 0
    for start_index, start in enumerate(starts):
        if scipy_available:
            result = minimize(
                evaluate,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={
                    "maxiter": optimizer_maxiter,
                    "ftol": optimizer_ftol,
                    "disp": False,
                },
            )
            parameters = result.x
            objective = result.fun
            iterations = int(result.nit)
            evaluations = int(result.nfev)
            converged = result.success
        else:
            parameters, objective, iterations, evaluations, converged = bounded_pattern_search(
                evaluate,
                start,
                np.asarray([lower for lower, _ in bounds]),
                np.asarray([upper for _, upper in bounds]),
                initial_step=0.1,
                tolerance=1.0e-5,
                max_sweeps=1500,
            )
        total_evaluations += evaluations
        constraint_values = constraints(parameters)
        candidate_details = evaluate(parameters, details=True)
        if (
            converged
            and np.isfinite(objective)
            and np.min(constraint_values) >= -constraint_tolerance
            and isinstance(candidate_details, dict)
        ):
            results.append(
                (
                    float(candidate_details["objective"]),
                    start_index,
                    parameters,
                    iterations,
                    evaluations,
                    candidate_details,
                )
            )
    if not results:
        raise RuntimeError(
            f"No all-pixel NP optimization start converged out of {len(starts)}"
        )
    results.sort(key=lambda item: item[0])
    (
        objective,
        best_start_index,
        best_parameters,
        best_iterations,
        best_evaluations,
        details,
    ) = results[0]
    loss = details["loss"]
    null_loss = all_pixel_loss_components(
        fit_observations, np.zeros(np.count_nonzero(fit_mask))
    )
    nonreference_mask = fit_mask & ~reference_observation_mask(observations)
    nonreference_observations = [
        row
        for row, selected in zip(observations, nonreference_mask)
        if selected
    ]
    nonreference_loss = all_pixel_loss_components(
        nonreference_observations,
        details["prediction"][nonreference_mask],
    )
    nonreference_null_loss = all_pixel_loss_components(
        nonreference_observations,
        np.zeros(np.count_nonzero(nonreference_mask)),
    )
    pixel_count = loss["pixel_count"]
    converged_objectives = np.asarray([item[0] for item in results])
    near_best = converged_objectives[
        converged_objectives <= converged_objectives[0] + 1.0e-8
    ]
    node_prior_thickness = (
        details["initial_thickness_nm"] + details["scale"] * base_node_thickness
    )
    return {
        "trajectory": "entropy_sqrt",
        "parameter": details["scale"],
        "initial_thickness_nm": details["initial_thickness_nm"],
        "fit_initial_thickness": bool(fit_initial_thickness),
        "fit_observations": int(np.count_nonzero(fit_mask)),
        "held_out_observations": int(np.count_nonzero(~fit_mask)),
        "prediction": details["prediction"],
        "thickness_nm": details["thickness_nm"],
        "incremental_thickness_nm": details["incremental_thickness_nm"],
        "step_log_offset": details["step_log_offset"],
        "step_percent": 100.0 * np.expm1(details["step_log_offset"]),
        "step_observations": (
            int(np.count_nonzero(step_design[fit_mask]))
            if step_design is not None
            else 0
        ),
        "mean_rate_nm_day": mean_endpoint_rate(
            observations, details["thickness_nm"]
        ),
        "pixel_count": pixel_count,
        "within_sse": loss["within_sse"],
        "between_sse": loss["between_sse"],
        "total_sse": loss["total_sse"],
        "null_between_sse": null_loss["between_sse"],
        "null_total_sse": null_loss["total_sse"],
        "pixel_rms_log": float(np.sqrt(loss["total_sse"] / pixel_count)),
        "null_pixel_rms_log": float(
            np.sqrt(null_loss["total_sse"] / pixel_count)
        ),
        "nonreference_pixel_count": nonreference_loss["pixel_count"],
        "nonreference_total_sse": nonreference_loss["total_sse"],
        "nonreference_null_total_sse": nonreference_null_loss["total_sse"],
        "nonreference_pixel_rms_log": float(
            np.sqrt(
                nonreference_loss["total_sse"]
                / nonreference_loss["pixel_count"]
            )
        ),
        "nonreference_null_pixel_rms_log": float(
            np.sqrt(
                nonreference_null_loss["total_sse"]
                / nonreference_null_loss["pixel_count"]
            )
        ),
        "between_rms_log": float(np.sqrt(loss["between_sse"] / pixel_count)),
        "null_between_rms_log": float(
            np.sqrt(null_loss["between_sse"] / pixel_count)
        ),
        "within_rms_log": float(np.sqrt(loss["within_sse"] / pixel_count)),
        "node_repeats": np.asarray(repeats, dtype=int),
        "node_days": node_days,
        "node_prior_thickness_nm": node_prior_thickness,
        "node_multipliers": details["node_multipliers"],
        "node_thickness_nm": details["node_thickness_nm"],
        "node_incremental_thickness_nm": details[
            "node_incremental_thickness_nm"
        ],
        "entropy_penalty": details["entropy"],
        "smoothness_penalty": details["smoothness"],
        "optical_parameters": details["optical_parameters"],
        "optical_penalty": details["optical_penalty"],
        "entropy_weight": entropy_weight,
        "smoothness_weight": smoothness_weight,
        "normalized_data_term": details["data_term"],
        "penalized_objective": objective,
        "optimizer_method": (
            "scipy.optimize SLSQP multistart"
            if scipy_available
            else "bounded pattern-search multistart fallback"
        ),
        "optimizer_starts": len(starts),
        "optimizer_converged_starts": len(results),
        "optimizer_best_start_index": best_start_index,
        "optimizer_iterations": best_iterations,
        "optimizer_evaluations": total_evaluations,
        "optimizer_objective_spread": float(np.ptp(converged_objectives)),
        "optimizer_near_best_starts": len(near_best),
        "optimizer_near_best_spread": float(np.ptp(near_best)),
        "optimizer_converged": True,
    }


def write_fit_csv(path, fits, selected_mode, selected_trajectory):
    fields = [
        "sed_mode",
        "selected",
        "trajectory",
        "scale",
        "initial_thickness_nm",
        "mean_rate_nm_day",
        "max_thickness_nm",
        "max_incremental_thickness_nm",
        "step_log_offset",
        "step_percent",
        "step_observations",
        "pixel_comparisons",
        "nonreference_pixel_comparisons",
        "null_pixel_rms_log",
        "model_pixel_rms_log",
        "nonreference_null_pixel_rms_log",
        "nonreference_model_pixel_rms_log",
        "total_sse_improvement_fraction",
        "within_sse",
        "null_between_sse",
        "model_between_sse",
        "null_total_sse",
        "model_total_sse",
        "within_pixel_rms_log",
        "null_between_exposure_rms_log",
        "model_between_exposure_rms_log",
        "between_sse_improvement_fraction",
        "coarse_grid_spacing",
        "coarse_local_minima",
        "coarse_local_minimum_scales",
        "second_basin_delta_sse",
        "entropy_relative_to_unity",
        "log_node_curvature",
        "entropy_weight",
        "smoothness_weight",
        "optimizer_method",
        "optimizer_starts",
        "optimizer_converged_starts",
        "optimizer_best_start_index",
        "optimizer_iterations",
        "optimizer_evaluations",
        "optimizer_objective_spread",
        "optimizer_near_best_starts",
        "optimizer_near_best_spread",
        "optimizer_converged",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            for trajectory in GLOBAL_TRAJECTORIES:
                result = fits[mode][trajectory]
                local_scales = result.get("coarse_local_minimum_scales", "")
                if isinstance(local_scales, list):
                    local_scales = ";".join(f"{value:.12g}" for value in local_scales)
                writer.writerow(
                    {
                        "sed_mode": mode,
                        "selected": (
                            mode == selected_mode and trajectory == selected_trajectory
                        ),
                        "trajectory": result["trajectory"],
                        "scale": f'{result["parameter"]:.12g}',
                        "initial_thickness_nm": f'{result.get("initial_thickness_nm", 0.0):.12g}',
                        "mean_rate_nm_day": f'{result["mean_rate_nm_day"]:.12g}',
                        "max_thickness_nm": f'{np.max(result["thickness_nm"]):.12g}',
                        "max_incremental_thickness_nm": (
                            f'{np.max(result.get("incremental_thickness_nm", result["thickness_nm"])):.12g}'
                        ),
                        "step_log_offset": f'{result["step_log_offset"]:.12g}',
                        "step_percent": f'{result["step_percent"]:.12g}',
                        "step_observations": result["step_observations"],
                        "pixel_comparisons": result["pixel_count"],
                        "nonreference_pixel_comparisons": result[
                            "nonreference_pixel_count"
                        ],
                        "null_pixel_rms_log": f'{result["null_pixel_rms_log"]:.12g}',
                        "model_pixel_rms_log": f'{result["pixel_rms_log"]:.12g}',
                        "nonreference_null_pixel_rms_log": (
                            f'{result["nonreference_null_pixel_rms_log"]:.12g}'
                        ),
                        "nonreference_model_pixel_rms_log": (
                            f'{result["nonreference_pixel_rms_log"]:.12g}'
                        ),
                        "total_sse_improvement_fraction": (
                            f'{1.0 - result["total_sse"] / result["null_total_sse"]:.12g}'
                        ),
                        "within_sse": f'{result["within_sse"]:.12g}',
                        "null_between_sse": f'{result["null_between_sse"]:.12g}',
                        "model_between_sse": f'{result["between_sse"]:.12g}',
                        "null_total_sse": f'{result["null_total_sse"]:.12g}',
                        "model_total_sse": f'{result["total_sse"]:.12g}',
                        "within_pixel_rms_log": f'{result["within_rms_log"]:.12g}',
                        "null_between_exposure_rms_log": (
                            f'{result["null_between_rms_log"]:.12g}'
                        ),
                        "model_between_exposure_rms_log": (
                            f'{result["between_rms_log"]:.12g}'
                        ),
                        "between_sse_improvement_fraction": (
                            f'{1.0 - result["between_sse"] / result["null_between_sse"]:.12g}'
                        ),
                        "coarse_grid_spacing": f'{result.get("coarse_grid_spacing", "")}',
                        "coarse_local_minima": result.get("coarse_local_minima", ""),
                        "coarse_local_minimum_scales": local_scales,
                        "second_basin_delta_sse": f'{result.get("second_basin_delta_sse", "")}',
                        "entropy_relative_to_unity": f'{result.get("entropy_penalty", "")}',
                        "log_node_curvature": f'{result.get("smoothness_penalty", "")}',
                        "entropy_weight": result.get("entropy_weight", ""),
                        "smoothness_weight": result.get("smoothness_weight", ""),
                        "optimizer_method": result.get("optimizer_method", "grid refinement"),
                        "optimizer_starts": result.get("optimizer_starts", ""),
                        "optimizer_converged_starts": result.get(
                            "optimizer_converged_starts", ""
                        ),
                        "optimizer_best_start_index": result.get(
                            "optimizer_best_start_index", ""
                        ),
                        "optimizer_iterations": result.get("optimizer_iterations", ""),
                        "optimizer_evaluations": result["optimizer_evaluations"],
                        "optimizer_objective_spread": result.get(
                            "optimizer_objective_spread", ""
                        ),
                        "optimizer_near_best_starts": result.get(
                            "optimizer_near_best_starts", ""
                        ),
                        "optimizer_near_best_spread": result.get(
                            "optimizer_near_best_spread", ""
                        ),
                        "optimizer_converged": result["optimizer_converged"],
                    }
                )


def write_node_csv(path, fits, selected_mode):
    fields = [
        "sed_mode",
        "selected",
        "repeat",
        "days",
        "hours",
        "sqrt_prior_thickness_nm",
        "node_multiplier",
        "initial_thickness_nm",
        "incremental_thickness_nm",
        "fitted_thickness_nm",
        "interval_rate_nm_day",
        "relative_entropy_contribution",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            result = fits[mode]["entropy_sqrt"]
            previous_day = None
            previous_thickness = None
            node_incremental = result.get(
                "node_incremental_thickness_nm",
                result["node_thickness_nm"],
            )
            for repeat, days, prior, multiplier, incremental, thickness in zip(
                result["node_repeats"],
                result["node_days"],
                result["node_prior_thickness_nm"],
                result["node_multipliers"],
                node_incremental,
                result["node_thickness_nm"],
            ):
                if previous_day is None or days == previous_day:
                    interval_rate = ""
                else:
                    interval_rate = f"{(thickness - previous_thickness) / (days - previous_day):.12g}"
                writer.writerow(
                    {
                        "sed_mode": mode,
                        "selected": mode == selected_mode,
                        "repeat": int(repeat),
                        "days": f"{days:.12g}",
                        "hours": f"{24.0 * days:.12g}",
                        "sqrt_prior_thickness_nm": f"{prior:.12g}",
                        "node_multiplier": f"{multiplier:.12g}",
                        "initial_thickness_nm": f'{result.get("initial_thickness_nm", 0.0):.12g}',
                        "incremental_thickness_nm": f"{incremental:.12g}",
                        "fitted_thickness_nm": f"{thickness:.12g}",
                        "interval_rate_nm_day": interval_rate,
                        "relative_entropy_contribution": (
                            f"{multiplier * np.log(multiplier) - multiplier + 1.0:.12g}"
                        ),
                    }
                )
                previous_day = days
                previous_thickness = thickness


def write_prediction_csv(path, observations, result, selected_mode, step_mask=None):
    if step_mask is None:
        step_mask = np.zeros(len(observations), dtype=bool)
    step_design = event_step_design(observations, step_mask)
    fields = [
        "sed_mode",
        "led",
        "repeat",
        "days",
        "pixel_count",
        "initial_thickness_nm",
        "incremental_thickness_nm",
        "thickness_nm",
        "step_active",
        "step_log_offset",
        "observed_geometric_mean_change_percent",
        "ice_model_change_percent",
        "model_change_percent",
        "observed_mean_log_change",
        "model_log_change",
        "mean_residual_log_change",
        "within_pixel_rms_log",
        "pixel_rms_about_global_model_log",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        incremental_thickness = result.get(
            "incremental_thickness_nm",
            result["thickness_nm"],
        )
        for index, (row, prediction, incremental, thickness) in enumerate(
            zip(
                observations,
                result["prediction"],
                incremental_thickness,
                result["thickness_nm"],
            )
        ):
            step_value = float(step_design[index])
            step_active = step_value != 0.0
            ice_prediction = prediction - result["step_log_offset"] * step_value
            residual = row["log_change"] - prediction
            within_rms = np.sqrt(row["within_sum_squares"] / row["pixel_count"])
            writer.writerow(
                {
                    "sed_mode": selected_mode,
                    "led": row["led"],
                    "repeat": row["repeat"],
                    "days": f'{row["days"]:.12g}',
                    "pixel_count": row["pixel_count"],
                    "initial_thickness_nm": f'{result.get("initial_thickness_nm", 0.0):.12g}',
                    "incremental_thickness_nm": f"{incremental:.12g}",
                    "thickness_nm": f"{thickness:.12g}",
                    "step_active": step_active,
                    "step_log_offset": (
                        f'{result["step_log_offset"] * step_value:.12g}'
                        if step_active
                        else "0"
                    ),
                    "observed_geometric_mean_change_percent": (
                        f"{100.0 * np.expm1(row['log_change']):.12g}"
                    ),
                    "ice_model_change_percent": (
                        f"{100.0 * np.expm1(ice_prediction):.12g}"
                    ),
                    "model_change_percent": f"{100.0 * np.expm1(prediction):.12g}",
                    "observed_mean_log_change": f'{row["log_change"]:.12g}',
                    "model_log_change": f"{prediction:.12g}",
                    "mean_residual_log_change": f"{residual:.12g}",
                    "within_pixel_rms_log": f"{within_rms:.12g}",
                    "pixel_rms_about_global_model_log": (
                        f"{np.sqrt(within_rms**2 + residual**2):.12g}"
                    ),
                }
            )


def write_report(
    path,
    statistics_path,
    model_path,
    observations,
    fits,
    selected_mode,
    selected_trajectory,
    max_scale,
    max_node_multiplier,
    step_time=None,
):
    selected = fits[selected_mode][selected_trajectory]
    fitted_leds = sorted({row["led"] for row in observations})
    fitted_led_label = ", ".join(str(led) for led in fitted_leds)
    next_best_mode = min(
        (mode for mode in MODES if mode != selected_mode),
        key=lambda mode: fits[mode][selected_trajectory]["between_sse"],
    )
    next_best = fits[next_best_mode][selected_trajectory]
    lines = [
        "# WFI01 all-pixel global SOC fit",
        "",
        "## Scope",
        "",
        f"- One global non-parametric trajectory is shared by every WFI01 pixel and fitted LED(s) `{fitted_led_label}`.",
        "- The NP trajectory is `d_i = d0 + scale * m_i * d_sqrt(t_i)`, with one shared initial physical thickness `d0`, one positive multiplier per observation epoch, the first multiplier fixed to one, and non-decreasing fitted thickness.",
        "- Each LED is normalized to its first exposure as `R(d_i) / R(d_first,LED)`; normalization does not force the physical baseline thickness to zero.",
        "- The square-root trajectory is retained as a baseline comparison with the same initial-thickness treatment; it is not the primary model.",
        "- Every comparison requires finite, positive data and `DQ == 0` in both the current and that LED's first exposure.",
        f"- Exact pixel comparisons: {selected['pixel_count']:,} across {len(observations)} LED/epoch pairs.",
        f"- The six identity-normalized reference pairs constrain no model parameter. Excluding them leaves {selected['nonreference_pixel_count']:,} comparisons; both RMS conventions are reported below.",
        "- Pixels and epochs are correlated, so this fit reports descriptive residuals rather than treating the pixel count as an inferential sample size.",
        "",
        "## Loss",
        "",
        "For exposure `g`, the shared trajectory predicts one value `p_g` and each pixel contributes `y_gp = log(current_gp / first_gp)`. The exact loss is:",
        "",
        "```text",
        "SSE = sum_g [sum(y_gp^2) - 2 p_g sum(y_gp) + n_g p_g^2]",
        "```",
        "",
        "The count, sum, and sum of squares in the statistics file therefore reproduce the loss over all pixels without approximation.",
        "",
        "## Results",
        "",
        "| SED treatment | Trajectory | Scale | Initial thickness (nm) | Accumulated thickness (nm) | Mean rate (nm/day) | Final thickness (nm) | All-pixel RMS (log) | Between-exposure SSE removed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if selected["step_observations"]:
        event_label = step_time or "the supplied event time"
        lines[10:10] = [
            f"- A common post-event log-throughput step was profiled after `{event_label}`; it applies to `{selected['step_observations']}` observations.",
            f"- The fitted common step is `{selected['step_percent']:.6f}%` in throughput (`{selected['step_log_offset']:.8f}` in log units).",
        ]
        results_index = lines.index("## Results")
        lines[results_index:results_index] = [
            "For an event-aware fit, the predicted value is `p_g = p_ice,g + b * [I(g after event) - I(g_ref,LED after event)]`; the step offset is therefore normalized to each LED's first exposure and profiled exactly for each candidate ice trajectory.",
            "",
        ]
        header_index = lines.index(
            "| SED treatment | Trajectory | Scale | Initial thickness (nm) | Accumulated thickness (nm) | Mean rate (nm/day) | Final thickness (nm) | All-pixel RMS (log) | Between-exposure SSE removed |"
        )
        lines[header_index] = "| SED treatment | Trajectory | Scale | Initial thickness (nm) | Accumulated thickness (nm) | Mean rate (nm/day) | Final thickness (nm) | Step (%) | All-pixel RMS (log) | Between-exposure SSE removed |"
        lines[header_index + 1] = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    for mode in MODES:
        for trajectory in GLOBAL_TRAJECTORIES:
            result = fits[mode][trajectory]
            selected_label = (
                " (selected)"
                if mode == selected_mode and trajectory == selected_trajectory
                else ""
            )
            if selected["step_observations"]:
                lines.append(
                    f"| {mode}{selected_label} | {trajectory} | "
                    f"{result['parameter']:.6f} | {result.get('initial_thickness_nm', 0.0):.6f} | "
                    f"{np.max(result.get('incremental_thickness_nm', result['thickness_nm'])):.6f} | "
                    f"{result['mean_rate_nm_day']:.6f} | "
                    f"{np.max(result['thickness_nm']):.6f} | {result['step_percent']:.6f} | "
                    f"{result['pixel_rms_log']:.8f} | "
                    f"{100.0 * (1.0 - result['between_sse'] / result['null_between_sse']):.3f}% |"
                )
            else:
                lines.append(
                    f"| {mode}{selected_label} | {trajectory} | "
                    f"{result['parameter']:.6f} | {result.get('initial_thickness_nm', 0.0):.6f} | "
                    f"{np.max(result.get('incremental_thickness_nm', result['thickness_nm'])):.6f} | "
                    f"{result['mean_rate_nm_day']:.6f} | "
                    f"{np.max(result['thickness_nm']):.6f} | {result['pixel_rms_log']:.8f} | "
                    f"{100.0 * (1.0 - result['between_sse'] / result['null_between_sse']):.3f}% |"
                )
    lines.extend(
        [
            "",
            f"The selected `{selected_mode}` NP response has all-pixel RMS `{selected['pixel_rms_log']:.8f}` versus null RMS `{selected['null_pixel_rms_log']:.8f}` when the six identity reference pairs are included.",
            f"Excluding those reference-vs-itself pairs, its all-pixel RMS is `{selected['nonreference_pixel_rms_log']:.8f}` versus null RMS `{selected['nonreference_null_pixel_rms_log']:.8f}`; this is the less optimistic descriptive residual metric.",
            f"It is `{next_best['pixel_rms_log'] - selected['pixel_rms_log']:.8f}` lower in RMS than the next-best `{next_best_mode}` NP treatment.",
            f"It removes `{100.0 * (1.0 - selected['total_sse'] / selected['null_total_sse']):.3f}%` of total all-pixel SSE and `{100.0 * (1.0 - selected['between_sse'] / selected['null_between_sse']):.3f}%` of the between-exposure SSE available to a global model.",
            f"Its within-exposure pixel RMS is `{selected['within_rms_log']:.8f}`; this irreducible spatial/noise term is included in the all-pixel RMS.",
            f"The RMS of the exposure means falls from `{selected['null_between_rms_log']:.8f}` to `{selected['between_rms_log']:.8f}` when weighted by valid pixel counts.",
            f"The NP multipliers range from `{np.min(selected['node_multipliers']):.6f}` to `{np.max(selected['node_multipliers']):.6f}`.",
            f"The jointly fitted initial physical ice thickness is `{selected.get('initial_thickness_nm', 0.0):.6f} nm`; the fitted accumulated thickness reaches `{np.max(selected.get('incremental_thickness_nm', selected['thickness_nm'])):.6f} nm`.",
            "",
            "## LED residuals",
            "",
            "| LED | Final observed change | Final model change | Final residual | RMS mean residual | Between-SSE improvement |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    led_residual_rms = []
    for led in fitted_leds:
        indices = [
            index for index, row in enumerate(observations) if row["led"] == led
        ]
        final_index = max(indices, key=lambda index: observations[index]["days"])
        observed = 100.0 * np.expm1(observations[final_index]["log_change"])
        predicted = 100.0 * np.expm1(selected["prediction"][final_index])
        observed_series = 100.0 * np.expm1(
            [observations[index]["log_change"] for index in indices]
        )
        predicted_series = 100.0 * np.expm1(selected["prediction"][indices])
        counts = np.asarray(
            [observations[index]["pixel_count"] for index in indices], dtype=float
        )
        residual_rms = float(
            np.sqrt(np.average((observed_series - predicted_series) ** 2, weights=counts))
        )
        led_residual_rms.append(residual_rms)
        null_sse = float(
            np.sum(
                counts
                * np.asarray(
                    [observations[index]["log_change"] for index in indices]
                )
                ** 2
            )
        )
        model_sse = float(
            np.sum(
                counts
                * (
                    np.asarray(
                        [observations[index]["log_change"] for index in indices]
                    )
                    - selected["prediction"][indices]
                )
                ** 2
            )
        )
        lines.append(
            f"| {led} | {observed:.4f}% | {predicted:.4f}% | "
            f"{observed - predicted:+.4f} points | {residual_rms:.4f} points | "
            f"{100.0 * (1.0 - model_sse / null_sse):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "The LED residuals are the diagnostic for whether a single shared NP trajectory is adequate. Residual structure that remains coherent by LED indicates a spectral-response or calibration mismatch rather than a failure of the temporal node flexibility.",
            "",
            "## Model quality and recommended use",
            "",
            f"This is a useful aggregate trend fit, but not a high-fidelity per-LED physical prediction. The non-reference all-pixel RMS is `{selected['nonreference_pixel_rms_log']:.8f}` in log units, compared with an irreducible within-exposure pixel RMS of `{selected['within_rms_log']:.8f}`; the weighted RMS of exposure means falls from `{selected['null_between_rms_log']:.8f}` for the null to `{selected['between_rms_log']:.8f}` for the model.",
            f"Across the fitted LEDs, the RMS of the mean residuals spans `{min(led_residual_rms):.4f}` to `{max(led_residual_rms):.4f}` percentage points. This coherent LED-dependent structure means the shared response model is not accurate enough to support a precision absolute ice-rate claim.",
            "The recommended default is the event-aware fixed empirical response with the core SED and entropy-regularized square-root trajectory: it is the best descriptive fit and avoids allowing weakly identified optical nuisance parameters to trade directly against thickness. Use the fixed-response square-root baseline as the conservative model-form check; its core fit gives `11.477738 nm/day` versus `12.544659 nm/day` for the selected entropy trajectory.",
            "The shared-magnitude nuisance fits are not the default: their small held-out improvement is accompanied by a large rate shift, and the magnitude-plus-phase fit reaches the zero-thickness boundary.",
            "",
            "## Optimization",
            "",
            f"- Square-root baseline: full bounded `0-{max_scale:g}` scale scan with coarse spacing `{fits[selected_mode]['soc_sqrt']['coarse_grid_spacing']:.6g}`, `{fits[selected_mode]['soc_sqrt']['coarse_local_minima']}` coarse local minima, and alias-aware initial-thickness refinement.",
            f"- NP objective: all-pixel between-exposure SSE divided by null SSE + `{selected['entropy_weight']}` times generalized relative entropy + `{selected['smoothness_weight']}` times time-aware log-node curvature.",
            f"- NP optimization: `{selected['optimizer_method']}`, `{selected['optimizer_starts']}` starts, `{selected['optimizer_converged_starts']}` converged starts, and `{selected['optimizer_near_best_starts']}` within `1e-8` objective units of the best.",
            f"- NP objective spread among converged starts: `{selected['optimizer_objective_spread']:.6g}`; near-best spread: `{selected['optimizer_near_best_spread']:.6g}`.",
            f"- NP multiplier bound: `1/{max_node_multiplier:g}` to `{max_node_multiplier:g}`; monotonic fitted thickness was enforced.",
        ]
    )
    if np.isfinite(fits[selected_mode]["soc_sqrt"]["second_basin_delta_sse"]):
        lines.append(
            "- For the square-root baseline, the second-best periodic basin has an all-pixel SSE larger by "
            f"`{fits[selected_mode]['soc_sqrt']['second_basin_delta_sse']:.6g}`."
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Pixel sufficient statistics: `{statistics_path}`",
            f"- SOC transmission model: `{model_path}`",
            "- Public mapping identifies WFI01/SCU 1 with SCA serial 22081; applicability of the local `SCA1` empirical response to that detector remains unverified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_fit(path, observations, result, baseline, selected_mode, step_day=None):
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 2.4),
        height_ratios=(2.0, 1.0),
    )
    thickness_axis = fig.add_subplot(grid[0, 0])
    rate_axis = fig.add_subplot(grid[1, 0], sharex=thickness_axis)
    data_axis = fig.add_subplot(grid[0, 1])
    residual_axis = fig.add_subplot(grid[1, 1], sharex=data_axis)
    colors = {led: plt.get_cmap("tab10")(led - 1) for led in range(1, 7)}
    fitted_leds = sorted({row["led"] for row in observations})
    repeat_baseline = []
    for repeat in result["node_repeats"]:
        indices = [
            index
            for index, row in enumerate(observations)
            if row["repeat"] == repeat
        ]
        repeat_baseline.append(
            np.mean(
                baseline.get(
                    "incremental_thickness_nm",
                    baseline["thickness_nm"],
                )[indices]
            )
        )

    for led in fitted_leds:
        indices = [index for index, row in enumerate(observations) if row["led"] == led]
        days = np.asarray([observations[index]["days"] for index in indices])
        observed = 100.0 * np.expm1(
            [observations[index]["log_change"] for index in indices]
        )
        predicted = 100.0 * np.expm1(result["prediction"][indices])
        baseline_predicted = 100.0 * np.expm1(baseline["prediction"][indices])
        data_axis.plot(days, observed, "o", color=colors[led], label=f"LED {led} data")
        data_axis.plot(
            days,
            predicted,
            "-",
            color="#111111",
            linewidth=2.4,
            label="NP model" if led == fitted_leds[0] else None,
        )
        data_axis.plot(
            days,
            baseline_predicted,
            "--",
            color="#777777",
            linewidth=1.6,
            label="Square-root baseline" if led == fitted_leds[0] else None,
        )
        residual_axis.plot(days, observed - predicted, "o-", color=colors[led])

    node_incremental_thickness = result.get(
        "node_incremental_thickness_nm",
        result["node_thickness_nm"],
    )
    thickness_axis.plot(
        result["node_days"],
        node_incremental_thickness,
        "o-",
        color="#111111",
        linewidth=2.0,
        label="Fitted NP accumulation",
    )
    thickness_axis.plot(
        result["node_days"],
        repeat_baseline,
        "--",
        color="#666666",
        linewidth=1.5,
        label="Square-root baseline",
    )
    thickness_axis.set_title(
        "Fitted accumulation\n"
        f"Initial physical thickness = {result.get('initial_thickness_nm', 0.0):.2f} nm"
    )
    thickness_axis.set_ylabel("Accumulated thickness (nm)")
    thickness_axis.set_xlabel("Days since first WFI01 exposure")
    thickness_axis.grid(alpha=0.2)
    thickness_axis.legend(frameon=False)

    node_days = np.asarray(result["node_days"])
    node_thickness = np.asarray(node_incremental_thickness)
    interval_days = np.diff(node_days)
    interval_days_midpoints = 0.5 * (node_days[1:] + node_days[:-1])
    fitted_rates = np.diff(node_thickness) / interval_days
    baseline_rates = np.diff(repeat_baseline) / interval_days
    rate_axis.plot(
        interval_days_midpoints,
        fitted_rates,
        "o-",
        color="#111111",
        linewidth=2.0,
        label="NP derivative",
    )
    rate_axis.plot(
        interval_days_midpoints,
        baseline_rates,
        "--",
        color="#777777",
        linewidth=1.5,
        label="Square-root derivative",
    )
    rate_axis.axhline(0.0, color="black", linewidth=0.8)
    rate_axis.set_title("Accumulation rate")
    rate_axis.set_xlabel("Days since first WFI01 exposure")
    rate_axis.set_ylabel("d(thickness)/dt (nm/day)")
    rate_axis.grid(alpha=0.2)
    rate_axis.legend(frameon=False, fontsize=8)

    if step_day is not None:
        for axis in (thickness_axis, rate_axis, data_axis, residual_axis):
            axis.axvline(
                step_day,
                color="#c1121f",
                linewidth=1.2,
                linestyle=":",
                alpha=0.8,
            )
        data_axis.plot(
            [],
            [],
            color="#c1121f",
            linewidth=1.2,
            linestyle=":",
            label="ACM deployment",
        )

    data_axis.axhline(0.0, color="black", linewidth=0.8)
    data_axis.set_ylabel("Geometric-mean pixel change (%)")
    data_axis.set_title(
        f"WFI01 all-pixel global NP SOC fit, LED(s) {','.join(map(str, fitted_leds))} ({selected_mode} SED)"
    )
    data_axis.legend(frameon=False, ncol=3)
    residual_axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    residual_axis.set_xlabel("Days since first WFI01 exposure")
    residual_axis.set_ylabel("Mean residual (percentage points)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--statistics-path", type=Path, default=STATISTICS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--rebuild-statistics", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--max-scale", type=float, default=20.0)
    parser.add_argument("--scale-steps", type=int, default=4001)
    parser.add_argument("--thickness-grid-steps", type=int, default=2401)
    parser.add_argument("--sqrt-ref-thickness-nm", type=float, default=122.7)
    parser.add_argument("--sqrt-ref-time-days", type=float, default=4.65)
    parser.add_argument("--reset-day", type=float, default=18.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--max-node-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--initial-thickness-nm",
        type=float,
        default=None,
        help="Fix the physical thickness at the first exposure; omitted means fit it",
    )
    parser.add_argument(
        "--initial-thickness-starts",
        default="0,15,40,80,120,160,200,240",
        help="Comma-separated multistart values used when fitting initial thickness",
    )
    parser.add_argument(
        "--max-initial-thickness-nm",
        type=float,
        default=250.0,
        help="Upper bound for a fitted initial physical thickness",
    )
    parser.add_argument("--led", type=int, choices=range(1, 7), default=None)
    parser.add_argument(
        "--max-epoch",
        dest="max_repeat",
        type=int,
        default=None,
        metavar="EPOCH",
        help="Optionally fit only observation epochs up to and including this index",
    )
    parser.add_argument(
        "--max-repeat",
        dest="max_repeat",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--step-time",
        default=DEFAULT_STEP_TIME,
        help="UTC ISO timestamp for the known common ACM throughput step",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_rows < 1 or args.thickness_grid_steps < 2:
        raise ValueError("Chunk and thickness-grid sizes must be positive")
    if args.entropy_weight < 0.0 or args.smoothness_weight < 0.0:
        raise ValueError("Entropy and smoothness weights must be non-negative")
    if args.max_node_multiplier <= 1.0:
        raise ValueError("--max-node-multiplier must be greater than one")
    if args.max_initial_thickness_nm < 0.0:
        raise ValueError("--max-initial-thickness-nm must be non-negative")
    if args.initial_thickness_nm is not None and args.initial_thickness_nm < 0.0:
        raise ValueError("--initial-thickness-nm must be non-negative")
    if args.max_repeat is not None and args.max_repeat < 2:
        raise ValueError("--max-epoch must retain at least epochs 0, 1, and 2")
    try:
        initial_thickness_starts = [
            float(value) for value in args.initial_thickness_starts.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            "--initial-thickness-starts must contain comma-separated numbers"
        ) from error
    if not initial_thickness_starts or not np.all(np.isfinite(initial_thickness_starts)):
        raise ValueError("Initial-thickness starts must be finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.statistics_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_statistics or not args.statistics_path.exists():
        build_all_pixel_statistics(
            args.index_path,
            args.data_dir,
            args.statistics_path,
            chunk_rows=args.chunk_rows,
        )
        print(f"Wrote {args.statistics_path}")
    observations = load_all_pixel_statistics(args.statistics_path)
    if args.led is not None:
        observations = [row for row in observations if row["led"] == args.led]
        if not observations:
            raise ValueError(f"No observations found for LED {args.led}")
    if args.max_repeat is not None:
        observations = [
            row for row in observations if row["repeat"] <= args.max_repeat
        ]
        if not observations:
            raise ValueError(
                f"No observations found through epoch {args.max_repeat}"
            )

    step_day = None
    step_mask = None
    if args.step_time is not None:
        event_time = parse_mast_time(args.step_time)
        global_start = min(
            parse_mast_time(row["exposure_start_time"]) for row in observations
        )
        step_day = (event_time - global_start).total_seconds() / 86400.0
        step_mask = np.asarray(
            [
                parse_mast_time(row["exposure_start_time"]) >= event_time
                for row in observations
            ],
            dtype=bool,
        )
        if not np.any(step_mask) and args.max_repeat is not None:
            # A deliberately truncated pre-event fit has no affected rows.
            step_day = None
            step_mask = None
        elif not np.any(step_mask) or np.all(step_mask):
            raise ValueError("The event time must split the supplied observations")

    model_wavelengths, thicknesses, ratios = read_model_table(args.model_path)
    coefficients, omegas = fit_periodic_model(model_wavelengths, thicknesses, ratios)
    model = (model_wavelengths, thicknesses, coefficients, omegas)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    initial_thickness_limit = (
        args.max_initial_thickness_nm
        if args.initial_thickness_nm is None
        else args.initial_thickness_nm
    )
    fit_initial_thickness = args.initial_thickness_nm is None
    initial_thickness_nm = (
        0.0 if args.initial_thickness_nm is None else args.initial_thickness_nm
    )
    max_initial_thickness_nm = max(
        args.max_initial_thickness_nm,
        initial_thickness_nm,
    )
    d_max = float(np.max(thicknesses))
    if initial_thickness_limit > d_max:
        raise ValueError(
            "Initial-thickness search exceeds the calibrated optical-model range "
            f"of {d_max:g} nm"
        )
    d_grid = np.linspace(0.0, d_max, args.thickness_grid_steps)
    responses = broadband_response_curves(
        model, spectra, qe_wavelengths, qe_values, d_grid
    )

    fits = {}
    for mode in MODES:
        sqrt_fit = fit_global_soc_sqrt(
            observations,
            responses[mode],
            d_grid,
            args.max_scale,
            args.scale_steps,
            args.sqrt_ref_thickness_nm,
            args.sqrt_ref_time_days,
            args.reset_day,
            step_mask=step_mask,
            initial_thickness_nm=initial_thickness_nm,
            fit_initial_thickness=fit_initial_thickness,
            initial_thickness_starts=initial_thickness_starts,
            max_initial_thickness_nm=max_initial_thickness_nm,
        )
        entropy_fit = fit_global_entropy_sqrt(
            observations,
            responses[mode],
            d_grid,
            sqrt_fit["parameter"],
            sqrt_fit["coarse_local_minimum_scales"],
            args.max_scale,
            args.sqrt_ref_thickness_nm,
            args.sqrt_ref_time_days,
            args.reset_day,
            args.entropy_weight,
            args.smoothness_weight,
            args.max_node_multiplier,
            step_mask=step_mask,
            initial_thickness_nm=initial_thickness_nm,
            fit_initial_thickness=fit_initial_thickness,
            initial_thickness_starts=initial_thickness_starts,
            max_initial_thickness_nm=max_initial_thickness_nm,
        )
        fits[mode] = {
            "soc_sqrt": sqrt_fit,
            "entropy_sqrt": entropy_fit,
        }
    selected_trajectory = "entropy_sqrt"
    selected_mode = min(
        MODES,
        key=lambda mode: fits[mode][selected_trajectory]["between_sse"],
    )
    fit_path = args.output_dir / FIT_PATH.name
    prediction_path = args.output_dir / PREDICTION_PATH.name
    node_path = args.output_dir / NODE_PATH.name
    report_path = args.output_dir / REPORT_PATH.name
    plot_path = args.output_dir / PLOT_PATH.name
    write_fit_csv(fit_path, fits, selected_mode, selected_trajectory)
    write_node_csv(node_path, fits, selected_mode)
    write_prediction_csv(
        prediction_path,
        observations,
        fits[selected_mode][selected_trajectory],
        selected_mode,
        step_mask=step_mask,
    )
    write_report(
        report_path,
        args.statistics_path,
        args.model_path,
        observations,
        fits,
        selected_mode,
        selected_trajectory,
        args.max_scale,
        args.max_node_multiplier,
        step_time=args.step_time,
    )
    plot_fit(
        plot_path,
        observations,
        fits[selected_mode][selected_trajectory],
        fits[selected_mode]["soc_sqrt"],
        selected_mode,
        step_day=step_day,
    )

    print(
        f"Compared {fits[selected_mode][selected_trajectory]['pixel_count']:,} valid pixel pairs "
        f"across {len(observations)} exposures"
    )
    for mode in MODES:
        for trajectory in GLOBAL_TRAJECTORIES:
            result = fits[mode][trajectory]
            print(
                f"{mode}/{trajectory}: scale={result['parameter']:.8g}; "
                f"d0={result.get('initial_thickness_nm', 0.0):.8g} nm; "
                f"accumulated={np.max(result.get('incremental_thickness_nm', result['thickness_nm'])):.8g} nm; "
                f"rate={result['mean_rate_nm_day']:.8g} nm/day; "
                f"all-pixel RMS={result['pixel_rms_log']:.8g}; "
                f"explainable SSE removed="
                f"{100.0 * (1.0 - result['between_sse'] / result['null_between_sse']):.3f}%"
            )
    print(
        f"Selected model: {selected_mode}/{selected_trajectory}; "
        f"weights entropy={args.entropy_weight:g}, smoothness={args.smoothness_weight:g}"
    )
    if step_mask is not None:
        selected = fits[selected_mode][selected_trajectory]
        print(
            f"Profiled step: {selected['step_percent']:.8g}% throughput "
            f"({selected['step_log_offset']:.8g} log units) after {args.step_time}"
        )
    print(f"Wrote {fit_path}")
    print(f"Wrote {node_path}")
    print(f"Wrote {prediction_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
