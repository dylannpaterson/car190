"""Build per-tile gain and three-known-ACM correction tables."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from analyze_wfi01_all_pixel_soc_model import DEFAULT_STEP_TIME, parse_mast_time
from plot_wfi_led_postday1_sinusoid import (
    evaluate_components,
    fit_sinusoid_plus_two_acm,
    fit_sinusoid_plus_three_acm,
)


CORRECTION_FIELDS = (
    "detector",
    "tile_id",
    "group",
    "days",
    "log_gain_correction",
    "acm_step_log_offset",
    "pre_reset_acm_step_log_offset",
    "additional_acm_step_log_offset",
    "late_acm_step_log_offset",
)
SUMMARY_FIELDS = (
    "detector",
    "tile_id",
    "reference_led",
    "minimum_day_exclusive",
    "reference_period_days",
    "reference_rmse_log",
    "reference_r2",
    "post_reset_period_days",
    "post_reset_rmse_log",
    "post_reset_r2",
    "acm_step_log_offset",
    "pre_reset_acm_step_log_offset",
    "additional_acm_step_log_offset",
    "late_acm_step_log_offset",
    "acm_step_mad_log",
    "pre_reset_acm_step_mad_log",
    "additional_acm_step_mad_log",
    "late_acm_step_mad_log",
)
KNOWN_FIRST_ACM_REPEAT = 14
KNOWN_PRE_RESET_ACM_REPEAT = 35
KNOWN_POST_RESET_ACM_REPEAT = 39


def detector_name(path):
    return path.name.split("_tile_log_statistics_", 1)[0].upper()


def load_statistics(
    path,
    additional_acm_step_repeat=None,
    pre_reset_acm_step_repeat=None,
):
    """Load one SCA's mean ratios once, grouped by tile and LED."""
    event_time = parse_mast_time(DEFAULT_STEP_TIME)
    rows_by_tile = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            count = float(row["n_pair_valid"])
            if count <= 0.0:
                continue
            tile_id = int(row["tile_id"])
            rows_by_tile.setdefault(tile_id, []).append(row)
    if not rows_by_tile:
        raise ValueError(f"No valid tile rows found in {path}")

    tiles = {}
    for tile_id, rows in rows_by_tile.items():
        rows = sorted(rows, key=lambda row: int(row["group"]))
        groups = [int(row["group"]) for row in rows]
        if groups != sorted(set(groups)):
            raise ValueError(f"Duplicate groups for tile {tile_id} in {path}")
        by_led = {}
        for row in rows:
            led = int(row["led"])
            by_led.setdefault(led, []).append(row)
        led_data = {}
        for led in range(1, 7):
            led_rows = sorted(by_led.get(led, []), key=lambda row: float(row["days"]))
            if len(led_rows) < 6:
                raise ValueError(f"Too few LED {led} rows for tile {tile_id}")
            days = np.asarray([float(row["days"]) for row in led_rows], dtype=float)
            values = np.asarray(
                [
                    float(row["sum_log_change"])
                    / float(row["n_pair_valid"])
                    for row in led_rows
                ],
                dtype=float,
            )
            after_event = np.asarray(
                [
                    parse_mast_time(row["exposure_start_time"]) >= event_time
                    for row in led_rows
                ],
                dtype=float,
            )
            first_step = after_event - after_event[0]
            late_repeat = max(int(row["repeat"]) for row in led_rows)
            late_step = np.asarray(
                [float(int(row["repeat"]) == late_repeat) for row in led_rows],
                dtype=float,
            )
            late_step -= late_step[0]
            if additional_acm_step_repeat is None:
                additional_step = np.zeros(len(led_rows), dtype=float)
            else:
                additional_step = np.asarray(
                    [
                        float(int(row["repeat"]) >= additional_acm_step_repeat)
                        for row in led_rows
                    ],
                    dtype=float,
                )
                if not np.any(additional_step == 0.0) or not np.any(
                    additional_step == 1.0
                ):
                    raise ValueError(
                        "Additional ACM step repeat must have observations on both sides"
                    )
                additional_step -= additional_step[0]
            if pre_reset_acm_step_repeat is None:
                pre_reset_step = np.zeros(len(led_rows), dtype=float)
            else:
                pre_reset_step = np.asarray(
                    [
                        float(int(row["repeat"]) >= pre_reset_acm_step_repeat)
                        for row in led_rows
                    ],
                    dtype=float,
                )
                if not np.any(pre_reset_step == 0.0) or not np.any(
                    pre_reset_step == 1.0
                ):
                    raise ValueError(
                        "Pre-reset ACM step repeat must have observations on both sides"
                    )
                pre_reset_step -= pre_reset_step[0]
            led_data[led] = {
                "days": days,
                "values": values,
                "repeat": np.asarray([int(row["repeat"]) for row in led_rows]),
                "first_step": first_step,
                "additional_step": additional_step,
                "pre_reset_step": pre_reset_step,
                "late_step": late_step,
            }
        tiles[tile_id] = {"rows": rows, "led": led_data}
    return tiles


def fit_post_reset_shared_acm_step(led_data, reset_day, acm_step_repeat):
    """Fit independent post-reset sinusoids with one shared ACM step."""
    series = []
    initial_periods = []
    bounds = []
    for led in range(1, 7):
        current = led_data[led]
        mask = current["days"] >= reset_day
        if np.count_nonzero(mask) < 5:
            raise ValueError(f"LED {led} has too few post-reset observations")
        days = current["days"][mask]
        values = current["values"][mask]
        repeats = current["repeat"][mask]
        step = (repeats >= acm_step_repeat).astype(float)
        if not np.any(step == 0.0) or not np.any(step == 1.0):
            raise ValueError(
                f"LED {led} does not have observations on both sides of "
                f"repeat {acm_step_repeat}"
            )
        initial_fit = fit_sinusoid_plus_two_acm(
            days,
            values,
            np.zeros_like(step),
            step,
        )
        span = float(np.ptp(days))
        minimum_period = max(2.0 * float(np.median(np.diff(days))), 0.05)
        maximum_period = 100.0 * span
        series.append(
            {
                "days": days,
                "values": values,
                "time": days - days[0],
                "step": step,
            }
        )
        initial_periods.append(initial_fit["period_days"])
        bounds.append((np.log(minimum_period), np.log(maximum_period)))

    def design_matrix(log_periods):
        n_rows = sum(item["days"].size for item in series)
        matrix = np.zeros((n_rows, 19), dtype=float)
        offset = 0
        for led_index, (item, log_period) in enumerate(zip(series, log_periods)):
            period = float(np.exp(log_period))
            angular_frequency = 2.0 * np.pi / period
            length = item["days"].size
            columns = slice(3 * led_index, 3 * led_index + 3)
            matrix[offset : offset + length, columns] = np.column_stack(
                (
                    np.ones(length),
                    np.sin(angular_frequency * item["time"]),
                    np.cos(angular_frequency * item["time"]),
                )
            )
            matrix[offset : offset + length, -1] = item["step"]
            offset += length
        return matrix

    values = np.concatenate([item["values"] for item in series])

    def solve(log_periods):
        matrix = design_matrix(log_periods)
        coefficients, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
        residual = values - matrix @ coefficients
        return float(np.dot(residual, residual)), coefficients, matrix @ coefficients

    initial = np.log(np.asarray(initial_periods, dtype=float))
    result = minimize(
        lambda log_periods: solve(log_periods)[0],
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 2000},
    )
    if not result.success:
        raise ValueError(f"Post-reset shared ACM fit failed: {result.message}")

    _, coefficients, fitted_values = solve(result.x)
    fits = {}
    offset = 0
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    for led_index, item in enumerate(series):
        length = item["days"].size
        observed = item["values"]
        fitted = fitted_values[offset : offset + length]
        residual = observed - fitted
        led_coefficients = coefficients[3 * led_index : 3 * led_index + 3]
        led_sse = float(np.dot(residual, residual))
        led_tss = float(np.sum((observed - np.mean(observed)) ** 2))
        fits[led_index + 1] = {
            "period_days": float(np.exp(result.x[led_index])),
            "offset_log_change": float(led_coefficients[0]),
            "amplitude_log_change": float(
                np.hypot(led_coefficients[1], led_coefficients[2])
            ),
            "phase_rad": float(np.arctan2(led_coefficients[2], led_coefficients[1])),
            "acm_step_log_change": float(coefficients[-1]),
            "late_acm_step_log_change": 0.0,
            "sse_log_change_squared": led_sse,
            "rmse_log_change": float(np.sqrt(np.mean(residual * residual))),
            "r2": float(1.0 - led_sse / led_tss) if led_tss > 0.0 else np.nan,
            "coefficients": np.asarray(
                [
                    led_coefficients[0],
                    led_coefficients[1],
                    led_coefficients[2],
                    coefficients[-1],
                ]
            ),
            "fitted": fitted,
            "residual": residual,
        }
        offset += length

    total_residual = values - fitted_values
    return {
        "shared_step": float(coefficients[-1]),
        "fits": fits,
        "sse_log_change_squared": float(np.dot(total_residual, total_residual)),
        "rmse_log_change": float(np.sqrt(np.mean(total_residual * total_residual))),
        "r2": (
            float(1.0 - np.dot(total_residual, total_residual) / total_sum_squares)
            if total_sum_squares > 0.0
            else np.nan
        ),
    }


def fit_tile(
    tile,
    reference_led,
    minimum_day,
    additional_acm_step_repeat=None,
    ice_reset_day=None,
    pre_reset_acm_step_repeat=None,
):
    known_three_step_fit = (
        pre_reset_acm_step_repeat is not None
        and additional_acm_step_repeat is not None
    )
    if known_three_step_fit and (
        pre_reset_acm_step_repeat != KNOWN_PRE_RESET_ACM_REPEAT
        or additional_acm_step_repeat != KNOWN_POST_RESET_ACM_REPEAT
    ):
        raise ValueError(
            "The known three-step WFI01 fit requires ACM repeats 14, 35, and 39"
        )
    if known_three_step_fit and ice_reset_day is None:
        raise ValueError(
            "The known three-step WFI01 fit requires --ice-reset-day so repeat "
            "39 is fit against independent post-reset LED sinusoids"
        )
    if (
        pre_reset_acm_step_repeat is not None
        and additional_acm_step_repeat is None
        and ice_reset_day is None
    ):
        raise ValueError(
            "A pre-reset ACM step requires --ice-reset-day so it is fit only "
            "on the pre-reset segment"
        )
    reference = tile["led"][reference_led]
    reference_days = reference["days"]
    reference_values = reference["values"]
    first_step_repeats = reference["repeat"][reference["first_step"] > 0.0]
    if known_three_step_fit and (
        not len(first_step_repeats)
        or int(np.min(first_step_repeats)) != KNOWN_FIRST_ACM_REPEAT
    ):
        raise ValueError("The known first ACM step is not at repeat 14")
    early = reference_days <= minimum_day
    if not np.any(early):
        raise ValueError("Reference LED must contain data on both sides of minimum_day")
    post_reset_shared_fit = None
    if known_three_step_fit and ice_reset_day is None:
        reference_fit_mask = (reference_days > minimum_day) & (
            reference["repeat"] <= pre_reset_acm_step_repeat
        )
        if not np.any(reference_fit_mask):
            raise ValueError("No sinusoid-fit observations through the epoch-35 step")
        reference_fit = fit_sinusoid_plus_two_acm(
            reference_days[reference_fit_mask],
            reference_values[reference_fit_mask],
            reference["first_step"][reference_fit_mask],
            reference["pre_reset_step"][reference_fit_mask],
        )
        _, reference_model = evaluate_components(
            reference_days - reference_days[reference_fit_mask][0],
            reference_fit,
            reference["first_step"],
            reference["pre_reset_step"],
        )
        reference_post_mask = reference["additional_step"] > 0.0
        if not np.any(reference_post_mask):
            raise ValueError("No observations follow the epoch-39 ACM step")
        reference_fit["additional_acm_step_log_change"] = float(
            np.mean(reference_values[reference_post_mask] - reference_model[reference_post_mask])
        )
        reference_fit["late_acm_step_log_change"] = 0.0
        post_reset_fit = None
    elif ice_reset_day is None:
        fit_mask = reference_days > minimum_day
        if not np.any(fit_mask):
            raise ValueError(
                "Reference LED must contain post-cutoff observations"
            )
        if additional_acm_step_repeat is None:
            reference_fit = fit_sinusoid_plus_two_acm(
                reference_days[fit_mask],
                reference_values[fit_mask],
                reference["first_step"][fit_mask],
                reference["late_step"][fit_mask],
            )
        else:
            reference_fit = fit_sinusoid_plus_three_acm(
                reference_days[fit_mask],
                reference_values[fit_mask],
                reference["first_step"][fit_mask],
                reference["additional_step"][fit_mask],
                reference["late_step"][fit_mask],
            )
        _, reference_model = evaluate_components(
            reference_days - reference_days[fit_mask][0],
            reference_fit,
            reference["first_step"],
            reference["late_step"],
            reference["additional_step"]
            if additional_acm_step_repeat is not None
            else None,
        )
        post_reset_fit = None
    else:
        pre_reset_fit_mask = (reference_days > minimum_day) & (
            reference_days < ice_reset_day
        )
        post_reset_fit_mask = reference_days >= ice_reset_day
        if not np.any(pre_reset_fit_mask):
            raise ValueError("Reset day leaves no post-cutoff pre-reset observations")
        if not np.any(post_reset_fit_mask):
            raise ValueError("Reset day leaves no post-reset observations")
        zero_steps = np.zeros_like(reference_days)
        reference_fit = fit_sinusoid_plus_three_acm(
            reference_days[pre_reset_fit_mask],
            reference_values[pre_reset_fit_mask],
            reference["first_step"][pre_reset_fit_mask],
            reference["pre_reset_step"][pre_reset_fit_mask],
            zero_steps[pre_reset_fit_mask],
        )
        _, reference_model = evaluate_components(
            reference_days - reference_days[pre_reset_fit_mask][0],
            reference_fit,
            reference["first_step"],
            zero_steps,
            reference["pre_reset_step"],
        )
        if known_three_step_fit:
            post_reset_shared_fit = fit_post_reset_shared_acm_step(
                tile["led"], ice_reset_day, additional_acm_step_repeat
            )
            post_reset_fit = post_reset_shared_fit["fits"][reference_led]
        else:
            post_reset_fit = fit_sinusoid_plus_three_acm(
                reference_days[post_reset_fit_mask],
                reference_values[post_reset_fit_mask],
                zero_steps[post_reset_fit_mask],
                reference["additional_step"][post_reset_fit_mask],
                reference["late_step"][post_reset_fit_mask],
            )
    gain_days = reference_days[early]
    gain_values = reference_model[early] - reference_values[early]

    acm_steps = []
    pre_reset_acm_steps = []
    additional_acm_steps = []
    late_acm_steps = []
    for led in range(1, 7):
        current = tile["led"][led]
        corrected_values = current["values"].copy()
        early_current = current["days"] <= minimum_day
        corrected_values[early_current] += np.interp(
            current["days"][early_current], gain_days, gain_values
        )
        if known_three_step_fit and ice_reset_day is None:
            current_fit_mask = (current["days"] > minimum_day) & (
                current["repeat"] <= pre_reset_acm_step_repeat
            )
            if not np.any(current_fit_mask):
                raise ValueError("No current-LED sinusoid-fit observations through epoch 35")
            corrected_fit = fit_sinusoid_plus_two_acm(
                current["days"][current_fit_mask],
                corrected_values[current_fit_mask],
                current["first_step"][current_fit_mask],
                current["pre_reset_step"][current_fit_mask],
            )
            _, current_model = evaluate_components(
                current["days"] - current["days"][current_fit_mask][0],
                corrected_fit,
                current["first_step"],
                current["pre_reset_step"],
            )
            post_mask = current["additional_step"] > 0.0
            if not np.any(post_mask):
                raise ValueError("No current-LED observations follow epoch 39")
            pre_reset_step = corrected_fit["late_acm_step_log_change"]
            corrected_fit["additional_acm_step_log_change"] = float(
                np.mean(corrected_values[post_mask] - current_model[post_mask])
            )
            corrected_fit["late_acm_step_log_change"] = 0.0
            acm_steps.append(corrected_fit["acm_step_log_change"])
            pre_reset_acm_steps.append(pre_reset_step)
            additional_acm_steps.append(
                corrected_fit["additional_acm_step_log_change"]
            )
            late_acm_steps.append(0.0)
        elif ice_reset_day is None:
            fit_mask = current["days"] > minimum_day
            if additional_acm_step_repeat is None:
                corrected_fit = fit_sinusoid_plus_two_acm(
                    current["days"][fit_mask],
                    corrected_values[fit_mask],
                    current["first_step"][fit_mask],
                    current["late_step"][fit_mask],
                )
            else:
                corrected_fit = fit_sinusoid_plus_three_acm(
                    current["days"][fit_mask],
                    corrected_values[fit_mask],
                    current["first_step"][fit_mask],
                    current["additional_step"][fit_mask],
                    current["late_step"][fit_mask],
                )
            acm_steps.append(corrected_fit["acm_step_log_change"])
            pre_reset_acm_steps.append(0.0)
            additional_acm_steps.append(
                corrected_fit.get("additional_acm_step_log_change", 0.0)
            )
            late_acm_steps.append(corrected_fit["late_acm_step_log_change"])
        else:
            pre_reset_fit_mask = (current["days"] > minimum_day) & (
                current["days"] < ice_reset_day
            )
            post_reset_fit_mask = current["days"] >= ice_reset_day
            zero_steps = np.zeros_like(current["days"])
            pre_reset_fit = fit_sinusoid_plus_three_acm(
                current["days"][pre_reset_fit_mask],
                corrected_values[pre_reset_fit_mask],
                current["first_step"][pre_reset_fit_mask],
                current["pre_reset_step"][pre_reset_fit_mask],
                zero_steps[pre_reset_fit_mask],
            )
            acm_steps.append(pre_reset_fit["acm_step_log_change"])
            pre_reset_acm_steps.append(pre_reset_fit["additional_acm_step_log_change"])
            if known_three_step_fit:
                additional_acm_steps.append(post_reset_shared_fit["shared_step"])
                late_acm_steps.append(0.0)
            else:
                post_reset_fit = fit_sinusoid_plus_three_acm(
                    current["days"][post_reset_fit_mask],
                    corrected_values[post_reset_fit_mask],
                    zero_steps[post_reset_fit_mask],
                    current["additional_step"][post_reset_fit_mask],
                    current["late_step"][post_reset_fit_mask],
                )
                additional_acm_steps.append(
                    post_reset_fit["additional_acm_step_log_change"]
                )
                late_acm_steps.append(post_reset_fit["late_acm_step_log_change"])

    acm_steps = np.asarray(acm_steps, dtype=float)
    pre_reset_acm_steps = np.asarray(pre_reset_acm_steps, dtype=float)
    additional_acm_steps = np.asarray(additional_acm_steps, dtype=float)
    late_acm_steps = np.asarray(late_acm_steps, dtype=float)
    return {
        "reference_fit": reference_fit,
        "post_reset_fit": post_reset_fit,
        "gain_days": gain_days,
        "gain_values": gain_values,
        "acm_step": float(np.median(acm_steps)),
        "pre_reset_acm_step": float(np.median(pre_reset_acm_steps)),
        "additional_acm_step": float(np.median(additional_acm_steps)),
        "late_acm_step": float(np.median(late_acm_steps)),
        "acm_step_mad": float(np.median(np.abs(acm_steps - np.median(acm_steps)))),
        "pre_reset_acm_step_mad": float(
            np.median(
                np.abs(
                    pre_reset_acm_steps - np.median(pre_reset_acm_steps)
                )
            )
        ),
        "additional_acm_step_mad": float(
            np.median(
                np.abs(additional_acm_steps - np.median(additional_acm_steps))
            )
        ),
        "late_acm_step_mad": float(
            np.median(np.abs(late_acm_steps - np.median(late_acm_steps)))
        ),
    }


def build_corrections(
    path,
    reference_led,
    minimum_day,
    additional_acm_step_repeat=None,
    ice_reset_day=None,
    pre_reset_acm_step_repeat=None,
):
    detector = detector_name(path)
    tiles = load_statistics(
        path,
        additional_acm_step_repeat,
        pre_reset_acm_step_repeat,
    )
    correction_rows = []
    summary_rows = []
    for tile_id in sorted(tiles):
        tile = tiles[tile_id]
        fit = fit_tile(
            tile,
            reference_led,
            minimum_day,
            additional_acm_step_repeat,
            ice_reset_day,
            pre_reset_acm_step_repeat,
        )
        gain_days = fit["gain_days"]
        gain_values = fit["gain_values"]
        for row in tile["rows"]:
            day = float(row["days"])
            gain = (
                float(np.interp(day, gain_days, gain_values))
                if day <= minimum_day
                else 0.0
            )
            correction_rows.append(
                {
                    "detector": detector,
                    "tile_id": tile_id,
                    "group": int(row["group"]),
                    "days": day,
                    "log_gain_correction": gain,
                    "acm_step_log_offset": fit["acm_step"],
                    "pre_reset_acm_step_log_offset": fit["pre_reset_acm_step"],
                    "additional_acm_step_log_offset": fit["additional_acm_step"],
                    "late_acm_step_log_offset": fit["late_acm_step"],
                }
            )
        reference_fit = fit["reference_fit"]
        post_reset_fit = fit["post_reset_fit"]
        summary_rows.append(
            {
                "detector": detector,
                "tile_id": tile_id,
                "reference_led": reference_led,
                "minimum_day_exclusive": minimum_day,
                "reference_period_days": reference_fit["period_days"],
                "reference_rmse_log": reference_fit["rmse_log_change"],
                "reference_r2": reference_fit["r2"],
                "post_reset_period_days": (
                    np.nan
                    if post_reset_fit is None
                    else post_reset_fit["period_days"]
                ),
                "post_reset_rmse_log": (
                    np.nan
                    if post_reset_fit is None
                    else post_reset_fit["rmse_log_change"]
                ),
                "post_reset_r2": (
                    np.nan
                    if post_reset_fit is None
                    else post_reset_fit["r2"]
                ),
                "acm_step_log_offset": fit["acm_step"],
                "pre_reset_acm_step_log_offset": fit["pre_reset_acm_step"],
                "additional_acm_step_log_offset": fit["additional_acm_step"],
                "late_acm_step_log_offset": fit["late_acm_step"],
                "acm_step_mad_log": fit["acm_step_mad"],
                "pre_reset_acm_step_mad_log": fit["pre_reset_acm_step_mad"],
                "additional_acm_step_mad_log": fit["additional_acm_step_mad"],
                "late_acm_step_mad_log": fit["late_acm_step_mad"],
            }
        )
    return correction_rows, summary_rows


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=Path("wfi_tile_log_statistics_512"),
    )
    parser.add_argument("--reference-led", type=int, default=3)
    parser.add_argument("--minimum-day", type=float, default=1.0)
    parser.add_argument(
        "--additional-acm-step-repeat",
        type=int,
        default=None,
        help=(
            "First repeat affected by an optional additional ACM step; use 39 "
            "for the WFI01 post-reset event."
        ),
    )
    parser.add_argument(
        "--pre-reset-acm-step-repeat",
        type=int,
        default=None,
        help=(
            "First repeat affected by an optional pre-reset ACM step; use 35 "
            "for the WFI01 day-11 event."
        ),
    )
    parser.add_argument(
        "--ice-reset-day",
        type=float,
        default=None,
        help=(
            "First post-reset observation day. When supplied, the sinusoid used "
            "for the gain/first-step fit is restricted to pre-reset data and a "
            "separate post-reset sinusoid is fit."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("refit_corrections_tile/wfi_all_tile_fixed_led3_gain_acm_corrections.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("refit_corrections_tile/wfi_all_tile_correction_fit_summary.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.reference_led <= 6:
        raise ValueError("Reference LED must be between 1 and 6")
    if args.minimum_day < 0.0 or not np.isfinite(args.minimum_day):
        raise ValueError("Minimum day must be finite and non-negative")
    if args.additional_acm_step_repeat is not None and args.additional_acm_step_repeat < 0:
        raise ValueError("Additional ACM step repeat must be non-negative")
    if args.pre_reset_acm_step_repeat is not None and args.pre_reset_acm_step_repeat < 0:
        raise ValueError("Pre-reset ACM step repeat must be non-negative")
    if args.ice_reset_day is not None and not np.isfinite(args.ice_reset_day):
        raise ValueError("Ice reset day must be finite")
    if (
        args.additional_acm_step_repeat is not None
        and args.pre_reset_acm_step_repeat is None
        and args.ice_reset_day is None
    ):
        raise ValueError(
            "An additional ACM step requires --ice-reset-day so the sinusoid is "
            "not fit across the reset"
        )
    if (
        args.pre_reset_acm_step_repeat is not None
        and args.additional_acm_step_repeat is None
        and args.ice_reset_day is None
    ):
        raise ValueError(
            "A pre-reset ACM step requires --ice-reset-day so it is fit only "
            "on the pre-reset segment"
        )
    statistics_paths = sorted(args.statistics_dir.glob("wfi*_tile_log_statistics_*.csv"))
    if not statistics_paths:
        raise ValueError(f"No SCA statistics CSVs found in {args.statistics_dir}")
    correction_rows = []
    summary_rows = []
    for path in statistics_paths:
        current_corrections, current_summary = build_corrections(
            path,
            args.reference_led,
            args.minimum_day,
            args.additional_acm_step_repeat,
            args.ice_reset_day,
            args.pre_reset_acm_step_repeat,
        )
        correction_rows.extend(current_corrections)
        summary_rows.extend(current_summary)
        print(f"processed {detector_name(path)}: {len(current_summary)} tiles")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, CORRECTION_FIELDS, correction_rows)
    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    print(f"wrote: {args.output} ({len(correction_rows)} rows)")
    print(f"wrote: {args.summary} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
