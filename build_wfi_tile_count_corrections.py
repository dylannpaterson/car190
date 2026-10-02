"""Fit gain/ACM corrections to tile count means and rebuild log ratios."""

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_wfi01_all_pixel_soc_model import DEFAULT_STEP_TIME, parse_mast_time
from plot_wfi_led_postday1_sinusoid import (
    evaluate_components,
    fit_sinusoid_plus_three_acm,
    fit_sinusoid_plus_two_acm,
)


CORRECTION_FIELDS = (
    "detector",
    "tile_id",
    "group",
    "days",
    "correction_space",
    "count_log_correction",
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
    "reference_rmse_counts",
    "reference_r2",
    "post_reset_period_days",
    "post_reset_rmse_counts",
    "post_reset_r2",
    "reference_acm_step_counts",
    "reference_pre_reset_acm_step_counts",
    "reference_additional_acm_step_counts",
    "reference_late_acm_step_counts",
)


def detector_name(path):
    return path.name.split("_tile_log_statistics_", 1)[0].upper()


def _fit(values, days, first_step, additional_step, late_step, fit_mask):
    if not np.any(fit_mask):
        raise ValueError("Count data has no observations in the requested fit segment")
    if additional_step is None:
        fit = fit_sinusoid_plus_two_acm(
            days[fit_mask],
            values[fit_mask],
            first_step[fit_mask],
            late_step[fit_mask],
        )
    else:
        fit = fit_sinusoid_plus_three_acm(
            days[fit_mask],
            values[fit_mask],
            first_step[fit_mask],
            additional_step[fit_mask],
            late_step[fit_mask],
        )
    return fit


def _components(days, fit, first_step, late_step, additional_step, origin):
    return evaluate_components(
        days - origin,
        fit,
        first_step,
        late_step,
        additional_step,
    )


def load_statistics(path, additional_acm_step_repeat, pre_reset_acm_step_repeat):
    event_time = parse_mast_time(DEFAULT_STEP_TIME)
    rows_by_tile = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sum_current_counts", "n_pair_valid"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Count-space correction fitting requires fields: "
                f"{sorted(missing)}"
            )
        for row in reader:
            if float(row["n_pair_valid"]) > 0.0:
                rows_by_tile.setdefault(int(row["tile_id"]), []).append(row)
    if not rows_by_tile:
        raise ValueError(f"No valid tile rows found in {path}")

    tiles = {}
    for tile_id, rows in rows_by_tile.items():
        rows = sorted(rows, key=lambda row: int(row["group"]))
        by_led = {}
        for row in rows:
            by_led.setdefault(int(row["led"]), []).append(row)
        led_data = {}
        for led in range(1, 7):
            led_rows = sorted(by_led[led], key=lambda row: float(row["days"]))
            days = np.asarray([float(row["days"]) for row in led_rows], dtype=float)
            values = np.asarray(
                [
                    float(row["sum_current_counts"])
                    / float(row["n_pair_valid"])
                    for row in led_rows
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"Non-positive count means for LED {led}, tile {tile_id}")
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
                additional_step = None
            else:
                additional_step = np.asarray(
                    [
                        float(int(row["repeat"]) >= additional_acm_step_repeat)
                        for row in led_rows
                    ],
                    dtype=float,
                )
                additional_step -= additional_step[0]
            if pre_reset_acm_step_repeat is None:
                pre_reset_step = None
            else:
                pre_reset_step = np.asarray(
                    [
                        float(int(row["repeat"]) >= pre_reset_acm_step_repeat)
                        for row in led_rows
                    ],
                    dtype=float,
                )
                pre_reset_step -= pre_reset_step[0]
            led_data[led] = {
                "rows": led_rows,
                "days": days,
                "values": values,
                "first_step": first_step,
                "additional_step": additional_step,
                "pre_reset_step": pre_reset_step,
                "late_step": late_step,
                "repeat": np.asarray([int(row["repeat"]) for row in led_rows]),
            }
        tiles[tile_id] = {"rows": rows, "led": led_data}
    return tiles


def _fit_led_count_curve(led, minimum_day, ice_reset_day):
    days = led["days"]
    values = led["values"]
    zero = np.zeros_like(days)
    if ice_reset_day is None:
        fit_mask = days > minimum_day
        fit = _fit(
            values,
            days,
            led["first_step"],
            led["additional_step"],
            led["late_step"],
            fit_mask,
        )
        sinusoid, full = _components(
            days,
            fit,
            led["first_step"],
            led["late_step"],
            led["additional_step"],
            days[fit_mask][0],
        )
        return fit, None, sinusoid, full, None

    pre_mask = (days > minimum_day) & (days < ice_reset_day)
    post_mask = days >= ice_reset_day
    pre_fit = fit_sinusoid_plus_three_acm(
        days[pre_mask],
        values[pre_mask],
        led["first_step"][pre_mask],
        led["pre_reset_step"][pre_mask],
        zero[pre_mask],
    )
    post_fit = _fit(
        values,
        days,
        zero,
        led["additional_step"],
        led["late_step"],
        post_mask,
    )
    pre_sinusoid, pre_full = _components(
        days,
        pre_fit,
        led["first_step"],
        zero,
        led["pre_reset_step"],
        days[pre_mask][0],
    )
    post_sinusoid, post_full = _components(
        days,
        post_fit,
        zero,
        led["late_step"],
        led["additional_step"],
        days[post_mask][0],
    )
    sinusoid = np.where(post_mask, post_sinusoid, pre_sinusoid)
    full = np.where(post_mask, post_full, pre_full)
    return pre_fit, post_fit, sinusoid, full, post_mask


def fit_tile(tile, reference_led, minimum_day, additional_acm_step_repeat,
             ice_reset_day, pre_reset_acm_step_repeat):
    reference = tile["led"][reference_led]
    reference_fit, reference_post_fit, reference_sinusoid, reference_full, post_mask = (
        _fit_led_count_curve(reference, minimum_day, ice_reset_day)
    )
    early = reference["days"] <= minimum_day
    gain_factor_days = reference["days"][early]
    gain_factors = reference_sinusoid[early] / reference["values"][early]
    if np.any(~np.isfinite(gain_factors)) or np.any(gain_factors <= 0.0):
        raise ValueError("Reference count fit produced invalid early gain factors")

    step_factors_by_repeat = {}
    led_fits = {}
    for led in range(1, 7):
        current = tile["led"][led]
        gain_factor = np.ones_like(current["days"])
        early_current = current["days"] <= minimum_day
        gain_factor[early_current] = np.interp(
            current["days"][early_current], gain_factor_days, gain_factors
        )
        corrected_values = current["values"] * gain_factor
        corrected_led = dict(current)
        corrected_led["values"] = corrected_values
        pre_fit, post_fit, sinusoid, full, led_post_mask = _fit_led_count_curve(
            corrected_led, minimum_day, ice_reset_day
        )
        step_factor = sinusoid / full
        if np.any(~np.isfinite(step_factor)) or np.any(step_factor <= 0.0):
            raise ValueError(f"LED {led} count fit produced invalid ACM factors")
        for repeat, factor in zip(current["repeat"], step_factor):
            step_factors_by_repeat.setdefault(int(repeat), []).append(float(factor))
        led_fits[led] = {
            "gain_factor": gain_factor,
            "step_factor": step_factor,
            "pre_fit": pre_fit,
            "post_fit": post_fit,
        }

    shared_step_factor = {
        repeat: float(np.exp(np.median(np.log(factors))))
        for repeat, factors in step_factors_by_repeat.items()
    }
    correction_rows = []
    rows_by_key = {}
    for led in range(1, 7):
        current = tile["led"][led]
        for index, row in enumerate(current["rows"]):
            factor = led_fits[led]["gain_factor"][index] * shared_step_factor[
                int(row["repeat"])
            ]
            correction_rows.append(
                {
                    "detector": "",
                    "tile_id": int(row["tile_id"]),
                    "group": int(row["group"]),
                    "days": float(row["days"]),
                    "correction_space": "counts",
                    "count_log_correction": float(np.log(factor)),
                    "log_gain_correction": float(
                        np.log(led_fits[led]["gain_factor"][index])
                    ),
                    "acm_step_log_offset": 0.0,
                    "pre_reset_acm_step_log_offset": 0.0,
                    "additional_acm_step_log_offset": 0.0,
                    "late_acm_step_log_offset": 0.0,
                }
            )
            rows_by_key[(int(row["tile_id"]), int(row["group"]))] = row

    reference_fit = reference_fit
    summary = {
        "tile_id": int(next(iter(rows_by_key.values()))["tile_id"]),
        "reference_period_days": reference_fit["period_days"],
        "reference_rmse_counts": reference_fit["rmse_log_change"],
        "reference_r2": reference_fit["r2"],
        "post_reset_period_days": (
            np.nan if reference_post_fit is None else reference_post_fit["period_days"]
        ),
        "post_reset_rmse_counts": (
            np.nan if reference_post_fit is None else reference_post_fit["rmse_log_change"]
        ),
        "post_reset_r2": (
            np.nan if reference_post_fit is None else reference_post_fit["r2"]
        ),
        "reference_acm_step_counts": reference_fit["acm_step_log_change"],
        "reference_pre_reset_acm_step_counts": 0.0,
        "reference_additional_acm_step_counts": (
            0.0 if reference_post_fit is None else reference_post_fit["additional_acm_step_log_change"]
        ),
        "reference_late_acm_step_counts": reference_fit["late_acm_step_log_change"],
    }
    if ice_reset_day is not None:
        summary["reference_pre_reset_acm_step_counts"] = reference_fit[
            "additional_acm_step_log_change"
        ]
    return correction_rows, summary


def build_corrections(path, reference_led, minimum_day, additional_acm_step_repeat,
                      ice_reset_day, pre_reset_acm_step_repeat):
    detector = detector_name(path)
    tiles = load_statistics(
        path, additional_acm_step_repeat, pre_reset_acm_step_repeat
    )
    correction_rows = []
    summary_rows = []
    for tile_id in sorted(tiles):
        rows, summary = fit_tile(
            tiles[tile_id],
            reference_led,
            minimum_day,
            additional_acm_step_repeat,
            ice_reset_day,
            pre_reset_acm_step_repeat,
        )
        for row in rows:
            row["detector"] = detector
        correction_rows.extend(rows)
        summary_rows.append(
            {
                "detector": detector,
                "tile_id": summary["tile_id"],
                "reference_led": reference_led,
                "minimum_day_exclusive": minimum_day,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "tile_id"
                },
            }
        )
    return correction_rows, summary_rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--reference-led", type=int, default=3)
    parser.add_argument("--minimum-day", type=float, default=1.0)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=None)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=None)
    parser.add_argument("--ice-reset-day", type=float, default=None)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows, summary = build_corrections(
        args.statistics,
        args.reference_led,
        args.minimum_day,
        args.additional_acm_step_repeat,
        args.ice_reset_day,
        args.pre_reset_acm_step_repeat,
    )
    write_csv(args.corrections, CORRECTION_FIELDS, rows)
    write_csv(args.summary, SUMMARY_FIELDS, summary)
    print(f"wrote: {args.corrections} ({len(rows)} rows)")
    print(f"wrote: {args.summary} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
