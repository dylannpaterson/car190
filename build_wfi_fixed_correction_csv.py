"""Build fixed ratio corrections from the LED 3 common-gain diagnostic."""

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_wfi_led_postday1_sinusoid import (
    evaluate_components,
    fit_sinusoid_plus_two_acm,
    load_led_data,
)


def all_groups(path, tile_id):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["tile_id"]) == tile_id:
                rows.append(row)
    rows.sort(key=lambda row: int(row["group"]))
    groups = [int(row["group"]) for row in rows]
    if groups != list(range(len(groups))):
        raise ValueError("Tile groups must span zero through the final group")
    return rows


def load_shared_acm(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError("The ACM summary must contain exactly six LED rows")
    first = np.asarray(
        [float(row["corrected_acm_step_log_change"]) for row in rows], dtype=float
    )
    late = np.asarray(
        [
            float(row["corrected_late_acm_step_log_change"])
            for row in rows
        ],
        dtype=float,
    )
    return float(np.median(first)), float(np.median(late))


def build_corrections(statistics, acm_summary, reference_led, tile_id, minimum_day):
    (
        days,
        values,
        first_step,
        late_step,
        _,
        _,
        _,
    ) = load_led_data(statistics, reference_led, tile_id, -1.0)
    fit_mask = days > minimum_day
    fit = fit_sinusoid_plus_two_acm(
        days[fit_mask],
        values[fit_mask],
        first_step[fit_mask],
        late_step[fit_mask],
    )
    _, model = evaluate_components(
        days - days[fit_mask][0], fit, first_step, late_step
    )
    gain_correction = np.where(days <= minimum_day, model - values, 0.0)
    acm_step, late_acm_step = load_shared_acm(acm_summary)
    all_rows = all_groups(statistics, tile_id)
    rows = []
    for row in all_rows:
        group = int(row["group"])
        day = float(row["days"])
        gain = (
            float(
                np.interp(
                    day,
                    days[days <= minimum_day],
                    gain_correction[days <= minimum_day],
                )
            )
            if day <= minimum_day
            else 0.0
        )
        rows.append(
            {
                "group": group,
                "days": day,
                "log_gain_correction": gain,
                "acm_step_log_offset": acm_step,
                "late_acm_step_log_offset": late_acm_step,
            }
        )
    return rows, fit, acm_step, late_acm_step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("wfi_tile_log_statistics_512/wfi01_tile_log_statistics_512.csv"),
    )
    parser.add_argument(
        "--acm-summary",
        type=Path,
        default=Path("wfi01_common_led3_gain_corrected_sinusoid_fits.csv"),
    )
    parser.add_argument("--reference-led", type=int, default=3)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--minimum-day", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wfi01_fixed_led3_gain_acm_corrections.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference_led < 1 or args.reference_led > 6:
        raise ValueError("Reference LED must be between 1 and 6")
    if args.minimum_day < 0.0 or not np.isfinite(args.minimum_day):
        raise ValueError("Minimum day must be finite and non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, fit, acm_step, late_acm_step = build_corrections(
        args.statistics,
        args.acm_summary,
        args.reference_led,
        args.tile_id,
        args.minimum_day,
    )
    fields = (
        "group",
        "days",
        "log_gain_correction",
        "acm_step_log_offset",
        "late_acm_step_log_offset",
    )
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.output}")
    print(
        f"reference LED {args.reference_led}: period={fit['period_days']:.8g} d; "
        f"fixed shared ACM steps={acm_step:.8g}, {late_acm_step:.8g} log"
    )


if __name__ == "__main__":
    main()
