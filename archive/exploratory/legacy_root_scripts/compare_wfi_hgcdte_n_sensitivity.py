"""Compare full-TMM HgCdTe-n sensitivity fits with a relative-response baseline."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wfi_production_layout import corrected_statistics_path, positive_integer
from wfi_production_plot_annotations import (
    ENCLOSURE_TEMPERATURE_CHANGE_LABEL,
    ENCLOSURE_TEMPERATURE_CHANGE_UTC,
    SCE_TEMPERATURE_CHANGE_LABEL,
    SCE_TEMPERATURE_CHANGE_UTC,
    load_time_annotations,
)


UTC = ZoneInfo("UTC")
STEP_TIME_TOLERANCE_DAYS = 1.0e-9
DEFAULT_STEP_UTC = "2026-10-17T22:37:21.121000+00:00"


def _parse_utc_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_fit(path):
    with np.load(path, allow_pickle=False) as source:
        days = np.asarray(source["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(source["tile_ice_thickness"], dtype=float).reshape(-1)
        rates = np.asarray(source["tile_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        n_log_scale = float(np.asarray(source["absolute_hgcdte_n_log_scale"]).reshape(-1)[0])
        reset_day = float(np.asarray(source["ice_reset_day"]).reshape(-1)[0])
    return {
        "days": days,
        "thickness": thickness,
        "rates": rates,
        "n_log_scale": n_log_scale,
        "reset_day": reset_day,
    }


def _residual_rms(path):
    values = []
    with (path / "tile_group_predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["residual_log"]))
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _step_values(days, rates, step_day, window_days):
    boundary = step_day - STEP_TIME_TOLERANCE_DAYS
    pre = (days >= step_day - window_days) & (days < boundary)
    post = np.flatnonzero(days >= boundary)
    if np.count_nonzero(pre) < 2 or post.size == 0:
        raise ValueError("Insufficient pre/post rate samples for step comparison")
    slope, intercept = np.polyfit(days[pre], rates[pre], 1)
    baseline = float(slope * step_day + intercept)
    first = int(post[0])
    observed = float(rates[first])
    return baseline, observed, observed - baseline


def _quantiles(values):
    return np.percentile(values, (16.0, 50.0, 84.0), axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument("--step-utc", default=DEFAULT_STEP_UTC)
    parser.add_argument("--window-hours", type=float, default=48.0)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = args.detector.upper()
    if args.window_hours <= 0.0:
        raise ValueError("window-hours must be positive")
    baseline_root = Path(args.baseline_root)
    sensitivity_root = Path(args.sensitivity_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = corrected_statistics_path(args.statistics_dir, detector, args.tile_size)
    annotations = load_time_annotations(statistics_path)
    step_utc = _parse_utc_timestamp(args.step_utc)
    step_day = (step_utc - annotations.start_utc).total_seconds() / 86400.0
    sce_day = (
        SCE_TEMPERATURE_CHANGE_UTC - annotations.start_utc
    ).total_seconds() / 86400.0
    enclosure_day = (
        ENCLOSURE_TEMPERATURE_CHANGE_UTC - annotations.start_utc
    ).total_seconds() / 86400.0
    window_days = float(args.window_hours) / 24.0

    baseline_fits = []
    sensitivity_fits = []
    rows = []
    for tile_id in range(256):
        baseline_path = baseline_root / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz"
        sensitivity_path = sensitivity_root / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz"
        baseline = _load_fit(baseline_path)
        sensitivity = _load_fit(sensitivity_path)
        if not np.array_equal(baseline["days"], sensitivity["days"]):
            raise ValueError(f"Observation days differ for tile {tile_id}")
        baseline_fits.append(baseline)
        sensitivity_fits.append(sensitivity)
        baseline_step = _step_values(
            baseline["days"], baseline["rates"], step_day, window_days
        )
        sensitivity_step = _step_values(
            sensitivity["days"], sensitivity["rates"], step_day, window_days
        )
        rows.append(
            {
                "detector": detector,
                "tile_id": tile_id,
                "baseline_n_log_scale": baseline["n_log_scale"],
                "sensitivity_n_log_scale": sensitivity["n_log_scale"],
                "delta_n_log_scale": sensitivity["n_log_scale"] - baseline["n_log_scale"],
                "baseline_initial_thickness_nm": baseline["thickness"][0],
                "sensitivity_initial_thickness_nm": sensitivity["thickness"][0],
                "delta_initial_thickness_nm": sensitivity["thickness"][0] - baseline["thickness"][0],
                "baseline_final_thickness_nm": baseline["thickness"][-1],
                "sensitivity_final_thickness_nm": sensitivity["thickness"][-1],
                "delta_final_thickness_nm": sensitivity["thickness"][-1] - baseline["thickness"][-1],
                "baseline_final_rate_nm_day": baseline["rates"][-1],
                "sensitivity_final_rate_nm_day": sensitivity["rates"][-1],
                "delta_final_rate_nm_day": sensitivity["rates"][-1] - baseline["rates"][-1],
                "baseline_rate_step_nm_day": baseline_step[2],
                "sensitivity_rate_step_nm_day": sensitivity_step[2],
                "delta_rate_step_nm_day": sensitivity_step[2] - baseline_step[2],
                "baseline_residual_rms_log": _residual_rms(
                    baseline_path.parent
                ),
                "sensitivity_residual_rms_log": _residual_rms(
                    sensitivity_path.parent
                ),
            }
        )

    fields = tuple(rows[0])
    output_csv = output_dir / "wfi_hgcdte_n_sensitivity_per_tile.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    baseline_thickness = np.asarray([fit["thickness"] for fit in baseline_fits])
    sensitivity_thickness = np.asarray([fit["thickness"] for fit in sensitivity_fits])
    baseline_rates = np.asarray([fit["rates"] for fit in baseline_fits])
    sensitivity_rates = np.asarray([fit["rates"] for fit in sensitivity_fits])
    thickness_delta = sensitivity_thickness - baseline_thickness
    rate_delta = sensitivity_rates - baseline_rates
    days = baseline_fits[0]["days"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for values, axis, title, ylabel in (
        (baseline_thickness, axes[0, 0], "Ice thickness", "Ice thickness (nm)"),
        (baseline_rates, axes[0, 1], "Instantaneous accumulation rate", "Rate (nm/day)"),
        (thickness_delta, axes[1, 0], "Sensitivity minus baseline", "Thickness difference (nm)"),
        (rate_delta, axes[1, 1], "Sensitivity minus baseline", "Rate difference (nm/day)"),
    ):
        quantiles = _quantiles(values)
        axis.fill_between(days, quantiles[0], quantiles[2], color="#0072B2", alpha=0.18)
        axis.plot(days, quantiles[1], color="#0072B2", linewidth=1.8, label="p50")
        axis.axvline(step_day, color="#009E73", linestyle="-.", linewidth=1.0, label="Observed rate step")
        axis.axvline(
            sce_day,
            color="#56B4E9",
            linestyle="--",
            linewidth=1.0,
            label=SCE_TEMPERATURE_CHANGE_LABEL,
        )
        axis.axvline(
            enclosure_day,
            color="#E69F00",
            linestyle="--",
            linewidth=1.0,
            label=ENCLOSURE_TEMPERATURE_CHANGE_LABEL,
        )
        axis.axvline(baseline_fits[0]["reset_day"], color="#CC79A7", linestyle=":", linewidth=1.0, label="Ice reset")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].plot(days, _quantiles(sensitivity_thickness)[1], color="#D55E00", linewidth=1.8, label="Sensitivity p50")
    axes[0, 1].plot(days, _quantiles(sensitivity_rates)[1], color="#D55E00", linewidth=1.8, label="Sensitivity p50")
    axes[1, 0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 1].axhline(0.0, color="0.35", linewidth=0.8)
    for axis in axes[1, :]:
        axis.set_xlabel("Days since first exposure")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False)
    figure.suptitle(
        f"{detector}: HgCdTe n prior sensitivity, sigma 0.25 vs relative-response baseline",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.958,
        f"Observed rate step: {step_utc:%Y-%m-%d %H:%M:%S.%f} UTC; "
        f"comparison uses the same 256-tile, 581-observation input",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.92))
    output_plot = output_dir / "wfi_hgcdte_n_sensitivity_trajectory_comparison.png"
    figure.savefig(output_plot, dpi=180)
    plt.close(figure)

    baseline_rms = np.asarray([float(row["baseline_residual_rms_log"]) for row in rows])
    sensitivity_rms = np.asarray([float(row["sensitivity_residual_rms_log"]) for row in rows])
    final_delta = np.asarray([float(row["delta_final_thickness_nm"]) for row in rows])
    rate_delta = np.asarray([float(row["delta_final_rate_nm_day"]) for row in rows])
    n_delta = np.asarray([float(row["delta_n_log_scale"]) for row in rows])
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_plot}")
    print("Initial thickness delta p16/p50/p84: " + str(np.percentile(
        [float(row["delta_initial_thickness_nm"]) for row in rows], [16, 50, 84]
    )))
    print("Final thickness delta p16/p50/p84: " + str(np.percentile(final_delta, [16, 50, 84])))
    print("Final rate delta p16/p50/p84: " + str(np.percentile(rate_delta, [16, 50, 84])))
    print("n log-scale delta p16/p50/p84: " + str(np.percentile(n_delta, [16, 50, 84])))
    print("Residual RMS baseline/sensitivity median: " + str((np.median(baseline_rms), np.median(sensitivity_rms))))


if __name__ == "__main__":
    main()
