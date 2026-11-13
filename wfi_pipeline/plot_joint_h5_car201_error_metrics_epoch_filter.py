"""Compare CAR201 ice-correction RMSE and NMAD by filter and UTC epoch.

Input is the matched-sample table written by ``compare_eric_v3_amber_car201.py``.
The three variants use the same saved Amber roll offsets. RMSE and NMAD are
calculated from residual magnitude in millimagnitudes (mmag).
"""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heatmap_car201_allstar_model_correction import LOG_FLUX_PER_MAG


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_DIR = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/unchanged_reader_port/"
    "current_gray_v2_car201_all_detectors"
)
DEFAULT_PREDICTIONS = (
    DEFAULT_COMPARISON_DIR / "amber_eric_v3_vs_joint_modulated_measurement_predictions.csv"
)
FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213")
VARIANTS = {
    "no_ice_roll_adjusted_delta_mag": (
        "uncorrected_no_ice", "No ice correction (roll adjusted)", "#777777"
    ),
    "eric_v3_roll_adjusted_delta_mag": (
        "eric_v3", "Eric v3 corrected", "#c55a11"
    ),
    "joint_modulated_roll_adjusted_delta_mag": (
        "joint_modulated_h5", "Joint-modulated H5 corrected", "#176b87"
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    return parser.parse_args(argv)


def residual_statistics(delta_mag):
    values = 1000.0 * np.asarray(delta_mag, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite residuals in a filter/epoch group")
    median = float(np.median(values))
    nmad = float(1.4826 * np.median(np.abs(values - median)))
    flux_change_percent = 100.0 * np.expm1(
        -LOG_FLUX_PER_MAG * values / 1000.0
    )
    rmse = float(np.sqrt(np.mean(values**2)))
    return {
        "n_measurements": int(values.size),
        "median_residual_mmag": median,
        "p16_residual_mmag": float(np.quantile(values, 0.16)),
        "p84_residual_mmag": float(np.quantile(values, 0.84)),
        "rmse_mmag": rmse,
        "nmad_mmag": nmad,
        "rmse_nmad_ratio": float(rmse / nmad) if nmad > 0.0 else np.nan,
        "fraction_within_0p5_percent": float(
            np.mean(np.abs(flux_change_percent) <= 0.5)
        ),
    }


def summarize_by_epoch(predictions):
    required = {"detector", "filter", "pass", "mjd_utc", *VARIANTS}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    detectors = set(predictions["detector"].astype(str))
    if len(detectors) != 18:
        raise ValueError(f"Expected 18 detectors in the matched sample; found {len(detectors)}")

    records = []
    for (band, epoch), group in predictions.groupby(["filter", "pass"], sort=False):
        record = {
            "filter": band,
            "pass": epoch,
            "epoch_mjd_utc": float(group["mjd_utc"].median()),
            "epoch_utc": pd.to_datetime(
                float(group["mjd_utc"].median()),
                unit="D",
                origin="1858-11-17",
                utc=True,
            ).isoformat(),
        }
        for column, (key, _label, _color) in VARIANTS.items():
            stats = residual_statistics(
                pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            )
            record[f"{key}_measurements"] = stats.pop("n_measurements")
            record.update({f"{key}_{name}": value for name, value in stats.items()})
        records.append(record)
    result = pd.DataFrame(records)
    result["filter_order"] = result["filter"].map(
        {band: index for index, band in enumerate(FILTERS)}
    )
    return result.sort_values(["filter_order", "epoch_mjd_utc"]).drop(
        columns="filter_order"
    ).reset_index(drop=True)


def summarize_by_filter(predictions):
    records = []
    for band in (*FILTERS, "ALL_FILTERS"):
        selected = predictions if band == "ALL_FILTERS" else predictions.loc[
            predictions["filter"] == band
        ]
        record = {
            "filter": band,
            "detectors": int(selected["detector"].nunique()),
        }
        for column, (key, _label, _color) in VARIANTS.items():
            stats = residual_statistics(
                pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=float)
            )
            record.update({f"{key}_{name}": value for name, value in stats.items()})
        records.append(record)
    return pd.DataFrame(records)


def plot_metric(summary, output_dir, metric):
    metric_label = "RMSE" if metric == "rmse_mmag" else "NMAD"
    filename = f"car201_{metric.removesuffix('_mmag')}_by_filter_epoch"
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), sharex=True, sharey=True)
    offsets = {
        "uncorrected_no_ice": -0.12,
        "eric_v3": 0.0,
        "joint_modulated_h5": 0.12,
    }

    for band, axis in zip(FILTERS, axes.flat, strict=True):
        subset = summary.loc[summary["filter"] == band]
        for _column, (key, label, color) in VARIANTS.items():
            x_date = pd.to_datetime(subset["epoch_mjd_utc"], unit="D", origin="1858-11-17", utc=True)
            x = mdates.date2num(x_date.dt.to_pydatetime()) + offsets[key]
            values = subset[f"{key}_{metric}"].to_numpy(dtype=float)
            axis.scatter(x, values, s=25, color=color, label=label, zorder=3)
        axis.set_title(band)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(bottom=0.0)
        axis.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=3, maxticks=5, interval_multiples=True)
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=timezone.utc))
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)

    for axis in axes[:, 0]:
        axis.set_ylabel(f"Residual {metric_label} (mmag)")
    for axis in axes[-1, :]:
        axis.set_xlabel("UTC date of ending visit")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                  ncol=3, frameon=False)
    figure.suptitle(
        f"CAR201 ice-correction residual {metric_label} by filter and epoch — WFI01–WFI18",
        y=0.995,
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        "Each point pools the same matched visits from all 18 detectors. All variants use the same "
        "saved Amber roll offsets; no lines connect epochs.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.025, 0.075, 1, 0.91))
    figure.savefig(output_dir / f"{filename}.png", dpi=180)
    figure.savefig(output_dir / f"{filename}.pdf")
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    predictions = pd.read_csv(args.predictions)
    by_epoch = summarize_by_epoch(predictions)
    by_filter = summarize_by_filter(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_epoch.to_csv(args.output_dir / "car201_rmse_by_filter_epoch.csv", index=False)
    by_filter.to_csv(args.output_dir / "car201_rmse_by_filter_summary.csv", index=False)
    plot_metric(by_epoch, args.output_dir, "rmse_mmag")
    plot_metric(by_epoch, args.output_dir, "nmad_mmag")
    print("Per-filter CAR201 metrics:")
    print(by_filter.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
