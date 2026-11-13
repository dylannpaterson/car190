"""Plot Eric-v3 versus joint-H5 CAR201 ice-correction residuals by filter/epoch.

The input is the exact matched-sample output from
``compare_eric_v3_amber_car201.py``. Both models use Eric's unchanged reader,
and both residuals include the same saved Amber roll offsets.
"""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
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
    "eric_v3_roll_adjusted_relative_flux_change_percent": "Eric v3",
    "joint_modulated_roll_adjusted_relative_flux_change_percent": "Joint-modulated H5",
}
COLORS = {"Eric v3": "#c55a11", "Joint-modulated H5": "#176b87"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--stem", default="ice_correction_by_filter_epoch")
    return parser.parse_args(argv)


def epoch_summary(predictions):
    required = {"detector", "filter", "pass", "mjd_utc", *VARIANTS}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    detectors = set(predictions["detector"].astype(str))
    if len(detectors) != 18:
        raise ValueError(f"Expected 18 detectors in the matched sample; found {len(detectors)}")

    records = []
    for column, label in VARIANTS.items():
        selected = predictions[["filter", "pass", "mjd_utc", column]].rename(
            columns={column: "residual_percent"}
        )
        for (band, epoch), group in selected.groupby(["filter", "pass"], sort=False):
            values = group["residual_percent"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            median = float(np.median(values))
            records.append({
                "filter": band,
                "pass": epoch,
                "variant": label,
                "mjd_utc": float(group["mjd_utc"].median()),
                "n_measurements": int(values.size),
                "median_residual_percent": median,
                "p16_residual_percent": float(np.quantile(values, 0.16)),
                "p84_residual_percent": float(np.quantile(values, 0.84)),
                "rms_residual_percent": float(np.sqrt(np.mean(values**2))),
                "robust_rms_residual_percent": float(
                    1.4826 * np.median(np.abs(values - median))
                ),
                "fraction_within_0p5_percent": float(np.mean(np.abs(values) <= 0.5)),
            })
    return pd.DataFrame(records)


def plot(summary, output_dir, stem):
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), sharex=True, sharey=True)
    x_shift = {"Eric v3": -0.08, "Joint-modulated H5": 0.08}

    for band, axis in zip(FILTERS, axes.flat, strict=True):
        axis.axhspan(-0.5, 0.5, color="#65ac68", alpha=0.14, zorder=0)
        axis.axhline(0.0, color="0.35", linewidth=0.8, zorder=1)
        subset = summary.loc[summary["filter"] == band]
        for variant in VARIANTS.values():
            current = subset.loc[subset["variant"] == variant].copy()
            current = current.sort_values("mjd_utc")
            utc = pd.to_datetime(
                current["mjd_utc"].to_numpy(dtype=float),
                unit="D",
                origin="1858-11-17",
                utc=True,
            )
            x = mdates.date2num(utc.to_pydatetime()) + x_shift[variant]
            median = current["median_residual_percent"].to_numpy(dtype=float)
            lower = median - current["p16_residual_percent"].to_numpy(dtype=float)
            upper = current["p84_residual_percent"].to_numpy(dtype=float) - median
            axis.errorbar(
                x,
                median,
                yerr=np.vstack((lower, upper)),
                fmt="o",
                color=COLORS[variant],
                markersize=4.0,
                elinewidth=0.8,
                capsize=2.0,
                label=variant,
                zorder=3,
            )
        axis.set_title(band)
        axis.grid(axis="y", alpha=0.22)
        axis.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=3, maxticks=5, interval_multiples=True)
        )
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d", tz=timezone.utc)
        )
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)

    for axis in axes[:, 0]:
        axis.set_ylabel("Residual relative flux change (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("UTC date of ending visit")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                  ncol=2, frameon=False)
    figure.suptitle(
        "Eric v3 versus joint-modulated H5 — same stars, epochs, and roll offsets",
        y=0.995,
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        "Each point pools the exact matched CAR201 measurements from WFI01–WFI18 at that filter/epoch. "
        "Bars show the 16th–84th percentiles; green band is ±0.5%. Residuals include the same saved "
        "Amber roll offsets for both models. No connecting lines.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.025, 0.075, 1, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=180)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    predictions = pd.read_csv(args.predictions)
    summary = epoch_summary(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / f"{args.stem}.csv", index=False)
    plot(summary, args.output_dir, args.stem)

    pooled = []
    for (band, variant), group in summary.groupby(["filter", "variant"], sort=False):
        weights = group["n_measurements"].to_numpy(dtype=float)
        pooled.append({
            "filter": band,
            "variant": variant,
            "epochs": int(len(group)),
            "measurements": int(weights.sum()),
            "weighted_mean_epoch_robust_rms_percent": float(
                np.average(group["robust_rms_residual_percent"], weights=weights)
            ),
            "weighted_mean_epoch_fraction_within_0p5_percent": float(
                np.average(group["fraction_within_0p5_percent"], weights=weights)
            ),
        })
    pd.DataFrame(pooled).to_csv(
        args.output_dir / f"{args.stem}_filter_summary.csv", index=False
    )
    print(f"Wrote {args.output_dir / (args.stem + '.png')}")
    print(f"Wrote {args.output_dir / (args.stem + '.pdf')}")
    print(f"Wrote {args.output_dir / (args.stem + '.csv')}")
    print(f"Wrote {args.output_dir / (args.stem + '_filter_summary.csv')}")


if __name__ == "__main__":
    main()
