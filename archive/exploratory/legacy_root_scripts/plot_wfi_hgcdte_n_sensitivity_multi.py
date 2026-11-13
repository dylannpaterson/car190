"""Plot HgCdTe-n sensitivity trajectories for multiple WFI detectors."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from compare_wfi_hgcdte_n_sensitivity import _load_fit
from wfi_production_layout import corrected_statistics_path, positive_integer
from wfi_production_plot_annotations import (
    ENCLOSURE_TEMPERATURE_CHANGE_LABEL,
    ENCLOSURE_TEMPERATURE_CHANGE_UTC,
    SCE_TEMPERATURE_CHANGE_LABEL,
    SCE_TEMPERATURE_CHANGE_UTC,
    load_time_annotations,
)


UTC = SCE_TEMPERATURE_CHANGE_UTC.tzinfo
DEFAULT_STEP_UTC = "2026-10-17T22:37:21.121000+00:00"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--baseline-production-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", action="append", required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=256)
    parser.add_argument("--step-utc", default=DEFAULT_STEP_UTC)
    return parser.parse_args()


def _parse_utc_timestamp(value):
    from datetime import datetime

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_detector(root, detector):
    fits = []
    for tile_id in range(256):
        fits.append(_load_fit(Path(root) / "fits" / f"tile_{tile_id:03d}" / "posterior_samples.npz"))
    days = fits[0]["days"]
    for fit in fits[1:]:
        if not np.array_equal(days, fit["days"]):
            raise ValueError(f"Observation days differ within {detector}")
    return {
        "days": days,
        "thickness": np.asarray([fit["thickness"] for fit in fits]),
        "rates": np.asarray([fit["rates"] for fit in fits]),
        "reset_day": fits[0]["reset_day"],
    }


def _plot_quantiles(axis, values, color, label, days, linestyle="-"):
    quantiles = np.percentile(values, (16.0, 50.0, 84.0), axis=0)
    axis.fill_between(days, quantiles[0], quantiles[2], color=color, alpha=0.10)
    axis.plot(days, quantiles[1], color=color, linewidth=1.8, linestyle=linestyle, label=label)


def main():
    args = parse_args()
    detectors = tuple(detector.upper() for detector in args.detector)
    if not detectors:
        raise ValueError("At least one detector is required")
    statistics_path = corrected_statistics_path(args.statistics_dir, detectors[0], args.tile_size)
    annotations = load_time_annotations(statistics_path)
    step_utc = _parse_utc_timestamp(args.step_utc)
    step_day = (step_utc - annotations.start_utc).total_seconds() / 86400.0
    sce_day = (SCE_TEMPERATURE_CHANGE_UTC - annotations.start_utc).total_seconds() / 86400.0
    enclosure_day = (
        ENCLOSURE_TEMPERATURE_CHANGE_UTC - annotations.start_utc
    ).total_seconds() / 86400.0

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    legend_handles = []
    for index, detector in enumerate(detectors):
        color = COLORS[index % len(COLORS)]
        baseline_root = Path(args.baseline_production_root) / detector.lower() / "relative_response"
        sensitivity_root = Path(args.sensitivity_root) / detector.lower()
        baseline = _load_detector(baseline_root, detector)
        sensitivity = _load_detector(sensitivity_root, detector)
        if not np.array_equal(baseline["days"], sensitivity["days"]):
            raise ValueError(f"Observation days differ between runs for {detector}")
        days = baseline["days"]
        thickness_delta = sensitivity["thickness"] - baseline["thickness"]
        rate_delta = sensitivity["rates"] - baseline["rates"]

        _plot_quantiles(axes[0, 0], baseline["thickness"], color, f"{detector} baseline", days, "--")
        _plot_quantiles(axes[0, 0], sensitivity["thickness"], color, f"{detector} sigma 0.25", days)
        _plot_quantiles(axes[0, 1], baseline["rates"], color, f"{detector} baseline", days, "--")
        _plot_quantiles(axes[0, 1], sensitivity["rates"], color, f"{detector} sigma 0.25", days)
        _plot_quantiles(axes[1, 0], thickness_delta, color, f"{detector} thickness delta", days)
        _plot_quantiles(axes[1, 1], rate_delta, color, f"{detector} rate delta", days)
        legend_handles.extend(
            [
                Line2D([0], [0], color=color, linestyle="--", linewidth=1.8, label=f"{detector} baseline"),
                Line2D([0], [0], color=color, linestyle="-", linewidth=1.8, label=f"{detector} sigma 0.25"),
            ]
        )

    for axis, title, ylabel in (
        (axes[0, 0], "Ice thickness", "Ice thickness (nm)"),
        (axes[0, 1], "Instantaneous accumulation rate", "Rate (nm/day)"),
        (axes[1, 0], "Sensitivity minus baseline", "Thickness difference (nm)"),
        (axes[1, 1], "Sensitivity minus baseline", "Rate difference (nm/day)"),
    ):
        axis.axvline(step_day, color="#009E73", linestyle="-.", linewidth=1.0)
        axis.axvline(sce_day, color="#56B4E9", linestyle="--", linewidth=1.0)
        axis.axvline(enclosure_day, color="#E69F00", linestyle="--", linewidth=1.0)
        axis.axvline(baseline["reset_day"], color="#CC79A7", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1, 0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 0].set_xlabel("Days since first exposure")
    axes[1, 1].set_xlabel("Days since first exposure")

    legend_handles.extend(
        [
            Line2D([0], [0], color="#009E73", linestyle="-.", label="Observed rate step"),
            Line2D([0], [0], color="#56B4E9", linestyle="--", label=SCE_TEMPERATURE_CHANGE_LABEL),
            Line2D([0], [0], color="#E69F00", linestyle="--", label=ENCLOSURE_TEMPERATURE_CHANGE_LABEL),
            Line2D([0], [0], color="#CC79A7", linestyle=":", label="Ice reset"),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"{' / '.join(detectors)}: HgCdTe n prior sensitivity, sigma 0.25",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.958,
        f"Observed rate step: {step_utc:%Y-%m-%d %H:%M:%S.%f} UTC; "
        "bands show p16-p84 across 256 tiles",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.92))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
