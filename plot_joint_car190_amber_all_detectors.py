"""Plot all-detector 2-day CAR190 + Amber joint-fit diagnostics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time

import heatmap_car201_allstar_model_correction as workflow
from experiment_joint_wfi01_weekly_ice import RESETS, cadence_output


ROOT = Path(__file__).resolve().parent
PRODUCTION = workflow.PRODUCTION_ROOT
SUMMARY = ROOT / "joint_car190_amber_2day_all_detectors_summary.csv"
OUTPUT = PRODUCTION / (
    "wfi01/spline_relative_response_normal/experiments/"
    "car190_amber_joint_2day_all_detectors_overview"
)
EVENTS = (
    "2026-10-17T03:30:00",
    "2026-10-17T23:00:00",
    "2026-10-30T01:43:00",
)


def load_detector_ice(detector):
    fit_dir = cadence_output(detector, 2.0)
    summary = json.loads((fit_dir / "summary.json").read_text())
    if not summary.get("optimizer_success"):
        raise ValueError(f"{detector}: 2-day fit did not converge")
    with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=float)
        delta = np.asarray(archive["fitted_thickness_nm"], dtype=float) - np.asarray(
            archive["baseline_thickness_nm"], dtype=float
        )
    if delta.shape[0] != 256:
        raise ValueError(f"{detector}: expected 256 tile histories, got {delta.shape}")
    return times, np.percentile(delta, [16, 50, 84], axis=0)


def plot_ice_histories(detector_data, output):
    reference = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    all_bounds = np.concatenate([stats[[0, 2]].ravel() for _, stats in detector_data.values()])
    ypad = max(0.2, 0.05 * (all_bounds.max() - all_bounds.min()))
    ymin, ymax = all_bounds.min() - ypad, all_bounds.max() + ypad
    fig, axes = plt.subplots(6, 3, figsize=(16, 20), sharex=True, sharey=True)
    axes = axes.ravel()
    reset_dates = [Time(reference + day, format="mjd", scale="utc").to_datetime() for day in RESETS]
    event_dates = [Time(value, format="isot", scale="utc").to_datetime() for value in EVENTS]

    for axis, detector in zip(axes, workflow.DETECTORS, strict=True):
        times, stats = detector_data[detector]
        dates = np.asarray(Time(reference + times, format="mjd", scale="utc").to_datetime())
        stages = np.searchsorted(RESETS, times, side="right")
        for stage in range(3):
            selected = stages == stage
            axis.fill_between(dates[selected], stats[0, selected], stats[2, selected],
                              color="#357fa1", alpha=0.17, linewidth=0)
            axis.plot(dates[selected], stats[1, selected], color="#246b8a", linewidth=1.1)
        for date in reset_dates:
            axis.axvline(date, color=".35", linestyle=":", linewidth=0.8)
        for date in event_dates:
            axis.axvline(date, color="#b66b2e", linestyle="--", linewidth=0.7, alpha=0.85)
        axis.axhline(0, color=".35", linewidth=0.65)
        axis.set_title(detector, fontsize=10)
        axis.set_ylim(ymin, ymax)
        axis.grid(alpha=0.18)

    for axis in axes[::3]:
        axis.set_ylabel("Ice change (nm)")
    for axis in axes[-3:]:
        axis.set_xlabel("UTC date")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    legend = [
        Line2D([0], [0], color="#246b8a", lw=1.2, label="Median across 256 tiles"),
        Line2D([0], [0], color="#357fa1", lw=5, alpha=0.2, label="16th–84th percentile"),
        Line2D([0], [0], color=".35", lw=1, linestyle=":", label="Ice reset"),
        Line2D([0], [0], color="#b66b2e", lw=1, linestyle="--", label="Confirmed temperature event"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("WFI01–WFI18: fitted minus baseline ice thickness", y=0.997, fontsize=15)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.96))
    fig.savefig(output, dpi=180)
    return fig


def load_amber_metrics(detectors):
    rows = []
    for detector in detectors:
        path = cadence_output(detector, 2.0) / "fixed_vs_ice_stellar_metrics.csv"
        table = pd.read_csv(path).set_index("model")
        fixed = table.loc["fixed ice"]
        fitted = table.loc["2-day per-tile ice"]
        rows.append({
            "detector": detector,
            "fixed_rms": float(fixed["residual_rms_mmag"]),
            "fitted_rms": float(fitted["residual_rms_mmag"]),
            "rms_gain": float(fixed["residual_rms_mmag"] - fitted["residual_rms_mmag"]),
            "fixed_fraction": 100.0 * float(fixed["fraction_within_0p5_percent"]),
            "fitted_fraction": 100.0 * float(fitted["fraction_within_0p5_percent"]),
        })
    return pd.DataFrame(rows)


def plot_amber_metrics(metrics, summary, output):
    positions = np.arange(len(metrics))
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    axes[0].scatter(positions - 0.12, metrics["fixed_rms"], marker="o", s=34,
                    color="#777777", label="Fixed ice")
    axes[0].scatter(positions + 0.12, metrics["fitted_rms"], marker="s", s=34,
                    color="#0072B2", label="2-day per-tile ice")
    axes[0].set_ylabel("Amber residual RMS (mmag)")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].scatter(positions, metrics["rms_gain"], marker="o", s=34, color="#009E73")
    axes[1].axhline(0, color=".35", linewidth=0.8)
    axes[1].set_ylabel("RMS improvement\n(fixed − 2-day, mmag)")

    axes[2].scatter(positions - 0.12, metrics["fixed_fraction"], marker="o", s=34,
                    color="#777777", label="Fixed ice")
    axes[2].scatter(positions + 0.12, metrics["fitted_fraction"], marker="s", s=34,
                    color="#0072B2", label="2-day per-tile ice")
    axes[2].set_ylabel("Within ±0.5% (%)")
    axes[2].set_xticks(positions, metrics["detector"], rotation=45, ha="right")
    axes[2].set_xlabel("Detector")

    max_ice = summary.set_index("detector").loc[metrics["detector"], "two_day_max_abs_ice_change_nm"].to_numpy()
    twin = axes[2].twinx()
    twin.scatter(positions, max_ice, marker="^", s=34, color="#D55E00", label="Max |ice change|")
    twin.set_ylabel("Max |ice change| (nm)", color="#D55E00")
    twin.tick_params(axis="y", colors="#D55E00")
    axes[2].legend(handles=[
        Line2D([0], [0], color="#777777", marker="o", linestyle="None", label="Fixed ice"),
        Line2D([0], [0], color="#0072B2", marker="s", linestyle="None", label="2-day Amber"),
        Line2D([0], [0], color="#D55E00", marker="^", linestyle="None", label="Max |ice change|"),
    ], frameon=False, ncol=3, loc="lower left", fontsize=8)
    axes[2].grid(alpha=0.18)
    for axis in axes[:2]:
        axis.grid(alpha=0.18)

    fig.suptitle("Amber fit changes from 2-day per-tile ice modulation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    return fig


def main():
    summary = pd.read_csv(SUMMARY)
    detectors = list(workflow.DETECTORS)
    if set(summary["detector"]) != set(detectors) or len(summary) != len(detectors):
        raise ValueError("The all-detector summary must contain exactly WFI01–WFI18")
    if not (summary["two_day_optimizer_success"].astype(bool).all()):
        raise ValueError("At least one detector does not have a successful 2-day fit")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    detector_data = {detector: load_detector_ice(detector) for detector in detectors}
    metrics = load_amber_metrics(detectors)
    ice_fig = plot_ice_histories(detector_data, OUTPUT / "ice_adjustments_by_detector.png")
    amber_fig = plot_amber_metrics(metrics, summary, OUTPUT / "amber_fit_changes_by_detector.png")
    with PdfPages(OUTPUT / "all_detector_joint_fit_plots.pdf") as pdf:
        pdf.savefig(ice_fig)
        pdf.savefig(amber_fig)
    plt.close(ice_fig)
    plt.close(amber_fig)
    metrics.to_csv(OUTPUT / "amber_metrics_plotted.csv", index=False)
    print(f"Wrote {OUTPUT / 'ice_adjustments_by_detector.png'}")
    print(f"Wrote {OUTPUT / 'amber_fit_changes_by_detector.png'}")
    print(f"Wrote {OUTPUT / 'all_detector_joint_fit_plots.pdf'}")


if __name__ == "__main__":
    main()
