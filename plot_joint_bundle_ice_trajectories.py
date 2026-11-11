"""Compare absolute ice histories in the preserved unmodulated and joint bundles."""

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
from wfi_wavelength_response_bundle import load_bundle


ROOT = Path(__file__).resolve().parent
PRODUCTION = workflow.PRODUCTION_ROOT
BUNDLE_DIR = PRODUCTION / "wavelength_response_bundle"
BASE_PATH = BUNDLE_DIR / "wfi_all_2nm_rank16.npz"
MODULATED_PATH = BUNDLE_DIR / "wfi_all_2nm_rank16_joint_car190_amber_2day_modulated.npz"
OUTPUT = (
    PRODUCTION / "wfi01/spline_relative_response_normal/experiments/"
    "car190_amber_joint_2day_all_detectors_overview"
)


def detector_statistics(bundle, detector):
    mask = np.asarray(bundle["detectors"]).astype(str) == detector
    ids = np.asarray(bundle["tile_ids"], dtype=int)[mask]
    order = np.argsort(ids)
    thickness = np.asarray(bundle["tile_ice_thickness_nm" if "tile_ice_thickness_nm" in bundle
                                 else "tile_ice_thickness"], dtype=float)[mask][order]
    return ids[order], thickness


def plot_overview(days, dates, base_by_detector, mod_by_detector, resets, ylim, output):
    fig, axes = plt.subplots(6, 3, figsize=(16, 19), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, detector in zip(axes, workflow.DETECTORS, strict=True):
        base = base_by_detector[detector]
        mod = mod_by_detector[detector]
        for stage in range(len(resets) + 1):
            stage_ids = np.searchsorted(resets, days, side="right") == stage
            if not np.any(stage_ids):
                continue
            for values, color, style in ((base, "0.45", "--"), (mod, "#d47727", "-")):
                stats = np.percentile(values, [16, 50, 84], axis=0)
                axis.fill_between(dates[stage_ids], stats[0, stage_ids], stats[2, stage_ids],
                                  color=color, alpha=0.12, linewidth=0)
                axis.plot(dates[stage_ids], stats[1, stage_ids], color=color,
                          linestyle=style, linewidth=1.0)
        for reset in resets:
            axis.axvline(Time(float(bundle_reference_mjd()) + reset, format="mjd", scale="utc").to_datetime(),
                         color="0.25", linestyle=":", linewidth=0.75)
        axis.set_title(detector, fontsize=10)
        axis.set_ylim(*ylim)
        axis.grid(alpha=0.18)
    for axis in axes[::3]:
        axis.set_ylabel("Ice thickness (nm)")
    for axis in axes[-3:]:
        axis.set_xlabel("UTC date")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    legend = [
        Line2D([0], [0], color="0.45", linestyle="--", lw=1.6, label="Original unmodulated bundle"),
        Line2D([0], [0], color="#d47727", linestyle="-", lw=1.6, label="Joint-modulated bundle"),
        Line2D([0], [0], color="#888888", alpha=0.2, lw=5, label="Tile 16th–84th percentile"),
        Line2D([0], [0], color="0.25", linestyle=":", lw=1, label="Reset / Decon boundary"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Unmodulated and joint-modulated WFI ice histories", y=0.997, fontsize=15)
    fig.text(0.5, 0.014,
             "Each panel summarizes 256 tiles. Both models are shown on the same unchanged CAR190 observation-time grid.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.96))
    fig.savefig(output, dpi=180)
    return fig


def bundle_reference_mjd():
    return float(Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd)


def plot_detector_page(detector, days, dates, base, mod, resets, ylim, pdf):
    delta = mod - base
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 0.8], "hspace": 0.08})
    for stage in range(len(resets) + 1):
        selected = np.searchsorted(resets, days, side="right") == stage
        if not np.any(selected):
            continue
        for values, color, style, label in (
            (base, "0.45", "--", "Original unmodulated bundle"),
            (mod, "#d47727", "-", "Joint-modulated bundle"),
        ):
            stats = np.percentile(values, [16, 50, 84], axis=0)
            axes[0].fill_between(dates[selected], stats[0, selected], stats[2, selected],
                                 color=color, alpha=0.14, linewidth=0)
            axes[0].plot(dates[selected], stats[1, selected], color=color,
                         linestyle=style, linewidth=1.8, label=label if stage == 0 else None)
        delta_stats = np.percentile(delta, [16, 50, 84], axis=0)
        axes[1].fill_between(dates[selected], delta_stats[0, selected], delta_stats[2, selected],
                             color="#0072B2", alpha=0.16, linewidth=0)
        axes[1].plot(dates[selected], delta_stats[1, selected], color="#0072B2", linewidth=1.5)

    for axis in axes:
        for reset in resets:
            axis.axvline(Time(bundle_reference_mjd() + reset, format="mjd", scale="utc").to_datetime(),
                         color="0.25", linestyle=":", linewidth=0.9)
        axis.axhline(0, color="0.4", linewidth=0.7)
        axis.grid(alpha=0.2)
    axes[0].set_ylim(*ylim)
    delta_limit = max(1.0, float(np.max(np.abs(delta))) * 1.08)
    axes[1].set_ylim(-delta_limit, delta_limit)
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[1].set_ylabel("Modulated − unmodulated (nm)")
    axes[1].set_xlabel("UTC date")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle(f"{detector}: ice thickness history", y=0.99, fontsize=14)
    fig.text(0.5, 0.02,
             "Lines are tile medians; shading is tile 16th–84th percentiles. "
             "Dotted lines mark the unchanged ice reset and Decon 3 boundaries.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = load_bundle(BASE_PATH)
    mod = load_bundle(MODULATED_PATH)
    for name in ("tile_ids", "detectors", "wavelengths_nm", "ice_thickness_nm", "observation_days"):
        np.testing.assert_array_equal(base[name], mod[name], err_msg=f"Bundle grid/schema mismatch: {name}")
    days = np.asarray(base["observation_days"], dtype=float)
    reference_mjd = bundle_reference_mjd()
    dates = np.asarray(Time(reference_mjd + days, format="mjd", scale="utc").to_datetime())
    resets = [float(np.asarray(base[name], dtype=float).reshape(-1)[0])
              for name in ("ice_reset_day", "third_decon_day")]

    base_by_detector = {}
    mod_by_detector = {}
    all_stats = []
    for detector in workflow.DETECTORS:
        base_ids, base_values = detector_statistics(base, detector)
        mod_ids, mod_values = detector_statistics(mod, detector)
        np.testing.assert_array_equal(base_ids, mod_ids)
        if base_values.shape != (256, days.size) or mod_values.shape != base_values.shape:
            raise ValueError(f"Unexpected tile-history dimensions for {detector}")
        base_by_detector[detector] = base_values
        mod_by_detector[detector] = mod_values
        base_stats = np.percentile(base_values, [16, 50, 84], axis=0)
        mod_stats = np.percentile(mod_values, [16, 50, 84], axis=0)
        delta_stats = np.percentile(mod_values - base_values, [16, 50, 84], axis=0)
        for day_index, day in enumerate(days):
            all_stats.append({
                "detector": detector,
                "day": float(day),
                "unmodulated_p16_nm": float(base_stats[0, day_index]),
                "unmodulated_median_nm": float(base_stats[1, day_index]),
                "unmodulated_p84_nm": float(base_stats[2, day_index]),
                "joint_modulated_p16_nm": float(mod_stats[0, day_index]),
                "joint_modulated_median_nm": float(mod_stats[1, day_index]),
                "joint_modulated_p84_nm": float(mod_stats[2, day_index]),
                "thickness_change_p16_nm": float(delta_stats[0, day_index]),
                "thickness_change_median_nm": float(delta_stats[1, day_index]),
                "thickness_change_p84_nm": float(delta_stats[2, day_index]),
            })

    limits_values = np.concatenate([
        values.ravel() for values in
        [base_by_detector[d] for d in workflow.DETECTORS]
        + [mod_by_detector[d] for d in workflow.DETECTORS]
    ])
    ymax = float(np.nanpercentile(limits_values, 99.8))
    ylim = (min(0.0, float(np.nanmin(limits_values)) - 1.0), ymax + 0.04 * ymax)
    pdf_path = OUTPUT / "ice_thickness_unmodulated_vs_joint_modulated_all_detectors.pdf"
    overview_path = OUTPUT / "ice_thickness_unmodulated_vs_joint_modulated_overview.png"
    with PdfPages(pdf_path) as pdf:
        overview = plot_overview(days, dates, base_by_detector, mod_by_detector, resets, ylim, overview_path)
        pdf.savefig(overview)
        plt.close(overview)
        for detector in workflow.DETECTORS:
            plot_detector_page(detector, days, dates, base_by_detector[detector],
                               mod_by_detector[detector], resets, ylim, pdf)
    pd.DataFrame(all_stats).to_csv(OUTPUT / "ice_thickness_unmodulated_vs_joint_modulated.csv", index=False)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {overview_path}")


if __name__ == "__main__":
    main()
