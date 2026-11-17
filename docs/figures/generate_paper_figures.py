"""Regenerate the WFI methodology paper's six figures in this directory.

Run ``python docs/figures/generate_paper_figures.py --figure all`` to rebuild
the complete set, or pass a figure number from 1 through 6 to rebuild one plot.
The scripts use the recorded statistics, model fits, and reference data in the
repository workspace; no fitting stage is rerun.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FIGURE_WIDTH_IN = 6.4
FIGURE_DPI = 240
PLOT_FONT_SIZE = 8
TICK_FONT_SIZE = 7
LEGEND_FONT_SIZE = 7

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent
STATS = ROOT / "20261111/corrected_statistics_4h_thinned"
TILE = 135
DETECTOR = "WFI02"
LED_FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184")
FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213")
RESET_DAYS = np.asarray([12.907397928240743, 38.79784510416666], dtype=float)
RCS_ORIGIN_UTC = "2026-09-12T01:25:50.714+00:00"


def save_figure(figure, stem: str, dpi: int = FIGURE_DPI) -> tuple[Path, Path]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf, dpi=dpi)
    plt.close(figure)
    print(f"Wrote {png} and {pdf}")
    return png, pdf


def plot_spectral_coverage() -> tuple[Path, Path]:
    from astropy.table import Table

    sys.path.insert(0, str(ROOT / "wfi_pipeline"))
    import fit_car201_zane_star_reflectance_modulation as stellar

    led_table = Table.read(
        ROOT / "external_reference_data/WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    # This ECSV has a free-text provenance line before the YAML metadata; the
    # comma-separated table body is therefore read after skipping comment lines.
    effective_area = pd.read_csv(
        ROOT / "external_reference_data/roman_effarea_v8_20240301/"
        "Roman_effarea_v8_SCA02_20240301.ecsv",
        comment="#",
        skipinitialspace=True,
    )
    star_id = "2128134408017300864"
    wavelengths_nm = np.arange(480.0, 2700.1, 5.0)
    sed_posteriors = stellar.load_sed_posteriors(
        ROOT / "car201_data/sed/v2/faint-posteriors-v2.h5"
    )
    photon_shape, parameters, fallback_count = stellar.reconstruct_sed_photon_shapes(
        [star_id], sed_posteriors,
        ROOT / "car201_data/sed/v2/btsettl-spectral-grid.h5", wavelengths_nm,
    )
    if fallback_count:
        raise ValueError(f"Unexpected nearest-grid SED fallbacks: {fallback_count}")

    figure, axes = plt.subplots(
        3, 1, figsize=(FIGURE_WIDTH_IN, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.0, 1.0]},
    )
    colors = plt.get_cmap("tab10")(np.arange(8))
    styles = ("-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1)))
    led_bands = np.asarray(led_table["RCSLEDBand"], dtype=int)
    led_wave = np.asarray(led_table["RCSLEDSpectrumWavelength"], dtype=float)
    led_output = np.asarray(led_table["RCSLEDSpectrumRelativeOutput"], dtype=float)
    for led in range(1, 7):
        selected = led_bands == led
        wave, output = led_wave[selected], led_output[selected]
        order = np.argsort(wave)
        wave, output = wave[order], output[order]
        peak = float(np.max(output))
        use = output >= 0.05 * peak
        axes[0].plot(
            wave[use], output[use] / peak, color=colors[led - 1],
            linestyle=styles[led - 1], linewidth=1.1,
            label=f"LED {led} ({LED_FILTERS[led - 1]})",
        )

    area_wave_nm = np.asarray(effective_area["Wave"], dtype=float) * 1000.0
    for index, band in enumerate(FILTERS):
        area = np.asarray(effective_area[band], dtype=float)
        axes[1].plot(
            area_wave_nm, area / np.max(area), color=colors[index], linewidth=1.0,
            label=band,
        )
    spectrum = np.asarray(photon_shape[0], dtype=float)
    axes[2].plot(wavelengths_nm, spectrum / np.max(spectrum), color="0.1", linewidth=1.0)
    led_centers_nm = np.asarray([623.521, 852.803, 1038.585, 1269.758, 1481.241, 1670.169])
    for center in led_centers_nm:
        axes[2].axvline(center, color="0.45", linestyle=":", linewidth=0.8)

    labels = ("a", "b", "c")
    ylabels = ("LED output / peak", "Effective area / filter peak", "Photon SED / peak")
    for axis, label, ylabel in zip(axes, labels, ylabels, strict=True):
        axis.text(-0.07, 1.02, label, transform=axis.transAxes,
                  fontsize=10, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xlim(470, 2700)
        axis.grid(alpha=0.2)
    led_handles, led_labels = axes[0].get_legend_handles_labels()
    area_handles, area_labels = axes[1].get_legend_handles_labels()
    led_legend = figure.legend(
        led_handles, led_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
        frameon=False, fontsize=LEGEND_FONT_SIZE, columnspacing=0.8,
        handlelength=1.6,
    )
    figure.add_artist(led_legend)
    figure.legend(
        area_handles, area_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4,
        frameon=False, fontsize=LEGEND_FONT_SIZE, columnspacing=0.8,
        handlelength=1.6,
    )
    axes[2].set_xlabel("Wavelength (nm)")
    figure.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.10, hspace=0.38)
    print(f"Example SED posterior-median T_eff/log g: {parameters[0, 0]:.0f} K/{parameters[0, 1]:.2f}")
    return save_figure(figure, "wfi02_spectral_coverage")


def plot_raw_corrected() -> tuple[Path, Path]:
    from astropy.time import Time

    path = ROOT / "20261111/correction_comparison/wfi02_tile135_raw_vs_corrected.csv"
    frame = pd.read_csv(path)
    frame["utc"] = pd.to_datetime(
        frame["exposure_start_time_corrected"], utc=True, format="mixed"
    ).dt.tz_convert(None)
    reset_dates = [pd.Timestamp("2026-09-24T23:12:29.895"),
                   pd.Timestamp("2026-10-20T20:34:44.531")]
    figure, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 3.9), sharex=True, sharey=True)
    date_ticks = pd.to_datetime(["2026-09-22", "2026-10-12", "2026-11-01"])
    colors = {"raw": "#2378b5", "corrected": "#d95f02"}
    markers = {1039: "o", 1063: "^"}
    for led, axis in enumerate(axes.flat, start=1):
        selected = frame.loc[frame["led_corrected"] == led]
        for program, marker in markers.items():
            rows = selected.loc[selected["program_corrected"] == program].sort_values("utc")
            axis.scatter(rows["utc"], rows["raw_mean_milli"], s=10, marker=marker,
                         color=colors["raw"], alpha=0.75, linewidths=0)
            axis.scatter(rows["utc"], rows["corrected_mean_milli"], s=10, marker=marker,
                         color=colors["corrected"], alpha=0.75, linewidths=0)
        for reset_index, event in enumerate(reset_dates):
            style = ":" if reset_index == 0 else "-."
            axis.axvline(event, color="0.35", linestyle=style, linewidth=0.8)
        axis.set_title(f"LED {led} ({LED_FILTERS[led - 1]})")
        axis.grid(alpha=0.2)
        axis.set_xticks(date_ticks)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    figure.supylabel(r"Reference-relative log response ($10^{-3}$)", fontsize=PLOT_FONT_SIZE)
    figure.supxlabel("UTC date", fontsize=PLOT_FONT_SIZE, y=0.035)
    handles = [
        Line2D([0], [0], marker="o", color=colors["raw"], linestyle="none",
               label="Raw / uncorrected"),
        Line2D([0], [0], marker="o", color=colors["corrected"], linestyle="none",
               label="Fully corrected"),
        Line2D([0], [0], marker="o", color="0.15", linestyle="none",
               label="CAR190 (Program 1039)"),
        Line2D([0], [0], marker="^", color="0.15", linestyle="none",
               label="CAR216 (Program 1063)"),
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=0.8,
               label="Decon2 reset"),
        Line2D([0], [0], color="0.35", linestyle="-.", linewidth=0.8,
               label="Decon3 reset"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                  ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE,
                  columnspacing=0.8, handlelength=1.6)
    figure.subplots_adjust(left=0.12, right=0.99, top=0.83, bottom=0.14,
                            hspace=0.24, wspace=0.20)
    return save_figure(figure, "wfi02_tile135_raw_vs_corrected")


def load_tile_history(path: Path, constant_rate: bool = False):
    with np.load(path, allow_pickle=False) as archive:
        days = np.asarray(archive["observation_days"], dtype=float)
        thickness = np.squeeze(np.asarray(archive["tile_ice_thickness"], dtype=float))
        rate = np.squeeze(np.asarray(archive["tile_instantaneous_rate_nm_day"], dtype=float))
        resets = np.asarray(archive["ice_reset_days"], dtype=float)
    if constant_rate and not np.allclose(rate, rate[0], rtol=0, atol=1e-10):
        raise ValueError(f"Expected a constant ice-deposition-rate history in {path}")
    return days, thickness, rate, resets


def plot_ice_history() -> tuple[Path, Path]:
    from datetime import datetime, timedelta

    production = STATS / "production/wfi02"
    warm_path = production / "relative_response_constant_rate_warmup_start40/fits/tile_135/constant_rate_map.npz"
    rapid_path = production / "relative_response/fits/tile_135/posterior_samples.npz"
    joint_path = production / "relative_response/experiments/car190_amber_joint_2day_reader_compatible_current_gray_v2/joint_weekly_ice_fit.npz"
    warm_days, warm_d, warm_r, resets = load_tile_history(warm_path, constant_rate=True)
    rapid_days, rapid_d, rapid_r, rapid_resets = load_tile_history(rapid_path)
    np.testing.assert_allclose(rapid_resets, resets, rtol=0, atol=1e-10)
    with np.load(joint_path, allow_pickle=False) as archive:
        joint_days = np.asarray(archive["times"], dtype=float)
        joint_d = np.asarray(archive["fitted_thickness_nm"], dtype=float)[TILE]
    joint_r = np.empty_like(joint_d)
    segments = np.searchsorted(resets, joint_days, side="right")
    for segment in np.unique(segments):
        use = segments == segment
        joint_r[use] = np.gradient(joint_d[use], joint_days[use])

    origin = datetime.fromisoformat(RCS_ORIGIN_UTC)

    def dates(days):
        return mdates.date2num([origin + timedelta(days=float(day)) for day in days])

    colors = {"initial": "#cb78a5", "rapid": "#177db5", "joint": "#db5b00"}
    series = [
        (warm_days, warm_d, warm_r, colors["initial"], "--",
         "Constant ice-deposition-rate initialization"),
        (rapid_days, rapid_d, rapid_r, colors["rapid"], "-", "Rapid RCS update"),
        (joint_days, joint_d, joint_r, colors["joint"], "-",
         "Joint RCS + stellar photometry fit"),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_IN, 3.9), sharex=True,
                                gridspec_kw={"height_ratios": [1.0, 0.9], "hspace": 0.08})
    for days, thickness, rate, color, style, label in series:
        masks = [days < resets[0], (days >= resets[0]) & (days < resets[1]), days >= resets[1]]
        for segment, mask in enumerate(masks):
            if not np.any(mask):
                continue
            axes[0].plot(dates(days[mask]), thickness[mask], color=color, linestyle=style,
                         linewidth=1.1, label=label if segment == 0 else None)
            axes[1].plot(dates(days[mask]), rate[mask], color=color, linestyle=style,
                         linewidth=1.1, label=label if segment == 0 else None)
    for axis in axes:
        for reset_index, reset in enumerate(resets):
            style = ":" if reset_index == 0 else "-."
            axis.axvline(dates([reset])[0], color="#263c63", linestyle=style, linewidth=0.9)
        axis.grid(color="#d5dfeb", linewidth=0.6)
    axes[0].set_ylabel("Thickness (nm)")
    axes[0].set_ylim(-5, 255)
    axes[1].set_ylabel("Rate (nm/day)")
    axes[1].set_xlabel("UTC date")
    axes[1].set_ylim(6, max(warm_r.max(), rapid_r.max(), joint_r.max()) + 1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="#263c63", linestyle=":", linewidth=0.9, label="Decon2 reset"),
        Line2D([0], [0], color="#263c63", linestyle="-.", linewidth=0.9, label="Decon3 reset"),
    ])
    legend_labels.extend(["Decon2 reset", "Decon3 reset"])
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                  ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE,
                  columnspacing=0.8, handlelength=1.6)
    for axis, label in zip(axes, ("a", "b"), strict=True):
        axis.text(-0.07, 1.02, label, transform=axis.transAxes, fontsize=9,
                  fontweight="bold", va="bottom")
    axes[1].set_xlim(dates([-1])[0], dates([61])[0])
    axes[1].set_xticks(dates([0, 20, 40, 60]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=timezone.utc))
    figure.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.12, hspace=0.10)
    return save_figure(figure, "wfi02_representative_ice_history")


def plot_rcs_model_data() -> tuple[Path, Path]:
    from astropy.time import Time

    sys.path[:0] = [str(ROOT / "wfi_pipeline"),
                    str(ROOT / "archive/exploratory/legacy_root_scripts")]
    import fit_wfi_tmm_production as production
    import plot_joint_amber_car190_detailed as plotter
    from experiment_joint_wfi01_fixed_ice import LedModel
    from support_aligned_modulation import effective_area_support

    relative_root = STATS / "production/wfi02/relative_response"
    warm_root = STATS / "production/wfi02/relative_response_constant_rate_warmup_start40"
    warm_path = warm_root / "fits/tile_135/constant_rate_map.npz"
    rapid_path = relative_root / "fits/tile_135/tile_group_predictions.csv"
    joint_root = relative_root / "experiments/car190_amber_joint_2day_reader_compatible_current_gray_v2"
    with np.load(warm_path, allow_pickle=False) as warm:
        warm_days = np.asarray(warm["observation_days"], dtype=float)
        warm_led = np.asarray(warm["observation_led"], dtype=int)
        warm_thickness = np.asarray(warm["tile_ice_thickness"], dtype=float).reshape(-1)
        cache_path = Path(str(np.asarray(warm["response_cache"]).reshape(-1)[0]))
        cache = np.load(cache_path, allow_pickle=False)
        grid_nm = np.asarray(cache["grid_nm"], dtype=float)
        curves = np.asarray(cache["relative_log_response"], dtype=float)
        offsets = (np.asarray(warm["led_log_offsets"], dtype=float).reshape(-1)
                   if bool(np.asarray(warm["fit_led_log_offsets"]).item())
                   else np.zeros(6, dtype=float))
    rapid = pd.read_csv(rapid_path).sort_values("group")
    np.testing.assert_array_equal(rapid["led"].to_numpy(int), warm_led)
    np.testing.assert_allclose(rapid["days"], warm_days, rtol=0, atol=1e-10)
    warm_prediction = np.empty_like(warm_thickness)
    for led in range(1, 7):
        indexes = np.flatnonzero(warm_led == led)
        reference_index = int(indexes[0])
        curve = curves[led - 1]
        current = np.interp(warm_thickness[indexes], grid_nm, curve)
        reference = float(np.interp(warm_thickness[reference_index], grid_nm, curve))
        warm_prediction[indexes] = current - reference + np.where(
            indexes != reference_index, offsets[led - 1], 0.0
        )

    joint_file = joint_root / "joint_weekly_ice_fit.npz"
    with np.load(joint_file, allow_pickle=False) as joint:
        times = np.asarray(joint["times"], dtype=float)
        thickness = np.asarray(joint["fitted_thickness_nm"], dtype=float)
        theta = np.asarray(joint["shared_parameters"], dtype=float)
    area_path = ROOT / "external_reference_data/roman_effarea_v8_20240301/" \
        "Roman_effarea_v8_SCA02_20240301.ecsv"
    led_model = LedModel(
        effective_area_support(area_path), detector=DETECTOR,
        relative_response_root=relative_root,
        optical_cache_root=Path(json.loads((joint_root / "summary.json").read_text())["optical_cache_root"]),
    )
    joint_rows, _ = plotter.predict_car190(led_model, theta, times, thickness)
    joint_tile = joint_rows.loc[joint_rows["tile"] == TILE]
    joint_prediction = np.empty_like(warm_thickness)
    observed = rapid["observed_log_change"].to_numpy(float)
    sigma = np.empty_like(observed)
    for led in range(1, 7):
        rows = joint_tile.loc[joint_tile["led"] == led].sort_values("day")
        use = rapid["led"].to_numpy(int) == led
        np.testing.assert_allclose(rows["day"], warm_days[use], rtol=0, atol=1e-10)
        joint_prediction[use] = rows["predicted"].to_numpy(float)
        sigma[use] = np.asarray(led_model.bands[led - 1][TILE][8], dtype=float)
        np.testing.assert_allclose(led_model.bands[led - 1][TILE][7], observed[use], rtol=0, atol=1e-10)

    reference = Time(RCS_ORIGIN_UTC.removesuffix("+00:00"), format="isot", scale="utc").mjd

    def dates(days):
        return mdates.date2num(Time(reference + np.asarray(days), format="mjd", scale="utc").to_datetime())

    colors = {"warm": "#cb78a5", "rapid": "#177db5", "joint": "#db5b00"}
    figure, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 4.2), sharex=True, sharey=True,
                                gridspec_kw={"hspace": 0.35, "wspace": 0.20})
    values = np.concatenate([observed, warm_prediction, rapid["model_log_change"].to_numpy(float), joint_prediction]) * 1000.0
    low, high = float(values.min()), float(values.max())
    margin = 0.05 * max(high-low, 1.0)
    for led, axis in enumerate(axes.flat, start=1):
        use = rapid["led"].to_numpy(int) == led
        days = warm_days[use]
        axis.errorbar(dates(days), observed[use]*1000, yerr=sigma[use]*1000,
                      fmt="o", color="0.12", markersize=2.8, elinewidth=0.65,
                      capsize=1.0, label="Corrected RCS group means" if led == 1 else None)
        masks = [days < RESET_DAYS[0], (days >= RESET_DAYS[0]) & (days < RESET_DAYS[1]), days >= RESET_DAYS[1]]
        for segment, mask in enumerate(masks):
            if not np.any(mask):
                continue
            axis.plot(dates(days[mask]), warm_prediction[use][mask]*1000,
                      color=colors["warm"], linestyle="--", linewidth=1.0,
                      label="Constant ice-deposition-rate initialization" if led == 1 and segment == 0 else None)
            axis.plot(dates(days[mask]), rapid.loc[use, "model_log_change"].to_numpy()[mask]*1000,
                      color=colors["rapid"], linewidth=1.0,
                      label="Rapid RCS update" if led == 1 and segment == 0 else None)
            axis.plot(dates(days[mask]), joint_prediction[use][mask]*1000,
                      color=colors["joint"], linewidth=1.0,
                      label="Joint RCS + stellar-photometry fit" if led == 1 and segment == 0 else None)
        for reset_index, reset in enumerate(RESET_DAYS):
            style = ":" if reset_index == 0 else "-."
            axis.axvline(dates([reset])[0], color="#263c63", linestyle=style, linewidth=0.8)
        axis.set_title(f"LED {led} ({LED_FILTERS[led-1]})")
        axis.set_ylim(low-margin,high+margin)
        axis.grid(color="#d5dfeb",linewidth=0.8)
    figure.supylabel(r"Relative log response ($10^{-3}$)", fontsize=PLOT_FONT_SIZE)
    figure.supxlabel("UTC date", fontsize=PLOT_FONT_SIZE, y=0.035)
    date_ticks = dates([10, 30, 50])
    date_limits = dates([0, 60])
    for axis in axes.flat:
        axis.set_xticks(date_ticks)
        axis.set_xlim(date_limits[0], date_limits[-1])
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=timezone.utc))
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="#263c63", linestyle=":", linewidth=0.8, label="Decon2 reset"),
        Line2D([0], [0], color="#263c63", linestyle="-.", linewidth=0.8, label="Decon3 reset"),
    ])
    labels.extend(["Decon2 reset", "Decon3 reset"])
    figure.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,0.995),
                  ncol=3,frameon=False,fontsize=LEGEND_FONT_SIZE,
                  columnspacing=0.8,handlelength=1.6)
    figure.subplots_adjust(left=0.12,right=0.99,top=0.84,bottom=0.14,hspace=0.35,wspace=0.20)
    return save_figure(figure,"wfi02_tile135_rcs_model_data")


def plot_car201_model_data() -> tuple[Path, Path]:
    # The joint-fit forward model supplies fixed optical calibration, the RCS-only
    # baseline histories, the fitted histories, the stellar data, and the fitted
    # post-roll offsets. Figure 5 moves that roll term to the observed series.
    sys.path.insert(0, str(ROOT / "wfi_pipeline"))
    from astropy.time import Time
    from fit_amber_wfi01_student_t import profile_student_locations
    from reader_compatible_joint_forward import ReaderCompatiblePhotometryForward
    import run_amber_student_t_detectors as run
    import run_joint_car190_amber_inverse_wavelength_maxent as joint_fit

    fit_dir = STATS / "production/wfi02/relative_response/experiments/" \
        "car190_amber_joint_2day_reader_compatible_current_gray_v2"
    source_ids, seds, raw_magnitude, raw_error, sample_valid, common, frames = joint_fit.raw_common_training_sample(DETECTOR)
    with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=float)
        fit_source_ids = np.asarray(archive["source_ids"]).astype(str)
        fit_valid = np.asarray(archive["stellar_valid"], dtype=bool)
        joint_prediction_with_roll = np.asarray(archive["stellar_prediction_mag"], dtype=float)*1000
    with np.load(fit_dir / "training_inputs.npz", allow_pickle=False) as archive:
        magnitude = np.asarray(archive["magnitude"], dtype=float)
        error = np.asarray(archive["error"], dtype=float)
        valid = np.asarray(archive["valid"], dtype=bool)
    with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
        history_times = np.asarray(archive["times"], dtype=float)
        baseline_history = np.asarray(archive["baseline_thickness_nm"], dtype=float)
        fitted_history = np.asarray(archive["fitted_thickness_nm"], dtype=float)
    np.testing.assert_array_equal(source_ids, fit_source_ids)
    np.testing.assert_array_equal(sample_valid, fit_valid)
    np.testing.assert_array_equal(valid, fit_valid)
    np.testing.assert_allclose(magnitude, raw_magnitude, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(error, raw_error, rtol=0, atol=0, equal_nan=True)
    summary = json.loads((fit_dir / "summary.json").read_text())
    history_root = Path(summary["ice_history_fit_root"])
    model = joint_fit.model_for_raw_sample(
        DETECTOR, source_ids, seds, valid, frames,
        ROOT / "external_reference_data/roman_effarea_v8_20240301/Roman_effarea_v8_SCA02_20240301.ecsv",
        history_root, Path(summary["optical_cache_root"]),
    )
    with np.load(history_root / "tile_000/posterior_samples.npz", allow_pickle=False) as archive:
        reset_days = np.asarray(archive["ice_reset_days"], dtype=float)
    reference_mjd = Time(run.workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    pass_days = np.asarray([
        [joint_fit.common_pass_mjd(frames, band, name) - reference_mjd for name in run.PASSES]
        for band in run.FILTERS
    ])
    pass_indices = np.searchsorted(history_times, pass_days)
    np.testing.assert_allclose(history_times[pass_indices], pass_days, rtol=0, atol=1e-10)
    mode = model.mode_count
    zero_points = parameters[2*mode:].reshape(len(run.FILTERS), model.zero_point_group_count)
    roll_mag = np.zeros((len(source_ids), len(run.FILTERS), len(run.PASSES)), dtype=float)
    for pass_index, group in enumerate(model.pass_zero_point_group_ids):
        if group:
            roll_mag[:, :, pass_index] = (-zero_points[:, group-1]/run.workflow.LOG_FLUX_PER_MAG)[None, :]
    reference = np.zeros(magnitude.shape[:2], dtype=float)
    for source in range(len(source_ids)):
        for filter_index in range(len(run.FILTERS)):
            values = magnitude[source, filter_index, valid[source, filter_index]]
            if values.size:
                reference[source, filter_index] = np.median(values)
    observed = np.where(valid, (magnitude-reference[:, :, None])*1000, 0.)
    roll_adjusted_data = np.where(valid, observed-roll_mag*1000, 0.)
    joint_no_roll = joint_prediction_with_roll-roll_mag*1000
    joint_residual_before = observed-joint_prediction_with_roll
    joint_residual_after = roll_adjusted_data-joint_no_roll
    joint_transfer_error = float(
        np.max(np.abs(joint_residual_before[valid]-joint_residual_after[valid]))
    )
    if joint_transfer_error > 1e-10:
        raise ValueError(f"Roll transfer changed joint-model residuals by {joint_transfer_error:g} mmag")
    extra_scale = float(summary["extra_student_t_scale_mmag"])
    scales = np.where(valid, np.hypot(error*1000, extra_scale), 1.)
    group_valid = valid.reshape((-1,len(run.PASSES)))
    active = group_valid.sum(axis=1)>=2
    group_scales = scales.reshape((-1,len(run.PASSES)))
    def profile(model_values):
        residual=(roll_adjusted_data-model_values).reshape((-1,len(run.PASSES)))
        location=np.zeros(len(residual))
        location[active]=profile_student_locations(residual[active],group_scales[active],group_valid[active])[0]
        return location.reshape(magnitude.shape[:2])
    joint_location=profile(joint_no_roll)

    model.reader_gray_log_by_tile=np.asarray(summary["reader_gray_log_offsets_by_tile"],float)[:,np.searchsorted(reset_days,pass_days,side="right")]
    dense_start=0.0
    dense_end=min(float(history_times.max()),60.0)
    dense_days=np.unique(np.r_[np.arange(dense_start,dense_end+1e-10,.25),pass_days.ravel()])
    source_tiles=np.asarray(model.source_tile_ids,int)
    source_count=len(source_ids); filter_count=len(run.FILTERS); tile_count=baseline_history.shape[0]
    curve_cohort_valid=np.zeros((source_count,filter_count,len(dense_days)),dtype=bool)
    for fi in range(filter_count):
        available_passes=np.flatnonzero(valid[:,fi,:].any(axis=0))
        nearest_passes=available_passes[np.argmin(
            np.abs(pass_days[fi,available_passes][None,:]-dense_days[:,None]),axis=1
        )]
        for pi in available_passes:
            times=nearest_passes==pi
            curve_cohort_valid[:,fi,times]=valid[:,fi,pi,None]
    history_segments=np.searchsorted(reset_days,history_times,side="right")
    def interp_history(history,target_days):
        result=np.empty((tile_count,len(target_days)))
        target_segments=np.searchsorted(reset_days,target_days,side="right")
        for segment in range(len(reset_days)+1):
            src=history_segments==segment; dst=target_segments==segment
            if not np.any(dst): continue
            for tile in range(tile_count):
                result[tile,dst]=np.interp(target_days[dst],history_times[src],history[tile,src])
        return result

    dense_model=copy.copy(model)
    dense_forward=ReaderCompatiblePhotometryForward(dense_model)
    baseline_curve=np.empty((source_count,filter_count,len(dense_days)))
    joint_curve=np.empty_like(baseline_curve)
    zero_modulation=np.zeros(2*mode)
    joint_modulation=parameters[:2*mode]
    chunk_size=20
    for start in range(0,len(dense_days),chunk_size):
        stop=min(start+chunk_size,len(dense_days)); days=dense_days[start:stop]
        targets=np.column_stack([pass_days[:,0],np.tile(days,(filter_count,1))])
        ntime=1+len(days)
        dense_model.pass_count=ntime
        dense_model.tile_thickness_by_id=np.empty((tile_count,filter_count,ntime))
        dense_model.source_tile_ids=np.empty((source_count,filter_count,ntime),dtype=int)
        dense_model.reader_gray_log_by_tile=np.empty((tile_count,filter_count,ntime))
        for fi in range(filter_count):
            dense_model.tile_thickness_by_id[:,fi,:]=interp_history(baseline_history,targets[fi])
            stages=np.searchsorted(reset_days,targets[fi],side="right")
            gray=np.asarray(summary["reader_gray_log_offsets_by_tile"],float)
            dense_model.reader_gray_log_by_tile[:,fi,:]=gray[:,stages]
            dense_model.source_tile_ids[:,fi,0]=source_tiles[:,fi,0]
            distance=np.abs(pass_days[fi][None,:,None]-days[None,None,:])
            distance=np.where(valid[:,fi,:,None],distance,np.inf)
            nearest=np.argmin(distance,axis=1)
            dense_model.source_tile_ids[:,fi,1:]=source_tiles[np.arange(source_count)[:,None],fi,nearest]
        base_log,_=dense_forward.evaluate(zero_modulation)
        baseline_curve[:,:,start:stop]=-1000*base_log[:,:,1:]/run.workflow.LOG_FLUX_PER_MAG
        for fi in range(filter_count):
            dense_model.tile_thickness_by_id[:,fi,:]=interp_history(fitted_history,targets[fi])
        joint_log,_=dense_forward.evaluate(joint_modulation)
        joint_curve[:,:,start:stop]=-1000*joint_log[:,:,1:]/run.workflow.LOG_FLUX_PER_MAG
    visit_index=np.searchsorted(dense_days,pass_days)
    np.testing.assert_allclose(dense_days[visit_index],pass_days,rtol=0,atol=1e-10)
    joint_visit=joint_curve[np.arange(source_count)[:,None,None],np.arange(filter_count)[None,:,None],visit_index[None,:,:]]
    joint_error=np.max(np.abs((joint_visit-joint_no_roll)[valid]))
    if joint_error>2e-5:
        raise ValueError(f"Roll transfer changed joint-fit residuals; max difference={joint_error} mmag")
    base_visit=baseline_curve[np.arange(source_count)[:,None,None],np.arange(filter_count)[None,:,None],visit_index[None,:,:]]
    base_transfer_error=np.max(np.abs(((observed-(base_visit+roll_mag*1000))-(roll_adjusted_data-base_visit))[valid]))
    if base_transfer_error>1e-10:
        raise ValueError(f"Roll transfer changed baseline residuals by {base_transfer_error} mmag")
    base_location=profile(base_visit)
    base_plot=baseline_curve+base_location[:,:,None]
    joint_plot=joint_curve+joint_location[:,:,None]
    # Figure styling and layout mirror the original paper panel: two columns,
    # four filter rows, shared axes, one shared y label, and full history span.
    reset_dates=mdates.date2num(Time(reference_mjd+reset_days,format="mjd",scale="utc").to_datetime())
    dense_dates=mdates.date2num(Time(reference_mjd+dense_days,format="mjd",scale="utc").to_datetime())
    pass_dates=mdates.date2num(Time(reference_mjd+pass_days,format="mjd",scale="utc").to_datetime())
    postroll_index=int(np.flatnonzero(np.asarray(model.pass_zero_point_group_ids)>0)[0])
    postroll_date=float(np.median(pass_dates[:,postroll_index]))
    active_source=active.reshape(source_count,filter_count)
    limits=[]; data_summaries=[]; model_summaries=[]
    for fi in range(filter_count):
        rows=[]
        for pi in range(len(run.PASSES)):
            use=valid[:,fi,pi]
            if use.any():
                values=roll_adjusted_data[use,fi,pi]
                rows.append((pass_dates[fi,pi],*np.quantile(values,[.16,.5,.84])))
        data_summaries.append(rows)
        bq=np.empty((2,len(dense_days)))
        jq=np.empty((2,len(dense_days)))
        for time_index in range(len(dense_days)):
            use_source=active_source[:,fi]&curve_cohort_valid[:,fi,time_index]
            bq[:,time_index]=np.quantile(base_plot[use_source,fi,time_index],[.16,.84])
            jq[:,time_index]=np.quantile(joint_plot[use_source,fi,time_index],[.16,.84])
        limits.extend([row[1] for row in rows]+[row[3] for row in rows])
        limits.extend([float(bq[0].min()),float(bq[1].max()),float(jq[0].min()),float(jq[1].max())])
    if min(limits) <= -60 or max(limits) >= 50:
        raise ValueError(
            "Figure 5 data or full-span model curves exceed the original shared y-axis range"
        )
    figure,axes=plt.subplots(4,2,figsize=(FIGURE_WIDTH_IN,6.2),sharex=True,sharey=True); axes=axes.ravel()
    blue,orange="#1f77b4","#d95f02"; segments=[dense_days<reset_days[0],(dense_days>=reset_days[0])&(dense_days<reset_days[1]),dense_days>=reset_days[1]]
    for fi,(band,axis) in enumerate(zip(run.FILTERS,axes,strict=True)):
        axis.axhline(0,color="0.4",linewidth=.6)
        for reset_index,date in enumerate(reset_dates):
            style=":" if reset_index==0 else "-."
            axis.axvline(date,color="0.35",linestyle=style,linewidth=.8)
        axis.axvline(postroll_date,color="0.45",linestyle="--",linewidth=.8)
        for date,p16,median,p84 in data_summaries[fi]:
            axis.errorbar(date,median,yerr=[[median-p16],[p84-median]],fmt="o",color="black",markersize=2.2,elinewidth=.45,capsize=.8,zorder=5)
        base_median=np.empty(len(dense_days))
        joint_quantiles=np.empty((3,len(dense_days)))
        for time_index in range(len(dense_days)):
            use_source=active_source[:,fi]&curve_cohort_valid[:,fi,time_index]
            base_median[time_index]=np.median(base_plot[use_source,fi,time_index])
            joint_quantiles[:,time_index]=np.quantile(
                joint_plot[use_source,fi,time_index],[.16,.5,.84]
            )
        j16,jmedian,j84=joint_quantiles
        for segment in segments:
            use=segment
            if not use.any(): continue
            x=dense_dates[use]
            axis.plot(x,base_median[use],color=blue,linestyle="--",linewidth=1.4)
            axis.fill_between(x,j16[use],j84[use],color=orange,alpha=.16,linewidth=0)
            axis.plot(x,jmedian[use],color=orange,linewidth=1.5)
        for pi in range(len(run.PASSES)):
            cohort=valid[:,fi,pi]
            if cohort.any():
                time_index=int(np.searchsorted(dense_days,pass_days[fi,pi]))
                data_median=float(np.median(roll_adjusted_data[cohort,fi,pi]))
                model_median=float(np.median(joint_plot[cohort,fi,time_index]))
                model_summaries.append((band,run.PASSES[pi],data_median,model_median))
        axis.set_title(band, fontsize=8)
        axis.set_ylim(-60, 50)
        axis.set_yticks([-50, -25, 0, 25])
        axis.tick_params(labelsize=TICK_FONT_SIZE)
        axis.grid(axis="y", color="#d5dfeb", linewidth=0.8)
    figure.supylabel("Median-centered magnitude change (mmag)", fontsize=PLOT_FONT_SIZE)
    figure.supxlabel("UTC date", fontsize=PLOT_FONT_SIZE, y=0.035)
    handles=[
        Line2D([0],[0],marker="o",linestyle="none",color="black",markersize=3,label="Observed CAR201"),
        Line2D([0],[0],color=blue,linestyle="--",linewidth=1.0,label="Base TMM + RCS-only ice"),
        Line2D([0],[0],color=orange,linewidth=1.0,label="Joint fit (16–84% band)"),
        Line2D([0],[0],color="0.35",linestyle=":",linewidth=.8,label="Decon2 reset"),
        Line2D([0],[0],color="0.35",linestyle="-.",linewidth=.8,label="Decon3 reset"),
        Line2D([0],[0],color="0.45",linestyle="--",linewidth=.8,label="Post-roll transition"),
    ]
    figure.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=3,
                  frameon=False,fontsize=LEGEND_FONT_SIZE,columnspacing=.8,handlelength=1.5)
    ticks=mdates.date2num(Time(reference_mjd+np.asarray([0.,20.,40.,60.]),format="mjd",scale="utc").to_datetime())
    for axis in axes:
        axis.set_xticks(ticks); axis.set_xlim(ticks[0]-.5,ticks[-1]+2.); axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d",tz=timezone.utc))
    figure.subplots_adjust(left=.105,right=.985,top=.87,bottom=.09,hspace=.30,wspace=.16)
    png=Path(__file__).resolve().parent/"wfi02_full_car201_model_data.png"; figure.savefig(png,dpi=FIGURE_DPI); figure.savefig(png.with_suffix(".pdf"),dpi=FIGURE_DPI); plt.close(figure)
    first_postroll=int(np.flatnonzero(np.asarray(model.pass_zero_point_group_ids)>0)[0])
    print("F062 post-roll data shift (mmag):",float(-1000*roll_mag[0,0,first_postroll]))
    print("Joint fit roll-transfer residual check (mmag):",float(joint_error))
    print("Baseline roll-transfer residual check (mmag):",float(base_transfer_error))
    f062_visit_differences=[abs(data-model) for band,_,data,model in model_summaries if band=="F062"]
    print("F062 max absolute visit-median data/model difference (mmag):",float(max(f062_visit_differences)))
    print("Saved WFI02 Figure 5",png,png.with_suffix(".pdf"))


def plot_complex_reflectance() -> tuple[Path, Path]:
    sys.path[:0] = [
        str(ROOT / "wfi_pipeline"),
        str(ROOT / "archive/exploratory/legacy_root_scripts"),
    ]
    import fit_car201_zane_star_reflectance_modulation as fitter
    import plot_joint_bundle_clean_complex_reflection as plotter

    bundle = plotter.load_bundle(
        STATS / "production/wavelength_response_bundle/wfi_all_2nm_rank16_modulated.npz"
    )
    wavelengths = np.asarray(bundle["modulation_wavelengths_nm"], dtype=float)
    base_n, base_k = fitter.production.hgcdte_reference_optical_constants(wavelengths)
    spectral_basis = fitter.production.spectral_modulation_basis(wavelengths)
    tile_ids, unmodulated, modulated = plotter.tile_reflections(
        bundle, DETECTOR, wavelengths, base_n, base_k, spectral_basis
    )
    selected = int(np.flatnonzero(tile_ids == TILE)[0])
    base = unmodulated[selected:selected+1]
    joint = modulated[selected:selected+1]
    reference_phase = plotter.phase_reference(base)
    base_phase = plotter.detailed_plotting.aligned_phase(base, reference_phase)[0]
    joint_phase = plotter.detailed_plotting.aligned_phase(joint, reference_phase)[0]
    reciprocal_wave = 1000.0 / wavelengths
    order = np.argsort(reciprocal_wave)

    figure, axes = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTH_IN, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9], "hspace": 0.08},
    )
    axes[0].plot(reciprocal_wave[order], np.abs(base[0, order]), color="#1f77b4",
                 linestyle="--", linewidth=1.1, label="Unmodulated ice-free TMM")
    axes[0].plot(reciprocal_wave[order], np.abs(joint[0, order]), color="#d95f02",
                 linewidth=1.1, label="Modulated ice-free reflection")
    axes[1].plot(reciprocal_wave[order], base_phase[order], color="#1f77b4",
                 linestyle="--", linewidth=1.1)
    axes[1].plot(reciprocal_wave[order], joint_phase[order], color="#d95f02", linewidth=1.1)
    led_centers_nm = np.asarray([623.521, 852.803, 1038.585, 1269.758, 1481.241, 1670.169])
    led_colors = plt.get_cmap("Dark2")(np.linspace(0.0, 1.0, len(led_centers_nm)))
    reciprocal_leds = 1000.0 / led_centers_nm
    for led_number, (position, color) in enumerate(zip(reciprocal_leds, led_colors, strict=True), start=1):
        if not reciprocal_wave.min() <= position <= reciprocal_wave.max():
            continue
        for axis in axes:
            axis.axvline(position, color=color, linestyle=":", linewidth=0.7, zorder=0)
        axes[0].text(
            position, 0.98, f"LED {led_number}",
            transform=axes[0].get_xaxis_transform(), rotation=90,
            ha="right", va="top", fontsize=6.5, fontweight="bold",
            color="black", bbox={"facecolor": "white", "edgecolor": color,
                                  "linewidth": 0.6, "alpha": 1.0, "pad": 1.0},
        )
    axes[0].set_ylabel(r"Amplitude $|r|$", fontsize=PLOT_FONT_SIZE)
    axes[1].set_ylabel("Phase (rad)", fontsize=PLOT_FONT_SIZE)
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top = axes[0].secondary_xaxis(
        "top", functions=(plotter.reciprocal_wavelength, plotter.reciprocal_wavelength)
    )
    ticks = np.asarray([2300.0, 1700.0, 1300.0, 1000.0, 800.0, 600.0, 500.0])
    top.set_xticks(ticks[(ticks >= wavelengths.min()) & (ticks <= wavelengths.max())])
    top.set_xlabel("Wavelength (nm)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                  ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE,
                  columnspacing=1.0, handlelength=1.7)
    figure.subplots_adjust(left=0.13, right=0.985, top=0.83, bottom=0.15, hspace=0.10)
    return save_figure(figure, "wfi02_tile135_complex_reflectance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", choices=("all", "1", "2", "3", "4", "5", "6"),
        default="all", help="Figure to regenerate; default regenerates all six",
    )
    args = parser.parse_args()
    generators = {
        "1": plot_spectral_coverage,
        "2": plot_raw_corrected,
        "3": plot_ice_history,
        "4": plot_rcs_model_data,
        "5": plot_car201_model_data,
        "6": plot_complex_reflectance,
    }
    if args.figure == "all":
        for generator in generators.values():
            generator()
    else:
        generators[args.figure]()


if __name__ == "__main__":
    main()
