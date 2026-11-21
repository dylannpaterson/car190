"""Regenerate the WFI methodology paper's seven figures in this directory.

Run ``python docs/figures/generate_paper_figures.py --figure all`` to rebuild
the complete set, or pass a figure number from 1 through 7 to rebuild one plot.
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
DEFAULT_RUN_ROOT = ROOT / "20261117/production/pipeline_runs/relative_response"
DEFAULT_INPUT_ROOT = ROOT / "20261117"
STATS = DEFAULT_RUN_ROOT / "corrected_statistics_4h_thinned"
PRODUCTION_ROOT = DEFAULT_RUN_ROOT / "production"
INPUT_ROOT = DEFAULT_INPUT_ROOT
PINNED_LED_QUADRATURE = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle/"
    "eric_v3_reference/ice_response_model_v3.h5"
)
LED_QUADRATURE_H5 = PINNED_LED_QUADRATURE
MODULATED_BUNDLE = PRODUCTION_ROOT / "wavelength_response_bundle/wfi_all_2nm_rank16_modulated.npz"
TILE = 135
DETECTOR = "WFI02"
LED_FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184")
FILTERS = ("F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213")
RESET_DAYS = np.asarray([12.907397928240743, 38.79784510416666], dtype=float)
RCS_ORIGIN_UTC = "2026-09-12T01:25:50.714+00:00"

sys.path.insert(0, str(ROOT))
from docs.paper_run_paths import resolve_pipeline_run


def warmup_root(production_root: Path, detector: str) -> Path:
    candidates = (
        production_root / detector.lower() / "relative_response_constant_rate_warmup_start40",
        production_root / detector.lower() / "relative_response_constant_rate_warmup_third_decon_start40",
    )
    for candidate in candidates:
        if (candidate / "fits/tile_135/constant_rate_map.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"No completed start-40 constant-rate warmup found for {detector}: {candidates}"
    )


def save_figure(figure, stem: str, dpi: int = FIGURE_DPI) -> tuple[Path, Path]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf, dpi=dpi)
    plt.close(figure)
    print(f"Wrote {png} and {pdf}")
    return png, pdf


def set_utc_date_axis(axis, max_ticks: int = 6) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks, tz=timezone.utc)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=timezone.utc))


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


def raw_corrected_comparison_frame() -> pd.DataFrame:
    if INPUT_ROOT is None:
        input_root = DEFAULT_INPUT_ROOT
    else:
        input_root = INPUT_ROOT

    raw_path = input_root / "raw_statistics/wfi02_tile_log_statistics_256.csv"
    corrected_path = (
        input_root
        / "corrected_statistics/wfi02_tile_log_statistics_256_count_corrected.csv"
    )
    missing = [path for path in (raw_path, corrected_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Raw-versus-corrected Figure 2 needs both source statistics files; "
            f"missing: {missing}"
        )

    columns = [
        "led", "program", "repeat", "group", "file_set_name", "exposure_start_time",
        "days", "tile_id", "n_pair_valid", "sum_log_change",
    ]
    raw = pd.read_csv(raw_path, usecols=columns)
    corrected = pd.read_csv(corrected_path, usecols=columns)
    raw = raw.loc[raw["tile_id"] == TILE].copy()
    corrected = corrected.loc[corrected["tile_id"] == TILE].copy()
    keys = ["led", "program", "repeat", "group", "file_set_name", "tile_id"]
    frame = raw.merge(
        corrected,
        on=keys,
        how="outer",
        suffixes=("_raw", "_corrected"),
        indicator=True,
        validate="one_to_one",
    )
    if not frame["_merge"].eq("both").all():
        raise ValueError("Raw and corrected source statistics do not have matching groups")
    np.testing.assert_array_equal(
        frame["exposure_start_time_raw"].to_numpy(),
        frame["exposure_start_time_corrected"].to_numpy(),
    )
    np.testing.assert_allclose(
        frame["days_raw"].to_numpy(float), frame["days_corrected"].to_numpy(float),
        rtol=0, atol=1e-12,
    )
    np.testing.assert_array_equal(
        frame["n_pair_valid_raw"].to_numpy(int),
        frame["n_pair_valid_corrected"].to_numpy(int),
    )
    raw_mean = (
        frame["sum_log_change_raw"].to_numpy(float)
        / frame["n_pair_valid_raw"].to_numpy(float)
    ) * 1000.0
    corrected_mean = (
        frame["sum_log_change_corrected"].to_numpy(float)
        / frame["n_pair_valid_corrected"].to_numpy(float)
    ) * 1000.0
    return pd.DataFrame({
        "group": frame["group"],
        "file_set_name": frame["file_set_name"],
        "program_corrected": frame["program"],
        "repeat_corrected": frame["repeat"],
        "led_corrected": frame["led"],
        "exposure_start_time_corrected": frame["exposure_start_time_corrected"],
        "days_corrected": frame["days_corrected"],
        "n_pair_valid_raw": frame["n_pair_valid_raw"],
        "raw_mean_milli": raw_mean,
        "n_pair_valid_corrected": frame["n_pair_valid_corrected"],
        "corrected_mean_milli": corrected_mean,
        "correction_delta_milli": corrected_mean - raw_mean,
    })


def plot_raw_corrected() -> tuple[Path, Path]:
    from astropy.time import Time

    frame = raw_corrected_comparison_frame()
    frame["utc"] = pd.to_datetime(
        frame["exposure_start_time_corrected"], utc=True, format="mixed"
    ).dt.tz_convert(None)
    reset_dates = [pd.Timestamp("2026-09-24T23:12:29.895"),
                   pd.Timestamp("2026-10-20T20:34:44.531")]
    figure, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 3.9), sharex=True, sharey=True)
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
        set_utc_date_axis(axis, max_ticks=4)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    first_date = frame["utc"].min()
    last_date = frame["utc"].max()
    axes.flat[0].set_xlim(first_date - pd.Timedelta(days=1), last_date + pd.Timedelta(days=1))
    print(f"Figure 2 raw/corrected input coverage (UTC): {first_date} to {last_date}")
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

    production = PRODUCTION_ROOT / "wfi02"
    warm_path = warmup_root(PRODUCTION_ROOT, DETECTOR) / "fits/tile_135/constant_rate_map.npz"
    rcs_only_path = production / "relative_response/fits/tile_135/posterior_samples.npz"
    joint_path = production / "relative_response/experiments/car190_amber_joint_2day_reader_compatible_current_gray_v2/joint_weekly_ice_fit.npz"
    warm_days, warm_d, warm_r, resets = load_tile_history(warm_path, constant_rate=True)
    rcs_only_days, rcs_only_d, rcs_only_r, rcs_only_resets = load_tile_history(rcs_only_path)
    np.testing.assert_allclose(rcs_only_resets, resets, rtol=0, atol=1e-10)
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

    history_start = min(float(warm_days.min()), float(rcs_only_days.min()), float(joint_days.min()))
    history_end = max(float(warm_days.max()), float(rcs_only_days.max()), float(joint_days.max()))

    colors = {"initial": "#cb78a5", "rcs_only": "#177db5", "joint": "#db5b00"}
    series = [
        (warm_days, warm_d, warm_r, colors["initial"], "--",
         "Constant-rate preliminary fit"),
        (rcs_only_days, rcs_only_d, rcs_only_r, colors["rcs_only"], "-", "RCS-only TMM"),
        (joint_days, joint_d, joint_r, colors["joint"], "-", "Joint RCS + stellar"),
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
        set_utc_date_axis(axis)
    axes[0].set_ylabel("Thickness (nm)")
    axes[0].set_ylim(-5, 255)
    axes[1].set_ylabel("Rate (nm/day)")
    axes[1].set_xlabel("UTC date")
    axes[1].set_ylim(6, max(warm_r.max(), rcs_only_r.max(), joint_r.max()) + 1)
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
    axes[1].set_xlim(dates([history_start - 1, history_end + 1]))
    print(
        "Figure 3 fitted-history coverage (UTC):",
        origin + timedelta(days=history_start),
        "to",
        origin + timedelta(days=history_end),
    )
    figure.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.12, hspace=0.10)
    return save_figure(figure, "wfi02_representative_ice_history")


def plot_rcs_model_data() -> tuple[Path, Path]:
    from astropy.time import Time
    import h5py

    sys.path[:0] = [str(ROOT / "wfi_pipeline"),
                    str(ROOT / "archive/exploratory/legacy_root_scripts")]
    import fit_wfi_tmm_production as production
    import plot_joint_amber_car190_detailed as plotter
    from experiment_joint_wfi01_fixed_ice import LedModel
    from support_aligned_modulation import effective_area_support

    relative_root = PRODUCTION_ROOT / "wfi02/relative_response"
    warm_root = warmup_root(PRODUCTION_ROOT, DETECTOR)
    warm_path = warm_root / "fits/tile_135/constant_rate_map.npz"
    rcs_only_path = relative_root / "fits/tile_135/tile_group_predictions.csv"
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
    rcs_only = pd.read_csv(rcs_only_path).sort_values("group")
    np.testing.assert_array_equal(rcs_only["led"].to_numpy(int), warm_led)
    np.testing.assert_allclose(rcs_only["days"], warm_days, rtol=0, atol=1e-10)
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
        saved_joint_chi2 = float(np.asarray(joint["car190_chi2"]).item())
    area_path = ROOT / "external_reference_data/roman_effarea_v8_20240301/" \
        "Roman_effarea_v8_SCA02_20240301.ecsv"
    joint_summary = json.loads((joint_root / "summary.json").read_text())
    with h5py.File(LED_QUADRATURE_H5, "r") as archive:
        led_wavelength_nm = np.asarray(archive["led_bands/wavelength_nm"][()], dtype=float)
        led_weights = np.asarray(archive["led_bands/weight"][()], dtype=float)
    led_model = LedModel(
        effective_area_support(area_path), detector=DETECTOR,
        relative_response_root=relative_root,
        optical_cache_root=Path(joint_summary["optical_cache_root"]),
        led_wavelength_nm=led_wavelength_nm, led_weights=led_weights,
    )
    joint_rows, joint_residual = plotter.predict_car190(led_model, theta, times, thickness)
    np.testing.assert_allclose(
        np.sum(joint_residual**2), saved_joint_chi2, rtol=2e-7, atol=2e-4
    )
    likelihood_rows = []
    for led_number, tile_items in enumerate(led_model.bands, start=1):
        rcs_chi2 = 0.0
        joint_model_chi2 = 0.0
        group_count = 0
        for tile, item in enumerate(tile_items):
            observed_groups = np.asarray(item[7], dtype=float)
            errors = np.asarray(item[8], dtype=float)
            days = np.asarray(item[9], dtype=float)
            rcs_prediction = np.asarray(item[6], dtype=float)
            rows = joint_rows.loc[
                (joint_rows["led"] == led_number) & (joint_rows["tile"] == tile)
            ].sort_values("day")
            np.testing.assert_allclose(rows["day"].to_numpy(float), days, rtol=0, atol=1e-10)
            np.testing.assert_allclose(rows["observed"].to_numpy(float), observed_groups, rtol=0, atol=1e-10)
            joint_prediction = rows["predicted"].to_numpy(float)
            rcs_chi2 += float(np.sum(((rcs_prediction - observed_groups) / errors) ** 2))
            joint_model_chi2 += float(np.sum(((joint_prediction - observed_groups) / errors) ** 2))
            group_count += len(days)
        likelihood_rows.append({
            "led": led_number,
            "filter": LED_FILTERS[led_number - 1],
            "n_tile_groups": group_count,
            "rcs_only_weighted_chi2_per_group": rcs_chi2 / group_count,
            "joint_weighted_chi2_per_group": joint_model_chi2 / group_count,
            "delta_mean_log_likelihood_joint_minus_rcs_only_per_group": (
                0.5 * (rcs_chi2 - joint_model_chi2) / group_count
            ),
        })
    np.testing.assert_allclose(
        sum(row["joint_weighted_chi2_per_group"] * row["n_tile_groups"] for row in likelihood_rows),
        saved_joint_chi2, rtol=2e-7, atol=2e-4,
    )
    likelihood_frame = pd.DataFrame(likelihood_rows)
    likelihood_path = FIGURES / "wfi02_rcs_likelihood_by_led.csv"
    likelihood_frame.to_csv(likelihood_path, index=False, float_format="%.8g")
    rcs_only_total_chi2 = float(sum(
        row["rcs_only_weighted_chi2_per_group"] * row["n_tile_groups"]
        for row in likelihood_rows
    ))
    print(
        "Total WFI02 RCS weighted chi-square and joint-minus-RCS-only log likelihood:",
        rcs_only_total_chi2, saved_joint_chi2,
        0.5 * (rcs_only_total_chi2 - saved_joint_chi2),
    )
    print("WFI02 per-LED RCS likelihood comparison, summed over 256 tiles:")
    print(likelihood_frame.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print("Wrote", likelihood_path)
    joint_tile = joint_rows.loc[joint_rows["tile"] == TILE]
    joint_prediction = np.empty_like(warm_thickness)
    observed = rcs_only["observed_log_change"].to_numpy(float)
    sigma = np.empty_like(observed)
    for led in range(1, 7):
        rows = joint_tile.loc[joint_tile["led"] == led].sort_values("day")
        use = rcs_only["led"].to_numpy(int) == led
        np.testing.assert_allclose(rows["day"], warm_days[use], rtol=0, atol=1e-10)
        joint_prediction[use] = rows["predicted"].to_numpy(float)
        sigma[use] = np.asarray(led_model.bands[led - 1][TILE][8], dtype=float)
        np.testing.assert_allclose(led_model.bands[led - 1][TILE][7], observed[use], rtol=0, atol=1e-10)

    reference = Time(RCS_ORIGIN_UTC.removesuffix("+00:00"), format="isot", scale="utc").mjd

    def dates(days):
        return mdates.date2num(Time(reference + np.asarray(days), format="mjd", scale="utc").to_datetime())

    colors = {"warm": "#cb78a5", "rcs_only": "#177db5", "joint": "#db5b00"}
    figure, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, 4.2), sharex=True, sharey=True,
                                gridspec_kw={"hspace": 0.35, "wspace": 0.20})
    coverage_start = float(warm_days.min())
    coverage_end = float(warm_days.max())
    values = np.concatenate([observed, warm_prediction, rcs_only["model_log_change"].to_numpy(float), joint_prediction]) * 1000.0
    low, high = float(values.min()), float(values.max())
    margin = 0.05 * max(high-low, 1.0)
    for led, axis in enumerate(axes.flat, start=1):
        use = rcs_only["led"].to_numpy(int) == led
        days = warm_days[use]
        axis.errorbar(dates(days), observed[use]*1000, yerr=sigma[use]*1000,
                      fmt="none", ecolor="0.18", elinewidth=0.55,
                      capsize=0.35, zorder=8,
                      label="Corrected RCS group means (error bars)" if led == 1 else None)
        masks = [days < RESET_DAYS[0], (days >= RESET_DAYS[0]) & (days < RESET_DAYS[1]), days >= RESET_DAYS[1]]
        for segment, mask in enumerate(masks):
            if not np.any(mask):
                continue
            axis.plot(dates(days[mask]), warm_prediction[use][mask]*1000,
                      color=colors["warm"], linestyle="--", linewidth=1.55, zorder=4,
                      label="Constant-rate preliminary fit" if led == 1 and segment == 0 else None)
            axis.plot(dates(days[mask]), joint_prediction[use][mask]*1000,
                      color=colors["joint"], linewidth=1.8, zorder=5,
                      label="Joint RCS + stellar" if led == 1 and segment == 0 else None)
            axis.plot(dates(days[mask]), rcs_only.loc[use, "model_log_change"].to_numpy()[mask]*1000,
                      color=colors["rcs_only"], linestyle=(0, (4, 2)),
                      linewidth=1.45, zorder=6,
                      label="RCS-only TMM" if led == 1 and segment == 0 else None)
        for reset_index, reset in enumerate(RESET_DAYS):
            style = ":" if reset_index == 0 else "-."
            axis.axvline(dates([reset])[0], color="#263c63", linestyle=style, linewidth=0.8)
        axis.set_title(f"LED {led} ({LED_FILTERS[led-1]})")
        axis.set_ylim(low-margin,high+margin)
        axis.grid(color="#d5dfeb",linewidth=0.8)
    figure.supylabel(r"Relative log response ($10^{-3}$)", fontsize=PLOT_FONT_SIZE)
    figure.supxlabel("UTC date", fontsize=PLOT_FONT_SIZE, y=0.035)
    date_limits = dates([coverage_start - 1.0, coverage_end + 1.0])
    for axis in axes.flat:
        axis.set_xlim(date_limits[0], date_limits[-1])
        set_utc_date_axis(axis, max_ticks=4)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    coverage_dates = Time(reference + np.asarray([coverage_start, coverage_end]), format="mjd", scale="utc").to_datetime()
    print(f"Figure 4 corrected RCS/fits coverage (UTC): {coverage_dates[0]} to {coverage_dates[1]}")
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
    from fit_amber_wfi01_student_t import NU, profile_student_locations
    from reader_compatible_joint_forward import ReaderCompatiblePhotometryForward
    import run_amber_student_t_detectors as run
    import run_joint_car190_amber_inverse_wavelength_maxent as joint_fit

    fit_dir = PRODUCTION_ROOT / "wfi02/relative_response/experiments/" \
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
    no_ice_location=profile(np.zeros_like(joint_no_roll))
    joint_location=profile(joint_no_roll)

    model.reader_gray_log_by_tile=np.asarray(summary["reader_gray_log_offsets_by_tile"],float)[:,np.searchsorted(reset_days,pass_days,side="right")]
    dense_start=0.0
    dense_end=float(history_times.max())
    dense_days=np.unique(
        np.r_[np.arange(dense_start,dense_end+1e-10,.25),pass_days.ravel(),dense_end]
    )
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
    baseline_ice_effect=np.empty_like(baseline_curve)
    joint_ice_effect=np.empty_like(baseline_curve)
    baseline_ice_reference=None
    joint_ice_reference=None
    log_flux_per_mag=float(run.workflow.LOG_FLUX_PER_MAG)
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
        baseline_curve[:,:,start:stop]=-1000*base_log[:,:,1:]/log_flux_per_mag
        gray_by_tile_time=dense_model.reader_gray_log_by_tile.copy()
        dense_model.reader_gray_log_by_tile=np.zeros_like(gray_by_tile_time)
        base_absolute_log=dense_forward.evaluate_absolute(zero_modulation)
        baseline_ice_effect[:,:,start:stop]=-1000*base_absolute_log[:,:,1:]/log_flux_per_mag
        base_reference=-1000*base_absolute_log[:,:,0]/log_flux_per_mag
        if baseline_ice_reference is None:
            baseline_ice_reference=base_reference
        else:
            np.testing.assert_allclose(base_reference,baseline_ice_reference,rtol=0,atol=1e-10)
        dense_model.reader_gray_log_by_tile=gray_by_tile_time
        for fi in range(filter_count):
            dense_model.tile_thickness_by_id[:,fi,:]=interp_history(fitted_history,targets[fi])
        joint_log,_=dense_forward.evaluate(joint_modulation)
        joint_curve[:,:,start:stop]=-1000*joint_log[:,:,1:]/log_flux_per_mag
        dense_model.reader_gray_log_by_tile=np.zeros_like(gray_by_tile_time)
        joint_absolute_log=dense_forward.evaluate_absolute(joint_modulation)
        joint_ice_effect[:,:,start:stop]=-1000*joint_absolute_log[:,:,1:]/log_flux_per_mag
        joint_reference=-1000*joint_absolute_log[:,:,0]/log_flux_per_mag
        if joint_ice_reference is None:
            joint_ice_reference=joint_reference
        else:
            np.testing.assert_allclose(joint_reference,joint_ice_reference,rtol=0,atol=1e-10)
        dense_model.reader_gray_log_by_tile=gray_by_tile_time
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
    no_ice_residual=roll_adjusted_data-no_ice_location[:,:,None]
    base_residual=roll_adjusted_data-base_visit-base_location[:,:,None]
    joint_residual=roll_adjusted_data-joint_no_roll-joint_location[:,:,None]
    source_tile_ids=np.asarray(model.source_tile_ids,dtype=int)
    reader_gray=np.asarray(model.reader_gray_log_by_tile,dtype=float)
    gray_log_delta=np.zeros_like(joint_no_roll)
    for fi in range(filter_count):
        reference_tiles=source_tile_ids[:,fi,0]
        reference_gray=reader_gray[reference_tiles,fi,0]
        for pi in range(len(run.PASSES)):
            current_tiles=source_tile_ids[:,fi,pi]
            gray_log_delta[:,fi,pi]=reader_gray[current_tiles,fi,pi]-reference_gray
    gray_magnitude_delta=-1000*gray_log_delta/log_flux_per_mag
    zero_ice_observed=(
        roll_adjusted_data
        - joint_location[:,:,None]
        - gray_magnitude_delta
        + joint_ice_reference[:,:,None]
    )
    joint_ice_at_visits=joint_ice_effect[
        np.arange(source_count)[:,None,None],
        np.arange(filter_count)[None,:,None],
        visit_index[None,:,:],
    ]
    zero_ice_residual=zero_ice_observed-joint_ice_at_visits
    zero_ice_transfer_error=float(
        np.max(np.abs(zero_ice_residual[valid]-joint_residual[valid]))
    )
    if zero_ice_transfer_error>2e-5:
        raise ValueError(
            "Zero-ice data transformation changed joint residuals; "
            f"max difference={zero_ice_transfer_error} mmag"
        )
    def robust_nmad(values):
        center=float(np.median(values))
        return float(1.4826*np.median(np.abs(values-center)))
    def student_nll_per_measurement(residual,filter_index):
        use=valid[:,filter_index,:]
        values=residual[:,filter_index,:][use]
        sigma=scales[:,filter_index,:][use]
        nll=np.log(sigma)+.5*(NU+1.)*np.log1p(values**2/(NU*sigma**2))
        return float(np.mean(nll))
    photometry_metrics=[]
    for fi,band in enumerate(run.FILTERS):
        use=valid[:,fi,:]
        n_measurements=int(use.sum())
        photometry_metrics.append({
            "filter":band,
            "n_stars":int(np.count_nonzero(use.any(axis=1))),
            "n_measurements":n_measurements,
            "nmad_no_ice_mmag":robust_nmad(no_ice_residual[:,fi,:][use]),
            "nmad_rcs_only_tmm_mmag":robust_nmad(base_residual[:,fi,:][use]),
            "nmad_joint_mmag":robust_nmad(joint_residual[:,fi,:][use]),
            "mean_student_nll_no_ice":student_nll_per_measurement(no_ice_residual,fi),
            "mean_student_nll_rcs_only":student_nll_per_measurement(base_residual,fi),
            "mean_student_nll_joint":student_nll_per_measurement(joint_residual,fi),
        })
    for row in photometry_metrics:
        row["delta_mean_log_likelihood_joint_minus_rcs_only_per_measurement"]=(
            row["mean_student_nll_rcs_only"]-row["mean_student_nll_joint"]
        )
    photometry_metrics_frame=pd.DataFrame(photometry_metrics)
    photometry_metrics_path=FIGURES/"wfi02_car201_residual_metrics.csv"
    photometry_metrics_frame.to_csv(photometry_metrics_path,index=False,float_format="%.8g")
    print("WFI02 CAR201 residual metrics (mmag NMAD; mean Student-t NLL per measurement):")
    print(photometry_metrics_frame.to_string(index=False,float_format=lambda value:f"{value:.5g}"))
    total_photometry_delta_loglike=float(np.sum(
        photometry_metrics_frame[
            "delta_mean_log_likelihood_joint_minus_rcs_only_per_measurement"
        ]
        * photometry_metrics_frame["n_measurements"]
    ))
    print("Total WFI02 CAR201 joint-minus-RCS-only log likelihood:",total_photometry_delta_loglike)
    print("Wrote",photometry_metrics_path)
    # Figure styling and layout mirror the original paper panel: two columns,
    # four filter rows, shared axes, one shared y label, and full history span.
    reset_dates=mdates.date2num(Time(reference_mjd+reset_days,format="mjd",scale="utc").to_datetime())
    dense_dates=mdates.date2num(Time(reference_mjd+dense_days,format="mjd",scale="utc").to_datetime())
    pass_dates=mdates.date2num(Time(reference_mjd+pass_days,format="mjd",scale="utc").to_datetime())
    observed_span=Time(
        reference_mjd+np.asarray([pass_days.min(),pass_days.max()]),
        format="mjd",scale="utc",
    ).to_datetime()
    model_span=Time(
        reference_mjd+np.asarray([dense_days.min(),dense_days.max()]),
        format="mjd",scale="utc",
    ).to_datetime()
    print("Figure 5 CAR201 visit coverage (UTC):",observed_span[0],"to",observed_span[1])
    print("Figure 5 joint-fit model coverage (UTC):",model_span[0],"to",model_span[1])
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
        axis.set_ylim(50, -60)
        axis.set_yticks([-50, -25, 0, 25])
        axis.tick_params(labelsize=TICK_FONT_SIZE)
        axis.grid(axis="y", color="#d5dfeb", linewidth=0.8)
    figure.supylabel(
        "Median-centered magnitude change (mmag; brighter upward)",
        fontsize=PLOT_FONT_SIZE,
    )
    figure.supxlabel("UTC date", fontsize=PLOT_FONT_SIZE, y=0.035)
    handles=[
        Line2D([0],[0],marker="o",linestyle="none",color="black",markersize=3,label="Observed CAR201"),
        Line2D([0],[0],color=blue,linestyle="--",linewidth=1.0,label="RCS-only TMM"),
        Line2D([0],[0],color=orange,linewidth=1.0,label="Joint RCS + stellar (16–84% band)"),
        Line2D([0],[0],color="0.35",linestyle=":",linewidth=.8,label="Decon2 reset"),
        Line2D([0],[0],color="0.35",linestyle="-.",linewidth=.8,label="Decon3 reset"),
        Line2D([0],[0],color="0.45",linestyle="--",linewidth=.8,label="Post-roll transition"),
    ]
    figure.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=3,
                  frameon=False,fontsize=LEGEND_FONT_SIZE,columnspacing=.8,handlelength=1.5)
    plot_start=float(min(dense_days.min(),pass_days.min()))
    plot_end=float(max(dense_days.max(),pass_days.max()))
    date_limits=mdates.date2num(
        Time(
            reference_mjd+np.asarray([plot_start-1.0,plot_end+1.0]),
            format="mjd",scale="utc",
        ).to_datetime()
    )
    for axis in axes:
        axis.set_xlim(date_limits[0],date_limits[-1])
        set_utc_date_axis(axis, max_ticks=4)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    figure.subplots_adjust(left=.105,right=.985,top=.87,bottom=.09,hspace=.30,wspace=.16)
    median_centered_png=FIGURES/"wfi02_full_car201_model_data_median_centered.png"
    figure.savefig(median_centered_png,dpi=FIGURE_DPI)
    figure.savefig(median_centered_png.with_suffix(".pdf"),dpi=FIGURE_DPI)
    plt.close(figure)

    # Companion view: report the optical ice contribution against a clean
    # zero-thickness stack. The observed series is translated into that frame
    # using the fitted star/filter baseline, reset-gray changes, and the joint
    # model's EP3-pass-1 ice response; it is not a direct zero-ice observation.
    zero_ice_data_summaries=[]
    zero_ice_limits=[0.0]
    zero_ice_model_quantiles=[]
    for fi in range(filter_count):
        rows=[]
        for pi in range(len(run.PASSES)):
            use=valid[:,fi,pi]
            if use.any():
                values=zero_ice_observed[use,fi,pi]
                rows.append((pass_dates[fi,pi],*np.quantile(values,[.16,.5,.84])))
        zero_ice_data_summaries.append(rows)
        zero_ice_limits.extend(
            value for row in rows for value in (float(row[1]),float(row[3]))
        )
        base_quantiles=np.empty((2,len(dense_days)))
        joint_quantiles=np.empty((2,len(dense_days)))
        base_median=np.empty(len(dense_days))
        joint_median=np.empty(len(dense_days))
        for time_index in range(len(dense_days)):
            use_source=active_source[:,fi]&curve_cohort_valid[:,fi,time_index]
            base_values=baseline_ice_effect[use_source,fi,time_index]
            joint_values=joint_ice_effect[use_source,fi,time_index]
            base_quantiles[:,time_index]=np.quantile(
                base_values,[.16,.84]
            )
            joint_quantiles[:,time_index]=np.quantile(
                joint_values,[.16,.84]
            )
            base_median[time_index]=np.median(base_values)
            joint_median[time_index]=np.median(joint_values)
        zero_ice_model_quantiles.append(
            (base_quantiles,joint_quantiles,base_median,joint_median)
        )
        zero_ice_limits.extend(base_quantiles.ravel().tolist())
        zero_ice_limits.extend(joint_quantiles.ravel().tolist())

    zero_ice_low=float(min(zero_ice_limits))
    zero_ice_high=float(max(zero_ice_limits))
    zero_ice_margin=.06*max(zero_ice_high-zero_ice_low,1.0)
    tick_step=25.0 if zero_ice_high-zero_ice_low<250.0 else 50.0
    tick_low=np.floor((zero_ice_low-zero_ice_margin)/tick_step)*tick_step
    tick_high=np.ceil((zero_ice_high+zero_ice_margin)/tick_step)*tick_step
    zero_ice_ticks=np.arange(tick_low,tick_high+0.5*tick_step,tick_step)
    ice_figure,ice_axes=plt.subplots(
        4,2,figsize=(FIGURE_WIDTH_IN,6.2),sharex=True,sharey=True
    )
    ice_axes=ice_axes.ravel()
    for fi,(band,axis) in enumerate(zip(run.FILTERS,ice_axes,strict=True)):
        axis.axhline(0.0,color="0.25",linewidth=.9,zorder=2)
        for reset_index,date in enumerate(reset_dates):
            style=":" if reset_index==0 else "-."
            axis.axvline(date,color="0.35",linestyle=style,linewidth=.8)
        axis.axvline(postroll_date,color="0.45",linestyle="--",linewidth=.8)
        for date,p16,median,p84 in zero_ice_data_summaries[fi]:
            axis.errorbar(
                date,median,yerr=[[median-p16],[p84-median]],fmt="o",
                color="black",markersize=2.2,elinewidth=.45,capsize=.8,zorder=5,
            )
        base_quantiles,joint_quantiles,base_median,joint_median=zero_ice_model_quantiles[fi]
        j16,j84=joint_quantiles
        for segment in segments:
            if not segment.any():
                continue
            x=dense_dates[segment]
            axis.plot(
                x,base_median[segment],
                color=blue,linestyle="--",linewidth=1.4,zorder=3,
            )
            axis.fill_between(
                x,j16[segment],j84[segment],color=orange,alpha=.16,
                linewidth=0,zorder=3,
            )
            axis.plot(
                x,joint_median[segment],
                color=orange,linewidth=1.5,zorder=4,
            )
        axis.set_title(band,fontsize=8)
        axis.set_ylim(zero_ice_high+zero_ice_margin,zero_ice_low-zero_ice_margin)
        axis.set_yticks(zero_ice_ticks)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
        axis.grid(axis="y",color="#d5dfeb",linewidth=.8)
    ice_figure.supylabel(
        "Ice-induced magnitude change from zero ice (mmag; brighter upward)",
        fontsize=PLOT_FONT_SIZE,
    )
    ice_figure.supxlabel("UTC date",fontsize=PLOT_FONT_SIZE,y=.035)
    ice_handles=[
        Line2D([0],[0],marker="o",linestyle="none",color="black",markersize=3,
               label="Observed (fit-shifted)"),
        Line2D([0],[0],color=blue,linestyle="--",linewidth=1.0,
               label="RCS-only TMM"),
        Line2D([0],[0],color=orange,linewidth=1.0,
               label="Joint RCS + stellar (16–84%)"),
        Line2D([0],[0],color="0.25",linewidth=.9,label="Zero ice"),
        Line2D([0],[0],color="0.35",linestyle=":",linewidth=.8,label="Decon2"),
        Line2D([0],[0],color="0.35",linestyle="-.",linewidth=.8,label="Decon3"),
        Line2D([0],[0],color="0.45",linestyle="--",linewidth=.8,label="Post-roll"),
    ]
    ice_figure.legend(
        handles=ice_handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=4,
        frameon=False,fontsize=LEGEND_FONT_SIZE,columnspacing=.8,handlelength=1.5,
    )
    for axis in ice_axes:
        axis.set_xlim(date_limits[0],date_limits[-1])
        set_utc_date_axis(axis, max_ticks=4)
        axis.tick_params(labelsize=TICK_FONT_SIZE)
    ice_figure.subplots_adjust(left=.14,right=.985,top=.88,bottom=.11,hspace=.30,wspace=.16)
    zero_ice_png=FIGURES/"wfi02_full_car201_model_data.png"
    zero_ice_pdf=zero_ice_png.with_suffix(".pdf")
    companion_png=FIGURES/"wfi02_car201_ice_effect_zero_ice_reference.png"
    ice_figure.savefig(zero_ice_png,dpi=FIGURE_DPI)
    ice_figure.savefig(zero_ice_pdf,dpi=FIGURE_DPI)
    ice_figure.savefig(companion_png,dpi=FIGURE_DPI)
    ice_figure.savefig(companion_png.with_suffix(".pdf"),dpi=FIGURE_DPI)
    plt.close(ice_figure)
    first_postroll=int(np.flatnonzero(np.asarray(model.pass_zero_point_group_ids)>0)[0])
    print("F062 post-roll data shift (mmag):",float(-1000*roll_mag[0,0,first_postroll]))
    print("Joint fit roll-transfer residual check (mmag):",float(joint_error))
    print("Baseline roll-transfer residual check (mmag):",float(base_transfer_error))
    print("Zero-ice residual transfer check (mmag):",zero_ice_transfer_error)
    f062_visit_differences=[abs(data-model) for band,_,data,model in model_summaries if band=="F062"]
    print("F062 max absolute visit-median data/model difference (mmag):",float(max(f062_visit_differences)))
    print("Saved median-centered diagnostic",median_centered_png,median_centered_png.with_suffix(".pdf"))
    print("Saved WFI02 Figure 5",zero_ice_png,zero_ice_pdf)
    print("Saved zero-ice-reference companion",companion_png,companion_png.with_suffix(".pdf"))


def plot_complex_reflectance() -> tuple[Path, Path]:
    sys.path[:0] = [
        str(ROOT / "wfi_pipeline"),
        str(ROOT / "archive/exploratory/legacy_root_scripts"),
    ]
    import fit_car201_zane_star_reflectance_modulation as fitter
    import plot_joint_bundle_clean_complex_reflection as plotter

    if not MODULATED_BUNDLE.is_file():
        raise FileNotFoundError(
            f"Missing modulated response bundle for the selected run: {MODULATED_BUNDLE}"
        )
    bundle = plotter.load_bundle(MODULATED_BUNDLE)
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
                 linestyle="--", linewidth=1.1, label="RCS-only TMM")
    axes[0].plot(reciprocal_wave[order], np.abs(joint[0, order]), color="#d95f02",
                 linewidth=1.1, label="Joint RCS + stellar")
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


def plot_focal_plane_ice_thickness() -> tuple[Path, Path]:
    from matplotlib.patches import Rectangle

    sys.path[:0] = [
        str(ROOT / "wfi_pipeline"),
        str(ROOT / "archive/exploratory/legacy_root_scripts"),
    ]
    import plot_joint_modulated_ice_trends as trends
    import plot_wfi_full_focal_plane_rates as focal

    bundle = trends.load_joint_modulated_plot_bundle(MODULATED_BUNDLE)
    trends.validate_joint_modulated_bundle(bundle)
    reset_days, _ = trends._reset_metadata(bundle)
    if len(reset_days) < 2:
        raise ValueError("Focal-plane reset maps require Decon2 and Decon3")
    days = np.asarray(bundle["observation_days"], dtype=float).reshape(-1)
    thickness = np.asarray(bundle["tile_ice_thickness"], dtype=float)
    rates = np.asarray(bundle["tile_instantaneous_rate_nm_day"], dtype=float)
    detector_ids = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int)
    tile_rows = np.asarray(bundle["tile_rows"], dtype=int)
    tile_cols = np.asarray(bundle["tile_cols"], dtype=int)

    segment2_nodes = np.flatnonzero((days >= reset_days[0]) & (days < reset_days[1]))
    segment3_nodes = np.flatnonzero(days >= reset_days[1])
    if not segment2_nodes.size or not segment3_nodes.size:
        raise ValueError("Joint fit lacks a fitted node in segment 2 or segment 3")
    segment2_index = int(segment2_nodes[-1])
    segment3_index = int(segment3_nodes[-1])
    endpoint_specs = (
        ("segment2_end", segment2_index),
        ("segment3_end", segment3_index),
    )

    maps = {name: {} for name, _index in endpoint_specs}
    for detector in focal.SCAS:
        selected = np.flatnonzero(detector_ids == detector)
        order = np.argsort(tile_ids[selected])
        selected = selected[order]
        if not np.array_equal(tile_ids[selected], np.arange(256)):
            raise ValueError(f"{detector}: joint bundle does not contain all 256 tiles")
        rows = tile_rows[selected]
        cols = tile_cols[selected]
        for name, time_index in endpoint_specs:
            values = np.full((16, 16), np.nan, dtype=float)
            values[rows, cols] = thickness[selected, time_index]
            if not np.isfinite(values).all():
                raise ValueError(f"{detector}/{name}: incomplete fitted thickness map")
            maps[name][detector] = {"ice_thickness_nm": values}

    all_values = np.concatenate(
        [
            maps[name][detector]["ice_thickness_nm"].ravel()
            for name, _index in endpoint_specs
            for detector in focal.SCAS
        ]
    )
    color_min, color_max = np.percentile(all_values, [1.0, 99.0])
    if not color_min < color_max:
        raise ValueError("Focal-plane thickness maps have degenerate color limits")

    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 3.25))
    figure.subplots_adjust(left=0.015, right=0.91, top=0.96, bottom=0.12, wspace=0.04)
    image = None
    for axis, (panel, (map_name, _time_index)) in zip(
        axes, zip(("a", "b"), endpoint_specs, strict=True), strict=True
    ):
        axis.text(-0.025, 1.015, panel, transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="bottom")
        for detector in focal.SCAS:
            x, y = focal.sca_position(detector)
            values = maps[map_name][detector]["ice_thickness_nm"]
            image = axis.imshow(
                values,
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                extent=(x, x + focal.SCA_SIZE, y, y + focal.SCA_SIZE),
            )
            axis.add_patch(
                Rectangle((x, y), focal.SCA_SIZE, focal.SCA_SIZE,
                          fill=False, edgecolor="black", linewidth=0.65)
            )
            axis.text(x + focal.SCA_SIZE / 2, y + focal.SCA_SIZE / 2, detector,
                      ha="center", va="center", fontsize=5.2, color="white")
            grid_size = values.shape[0]
            for tile_index in range(1, grid_size):
                position = focal.SCA_SIZE * tile_index / grid_size
                axis.plot((x + position, x + position), (y, y + focal.SCA_SIZE),
                          color="white", alpha=0.32, linewidth=0.22)
                axis.plot((x, x + focal.SCA_SIZE), (y + position, y + position),
                          color="white", alpha=0.32, linewidth=0.22)

        axis.set_aspect("equal")
        axis.set_xlim(-1.0, focal.SCA_PITCH * 6 - 1.0)
        axis.set_ylim(
            min(focal.COLUMN_Y_OFFSET) - 1.0,
            focal.SCA_PITCH * 2 + max(focal.COLUMN_Y_OFFSET) + focal.SCA_SIZE + 1.0,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("Roman WFI focal-plane mosaic", fontsize=TICK_FONT_SIZE)
        center_x = focal.SCA_PITCH * 3 - 0.5
        center_y = focal.SCA_PITCH * 2 + focal.COLUMN_Y_OFFSET[2] + 1.2
        axis.plot(center_x, center_y, marker="x", color="#d62728",
                  markersize=5, markeredgewidth=1.0)
        axis.text(center_x + 0.3, center_y + 0.5, "WFI_CEN", color="#d62728",
                  fontsize=5, va="bottom")

    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Fitted ice thickness (nm)", fontsize=PLOT_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    origin = datetime.fromisoformat(RCS_ORIGIN_UTC)
    segment2_date = origin + timedelta(days=float(days[segment2_index]))
    segment3_date = origin + timedelta(days=float(days[segment3_index]))
    print("Focal-plane thickness-map times (UTC):", segment2_date, segment3_date)

    detector_mean_rates = []
    segment_specs = (
        ("before Decon2", (days >= days[0]) & (days < reset_days[0])),
        ("segment 2", (days >= reset_days[0]) & (days < reset_days[1])),
        ("segment 3", days >= reset_days[1]),
    )
    for segment_name, mask in segment_specs:
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            raise ValueError(f"Not enough fitted nodes to summarize {segment_name}")
        first, last = int(indices[0]), int(indices[-1])
        duration = float(days[last] - days[first])
        tile_mean_rates = (thickness[:, last] - thickness[:, first]) / duration
        per_detector = np.asarray(
            [np.mean(tile_mean_rates[detector_ids == detector]) for detector in focal.SCAS],
            dtype=float,
        )
        detector_mean_rates.append(per_detector)
        print(
            f"Mean {segment_name} rate across detector tile means: "
            f"{np.mean(per_detector):.2f} nm/day "
            f"(detector min–max {np.min(per_detector):.2f}–{np.max(per_detector):.2f})"
        )

    for reset_index, reset_day in enumerate(reset_days):
        next_reset = reset_days[reset_index + 1] if reset_index + 1 < len(reset_days) else np.inf
        segment_indices = np.flatnonzero((days >= reset_day) & (days < next_reset))
        segment_days = days[segment_indices]
        for offset in (0.5, 7.0, 20.0):
            target_day = float(reset_day + offset)
            if not segment_days.size or target_day > segment_days[-1]:
                continue
            upper = int(np.searchsorted(segment_days, target_day, side="right"))
            upper = min(max(upper, 1), segment_days.size - 1)
            lower = upper - 1
            fraction = (target_day - segment_days[lower]) / (
                segment_days[upper] - segment_days[lower]
            )
            values = (
                rates[:, segment_indices[lower]] * (1.0 - fraction)
                + rates[:, segment_indices[upper]] * fraction
            )
            per_detector = np.asarray(
                [np.median(values[detector_ids == detector]) for detector in focal.SCAS],
                dtype=float,
            )
            print(
                f"Post-reset rate, reset {reset_index + 2}, +{offset:g} days: "
                f"median detector median {np.median(per_detector):.2f} nm/day "
                f"(detector p10–p90 {np.percentile(per_detector, 10):.2f}–"
                f"{np.percentile(per_detector, 90):.2f})"
            )

    return save_figure(figure, "wfi_full_focal_plane_ice_thickness_segment_ends")


def main(default_figure: str = "all") -> None:
    global FIGURES, STATS, PRODUCTION_ROOT, INPUT_ROOT, LED_QUADRATURE_H5, MODULATED_BUNDLE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", choices=("all", "1", "2", "3", "4", "5", "6", "7"),
        default=default_figure, help="Figure to regenerate; default regenerates all seven",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="pipeline output root or date root; defaults to the completed 20261117 run",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="date root containing raw_statistics/ and corrected_statistics/ for Figure 2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for generated figures and metric CSVs (defaults to docs/figures)",
    )
    parser.add_argument(
        "--led-quadrature-h5",
        type=Path,
        help="override the pinned six-LED quadrature reference HDF5",
    )
    parser.add_argument(
        "--modulated-bundle",
        type=Path,
        help="override the run's modulated wavelength-response bundle for Figure 6",
    )
    args = parser.parse_args()

    paths = resolve_pipeline_run(args.run_root or DEFAULT_RUN_ROOT)
    STATS = paths.statistics_dir
    PRODUCTION_ROOT = paths.production_dir
    INPUT_ROOT = (
        args.input_root.expanduser().resolve()
        if args.input_root is not None
        else paths.input_dir
    )
    print("Using pipeline run:", paths.run_root)

    reference_candidate = (
        PRODUCTION_ROOT
        / "wavelength_response_bundle/eric_v3_reference/ice_response_model_v3.h5"
    )
    LED_QUADRATURE_H5 = (
        args.led_quadrature_h5.expanduser().resolve()
        if args.led_quadrature_h5 is not None
        else reference_candidate if reference_candidate.is_file() else PINNED_LED_QUADRATURE
    )
    MODULATED_BUNDLE = (
        args.modulated_bundle.expanduser().resolve()
        if args.modulated_bundle is not None
        else PRODUCTION_ROOT / "wavelength_response_bundle/wfi_all_2nm_rank16_modulated.npz"
    )
    FIGURES = args.output_dir.expanduser().resolve() if args.output_dir else Path(__file__).resolve().parent
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not LED_QUADRATURE_H5.is_file():
        raise FileNotFoundError(f"Missing LED quadrature reference: {LED_QUADRATURE_H5}")
    generators = {
        "1": plot_spectral_coverage,
        "2": plot_raw_corrected,
        "3": plot_ice_history,
        "4": plot_rcs_model_data,
        "5": plot_car201_model_data,
        "6": plot_complex_reflectance,
        "7": plot_focal_plane_ice_thickness,
    }
    if args.figure == "all":
        for generator in generators.values():
            generator()
    else:
        generators[args.figure]()


if __name__ == "__main__":
    main()
