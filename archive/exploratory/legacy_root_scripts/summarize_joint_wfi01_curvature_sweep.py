"""Summarize in-sample Amber corrections and plot clean reflection by curvature strength."""

import argparse
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mock_fit_wfi_star_reflectance_modulation as mock
import run_amber_student_t_detectors as run
from experiment_joint_wfi01_inverse_wavelength_curvature import weight_label
from plot_wfi01_free_led_complex_reflection import aligned_phase, reciprocal


def parse_weights(value):
    try:
        weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("weights must be comma-separated numbers") from error
    if not weights or any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise argparse.ArgumentTypeError("weights must be finite and positive")
    if len(set(weights)) != len(weights):
        raise argparse.ArgumentTypeError("weights must be unique")
    return weights


def correction_metrics(corrected, magnitude, valid, label):
    comparisons = valid & valid[:, :, :1]
    comparisons[:, :, 0] = False
    delta = corrected - corrected[:, :, :1]
    flux_percent = 100.0 * np.expm1(-run.workflow.LOG_FLUX_PER_MAG * delta)
    records = []
    values = flux_percent[comparisons]
    finite = np.isfinite(values)
    values = values[finite]
    records.append({
        "model": label,
        "filter": "ALL",
        "training_stars": int(len(magnitude)),
        "comparison_count": int(len(values)),
        "within_0p5_percent_count": int(np.count_nonzero(np.abs(values) <= 0.5)),
        "fraction_within_0p5_percent": float(np.mean(np.abs(values) <= 0.5)),
        "rms_relative_change_percent": float(np.sqrt(np.mean(values**2))),
        "median_relative_change_percent": float(np.median(values)),
    })
    for filter_index, band in enumerate(run.FILTERS):
        selected = comparisons[:, filter_index, :]
        values = flux_percent[:, filter_index, :][selected]
        values = values[np.isfinite(values)]
        records.append({
            "model": label,
            "filter": band,
            "training_stars": int(len(magnitude)),
            "comparison_count": int(len(values)),
            "within_0p5_percent_count": int(np.count_nonzero(np.abs(values) <= 0.5)),
            "fraction_within_0p5_percent": float(np.mean(np.abs(values) <= 0.5)) if len(values) else np.nan,
            "rms_relative_change_percent": float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
            "median_relative_change_percent": float(np.median(values)) if len(values) else np.nan,
        })
    return records


def training_magnitudes(source_ids, valid):
    """Reconstruct the exact fitted star/filter/pass cells from Amber photometry."""
    frame = pd.read_parquet(run.AMBER)
    frame = frame.loc[frame["sca"] == "WFI01"].copy()
    epoch_map = {f"r010430300{i}00": f"ep3_pass{i}" for i in range(1, 7)}
    epoch_map.update({f"r010430400{i}00": f"ep4_pass{i}" for i in range(1, 6)})
    frame["pass"] = frame["epoch"].map(epoch_map)
    frame = frame.loc[frame["pass"].notna()].copy()
    frame["source_id"] = frame["source_id"].astype(str)
    indexed = frame.set_index(["source_id", "filter", "pass"])
    if not indexed.index.is_unique:
        raise ValueError("Amber source/filter/pass measurements are not unique")

    magnitude = np.full(valid.shape, np.nan)
    for source_index, filter_index, pass_index in np.argwhere(valid):
        key = (source_ids[source_index], run.FILTERS[filter_index], run.PASSES[pass_index])
        try:
            magnitude[source_index, filter_index, pass_index] = float(indexed.loc[key, "mag_obs"])
        except KeyError as error:
            raise KeyError(f"Missing fitted Amber measurement {key}") from error
    if not np.all(np.isfinite(magnitude[valid])):
        raise ValueError("A fitted Amber measurement is non-finite")
    return magnitude


def modulation_curves(clean, basis, parameters, mode_count):
    ratio = np.exp(
        parameters[:mode_count] @ basis
        + 1j * (parameters[mode_count:2 * mode_count] @ basis)
    )
    return clean * ratio[None, :]


def plot_reflection(detector, wavelengths, curves, output_stem, csv_path,
                   x_limits=None, zoom_y=False, title_suffix=""):
    reciprocal_wave = reciprocal(wavelengths)
    order = np.argsort(reciprocal_wave)
    x = reciprocal_wave[order]
    display_indices = order
    if x_limits is not None:
        selected = (x >= x_limits[0]) & (x <= x_limits[1])
        display_indices = order[selected]
        if not len(display_indices):
            raise ValueError("The requested reciprocal-wavelength plot range is empty")
        x = reciprocal_wave[display_indices]
    clean = curves[0][1]
    reference_phase = np.angle(clean[:, -1])
    reference_phase += 2.0 * np.pi * np.round(
        (np.median(reference_phase) - reference_phase) / (2.0 * np.pi)
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    palette = ["#555555", "#2878a5", "#d97721", "#3a936f", "#a24e9c", "#a07822"]
    line_styles = ["--", "-", "-.", ":", (0, (5, 1)), (0, (1, 1))]
    colors = [palette[index % len(palette)] for index in range(len(curves))]
    styles = [line_styles[index % len(line_styles)] for index in range(len(curves))]
    rows = []
    curve_stats = []
    for (label, reflection), color, style in zip(curves, colors, styles, strict=False):
        magnitude = np.percentile(np.abs(reflection), [16, 50, 84], axis=0)
        phase = np.percentile(aligned_phase(reflection, reference_phase), [16, 50, 84], axis=0)
        curve_stats.append((label, color, style, magnitude, phase))

    for label, color, style, magnitude, phase in curve_stats:
        for axis, stats in zip(axes, (magnitude, phase), strict=True):
            axis.fill_between(x, stats[0, display_indices], stats[2, display_indices],
                              color=color, alpha=0.055, linewidth=0)
            axis.plot(x, stats[1, display_indices], color=color, linestyle=style,
                      linewidth=1.9, label=label)
        for index in display_indices:
            wavelength = wavelengths[index]
            rows.append({
                "detector": detector,
                "variant": label,
                "wavelength_nm": float(wavelength),
                "reciprocal_wavelength_um_inverse": float(reciprocal_wave[index]),
                **{f"magnitude_p{percentile}": float(magnitude[quantile, index])
                   for quantile, percentile in enumerate((16, 50, 84))},
                **{f"phase_p{percentile}_rad": float(phase[quantile, index])
                   for quantile, percentile in enumerate((16, 50, 84))},
            })

    led_colors = plt.get_cmap("Dark2")(np.linspace(0.0, 1.0, len(mock.LED_CENTERS_NM)))
    for axis in axes:
        axis.set_xlim(*(x_limits if x_limits is not None else (x[0], x[-1])))
        axis.grid(alpha=0.2)
    for led_number, (center, color) in enumerate(zip(mock.LED_CENTERS_NM, led_colors, strict=True), start=1):
        position = reciprocal(center)
        if x_limits is not None and not x_limits[0] <= position <= x_limits[1]:
            continue
        for axis in axes:
            axis.axvline(position, color=color, linestyle=":", linewidth=1.0, alpha=0.9, zorder=0)
        axes[0].text(
            position, 0.98, f"LED {led_number}", transform=axes[0].get_xaxis_transform(),
            rotation=90, ha="right", va="top", fontsize=9.5, fontweight="bold", color="black",
            bbox={"facecolor": "white", "edgecolor": color, "linewidth": 0.8, "alpha": 0.95, "pad": 1.5},
            zorder=5, clip_on=False,
        )
    axes[0].set_ylabel(r"Clean reflection magnitude $|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top = axes[0].secondary_xaxis("top", functions=(reciprocal, reciprocal))
    top_ticks = np.asarray([2300, 1700, 1500, 1300, 1000, 800, 600, 500], dtype=float)
    top_positions = reciprocal(top_ticks)
    visible = ((top_ticks >= wavelengths.min()) & (top_ticks <= wavelengths.max()))
    if x_limits is not None:
        visible &= (top_positions >= x_limits[0]) & (top_positions <= x_limits[1])
    top_ticks = top_ticks[visible]
    top.set_xticks(top_ticks)
    top.set_xlabel("Wavelength (nm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False)
    suffix = f" — {title_suffix}" if title_suffix else ""
    fig.suptitle(f"{detector}: clean complex reflection across curvature strengths{suffix}",
                 y=0.99, fontsize=14)
    note = ("Lines are tile medians; shading is tile 16th–84th percentiles. "
            "All curves show the clean boundary response; no ice film is included.")
    if zoom_y:
        note += " Red-end view; y limits use the tile percentile envelope to expose fit differences."
    fig.text(0.5, 0.025, note,
             ha="center", fontsize=9)
    if zoom_y:
        mag_low = min(np.min(item[3][0, display_indices]) for item in curve_stats)
        mag_high = max(np.max(item[3][2, display_indices]) for item in curve_stats)
        phase_low = min(np.min(item[4][0, display_indices]) for item in curve_stats)
        phase_high = max(np.max(item[4][2, display_indices]) for item in curve_stats)
        mag_pad = max(0.08 * (mag_high - mag_low), 0.002)
        phase_pad = max(0.08 * (phase_high - phase_low), 0.02)
        axes[0].set_ylim(mag_low - mag_pad, mag_high + mag_pad)
        axes[1].set_ylim(phase_low - phase_pad, phase_high + phase_pad)
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.86))
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--optical-cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0.0001,0.001,0.01"))
    parser.add_argument("--additional-fit-dir", type=Path)
    parser.add_argument("--additional-label", default="Reciprocal MaxEnt + curvature 0.01")
    args = parser.parse_args()
    detector = "WFI01"
    baseline_dir = args.sweep_root.parent / "baseline_fixed_ice"
    fit_dirs = [("No added curvature", baseline_dir)]
    fit_dirs.extend((f"Curvature weight {weight:g}", args.sweep_root / weight_label(weight))
                    for weight in args.weights)
    if args.additional_fit_dir is not None:
        fit_dirs.append((args.additional_label, args.additional_fit_dir))
    metric_rows = []
    source_ids = None
    valid = None
    magnitude = None
    parameter_shape = None
    curves = []
    wavelengths = None
    basis = None
    clean = None

    for label, fit_dir in fit_dirs:
        with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as archive:
            parameters = np.asarray(archive["parameters"], dtype=float)
            saved_ids = np.asarray(archive["source_ids"]).astype(str)
            saved_valid = np.asarray(archive["stellar_valid"], dtype=bool)
            saved_prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float)
            local_wavelengths = np.asarray(archive["wavelength_nm"], dtype=float)
            local_basis = np.asarray(archive["basis"], dtype=float)
        if source_ids is None:
            source_ids = saved_ids
            valid = saved_valid
            magnitude = training_magnitudes(source_ids, valid)
            wavelengths = local_wavelengths
            basis = local_basis
            if basis.shape[0] * 2 > parameters.size:
                raise ValueError("Saved modulation basis is larger than the fitted parameter vector")
            clean, *_ = run.workflow._optical_tile_arrays(
                detector, wavelengths, args.optical_cache_root
            )
            curves.append(("Unmodulated clean TMM", clean))
        else:
            np.testing.assert_array_equal(saved_ids, source_ids)
            np.testing.assert_array_equal(saved_valid, valid)
            np.testing.assert_array_equal(local_wavelengths, wavelengths)
            np.testing.assert_allclose(local_basis, basis, rtol=0.0, atol=0.0)
        if parameter_shape is None:
            parameter_shape = parameters.shape
        elif parameters.shape != parameter_shape:
            raise ValueError("The baseline and curvature fit parameter dimensions differ")
        mode = basis.shape[0]
        if parameters.size < 2 * mode:
            raise ValueError(f"Invalid fitted parameter vector in {fit_dir}")
        curves.append((label, modulation_curves(clean, basis, parameters, mode)))
        metric_rows.extend(correction_metrics(magnitude - saved_prediction, magnitude, valid, label))

    # Keep a compact correction table and a filter-level table from the same rows.
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.sweep_root / "amber_in_sample_correction_metrics.csv", index=False)
    filter_metrics = metrics.loc[metrics["filter"] != "ALL"].copy()
    filter_metrics.to_csv(args.sweep_root / "amber_in_sample_filter_within_0p5.csv", index=False)
    plot_reflection(
        detector,
        wavelengths,
        curves,
        args.sweep_root / "curvature_strength_clean_complex_reflection",
        args.sweep_root / "curvature_strength_clean_complex_reflection_quantiles.csv",
    )
    red_end = reciprocal(mock.LED_CENTERS_NM[4])
    red_min = reciprocal(float(wavelengths.max()))
    red_max = red_end * 1.02
    plot_reflection(
        detector,
        wavelengths,
        curves,
        args.sweep_root / "curvature_strength_clean_complex_reflection_led5_red_end",
        args.sweep_root / "curvature_strength_clean_complex_reflection_led5_red_end_quantiles.csv",
        x_limits=(red_min, red_max),
        zoom_y=True,
        title_suffix="LED 5 and longer wavelengths",
    )
    print(metrics.loc[metrics["filter"] == "ALL"].to_string(index=False))
    print(f"Wrote in-sample correction metrics and complex-reflection plot under {args.sweep_root}")


if __name__ == "__main__":
    main()
