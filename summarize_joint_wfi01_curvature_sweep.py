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
from experiment_joint_wfi01_fixed_ice import add_full_basis
from experiment_joint_wfi01_inverse_wavelength_curvature import weight_label
from fit_amber_wfi01_student_t import StudentObjective
from plot_wfi01_free_led_complex_reflection import aligned_phase, reciprocal
from support_aligned_modulation import effective_area_support


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


def modulation_curves(clean, basis, parameters, mode_count):
    ratio = np.exp(
        parameters[:mode_count] @ basis
        + 1j * (parameters[mode_count:2 * mode_count] @ basis)
    )
    return clean * ratio[None, :]


def plot_reflection(detector, wavelengths, curves, output_stem, csv_path):
    reciprocal_wave = reciprocal(wavelengths)
    order = np.argsort(reciprocal_wave)
    x = reciprocal_wave[order]
    clean = curves[0][1]
    reference_phase = np.angle(clean[:, -1])
    reference_phase += 2.0 * np.pi * np.round(
        (np.median(reference_phase) - reference_phase) / (2.0 * np.pi)
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors = ["#555555", "#2878a5", "#d97721", "#3a936f", "#a24e9c"]
    styles = ["--", "-", "-", "-", "-"]
    rows = []
    for (label, reflection), color, style in zip(curves, colors, styles, strict=False):
        magnitude = np.percentile(np.abs(reflection), [16, 50, 84], axis=0)
        phase = np.percentile(aligned_phase(reflection, reference_phase), [16, 50, 84], axis=0)
        for axis, stats in zip(axes, (magnitude, phase), strict=True):
            axis.fill_between(x, stats[0, order], stats[2, order], color=color, alpha=0.11, linewidth=0)
            axis.plot(x, stats[1, order], color=color, linestyle=style, linewidth=1.5, label=label)
        for index, wavelength in enumerate(wavelengths):
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
        axis.set_xlim(x[0], x[-1])
        axis.grid(alpha=0.2)
    for led_number, (center, color) in enumerate(zip(mock.LED_CENTERS_NM, led_colors, strict=True), start=1):
        position = reciprocal(center)
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
    top_ticks = np.asarray([2300, 1700, 1300, 1000, 800, 600, 500], dtype=float)
    top_ticks = top_ticks[(top_ticks >= wavelengths.min()) & (top_ticks <= wavelengths.max())]
    top.set_xticks(top_ticks)
    top.set_xlabel("Wavelength (nm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False)
    fig.suptitle(f"{detector}: clean complex reflection across curvature strengths", y=0.99, fontsize=14)
    fig.text(0.5, 0.025,
             "Lines are tile medians; shading is tile 16th–84th percentiles. "
             "All curves show the clean boundary response; no ice film is included.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.86))
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--ice-history-fit-root", type=Path, required=True)
    parser.add_argument("--optical-cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0.0001,0.001,0.01"))
    args = parser.parse_args()
    detector = "WFI01"
    baseline_dir = args.sweep_root.parent / "baseline_fixed_ice"
    area, _ = run.canonical_configuration(detector)
    _, _, source_ids, magnitude, error, valid, _, model, _ = run.prepare(
        detector,
        write_inputs=False,
        effective_area_path=area,
        output_dir=args.sweep_root,
        full_effective_area_support=True,
        ice_history_fit_root=args.ice_history_fit_root,
        optical_cache_root=args.optical_cache_root,
    )
    add_full_basis(model, effective_area_support(area))
    objective = StudentObjective(model, magnitude, error, valid)
    mode = model.mode_count
    clean = np.asarray(model.clean_reflection_by_tile, dtype=complex)
    curves = [("Unmodulated clean TMM", clean)]
    fit_dirs = [("No added curvature", baseline_dir)]
    fit_dirs.extend((f"Curvature weight {weight:g}", args.sweep_root / weight_label(weight))
                    for weight in args.weights)
    metric_rows = []
    per_filter_rows = []

    for label, fit_dir in fit_dirs:
        with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as archive:
            parameters = np.asarray(archive["parameters"], dtype=float)
            saved_ids = np.asarray(archive["source_ids"]).astype(str)
            saved_valid = np.asarray(archive["stellar_valid"], dtype=bool)
            saved_prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float)
        np.testing.assert_array_equal(saved_ids, np.asarray(source_ids).astype(str))
        np.testing.assert_array_equal(saved_valid, valid)
        coordinates = parameters / objective.coordinate_scale
        prediction_mmag = objective.prediction(coordinates)
        np.testing.assert_allclose(prediction_mmag / 1000.0, saved_prediction, atol=2e-9, rtol=0.0)
        prediction_mag = prediction_mmag / 1000.0
        curves.append((label, modulation_curves(clean, model.basis, parameters, mode)))

        modulated = correction_metrics(magnitude - prediction_mag, magnitude, valid, label)
        metric_rows.extend(modulated)
        theta_unmodulated = parameters.copy()
        theta_unmodulated[:2 * mode] = 0.0
        unmodulated_mag = objective.prediction(theta_unmodulated / objective.coordinate_scale) / 1000.0
        metric_rows.extend(correction_metrics(magnitude - unmodulated_mag, magnitude, valid,
                                             f"{label} / unmodulated optics"))

        zero_points = parameters[2 * mode:].reshape(len(run.FILTERS), model.zero_point_group_count)
        roll_mmag = np.zeros_like(magnitude)
        for pass_index, group in enumerate(model.pass_zero_point_group_ids):
            if group:
                roll_mmag[:, :, pass_index] = (
                    -1000.0 * zero_points[:, group - 1] / run.workflow.LOG_FLUX_PER_MAG
                )[None, :]
        metric_rows.extend(correction_metrics(magnitude - roll_mmag / 1000.0, magnitude, valid,
                                             f"{label} / roll-only"))

    # Keep a compact correction table and a filter-level table from the same rows.
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.sweep_root / "amber_in_sample_correction_metrics.csv", index=False)
    filter_metrics = metrics.loc[metrics["filter"] != "ALL"].copy()
    filter_metrics.to_csv(args.sweep_root / "amber_in_sample_filter_within_0p5.csv", index=False)
    plot_reflection(
        detector,
        np.asarray(model.wavelengths_nm, dtype=float),
        curves,
        args.sweep_root / "curvature_strength_clean_complex_reflection",
        args.sweep_root / "curvature_strength_clean_complex_reflection_quantiles.csv",
    )
    print(metrics.loc[metrics["filter"] == "ALL"].to_string(index=False))
    print(f"Wrote in-sample correction metrics and complex-reflection plot under {args.sweep_root}")


if __name__ == "__main__":
    main()
