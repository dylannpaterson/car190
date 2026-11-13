"""Plot in-sample WFI01 CAR201 photometry before/after each curvature fit."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import run_amber_student_t_detectors as run
from fit_car201_zane_star_reflectance_modulation import PASS_SET_CONFIG
from plot_joint_amber_car190_detailed import amber_variant_cells
from summarize_joint_wfi01_curvature_sweep import training_magnitudes


ROOT = Path(__file__).resolve().parent / (
    "20261111/corrected_statistics_4h_thinned/production/wfi01/relative_response/"
    "experiments/car190_amber_fixed_ice_inverse_wavelength_curvature/sweep"
)
BASELINE = ROOT.parent / "baseline_fixed_ice"
WEIGHTS = (0.0001, 0.001, 0.01)


def strength_label(weight):
    return "No added curvature" if weight is None else f"Curvature weight {weight:g}"


def load_training_rows(source_ids, valid):
    frame = pd.read_parquet(run.AMBER)
    frame = frame.loc[frame["sca"] == "WFI01"].copy()
    epoch_map = {f"r010430300{i}00": f"ep3_pass{i}" for i in range(1, 7)}
    epoch_map.update({f"r010430400{i}00": f"ep4_pass{i}" for i in range(1, 6)})
    frame["pass"] = frame["epoch"].map(epoch_map)
    frame = frame.loc[frame["pass"].notna()].copy()
    frame["source_id"] = frame["source_id"].astype(str)

    source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    filter_lookup = {name: index for index, name in enumerate(run.FILTERS)}
    pass_lookup = {name: index for index, name in enumerate(run.PASSES)}
    row_index = (
        frame["source_id"].map(source_lookup),
        frame["filter"].map(filter_lookup),
        frame["pass"].map(pass_lookup),
    )
    present = np.logical_and.reduce([index.notna().to_numpy() for index in row_index])
    frame = frame.loc[present].copy()
    row_index = (
        frame["source_id"].map(source_lookup).to_numpy(int),
        frame["filter"].map(filter_lookup).to_numpy(int),
        frame["pass"].map(pass_lookup).to_numpy(int),
    )
    selected = valid[row_index]
    frame = frame.loc[selected].copy()
    if frame.duplicated(["source_id", "filter", "pass"]).any():
        raise ValueError("Training Amber rows contain duplicate source/filter/pass cells")
    frame["mjd_amber"] = pd.to_numeric(frame["mjd"], errors="raise")
    return frame


def predictions_for_strengths(sweep_root, source_ids, valid, magnitude,
                              additional_fit_dir=None, additional_label=None):
    fit_dirs = [(None, BASELINE)] + [
        (weight, sweep_root / f"weight_{format(weight, '.6g').replace('-', 'm').replace('.', 'p')}")
        for weight in WEIGHTS
    ]
    if additional_fit_dir is not None:
        fit_dirs.append((additional_label or "Reciprocal MaxEnt + curvature 0.01",
                         Path(additional_fit_dir)))
    pass_groups = np.asarray(PASS_SET_CONFIG["ep3v2_ep4"]["zero_point_group_ids"], dtype=int)
    if pass_groups.shape != (len(run.PASSES),):
        raise ValueError("Configured photometry pass groups do not match the fit visit axis")

    corrected_variants = {}
    uncorrected_variants = {}
    for weight, fit_dir in fit_dirs:
        label = weight if isinstance(weight, str) else strength_label(weight)
        with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as archive:
            fitted_ids = np.asarray(archive["source_ids"]).astype(str)
            fitted_valid = np.asarray(archive["stellar_valid"], dtype=bool)
            prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float)
            parameters = np.asarray(archive["parameters"], dtype=float)
        np.testing.assert_array_equal(fitted_ids, source_ids)
        np.testing.assert_array_equal(fitted_valid, valid)
        if prediction.shape != magnitude.shape:
            raise ValueError(f"Saved photometry prediction shape mismatch in {fit_dir}")
        mode_count = 15
        zero_point_count = parameters.size - 2 * mode_count
        expected_count = len(run.FILTERS) * (int(pass_groups.max()))
        if zero_point_count != expected_count:
            raise ValueError(f"Unexpected roll-offset parameter count in {fit_dir}")
        zero_points = parameters[2 * mode_count:].reshape(len(run.FILTERS), int(pass_groups.max()))
        roll = np.zeros_like(magnitude)
        for pass_index, group in enumerate(pass_groups):
            if group:
                roll[:, :, pass_index] = (
                    -zero_points[:, group - 1] / run.workflow.LOG_FLUX_PER_MAG
                )[None, :]
        corrected_variants[label] = magnitude - prediction
        # Match the existing plot's uncorrected control: roll offsets applied,
        # no ice/spectral correction applied.
        uncorrected_variants[label] = magnitude - roll
    return corrected_variants, uncorrected_variants


def plot_strength_page(cells, variant_order, output_path, title, note):
    detectors = list(cells["detector"].unique())
    if detectors != ["WFI01"]:
        raise ValueError(f"Expected only WFI01 data, found {detectors}")
    lower = float(cells["p16_percent"].min())
    upper = float(cells["p84_percent"].max())
    margin = 0.05 * max(upper - lower, 1.0)
    colors = plt.get_cmap("tab10")(np.arange(len(variant_order)))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    for band, axis in zip(run.FILTERS, axes.flat, strict=True):
        axis.axhspan(-0.5, 0.5, color="#65ac68", alpha=0.14)
        axis.axhline(0, color="0.4", linewidth=0.6)
        for index, (label, color) in enumerate(zip(variant_order, colors, strict=True)):
            selected = cells.loc[(cells["variant"] == label) & (cells["filter"] == band)]
            selected = selected.sort_values("utc_date_number")
            offset = (index - (len(variant_order) - 1) / 2) / 96.0
            axis.errorbar(
                selected["utc_date_number"] + offset,
                selected["median_percent"],
                yerr=np.asarray([
                    selected["median_percent"] - selected["p16_percent"],
                    selected["p84_percent"] - selected["median_percent"],
                ]),
                fmt="o", color=color, markersize=3.0, elinewidth=0.7, capsize=1.0,
            )
        axis.set_title(band)
        axis.set_ylim(lower - margin, upper + margin)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x", labelrotation=50, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("UTC date of ending visit")
    handles = [Line2D([0], [0], color=color, marker="o", linestyle="none", label=label)
               for label, color in zip(variant_order, colors, strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=4, frameon=False)
    fig.suptitle(title, y=0.985, fontsize=15)
    fig.supylabel("Relative flux change from measured EP3 pass 1 (%)")
    fig.text(0.5, 0.025,
             note + " Points are WFI01 medians; bars are training-star 16th–84th percentiles; "
             "green band is ±0.5%. Same training stars and axes across strengths.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.025, 0.1, 1, 0.87))
    fig.savefig(output_path, dpi=180)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--additional-fit-dir", type=Path)
    parser.add_argument("--additional-label", default="Reciprocal MaxEnt + curvature 0.01")
    args = parser.parse_args()
    baseline_fit = args.sweep_root.parent / "baseline_fixed_ice"
    with np.load(baseline_fit / "joint_fit.npz", allow_pickle=False) as archive:
        source_ids = np.asarray(archive["source_ids"]).astype(str)
        valid = np.asarray(archive["stellar_valid"], dtype=bool)
    magnitude = training_magnitudes(source_ids, valid)
    common = load_training_rows(source_ids, valid)
    corrected_variants, uncorrected_variants = predictions_for_strengths(
        args.sweep_root, source_ids, valid, magnitude,
        additional_fit_dir=args.additional_fit_dir,
        additional_label=args.additional_label,
    )
    corrected_cells = []
    uncorrected_cells = []
    for label, corrected in corrected_variants.items():
        corrected_cells.extend(amber_variant_cells("WFI01", common, magnitude, valid, {label: corrected}))
        uncorrected_cells.extend(amber_variant_cells(
            "WFI01", common, magnitude, valid, {label: uncorrected_variants[label]}
        ))
    corrected_table = pd.DataFrame(corrected_cells)
    uncorrected_table = pd.DataFrame(uncorrected_cells)
    corrected_table.to_csv(args.output_dir / "amber_in_sample_curvature_corrected_epoch_percentiles.csv", index=False)
    uncorrected_table.to_csv(args.output_dir / "amber_in_sample_curvature_uncorrected_epoch_percentiles.csv", index=False)

    labels = list(corrected_variants)
    corrected_png = args.output_dir / "amber_in_sample_corrected_by_curvature_strength.png"
    uncorrected_png = args.output_dir / "amber_in_sample_uncorrected_by_curvature_strength.png"
    corrected_fig = plot_strength_page(
        corrected_table,
        labels,
        corrected_png,
        "WFI01 CAR201 photometry after joint correction",
        "Corrected by the joint optical model with fixed production ice histories.",
    )
    uncorrected_fig = plot_strength_page(
        uncorrected_table,
        labels,
        uncorrected_png,
        "WFI01 CAR201 photometry before ice correction",
        "Roll offsets applied; ice and spectral modulation corrections omitted.",
    )
    with PdfPages(args.output_dir / "amber_in_sample_photometry_by_curvature_strength.pdf") as pdf:
        pdf.savefig(corrected_fig, bbox_inches="tight")
        pdf.savefig(uncorrected_fig, bbox_inches="tight")
    plt.close(corrected_fig)
    plt.close(uncorrected_fig)
    print(f"Wrote corrected and uncorrected WFI01 CAR201 plots under {args.output_dir}")


if __name__ == "__main__":
    main()
