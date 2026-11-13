"""Summarize all-detector reciprocal-wavelength MaxEnt joint fits and heatmaps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import heatmap_car201_allstar_model_correction as workflow
from plot_joint_amber_filter_heatmaps import plot_heatmap
import run_amber_student_t_detectors as amber


ROOT = Path(__file__).resolve().parent
DEFAULT_STATISTICS = ROOT / "20261111/corrected_statistics_4h_thinned"
FIT_NAME = "relative_response/experiments/car190_amber_joint_2day_reciprocal_maxent"
OOS_NAME = "joint_car190_amber_inverse_wavelength_maxent_oos"


def metrics(corrected, valid, label, detector):
    comparisons = valid & valid[:, :, :1]
    comparisons[:, :, 0] = False
    delta = corrected - corrected[:, :, :1]
    flux = 100.0 * np.expm1(-workflow.LOG_FLUX_PER_MAG * delta)
    rows = []
    values = flux[comparisons]
    values = values[np.isfinite(values)]
    rows.append({
        "detector": detector,
        "filter": "ALL",
        "variant": label,
        "comparison_count": int(len(values)),
        "within_0p5_percent_count": int(np.count_nonzero(np.abs(values) <= 0.5)),
        "fraction_within_0p5_percent": float(np.mean(np.abs(values) <= 0.5)),
        "rms_relative_change_percent": float(np.sqrt(np.mean(values**2))),
        "median_relative_change_percent": float(np.median(values)),
    })
    for filter_index, band in enumerate(amber.FILTERS):
        selected = comparisons[:, filter_index, :]
        values = flux[:, filter_index, :][selected]
        values = values[np.isfinite(values)]
        rows.append({
            "detector": detector,
            "filter": band,
            "variant": label,
            "comparison_count": int(len(values)),
            "within_0p5_percent_count": int(np.count_nonzero(np.abs(values) <= 0.5)),
            "fraction_within_0p5_percent": float(np.mean(np.abs(values) <= 0.5)) if len(values) else np.nan,
            "rms_relative_change_percent": float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
            "median_relative_change_percent": float(np.median(values)) if len(values) else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    production = args.statistics_dir / "production"
    output = args.output_dir or production / "wavelength_response_bundle" / OOS_NAME
    output.mkdir(parents=True, exist_ok=True)
    oos_root = production / "wavelength_response_bundle" / OOS_NAME
    pass_groups = np.asarray(
        workflow.fit_model.PASS_SET_CONFIG["ep3v2_ep4"]["zero_point_group_ids"], dtype=int
    )

    fit_summaries = []
    in_records = []
    roll_records = []
    per_filter_oos = pd.read_csv(oos_root / "amber_out_of_sample_filter_within_0p5.csv")
    oos_metrics = pd.read_csv(oos_root / "amber_out_of_sample_correction_metrics.csv")
    oos_metrics = oos_metrics.loc[oos_metrics["variant"] == "joint_modulated"]

    for detector in workflow.DETECTORS:
        fit_dir = production / detector.lower() / FIT_NAME
        summary = json.loads((fit_dir / "summary.json").read_text())
        if not summary.get("optimizer_success"):
            raise RuntimeError(f"{detector} joint fit did not converge")
        with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as fit_archive:
            parameters = np.asarray(fit_archive["parameters"], dtype=float)
            prediction = np.asarray(fit_archive["stellar_prediction_mag"], dtype=float)
            valid = np.asarray(fit_archive["stellar_valid"], dtype=bool)
            source_ids = np.asarray(fit_archive["source_ids"]).astype(str)
            saved_basis = np.asarray(fit_archive["basis"], dtype=float)
        with np.load(fit_dir / "training_inputs.npz", allow_pickle=False) as input_archive:
            magnitude = np.asarray(input_archive["magnitude"], dtype=float)
            input_valid = np.asarray(input_archive["valid"], dtype=bool)
        np.testing.assert_array_equal(input_valid, valid)
        if len(source_ids) != int(summary["training_stars"]):
            raise ValueError(f"{detector} source count differs from fit summary")

        mode_count = saved_basis.shape[0]
        zero_point_count = parameters.size - 2 * mode_count
        if zero_point_count != len(amber.FILTERS) * int(pass_groups.max()):
            raise ValueError(f"{detector} roll-offset layout is not supported")
        zero_points = parameters[2 * mode_count:].reshape(len(amber.FILTERS), int(pass_groups.max()))
        roll = np.zeros_like(magnitude)
        for pass_index, group in enumerate(pass_groups):
            if group:
                roll[:, :, pass_index] = (
                    -zero_points[:, group - 1] / workflow.LOG_FLUX_PER_MAG
                )[None, :]
        in_records.extend(metrics(magnitude - prediction, valid, "joint_modulated", detector))
        roll_records.extend(metrics(magnitude - roll, valid, "no_ice_correction", detector))

        held = oos_metrics.loc[oos_metrics["detector"] == detector]
        if len(held) != 1:
            raise ValueError(f"Missing held-out direct-SED summary for {detector}")
        fit_summaries.append({
            "detector": detector,
            "optimizer_success": bool(summary["optimizer_success"]),
            "training_stars": len(source_ids),
            "training_measurements": int(valid.sum()),
            "in_sample_comparisons": int(in_records[-9]["comparison_count"]),
            "in_sample_within_0p5_percent": float(in_records[-9]["fraction_within_0p5_percent"]),
            "in_sample_rms_relative_change_percent": float(in_records[-9]["rms_relative_change_percent"]),
            "car190_chi2": float(summary["car190_chi2"]),
            "held_out_direct_sed_stars": int(held.iloc[0]["held_out_stars_with_direct_sed"]),
            "held_out_comparisons": int(held.iloc[0]["matched_source_filter_visit_values"]),
            "held_out_within_0p5_percent_count": int(np.rint(
                float(held.iloc[0]["fraction_within_0p5_percent"])
                * int(held.iloc[0]["matched_source_filter_visit_values"])
            )),
            "held_out_within_0p5_percent": float(held.iloc[0]["fraction_within_0p5_percent"]),
            "held_out_rms_relative_change_percent": float(held.iloc[0]["relative_flux_change_rms_percent"]),
            "max_abs_ice_change_nm": float(summary["max_abs_ice_change_nm"]),
        })

    in_table = pd.DataFrame(in_records)
    roll_table = pd.DataFrame(roll_records)
    in_overall = in_table.loc[in_table["filter"] == "ALL"]
    total_in = int(in_overall["comparison_count"].sum())
    total_in_within = int(in_overall["within_0p5_percent_count"].sum())
    total_oos = int(sum(row["held_out_comparisons"] for row in fit_summaries))
    total_oos_within = int(sum(row["held_out_within_0p5_percent_count"] for row in fit_summaries))
    fleet_summary = {
        "detector": "ALL",
        "optimizer_success": bool(all(row["optimizer_success"] for row in fit_summaries)),
        "training_stars": int(sum(row["training_stars"] for row in fit_summaries)),
        "training_measurements": int(sum(row["training_measurements"] for row in fit_summaries)),
        "in_sample_comparisons": total_in,
        "in_sample_within_0p5_percent": total_in_within / total_in,
        "in_sample_rms_relative_change_percent": float(np.sqrt(np.sum(
            in_overall["comparison_count"].to_numpy(float)
            * in_overall["rms_relative_change_percent"].to_numpy(float)**2
        ) / total_in)),
        "car190_chi2": float(sum(row["car190_chi2"] for row in fit_summaries)),
        "held_out_direct_sed_stars": int(sum(row["held_out_direct_sed_stars"] for row in fit_summaries)),
        "held_out_comparisons": total_oos,
        "held_out_within_0p5_percent_count": total_oos_within,
        "held_out_within_0p5_percent": total_oos_within / total_oos,
        "held_out_rms_relative_change_percent": float(np.sqrt(np.sum(
            np.asarray([row["held_out_comparisons"] for row in fit_summaries], dtype=float)
            * np.asarray([row["held_out_rms_relative_change_percent"] for row in fit_summaries], dtype=float)**2
        ) / total_oos)),
        "max_abs_ice_change_nm": float(max(row["max_abs_ice_change_nm"] for row in fit_summaries)),
    }
    fleet_fit_table = pd.DataFrame([*fit_summaries, fleet_summary])
    oos_filter = per_filter_oos.loc[
        (per_filter_oos["variant"] == "joint_modulated")
        & (per_filter_oos["cohort"] == "direct_sed")
    ].copy()
    in_filter = in_table.loc[in_table["filter"] != "ALL"].drop(columns="variant")
    oos_filter = oos_filter.rename(columns={
        "within_0p5_percent_count": "out_of_sample_within_0p5_percent_count",
        "comparison_count": "out_of_sample_comparison_count",
        "fraction_within_0p5_percent": "out_of_sample_fraction_within_0p5_percent",
    })
    combined = in_filter.merge(
        oos_filter[["detector", "filter", "out_of_sample_within_0p5_percent_count",
                    "out_of_sample_comparison_count", "out_of_sample_fraction_within_0p5_percent"]],
        on=["detector", "filter"], how="left", validate="one_to_one",
    )
    in_table.to_csv(output / "amber_in_sample_correction_metrics.csv", index=False)
    roll_table.to_csv(output / "amber_in_sample_roll_only_metrics.csv", index=False)
    combined.to_csv(output / "amber_in_vs_out_sample_within_0p5_by_filter.csv", index=False)
    fleet_fit_table.to_csv(output / "all_detector_fit_and_validation_summary.csv", index=False)

    detectors = list(workflow.DETECTORS)
    filters = list(amber.FILTERS)
    figure_in = plot_heatmap(
        combined, detectors, filters,
        "fraction_within_0p5_percent", "within_0p5_percent_count", "comparison_count",
        "Inverse-wavelength MaxEnt + curvature: in-sample observations within ±0.5%",
        output / "amber_in_sample_within_0p5_by_filter.png",
    )
    figure_oos = plot_heatmap(
        combined, detectors, filters,
        "out_of_sample_fraction_within_0p5_percent",
        "out_of_sample_within_0p5_percent_count", "out_of_sample_comparison_count",
        "Inverse-wavelength MaxEnt + curvature: held-out direct-SED observations within ±0.5%",
        output / "amber_out_of_sample_within_0p5_by_filter.png",
        out_of_sample=True,
    )
    with PdfPages(output / "amber_filter_within_0p5_heatmaps.pdf") as pdf:
        pdf.savefig(figure_in, bbox_inches="tight")
        pdf.savefig(figure_oos, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure_in)
    plt.close(figure_oos)
    print(fleet_fit_table.to_string(index=False))
    print(f"Wrote all-detector summaries and heatmaps under {output}")


if __name__ == "__main__":
    main()
