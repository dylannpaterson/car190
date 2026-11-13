"""Count in-sample, direct-SED holdout, and NN-SED sensitivity changes within 0.5%."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

import heatmap_car201_allstar_model_correction as workflow
import run_amber_student_t_detectors as amber
from experiment_joint_wfi01_fixed_ice import add_full_basis
from experiment_joint_wfi01_weekly_ice import cadence_output
from plot_joint_amber_car190_detailed import prediction_with_roll
from support_aligned_modulation import effective_area_support


ROOT = Path(__file__).resolve().parent
OUTPUT = workflow.PRODUCTION_ROOT / (
    "wfi01/spline_relative_response_normal/experiments/"
    "car190_amber_joint_2day_all_detectors_overview"
)


def main():
    oos_metrics = pd.read_csv(OUTPUT / "amber_out_of_sample_correction_metrics.csv")
    oos_counts = pd.read_csv(OUTPUT / "amber_out_of_sample_sample_counts.csv")
    oos_filter_counts = pd.read_csv(OUTPUT / "amber_out_of_sample_filter_within_0p5.csv")
    nn_metrics = pd.read_csv(OUTPUT / "amber_out_of_sample_nn_sed_metrics.csv")
    nn_counts = pd.read_csv(OUTPUT / "amber_out_of_sample_nn_sed_sample_counts.csv")
    oos_metrics = oos_metrics.loc[oos_metrics["variant"] == "joint_modulated"].set_index("detector")
    oos_counts = oos_counts.set_index("detector")
    nn_metrics = nn_metrics.loc[nn_metrics["variant"] == "joint_modulated"].set_index("detector")
    nn_counts = nn_counts.set_index("detector")
    records = []
    in_filter_records = []

    for detector in workflow.DETECTORS:
        fit_dir = cadence_output(detector, 2.0)
        area, _ = amber.canonical_configuration(detector)
        _, common, ids, magnitude, _error, valid, _, model, _ = amber.prepare(
            detector, write_inputs=False, effective_area_path=area, output_dir=fit_dir,
            full_effective_area_support=True,
        )
        add_full_basis(model, effective_area_support(area))
        with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=float)
            fitted_thickness = np.asarray(archive["fitted_thickness_nm"], dtype=float)
            theta = np.asarray(archive["shared_parameters"], dtype=float)
        reference_mjd = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
        star_days = np.array([
            [common.loc[(common["filter"] == band) & (common["pass"] == epoch), "mjd_amber"].median()
             - reference_mjd for epoch in amber.PASSES]
            for band in amber.FILTERS
        ])
        star_indices = np.searchsorted(times, star_days)
        np.testing.assert_allclose(times[star_indices], star_days, rtol=0, atol=1e-10)
        prediction, roll = prediction_with_roll(model, theta, fitted_thickness, star_indices)
        corrected = magnitude - prediction
        roll_adjusted_no_ice = magnitude - roll
        delta = corrected - corrected[:, :, :1]
        flux_percent = 100.0 * np.expm1(-workflow.LOG_FLUX_PER_MAG * delta)
        no_ice_delta = roll_adjusted_no_ice - roll_adjusted_no_ice[:, :, :1]
        no_ice_flux_percent = 100.0 * np.expm1(-workflow.LOG_FLUX_PER_MAG * no_ice_delta)
        comparisons = valid & valid[:, :, :1]
        comparisons[:, :, 0] = False
        in_values = flux_percent[comparisons]
        in_finite = np.isfinite(in_values)
        in_count = int(np.count_nonzero(in_finite & (np.abs(in_values) <= 0.5)))
        in_total = int(np.count_nonzero(in_finite))
        for filter_index, band in enumerate(amber.FILTERS):
            selected = comparisons[:, filter_index, :]
            for variant, values_grid in (("joint_modulated", flux_percent),
                                         ("no_ice_correction", no_ice_flux_percent)):
                values = values_grid[:, filter_index, :][selected]
                finite = np.isfinite(values)
                filter_total = int(np.count_nonzero(finite))
                filter_count = int(np.count_nonzero(finite & (np.abs(values) <= 0.5)))
                in_filter_records.append({
                    "detector": detector,
                    "filter": band,
                    "variant": variant,
                    "within_0p5_percent_count": filter_count,
                    "comparison_count": filter_total,
                    "fraction_within_0p5_percent": filter_count / filter_total if filter_total else np.nan,
                })

        out_row = oos_metrics.loc[detector]
        out_total = int(out_row["matched_source_filter_visit_values"])
        out_within = int(np.rint(float(out_row["fraction_within_0p5_percent"]) * out_total))
        nn_row = nn_metrics.loc[detector]
        nn_total = int(nn_row["matched_source_filter_visit_values"])
        nn_within = int(np.rint(float(nn_row["fraction_within_0p5_percent"]) * nn_total))
        records.append({
            "detector": detector,
            "in_sample_within_0p5_percent_count": in_count,
            "in_sample_comparison_count": in_total,
            "in_sample_fraction": in_count / in_total,
            "out_of_sample_direct_sed_within_0p5_count": out_within,
            "out_of_sample_direct_sed_comparison_count": out_total,
            "out_of_sample_direct_sed_fraction": out_within / out_total,
            "out_of_sample_direct_sed_star_count": int(oos_counts.loc[detector, "held_out_stars_with_direct_successful_sed"]),
            "out_of_sample_nn_sed_within_0p5_count": nn_within,
            "out_of_sample_nn_sed_comparison_count": nn_total,
            "out_of_sample_nn_sed_fraction": nn_within / nn_total if nn_total else np.nan,
            "out_of_sample_nn_sed_star_count": int(nn_counts.loc[detector, "held_out_stars_with_nn_imputed_sed"]),
            "out_of_sample_direct_plus_nn_within_0p5_count": out_within + nn_within,
            "out_of_sample_direct_plus_nn_comparison_count": out_total + nn_total,
            "out_of_sample_direct_plus_nn_fraction": (out_within + nn_within) / (out_total + nn_total),
        })
        print(f"{detector}: in {in_count}/{in_total}; direct OOS {out_within}/{out_total}; "
              f"NN OOS {nn_within}/{nn_total}", flush=True)

    table = pd.DataFrame(records)
    totals = {
        "detector": "ALL",
        "in_sample_within_0p5_percent_count": int(table["in_sample_within_0p5_percent_count"].sum()),
        "in_sample_comparison_count": int(table["in_sample_comparison_count"].sum()),
        "out_of_sample_direct_sed_within_0p5_count": int(table["out_of_sample_direct_sed_within_0p5_count"].sum()),
        "out_of_sample_direct_sed_comparison_count": int(table["out_of_sample_direct_sed_comparison_count"].sum()),
        "out_of_sample_direct_sed_star_count": int(table["out_of_sample_direct_sed_star_count"].sum()),
        "out_of_sample_nn_sed_within_0p5_count": int(table["out_of_sample_nn_sed_within_0p5_count"].sum()),
        "out_of_sample_nn_sed_comparison_count": int(table["out_of_sample_nn_sed_comparison_count"].sum()),
        "out_of_sample_nn_sed_star_count": int(table["out_of_sample_nn_sed_star_count"].sum()),
        "out_of_sample_direct_plus_nn_within_0p5_count": int(table["out_of_sample_direct_plus_nn_within_0p5_count"].sum()),
        "out_of_sample_direct_plus_nn_comparison_count": int(table["out_of_sample_direct_plus_nn_comparison_count"].sum()),
    }
    totals["in_sample_fraction"] = totals["in_sample_within_0p5_percent_count"] / totals["in_sample_comparison_count"]
    totals["out_of_sample_direct_sed_fraction"] = (
        totals["out_of_sample_direct_sed_within_0p5_count"]
        / totals["out_of_sample_direct_sed_comparison_count"]
    )
    totals["out_of_sample_nn_sed_fraction"] = (
        totals["out_of_sample_nn_sed_within_0p5_count"]
        / totals["out_of_sample_nn_sed_comparison_count"]
    )
    totals["out_of_sample_direct_plus_nn_fraction"] = (
        totals["out_of_sample_direct_plus_nn_within_0p5_count"]
        / totals["out_of_sample_direct_plus_nn_comparison_count"]
    )
    table = pd.concat([table, pd.DataFrame([totals])], ignore_index=True)
    output = OUTPUT / "amber_in_vs_out_sample_within_0p5_counts.csv"
    table.to_csv(output, index=False)
    in_filter = pd.DataFrame(in_filter_records)
    direct_oos_filter = oos_filter_counts.loc[
        (oos_filter_counts["variant"] == "joint_modulated")
        & (oos_filter_counts["cohort"] == "direct_sed")
    ].copy()
    direct_oos_filter = direct_oos_filter.rename(columns={
        "within_0p5_percent_count": "out_of_sample_within_0p5_percent_count",
        "comparison_count": "out_of_sample_comparison_count",
        "fraction_within_0p5_percent": "out_of_sample_fraction_within_0p5_percent",
    })
    in_modulated = in_filter.loc[in_filter["variant"] == "joint_modulated"].drop(columns="variant")
    filter_table = in_modulated.merge(
        direct_oos_filter[["detector", "filter", "out_of_sample_within_0p5_percent_count",
                           "out_of_sample_comparison_count", "out_of_sample_fraction_within_0p5_percent"]],
        on=["detector", "filter"], how="left", validate="one_to_one")
    filter_table.to_csv(OUTPUT / "amber_in_vs_out_sample_within_0p5_by_filter.csv", index=False)
    in_uncorrected = in_filter.loc[in_filter["variant"] == "no_ice_correction"].drop(columns="variant")
    in_uncorrected = in_uncorrected.rename(columns={
        "within_0p5_percent_count": "in_sample_within_0p5_percent_count",
        "comparison_count": "in_sample_comparison_count",
        "fraction_within_0p5_percent": "in_sample_fraction_within_0p5_percent",
    })
    out_uncorrected = oos_filter_counts.loc[
        (oos_filter_counts["variant"] == "no_ice_correction")
        & (oos_filter_counts["cohort"] == "direct_sed")
    ].copy()
    out_uncorrected = out_uncorrected.rename(columns={
        "within_0p5_percent_count": "out_of_sample_within_0p5_percent_count",
        "comparison_count": "out_of_sample_comparison_count",
        "fraction_within_0p5_percent": "out_of_sample_fraction_within_0p5_percent",
    })
    uncorrected_table = in_uncorrected.merge(
        out_uncorrected[["detector", "filter", "out_of_sample_within_0p5_percent_count",
                         "out_of_sample_comparison_count", "out_of_sample_fraction_within_0p5_percent"]],
        on=["detector", "filter"], how="left", validate="one_to_one")
    uncorrected_table.to_csv(
        OUTPUT / "amber_in_vs_out_sample_uncorrected_within_0p5_by_filter.csv", index=False)
    metadata = {
        "model": "joint modulated optics plus fitted 2-day ice history",
        "threshold": "absolute relative visit-to-visit flux change <= 0.5 percent",
        "comparison_definition": "same source/filter with measured EP3 pass 1 reference; reference rows excluded",
        "in_sample": "joint-fit source lists; saved fit parameters held fixed during count",
        "out_of_sample_direct_sed": "source IDs absent from joint-fit source lists; direct successful faint-v2/original SEDs only; saved fit parameters held fixed",
        "out_of_sample_nn_sed": "separately reported sensitivity cohort; nearest same-detector SED donors restricted to joint-fit training stars",
        "out_of_sample_nn_color_selection_uses_test_photometry": True,
        "output_csv": str(output.resolve()),
    }
    (OUTPUT / "amber_in_vs_out_sample_within_0p5_counts.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
