"""Supplement held-out joint-correction plots with nearest-neighbor SEDs.

These are a separately labeled sensitivity cohort: test stars remain excluded
from the joint fit, but their CAR201 colors are used to select a same-detector
successful SED donor, as in the historical all-star diagnostic workflow.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from astropy.time import Time

import heatmap_car201_allstar_model_correction as workflow
import run_amber_student_t_detectors as amber
from car201_photometry_forward_jacobian import CachedPhotometryForward
from experiment_joint_wfi01_fixed_ice import add_full_basis
from experiment_joint_wfi01_weekly_ice import cadence_output
from plot_joint_amber_car190_detailed import amber_variant_cells, plot_amber_variants
from plot_joint_amber_out_of_sample import (
    OUTPUT,
    all_star_rows,
    arrays_for_stars,
    model_predictions,
    oos_ids_for_detector,
    summarize_oos_detector,
    thickness_at_days,
    visits_and_frames,
)
from support_aligned_modulation import effective_area_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True, choices=workflow.DETECTORS)
    args = parser.parse_args()
    detectors = list(dict.fromkeys(args.detector))
    rows = all_star_rows()
    all_source_ids = sorted(rows["source_id"].astype(str).unique())
    direct_seds = workflow._load_successful_seds(
        ((workflow.FAINT_SED_PATH, "faint_v2"),
         (workflow.ORIGINAL_SED_PATH, "original_successful")),
        all_source_ids,
    )
    percentile_rows = []
    metric_rows = []
    sample_rows = []
    donor_records = []

    for detector_index, detector in enumerate(detectors, start=1):
        print(f"=== NN-SED held-out validation {detector} ({detector_index}/{len(detectors)}) ===", flush=True)
        detector_rows = rows.loc[rows["detector"] == detector].copy()
        detector_rows["source_id"] = detector_rows["source_id"].astype(str)
        candidate_ids = set(oos_ids_for_detector(rows, detector))
        if not candidate_ids:
            continue
        with np.load(cadence_output(detector, 2.0) / "joint_weekly_ice_fit.npz",
                     allow_pickle=False) as archive:
            trained_ids = set(np.asarray(archive["source_ids"]).astype(str))
        direct_oos_ids = candidate_ids & set(direct_seds)
        nn_target_ids = candidate_ids - direct_oos_ids
        donor_ids = sorted(trained_ids & set(direct_seds))
        assignment_ids = sorted(set(donor_ids) | set(nn_target_ids))
        donor_measurements = detector_rows.loc[detector_rows["source_id"].isin(assignment_ids)]
        donor_frames = [
            donor_measurements.loc[donor_measurements["pass"] == pass_name,
                                   ["source_id", "filter", "mag_obs", "mag_err"]].copy()
            for pass_name in amber.PASSES
        ]
        donor_seds = {source: direct_seds[source] for source in donor_ids}
        assignments, assignment_records = workflow._assign_seds_for_detector(
            detector, donor_frames, assignment_ids, donor_seds
        )
        nn_ids = sorted(source for source in nn_target_ids if source in assignments)
        if not nn_ids:
            print(f"{detector}: no quality-clean held-out stars need SED imputation", flush=True)
            continue

        held_rows = detector_rows.loc[detector_rows["source_id"].isin(nn_ids)].copy()
        counts = held_rows.groupby(["source_id", "filter"]).size()
        repeated = set(counts[counts >= 2].index)
        row_keys = pd.MultiIndex.from_frame(held_rows[["source_id", "filter"]])
        held_rows = held_rows.loc[row_keys.isin(repeated)].copy()
        nn_ids = sorted(held_rows["source_id"].unique())
        if not nn_ids:
            continue

        held_rows, frames, target_mjd = visits_and_frames(detector_rows, nn_ids)
        magnitude, error, valid = arrays_for_stars(held_rows, nn_ids)
        area, _ = amber.canonical_configuration(detector)
        old_fit, _ = amber.paths(detector)
        old_summary = json.loads((old_fit / "car201_actual_star_modulation_fit_summary.json").read_text())
        reference_mjd = Time(workflow.fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
        waves, areas = workflow._make_wavelength_grid(
            float(old_summary["modulation"]["wavelength_stride_nm"]),
            effective_area_path=area,
            full_effective_area_support=True,
        )
        seds = {source: assignments[source] for source in nn_ids}
        _, model = workflow._predict_detector(
            detector, frames, nn_ids, seds, workflow.SED_GRID_PATH, old_fit,
            waves, areas, reference_mjd, unmodulated=True, return_model=True,
            effective_area_path=area, full_effective_area_support=True,
        )
        add_full_basis(model, effective_area_support(area))

        fit_dir = cadence_output(detector, 2.0)
        with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
            joint_times = np.asarray(archive["times"], dtype=float)
            joint_fit_history = np.asarray(archive["fitted_thickness_nm"], dtype=float)
            baseline_history = np.asarray(archive["baseline_thickness_nm"], dtype=float)
            theta = np.asarray(archive["shared_parameters"], dtype=float)
            trained_ids = set(np.asarray(archive["source_ids"]).astype(str))
        overlap = trained_ids & set(nn_ids)
        if overlap:
            raise ValueError(f"NN-SED held-out source selection leaked {len(overlap)} fit stars for {detector}")
        if theta.size != 2 * model.mode_count + model.zero_point_count:
            raise ValueError(f"Joint parameters do not match {detector} NN-SED model basis")

        relative_days = target_mjd - reference_mjd
        baseline_depth = thickness_at_days(joint_times, baseline_history, relative_days)
        fitted_depth = thickness_at_days(joint_times, joint_fit_history, relative_days)
        unmodulated_theta = theta.copy()
        unmodulated_theta[:2 * model.mode_count] = 0.0
        unmodulated_prediction, roll = model_predictions(model, unmodulated_theta, baseline_depth)
        modulated_prediction, _ = model_predictions(model, theta, fitted_depth)
        variants = {
            "no_ice_correction": magnitude - roll,
            "unmodulated": magnitude - unmodulated_prediction,
            "joint_modulated": magnitude - modulated_prediction,
        }
        for name, corrected in variants.items():
            percentile_rows.extend(amber_variant_cells(
                detector, held_rows.rename(columns={"mjd": "mjd_amber"}), magnitude,
                valid, {name: corrected}))
        detector_metrics = summarize_oos_detector(detector, held_rows, nn_ids, magnitude, valid, variants)
        for record in detector_metrics:
            record["sed_assignment"] = "nearest_neighbor_magnitudes"
            record["held_out_stars_with_nn_imputed_sed"] = record.pop("held_out_stars_with_direct_sed")
        metric_rows.extend(detector_metrics)
        sample_rows.append({
            "detector": detector,
            "held_out_stars_with_nn_imputed_sed": len(nn_ids),
            "successful_training_sed_donors": len(donor_ids),
            "held_out_measurements": int(valid.sum()),
            "nearest_neighbor_seds_used": True,
            "SED_donor_policy": "successful SEDs from joint-fit training stars only; nearest same-detector CAR201 reference colors",
        })
        donor_records.extend(
            row for row in assignment_records
            if row["source_id"] in set(nn_ids)
        )
        print(f"{detector}: {len(nn_ids)} held-out stars assigned direct-donor NN SEDs", flush=True)

    if not percentile_rows:
        raise ValueError("No held-out stars requiring nearest-neighbor SEDs were available")
    cells = pd.DataFrame(percentile_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUTPUT / "amber_out_of_sample_nn_sed_epoch_percentiles.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "amber_out_of_sample_nn_sed_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUTPUT / "amber_out_of_sample_nn_sed_sample_counts.csv", index=False)
    pd.DataFrame(donor_records).to_csv(OUTPUT / "amber_out_of_sample_nn_sed_donors.csv", index=False)
    pdf_path = OUTPUT / "amber_out_of_sample_nn_sed_corrections_all_detectors.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_amber_variants(
            cells, OUTPUT, pdf, out_of_sample=True,
            cohort_label="held-out stars with nearest-neighbor SEDs",
            output_suffix="_out_of_sample_nn_sed",
            cohort_note=("Stars absent from joint-fit source lists; SEDs assigned from same-detector CAR201-color donors. "
                         "This supplemental NN cohort can share photometric color information with the diagnostic. "),
        )
    print(f"Wrote separate NN-SED sensitivity plots and donor records under {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
