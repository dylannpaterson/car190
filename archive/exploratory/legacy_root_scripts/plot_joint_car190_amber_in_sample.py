"""Plot in-sample Amber visit changes for joint CAR190/Amber detector fits."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages

import plot_joint_amber_car190_detailed as plotter
import plot_joint_amber_out_of_sample as oos
import run_amber_student_t_detectors as amber
import run_joint_car190_amber_inverse_wavelength_maxent as joint_fit
from heatmap_car201_allstar_model_correction import DETECTORS


ROOT = Path(__file__).resolve().parent
DEFAULT_STATISTICS = ROOT / "20261111/corrected_statistics_4h_thinned"
FIT_NAME = "relative_response/experiments/car190_amber_joint_2day_reciprocal_maxent"
DEFAULT_OUTPUT = (
    DEFAULT_STATISTICS
    / "production/wavelength_response_bundle/"
    "joint_car190_amber_inverse_wavelength_maxent_in_sample"
)


def detector_cells(detector, fit_root):
    source_ids, seds, magnitude, error, valid, common, frames = (
        joint_fit.raw_common_training_sample(detector)
    )
    fit_dir = fit_root / detector.lower() / FIT_NAME
    with np.load(fit_dir / "joint_fit.npz", allow_pickle=False) as archive:
        fit_ids = np.asarray(archive["source_ids"]).astype(str)
        saved_prediction = np.asarray(archive["stellar_prediction_mag"], dtype=float)
        saved_valid = np.asarray(archive["stellar_valid"], dtype=bool)
        parameters = np.asarray(archive["parameters"], dtype=float)
    with np.load(fit_dir / "training_inputs.npz", allow_pickle=False) as archive:
        saved_magnitude = np.asarray(archive["magnitude"], dtype=float)
        saved_input_valid = np.asarray(archive["valid"], dtype=bool)
    np.testing.assert_array_equal(source_ids, fit_ids)
    np.testing.assert_array_equal(valid, saved_valid)
    np.testing.assert_array_equal(valid, saved_input_valid)
    np.testing.assert_allclose(magnitude, saved_magnitude, rtol=0.0, atol=1e-10)

    summary = json.loads((fit_dir / "summary.json").read_text())
    if not summary.get("optimizer_success"):
        raise RuntimeError(f"{detector} joint fit was not successful")
    history_root = Path(summary["ice_history_fit_root"])
    cache_root = Path(summary["optical_cache_root"])
    area, _ = amber.canonical_configuration(detector)
    model = joint_fit.model_for_raw_sample(
        detector, source_ids, seds, valid, frames,
        area, history_root, cache_root,
    )

    with np.load(fit_dir / "joint_weekly_ice_fit.npz", allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=float)
        baseline_history = np.asarray(archive["baseline_thickness_nm"], dtype=float)
        fitted_history = np.asarray(archive["fitted_thickness_nm"], dtype=float)
    reference_mjd = Time(
        joint_fit.run.workflow.fit_model.ICE_REFERENCE_UTC,
        format="isot", scale="utc",
    ).mjd
    star_days = np.asarray([
        [
            common.loc[
                (common["filter"] == band) & (common["pass"] == pass_name),
                "mjd_amber",
            ].median() - reference_mjd
            for pass_name in amber.PASSES
        ]
        for band in amber.FILTERS
    ], dtype=float)
    star_indices = np.searchsorted(times, star_days)
    if np.any(star_indices >= times.size):
        raise ValueError(f"{detector}: Amber training epoch exceeds the fitted ice history")
    np.testing.assert_allclose(times[star_indices], star_days, rtol=0.0, atol=1e-10)
    baseline_thickness = baseline_history[:, star_indices]
    fitted_thickness = fitted_history[:, star_indices]

    mode_count = model.mode_count
    theta_unmodulated = parameters.copy()
    theta_unmodulated[:2 * mode_count] = 0.0
    unmodulated_prediction, roll = oos.model_predictions(
        model, theta_unmodulated, baseline_thickness
    )
    joint_prediction, _ = oos.model_predictions(model, parameters, fitted_thickness)
    np.testing.assert_allclose(
        joint_prediction, saved_prediction, rtol=2e-6, atol=2e-6,
        err_msg=f"{detector}: reconstructed in-sample prediction differs from saved fit",
    )
    variants = {
        "no_ice_correction": magnitude - roll,
        "unmodulated": magnitude - unmodulated_prediction,
        "joint_modulated": magnitude - joint_prediction,
    }
    rows = plotter.amber_variant_cells(
        detector, common, magnitude, valid, variants
    )
    info = {
        "detector": detector,
        "training_stars": int(len(source_ids)),
        "training_measurements": int(valid.sum()),
        "matched_source_filter_visit_values": int(
            (valid & valid[:, :, :1])[:, :, 1:].sum()
        ),
        "optimizer_success": bool(summary["optimizer_success"]),
        "maxent_weight": float(summary["reciprocal_maxent_weight"]),
        "curvature_weight": float(summary["curvature_weight"]),
        "training_selection": summary["training_selection"],
    }
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detector", action="append", required=True, choices=DETECTORS,
        help="Detector to include; repeat explicitly for multiple detectors.",
    )
    args = parser.parse_args()
    detectors = list(dict.fromkeys(args.detector))
    production = args.statistics_dir / "production"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    fit_info = []
    for index, detector in enumerate(detectors, start=1):
        print(f"=== In-sample plot data {detector} ({index}/{len(detectors)}) ===", flush=True)
        rows, info = detector_cells(detector, production)
        all_rows.extend(rows)
        fit_info.append(info)
        print(
            f"{detector}: {info['training_stars']} stars, "
            f"{info['training_measurements']} measurements, "
            f"{info['matched_source_filter_visit_values']} visit comparisons",
            flush=True,
        )

    cells = pd.DataFrame(all_rows)
    cells_path = args.output_dir / "amber_in_sample_epoch_percentiles.csv"
    cells.to_csv(cells_path, index=False)
    pd.DataFrame(fit_info).to_csv(args.output_dir / "detector_fit_sample_counts.csv", index=False)
    with PdfPages(args.output_dir / "amber_in_sample_corrections_all_detectors.pdf") as pdf:
        plotter.plot_amber_variants(
            cells,
            args.output_dir,
            pdf,
            out_of_sample=False,
            output_suffix="_raw_joint_fit_cohort",
            cohort_note=(
                "Exact raw Zane/Amber joint-fit cohort; no prior fit, manual-exclusion, "
                "or repeat-outlier tables used. "
            ),
        )
    print(f"Wrote in-sample correction comparisons under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
