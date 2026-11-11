"""Summarize the completed Amber comparison without running any fits."""

import json

import numpy as np
import pandas as pd

from check_amber_wfi01_photometry import OUTPUT, BASE, profile_residual


def main():
    archive = np.load(OUTPUT / "predictions.npz", allow_pickle=False)
    valid = archive["valid"]
    change = profile_residual(archive["amber_refitted_floor"], archive["amber_frozen_floor"],
                              archive["zane_error"], valid)[valid] * 1000
    comparison = {
        "amber_refit_vs_frozen_zane_floor_prediction_rms_mmag": float(np.sqrt(np.mean(change**2))),
        "amber_refit_vs_frozen_zane_floor_prediction_p95_absolute_mmag": float(np.quantile(np.abs(change), .95)),
        "prediction_comparison_centering": "same per-star/filter catalog-error-weighted mean removed from both predictions; Zane errors used for common weights",
    }
    floor_summary = json.loads((BASE.with_name(BASE.name + "_fitted_floor")
                               / "car201_actual_star_modulation_fit_summary.json").read_text())
    comparison["zane_fitted_sigma_sys_mmag"] = floor_summary["data_quality"]["systematic_floor_mmag"]
    for label, magnitude, prediction, error in (
        ("zane_original", "zane_magnitude", "zane_frozen", "zane_error"),
        ("amber_frozen_original", "amber_magnitude", "amber_frozen", "amber_error"),
        ("amber_frozen_floor", "amber_magnitude", "amber_frozen_floor", "amber_error"),
        ("amber_refit_floor", "amber_magnitude", "amber_refitted_floor", "amber_error"),
    ):
        residual = profile_residual(archive[magnitude], archive[prediction], archive[error], valid)[valid]
        flux = 100 * np.expm1(-.4 * np.log(10.) * residual)
        comparison[label + "_rms_mmag"] = float(1000*np.sqrt(np.mean(residual**2)))
        comparison[label + "_fraction_within_0p5_percent"] = float(np.mean(np.abs(flux)<=.5))
    old = pd.read_csv(BASE.with_name(BASE.name + "_fitted_floor")
                     / "car201_actual_star_modulation_parameters.csv")
    new = pd.read_csv(OUTPUT / "amber_modulation_parameters.csv")
    parameters = new.merge(old, on="mode", suffixes=("_amber", "_zane"), validate="one_to_one")
    parameters.to_csv(OUTPUT / "amber_vs_zane_fitted_floor_parameters.csv", index=False)
    (OUTPUT / "aggregate_comparison_metrics.json").write_text(json.dumps(comparison, indent=2)+"\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
