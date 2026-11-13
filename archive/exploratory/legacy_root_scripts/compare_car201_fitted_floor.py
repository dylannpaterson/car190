"""Compare saved CAR201 fitted-floor experiments against no-floor fits."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heatmap_car201_allstar_model_correction import DETECTORS, FILTERS, PRODUCTION_ROOT
import mock_fit_wfi_star_reflectance_modulation as mock


def _profile(values, errors, frame):
    """Identical catalog-error weighting for both fits, for an apples-to-apples diagnostic."""
    temporary = frame[["source_id", "filter"]].copy()
    temporary["weight"] = 1.0 / np.square(errors)
    temporary["weighted_value"] = temporary["weight"] * values
    grouped = temporary.groupby(["source_id", "filter"])
    offset = (grouped["weighted_value"].transform("sum")
              / grouped["weight"].transform("sum")).to_numpy(float)
    return values - offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = PRODUCTION_ROOT / "wavelength_response_bundle"
    records, filter_records, parameter_records, roll_records = [], [], [], []
    missing = []
    for detector in DETECTORS:
        lower = detector.lower()
        experiments = PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
        old_dir = experiments / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched"
        new_dir = old_dir.with_name(old_dir.name + "_fitted_floor")
        new_summary_path = new_dir / "car201_actual_star_modulation_fit_summary.json"
        if not new_summary_path.is_file():
            missing.append(detector)
            continue
        old_summary = json.loads((old_dir / new_summary_path.name).read_text())
        new_summary = json.loads(new_summary_path.read_text())
        if not new_summary["fit"]["success"]:
            raise RuntimeError(f"{detector}: fitted-floor fit did not converge")
        frames = []
        for directory in (old_dir, new_dir):
            data = pd.read_csv(directory / "car201_actual_star_modulation_measurements.csv",
                               dtype={"source_id": str})
            frames.append(data.sort_values(["source_id", "filter", "pass"]).reset_index(drop=True))
        old, new = frames
        keys = ["source_id", "filter", "pass"]
        if not old[keys].equals(new[keys]) or not old.likelihood_used.equals(new.likelihood_used):
            raise RuntimeError(f"{detector}: source/measurement selection changed")
        mask = old.likelihood_used.astype(bool)
        old, new = old.loc[mask].reset_index(drop=True), new.loc[mask].reset_index(drop=True)
        obs = old.mag_obs.to_numpy(float)
        errors = old.mag_err.to_numpy(float)
        old_prediction = old.fitted_delta_mag_including_zeropoint.to_numpy(float)
        new_prediction = new.fitted_delta_mag_including_zeropoint.to_numpy(float)
        old_residual = _profile(obs - old_prediction, errors, old)
        new_residual = _profile(obs - new_prediction, errors, old)
        prediction_change = _profile(new_prediction - old_prediction, errors, old)
        old_flux = 100 * np.expm1(-mock.LOG_FLUX_PER_MAG * old_residual)
        new_flux = 100 * np.expm1(-mock.LOG_FLUX_PER_MAG * new_residual)
        floor = new_summary["data_quality"]["systematic_floor_mmag"]
        records.append({
            "detector": detector,
            "sigma_sys_mmag": floor,
            "sigma_sys_small_signal_flux_percent": floor * mock.LOG_FLUX_PER_MAG / 10.,
            "used_stars": new_summary["active_source_count_with_usable_photometry"],
            "used_observations": len(old),
            "old_reduced_chi2": old_summary["fit"]["reduced_chi2_approx"],
            "new_reduced_chi2_with_floor": new_summary["fit"]["reduced_chi2_approx"],
            "old_profiled_rms_mmag": 1000 * np.sqrt(np.mean(old_residual**2)),
            "new_profiled_rms_mmag": 1000 * np.sqrt(np.mean(new_residual**2)),
            "old_catalog_error_chi2": np.sum(np.square(old_residual / errors)),
            "new_catalog_error_chi2": np.sum(np.square(new_residual / errors)),
            "profiled_prediction_change_rms_mmag": 1000 * np.sqrt(np.mean(prediction_change**2)),
            "profiled_prediction_change_p95_abs_mmag": 1000 * np.quantile(np.abs(prediction_change), .95),
            "old_fraction_within_0p5_percent": np.mean(np.abs(old_flux) <= .5),
            "new_fraction_within_0p5_percent": np.mean(np.abs(new_flux) <= .5),
            "old_maxent_penalty": old_summary["fit"]["maxent_penalty_chi2"],
            "new_maxent_penalty": new_summary["fit"]["maxent_penalty_chi2"],
            "old_phase_penalty": old_summary["fit"]["phase_smoothness_penalty_chi2"],
            "new_phase_penalty": new_summary["fit"]["phase_smoothness_penalty_chi2"],
        })
        for band in FILTERS:
            selected = old["filter"].to_numpy() == band
            filter_records.append({
                "detector": detector, "filter": band, "observations": int(selected.sum()),
                "old_profiled_rms_mmag": 1000 * np.sqrt(np.mean(old_residual[selected]**2)),
                "new_profiled_rms_mmag": 1000 * np.sqrt(np.mean(new_residual[selected]**2)),
                "prediction_change_rms_mmag": 1000 * np.sqrt(np.mean(prediction_change[selected]**2)),
                "old_fraction_within_0p5_percent": np.mean(np.abs(old_flux[selected]) <= .5),
                "new_fraction_within_0p5_percent": np.mean(np.abs(new_flux[selected]) <= .5),
            })
        param_name = "car201_actual_star_modulation_parameters.csv"
        old_params = pd.read_csv(old_dir / param_name).set_index("mode")
        new_params = pd.read_csv(new_dir / param_name).set_index("mode").loc[old_params.index]
        for mode in old_params.index:
            parameter_records.append({
                "detector": detector, "mode": mode,
                "old_log_amplitude": old_params.loc[mode, "fit_log_amplitude"],
                "new_log_amplitude": new_params.loc[mode, "fit_log_amplitude"],
                "old_phase_rad": old_params.loc[mode, "fit_phase_rad"],
                "new_phase_rad": new_params.loc[mode, "fit_phase_rad"],
            })
        old_roll = pd.read_csv(old_dir / "car201_actual_roll_zero_points.csv")
        new_roll = pd.read_csv(new_dir / "car201_actual_roll_zero_points.csv")
        roll = old_roll.merge(new_roll, on=["filter", "offset_group"], suffixes=("_old", "_new"), validate="one_to_one")
        for _, row in roll.iterrows():
            roll_records.append({
                "detector": detector, "filter": row["filter"],
                "old_offset_mmag": row["equivalent_magnitude_offset_mmag_old"],
                "new_offset_mmag": row["equivalent_magnitude_offset_mmag_new"],
                "change_mmag": row["equivalent_magnitude_offset_mmag_new"] - row["equivalent_magnitude_offset_mmag_old"],
            })
    if missing and args.require_complete:
        raise RuntimeError(f"Fits still missing: {', '.join(missing)}")
    if not records:
        raise RuntimeError("No completed fitted-floor fits")
    result = pd.DataFrame(records)
    result.to_csv(output / "car201_fitted_floor_comparison.csv", index=False)
    pd.DataFrame(filter_records).to_csv(output / "car201_fitted_floor_filter_comparison.csv", index=False)
    pd.DataFrame(parameter_records).to_csv(output / "car201_fitted_floor_modulation_comparison.csv", index=False)
    pd.DataFrame(roll_records).to_csv(output / "car201_fitted_floor_roll_offset_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].hist(result.sigma_sys_mmag, bins=min(9, max(3, len(result))), color="#3787a5", edgecolor="white")
    axes[0].set_xlabel(r"Fitted $\sigma_{sys}$ (mmag)")
    axes[0].set_ylabel("Detector count")
    axes[1].bar(result.detector, result.sigma_sys_mmag, color="#3787a5")
    axes[1].tick_params(axis="x", labelrotation=60)
    axes[1].set_ylabel(r"Fitted $\sigma_{sys}$ (mmag)")
    fig.suptitle(f"CAR201 fitted detector-level systematic floors ({len(result)}/18 completed)")
    fig.savefig(output / "car201_fitted_sigma_sys_histogram.png", dpi=180)
    plt.close(fig)
    # Directly expose changes in the inferred optical modulation, separately
    # from the photometric response where spectral changes can be degenerate.
    wave = np.arange(470., 2321., 2.)
    basis, names, *_ = mock.modulation_basis(wave, grouping="per-gap")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    parameter_table = pd.DataFrame(parameter_records)
    for detector, color in zip(result.detector, plt.get_cmap("tab10").colors):
        selected = parameter_table.loc[parameter_table.detector == detector].set_index("mode").loc[names]
        for label, style in (("old", "--"), ("new", "-")):
            amplitude = np.expm1(selected[f"{label}_log_amplitude"].to_numpy() @ basis) * 100.
            phase = selected[f"{label}_phase_rad"].to_numpy() @ basis
            axes[0].plot(wave, amplitude, color=color, linestyle=style,
                         label=f"{detector}: {'no floor' if label == 'old' else 'fitted floor'}")
            axes[1].plot(wave, phase, color=color, linestyle=style)
    axes[0].set_ylabel("Reflection-amplitude modulation (%)")
    axes[1].set_ylabel("Reflection-phase modulation (rad)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.axhline(0, color="0.5", linewidth=.6)
        axis.grid(alpha=.2)
    fig.suptitle("Effect of fitting sigma_sys on the inferred optical modulation")
    fig.tight_layout()
    fig.savefig(output / "car201_fitted_floor_modulation_change.png", dpi=180)
    plt.close(fig)
    print(result[["detector", "sigma_sys_mmag", "old_reduced_chi2", "new_reduced_chi2_with_floor",
                  "profiled_prediction_change_rms_mmag", "old_fraction_within_0p5_percent",
                  "new_fraction_within_0p5_percent"]].to_string(index=False))
    if missing:
        print("Pending:", ", ".join(missing))


if __name__ == "__main__":
    main()
