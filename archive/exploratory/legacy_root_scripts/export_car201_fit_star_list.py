"""Export the exact detector/source cohort used in the saved CAR201 fits."""

import json

import pandas as pd

from heatmap_car201_allstar_model_correction import DETECTORS, PRODUCTION_ROOT, FILTERS, PASSES


def main():
    records = []
    for detector in DETECTORS:
        lower = detector.lower()
        directory = (PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
                     / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched")
        data = pd.read_csv(directory / "car201_actual_star_modulation_measurements.csv",
                           dtype={"source_id": str})
        used = data.loc[data["likelihood_used"].astype(bool)]
        summary = json.loads((directory / "car201_actual_star_modulation_fit_summary.json").read_text())
        expected = summary["active_source_count_with_usable_photometry"]
        if used.source_id.nunique() != expected:
            raise ValueError(f"{detector}: star count does not match saved fit summary")
        for source_id, rows in used.groupby("source_id", sort=True):
            records.append({
                "detector": detector,
                "gaia_dr3_source_id": source_id,
                "used_measurement_count": len(rows),
                "used_filter_count": rows["filter"].nunique(),
                **{f"used_{band}": bool((rows["filter"] == band).any()) for band in FILTERS},
                "used_epoch_count": rows["pass"].nunique(),
                **{f"used_{epoch}": bool((rows["pass"] == epoch).any()) for epoch in PASSES},
                "used_tile_count": rows["tile_id"].nunique(),
                "sed_teff_k_median": rows["sed_teff_k_median"].iloc[0],
                "sed_av_mag_median": rows["sed_av_mag_median"].iloc[0],
                "sed_band_count": rows["sed_band_count"].iloc[0],
                "sed_chi2": rows["sed_chi2"].iloc[0],
            })
    output = PRODUCTION_ROOT / "wavelength_response_bundle/car201_modulation_fit_stars.csv"
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(output)
    print(f"{len(frame)} detector/source pairs; {frame.gaia_dr3_source_id.nunique()} unique Gaia sources")


if __name__ == "__main__":
    main()
