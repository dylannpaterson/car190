"""Export roll offsets with cross-roll source counts and identifiability flags."""

import json

import pandas as pd

from heatmap_car201_allstar_model_correction import DETECTORS, PRODUCTION_ROOT, PASSES


def main():
    records = []
    for detector in DETECTORS:
        lower = detector.lower()
        directory = (PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
                     / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched")
        summary = json.loads((directory / "car201_actual_star_modulation_fit_summary.json").read_text())
        group_ids = summary["modulation"]["zeropoint_model"]["pass_group_ids"]
        pre_epochs = [epoch for epoch, group in zip(PASSES, group_ids, strict=True) if group == 0]
        post_epochs = [epoch for epoch, group in zip(PASSES, group_ids, strict=True) if group > 0]
        data = pd.read_csv(directory / "car201_actual_star_modulation_measurements.csv", dtype={"source_id": str})
        data = data.loc[data["likelihood_used"].astype(bool)]
        offsets = pd.read_csv(directory / "car201_actual_roll_zero_points.csv")
        for _, row in offsets.iterrows():
            band = data.loc[data["filter"] == row["filter"]]
            pre = set(band.loc[band["pass"].isin(pre_epochs), "source_id"])
            post = set(band.loc[band["pass"].isin(post_epochs), "source_id"])
            records.append({"detector": detector, **row.to_dict(),
                            "cross_roll_fit_star_count": len(pre & post),
                            "identifiable_from_cross_roll_stars": bool(pre & post)})
    result = pd.DataFrame(records)
    output = PRODUCTION_ROOT / "wavelength_response_bundle/car201_all_detector_roll_offsets.csv"
    result.to_csv(output, index=False)
    identifiable = result.loc[result["identifiable_from_cross_roll_stars"]]
    print("Identifiable offsets: min / median / max (mmag)")
    print(identifiable.groupby("filter")["equivalent_magnitude_offset_mmag"].agg(["min", "median", "max"]).round(3).to_string())
    print("Median absolute identifiable offset (mmag):", identifiable.equivalent_magnitude_offset_mmag.abs().median())
    print(output)


if __name__ == "__main__":
    main()
