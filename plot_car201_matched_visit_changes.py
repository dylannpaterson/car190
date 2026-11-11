"""Compare same-star changes between adjacent visits without an inferred baseline."""

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time

from heatmap_car201_allstar_model_correction import (
    DEFAULT_OUTPUT_PREFIX, DETECTORS, FILTERS, PASSES, PRODUCTION_ROOT,
)


def quality_filter(data):
    dq = data["dq_flag"].to_numpy(np.int64)
    data["saturated"] = (dq >= 0) & ((dq & 2) != 0)
    saturated = data.groupby(["detector", "source_id"])["saturated"].transform("any")
    data = data.loc[~saturated & ~((dq >= 0) & ((dq & 1) != 0))].copy()
    exclusions = []
    offsets = []
    for detector in DETECTORS:
        lower = detector.lower()
        directory = (PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
                     / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched")
        saved = pd.read_csv(directory / "car201_actual_star_modulation_measurements.csv",
                            dtype={"source_id": str})
        mask = saved["repeat_outlier_excluded"] | saved["manual_measurement_excluded"]
        excluded = saved.loc[mask, ["source_id", "filter", "pass"]].copy()
        excluded["detector"] = detector
        exclusions.append(excluded)
        summary = json.loads((directory / "car201_actual_star_modulation_fit_summary.json").read_text())
        zp = summary["modulation"]["zeropoint_model"]
        saved_offsets = pd.read_csv(directory / "car201_actual_roll_zero_points.csv")
        for epoch, group in zip(PASSES, zp["pass_group_ids"], strict=True):
            for band in FILTERS:
                offset = 0.
                if group:
                    selected = saved_offsets.loc[(saved_offsets["filter"] == band)
                                                  & (saved_offsets["offset_group"] == zp["group_names"][group])]
                    if len(selected) != 1:
                        raise ValueError("Missing unique saved roll offset")
                    offset = float(selected["equivalent_magnitude_offset_mmag"].iloc[0]) / 1000.
                offsets.append({"detector": detector, "filter": band, "pass": epoch,
                                "roll_offset_mag": offset})
    keys = ["detector", "source_id", "filter", "pass"]
    data = data.merge(pd.concat(exclusions).drop_duplicates().assign(exclude=True), on=keys, how="left")
    data = data.loc[data["exclude"].isna()].drop(columns="exclude")
    return data.merge(pd.DataFrame(offsets), on=["detector", "filter", "pass"], validate="many_to_one")


def main():
    prefix = DEFAULT_OUTPUT_PREFIX
    data = pd.read_csv(prefix.with_name(prefix.name + "_all_observations.csv"), dtype={"source_id": str})
    unmod = pd.read_csv(prefix.with_name("car201_allstar_unmodulated_all_observations.csv"),
                        dtype={"source_id": str})
    keys = ["detector", "source_id", "filter", "pass"]
    data = data.merge(unmod[keys + ["model_delta_mag"]].rename(
        columns={"model_delta_mag": "unmodulated_model_mag"}), on=keys, validate="one_to_one")
    data = quality_filter(data)
    data["adjusted_observed_mag"] = data["mag_obs"] - data["roll_offset_mag"]
    data["modulated_ice_mag"] = data["model_delta_mag"] - data["roll_offset_mag"]
    data["unmodulated_ice_mag"] = data["unmodulated_model_mag"] - data["roll_offset_mag"]
    pair_keys = ["detector", "source_id", "filter", "pass_index"]
    columns = pair_keys + ["adjusted_observed_mag", "modulated_ice_mag", "unmodulated_ice_mag",
                          "day_from_fixed_ice_reference", "pass"]
    previous = data[columns].copy()
    previous["pass_index"] += 1
    pairs = data[columns].merge(previous, on=pair_keys, suffixes=("", "_previous"), validate="one_to_one")
    raw_change = pairs["adjusted_observed_mag"] - pairs["adjusted_observed_mag_previous"]
    for mode, model_column in (("no_ice_correction", None),
                               ("unmodulated", "unmodulated_ice_mag"),
                               ("modulated", "modulated_ice_mag")):
        correction = 0. if model_column is None else pairs[model_column] - pairs[model_column + "_previous"]
        pairs[mode + "_percent"] = 100 * np.expm1(-0.4 * np.log(10.) * (raw_change - correction))
    all_stats = []
    for mode in ("no_ice_correction", "unmodulated", "modulated"):
        grouped = pairs.groupby(["detector", "filter", "pass", "pass_previous"])
        stats = grouped[mode + "_percent"].quantile([.16, .5, .84]).unstack()
        stats.columns = ["p16_percent", "median_percent", "p84_percent"]
        stats["matched_stars"] = grouped.size()
        stats["ending_day"] = grouped["day_from_fixed_ice_reference"].median()
        stats["mode"] = mode
        all_stats.append(stats.reset_index())
    summary = pd.concat(all_stats, ignore_index=True)
    reference = Time("2026-09-12T01:25:50.714", format="isot", scale="utc").mjd
    summary["utc_date_number"] = mdates.date2num(Time(reference + summary["ending_day"].to_numpy(),
                                                      format="mjd", scale="utc").to_datetime())
    low, high = summary["p16_percent"].min(), summary["p84_percent"].max()
    margin = .05 * max(high-low, 1.)
    colors = plt.get_cmap("tab20")(np.arange(len(DETECTORS)))
    for mode in ("no_ice_correction", "unmodulated", "modulated"):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
        for band, axis in zip(FILTERS, axes.flat, strict=True):
            axis.axhspan(-.5, .5, color="#69ac67", alpha=.14)
            axis.axhline(0, color="0.4", linewidth=.6)
            for index, (detector, color) in enumerate(zip(DETECTORS, colors, strict=True)):
                rows = summary.loc[(summary["mode"] == mode) & (summary["filter"] == band)
                                   & (summary["detector"] == detector)]
                axis.errorbar(rows["utc_date_number"] + (index-8.5)/96., rows["median_percent"],
                              yerr=np.array([rows["median_percent"]-rows["p16_percent"],
                                             rows["p84_percent"]-rows["median_percent"]]),
                              fmt="o", color=color, markersize=2.5, elinewidth=.65, capsize=1.2)
            axis.set_title(band)
            axis.set_ylim(low-margin, high+margin)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            axis.tick_params(axis="x", labelrotation=50, labelsize=8)
            axis.grid(axis="y", alpha=.2)
        for axis in axes[-1]:
            axis.set_xlabel("UTC date of ending visit")
        handles = [Line2D([0], [0], color=color, label=detector)
                   for detector, color in zip(DETECTORS, colors, strict=True)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.945), ncol=9, frameon=False)
        label = {"no_ice_correction": "Roll-adjusted observations; no ice correction",
                 "unmodulated": "Unmodulated ice correction", "modulated": "Modulated ice correction"}[mode]
        fig.suptitle(label + " — same-star changes between consecutive visits", y=.985, fontsize=15)
        fig.supylabel("Visit-to-visit relative flux change (%)")
        fig.text(.5, .025,
                 "Each sample compares the same star/filter in two consecutive scheduled visits. "
                 "Points: detector medians; bars: star 16–84%; green: ±0.5%.\n"
                 "Fit quality cuts applied; missing pairs omitted. Saved roll offsets applied to data. "
                 "No inferred EP3 baselines or forced-zero anchors. All three figures share axes.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(.025,.1,1,.87))
        output = prefix.with_name(f"car201_matched_visit_changes_{mode}.png")
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(output)
    summary.to_csv(prefix.with_name("car201_matched_visit_changes_summary.csv"), index=False)
    pairs.to_csv(prefix.with_name("car201_matched_visit_changes_observations.csv"), index=False)
    print(f"Matched observations: {len(pairs)}; detector/star pairs: "
          f"{len(pairs[['detector','source_id']].drop_duplicates())}")


if __name__ == "__main__":
    main()
