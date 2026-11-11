"""Per-detector epoch medians and individual-star percentile envelopes."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import argparse
from astropy.time import Time

from heatmap_car201_allstar_model_correction import DEFAULT_OUTPUT_PREFIX, DETECTORS, FILTERS, PASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmodulated", action="store_true")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--nn-only", action="store_true")
    parser.add_argument("--roll-offset-to-data", action="store_true")
    parser.add_argument("--ep3-reference-only", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    args = parser.parse_args()
    prefix = (DEFAULT_OUTPUT_PREFIX.with_name("car201_allstar_unmodulated")
              if args.unmodulated else DEFAULT_OUTPUT_PREFIX)
    data = pd.read_csv(prefix.with_name(prefix.name + "_all_observations.csv"))
    # Match production photometry quality policy: exclude an entire source if
    # any measurement is saturated, then exclude individual DO_NOT_USE rows.
    dq = data["dq_flag"].to_numpy(np.int64)
    data["saturated"] = (dq >= 0) & ((dq & 2) != 0)
    saturated_source = data.groupby(["detector", "source_id"])["saturated"].transform("any")
    data = data.loc[~saturated_source & ~((dq >= 0) & ((dq & 1) != 0))].copy()
    # Reuse the fit's explicit manual/repeat-outlier row exclusions for sources
    # present in its tables, without restricting to the SED-selected fit cohort.
    from heatmap_car201_allstar_model_correction import PRODUCTION_ROOT
    exclusions = []
    for detector in DETECTORS:
        lower = detector.lower()
        path = (PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
                / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched"
                / "car201_actual_star_modulation_measurements.csv")
        saved = pd.read_csv(path, dtype={"source_id": str})
        mask = saved["repeat_outlier_excluded"].astype(bool) | saved["manual_measurement_excluded"].astype(bool)
        rows = saved.loc[mask, ["source_id", "filter", "pass"]].copy()
        rows["detector"] = detector
        exclusions.append(rows)
    data["source_id"] = data["source_id"].astype(str)
    excluded = pd.concat(exclusions, ignore_index=True).assign(exclude=True)
    data = data.merge(excluded, on=["detector", "source_id", "filter", "pass"], how="left")
    data = data.loc[data["exclude"].isna()].copy()
    if args.nn_only:
        assignments = pd.read_csv(
            prefix.with_name(prefix.name + "_sed_assignments.csv"),
            dtype={"source_id": str},
        )
        nn_sources = assignments.loc[
            assignments["sed_source"] == "nearest_neighbor_magnitudes",
            ["detector", "source_id"],
        ].drop_duplicates()
        data = data.merge(nn_sources, on=["detector", "source_id"], how="inner", validate="many_to_one")
        print(f"NN-only quality-filtered sample: {len(data)} observations; "
              f"{len(data[['detector', 'source_id']].drop_duplicates())} detector/star pairs")
    from heatmap_car201_allstar_model_correction import _assign_relative_changes
    if args.roll_offset_to_data:
        import json
        offset_tables = []
        for detector in DETECTORS:
            lower = detector.lower()
            directory = (PRODUCTION_ROOT / lower / "spline_relative_response_normal/experiments"
                         / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched")
            fit_summary = json.loads((directory / "car201_actual_star_modulation_fit_summary.json").read_text())
            zp = fit_summary["modulation"]["zeropoint_model"]
            offsets = pd.read_csv(directory / "car201_actual_roll_zero_points.csv")
            for epoch, group_id in zip(PASSES, zp["pass_group_ids"], strict=True):
                for band in FILTERS:
                    offset = 0.0
                    if group_id:
                        selected = offsets.loc[(offsets["filter"] == band)
                                               & (offsets["offset_group"] == zp["group_names"][group_id])]
                        if len(selected) != 1:
                            raise ValueError(f"Missing roll offset: {detector}/{band}/{epoch}")
                        offset = float(selected["equivalent_magnitude_offset_mmag"].iloc[0]) / 1000.
                    offset_tables.append({"detector": detector, "filter": band,
                                          "pass": epoch, "applied_roll_offset_mag": offset})
        data = data.merge(pd.DataFrame(offset_tables), on=["detector", "filter", "pass"],
                          validate="many_to_one")
        if data["applied_roll_offset_mag"].isna().any():
            raise ValueError("A measurement lacks a roll-offset assignment")
        before = _assign_relative_changes(data.copy())
        data["mag_obs"] -= data["applied_roll_offset_mag"]
        data["model_delta_mag"] -= data["applied_roll_offset_mag"]
    data = _assign_relative_changes(data)
    if args.roll_offset_to_data:
        difference = np.max(np.abs(data["corrected_delta_mag"].to_numpy()
                                   - before["corrected_delta_mag"].to_numpy()))
        if difference > 1e-10:
            raise ValueError(f"Moving roll offsets changed corrected residuals by {difference} mag")
        print(f"Roll-offset transfer verified: maximum corrected residual change {difference:.3g} mag")
    if args.ep3_reference_only:
        all_repeated = data.loc[data["observations_per_source_filter"] >= 2]
        counts = []
        for detector in DETECTORS:
            for band in FILTERS:
                rows = all_repeated.loc[(all_repeated["detector"] == detector)
                                        & (all_repeated["filter"] == band)]
                kept = rows.loc[rows["anchor_pass"] == "ep3_pass1"]
                counts.append({"detector": detector, "filter": band,
                               "all_quality_repeated_stars": rows["source_id"].nunique(),
                               "ep3_reference_stars": kept["source_id"].nunique(),
                               "all_nonreference_observations": int((~rows["is_anchor_observation"]).sum()),
                               "ep3_reference_nonreference_observations": int((~kept["is_anchor_observation"]).sum())})
        data = data.loc[data["anchor_pass"] == "ep3_pass1"].copy()
        coverage = pd.DataFrame(counts)
        coverage.to_csv(DEFAULT_OUTPUT_PREFIX.with_name("car201_ep3_reference_sample_coverage.csv"), index=False)
        print("EP3-reference plot sample:",
              int(coverage.ep3_reference_stars.sum()), "detector/star/filter groups;",
              int(coverage.ep3_reference_nonreference_observations.sum()), "non-reference observations of",
              int(coverage.all_nonreference_observations.sum()))
    if args.no_correction:
        data["corrected_flux_change_percent"] = data["raw_flux_change_percent"]
    data = data.loc[(data["observations_per_source_filter"] >= 2)
                    & ~data["is_anchor_observation"].astype(bool)].copy()
    # Never combine changes measured relative to different reference epochs.
    # Retain every star, but summarize each exact anchor-pass cohort separately.
    group_keys = ["filter", "detector", "pass", "anchor_pass"]
    grouped = data.groupby(group_keys)["corrected_flux_change_percent"]
    summary = grouped.quantile([0.16, 0.5, 0.84]).unstack()
    summary.columns = ["p16_percent", "median_percent", "p84_percent"]
    summary["observations"] = grouped.size()
    summary = summary.reset_index()
    from heatmap_car201_allstar_model_correction import fit_model
    reference_mjd = Time(fit_model.ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    observation_days = data.groupby(group_keys)[
        "day_from_fixed_ice_reference"
    ].median().rename("observation_day").reset_index()
    summary = summary.merge(observation_days, on=group_keys, validate="one_to_one")
    summary["utc_date_number"] = mdates.date2num(
        Time(reference_mjd + summary["observation_day"].to_numpy(float),
             format="mjd", scale="utc").to_datetime()
    )
    if args.ylim is not None:
        shared_limits = args.ylim
    else:
        tables = [summary]
        counterpart = DEFAULT_OUTPUT_PREFIX.with_name(
            "car201_corrected_detector_epoch_errorbars.csv"
            if args.unmodulated else "car201_unmodulated_detector_epoch_errorbars.csv"
        )
        if counterpart.is_file():
            tables.append(pd.read_csv(counterpart))
        for name in ("car201_corrected_detector_epoch_errorbars.csv",
                     "car201_unmodulated_detector_epoch_errorbars.csv",
                     "car201_uncorrected_detector_epoch_errorbars.csv"):
            path = DEFAULT_OUTPUT_PREFIX.with_name(name)
            if path.is_file():
                tables.append(pd.read_csv(path))
        if args.roll_offset_to_data:
            for stem in ("car201_corrected_detector_epoch_errorbars",
                         "car201_unmodulated_detector_epoch_errorbars",
                         "car201_uncorrected_detector_epoch_errorbars"):
                path = DEFAULT_OUTPUT_PREFIX.with_name(stem + "_roll_offset_on_data.csv")
                if path.is_file():
                    tables.append(pd.read_csv(path))
        lower = min(float(table["p16_percent"].min()) for table in tables)
        upper = max(float(table["p84_percent"].max()) for table in tables)
        margin = 0.05 * max(upper - lower, 1.0)
        shared_limits = [lower - margin, upper + margin]
    colors = plt.get_cmap("tab20")(np.arange(len(DETECTORS)))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    for band, axis in zip(FILTERS, axes.flat, strict=True):
        axis.axhspan(-0.5, 0.5, color="#69ac67", alpha=0.14, zorder=0)
        axis.axhline(0, color="0.4", linewidth=0.65, zorder=0)
        for detector_index, (detector, color) in enumerate(zip(DETECTORS, colors, strict=True)):
            detector_rows = summary.loc[(summary["filter"] == band)
                                        & (summary["detector"] == detector)]
            for anchor, rows in detector_rows.groupby("anchor_pass"):
                rows = rows.set_index("pass").reindex(PASSES)
                low = rows["p16_percent"].to_numpy(float)
                median = rows["median_percent"].to_numpy(float)
                high = rows["p84_percent"].to_numpy(float)
                marker = "o" if anchor == "ep3_pass1" else "^" if anchor == "ep4_pass2" else "x"
                cohort_offset = -0.12 if anchor == "ep3_pass1" else 0.12 if anchor == "ep4_pass2" else 0.
                detector_offset_days = (detector_index - (len(DETECTORS) - 1) / 2) / 96.0
                axis.errorbar(
                    rows["utc_date_number"].to_numpy(float) + detector_offset_days + cohort_offset,
                    median, yerr=np.asarray([median - low, high - median]),
                    fmt=marker, color=color, markersize=2.8,
                    markerfacecolor="none" if marker == "^" else color,
                    elinewidth=0.65, capsize=1.2, capthick=0.65, alpha=0.85,
                )
        axis.set_title(band)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x", labelrotation=50, labelsize=8)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlim(summary["utc_date_number"].min() - 0.6,
                      summary["utc_date_number"].max() + 0.6)
        axis.set_ylim(*shared_limits)
    for axis in axes[-1, :]:
        axis.set_xlabel("UTC date")
    handles = [Line2D([0], [0], color=color, label=detector, linewidth=2)
               for detector, color in zip(DETECTORS, colors, strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=9, fontsize=9, frameon=False)
    cohort_handles = [
        Line2D([0], [0], color="0.25", marker="o", linestyle="none", label="Reference: EP3 pass 1"),
    ]
    if (summary["anchor_pass"] == "ep4_pass2").any():
        cohort_handles.append(Line2D([0], [0], color="0.25", marker="^", markerfacecolor="none",
                                     linestyle="none", label="Reference: EP4 pass 2 (post-roll)"))
    if (~summary["anchor_pass"].isin(["ep3_pass1", "ep4_pass2"])).any():
        cohort_handles.append(Line2D([0], [0], color="0.25", marker="x", linestyle="none",
                                     label="Other references (each epoch kept separate)"))
    fig.legend(handles=cohort_handles, loc="upper center", bbox_to_anchor=(0.5, 0.895),
               ncol=3, fontsize=9, frameon=False)
    title = ("Unmodulated ice model: corrected star changes by epoch and detector"
             if args.unmodulated else "Corrected star changes by epoch and detector")
    if args.no_correction:
        title = "No correction: observed star changes by epoch and detector"
    if args.roll_offset_to_data:
        title = ("Roll-offset-adjusted data; no ice correction" if args.no_correction
                 else "Unmodulated ice correction; roll offsets applied to data" if args.unmodulated
                 else "Modulated ice correction; roll offsets applied to data")
    if args.nn_only:
        title += " — NN-imputed SED stars only"
    if args.ep3_reference_only:
        title += " — measured EP3-pass-1 references only"
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.supylabel("Observed relative flux change (%)" if args.no_correction
                  else "Corrected relative flux change (%)")
    fig.text(0.5, 0.025,
             "Points: detector medians within each reference cohort. Error bars: star 16th–84th percentiles. Green band: ±0.5%.\n"
             "Fit DQ policy: saturated stars excluded entirely; DO_NOT_USE and saved fit-outlier rows excluded. References recomputed after cuts.\n"
             + ("NN-imputed SED stars only; forced-zero reference observations omitted."
               if args.nn_only else "Quality-filtered EP3-reference stars, including NN SEDs; reference observations omitted."
               if args.ep3_reference_only else "All remaining catalog stars, including NN SEDs; forced-zero reference observations omitted.")
             + ("\nSaved roll offset subtracted from observed magnitude and removed from the model; no new offsets fitted."
                if args.roll_offset_to_data else "")
             + (f"\nShared display: {shared_limits[0]:.2f}% to {shared_limits[1]:.2f}%; full cohort quantiles and counts retained in CSV."
                if args.ylim is not None else "")
             + ("\nPlot subset only: each star/filter has a quality-valid EP3-pass-1 measurement; fit cohort unchanged."
                if args.ep3_reference_only else ""),
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0.025, 0.12, 1, 0.84))
    output = prefix.with_name("car201_unmodulated_detector_epoch_errorbars.png"
                              if args.unmodulated else "car201_corrected_detector_epoch_errorbars.png")
    if args.no_correction:
        output = prefix.with_name("car201_uncorrected_detector_epoch_errorbars.png")
    if args.nn_only:
        output = output.with_name(output.stem + "_nn_only.png")
    if args.roll_offset_to_data:
        output = output.with_name(output.stem + "_roll_offset_on_data.png")
    if args.ep3_reference_only:
        output = output.with_name(output.stem + "_ep3_reference_only.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    summary.to_csv(output.with_suffix(".csv"), index=False)
    print(output)


if __name__ == "__main__":
    main()
