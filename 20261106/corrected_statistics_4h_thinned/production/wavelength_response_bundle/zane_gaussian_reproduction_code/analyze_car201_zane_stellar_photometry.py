"""Inspect Zane's CAR201 WFI16 stellar photometry over Sep 28--Oct 14.

This is a data-only diagnostic: it matches Gaia source IDs across the selected
ep3 passes, keeps every matched source with a successful stellar-SED posterior,
and summarizes each filter's magnitude change relative to ep3_pass1. No ice
parameters are refit and DQ flags are reported, not interpreted or used as an
undocumented cut.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time


ROOT = Path(__file__).resolve().parent
CAR201_ROOT = ROOT.parent / "CAR201"
DEFAULT_PHOTOMETRY_DIR = CAR201_ROOT / "zanes_photometry"
DEFAULT_SED_POSTERIORS = CAR201_ROOT / "sed/car201-btsettl-posteriors.h5"
PASS_NAMES = (
    "ep3_pass1",
    "ep3_pass3",
    "ep3_pass4",
    "ep3_pass5",
    "ep3_pass6",
)
FILTERS = ("F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213")
ICE_REFERENCE_UTC = "2026-09-12T01:25:50.714"
WINDOW_START_UTC = "2026-09-28T00:00:00"
WINDOW_END_UTC = "2026-10-15T00:00:00"


def load_pass(path, start_mjd, end_mjd, detector, *, include_invalid=False):
    frame = pd.read_parquet(path)
    required = {
        "gaia_id",
        "filter",
        "sca",
        "mjd",
        "dq_flag",
        "mag_obs",
        "mag_err",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    mask = (
        (frame["sca"].astype(str) == detector)
        & frame["mjd"].between(start_mjd, end_mjd, inclusive="left")
        & frame["filter"].isin(FILTERS)
    )
    if not include_invalid:
        mask &= (
            np.isfinite(frame["mag_obs"])
            & np.isfinite(frame["mag_err"])
            & (frame["mag_err"] > 0.0)
        )
    frame = frame.loc[mask].copy()
    frame["source_id"] = (
        frame["gaia_id"].astype(str).str.replace("Gaia DR3 ", "", regex=False)
    )
    duplicate = frame.duplicated(["source_id", "filter"], keep=False)
    if duplicate.any():
        raise ValueError(f"{path} has duplicate source/filter measurements")
    return frame


def load_sed_posteriors(path):
    with h5py.File(path, "r") as archive:
        source_ids = [
            value.decode("ascii").rstrip("\x00")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in archive["source_id"][:]
        ]
        if "bands" in archive:
            band_counts = archive["bands"][:]
        elif "band_count" in archive:
            band_counts = archive["band_count"][:]
        else:
            band_counts = np.asarray(archive["band_used"][:]).sum(axis=1)
        effective_samples = (
            archive["effective_samples"][:]
            if "effective_samples" in archive
            else np.full(len(source_ids), np.nan)
        )
        return {
            source_id: {
                "parameters_median": np.median(samples, axis=0),
                "state": int(state),
                "bands": int(bands),
                "chi2": float(chi2),
                "effective_samples": float(effective_samples),
            }
            for source_id, samples, state, bands, chi2, effective_samples in zip(
                source_ids,
                archive["samples"][:],
                archive["state"][:],
                band_counts,
                archive["chi2"][:],
                effective_samples,
            )
        }


def select_sources_with_seds(common_ids, sed_by_id):
    selected = {}
    for source_id in sorted(common_ids):
        sed = sed_by_id.get(source_id)
        if sed is not None and sed["state"] == 1:
            selected[source_id] = sed
    return selected


def summarize_photometry(pass_frames, selected_ids, reference_pass, ice_reference_mjd):
    reference = pass_frames[reference_pass].set_index(["source_id", "filter"])
    records = []
    for pass_name, frame in pass_frames.items():
        current = frame.set_index(["source_id", "filter"])
        for filter_name in FILTERS:
            source_filter_ids = [
                source_id
                for source_id in selected_ids
                if (source_id, filter_name) in reference.index
                and (source_id, filter_name) in current.index
            ]
            ref_rows = reference.loc[
                [(source_id, filter_name) for source_id in source_filter_ids]
            ]
            current_rows = current.loc[
                [(source_id, filter_name) for source_id in source_filter_ids]
            ]
            delta_mag = current_rows["mag_obs"].to_numpy() - ref_rows["mag_obs"].to_numpy()
            delta_error = (
                np.zeros_like(delta_mag)
                if pass_name == reference_pass
                else np.hypot(
                    current_rows["mag_err"].to_numpy(),
                    ref_rows["mag_err"].to_numpy(),
                )
            )
            mjd = float(np.median(current_rows["mjd"].to_numpy()))
            p16, median, p84 = np.percentile(delta_mag, [16, 50, 84])
            records.append(
                {
                    "pass": pass_name,
                    "filter": filter_name,
                    "mjd": mjd,
                    "utc": Time(mjd, format="mjd", scale="utc").isot,
                    "day_from_fixed_ice_reference": mjd - ice_reference_mjd,
                "matched_sed_sources": len(delta_mag),
                    "median_delta_mag": median,
                    "population_p16_delta_mag": p16,
                    "population_p84_delta_mag": p84,
                    "median_per_star_delta_error_mag": float(np.median(delta_error)),
                    "median_current_mag_error_mag": float(
                        np.median(current_rows["mag_err"].to_numpy())
                    ),
                    "dq_zero_fraction": float(
                        np.mean(current_rows["dq_flag"].to_numpy() == 0)
                    ),
                }
            )
    return pd.DataFrame(records)


def plot_photometry(summary, output_path, detector, star_count, first_utc, last_utc):
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 10.0), sharex=True)
    palette = plt.get_cmap("viridis")
    for filter_index, (axis, filter_name) in enumerate(zip(axes.flat, FILTERS)):
        values = summary.loc[summary["filter"] == filter_name].sort_values(
            "day_from_fixed_ice_reference"
        )
        x = values["day_from_fixed_ice_reference"].to_numpy()
        median_mmag = 1000.0 * values["median_delta_mag"].to_numpy()
        lower_mmag = median_mmag - 1000.0 * values["population_p16_delta_mag"].to_numpy()
        upper_mmag = 1000.0 * values["population_p84_delta_mag"].to_numpy() - median_mmag
        axis.errorbar(
            x,
            median_mmag,
            yerr=np.vstack((lower_mmag, upper_mmag)),
            fmt="o",
            linestyle="none",
            markersize=4.5,
            color=palette(filter_index / max(1, len(FILTERS) - 1)),
            ecolor="0.55",
            elinewidth=1.0,
            capsize=2,
            label="median across matched stars; bars are population p16–p84",
        )
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        axis.set_title(filter_name)
        axis.grid(True, alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Magnitude change (mmag)")
    axes[-1, 0].set_xlabel("Days from fixed ice-history reference (2026-09-12)")
    axes[-1, 1].set_xlabel("Days from fixed ice-history reference (2026-09-12)")
    axes[0, 0].legend(fontsize=7, loc="best")
    figure.suptitle(
        f"CAR201 Zane photometry: {detector} SED-matched stars, {star_count} matched\n"
        f"{first_utc} to {last_utc}; changes relative to ep3_pass1; data-only diagnostic",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photometry-dir", type=Path, default=DEFAULT_PHOTOMETRY_DIR)
    parser.add_argument("--sed-posteriors", type=Path, default=DEFAULT_SED_POSTERIORS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI16")
    args = parser.parse_args(argv)

    start_mjd = float(Time(WINDOW_START_UTC, format="isot", scale="utc").mjd)
    end_mjd = float(Time(WINDOW_END_UTC, format="isot", scale="utc").mjd)
    ice_reference_mjd = float(
        Time(ICE_REFERENCE_UTC, format="isot", scale="utc").mjd
    )
    pass_frames = {
        pass_name: load_pass(
            args.photometry_dir / f"{pass_name}_master.parquet",
            start_mjd,
            end_mjd,
            args.detector,
        )
        for pass_name in PASS_NAMES
    }
    common_ids = set.intersection(
        *(
            set(frame["source_id"].unique())
            for frame in pass_frames.values()
        )
    )
    sed_by_id = load_sed_posteriors(args.sed_posteriors)
    selected_seds = select_sources_with_seds(common_ids, sed_by_id)
    if not selected_seds:
        raise ValueError("No common sources have successful SED posterior fits")

    summary = summarize_photometry(
        pass_frames,
        set(selected_seds),
        PASS_NAMES[0],
        ice_reference_mjd,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "car201_zane_wfi16_all_sed_photometry_sep28_oct14.csv"
    summary.to_csv(summary_csv, index=False)

    sed_csv = args.output_dir / "car201_zane_wfi16_all_sed_matches.csv"
    with sed_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gaia_source_id",
                "teff_k_median",
                "logg_median",
                "metallicity_median_dex",
                "av_median_mag",
                "sed_band_count",
                "sed_chi2",
                "sed_effective_samples",
            ]
        )
        for source_id, sed in selected_seds.items():
            parameters = sed["parameters_median"]
            writer.writerow(
                [
                    source_id,
                    parameters[0],
                    parameters[1],
                    parameters[2],
                    parameters[4],
                    sed["bands"],
                    sed["chi2"],
                    sed["effective_samples"],
                ]
            )

    plot_path = args.output_dir / "car201_zane_wfi16_all_sed_photometry_sep28_oct14.png"
    plot_photometry(
        summary,
        plot_path,
        args.detector,
        len(selected_seds),
        WINDOW_START_UTC,
        "2026-10-14T23:59:59",
    )

    first_mjd = float(min(frame["mjd"].min() for frame in pass_frames.values()))
    last_mjd = float(max(frame["mjd"].max() for frame in pass_frames.values()))
    seds = np.asarray([item["parameters_median"] for item in selected_seds.values()])
    dq_zero_fraction = float(
        np.mean(
            np.concatenate(
                [frame["dq_flag"].to_numpy() == 0 for frame in pass_frames.values()]
            )
        )
    )
    report = {
        "photometry_source": str(args.photometry_dir),
        "sed_source": str(args.sed_posteriors),
        "detector": args.detector,
        "pass_files": [f"{name}_master.parquet" for name in PASS_NAMES],
        "date_window_utc": [WINDOW_START_UTC, WINDOW_END_UTC],
        "mjd_range": [first_mjd, last_mjd],
        "day_from_ice_reference_range": [
            first_mjd - ice_reference_mjd,
            last_mjd - ice_reference_mjd,
        ],
        "fixed_ice_reference_utc": ICE_REFERENCE_UTC,
        "common_source_ids_in_all_passes": len(common_ids),
        "matched_g_type_sed_sources": len(selected_seds),
        "sed_selection": "all matched sources with successful posterior state; no Teff, band-count, or chi-square cut",
        "matched_sed_median_parameters": {
            "teff_k": float(np.median(seds[:, 0])),
            "logg": float(np.median(seds[:, 1])),
            "metallicity_dex": float(np.median(seds[:, 2])),
            "av_mag": float(np.median(seds[:, 4])),
        },
        "matched_sed_quality_counts": {
            "band_count_ge_5": int(sum(item["bands"] >= 5 for item in selected_seds.values())),
            "chi2_lt_25": int(sum(item["chi2"] < 25.0 for item in selected_seds.values())),
            "both": int(
                sum(
                    item["bands"] >= 5 and item["chi2"] < 25.0
                    for item in selected_seds.values()
                )
            ),
        },
        "dq_zero_fraction_reported_not_filtered": dq_zero_fraction,
        "dq_interpretation": "No mask applied; preserve flags until the parquet producer's flag definitions are confirmed.",
        "photometry_plot": str(plot_path),
        "photometry_summary_csv": str(summary_csv),
        "matched_sed_csv": str(sed_csv),
        "note": "Data-only summary of all successful SED matches; no modulation or ice fit is performed in this diagnostic.",
    }
    summary_path = args.output_dir / "car201_zane_wfi16_all_sed_photometry_sep28_oct14_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WFI16: {len(common_ids)} common Gaia IDs, {len(selected_seds)} successful SED matches")
    print(f"Data span: MJD {first_mjd:.5f}–{last_mjd:.5f}, relative day {first_mjd-ice_reference_mjd:.2f}–{last_mjd-ice_reference_mjd:.2f}")
    print(f"DQ==0 fraction reported only: {dq_zero_fraction:.1%}")
    for path in (plot_path, summary_csv, sed_csv, summary_path):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
