"""Histogram Roman DQ flags for v2-SED-matched CAR201 post-roll photometry."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

from analyze_car201_zane_stellar_photometry import (
    DEFAULT_PHOTOMETRY_DIR,
    DEFAULT_SED_POSTERIORS,
    FILTERS,
    load_sed_posteriors,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_POSTROLL_PHOTOMETRY = DEFAULT_PHOTOMETRY_DIR / "20261106"
DEFAULT_SED_V2 = DEFAULT_SED_POSTERIORS.parent / "v2/faint-posteriors-v2.h5"
POSTROLL_PASSES = ("ep4_pass2", "ep4_pass3", "ep4_pass4", "ep4_pass5")
START_UTC = "2026-10-18T00:00:00"
END_UTC = "2026-11-01T00:00:00"

# Standard Roman PIXELDQ flags documented by roman_datamodels.dqflags.pixel.
DQ_FLAGS = (
    (0, "DO_NOT_USE", "Bad pixel; do not use"),
    (1, "SATURATED", "Pixel saturated during exposure"),
    (2, "JUMP_DET", "Jump detected during exposure"),
    (3, "DROPOUT", "Data lost in transmission"),
    (4, "GW_AFFECTED_DATA", "Affected by the guide-window read window"),
    (5, "PERSISTENCE", "High persistence"),
    (6, "AD_FLOOR", "Below the A/D floor"),
    (7, "OUTLIER", "Flagged by outlier detection"),
    (8, "UNRELIABLE_ERROR", "Uncertainty exceeds quoted error"),
    (9, "NON_SCIENCE", "Not on the science portion of the detector"),
    (10, "DEAD", "Dead pixel"),
    (11, "HOT", "Hot pixel"),
    (12, "WARM", "Warm pixel"),
    (13, "LOW_QE", "Low quantum efficiency"),
    (15, "TELEGRAPH", "Telegraph pixel"),
    (16, "NONLINEAR", "Highly nonlinear pixel"),
    (17, "BAD_REF_PIXEL", "Bad reference pixel"),
    (18, "NO_FLAT_FIELD", "Flat field cannot be measured"),
    (19, "NO_GAIN_VALUE", "Gain cannot be measured"),
    (20, "NO_LIN_CORR", "Linearity correction unavailable"),
    (21, "NO_SAT_CHECK", "Saturation check unavailable"),
    (22, "UNRELIABLE_BIAS", "Bias variance is large"),
    (23, "UNRELIABLE_DARK", "Dark variance is large"),
    (24, "UNRELIABLE_SLOPE", "Slope variance is large/noisy pixel"),
    (25, "UNRELIABLE_FLAT", "Flat-field variance is large"),
    (28, "UNRELIABLE_RESET", "Sensitive to reset anomaly"),
    (30, "OTHER_BAD_PIXEL", "Catch-all bad-pixel flag"),
    (31, "REFERENCE_PIXEL", "Reference pixel"),
)
RESERVED_BITS = (14, 26, 27, 29)


def load_postroll_sed_rows(photometry_dir, sed_path):
    start_mjd = float(Time(START_UTC, format="isot", scale="utc").mjd)
    end_mjd = float(Time(END_UTC, format="isot", scale="utc").mjd)
    sed_by_id = load_sed_posteriors(sed_path)
    successful_ids = {
        source_id
        for source_id, sed in sed_by_id.items()
        if sed["state"] == 1
    }
    frames = []
    for pass_name in POSTROLL_PASSES:
        frame = pd.read_parquet(
            Path(photometry_dir) / f"{pass_name}_master.parquet",
            columns=("gaia_id", "filter", "sca", "mjd", "dq_flag", "mag_obs", "mag_err"),
        )
        frame = frame.loc[
            frame["mjd"].between(start_mjd, end_mjd, inclusive="left")
            & frame["filter"].isin(FILTERS)
        ].copy()
        frame["source_id"] = frame["gaia_id"].astype(str).str.replace(
            "Gaia DR3 ", "", regex=False
        )
        frame["pass"] = pass_name
        frame = frame.loc[frame["source_id"].isin(successful_ids)]
        frames.append(frame)
    if not frames:
        raise ValueError("No post-roll photometry frames were loaded")
    return pd.concat(frames, ignore_index=True)


def summarize_flags(rows):
    dq = pd.to_numeric(rows["dq_flag"], errors="coerce").fillna(-1).astype(np.int64)
    recorded = dq >= 0
    total_rows = len(rows)
    bit_records = []
    detector_rows = []
    named_mask = 0
    for bit, name, description in DQ_FLAGS:
        flag_value = 1 << bit
        named_mask |= flag_value
        flagged = recorded & ((dq & flag_value) != 0)
        bit_records.append(
            {
                "bit": bit,
                "value": flag_value,
                "name": name,
                "description": description,
                "measurement_rows": int(flagged.sum()),
                "fraction_of_sed_matched_rows": float(flagged.sum() / total_rows)
                if total_rows
                else 0.0,
                "unique_detector_source_pairs": int(
                    rows.loc[flagged, ["sca", "source_id"]].drop_duplicates().shape[0]
                ),
            }
        )

    reserved_mask = sum(1 << bit for bit in RESERVED_BITS)
    all_known_mask = named_mask | reserved_mask
    unknown_flags = recorded & ((dq & np.int64(~all_known_mask)) != 0)
    for detector, frame in rows.groupby("sca", sort=True):
        values = pd.to_numeric(frame["dq_flag"], errors="coerce").fillna(-1).astype(np.int64)
        known = values >= 0
        detector_rows.append(
            {
                "detector": detector,
                "sed_matched_measurement_rows": int(len(frame)),
                "rows_with_recorded_dq": int(known.sum()),
                "rows_dq_zero": int((known & (values == 0)).sum()),
                "rows_dq_nonzero": int((known & (values != 0)).sum()),
                "rows_dq_missing": int((~known).sum()),
                **{
                    name: int((known & ((values & (1 << bit)) != 0)).sum())
                    for bit, name, _ in DQ_FLAGS
                },
            }
        )

    mask_rows = []
    known_dq = rows.loc[recorded].copy()
    known_dq["dq_mask"] = dq[recorded].astype(np.uint32)
    for mask, group in known_dq.groupby("dq_mask", sort=True):
        mask_value = int(mask)
        bits = [
            name
            for bit, name, _ in DQ_FLAGS
            if mask_value & (1 << bit)
        ]
        bits.extend(f"RESERVED_{bit}" for bit in RESERVED_BITS if mask_value & (1 << bit))
        unknown = mask_value & ~all_known_mask
        if unknown:
            bits.append(f"UNKNOWN_BITS_0x{unknown:x}")
        mask_rows.append(
            {
                "dq_flag": mask_value,
                "flag_names": ";".join(bits) if bits else "GOOD",
                "measurement_rows": int(len(group)),
                "unique_detector_source_pairs": int(
                    group[["sca", "source_id"]].drop_duplicates().shape[0]
                ),
            }
        )
    return bit_records, detector_rows, mask_rows, int(unknown_flags.sum())


def plot_histogram(bit_records, detector_rows, output_path):
    counts = np.asarray([row["measurement_rows"] for row in bit_records], dtype=int)
    labels = [f"{row['name']} (bit {row['bit']})" for row in bit_records]
    detectors = [row["detector"] for row in detector_rows]
    rates = np.asarray(
        [
            [row[name] / row["sed_matched_measurement_rows"] for _, name, _ in DQ_FLAGS]
            for row in detector_rows
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(18.0, 10.5), gridspec_kw={"width_ratios": (1.2, 1.8)}
    )
    y = np.arange(len(bit_records))
    axes[0].barh(y, counts, color="#507fa6")
    axes[0].set_yticks(y, labels=labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("SED-matched measurement rows with bit set")
    axes[0].set_title("DQ-bit counts (bits may overlap)")
    axes[0].grid(True, axis="x", alpha=0.25)

    image = axes[1].imshow(rates, aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_yticks(np.arange(len(detectors)), labels=detectors)
    axes[1].set_xticks(
        np.arange(len(DQ_FLAGS)),
        labels=[f"{name}\n{bit}" for bit, name, _ in DQ_FLAGS],
        rotation=70,
        ha="right",
        fontsize=7,
    )
    axes[1].set_title("Fraction of each detector's SED-matched rows flagged")
    colorbar = figure.colorbar(image, ax=axes[1], pad=0.02)
    colorbar.set_label("Fraction of measurement rows")
    figure.suptitle(
        "CAR201 v2-SED-matched post-roll photometry: raw Roman DQ flags\n"
        "ep4_pass2–ep4_pass5; no SATURATED/DO_NOT_USE filtering applied",
        y=0.995,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photometry-dir", type=Path, default=DEFAULT_POSTROLL_PHOTOMETRY)
    parser.add_argument("--sed-posteriors", type=Path, default=DEFAULT_SED_V2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "20261106_postroll_faint_v2_sed_dq",
    )
    args = parser.parse_args(argv)

    rows = load_postroll_sed_rows(args.photometry_dir, args.sed_posteriors)
    bit_records, detector_rows, mask_rows, unknown_rows = summarize_flags(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bit_csv = args.output_prefix.with_name(args.output_prefix.name + "_bit_counts.csv")
    detector_csv = args.output_prefix.with_name(
        args.output_prefix.name + "_detector_counts.csv"
    )
    masks_csv = args.output_prefix.with_name(args.output_prefix.name + "_mask_counts.csv")
    plot_path = args.output_prefix.with_name(args.output_prefix.name + "_histogram.png")
    pd.DataFrame(bit_records).to_csv(bit_csv, index=False)
    pd.DataFrame(detector_rows).to_csv(detector_csv, index=False)
    pd.DataFrame(mask_rows).sort_values(
        "measurement_rows", ascending=False
    ).to_csv(masks_csv, index=False)
    plot_histogram(bit_records, detector_rows, plot_path)

    print(f"Post-roll passes: {', '.join(POSTROLL_PASSES)}")
    print(f"Successful v2 SED-matched measurement rows: {len(rows)}")
    print(
        "Unique detector/source pairs: "
        f"{rows[['sca', 'source_id']].drop_duplicates().shape[0]}"
    )
    print(f"Rows with missing DQ: {int(rows['dq_flag'].isna().sum())}")
    print(f"Rows with reserved/unknown DQ bits: {unknown_rows}")
    for path in (plot_path, bit_csv, detector_csv, masks_csv):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
