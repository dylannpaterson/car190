"""Plot Amber CAR201 differential-color data against full-history and H5 models.

The default comparison is EP3 pass 5 relative to the EP3 pass-1 reference:
F129-F184 change versus the reference F087-F158 color. It compares the saved
full-history joint-fit predictions with predictions from the reader-compatible
H5 export; no model fit is performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import odr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    ROOT
    / "20261117/production/pipeline_runs/relative_response/production/plots/"
    "relative_response/car201/amber_eric_v3_vs_joint_modulated_measurement_predictions.csv"
)
DEFAULT_AMBER_PHOTOMETRY = (
    ROOT / "car201_data/ambers_photometry/photometry-all-star-epochs.parquet"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "20261117/production/pipeline_runs/relative_response/analysis-figures"
)
DEFAULT_MODEL_VALIDATION = (
    ROOT
    / "20261117/production/pipeline_runs/relative_response/production/"
    "wavelength_response_bundle/final/joint_modulated_v2.validation.json"
)
KEYS = ["detector", "source_id"]
FILTERS = ("F087", "F158", "F129", "F184")
REFERENCE_PASS = "ep3_pass1"
DEFAULT_TARGET_PASS = "ep3_pass5"
MODEL_COLUMN = "joint_modulated_model_delta_mag"
FULL_HISTORY_COLUMN = "saved_joint_fit_model_delta_mag"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--amber-photometry", type=Path, default=DEFAULT_AMBER_PHOTOMETRY)
    parser.add_argument("--model-validation", type=Path, default=DEFAULT_MODEL_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stem", default="amber_ep3p5_f129_f184_vs_f087_f158_anomaly"
    )
    parser.add_argument("--target-pass", default=DEFAULT_TARGET_PASS)
    parser.add_argument("--bins", type=int, default=6)
    return parser.parse_args(argv)


def _pivot_field(frame: pd.DataFrame, field: str, prefix: str) -> pd.DataFrame:
    values = frame.pivot(index=KEYS, columns="filter", values=field)
    values.columns = [f"{prefix}_{name}" for name in values.columns]
    return values


def _reference_errors(
    amber_path: Path, target_rows: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Match reference magnitudes back to Amber rows to recover their errors."""
    reference_rows = target_rows[
        KEYS + ["filter", "reference_mag_obs"]
    ].drop_duplicates()
    raw = pd.read_parquet(
        amber_path,
        columns=["source_id", "sca", "filter", "mag_obs", "mag_err"],
    )
    raw = raw.loc[raw["filter"].isin(FILTERS)].copy()
    raw["source_id"] = raw["source_id"].astype(str)
    raw["detector"] = raw["sca"].astype(str)
    raw = raw[KEYS + ["filter", "mag_obs", "mag_err"]]

    candidates = reference_rows.merge(
        raw, on=KEYS + ["filter"], how="left", validate="one_to_many"
    )
    candidates["reference_match_abs_diff"] = np.abs(
        candidates["mag_obs"] - candidates["reference_mag_obs"]
    )
    candidates = candidates.sort_values(
        ["reference_match_abs_diff", "mag_err"], na_position="last"
    ).drop_duplicates(KEYS + ["filter"], keep="first")
    matched = candidates["reference_match_abs_diff"].to_numpy(dtype=float)
    finite = np.isfinite(matched)
    if not np.all(finite):
        missing = int(np.count_nonzero(~finite))
        raise ValueError(f"Could not match {missing} Amber reference magnitudes to input rows")
    max_difference = float(np.max(matched)) if matched.size else 0.0
    if max_difference > 1.0e-5:
        raise ValueError(
            "Amber reference-magnitude/error match is not exact enough: "
            f"maximum absolute difference is {max_difference:.6g} mag"
        )

    errors = candidates.pivot(index=KEYS, columns="filter", values="mag_err")
    errors.columns = [f"referr_{name}" for name in errors.columns]
    return errors, max_difference


def _assemble_sample(
    predictions_path: Path,
    amber_path: Path,
    target_pass: str,
) -> tuple[pd.DataFrame, float]:
    predictions = pd.read_csv(predictions_path, dtype={"source_id": str})
    required = {
        *KEYS,
        "filter",
        "pass",
        "mag_obs",
        "mag_err",
        "reference_mag_obs",
        "observed_delta_mag",
        MODEL_COLUMN,
        FULL_HISTORY_COLUMN,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    target = predictions.loc[
        predictions["pass"].eq(target_pass)
        & predictions["filter"].isin(FILTERS)
    ].copy()
    if target.empty:
        raise ValueError(f"No Amber prediction rows found for {target_pass}")
    if target.duplicated(KEYS + ["filter"]).any():
        raise ValueError("Prediction table has duplicate detector/source/filter rows")

    wide = _pivot_field(target, "reference_mag_obs", "ref")
    for column, prefix in (
        ("observed_delta_mag", "obs"),
        (MODEL_COLUMN, "model"),
        (FULL_HISTORY_COLUMN, "full"),
        ("mag_err", "err"),
    ):
        wide = wide.join(_pivot_field(target, column, prefix), how="outer")

    reference_errors, maximum_reference_match_difference = _reference_errors(
        amber_path, target
    )
    wide = wide.join(reference_errors, how="left")

    complete_fields = [
        "ref_F087",
        "ref_F158",
        "ref_F129",
        "obs_F129",
        "obs_F184",
        "model_F129",
        "model_F184",
        "full_F129",
        "full_F184",
        "err_F129",
        "err_F184",
        "referr_F087",
        "referr_F158",
        "referr_F129",
        "referr_F184",
    ]
    sample = wide.dropna(subset=complete_fields).copy()
    numeric = sample[complete_fields].to_numpy(dtype=float)
    sample = sample.loc[np.all(np.isfinite(numeric), axis=1)].copy()
    sample["reference_color_mag"] = sample["ref_F087"] - sample["ref_F158"]
    sample["observed_anomaly_mag"] = sample["obs_F129"] - sample["obs_F184"]
    sample["model_anomaly_mag"] = sample["model_F129"] - sample["model_F184"]
    sample["full_history_anomaly_mag"] = sample["full_F129"] - sample["full_F184"]
    sample["source_brightness_mag"] = sample["ref_F129"]
    sample["sigma_color_mag"] = np.hypot(
        sample["referr_F087"], sample["referr_F158"]
    )
    sample["sigma_anomaly_mag"] = np.sqrt(
        sample["err_F129"] ** 2
        + sample["referr_F129"] ** 2
        + sample["err_F184"] ** 2
        + sample["referr_F184"] ** 2
    )
    positive_errors = (sample["sigma_color_mag"] > 0.0) & (
        sample["sigma_anomaly_mag"] > 0.0
    )
    sample = sample.loc[positive_errors].copy()
    if sample.empty:
        raise ValueError("No complete, finite Amber/model rows remain for the plot")
    sample = sample.reset_index()
    return sample, maximum_reference_match_difference


def _weighted_bins(x, y, y_error, bin_count):
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bin_count + 1)))
    if edges.size < 3:
        return []
    result = []
    assignments = np.searchsorted(edges[1:-1], x, side="right")
    for bin_index in range(edges.size - 1):
        selected = assignments == bin_index
        if bin_index == edges.size - 2:
            selected |= x == edges[-1]
        if np.count_nonzero(selected) < 2:
            continue
        local_x = x[selected]
        local_y = y[selected]
        local_error = y_error[selected]
        weights = 1.0 / local_error**2
        mean = float(np.sum(weights * local_y) / np.sum(weights))
        effective_n = float(np.sum(weights) ** 2 / np.sum(weights**2))
        scatter2 = float(np.sum(weights * (local_y - mean) ** 2) / np.sum(weights))
        standard_error = np.sqrt(max(scatter2, 0.0) / effective_n)
        result.append(
            {
                "x": float(np.average(local_x, weights=weights)),
                "y": mean,
                "three_sigma": 3.0 * standard_error,
                "count": int(np.count_nonzero(selected)),
            }
        )
    return result


def _fit_odr(x, y, x_error, y_error):
    def line(parameters, color):
        return parameters[0] * color + parameters[1]

    data = odr.RealData(x, y, sx=x_error, sy=y_error)
    fit = odr.ODR(
        data,
        odr.Model(line),
        beta0=[0.0, float(np.median(y))],
        maxit=1000,
    ).run()
    if fit.info > 4 or np.any(~np.isfinite(fit.beta)):
        raise RuntimeError(f"ODR did not converge: {fit.stopreason}")
    return fit


def _plot(sample, output_dir: Path, stem: str, target_pass: str, bins: int):
    x = sample["reference_color_mag"].to_numpy(dtype=float)
    y = sample["observed_anomaly_mag"].to_numpy(dtype=float)
    model_y = sample["model_anomaly_mag"].to_numpy(dtype=float)
    full_history_y = sample["full_history_anomaly_mag"].to_numpy(dtype=float)
    x_error = sample["sigma_color_mag"].to_numpy(dtype=float)
    y_error = sample["sigma_anomaly_mag"].to_numpy(dtype=float)
    brightness = sample["source_brightness_mag"].to_numpy(dtype=float)

    fit = _fit_odr(x, y, x_error, y_error)
    weights = 1.0 / y_error**2

    def comparison_metrics(prediction):
        residual = y - prediction
        offset = float(np.sum(weights * residual) / np.sum(weights))
        chi2 = float(np.sum(weights * (residual - offset) ** 2))
        reduced = chi2 / max(1, len(sample) - 1)
        return offset, reduced

    h5_offset, h5_reduced_chi2 = comparison_metrics(model_y)
    full_offset, full_reduced_chi2 = comparison_metrics(full_history_y)
    model_difference_mmag = 1000.0 * (full_history_y - model_y)
    model_difference_summary = {
        "median_mmag": float(np.median(model_difference_mmag)),
        "mean_mmag": float(np.mean(model_difference_mmag)),
        "rms_mmag": float(np.sqrt(np.mean(model_difference_mmag**2))),
        "p95_absolute_mmag": float(np.quantile(np.abs(model_difference_mmag), 0.95)),
        "max_absolute_mmag": float(np.max(np.abs(model_difference_mmag))),
    }
    binned = _weighted_bins(x, y, y_error, bins)

    figure, axis = plt.subplots(figsize=(12.5, 7.6))
    points = axis.scatter(
        x,
        y,
        c=brightness,
        cmap="cool_r",
        s=9,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
        label="Amber observed data",
        zorder=2,
    )
    axis.scatter(
        x,
        model_y,
        color="#13851b",
        s=11,
        alpha=0.60,
        linewidths=0,
        rasterized=True,
        label="Reader H5 (rank-one history export)",
        zorder=3,
    )
    axis.scatter(
        x,
        full_history_y,
        color="#df7810",
        marker="x",
        s=13,
        alpha=0.58,
        linewidths=0.7,
        rasterized=True,
        label="Saved joint fit (full tile histories)",
        zorder=4,
    )
    line_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    slope, intercept = fit.beta
    slope_error, intercept_error = fit.sd_beta
    axis.plot(
        line_x,
        slope * line_x + intercept,
        color="black",
        linewidth=1.8,
        label=(
            f"ODR data fit (slope: {slope:+.4f} ± {slope_error:.4f}, "
            f"offset: {1000.0 * intercept:+.1f} ± {1000.0 * intercept_error:.1f} mmag)"
        ),
        zorder=5,
    )
    if binned:
        bin_x = np.asarray([item["x"] for item in binned])
        bin_y = np.asarray([item["y"] for item in binned])
        bin_error = np.asarray([item["three_sigma"] for item in binned])
        axis.errorbar(
            bin_x,
            bin_y,
            yerr=bin_error,
            fmt="o-",
            color="#f01818",
            markerfacecolor="#f01818",
            markeredgecolor="#f01818",
            linewidth=1.4,
            markersize=6,
            capsize=3,
            label="Inverse-variance bin mean ±3σ",
            zorder=6,
        )
    axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0, zorder=1)
    axis.set_xlim(-2.0, 2.0)
    axis.set_ylim(-0.15, 0.15)
    axis.set_xlabel(
        r"Reference color: $m_{F087}-m_{F158}$ (EP3 pass 1; mag)", fontsize=12
    )
    axis.set_ylabel(
        r"Differential anomaly: $\Delta(m_{F129}-m_{F184})$ "
        f"(EP3 pass 5 − pass 1; mag)",
        fontsize=12,
    )
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper right", fontsize=9, framealpha=0.94)
    colorbar = figure.colorbar(points, ax=axis, pad=0.025)
    colorbar.set_label(r"Source brightness: reference $m_{F129}$ (mag)", fontsize=11)

    figure.suptitle(
        f"Amber CAR201 differential-color anomaly: F129−F184 | {target_pass} vs EP3 pass 1\n"
        f"2026-11-17 joint fit: full histories vs reader-H5 export; "
        f"{len(sample):,} matched stars (in-sample)",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.015,
        0.93,
        f"Reader H5: offset {1000.0 * h5_offset:+.2f} mmag; "
        f"χ²ν {h5_reduced_chi2:.2f}\n"
        f"Full histories: offset {1000.0 * full_offset:+.2f} mmag; "
        f"χ²ν {full_reduced_chi2:.2f}\n"
        f"Full−H5 anomaly: RMS {model_difference_summary['rms_mmag']:.2f} mmag",
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    figure.tight_layout(rect=(0.015, 0.02, 0.99, 0.90))
    figure.text(
        0.5,
        0.008,
        "Saved fit retains independent per-tile histories on its native 5-nm grid; "
        "the reader H5 uses a rank-one history export on its 1-nm grid.",
        ha="center",
        fontsize=8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "target_pass": target_pass,
        "reference_pass": REFERENCE_PASS,
        "reference_color": "F087 - F158",
        "anomaly": "observed/model delta(F129) - delta(F184)",
        "source_brightness": "reference F129 magnitude",
        "model": "2026-11-17 joint CAR190 + Amber fit",
        "reader_h5_prediction_column": MODEL_COLUMN,
        "full_history_prediction_column": FULL_HISTORY_COLUMN,
        "full_history_source": (
            "saved joint_fit.npz predictions before unchanged-reader H5 packaging; "
            "independent per-tile histories"
        ),
        "sample_size": int(len(sample)),
        "odr_slope": float(slope),
        "odr_slope_standard_error": float(slope_error),
        "odr_intercept_mag": float(intercept),
        "odr_intercept_standard_error_mag": float(intercept_error),
        "model_comparisons": {
            "reader_h5_rank_one_history_export": {
                "weighted_data_minus_model_offset_mag": h5_offset,
                "weighted_data_minus_model_offset_mmag": 1000.0 * h5_offset,
                "reduced_chi2_after_fitted_offset": h5_reduced_chi2,
            },
            "saved_fit_full_tile_histories": {
                "weighted_data_minus_model_offset_mag": full_offset,
                "weighted_data_minus_model_offset_mmag": 1000.0 * full_offset,
                "reduced_chi2_after_fitted_offset": full_reduced_chi2,
            },
        },
        "full_history_minus_h5_differential_anomaly": model_difference_summary,
        "chi2_degrees_of_freedom": int(max(1, len(sample) - 1)),
        "bin_count_requested": int(bins),
        "bin_count_used": int(len(binned)),
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    return summary, png_path, pdf_path


def main(argv=None):
    args = _parse_args(argv)
    sample, reference_match_difference = _assemble_sample(
        args.predictions, args.amber_photometry, args.target_pass
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, png_path, pdf_path = _plot(
        sample, args.output_dir, args.stem, args.target_pass, args.bins
    )
    summary.update(
        {
            "predictions_input": str(args.predictions.resolve()),
            "amber_photometry_input": str(args.amber_photometry.resolve()),
            "model_validation_input": str(args.model_validation.resolve()),
            "reference_magnitude_max_match_difference_mag": reference_match_difference,
        }
    )
    validation = json.loads(args.model_validation.read_text())
    summary["model_h5"] = validation["output"]
    summary["model_h5_sha256"] = validation["output_sha256"]
    summary["reader_h5_limitations"] = validation.get("outstanding_limitations", [])
    sample_path = args.output_dir / f"{args.stem}_stars.csv"
    summary_path = args.output_dir / f"{args.stem}_summary.json"
    sample.to_csv(sample_path, index=False)
    summary["sample_csv"] = str(sample_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Matched stars: {len(sample):,}")
    for label, values in summary["model_comparisons"].items():
        print(
            f"{label}: offset {values['weighted_data_minus_model_offset_mmag']:+.2f} mmag; "
            f"reduced chi2 {values['reduced_chi2_after_fitted_offset']:.2f}"
        )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {sample_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
