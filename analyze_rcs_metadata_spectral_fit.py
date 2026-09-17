"""Fit a conservative spectral interaction to all-detector RCS metadata.

The metadata file contains one calibrated image median for each detector,
LED, and repeat. This script fits log median changes with a free source term
for every LED/repeat and a detector offset. The only common time-dependent
terms are response-weighted wavelength slope and curvature proxies from
``assess_rcs_spectral_identifiability.py``.

The fit tests spectral interaction only. Its coefficients are not ice
thickness or accumulation rates because no optical-constant model maps the
proxies to nanometers.
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from assess_rcs_spectral_identifiability import (
    MAPPING_URL,
    QE_URL,
    SED_URL,
    load_mapping,
    load_qe,
    load_sed,
    response_features,
)


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "all_wfi_rcs_flat_metadata.csv"


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def detector_number(detector):
    prefix = "WFI"
    if not detector.startswith(prefix):
        raise ValueError(f"Unexpected detector name: {detector}")
    return int(detector[len(prefix) :])


def load_metadata(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        led = int(row["led"])
        detector = row["detector"]
        time_value = row["exposure_start_time"]
        key = (led, time_value)
        group = groups.setdefault(key, {})
        if detector in group:
            raise ValueError(f"Duplicate metadata row for {key}, {detector}")
        median = float(row["image_median"])
        if not np.isfinite(median) or median <= 0.0:
            raise ValueError(f"Nonpositive image median for {key}, {detector}")
        group[detector] = median

    detectors = sorted(
        {detector for group in groups.values() for detector in group},
        key=detector_number,
    )
    if detectors != [f"WFI{index:02d}" for index in range(1, 19)]:
        raise ValueError(f"Expected WFI01 through WFI18, found {detectors}")

    global_start = min(parse_mast_time(time_value) for _, time_value in groups)
    observations = []
    for led in range(1, 7):
        led_times = sorted(
            [time_value for group_led, time_value in groups if group_led == led],
            key=parse_mast_time,
        )
        if len(led_times) != 13:
            raise ValueError(f"LED {led} has {len(led_times)} repeats, expected 13")
        first_group = groups[(led, led_times[0])]
        for repeat, time_value in enumerate(led_times):
            group = groups[(led, time_value)]
            if set(group) != set(detectors):
                raise ValueError(f"LED {led}, repeat {repeat} does not contain all detectors")
            days = (
                parse_mast_time(time_value) - global_start
            ).total_seconds() / 86400.0
            for detector in detectors:
                observations.append(
                    {
                        "led": led,
                        "repeat": repeat,
                        "group": (led - 1) * 13 + repeat,
                        "detector": detector_number(detector),
                        "days": days,
                        "log_change": np.log(group[detector] / first_group[detector]),
                    }
                )
    return observations, detectors, global_start


def nuisance_basis(observations):
    n_groups = max(row["group"] for row in observations) + 1
    n_detectors = 18
    basis = np.zeros((len(observations), n_groups + n_detectors), dtype=float)
    for index, row in enumerate(observations):
        basis[index, row["group"]] = 1.0
        basis[index, n_groups + row["detector"] - 1] = 1.0
    return basis


def residualize(values, nuisance):
    u, singular_values, _ = np.linalg.svd(nuisance, full_matrices=False)
    if singular_values.size == 0:
        return values
    tolerance = max(nuisance.shape) * singular_values[0] * np.finfo(float).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    projection_basis = u[:, :rank]
    return values - projection_basis @ (projection_basis.T @ values)


def fit_features(observations, feature_matrices):
    nuisance = nuisance_basis(observations)
    y = np.asarray([row["log_change"] for row in observations])
    days = np.asarray([row["days"] for row in observations])
    physical = np.asarray(
        [
            [
                days[index] * matrix[row["detector"] - 1, row["led"] - 1]
                for matrix in feature_matrices
            ]
            for index, row in enumerate(observations)
        ]
    )
    y_residual = residualize(y, nuisance)
    physical_residual = residualize(physical, nuisance)
    coefficients, _, physical_rank, _ = np.linalg.lstsq(
        physical_residual, y_residual, rcond=None
    )
    residual = y_residual - physical_residual @ coefficients
    nuisance_rank = np.linalg.matrix_rank(nuisance)
    degrees_of_freedom = len(y) - nuisance_rank - physical_rank
    variance = float(np.sum(residual**2) / degrees_of_freedom)
    covariance = variance * np.linalg.pinv(
        physical_residual.T @ physical_residual
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "t_statistics": coefficients / standard_errors,
        "null_residual_rms": float(np.sqrt(np.mean(y_residual**2))),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "observed_residualized": y_residual,
        "fitted_residualized": physical_residual @ coefficients,
        "fit_residualized": residual,
        "n_observations": len(y),
        "nuisance_rank": nuisance_rank,
        "physical_rank": int(physical_rank),
        "degrees_of_freedom": degrees_of_freedom,
    }


def write_fit_csv(path, fits):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sed_mode",
                "basis",
                "coefficient_per_day",
                "standard_error_per_day",
                "t_statistic",
                "null_residual_rms_log",
                "residual_rms_log",
                "rms_improvement_fraction",
                "n_observations",
                "nuisance_rank",
                "physical_rank",
                "degrees_of_freedom",
            ]
        )
        for sed_mode, fit in fits.items():
            for index, basis in enumerate(("linear_slope", "quadratic_curvature")):
                writer.writerow(
                    [
                        sed_mode,
                        basis,
                        f"{fit['coefficients'][index]:.12g}",
                        f"{fit['standard_errors'][index]:.12g}",
                        f"{fit['t_statistics'][index]:.12g}",
                        f"{fit['null_residual_rms']:.12g}",
                        f"{fit['residual_rms']:.12g}",
                        f"{1.0 - fit['residual_rms'] / fit['null_residual_rms']:.12g}",
                        fit["n_observations"],
                        fit["nuisance_rank"],
                        fit["physical_rank"],
                        fit["degrees_of_freedom"],
                    ]
                )


def write_report(path, metadata_path, fits, mapping, global_start):
    lines = [
        "# All-Detector RCS Spectral Interaction Fit",
        "",
        "This is a conservative metadata-level regression, not an ice",
        "thickness or accumulation-rate measurement.",
        "",
        f"- Metadata: `{metadata_path}`",
        f"- Sequence start: `{global_start.isoformat()}`",
        "- Detector mapping assumption: `WFI01` through `WFI18` correspond to SCU 1 through 18.",
        f"- Public SCU 1 maps to SCA `{mapping[1]}`; see the response summary for all mappings.",
        "- Model: log image median = free LED/repeat source term + detector offset + time x spectral interaction.",
        "- Source terms are completely free for each LED/repeat, so source-output drift is not interpreted as ice.",
        "- Reported standard errors are nominal OLS residual errors; they do not include correlated calibration systematics.",
        "",
        "| SED mode | Basis | Coefficient per day | Standard error | t statistic | Null RMS | Fit RMS | RMS improvement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sed_mode, fit in fits.items():
        for index, basis in enumerate(("linear slope", "quadratic curvature")):
            lines.append(
                f"| {sed_mode} | {basis} | {fit['coefficients'][index]:.6g} | "
                f"{fit['standard_errors'][index]:.6g} | {fit['t_statistics'][index]:.6g} | "
                f"{fit['null_residual_rms']:.6g} | {fit['residual_rms']:.6g} | "
                f"{1.0 - fit['residual_rms'] / fit['null_residual_rms']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The coefficient units are log-response change per day per unit of the",
            "arbitrary spectral basis. They cannot be converted to nm/day without",
            "an ice optical model. A useful physical rate fit should replace these",
            "proxies with the derivative of a specified ice transmission model and",
            "should include independent source and temperature telemetry. The fit",
            "improvement is sensitive to the noisy SED-wing treatment, so it is not",
            "evidence for an ice rate by itself.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, detectors, global_start = load_metadata(args.metadata)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    mapping = load_mapping(MAPPING_URL)
    features_by_mode = {
        mode: response_features(
            spectra,
            qe_wavelengths,
            qe_values,
            weighting="photon",
            sed_mode=mode,
            core_threshold_percent=5.0,
        )
        for mode in ("signed", "clipped", "core")
    }
    fits = {
        mode: fit_features(
            observations,
            [features["slope"], features["curvature"]],
        )
        for mode, features in features_by_mode.items()
    }
    fit_path = args.output_dir / "rcs_metadata_spectral_fit.csv"
    report_path = args.output_dir / "rcs_metadata_spectral_fit.md"
    write_fit_csv(fit_path, fits)
    write_report(report_path, args.metadata, fits, mapping, global_start)
    print(f"Metadata observations: {len(observations)} ({len(detectors)} detectors)")
    print(f"Wrote {fit_path}")
    print(f"Wrote {report_path}")
    for mode, fit in fits.items():
        print(
            f"{mode}: slope={fit['coefficients'][0]:.6g} +/- "
            f"{fit['standard_errors'][0]:.6g} per day; "
            f"curvature={fit['coefficients'][1]:.6g} +/- "
            f"{fit['standard_errors'][1]:.6g} per day"
        )


if __name__ == "__main__":
    main()
