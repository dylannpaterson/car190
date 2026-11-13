"""Screen RCS spectral leverage for a common ice signal.

The RCS LED spectra are relative spectral energy outputs, while the public
WFI QE table supplies one wavelength response curve for each of 18 SCUs. This
script combines them and reports response-weighted effective wavelengths plus
the part of simple spectral ice signatures that cannot be absorbed by free
LED and detector offsets.

This is an identifiability screen, not a thickness or accumulation-rate fit.
The spectral slope and curvature bases are deliberately model-independent
proxies until an ice optical-constant model is selected.
"""

import argparse
import csv
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parent
SED_URL = (
    "https://raw.githubusercontent.com/RomanSpaceTelescope/"
    "roman-technical-information/ff3bcfea58b5693169d7c30fa71dc90fc8346ea3/"
    "roman_technical_information/data/WideFieldInstrument/"
    "RelativeCalibrationSystem/WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
)
QE_URL = (
    "https://raw.githubusercontent.com/RomanSpaceTelescope/"
    "roman-technical-information/11fc10cb987ff1a8745aa9e2952053a1e05a9ed9/"
    "roman_technical_information/data/"
    "WideFieldInstrument/FPSPerformance/WFI_Quantum_efficiency.ecsv"
)
MAPPING_URL = (
    "https://raw.githubusercontent.com/RomanSpaceTelescope/"
    "roman-technical-information/11fc10cb987ff1a8745aa9e2952053a1e05a9ed9/"
    "roman_technical_information/data/"
    "WideFieldInstrument/FPSPerformance/WFI_SCU_to_SCA.ecsv"
)

BAND_FILTERS = {
    1: "F062",
    2: "F087",
    3: "F106",
    4: "F129",
    5: "F158",
    6: "F184",
}
EXPECTED_SED_SAMPLES = {1: 3648, 2: 3648, 3: 128, 4: 128, 5: 153, 6: 153}


def read_source(source):
    if source.startswith(("http://", "https://")):
        request = Request(source, headers={"User-Agent": "WFI-RCS-analysis"})
        with urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text()


def read_csv_rows(text):
    lines = (
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return list(csv.DictReader(lines))


def load_sed(source):
    rows_by_band = {band: [] for band in BAND_FILTERS}
    for row in read_csv_rows(read_source(source)):
        band = int(row["RCSLEDBand"])
        if band not in rows_by_band:
            raise ValueError(f"Unexpected RCS LED band: {band}")
        if row["filter"] != BAND_FILTERS[band]:
            raise ValueError(
                f"Band {band} is labeled {row['filter']}, "
                f"expected {BAND_FILTERS[band]}"
            )
        rows_by_band[band].append(
            (
                float(row["RCSLEDSpectrumWavelength"]),
                float(row["RCSLEDSpectrumRelativeOutput"]),
            )
        )

    spectra = {}
    for band, rows in rows_by_band.items():
        if not rows:
            raise ValueError(f"No SED rows found for band {band}")
        if len(rows) != EXPECTED_SED_SAMPLES[band]:
            raise ValueError(
                f"Band {band} has {len(rows)} SED samples, "
                f"expected {EXPECTED_SED_SAMPLES[band]}"
            )
        rows.sort()
        wavelengths, output = np.asarray(rows, dtype=float).T
        spectra[band] = {
            "wavelength": wavelengths,
            "raw_output": output,
            "negative_samples": int(np.count_nonzero(output < 0.0)),
        }
    return spectra


def load_qe(source):
    rows = read_csv_rows(read_source(source))
    if not rows:
        raise ValueError("No QE rows found")
    fieldnames = [f"SCU {index}" for index in range(1, 19)]
    missing = [name for name in ["wavelength", *fieldnames] if name not in rows[0]]
    if missing:
        raise ValueError(f"QE table is missing columns: {', '.join(missing)}")

    wavelengths = np.asarray([float(row["wavelength"]) for row in rows])
    values = np.asarray(
        [[float(row[name]) for name in fieldnames] for row in rows], dtype=float
    )
    order = np.argsort(wavelengths)
    return wavelengths[order], values[order], fieldnames


def load_mapping(source):
    mapping = {}
    for row in read_csv_rows(read_source(source)):
        scu = int(row["SCU#"])
        mapping[scu] = str(int(float(row["SCA"])))
    expected = set(range(1, 19))
    if set(mapping) != expected:
        raise ValueError("SCU-to-SCA mapping does not contain exactly SCUs 1 through 18")
    return mapping


def integrate(values, wavelengths):
    return float(np.trapezoid(values, wavelengths))


def response_features(
    spectra, qe_wavelengths, qe_values, weighting, sed_mode, core_threshold_percent
):
    effective_wavelength = np.zeros((18, 6))
    integrated_response = np.zeros((18, 6))
    qe_at_effective_wavelength = np.zeros((18, 6))
    slope = np.zeros((18, 6))
    curvature = np.zeros((18, 6))

    # The wavelength multiplier converts relative spectral power per wavelength
    # to relative photon rate. Use --weighting energy if the source table is
    # interpreted as a photon spectrum instead.
    center = 1200.0
    scale = 1000.0
    for band in range(1, 7):
        wavelengths = spectra[band]["wavelength"]
        raw_output = spectra[band]["raw_output"]
        if sed_mode == "signed":
            output = raw_output
        elif sed_mode == "clipped":
            # Negative values are documented measurement noise in spectral wings.
            output = np.maximum(raw_output, 0.0)
        elif sed_mode == "core":
            threshold = np.max(raw_output) * core_threshold_percent / 100.0
            output = np.where(raw_output >= threshold, raw_output, 0.0)
        else:
            raise ValueError(f"Unknown SED mode: {sed_mode}")
        photon_weight = wavelengths if weighting == "photon" else np.ones_like(wavelengths)
        for scu in range(18):
            qe = np.interp(
                wavelengths,
                qe_wavelengths,
                qe_values[:, scu],
                left=0.0,
                right=0.0,
            )
            weighted_output = output * qe * photon_weight
            response = integrate(weighted_output, wavelengths)
            if response <= 0.0:
                raise ValueError(f"No positive response for SCU {scu + 1}, band {band}")
            effective_wavelength[scu, band - 1] = integrate(
                weighted_output * wavelengths, wavelengths
            ) / response
            integrated_response[scu, band - 1] = response
            qe_at_effective_wavelength[scu, band - 1] = np.interp(
                effective_wavelength[scu, band - 1], qe_wavelengths, qe_values[:, scu]
            )
            normalized_wavelength = (wavelengths - center) / scale
            slope[scu, band - 1] = integrate(
                weighted_output * normalized_wavelength, wavelengths
            ) / response
            curvature[scu, band - 1] = integrate(
                weighted_output * normalized_wavelength**2, wavelengths
            ) / response

    return {
        "effective_wavelength": effective_wavelength,
        "integrated_response": integrated_response,
        "qe_at_effective_wavelength": qe_at_effective_wavelength,
        "slope": slope,
        "curvature": curvature,
    }


def interaction_residual(matrix):
    """Remove arbitrary log-response detector and LED offsets from a D x L matrix."""

    row_mean = np.mean(matrix, axis=1, keepdims=True)
    column_mean = np.mean(matrix, axis=0, keepdims=True)
    return matrix - row_mean - column_mean + np.mean(matrix)


def write_response_csv(path, features_by_mode, mapping):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scu",
                "sca",
                "sed_mode",
                "band",
                "filter",
                "integrated_response_arbitrary",
                "effective_wavelength_nm",
                "qe_at_effective_wavelength",
            ]
        )
        for sed_mode, features in features_by_mode.items():
            for scu in range(18):
                for band in range(1, 7):
                    writer.writerow(
                        [
                            scu + 1,
                            mapping[scu + 1],
                            sed_mode,
                            band,
                            BAND_FILTERS[band],
                            f"{features['integrated_response'][scu, band - 1]:.12g}",
                            f"{features['effective_wavelength'][scu, band - 1]:.9f}",
                            f"{features['qe_at_effective_wavelength'][scu, band - 1]:.9g}",
                        ]
                    )


def write_identifiability_csv(path, features_by_mode):
    basis_names = ["grey", "linear_slope", "quadratic_curvature"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "basis",
                "sed_mode",
                "raw_frobenius_norm",
                "after_detector_and_led_offsets_norm",
                "residual_fraction",
                "residual_rms_per_scu_band",
            ]
        )
        for sed_mode, features in features_by_mode.items():
            basis_matrices = [
                np.ones((18, 6)),
                features["slope"],
                features["curvature"],
            ]
            for name, matrix in zip(basis_names, basis_matrices):
                residual = interaction_residual(matrix)
                raw_norm = float(np.linalg.norm(matrix))
                residual_norm = float(np.linalg.norm(residual))
                writer.writerow(
                    [
                        name,
                        sed_mode,
                        f"{raw_norm:.12g}",
                        f"{residual_norm:.12g}",
                        f"{residual_norm / raw_norm:.12g}",
                        f"{residual_norm / np.sqrt(matrix.size):.12g}",
                    ]
                )


def write_report(
    path,
    sed_source,
    qe_source,
    mapping_source,
    mapping,
    spectra,
    qe_wavelengths,
    features_by_mode,
    weighting,
    core_threshold_percent,
):
    lines = [
        "# RCS Spectral Identifiability Screen",
        "",
        "This is a response-weighting and identifiability screen, not an ice",
        "thickness or accumulation-rate measurement.",
        "",
        f"- SED source: `{sed_source}`",
        f"- QE source: `{qe_source}`",
        f"- SCU-to-SCA mapping source: `{mapping_source}`",
        f"- SCU-to-SCA rows: {len(mapping)}",
        f"- Spectral weighting: `{weighting}`",
        f"- QE wavelength range: {qe_wavelengths[0]:.0f}-{qe_wavelengths[-1]:.0f} nm",
        "- QE metadata: median QE measurements for individual SCAs; temperature compatibility with the flats must be checked.",
        "- SED modes: `signed` preserves published values, `clipped` sets negative values to zero,",
        f"  and `core` keeps values at or above {core_threshold_percent:g}% of each peak.",
        "",
        "## Response-Weighted Means",
        "",
        "| Mode | Band | Filter | SED samples | Mean range (nm) | Mean spread (nm) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for sed_mode, features in features_by_mode.items():
        effective = features["effective_wavelength"]
        for band in range(1, 7):
            values = effective[:, band - 1]
            lines.append(
                f"| {sed_mode} | {band} | {BAND_FILTERS[band]} | "
                f"{len(spectra[band]['wavelength'])} | "
                f"{values.min():.3f}-{values.max():.3f} | {values.std(ddof=1):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Nuisance-Projected Spectral Leverage",
            "",
            "The residual removes an arbitrary additive log-response offset for each",
            "SCU and each LED band. A grey transmission change is therefore exactly",
            "degenerate with source/detector offsets; wavelength-dependent leverage",
            "survives only through the SCU-by-band interaction pattern.",
            "",
            "| Mode | Basis | Residual norm | Residual fraction of basis norm |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for sed_mode, features in features_by_mode.items():
        for name, matrix in zip(
            ["grey", "linear slope", "quadratic curvature"],
            [np.ones((18, 6)), features["slope"], features["curvature"]],
        ):
            residual = interaction_residual(matrix)
            lines.append(
                f"| {sed_mode} | {name} | {np.linalg.norm(residual):.6g} | "
                f"{np.linalg.norm(residual) / np.linalg.norm(matrix):.6g} |"
            )

    lines.extend(["", "## Projected Basis Conditioning", ""])
    for sed_mode, features in features_by_mode.items():
        projected = np.column_stack(
            [
                interaction_residual(features["slope"]).ravel(),
                interaction_residual(features["curvature"]).ravel(),
            ]
        )
        column_norms = np.linalg.norm(projected, axis=0)
        scaled = projected / column_norms
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        condition = singular_values[0] / singular_values[-1]
        lines.append(
            f"- `{sed_mode}`: singular values `{singular_values[0]:.6g}`, "
            f"`{singular_values[1]:.6g}`; condition number `{condition:.6g}`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The exact SED and public QE curves are sufficient to calculate the",
            "forward response and test a specified ice transmission law. The",
            "published table contains noisy wings, so the signed/clipped/core",
            "comparison is a required preprocessing sensitivity check, especially",
            "for the broad scan windows. The relative SED has no absolute source",
            "calibration, and a physical ice model still needs optical constants,",
            "film geometry, initial thickness, and source monitoring. These metrics",
            "quantify spectral leverage before those assumptions are introduced.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sed", default=SED_URL, help="SED ECSV path or URL")
    parser.add_argument("--qe", default=QE_URL, help="QE ECSV path or URL")
    parser.add_argument(
        "--mapping", default=MAPPING_URL, help="SCU-to-SCA mapping ECSV path or URL"
    )
    parser.add_argument(
        "--weighting",
        choices=("photon", "energy"),
        default="photon",
        help="Interpret relative SED output as energy or photon weighting",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT, help="Directory for generated outputs"
    )
    parser.add_argument(
        "--core-threshold-percent",
        type=float,
        default=5.0,
        help="Core SED threshold as a percentage of each band's peak",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectra = load_sed(args.sed)
    qe_wavelengths, qe_values, _ = load_qe(args.qe)
    mapping = load_mapping(args.mapping)
    features_by_mode = {
        mode: response_features(
            spectra,
            qe_wavelengths,
            qe_values,
            weighting=args.weighting,
            sed_mode=mode,
            core_threshold_percent=args.core_threshold_percent,
        )
        for mode in ("signed", "clipped", "core")
    }

    response_path = args.output_dir / "rcs_spectral_response_summary.csv"
    identifiability_path = args.output_dir / "rcs_spectral_identifiability.csv"
    report_path = args.output_dir / "rcs_spectral_identifiability.md"
    write_response_csv(response_path, features_by_mode, mapping)
    write_identifiability_csv(identifiability_path, features_by_mode)
    write_report(
        report_path,
        args.sed,
        args.qe,
        args.mapping,
        mapping,
        spectra,
        qe_wavelengths,
        features_by_mode,
        args.weighting,
        args.core_threshold_percent,
    )

    print(f"SED samples: {sum(len(spectra[band]['wavelength']) for band in spectra)}")
    print(f"QE samples: {len(qe_wavelengths)} x 18 SCUs")
    print(f"Wrote {response_path}")
    print(f"Wrote {identifiability_path}")
    print(f"Wrote {report_path}")
    for sed_mode, features in features_by_mode.items():
        for band in range(1, 7):
            values = features["effective_wavelength"][:, band - 1]
            print(
                f"{sed_mode} band {band} ({BAND_FILTERS[band]}): "
                f"response-weighted mean {values.min():.3f}-{values.max():.3f} nm"
            )


if __name__ == "__main__":
    main()
