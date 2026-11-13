"""Regenerate clean-only complex-reflection figures from saved CAR201 fits."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import compare_wfi_tile136_complex_reflectance as complex_model
import fit_wfi_tmm_production as production
import mock_fit_wfi_star_reflectance_modulation as mock_model
from fit_car201_zane_star_reflectance_modulation import (
    detector_default_output_dir,
    detector_optical_cache,
    plot_clean_complex_reflection,
    plot_clean_reflectance_modulation_ratio,
)


def regenerate_detector_plot(detector, pass_set="ep3", allow_partial_source_coverage=False):
    output_dir = detector_default_output_dir(
        detector,
        pass_set,
        allow_partial_source_coverage=allow_partial_source_coverage,
    )
    summary_path = output_dir / "car201_actual_star_modulation_fit_summary.json"
    parameters_path = output_dir / "car201_actual_star_modulation_parameters.csv"
    with summary_path.open() as handle:
        summary = json.load(handle)

    stride_nm = float(summary["modulation"].get("wavelength_stride_nm", 5.0))
    original_wavelengths_nm, effective_area = mock_model.load_effective_area(
        mock_model.DEFAULT_EFFECTIVE_AREA
    )
    wavelength_grid_nm = np.arange(
        400.0, 2321.0 + 0.5 * stride_nm, stride_nm
    )
    area_on_grid = np.asarray(
        [
            np.interp(wavelength_grid_nm, original_wavelengths_nm, band)
            for band in effective_area
        ]
    )
    supported = np.any(area_on_grid > 0.0, axis=0)
    wavelengths_nm = wavelength_grid_nm[supported]

    optical_cache = detector_optical_cache(detector)
    source_fit = mock_model.source_fit_from_cache(optical_cache, detector, 136)
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths_nm)
    spectral_basis = production.spectral_modulation_basis(wavelengths_nm)
    state = complex_model.load_state(
        source_fit, wavelengths_nm, base_n, base_k, spectral_basis
    )
    clean_first, clean_second, incident_y, _, _ = mock_model.clean_stack_boundary(
        state, wavelengths_nm
    )
    clean_reflection = (incident_y * clean_first - clean_second) / (
        incident_y * clean_first + clean_second
    )

    basis_name = summary["modulation"]["basis"]
    basis, basis_names, *_ = mock_model.modulation_basis(
        wavelengths_nm, grouping=basis_name
    )
    parameters = pd.read_csv(parameters_path).set_index("mode").loc[basis_names]
    output_path = output_dir / "car201_actual_clean_complex_reflection_fit.png"
    plot_clean_complex_reflection(
        output_path,
        detector,
        wavelengths_nm,
        clean_reflection,
        basis,
        parameters["fit_log_amplitude"].to_numpy(float),
        parameters["fit_phase_rad"].to_numpy(float),
        float(summary["fit"]["chi2"]),
        int(summary["fit"]["degrees_of_freedom_approx"]),
        float(summary["modulation"]["maxent_regularization"]["weight"]),
        float(summary["modulation"]["phase_smoothness_regularization"]["weight"]),
    )
    ratio_path = output_dir / "car201_actual_reflectance_modulation_ratio.png"
    plot_clean_reflectance_modulation_ratio(
        ratio_path,
        detector,
        wavelengths_nm,
        basis,
        parameters["fit_log_amplitude"].to_numpy(float),
        parameters["fit_phase_rad"].to_numpy(float),
        float(summary["modulation"]["maxent_regularization"]["weight"]),
        float(summary["modulation"]["phase_smoothness_regularization"]["weight"]),
    )
    summary.setdefault("products", {})[
        "clean_reflectance_modulation_ratio_plot"
    ] = str(ratio_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output_path, ratio_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        action="append",
        required=True,
        help="Detector to plot; repeat explicitly for multiple detectors.",
    )
    parser.add_argument(
        "--pass-set", choices=("ep3", "ep4", "ep3_ep4"), default="ep3"
    )
    parser.add_argument("--allow-partial-source-coverage", action="store_true")
    args = parser.parse_args(argv)
    for detector in args.detector:
        detector = detector.upper()
        clean_path, ratio_path = regenerate_detector_plot(
            detector, args.pass_set, args.allow_partial_source_coverage
        )
        print(f"Wrote clean-only complex-reflection figure: {clean_path}")
        print(f"Wrote reflectance-modulation ratio figure: {ratio_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
