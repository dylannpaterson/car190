"""Grid-search an absolute-power TMM model for WFI ratio data.

The six optical parameters are the thickness scale and real refractive index
for each of three lossless AR layers.  HgCdTe is treated as a semi-infinite
terminal substrate, and the ice thickness follows a constant-rate trajectory.
This is deliberately a diagnostic model, separate from the production
NumPyro fit.
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_rcs_soc_ice_model import sed_output
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_wfi_tmm_numpyro import (
    AR_LOWER,
    AR_UPPER,
    HGCDTE_REFERENCE_COMPOSITION,
    HGCDTE_REFERENCE_TEMPERATURE_K,
    apply_fixed_corrections,
    hgcdte_reference_optical_constants,
    load_tile_statistics,
)


DEFAULT_STATISTICS = Path(
    "wfi_tile_log_statistics_512/wfi01_tile_log_statistics_512.csv"
)
DEFAULT_FIXED_CORRECTIONS = Path("wfi01_fixed_led3_gain_acm_corrections.csv")


def trapezoid_sample_weights(wavelengths):
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("At least two ordered wavelength samples are required")
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Wavelength samples must be strictly increasing")
    weights = np.empty_like(wavelengths)
    weights[0] = 0.5 * (wavelengths[1] - wavelengths[0])
    weights[-1] = 0.5 * (wavelengths[-1] - wavelengths[-2])
    weights[1:-1] = 0.5 * (wavelengths[2:] - wavelengths[:-2])
    return weights


def build_source_integrations(
    spectra,
    sed_mode="core",
    sed_weighting="energy",
    qe_wavelengths=None,
    qe_values=None,
    qe_column=0,
):
    """Return native-wavelength source integration weights with optional QE."""
    if sed_weighting not in ("energy", "photon"):
        raise ValueError("SED weighting must be 'energy' or 'photon'")
    integrations = {}
    for led in range(1, 7):
        spectrum = spectra[led]
        wavelengths = np.asarray(spectrum["wavelength"], dtype=float)
        output = sed_output(spectrum, sed_mode)
        spectral_factor = (
            wavelengths if sed_weighting == "photon" else np.ones_like(wavelengths)
        )
        weights = output * spectral_factor * trapezoid_sample_weights(wavelengths)
        if qe_wavelengths is not None or qe_values is not None:
            if qe_wavelengths is None or qe_values is None:
                raise ValueError("QE wavelengths and values must be supplied together")
            qe = np.interp(
                wavelengths,
                np.asarray(qe_wavelengths, dtype=float),
                np.asarray(qe_values, dtype=float)[:, int(qe_column)],
                left=0.0,
                right=0.0,
            )
            weights *= qe
        if not np.isfinite(weights).all() or np.sum(weights) <= 0.0:
            raise ValueError(f"LED {led} has no positive finite source weight")
        integrations[led] = (wavelengths, weights)
    return integrations


def _layer_elements(wavelengths, index, thickness_nm):
    phase = 2.0 * np.pi * index * thickness_nm / wavelengths
    cosine = np.cos(phase)
    sine = -1j * np.sin(phase)
    return cosine, sine / index, sine * index, cosine


def _apply_layer_elements(first, second, elements):
    m00, m01, m10, m11 = elements
    return (
        m00 * first + m01 * second,
        m10 * first + m11 * second,
    )


def _ar_parameters_to_stack(optical_parameters):
    optical_parameters = np.asarray(optical_parameters, dtype=float).reshape(-1)
    if optical_parameters.size != 6:
        raise ValueError("The three-layer lossless AR stack requires six parameters")
    scales = optical_parameters[0::2]
    indices = optical_parameters[1::2]
    if np.any(scales <= 0.0) or np.any(indices <= 0.0):
        raise ValueError("AR thickness scales and indices must be positive")
    return 100.0 * scales, indices.astype(complex)


def _substrate_index(
    wavelengths,
    use_hgcdte_optical_constants,
    hgcdte_n_scale=1.0,
    hgcdte_k_scale=1.0,
):
    if (
        not np.isfinite(hgcdte_n_scale)
        or not np.isfinite(hgcdte_k_scale)
        or hgcdte_n_scale <= 0.0
        or hgcdte_k_scale <= 0.0
    ):
        raise ValueError("HgCdTe n/k scales must be positive and finite")
    if use_hgcdte_optical_constants:
        real, imaginary = hgcdte_reference_optical_constants(wavelengths)
        return hgcdte_n_scale * real + 1j * hgcdte_k_scale * imaginary
    return np.full_like(
        wavelengths,
        hgcdte_n_scale * 3.6 + 1j * hgcdte_k_scale * 0.7,
        dtype=complex,
    )


def tmm_power_curves(
    wavelengths,
    optical_parameters,
    ice_thickness_nm,
    ice_index=1.25 + 0.0j,
    use_hgcdte_optical_constants=True,
    hgcdte_n_scale=1.0,
    hgcdte_k_scale=1.0,
):
    """Return clean and ice-coated power entering a terminal substrate."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    ice_thickness_nm = np.asarray(ice_thickness_nm, dtype=float)
    if ice_thickness_nm.ndim > 1:
        raise ValueError("Ice thickness must be a scalar or one-dimensional array")
    vectorized_thickness = ice_thickness_nm.ndim == 1
    thicknesses, ar_indices = _ar_parameters_to_stack(optical_parameters)
    substrate = _substrate_index(
        wavelengths,
        use_hgcdte_optical_constants,
        hgcdte_n_scale,
        hgcdte_k_scale,
    )

    clean_first = np.ones_like(wavelengths, dtype=complex)
    clean_second = substrate.copy()
    for index, thickness in zip(ar_indices[::-1], thicknesses[::-1]):
        clean_first, clean_second = _apply_layer_elements(
            clean_first,
            clean_second,
            _layer_elements(wavelengths, index, thickness),
        )
    clean = np.real(substrate) * np.abs(2.0 / (clean_first + clean_second)) ** 2

    if vectorized_thickness:
        ice_wavelengths = wavelengths[:, None]
        ice_thickness = ice_thickness_nm[None, :]
        boundary_first = clean_first[:, None]
        boundary_second = clean_second[:, None]
        substrate_power = np.real(substrate)[:, None]
    else:
        ice_wavelengths = wavelengths
        ice_thickness = float(ice_thickness_nm)
        boundary_first = clean_first
        boundary_second = clean_second
        substrate_power = np.real(substrate)
    ice_m00, ice_m01, ice_m10, ice_m11 = _layer_elements(
        ice_wavelengths, complex(ice_index), ice_thickness
    )
    # The characteristic-matrix boundary vector for air followed by ice is
    # (m00 + m10, m01 + m11), matching the production normal-incidence path.
    ice_boundary_first = ice_m00 + ice_m10
    ice_boundary_second = ice_m01 + ice_m11
    coated_denominator = (
        boundary_first * ice_boundary_first + boundary_second * ice_boundary_second
    )
    coated = substrate_power * np.abs(2.0 / coated_denominator) ** 2
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(coated)):
        raise ValueError("TMM power calculation produced non-finite values")
    return clean, coated


def integrated_absolute_response(
    source_integrations,
    optical_parameters,
    thicknesses_nm,
    ice_index=1.25 + 0.0j,
    use_hgcdte_optical_constants=True,
    hgcdte_n_scale=1.0,
    hgcdte_k_scale=1.0,
):
    """Integrate absolute substrate power for every LED and thickness."""
    thicknesses_nm = np.asarray(thicknesses_nm, dtype=float).reshape(-1)
    response = np.empty((6, len(thicknesses_nm)), dtype=float)
    for led in range(1, 7):
        wavelengths, weights = source_integrations[led]
        clean, coated = tmm_power_curves(
            wavelengths,
            optical_parameters,
            0.0,
            ice_index=ice_index,
            use_hgcdte_optical_constants=use_hgcdte_optical_constants,
            hgcdte_n_scale=hgcdte_n_scale,
            hgcdte_k_scale=hgcdte_k_scale,
        )
        _, coated = tmm_power_curves(
            wavelengths,
            optical_parameters,
            thicknesses_nm,
            ice_index=ice_index,
            use_hgcdte_optical_constants=use_hgcdte_optical_constants,
            hgcdte_n_scale=hgcdte_n_scale,
            hgcdte_k_scale=hgcdte_k_scale,
        )
        coated_values = np.sum(weights[:, None] * coated, axis=0)
        clean_integrated = np.sum(weights * clean)
        response[led - 1] = coated_values / clean_integrated
    return response


def predict_constant_rate_log_ratios(
    data,
    source_integrations,
    optical_parameters,
    rate_nm_day,
    initial_thickness_nm=0.0,
    ice_index=1.25 + 0.0j,
    use_hgcdte_optical_constants=True,
    hgcdte_n_scale=1.0,
    hgcdte_k_scale=1.0,
):
    """Predict per-LED baseline-relative log ratios for constant accumulation."""
    if not np.isfinite(rate_nm_day) or rate_nm_day <= 0.0:
        raise ValueError("Accumulation rate must be positive and finite")
    days = np.asarray(data["days"], dtype=float)
    led = np.asarray(data["led"], dtype=int)
    thickness = initial_thickness_nm + rate_nm_day * (days - np.min(days))
    prediction = np.empty(len(days), dtype=float)
    observed = None
    if "counts" in data and "sums" in data:
        observed = np.asarray(data["sums"], dtype=float) / np.asarray(
            data["counts"], dtype=float
        )
    for led_value in range(1, 7):
        groups = np.flatnonzero(led == led_value)
        if not groups.size:
            raise ValueError(f"Missing LED {led_value} observations")
        baseline_group = int(np.asarray(data["first_by_led"])[led_value - 1])
        thickness_values = np.unique(
            np.concatenate((thickness[groups], [thickness[baseline_group]]))
        )
        response = integrated_absolute_response(
            source_integrations,
            optical_parameters,
            thickness_values,
            ice_index=ice_index,
            use_hgcdte_optical_constants=use_hgcdte_optical_constants,
            hgcdte_n_scale=hgcdte_n_scale,
            hgcdte_k_scale=hgcdte_k_scale,
        )[led_value - 1]
        response_by_thickness = dict(zip(thickness_values, response))
        baseline_response = response_by_thickness[thickness[baseline_group]]
        baseline_offset = (
            float(np.average(observed[:, baseline_group], weights=data["counts"][:, baseline_group]))
            if observed is not None
            else 0.0
        )
        prediction[groups] = np.log(
            np.asarray([response_by_thickness[value] for value in thickness[groups]])
            / baseline_response
        ) + baseline_offset
    return prediction


def score_prediction(data, prediction):
    prediction = np.asarray(prediction, dtype=float)
    counts = np.asarray(data["counts"], dtype=float)
    sums = np.asarray(data["sums"], dtype=float)
    sum_squared = np.asarray(data["sum_squared"], dtype=float)
    if prediction.shape != (counts.shape[1],):
        raise ValueError("Prediction groups do not match tile statistics")
    residual_sse = sum_squared - 2.0 * sums * prediction[None, :]
    residual_sse += counts * prediction[None, :] ** 2
    observed = sums / counts
    group_residual = observed - prediction[None, :]
    return {
        "sse": float(max(np.sum(residual_sse), 0.0)),
        "pixel_rms": float(np.sqrt(max(np.sum(residual_sse), 0.0) / np.sum(counts))),
        "group_rms": float(np.sqrt(np.mean(group_residual * group_residual))),
        "prediction": prediction,
    }


def optical_parameter_grid(grid_points):
    if grid_points < 2:
        raise ValueError("Grid points must be at least two")
    values = []
    for layer in range(3):
        scale_index = 3 * layer
        index_index = scale_index + 1
        values.append(
            np.logspace(
                np.log10(AR_LOWER[0]),
                np.log10(AR_UPPER[0]),
                grid_points,
            )
        )
        values.append(
            np.linspace(AR_LOWER[1], AR_UPPER[1], grid_points)
        )
        del scale_index, index_index
    return itertools.product(*values)


def parse_float_list(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite comma-separated values")
    return values


def run_grid_search(
    data,
    source_integrations,
    grid_points,
    rates,
    initial_thickness_nm,
    ice_index,
    use_hgcdte_optical_constants,
    top_count,
):
    results = []
    candidate_count = grid_points**6 * len(rates)
    for candidate_index, optical_parameters in enumerate(
        optical_parameter_grid(grid_points), start=1
    ):
        optical_parameters = np.asarray(optical_parameters, dtype=float)
        for rate in rates:
            prediction = predict_constant_rate_log_ratios(
                data,
                source_integrations,
                optical_parameters,
                rate,
                initial_thickness_nm=initial_thickness_nm,
                ice_index=ice_index,
                use_hgcdte_optical_constants=use_hgcdte_optical_constants,
            )
            score = score_prediction(data, prediction)
            results.append(
                {
                    "optical_parameters": optical_parameters.copy(),
                    "rate_nm_day": float(rate),
                    **score,
                }
            )
        if candidate_index == 1 or candidate_index % 25 == 0:
            print(
                f"evaluated optical grid {candidate_index}/{grid_points**6} "
                f"({candidate_index * len(rates)}/{candidate_count} total)"
            )
    results.sort(key=lambda row: row["sse"])
    return results[:top_count], results[0]


def write_outputs(output_dir, data, best_results, best_result, settings):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "grid_results.csv"
    fields = [
        "rank",
        "sse",
        "pixel_rms",
        "group_rms",
        "rate_nm_day",
        "layer1_thickness_scale",
        "layer1_n",
        "layer2_thickness_scale",
        "layer2_n",
        "layer3_thickness_scale",
        "layer3_n",
    ]
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(best_results, start=1):
            parameters = result["optical_parameters"]
            writer.writerow(
                {
                    "rank": rank,
                    "sse": result["sse"],
                    "pixel_rms": result["pixel_rms"],
                    "group_rms": result["group_rms"],
                    "rate_nm_day": result["rate_nm_day"],
                    **{
                        fields[index + 5]: parameters[index]
                        for index in range(6)
                    },
                }
            )

    prediction_path = output_dir / "best_prediction.csv"
    observed = np.asarray(data["sums"], dtype=float) / data["counts"]
    with prediction_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "days", "led", "observed_log_ratio", "predicted_log_ratio", "residual_log_ratio"])
        for group, (day, led, prediction) in enumerate(
            zip(data["days"], data["led"], best_result["prediction"])
        ):
            writer.writerow(
                [
                    group,
                    f"{day:.12g}",
                    int(led),
                    f"{np.mean(observed[:, group]):.12g}",
                    f"{prediction:.12g}",
                    f"{np.mean(observed[:, group]) - prediction:.12g}",
                ]
            )

    summary_path = output_dir / "grid_search_summary.md"
    parameters = best_result["optical_parameters"]
    lines = [
        "# QE-free absolute-power TMM grid search",
        "",
        "The model integrates absolute TMM power entering a semi-infinite HgCdTe substrate over the published LED SED.",
        "",
        f"- Grid points per optical parameter: `{settings['grid_points']}`",
        f"- Optical candidates: `{settings['grid_points'] ** 6}`",
        f"- Accumulation rates searched (nm/day): `{settings['rates']}`",
        f"- Initial ice thickness (nm): `{settings['initial_thickness_nm']}`",
        f"- SED mode/weighting: `{settings['sed_mode']}` / `{settings['sed_weighting']}`",
        f"- QE weighting: `{settings['qe_weighted']}`",
        f"- Wavelength-dependent HgCdTe constants: `{settings['use_hgcdte_optical_constants']}`",
        f"- Fixed corrections: `{settings['fixed_corrections']}`",
        "",
        "## Best result",
        "",
        f"- SSE: `{best_result['sse']:.12g}`",
        f"- Pixel RMS: `{best_result['pixel_rms']:.12g}` log",
        f"- Group RMS: `{best_result['group_rms']:.12g}` log",
        f"- Rate: `{best_result['rate_nm_day']:.12g}` nm/day",
        "",
        "| Layer | Thickness scale | Thickness (nm) | n |",
        "|---:|---:|---:|---:|",
    ]
    for layer in range(3):
        scale = parameters[2 * layer]
        lines.append(
            f"| {layer + 1} | {scale:.12g} | {100.0 * scale:.12g} | "
            f"{parameters[2 * layer + 1]:.12g} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, constrained_layout=True)
    prediction = best_result["prediction"]
    observed_mean = np.mean(observed, axis=0)
    for led_value, axis in enumerate(axes.flat, start=1):
        groups = np.flatnonzero(np.asarray(data["led"]) == led_value)
        axis.plot(data["days"][groups], observed_mean[groups], "ko", ms=3, label="Observed")
        axis.plot(data["days"][groups], prediction[groups], color="#D55E00", label="Absolute-power TMM")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_title(f"LED {led_value}")
        axis.set_xlabel("Days since first exposure")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Log ratio")
    axes[1, 0].set_ylabel("Log ratio")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Constant-rate, absolute substrate-power TMM")
    plot_path = output_dir / "best_fit.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return summary_path, result_path, prediction_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--output-dir", type=Path, default=Path("wfi01_tmm_power_grid"))
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--rate-grid", default="14.0")
    parser.add_argument("--initial-thickness-nm", type=float, default=0.0)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--sed-weighting", choices=("energy", "photon"), default="energy")
    parser.add_argument("--qe-weighted", action="store_true")
    parser.add_argument("--ice-index", type=float, default=1.25)
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--top-count", type=int, default=25)
    parser.add_argument(
        "--fixed-corrections",
        type=Path,
        default=DEFAULT_FIXED_CORRECTIONS,
        help="Fixed gain/ACM sidecar applied before the grid search.",
    )
    parser.add_argument("--no-hgcdte-optical-constants", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not np.isfinite(args.initial_thickness_nm) or args.initial_thickness_nm < 0.0:
        raise ValueError("Initial thickness must be finite and non-negative")
    if not np.isfinite(args.ice_index) or args.ice_index <= 0.0:
        raise ValueError("Ice index must be positive and finite")
    if args.top_count < 1:
        raise ValueError("Top-count must be positive")
    rates = parse_float_list(args.rate_grid)
    if any(rate <= 0.0 for rate in rates):
        raise ValueError("All accumulation rates must be positive")
    data = load_tile_statistics(args.statistics, max_tiles=args.max_tiles)
    if args.fixed_corrections is not None:
        data = apply_fixed_corrections(data, args.fixed_corrections)
    spectra = load_sed(SED_URL)
    qe_wavelengths = qe_values = None
    if args.qe_weighted:
        qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    source_integrations = build_source_integrations(
        spectra,
        args.sed_mode,
        args.sed_weighting,
        qe_wavelengths=qe_wavelengths,
        qe_values=qe_values,
        qe_column=0,
    )
    settings = {
        "grid_points": args.grid_points,
        "rates": rates,
        "initial_thickness_nm": args.initial_thickness_nm,
        "sed_mode": args.sed_mode,
        "sed_weighting": args.sed_weighting,
        "qe_weighted": args.qe_weighted,
        "use_hgcdte_optical_constants": not args.no_hgcdte_optical_constants,
        "fixed_corrections": None if args.fixed_corrections is None else str(args.fixed_corrections),
    }
    best_results, best_result = run_grid_search(
        data,
        source_integrations,
        args.grid_points,
        rates,
        args.initial_thickness_nm,
        args.ice_index,
        not args.no_hgcdte_optical_constants,
        args.top_count,
    )
    paths = write_outputs(args.output_dir, data, best_results, best_result, settings)
    for path in paths:
        print(f"Wrote {path}")
    print(
        f"Best: SSE={best_result['sse']:.8g}; "
        f"group RMS={best_result['group_rms']:.8g} log; "
        f"rate={best_result['rate_nm_day']:.8g} nm/day"
    )


if __name__ == "__main__":
    main()
