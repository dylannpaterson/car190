#!/usr/bin/env python3
"""Fit a smooth wavelength modulation of the WFI ice-response ratio.

The source SED is fixed to the measured CALSPEC WD 1657+343 photon shape.  A
photon-counting blackbody tail, anchored at the final CALSPEC sample, supplies
the unmeasured long-wavelength part needed by F213.  Unlike
``fit_wfi_blackbody_sed.py``, the free smooth modulation multiplies the ice
response ratio rather than the source SED.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import emcee
import matplotlib
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_blackbody_sed as base


ROOT = Path(__file__).resolve().parent
DEFAULT_RESPONSE_RATIO = base.DEFAULT_RESPONSE_RATIO
DEFAULT_DATA = base.DEFAULT_DATA
DEFAULT_BANDPASS_FILE = base.ROMAN_EFFECTIVE_AREA_PATH
DEFAULT_CALSPEC_SPECTRUM = base.DEFAULT_CALSPEC_SPECTRUM
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_098/plots/calspec_response_modulation"
)


def tile_label(path):
    for part in reversed(Path(path).parts):
        if part.startswith("tile_"):
            return part.replace("_", " ")
    return "WFI05"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--response-ratio", type=Path, default=DEFAULT_RESPONSE_RATIO)
    parser.add_argument("--bandpass-file", type=Path, default=DEFAULT_BANDPASS_FILE)
    parser.add_argument("--calspec-spectrum", type=Path, default=DEFAULT_CALSPEC_SPECTRUM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--extrapolation-temperature",
        type=float,
        default=46750.0,
        help="Temperature in K for the blackbody extrapolation outside CALSPEC.",
    )
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--response-modulation-sigma",
        type=float,
        default=None,
        help="Gaussian prior width for response log-modulation nodes.",
    )
    parser.add_argument(
        "--response-maxent-strength",
        type=float,
        default=None,
        help="Maximum-entropy regularization strength for response nodes.",
    )
    parser.add_argument(
        "--response-maxent-log-bound",
        type=float,
        default=5.0,
        help="Finite numerical support for response log-modulation nodes.",
    )
    parser.add_argument(
        "--response-modulation-node-spacing",
        type=float,
        default=None,
        help="Full-area response-node spacing in nm.",
    )
    parser.add_argument(
        "--response-modulation-full-area",
        action="store_true",
        help="Place response nodes across the full Roman effective-area support.",
    )
    parser.add_argument(
        "--map-initialize",
        action="store_true",
        help="Optimize the posterior first and initialize walkers around its MAP state.",
    )
    parser.add_argument(
        "--map-maxiter",
        type=int,
        default=3000,
        help="Maximum L-BFGS-B iterations for MAP initialization.",
    )
    return parser.parse_args(argv)


def calspec_blackbody_extrapolated_shape(
    wavelength_nm,
    calspec_wavelengths_nm,
    calspec_shape,
    extrapolation_temperature_k,
):
    """Return CALSPEC inside its range and a continuous blackbody tail outside."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    scalar = wavelengths.ndim == 0
    wavelengths = np.atleast_1d(wavelengths)
    calspec_wavelengths_nm = np.asarray(calspec_wavelengths_nm, dtype=float)
    calspec_shape = np.asarray(calspec_shape, dtype=float)
    blackbody = base.photon_blackbody_shape(wavelengths, extrapolation_temperature_k)
    endpoint_blackbody = base.photon_blackbody_shape(
        np.asarray(
            [calspec_wavelengths_nm[0], calspec_wavelengths_nm[-1]],
            dtype=float,
        ),
        extrapolation_temperature_k,
    )
    endpoint_scales = np.asarray(
        [calspec_shape[0], calspec_shape[-1]], dtype=float
    ) / endpoint_blackbody
    shape = np.interp(wavelengths, calspec_wavelengths_nm, calspec_shape)
    below = wavelengths < calspec_wavelengths_nm[0]
    above = wavelengths > calspec_wavelengths_nm[-1]
    shape[below] = blackbody[below] * endpoint_scales[0]
    shape[above] = blackbody[above] * endpoint_scales[1]
    return float(shape[0]) if scalar else shape


def response_modulation_factor(wavelength_nm, node_log_amplitudes, node_wavelengths):
    return base.sed_modulation_factor(
        wavelength_nm,
        node_log_amplitudes,
        node_wavelengths=node_wavelengths,
        anchor_wavelengths=np.asarray([], dtype=float),
    )


def bandpass_ratios(
    fixed_sed,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    filters,
    response_node_log_amplitudes,
    response_node_wavelengths,
):
    """Fold a fixed photon SED through a modulated ice-response ratio."""
    predictions = []
    for name in filters:
        area = np.asarray(bandpasses[name], dtype=float)
        support = area > 0.0
        wavelengths = np.asarray(area_wavelengths_nm[support], dtype=float)
        area = area[support]
        ice_ratio = np.interp(
            wavelengths,
            response_wavelengths_nm,
            response_ratio,
            left=float(response_ratio[0]),
            right=float(response_ratio[-1]),
        )
        ice_ratio *= response_modulation_factor(
            wavelengths,
            response_node_log_amplitudes,
            response_node_wavelengths,
        )
        photon_sed = fixed_sed(wavelengths)
        clean_counts = np.trapezoid(area * photon_sed, wavelengths)
        ice_counts = np.trapezoid(area * photon_sed * ice_ratio, wavelengths)
        if clean_counts <= 0.0 or not np.isfinite(clean_counts):
            raise ValueError(f"Invalid clean count integral for {name}")
        predictions.append(ice_counts / clean_counts)
    return np.asarray(predictions, dtype=float)


def log_probability(
    theta,
    fixed_sed,
    observations,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    response_node_wavelengths,
    response_modulation_sigma,
    response_maxent_strength,
    response_maxent_log_bound,
    entropy_wavelengths,
    entropy_weights,
):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != len(response_node_wavelengths):
        return -np.inf
    if response_maxent_strength is not None and np.any(
        np.abs(theta) > response_maxent_log_bound
    ):
        return -np.inf
    log_prior = 0.0
    if response_modulation_sigma is not None:
        log_prior += float(
            -0.5
            * np.sum(
                (theta / response_modulation_sigma) ** 2
                + np.log(2.0 * np.pi * response_modulation_sigma**2)
            )
        )
    elif response_maxent_strength is not None:
        log_prior += response_maxent_strength * base.relative_maxent_score(
            entropy_wavelengths,
            theta,
            response_node_wavelengths,
            np.asarray([], dtype=float),
            entropy_weights,
        )
    filters = [row["filter"] for row in observations]
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    model = bandpass_ratios(
        fixed_sed,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        filters,
        theta,
        response_node_wavelengths,
    )
    residual = (np.log(observed) - np.log(model)) / (sigma / observed)
    return float(
        log_prior
        - 0.5 * np.sum(residual**2 + np.log(2.0 * np.pi * (sigma / observed) ** 2))
    )


def write_model_csv(path, observations, model_quantiles):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "filter",
                "pivot_wavelength_nm",
                "observed_cr_ratio",
                "ratio_sigma",
                "model_p16",
                "model_median",
                "model_p84",
            ),
        )
        writer.writeheader()
        for index, row in enumerate(observations):
            writer.writerow(
                {
                    "filter": row["filter"],
                    "pivot_wavelength_nm": f"{row['pivot_wavelength_nm']:.9g}",
                    "observed_cr_ratio": f"{row['cr_ratio']:.9g}",
                    "ratio_sigma": f"{row['ratio_sigma']:.9g}",
                    "model_p16": f"{model_quantiles[0, index]:.9g}",
                    "model_median": f"{model_quantiles[1, index]:.9g}",
                    "model_p84": f"{model_quantiles[2, index]:.9g}",
                }
            )


def write_response_csv(
    path,
    response_wavelengths_nm,
    response_ratio,
    response_quantiles,
):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "wavelength_nm",
                "unmodulated_response_ratio",
                "modulated_response_p16",
                "modulated_response_median",
                "modulated_response_p84",
            ),
        )
        writer.writeheader()
        for index, wavelength in enumerate(response_wavelengths_nm):
            writer.writerow(
                {
                    "wavelength_nm": f"{wavelength:.9g}",
                    "unmodulated_response_ratio": f"{response_ratio[index]:.9g}",
                    "modulated_response_p16": f"{response_quantiles[0, index]:.9g}",
                    "modulated_response_median": f"{response_quantiles[1, index]:.9g}",
                    "modulated_response_p84": f"{response_quantiles[2, index]:.9g}",
                }
            )


def chain_convergence_diagnostics(sampler, burn, response_node_wavelengths):
    """Summarize raw post-burn chain convergence without flattening walkers."""
    chain = np.asarray(sampler.get_chain(discard=burn, flat=False), dtype=float)
    log_probability = np.asarray(
        sampler.get_log_prob(discard=burn, flat=False), dtype=float
    )
    autocorrelation_time = np.asarray(
        emcee.autocorr.integrated_time(chain, quiet=True), dtype=float
    ).reshape(-1)
    half = chain.shape[0] // 2
    first = chain[:half]
    second = chain[-half:]
    first_median = np.median(first, axis=(0, 1))
    second_median = np.median(second, axis=(0, 1))
    first_quantiles = np.percentile(
        first.reshape(-1, first.shape[-1]), [16.0, 50.0, 84.0], axis=0
    )
    second_quantiles = np.percentile(
        second.reshape(-1, second.shape[-1]), [16.0, 50.0, 84.0], axis=0
    )
    first_log_probability = log_probability[:half]
    second_log_probability = log_probability[-half:]
    return {
        "post_burn_steps": int(chain.shape[0]),
        "walkers": int(chain.shape[1]),
        "dimensions": int(chain.shape[2]),
        "finite_chain": bool(np.all(np.isfinite(chain))),
        "finite_log_probability": bool(np.all(np.isfinite(log_probability))),
        "integrated_autocorrelation_time_steps": autocorrelation_time.tolist(),
        "autocorrelation_time_min_steps": float(np.min(autocorrelation_time)),
        "autocorrelation_time_median_steps": float(np.median(autocorrelation_time)),
        "autocorrelation_time_max_steps": float(np.max(autocorrelation_time)),
        "largest_autocorrelation_node_nm": float(
            response_node_wavelengths[int(np.argmax(autocorrelation_time))]
        ),
        "post_steps_over_50_max_tau": float(
            chain.shape[0] / (50.0 * np.max(autocorrelation_time))
        ),
        "effective_samples_min": float(
            chain.shape[0] * chain.shape[1] / np.max(autocorrelation_time)
        ),
        "effective_samples_median": float(
            chain.shape[0] * chain.shape[1] / np.median(autocorrelation_time)
        ),
        "effective_samples_max": float(
            chain.shape[0] * chain.shape[1] / np.min(autocorrelation_time)
        ),
        "split_max_abs_node_median_difference": float(
            np.max(np.abs(first_median - second_median))
        ),
        "split_rms_node_median_difference": float(
            np.sqrt(np.mean((first_median - second_median) ** 2))
        ),
        "split_max_abs_p16_difference": float(
            np.max(np.abs(first_quantiles[0] - second_quantiles[0]))
        ),
        "split_max_abs_p84_difference": float(
            np.max(np.abs(first_quantiles[2] - second_quantiles[2]))
        ),
        "split_log_probability_median_first": float(
            np.median(first_log_probability)
        ),
        "split_log_probability_median_second": float(
            np.median(second_log_probability)
        ),
        "split_log_probability_median_difference": float(
            np.median(first_log_probability) - np.median(second_log_probability)
        ),
        "split_log_probability_mean_first": float(np.mean(first_log_probability)),
        "split_log_probability_mean_second": float(np.mean(second_log_probability)),
        "split_log_probability_mean_difference": float(
            np.mean(first_log_probability) - np.mean(second_log_probability)
        ),
    }


def write_outputs(
    args,
    observations,
    fixed_sed,
    calspec_wavelengths_nm,
    calspec_shape,
    posterior_node_log_amplitudes,
    sampler,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    response_node_wavelengths,
    entropy_wavelengths,
    entropy_weights,
    map_initialization=None,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = [row["filter"] for row in observations]
    sample_stride = max(1, len(posterior_node_log_amplitudes) // 2000)
    model_samples = np.asarray(
        [
            bandpass_ratios(
                fixed_sed,
                area_wavelengths_nm,
                bandpasses,
                response_wavelengths_nm,
                response_ratio,
                filters,
                node_logs,
                response_node_wavelengths,
            )
            for node_logs in posterior_node_log_amplitudes[::sample_stride]
        ]
    )
    model_quantiles = np.percentile(model_samples, [16.0, 50.0, 84.0], axis=0)
    write_model_csv(output_dir / "calspec_response_modulation_model.csv", observations, model_quantiles)
    response_modulation_samples = np.asarray(
        [
            response_ratio
            * response_modulation_factor(
                response_wavelengths_nm,
                node_logs,
                response_node_wavelengths,
            )
            for node_logs in posterior_node_log_amplitudes[::sample_stride]
        ]
    )
    response_quantiles = np.percentile(
        response_modulation_samples,
        [16.0, 50.0, 84.0],
        axis=0,
    )
    write_response_csv(
        output_dir / "calspec_modulated_response.csv",
        response_wavelengths_nm,
        response_ratio,
        response_quantiles,
    )
    np.savez_compressed(
        output_dir / "calspec_response_modulation_chain.npz",
        chain=sampler.get_chain(),
        log_probability=sampler.get_log_prob(),
        response_node_wavelength_nm=np.asarray(response_node_wavelengths, dtype=float),
        response_node_log_amplitude=posterior_node_log_amplitudes,
    )

    median_node_log_amplitudes = np.median(posterior_node_log_amplitudes, axis=0)
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    median_model = bandpass_ratios(
        fixed_sed,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        filters,
        median_node_log_amplitudes,
        response_node_wavelengths,
    )
    chi_squared_at_node_median = float(
        np.sum(((observed - median_model) / sigma) ** 2)
    )
    chi_squared = float(
        np.sum(((observed - model_quantiles[1]) / sigma) ** 2)
    )
    maxent_score = (
        None
        if args.response_maxent_strength is None
        else base.relative_maxent_score(
            entropy_wavelengths,
            median_node_log_amplitudes,
            response_node_wavelengths,
            np.asarray([], dtype=float),
            entropy_weights,
        )
    )
    convergence_diagnostics = chain_convergence_diagnostics(
        sampler,
        args.burn,
        response_node_wavelengths,
    )
    summary = {
        "data": str(Path(args.data).resolve()),
        "response_ratio": str(Path(args.response_ratio).resolve()),
        "roman_effective_area": str(Path(args.bandpass_file).resolve()),
        "fit_parameter": "full_area_log_response_modulation",
        "fixed_sed": {
            "path": str(Path(args.calspec_spectrum).resolve()),
            "target": "WD1657+343",
            "flux_units": "erg s-1 cm-2 Angstrom-1",
            "photon_shape": "CALSPEC FLAM multiplied by wavelength, normalized at 1000 nm",
            "measured_range_nm": [
                float(calspec_wavelengths_nm[0]),
                float(calspec_wavelengths_nm[-1]),
            ],
            "extrapolation": "photon blackbody anchored continuously at each CALSPEC endpoint",
            "extrapolation_temperature_k": float(args.extrapolation_temperature),
        },
        "response_modulation": {
            "node_wavelength_nm": np.asarray(response_node_wavelengths, dtype=float).tolist(),
            "node_spacing_nm": float(args.response_modulation_node_spacing),
            "regularization": (
                "maximum_entropy"
                if args.response_maxent_strength is not None
                else "gaussian_log_amplitude"
            ),
            "log_amplitude_prior_sigma": (
                None
                if args.response_modulation_sigma is None
                else float(args.response_modulation_sigma)
            ),
            "maxent_strength": (
                None
                if args.response_maxent_strength is None
                else float(args.response_maxent_strength)
            ),
            "maxent_log_bound": (
                None
                if args.response_maxent_strength is None
                else float(args.response_maxent_log_bound)
            ),
            "log_amplitude_p16": np.percentile(
                posterior_node_log_amplitudes, 16.0, axis=0
            ).tolist(),
            "log_amplitude_median": median_node_log_amplitudes.tolist(),
            "log_amplitude_p84": np.percentile(
                posterior_node_log_amplitudes, 84.0, axis=0
            ).tolist(),
            "maxent_score_at_median": maxent_score,
        },
        "chi_squared_at_median": chi_squared,
        "chi_squared_at_node_median": chi_squared_at_node_median,
        "data_points": len(observations),
        "image_digitization": True,
        "input_data_note": (
            "CR ratios were digitized from the supplied image; each point was assigned "
            "a fixed 0.005 absolute-ratio uncertainty"
        ),
        "modulation_interpretation": (
            "The fitted log modulation multiplies the ice response ratio in the "
            "ice-count integral; it does not modulate the clean-band denominator."
        ),
        "model_plot": str((output_dir / "calspec_response_modulation_fit.png").resolve()),
        "modulated_response_csv": str(
            (output_dir / "calspec_modulated_response.csv").resolve()
        ),
        "sed_plot": str((output_dir / "calspec_extrapolated_sed.png").resolve()),
        "response_modulation_posterior_plot": str(
            (output_dir / "response_modulation_posterior.png").resolve()
        ),
        "response_modulation_trace_plot": str(
            (output_dir / "response_modulation_trace.png").resolve()
        ),
        "emcee_walkers": int(sampler.nwalkers),
        "emcee_steps": int(sampler.iteration),
        "emcee_burn": int(args.burn),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "convergence_diagnostics": convergence_diagnostics,
        "emcee_initialization": (
            "map_centered" if map_initialization is not None else "zero_centered"
        ),
        "map_initialization": map_initialization,
    }

    wavelength_upper = max(
        2200.0,
        max(
            float(np.max(area_wavelengths_nm[np.asarray(area) >= 0.1 * np.max(area)]))
            for area in bandpasses.values()
        ),
    )
    response_plot_mask = (response_wavelengths_nm >= 450.0) & (
        response_wavelengths_nm <= wavelength_upper
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    pivot = np.asarray([row["pivot_wavelength_nm"] for row in observations])
    order = np.argsort(pivot)
    axes[0].errorbar(
        pivot[order],
        observed[order],
        yerr=sigma[order],
        fmt="o",
        color="black",
        capsize=3,
        label="Digitized CR ratios",
    )
    axes[0].fill_between(
        pivot[order],
        model_quantiles[0, order],
        model_quantiles[2, order],
        color="#56B4E9",
        alpha=0.35,
        label="Response-modulation posterior 16th-84th percentile",
    )
    axes[0].plot(
        pivot[order],
        model_quantiles[1, order],
        color="#0072B2",
        linewidth=2.0,
        label="CALSPEC + blackbody-tail response model",
    )
    axes[0].set_ylabel("Oct 15 / Sep 28 CR ratio")
    axes[0].set_xlabel("Roman effective wavelength (nm)")
    axes[0].set_title(
        f"{tile_label(args.response_ratio)} response modulation; "
        f"$\\chi^2$={chi_squared:.2f}"
    )
    axes[0].set_xlim(450.0, wavelength_upper)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].fill_between(
        response_wavelengths_nm[response_plot_mask],
        response_quantiles[0, response_plot_mask],
        response_quantiles[2, response_plot_mask],
        color="#56B4E9",
        alpha=0.35,
        label="Modulated response posterior 16th-84th percentile",
    )
    axes[1].plot(
        response_wavelengths_nm[response_plot_mask],
        response_ratio[response_plot_mask],
        color="#6F6F6F",
        linestyle="--",
        linewidth=1.4,
        label="Unmodulated ice response ratio",
    )
    axes[1].plot(
        response_wavelengths_nm[response_plot_mask],
        response_quantiles[1, response_plot_mask],
        color="#009E73",
        linewidth=2.0,
        label="Modulated response median",
    )
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Ice response ratio")
    axes[1].set_title("Unmodulated and modulated spectral response")
    axes[1].set_xlim(450.0, wavelength_upper)
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.savefig(output_dir / "calspec_response_modulation_fit.png", dpi=180)
    plt.close(figure)

    wavelengths = np.linspace(450.0, wavelength_upper, 900)
    fixed_shape = fixed_sed(wavelengths)
    blackbody_shape = base.photon_blackbody_shape(wavelengths, args.extrapolation_temperature)
    blackbody_shape /= base.photon_blackbody_shape(1000.0, args.extrapolation_temperature)
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.plot(wavelengths, fixed_shape, color="#D55E00", linewidth=2.0, label="CALSPEC + BB extrapolation")
    axis.plot(
        wavelengths,
        blackbody_shape,
        color="#009E73",
        linestyle="--",
        linewidth=1.4,
        label=f"{args.extrapolation_temperature:.0f} K photon blackbody",
    )
    for name in base.ROMAN_FILTERS:
        area = np.asarray(bandpasses[name], dtype=float)
        support = area >= 0.1 * np.max(area)
        if np.any(support):
            lower = float(np.min(area_wavelengths_nm[support]))
            upper = float(np.max(area_wavelengths_nm[support]))
            axis.axvspan(lower, upper, color="0.5", alpha=0.045, linewidth=0.0)
            axis.text(
                0.5 * (lower + upper),
                0.98,
                name,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="0.35",
            )
    axis.axvline(calspec_wavelengths_nm[-1], color="#D55E00", linestyle=":", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_xlim(450.0, wavelength_upper)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Relative photon-counting SED\n(normalized at 1000 nm)")
    axis.set_title("Fixed CALSPEC SED with blackbody extrapolation")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output_dir / "calspec_extrapolated_sed.png", dpi=180)
    plt.close(figure)

    columns = min(6, len(response_node_wavelengths))
    rows = (len(response_node_wavelengths) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.8 * columns, 2.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    posterior_values = posterior_node_log_amplitudes
    for index, (axis, wavelength) in enumerate(
        zip(axes.flat, response_node_wavelengths)
    ):
        values = posterior_values[:, index]
        axis.hist(values, bins=35, density=True, color="#009E73", alpha=0.75)
        axis.axvline(0.0, color="0.25", linewidth=1.0)
        axis.axvline(np.median(values), color="#D55E00", linewidth=1.8)
        if args.response_modulation_sigma is not None:
            prior_grid = np.linspace(
                -4.0 * args.response_modulation_sigma,
                4.0 * args.response_modulation_sigma,
                300,
            )
            prior_density = np.exp(
                -0.5 * (prior_grid / args.response_modulation_sigma) ** 2
            ) / (args.response_modulation_sigma * np.sqrt(2.0 * np.pi))
            axis.plot(prior_grid, prior_density, color="#D55E00", linestyle="--")
        axis.set_title(f"{wavelength:.0f} nm")
        axis.set_xlabel("Log response amplitude")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(response_node_wavelengths) :]:
        axis.set_visible(False)
    axes.flat[0].set_ylabel("Posterior density")
    figure.savefig(output_dir / "response_modulation_posterior.png", dpi=180)
    plt.close(figure)

    trace = sampler.get_chain()[:, :, :]
    trace_columns = min(3, len(response_node_wavelengths))
    trace_rows = (len(response_node_wavelengths) + trace_columns - 1) // trace_columns
    figure, axes = plt.subplots(
        trace_rows,
        trace_columns,
        figsize=(12, 2.3 * trace_rows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for index, (axis, wavelength) in enumerate(zip(axes.flat, response_node_wavelengths)):
        axis.plot(trace[:, :, index], alpha=0.12, linewidth=0.5)
        axis.axhline(0.0, color="#D55E00", linewidth=1.0)
        axis.set_ylabel(f"{wavelength:.0f} nm")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(response_node_wavelengths) :]:
        axis.set_visible(False)
    for axis in axes.flat[-trace_columns:]:
        axis.set_xlabel("Emcee step")
    figure.savefig(output_dir / "response_modulation_trace.png", dpi=180)
    plt.close(figure)

    summary_path = output_dir / "calspec_response_modulation_fit_summary.json"
    convergence_path = output_dir / "calspec_response_modulation_convergence.json"
    convergence_path.write_text(
        json.dumps(convergence_diagnostics, indent=2, sort_keys=True) + "\n"
    )
    summary["convergence_diagnostics_path"] = str(convergence_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.nwalkers < 4 or args.steps <= args.burn or args.thin < 1:
        raise ValueError("Use at least four walkers and steps greater than burn")
    if args.extrapolation_temperature <= 0.0:
        raise ValueError("Extrapolation temperature must be positive")
    if args.response_modulation_sigma is not None and args.response_modulation_sigma <= 0.0:
        raise ValueError("Response modulation prior width must be positive")
    if args.response_maxent_strength is not None and args.response_maxent_strength <= 0.0:
        raise ValueError("Maximum-entropy strength must be positive")
    if args.response_maxent_log_bound <= 0.0:
        raise ValueError("Maximum-entropy log bound must be positive")
    if args.map_maxiter < 1:
        raise ValueError("MAP maximum iterations must be positive")
    if args.response_modulation_sigma is None and args.response_maxent_strength is None:
        raise ValueError("Choose Gaussian or maximum-entropy response regularization")
    if args.response_modulation_sigma is not None and args.response_maxent_strength is not None:
        raise ValueError("Choose either Gaussian response regularization or maximum entropy")
    if not args.response_modulation_full_area:
        raise ValueError("Use --response-modulation-full-area for this response test")
    if args.response_modulation_node_spacing is None:
        raise ValueError("Response modulation requires --response-modulation-node-spacing")

    area_wavelengths_nm, bandpasses = base.load_bandpasses(args.bandpass_file)
    response_wavelengths_nm, response_ratio = base.load_response_ratio(args.response_ratio)
    observations = base.load_observations(args.data)
    calspec = base.load_calspec_spectrum(args.calspec_spectrum)
    if calspec is None:
        raise FileNotFoundError(f"CALSPEC spectrum not found: {args.calspec_spectrum}")
    calspec_wavelengths_nm, calspec_shape = calspec
    fixed_sed = lambda wavelengths: calspec_blackbody_extrapolated_shape(
        wavelengths,
        calspec_wavelengths_nm,
        calspec_shape,
        args.extrapolation_temperature,
    )
    response_node_wavelengths = base.build_full_area_modulation_nodes(
        area_wavelengths_nm,
        bandpasses,
        args.response_modulation_node_spacing,
    )
    entropy_wavelengths = area_wavelengths_nm
    entropy_weights = np.zeros_like(area_wavelengths_nm, dtype=float)
    for area in bandpasses.values():
        entropy_weights += np.asarray(area, dtype=float) / np.max(area)
    entropy_weights *= fixed_sed(entropy_wavelengths)

    ndim = len(response_node_wavelengths)
    if args.nwalkers < max(4, 2 * ndim):
        raise ValueError(f"Use at least {max(4, 2 * ndim)} walkers for {ndim} parameters")
    log_probability_args = (
        fixed_sed,
        observations,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        response_node_wavelengths,
        args.response_modulation_sigma,
        args.response_maxent_strength,
        args.response_maxent_log_bound,
        entropy_wavelengths,
        entropy_weights,
    )
    map_initialization = None
    map_start = np.zeros(ndim, dtype=float)
    if args.map_initialize:
        def map_objective(theta):
            value = log_probability(theta, *log_probability_args)
            return 1.0e100 if not np.isfinite(value) else -value

        bounds = [
            (-args.response_maxent_log_bound + 1.0e-8, args.response_maxent_log_bound - 1.0e-8)
        ] * ndim
        map_result = minimize(
            map_objective,
            map_start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": args.map_maxiter,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "maxls": 50,
            },
        )
        if not np.all(np.isfinite(map_result.x)) or not np.isfinite(map_result.fun):
            raise RuntimeError("MAP initialization returned a non-finite state")
        map_start = np.asarray(map_result.x, dtype=float)
        map_initialization = {
            "success": bool(map_result.success),
            "message": str(map_result.message),
            "iterations": int(map_result.nit),
            "function_evaluations": int(map_result.nfev),
            "log_posterior": float(-map_result.fun),
            "max_abs_log_amplitude": float(np.max(np.abs(map_start))),
        }
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    positions = map_start + 1.0e-3 * rng.normal(size=(args.nwalkers, ndim))
    if args.response_maxent_strength is not None:
        positions = np.clip(
            positions,
            -args.response_maxent_log_bound + 1.0e-6,
            args.response_maxent_log_bound - 1.0e-6,
        )
    sampler = emcee.EnsembleSampler(
        args.nwalkers,
        ndim,
        log_probability,
        args=log_probability_args,
    )
    sampler.run_mcmc(positions, args.steps, progress=False)
    posterior_node_log_amplitudes = sampler.get_chain(
        discard=args.burn,
        thin=args.thin,
        flat=True,
    )
    summary = write_outputs(
        args,
        observations,
        fixed_sed,
        calspec_wavelengths_nm,
        calspec_shape,
        posterior_node_log_amplitudes,
        sampler,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        response_node_wavelengths,
        entropy_wavelengths,
        entropy_weights,
        map_initialization,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
