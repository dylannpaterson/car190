#!/usr/bin/env python3
"""Fit a blackbody temperature to an image-derived WFI ice-response ratio.

The observed values are assumed to be independent per-filter count-rate ratios
between two ice states.  The model is a photon-counting blackbody folded
through the Roman effective area and the production-TMM ice response ratio.
The blackbody normalization cancels independently in every filter ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ROMAN_FILTERS = ("F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213")
EFFECTIVE_LED_WAVELENGTHS_NM = np.asarray(
    [
        622.344620431963,
        846.342064885868,
        1031.7737605274665,
        1261.2256752723229,
        1482.4771405873898,
        1658.734259886027,
    ],
    dtype=float,
)
SED_MODULATION_NODE_WAVELENGTHS_NM = 0.5 * (
    EFFECTIVE_LED_WAVELENGTHS_NM[:-1] + EFFECTIVE_LED_WAVELENGTHS_NM[1:]
)
ROMAN_EFFECTIVE_AREA_PATH = (
    ROOT / "external_reference_data/Roman_effarea_v8_SCA05_20240301.ecsv"
)
DEFAULT_RESPONSE_RATIO = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_136/plots/ice_response_ratio_20261015_vs_20260928.csv"
)
DEFAULT_DATA = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_136/plots/blackbody_sed_digitized.csv"
)
DEFAULT_CALSPEC_SPECTRUM = (
    ROOT / "external_reference_data/wd1657_343_stisnic_003.ascii"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "20261014/corrected_statistics_4h_thinned/production/tile_size_256/wfi05/"
    "fits/tile_136/plots/blackbody_sed"
)

PLANCK_H = 6.62607015e-34
PLANCK_C = 299792458.0
PLANCK_K = 1.380649e-23


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--response-ratio", type=Path, default=DEFAULT_RESPONSE_RATIO)
    parser.add_argument("--bandpass-file", type=Path, default=ROMAN_EFFECTIVE_AREA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--initial-temperature", type=float, default=15000.0)
    parser.add_argument("--min-temperature", type=float, default=3000.0)
    parser.add_argument("--max-temperature", type=float, default=100000.0)
    parser.add_argument(
        "--prior-center-temperature",
        type=float,
        default=None,
        help="Optional Gaussian white-dwarf temperature prior center in K.",
    )
    parser.add_argument(
        "--prior-sigma-temperature",
        type=float,
        default=None,
        help="Optional Gaussian white-dwarf temperature prior width in K.",
    )
    parser.add_argument(
        "--sed-modulation-sigma",
        type=float,
        default=None,
        help=(
            "Optional zero-centered Gaussian prior width for inter-LED log SED "
            "modulation amplitudes; omit to disable modulation."
        ),
    )
    parser.add_argument(
        "--sed-modulation-node-spacing",
        type=float,
        default=None,
        help=(
            "Optional spacing in nm for additional inter-LED SED nodes. "
            "The default uses one midpoint per LED gap."
        ),
    )
    parser.add_argument(
        "--sed-modulation-full-area",
        action="store_true",
        help="Place free SED nodes across the full Roman effective-area support.",
    )
    parser.add_argument(
        "--sed-maxent-strength",
        type=float,
        default=None,
        help=(
            "Optional maximum-entropy regularization strength relative to the "
            "blackbody reference SED; mutually exclusive with --sed-modulation-sigma."
        ),
    )
    parser.add_argument(
        "--sed-maxent-log-bound",
        type=float,
        default=5.0,
        help="Finite numerical support for maximum-entropy log modulation amplitudes.",
    )
    return parser.parse_args(argv)


def load_bandpasses(path):
    """Load raw Roman effective areas without peak normalization."""
    path = Path(path)
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = list(csv.DictReader(lines, skipinitialspace=True))
    if not rows:
        raise ValueError(f"Roman effective-area file has no rows: {path}")
    if "Wave" not in rows[0] or any(name not in rows[0] for name in ROMAN_FILTERS):
        raise ValueError("Roman effective-area file is missing required filters")
    wavelengths_nm = np.asarray(
        [float(row["Wave"]) * 1000.0 for row in rows], dtype=float
    )
    if np.any(~np.isfinite(wavelengths_nm)) or np.any(np.diff(wavelengths_nm) <= 0.0):
        raise ValueError("Roman effective-area wavelengths must increase")
    bandpasses = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in ROMAN_FILTERS
    }
    for name, values in bandpasses.items():
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"Invalid effective area for {name}")
        if not np.any(values > 0.0):
            raise ValueError(f"Effective area has no positive support for {name}")
    return wavelengths_nm, bandpasses


def load_response_ratio(path):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"wavelength_nm", "relative_sensitivity"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Response-ratio CSV is missing required columns: {path}")
    wavelengths_nm = np.asarray([float(row["wavelength_nm"]) for row in rows])
    response = np.asarray([float(row["relative_sensitivity"]) for row in rows])
    if (
        np.any(~np.isfinite(wavelengths_nm))
        or np.any(~np.isfinite(response))
        or np.any(response <= 0.0)
        or np.any(np.diff(wavelengths_nm) <= 0.0)
    ):
        raise ValueError(f"Response-ratio grid is invalid: {path}")
    return wavelengths_nm, response


def load_observations(path):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filter", "pivot_wavelength_nm", "cr_ratio", "ratio_sigma"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Digitized data is missing required columns: {path}")
    observations = []
    for row in rows:
        name = str(row["filter"]).strip().upper()
        if name not in ROMAN_FILTERS:
            raise ValueError(f"Unknown Roman filter in {path}: {name}")
        ratio = float(row["cr_ratio"])
        sigma = float(row["ratio_sigma"])
        if not np.isfinite(ratio) or ratio <= 0.0 or not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"Invalid ratio or uncertainty in {path}: {row}")
        observations.append(
            {
                "filter": name,
                "pivot_wavelength_nm": float(row["pivot_wavelength_nm"]),
                "cr_ratio": ratio,
                "ratio_sigma": sigma,
            }
        )
    observations.sort(key=lambda row: row["pivot_wavelength_nm"])
    if {row["filter"] for row in observations} != set(ROMAN_FILTERS):
        raise ValueError("Digitized data must contain exactly one row per Roman filter")
    return observations


def load_calspec_spectrum(path):
    """Load the CALSPEC FLAM spectrum and convert it to photon-counting shape."""
    path = Path(path)
    if not path.is_file():
        return None
    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"CALSPEC spectrum must have two columns: {path}")
    wavelengths_nm = values[:, 0] * 0.1
    flam = values[:, 1]
    valid = np.isfinite(wavelengths_nm) & np.isfinite(flam) & (flam > 0.0)
    wavelengths_nm = wavelengths_nm[valid]
    flam = flam[valid]
    if len(wavelengths_nm) < 2 or np.any(np.diff(wavelengths_nm) <= 0.0):
        raise ValueError(f"CALSPEC spectrum wavelength grid is invalid: {path}")
    photon_shape = flam * wavelengths_nm
    normalization = float(np.interp(1000.0, wavelengths_nm, photon_shape))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(f"CALSPEC spectrum does not cover 1000 nm: {path}")
    return wavelengths_nm, photon_shape / normalization


def photon_blackbody_shape(wavelength_nm, temperature_k):
    """Return B_lambda(T) multiplied by photon energy conversion."""
    wavelength_m = np.asarray(wavelength_nm, dtype=float) * 1.0e-9
    exponent = PLANCK_H * PLANCK_C / (wavelength_m * PLANCK_K * temperature_k)
    exponent = np.clip(exponent, 1.0e-9, 700.0)
    # Constants independent of wavelength cancel in each clean/ice ratio.
    return wavelength_m**-4 / np.expm1(exponent)


def energy_blackbody_shape(wavelength_nm, temperature_k):
    """Return the energy-per-wavelength Planck shape for plot diagnostics."""
    wavelength_m = np.asarray(wavelength_nm, dtype=float) * 1.0e-9
    exponent = PLANCK_H * PLANCK_C / (wavelength_m * PLANCK_K * temperature_k)
    exponent = np.clip(exponent, 1.0e-9, 700.0)
    return wavelength_m**-5 / np.expm1(exponent)


def build_sed_modulation_nodes(spacing_nm=None):
    if spacing_nm is None:
        return SED_MODULATION_NODE_WAVELENGTHS_NM.copy()
    if spacing_nm <= 0.0:
        raise ValueError("SED modulation node spacing must be positive")
    nodes = []
    for lower, upper in zip(EFFECTIVE_LED_WAVELENGTHS_NM[:-1], EFFECTIVE_LED_WAVELENGTHS_NM[1:]):
        nodes.extend(np.arange(lower + spacing_nm, upper, spacing_nm))
    return np.asarray(nodes, dtype=float)


def build_full_area_modulation_nodes(area_wavelengths_nm, bandpasses, spacing_nm):
    if spacing_nm is None or spacing_nm <= 0.0:
        raise ValueError("Full-area SED modulation requires a positive node spacing")
    support_bounds = []
    for area in bandpasses.values():
        area = np.asarray(area, dtype=float)
        support = area >= 0.1 * np.max(area)
        if np.any(support):
            support_bounds.append(
                (
                    float(np.min(area_wavelengths_nm[support])),
                    float(np.max(area_wavelengths_nm[support])),
                )
            )
    if not support_bounds:
        raise ValueError("Effective-area file has no positive filter support")
    lower = min(bounds[0] for bounds in support_bounds)
    upper = max(bounds[1] for bounds in support_bounds)
    nodes = np.arange(lower, upper, spacing_nm, dtype=float)
    return np.unique(np.r_[nodes, upper])


def sed_modulation_factor(
    wavelength_nm,
    node_log_amplitudes,
    node_wavelengths=SED_MODULATION_NODE_WAVELENGTHS_NM,
    anchor_wavelengths=EFFECTIVE_LED_WAVELENGTHS_NM,
):
    """Return a log-linear SED modulation, optionally anchored at LED centers."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    amplitudes = np.asarray(node_log_amplitudes, dtype=float).reshape(-1)
    node_wavelengths = np.asarray(node_wavelengths, dtype=float).reshape(-1)
    anchor_wavelengths = np.asarray(anchor_wavelengths, dtype=float).reshape(-1)
    if amplitudes.size == 0:
        return np.ones_like(wavelengths, dtype=float)
    if amplitudes.size != node_wavelengths.size:
        raise ValueError("SED modulation amplitudes and node wavelengths must have equal lengths")
    if anchor_wavelengths.size:
        if np.any(node_wavelengths <= anchor_wavelengths[0]) or np.any(
            node_wavelengths >= anchor_wavelengths[-1]
        ):
            raise ValueError("Anchored SED modulation nodes must lie inside the anchor range")
        knot_wavelengths = np.concatenate((anchor_wavelengths, node_wavelengths))
        knot_log_amplitudes = np.concatenate((np.zeros(anchor_wavelengths.size), amplitudes))
    else:
        knot_wavelengths = node_wavelengths
        knot_log_amplitudes = amplitudes
    order = np.argsort(knot_wavelengths)
    knot_wavelengths = knot_wavelengths[order]
    knot_log_amplitudes = knot_log_amplitudes[order]
    if np.any(np.diff(knot_wavelengths) <= 0.0):
        raise ValueError("SED modulation nodes must be distinct from LED centers and each other")
    log_modulation = np.interp(
        wavelengths,
        knot_wavelengths,
        knot_log_amplitudes,
        left=float(knot_log_amplitudes[0]),
        right=float(knot_log_amplitudes[-1]),
    )
    return np.exp(np.clip(log_modulation, -50.0, 50.0))


def relative_maxent_score(
    wavelength_nm,
    node_log_amplitudes,
    node_wavelengths,
    anchor_wavelengths,
    weights,
):
    """Return normalized relative entropy, maximized by an unmodulated SED."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    weights = np.asarray(weights, dtype=float)
    modulation = sed_modulation_factor(
        wavelengths,
        node_log_amplitudes,
        node_wavelengths=node_wavelengths,
        anchor_wavelengths=anchor_wavelengths,
    )
    normalization = np.trapezoid(weights, wavelengths)
    if normalization <= 0.0 or not np.isfinite(normalization):
        raise ValueError("Maximum-entropy weights must have positive finite support")
    entropy_density = modulation - 1.0 - modulation * np.log(
        np.maximum(modulation, np.finfo(float).tiny)
    )
    return float(np.trapezoid(weights * entropy_density, wavelengths) / normalization)


def photon_sed_shape(
    wavelength_nm,
    temperature_k,
    node_log_amplitudes=(),
    node_wavelengths=SED_MODULATION_NODE_WAVELENGTHS_NM,
    anchor_wavelengths=EFFECTIVE_LED_WAVELENGTHS_NM,
):
    return photon_blackbody_shape(wavelength_nm, temperature_k) * sed_modulation_factor(
        wavelength_nm,
        node_log_amplitudes,
        node_wavelengths=node_wavelengths,
        anchor_wavelengths=anchor_wavelengths,
    )


def bandpass_ratios(
    temperature_k,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    filters=ROMAN_FILTERS,
    sed_node_log_amplitudes=None,
    sed_node_wavelengths=SED_MODULATION_NODE_WAVELENGTHS_NM,
    sed_anchor_wavelengths=EFFECTIVE_LED_WAVELENGTHS_NM,
):
    """Fold the ice response through each photon-counting bandpass."""
    if temperature_k <= 0.0 or not np.isfinite(temperature_k):
        raise ValueError("Temperature must be positive and finite")
    predictions = []
    for name in filters:
        area = np.asarray(bandpasses[name], dtype=float)
        support = area > 0.0
        wavelengths = np.asarray(area_wavelengths_nm[support], dtype=float)
        area = area[support]
        # The response grid starts at 480 nm while F062 has a very small tail
        # below it. Edge extrapolation keeps the clean and ice integrals on the
        # same full effective-area support instead of silently renormalizing it.
        ice_ratio = np.interp(
            wavelengths,
            response_wavelengths_nm,
            response_ratio,
            left=float(response_ratio[0]),
            right=float(response_ratio[-1]),
        )
        photon_sed = photon_blackbody_shape(wavelengths, temperature_k)
        if sed_node_log_amplitudes is not None:
            photon_sed *= sed_modulation_factor(
                wavelengths,
                sed_node_log_amplitudes,
                node_wavelengths=sed_node_wavelengths,
                anchor_wavelengths=sed_anchor_wavelengths,
            )
        clean_counts = np.trapezoid(area * photon_sed, wavelengths)
        ice_counts = np.trapezoid(area * photon_sed * ice_ratio, wavelengths)
        if clean_counts <= 0.0 or not np.isfinite(clean_counts):
            raise ValueError(f"Invalid clean count integral for {name}")
        predictions.append(ice_counts / clean_counts)
    return np.asarray(predictions, dtype=float)


def log_probability(
    theta,
    observations,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    min_temperature,
    max_temperature,
    prior_center_temperature,
    prior_sigma_temperature,
    sed_node_wavelengths,
    sed_modulation_sigma,
    sed_anchor_wavelengths,
    sed_maxent_strength,
    sed_maxent_log_bound,
    entropy_wavelengths,
    entropy_weights,
):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != 1 + len(sed_node_wavelengths):
        return -np.inf
    log10_temperature = float(theta[0])
    sed_node_log_amplitudes = theta[1:]
    lower = np.log10(min_temperature)
    upper = np.log10(max_temperature)
    if not lower < log10_temperature < upper:
        return -np.inf
    temperature = 10.0**log10_temperature
    if (prior_center_temperature is None) != (prior_sigma_temperature is None):
        raise ValueError("Both white-dwarf prior parameters must be supplied together")
    log_prior = 0.0
    if prior_center_temperature is not None:
        standardized = (temperature - prior_center_temperature) / prior_sigma_temperature
        # The sampler coordinate is log10(T), so include the T-to-log10(T)
        # Jacobian for a Gaussian prior defined in kelvin.
        log_prior = (
            -0.5 * standardized**2
            - np.log(prior_sigma_temperature * np.sqrt(2.0 * np.pi))
            + np.log(np.log(10.0) * temperature)
        )
    if sed_node_log_amplitudes.size:
        if sed_modulation_sigma is not None and sed_maxent_strength is not None:
            raise ValueError("Choose either Gaussian SED regularization or maximum entropy")
        if sed_modulation_sigma is not None:
            if sed_modulation_sigma <= 0.0:
                raise ValueError("SED modulation requires a positive prior width")
            log_prior += float(
                -0.5
                * np.sum(
                    (sed_node_log_amplitudes / sed_modulation_sigma) ** 2
                    + np.log(2.0 * np.pi * sed_modulation_sigma**2)
                )
            )
        elif sed_maxent_strength is not None:
            if np.any(np.abs(sed_node_log_amplitudes) > sed_maxent_log_bound):
                return -np.inf
            log_prior += sed_maxent_strength * relative_maxent_score(
                entropy_wavelengths,
                sed_node_log_amplitudes,
                sed_node_wavelengths,
                sed_anchor_wavelengths,
                entropy_weights,
            )
        else:
            raise ValueError("SED modulation requires Gaussian or maximum-entropy regularization")
    filters = [row["filter"] for row in observations]
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    model = bandpass_ratios(
        temperature,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        filters=filters,
        sed_node_log_amplitudes=sed_node_log_amplitudes,
        sed_node_wavelengths=sed_node_wavelengths,
        sed_anchor_wavelengths=sed_anchor_wavelengths,
    )
    residual = (np.log(observed) - np.log(model)) / (sigma / observed)
    return float(
        log_prior
        - 0.5 * np.sum(residual**2 + np.log(2.0 * np.pi * (sigma / observed) ** 2))
    )


def write_posterior_shape_plot(
    output_dir,
    posterior_temperatures,
    posterior_node_log_amplitudes,
    area_wavelengths_nm,
    bandpasses,
    observations,
    model_samples,
    prior_center_temperature,
    prior_sigma_temperature,
    sed_node_wavelengths,
    sed_anchor_wavelengths,
):
    """Plot the fitted SED shape and its posterior sensitivity over Roman bands."""
    output_dir = Path(output_dir)
    calspec_spectrum = load_calspec_spectrum(DEFAULT_CALSPEC_SPECTRUM)
    filter_support_bounds = []
    for name in ROMAN_FILTERS:
        area = np.asarray(bandpasses[name], dtype=float)
        support = area >= 0.1 * np.max(area)
        if np.any(support):
            filter_support_bounds.append(
                (
                    float(np.min(area_wavelengths_nm[support])),
                    float(np.max(area_wavelengths_nm[support])),
                )
            )
    wavelength_upper = max(
        2200.0,
        max((bounds[1] for bounds in filter_support_bounds), default=2200.0),
    )
    wavelengths_nm = np.linspace(450.0, wavelength_upper, 900)
    posterior_node_log_amplitudes = np.asarray(posterior_node_log_amplitudes, dtype=float)
    if posterior_node_log_amplitudes.ndim == 1:
        if posterior_node_log_amplitudes.size == 0:
            posterior_node_log_amplitudes = np.empty((len(posterior_temperatures), 0))
        else:
            posterior_node_log_amplitudes = posterior_node_log_amplitudes.reshape(-1, 1)
    draw_count = min(120, len(posterior_temperatures))
    draw_indices = np.linspace(0, len(posterior_temperatures) - 1, draw_count, dtype=int)
    draw_temperatures = posterior_temperatures[draw_indices]
    draw_node_log_amplitudes = posterior_node_log_amplitudes[draw_indices]
    sed_draws = np.asarray(
        [
            photon_sed_shape(
                wavelengths_nm,
                temperature,
                node_logs,
                node_wavelengths=sed_node_wavelengths,
                anchor_wavelengths=sed_anchor_wavelengths,
            )
            / photon_sed_shape(
                1000.0,
                temperature,
                node_logs,
                node_wavelengths=sed_node_wavelengths,
                anchor_wavelengths=sed_anchor_wavelengths,
            )
            for temperature, node_logs in zip(draw_temperatures, draw_node_log_amplitudes)
        ]
    )
    sed_quantiles = np.percentile(sed_draws, [16.0, 50.0, 84.0], axis=0)
    sed_posterior_samples = np.asarray(
        [
            photon_sed_shape(
                wavelengths_nm,
                temperature,
                node_logs,
                node_wavelengths=sed_node_wavelengths,
                anchor_wavelengths=sed_anchor_wavelengths,
            )
            / photon_sed_shape(
                1000.0,
                temperature,
                node_logs,
                node_wavelengths=sed_node_wavelengths,
                anchor_wavelengths=sed_anchor_wavelengths,
            )
            for temperature, node_logs in zip(
                posterior_temperatures[:: max(1, len(posterior_temperatures) // 2000)],
                posterior_node_log_amplitudes[
                    :: max(1, len(posterior_temperatures) // 2000)
                ],
            )
        ]
    )
    sed_posterior_quantiles = np.percentile(
        sed_posterior_samples, [16.0, 50.0, 84.0], axis=0
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    sed_axis = axes[0, 0]
    for sed in sed_draws:
        sed_axis.plot(wavelengths_nm, sed, color="#56B4E9", alpha=0.14, linewidth=0.8)
    sed_axis.fill_between(
        wavelengths_nm,
        sed_quantiles[0],
        sed_quantiles[2],
        color="#56B4E9",
        alpha=0.35,
        label="SED posterior 16th-84th percentile",
    )
    sed_axis.plot(
        wavelengths_nm,
        sed_quantiles[1],
        color="#0072B2",
        linewidth=2.2,
        label="SED posterior median",
    )
    if calspec_spectrum is not None:
        calspec_wavelengths_nm, calspec_shape = calspec_spectrum
        sed_axis.plot(
            calspec_wavelengths_nm,
            calspec_shape,
            color="#D55E00",
            linewidth=1.2,
            alpha=0.9,
            label="CALSPEC WD 1657+343",
        )
    reference_temperature = float(np.median(posterior_temperatures))
    photon_reference = photon_blackbody_shape(wavelengths_nm, reference_temperature)
    photon_reference /= photon_blackbody_shape(1000.0, reference_temperature)
    energy_reference = energy_blackbody_shape(wavelengths_nm, reference_temperature)
    energy_reference /= energy_blackbody_shape(1000.0, reference_temperature)
    sed_axis.plot(
        wavelengths_nm,
        photon_reference,
        color="#009E73",
        linestyle="--",
        linewidth=1.5,
        label=f"Photon blackbody ({reference_temperature:.0f} K)",
    )
    sed_axis.plot(
        wavelengths_nm,
        energy_reference,
        color="#6F6F6F",
        linestyle=":",
        linewidth=1.5,
        label="Energy FLAM blackbody (diagnostic)",
    )
    for name in ROMAN_FILTERS:
        area = np.asarray(bandpasses[name], dtype=float)
        support = area >= 0.1 * np.max(area)
        if np.any(support):
            lower = float(np.min(area_wavelengths_nm[support]))
            upper = float(np.max(area_wavelengths_nm[support]))
            sed_axis.axvspan(lower, upper, color="0.5", alpha=0.045, linewidth=0.0)
            sed_axis.text(
                0.5 * (lower + upper),
                0.98,
                name,
                transform=sed_axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="0.35",
            )
    sed_axis.set_yscale("log")
    sed_axis.set_xlim(450.0, wavelength_upper)
    sed_axis.set_ylabel("Relative photon-counting SED\n(normalized at 1000 nm)")
    sed_axis.set_title("Posterior SED shapes across the Roman wavelength range")
    sed_axis.grid(alpha=0.2)
    sed_axis.legend(frameon=False, fontsize=9)

    ratio_axis = axes[0, 1]
    pivot = np.asarray([row["pivot_wavelength_nm"] for row in observations])
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    ratio_quantiles = np.percentile(model_samples, [16.0, 50.0, 84.0], axis=0)
    order = np.argsort(pivot)
    ratio_axis.errorbar(
        pivot[order],
        observed[order],
        yerr=sigma[order],
        fmt="o",
        color="black",
        capsize=3,
        label="Digitized CR ratios",
    )
    ratio_axis.fill_between(
        pivot[order],
        ratio_quantiles[0, order],
        ratio_quantiles[2, order],
        color="#56B4E9",
        alpha=0.35,
        label="Ratio posterior 16th-84th percentile",
    )
    ratio_axis.plot(
        pivot[order],
        ratio_quantiles[1, order],
        color="#0072B2",
        linewidth=2.0,
        label="Ratio posterior median",
    )
    ratio_axis.set_xlabel("Roman effective wavelength (nm)")
    ratio_axis.set_ylabel("Oct 15 / Sep 28 CR ratio")
    ratio_axis.set_title("Folded ice-response posterior")
    ratio_axis.grid(alpha=0.2)
    ratio_axis.legend(frameon=False, fontsize=9)

    shape_axis = axes[1, 0]
    fractional_shape = 100.0 * (sed_posterior_quantiles / sed_posterior_quantiles[1] - 1.0)
    shape_axis.fill_between(
        wavelengths_nm,
        fractional_shape[0],
        fractional_shape[2],
        color="#56B4E9",
        alpha=0.45,
        label="SED posterior 16th-84th percentile",
    )
    shape_axis.plot(
        wavelengths_nm,
        fractional_shape[1],
        color="#0072B2",
        linewidth=2.0,
        label="Posterior median",
    )
    shape_axis.axhline(0.0, color="0.25", linewidth=1.0)
    shape_axis.set_xlim(450.0, wavelength_upper)
    shape_axis.set_xlabel("Wavelength (nm)")
    shape_axis.set_ylabel("Shape difference from median (%)")
    shape_axis.set_title("Temperature sensitivity after normalization")
    shape_axis.grid(alpha=0.2)
    shape_axis.legend(frameon=False, fontsize=9)
    if filter_support_bounds:
        support_lower = min(bounds[0] for bounds in filter_support_bounds)
        support_upper = max(bounds[1] for bounds in filter_support_bounds)
        support_mask = (wavelengths_nm >= support_lower) & (wavelengths_nm <= support_upper)
    else:
        support_lower = float(wavelengths_nm[0])
        support_upper = float(wavelengths_nm[-1])
        support_mask = np.ones(wavelengths_nm.shape, dtype=bool)
    maximum_shape_difference = float(
        np.max(np.abs(fractional_shape[[0, 2]][:, support_mask]))
    )
    shape_axis.text(
        0.03,
        0.94,
        f"Maximum over {support_lower:.0f}-{support_upper:.0f} nm: "
        f"{maximum_shape_difference:.2f}%",
        transform=shape_axis.transAxes,
        va="top",
        fontsize=9,
    )

    temperature_axis = axes[1, 1]
    temperature_axis.hist(
        posterior_temperatures,
        bins=40,
        density=True,
        color="#0072B2",
        alpha=0.75,
        label="Temperature posterior",
    )
    temperature_axis.axvline(
        float(np.median(posterior_temperatures)),
        color="#D55E00",
        linewidth=2.0,
        label="Posterior median",
    )
    if prior_center_temperature is not None:
        temperature_min = max(
            0.0,
            min(
                float(np.percentile(posterior_temperatures, 0.1)),
                prior_center_temperature - 4.0 * prior_sigma_temperature,
            ),
        )
        temperature_max = max(
            float(np.percentile(posterior_temperatures, 99.9)),
            prior_center_temperature + 4.0 * prior_sigma_temperature,
        )
        prior_temperatures = np.linspace(temperature_min, temperature_max, 500)
        prior_density = np.exp(
            -0.5 * ((prior_temperatures - prior_center_temperature) / prior_sigma_temperature) ** 2
        ) / (prior_sigma_temperature * np.sqrt(2.0 * np.pi))
        temperature_axis.plot(
            prior_temperatures,
            prior_density,
            color="#D55E00",
            linestyle="--",
            linewidth=1.5,
            label="External WD prior",
        )
        temperature_axis.axvline(
            prior_center_temperature,
            color="#D55E00",
            linestyle=":",
            linewidth=1.5,
        )
    temperature_axis.set_xlabel("Blackbody temperature (K)")
    temperature_axis.set_ylabel("Posterior density")
    temperature_axis.set_title("Temperature posterior")
    temperature_axis.grid(alpha=0.2)
    temperature_axis.legend(frameon=False, fontsize=9)

    figure.savefig(output_dir / "blackbody_sed_posterior_shapes.png", dpi=180)
    plt.close(figure)


def write_outputs(
    output_dir,
    observations,
    posterior_temperatures,
    posterior_node_log_amplitudes,
    sampler,
    area_wavelengths_nm,
    bandpasses,
    response_wavelengths_nm,
    response_ratio,
    min_temperature,
    max_temperature,
    burn,
    prior_center_temperature,
    prior_sigma_temperature,
    response_path,
    bandpass_path,
    data_path,
    sed_node_wavelengths,
    sed_modulation_sigma,
    sed_modulation_node_spacing,
    sed_anchor_wavelengths,
    sed_maxent_strength,
    sed_maxent_log_bound,
    entropy_wavelengths,
    entropy_weights,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = [row["filter"] for row in observations]
    sample_stride = max(1, len(posterior_temperatures) // 2000)
    model_samples = np.asarray(
        [
            bandpass_ratios(
                temperature,
                area_wavelengths_nm,
                bandpasses,
                response_wavelengths_nm,
                response_ratio,
                filters=filters,
                sed_node_log_amplitudes=node_logs,
                sed_node_wavelengths=sed_node_wavelengths,
                sed_anchor_wavelengths=sed_anchor_wavelengths,
            )
            for temperature, node_logs in zip(
                posterior_temperatures[::sample_stride],
                posterior_node_log_amplitudes[::sample_stride],
            )
        ]
    )
    model_quantiles = np.percentile(model_samples, [16.0, 50.0, 84.0], axis=0)
    model_path = output_dir / "blackbody_sed_model.csv"
    with model_path.open("w", newline="") as handle:
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

    chain_path = output_dir / "blackbody_sed_chain.npz"
    np.savez_compressed(
        chain_path,
        temperature_k=posterior_temperatures,
        log10_temperature=np.log10(posterior_temperatures),
        chain=sampler.get_chain(),
        log_probability=sampler.get_log_prob(),
        sed_node_wavelength_nm=np.asarray(sed_node_wavelengths, dtype=float),
        sed_node_log_amplitude=posterior_node_log_amplitudes,
        sed_anchor_wavelength_nm=np.asarray(sed_anchor_wavelengths, dtype=float),
    )
    median_temperature = float(np.median(posterior_temperatures))
    median_node_log_amplitudes = np.median(posterior_node_log_amplitudes, axis=0)
    observed = np.asarray([row["cr_ratio"] for row in observations])
    sigma = np.asarray([row["ratio_sigma"] for row in observations])
    median_model = bandpass_ratios(
        median_temperature,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        filters=filters,
        sed_node_log_amplitudes=median_node_log_amplitudes,
        sed_node_wavelengths=sed_node_wavelengths,
        sed_anchor_wavelengths=sed_anchor_wavelengths,
    )
    chi_squared = float(np.sum(((observed - median_model) / sigma) ** 2))
    median_maxent_score = (
        None
        if sed_maxent_strength is None
        else relative_maxent_score(
            entropy_wavelengths,
            median_node_log_amplitudes,
            sed_node_wavelengths,
            sed_anchor_wavelengths,
            entropy_weights,
        )
    )
    summary = {
        "data": str(Path(data_path).resolve()),
        "response_ratio": str(Path(response_path).resolve()),
        "roman_effective_area": str(Path(bandpass_path).resolve()),
        "fit_parameter": (
            "log10_temperature"
            if not len(sed_node_wavelengths)
            else (
                "log10_temperature + inter_led_log_sed_modulation"
                if len(sed_anchor_wavelengths)
                else "log10_temperature + full_area_log_sed_modulation"
            )
        ),
        "temperature_prior_k": [float(min_temperature), float(max_temperature)],
        "white_dwarf_temperature_prior_k": (
            None
            if prior_center_temperature is None
            else [float(prior_center_temperature), float(prior_sigma_temperature)]
        ),
        "temperature_k_p16": float(np.percentile(posterior_temperatures, 16.0)),
        "temperature_k_median": median_temperature,
        "temperature_k_p84": float(np.percentile(posterior_temperatures, 84.0)),
        "sed_modulation": {
            "enabled": bool(len(sed_node_wavelengths)),
            "effective_led_wavelength_nm": EFFECTIVE_LED_WAVELENGTHS_NM.tolist(),
            "anchor_wavelength_nm": np.asarray(sed_anchor_wavelengths, dtype=float).tolist(),
            "full_area_nodes": bool(
                len(sed_node_wavelengths) and len(sed_anchor_wavelengths) == 0
            ),
            "node_wavelength_nm": np.asarray(sed_node_wavelengths, dtype=float).tolist(),
            "node_spacing_nm": (
                None
                if sed_modulation_node_spacing is None
                else float(sed_modulation_node_spacing)
            ),
            "log_amplitude_prior_sigma": (
                None if sed_modulation_sigma is None else float(sed_modulation_sigma)
            ),
            "regularization": (
                "maximum_entropy"
                if sed_maxent_strength is not None
                else "gaussian_log_amplitude"
                if sed_modulation_sigma is not None
                else None
            ),
            "maxent_strength": (
                None if sed_maxent_strength is None else float(sed_maxent_strength)
            ),
            "maxent_log_bound": (
                None if sed_maxent_strength is None else float(sed_maxent_log_bound)
            ),
            "maxent_score_at_median": median_maxent_score,
            "log_amplitude_p16": np.percentile(
                posterior_node_log_amplitudes, 16.0, axis=0
            ).tolist(),
            "log_amplitude_median": median_node_log_amplitudes.tolist(),
            "log_amplitude_p84": np.percentile(
                posterior_node_log_amplitudes, 84.0, axis=0
            ).tolist(),
        },
        "chi_squared_at_median": chi_squared,
        "data_points": len(observations),
        "blackbody_normalization": "cancels independently in every filter ratio",
        "spectral_weighting": "photon_counting_B_lambda(T) * wavelength / (h c)",
        "effective_area_normalization": "raw Roman effective areas; no peak normalization",
        "input_data_note": (
            "CR ratios were digitized from the supplied image; each point was assigned "
            "a fixed 0.005 absolute-ratio uncertainty"
        ),
        "calspec_spectrum": (
            None
            if not DEFAULT_CALSPEC_SPECTRUM.is_file()
            else {
                "path": str(DEFAULT_CALSPEC_SPECTRUM.resolve()),
                "target": "WD1657+343",
                "flux_units": "erg s-1 cm-2 Angstrom-1",
                "plot_shape": "FLAM multiplied by wavelength, normalized at 1000 nm",
            }
        ),
        "response_edge_handling": "edge-value extrapolation over tiny effective-area tails",
        "posterior_shape_plot": str(
            (output_dir / "blackbody_sed_posterior_shapes.png").resolve()
        ),
        "sed_modulation_posterior_plot": (
            None
            if len(sed_node_wavelengths) == 0
            else str((output_dir / "sed_modulation_posterior.png").resolve())
        ),
        "sed_modulation_trace_plot": (
            None
            if len(sed_node_wavelengths) == 0
            else str((output_dir / "sed_modulation_trace.png").resolve())
        ),
        "emcee_walkers": int(sampler.nwalkers),
        "emcee_steps": int(sampler.iteration),
        "emcee_burn": int(burn),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "image_digitization": True,
    }
    response_metadata_path = Path(response_path).with_suffix(".json")
    if response_metadata_path.is_file():
        with response_metadata_path.open() as handle:
            response_metadata = json.load(handle)
        summary["ice_model"] = {
            "metadata": str(response_metadata_path.resolve()),
            "date_a": response_metadata.get("date_a"),
            "date_b": response_metadata.get("date_b"),
            "ice_thickness_a_nm": response_metadata.get("ice_thickness_a_nm"),
            "ice_thickness_b_nm": response_metadata.get("ice_thickness_b_nm"),
            "ice_n_constant": response_metadata.get("ice_n_constant"),
            "fit_dir": response_metadata.get("fit_dir"),
        }
    summary_path = output_dir / "blackbody_sed_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    pivot = np.asarray([row["pivot_wavelength_nm"] for row in observations])
    order = np.argsort(pivot)
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].errorbar(
        pivot[order],
        observed[order],
        yerr=sigma[order],
        fmt="o",
        color="black",
        capsize=3,
        label="Digitized CR ratios",
    )
    sed_model_label = (
        "Blackbody-folded ice model"
        if not len(sed_node_wavelengths)
        else (
            "Blackbody + inter-LED SED modulation"
            if len(sed_anchor_wavelengths)
            else "Blackbody + full-area SED modulation"
        )
    )
    axes[0].plot(
        pivot[order],
        model_quantiles[1, order],
        color="#0072B2",
        linewidth=2.0,
        label=sed_model_label,
    )
    axes[0].fill_between(
        pivot[order],
        model_quantiles[0, order],
        model_quantiles[2, order],
        color="#56B4E9",
        alpha=0.35,
        label="Temperature posterior 16th-84th percentile",
    )
    axes[0].set_ylabel("Oct 15 / Sep 28 CR ratio")
    axes[0].set_xlabel("Roman effective wavelength (nm)")
    axes[0].set_title(
        f"WFI05 tile 136 ice-response ratio; T={median_temperature:.0f} K "
        f"(+{summary['temperature_k_p84'] - median_temperature:.0f}/"
        f"-{median_temperature - summary['temperature_k_p16']:.0f})"
        + (f"; {len(sed_node_wavelengths)} SED nodes" if len(sed_node_wavelengths) else "")
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    log_temperature = np.log10(posterior_temperatures)
    axes[1].hist(log_temperature, bins=40, color="#0072B2", alpha=0.75)
    axes[1].axvline(np.log10(median_temperature), color="#D55E00", linewidth=2.0)
    axes[1].set_xlabel("log10 blackbody temperature (K)")
    axes[1].set_ylabel("Posterior samples")
    axes[1].grid(alpha=0.2)
    figure.savefig(output_dir / "blackbody_sed_fit.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    chain = sampler.get_chain()[:, :, 0]
    axis.plot(chain, alpha=0.18, linewidth=0.5)
    axis.axhline(np.log10(median_temperature), color="#D55E00", linewidth=1.5)
    axis.set_xlabel("Emcee step")
    axis.set_ylabel("log10 temperature (K)")
    axis.set_title("Blackbody-temperature ensemble trace")
    axis.grid(alpha=0.2)
    figure.savefig(output_dir / "blackbody_sed_trace.png", dpi=180)
    plt.close(figure)

    write_posterior_shape_plot(
        output_dir,
        posterior_temperatures,
        posterior_node_log_amplitudes,
        area_wavelengths_nm,
        bandpasses,
        observations,
        model_samples,
        prior_center_temperature,
        prior_sigma_temperature,
        sed_node_wavelengths,
        sed_anchor_wavelengths,
    )
    if len(sed_node_wavelengths):
        posterior_columns = min(6, len(sed_node_wavelengths))
        posterior_rows = (len(sed_node_wavelengths) + posterior_columns - 1) // posterior_columns
        modulation_figure, modulation_axes = plt.subplots(
            posterior_rows,
            posterior_columns,
            figsize=(2.8 * posterior_columns, 2.8 * posterior_rows),
            squeeze=False,
            constrained_layout=True,
        )
        for index, (axis, wavelength) in enumerate(
            zip(modulation_axes.flat, sed_node_wavelengths)
        ):
            values = posterior_node_log_amplitudes[:, index]
            axis.hist(values, bins=35, density=True, color="#009E73", alpha=0.75)
            axis.axvline(0.0, color="0.25", linewidth=1.0)
            axis.axvline(np.median(values), color="#D55E00", linewidth=1.8)
            if sed_modulation_sigma is not None:
                prior_grid = np.linspace(
                    -4.0 * sed_modulation_sigma,
                    4.0 * sed_modulation_sigma,
                    300,
                )
                prior_density = np.exp(
                    -0.5 * (prior_grid / sed_modulation_sigma) ** 2
                ) / (sed_modulation_sigma * np.sqrt(2.0 * np.pi))
                axis.plot(prior_grid, prior_density, color="#D55E00", linestyle="--")
            axis.set_title(f"{wavelength:.0f} nm")
            axis.set_xlabel("Log SED amplitude")
            axis.grid(alpha=0.2)
        for axis in modulation_axes.flat[len(sed_node_wavelengths) :]:
            axis.set_visible(False)
        modulation_axes.flat[0].set_ylabel("Posterior density")
        modulation_figure.savefig(output_dir / "sed_modulation_posterior.png", dpi=180)
        plt.close(modulation_figure)

        trace = sampler.get_chain()[:, :, 1:]
        trace_columns = min(3, len(sed_node_wavelengths))
        trace_rows = (len(sed_node_wavelengths) + trace_columns - 1) // trace_columns
        trace_figure, trace_axes = plt.subplots(
            trace_rows,
            trace_columns,
            figsize=(12, 2.3 * trace_rows),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        for index, (axis, wavelength) in enumerate(
            zip(trace_axes.flat, sed_node_wavelengths)
        ):
            axis.plot(trace[:, :, index], alpha=0.12, linewidth=0.5)
            axis.axhline(0.0, color="#D55E00", linewidth=1.0)
            axis.set_ylabel(f"{wavelength:.0f} nm")
            axis.grid(alpha=0.2)
        for axis in trace_axes.flat[len(sed_node_wavelengths) :]:
            axis.set_visible(False)
        for axis in trace_axes.flat[-trace_columns:]:
            axis.set_xlabel("Emcee step")
        trace_figure.savefig(output_dir / "sed_modulation_trace.png", dpi=180)
        plt.close(trace_figure)

    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.nwalkers < 4 or args.steps <= args.burn or args.thin < 1:
        raise ValueError("Use at least four walkers and steps greater than burn")
    if args.min_temperature <= 0.0 or args.max_temperature <= args.min_temperature:
        raise ValueError("Temperature prior bounds are invalid")
    if not args.min_temperature < args.initial_temperature < args.max_temperature:
        raise ValueError("Initial temperature must lie inside the prior")
    if (args.prior_center_temperature is None) != (
        args.prior_sigma_temperature is None
    ):
        raise ValueError(
            "--prior-center-temperature and --prior-sigma-temperature must be used together"
        )
    if args.prior_center_temperature is not None and (
        args.prior_center_temperature <= 0.0 or args.prior_sigma_temperature <= 0.0
    ):
        raise ValueError("White-dwarf prior center and width must be positive")
    if args.sed_modulation_sigma is not None and args.sed_modulation_sigma <= 0.0:
        raise ValueError("SED modulation prior width must be positive")
    if args.sed_maxent_strength is not None and args.sed_maxent_strength <= 0.0:
        raise ValueError("Maximum-entropy strength must be positive")
    if args.sed_maxent_log_bound <= 0.0:
        raise ValueError("Maximum-entropy log bound must be positive")
    if args.sed_modulation_sigma is not None and args.sed_maxent_strength is not None:
        raise ValueError("Choose either --sed-modulation-sigma or --sed-maxent-strength")
    if args.sed_modulation_node_spacing is not None and (
        args.sed_modulation_sigma is None and args.sed_maxent_strength is None
    ):
        raise ValueError("SED modulation node spacing requires SED regularization")
    if args.sed_modulation_full_area and (
        args.sed_modulation_sigma is None and args.sed_maxent_strength is None
    ):
        raise ValueError("Full-area SED modulation requires SED regularization")
    if args.sed_modulation_full_area and args.sed_modulation_node_spacing is None:
        raise ValueError("Full-area SED modulation requires --sed-modulation-node-spacing")

    area_wavelengths_nm, bandpasses = load_bandpasses(args.bandpass_file)
    response_wavelengths_nm, response_ratio = load_response_ratio(args.response_ratio)
    observations = load_observations(args.data)
    regularized_modulation = (
        args.sed_modulation_sigma is not None or args.sed_maxent_strength is not None
    )
    if not regularized_modulation:
        sed_node_wavelengths = np.asarray([], dtype=float)
        sed_anchor_wavelengths = np.asarray([], dtype=float)
    elif args.sed_modulation_full_area:
        sed_node_wavelengths = build_full_area_modulation_nodes(
            area_wavelengths_nm,
            bandpasses,
            args.sed_modulation_node_spacing,
        )
        sed_anchor_wavelengths = np.asarray([], dtype=float)
    else:
        sed_node_wavelengths = build_sed_modulation_nodes(args.sed_modulation_node_spacing)
        sed_anchor_wavelengths = EFFECTIVE_LED_WAVELENGTHS_NM.copy()
    if args.sed_modulation_sigma is not None and len(sed_node_wavelengths) == 0:
        raise ValueError("SED modulation spacing produced no nodes")
    if args.sed_maxent_strength is not None and len(sed_node_wavelengths) == 0:
        raise ValueError("Maximum-entropy spacing produced no nodes")
    entropy_wavelengths = area_wavelengths_nm
    entropy_weight = np.zeros_like(area_wavelengths_nm, dtype=float)
    for area in bandpasses.values():
        entropy_weight += np.asarray(area, dtype=float) / np.max(area)
    reference_temperature = (
        args.prior_center_temperature
        if args.prior_center_temperature is not None
        else args.initial_temperature
    )
    entropy_weights = entropy_weight * photon_blackbody_shape(
        entropy_wavelengths, reference_temperature
    )
    ndim = 1 + len(sed_node_wavelengths)
    if args.nwalkers < max(4, 2 * ndim):
        raise ValueError(f"Use at least {max(4, 2 * ndim)} walkers for {ndim} parameters")
    log_probability_args = (
        observations,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        args.min_temperature,
        args.max_temperature,
        args.prior_center_temperature,
        args.prior_sigma_temperature,
        sed_node_wavelengths,
        args.sed_modulation_sigma,
        sed_anchor_wavelengths,
        args.sed_maxent_strength,
        args.sed_maxent_log_bound,
        entropy_wavelengths,
        entropy_weights,
    )
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    initial_log_temperature = np.log10(args.initial_temperature)
    positions = np.zeros((args.nwalkers, ndim), dtype=float)
    positions[:, 0] = initial_log_temperature + 0.08 * rng.normal(size=args.nwalkers)
    if ndim > 1:
        positions[:, 1:] = 1.0e-3 * rng.normal(size=(args.nwalkers, ndim - 1))
    positions[:, 0] = np.clip(
        positions[:, 0],
        np.log10(args.min_temperature) + 1.0e-6,
        np.log10(args.max_temperature) - 1.0e-6,
    )
    sampler = emcee.EnsembleSampler(
        args.nwalkers,
        ndim,
        log_probability,
        args=log_probability_args,
    )
    sampler.run_mcmc(positions, args.steps, progress=False)
    posterior_samples = sampler.get_chain(
        discard=args.burn,
        thin=args.thin,
        flat=True,
    )
    posterior_log_temperature = posterior_samples[:, 0]
    posterior_temperatures = 10.0**posterior_log_temperature
    posterior_node_log_amplitudes = posterior_samples[:, 1:]
    summary = write_outputs(
        args.output_dir,
        observations,
        posterior_temperatures,
        posterior_node_log_amplitudes,
        sampler,
        area_wavelengths_nm,
        bandpasses,
        response_wavelengths_nm,
        response_ratio,
        args.min_temperature,
        args.max_temperature,
        args.burn,
        args.prior_center_temperature,
        args.prior_sigma_temperature,
        args.response_ratio,
        args.bandpass_file,
        args.data,
        sed_node_wavelengths,
        args.sed_modulation_sigma,
        args.sed_modulation_node_spacing,
        sed_anchor_wavelengths,
        args.sed_maxent_strength,
        args.sed_maxent_log_bound,
        entropy_wavelengths,
        entropy_weights,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
