"""Pilot emcee posterior for the one-pixel TMM fit."""

import argparse
import csv
from pathlib import Path

import emcee
import numpy as np

from analyze_rcs_soc_ice_model import (
    generalized_relative_entropy,
    mean_endpoint_rate,
    mean_squared_log_node_curvature,
    response_prediction,
    soc_sqrt_thickness,
)
from analyze_wfi01_all_pixel_soc_model import (
    load_all_pixel_statistics,
    event_step_design,
)
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_effective_sca1_tmm import load_empirical_table
from fit_wfi01_single_pixel_tmm import (
    AR_THICKNESS_SCALE_NM,
    HGCDTE_MODEL_CHOICES,
    TMM_SIGMAS,
    build_tmm_response,
    tmm_configuration,
)
from fit_wfi01_joint_complex_q_soc import prepare_response_integration
from analyze_rcs_soc_ice_model import read_model_table
from fit_wfi01_joint_optical_soc import DEFAULT_STEP_TIME, parse_step_mask


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path.home() / "Documents/rgespit-comissioning-ice-simulations/data/ice_optical_constants/spectral_response_model_SCA1.ecsv"
DEFAULT_STATISTICS = ROOT / "wfi01_single_pixel_1000_1000/single_pixel_log_statistics.csv"
DEFAULT_MAP_OUTPUT = ROOT / "wfi01_single_pixel_1000_1000_tmm_medium"
DEFAULT_OUTPUT = ROOT / "wfi01_single_pixel_1000_1000_tmm_emcee"
REFERENCE_THICKNESS_NM = 122.7
REFERENCE_TIME_DAYS = 4.65
RESET_DAY = 18.0
ENTROPY_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 4.64158883361
MAX_NODE_MULTIPLIER = 2.0
MAX_SCALE = 10.0
# Exploratory upper support while testing the physical initial-thickness prior.
MAX_INITIAL_THICKNESS_NM = 50.0
INITIAL_THICKNESS_PRIOR_SCALE_NM = 10.0
PARAMETER_NAMES = (
    "initial_thickness_nm",
    "log_trajectory_scale",
    *[f"log_node_multiplier_{repeat}" for repeat in range(1, 19)],
    "ar_thickness_scaled",
    "ar_refractive_index",
    "ar_absorption_index",
    "hgcdte_refractive_index",
    "hgcdte_absorption_index",
    "acm_step_log_offset",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-path", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--map-output", type=Path, default=DEFAULT_MAP_OUTPUT)
    parser.add_argument("--secondary-map-output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step-time", default=DEFAULT_STEP_TIME)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--hgcdte-model", choices=HGCDTE_MODEL_CHOICES, default="free")
    parser.add_argument("--thickness-grid-steps", type=int, default=101)
    parser.add_argument("--wavelength-stride", type=int, default=10)
    parser.add_argument("--log-noise-sigma", type=float, default=0.005)
    parser.add_argument(
        "--max-initial-thickness-nm",
        type=float,
        default=MAX_INITIAL_THICKNESS_NM,
    )
    parser.add_argument(
        "--initial-thickness-prior-scale-nm",
        type=float,
        default=INITIAL_THICKNESS_PRIOR_SCALE_NM,
    )
    parser.add_argument("--walkers", type=int, default=56)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--move",
        choices=("stretch", "differential", "snooker", "mixed"),
        default="differential",
    )
    parser.add_argument("--de-gamma", type=float, default=0.2)
    return parser.parse_args()


def read_map_fit(path):
    summary_path = path / "single_pixel_tmm_fit.md"
    lines = summary_path.read_text().splitlines()
    tmm_row = next(line for line in lines if line.startswith("| TMM response |"))
    values = [value.strip() for value in tmm_row.strip("|").split("|")]
    initial_thickness = float(values[1])
    step_percent = float(values[5])
    parameters = {}
    parameter_labels = {
        "AR thickness (nm)": "ar_thickness_scaled",
        "AR refractive index": "ar_refractive_index",
        "AR absorption index": "ar_absorption_index",
        "HgCdTe refractive index": "hgcdte_refractive_index",
        "HgCdTe absorption index": "hgcdte_absorption_index",
    }
    for line in lines:
        if not line.startswith("|"):
            continue
        fields = [value.strip() for value in line.strip("|").split("|")]
        if fields and fields[0] in parameter_labels:
            key = parameter_labels[fields[0]]
            value = float(fields[1])
            parameters[key] = value / AR_THICKNESS_SCALE_NM if key == "ar_thickness_scaled" else value

    nodes_path = path / "tmm_nodes.csv"
    with nodes_path.open(newline="") as handle:
        nodes = [row for row in csv.DictReader(handle) if row["model"] == "tmm"]
    nodes.sort(key=lambda row: int(row["repeat"]))
    node_multipliers = np.asarray([float(row["node_multiplier"]) for row in nodes])
    node_days = np.asarray([float(row["days"]) for row in nodes])
    incremental = np.asarray([float(row["incremental_thickness_nm"]) for row in nodes])
    base = soc_sqrt_thickness(
        node_days,
        REFERENCE_THICKNESS_NM,
        REFERENCE_TIME_DAYS,
        RESET_DAY,
    )
    scale_candidates = incremental[1:] / (base[1:] * node_multipliers[1:])
    log_scale = float(np.log(np.median(scale_candidates)))

    values_by_name = {
        "initial_thickness_nm": initial_thickness,
        "log_trajectory_scale": log_scale,
        **{
            f"log_node_multiplier_{repeat}": np.log(node_multipliers[repeat])
            for repeat in range(1, 19)
        },
        **parameters,
        "acm_step_log_offset": float(np.log1p(step_percent / 100.0)),
    }
    return np.asarray([values_by_name[name] for name in PARAMETER_NAMES], dtype=float)


def parameter_bounds(max_initial_thickness_nm=MAX_INITIAL_THICKNESS_NM, hgcdte_model="free"):
    _, tmm_bounds, _ = tmm_configuration(hgcdte_model)
    lower = [0.0, np.log(1.0e-6)] + [-np.log(MAX_NODE_MULTIPLIER)] * 18
    upper = [max_initial_thickness_nm, np.log(MAX_SCALE)] + [
        np.log(MAX_NODE_MULTIPLIER)
    ] * 18
    lower.extend(bound[0] for bound in tmm_bounds)
    upper.extend(bound[1] for bound in tmm_bounds)
    lower.append(-0.2)
    upper.append(0.2)
    return np.asarray(lower), np.asarray(upper)


def unpack_trajectory(theta, observations):
    repeats = sorted({row["repeat"] for row in observations})
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeats)}
    observation_nodes = np.asarray(
        [repeat_to_index[row["repeat"]] for row in observations], dtype=int
    )
    node_days = np.asarray(
        [
            np.mean([row["days"] for row in observations if row["repeat"] == repeat])
            for repeat in repeats
        ],
        dtype=float,
    )
    base_thickness = soc_sqrt_thickness(
        np.asarray([row["days"] for row in observations]),
        REFERENCE_THICKNESS_NM,
        REFERENCE_TIME_DAYS,
        RESET_DAY,
    )
    base_node_thickness = soc_sqrt_thickness(
        node_days,
        REFERENCE_THICKNESS_NM,
        REFERENCE_TIME_DAYS,
        RESET_DAY,
    )
    node_multipliers = np.exp(np.concatenate(([0.0], theta[2:20])))
    scale = np.exp(theta[1])
    thickness = theta[0] + scale * base_thickness * node_multipliers[observation_nodes]
    node_thickness = theta[0] + scale * base_node_thickness * node_multipliers
    return thickness, node_thickness, node_days, node_multipliers


def make_log_probability(
    observations,
    step_mask,
    d_grid,
    response_factory,
    sigma,
    max_initial_thickness_nm,
    initial_thickness_prior_scale_nm,
    hgcdte_model,
):
    lower, upper = parameter_bounds(max_initial_thickness_nm, hgcdte_model)
    tmm_center, _, _ = tmm_configuration(hgcdte_model)
    step_design = event_step_design(observations, step_mask)

    def log_probability(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(PARAMETER_NAMES),) or np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        thickness, node_thickness, _, node_multipliers = unpack_trajectory(theta, observations)
        chronological_order = np.argsort(
            np.asarray([row["days"] for row in observations]),
            kind="stable",
        )
        if np.any(np.diff(node_thickness) < 0.0) or np.any(
            np.diff(thickness[chronological_order]) < 0.0
        ):
            return -np.inf
        if np.min(thickness) < d_grid[0] or np.max(thickness) > d_grid[-1]:
            return -np.inf
        response = response_factory(theta[20:25])
        if response is None:
            return -np.inf
        prediction = response_prediction(observations, response, d_grid, thickness)
        prediction = prediction + theta[25] * step_design
        residuals = np.asarray(
            [row["log_change"] for row in observations], dtype=float
        ) - prediction
        if not np.all(np.isfinite(residuals)):
            return -np.inf
        entropy = generalized_relative_entropy(node_multipliers)
        smoothness = mean_squared_log_node_curvature(
            np.log(node_multipliers),
            unpack_trajectory(theta, observations)[2],
        )
        optical_penalty = np.mean(
            ((theta[20:25] - tmm_center) / TMM_SIGMAS) ** 2
        )
        return float(
            -0.5 * np.sum((residuals / sigma) ** 2)
            - ENTROPY_WEIGHT * entropy
            - SMOOTHNESS_WEIGHT * smoothness
            - 0.5 * optical_penalty
            - 0.5 * (theta[0] / initial_thickness_prior_scale_nm) ** 2
        )

    return log_probability


def write_outputs(
    output_dir,
    sampler,
    observations,
    theta_map,
    sigma,
    sampler_details,
    fixed_hgcdte=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    chain = sampler.get_chain().copy()
    log_prob = sampler.get_log_prob()
    if fixed_hgcdte is not None:
        chain[:, :, 23:25] = fixed_hgcdte
        theta_map = np.asarray(theta_map, dtype=float).copy()
        theta_map[23:25] = fixed_hgcdte
    burn = max(0, chain.shape[0] // 2)
    flat = chain[burn:].reshape(-1, chain.shape[-1])
    flat_log_probability = log_prob[burn:].reshape(-1)
    posterior_path = output_dir / "posterior_samples.csv"
    fields = [*PARAMETER_NAMES, "mean_rate_nm_day", "log_probability"]
    with posterior_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample, value in zip(flat, flat_log_probability):
            thickness, _, _, _ = unpack_trajectory(sample, observations)
            writer.writerow(
                {
                    **{name: f"{item:.12g}" for name, item in zip(PARAMETER_NAMES, sample)},
                    "mean_rate_nm_day": f"{mean_endpoint_rate(observations, thickness):.12g}",
                    "log_probability": f"{value:.12g}",
                }
            )

    rate_samples = np.asarray(
        [mean_endpoint_rate(observations, unpack_trajectory(sample, observations)[0]) for sample in flat]
    )
    quantile = np.percentile(rate_samples, [16, 50, 84])
    acceptance = np.mean(sampler.acceptance_fraction)
    summary_path = output_dir / "emcee_summary.md"
    parameter_note = (
        f"Parameter count: `{len(PARAMETER_NAMES)}` including the ACM step; "
        f"fixed log-noise sigma: `{sigma}`."
    )
    if fixed_hgcdte is not None:
        parameter_note += (
            f" HgCdTe n/k are frozen at `{fixed_hgcdte[0]:g}`, `{fixed_hgcdte[1]:g}` "
            "in the response model."
        )
    summary_path.write_text(
        "\n".join(
            [
                "# WFI01 single-pixel TMM emcee pilot",
                "",
                parameter_note,
                f"Walkers: `{sampler.nwalkers}`; steps: `{chain.shape[0]}`; mean acceptance fraction: `{acceptance:.6g}`.",
                f"Sampler: `{sampler_details}`.",
                "",
                "| Quantity | 16th percentile | Median | 84th percentile |",
                "|---|---:|---:|---:|",
                f"| Mean ice rate (nm/day) | {quantile[0]:.8f} | {quantile[1]:.8f} | {quantile[2]:.8f} |",
                "",
                "The chain is a pilot. Convergence and multimodality require longer, independent chains.",
                "",
                f"MAP start rate (nm/day): `{mean_endpoint_rate(observations, unpack_trajectory(theta_map, observations)[0]):.8f}`.",
            ]
        )
        + "\n"
    )
    np.savez_compressed(
        output_dir / "chain.npz",
        chain=chain,
        log_probability=log_prob,
        theta_map=theta_map,
    )
    return posterior_path, summary_path


def build_moves(name, de_gamma):
    if name == "stretch":
        return emcee.moves.StretchMove()
    differential = emcee.moves.DEMove(gamma0=de_gamma)
    if name == "differential":
        return differential
    if name == "snooker":
        return emcee.moves.DESnookerMove()
    return [
        (differential, 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ]


def main():
    args = parse_args()
    if args.walkers < 2 * len(PARAMETER_NAMES):
        raise ValueError(f"--walkers must be at least {2 * len(PARAMETER_NAMES)}")
    if args.steps < 1 or args.wavelength_stride < 1:
        raise ValueError("Steps and wavelength stride must be positive")
    if (
        args.max_initial_thickness_nm <= 0.0
        or args.initial_thickness_prior_scale_nm <= 0.0
    ):
        raise ValueError("Initial-thickness prior values must be positive")
    observations = load_all_pixel_statistics(args.statistics_path)
    tmm_center, _, fixed_hgcdte = tmm_configuration(args.hgcdte_model)
    step_mask = parse_step_mask(observations, args.step_time)
    full_wavelengths, _ = load_empirical_table(
        args.model_path
    )
    model_wavelengths, calibrated_thicknesses, _ = read_model_table(
        args.model_path
    )
    if not np.allclose(full_wavelengths, model_wavelengths):
        raise ValueError("TMM and response grids differ")
    model_wavelengths = model_wavelengths[:: args.wavelength_stride]
    d_grid = np.linspace(0.0, float(np.max(calibrated_thicknesses)), args.thickness_grid_steps)
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    integration = prepare_response_integration(
        model_wavelengths,
        spectra,
        qe_wavelengths,
        qe_values,
        args.sed_mode,
    )
    cache = {}

    def response_factory(parameters):
        key = tuple(np.asarray(parameters, dtype=float))
        if key not in cache:
            cache[key] = build_tmm_response(
                model_wavelengths,
                spectra,
                qe_wavelengths,
                qe_values,
                d_grid,
                key,
                integration,
                fixed_hgcdte=fixed_hgcdte,
            )
        return cache[key]

    lower, upper = parameter_bounds(args.max_initial_thickness_nm, args.hgcdte_model)
    theta_map = np.clip(
        read_map_fit(args.map_output),
        lower + 1.0e-8,
        upper - 1.0e-8,
    )
    theta_centers = [theta_map]
    if args.secondary_map_output is not None:
        theta_centers.append(
            np.clip(
                read_map_fit(args.secondary_map_output),
                lower + 1.0e-8,
                upper - 1.0e-8,
            )
        )
    log_probability = make_log_probability(
        observations,
        step_mask,
        d_grid,
        response_factory,
        args.log_noise_sigma,
        args.max_initial_thickness_nm,
        args.initial_thickness_prior_scale_nm,
        args.hgcdte_model,
    )
    rng = np.random.default_rng(args.seed)
    perturbation = np.full(len(PARAMETER_NAMES), 1.0e-4)
    perturbation[0] = 0.05
    perturbation[1:20] = 0.002
    perturbation[20:25] = 0.005
    if fixed_hgcdte is not None:
        perturbation[23:25] = 1.0e-7
    perturbation[25] = 0.001
    center_indices = np.arange(args.walkers) % len(theta_centers)
    centers = np.asarray(theta_centers)[center_indices]
    positions = centers + rng.normal(
        0.0,
        perturbation,
        size=(args.walkers, len(PARAMETER_NAMES)),
    )
    positions = np.clip(positions, lower + 1.0e-8, upper - 1.0e-8)
    sampler = emcee.EnsembleSampler(
        args.walkers,
        len(PARAMETER_NAMES),
        log_probability,
        moves=build_moves(args.move, args.de_gamma),
    )
    sampler.run_mcmc(positions, args.steps, progress=True)
    paths = write_outputs(
        args.output_dir,
        sampler,
        observations,
        theta_map,
        args.log_noise_sigma,
        f"{args.move}, de_gamma={args.de_gamma}, secondary_map={args.secondary_map_output is not None}, "
        f"d0_max={args.max_initial_thickness_nm} nm, d0_prior_scale={args.initial_thickness_prior_scale_nm} nm, "
        f"hgcdte_model={args.hgcdte_model}",
        fixed_hgcdte=fixed_hgcdte,
    )
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")


if __name__ == "__main__":
    main()
