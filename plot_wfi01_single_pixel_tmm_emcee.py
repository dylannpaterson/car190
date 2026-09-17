"""Plot diagnostics for a WFI01 single-pixel emcee chain."""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_rcs_soc_ice_model import (
    read_model_table,
    response_prediction,
    soc_sqrt_thickness,
)
from analyze_wfi01_all_pixel_soc_model import event_step_design, load_all_pixel_statistics
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_effective_sca1_tmm import load_empirical_table
from fit_wfi01_joint_complex_q_soc import prepare_response_integration
from fit_wfi01_joint_optical_soc import DEFAULT_STEP_TIME, parse_step_mask
from sample_wfi01_single_pixel_tmm_emcee import (
    DEFAULT_STATISTICS,
    HGCDTE_MODEL_CHOICES,
    MODEL_PATH,
    build_tmm_response,
    tmm_configuration,
    unpack_trajectory,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CHAIN = ROOT / "wfi01_single_pixel_1000_1000_tmm_emcee_3000_bimodal"
DEFAULT_PLOTS = DEFAULT_CHAIN / "plots"
REFERENCE_THICKNESS_NM = 122.7
REFERENCE_TIME_DAYS = 4.65
RESET_DAY = 18.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-output", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--statistics-path", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--burn-fraction", type=float, default=0.5)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--thickness-grid-steps", type=int, default=101)
    parser.add_argument("--wavelength-stride", type=int, default=10)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--hgcdte-model", choices=HGCDTE_MODEL_CHOICES, default="free")
    parser.add_argument("--mode-threshold-nm", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def load_chain(path, burn_fraction):
    if not 0.0 <= burn_fraction < 1.0:
        raise ValueError("Burn fraction must be in [0, 1)")
    archive = np.load(path / "chain.npz")
    chain = np.asarray(archive["chain"], dtype=float)
    log_probability = np.asarray(archive["log_probability"], dtype=float)
    if chain.ndim != 3 or log_probability.shape != chain.shape[:2]:
        raise ValueError("Chain archive has incompatible shapes")
    burn = int(chain.shape[0] * burn_fraction)
    posterior = chain[burn:]
    posterior_log_probability = log_probability[burn:]
    flat = posterior.reshape(-1, posterior.shape[-1])
    flat_log_probability = posterior_log_probability.reshape(-1)
    valid = np.isfinite(flat).all(axis=1) & np.isfinite(flat_log_probability)
    return chain, log_probability, burn, flat[valid], flat_log_probability[valid]


def trajectory_context(observations):
    repeats = sorted({row["repeat"] for row in observations})
    repeat_to_index = {repeat: index for index, repeat in enumerate(repeats)}
    node_index = np.asarray(
        [repeat_to_index[row["repeat"]] for row in observations], dtype=int
    )
    days = np.asarray([row["days"] for row in observations], dtype=float)
    node_days = np.asarray(
        [
            np.mean([row["days"] for row in observations if row["repeat"] == repeat])
            for repeat in repeats
        ],
        dtype=float,
    )
    base = soc_sqrt_thickness(
        days,
        REFERENCE_THICKNESS_NM,
        REFERENCE_TIME_DAYS,
        RESET_DAY,
    )
    node_base = soc_sqrt_thickness(
        node_days,
        REFERENCE_THICKNESS_NM,
        REFERENCE_TIME_DAYS,
        RESET_DAY,
    )
    return days, node_index, node_days, base, node_base


def rate_samples(parameters, context):
    days, node_index, _, base, _ = context
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    multipliers = np.exp(
        np.concatenate(
            (np.zeros((parameters.shape[0], 1)), parameters[:, 2:20]), axis=1
        )
    )
    scale = np.exp(parameters[:, 1])
    first = int(np.argmin(days))
    last = int(np.argmax(days))
    first_thickness = (
        parameters[:, 0]
        + scale * base[first] * multipliers[:, node_index[first]]
    )
    last_thickness = (
        parameters[:, 0]
        + scale * base[last] * multipliers[:, node_index[last]]
    )
    span = float(np.max(days) - np.min(days))
    return (last_thickness - first_thickness) / span


def node_thickness_samples(parameters, context):
    _, _, _, _, node_base = context
    parameters = np.asarray(parameters, dtype=float)
    multipliers = np.exp(
        np.concatenate(
            (np.zeros((parameters.shape[0], 1)), parameters[:, 2:20]), axis=1
        )
    )
    return parameters[:, [0]] + np.exp(parameters[:, [1]]) * node_base[None, :] * multipliers


def build_response_context(model_path, sed_mode, wavelength_stride, thickness_grid_steps):
    full_wavelengths, _ = load_empirical_table(model_path)
    model_wavelengths, calibrated_thicknesses, _ = read_model_table(model_path)
    if not np.allclose(full_wavelengths, model_wavelengths):
        raise ValueError("TMM and response grids differ")
    model_wavelengths = model_wavelengths[::wavelength_stride]
    d_grid = np.linspace(
        0.0,
        float(np.max(calibrated_thicknesses)),
        thickness_grid_steps,
    )
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    integration = prepare_response_integration(
        model_wavelengths,
        spectra,
        qe_wavelengths,
        qe_values,
        sed_mode,
    )
    return (
        model_wavelengths,
        d_grid,
        spectra,
        qe_wavelengths,
        qe_values,
        integration,
    )


def posterior_predictions(
    parameters,
    observations,
    step_design,
    response_context,
    draws,
    seed,
    fixed_hgcdte=None,
):
    model_wavelengths, d_grid, spectra, qe_wavelengths, qe_values, integration = response_context
    rng = np.random.default_rng(seed)
    count = min(draws, len(parameters))
    indices = rng.choice(len(parameters), size=count, replace=False)
    predictions = []
    selected = []
    for index in indices:
        sample = parameters[index]
        thickness = unpack_trajectory(sample, observations)[0]
        response = build_tmm_response(
            model_wavelengths,
            spectra,
            qe_wavelengths,
            qe_values,
            d_grid,
            sample[20:25],
            integration,
            fixed_hgcdte=fixed_hgcdte,
        )
        if response is None:
            continue
        prediction = response_prediction(observations, response, d_grid, thickness)
        predictions.append(prediction + sample[25] * step_design)
        selected.append(index)
    if len(predictions) < 10:
        raise ValueError("Too few valid posterior-predictive draws")
    return np.asarray(predictions), np.asarray(selected)


def save_trace_plot(path, chain, log_probability, burn, context, seed):
    steps, walkers, _ = chain.shape
    flat_chain = chain.reshape(-1, chain.shape[-1])
    rate_chain = rate_samples(flat_chain, context).reshape(steps, walkers)
    rng = np.random.default_rng(seed)
    walker_count = min(24, walkers)
    walker_indices = np.sort(rng.choice(walkers, size=walker_count, replace=False))
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for walker in walker_indices:
        axes[0].plot(rate_chain[:, walker], lw=0.6, alpha=0.55)
        axes[1].plot(chain[:, walker, 0], lw=0.6, alpha=0.55)
        axes[2].plot(100.0 * chain[:, walker, 20], lw=0.6, alpha=0.55)
        axes[3].plot(log_probability[:, walker], lw=0.6, alpha=0.55)
    labels = (
        "Mean ice rate (nm/day)",
        "Initial thickness (nm)",
        "AR thickness (nm)",
        "Log probability",
    )
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.axvline(burn, color="black", ls="--", lw=0.8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Step")
    figure.suptitle("WFI01 pixel (1000, 1000): emcee traces")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_parameter_plot(path, parameters, log_probability, rates, seed, mode_threshold_nm):
    rng = np.random.default_rng(seed)
    plot_count = min(25000, len(parameters))
    plot_indices = rng.choice(len(parameters), size=plot_count, replace=False)
    sampled = parameters[plot_indices]
    sampled_rates = rates[plot_indices]
    sampled_log_probability = log_probability[plot_indices]
    mode = sampled[:, 0] < mode_threshold_nm
    colors = np.where(mode, "#0072B2", "#D55E00")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].scatter(sampled[:, 0], 100.0 * sampled[:, 20], c=colors, s=3, alpha=0.25)
    axes[0, 0].set_xlabel("Initial thickness (nm)")
    axes[0, 0].set_ylabel("AR thickness (nm)")
    axes[0, 1].scatter(sampled[:, 0], sampled[:, 23], c=colors, s=3, alpha=0.25)
    axes[0, 1].set_xlabel("Initial thickness (nm)")
    axes[0, 1].set_ylabel("HgCdTe n")
    axes[0, 2].scatter(100.0 * sampled[:, 20], sampled[:, 23], c=colors, s=3, alpha=0.25)
    axes[0, 2].set_xlabel("AR thickness (nm)")
    axes[0, 2].set_ylabel("HgCdTe n")
    axes[1, 0].hist(sampled_rates, bins=60, color="#56B4E9", alpha=0.85)
    axes[1, 0].set_xlabel("Mean ice rate (nm/day)")
    axes[1, 1].hist(parameters[:, 0], bins=60, color="#999999", alpha=0.85)
    axes[1, 1].set_xlabel("Initial thickness (nm)")
    axes[1, 2].hist(100.0 * parameters[:, 20], bins=60, color="#999999", alpha=0.85)
    axes[1, 2].set_xlabel("AR thickness (nm)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Posterior parameter space; blue d0 < {mode_threshold_nm:g} nm, "
        f"orange d0 >= {mode_threshold_nm:g} nm"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_trajectory_plot(path, parameters, rates, context, seed):
    node_days = context[2]
    node_samples = node_thickness_samples(parameters, context)
    rng = np.random.default_rng(seed)
    draw_count = min(150, len(parameters))
    draw_indices = rng.choice(len(parameters), size=draw_count, replace=False)
    quantiles = np.percentile(node_samples, [16, 50, 84], axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index in draw_indices:
        axes[0].plot(node_days, node_samples[index], color="#0072B2", alpha=0.08, lw=0.7)
    axes[0].fill_between(node_days, quantiles[0], quantiles[2], color="#56B4E9", alpha=0.35)
    axes[0].plot(node_days, quantiles[1], color="#0072B2", lw=2.0)
    axes[0].set_xlabel("Days since first exposure")
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].set_title("Marginalized ice trajectories")
    axes[0].grid(alpha=0.2)
    axes[1].hist(rates, bins=70, color="#56B4E9", alpha=0.9)
    axes[1].axvline(np.median(rates), color="#D55E00", lw=2.0)
    axes[1].set_xlabel("Mean ice rate (nm/day)")
    axes[1].set_ylabel("Posterior samples")
    axes[1].set_title("Marginalized rate")
    axes[1].grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_prediction_plot(path, observations, predictions):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
    for led, axis in enumerate(axes.flat, start=1):
        indices = [index for index, row in enumerate(observations) if row["led"] == led]
        order = np.argsort([observations[index]["days"] for index in indices])
        indices = np.asarray(indices)[order]
        days = np.asarray([observations[index]["days"] for index in indices])
        observed = 100.0 * np.expm1(
            [observations[index]["log_change"] for index in indices]
        )
        posterior = 100.0 * np.expm1(predictions[:, indices])
        quantiles = np.percentile(posterior, [16, 50, 84], axis=0)
        for draw in posterior[:: max(1, len(posterior) // 25)]:
            axis.plot(days, draw, color="#999999", alpha=0.12, lw=0.7)
        axis.fill_between(days, quantiles[0], quantiles[2], color="#56B4E9", alpha=0.35)
        axis.plot(days, quantiles[1], color="#0072B2", lw=2.0, label="Posterior median")
        axis.plot(days, observed, "ko", ms=3, label="Observed")
        axis.axhline(0.0, color="black", lw=0.6)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Change (%)")
    axes[1, 0].set_ylabel("Change (%)")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Posterior predictive curves for pixel (1000, 1000)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_diagnostics(
    path,
    chain,
    log_probability,
    burn,
    parameters,
    rates,
    mode_threshold_nm,
    fixed_hgcdte=None,
):
    mode = parameters[:, 0] < mode_threshold_nm
    chain_mode = chain[:, :, 0] < mode_threshold_nm
    transitions = int(np.sum(chain_mode[1:] != chain_mode[:-1]))
    try:
        from emcee.autocorr import integrated_time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autocorrelation_chain = chain[burn:]
            if fixed_hgcdte is not None:
                autocorrelation_chain = np.delete(autocorrelation_chain, (23, 24), axis=2)
            tau = integrated_time(autocorrelation_chain, quiet=True)
        tau_note = f"{np.min(tau):.4g} to {np.max(tau):.4g} steps"
    except Exception as error:
        tau_note = f"unavailable ({type(error).__name__})"
    quantile = np.percentile(rates, [16, 50, 84])
    lines = [
        "# WFI01 single-pixel emcee diagnostics",
        "",
        f"Chain shape: `{chain.shape}`; discarded steps: `{burn}`; retained samples: `{len(parameters)}`.",
        f"Finite log probabilities: `{np.isfinite(log_probability).mean():.6g}`.",
        f"Low-d0 mode fraction (d0 < {mode_threshold_nm:g} nm): `{np.mean(mode):.6g}`.",
        f"Mode transitions across walkers: `{transitions}`.",
        f"Estimated autocorrelation time range: `{tau_note}`.",
        "",
        "| Quantity | 16th percentile | Median | 84th percentile |",
        "|---|---:|---:|---:|",
        f"| Mean ice rate (nm/day) | {quantile[0]:.8f} | {quantile[1]:.8f} | {quantile[2]:.8f} |",
        "",
        "The imposed initial-thickness support is retained; no posterior mode was discarded.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.draws < 10 or args.wavelength_stride < 1 or args.thickness_grid_steps < 2:
        raise ValueError("Draws, wavelength stride, and grid steps are too small")
    observations = load_all_pixel_statistics(args.statistics_path)
    _, _, fixed_hgcdte = tmm_configuration(args.hgcdte_model)
    chain, log_probability, burn, parameters, posterior_log_probability = load_chain(
        args.chain_output,
        args.burn_fraction,
    )
    context = trajectory_context(observations)
    rates = rate_samples(parameters, context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_trace_plot(
        args.output_dir / "emcee_traces.png",
        chain,
        log_probability,
        burn,
        context,
        args.seed,
    )
    save_parameter_plot(
        args.output_dir / "emcee_parameter_space.png",
        parameters,
        posterior_log_probability,
        rates,
        args.seed,
        args.mode_threshold_nm,
    )
    save_trajectory_plot(
        args.output_dir / "emcee_trajectory_and_rate.png",
        parameters,
        rates,
        context,
        args.seed,
    )
    step_mask = parse_step_mask(observations, DEFAULT_STEP_TIME)
    step_design = event_step_design(observations, step_mask)
    response_context = build_response_context(
        args.model_path,
        args.sed_mode,
        args.wavelength_stride,
        args.thickness_grid_steps,
    )
    predictions, _ = posterior_predictions(
        parameters,
        observations,
        step_design,
        response_context,
        args.draws,
        args.seed,
        fixed_hgcdte=fixed_hgcdte,
    )
    save_prediction_plot(
        args.output_dir / "emcee_posterior_predictive.png",
        observations,
        predictions,
    )
    write_diagnostics(
        args.output_dir / "emcee_plot_diagnostics.md",
        chain,
        log_probability,
        burn,
        parameters,
        rates,
        args.mode_threshold_nm,
        fixed_hgcdte=fixed_hgcdte,
    )
    for name in (
        "emcee_traces.png",
        "emcee_parameter_space.png",
        "emcee_trajectory_and_rate.png",
        "emcee_posterior_predictive.png",
        "emcee_plot_diagnostics.md",
    ):
        print(f"Wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
