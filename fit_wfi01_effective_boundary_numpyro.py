"""Hierarchical WFI01 fit using a per-tile effective complex boundary."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_rcs_soc_ice_model import read_model_table
from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_wfi01_joint_complex_q_soc import (
    load_effective_q,
    prepare_response_integration,
)
from fit_wfi_tmm_numpyro import (
    DEFAULT_MODEL_PATH,
    INITIAL_THICKNESS_INIT_NM,
    REFERENCE_TIME_DAYS,
    REFERENCE_THICKNESS_NM,
    RESET_DAY,
    TRAJECTORY_SCALE_INIT,
    load_tile_statistics,
    require_numpyro,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TILE_STATISTICS = ROOT / "wfi01_tile_log_statistics_512.csv"
DEFAULT_EFFECTIVE_Q_TABLE = (
    ROOT / "effective_sca1_reflectance_physical" / "effective_sca1_reflectance_table.csv"
)
DEFAULT_OUTPUT = ROOT / "wfi01_effective_boundary_numpyro_tile_pilot"

BASE_BOUNDARY_LOWER = np.asarray([-0.8, -0.8, -1.5], dtype=float)
BASE_BOUNDARY_UPPER = np.asarray([0.8, 0.8, 1.5], dtype=float)
BASE_BOUNDARY_PARAMETER_NAMES = (
    "log_q_magnitude_scale",
    "q_phase_offset_rad",
    "log_q_absorption_scale",
)
BOUNDARY_LOWER = BASE_BOUNDARY_LOWER.copy()
BOUNDARY_UPPER = BASE_BOUNDARY_UPPER.copy()
BOUNDARY_REFERENCE = np.zeros(3, dtype=float)
BOUNDARY_PARAMETER_NAMES = BASE_BOUNDARY_PARAMETER_NAMES


def configure_boundary_parameters(include_phase_slope):
    """Select the three- or four-parameter boundary response."""
    global BOUNDARY_LOWER, BOUNDARY_UPPER, BOUNDARY_REFERENCE
    global BOUNDARY_PARAMETER_NAMES
    if include_phase_slope:
        BOUNDARY_LOWER = np.concatenate((BASE_BOUNDARY_LOWER, [-1.5]))
        BOUNDARY_UPPER = np.concatenate((BASE_BOUNDARY_UPPER, [1.5]))
        BOUNDARY_PARAMETER_NAMES = BASE_BOUNDARY_PARAMETER_NAMES + (
            "q_phase_slope_rad_per_normalized_wavelength",
        )
    else:
        BOUNDARY_LOWER = BASE_BOUNDARY_LOWER.copy()
        BOUNDARY_UPPER = BASE_BOUNDARY_UPPER.copy()
        BOUNDARY_PARAMETER_NAMES = BASE_BOUNDARY_PARAMETER_NAMES
    BOUNDARY_REFERENCE = np.zeros(len(BOUNDARY_PARAMETER_NAMES), dtype=float)


def load_response_context(model_path, effective_q_table, sed_mode, wavelength_stride):
    model_wavelengths, _, _ = read_model_table(model_path)
    q_wavelengths, q_base, alpha_base = load_effective_q(effective_q_table)
    if not np.allclose(model_wavelengths, q_wavelengths):
        raise ValueError("The effective q table must use the model wavelength grid")
    if wavelength_stride < 1:
        raise ValueError("wavelength_stride must be positive")
    model_wavelengths = model_wavelengths[::wavelength_stride]
    q_base = q_base[::wavelength_stride]
    alpha_base = alpha_base[::wavelength_stride]
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    integration = prepare_response_integration(
        model_wavelengths,
        spectra,
        qe_wavelengths,
        qe_values,
        sed_mode,
    )
    return {
        "wavelengths": model_wavelengths,
        "q_base": q_base,
        "alpha_base": alpha_base,
        "model_weights": integration[0],
        "outside_output": integration[1],
        "clean_output": integration[2],
    }


def build_prediction_function(data, context):
    _, jnp, _, _, _, _, _ = require_numpyro()
    wavelengths = jnp.asarray(context["wavelengths"])
    q_base = jnp.asarray(context["q_base"])
    alpha_base = jnp.asarray(context["alpha_base"])
    model_weights = jnp.asarray(context["model_weights"])
    outside_output = jnp.asarray(context["outside_output"])
    clean_output = jnp.asarray(context["clean_output"])
    led_index = jnp.asarray(data["led"] - 1, dtype=jnp.int32)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    step_design = jnp.asarray(data["step_design"])
    phase_coordinate = (wavelengths - 1400.0) / 1000.0

    def response_curves(ice_thickness, boundary_parameters):
        phase_delta = boundary_parameters[:, 1, None]
        if len(BOUNDARY_PARAMETER_NAMES) > 3:
            phase_delta = phase_delta + (
                boundary_parameters[:, 3, None] * phase_coordinate[None, :]
            )
        q = q_base[None, :] * jnp.exp(
            boundary_parameters[:, 0, None]
            + 1j * phase_delta
        )
        alpha = alpha_base[None, :] * jnp.exp(boundary_parameters[:, 2, None])
        phase = 4.0 * jnp.pi * 1.25 * ice_thickness[None, :, None] / wavelengths[None, None, :]
        denominator = jnp.abs(1.0 - q[:, None, :] * jnp.exp(1j * phase)) ** 2
        numerator = jnp.abs(1.0 - q[:, None, :]) ** 2
        factors = jnp.exp(-alpha[:, None, :] * ice_thickness[None, :, None])
        factors = factors * numerator / denominator
        integrated = jnp.einsum("tgw,lw->tgl", factors, model_weights)
        integrated = integrated + outside_output[None, None, :]
        return integrated / clean_output[None, None, :]

    def prediction(ice_thickness, boundary_parameters, acm_step):
        response = response_curves(ice_thickness, boundary_parameters)
        current = response[:, jnp.arange(response.shape[1]), led_index]
        baseline_thickness = ice_thickness[first_by_led[led_index]]
        baseline_response = response_curves(baseline_thickness, boundary_parameters)
        baseline = baseline_response[:, jnp.arange(response.shape[1]), led_index]
        return jnp.log(current / baseline) + acm_step * step_design[None, :]

    return prediction


def boundary_reference_raw(jnp):
    probability = (BOUNDARY_REFERENCE - BOUNDARY_LOWER) / (
        BOUNDARY_UPPER - BOUNDARY_LOWER
    )
    return jnp.asarray(jnp.log(probability / (1.0 - probability)))


def build_model(data, context, log_noise_sigma, likelihood_weight):
    jax, jnp, numpyro, dist, _, _, _ = require_numpyro()
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    days = jnp.asarray(data["days"])
    base_thickness = REFERENCE_THICKNESS_NM * (
        jnp.sqrt((RESET_DAY + days) / REFERENCE_TIME_DAYS)
        - jnp.sqrt(RESET_DAY / REFERENCE_TIME_DAYS)
    )
    reference_raw = boundary_reference_raw(jnp)
    lower = jnp.asarray(BOUNDARY_LOWER)
    upper = jnp.asarray(BOUNDARY_UPPER)
    prediction = build_prediction_function(data, context)
    parameter_count = len(BOUNDARY_PARAMETER_NAMES)

    def model():
        boundary_mu_raw = numpyro.sample(
            "boundary_mu_raw",
            dist.Normal(reference_raw, jnp.ones(parameter_count)).to_event(1),
        )
        boundary_sigma_raw = numpyro.sample(
            "boundary_sigma_raw",
            dist.HalfNormal(0.5 * jnp.ones(parameter_count)).to_event(1),
        )
        boundary_z = numpyro.sample(
            "boundary_z",
            dist.Normal(0.0, 1.0)
            .expand((counts.shape[0], parameter_count))
            .to_event(2),
        )
        boundary_raw = boundary_mu_raw[None, :] + boundary_sigma_raw[None, :] * boundary_z
        boundary_parameters = lower + (upper - lower) * jax.nn.sigmoid(boundary_raw)
        d0 = numpyro.sample(
            "initial_thickness_nm",
            dist.TruncatedNormal(35.0, 15.0, low=0.0, high=50.0),
        )
        log_scale = numpyro.sample(
            "log_trajectory_scale", dist.Normal(jnp.log(TRAJECTORY_SCALE_INIT), 0.75)
        )
        acm_step = numpyro.sample("acm_step_log_offset", dist.Normal(0.0, 0.05))
        ice_thickness = d0 + jnp.exp(log_scale) * base_thickness
        fitted = prediction(ice_thickness, boundary_parameters, acm_step)
        residual_sse = (
            sum_squared
            - 2.0 * fitted * sums
            + counts * fitted * fitted
        )
        total_sse = jnp.sum(residual_sse)
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5 * likelihood_weight * total_sse / (log_noise_sigma**2),
        )
        numpyro.deterministic("boundary_parameters", boundary_parameters)
        numpyro.deterministic(
            "boundary_mean",
            lower + (upper - lower) * jax.nn.sigmoid(boundary_mu_raw),
        )
        numpyro.deterministic("ice_thickness", ice_thickness)
        numpyro.deterministic(
            "mean_rate_nm_day",
            jnp.exp(log_scale)
            * (base_thickness[-1] - base_thickness[0])
            / (jnp.max(days) - jnp.min(days)),
        )

    return model


def quantile(values):
    return np.percentile(np.asarray(values), [16, 50, 84])


def posterior_metrics(data, predictions):
    median_prediction = np.percentile(predictions, 50, axis=0)
    residual_sse = (
        data["sum_squared"]
        - 2.0 * median_prediction * data["sums"]
        + data["counts"] * median_prediction * median_prediction
    )
    total_sse = float(np.sum(residual_sse))
    pixel_count = float(np.sum(data["counts"]))
    first = data["first_by_led"]
    nonreference = np.ones(len(data["days"]), dtype=bool)
    nonreference[first] = False
    nonreference_sse = float(np.sum(residual_sse[:, nonreference]))
    nonreference_count = float(np.sum(data["counts"][:, nonreference]))
    return {
        "median_prediction": median_prediction,
        "total_sse": total_sse,
        "pixel_rms_log": float(np.sqrt(total_sse / pixel_count)),
        "nonreference_pixel_rms_log": float(
            np.sqrt(nonreference_sse / nonreference_count)
        ),
        "mean_tile_exposure_rms_log": float(
            np.sqrt(np.mean((data["sums"] / data["counts"] - median_prediction) ** 2))
        ),
    }


def write_tile_parameters(path, data, samples):
    parameters = np.asarray(samples["boundary_parameters"])
    quantiles = np.percentile(parameters, [16, 50, 84], axis=0)
    fields = ["tile_id", "tile_row", "tile_col"]
    for name in BOUNDARY_PARAMETER_NAMES:
        fields.extend((f"{name}_p16", f"{name}_median", f"{name}_p84"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tile_index, tile_id in enumerate(data["tile_ids"]):
            metadata = data["metadata"][int(tile_id)]
            row = {
                "tile_id": int(tile_id),
                "tile_row": metadata["tile_row"],
                "tile_col": metadata["tile_col"],
            }
            for parameter_index, name in enumerate(BOUNDARY_PARAMETER_NAMES):
                row[f"{name}_p16"] = quantiles[0, tile_index, parameter_index]
                row[f"{name}_median"] = quantiles[1, tile_index, parameter_index]
                row[f"{name}_p84"] = quantiles[2, tile_index, parameter_index]
            writer.writerow(row)


def write_predictions(path, data, metrics):
    fields = (
        "tile_id",
        "group",
        "led",
        "repeat",
        "days",
        "observed_percent",
        "predicted_percent",
        "residual_log",
    )
    observed = data["sums"] / data["counts"]
    prediction = metrics["median_prediction"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tile_index, tile_id in enumerate(data["tile_ids"]):
            for group in range(len(data["days"])):
                writer.writerow(
                    {
                        "tile_id": int(tile_id),
                        "group": group,
                        "led": int(data["led"][group]),
                        "repeat": int(data["repeat"][group]),
                        "days": f"{data['days'][group]:.12g}",
                        "observed_percent": f"{100.0 * np.expm1(observed[tile_index, group]):.12g}",
                        "predicted_percent": f"{100.0 * np.expm1(prediction[tile_index, group]):.12g}",
                        "residual_log": f"{observed[tile_index, group] - prediction[tile_index, group]:.12g}",
                    }
                )


def write_fit_plot(path, data, metrics):
    observed = data["sums"] / data["counts"]
    prediction = metrics["median_prediction"]
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharex=False, sharey=True, constrained_layout=True
    )
    for led, axis in enumerate(axes.flat, start=1):
        groups = np.flatnonzero(data["led"] == led)
        order = groups[np.argsort(data["days"][groups], kind="stable")]
        x = data["days"][order]
        axis.scatter(
            np.broadcast_to(x, observed[:, order].shape).ravel(),
            (100.0 * np.expm1(observed[:, order])).ravel(),
            color="#555555",
            alpha=0.16,
            s=7,
            linewidths=0,
        )
        for tile in range(observed.shape[0]):
            axis.plot(
                x,
                100.0 * np.expm1(prediction[tile, order]),
                color="#0072B2",
                alpha=0.22,
                linewidth=0.8,
            )
        residual = observed[:, order] - prediction[:, order]
        axis.set_title(f"LED {led}; RMS={np.sqrt(np.mean(residual**2)):.4g} log")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xlabel("Days since first exposure")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    axes[0, 0].scatter([], [], color="#555555", s=18, label="Observed tile data")
    axes[0, 0].plot([], [], color="#0072B2", linewidth=2.0, label="Boundary model")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("WFI01 tile observations versus hierarchical effective boundary")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_corner_plot(path, values, labels, title):
    import corner

    figure = corner.corner(
        values,
        labels=labels,
        bins=30,
        color="#0072B2",
        fill_contours=True,
        plot_datapoints=False,
        smooth=0.8,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".4g",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
    )
    figure.suptitle(title, y=1.02, fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_corner_plots(output_dir, samples):
    boundary_mean = np.asarray(samples["boundary_mean"], dtype=float)
    boundary_labels = [
        "q magnitude mean",
        "q phase mean (rad)",
        "q absorption mean",
    ]
    tile_labels = [
        "tile log q magnitude",
        "tile q phase (rad)",
        "tile log q absorption",
    ]
    hyper_labels = [
        "q magnitude mean",
        "q phase mean (rad)",
        "q absorption mean",
        "latent sigma magnitude",
        "latent sigma phase",
        "latent sigma absorption",
    ]
    if len(BOUNDARY_PARAMETER_NAMES) > 3:
        boundary_labels.append("q phase slope")
        tile_labels.append("tile q phase slope")
        hyper_labels.extend(("q phase slope mean", "latent sigma phase slope"))
    global_values = np.column_stack(
        (
            samples["initial_thickness_nm"],
            samples["mean_rate_nm_day"],
            100.0 * np.expm1(samples["acm_step_log_offset"]),
            boundary_mean,
        )
    )
    save_corner_plot(
        output_dir / "numpyro_boundary_global_corner.png",
        global_values,
        ("d0 (nm)", "mean rate (nm/day)", "ACM step (%)", *boundary_labels),
        "WFI01 global posterior, effective boundary model",
    )

    hyper_values = np.column_stack(
        (boundary_mean, np.asarray(samples["boundary_sigma_raw"], dtype=float))
    )
    save_corner_plot(
        output_dir / "numpyro_boundary_hyper_corner.png",
        hyper_values,
        hyper_labels,
        "WFI01 effective-boundary population hyperparameters",
    )

    tile_values = np.asarray(samples["boundary_parameters"], dtype=float).reshape(
        -1, len(BOUNDARY_PARAMETER_NAMES)
    )
    save_corner_plot(
        output_dir / "numpyro_boundary_tile_corner.png",
        tile_values,
        tile_labels,
        "WFI01 effective-boundary tile parameters",
    )


def write_summary(path, args, data, samples, metrics):
    parameter_description = ", ".join(BOUNDARY_PARAMETER_NAMES)
    comparison_note = (
        "This is a matched three-parameter response-model comparison."
        if len(BOUNDARY_PARAMETER_NAMES) == 3
        else "This adds one phase-slope parameter per tile as a flexibility diagnostic; it is not a matched three-parameter comparison."
    )
    lines = [
        "# WFI01 hierarchical effective-boundary fit",
        "",
        f"This model uses the existing effective complex-q equation with static per-tile parameters: {parameter_description}. The shared square-root ice trajectory, ACM step, tile likelihood, and hierarchical structure match the spatial TMM fit.",
        "",
        f"Tiles: `{len(data['tile_ids'])}`; groups: `{len(data['days'])}`.",
        f"Effective q table: `{args.effective_q_table}`.",
        f"HgCdTe is not used by this boundary model; wavelength stride is `{args.wavelength_stride}`.",
        "",
        "| Quantity | 16th percentile | Median | 84th percentile |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Initial thickness (nm)", "initial_thickness_nm"),
        ("Mean rate (nm/day)", "mean_rate_nm_day"),
        ("ACM step (%)", "acm_step_log_offset"),
    ):
        values = np.asarray(samples[key])
        if key == "acm_step_log_offset":
            values = 100.0 * np.expm1(values)
        values = quantile(values)
        lines.append(
            f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
        )
    boundary_mean = np.asarray(samples["boundary_mean"])
    for index, name in enumerate(BOUNDARY_PARAMETER_NAMES):
        values = quantile(boundary_mean[:, index])
        lines.append(
            f"| Population mean `{name}` | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
        )
    lines.extend(
        [
            "",
            f"Posterior-median total RMS: `{metrics['pixel_rms_log']:.10g}` log.",
            f"Posterior-median non-reference RMS: `{metrics['nonreference_pixel_rms_log']:.10g}` log.",
            f"Posterior-median tile-exposure RMS: `{metrics['mean_tile_exposure_rms_log']:.10g}` log.",
            "",
            f"{comparison_note} It is not a claim that the effective q table is Eric's exact physical boundary parameterization.",
            "",
            "## Plots",
            "",
            f"- Tile fit: `{path.parent / 'tile_fit.png'}`",
            f"- Global corner: `{path.parent / 'numpyro_boundary_global_corner.png'}`",
            f"- Hyperparameter corner: `{path.parent / 'numpyro_boundary_hyper_corner.png'}`",
            f"- Tile-parameter corner: `{path.parent / 'numpyro_boundary_tile_corner.png'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-statistics", type=Path, default=DEFAULT_TILE_STATISTICS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--effective-q-table", type=Path, default=DEFAULT_EFFECTIVE_Q_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--wavelength-stride", type=int, default=20)
    parser.add_argument("--log-noise-sigma", type=float, default=0.005)
    parser.add_argument("--likelihood-weight", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--include-phase-slope",
        action="store_true",
        help="Add one per-tile wavelength-dependent q phase-slope parameter",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_noise_sigma <= 0.0 or args.likelihood_weight <= 0.0:
        raise ValueError("Noise sigma and likelihood weight must be positive")
    if args.warmup < 1 or args.samples < 1 or args.chains < 1 or args.max_tree_depth < 1:
        raise ValueError("Warmup, samples, chains, and max tree depth must be positive")
    if args.max_tiles is not None and args.max_tiles < 1:
        raise ValueError("max_tiles must be positive")

    jax, _, numpyro, _, MCMC, NUTS, init_to_value = require_numpyro()
    jax.config.update("jax_enable_x64", True)
    configure_boundary_parameters(args.include_phase_slope)
    data = load_tile_statistics(args.tile_statistics, args.max_tiles)
    context = load_response_context(
        args.model_path,
        args.effective_q_table,
        args.sed_mode,
        args.wavelength_stride,
    )
    model = build_model(data, context, args.log_noise_sigma, args.likelihood_weight)
    parameter_count = len(BOUNDARY_PARAMETER_NAMES)
    reference_raw = np.log(
        (BOUNDARY_REFERENCE - BOUNDARY_LOWER)
        / (BOUNDARY_UPPER - BOUNDARY_REFERENCE)
    )
    initial_values = {
        "boundary_mu_raw": reference_raw,
        "boundary_sigma_raw": np.full(parameter_count, 0.2),
        "boundary_z": np.zeros((len(data["tile_ids"]), parameter_count)),
        "initial_thickness_nm": INITIAL_THICKNESS_INIT_NM,
        "log_trajectory_scale": np.log(TRAJECTORY_SCALE_INIT),
        "acm_step_log_offset": 0.0,
    }
    kernel = NUTS(
        model,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        init_strategy=init_to_value(values=initial_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(args.seed))
    samples = mcmc.get_samples(group_by_chain=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_prediction_function(data, context)
    batched_predictor = jax.jit(jax.vmap(predictor, in_axes=(0, 0, 0)))
    predictions = []
    for start in range(0, len(samples["ice_thickness"]), 16):
        stop = min(start + 16, len(samples["ice_thickness"]))
        predictions.append(
            np.asarray(
                batched_predictor(
                    samples["ice_thickness"][start:stop],
                    samples["boundary_parameters"][start:stop],
                    samples["acm_step_log_offset"][start:stop],
                )
            )
        )
    predictions = np.concatenate(predictions, axis=0)
    metrics = posterior_metrics(data, predictions)
    np.savez_compressed(
        args.output_dir / "posterior_samples.npz",
        **{key: np.asarray(value) for key, value in samples.items()},
        tile_ids=data["tile_ids"],
    )
    write_tile_parameters(args.output_dir / "tile_boundary_posterior.csv", data, samples)
    write_predictions(args.output_dir / "tile_predictions.csv", data, metrics)
    write_fit_plot(args.output_dir / "tile_fit.png", data, metrics)
    write_corner_plots(args.output_dir, samples)
    write_summary(args.output_dir / "summary.md", args, data, samples, metrics)
    mcmc.print_summary()
    print(f"Posterior-median rate: {np.median(samples['mean_rate_nm_day']):.8g} nm/day")
    print(f"Posterior-median total RMS: {metrics['pixel_rms_log']:.8g} log")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
