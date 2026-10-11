"""Fit one common count-space time shift with independent detector/LED scales."""

import argparse
import csv
import json
from pathlib import Path

import emcee
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fit_wfi_count_shift_scale_mcmc import _segment_prediction, load_model_data


RESET_DAY = 15.064405023148149
SEED = 20260930
WARMUP_STEPS = 2500
PRODUCTION_STEPS = 6000
NWALKERS = 128


def common_delta_bounds(model):
    lower = max(
        model["reset_day"] + segment["shift_prior_lower"]
        for segment in model["segments"]
    )
    upper = min(
        model["reset_day"] + segment["shift_prior_upper"]
        for segment in model["segments"]
    )
    if not lower < upper:
        raise ValueError(f"No common full-overlap delta_t prior: [{lower}, {upper}]")
    return float(lower), float(upper)


def _parameter_arrays(model, parameters):
    parameters = np.asarray(parameters, dtype=float)
    scalar = parameters.ndim == 1
    parameters = np.atleast_2d(parameters)
    n_segments = model["n_segments"]
    expected = 1 + 2 * n_segments
    if parameters.shape[1] != expected:
        raise ValueError(f"Expected {expected} parameters, got {parameters.shape[1]}")
    return (
        scalar,
        parameters,
        parameters[:, 0],
        parameters[:, 1 : 1 + n_segments],
        parameters[:, 1 + n_segments :],
    )


def log_probability(model, parameters):
    scalar, parameters, delta_t, log_scale, log_sigma = _parameter_arrays(model, parameters)
    lower, upper = common_delta_bounds(model)
    valid = (
        np.isfinite(parameters).all(axis=1)
        & (delta_t >= lower)
        & (delta_t <= upper)
        & np.all((log_scale > -2.0) & (log_scale < 2.0), axis=1)
        & np.all((log_sigma > -5.0) & (log_sigma < 8.0), axis=1)
    )
    result = np.full(len(parameters), -np.inf, dtype=float)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        total = np.zeros(len(valid_indices), dtype=float)
        for index, segment in enumerate(model["segments"]):
            prediction = _segment_prediction(segment, delta_t[valid_indices])
            residual = segment["post_values"][None, :, :] - np.exp(
                log_scale[valid_indices, index, None, None]
            ) * prediction
            weighted_sse = np.sum(
                segment["post_weights"][None, :, :] * residual**2,
                axis=(1, 2),
            )
            sigma = np.exp(log_sigma[valid_indices, index])
            n_data = float(np.sum(segment["post_weights"]))
            total += -0.5 * (
                weighted_sse / sigma**2
                + n_data * np.log(2.0 * np.pi * sigma**2)
            )
        result[valid_indices] = total
    return float(result[0]) if scalar else result


def _scale_and_sse(segment, prediction):
    weights = segment["post_weights"][None, :, :]
    observed = segment["post_values"][None, :, :]
    numerator = np.sum(weights * observed * prediction, axis=(1, 2))
    denominator = np.sum(weights * prediction**2, axis=(1, 2))
    scale = numerator / np.maximum(denominator, 1.0e-30)
    scale = np.clip(scale, np.exp(-1.999), np.exp(1.999))
    residual = observed - scale[:, None, None] * prediction
    weighted_sse = np.sum(weights * residual**2, axis=(1, 2))
    return scale, weighted_sse


def initial_center(model, grid_size=801):
    lower, upper = common_delta_bounds(model)
    candidates = np.linspace(lower + 1.0e-8, upper - 1.0e-8, grid_size)
    score = np.zeros(grid_size, dtype=float)
    for segment in model["segments"]:
        prediction = _segment_prediction(segment, candidates)
        _, weighted_sse = _scale_and_sse(segment, prediction)
        n_data = float(np.sum(segment["post_weights"]))
        score += n_data * np.log(np.maximum(weighted_sse, 1.0e-30) / n_data)
    best_index = int(np.argmin(score))
    delta_t = candidates[best_index]
    scales = []
    sigmas = []
    prediction = []
    for segment in model["segments"]:
        segment_prediction = _segment_prediction(segment, delta_t)
        scale, weighted_sse = _scale_and_sse(segment, segment_prediction[None, ...])
        n_data = float(np.sum(segment["post_weights"]))
        scales.append(float(scale[0]))
        sigmas.append(max(float(np.sqrt(weighted_sse[0] / n_data)), np.exp(-4.0)))
        prediction.append(segment_prediction)
    return np.asarray([delta_t, *np.log(scales), *np.log(sigmas)]), prediction


def fit_common_delta(model, seed, warmup_steps, production_steps, nwalkers):
    n_segments = model["n_segments"]
    ndim = 1 + 2 * n_segments
    if nwalkers < 2 * ndim:
        raise ValueError(f"nwalkers must be at least {2 * ndim} for {ndim} parameters")
    center, _ = initial_center(model)
    lower, upper = common_delta_bounds(model)
    widths = np.concatenate(
        (
            [max((upper - lower) * 1.0e-3, 1.0e-5)],
            np.full(n_segments, 1.0e-4),
            np.full(n_segments, 1.0e-3),
        )
    )
    rng = np.random.default_rng(seed)
    positions = center + rng.normal(size=(nwalkers, ndim)) * widths
    positions[:, 0] = np.clip(positions[:, 0], lower + 1.0e-8, upper - 1.0e-8)
    positions[:, 1 : 1 + n_segments] = np.clip(
        positions[:, 1 : 1 + n_segments], -1.999, 1.999
    )
    positions[:, 1 + n_segments :] = np.clip(
        positions[:, 1 + n_segments :], -4.999, 7.999
    )
    np.random.seed(seed)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lambda theta: log_probability(model, theta),
        vectorize=True,
    )
    sampler.run_mcmc(positions, warmup_steps, progress=True, skip_initial_state_check=True)
    sampler.reset()
    sampler.run_mcmc(None, production_steps, progress=True, skip_initial_state_check=True)
    try:
        autocorrelation_time = sampler.get_autocorr_time(tol=0)
    except emcee.autocorr.AutocorrError:
        autocorrelation_time = np.full(ndim, np.nan)
    return {
        "chain": sampler.get_chain(),
        "log_probability": sampler.get_log_prob(),
        "acceptance_fraction": sampler.acceptance_fraction,
        "autocorrelation_time": autocorrelation_time,
        "initial_center": center,
    }


def _quantiles(values):
    return {
        str(level): float(np.quantile(values, level))
        for level in (0.025, 0.16, 0.5, 0.84, 0.975)
    }


def _summary(model, result):
    chain = result["chain"]
    flat = chain.reshape(-1, chain.shape[-1])
    n_segments = model["n_segments"]
    delta_t = flat[:, 0]
    scales = np.exp(flat[:, 1 : 1 + n_segments])
    sigmas = np.exp(flat[:, 1 + n_segments :])
    parameters = {}
    for index, segment in enumerate(model["segments"]):
        parameters[f"{segment['detector']}_LED{segment['led']}"] = {
            "detector": segment["detector"],
            "led": segment["led"],
            "common_delta_t_from_t0_days": _quantiles(delta_t),
            "common_shift_days": _quantiles(delta_t - model["reset_day"]),
            "multiplicative_scale": _quantiles(scales[:, index]),
            "scale_change_percent": _quantiles(100.0 * (scales[:, index] - 1.0)),
            "sigma_counts": _quantiles(sigmas[:, index]),
        }
    lower, upper = common_delta_bounds(model)
    return {
        "parameterization": "one common delta_t_from_t0 with independent multiplicative scale and sigma per detector/LED",
        "reset_day": model["reset_day"],
        "common_delta_t_prior_days": [lower, upper],
        "detectors": model["detectors"],
        "n_tiles_per_detector": model["n_tiles"],
        "n_data": model["n_data"],
        "correction_paths": model["correction_paths"],
        "posterior_quantiles": {
            "common_delta_t_from_t0_days": _quantiles(delta_t),
            "common_shift_days": _quantiles(delta_t - model["reset_day"]),
            "parameters": parameters,
        },
        "acceptance_fraction_mean": float(np.mean(result["acceptance_fraction"])),
        "autocorrelation_time": [float(value) for value in result["autocorrelation_time"]],
        "chain_shape": list(chain.shape),
    }


def _write_parameter_csv(path, model, result):
    flat = result["chain"].reshape(-1, result["chain"].shape[-1])
    n_segments = model["n_segments"]
    delta_t = flat[:, 0]
    scales = np.exp(flat[:, 1 : 1 + n_segments])
    sigmas = np.exp(flat[:, 1 + n_segments :])
    fields = (
        "detector",
        "led",
        "common_delta_t_p025",
        "common_delta_t_p16",
        "common_delta_t_median",
        "common_delta_t_p84",
        "common_delta_t_p975",
        "scale_p025",
        "scale_p16",
        "scale_median",
        "scale_p84",
        "scale_p975",
        "sigma_p16",
        "sigma_median",
        "sigma_p84",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        delta_q = np.quantile(delta_t, (0.025, 0.16, 0.5, 0.84, 0.975))
        for index, segment in enumerate(model["segments"]):
            scale_q = np.quantile(scales[:, index], (0.025, 0.16, 0.5, 0.84, 0.975))
            sigma_q = np.quantile(sigmas[:, index], (0.16, 0.5, 0.84))
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [segment["detector"], segment["led"], *delta_q, *scale_q, *sigma_q],
                    )
                )
            )


def save_plots(output_dir, model, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chain = result["chain"]
    flat = chain.reshape(-1, chain.shape[-1])
    best = flat[np.argmax(result["log_probability"].reshape(-1))]
    n_segments = model["n_segments"]
    delta_t = best[0]
    scales = np.exp(best[1 : 1 + n_segments])
    figure, axes = plt.subplots(
        model["n_detectors"],
        6,
        figsize=(20, max(4, 3.0 * model["n_detectors"])),
        squeeze=False,
        sharey=False,
    )
    shift = delta_t - model["reset_day"]
    for index, segment in enumerate(model["segments"]):
        axis = axes[segment["detector_index"], segment["led"] - 1]
        for tile_index in range(segment["pre_values"].shape[0]):
            axis.plot(segment["pre_days"], segment["pre_values"][tile_index], color="tab:blue", alpha=0.10, linewidth=0.4)
            axis.plot(segment["post_days"] + shift, segment["post_values"][tile_index] / scales[index], color="tab:green", alpha=0.10, linewidth=0.4)
        axis.plot(segment["pre_days"], np.median(segment["pre_values"], axis=0), color="tab:blue", marker="o", linewidth=1.5)
        axis.plot(segment["post_days"] + shift, np.median(segment["post_values"], axis=0) / scales[index], color="tab:green", marker="o", linewidth=1.5)
        axis.set_title(f"{segment['detector']} LED {segment['led']}")
        axis.set_xlabel("Time from t=0 (days)")
        axis.grid(alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Corrected current counts")
    figure.suptitle(f"Common delta_t={delta_t:.6f} days; free detector/LED scales")
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_common_delta_overlay_best_fit_counts.png", dpi=160)
    plt.close(figure)

    detectors = [str(detector) for detector in model["detectors"]]
    palette = plt.get_cmap("tab10").colors
    detector_colors = {
        detector: palette[index % len(palette)]
        for index, detector in enumerate(detectors)
    }
    delta_t_values = flat[:, 0]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(
        delta_t_values,
        bins=80,
        histtype="step",
        linewidth=1.8,
        color="tab:blue",
        label="Common delta_t",
    )
    scatter_step = max(1, len(delta_t_values) // 30000)
    for index, segment in enumerate(model["segments"]):
        color = detector_colors[str(segment["detector"])]
        scale_change = 100.0 * (np.exp(flat[:, 1 + index]) - 1.0)
        axes[1].hist(
            scale_change,
            bins=80,
            histtype="step",
            linewidth=1.0,
            color=color,
            alpha=0.85,
        )
        axes[2].scatter(
            delta_t_values[::scatter_step],
            scale_change[::scatter_step],
            s=1,
            alpha=0.12,
            color=color,
        )
    axes[0].set_xlabel("delta_t from t=0 (days)")
    axes[0].set_ylabel("Posterior samples")
    axes[1].set_xlabel("Multiplicative scale change (%)")
    axes[1].set_ylabel("Posterior samples")
    axes[2].set_xlabel("delta_t from t=0 (days)")
    axes[2].set_ylabel("Multiplicative scale change (%)")
    axes[0].legend(frameon=False)
    detector_handles = [
        Line2D([0], [0], color=detector_colors[detector], label=detector)
        for detector in detectors
    ]
    for axis in axes[1:]:
        axis.legend(handles=detector_handles, frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("WFI01-WFI05 common-shift posterior measured from t=0")
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_common_delta_posterior.png", dpi=160)
    plt.close(figure)

    leds = sorted({int(segment["led"]) for segment in model["segments"]})
    detector_indices = {detector: index for index, detector in enumerate(detectors)}
    led_indices = {led: index for index, led in enumerate(leds)}
    median_grid = np.full((len(detectors), len(leds)), np.nan)
    lower_grid = np.full_like(median_grid, np.nan)
    upper_grid = np.full_like(median_grid, np.nan)
    for index, segment in enumerate(model["segments"]):
        row = detector_indices[str(segment["detector"])]
        column = led_indices[int(segment["led"])]
        scale_change = 100.0 * (np.exp(flat[:, 1 + index]) - 1.0)
        lower, median, upper = np.quantile(scale_change, (0.025, 0.5, 0.975))
        median_grid[row, column] = median
        lower_grid[row, column] = lower
        upper_grid[row, column] = upper

    figure, axis = plt.subplots(figsize=(9, 5.5))
    image = axis.imshow(
        median_grid,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=np.nanmin(median_grid),
        vmax=np.nanmax(median_grid),
    )
    for row in range(len(detectors)):
        for column in range(len(leds)):
            if np.isfinite(median_grid[row, column]):
                axis.text(
                    column,
                    row,
                    f"{median_grid[row, column]:+.2f}%",
                    ha="center",
                    va="center",
                    color="black",
                )
    axis.set_xticks(range(len(leds)), [f"LED {led}" for led in leds])
    axis.set_yticks(range(len(detectors)), detectors)
    axis.set_xlabel("LED")
    axis.set_ylabel("Detector")
    axis.set_title("Median multiplicative scale change by detector and LED")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Scale change (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_common_delta_scale_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    led_values = np.asarray(leds)
    for detector in detectors:
        row = detector_indices[detector]
        valid = np.isfinite(median_grid[row])
        axis.errorbar(
            led_values[valid],
            median_grid[row, valid],
            yerr=np.vstack(
                (
                    median_grid[row, valid] - lower_grid[row, valid],
                    upper_grid[row, valid] - median_grid[row, valid],
                )
            ),
            color=detector_colors[detector],
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=detector,
        )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set_xticks(leds, [f"LED {led}" for led in leds])
    axis.set_xlabel("LED")
    axis.set_ylabel("Multiplicative scale change (%)")
    axis.set_title("LED dependence of scale change (95% posterior intervals)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    axis.set_ylim(np.nanmin(lower_grid) - 0.15, max(0.15, np.nanmax(upper_grid) + 0.15))
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_common_delta_scale_trends.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Count-space correction CSV or directory; omit for *_count_corrected.csv inputs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", default=None)
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=39)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=35)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--production-steps", type=int, default=PRODUCTION_STEPS)
    parser.add_argument("--nwalkers", type=int, default=NWALKERS)
    args = parser.parse_args()
    model = load_model_data(
        args.statistics,
        args.corrections,
        args.reset_day,
        args.additional_acm_step_repeat,
        args.pre_reset_acm_step_repeat,
        args.detectors,
    )
    result = fit_common_delta(
        model,
        args.seed,
        args.warmup_steps,
        args.production_steps,
        args.nwalkers,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(model, result)
    summary.update(
        {
            "seed": args.seed,
            "nwalkers": args.nwalkers,
            "warmup_steps": args.warmup_steps,
            "production_steps_per_walker": args.production_steps,
        }
    )
    np.savez(
        output_dir / "posterior_samples.npz",
        chain=result["chain"],
        log_probability=result["log_probability"],
        detectors=np.asarray(model["detectors"]),
        detector_led=np.asarray([f"{segment['detector']}_LED{segment['led']}" for segment in model["segments"]]),
        reset_day=model["reset_day"],
    )
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_parameter_csv(output_dir / "detector_led_shift_scale_posterior.csv", model, result)
    (output_dir / "summary.md").write_text(
        "# Common-delta count-space shift/free-scale MCMC\n\n"
        "One delta_t is shared by all detector/LED combinations. Each combination has "
        "an independent multiplicative scale and noise term.\n"
    )
    save_plots(output_dir, model, result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
